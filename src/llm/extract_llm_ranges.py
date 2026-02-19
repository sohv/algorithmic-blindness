#!/usr/bin/env python3
"""
Extract LLM Ranges from Raw Responses
=====================================
Reads raw LLM responses and extracts metric ranges (precision, recall, f1, shd).
Much easier to debug and iterate than parsing during querying.

Usage:
    python extract_llm_ranges.py --input_dir variance/raw_responses --output_dir variance/extracted_ranges
"""

import re
import json
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class MetricRange:
    """Extracted metric range."""
    lower: float
    upper: float
    confidence: str  # 'high', 'medium', 'low'
    source_pattern: str  # Which pattern matched


def strip_verbose_reasoning(response_text: str) -> str:
    """
    Extract ONLY the numerical metric lines from verbose responses.
    
    DeepSeek often provides extensive reasoning/explanations. This function 
    preserves only the final predictions in the required format:
        Precision: [X.XX, X.XX]
        Recall: [X.XX, X.XX]
        F1: [X.XX, X.XX]
        SHD: [X, X]
    
    Args:
        response_text: Full response that may contain explanations
        
    Returns:
        Cleaned response with only metric lines
    """
    lines = response_text.split('\n')
    metric_lines = []
    
    # Metrics to look for (case-insensitive)
    metric_keywords = ['precision', 'recall', 'f1', 'shd']
    
    for line in lines:
        # Check if line contains any metric keyword
        line_lower = line.lower().strip()
        
        # Skip empty lines and non-metric lines
        if not line_lower:
            continue
            
        # Keep lines that have a metric keyword followed by a bracket/colon and range
        has_metric = any(metric in line_lower for metric in metric_keywords)
        has_range = '[' in line and ']' in line
        
        if has_metric and has_range:
            metric_lines.append(line.strip())
    
    # If we found metric lines, return them; otherwise return original text
    if metric_lines:
        return '\n'.join(metric_lines)
    else:
        return response_text


def extract_ranges_from_text(text: str) -> Dict[str, MetricRange]:
    """Extract all metric ranges from response text."""
    
    # FIRST: Strip verbose reasoning if present (for DeepSeek and other verbose models)
    text = strip_verbose_reasoning(text)
    
    # Clean text
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'\$[^$]*\$', '', text)  # LaTeX
    text = re.sub(r'#+\s+', '', text)  # Headers
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
    
    results = {}
    
    # Patterns in order of confidence
    patterns = {
        'precision': [
            # High confidence: explicit format
            (r'\*?precision\*?\s*[:\*\.]+\s*\[?([0-9.]+)\s*,\s*([0-9.]+)\]?', 'high'),
            (r'\d+\.\s*\*?precision\*?\s*[:\s]+([0-9.]+)[\s\-,]*([0-9.]+)', 'high'),
            # Medium confidence: narrative with metric + range
            (r'precision\s+(?:ranging|from|between|is|around|approximately)\s+(?:from\s+)?([0-9.]+)\s+(?:to|-)\s+([0-9.]+)', 'medium'),
            # Low confidence: just metric word + two numbers
            (r'precision[^0-9]*([0-9.]+)[^0-9]*([0-9.]+)', 'low'),
        ],
        'recall': [
            (r'\*?recall\*?\s*[:\*\.]+\s*\[?([0-9.]+)\s*,\s*([0-9.]+)\]?', 'high'),
            (r'\d+\.\s*\*?recall\*?\s*[:\s]+([0-9.]+)[\s\-,]*([0-9.]+)', 'high'),
            (r'recall\s+(?:ranging|from|between|is|around|approximately)\s+(?:from\s+)?([0-9.]+)\s+(?:to|-)\s+([0-9.]+)', 'medium'),
            (r'recall[^0-9]*([0-9.]+)[^0-9]*([0-9.]+)', 'low'),
        ],
        'f1': [
            (r'\*?f[1-]\s*(?:score)?\*?\s*[:\*\.]+\s*\[?([0-9.]+)\s*,\s*([0-9.]+)\]?', 'high'),
            (r'\d+\.\s*\*?f[1-]\*?\s*[:\s]+([0-9.]+)[\s\-,]*([0-9.]+)', 'high'),
            (r'f[1-]\s+(?:ranging|from|between|is|around|approximately)\s+(?:from\s+)?([0-9.]+)\s+(?:to|-)\s+([0-9.]+)', 'medium'),
            (r'f[1-][^0-9]*([0-9.]+)[^0-9]*([0-9.]+)', 'low'),
        ],
        'shd': [
            (r'(?:\*?shd\*?|structural\s+hamming\s+distance)\s*[:\*\.]+\s*\[?([0-9.]+)\s*,\s*([0-9.]+)\]?', 'high'),
            (r'\d+\.\s*(?:\*?shd\*?|distance)\s*[:\s]+([0-9.]+)[\s\-,]*([0-9.]+)', 'high'),
            (r'(?:shd|distance)\s+(?:ranging|from|between|is|around|approximately)\s+(?:from\s+)?([0-9.]+)\s+(?:to|-)\s+([0-9.]+)', 'medium'),
            (r'(?:shd|distance)[^0-9]*([0-9.]+)[^0-9]*([0-9.]+)', 'low'),
        ],
    }
    
    for metric_name, metric_patterns in patterns.items():
        for pattern_str, confidence in metric_patterns:
            if metric_name in results:
                break  # Already found this metric
            
            pattern = re.compile(pattern_str, re.IGNORECASE | re.DOTALL)
            match = pattern.search(text)
            
            if match:
                try:
                    lower = float(match.group(1))
                    upper = float(match.group(2))
                    
                    # Normalize percentages
                    if lower > 1.0 and upper > 1.0 and metric_name != 'shd':
                        if lower <= 100 and upper <= 100:
                            lower /= 100.0
                            upper /= 100.0
                    
                    # Auto-fix swapped
                    if lower > upper:
                        lower, upper = upper, lower
                    
                    results[metric_name] = MetricRange(
                        lower=lower,
                        upper=upper,
                        confidence=confidence,
                        source_pattern=pattern_str[:50]
                    )
                    break
                except (ValueError, IndexError):
                    continue
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Extract metric ranges from raw LLM responses")
    parser.add_argument('--input_dir', type=str, default='variance/raw_responses',
        help='Directory containing raw response files')
    parser.add_argument('--output_dir', type=str, default='variance/extracted_ranges',
        help='Output directory for extracted ranges')
    parser.add_argument('--verbose', action='store_true', help='Show extraction details')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("="*80)
    print("EXTRACTING METRIC RANGES FROM RAW RESPONSES")
    print("="*80)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print()
    
    # Find all raw response files
    raw_files = sorted(input_dir.glob("*_raw.txt"))
    print(f"Found {len(raw_files)} raw response files\n")
    
    # Group by (dataset, algorithm, formulation)
    responses_by_exp = {}
    
    for raw_file in raw_files:
        # Parse filename: {dataset}_{algorithm}_f{formulation}_{model}_raw.txt
        # Note: dataset can have underscores (e.g., synthetic_12) so we need to find formulation pattern
        stem = raw_file.stem.replace('_raw', '')
        parts = stem.split('_')
        
        if len(parts) < 4:
            print(f"⚠ Skipping {raw_file.name} - unexpected filename format")
            continue
        
        # Find formulation by looking for f1/f2/f3 pattern from the end
        # Format is: ..._{algorithm}_f{N}_{model}
        model = parts[-1]  # Last part is model
        formulation = parts[-2]  # Second to last should be f1/f2/f3
        
        # Everything before algorithm_formulation is dataset_algorithm
        # The part before formulation is algorithm
        algorithm = parts[-3]  # Third from last is algorithm
        dataset = '_'.join(parts[:-3])  # Everything before algorithm is dataset
        dataset_algo = f"{dataset}_{algorithm}"  # Reconstruct dataset_algorithm
        
        exp_key = (dataset_algo, formulation)
        if exp_key not in responses_by_exp:
            responses_by_exp[exp_key] = {}
        
        # Read and extract
        with open(raw_file, 'r') as f:
            text = f.read()
        
        ranges = extract_ranges_from_text(text)
        responses_by_exp[exp_key][model] = ranges
        
        if args.verbose:
            print(f"✓ {raw_file.name}")
            for metric, rng in ranges.items():
                print(f"  {metric}: {rng.lower:.3f}-{rng.upper:.3f} ({rng.confidence})")
    
    # Save structured output
    total_extracted = 0
    for (exp_key, formulation), llm_responses in responses_by_exp.items():
        output_file = output_dir / f"{exp_key}_{formulation}_ranges.json"
        
        # Convert to JSON-serializable format
        output_data = {
            'experiment': exp_key,
            'formulation': formulation,
            'llm_estimates': {}
        }
        
        for model, ranges in llm_responses.items():
            output_data['llm_estimates'][model] = {
                metric: {
                    'lower': rng.lower,
                    'upper': rng.upper,
                    'confidence': rng.confidence,
                    'source': rng.source_pattern
                }
                for metric, rng in ranges.items()
            }
            total_extracted += len(ranges)
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f" {output_file.name}")
    
    print()
    print("="*80)
    print(f"Extracted {total_extracted} metric ranges from {len(raw_files)} responses")
    print(f"Saved to: {output_dir}/")
    print(f"Next step: python compare_llm_vs_algorithmic.py --llm_dir {output_dir} --algorithmic_dir ../experiments/results")


if __name__ == "__main__":
    main()
