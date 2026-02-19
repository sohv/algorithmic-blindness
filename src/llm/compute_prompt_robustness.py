#!/usr/bin/env python3
"""
Compute Prompt Robustness via Coefficient of Variation
=========================================================
Analyzes stability of LLM estimates across the 3 prompt formulations.

For each LLM and metric, computes CV across formulations:
    CV% = (std_dev / mean) * 100

Usage:
    python compute_prompt_robustness.py --input_dir results/raw_responses --output_dir results/robustness_analysis

Output:
    - robustness_summary.json: CV for each LLM per metric
    - robustness_report.txt: Human-readable robustness assessment
"""

import re
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import statistics


@dataclass
class MetricEstimate:
    """Single metric estimate from a formulation."""
    lower: float
    upper: float
    
    def midpoint(self) -> float:
        return (self.lower + self.upper) / 2
    
    def width(self) -> float:
        return self.upper - self.lower


def extract_ranges_from_text(text: str) -> Dict[str, MetricEstimate]:
    """Extract metric ranges from raw response."""
    results = {}
    
    # Strip verbose reasoning first
    lines = text.split('\n')
    metric_lines = []
    metric_keywords = ['precision', 'recall', 'f1', 'shd']
    
    for line in lines:
        line_lower = line.lower().strip()
        if not line_lower:
            continue
        has_metric = any(metric in line_lower for metric in metric_keywords)
        has_range = '[' in line and ']' in line
        if has_metric and has_range:
            metric_lines.append(line.strip())
    
    if not metric_lines:
        metric_lines = lines  # Fall back to original if no metric lines found
    
    clean_text = '\n'.join(metric_lines)
    
    # Extract metrics
    patterns = [
        (r'[Pp]recision\s*:\s*\[([\d.]+)\s*,\s*([\d.]+)\]', 'precision'),
        (r'[Rr]ecall\s*:\s*\[([\d.]+)\s*,\s*([\d.]+)\]', 'recall'),
        (r'[Ff]1\s*:\s*\[([\d.]+)\s*,\s*([\d.]+)\]', 'f1'),
        (r'[Ss][Hh][Dd]\s*:\s*\[([\d.]+)\s*,\s*([\d.]+)\]', 'shd'),
    ]
    
    for pattern, metric_name in patterns:
        match = re.search(pattern, clean_text)
        if match:
            try:
                lower = float(match.group(1))
                upper = float(match.group(2))
                results[metric_name] = MetricEstimate(lower, upper)
            except ValueError:
                pass
    
    return results


def compute_cv(values: List[float]) -> Tuple[float, float]:
    """Compute coefficient of variation (CV%) and standard deviation.
    
    Args:
        values: List of values (e.g., midpoints across formulations)
        
    Returns:
        (cv_percent, std_dev)
    """
    if len(values) < 2:
        return 0.0, 0.0
    
    mean = statistics.mean(values)
    if mean == 0:
        return 0.0, 0.0
    
    std_dev = statistics.stdev(values)
    cv_percent = (std_dev / mean) * 100
    
    return cv_percent, std_dev


def main():
    parser = argparse.ArgumentParser(description="Compute prompt robustness via coefficient of variation")
    parser.add_argument('--input_dir', type=str, default='results/raw_responses',
        help='Directory with raw LLM responses')
    parser.add_argument('--output_dir', type=str, default='results/robustness_analysis',
        help='Output directory for robustness analysis')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return
    
    # Group files by (dataset, algorithm, model)
    # File format: {dataset}_{algorithm}_f{formulation}_{model}_raw.txt
    grouped = defaultdict(dict)  # (dataset, algorithm, model) -> {formulation: metrics}
    
    for raw_file in sorted(input_dir.glob("*_raw.txt")):
        # Parse filename
        parts = raw_file.stem.replace('_raw', '').split('_')
        
        # Format: dataset_algorithm_f{formulation}_model
        # But dataset name might have underscores (e.g., synthetic_12)
        # So we parse from the end: model, formulation, then algorithm and dataset
        
        try:
            model = parts[-1]  # Last part is model
            formulation_str = parts[-2]  # Second to last should be f{formulation}
            
            if not formulation_str.startswith('f'):
                continue
            
            formulation = int(formulation_str[1])
            
            # Everything before formulation is dataset_algorithm
            remaining = '_'.join(parts[:-2])
            
            # Split from end: last part is algorithm, rest is dataset
            remaining_parts = remaining.split('_')
            algorithm = remaining_parts[-1]
            dataset = '_'.join(remaining_parts[:-1])
            
            key = (dataset, algorithm, model)
            
            # Extract metrics from file
            with open(raw_file, 'r') as f:
                content = f.read()
            
            metrics = extract_ranges_from_text(content)
            if metrics:  # Only add if we found metrics
                grouped[key][formulation] = metrics
        
        except Exception as e:
            print(f"Warning: Could not parse {raw_file.name}: {e}")
            continue
    
    # Compute CV for each (dataset, algorithm, model)
    robustness_data = {}  # (dataset, algorithm, model) -> {metric -> cv_info}
    
    for (dataset, algorithm, model), formulation_metrics in grouped.items():
        # Need data from at least 2 formulations to compute CV
        if len(formulation_metrics) < 2:
            continue
        
        # Get metrics present in all formulations
        all_metrics_per_f = [set(m.keys()) for m in formulation_metrics.values()]
        common_metrics = set.intersection(*all_metrics_per_f) if all_metrics_per_f else set()
        
        cv_info = {}
        
        for metric_name in sorted(common_metrics):
            # Collect midpoints and widths across formulations
            midpoints = []
            widths = []
            
            for formulation in sorted(formulation_metrics.keys()):
                estimate = formulation_metrics[formulation].get(metric_name)
                if estimate:
                    midpoints.append(estimate.midpoint())
                    widths.append(estimate.width())
            
            if len(midpoints) >= 2:
                cv_midpoint, std_midpoint = compute_cv(midpoints)
                cv_width, std_width = compute_cv(widths)
                
                cv_info[metric_name] = {
                    'cv_midpoint_percent': round(cv_midpoint, 2),
                    'std_midpoint': round(std_midpoint, 4),
                    'mean_midpoint': round(statistics.mean(midpoints), 4),
                    'cv_width_percent': round(cv_width, 2),
                    'std_width': round(std_width, 4),
                    'mean_width': round(statistics.mean(widths), 4),
                    'formulations_available': len(midpoints),
                    'midpoints': [round(m, 4) for m in midpoints],
                    'widths': [round(w, 4) for w in widths],
                }
        
        if cv_info:
            key_str = f"{dataset}_{algorithm}_{model}"
            robustness_data[key_str] = cv_info
    
    # Save JSON results
    json_output = output_dir / "robustness_summary.json"
    with open(json_output, 'w') as f:
        json.dump(robustness_data, f, indent=2)
    print(f"✓ Saved robustness summary: {json_output}")
    
    # Generate human-readable report
    report_output = output_dir / "robustness_report.txt"
    
    with open(report_output, 'w') as f:
        f.write("="*90 + "\n")
        f.write("COEFFICIENT OF VARIATION ACROSS PROMPT FORMULATIONS\n")
        f.write("="*90 + "\n\n")
        
        f.write("CV% = (std_dev / mean) * 100\n\n")
        
        # Group by model for summary
        model_cv_summary = defaultdict(list)
        
        for key_str in sorted(robustness_data.keys()):
            dataset, algorithm, model = key_str.rsplit('_', 2)
            cv_info = robustness_data[key_str]
            
            f.write(f"\n{key_str}\n")
            f.write("-" * 70 + "\n")
            
            for metric_name in sorted(cv_info.keys()):
                metric_cv = cv_info[metric_name]
                f.write(f"  {metric_name.upper()}:\n")
                f.write(f"    CV (midpoint):  {metric_cv['cv_midpoint_percent']:.2f}%  ")
                f.write(f"(mean={metric_cv['mean_midpoint']:.4f}, std={metric_cv['std_midpoint']:.4f})\n")
                f.write(f"    CV (width):     {metric_cv['cv_width_percent']:.2f}%  ")
                f.write(f"(mean={metric_cv['mean_width']:.4f}, std={metric_cv['std_width']:.4f})\n")
                
                # Collect for model summary
                model_cv_summary[model].append(metric_cv['cv_midpoint_percent'])
        
        # Summary by model
        f.write("\n\n" + "="*90 + "\n")
        f.write("COEFFICIENT OF VARIATION (%) BY MODEL\n")
        f.write("="*90 + "\n\n")
        
        for model in sorted(model_cv_summary.keys()):
            cv_values = model_cv_summary[model]
            avg_cv = statistics.mean(cv_values)
            max_cv = max(cv_values)
            
            f.write(f"{model.upper():<20} Avg: {avg_cv:6.2f}%   Max: {max_cv:6.2f}%\n")
    
    print(f"✓ Saved robustness report: {report_output}")
    
    # Print summary to console
    print("\n" + "="*90)
    print("COEFFICIENT OF VARIATION (%) BY MODEL\n")
    
    model_cv_summary = defaultdict(list)
    for key_str in robustness_data.keys():
        dataset, algorithm, model = key_str.rsplit('_', 2)
        cv_info = robustness_data[key_str]
        for metric_cv in cv_info.values():
            model_cv_summary[model].append(metric_cv['cv_midpoint_percent'])
    
    for model in sorted(model_cv_summary.keys()):
        cv_values = model_cv_summary[model]
        avg_cv = statistics.mean(cv_values)
        max_cv = max(cv_values)
        print(f"{model.upper():<15} → Avg: {avg_cv:6.2f}%   Max: {max_cv:6.2f}%")
    
    print("\n" + "="*90)


if __name__ == "__main__":
    main()
