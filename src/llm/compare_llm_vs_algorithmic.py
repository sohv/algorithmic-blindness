#!/usr/bin/env python3
"""
Compare LLM Estimates vs Algorithmic Confidence Intervals
===========================================================
Compares LLM-extracted metric ranges with algorithmic confidence intervals.

Usage:
    python compare_llm_vs_algorithmic.py --llm_dir variance/extracted_ranges --algorithmic_dir ../experiments/results
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import statistics


@dataclass
class ComparisonResult:
    """Result of comparing LLM estimate to algorithmic CI."""
    metric: str
    llm_range: Tuple[float, float]
    algorithmic_ci: Tuple[float, float]
    algorithmic_mean: float  # True value from algorithms
    
    # PRIMARY: Calibrated Coverage
    calibrated_coverage: bool  # Does LLM range contain algorithmic mean?
    calibrated_coverage_score: float  # 0-1 score: how well LLM range brackets the truth
    
    # Secondary metrics
    overlap: bool
    containment: str  # 'llm_contains_algo', 'algo_contains_llm', 'partial', 'none'
    overlap_fraction: float  # Jaccard similarity
    center_distance: float
    
    # Widths
    llm_width: float
    algo_width: float
    
    # Scores
    accuracy_score: float  # How well LLM estimated, 0-1
    confidence_score: str  # 'high', 'medium', 'low'


def compute_overlap(range1: Tuple[float, float], range2: Tuple[float, float]) -> Tuple[bool, float]:
    """Check if ranges overlap and compute overlap fraction (Jaccard)."""
    a_low, a_high = range1
    b_low, b_high = range2
    
    # Intersection
    inter_low = max(a_low, b_low)
    inter_high = min(a_high, b_high)
    
    if inter_high < inter_low:
        return False, 0.0
    
    # Jaccard: intersection / union
    inter_size = inter_high - inter_low
    union_low = min(a_low, b_low)
    union_high = max(a_high, b_high)
    union_size = union_high - union_low
    
    jaccard = inter_size / union_size if union_size > 0 else 0.0
    return True, jaccard


def compute_containment(llm_range: Tuple[float, float], algo_range: Tuple[float, float]) -> str:
    """Determine containment relationship."""
    llm_low, llm_high = llm_range
    algo_low, algo_high = algo_range
    
    llm_contains = llm_low <= algo_low and llm_high >= algo_high
    algo_contains = algo_low <= llm_low and algo_high >= llm_high
    
    if llm_contains:
        return 'llm_contains_algo'
    elif algo_contains:
        return 'algo_contains_llm'
    else:
        # Check partial overlap
        if max(llm_low, algo_low) <= min(llm_high, algo_high):
            return 'partial'
        else:
            return 'none'


def compare_ranges(metric: str, llm_range: Dict, algo_ci: Tuple[float, float], 
                   algo_mean: float) -> Optional[ComparisonResult]:
    """Compare a single LLM range estimate with algorithmic CI.
    
    Args:
        metric: Metric name (e.g., 'precision')
        llm_range: LLM predicted range {'lower': X, 'upper': Y, ...}
        algo_ci: Algorithmic 95% CI as (lower, upper)
        algo_mean: True algorithmic mean value
    """
    
    if not llm_range or 'lower' not in llm_range:
        return None
    
    llm_low = llm_range['lower']
    llm_high = llm_range['upper']
    algo_low, algo_high = algo_ci
    
    # PRIMARY: Calibrated Coverage (does true mean fall in LLM range?)
    calibrated_coverage = bool(llm_low <= algo_mean <= llm_high)
    
    # CONTINUOUS COVERAGE SCORE (0-1): How well does LLM range bracket the truth?
    # - 1.0: algo_mean at center of LLM range (perfect)
    # - 0.5: algo_mean at edge of LLM range (acceptable)
    # - 0.0: algo_mean far outside LLM range (terrible)
    llm_center = (llm_low + llm_high) / 2
    llm_half_width = (llm_high - llm_low) / 2
    
    if llm_half_width == 0:
        # Zero-width range: score based on exact match
        calibrated_coverage_score = 1.0 if algo_mean == llm_center else 0.0
    else:
        distance_from_center = abs(algo_mean - llm_center)
        if distance_from_center <= llm_half_width:
            # Inside range: score = 1 - (fraction of half-width to edge)
            calibrated_coverage_score = 1.0 - (distance_from_center / llm_half_width) * 0.5
        else:
            # Outside range: penalty based on how far outside
            distance_outside = distance_from_center - llm_half_width
            calibrated_coverage_score = max(0.0, 1.0 - (distance_outside / llm_half_width))
    
    calibrated_coverage_score = float(round(calibrated_coverage_score, 4))
    
    
    # Compute secondary metrics
    overlap, jaccard = compute_overlap((llm_low, llm_high), (algo_low, algo_high))
    
    containment = compute_containment((llm_low, llm_high), (algo_low, algo_high))
    
    center_distance = abs(
        ((llm_low + llm_high) / 2) - 
        ((algo_low + algo_high) / 2)
    )
    
    llm_width = llm_high - llm_low
    algo_width = algo_high - algo_low
    
    # Accuracy score: penalize distance and width mismatch
    accuracy = 1.0
    if center_distance > 0.1:
        accuracy -= min(0.3, center_distance)
    if abs(llm_width - algo_width) > 0.2:
        accuracy -= 0.2
    accuracy = max(0.0, accuracy)
    
    # Confidence based on LLM extraction confidence
    confidence = llm_range.get('confidence', 'low')
    
    return ComparisonResult(
        metric=metric,
        llm_range=(llm_low, llm_high),
        algorithmic_ci=(algo_low, algo_high),
        algorithmic_mean=algo_mean,
        calibrated_coverage=calibrated_coverage,
        calibrated_coverage_score=calibrated_coverage_score,
        overlap=overlap,
        containment=containment,
        overlap_fraction=jaccard,
        center_distance=center_distance,
        llm_width=llm_width,
        algo_width=algo_width,
        accuracy_score=accuracy,
        confidence_score=confidence,
    )


def load_algorithmic_ci(algo_results_dir: Path, dataset: str, algorithm: str, 
                         formulation: str) -> Optional[Dict]:
    """Load algorithmic variance results for a given experiment."""
    
    # Look for {dataset}_{algorithm}_variance.json in results/
    variance_file = algo_results_dir / f"{dataset}_{algorithm}_variance.json"
    
    if not variance_file.exists():
        return None
    
    with open(variance_file, 'r') as f:
        data = json.load(f)
    
    # Algorithmic results have direct metrics at top level (not formulation-specific)
    # Since LLM aggregates across formulations, we use the overall algorithmic results
    if 'results' in data:
        metrics_data = data['results']
    elif 'formulations' in data:
        # Fallback for legacy format with formulations
        formulation_str = f"f{formulation}" if isinstance(formulation, int) else formulation
        if formulation_str in data['formulations']:
            metrics_data = data['formulations'][formulation_str].get('metrics', {})
        else:
            return None
    else:
        return None
    
    # Extract CI bounds for each metric
    extracted_metrics = {}
    for metric in ['precision', 'recall', 'f1', 'shd']:
        if metric in metrics_data:
            metric_data = metrics_data[metric]
            extracted_metrics[metric] = {
                'ci': (metric_data.get('ci_95_lower'), metric_data.get('ci_95_upper')),
                'mean': metric_data.get('mean')
            }
    
    return extracted_metrics if extracted_metrics else None


def main():
    parser = argparse.ArgumentParser(description="Compare LLM estimates with algorithmic CIs")
    parser.add_argument('--llm_dir', type=str, default='variance/aggregated_ranges',
        help='Directory with aggregated LLM ranges (averaged across formulations)')
    parser.add_argument('--algorithmic_dir', type=str, default='../experiments/results',
        help='Directory with algorithmic variance results')
    parser.add_argument('--output_dir', type=str, default='variance/comparisons',
        help='Output directory for comparison results')
    parser.add_argument('--verbose', action='store_true', help='Show detailed comparison')
    
    args = parser.parse_args()
    
    llm_dir = Path(args.llm_dir)
    algo_dir = Path(args.algorithmic_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("="*80)
    print("COMPARING LLM ESTIMATES VS ALGORITHMIC CONFIDENCE INTERVALS")
    print("="*80)
    print(f"LLM ranges:       {llm_dir} (aggregated across formulations)")
    print(f"Algorithmic CIs:  {algo_dir}")
    print(f"Output:           {output_dir}")
    print()
    
    # Load all LLM aggregated ranges
    llm_files = sorted(llm_dir.glob("*_aggregated.json"))
    print(f"Found {len(llm_files)} LLM aggregation files\n")
    
    if not llm_files:
        print("❌ No LLM aggregation files found. Run aggregate_formulations.py first.")
        return
    
    # Comparison results
    all_comparisons = {}
    all_models_seen = set()
    
    for llm_file in llm_files:
        # Parse: {dataset}_{algorithm}_aggregated.json
        # Handle datasets with underscores (e.g., synthetic_12_fci_aggregated.json)
        stem = llm_file.stem
        if not stem.endswith('_aggregated'):
            print(f" Skipping {llm_file.name} - unexpected format")
            continue
        
        stem_without_suffix = stem[:-len('_aggregated')]  # Remove "_aggregated"
        parts = stem_without_suffix.split('_')
        
        if len(parts) < 2:
            print(f" Skipping {llm_file.name} - unexpected format")
            continue
        
        # Last part is always algorithm, everything before it is dataset
        algorithm = parts[-1]
        dataset = '_'.join(parts[:-1])
        
        # Load data
        with open(llm_file, 'r') as f:
            llm_data = json.load(f)
        
        # Track all models seen
        for model_name in llm_data['llm_estimates'].keys():
            all_models_seen.add(model_name)
        
        # Load algorithmic CI (use any formulation, they all have same algorithmic results)
        algo_metrics = load_algorithmic_ci(algo_dir, dataset, algorithm, '1')
        
        if not algo_metrics:
            if args.verbose:
                print(f" Skipping {dataset} {algorithm} - no algorithmic data found")
            continue
        
        # Compare each LLM model
        exp_key = f"{dataset}_{algorithm}"
        all_comparisons[exp_key] = {
            'dataset': dataset,
            'algorithm': algorithm,
            'aggregation_source': 'mean of f1, f2, f3',
            'models': {}
        }
        
        for model, model_estimates in llm_data['llm_estimates'].items():
            model_comparisons = {}
            
            for metric in ['precision', 'recall', 'f1', 'shd']:
                if metric not in model_estimates or metric not in algo_metrics:
                    continue
                
                llm_estimate = model_estimates[metric]
                algo_data = algo_metrics[metric]
                algo_ci = algo_data['ci']
                algo_mean = algo_data['mean']
                
                comparison = compare_ranges(metric, llm_estimate, algo_ci, algo_mean)
                
                if comparison:
                    model_comparisons[metric] = asdict(comparison)
            
            all_comparisons[exp_key]['models'][model] = model_comparisons
            
            if args.verbose:
                print(f"✓ {exp_key} / {model}: {len(model_comparisons)} metrics compared")
    
    # Save results
    output_file = output_dir / "comparison_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_comparisons, f, indent=2)
    
    # Format LLM names for filename
    expected_all_models = {'gpt5', 'claude', 'geminiflash', 'gemini3', 'deepseek', 'llama', 'qwen', 'qwenthink'}
    all_models_normalized = {m.lower() for m in all_models_seen}
    
    # Check if we have "all" common models (at least 5-6 of them)
    if len(all_models_normalized) >= 5:
        llm_suffix = "all"
    else:
        # Format specific models with underscore separation, sorted for consistency
        sorted_models = sorted(all_models_seen)
        llm_suffix = "_".join(sorted_models)
    
    # Save detailed result with LLM names in filename
    detailed_output_file = output_dir / f"comparison_results_{llm_suffix}.json"
    with open(detailed_output_file, 'w') as f:
        json.dump(all_comparisons, f, indent=2)
    
    # Generate summary
    print()
    print("="*80)
    print("COMPARISON SUMMARY (Aggregated across 3 formulations)")
    print("="*80)
    
    total_comparisons = 0
    total_overlaps = 0
    total_high_accuracy = 0
    
    for exp_key, exp_data in all_comparisons.items():
        print(f"\n{exp_key}:")
        for model, metrics in exp_data['models'].items():
            overlaps = sum(1 for m in metrics.values() if m['overlap'])
            high_acc = sum(1 for m in metrics.values() if m['accuracy_score'] >= 0.7)
            
            print(f"  {model}: {len(metrics)} metrics, {overlaps} overlaps, {high_acc} high accuracy")
            
            total_comparisons += len(metrics)
            total_overlaps += overlaps
            total_high_accuracy += high_acc
    
    if total_comparisons > 0:
        print()
        print(f"Overall: {total_overlaps}/{total_comparisons} overlapping ({100*total_overlaps/total_comparisons:.1f}%)")
        print(f"High accuracy: {total_high_accuracy}/{total_comparisons} ({100*total_high_accuracy/total_comparisons:.1f}%)")
    
    print()
    print("="*80)
    print(f"Results saved to: {detailed_output_file}")


if __name__ == "__main__":
    main()
