#!/usr/bin/env python3
"""
Compare Baselines vs LLM Performance
====================================

Runs Random and Heuristic baselines on all datasets/algorithms,
then compares calibrated coverage against LLM predictions.

Usage:
    python compare_baselines.py --llm_dir ../llm/variance/aggregated_ranges \
                                --algorithmic_dir ../experiments/results
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import statistics

from simple_baselines import RandomBaseline, HeuristicBaseline


@dataclass
class BaselineComparison:
    """Result of comparing baseline to algorithmic ground truth."""
    dataset: str
    algorithm: str
    metric: str
    baseline_name: str
    
    # Predictions
    baseline_range: Tuple[float, float]
    algorithmic_mean: float
    algorithmic_ci: Tuple[float, float]
    
    # Calibrated coverage
    calibrated_coverage: bool
    calibrated_coverage_score: float


def load_algorithmic_metrics(algo_results_dir: Path, dataset: str, algorithm: str) -> Optional[Dict]:
    """Load algorithmic variance results."""
    variance_file = algo_results_dir / f"{dataset}_{algorithm}_variance.json"
    
    if not variance_file.exists():
        return None
    
    with open(variance_file, 'r') as f:
        data = json.load(f)
    
    if 'results' not in data:
        return None
    
    metrics_data = data['results']
    extracted_metrics = {}
    
    for metric in ['precision', 'recall', 'f1', 'shd']:
        if metric in metrics_data:
            metric_data = metrics_data[metric]
            extracted_metrics[metric] = {
                'ci': (metric_data.get('ci_95_lower'), metric_data.get('ci_95_upper')),
                'mean': metric_data.get('mean')
            }
    
    return extracted_metrics if extracted_metrics else None


def get_dataset_info(dataset_name: str) -> Tuple[int, int]:
    """Get (n_samples, n_variables) for dataset."""
    # Hardcoded from benchmark definitions
    dataset_info = {
        'alarm': (10000, 37),
        'asia': (10000, 8),
        'cancer': (5000, 5),
        'child': (5000, 20),
        'earthquake': (5000, 5),
        'hepar2': (5000, 70),
        'insurance': (5000, 27),
        'sachs': (5000, 11),
        'survey': (5000, 43),
        'synthetic_12': (1000, 12),
        'synthetic_30': (1000, 30),
        'synthetic_50': (1000, 50),
        'synthetic_60': (1000, 60),
    }
    return dataset_info.get(dataset_name, (1000, 10))  # Default fallback


def compute_calibrated_coverage_score(llm_range: Tuple[float, float], 
                                      algo_mean: float) -> float:
    """Compute continuous calibrated coverage score (0-1)."""
    llm_low, llm_high = llm_range
    llm_center = (llm_low + llm_high) / 2
    llm_half_width = (llm_high - llm_low) / 2
    
    if llm_half_width == 0:
        return 1.0 if algo_mean == llm_center else 0.0
    
    distance_from_center = abs(algo_mean - llm_center)
    if distance_from_center <= llm_half_width:
        score = 1.0 - (distance_from_center / llm_half_width) * 0.5
    else:
        distance_outside = distance_from_center - llm_half_width
        score = max(0.0, 1.0 - (distance_outside / llm_half_width))
    
    return float(round(score, 4))


def compare_baseline(baseline_range: Tuple[float, float],
                     algo_mean: float, 
                     algo_ci: Tuple[float, float]) -> BaselineComparison:
    """Create comparison result for baseline prediction."""
    score = compute_calibrated_coverage_score(baseline_range, algo_mean)
    coverage = bool(baseline_range[0] <= algo_mean <= baseline_range[1])
    
    return {
        'calibrated_coverage': coverage,
        'calibrated_coverage_score': score,
        'baseline_range': baseline_range,
        'algorithmic_mean': algo_mean,
        'algorithmic_ci': algo_ci
    }


def main():
    parser = argparse.ArgumentParser(description="Compare baselines vs LLM performance")
    parser.add_argument('--llm_dir', type=str, default='../llm/variance/aggregated_ranges',
                        help='Directory with LLM aggregated ranges')
    parser.add_argument('--algorithmic_dir', type=str, default='../experiments/results',
                        help='Directory with algorithmic variance results')
    parser.add_argument('--output_file', type=str, default='baseline_comparison_results.json',
                        help='Output file for comparison results')
    
    args = parser.parse_args()
    
    llm_dir = Path(args.llm_dir)
    algo_dir = Path(args.algorithmic_dir)
    output_file = Path(args.output_file)
    
    print("="*80)
    print("COMPARING BASELINES VS LLM")
    print("="*80)
    print(f"LLM aggregates:      {llm_dir}")
    print(f"Algorithmic results: {algo_dir}")
    print()
    
    # Initialize baselines
    random_baseline = RandomBaseline(seed=42)
    heuristic_baseline = HeuristicBaseline()
    
    # Load LLM results for reference
    llm_files = sorted(llm_dir.glob("*_aggregated.json"))
    llm_data_cache = {}
    
    for llm_file in llm_files:
        stem = llm_file.stem
        if not stem.endswith('_aggregated'):
            continue
        stem_without_suffix = stem[:-len('_aggregated')]
        parts = stem_without_suffix.split('_')
        algorithm = parts[-1]
        dataset = '_'.join(parts[:-1])
        
        with open(llm_file, 'r') as f:
            data = json.load(f)
        
        exp_key = f"{dataset}_{algorithm}"
        llm_data_cache[exp_key] = data['llm_estimates']
    
    # Compare baselines on all experiments
    all_baselines = {
        'random': {},
        'heuristic': {},
        'llm': {}
    }
    
    experiment_count = 0
    
    for dataset_algo, llm_models in sorted(llm_data_cache.items()):
        parts = dataset_algo.rsplit('_', 1)
        if len(parts) != 2:
            continue
        
        dataset, algorithm = parts
        
        # Load ground truth
        algo_metrics = load_algorithmic_metrics(algo_dir, dataset, algorithm)
        if not algo_metrics:
            continue
        
        experiment_count += 1
        
        # Get dataset characteristics
        n_samples, n_variables = get_dataset_info(dataset)
        
        # Generate baseline predictions
        random_pred = random_baseline.predict(dataset, algorithm, n_samples, n_variables)
        heuristic_pred = heuristic_baseline.predict(dataset, algorithm, n_samples, n_variables)
        
        # Compare each metric
        for metric in ['precision', 'recall', 'f1', 'shd']:
            if metric not in algo_metrics:
                continue
            
            algo_data = algo_metrics[metric]
            algo_mean = algo_data['mean']
            algo_ci = algo_data['ci']
            
            # Random baseline
            if metric in random_pred:
                result = compare_baseline(random_pred[metric], algo_mean, algo_ci)
                key = f"{dataset_algo}_{metric}"
                all_baselines['random'][key] = result
            
            # Heuristic baseline
            if metric in heuristic_pred:
                result = compare_baseline(heuristic_pred[metric], algo_mean, algo_ci)
                key = f"{dataset_algo}_{metric}"
                all_baselines['heuristic'][key] = result
            
            # LLM (extract from cached data)
            if metric in llm_models and 'claude' in llm_models:
                # Use Claude as representative LLM
                claude_pred = llm_models['claude']
                if metric in claude_pred:
                    llm_metric = claude_pred[metric]
                    result = compare_baseline(
                        (llm_metric['lower'], llm_metric['upper']),
                        algo_mean,
                        algo_ci
                    )
                    key = f"{dataset_algo}_{metric}"
                    all_baselines['llm'][key] = result
    
    # Compute statistics
    print(f"Tested {experiment_count} experiments\n")
    print("="*80)
    print("CALIBRATED COVERAGE COMPARISON")
    print("="*80)
    
    results_summary = {}
    
    for baseline_type in ['random', 'heuristic', 'llm']:
        results = all_baselines[baseline_type]
        if not results:
            continue
        
        coverage_list = [1 if r['calibrated_coverage'] else 0 for r in results.values()]
        score_list = [r['calibrated_coverage_score'] for r in results.values()]
        
        coverage_pct = 100 * sum(coverage_list) / len(coverage_list)
        mean_score = sum(score_list) / len(score_list)
        
        results_summary[baseline_type] = {
            'count': len(results),
            'calibrated_coverage_pct': coverage_pct,
            'mean_coverage_score': mean_score,
            'median_score': statistics.median(score_list),
            'perfect_0_8_1_0': sum(1 for x in score_list if 0.8 <= x <= 1.0),
            'terrible_0_0_0_2': sum(1 for x in score_list if 0.0 <= x <= 0.2)
        }
        
        print(f"\n{baseline_type.upper()}:")
        print(f"  Calibrated Coverage: {coverage_pct:.1f}% ({sum(coverage_list)}/{len(coverage_list)})")
        print(f"  Mean Coverage Score: {mean_score:.3f}")
        print(f"  Median Score:        {statistics.median(score_list):.3f}")
        print(f"  Perfect (0.8-1.0):   {results_summary[baseline_type]['perfect_0_8_1_0']} ({100*results_summary[baseline_type]['perfect_0_8_1_0']/len(coverage_list):.1f}%)")
        print(f"  Terrible (0.0-0.2):  {results_summary[baseline_type]['terrible_0_0_0_2']} ({100*results_summary[baseline_type]['terrible_0_0_0_2']/len(coverage_list):.1f}%)")
    
    # Comparison table
    print("\n" + "="*80)
    print("RANKING")
    print("="*80)
    
    sorted_results = sorted(results_summary.items(), 
                           key=lambda x: x[1]['calibrated_coverage_pct'], 
                           reverse=True)
    
    for i, (name, stats) in enumerate(sorted_results, 1):
        print(f"{i}. {name.upper():12s}: {stats['calibrated_coverage_pct']:5.1f}% coverage | Score {stats['mean_coverage_score']:.3f}")
    
    # Save detailed results
    output_data = {
        'summary': results_summary,
        'details': {
            'random': all_baselines['random'],
            'heuristic': all_baselines['heuristic'],
            'llm_claude': all_baselines['llm']
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    
    print(f"\nDetailed results saved to: {output_file}")
    print("="*80)


if __name__ == "__main__":
    main()
