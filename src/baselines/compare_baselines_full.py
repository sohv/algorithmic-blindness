#!/usr/bin/env python3
"""
Complete Baseline Comparison (Full LLM Dataset)
================================================

Compares Random and Heuristic baselines against ALL 1,632 LLM metrics,
not just a subset. This ensures fair comparison across all models/datasets.

Usage:
    python compare_baselines_full.py \
        --llm_results ../llm/variance/comparisons/comparison_results_all.json \
        --algorithmic_dir ../experiments/results
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional
from collections import defaultdict
import statistics

from simple_baselines import RandomBaseline, HeuristicBaseline


def get_dataset_info(dataset_name: str) -> Tuple[int, int]:
    """Get (n_samples, n_variables) for dataset."""
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
    return dataset_info.get(dataset_name, (1000, 10))


def compute_calibrated_coverage_score(baseline_range: Tuple[float, float], 
                                      algo_mean: float) -> float:
    """Compute continuous calibrated coverage score (0-1)."""
    low, high = baseline_range
    center = (low + high) / 2
    half_width = (high - low) / 2
    
    if half_width == 0:
        return 1.0 if algo_mean == center else 0.0
    
    distance_from_center = abs(algo_mean - center)
    if distance_from_center <= half_width:
        score = 1.0 - (distance_from_center / half_width) * 0.5
    else:
        distance_outside = distance_from_center - half_width
        score = max(0.0, 1.0 - (distance_outside / half_width))
    
    return float(round(score, 4))


def main():
    parser = argparse.ArgumentParser(
        description="Complete baseline comparison on full LLM dataset (1,632 metrics)")
    parser.add_argument('--llm_results', type=str, 
                        default='../llm/variance/comparisons/comparison_results_all.json',
                        help='Full LLM comparison results file')
    parser.add_argument('--algorithmic_dir', type=str, 
                        default='../experiments/results',
                        help='Directory with algorithmic variance results')
    parser.add_argument('--output_file', type=str, 
                        default='baseline_comparison_full_results.json',
                        help='Output file for complete comparison')
    
    args = parser.parse_args()
    
    llm_results_file = Path(args.llm_results)
    algo_dir = Path(args.algorithmic_dir)
    output_file = Path(args.output_file)
    
    print("="*80)
    print("COMPLETE BASELINE COMPARISON (Full 1,632 LLM Metrics)")
    print("="*80)
    print(f"LLM results: {llm_results_file}")
    print(f"Algorithmic: {algo_dir}")
    print()
    
    # Load full LLM comparison results
    with open(llm_results_file, 'r') as f:
        llm_full = json.load(f)
    
    print(f"Loaded {len(llm_full)} experiments from LLM results")
    
    # Initialize baselines
    random_baseline = RandomBaseline(seed=42)
    heuristic_baseline = HeuristicBaseline()
    
    # Collect comparisons
    all_comparisons = {
        'random': [],
        'heuristic': [],
        'llm_models': {}
    }
    
    metric_count = 0
    
    # Iterate through all LLM experiments and all models
    for exp_key, exp_data in sorted(llm_full.items()):
        dataset = exp_data['dataset']
        algorithm = exp_data['algorithm']
        
        # Get dataset characteristics
        n_samples, n_variables = get_dataset_info(dataset)
        
        # Generate baseline predictions once per (dataset, algorithm) pair
        random_pred = random_baseline.predict(dataset, algorithm, n_samples, n_variables)
        heuristic_pred = heuristic_baseline.predict(dataset, algorithm, n_samples, n_variables)
        
        # Process each model's predictions
        for model_name, model_metrics in exp_data['models'].items():
            if model_name not in all_comparisons['llm_models']:
                all_comparisons['llm_models'][model_name] = {
                    'coverage_scores': [],
                    'binary_coverage': [],
                    'count': 0
                }
            
            for metric_name, metric_data in model_metrics.items():
                metric_count += 1
                
                algo_mean = metric_data['algorithmic_mean']
                algo_ci = metric_data['algorithmic_ci']
                llm_range = metric_data['llm_range']
                
                # Random baseline
                if metric_name in random_pred:
                    baseline_range = random_pred[metric_name]
                    score = compute_calibrated_coverage_score(baseline_range, algo_mean)
                    coverage = bool(baseline_range[0] <= algo_mean <= baseline_range[1])
                    
                    all_comparisons['random'].append({
                        'exp': exp_key,
                        'metric': metric_name,
                        'score': score,
                        'coverage': coverage,
                        'baseline_range': baseline_range,
                        'algo_mean': algo_mean
                    })
                
                # Heuristic baseline
                if metric_name in heuristic_pred:
                    baseline_range = heuristic_pred[metric_name]
                    score = compute_calibrated_coverage_score(baseline_range, algo_mean)
                    coverage = bool(baseline_range[0] <= algo_mean <= baseline_range[1])
                    
                    all_comparisons['heuristic'].append({
                        'exp': exp_key,
                        'metric': metric_name,
                        'score': score,
                        'coverage': coverage,
                        'baseline_range': baseline_range,
                        'algo_mean': algo_mean
                    })
                
                # Track LLM model
                all_comparisons['llm_models'][model_name]['coverage_scores'].append(
                    metric_data['calibrated_coverage_score']
                )
                all_comparisons['llm_models'][model_name]['binary_coverage'].append(
                    metric_data['calibrated_coverage']
                )
                all_comparisons['llm_models'][model_name]['count'] += 1
    
    print(f"Processed {metric_count} total metrics\n")
    
    # Compute statistics
    print("="*80)
    print("RESULTS: Full Dataset Calibrated Coverage Comparison")
    print("="*80)
    
    summary = {}
    
    # Random baseline
    random_coverage = [c['coverage'] for c in all_comparisons['random']]
    random_scores = [c['score'] for c in all_comparisons['random']]
    summary['random'] = {
        'count': len(random_coverage),
        'calibrated_coverage_pct': 100 * sum(random_coverage) / len(random_coverage),
        'mean_score': sum(random_scores) / len(random_scores),
        'median_score': statistics.median(random_scores),
        'perfect_0_8_1_0': sum(1 for x in random_scores if 0.8 <= x <= 1.0),
        'terrible_0_0_0_2': sum(1 for x in random_scores if 0.0 <= x <= 0.2)
    }
    
    # Heuristic baseline
    heuristic_coverage = [c['coverage'] for c in all_comparisons['heuristic']]
    heuristic_scores = [c['score'] for c in all_comparisons['heuristic']]
    summary['heuristic'] = {
        'count': len(heuristic_coverage),
        'calibrated_coverage_pct': 100 * sum(heuristic_coverage) / len(heuristic_coverage),
        'mean_score': sum(heuristic_scores) / len(heuristic_scores),
        'median_score': statistics.median(heuristic_scores),
        'perfect_0_8_1_0': sum(1 for x in heuristic_scores if 0.8 <= x <= 1.0),
        'terrible_0_0_0_2': sum(1 for x in heuristic_scores if 0.0 <= x <= 0.2)
    }
    
    # LLM models
    summary['llm'] = {}
    for model_name, model_data in sorted(all_comparisons['llm_models'].items()):
        scores = model_data['coverage_scores']
        binary = model_data['binary_coverage']
        summary['llm'][model_name] = {
            'count': model_data['count'],
            'calibrated_coverage_pct': 100 * sum(binary) / len(binary),
            'mean_score': sum(scores) / len(scores),
            'median_score': statistics.median(scores),
            'perfect_0_8_1_0': sum(1 for x in scores if 0.8 <= x <= 1.0),
            'terrible_0_0_0_2': sum(1 for x in scores if 0.0 <= x <= 0.2)
        }
    
    # Print results
    print("\nBASELINES:")
    for baseline_type in ['random', 'heuristic']:
        stats = summary[baseline_type]
        print(f"\n{baseline_type.upper()}:")
        print(f"  Metrics tested: {stats['count']}")
        print(f"  Calibrated Coverage: {stats['calibrated_coverage_pct']:.1f}%")
        print(f"  Mean Coverage Score: {stats['mean_score']:.3f}")
        print(f"  Median Score: {stats['median_score']:.3f}")
        print(f"  Perfect (0.8-1.0): {stats['perfect_0_8_1_0']} ({100*stats['perfect_0_8_1_0']/stats['count']:.1f}%)")
        print(f"  Terrible (0.0-0.2): {stats['terrible_0_0_0_2']} ({100*stats['terrible_0_0_0_2']/stats['count']:.1f}%)")
    
    print("\n" + "="*80)
    print("LLM MODELS (for comparison):")
    print("="*80)
    
    # Sort by coverage
    sorted_llms = sorted(summary['llm'].items(), 
                        key=lambda x: x[1]['calibrated_coverage_pct'], 
                        reverse=True)
    
    print(f"\n{'Model':<15s} {'Coverage%':>10s} {'Mean Score':>12s} {'Perfect':>10s} {'Terrible':>10s}")
    print("-" * 70)
    
    for model_name, stats in sorted_llms:
        print(f"{model_name:<15s} {stats['calibrated_coverage_pct']:>9.1f}% {stats['mean_score']:>12.3f} "
              f"{stats['perfect_0_8_1_0']:>9d} {stats['terrible_0_0_0_2']:>10d}")
    
    # Three-way comparison for best/worst LLMs
    print("\n" + "="*80)
    print("THREE-WAY COMPARISON: Best/Worst LLM vs Baselines")
    print("="*80)
    
    best_llm_name = sorted_llms[0][0]
    worst_llm_name = sorted_llms[-1][0]
    
    print(f"\n{'Method':<20s} {'Coverage%':>10s} {'Mean Score':>12s}")
    print("-" * 50)
    print(f"{'Random Baseline':<20s} {summary['random']['calibrated_coverage_pct']:>9.1f}% "
          f"{summary['random']['mean_score']:>12.3f}")
    print(f"{'Heuristic Baseline':<20s} {summary['heuristic']['calibrated_coverage_pct']:>9.1f}% "
          f"{summary['heuristic']['mean_score']:>12.3f}")
    print(f"{best_llm_name + ' (best)':<20s} {summary['llm'][best_llm_name]['calibrated_coverage_pct']:>9.1f}% "
          f"{summary['llm'][best_llm_name]['mean_score']:>12.3f}")
    print(f"{worst_llm_name + ' (worst)':<20s} {summary['llm'][worst_llm_name]['calibrated_coverage_pct']:>9.1f}% "
          f"{summary['llm'][worst_llm_name]['mean_score']:>12.3f}")
    
    # Key finding
    print("\n" + "="*80)
    print("KEY FINDING")
    print("="*80)
    
    claude_cov = summary['llm'].get('claude', {}).get('calibrated_coverage_pct', 0)
    random_cov = summary['random']['calibrated_coverage_pct']
    
    if claude_cov >= random_cov - 0.5:  # Within rounding error
        print(f"\n⚠️  Best LLM ({best_llm_name}): {summary['llm'][best_llm_name]['calibrated_coverage_pct']:.1f}%")
        print(f"🎲 Random Baseline: {random_cov:.1f}%")
        print(f"\n→ LLMs perform AT MOST ON PAR with random guessing")
        print(f"→ No LLM meaningfully beats baseline approaches")
    else:
        print(f"\n✓ Best LLM ({best_llm_name}): {summary['llm'][best_llm_name]['calibrated_coverage_pct']:.1f}%")
        print(f"🎲 Random Baseline: {random_cov:.1f}%")
        print(f"→ LLMs beat random by {summary['llm'][best_llm_name]['calibrated_coverage_pct'] - random_cov:.1f}pp")
    
    # Save results
    output_data = {
        'metadata': {
            'dataset_size': metric_count,
            'experiments': len(llm_full),
            'models_evaluated': len(all_comparisons['llm_models'])
        },
        'summary': summary,
        'details': {
            'random': all_comparisons['random'],
            'heuristic': all_comparisons['heuristic'],
            'llm_models': all_comparisons['llm_models']
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    
    print(f"\nFull results saved to: {output_file}")
    print("="*80)


if __name__ == "__main__":
    main()
