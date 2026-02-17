#!/usr/bin/env python3
"""
Compute Evaluation Metrics for All LLMs
========================================

Day 4 Main Script: Evaluate all LLM predictions against ground truth.

Usage:
    python compute_metrics.py --ground_truth results/variance \
                              --llm_results results/llm_comparisons \
                              --output results/evaluation/

Outputs:
    - main_results.csv: Table 1 for paper
    - detailed_results.json: Full breakdown
    - coverage_by_dataset.csv: Per-dataset analysis
    - coverage_by_algorithm.csv: Per-algorithm analysis
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.evaluation.metrics import evaluate_all_llms, evaluate_llm
from src.baselines.simple_baselines import generate_baseline_predictions


def load_ground_truth(results_dir: Path) -> Dict:
    """
    Load ground truth from variance analysis results.
    
    Args:
        results_dir: Directory containing *_variance.json files
        
    Returns:
        {dataset: {algorithm: {metric: {'mean': X, 'ci_lower': Y, 'ci_upper': Z}}}}
    """
    ground_truth = {}
    
    variance_files = list(results_dir.glob('*_variance.json'))
    
    if not variance_files:
        raise FileNotFoundError(f"No variance files found in {results_dir}")
    
    print(f"Loading ground truth from {len(variance_files)} files...")
    
    for var_file in variance_files:
        with open(var_file, 'r') as f:
            data = json.load(f)
        
        dataset = data['dataset']
        algorithm = data['algorithm']
        
        if dataset not in ground_truth:
            ground_truth[dataset] = {}
        
        if algorithm not in ground_truth[dataset]:
            ground_truth[dataset][algorithm] = {}
        
        # Extract results (handle both old and new format)
        if 'results' in data:
            results = data['results']
        else:
            results = data
        
        for metric in ['precision', 'recall', 'f1', 'shd']:
            if metric in results:
                ground_truth[dataset][algorithm][metric] = {
                    'mean': results[metric].get('mean', 0.0),
                    'ci_lower': results[metric].get('ci_95_lower', 0.0),
                    'ci_upper': results[metric].get('ci_95_upper', 1.0),
                    'std': results[metric].get('std', 0.0)
                }
    
    print(f"Loaded ground truth for {len(ground_truth)} datasets")
    return ground_truth


def load_llm_predictions(llm_dir: Path, llm_name: str) -> Dict:
    """
    Load LLM predictions from comparison files.
    
    Args:
        llm_dir: Directory containing LLM comparison results
        llm_name: Name of LLM (e.g., 'gpt4', 'claude')
        
    Returns:
        {dataset: {algorithm: {metric: (lower, upper)}}}
    """
    predictions = {}
    
    # Look for files containing this LLM's predictions
    comparison_files = list(llm_dir.glob('*_llm_comparison.json'))
    
    if not comparison_files:
        print(f"Warning: No comparison files found for {llm_name}")
        return predictions
    
    for comp_file in comparison_files:
        with open(comp_file, 'r') as f:
            data = json.load(f)
        
        # Extract dataset and algorithm from filename
        # Format: {dataset}_{algorithm}_llm_comparison.json
        stem = comp_file.stem.replace('_llm_comparison', '')
        parts = stem.rsplit('_', 1)
        
        if len(parts) == 2:
            dataset, algorithm = parts
        else:
            # Try alternative format
            dataset = data.get('dataset', 'unknown')
            algorithm = data.get('algorithm', 'unknown')
        
        # Find this LLM's predictions in the file
        llm_key = f"{llm_name}_comparison"
        if llm_key not in data:
            # Try alternative keys
            llm_key = llm_name
            if llm_key not in data:
                continue
        
        if dataset not in predictions:
            predictions[dataset] = {}
        
        if algorithm not in predictions[dataset]:
            predictions[dataset][algorithm] = {}
        
        llm_data = data[llm_key]
        
        for metric in ['precision', 'recall', 'f1', 'shd']:
            if metric in llm_data:
                # Handle different formats
                if isinstance(llm_data[metric], dict) and 'llm_range' in llm_data[metric]:
                    llm_range = llm_data[metric]['llm_range']
                elif isinstance(llm_data[metric], (list, tuple)) and len(llm_data[metric]) == 2:
                    llm_range = llm_data[metric]
                else:
                    continue
                
                predictions[dataset][algorithm][metric] = tuple(llm_range)
    
    return predictions


def extract_dataset_info(ground_truth: Dict) -> Dict[str, Dict]:
    """
    Extract dataset metadata for baseline generation.
    
    Args:
        ground_truth: Ground truth results
        
    Returns:
        {dataset: {'n_samples': X, 'n_variables': Y}}
    """
    # Hardcoded for UAI submission (5 datasets)
    dataset_info = {
        'asia': {'n_samples': 10000, 'n_variables': 8},
        'sachs': {'n_samples': 1000, 'n_variables': 11},
        'cancer': {'n_samples': 5000, 'n_variables': 5},
        'synthetic_12': {'n_samples': 1000, 'n_variables': 12},
        'synthetic_30': {'n_samples': 1000, 'n_variables': 30}
    }
    
    # Filter to only datasets in ground truth
    return {k: v for k, v in dataset_info.items() if k in ground_truth}


def main():
    parser = argparse.ArgumentParser(description='Compute evaluation metrics for LLMs')
    parser.add_argument('--ground_truth', type=str, default='results/variance',
                        help='Directory with ground truth variance results')
    parser.add_argument('--llm_results', type=str, default='results/llm_comparisons',
                        help='Directory with LLM comparison results')
    parser.add_argument('--output', type=str, default='results/evaluation',
                        help='Output directory for evaluation results')
    parser.add_argument('--llms', nargs='+', 
                        default=['gpt5', 'claude', 'gemini', 'deepseek', 'llama', 'qwen'],
                        help='LLM names to evaluate')
    
    args = parser.parse_args()
    
    # Setup paths
    ground_truth_dir = Path(args.ground_truth)
    llm_dir = Path(args.llm_results)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("DAY 4: COMPUTE CALIBRATED COVERAGE & EVALUATION METRICS")
    print("="*70)
    
    # Load ground truth
    print("\n[1/5] Loading ground truth...")
    ground_truth = load_ground_truth(ground_truth_dir)
    print(f"      Loaded: {len(ground_truth)} datasets, "
          f"{sum(len(algs) for algs in ground_truth.values())} dataset-algorithm pairs")
    
    # Extract dataset info for baselines
    dataset_info = extract_dataset_info(ground_truth)
    algorithms = list(set(
        alg for dataset in ground_truth.values() 
        for alg in dataset.keys()
    ))
    
    # Generate baseline predictions
    print("\n[2/5] Generating baseline predictions...")
    baselines = generate_baseline_predictions(dataset_info, algorithms, baseline_type='both')
    print(f"      Generated: {len(baselines)} baselines (random, heuristic)")
    
    # Load LLM predictions
    print("\n[3/5] Loading LLM predictions...")
    llm_predictions = {}
    
    for llm_name in args.llms:
        preds = load_llm_predictions(llm_dir, llm_name)
        if preds:
            llm_predictions[llm_name] = preds
            n_preds = sum(
                len(algs) * len(next(iter(algs.values())))
                for algs in preds.values()
            )
            print(f"      {llm_name:12s}: {len(preds)} datasets loaded")
        else:
            print(f"      {llm_name:12s}: No predictions found (skipping)")
    
    # Combine LLMs and baselines
    all_predictions = {**llm_predictions, **baselines}
    
    # Evaluate all
    print("\n[4/5] Computing metrics...")
    results_df = evaluate_all_llms(ground_truth, all_predictions)
    
    # Save main results
    print("\n[5/5] Saving results...")
    
    # Main table (for paper)
    main_csv = output_dir / 'main_results.csv'
    results_df.to_csv(main_csv, index=False)
    print(f"      Saved: {main_csv}")
    
    # LaTeX table
    main_tex = output_dir / 'main_results.tex'
    latex_str = results_df.to_latex(index=False, float_format='%.3f')
    with open(main_tex, 'w') as f:
        f.write(latex_str)
    print(f"      Saved: {main_tex}")
    
    # Detailed per-dataset and per-algorithm breakdown
    detailed_results = {}
    
    for llm_name in all_predictions.keys():
        eval_result = evaluate_llm(llm_name, all_predictions[llm_name], ground_truth)
        detailed_results[llm_name] = {
            'overall': {
                'calibrated_coverage': eval_result.calibrated_coverage,
                'mae': eval_result.mae,
                'mean_width': eval_result.mean_width,
                'ranking_correlation': eval_result.ranking_correlation
            },
            'by_dataset': eval_result.coverage_by_dataset,
            'by_algorithm': eval_result.coverage_by_algorithm,
            'by_metric': eval_result.coverage_by_metric
        }
    
    detailed_json = output_dir / 'detailed_results.json'
    with open(detailed_json, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    print(f"      Saved: {detailed_json}")
    
    # Per-dataset table
    dataset_rows = []
    for llm_name in all_predictions.keys():
        for dataset, coverage in detailed_results[llm_name]['by_dataset'].items():
            dataset_rows.append({
                'LLM': llm_name,
                'Dataset': dataset,
                'Coverage': f"{coverage:.3f}"
            })
    
    dataset_df = pd.DataFrame(dataset_rows)
    dataset_csv = output_dir / 'coverage_by_dataset.csv'
    dataset_df.to_csv(dataset_csv, index=False)
    print(f"      Saved: {dataset_csv}")
    
    # Per-algorithm table
    algo_rows = []
    for llm_name in all_predictions.keys():
        for algorithm, coverage in detailed_results[llm_name]['by_algorithm'].items():
            algo_rows.append({
                'LLM': llm_name,
                'Algorithm': algorithm,
                'Coverage': f"{coverage:.3f}"
            })
    
    algo_df = pd.DataFrame(algo_rows)
    algo_csv = output_dir / 'coverage_by_algorithm.csv'
    algo_df.to_csv(algo_csv, index=False)
    print(f"      Saved: {algo_csv}")
    
    # Print summary to console
    print("\n" + "="*70)
    print("MAIN RESULTS (Table 1 for Paper)")
    print("="*70)
    print(results_df.to_string(index=False))
    
    # Identify best LLM
    best_row = results_df.iloc[0]
    best_llm = best_row['LLM']
    best_coverage = float(best_row['Calibrated_Coverage'])
    
    # Get random baseline for comparison
    random_row = results_df[results_df['LLM'] == 'random']
    if not random_row.empty:
        random_coverage = float(random_row.iloc[0]['Calibrated_Coverage'])
        improvement = (best_coverage - random_coverage) / random_coverage * 100
        
        print("\n" + "="*70)
        print("KEY FINDINGS")
        print("="*70)
        print(f"Best LLM: {best_llm} with {best_coverage:.1%} coverage")
        print(f"Random baseline: {random_coverage:.1%} coverage")
        print(f"Improvement: {improvement:.1f}% better than random")
        
        if best_coverage > 0.60:
            print("\n✓ LLMs demonstrate genuine algorithmic understanding (>60%)")
        elif best_coverage > 0.40:
            print("\n⚠ LLMs show moderate understanding (40-60%)")
        else:
            print("\n✗ LLMs perform poorly (<40%) - mostly pattern matching")
    
    print("\n" + "="*70)
    print("DAY 4 COMPLETE - Ready for statistical tests (Day 5)")
    print("="*70)


if __name__ == '__main__':
    main()
