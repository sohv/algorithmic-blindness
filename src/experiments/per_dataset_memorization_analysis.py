#!/usr/bin/env python3
"""
Per-Dataset Memorization Analysis

Tests which specific benchmark datasets are memorized by which LLMs.
This reveals:
- Which dataset names trigger memorization (e.g., 'asia' is memorized)
- Which models memorize which datasets
- Strength of memorization signal per dataset

Null Hypothesis: Dataset name doesn't affect model behavior
Alternative: Models respond differently to known vs unknown dataset names
"""

import json
import os
import sys
from typing import Dict, List, Tuple
from scipy import stats
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def load_perturbation_results(model_name: str, results_dir: str) -> Dict:
    """Load perturbation test results for a model."""
    filepath = os.path.join(results_dir, f'perturbation_{model_name}.json')
    if not os.path.exists(filepath):
        return None
    
    with open(filepath, 'r') as f:
        return json.load(f)


def extract_per_dataset_precision(data: Dict) -> Dict[str, List[Tuple[float, float]]]:
    """
    Extract precision pairs organized by dataset.
    Returns: {dataset: [(original_precision, perturbed_precision), ...]}
    """
    dataset_pairs = defaultdict(list)
    
    for test in data:
        dataset = test.get('dataset', 'unknown')
        original_nums = test.get('original_extracted_numbers', [])
        perturbed_nums = test.get('perturbed_extracted_numbers', [])
        
        if original_nums and perturbed_nums:
            original_precision = original_nums[0]
            perturbed_precision = perturbed_nums[0]
            
            if 0 <= original_precision <= 1 and 0 <= perturbed_precision <= 1:
                dataset_pairs[dataset].append((original_precision, perturbed_precision))
    
    return dict(dataset_pairs)


def test_dataset_memorization(original: List[float], perturbed: List[float]) -> Dict:
    """Test if a specific dataset shows memorization."""
    if len(original) < 1:
        return {'n': 0, 'error': 'Insufficient data'}
    
    if len(original) == 1:
        # Single sample: just report the difference
        diff = original[0] - perturbed[0]
        return {
            'n': 1,
            'mean_original': original[0],
            'mean_perturbed': perturbed[0],
            'mean_difference': diff,
            'p_value': None,
            'cohens_d': None,
            'significant': diff > 0.1  # Heuristic for single sample
        }
    
    # Multiple samples: t-test
    differences = [o - p for o, p in zip(original, perturbed)]
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    
    t_stat, p_value_two_tailed = stats.ttest_rel(original, perturbed)
    p_value_one_tailed = p_value_two_tailed / 2 if t_stat > 0 else 1 - (p_value_two_tailed / 2)
    
    cohens_d = mean_diff / std_diff if std_diff > 0 else 0
    
    return {
        'n': len(original),
        'mean_original': float(np.mean(original)),
        'mean_perturbed': float(np.mean(perturbed)),
        'mean_difference': float(mean_diff),
        'std_difference': float(std_diff),
        't_statistic': float(t_stat),
        'p_value': float(p_value_one_tailed),
        'cohens_d': float(cohens_d),
        'significant_at_0_05': p_value_one_tailed < 0.05,
        'memorization_strength': classify_memorization(mean_diff, cohens_d)
    }


def classify_memorization(mean_diff: float, cohens_d: float) -> str:
    """Classify strength of memorization signal."""
    if mean_diff < 0.05:
        return 'none'
    elif mean_diff < 0.1 or abs(cohens_d) < 0.3:
        return 'weak'
    elif mean_diff < 0.2 or abs(cohens_d) < 0.6:
        return 'moderate'
    else:
        return 'strong'


def main():
    """Run per-dataset memorization analysis."""
    
    results_dir = '/home/ece/hdd/sohan/algorithmic-blindness/src/experiments/results/memorization'
    models = ['gpt5', 'deepseek', 'deepseekthink', 'claude', 'gemini3', 'llama', 'qwen', 'qwenthink']
    datasets = ['asia', 'cancer', 'child', 'earthquake', 'alarm', 'sachs']
    
    print("\n" + "=" * 120)
    print("PER-DATASET MEMORIZATION ANALYSIS")
    print("=" * 120)
    print("\nTests which specific benchmark datasets are memorized by which models\n")
    
    # Store all results for heatmap generation
    all_results = {}
    heatmap_data = {}  # For visualization
    
    for model_name in models:
        print(f"\n{'=' * 120}")
        print(f"MODEL: {model_name.upper()}")
        print(f"{'=' * 120}")
        
        data = load_perturbation_results(model_name, results_dir)
        if not data:
            print(f"⚠ No data found for {model_name}")
            continue
        
        dataset_precision_pairs = extract_per_dataset_precision(data)
        model_results = {}
        heatmap_data[model_name] = {}
        
        # Test each dataset
        for dataset in datasets:
            if dataset not in dataset_precision_pairs:
                print(f"\n  {dataset:12} - NO DATA")
                model_results[dataset] = {'error': 'No data'}
                heatmap_data[model_name][dataset] = None
                continue
            
            pairs = dataset_precision_pairs[dataset]
            original_vals = [p[0] for p in pairs]
            perturbed_vals = [p[1] for p in pairs]
            
            result = test_dataset_memorization(original_vals, perturbed_vals)
            model_results[dataset] = result
            
            # Store for heatmap (mean difference)
            heatmap_data[model_name][dataset] = result.get('mean_difference', 0)
            
            # Print results
            sig_marker = "✓ MEMORIZED" if result.get('significant_at_0_05') else "○ no signal"
            strength = result.get('memorization_strength', 'none')
            
            print(f"\n  {dataset:12} | n={result['n']:<2} | " +
                  f"orig={result['mean_original']:.3f} → pert={result['mean_perturbed']:.3f} | " +
                  f"Δ={result['mean_difference']:+.4f} | " +
                  f"d={result.get('cohens_d', 0):.3f} | " +
                  f"{strength:8} | {sig_marker}")
            
            if result.get('p_value') is not None:
                print(f"           p={result['p_value']:.6f}")
        
        all_results[model_name] = model_results
    
    # Generate heatmap
    print("\n" + "=" * 120)
    print("HEATMAP: Mean Precision Difference (Original - Perturbed) by Model × Dataset")
    print("=" * 120)
    print("\nHigher values = stronger memorization signal")
    print("Positive = model drops precision on unknown names (memorization)\n")
    
    # Print header
    print(f"{'Model':<15}", end='')
    for dataset in datasets:
        print(f"{dataset:>12}", end='')
    print()
    print("-" * 15 + "-" * (12 * len(datasets)))
    
    # Print rows
    for model_name in models:
        print(f"{model_name:<15}", end='')
        for dataset in datasets:
            if model_name in heatmap_data and dataset in heatmap_data[model_name]:
                val = heatmap_data[model_name][dataset]
                if val is not None:
                    # Color coding: strong (>0.15), moderate (>0.05), weak (>0), none
                    if val > 0.15:
                        marker = "████"
                    elif val > 0.05:
                        marker = "██  "
                    elif val > 0:
                        marker = "█   "
                    else:
                        marker = "    "
                    print(f"{val:>8.4f} {marker}", end='')
                else:
                    print(f"{'N/A':>13}", end='')
            else:
                print(f"{'?':>13}", end='')
        print()
    
    # Summary statistics
    print("\n" + "=" * 120)
    print("MEMORIZATION SUMMARY")
    print("=" * 120)
    
    memorization_count = defaultdict(lambda: defaultdict(int))
    
    for model_name in models:
        strong_count = 0
        moderate_count = 0
        weak_count = 0
        
        if model_name in all_results:
            for dataset, result in all_results[model_name].items():
                strength = result.get('memorization_strength', 'none')
                if strength == 'strong':
                    strong_count += 1
                    memorization_count[model_name]['strong'] += 1
                elif strength == 'moderate':
                    moderate_count += 1
                    memorization_count[model_name]['moderate'] += 1
                elif strength == 'weak':
                    weak_count += 1
                    memorization_count[model_name]['weak'] += 1
        
        print(f"\n{model_name:15} → Strong: {strong_count}/6  |  Moderate: {moderate_count}/6  |  Weak: {weak_count}/6")
    
    # Save detailed results
    output_file = os.path.join(results_dir, 'per_dataset_memorization_analysis.json')
    
    # Convert to JSON-serializable format
    json_results = {}
    for model, datasets_result in all_results.items():
        json_results[model] = {}
        for dataset, result in datasets_result.items():
            json_results[model][dataset] = {}
            for k, v in result.items():
                if isinstance(v, (np.floating, np.integer)):
                    json_results[model][dataset][k] = float(v)
                elif isinstance(v, bool):
                    json_results[model][dataset][k] = int(v)
                elif isinstance(v, np.bool_):
                    json_results[model][dataset][k] = int(v)
                else:
                    json_results[model][dataset][k] = v
    
    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\n{'=' * 120}")
    print(f"Detailed results saved to: {output_file}")
    print(f"{'=' * 120}\n")


if __name__ == '__main__':
    main()
