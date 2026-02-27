#!/usr/bin/env python3
"""
Statistical Testing for LLM Memorization Detection

Tests whether precision drops from original→perturbed datasets are statistically significant.
This indicates memorization: models that memorize dataset names will show larger drops
when those names are changed to unknown names.

Null Hypothesis (H0): Original and perturbed precision values come from the same distribution
Alternative Hypothesis (H1): Perturbed values are lower (one-tailed test)
"""

import json
import os
import sys
import statistics
from typing import Dict, List, Tuple
from scipy import stats
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def load_perturbation_results(model_name: str, results_dir: str) -> Dict:
    """Load perturbation test results for a model."""
    filepath = os.path.join(results_dir, f'perturbation_{model_name}.json')
    if not os.path.exists(filepath):
        print(f"⚠ File not found: {filepath}")
        return None
    
    with open(filepath, 'r') as f:
        return json.load(f)


def extract_precision_pairs(data: Dict) -> List[Tuple[float, float]]:
    """
    Extract (original_precision, perturbed_precision) pairs from results.
    
    Precision is typically the first extracted number (index 0).
    """
    pairs = []
    
    for test in data:
        original_nums = test.get('original_extracted_numbers', [])
        perturbed_nums = test.get('perturbed_extracted_numbers', [])
        
        # Extract first number (precision) from each
        if original_nums and perturbed_nums:
            original_precision = original_nums[0]
            perturbed_precision = perturbed_nums[0]
            
            # Only include valid pairs (0-1 range)
            if 0 <= original_precision <= 1 and 0 <= perturbed_precision <= 1:
                pairs.append((original_precision, perturbed_precision))
    
    return pairs


def perform_paired_ttest(original: List[float], perturbed: List[float]) -> Dict:
    """
    Perform paired t-test comparing original vs perturbed precision.
    One-tailed: tests if perturbed < original (memorization signal)
    """
    if len(original) < 2 or len(perturbed) < 2:
        return {'error': 'Insufficient data for testing'}
    
    # Calculate differences
    differences = [o - p for o, p in zip(original, perturbed)]
    mean_diff = statistics.mean(differences)
    stdev_diff = statistics.stdev(differences) if len(differences) > 1 else 0
    
    # Paired t-test (one-tailed: alternative='greater' means original > perturbed)
    t_statistic, p_value_two_tailed = stats.ttest_rel(original, perturbed)
    p_value_one_tailed = p_value_two_tailed / 2 if t_statistic > 0 else 1 - (p_value_two_tailed / 2)
    
    # Calculate Cohen's d (effect size)
    cohens_d = mean_diff / stdev_diff if stdev_diff > 0 else 0
    
    # 95% Confidence interval for mean difference
    n = len(differences)
    t_critical = stats.t.ppf(0.975, n - 1)  # 95% CI
    se_diff = stdev_diff / np.sqrt(n) if stdev_diff > 0 else 0
    ci_lower = mean_diff - t_critical * se_diff
    ci_upper = mean_diff + t_critical * se_diff
    
    return {
        'n': n,
        'mean_original': statistics.mean(original),
        'mean_perturbed': statistics.mean(perturbed),
        'mean_difference': mean_diff,
        'std_difference': stdev_diff,
        't_statistic': t_statistic,
        'p_value_one_tailed': p_value_one_tailed,
        'p_value_two_tailed': p_value_two_tailed,
        'cohens_d': cohens_d,
        'ci_95': [ci_lower, ci_upper],
        'significant_at_0_05': p_value_one_tailed < 0.05,
        'significant_at_0_01': p_value_one_tailed < 0.01,
        'interpretation': interpret_result(p_value_one_tailed, mean_diff, cohens_d)
    }


def interpret_result(p_value: float, mean_diff: float, cohens_d: float) -> str:
    """Provide interpretation of statistical test results."""
    sig_level = "***" if p_value < 0.01 else "**" if p_value < 0.05 else "*" if p_value < 0.10 else "ns"
    
    effect_size = "negligible" if abs(cohens_d) < 0.2 else "small" if abs(cohens_d) < 0.5 else "medium" if abs(cohens_d) < 0.8 else "large"
    
    if p_value < 0.05 and mean_diff > 0:
        return f"{sig_level} | Strong memorization signal (effect size: {effect_size}, d={cohens_d:.3f})"
    elif p_value < 0.10 and mean_diff > 0:
        return f"{sig_level} | Weak memorization signal (effect size: {effect_size}, d={cohens_d:.3f})"
    elif p_value >= 0.10:
        return f"{sig_level} | No significant memorization signal detected"
    else:
        return f"{sig_level} | Unexpected: perturbed > original (anti-memorization?)"


def main():
    """Run statistical analysis on all perturbation test results."""
    
    results_dir = '/home/ece/hdd/sohan/algorithmic-blindness/src/experiments/results/memorization'
    models = ['gpt5', 'deepseek', 'deepseekthink', 'claude', 'gemini3', 'llama', 'qwen', 'qwenthink']
    
    all_results = {}
    
    print("\n" + "=" * 100)
    print("STATISTICAL TESTING: LLM Memorization Detection via Perturbation")
    print("=" * 100)
    print("\nNull Hypothesis (H0): Precision is invariant to dataset name changes")
    print("Alt Hypothesis (H1): Precision drops when unknown dataset names are used (memorization)\n")
    
    for model_name in models:
        print(f"\n{'=' * 100}")
        print(f"Testing: {model_name.upper()}")
        print(f"{'=' * 100}")
        
        data = load_perturbation_results(model_name, results_dir)
        if not data:
            print(f"⚠ Skipping {model_name} - no data found")
            continue
        
        pairs = extract_precision_pairs(data)
        
        if not pairs:
            print(f"⚠ No valid precision pairs found for {model_name}")
            continue
        
        original_vals = [p[0] for p in pairs]
        perturbed_vals = [p[1] for p in pairs]
        
        # Perform statistical test
        test_result = perform_paired_ttest(original_vals, perturbed_vals)
        
        all_results[model_name] = {
            'pairs_analyzed': len(pairs),
            **test_result,
            'raw_pairs': pairs
        }
        
        # Print results
        print(f"Sample size: {test_result['n']} test cases")
        print(f"Mean original precision: {test_result['mean_original']:.4f}")
        print(f"Mean perturbed precision: {test_result['mean_perturbed']:.4f}")
        print(f"Mean difference: {test_result['mean_difference']:.4f} (95% CI: [{test_result['ci_95'][0]:.4f}, {test_result['ci_95'][1]:.4f}])")
        print(f"T-statistic: {test_result['t_statistic']:.4f}")
        print(f"P-value (one-tailed): {test_result['p_value_one_tailed']:.6f}")
        print(f"Cohen's d: {test_result['cohens_d']:.4f}")
        print(f"\nSignificance: {test_result['interpretation']}")
        
        if test_result['significant_at_0_05']:
            print("✓ SIGNIFICANT at α=0.05 → Evidence of memorization")
        elif test_result['significant_at_0_01']:
            print("✓ SIGNIFICANT at α=0.01 → Strong evidence of memorization")
        else:
            print("✗ NOT SIGNIFICANT → No evidence of memorization")
    
    # Save full results
    output_file = os.path.join(results_dir, 'statistical_analysis.json')
    
    # Create summary for JSON
    summary = {}
    for model, result in all_results.items():
        if 'error' not in result:
            summary[model] = {
                'sample_size': result.get('n', 0),
                'mean_difference': float(result.get('mean_difference', 0)),
                'p_value': float(result.get('p_value_one_tailed', 1)),
                'cohens_d': float(result.get('cohens_d', 0)),
                'significant': int(result.get('significant_at_0_05', False)),
                'interpretation': result.get('interpretation', '')
            }
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'=' * 100}")
    print(f"Full results saved to: {output_file}")
    print(f"{'=' * 100}\n")
    
    # Summary table
    print("\nSUMMARY TABLE:")
    print("-" * 100)
    cohens_label = "Cohen's d"
    print(f"{'Model':<20} {'N':<5} {'Mean Diff':<12} {'P-value':<12} {cohens_label:<12} {'Significant?':<15}")
    print("-" * 100)
    
    for model in models:
        if model in all_results and 'error' not in all_results[model]:
            result = all_results[model]
            sig = "YES ✓" if result['significant_at_0_05'] else "NO ✗"
            print(f"{model:<20} {result['n']:<5} {result['mean_difference']:>10.4f}  {result['p_value_one_tailed']:>10.6f}  {result['cohens_d']:>10.4f}  {sig:<15}")
    
    print("-" * 100)


if __name__ == '__main__':
    main()
