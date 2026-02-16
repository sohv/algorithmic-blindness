#!/usr/bin/env python3
"""
Statistical Significance Testing (Day 5)
=========================================

Tests whether LLM performance is statistically significant compared to baselines.

Statistical Tests:
1. Wilcoxon signed-rank test: LLM vs Random (paired, non-parametric)
2. False Discovery Rate (FDR) correction for multiple comparisons
3. Effect size computation (Cohen's d)

Usage:
    python statistical_tests.py --input results/evaluation \
                                 --output results/statistics
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon, ttest_rel
from statsmodels.stats.multitest import multipletests
from typing import Dict, List, Tuple


def load_evaluation_results(input_dir: Path) -> Dict:
    """Load evaluation results from Day 4."""
    detailed_file = input_dir / 'detailed_results.json'
    
    if not detailed_file.exists():
        raise FileNotFoundError(f"{detailed_file} not found. Run compute_metrics.py first.")
    
    with open(detailed_file, 'r') as f:
        return json.load(f)


def extract_paired_scores(detailed_results: Dict, 
                          llm_name: str,
                          baseline_name: str = 'random') -> Tuple[List[float], List[float]]:
    """
    Extract paired coverage scores for statistical testing.
    
    For each dataset-algorithm pair, get coverage from LLM and baseline.
    
    Args:
        detailed_results: Detailed results dict
        llm_name: Name of LLM to test
        baseline_name: Name of baseline (default: 'random')
        
    Returns:
        (llm_scores, baseline_scores) as paired lists
    """
    llm_scores = []
    baseline_scores = []
    
    # Get all dataset-algorithm pairs
    llm_by_dataset = detailed_results[llm_name]['by_dataset']
    baseline_by_dataset = detailed_results[baseline_name]['by_dataset']
    
    # Match by dataset
    for dataset in llm_by_dataset.keys():
        if dataset in baseline_by_dataset:
            llm_scores.append(llm_by_dataset[dataset])
            baseline_scores.append(baseline_by_dataset[dataset])
    
    return llm_scores, baseline_scores


def wilcoxon_test(llm_scores: List[float], 
                  baseline_scores: List[float],
                  alternative: str = 'greater') -> Tuple[float, float]:
    """
    Wilcoxon signed-rank test (non-parametric paired test).
    
    H0: LLM and baseline have same median performance
    H1: LLM has higher median performance
    
    Args:
        llm_scores: LLM coverage scores
        baseline_scores: Baseline coverage scores
        alternative: 'greater' (LLM > baseline), 'two-sided', 'less'
        
    Returns:
        (test_statistic, p_value)
    """
    if len(llm_scores) != len(baseline_scores):
        raise ValueError("Score lists must have same length")
    
    if len(llm_scores) < 3:
        return np.nan, 1.0  # Not enough data
    
    try:
        stat, p_value = wilcoxon(llm_scores, baseline_scores, 
                                 alternative=alternative,
                                 zero_method='zsplit')
        return stat, p_value
    except Exception as e:
        print(f"Warning: Wilcoxon test failed: {e}")
        return np.nan, 1.0


def compute_cohens_d(llm_scores: List[float], 
                     baseline_scores: List[float]) -> float:
    """
    Compute Cohen's d effect size.
    
    Effect size interpretation:
    - |d| < 0.2: negligible
    - 0.2 ≤ |d| < 0.5: small
    - 0.5 ≤ |d| < 0.8: medium  
    - |d| ≥ 0.8: large
    
    Args:
        llm_scores: LLM coverage scores
        baseline_scores: Baseline coverage scores
        
    Returns:
        Cohen's d
    """
    llm_array = np.array(llm_scores)
    baseline_array = np.array(baseline_scores)
    
    mean_diff = np.mean(llm_array) - np.mean(baseline_array)
    
    # Pooled standard deviation
    n1, n2 = len(llm_array), len(baseline_array)
    var1, var2 = np.var(llm_array, ddof=1), np.var(baseline_array, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    
    return mean_diff / pooled_std


def run_all_tests(detailed_results: Dict, 
                  llm_names: List[str],
                  baseline_name: str = 'random') -> pd.DataFrame:
    """
    Run statistical tests for all LLMs vs baseline.
    
    Args:
        detailed_results: Detailed results from Day 4
        llm_names: List of LLM names to test
        baseline_name: Baseline to compare against
        
    Returns:
        DataFrame with test results
    """
    results = []
    p_values_for_correction = []
    
    print(f"\nRunning Wilcoxon signed-rank tests vs '{baseline_name}' baseline...")
    print("="*70)
    
    for llm_name in llm_names:
        # Skip baselines
        if llm_name in ['random', 'heuristic']:
            continue
        
        if llm_name not in detailed_results:
            print(f"Warning: {llm_name} not in results, skipping")
            continue
        
        # Extract paired scores
        llm_scores, baseline_scores = extract_paired_scores(
            detailed_results, llm_name, baseline_name
        )
        
        # Wilcoxon test
        stat, p_value = wilcoxon_test(llm_scores, baseline_scores, alternative='greater')
        
        # Cohen's d
        cohens_d = compute_cohens_d(llm_scores, baseline_scores)
        
        # Mean difference
        mean_diff = np.mean(llm_scores) - np.mean(baseline_scores)
        
        results.append({
            'LLM': llm_name,
            'Mean_LLM': np.mean(llm_scores),
            'Mean_Baseline': np.mean(baseline_scores),
            'Mean_Diff': mean_diff,
            'Cohens_d': cohens_d,
            'Test_Statistic': stat,
            'p_value': p_value,
            'n_pairs': len(llm_scores)
        })
        
        p_values_for_correction.append(p_value)
        
        print(f"{llm_name:12s}: mean_diff={mean_diff:+.3f}, d={cohens_d:.2f}, p={p_value:.4f}")
    
    # FDR correction (Benjamini-Hochberg)
    if p_values_for_correction:
        rejected, p_adjusted, _, _ = multipletests(
            p_values_for_correction,
            method='fdr_bh',
            alpha=0.05
        )
        
        # Add to results
        for i, result in enumerate(results):
            result['p_adjusted_FDR'] = p_adjusted[i]
            result['significant_FDR'] = rejected[i]
    
    df = pd.DataFrame(results)
    df = df.sort_values('Mean_Diff', ascending=False)
    
    return df


def interpret_results(stats_df: pd.DataFrame) -> str:
    """
    Generate interpretation text for paper.
    
    Args:
        stats_df: Statistical test results
        
    Returns:
        Formatted interpretation string
    """
    interpretation = "\n" + "="*70 + "\n"
    interpretation += "STATISTICAL SIGNIFICANCE INTERPRETATION\n"
    interpretation += "="*70 + "\n"
    
    # Count significant results
    n_total = len(stats_df)
    n_significant = stats_df['significant_FDR'].sum() if 'significant_FDR' in stats_df.columns else 0
    
    interpretation += f"\nTested {n_total} LLMs against random baseline\n"
    interpretation += f"Significant after FDR correction (α=0.05): {n_significant}/{n_total}\n"
    
    if n_significant == n_total:
        interpretation += "\n✓ ALL LLMs significantly outperform random (strong evidence)\n"
    elif n_significant > 0:
        interpretation += f"\n⚠ {n_significant}/{n_total} LLMs significantly better than random\n"
    else:
        interpretation += "\n✗ No LLM significantly better than random\n"
    
    # Effect sizes
    interpretation += "\nEffect Sizes (Cohen's d):\n"
    for _, row in stats_df.iterrows():
        d = row['Cohens_d']
        
        if d < 0.2:
            effect = "negligible"
        elif d < 0.5:
            effect = "small"
        elif d < 0.8:
            effect = "medium"
        else:
            effect = "large"
        
        sig_marker = "**" if row.get('significant_FDR', False) else "  "
        interpretation += f"  {sig_marker} {row['LLM']:12s}: d={d:+.2f} ({effect})\n"
    
    interpretation += "\nFor paper:\n"
    interpretation += "-" * 70 + "\n"
    
    # Generate paper text
    best_row = stats_df.iloc[0]
    best_llm = best_row['LLM']
    best_diff = best_row['Mean_Diff']
    best_p = best_row.get('p_adjusted_FDR', best_row['p_value'])
    best_d = best_row['Cohens_d']
    
    interpretation += f'"{best_llm} achieves {best_diff:.1%} higher coverage than random '\
                      f'(p={best_p:.4f}, FDR-corrected; Cohen\'s d={best_d:.2f}). "'
    
    if n_significant == n_total:
        interpretation += f'All {n_total} LLMs significantly outperform random guessing, '\
                          'suggesting genuine algorithmic knowledge."\n'
    else:
        interpretation += f'{n_significant} of {n_total} LLMs significantly outperform random."\n'
    
    return interpretation


def main():
    parser = argparse.ArgumentParser(description='Statistical significance testing')
    parser.add_argument('--input', type=str, default='results/evaluation',
                        help='Input directory with evaluation results')
    parser.add_argument('--output', type=str, default='results/statistics',
                        help='Output directory for statistical test results')
    parser.add_argument('--baseline', type=str, default='random',
                        choices=['random', 'heuristic'],
                        help='Baseline to test against')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("DAY 5: STATISTICAL SIGNIFICANCE TESTING")
    print("="*70)
    
    # Load results
    print(f"\n[1/3] Loading evaluation results from {input_dir}...")
    detailed_results = load_evaluation_results(input_dir)
    
    llm_names = [name for name in detailed_results.keys() 
                 if name not in ['random', 'heuristic']]
    print(f"      Found {len(llm_names)} LLMs: {', '.join(llm_names)}")
    
    # Run tests
    print(f"\n[2/3] Running statistical tests...")
    stats_df = run_all_tests(detailed_results, llm_names, baseline_name=args.baseline)
    
    # Save results
    print(f"\n[3/3] Saving results...")
    
    csv_file = output_dir / 'statistical_tests.csv'
    stats_df.to_csv(csv_file, index=False)
    print(f"      Saved: {csv_file}")
    
    latex_file = output_dir / 'statistical_tests.tex'
    latex_str = stats_df.to_latex(index=False, float_format='%.4f')
    with open(latex_file, 'w') as f:
        f.write(latex_str)
    print(f"      Saved: {latex_file}")
    
    # Generate interpretation
    interpretation = interpret_results(stats_df)
    
    interp_file = output_dir / 'interpretation.txt'
    with open(interp_file, 'w') as f:
        f.write(interpretation)
    print(f"      Saved: {interp_file}")
    
    # Print to console
    print(interpretation)
    
    print("\n" + "="*70)
    print("DAY 5 COMPLETE - Statistical tests ready for paper")
    print("="*70)


if __name__ == '__main__':
    main()
