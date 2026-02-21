#!/usr/bin/env python3
"""
Analyze Variance Across Prompt Formulations
============================================

This script analyzes how LLM estimates vary across the three prompt formulations.

Key metrics:
1. Cross-prompt variance (std across formulations)
2. Cross-prompt range (max - min)
3. Coefficient of variation (CV = std / mean)
4. Robustness score (1 - CV)

Interpretation:
- Low variance (<20% difference): ROBUST - results hold across formulations
- High variance (>20% difference): SENSITIVE - must discuss why

Either outcome is publishable if discussed properly.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import json

# ACL one-column format with crisp serif fonts for publication quality
plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "lines.linewidth": 1.5,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    # Times New Roman serif font - standard for academic papers, crisp in PDFs
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Liberation Sans", "Helvetica", "Arial"],
    "text.antialiased": True,
    "figure.dpi": 100,
    # Very dark black for all text elements
    "text.color": "#000000",
    "axes.labelcolor": "#000000",
    "xtick.color": "#000000",
    "ytick.color": "#000000",
    "axes.edgecolor": "#1a1a1a",
    "grid.color": "#4a4a4a",
})


def analyze_prompt_variance(llm_results: Dict[str, Dict[str, Tuple[float, float]]]) -> Dict:
    """
    Analyze variance across prompt formulations for a single LLM.

    Args:
        llm_results: Dictionary mapping formulation_id -> {metric: (lower, upper)}
            Example:
            {
                'formulation_1': {
                    'precision': (0.70, 0.85),
                    'recall': (0.65, 0.80),
                    ...
                },
                'formulation_2': {...},
                'formulation_3': {...}
            }

    Returns:
        Dictionary with variance analysis
    """
    metrics = ['precision', 'recall', 'f1', 'shd']
    results = {}

    for metric in metrics:
        # Extract midpoints from ranges
        midpoints = []
        ranges = []

        for form_id in ['formulation_1', 'formulation_2', 'formulation_3']:
            if form_id in llm_results and metric in llm_results[form_id]:
                lower, upper = llm_results[form_id][metric]
                midpoint = (lower + upper) / 2
                range_width = upper - lower

                midpoints.append(midpoint)
                ranges.append(range_width)

        if len(midpoints) < 2:
            continue

        # Compute variance metrics
        mean_midpoint = np.mean(midpoints)
        std_midpoint = np.std(midpoints)
        cv = std_midpoint / mean_midpoint if mean_midpoint != 0 else 0

        # Percent difference: (max - min) / mean * 100
        pct_difference = (np.max(midpoints) - np.min(midpoints)) / mean_midpoint * 100 if mean_midpoint != 0 else 0

        # Robustness score: higher is better (0 = completely inconsistent, 1 = perfectly consistent)
        robustness = max(0, 1 - cv)

        results[metric] = {
            'midpoints': midpoints,
            'mean': float(mean_midpoint),
            'std': float(std_midpoint),
            'min': float(np.min(midpoints)),
            'max': float(np.max(midpoints)),
            'range': float(np.max(midpoints) - np.min(midpoints)),
            'cv': float(cv),
            'percent_difference': float(pct_difference),
            'robustness_score': float(robustness),
            'is_robust': pct_difference < 20,  # Threshold from PDF
            'interpretation': 'Robust (<20% variation)' if pct_difference < 20 else 'Sensitive (>20% variation)'
        }

    return results


def compare_prompt_formulations(all_llm_results: Dict[str, Dict],
                                dataset_name: str,
                                algorithm_name: str) -> pd.DataFrame:
    """
    Compare all LLMs across all prompt formulations.

    Args:
        all_llm_results: Dictionary mapping llm_name -> formulation results
        dataset_name: Name of dataset
        algorithm_name: Name of algorithm

    Returns:
        DataFrame with comparison statistics
    """
    rows = []

    for llm_name, formulations in all_llm_results.items():
        analysis = analyze_prompt_variance(formulations)

        for metric, stats in analysis.items():
            rows.append({
                'llm': llm_name,
                'dataset': dataset_name,
                'algorithm': algorithm_name,
                'metric': metric,
                'mean_estimate': stats['mean'],
                'std_across_prompts': stats['std'],
                'cv': stats['cv'],
                'percent_difference': stats['percent_difference'],
                'robustness_score': stats['robustness_score'],
                'is_robust': stats['is_robust'],
                'interpretation': stats['interpretation']
            })

    df = pd.DataFrame(rows)
    return df


def compute_prompt_robustness_score(variance_df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute overall robustness scores for each LLM.

    Args:
        variance_df: DataFrame from compare_prompt_formulations()

    Returns:
        Dictionary mapping llm_name -> overall robustness score (0-1)
    """
    robustness_by_llm = {}

    for llm in variance_df['llm'].unique():
        llm_data = variance_df[variance_df['llm'] == llm]
        # Average robustness across all metrics
        avg_robustness = llm_data['robustness_score'].mean()
        robustness_by_llm[llm] = float(avg_robustness)

    return robustness_by_llm


def visualize_prompt_variance(variance_df: pd.DataFrame, output_dir: Path):
    """
    Create visualizations of prompt variance analysis.

    Args:
        variance_df: DataFrame from compare_prompt_formulations()
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)

    # 1. Robustness Scores Heatmap
    fig, ax = plt.subplots(figsize=(10, 6))

    pivot = variance_df.pivot_table(
        values='robustness_score',
        index='llm',
        columns='metric',
        aggfunc='mean'
    )

    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn',
                vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Robustness Score'})
    ax.set_title('Prompt Robustness Scores Across LLMs and Metrics\n'
                 '(1.0 = Perfectly consistent, 0.0 = Highly variable)',
                 fontsize=14, fontweight='bold', color='#000000')
    ax.set_xlabel('Metric', fontsize=12, color='#000000')
    ax.set_ylabel('LLM', fontsize=12, color='#000000')

    plt.tight_layout()
    plt.savefig(output_dir / 'all_llms_prompt_robustness_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Percent Difference Across Prompts
    fig, ax = plt.subplots(figsize=(12, 6))

    # Bar plot of percent differences
    pivot_pct = variance_df.pivot_table(
        values='percent_difference',
        index='metric',
        columns='llm',
        aggfunc='mean'
    )

    pivot_pct.plot(kind='bar', ax=ax, width=0.8)

    # Add horizontal line at 20% threshold
    ax.axhline(y=20, color='#d72f2f', linestyle='--', linewidth=2,
               label='20% threshold (robust/sensitive boundary)')

    ax.set_title('Percent Difference Across Prompt Formulations\n'
                 '(<20% = Robust, >20% = Sensitive)',
                 fontsize=14, fontweight='bold', color='#000000')
    ax.set_xlabel('Metric', fontsize=12, color='#000000')
    ax.set_ylabel('Percent Difference (%)', fontsize=12, color='#000000')
    ax.legend(title='LLM', bbox_to_anchor=(1.05, 1), loc='upper left', labelcolor='#000000')
    ax.grid(axis='y', alpha=0.3, color='#4a4a4a')
    plt.xticks(rotation=0)

    plt.tight_layout()
    plt.savefig(output_dir / 'all_llms_prompt_percent_difference.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Overall Robustness by LLM
    robustness_scores = compute_prompt_robustness_score(variance_df)

    fig, ax = plt.subplots(figsize=(10, 6))

    llms = list(robustness_scores.keys())
    scores = list(robustness_scores.values())
    colors = ['green' if s > 0.8 else 'orange' if s > 0.6 else 'red' for s in scores]

    bars = ax.barh(llms, scores, color=colors, alpha=0.7)

    # Add value labels
    for i, (llm, score) in enumerate(zip(llms, scores)):
        ax.text(score + 0.02, i, f'{score:.3f}',
                va='center', fontsize=10, fontweight='bold', color='#000000')

    ax.set_xlim(0, 1.1)
    ax.set_xlabel('Overall Robustness Score', fontsize=12, color='#000000')
    ax.set_title('Overall Prompt Robustness by LLM\n'
                 '(Average across all metrics)',
                 fontsize=14, fontweight='bold', color='#000000')
    ax.grid(axis='x', alpha=0.3, color='#4a4a4a')

    plt.tight_layout()
    plt.savefig(output_dir / 'all_llms_overall_robustness.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nVisualizations saved to {output_dir}/")
    print("  - prompt_robustness_heatmap.png")
    print("  - prompt_percent_difference.png")
    print("  - llm_overall_robustness.png")


def generate_latex_table(variance_df: pd.DataFrame, output_path: Path):
    """
    Generate LaTeX table for paper.

    Args:
        variance_df: DataFrame from compare_prompt_formulations()
        output_path: Path to save .tex file
    """
    # Create summary table
    summary = variance_df.groupby(['llm', 'metric']).agg({
        'mean_estimate': 'mean',
        'percent_difference': 'mean',
        'is_robust': lambda x: '✓' if x.iloc[0] else '✗'
    }).reset_index()

    # Pivot for better layout
    table_data = []
    for llm in summary['llm'].unique():
        llm_data = summary[summary['llm'] == llm]
        row = {'LLM': llm}

        for _, metric_row in llm_data.iterrows():
            metric = metric_row['metric']
            pct_diff = metric_row['percent_difference']
            robust = metric_row['is_robust']

            row[f'{metric}_diff'] = f"{pct_diff:.1f}%"
            row[f'{metric}_robust'] = robust

        table_data.append(row)

    df_table = pd.DataFrame(table_data)

    # Generate LaTeX
    latex = df_table.to_latex(index=False, escape=False)

    with open(output_path, 'w') as f:
        f.write("% Prompt Variance Analysis Table\n")
        f.write("% Generated automatically\n\n")
        f.write(latex)

    print(f"\nLaTeX table saved to {output_path}")


if __name__ == "__main__":
    print("Prompt Variance Analysis Module")
    print("="*80)
    print("This module analyzes variance across prompt formulations.")
    print("\nUsage:")
    print("  from prompt_variations import analyze_prompt_variance")
    print("\n  results = analyze_prompt_variance(llm_results)")
    print("  df = compare_prompt_formulations(all_results, 'titanic', 'PC')")
