#!/usr/bin/env python3
"""
Visualization script for variance analysis results.

Creates publication-quality plots comparing:
1. Algorithmic variance (CIs) vs LLM prediction ranges
2. Overlap analysis across datasets
3. Performance degradation with graph complexity
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import pandas as pd

# Set publication-quality style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']


def load_results(results_dir: Path) -> Dict:
    """Load all JSON results from directory."""
    results = {}
    
    for json_file in results_dir.glob("*.json"):
        with open(json_file, 'r') as f:
            data = json.load(f)
            key = json_file.stem
            results[key] = data
    
    return results


def plot_metric_comparison(results: Dict, metric: str, output_dir: Path, results_dir: Path = None):
    """
    Create comparison plot for a specific metric across all experiments.

    Shows:
    - Algorithmic mean with 95% CI (error bars)
    - LLM prediction range (shaded region)
    - Overlap indicator
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    datasets = ['titanic_lingam_variance', 'asia_pc_variance', 'cancer_pc_variance',
                'asia_pc_variance', 'alarm_pc_variance', 'sachs_pc_variance', 'survey_pc_variance', 'child_pc_variance', 'hepar2_pc_variance']
    
    for idx, dataset_key in enumerate(datasets):
        ax = axes[idx]
        
        if dataset_key not in results:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(dataset_key.replace('_', ' ').title())
            continue
        
        data = results[dataset_key]
        
        if 'results' not in data:
            continue
        
        metric_data = data['results'].get(metric, {})
        
        mean = metric_data.get('mean', 0)
        ci_lower = metric_data.get('ci_95_lower', mean)
        ci_upper = metric_data.get('ci_95_upper', mean)
        
        # Plot algorithmic result
        ax.errorbar([1], [mean], 
                   yerr=[[mean - ci_lower], [ci_upper - mean]],
                   fmt='o', markersize=8, capsize=5, capthick=2,
                   color='#2E86AB', label='Algorithm (95% CI)')
        
        # Load LLM comparison if available
        if results_dir is not None:
            comparison_file = results_dir / f"{dataset_key}_llm_comparison.json"
        else:
            comparison_file = None

        if comparison_file and comparison_file.exists():
            with open(comparison_file, 'r') as f:
                comp_data = json.load(f)
            
            if metric in comp_data.get('gpt5_comparison', {}):
                llm_lower, llm_upper = comp_data['gpt5_comparison'][metric]['llm_range']
                
                # Plot LLM range as horizontal band
                ax.axhspan(llm_lower, llm_upper, alpha=0.3, 
                          color='#A23B72', label='GPT-5 Estimate')
                
                # Check overlap
                overlaps = comp_data['gpt5_comparison'][metric]['overlaps']
                if overlaps:
                    ax.text(1, mean, '✓', ha='center', va='bottom', 
                           fontsize=16, color='green', fontweight='bold')
                else:
                    ax.text(1, mean, '✗', ha='center', va='bottom',
                           fontsize=16, color='red', fontweight='bold')
        
        ax.set_xlim(0.5, 1.5)
        ax.set_xticks([])
        ax.set_ylabel(metric.upper())
        ax.set_title(dataset_key.replace('_', ' ').title())
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'all_datasets_algorithms_{metric}_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved {metric} comparison plot")


def plot_overlap_heatmap(results_dir: Path, output_dir: Path):
    """
    Create heatmap showing overlap percentage between algorithmic CIs
    and LLM prediction ranges across all experiments.
    """
    # Collect overlap data
    overlap_data = []
    
    for comp_file in results_dir.glob("*_llm_comparison.json"):
        with open(comp_file, 'r') as f:
            data = json.load(f)
        
        dataset = data.get('dataset', 'unknown')
        algorithm = data.get('algorithm', 'unknown')
        
        # Updated to support 6 LLMs
        llm_mapping = {
            'gpt5_comparison': 'GPT-5.2',
            'deepseek_comparison': 'DeepSeek R1',
            'claude_comparison': 'Claude Opus 4.6',
            'gemini_comparison': 'Gemini 2.5 Pro',
            'llama_comparison': 'Llama 3.3 70B',
            'qwen_comparison': 'Qwen 3 32B'
        }

        for llm, llm_name in llm_mapping.items():
            if llm not in data:
                continue
            
            for metric in ['precision', 'recall', 'f1', 'shd']:
                if metric not in data[llm]:
                    continue
                
                overlap_pct = data[llm][metric].get('overlap_percentage', 0)
                
                overlap_data.append({
                    'dataset': f"{dataset}_{algorithm}",
                    'llm': llm_name,
                    'metric': metric,
                    'overlap_pct': overlap_pct
                })
    
    if not overlap_data:
        print("No overlap data found")
        return
    
    df = pd.DataFrame(overlap_data)
    
    # Create pivot table
    pivot = df.pivot_table(
        values='overlap_pct',
        index='dataset',
        columns=['llm', 'metric'],
        aggfunc='mean'
    )
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(14, 8))
    
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn', 
                vmin=0, vmax=100, cbar_kws={'label': 'Overlap %'},
                linewidths=0.5, ax=ax)
    
    ax.set_title('LLM Prediction Accuracy: Overlap with Algorithmic 95% CI', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('LLM Model / Metric', fontsize=12)
    ax.set_ylabel('Dataset / Algorithm', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'all_datasets_algorithms_overlap_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Saved overlap heatmap")


def plot_shd_vs_complexity(results: Dict, output_dir: Path):
    """
    Plot SHD as a function of graph complexity (number of nodes).
    Shows performance degradation for both algorithms and LLMs.
    """
    complexity_data = []
    
    # Extract data for different graph sizes
    for key, data in results.items():
        if 'results' not in data:
            continue
        
        # Parse dataset name to get node count
        dataset = data.get('dataset', '')
        algorithm = data.get('algorithm', '')
        
        n_nodes = None
        if 'synthetic' in dataset:
            n_nodes = int(dataset.split('_')[1])
        elif dataset == 'asia':
            n_nodes = 8
        elif dataset == 'alarm':
            n_nodes = 37
        elif dataset == 'sachs':
            n_nodes = 11
        elif dataset == 'survey':
            n_nodes = 6
        elif dataset == 'child':
            n_nodes = 20
        elif dataset == 'hepar2':
            n_nodes = 70
        elif dataset == 'titanic':
            n_nodes = 7
        
        if n_nodes is None:
            continue
        
        shd = data['results'].get('shd', {})
        
        complexity_data.append({
            'n_nodes': n_nodes,
            'algorithm': algorithm.upper(),
            'mean_shd': shd.get('mean', 0),
            'ci_lower': shd.get('ci_95_lower', 0),
            'ci_upper': shd.get('ci_95_upper', 0),
            'dataset': dataset
        })
    
    if not complexity_data:
        print("No complexity data found")
        return
    
    df = pd.DataFrame(complexity_data)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for algorithm in df['algorithm'].unique():
        subset = df[df['algorithm'] == algorithm]
        
        ax.errorbar(subset['n_nodes'], subset['mean_shd'],
                   yerr=[subset['mean_shd'] - subset['ci_lower'],
                         subset['ci_upper'] - subset['mean_shd']],
                   fmt='o-', label=algorithm, capsize=5, capthick=2,
                   markersize=8, linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Number of Nodes', fontsize=12, fontweight='bold')
    ax.set_ylabel('Structural Hamming Distance (SHD)', fontsize=12, fontweight='bold')
    ax.set_title('Performance Degradation with Graph Complexity', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'all_datasets_algorithms_shd_vs_complexity.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Saved complexity analysis plot")


def create_summary_table(results: Dict, output_dir: Path):
    """Create LaTeX table summarizing all results."""
    
    table_data = []
    
    for key, data in results.items():
        if 'results' not in data:
            continue
        
        dataset = data.get('dataset', 'unknown')
        algorithm = data.get('algorithm', 'unknown')
        
        res = data['results']
        
        row = {
            'Dataset': dataset.replace('_', ' ').title(),
            'Algorithm': algorithm.upper(),
            'Precision': f"{res['precision']['mean']:.3f} "
                        f"[{res['precision']['ci_95_lower']:.3f}, "
                        f"{res['precision']['ci_95_upper']:.3f}]",
            'Recall': f"{res['recall']['mean']:.3f} "
                     f"[{res['recall']['ci_95_lower']:.3f}, "
                     f"{res['recall']['ci_95_upper']:.3f}]",
            'F1': f"{res['f1']['mean']:.3f} "
                 f"[{res['f1']['ci_95_lower']:.3f}, "
                 f"{res['f1']['ci_95_upper']:.3f}]",
            'SHD': f"{res['shd']['mean']:.1f} "
                  f"[{res['shd']['ci_95_lower']:.1f}, "
                  f"{res['shd']['ci_95_upper']:.1f}]"
        }
        
        table_data.append(row)
    
    df = pd.DataFrame(table_data)
    
    # Save as CSV
    df.to_csv(output_dir / 'results_summary.csv', index=False)
    
    # Generate LaTeX
    latex = df.to_latex(index=False, escape=False, 
                       caption="Algorithm Performance with 95% Confidence Intervals",
                       label="tab:results")
    
    with open(output_dir / 'results_summary.tex', 'w') as f:
        f.write(latex)
    
    print("Saved summary table (CSV + LaTeX)")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize variance analysis results")
    parser.add_argument('--results', type=str, default='results',
                       help='Directory containing result JSON files')
    parser.add_argument('--output', type=str, default='plots',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("Loading results...")
    results = load_results(results_dir)
    print(f"Loaded {len(results)} result files")
    
    print("\nGenerating visualizations...")
    
    # Individual metric comparisons
    for metric in ['precision', 'recall', 'f1', 'shd']:
        plot_metric_comparison(results, metric, output_dir, results_dir)
    
    # Overlap heatmap
    plot_overlap_heatmap(results_dir, output_dir)
    
    # Complexity analysis
    plot_shd_vs_complexity(results, output_dir)
    
    # Summary table
    create_summary_table(results, output_dir)
    
    print(f"\nAll visualizations saved to: {output_dir}/")


if __name__ == "__main__":
    main()
