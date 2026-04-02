#!/usr/bin/env python3
"""
Unified plotting script for all visualizations.

Generates publication-quality plots from experimental results.

Usage:
    python plot.py --list                          # List all available experiments
    python plot.py --experiments baseline.asia algorithmic.pc llm.qwen
    python plot.py --experiments all               # Plot all available results
    python plot.py --experiments baseline --style config/style.yaml
"""

import sys
import argparse
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# Add src/ to path for imports (allows running from anywhere)
sys.path.insert(0, str(Path(__file__).parent.parent))


def discover_experiments(results_dir: Path) -> Dict[str, Path]:
    """
    Discover all available experiment result files.
    
    Returns:
        Dict mapping experiment name to JSON file path
        Format for names: {source}.{dataset}_{algo} or {source}.{analysis}
        Sources: baseline, llm
    """
    experiments = {}
    
    # Discover baseline/algorithmic results
    algo_dir = results_dir / "algorithms"
    if algo_dir.exists():
        for json_file in algo_dir.glob("*.json"):
            # Extract experiment name: dataset_algo_runs_N_variance.json
            name = json_file.stem.replace("_runs_", "_").replace("_variance", "")
            exp_name = f"baseline.{name}"
            experiments[exp_name] = json_file
    
    # Discover LLM results
    llm_dir = results_dir / "llm"
    if llm_dir.exists():
        # Recursive search for all JSON files in llm subdirectories
        for json_file in llm_dir.rglob("*.json"):
            # Create hierarchical names: llm.analysis.summary, llm.comparisons.results, etc.
            rel_path = json_file.relative_to(llm_dir)
            name_parts = list(rel_path.parts[:-1]) + [json_file.stem]
            exp_name = "llm." + ".".join(name_parts)
            experiments[exp_name] = json_file
    
    return experiments


def load_style_config(style_path: Path) -> Dict:
    """Load matplotlib style configuration from YAML."""
    if not style_path.exists():
        print(f"  ⚠ Style config not found at {style_path}, using defaults")
        return {}
    
    try:
        with open(style_path) as f:
            config = yaml.safe_load(f)
        print(f"  ✓ Loaded style config from {style_path}")
        return config
    except Exception as e:
        print(f"  ⚠ Error loading style config: {e}")
        return {}


def apply_style_config(style_config: Dict):
    """Apply style configuration from YAML to matplotlib."""
    if not style_config:
        # Fallback defaults
        style_config = {
            "fonts": {"title": 11, "axis_label": 10, "tick_label": 9, "legend": 9},
            "figure": {"dpi": 300}
        }
    
    # Extract configuration
    fonts = style_config.get("fonts", {})
    figure = style_config.get("figure", {})
    colors = style_config.get("colors", {})
    
    rcparams = {
        "font.size": fonts.get("title", 11),
        "axes.titlesize": fonts.get("title", 11),
        "axes.labelsize": fonts.get("axis_label", 10),
        "xtick.labelsize": fonts.get("tick_label", 9),
        "ytick.labelsize": fonts.get("tick_label", 9),
        "legend.fontsize": fonts.get("legend", 9),
        "lines.linewidth": 1.5,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans", "Liberation Sans"],
        "text.antialiased": True,
        "figure.dpi": figure.get("dpi", 300),
        "text.color": "#000000",
        "axes.labelcolor": "#000000",
        "xtick.color": "#000000",
        "ytick.color": "#000000",
        "axes.edgecolor": "#1a1a1a",
        "grid.color": "#4a4a4a",
    }
    
    plt.rcParams.update(rcparams)


def save_plot_hq(fig, output_dir: Path, filename: str):
    """Save plot in both PDF and PNG formats at high quality."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pdf_path = output_dir / f"{filename}.pdf"
    png_path = output_dir / f"{filename}.png"
    
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight', format='pdf')
    fig.savefig(png_path, dpi=300, bbox_inches='tight', format='png')
    
    return str(pdf_path)


def load_json(path: Path) -> Dict:
    """Load JSON file with error handling."""
    if not path.exists():
        print(f"  ⚠ File not found: {path}")
        return {}
    
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"  ⚠ Error loading {path}: {e}")
        return {}


def plot_baseline_variance(exp_name: str, json_file: Path, output_dir: Path):
    """Generate variance analysis plots for baseline algorithms."""
    print(f"\n[PLOT] {exp_name}")
    print("-" * 80)
    
    data = load_json(json_file)
    if not data:
        return
    
    # Extract dataset and algorithm from experiment name
    # Format: baseline.dataset_algorithm
    parts = exp_name.replace("baseline.", "").rsplit("_", 1)
    if len(parts) != 2:
        print(f"  ⚠ Cannot parse experiment name: {exp_name}")
        return
    
    dataset_name, algo_name = parts
    
    # Handle two possible JSON structures:
    # 1. New format with aggregated results (mean, std, ci_95, etc.)
    # 2. Old format with individual runs list
    
    if "results" in data:
        # New aggregated format
        results = data.get("results", {})
        n_runs = data.get("n_runs", 0)
        
        if not results:
            print(f"  ⚠ No results data found in {json_file}")
            return
        
        # Extract metrics from aggregated results
        metrics = {}
        for metric_name in results:
            if metric_name in results and isinstance(results[metric_name], dict):
                metrics[metric_name] = results[metric_name]
        
        if not metrics:
            print(f"  ⚠ No metric data found")
            return
        
        # Plot 1: Metrics summary with error bars (confidence intervals)
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle(f'{dataset_name} - {algo_name.upper()}: Performance Summary ({n_runs} runs)', 
                     fontweight='bold', fontsize=12)
        
        axes = axes.flatten()
        colors = ['#0072B2', '#E69F00', '#D55E00', '#CC79A7']
        metric_names_list = list(metrics.keys())
        
        for idx, metric_name in enumerate(metric_names_list[:4]):
            ax = axes[idx]
            metric_data = metrics[metric_name]
            
            mean_val = metric_data.get('mean', 0)
            ci_lower = metric_data.get('ci_95_lower', mean_val)
            ci_upper = metric_data.get('ci_95_upper', mean_val)
            std_val = metric_data.get('std', 0)
            
            # Create bar plot with error bars
            x_pos = 0
            ax.bar(x_pos, mean_val, color=colors[idx], edgecolor='black', 
                   linewidth=1.5, alpha=0.8, width=0.5)
            
            # Error bars showing 95% CI
            error_lower = mean_val - ci_lower
            error_upper = ci_upper - mean_val
            ax.errorbar(x_pos, mean_val, 
                       yerr=[[error_lower], [error_upper]],
                       fmt='none', color='black', linewidth=2, capsize=10)
            
            ax.set_ylabel(metric_name.capitalize(), fontweight='bold')
            ax.set_title(f'{metric_name.capitalize()}', fontweight='bold')
            ax.set_xlim(-0.5, 0.5)
            ax.set_xticks([])
            
            # Add statistics text
            stats_text = f'μ={mean_val:.4f}\nσ={std_val:.4f}\nCI95=[{ci_lower:.4f}, {ci_upper:.4f}]'
            ax.text(0.5, 0.98, stats_text, transform=ax.transAxes, 
                   ha='right', va='top', fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        safe_name = f"{dataset_name}_{algo_name}_summary"
        save_plot_hq(fig, output_dir, safe_name)
        plt.close()
        print(f"  ✓ Generated: {safe_name}.pdf/png (summary with confidence intervals)")
        
        # Plot 2: Coefficient of Variation for each metric
        fig, ax = plt.subplots(figsize=(8, 5))
        
        cv_values = []
        metric_labels = []
        
        for metric_name, metric_data in metrics.items():
            mean_val = metric_data.get('mean', 0)
            std_val = metric_data.get('std', 0)
            if mean_val > 0:
                cv = 100 * std_val / mean_val
                cv_values.append(cv)
                metric_labels.append(metric_name.capitalize())
        
        if cv_values:
            bars = ax.bar(range(len(metric_labels)), cv_values, 
                         color=colors[:len(metric_labels)],
                         edgecolor='black', linewidth=1.2, alpha=0.8)
            
            ax.set_ylabel('Coefficient of Variation (%)', fontweight='bold')
            ax.set_title(f'{dataset_name} - {algo_name.upper()}: Metric Stability', fontweight='bold')
            ax.set_xticks(range(len(metric_labels)))
            ax.set_xticklabels(metric_labels, rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.3)
            
            for bar, val in zip(bars, cv_values):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                        f'{val:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            plt.tight_layout()
            safe_name = f"{dataset_name}_{algo_name}_cv_analysis"
            save_plot_hq(fig, output_dir, safe_name)
            plt.close()
            print(f"  ✓ Generated: {safe_name}.pdf/png (CV Analysis)")
    
    elif "runs" in data:
        # Legacy format with individual runs
        runs = data.get("runs", [])
        if not runs:
            print(f"  ⚠ No runs data found in {json_file}")
            return
        
        # Extract metrics
        metrics = {
            'precision': [r.get('precision', 0) for r in runs],
            'recall': [r.get('recall', 0) for r in runs],
            'f1': [r.get('f1', 0) for r in runs],
            'shd': [r.get('shd', 0) for r in runs],
        }
        
        # Plot 1: Metrics across runs
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle(f'{dataset_name} - {algo_name.upper()}: Performance Across {len(runs)} Runs', 
                     fontweight='bold', fontsize=12)
        
        axes = axes.flatten()
        colors = ['#0072B2', '#E69F00', '#D55E00', '#CC79A7']
        
        for idx, (metric_name, values) in enumerate(metrics.items()):
            ax = axes[idx]
            ax.plot(range(len(values)), values, marker='o', color=colors[idx], 
                    linewidth=2, markersize=6, label=metric_name)
            ax.fill_between(range(len(values)), values, alpha=0.2, color=colors[idx])
            ax.set_xlabel('Run Number', fontweight='bold')
            ax.set_ylabel(metric_name.capitalize(), fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            mean_val = np.mean(values)
            std_val = np.std(values)
            ax.axhline(y=mean_val, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
            ax.text(0.98, 0.95, f'μ={mean_val:.3f}\nσ={std_val:.3f}', 
                   transform=ax.transAxes, ha='right', va='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        safe_name = f"{dataset_name}_{algo_name}_variance_runs"
        save_plot_hq(fig, output_dir, safe_name)
        plt.close()
        print(f"  ✓ Generated: {safe_name}.pdf/png")
        
        # Plot 2: Coefficient of Variation
        fig, ax = plt.subplots(figsize=(8, 5))
        
        cv_values = []
        metric_names = []
        
        for metric_name, values in metrics.items():
            if np.mean(values) > 0:
                cv = 100 * np.std(values) / np.mean(values)
                cv_values.append(cv)
                metric_names.append(metric_name.capitalize())
        
        bars = ax.bar(range(len(metric_names)), cv_values, color=colors[:len(metric_names)],
                      edgecolor='black', linewidth=1.2, alpha=0.8)
        
        ax.set_ylabel('Coefficient of Variation (%)', fontweight='bold')
        ax.set_title(f'{dataset_name} - {algo_name.upper()}: Metric Stability', fontweight='bold')
        ax.set_xticks(range(len(metric_names)))
        ax.set_xticklabels(metric_names, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars, cv_values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        safe_name = f"{dataset_name}_{algo_name}_cv_analysis"
        save_plot_hq(fig, output_dir, safe_name)
        plt.close()
        print(f"  ✓ Generated: {safe_name}.pdf/png (CV Analysis)")
    else:
        print(f"  ⚠ Cannot determine data format in {json_file}")



def plot_llm_results(exp_name: str, json_file: Path, output_dir: Path):
    """Generate plots for LLM experimental results."""
    print(f"\n[PLOT] {exp_name}")
    print("-" * 80)
    
    data = load_json(json_file)
    if not data:
        return
    
    # Determine plot type based on content structure
    if isinstance(data, dict):
        # Check if it's a comparison result with models
        if any(isinstance(v, dict) and "models" in v for v in data.values()):
            plot_llm_comparison(exp_name, data, output_dir)
        # Check if it's a robustness summary
        elif any("cv" in str(k).lower() for v in data.values() if isinstance(v, dict) for k in v.keys()):
            plot_llm_robustness(exp_name, data, output_dir)
        # Check if it's coverage data
        elif any("coverage" in str(k).lower() for v in data.values() if isinstance(v, dict) for k in v.keys()):
            plot_llm_coverage(exp_name, data, output_dir)
        else:
            print(f"  ⚠ Unknown LLM results format in {json_file}")
    
    print(f"  ✓ Processed LLM results: {exp_name}")


def plot_llm_comparison(exp_name: str, data: Dict, output_dir: Path):
    """Plot LLM vs algorithmic comparison results."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    results_summary = {}
    for key, exp_data in data.items():
        if "models" in exp_data:
            results_summary[key] = {
                'algo_mean': exp_data.get('algorithmic_mean', 0),
                'models': exp_data['models']
            }
    
    if not results_summary:
        print(f"  ⚠ No comparison data found")
        return
    
    # Plot first 10 experiments for clarity
    exp_keys = list(results_summary.keys())[:10]
    algo_means = [results_summary[k]['algo_mean'] for k in exp_keys]
    
    x = np.arange(len(exp_keys))
    width = 0.35
    
    ax.bar(x - width/2, algo_means, width, label='Algorithm', 
           color='#0072B2', edgecolor='black', linewidth=1.0, alpha=0.85)
    
    ax.set_ylabel('Performance Score', fontweight='bold')
    ax.set_title(f'LLM vs Algorithm Comparison: {exp_name}', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([k[:15] for k in exp_keys], rotation=45, ha='right', fontsize=8)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    safe_name = exp_name.replace(".", "_")
    save_plot_hq(fig, output_dir, safe_name)
    plt.close()


def plot_llm_robustness(exp_name: str, data: Dict, output_dir: Path):
    """Plot LLM robustness analysis (CV%)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models_cv = {}
    for model, metrics in data.items():
        if isinstance(metrics, dict) and "precision_cv" in metrics:
            models_cv[model] = {
                'precision': metrics.get('precision_cv', 0),
                'recall': metrics.get('recall_cv', 0),
                'f1': metrics.get('f1_cv', 0),
            }
    
    if not models_cv:
        print(f"  ⚠ No robustness CV data found")
        return
    
    models = sorted(models_cv.keys())
    precision_vals = [models_cv[m]['precision'] for m in models]
    recall_vals = [models_cv[m]['recall'] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    ax.bar(x - width/2, precision_vals, width, label='Precision CV', 
           color='#0072B2', edgecolor='black', linewidth=1.0, alpha=0.85)
    ax.bar(x + width/2, recall_vals, width, label='Recall CV', 
           color='#E69F00', edgecolor='black', linewidth=1.0, alpha=0.85)
    
    ax.set_ylabel('Coefficient of Variation (%)', fontweight='bold')
    ax.set_title(f'LLM Robustness Analysis: {exp_name}', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    safe_name = exp_name.replace(".", "_")
    save_plot_hq(fig, output_dir, safe_name)
    plt.close()


def plot_llm_coverage(exp_name: str, data: Dict, output_dir: Path):
    """Plot LLM coverage analysis."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    coverage_data = {}
    for key, metrics in data.items():
        if isinstance(metrics, dict) and any("coverage" in str(k).lower() for k in metrics.keys()):
            coverage_data[key] = metrics
    
    if not coverage_data:
        print(f"  ⚠ No coverage data found")
        return
    
    keys = list(coverage_data.keys())[:15]
    values = [coverage_data[k].get('calibrated_coverage_score', 0) * 100 for k in keys]
    
    ax.barh(range(len(keys)), values, color='steelblue', edgecolor='black', linewidth=1.0, alpha=0.85)
    ax.set_xlabel('Coverage (%)', fontweight='bold')
    ax.set_title(f'LLM Coverage Analysis: {exp_name}', fontweight='bold')
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels(keys, fontsize=8)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    safe_name = exp_name.replace(".", "_")
    save_plot_hq(fig, output_dir, safe_name)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-quality plots from experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --list                                    # List all available experiments
  %(prog)s --experiments baseline.asia_pc            # Plot specific experiment
  %(prog)s --experiments baseline llm                # Plot all baseline + llm experiments
  %(prog)s --experiments all                         # Plot everything
  %(prog)s --experiments baseline.* --output plots/  # Plot all baseline with custom output
        """)
    
    parser.add_argument('--experiments', nargs='+', 
                       default=['all'],
                       help='Experiments to plot (supports glob patterns with *)')
    parser.add_argument('--results_dir', type=Path, 
                       default=None,
                       help='Results directory (default: ../results relative to this script)')
    parser.add_argument('--output', type=Path,
                       default=None,
                       help='Output directory for plots (default: ../results/plots)')
    parser.add_argument('--style', type=Path,
                       default=None,
                       help='Style config YAML (default: ../../config/style.yaml)')
    parser.add_argument('--list', action='store_true',
                       help='List all available experiments and exit')
    
    args = parser.parse_args()
    
    # Set defaults based on script location
    results_dir = args.results_dir or (Path(__file__).parent.parent / 'results')
    output_dir = args.output or (Path(__file__).parent.parent / 'results' / 'plots')
    style_path = args.style or (Path(__file__).parent.parent.parent / 'config' / 'style.yaml')
    
    # Discover available experiments
    all_experiments = discover_experiments(results_dir)
    
    # List available experiments
    if args.list or not all_experiments:
        print("\n" + "=" * 80)
        print("AVAILABLE EXPERIMENTS")
        print("=" * 80)
        if all_experiments:
            for exp_name in sorted(all_experiments.keys()):
                print(f"  • {exp_name}")
            print(f"\nTotal: {len(all_experiments)} experiments found")
        else:
            print("  No experiments found in:", results_dir)
            print("  Run src/algorithms/run_experiments.py to generate baseline results")
            print("  Run src/llm/query_all_llms.py to generate LLM results")
        print("=" * 80)
        
        if args.list:
            return
    
    # Filter experiments based on user input
    selected_experiments = {}
    
    if 'all' in args.experiments:
        selected_experiments = all_experiments
    else:
        # Support glob patterns
        for pattern in args.experiments:
            if '*' in pattern:
                # Glob pattern matching
                import fnmatch
                for exp_name, exp_path in all_experiments.items():
                    if fnmatch.fnmatch(exp_name, pattern):
                        selected_experiments[exp_name] = exp_path
            else:
                # Exact match or prefix match
                if pattern in all_experiments:
                    selected_experiments[pattern] = all_experiments[pattern]
                else:
                    # Prefix match (e.g., "baseline" matches "baseline.asia_pc")
                    for exp_name, exp_path in all_experiments.items():
                        if exp_name.startswith(pattern + ".") or exp_name.startswith(pattern + "_"):
                            selected_experiments[exp_name] = exp_path
    
    if not selected_experiments:
        print(f"✗ No matching experiments found for: {args.experiments}")
        print(f"  Run with --list to see available experiments")
        return
    
    print("=" * 80)
    print("PLOTTING EXPERIMENTS")
    print("=" * 80)
    print(f"\nResults directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Style config: {style_path}")
    print(f"Experiments to plot: {len(selected_experiments)}")
    print()
    
    # Load and apply style configuration
    style_config = load_style_config(style_path)
    apply_style_config(style_config)
    
    # Plot each experiment
    success_count = 0
    for exp_name in sorted(selected_experiments.keys()):
        json_file = selected_experiments[exp_name]
        try:
            if exp_name.startswith("baseline."):
                plot_baseline_variance(exp_name, json_file, output_dir)
            elif exp_name.startswith("llm."):
                plot_llm_results(exp_name, json_file, output_dir)
            else:
                print(f"\n[PLOT] {exp_name}")
                print("  ⚠ Unknown experiment source")
            success_count += 1
        except Exception as e:
            print(f"\n[PLOT] {exp_name}")
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print(f"✓ Completed: {success_count}/{len(selected_experiments)} experiments plotted")
    print(f"✓ Output saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
