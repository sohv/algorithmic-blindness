#!/usr/bin/env python3
"""
Generate publication-quality plots for LLM Calibrated Coverage Analysis

This script generates three critical visualizations of LLM performance:
1. Calibrated Coverage by LLM (PRIMARY RESULT)
2. Real vs Synthetic Ablation Study
3. LiNGAM Failure Mode Analysis

Usage:
    python plot_llm_results.py [--output_dir ./plots]
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ACL one-column format with LARGER, more readable fonts
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "lines.linewidth": 1.5,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    # Professional font and high-quality rendering
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "DejaVu Sans", "Arial", "Helvetica"],
    "text.antialiased": True,
    "figure.dpi": 100,
})


def save_plots_hq(fig, plots_dir: Path, name: str):
    """Save plot as both PDF (300dpi) and PNG (300dpi) for publication quality."""
    # Save as PDF (paper-ready)
    pdf_path = plots_dir / f"{name}.pdf"
    fig.savefig(str(pdf_path), dpi=300, bbox_inches='tight', pad_inches=0.08, format='pdf')
    print(f"    ✓ {name}.pdf (300dpi)")
    
    # Save as PNG (preview/web)
    png_path = plots_dir / f"{name}.png"
    fig.savefig(str(png_path), dpi=300, bbox_inches='tight', pad_inches=0.08, format='png')
    print(f"    ✓ {name}.png (300dpi)")


def load_llm_comparison_data(comparisons_dir: Path) -> Tuple[Dict, Dict, Dict]:
    """Load LLM comparison results from JSON files.
    
    Aggregates calibrated coverage scores and breakdowns by:
    - Real vs Synthetic datasets
    - LiNGAM failure modes
    
    Returns:
        llm_stats: Overall calibrated coverage
        real_synthetic_data: Coverage breakdown by dataset type
        lingam_data: LiNGAM-specific algorithm understanding metrics
    """
    results_file = comparisons_dir / "comparison_results.json"
    
    if not results_file.exists():
        raise FileNotFoundError(f"Comparison results not found: {results_file}")
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    # Overall aggregation
    llm_coverage = {}
    llm_coverage_binary = {}
    
    # Real vs Synthetic breakdown
    real_coverage = {}  # {llm: count of covered}
    real_total = {}     # {llm: total}
    synthetic_coverage = {}
    synthetic_total = {}
    
    # LiNGAM breakdown
    lingam_discrete_coverage = {}  # {llm: count}
    lingam_discrete_total = {}
    lingam_synthetic_coverage = {}
    lingam_synthetic_total = {}
    
    for dataset_algo, exp_data in data.items():
        if "models" not in exp_data:
            continue
        
        dataset = exp_data.get("dataset", "")
        algorithm = exp_data.get("algorithm", "")
        
        # Determine if dataset is real or synthetic
        is_synthetic = "synthetic" in dataset.lower()
        
        for llm, metrics in exp_data["models"].items():
            # ===== Overall coverage =====
            if llm not in llm_coverage:
                llm_coverage[llm] = []
                llm_coverage_binary[llm] = []
            
            # ===== Real vs Synthetic =====
            if is_synthetic:
                if llm not in synthetic_coverage:
                    synthetic_coverage[llm] = 0
                    synthetic_total[llm] = 0
            else:
                if llm not in real_coverage:
                    real_coverage[llm] = 0
                    real_total[llm] = 0
            
            # ===== LiNGAM breakdown =====
            if algorithm.lower() == "lingam":
                is_discrete_data = "true" not in dataset.lower() and "synthetic" not in dataset.lower()
                
                if is_discrete_data:
                    if llm not in lingam_discrete_coverage:
                        lingam_discrete_coverage[llm] = 0
                        lingam_discrete_total[llm] = 0
                else:
                    if llm not in lingam_synthetic_coverage:
                        lingam_synthetic_coverage[llm] = 0
                        lingam_synthetic_total[llm] = 0
            
            # Process metrics
            for metric_name, metric_data in metrics.items():
                if isinstance(metric_data, dict) and "calibrated_coverage_score" in metric_data:
                    llm_coverage[llm].append(metric_data["calibrated_coverage_score"])
                    is_covered = metric_data.get("calibrated_coverage", False)
                    llm_coverage_binary[llm].append(1 if is_covered else 0)
                    
                    # Track Real vs Synthetic
                    if is_synthetic:
                        synthetic_total[llm] = synthetic_total.get(llm, 0) + 1
                        if is_covered:
                            synthetic_coverage[llm] = synthetic_coverage.get(llm, 0) + 1
                    else:
                        real_total[llm] = real_total.get(llm, 0) + 1
                        if is_covered:
                            real_coverage[llm] = real_coverage.get(llm, 0) + 1
                    
                    # Track LiNGAM breakdown
                    if algorithm.lower() == "lingam":
                        is_discrete_data = "true" not in dataset.lower() and "synthetic" not in dataset.lower()
                        if is_discrete_data:
                            lingam_discrete_total[llm] = lingam_discrete_total.get(llm, 0) + 1
                            if is_covered:
                                lingam_discrete_coverage[llm] = lingam_discrete_coverage.get(llm, 0) + 1
                        else:
                            lingam_synthetic_total[llm] = lingam_synthetic_total.get(llm, 0) + 1
                            if is_covered:
                                lingam_synthetic_coverage[llm] = lingam_synthetic_coverage.get(llm, 0) + 1
    
    # Compute overall stats
    llm_stats = {}
    for llm in llm_coverage.keys():
        coverage_pct = 100 * np.mean(llm_coverage_binary[llm]) if llm_coverage_binary[llm] else 0
        llm_stats[llm] = {
            "coverage_pct": coverage_pct,
            "mean_score": np.mean(llm_coverage[llm]) if llm_coverage[llm] else 0,
            "std_score": np.std(llm_coverage[llm]) if len(llm_coverage[llm]) > 1 else 0,
            "n_metrics": len(llm_coverage[llm])
        }
    
    # Real vs Synthetic stats
    rs_data = {}
    for llm in set(list(real_total.keys()) + list(synthetic_total.keys())):
        real_pct = 100 * real_coverage.get(llm, 0) / real_total.get(llm, 1) if real_total.get(llm, 0) > 0 else 0
        synthetic_pct = 100 * synthetic_coverage.get(llm, 0) / synthetic_total.get(llm, 1) if synthetic_total.get(llm, 0) > 0 else 0
        rs_data[llm] = {"real": real_pct, "synthetic": synthetic_pct}
    
    # LiNGAM stats
    lingam_data = {}
    for llm in set(list(lingam_discrete_total.keys()) + list(lingam_synthetic_total.keys())):
        discrete_pct = 100 * lingam_discrete_coverage.get(llm, 0) / lingam_discrete_total.get(llm, 1) if lingam_discrete_total.get(llm, 0) > 0 else 0
        synthetic_pct = 100 * lingam_synthetic_coverage.get(llm, 0) / lingam_synthetic_total.get(llm, 1) if lingam_synthetic_total.get(llm, 0) > 0 else 0
        lingam_data[llm] = {"discrete": discrete_pct, "synthetic": synthetic_pct}
    
    return llm_stats, rs_data, lingam_data


def analyze_scalability(comparison_data: Dict) -> Dict:
    """Analyze LLM vs algorithmic performance degradation by dataset complexity.
    
    Compares simple synthetic (12 nodes) vs complex synthetic (30 nodes).
    Returns both LLM and algorithmic degradation metrics.
    """
    llm_simple_flags = {}  # {llm: [0/1 list]}
    llm_complex_flags = {}
    
    algo_simple_scores = {}  # {llm: [algorithmic_mean values]}
    algo_complex_scores = {}
    
    for dataset_algo, exp_data in comparison_data.items():
        if "models" not in exp_data:
            continue
        
        dataset = exp_data.get("dataset", "")
        
        # Categorize by complexity
        is_simple = "synthetic_12" in dataset
        is_complex = "synthetic_30" in dataset
        
        if not (is_simple or is_complex):
            continue
        
        for llm, metrics in exp_data["models"].items():
            for metric_name, metric_data in metrics.items():
                if isinstance(metric_data, dict) and "calibrated_coverage" in metric_data:
                    # Track LLM coverage
                    if is_simple:
                        if llm not in llm_simple_flags:
                            llm_simple_flags[llm] = []
                        llm_simple_flags[llm].append(1 if metric_data["calibrated_coverage"] else 0)
                        
                        if llm not in algo_simple_scores:
                            algo_simple_scores[llm] = []
                        algo_simple_scores[llm].append(metric_data.get("algorithmic_mean", 0))
                    
                    elif is_complex:
                        if llm not in llm_complex_flags:
                            llm_complex_flags[llm] = []
                        llm_complex_flags[llm].append(1 if metric_data["calibrated_coverage"] else 0)
                        
                        if llm not in algo_complex_scores:
                            algo_complex_scores[llm] = []
                        algo_complex_scores[llm].append(metric_data.get("algorithmic_mean", 0))
    
    # Compute percentages and degradation rates
    result = {}
    all_llms = set(llm_simple_flags.keys()) | set(llm_complex_flags.keys())
    
    for llm in all_llms:
        # LLM calibrated coverage degradation
        llm_simple_pct = 100 * np.mean(llm_simple_flags[llm]) if llm in llm_simple_flags and llm_simple_flags[llm] else 0
        llm_complex_pct = 100 * np.mean(llm_complex_flags[llm]) if llm in llm_complex_flags and llm_complex_flags[llm] else 0
        llm_degradation = llm_simple_pct - llm_complex_pct
        
        # Algorithmic performance degradation
        algo_simple_mean = np.mean(algo_simple_scores[llm]) if llm in algo_simple_scores and algo_simple_scores[llm] else 0
        algo_complex_mean = np.mean(algo_complex_scores[llm]) if llm in algo_complex_scores and algo_complex_scores[llm] else 0
        algo_degradation = algo_simple_mean - algo_complex_mean
        
        result[llm] = {
            "llm_simple": llm_simple_pct,
            "llm_complex": llm_complex_pct,
            "llm_degradation": llm_degradation,
            "algo_simple": algo_simple_mean,
            "algo_complex": algo_complex_mean,
            "algo_degradation": algo_degradation
        }
    
    return result


def generate_plots(output_dir: Path, llm_stats: Dict, rs_data: Dict, lingam_data: Dict, comparison_data: Dict):
    """Generate four critical LLM visualization plots.
    
    Args:
        output_dir: Directory to save plots
        llm_stats: Dictionary of LLM statistics with coverage percentages
        rs_data: Real vs Synthetic coverage breakdown
        lingam_data: LiNGAM discrete vs synthetic breakdown
        comparison_data: Raw comparison data for scalability analysis
    """
    plots_dir = output_dir / "plots" / "llm_results"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating LLM results plots in {plots_dir}...")
    
    # Sort LLMs by coverage for better visualization
    sorted_llms = sorted(llm_stats.keys(), 
                         key=lambda x: llm_stats[x]["coverage_pct"], 
                         reverse=True)
    coverage_values = [llm_stats[llm]["coverage_pct"] for llm in sorted_llms]
    
    # Color scheme based on performance
    colors = []
    for cov in coverage_values:
        if cov >= 70:
            colors.append('#1f77b4')  # Strong blue for good performance
        elif cov >= 50:
            colors.append('#ff7f0e')  # Orange for moderate
        else:
            colors.append('#d62728')  # Red for weak
    
    # ====================================================================
    # Plot 1: Calibrated Coverage (PRIMARY RESULT)
    # ====================================================================
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    
    bars = ax.bar(range(len(sorted_llms)), coverage_values, color=colors, 
                   edgecolor='#333333', linewidth=1.2, alpha=0.85)
    
    ax.set_ylabel('Calibrated Coverage (%)', fontweight='bold', fontsize=10)
    ax.set_title('Calibrated Coverage: LLM Predictions vs Ground Truth', fontweight='bold', fontsize=13)
    ax.set_xticks(range(len(sorted_llms)))
    ax.set_xticklabels(sorted_llms, rotation=45, ha='right', fontsize=9)
    ax.set_ylim([0, 105])
    ax.axhline(y=50, color='#999999', linestyle='--', linewidth=1, alpha=0.5, label='Random baseline')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bar, val in zip(bars, coverage_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add legend at top right
    ax.legend(['Random baseline'], loc='upper right', fontsize=8, framealpha=0.9, edgecolor='#333333')
    
    plt.tight_layout()
    save_plots_hq(fig, plots_dir, "01_calibrated_coverage_primary")
    plt.close()
    print(f"  ✓ 01_calibrated_coverage_primary.pdf/png")
    
    # ====================================================================
    # Plot 2: Real vs Synthetic Ablation
    # ====================================================================
    if rs_data:
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        
        # Filter LLMs that have real/synthetic data
        rs_llms = sorted(rs_data.keys(), 
                        key=lambda x: (rs_data[x]["real"] + rs_data[x]["synthetic"])/2, 
                        reverse=True)
        real_values = [rs_data[llm]["real"] for llm in rs_llms]
        synthetic_values = [rs_data[llm]["synthetic"] for llm in rs_llms]
        

        
        x = np.arange(len(rs_llms))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, real_values, width, label='Real Datasets', 
                        color='#1f77b4', edgecolor='#333333', linewidth=1.0, alpha=0.85)
        bars2 = ax.bar(x + width/2, synthetic_values, width, label='Synthetic Datasets',
                        color='#ff7f0e', edgecolor='#333333', linewidth=1.0, alpha=0.85)
        
        ax.set_ylabel('Coverage (%)', fontweight='bold', fontsize=10)
        ax.set_title('Ablation Study: Coverage on Real vs Synthetic Data', fontweight='bold', fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(rs_llms, rotation=45, ha='right', fontsize=9)
        ax.set_ylim([0, 105])
        ax.legend(loc='upper right', fontsize=9, framealpha=0.95, edgecolor='#333333')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1.0,
                        f'{height:.0f}%', ha='center', va='bottom', fontsize=7)
        
        plt.tight_layout()
        save_plots_hq(fig, plots_dir, "02_real_vs_synthetic_ablation")
        plt.close()
        print(f"  ✓ 02_real_vs_synthetic_ablation.pdf/png")
    else:
        print(f"  ⚠ 02_real_vs_synthetic_ablation (no real/synthetic breakdown data)")
    
    # ====================================================================
    # Plot 3: LiNGAM Failure Mode Analysis
    # ====================================================================
    if lingam_data:
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        
        # Filter LLMs that have LiNGAM data
        lingam_llms = sorted(lingam_data.keys(), 
                            key=lambda x: (lingam_data[x]["discrete"] + lingam_data[x]["synthetic"])/2, 
                            reverse=True)
        discrete_acc = [lingam_data[llm]["discrete"] for llm in lingam_llms]
        synthetic_acc = [lingam_data[llm]["synthetic"] for llm in lingam_llms]
        

        
        x = np.arange(len(lingam_llms))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, discrete_acc, width, label='Discrete Data (expect LOW)', 
                        color='#d62728', edgecolor='#333333', linewidth=1.0, alpha=0.85)
        bars2 = ax.bar(x + width/2, synthetic_acc, width, label='Synthetic Data (expect HIGH)',
                        color='#2ca02c', edgecolor='#333333', linewidth=1.0, alpha=0.85)
        
        ax.set_ylabel('Prediction Accuracy (%)', fontweight='bold', fontsize=10)
        ax.set_title('LiNGAM Failure Mode: Algorithm Understanding Test', fontweight='bold', fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(lingam_llms, rotation=45, ha='right', fontsize=9)
        ax.set_ylim([0, 105])
        ax.axhline(y=50, color='#999999', linestyle='--', linewidth=1, alpha=0.5, label='Random (~50%)')
        ax.legend(loc='upper right', fontsize=8, framealpha=0.95, edgecolor='#333333')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1.0,
                        f'{height:.0f}%', ha='center', va='bottom', fontsize=7)
        
        plt.tight_layout()
        save_plots_hq(fig, plots_dir, "03_lingam_failure_mode")
        plt.close()
        print(f"  ✓ 03_lingam_failure_mode.pdf/png")
    else:
        print(f"  ⚠ 03_lingam_failure_mode (no LiNGAM-specific data)")
    
    # ====================================================================
    # Plot 4: Scalability Analysis (Simple vs Complex Synthetic)
    # ====================================================================
    scalability_data = analyze_scalability(comparison_data)
    

    
    if scalability_data:
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        
        # Sort by LLM simple coverage (worst to best)
        scale_llms = sorted(scalability_data.keys(), 
                           key=lambda x: scalability_data[x]["llm_simple"], 
                           reverse=True)
        simple_coverage = [scalability_data[llm]["llm_simple"] for llm in scale_llms]
        complex_coverage = [scalability_data[llm]["llm_complex"] for llm in scale_llms]
        
        x = np.arange(len(scale_llms))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, simple_coverage, width, label='Simple (12-node)', 
                        color='#2ca02c', edgecolor='#333333', linewidth=1.0, alpha=0.85)
        bars2 = ax.bar(x + width/2, complex_coverage, width, label='Complex (30-node)',
                        color='#d62728', edgecolor='#333333', linewidth=1.0, alpha=0.85)
        
        ax.set_ylabel('Calibrated Coverage (%)', fontweight='bold', fontsize=10)
        ax.set_title('Scalability: LLM Performance on Graph Complexity', fontweight='bold', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(scale_llms, rotation=45, ha='right', fontsize=9)
        ax.set_ylim([0, 105])
        ax.legend(loc='upper right', fontsize=9, framealpha=0.95, edgecolor='#333333')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1.0,
                        f'{height:.0f}%', ha='center', va='bottom', fontsize=7, fontweight='bold')
        
        plt.tight_layout()
        save_plots_hq(fig, plots_dir, "04_scalability_analysis")
        plt.close()
        print(f"  ✓ 04_scalability_analysis.pdf/png")
    else:
        print(f"  ⚠ 04_scalability_analysis (no complexity breakdown data)")
    
    # ====================================================================
    # Summary
    # ====================================================================
    num_plots = 4
    print(f"\n✓ Generated {num_plots} plots × 2 formats (PDF + PNG) = {num_plots * 2} files")
    print(f"✓ All plots saved to {plots_dir} (300dpi publication quality)")
    
    return plots_dir


def main():
    print("=" * 80)
    print("LLM CALIBRATED COVERAGE ANALYSIS - PLOT GENERATION")
    print("=" * 80)
    
    # Load real comparison data
    comparisons_dir = Path(__file__).parent / "variance" / "comparisons"
    
    if not comparisons_dir.exists():
        print(f"\n✗ Comparisons directory not found: {comparisons_dir}")
        return
    
    try:
        with open(comparisons_dir / "comparison_results.json", 'r') as f:
            comparison_data = json.load(f)
        
        llm_stats, rs_data, lingam_data = load_llm_comparison_data(comparisons_dir)
        print(f"\n✓ Loaded comparison data from {comparisons_dir}")
        print(f"  Found {len(llm_stats)} LLM models with coverage scores")
        for llm, stats in sorted(llm_stats.items(), key=lambda x: x[1]["coverage_pct"], reverse=True):
            print(f"    • {llm}: {stats['coverage_pct']:.1f}% coverage ({int(stats['n_metrics'])} metrics)")
    except Exception as e:
        print(f"\n✗ Error loading comparison data: {e}")
        return
    
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    generate_plots(output_dir, llm_stats, rs_data, lingam_data, comparison_data)
    
    print("\n" + "=" * 80)
    print("✓ Plot generation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
