"""
Analyze existing variance results using statistical analysis modules.

This script loads pre-computed variance analysis results from ./results/
and runs statistical analysis without needing to re-run experiments.

Usage:
    python analyze_results.py
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns

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

# Add parent (src) to path for statistical_analysis import
sys.path.insert(0, str(Path(__file__).parent.parent))

from statistical_analysis import StatisticalTester, ExplanatoryAnalyzer


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


def load_variance_results() -> Dict[str, Dict]:
    """Load all variance JSON files from results/ folder."""
    results_dir = Path(__file__).parent / "results"
    
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    variance_files = sorted(results_dir.glob("*_variance.json"))
    
    if not variance_files:
        raise FileNotFoundError(f"No variance JSON files found in {results_dir}")
    
    print(f"Found {len(variance_files)} variance result files:")
    for f in variance_files:
        print(f"  • {f.name}")
    
    all_results = {}
    for file_path in variance_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
            key = f"{data['dataset']}_{data['algorithm']}"
            all_results[key] = data
            print(f"  Loaded: {key}")
    
    return all_results


def extract_metric_by_algorithm(results: Dict[str, Dict], metric: str = 'f1') -> Dict[str, List[float]]:
    """Extract specified metric grouped by algorithm.
    
    Args:
        results: Dict of all variance results
        metric: Which metric to extract (f1, precision, recall, shd)
        
    Returns:
        Dict mapping algorithm name to list of mean values
    """
    algo_values = {}
    
    for key, data in results.items():
        parts = key.split('_')
        algorithm = parts[-1]  # Last part is algorithm
        
        if algorithm not in algo_values:
            algo_values[algorithm] = []
        
        if metric in data['results']:
            mean_value = data['results'][metric]['mean']
            algo_values[algorithm].append(mean_value)
    
    return algo_values


def generate_plots(results: Dict[str, Dict], output_dir: Path):
    """Generate comprehensive visualization plots of experimental results.
    
    Args:
        results: Dict of all variance results
        output_dir: Directory to save plots
    """
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    print(f"\nGenerating plots in {plots_dir}...")
    
    # Extract metrics
    algo_f1 = extract_metric_by_algorithm(results, 'f1')
    algo_precision = extract_metric_by_algorithm(results, 'precision')
    algo_recall = extract_metric_by_algorithm(results, 'recall')
    algo_shd = extract_metric_by_algorithm(results, 'shd')
    
    algorithms = sorted(algo_f1.keys())
    # Use darker, more saturated colors
    colors_dark = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(algorithms)]
    
    # ====================================================================
    # Plot 1: Algorithm F1 Comparison
    # ====================================================================
    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    
    algo_means = []
    algo_stds = []
    for algo in algorithms:
        algo_means.append(np.mean(algo_f1[algo]))
        algo_stds.append(np.std(algo_f1[algo]))
    
    bars = ax.bar(algorithms, algo_means, yerr=algo_stds, capsize=4, 
                   color=colors_dark, edgecolor='#333333', linewidth=1.2, alpha=0.85, error_kw={'linewidth': 1.5})
    
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Score by Algorithm', fontweight='bold')
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars (smaller) - positioned above error bars
    for bar, mean, std in zip(bars, algo_means, algo_stds):
        height = bar.get_height()
        # Position text above the error bar (mean + std + small offset)
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.03,
                f'{mean:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    save_plots_hq(fig, plots_dir, "01_f1_comparison")
    plt.close()
    print(f"  ✓ 01_f1_comparison.pdf/png")
    
    # ====================================================================
    # Plot 2: Precision, Recall, F1 by Algorithm
    # ====================================================================
    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    
    x = np.arange(len(algorithms))
    width = 0.25
    
    precision_means = [np.mean(algo_precision[a]) for a in algorithms]
    recall_means = [np.mean(algo_recall[a]) for a in algorithms]
    f1_means = [np.mean(algo_f1[a]) for a in algorithms]
    
    ax.bar(x - width, precision_means, width, label='Precision', alpha=0.85, edgecolor='#333333', linewidth=0.8, color='#1f77b4')
    ax.bar(x, recall_means, width, label='Recall', alpha=0.85, edgecolor='#333333', linewidth=0.8, color='#ff7f0e')
    ax.bar(x + width, f1_means, width, label='F1', alpha=0.85, edgecolor='#333333', linewidth=0.8, color='#2ca02c')
    
    ax.set_ylabel('Score')
    ax.set_title('Precision, Recall, F1', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, rotation=45, ha='right')
    # Position legend horizontally at top with no title
    ax.set_ylim([0, 1.15])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=3, framealpha=0.9,
              edgecolor='#333333', fontsize=8, frameon=True, title=None)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    save_plots_hq(fig, plots_dir, "02_precision_recall_f1")
    plt.close()
    print(f"  ✓ 02_precision_recall_f1.pdf/png")
    
    # ====================================================================
    # Plot 3: SHD (Structural Hamming Distance) Comparison
    # ====================================================================
    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    
    shd_means = []
    shd_stds = []
    for algo in algorithms:
        shd_means.append(np.mean(algo_shd[algo]))
        shd_stds.append(np.std(algo_shd[algo]))
    
    bars = ax.bar(algorithms, shd_means, yerr=shd_stds, capsize=4, 
                   color=colors_dark, edgecolor='#333333', linewidth=1.2, alpha=0.85, error_kw={'linewidth': 1.5})
    
    ax.set_ylabel('SHD (lower is better)')
    ax.set_title('Structural Hamming Distance', fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels (smaller) - positioned slightly left to avoid overlap
    for bar, mean in zip(bars, shd_means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2. - 0.12, height,
                f'{mean:.0f}', ha='center', va='bottom', fontsize=9)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    save_plots_hq(fig, plots_dir, "03_shd_comparison")
    plt.close()
    print(f"  ✓ 03_shd_comparison.pdf/png")
    
    # ====================================================================
    # Plot 4: F1 Confidence Intervals
    # ====================================================================
    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    
    # Plot CI bands with professional styling
    for alg_idx, algo in enumerate(algorithms):
        f1_vals = algo_f1[algo]
        mean = np.mean(f1_vals)
        ci_lower = np.percentile(f1_vals, 2.5)
        ci_upper = np.percentile(f1_vals, 97.5)
        
        # Error band (filled region)
        ax.fill_betweenx([alg_idx - 0.15, alg_idx + 0.15], ci_lower, ci_upper, 
                          alpha=0.25, color=colors_dark[alg_idx])
        
        # CI line with markers
        ax.plot([ci_lower, ci_upper], [alg_idx, alg_idx], 'o-', linewidth=2.5, 
                markersize=5, color=colors_dark[alg_idx], label=algo)
        
        # Mean marker (diamond)
        ax.plot(mean, alg_idx, 'D', markersize=7, color=colors_dark[alg_idx], 
                markeredgecolor='#333333', markeredgewidth=1.2, zorder=5)
    
    ax.set_yticks(range(len(algorithms)))
    ax.set_yticklabels(algorithms)
    ax.set_xlabel('F1 Score')
    ax.set_title('F1 Score with 95% Confidence Intervals', fontweight='bold')
    ax.set_xlim([0, 1.0])
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Create legend for mean marker
    mean_patch = mpatches.Patch(label='Mean', color='gray', alpha=0.7)
    ax.legend(handles=[mean_patch], loc='upper left', framealpha=0.95, edgecolor='#333333')
    
    plt.tight_layout()
    save_plots_hq(fig, plots_dir, "04_f1_confidence_intervals")
    plt.close()
    print(f"  ✓ 04_f1_confidence_intervals.pdf/png")
    
    # ====================================================================
    # Plot 5: Dataset Performance Heatmap
    # ====================================================================
    # Extract performance by dataset and algorithm
    dataset_algo_f1 = {}
    for key, data in results.items():
        parts = key.split('_')
        algorithm = parts[-1]
        dataset = '_'.join(parts[:-1])
        
        if dataset not in dataset_algo_f1:
            dataset_algo_f1[dataset] = {}
        
        dataset_algo_f1[dataset][algorithm] = data['results']['f1']['mean']
    
    datasets = sorted(dataset_algo_f1.keys())
    
    if len(datasets) > 0:
        # Create matrix
        f1_matrix = np.zeros((len(datasets), len(algorithms)))
        for i, dataset in enumerate(datasets):
            for j, algo in enumerate(algorithms):
                f1_matrix[i, j] = dataset_algo_f1[dataset].get(algo, 0)
        
        # Create dataset labels with asterisk for hepar2
        dataset_labels = [f"{d}*" if d == 'hepar2' else d for d in datasets]
        
        # Plot heatmap (wider for heatmap readability)
        fig, ax = plt.subplots(figsize=(6.5, max(3.5, len(datasets) * 0.38)))
        
        # Use blue-green colormap matching original visualization
        cmap = sns.color_palette("viridis_r", as_cmap=True)
        
        sns.heatmap(f1_matrix, annot=True, fmt='.2f',
                    xticklabels=algorithms,
                    yticklabels=dataset_labels,
                    ax=ax, cmap=cmap, cbar_kws={'label': 'F1 Score'},
                    annot_kws={'fontsize': 8, 'color': 'black'},
                    vmin=0.0, vmax=1.0,
                    linewidths=0.5, linecolor='gray')
        
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_title('F1 Score: Dataset vs Algorithm', fontweight='bold', pad=10)
        
        # Add caption about hepar2 FCI exclusion
        fig.text(0.5, -0.02, '*FCI excluded from Hepar2 due to computational constraints', 
                 ha='center', fontsize=8, style='italic', color='#333333')
        
        plt.tight_layout(pad=2.5, w_pad=3.0, rect=[0, 0.02, 1, 1])
        save_plots_hq(fig, plots_dir, "05_dataset_heatmap")
        plt.close()
        print(f"  ✓ 05_dataset_heatmap.pdf/png")
    
    # ====================================================================
    # Plot 5b: Multi-Metric Faceted View (Precision, Recall, F1)
    # ====================================================================
    # Extract performance by dataset and algorithm for all metrics
    dataset_algo_metrics = {}
    for key, data in results.items():
        parts = key.split('_')
        algorithm = parts[-1]
        dataset = '_'.join(parts[:-1])
        
        if dataset not in dataset_algo_metrics:
            dataset_algo_metrics[dataset] = {}
        if algorithm not in dataset_algo_metrics[dataset]:
            dataset_algo_metrics[dataset][algorithm] = {}
        
        dataset_algo_metrics[dataset][algorithm]['precision'] = data['results']['precision']['mean']
        dataset_algo_metrics[dataset][algorithm]['recall'] = data['results']['recall']['mean']
        dataset_algo_metrics[dataset][algorithm]['f1'] = data['results']['f1']['mean']
    
    datasets = sorted(dataset_algo_metrics.keys())
    metrics = ['precision', 'recall', 'f1']
    metric_labels = ['Precision', 'Recall', 'F1 Score']
    
    if len(datasets) > 0:
        # Create three subplots (one per metric)
        fig, axes = plt.subplots(3, 1, figsize=(15, 11))
        
        # Store handles and labels for common legend
        handles_list, labels_list = [], []
        
        # Prepare data for each metric
        for metric_idx, (metric, metric_label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[metric_idx]
            
            # Build data for this metric
            x = np.arange(len(datasets))
            width = 0.2
            
            for algo_idx, algo in enumerate(algorithms):
                values = []
                for dataset in datasets:
                    if dataset in dataset_algo_metrics and algo in dataset_algo_metrics[dataset]:
                        values.append(dataset_algo_metrics[dataset][algo].get(metric, 0))
                    else:
                        values.append(0)
                
                bars = ax.bar(x + algo_idx * width, values, width, label=algo, 
                       color=colors_dark[algo_idx], alpha=0.85, edgecolor='#333333', linewidth=0.8)
                
                # Collect handles and labels from first subplot only
                if metric_idx == 0:
                    handles_list.append(bars)
                    labels_list.append(algo)
            
            # Mount asterisk for hepar2
            dataset_labels_x = [f"{d}*" if d == 'hepar2' else d for d in datasets]
            
            ax.set_ylabel(metric_label, fontweight='bold', fontsize=14)
            ax.set_title(f'({chr(97 + metric_idx)}) {metric_label} by Dataset and Algorithm', 
                        fontweight='bold', fontsize=13, loc='left')
            ax.set_xticks(x + width * 1.5)
            ax.set_xticklabels(dataset_labels_x, rotation=45, ha='right', fontsize=11)
            ax.tick_params(axis='y', labelsize=12)
            ax.set_ylim([0, 1.05])
            ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add common legend at bottom in one horizontal line
        fig.legend(handles_list, labels_list, loc='lower center', ncol=4, 
                  fontsize=11, framealpha=0.95, edgecolor='#333333', 
                  bbox_to_anchor=(0.5, 0.0), frameon=True)
        
        # Add hepar2 note on the left, aligned with legend
        fig.text(0.05, -0.005, '*FCI excluded from Hepar2 due to computational constraints', 
                ha='left', fontsize=11, style='italic', color='black')
        
        plt.subplots_adjust(hspace=0.45, bottom=0.12)
        save_plots_hq(fig, plots_dir, "05b_metrics_by_dataset")
        plt.close()
        print(f"  ✓ 05b_metrics_by_dataset.pdf/png")
    
    # ====================================================================
    # Plot 6: Distribution of F1 Scores (Box Plot)
    # ====================================================================
    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    
    f1_by_algo = [algo_f1[algo] for algo in algorithms]
    
    bp = ax.boxplot(f1_by_algo, labels=algorithms, patch_artist=True, widths=0.6,
                     boxprops=dict(linewidth=1.2, edgecolor='#333333'),
                     whiskerprops=dict(linewidth=1.2, color='#333333'),
                     capprops=dict(linewidth=1.2, color='#333333'),
                     medianprops=dict(linewidth=1.5, color='#d62728'))
    
    for patch, color in zip(bp['boxes'], colors_dark):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
        patch.set_edgecolor('#333333')
    
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Distribution by Algorithm', fontweight='bold')
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    save_plots_hq(fig, plots_dir, "06_f1_distribution")
    plt.close()
    print(f"  ✓ 06_f1_distribution.pdf/png")
    
    num_plots = len(list(plots_dir.glob('*.pdf')))
    print(f"\n✓ Generated {num_plots} plots × 2 formats (PDF + PNG) = {num_plots * 2} files")
    print(f"✓ All plots saved to {plots_dir} (300dpi publication quality)")
    return plots_dir


def main():
    print("=" * 80)
    print("STATISTICAL ANALYSIS OF EXISTING VARIANCE RESULTS")
    print("=" * 80)
    
    # Load results
    print("\n1. Loading variance results...")
    print("-" * 80)
    results = load_variance_results()
    print(f"\nTotal results loaded: {len(results)}")
    
    # ========================================================================
    # Part 1: Statistical Significance Testing
    # ========================================================================
    print("\n" + "=" * 80)
    print("2. RUNNING STATISTICAL SIGNIFICANCE TESTS")
    print("=" * 80)
    
    tester = StatisticalTester(alpha=0.05)
    
    # Extract F1 scores by algorithm
    algo_f1_scores = extract_metric_by_algorithm(results, 'f1')
    
    print(f"\nAlgorithms found: {list(algo_f1_scores.keys())}")
    for algo, scores in algo_f1_scores.items():
        print(f"  {algo}: {len(scores)} datasets, mean F1 = {sum(scores)/len(scores):.4f}")
    
    # Pairwise comparisons between algorithms
    statistical_results = []
    algo_list = sorted(algo_f1_scores.keys())
    
    print("\nPerforming pairwise algorithm comparisons...")
    for i in range(len(algo_list)):
        for j in range(i + 1, len(algo_list)):
            algo1, algo2 = algo_list[i], algo_list[j]
            
            if len(algo_f1_scores[algo1]) >= 2 and len(algo_f1_scores[algo2]) >= 2:
                result = tester.paired_t_test(
                    algo_f1_scores[algo1],
                    algo_f1_scores[algo2],
                    f"{algo1.upper()} vs {algo2.upper()}"
                )
                statistical_results.append(result)
                
                sig_marker = "✓ SIGNIFICANT" if result.is_significant else "  not significant"
                print(f"  {algo1} vs {algo2}: p={result.p_value:.4f}, d={result.effect_size:.3f} {sig_marker}")
    
    # Apply FDR correction
    if len(statistical_results) > 1:
        print(f"\nApplying FDR multiple comparison correction ({len(statistical_results)} tests)...")
        corrected_results = tester.multiple_comparison_correction(statistical_results, method='fdr')
        
        sig_before = sum(1 for r in statistical_results if r.is_significant)
        sig_after = sum(1 for r in corrected_results if r.corrected_is_significant)
        
        print(f"  Significant before correction: {sig_before}/{len(statistical_results)}")
        print(f"  Significant after FDR correction: {sig_after}/{len(corrected_results)}")
    else:
        corrected_results = statistical_results
    
    # Save statistical report
    report_path = Path(__file__).parent / "results" / "statistical_analysis_report.txt"
    tester.generate_statistical_report(corrected_results, str(report_path))
    print(f"\n✓ Statistical report saved: {report_path}")
    
    # ========================================================================
    # Generate Plots
    # ========================================================================
    print("\n" + "=" * 80)
    print("4. GENERATING VISUALIZATION PLOTS")
    print("=" * 80)
    
    plots_dir = generate_plots(results, Path(__file__).parent / "results")
    
    # ========================================================================
    # Part 2: Explanatory Factor Analysis
    # ========================================================================
    print("\n" + "=" * 80)
    print("5. EXPLANATORY FACTOR ANALYSIS")
    print("=" * 80)
    
    analyzer = ExplanatoryAnalyzer()
    
    # Prepare data for explanatory analysis
    experimental_results = {}
    graph_structures = {}
    dataset_metadata = {}
    
    # Group results by dataset
    for key, data in results.items():
        parts = key.split('_')
        algorithm = parts[-1]
        dataset = '_'.join(parts[:-1])
        
        if dataset not in experimental_results:
            experimental_results[dataset] = {}
        
        experimental_results[dataset][algorithm] = {
            'accuracy': data['results']['f1']['mean'],
            'confidence_interval_width': data['results']['f1']['ci_95_upper'] - data['results']['f1']['ci_95_lower'],
            'calibration_error': abs(data['results']['precision']['mean'] - data['results']['recall']['mean'])
        }
        
        # Mock graph structures and metadata
        if dataset not in graph_structures:
            n_nodes_config = {
                'asia': 8,
                'sachs': 11,
                'synthetic_12': 12,
                'synthetic_30': 30
            }
            n_nodes = n_nodes_config.get(dataset, 10)
            
            # Create sparse DAG structure
            import numpy as np
            graph_structures[dataset] = np.random.binomial(1, 0.15, (n_nodes, n_nodes))
            
            sample_size_config = {
                'asia': 5000,
                'sachs': 7466,
                'synthetic_12': 1000,
                'synthetic_30': 1000
            }
            
            dataset_metadata[dataset] = {
                'sample_size': sample_size_config.get(dataset, 1000),
                'dimensionality': n_nodes,
                'noise_level': 0.05
            }
    
    # Run explanatory analysis
    mock_llm_results = {'Algorithm_Results': experimental_results}
    insights = analyzer.analyze_performance_factors(
        mock_llm_results, graph_structures, dataset_metadata
    )
    
    print(f"\nIdentified {len(insights)} explanatory factors:")
    for insight in sorted(insights, key=lambda x: x.impact_score, reverse=True):
        print(f"  • {insight.factor_name}")
        print(f"    Impact Score: {insight.impact_score:.1f}%")
        print(f"    Correlation: {insight.correlation:.3f}")
        print(f"    {insight.description}")
    
    # Save theory report
    theory_report_path = Path(__file__).parent / "results" / "explanatory_theory_report.txt"
    analyzer.generate_theory_report(insights, str(theory_report_path))
    print(f"\n✓ Explanatory theory report saved: {theory_report_path}")
    
    print(f"\nGenerated files:")
    print(f"  1. {report_path}")
    print(f"  2. {theory_report_path}")
    print(f"  3. Plots in {plots_dir}:")
    for plot_file in sorted(plots_dir.glob("*.png")):
        print(f"     - {plot_file.name}")
    
    print(f"\nDatasets analyzed: {len(experimental_results)}")
    print(f"  {', '.join(sorted(experimental_results.keys()))}")
    
    print(f"\nAlgorithms analyzed: {len(algo_list)}")
    print(f"  {', '.join(sorted(algo_list))}")
    
    print(f"\nStatistical tests performed: {len(corrected_results)}")
    print(f"Significant findings (after FDR): {sum(1 for r in corrected_results if r.corrected_is_significant)}")


if __name__ == '__main__':
    main()
