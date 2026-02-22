"""
Dumbbell + Interval plot for LLM vs Algorithmic comparison.

Visualizes:
- Algorithmic mean (circle) with CI (thin error bar)
- LLM midpoint (dot) with range (thick error bar)
- Deviation connector (line between the two)
- Faceted by metric (precision, recall, f1, shd)
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

# Configuration
RESULTS_PATH = Path(__file__).parent / "results" / "comparisons" / "comparison_results.json"
OUTPUT_DIR = Path(__file__).parent / "results" / "plots"
OUTPUT_DIR.mkdir(exist_ok=True)

# Plotting constants
METRICS = ["precision", "recall", "f1", "shd"]
METRIC_LABELS = {
    "precision": "Precision",
    "recall": "Recall", 
    "f1": "F1-Score",
    "shd": "Structural Hamming Distance"
}
FIGSIZE = (16, 12)
COLORS = {
    "algo": "#2E86AB",
    "llm": "#A23B72",
    "overlap_yes": "#52B788",
    "overlap_no": "#D62828"
}


def load_comparison_data() -> Dict:
    """Load comparison results from JSON."""
    with open(RESULTS_PATH) as f:
        return json.load(f)


def extract_metric_data(data: Dict, dataset_algo: str, metric: str) -> pd.DataFrame:
    """
    Extract data for a specific dataset/algorithm and metric.
    
    Returns:
        DataFrame with columns: model, algo_mean, algo_ci_lower, algo_ci_upper,
                                llm_lower, llm_upper, llm_mid, center_distance,
                                overlap, llm_width, algo_width
    """
    rows = []
    
    if dataset_algo not in data:
        return pd.DataFrame()
    
    dataset_data = data[dataset_algo]
    
    for model, metrics_dict in dataset_data.get("models", {}).items():
        if metric not in metrics_dict:
            continue
        
        m = metrics_dict[metric]
        
        algo_mean = m.get("algorithmic_mean")
        algo_ci = m.get("algorithmic_ci", [algo_mean, algo_mean])
        llm_range = m.get("llm_range", [0, 0])
        
        llm_mid = np.mean(llm_range)
        
        rows.append({
            "model": model,
            "algo_mean": algo_mean,
            "algo_ci_lower": algo_ci[0],
            "algo_ci_upper": algo_ci[1],
            "llm_lower": llm_range[0],
            "llm_upper": llm_range[1],
            "llm_mid": llm_mid,
            "center_distance": m.get("center_distance"),
            "overlap": m.get("overlap"),
            "llm_width": m.get("llm_width"),
            "algo_width": m.get("algo_width"),
        })
    
    df = pd.DataFrame(rows)
    # Sort by center_distance (largest deviations first for visual prominence)
    if not df.empty:
        df = df.sort_values("center_distance", ascending=True).reset_index(drop=True)
    return df


def plot_single_dataset_algorithm(data: Dict, dataset_algo: str, save=True):
    """
    Create dumbbell plot for all 4 metrics of a single dataset/algorithm.
    
    Args:
        data: Loaded comparison data
        dataset_algo: Key like "alarm_fci"
        save: Whether to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE)
    fig.suptitle(f"LLM vs Algorithmic Performance: {dataset_algo.replace('_', ' ').title()}",
                 fontsize=18, fontweight="bold", y=0.995)
    
    axes_flat = axes.flatten()
    
    for ax, metric in zip(axes_flat, METRICS):
        df = extract_metric_data(data, dataset_algo, metric)
        
        if df.empty:
            ax.text(0.5, 0.5, f"No data for {metric}", 
                   ha="center", va="center", transform=ax.transAxes)
            continue
        
        n_models = len(df)
        y_positions = np.arange(n_models)
        
        # Draw algorithmic CI (thin bar)
        ax.barh(y_positions, 
               df["algo_ci_upper"] - df["algo_ci_lower"],
               left=df["algo_ci_lower"],
               height=0.15,
               color=COLORS["algo"],
               alpha=0.4,
               label="Algo CI",
               linewidth=0)
        
        # Draw LLM range (thick bar)
        ax.barh(y_positions,
               df["llm_upper"] - df["llm_lower"],
               left=df["llm_lower"],
               height=0.35,
               color=COLORS["llm"],
               alpha=0.3,
               label="LLM Range",
               linewidth=0)
        
        # Draw connector lines (deviation)
        for i, row in df.iterrows():
            color = COLORS["overlap_yes"] if row["overlap"] else COLORS["overlap_no"]
            ax.plot([row["algo_mean"], row["llm_mid"]], 
                   [i, i], 
                   color=color,
                   linewidth=2,
                   alpha=0.6,
                   zorder=2)
        
        # Algorithmic mean (circle)
        ax.scatter(df["algo_mean"], y_positions,
                  s=120, color=COLORS["algo"], marker="o",
                  edgecolors="black", linewidth=1.5, zorder=3,
                  label="Algo Mean")
        
        # LLM midpoint (dot)
        ax.scatter(df["llm_mid"], y_positions,
                  s=100, color=COLORS["llm"], marker="D",
                  edgecolors="black", linewidth=1.5, zorder=3,
                  label="LLM Midpoint")
        
        # Set labels and formatting
        ax.set_yticks(y_positions)
        ax.set_yticklabels(df["model"], fontsize=15)
        ax.set_xlabel(METRIC_LABELS[metric], fontsize=16, fontweight="bold")
        ax.grid(axis="x", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
    
    # Add legend
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["algo"],
               markersize=8, markeredgecolor="black", markeredgewidth=1, label="Algorithm Mean"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=COLORS["llm"],
               markersize=8, markeredgecolor="black", markeredgewidth=1, label="LLM Midpoint"),
        mpatches.Patch(color=COLORS["algo"], alpha=0.4, label="Algorithm CI (thin)"),
        mpatches.Patch(color=COLORS["llm"], alpha=0.3, label="LLM Range (thick)"),
    ]
    
    fig.legend(handles=legend_elements, loc="lower center", ncol=4, 
              fontsize=15, frameon=True, bbox_to_anchor=(0.5, -0.005))
    
    plt.tight_layout(rect=[0, 0.06, 1, 0.99])
    
    if save:
        output_path = OUTPUT_DIR / f"dumbbell_{dataset_algo}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved: {output_path}")
    
    return fig, axes





def plot_all_datasets(data: Dict, save=True):
    """
    Create plots for all dataset/algorithm combinations.
    
    Args:
        data: Loaded comparison data
        save: Whether to save figures
    """
    dataset_algos = list(data.keys())
    print(f"Found {len(dataset_algos)} dataset/algorithm combinations")
    print(f"Datasets: {dataset_algos}\n")
    
    for dataset_algo in dataset_algos:
        print(f"Plotting {dataset_algo}...")
        plot_single_dataset_algorithm(data, dataset_algo, save=save)
        plt.close("all")
    
    print(f"\n✓ All plots saved to {OUTPUT_DIR}")


def create_summary_table(data: Dict):
    """
    Create a summary table showing calibration stats across all datasets.
    """
    summary_rows = []
    
    for dataset_algo, dataset_data in data.items():
        for model, metrics_dict in dataset_data.get("models", {}).items():
            for metric, m in metrics_dict.items():
                summary_rows.append({
                    "Dataset-Algorithm": dataset_algo,
                    "Model": model,
                    "Metric": metric,
                    "Center Distance": m.get("center_distance", np.nan),
                    "Overlap": m.get("overlap", False),
                    "LLM Width": m.get("llm_width", np.nan),
                    "Algo Width": m.get("algo_width", np.nan),
                    "Width Ratio": m.get("llm_width", 1) / (m.get("algo_width", 1) + 1e-10),
                })
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Save summary
    summary_path = OUTPUT_DIR / "calibration_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"✓ Summary table saved: {summary_path}")
    
    # Print calibration statistics
    print("\n" + "="*80)
    print("CALIBRATION SUMMARY")
    print("="*80)
    print(f"\nTotal comparisons: {len(summary_df)}")
    print(f"Overlap cases: {summary_df['Overlap'].sum()} ({100*summary_df['Overlap'].mean():.1f}%)")
    print(f"\nWidth Ratio Statistics (LLM/Algo):")
    print(summary_df["Width Ratio"].describe())
    
    return summary_df


if __name__ == "__main__":
    print("Loading comparison data...")
    data = load_comparison_data()
    
    print("\n" + "="*80)
    print("CREATING DUMBBELL PLOTS")
    print("="*80 + "\n")
    
    # Create plots for all datasets
    plot_all_datasets(data, save=True)
    
    # Create summary table
    print("\n" + "="*80)
    print("CREATING SUMMARY STATISTICS")
    print("="*80 + "\n")
    summary_df = create_summary_table(data)
    
    print("\n✓ All visualizations complete!")
