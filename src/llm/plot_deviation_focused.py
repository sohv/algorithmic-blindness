"""
Deviation-Focused Plot: Center Distance Bar Charts

Visualizes how wrong LLM estimates are by showing:
- X-axis: center_distance (deviation magnitude)
- Y-axis: LLM models
- Color: green (overlap with algo CI) vs red (no overlap)
- Faceted by metric (precision, recall, f1, shd)

This highlights calibration issues and ranks models by estimation error.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict

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
    "overlap_yes": "#52B788",
    "overlap_no": "#D62828"
}


def load_comparison_data() -> Dict:
    """Load comparison results from JSON."""
    with open(RESULTS_PATH) as f:
        return json.load(f)


def extract_center_distance_data(data: Dict, dataset_algo: str, metric: str) -> pd.DataFrame:
    """
    Extract center_distance data for a specific dataset/algorithm and metric.
    
    Returns:
        DataFrame with columns: model, center_distance, overlap
    """
    rows = []
    
    if dataset_algo not in data:
        return pd.DataFrame()
    
    dataset_data = data[dataset_algo]
    
    for model, metrics_dict in dataset_data.get("models", {}).items():
        if metric not in metrics_dict:
            continue
        
        m = metrics_dict[metric]
        
        rows.append({
            "model": model,
            "center_distance": m.get("center_distance"),
            "overlap": m.get("overlap"),
        })
    
    df = pd.DataFrame(rows)
    # Sort by center_distance (largest deviations first)
    if not df.empty:
        df = df.sort_values("center_distance", ascending=True).reset_index(drop=True)
    return df


def plot_single_dataset_deviation(data: Dict, dataset_algo: str, save=True):
    """
    Create deviation-focused bar plot for all 4 metrics of a single dataset/algorithm.
    
    Args:
        data: Loaded comparison data
        dataset_algo: Key like "alarm_fci"
        save: Whether to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE)
    fig.suptitle(f"LLM Estimation Deviation (How Wrong?): {dataset_algo.replace('_', ' ').title()}",
                 fontsize=18, fontweight="bold", y=0.995)
    
    axes_flat = axes.flatten()
    
    for ax, metric in zip(axes_flat, METRICS):
        df = extract_center_distance_data(data, dataset_algo, metric)
        
        if df.empty:
            ax.text(0.5, 0.5, f"No data for {metric}", 
                   ha="center", va="center", transform=ax.transAxes)
            continue
        
        n_models = len(df)
        y_positions = np.arange(n_models)
        
        # Color bars based on overlap
        colors = [COLORS["overlap_yes"] if overlap else COLORS["overlap_no"] 
                 for overlap in df["overlap"]]
        
        bars = ax.barh(y_positions, df["center_distance"], color=colors, 
                      alpha=0.75, edgecolor="black", linewidth=1)
        
        # Set labels and formatting
        ax.set_yticks(y_positions)
        ax.set_yticklabels(df["model"], fontsize=15)
        ax.set_xlabel(METRIC_LABELS[metric], fontsize=16, fontweight="bold")
        ax.grid(axis="x", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        
        # Add value labels on bars
        for i, (idx, row) in enumerate(df.iterrows()):
            val = row["center_distance"]
            ax.text(val * 1.02, i, f"{val:.3f}",
                   va="center", fontsize=13, fontweight="bold")
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color=COLORS["overlap_yes"], alpha=0.75, edgecolor="black", linewidth=1, label="Within Algo CI (Good)"),
        mpatches.Patch(color=COLORS["overlap_no"], alpha=0.75, edgecolor="black", linewidth=1, label="Outside Algo CI (Poor)"),
    ]
    
    fig.legend(handles=legend_elements, loc="lower center", ncol=2, 
              fontsize=15, frameon=True, bbox_to_anchor=(0.5, -0.015))
    
    plt.tight_layout(rect=[0, 0.035, 1, 0.99])
    
    if save:
        output_path = OUTPUT_DIR / f"deviation_focused_{dataset_algo}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved: {output_path}")
    
    return fig, axes


def plot_all_datasets_deviation(data: Dict, save=True):
    """
    Create deviation plots for all dataset/algorithm combinations.
    
    Args:
        data: Loaded comparison data
        save: Whether to save figures
    """
    dataset_algos = list(data.keys())
    print(f"Found {len(dataset_algos)} dataset/algorithm combinations")
    print(f"Datasets: {dataset_algos}\n")
    
    for dataset_algo in dataset_algos:
        print(f"Plotting deviation {dataset_algo}...")
        plot_single_dataset_deviation(data, dataset_algo, save=save)
        plt.close("all")
    
    print(f"\n✓ All deviation plots saved to {OUTPUT_DIR}")


def create_deviation_ranking(data: Dict):
    """
    Create a summary showing which models have the worst/best calibration.
    """
    ranking_rows = []
    
    for dataset_algo, dataset_data in data.items():
        for model, metrics_dict in dataset_data.get("models", {}).items():
            for metric, m in metrics_dict.items():
                ranking_rows.append({
                    "Dataset-Algorithm": dataset_algo,
                    "Model": model,
                    "Metric": metric,
                    "Center Distance": m.get("center_distance", np.nan),
                    "Overlap": m.get("overlap", False),
                })
    
    ranking_df = pd.DataFrame(ranking_rows)
    
    # Save ranking
    ranking_path = OUTPUT_DIR / "deviation_ranking.csv"
    ranking_df.to_csv(ranking_path, index=False)
    print(f"✓ Deviation ranking saved: {ranking_path}")
    
    # Print statistics
    print("\n" + "="*80)
    print("DEVIATION RANKING (How Wrong Are LLM Estimates?)")
    print("="*80)
    print(f"\nTotal comparisons: {len(ranking_df)}")
    print(f"Calibrated (within CI): {ranking_df['Overlap'].sum()} ({100*ranking_df['Overlap'].mean():.1f}%)")
    print(f"\nCenter Distance Statistics (Lower = Better):")
    print(ranking_df["Center Distance"].describe())
    
    # Top worst models
    print("\n" + "-"*80)
    print("WORST CALIBRATED (Top 10 Largest Deviations):")
    print("-"*80)
    worst = ranking_df.nlargest(10, "Center Distance")[["Dataset-Algorithm", "Model", "Metric", "Center Distance", "Overlap"]]
    print(worst.to_string(index=False))
    
    # Top best models
    print("\n" + "-"*80)
    print("BEST CALIBRATED (Top 10 Smallest Deviations):")
    print("-"*80)
    best = ranking_df.nsmallest(10, "Center Distance")[["Dataset-Algorithm", "Model", "Metric", "Center Distance", "Overlap"]]
    print(best.to_string(index=False))
    
    return ranking_df


if __name__ == "__main__":
    print("Loading comparison data...")
    data = load_comparison_data()
    
    print("\n" + "="*80)
    print("CREATING DEVIATION-FOCUSED PLOTS")
    print("="*80 + "\n")
    
    # Create plots for all datasets
    plot_all_datasets_deviation(data, save=True)
    
    # Create ranking table
    print("\n" + "="*80)
    print("CREATING DEVIATION RANKING")
    print("="*80 + "\n")
    ranking_df = create_deviation_ranking(data)
    
    print("\n✓ All deviation visualizations complete!")
