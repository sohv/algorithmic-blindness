#!/usr/bin/env python3
"""
Generate publication-quality plot for Prompt Robustness (CV%) Analysis.

Reads robustness_summary.json and produces a grouped bar chart showing
mean CV% per model across metrics, with per-metric breakdown.

Usage:
    python plot_robustness.py
"""

import json
import statistics
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ACL one-column format with LARGER, more readable fonts
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
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "DejaVu Sans", "Arial", "Helvetica"],
    "text.antialiased": True,
    "figure.dpi": 100,
})

# Okabe-Ito colorblind-safe palette
COLORS = {
    'precision': '#0072B2',   # blue
    'recall':    '#E69F00',   # orange
    'f1':        '#009E73',   # bluish green
    'shd':       '#D55E00',   # vermillion
}

METRIC_ORDER = ['precision', 'recall', 'f1', 'shd']
METRIC_LABELS = {'precision': 'Precision', 'recall': 'Recall', 'f1': 'F1', 'shd': 'SHD'}


def save_plots_hq(fig, plots_dir: Path, name: str):
    """Save plot as both PDF (300dpi) and PNG (300dpi) for publication quality."""
    fig.savefig(str(plots_dir / f"{name}.pdf"), dpi=300, bbox_inches='tight', pad_inches=0.08, format='pdf')
    print(f"    {name}.pdf (300dpi)")
    fig.savefig(str(plots_dir / f"{name}.png"), dpi=300, bbox_inches='tight', pad_inches=0.08, format='png')
    print(f"    {name}.png (300dpi)")


def load_robustness_data(json_path: Path):
    """Load and aggregate robustness data by model and metric."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Aggregate CV% by model and metric
    model_metric_cvs = defaultdict(lambda: defaultdict(list))

    for key_str, metrics in data.items():
        parts = key_str.rsplit('_', 2)
        if len(parts) < 3:
            continue
        model = parts[-1]

        for metric_name, metric_data in metrics.items():
            cv = metric_data.get('cv_midpoint_percent', 0)
            model_metric_cvs[model][metric_name].append(cv)

    return model_metric_cvs


def main():
    robustness_dir = Path(__file__).parent / "results" / "robustness_analysis"
    json_path = robustness_dir / "robustness_summary.json"

    if not json_path.exists():
        print(f"Error: {json_path} not found")
        return

    print("Loading robustness data...")
    model_metric_cvs = load_robustness_data(json_path)

    models = sorted(model_metric_cvs.keys())
    print(f"Found {len(models)} models: {', '.join(models)}")

    # Compute mean CV% per model (across all metrics) for sorting
    model_avg_cv = {}
    for model in models:
        all_cvs = []
        for metric in METRIC_ORDER:
            all_cvs.extend(model_metric_cvs[model].get(metric, []))
        model_avg_cv[model] = statistics.mean(all_cvs) if all_cvs else 0

    # Sort models by average CV (most robust = lowest CV first)
    models_sorted = sorted(models, key=lambda m: model_avg_cv[m])

    # ====================================================================
    # Plot: Per-Model CV% by Metric (grouped bar chart)
    # ====================================================================
    fig, ax = plt.subplots(figsize=(5.5, 3.2))

    x = np.arange(len(models_sorted))
    n_metrics = len(METRIC_ORDER)
    width = 0.18

    for i, metric in enumerate(METRIC_ORDER):
        means = []
        for model in models_sorted:
            cvs = model_metric_cvs[model].get(metric, [])
            means.append(statistics.mean(cvs) if cvs else 0)

        offset = (i - (n_metrics - 1) / 2) * width
        ax.bar(x + offset, means, width, label=METRIC_LABELS[metric],
               color=COLORS[metric], edgecolor='#333333', linewidth=0.8, alpha=0.85)

    # Add overall average CV as diamond markers
    avg_cvs = [model_avg_cv[m] for m in models_sorted]
    ax.scatter(x, avg_cvs, marker='D', s=30, color='#000000', zorder=5, label='Overall Mean')

    ax.set_ylabel('CV (%)', fontweight='bold')
    ax.set_title('Prompt Robustness: Coefficient of Variation by Model', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models_sorted, rotation=45, ha='right')
    ax.set_ylim(bottom=0)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', fontsize=7, framealpha=0.95, edgecolor='#333333', ncol=3)

    plt.tight_layout()
    save_plots_hq(fig, robustness_dir, "prompt_robustness_cv")
    plt.close()

    print(f"\nPlot saved to {robustness_dir}")


if __name__ == "__main__":
    main()
