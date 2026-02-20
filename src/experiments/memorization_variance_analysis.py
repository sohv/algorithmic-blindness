#!/usr/bin/env python3
"""
Variance Analysis: Memorization Detection via Prediction Confidence
===================================================================
Hypothesis: LLMs should have lower variance (tighter ranges) on benchmark
datasets they've likely memorized vs synthetic datasets never in training data.

Usage:
    python memorization_variance_analysis.py
"""

import json
import statistics
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
    import numpy as np
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

# Consistent formatting with analyze_results.py
if PLOTTING_AVAILABLE:
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


BENCHMARK_DATASETS = {"asia", "cancer", "earthquake", "child", "alarm", "insurance", "survey", "sachs", "hepar2"}
SYNTHETIC_DATASETS = {"synthetic_12", "synthetic_30", "synthetic_50", "synthetic_60"}

LLM_MODELS = ["claude", "gpt5", "deepseekthink", "deepseek", "qwenthink", "qwen", "gemini3", "llama"]
METRICS = ["precision", "recall", "f1", "shd"]
ALGORITHMS = ["pc", "fci", "lingam", "notears"]


def load_aggregated_ranges(results_dir: Path) -> Dict:
    """Load all aggregated LLM predictions."""
    aggregated = {}
    ranges_dir = results_dir / "aggregated_ranges"

    for f in ranges_dir.glob("*_aggregated.json"):
        with open(f) as file:
            data = json.load(file)
            key = f"{data['dataset']}_{data['algorithm']}"
            aggregated[key] = data

    return aggregated


def compute_range_width(lower: float, upper: float) -> float:
    """Compute width of a prediction range."""
    return abs(upper - lower)


def analyze_variance_by_dataset_type(aggregated: Dict) -> Dict:
    """Compare prediction variance on benchmark vs synthetic datasets."""

    results = {
        "benchmark": {"widths": [], "by_model": defaultdict(list), "by_metric": defaultdict(list)},
        "synthetic": {"widths": [], "by_model": defaultdict(list), "by_metric": defaultdict(list)}
    }

    for dataset_algo, data in aggregated.items():
        dataset = data["dataset"]
        dataset_type = "benchmark" if dataset in BENCHMARK_DATASETS else "synthetic"

        for model in LLM_MODELS:
            if model not in data["llm_estimates"]:
                continue

            model_data = data["llm_estimates"][model]

            for metric in METRICS:
                if metric not in model_data:
                    continue

                metric_data = model_data[metric]
                width = compute_range_width(metric_data["lower"], metric_data["upper"])

                results[dataset_type]["widths"].append(width)
                results[dataset_type]["by_model"][model].append(width)
                results[dataset_type]["by_metric"][metric].append(width)

    return results


def compute_statistics(widths: List[float]) -> Dict:
    """Compute mean, median, std for a list of widths."""
    if not widths:
        return {}

    return {
        "mean": statistics.mean(widths),
        "median": statistics.median(widths),
        "stdev": statistics.stdev(widths) if len(widths) > 1 else 0,
        "min": min(widths),
        "max": max(widths),
        "count": len(widths)
    }


def save_plots_hq(fig, plots_dir: Path, name: str):
    """Save plot as both PDF (300dpi) and PNG (300dpi) for publication quality."""
    fig.savefig(str(plots_dir / f"{name}.pdf"), dpi=300, bbox_inches='tight', pad_inches=0.08, format='pdf')
    fig.savefig(str(plots_dir / f"{name}.png"), dpi=300, bbox_inches='tight', pad_inches=0.08, format='png')
    print(f"    ✓ {name}.pdf/png (300dpi)")


def print_results(results: Dict):
    """Print variance analysis results."""

    print("\n" + "="*70)
    print("MEMORIZATION VARIANCE ANALYSIS")
    print("="*70)
    print("\nHypothesis: Tighter ranges on memorized (benchmark) datasets")
    print("            vs wider ranges on novel (synthetic) datasets\n")

    # Overall comparison
    print("OVERALL COMPARISON")
    print("-" * 70)
    benchmark_stats = compute_statistics(results["benchmark"]["widths"])
    synthetic_stats = compute_statistics(results["synthetic"]["widths"])

    print(f"\nBenchmark Datasets (n={benchmark_stats['count']}):")
    print(f"  Mean range width:    {benchmark_stats['mean']:.4f}")
    print(f"  Median range width:  {benchmark_stats['median']:.4f}")
    print(f"  Std deviation:       {benchmark_stats['stdev']:.4f}")

    print(f"\nSynthetic Datasets (n={synthetic_stats['count']}):")
    print(f"  Mean range width:    {synthetic_stats['mean']:.4f}")
    print(f"  Median range width:  {synthetic_stats['median']:.4f}")
    print(f"  Std deviation:       {synthetic_stats['stdev']:.4f}")

    ratio = benchmark_stats['mean'] / synthetic_stats['mean'] if synthetic_stats['mean'] > 0 else 0
    print(f"\nRatio (Benchmark / Synthetic): {ratio:.2f}x")

    if ratio < 1:
        print(f"✓ Evidence of memorization: Benchmark ranges are {1/ratio:.2f}x tighter")
    else:
        print(f"✗ No memorization signal: Benchmark ranges are {ratio:.2f}x wider")

    # Per-model comparison
    print("\n" + "-" * 70)
    print("PER-MODEL VARIANCE")
    print("-" * 70)

    model_ratios = []
    for model in sorted(LLM_MODELS):
        bench_widths = results["benchmark"]["by_model"][model]
        synth_widths = results["synthetic"]["by_model"][model]

        bench_mean = statistics.mean(bench_widths) if bench_widths else 0
        synth_mean = statistics.mean(synth_widths) if synth_widths else 0

        ratio = bench_mean / synth_mean if synth_mean > 0 else 0
        model_ratios.append((model, ratio, bench_mean, synth_mean))

    # Sort by ratio (lower means more memorization signal)
    model_ratios.sort(key=lambda x: x[1])

    print(f"\n{'Model':<20} {'Benchmark':<12} {'Synthetic':<12} {'Ratio':<10} {'Signal'}")
    print("-" * 70)

    for model, ratio, bench_mean, synth_mean in model_ratios:
        signal = "✓ Strong" if ratio < 0.85 else "• Weak" if ratio > 1.0 else "• Moderate"
        print(f"{model:<20} {bench_mean:<12.4f} {synth_mean:<12.4f} {ratio:<10.2f}x {signal}")

    # Per-metric comparison
    print("\n" + "-" * 70)
    print("PER-METRIC VARIANCE")
    print("-" * 70)

    metric_ratios = []
    for metric in METRICS:
        bench_widths = results["benchmark"]["by_metric"][metric]
        synth_widths = results["synthetic"]["by_metric"][metric]

        bench_mean = statistics.mean(bench_widths) if bench_widths else 0
        synth_mean = statistics.mean(synth_widths) if synth_widths else 0

        ratio = bench_mean / synth_mean if synth_mean > 0 else 0
        metric_ratios.append((metric, ratio, bench_mean, synth_mean))

    metric_ratios.sort(key=lambda x: x[1])

    print(f"\n{'Metric':<15} {'Benchmark':<12} {'Synthetic':<12} {'Ratio':<10}")
    print("-" * 70)

    for metric, ratio, bench_mean, synth_mean in metric_ratios:
        print(f"{metric:<15} {bench_mean:<12.4f} {synth_mean:<12.4f} {ratio:<10.2f}x")

    # By network size (synthetic)
    print("\n" + "-" * 70)
    print("SYNTHETIC NETWORK SIZE EFFECT")
    print("-" * 70)

    synthetic_by_size = defaultdict(list)
    for dataset_algo, data in aggregated.items():
        dataset = data["dataset"]
        if dataset not in SYNTHETIC_DATASETS:
            continue

        # Extract size from synthetic_XX
        size = int(dataset.split("_")[1])

        for model in LLM_MODELS:
            if model not in data["llm_estimates"]:
                continue

            model_data = data["llm_estimates"][model]
            for metric in METRICS:
                if metric in model_data:
                    width = compute_range_width(
                        model_data[metric]["lower"],
                        model_data[metric]["upper"]
                    )
                    synthetic_by_size[size].append(width)

    print(f"\n{'Size':<10} {'Mean Width':<15} {'Count':<10}")
    print("-" * 70)
    for size in sorted(synthetic_by_size.keys()):
        widths = synthetic_by_size[size]
        mean_width = statistics.mean(widths)
        print(f"{size:<10} {mean_width:<15.4f} {len(widths):<10}")

    print("\n" + "="*70)


def generate_plots(results: Dict, aggregated: Dict, output_dir: Path):
    """Generate and save plots for variance analysis."""
    if not PLOTTING_AVAILABLE:
        print("Warning: matplotlib not available, skipping plots")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating plots in {output_dir}...")

    # 1. Overall variance comparison
    fig, ax = plt.subplots(figsize=(4.1, 2.5))
    bench_stats = compute_statistics(results["benchmark"]["widths"])
    synth_stats = compute_statistics(results["synthetic"]["widths"])

    categories = ["Benchmark", "Synthetic"]
    means = [bench_stats["mean"], synth_stats["mean"]]
    stdevs = [bench_stats["stdev"], synth_stats["stdev"]]

    x = np.arange(len(categories))
    width = 0.35

    colors_dark = ['#0072B2', '#E69F00']
    bars = ax.bar(x, means, width, yerr=stdevs, capsize=4, color=colors_dark,
                   edgecolor='#333333', linewidth=1.2, alpha=0.85, error_kw={'linewidth': 1.5})

    ax.set_ylabel("Prediction Range Width")
    ax.set_title("Memorization Signal: Benchmark vs Synthetic", fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stdevs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2 - 0.06, height + 0.02,
                f'{mean:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    save_plots_hq(fig, output_dir, "01_variance_benchmark_vs_synthetic")
    plt.close()

    # 2. Per-model variance comparison
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    model_data = []

    for model in sorted(LLM_MODELS):
        bench_widths = results["benchmark"]["by_model"][model]
        synth_widths = results["synthetic"]["by_model"][model]
        bench_mean = statistics.mean(bench_widths) if bench_widths else 0
        synth_mean = statistics.mean(synth_widths) if synth_widths else 0
        ratio = bench_mean / synth_mean if synth_mean > 0 else 0
        model_data.append((model, ratio))

    models = [m[0] for m in model_data]
    ratios = [m[1] for m in model_data]
    colors = ['#009E73' if r < 0.85 else '#D55E00' for r in ratios]

    # Normalize ratios to 0-1 scale for cleaner visualization
    max_ratio = max(ratios) if ratios else 1
    normalized_ratios = [r / max_ratio for r in ratios]

    bars = ax.barh(models, normalized_ratios, color=colors, edgecolor='#333333',
                    linewidth=1.0, alpha=0.85, height=0.6)
    ax.axvline(x=1.0, color='#666666', linestyle='--', linewidth=1.5, alpha=0.7)

    ax.set_xlabel("Benchmark/Synthetic Width Ratio (0-1 scale)")
    ax.set_title("Per-Model Memorization Signal", fontweight='bold')
    ax.set_xlim(0, 1.0)
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    # Add value labels showing real ratios
    for bar, norm_ratio, real_ratio in zip(bars, normalized_ratios, ratios):
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height()/2.,
                f'{norm_ratio:.2f}', ha='left', va='center', fontsize=9)

    plt.tight_layout()
    save_plots_hq(fig, output_dir, "02_variance_per_model")
    plt.close()

    # 3. Network size effect
    fig, ax = plt.subplots(figsize=(4.1, 2.5))
    synthetic_by_size = defaultdict(list)
    for dataset_algo, data in aggregated.items():
        dataset = data["dataset"]
        if dataset not in SYNTHETIC_DATASETS:
            continue
        size = int(dataset.split("_")[1])
        for model in LLM_MODELS:
            if model not in data["llm_estimates"]:
                continue
            model_data = data["llm_estimates"][model]
            for metric in METRICS:
                if metric in model_data:
                    width = compute_range_width(model_data[metric]["lower"], model_data[metric]["upper"])
                    synthetic_by_size[size].append(width)

    sizes = sorted(synthetic_by_size.keys())
    means = [statistics.mean(synthetic_by_size[s]) for s in sizes]

    ax.plot(sizes, means, marker='o', linewidth=2, markersize=7, color='#D55E00', label='Mean Width')
    ax.fill_between(sizes, means, alpha=0.2, color='#D55E00')

    ax.set_xlabel("Network Size (nodes)")
    ax.set_ylabel("Mean Prediction Range Width")
    ax.set_title("Network Size Effect on Confidence", fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    save_plots_hq(fig, output_dir, "03_variance_network_size_effect")
    plt.close()

    print(f"  ✓ Generated 3 plots (PDF + PNG)")


def save_results_json(results: Dict, aggregated: Dict, output_dir: Path):
    """Save analysis results in JSON format."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute overall statistics
    bench_stats = compute_statistics(results["benchmark"]["widths"])
    synth_stats = compute_statistics(results["synthetic"]["widths"])

    # Per-model stats
    per_model = {}
    for model in sorted(LLM_MODELS):
        bench_widths = results["benchmark"]["by_model"][model]
        synth_widths = results["synthetic"]["by_model"][model]
        bench_mean = statistics.mean(bench_widths) if bench_widths else 0
        synth_mean = statistics.mean(synth_widths) if synth_widths else 0
        ratio = bench_mean / synth_mean if synth_mean > 0 else 0

        per_model[model] = {
            "benchmark_mean_width": bench_mean,
            "synthetic_mean_width": synth_mean,
            "ratio_bench_synth": ratio,
            "memorization_signal": ratio < 0.85
        }

    # Per-metric stats
    per_metric = {}
    for metric in METRICS:
        bench_widths = results["benchmark"]["by_metric"][metric]
        synth_widths = results["synthetic"]["by_metric"][metric]
        bench_mean = statistics.mean(bench_widths) if bench_widths else 0
        synth_mean = statistics.mean(synth_widths) if synth_widths else 0
        ratio = bench_mean / synth_mean if synth_mean > 0 else 0

        per_metric[metric] = {
            "benchmark_mean_width": bench_mean,
            "synthetic_mean_width": synth_mean,
            "ratio_bench_synth": ratio
        }

    # Network size effect
    synthetic_by_size = defaultdict(list)
    for dataset_algo, data in aggregated.items():
        dataset = data["dataset"]
        if dataset not in SYNTHETIC_DATASETS:
            continue
        size = int(dataset.split("_")[1])
        for model in LLM_MODELS:
            if model not in data["llm_estimates"]:
                continue
            model_data = data["llm_estimates"][model]
            for metric in METRICS:
                if metric in model_data:
                    width = compute_range_width(model_data[metric]["lower"], model_data[metric]["upper"])
                    synthetic_by_size[size].append(width)

    network_size_effect = {}
    for size in sorted(synthetic_by_size.keys()):
        widths = synthetic_by_size[size]
        network_size_effect[str(size)] = {
            "mean_width": statistics.mean(widths),
            "count": len(widths)
        }

    # Compile final results
    final_results = {
        "analysis": "memorization_variance_analysis",
        "hypothesis": "LLMs have tighter prediction ranges on benchmark (memorized) datasets vs synthetic (novel) datasets",
        "overall": {
            "benchmark": bench_stats,
            "synthetic": synth_stats,
            "ratio_benchmark_to_synthetic": bench_stats["mean"] / synth_stats["mean"] if synth_stats["mean"] > 0 else 0,
            "memorization_signal": bench_stats["mean"] / synth_stats["mean"] < 1 if synth_stats["mean"] > 0 else False
        },
        "per_model": per_model,
        "per_metric": per_metric,
        "network_size_effect": network_size_effect
    }

    output_file = output_dir / "memorization_variance_analysis_results.json"
    with open(output_file, "w") as f:
        json.dump(final_results, f, indent=2)

    print(f"✓ Saved results to {output_file}")


if __name__ == "__main__":
    results_dir = Path("src/llm/results")
    output_dir = Path("src/experiments/results")

    print("Loading aggregated LLM predictions...")
    aggregated = load_aggregated_ranges(results_dir)
    print(f"Loaded {len(aggregated)} dataset-algorithm combinations")

    print("Analyzing variance by dataset type...")
    results = analyze_variance_by_dataset_type(aggregated)

    print_results(results)

    print("\nGenerating plots...")
    generate_plots(results, aggregated, output_dir / "plots")

    print("Saving results to JSON...")
    save_results_json(results, aggregated, output_dir)
