#!/usr/bin/env python3
"""
Consistency Check: Memorization via Cross-Model Agreement
==========================================================
Hypothesis: LLMs should give similar predictions (high agreement) on benchmark
datasets they've memorized, but diverge on synthetic datasets never in training.

Uses pairwise distance between LLM predictions to measure consistency.

Usage:
    python memorization_consistency_check.py
"""

import json
import statistics
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import itertools

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


def compute_range_distance(range1: Tuple[float, float], range2: Tuple[float, float]) -> float:
    """
    Compute distance between two ranges (0 = identical, 1 = completely disjoint).

    Uses Hausdorff distance: max of center distance and width difference.
    Higher values = less agreement.
    """
    low1, high1 = range1
    low2, high2 = range2

    center1 = (low1 + high1) / 2
    center2 = (low2 + high2) / 2

    width1 = high1 - low1
    width2 = high2 - low2

    center_dist = abs(center1 - center2)
    width_diff = abs(width1 - width2)

    # Normalize by typical magnitude (0.5 for unit ranges)
    return (center_dist + width_diff / 2) / 0.5


def analyze_consistency_by_dataset_type(aggregated: Dict) -> Dict:
    """Compare cross-model consistency on benchmark vs synthetic datasets."""

    results = {
        "benchmark": {
            "distances": [],
            "by_dataset": defaultdict(list),
            "by_metric": defaultdict(list),
            "agreement_counts": []
        },
        "synthetic": {
            "distances": [],
            "by_dataset": defaultdict(list),
            "by_metric": defaultdict(list),
            "agreement_counts": []
        }
    }

    for dataset_algo, data in aggregated.items():
        dataset = data["dataset"]
        algorithm = data["algorithm"]
        dataset_type = "benchmark" if dataset in BENCHMARK_DATASETS else "synthetic"

        # For each metric, compute pairwise distances between all model pairs
        for metric in METRICS:
            # Extract all model predictions for this metric
            predictions = {}
            for model in LLM_MODELS:
                if model in data["llm_estimates"] and metric in data["llm_estimates"][model]:
                    metric_data = data["llm_estimates"][model][metric]
                    predictions[model] = (metric_data["lower"], metric_data["upper"])

            if len(predictions) < 2:
                continue

            # Compute pairwise distances
            pairwise_distances = []
            for model1, model2 in itertools.combinations(sorted(predictions.keys()), 2):
                dist = compute_range_distance(predictions[model1], predictions[model2])
                pairwise_distances.append(dist)

            # Store results
            for dist in pairwise_distances:
                results[dataset_type]["distances"].append(dist)
                results[dataset_type]["by_dataset"][dataset].append(dist)
                results[dataset_type]["by_metric"][metric].append(dist)

            # Count high agreement (distance < 0.5)
            high_agreement = sum(1 for d in pairwise_distances if d < 0.5)
            total_pairs = len(pairwise_distances)
            results[dataset_type]["agreement_counts"].append(high_agreement / total_pairs if total_pairs > 0 else 0)

    return results


def compute_statistics(distances: List[float]) -> Dict:
    """Compute mean, median, std for a list of distances."""
    if not distances:
        return {}

    return {
        "mean": statistics.mean(distances),
        "median": statistics.median(distances),
        "stdev": statistics.stdev(distances) if len(distances) > 1 else 0,
        "min": min(distances),
        "max": max(distances),
        "count": len(distances)
    }


def save_plots_hq(fig, plots_dir: Path, name: str):
    """Save plot as both PDF (300dpi) and PNG (300dpi) for publication quality."""
    fig.savefig(str(plots_dir / f"{name}.pdf"), dpi=300, bbox_inches='tight', pad_inches=0.08, format='pdf')
    fig.savefig(str(plots_dir / f"{name}.png"), dpi=300, bbox_inches='tight', pad_inches=0.08, format='png')
    print(f"    ✓ {name}.pdf/png (300dpi)")


def print_results(results: Dict, aggregated: Dict):
    """Print consistency analysis results."""

    print("\n" + "="*70)
    print("MEMORIZATION CONSISTENCY CHECK")
    print("="*70)
    print("\nHypothesis: High model agreement (low distance) on memorized datasets")
    print("            vs low agreement (high distance) on novel datasets\n")

    # Overall comparison
    print("OVERALL CONSISTENCY")
    print("-" * 70)
    bench_stats = compute_statistics(results["benchmark"]["distances"])
    synth_stats = compute_statistics(results["synthetic"]["distances"])

    print(f"\nBenchmark Datasets (n={bench_stats['count']} pairwise distances):")
    print(f"  Mean distance:  {bench_stats['mean']:.4f}  (0=identical, 1=disjoint)")
    print(f"  Median distance: {bench_stats['median']:.4f}")
    print(f"  Std deviation:  {bench_stats['stdev']:.4f}")

    bench_agreement = statistics.mean(results["benchmark"]["agreement_counts"]) if results["benchmark"]["agreement_counts"] else 0
    print(f"  High agreement: {bench_agreement*100:.1f}% (distance < 0.5)")

    print(f"\nSynthetic Datasets (n={synth_stats['count']} pairwise distances):")
    print(f"  Mean distance:  {synth_stats['mean']:.4f}")
    print(f"  Median distance: {synth_stats['median']:.4f}")
    print(f"  Std deviation:  {synth_stats['stdev']:.4f}")

    synth_agreement = statistics.mean(results["synthetic"]["agreement_counts"]) if results["synthetic"]["agreement_counts"] else 0
    print(f"  High agreement: {synth_agreement*100:.1f}% (distance < 0.5)")

    diff = synth_stats['mean'] - bench_stats['mean']
    print(f"\nDistance Difference: {diff:.4f}")

    if diff > 0.1:
        print(f"✓ Evidence of memorization: Benchmark models agree {diff:.4f} better (lower distance)")
    else:
        print(f"✗ No memorization signal: Synthetic models actually agree {abs(diff):.4f} better")

    # Per-dataset consistency
    print("\n" + "-" * 70)
    print("PER-DATASET CONSISTENCY")
    print("-" * 70)

    dataset_consistency = []

    for dataset in sorted(set(list(results["benchmark"]["by_dataset"].keys()) +
                              list(results["synthetic"]["by_dataset"].keys()))):
        is_benchmark = dataset in BENCHMARK_DATASETS
        dataset_type = "benchmark" if is_benchmark else "synthetic"
        distances = results[dataset_type]["by_dataset"][dataset]

        if distances:
            mean_dist = statistics.mean(distances)
            agreement = sum(1 for d in distances if d < 0.5) / len(distances)
            dataset_consistency.append((dataset, is_benchmark, mean_dist, agreement, len(distances)))

    # Sort by type then by distance
    dataset_consistency.sort(key=lambda x: (not x[1], x[2]))

    print(f"\n{'Dataset':<15} {'Type':<10} {'Mean Dist':<12} {'Agree':<10} {'Samples'}")
    print("-" * 70)

    bench_dists = []
    synth_dists = []
    for dataset, is_bench, mean_dist, agreement, samples in dataset_consistency:
        dtype = "Bench" if is_bench else "Synth"
        print(f"{dataset:<15} {dtype:<10} {mean_dist:<12.4f} {agreement*100:>7.1f}% {samples:<10}")
        if is_bench:
            bench_dists.append(mean_dist)
        else:
            synth_dists.append(mean_dist)

    # Per-metric consistency
    print("\n" + "-" * 70)
    print("PER-METRIC CONSISTENCY")
    print("-" * 70)

    print(f"\n{'Metric':<15} {'Benchmark':<15} {'Synthetic':<15} {'Diff':<10}")
    print("-" * 70)

    for metric in METRICS:
        bench_dists = results["benchmark"]["by_metric"][metric]
        synth_dists = results["synthetic"]["by_metric"][metric]

        bench_mean = statistics.mean(bench_dists) if bench_dists else 0
        synth_mean = statistics.mean(synth_dists) if synth_dists else 0
        diff = synth_mean - bench_mean

        print(f"{metric:<15} {bench_mean:<15.4f} {synth_mean:<15.4f} {diff:<10.4f}")

    # Network size effect (synthetic)
    print("\n" + "-" * 70)
    print("SYNTHETIC NETWORK SIZE EFFECT")
    print("-" * 70)

    synthetic_by_size = defaultdict(list)
    for dataset_algo, data in aggregated.items():
        dataset = data["dataset"]
        if dataset not in SYNTHETIC_DATASETS:
            continue

        size = int(dataset.split("_")[1])

        for metric in METRICS:
            predictions = {}
            for model in LLM_MODELS:
                if model in data["llm_estimates"] and metric in data["llm_estimates"][model]:
                    metric_data = data["llm_estimates"][model][metric]
                    predictions[model] = (metric_data["lower"], metric_data["upper"])

            if len(predictions) < 2:
                continue

            for model1, model2 in itertools.combinations(sorted(predictions.keys()), 2):
                dist = compute_range_distance(predictions[model1], predictions[model2])
                synthetic_by_size[size].append(dist)

    print(f"\n{'Size':<10} {'Mean Distance':<15} {'Count':<10}")
    print("-" * 70)
    for size in sorted(synthetic_by_size.keys()):
        distances = synthetic_by_size[size]
        mean_dist = statistics.mean(distances)
        print(f"{size:<10} {mean_dist:<15.4f} {len(distances):<10}")

    print("\n" + "="*70)


def generate_plots(results: Dict, aggregated: Dict, output_dir: Path):
    """Generate and save plots for consistency analysis."""
    if not PLOTTING_AVAILABLE:
        print("Warning: matplotlib not available, skipping plots")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating plots in {output_dir}...")

    # 1. Overall consistency comparison
    fig, ax = plt.subplots(figsize=(4.1, 2.5))
    bench_stats = compute_statistics(results["benchmark"]["distances"])
    synth_stats = compute_statistics(results["synthetic"]["distances"])

    categories = ["Benchmark", "Synthetic"]
    means = [bench_stats["mean"], synth_stats["mean"]]
    stdevs = [bench_stats["stdev"], synth_stats["stdev"]]

    x = np.arange(len(categories))
    width = 0.35

    colors_dark = ['#0072B2', '#E69F00']
    bars = ax.bar(x, means, width, yerr=stdevs, capsize=4, color=colors_dark,
                   edgecolor='#333333', linewidth=1.2, alpha=0.85, error_kw={'linewidth': 1.5})

    ax.set_ylabel("Mean Model Distance")
    ax.set_title("Memorization Signal: Cross-Model Agreement", fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stdevs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2 - 0.06, height + 0.5,
                f'{mean:.1f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    save_plots_hq(fig, output_dir, "01_consistency_benchmark_vs_synthetic")
    plt.close()

    # 2. Per-dataset consistency
    fig, ax = plt.subplots(figsize=(4.5, 3.6))
    dataset_consistency = []

    for dataset in sorted(set(list(results["benchmark"]["by_dataset"].keys()) +
                              list(results["synthetic"]["by_dataset"].keys()))):
        is_benchmark = dataset in BENCHMARK_DATASETS
        dataset_type = "benchmark" if is_benchmark else "synthetic"
        distances = results[dataset_type]["by_dataset"][dataset]

        if distances:
            mean_dist = statistics.mean(distances)
            dataset_consistency.append((dataset, is_benchmark, mean_dist))

    dataset_consistency.sort(key=lambda x: x[2])

    datasets = [d[0] for d in dataset_consistency]
    distances = [d[2] for d in dataset_consistency]
    colors = ['#0072B2' if d[1] else '#E69F00' for d in dataset_consistency]

    # Scale distances to 0-1 range for cleaner axis
    max_dist = max(distances) if distances else 1
    normalized_distances = [d / max_dist for d in distances]

    bars = ax.barh(datasets, normalized_distances, color=colors, edgecolor='#333333',
                    linewidth=0.8, alpha=0.85, height=0.6)

    ax.set_xlabel("Mean Distance (0-1 scale)")
    ax.set_title("Per-Dataset Cross-Model Agreement", fontweight='bold')
    ax.set_xlim(0, 1.0)
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    # Add value labels (normalized to 0-1 scale)
    for bar, norm_dist, real_dist in zip(bars, normalized_distances, distances):
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height()/2.,
                f'{norm_dist:.2f}', ha='left', va='center', fontsize=9)

    plt.tight_layout()
    save_plots_hq(fig, output_dir, "02_consistency_per_dataset")
    plt.close()

    # 3. Network size effect
    fig, ax = plt.subplots(figsize=(4.1, 2.5))
    synthetic_by_size = defaultdict(list)
    for dataset_algo, data in aggregated.items():
        dataset = data["dataset"]
        if dataset not in SYNTHETIC_DATASETS:
            continue

        size = int(dataset.split("_")[1])

        for metric in METRICS:
            predictions = {}
            for model in LLM_MODELS:
                if model in data["llm_estimates"] and metric in data["llm_estimates"][model]:
                    metric_data = data["llm_estimates"][model][metric]
                    predictions[model] = (metric_data["lower"], metric_data["upper"])

            if len(predictions) < 2:
                continue

            for model1, model2 in itertools.combinations(sorted(predictions.keys()), 2):
                dist = compute_range_distance(predictions[model1], predictions[model2])
                synthetic_by_size[size].append(dist)

    sizes = sorted(synthetic_by_size.keys())
    means = [statistics.mean(synthetic_by_size[s]) for s in sizes]

    ax.plot(sizes, means, marker='o', linewidth=2, markersize=7, color='#D55E00', label='Mean Distance')
    ax.fill_between(sizes, means, alpha=0.2, color='#D55E00')

    ax.set_xlabel("Network Size (nodes)")
    ax.set_ylabel("Mean Cross-Model Distance")
    ax.set_title("Network Size Effect on Model Agreement", fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    save_plots_hq(fig, output_dir, "03_consistency_network_size_effect")
    plt.close()

    print(f"  ✓ Generated 3 plots (PDF + PNG)")


def save_results_json(results: Dict, aggregated: Dict, output_dir: Path):
    """Save analysis results in JSON format."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute overall statistics
    bench_stats = compute_statistics(results["benchmark"]["distances"])
    synth_stats = compute_statistics(results["synthetic"]["distances"])

    bench_agreement = statistics.mean(results["benchmark"]["agreement_counts"]) if results["benchmark"]["agreement_counts"] else 0
    synth_agreement = statistics.mean(results["synthetic"]["agreement_counts"]) if results["synthetic"]["agreement_counts"] else 0

    # Per-dataset stats
    per_dataset = {}
    for dataset in sorted(set(list(results["benchmark"]["by_dataset"].keys()) +
                              list(results["synthetic"]["by_dataset"].keys()))):
        is_benchmark = dataset in BENCHMARK_DATASETS
        dataset_type = "benchmark" if is_benchmark else "synthetic"
        distances = results[dataset_type]["by_dataset"][dataset]

        if distances:
            mean_dist = statistics.mean(distances)
            agreement = sum(1 for d in distances if d < 0.5) / len(distances)
            per_dataset[dataset] = {
                "type": dataset_type,
                "mean_distance": mean_dist,
                "high_agreement_pct": agreement * 100,
                "sample_count": len(distances)
            }

    # Per-metric stats
    per_metric = {}
    for metric in METRICS:
        bench_dists = results["benchmark"]["by_metric"][metric]
        synth_dists = results["synthetic"]["by_metric"][metric]

        bench_mean = statistics.mean(bench_dists) if bench_dists else 0
        synth_mean = statistics.mean(synth_dists) if synth_dists else 0

        per_metric[metric] = {
            "benchmark_mean_distance": bench_mean,
            "synthetic_mean_distance": synth_mean,
            "difference": synth_mean - bench_mean
        }

    # Network size effect
    synthetic_by_size = defaultdict(list)
    for dataset_algo, data in aggregated.items():
        dataset = data["dataset"]
        if dataset not in SYNTHETIC_DATASETS:
            continue

        size = int(dataset.split("_")[1])

        for metric in METRICS:
            predictions = {}
            for model in LLM_MODELS:
                if model in data["llm_estimates"] and metric in data["llm_estimates"][model]:
                    metric_data = data["llm_estimates"][model][metric]
                    predictions[model] = (metric_data["lower"], metric_data["upper"])

            if len(predictions) < 2:
                continue

            for model1, model2 in itertools.combinations(sorted(predictions.keys()), 2):
                dist = compute_range_distance(predictions[model1], predictions[model2])
                synthetic_by_size[size].append(dist)

    network_size_effect = {}
    for size in sorted(synthetic_by_size.keys()):
        distances = synthetic_by_size[size]
        network_size_effect[str(size)] = {
            "mean_distance": statistics.mean(distances),
            "sample_count": len(distances)
        }

    # Compile final results
    final_results = {
        "analysis": "memorization_consistency_check",
        "hypothesis": "LLMs show high cross-model agreement on memorized (benchmark) datasets vs low agreement on novel (synthetic) datasets",
        "overall": {
            "benchmark": bench_stats,
            "synthetic": synth_stats,
            "benchmark_high_agreement_pct": bench_agreement * 100,
            "synthetic_high_agreement_pct": synth_agreement * 100,
            "memorization_signal": bench_stats["mean"] < synth_stats["mean"]
        },
        "per_dataset": per_dataset,
        "per_metric": per_metric,
        "network_size_effect": network_size_effect
    }

    output_file = output_dir / "memorization_consistency_analysis_results.json"
    with open(output_file, "w") as f:
        json.dump(final_results, f, indent=2)

    print(f"✓ Saved results to {output_file}")


if __name__ == "__main__":
    results_dir = Path("src/llm/results")
    output_dir = Path("src/experiments/results")

    print("Loading aggregated LLM predictions...")
    aggregated = load_aggregated_ranges(results_dir)
    print(f"Loaded {len(aggregated)} dataset-algorithm combinations")

    print("Analyzing cross-model consistency...")
    results = analyze_consistency_by_dataset_type(aggregated)

    print_results(results, aggregated)

    print("\nGenerating plots...")
    generate_plots(results, aggregated, output_dir / "plots")

    print("Saving results to JSON...")
    save_results_json(results, aggregated, output_dir)
