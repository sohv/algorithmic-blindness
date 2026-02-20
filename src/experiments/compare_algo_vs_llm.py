#!/usr/bin/env python3
"""
Compare Algorithmic Results vs LLM Predictions
===============================================
Creates a comprehensive comparison table from comparison_results.json
showing algorithmic results, LLM predictions, and coverage for each
dataset-algorithm-metric combination.

Usage:
    python compare_algo_vs_llm.py
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

BENCHMARK_DATASETS = {"asia", "cancer", "earthquake", "child", "alarm", "insurance", "survey", "sachs", "hepar2"}
SYNTHETIC_DATASETS = {"synthetic_12", "synthetic_30", "synthetic_50", "synthetic_60"}


def load_comparison_results(results_file: Path) -> Dict:
    """Load pre-computed comparison results."""
    with open(results_file) as f:
        data = json.load(f)
    return data


def create_comparison_table(comparison_data: Dict, output_dir: Path):
    """Create comparison table from comparison_results.json."""

    rows = []

    for exp_key, exp_data in comparison_data.items():
        dataset = exp_data.get("dataset")
        algorithm = exp_data.get("algorithm")
        dataset_type = "Benchmark" if dataset in BENCHMARK_DATASETS else "Synthetic"

        # Aggregate across all LLM models for each metric
        metrics_data = {}
        models = exp_data.get("models", {})

        for metric in ["precision", "recall", "f1", "shd"]:
            # Collect data across all models for this metric
            llm_lowers = []
            llm_uppers = []
            algo_vals = []
            coverage_results = []

            for model_name, model_metrics in models.items():
                if metric in model_metrics:
                    metric_info = model_metrics[metric]
                    llm_range = metric_info.get("llm_range")
                    algo_val = metric_info.get("algorithmic_mean")
                    covered = metric_info.get("calibrated_coverage")

                    if llm_range:
                        llm_lowers.append(llm_range[0])
                        llm_uppers.append(llm_range[1])
                    if algo_val is not None:
                        algo_vals.append(algo_val)
                    if covered is not None:
                        coverage_results.append(covered)

            # Average across models
            if algo_vals and llm_lowers and llm_uppers:
                algo_mean = sum(algo_vals) / len(algo_vals)
                llm_lower_mean = sum(llm_lowers) / len(llm_lowers)
                llm_upper_mean = sum(llm_uppers) / len(llm_uppers)
                coverage_pct = (sum(coverage_results) / len(coverage_results) * 100) if coverage_results else None

                row = {
                    "Dataset": dataset,
                    "Dataset_Type": dataset_type,
                    "Algorithm": algorithm,
                    "Metric": metric,
                    "Algo_Mean": algo_mean,
                    "LLM_Lower_Avg": llm_lower_mean,
                    "LLM_Upper_Avg": llm_upper_mean,
                    "Num_Models": len(models),
                    "Coverage_Pct": coverage_pct
                }
                rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Save to CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "algo_vs_llm_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved comparison to {csv_path}")

    # Save to JSON
    json_path = output_dir / "algo_vs_llm_comparison.json"
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"✓ Saved JSON to {json_path}")

    return df, rows


def print_summary(df: pd.DataFrame, rows: List[Dict]):
    """Print summary statistics."""

    print("\n" + "="*90)
    print("ALGORITHM VS LLM COMPARISON SUMMARY")
    print("="*90)

    print(f"\nTotal Comparisons: {len(rows)}")
    print(f"Unique Datasets: {df['Dataset'].nunique()}")
    print(f"Unique Algorithms: {df['Algorithm'].nunique()}")

    # Average coverage by metric
    print("\n" + "-"*90)
    print("AVERAGE LLM COVERAGE BY METRIC:")
    print("-"*90)
    for metric in sorted(df['Metric'].unique()):
        metric_df = df[df['Metric'] == metric]
        avg_coverage = metric_df['Coverage_Pct'].mean()
        print(f"  {metric.upper():<12} {avg_coverage:>6.1f}%")

    # Coverage by algorithm
    print("\n" + "-"*90)
    print("AVERAGE LLM COVERAGE BY ALGORITHM:")
    print("-"*90)
    for algo in sorted(df['Algorithm'].unique()):
        algo_df = df[df['Algorithm'] == algo]
        avg_coverage = algo_df['Coverage_Pct'].mean()
        print(f"  {algo.upper():<12} {avg_coverage:>6.1f}%")

    # Coverage by dataset type
    print("\n" + "-"*90)
    print("AVERAGE LLM COVERAGE BY DATASET TYPE:")
    print("-"*90)
    for dtype in ["Benchmark", "Synthetic"]:
        dtype_df = df[df['Dataset_Type'] == dtype]
        avg_coverage = dtype_df['Coverage_Pct'].mean()
        print(f"  {dtype:<12} {avg_coverage:>6.1f}%")

    # Sample rows
    print("\n" + "-"*90)
    print("SAMPLE COMPARISONS (First 10 rows):")
    print("-"*90)
    sample_df = df.head(10).copy()
    sample_df["Algo_Mean"] = sample_df["Algo_Mean"].apply(lambda x: f"{x:.3f}")
    sample_df["LLM_Lower_Avg"] = sample_df["LLM_Lower_Avg"].apply(lambda x: f"{x:.3f}")
    sample_df["LLM_Upper_Avg"] = sample_df["LLM_Upper_Avg"].apply(lambda x: f"{x:.3f}")
    sample_df["Coverage_Pct"] = sample_df["Coverage_Pct"].apply(lambda x: f"{x:.1f}%" if x is not None else "N/A")

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(sample_df.to_string(index=False))

    print("\n" + "="*90)


if __name__ == "__main__":
    # Use absolute paths that work from any directory
    script_dir = Path(__file__).parent
    comparison_file = script_dir / "../llm/results/comparisons/comparison_results.json"
    output_dir = script_dir / "results/comparisons"

    print("Loading comparison results...")
    comparison_data = load_comparison_results(comparison_file)
    print(f"✓ Loaded {len(comparison_data)} dataset-algorithm combinations")

    print("Creating comparison table...")
    df, rows = create_comparison_table(comparison_data, output_dir)
    print(f"✓ Created table with {len(rows)} rows")

    print_summary(df, rows)

    print(f"\n✓ Results saved to {output_dir}/")
