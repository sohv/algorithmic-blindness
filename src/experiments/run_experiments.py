#!/usr/bin/env python3
"""
Run all experiments from the paper with proper variance analysis.

This script reproduces all experiments but with 100 runs per algorithm
to establish proper confidence intervals.

Coverage:
- 9 Benchmark datasets: Asia, Alarm, Sachs, Survey, Child, Cancer, Hepar2, 
  Earthquake, Insurance
- 4 Linear synthetic datasets: Synthetic-12, Synthetic-30, Synthetic-50, Synthetic-60 (held-out OOD)
- 4 algorithms: PC, LiNGAM, FCI, NOTEARS
- Total: 14 datasets x 4 algorithms x 100 runs = 5,600 algorithmic runs
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import networkx as nx
import json

# Add parent directory to path so imports work when run as: python run_experiments.py
sys.path.insert(0, str(Path(__file__).parent.parent))

from algorithms.variance_analysis import VarianceAnalyzer

# Note: All benchmark datasets (Asia, Alarm, Sachs, Survey, Child, Cancer, Hepar2, Earthquake, Insurance)
# are loaded from local .bif files using bnlearn - no module imports needed

# ============================================================================
# Dataset Loaders
# ============================================================================

def load_bnlearn_network(name: str):
    """Load a benchmark network from local .bif files, no downloads."""
    try:
        import bnlearn as bn
    except ImportError:
        print("Installing bnlearn...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'bnlearn'])
        import bnlearn as bn
    
    from sklearn.preprocessing import LabelEncoder

    # Map network names to bnlearn dataset names
    network_map = {
        'asia': 'asia',
        'alarm': 'alarm', 
        'sachs': 'sachs',
        'survey': 'survey',
        'child': 'child',
        'cancer': 'cancer',
        'hepar2': 'hepar2',
        'earthquake': 'earthquake',
        'insurance': 'insurance'
    }

    if name.lower() not in network_map:
        raise ValueError(f"Unknown network: {name}")

    dataset_name = network_map[name.lower()]
    
    # Load from local datasets folder
    project_root = Path(__file__).parent.parent.parent
    local_bif_file = project_root / 'datasets' / f'{dataset_name}.bif'
    
    if not local_bif_file.exists():
        raise FileNotFoundError(
            f"Local BIF file not found: {local_bif_file}\n"
            f"Please ensure {dataset_name}.bif exists in datasets/ folder"
        )
    
    print(f"Loading {dataset_name} from local BIF file: {local_bif_file}")
    
    # Import DAG from BIF file
    dag = bn.import_DAG(str(local_bif_file))
    
    # Extract node names in sorted order
    nodes = sorted(dag['model'].nodes())
    print(f"Loaded DAG: {len(nodes)} nodes")
    
    # Generate synthetic data directly from DAG (no downloads, no datazets)
    print(f"Generating data from DAG...")
    data = bn.sampling(dag, n=10000)
    
    # Handle different output formats from bn.sampling
    if hasattr(data, '__getitem__') and 'df' in data:
        data = data['df']
    elif not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data, columns=nodes)
    
    print(f"Generated data: {data.shape[0]} samples, {data.shape[1]} variables")
    
    # Ensure columns match node names from DAG
    if list(data.columns) != nodes:
        # Reorder data columns to match DAG node order
        data = data[nodes]
    
    # Encode all categorical/string columns to numeric
    for col in data.columns:
        if data[col].dtype == 'object' or str(data[col].dtype) == 'category':
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
    
    # Ensure all data is numeric
    data = data.apply(pd.to_numeric, errors='coerce')
    data = data.dropna()
    data = data.astype(np.float64)
    
    # Extract true graph as adjacency matrix
    n = len(nodes)
    true_graph = np.zeros((n, n))
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    for edge in dag['model'].edges():
        i = node_to_idx[edge[0]]
        j = node_to_idx[edge[1]]
        true_graph[i, j] = 1
    
    print(f"Extracted graph: {int(np.sum(true_graph))} edges")

    return data, true_graph, nodes


def generate_synthetic_dag(n_nodes: int, edge_prob: float = 0.2, seed: int = 42):
    """Generate synthetic DAG with linear Gaussian data."""
    np.random.seed(seed)
    
    # Generate random DAG
    G = nx.DiGraph()
    G.add_nodes_from(range(n_nodes))
    
    # Add edges respecting topological order
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if np.random.rand() < edge_prob:
                G.add_edge(i, j)
    
    # Convert to adjacency matrix
    true_graph = nx.to_numpy_array(G, dtype=int)
    
    # Generate data using structural equation model
    n_samples = 1000
    data = np.zeros((n_samples, n_nodes))
    
    # Topological sort to generate data in causal order
    topo_order = list(nx.topological_sort(G))
    
    for node in topo_order:
        parents = list(G.predecessors(node))
        
        if len(parents) == 0:
            # Exogenous variable
            data[:, node] = np.random.randn(n_samples)
        else:
            # Linear combination of parents + noise
            weights = np.random.randn(len(parents))
            data[:, node] = data[:, parents] @ weights + np.random.randn(n_samples)
    
    df = pd.DataFrame(data, columns=[f"X{i}" for i in range(n_nodes)])
    
    return df, true_graph


# ============================================================================
# Experiment Runners
# ============================================================================

def run_all_algorithms_on_dataset(analyzer: VarianceAnalyzer, data, true_graph, dataset_name: str, all_results: dict, algorithms_to_run=None):
    """Run PC, LiNGAM, FCI, and NOTEARS algorithms on a single dataset.
    
    Args:
        algorithms_to_run: List of algorithm names to run (e.g., ['fci', 'notears']). If None, runs all.
    """
    all_algorithms = [
        ('pc', analyzer.run_pc_multiple),
        ('lingam', analyzer.run_lingam_multiple),
        ('fci', analyzer.run_fci_multiple),
        ('notears', analyzer.run_notears_multiple),
    ]
    
    # Filter algorithms if specified
    if algorithms_to_run is not None:
        algorithms = [(name, fn) for name, fn in all_algorithms if name in algorithms_to_run]
    else:
        algorithms = all_algorithms

    for algo_name, algo_fn in algorithms:
        print(f"\n--- Running {algo_name.upper()} Algorithm ---")
        try:
            results = algo_fn(data, true_graph)
            print(f"{algo_name.upper()} - Precision: {results.precision.mean:.4f} "
                  f"[{results.precision.ci_lower:.4f}, {results.precision.ci_upper:.4f}]")
            print(f"{algo_name.upper()} - Recall:    {results.recall.mean:.4f} "
                  f"[{results.recall.ci_lower:.4f}, {results.recall.ci_upper:.4f}]")
            print(f"{algo_name.upper()} - F1:        {results.f1.mean:.4f} "
                  f"[{results.f1.ci_lower:.4f}, {results.f1.ci_upper:.4f}]")
            print(f"{algo_name.upper()} - SHD:       {results.shd.mean:.1f} "
                  f"[{results.shd.ci_lower:.1f}, {results.shd.ci_upper:.1f}]")
            analyzer.save_results(results, dataset_name, algo_name)
            all_results[f"{dataset_name}_{algo_name}"] = results
        except Exception as e:
            print(f"{algo_name.upper()} failed on {dataset_name}: {e}")

    return all_results


def run_benchmark_experiments(analyzer: VarianceAnalyzer, datasets_to_run=None, algorithms_to_run=None):
    """Run benchmarks for specified datasets."""
    benchmarks = ['asia', 'alarm', 'sachs', 'survey', 'child', 'cancer', 'hepar2', 'earthquake', 'insurance']
    
    # If no specific datasets requested, run all benchmarks
    if datasets_to_run is None:
        datasets_to_run = set(benchmarks)
    
    # Filter to only requested benchmarks
    benchmarks_to_run = [b for b in benchmarks if b in datasets_to_run]

    all_results = {}

    for bench_name in benchmarks_to_run:
        print("\n" + "="*80)
        print(f"BENCHMARK: {bench_name.upper()}")
        print("="*80)

        try:
            data, true_graph, nodes = load_bnlearn_network(bench_name)
            print(f"Loaded {bench_name}: {len(nodes)} nodes, {data.shape[0]} samples")

            run_all_algorithms_on_dataset(analyzer, data, true_graph, bench_name, all_results, algorithms_to_run)

        except Exception as e:
            print(f"Error processing {bench_name}: {e}")
            continue

    return all_results


def run_synthetic_experiments(analyzer: VarianceAnalyzer, datasets_to_run=None, algorithms_to_run=None):
    """Run specified algorithms on synthetic DAGs."""
    synthetic_datasets = [('synthetic_12', 12), ('synthetic_30', 30), ('synthetic_50', 50), ('synthetic_60', 60)]
    
    if datasets_to_run:
        synthetic_datasets = [(name, nodes) for name, nodes in synthetic_datasets 
                             if name in datasets_to_run]
    
    if not synthetic_datasets:
        print("\nNo synthetic datasets requested, skipping synthetic experiments")
        return {}

    all_results = {}

    for dataset_name, n_nodes in synthetic_datasets:
        # Use edge_prob=0.2 for small/medium (12, 30 nodes), 0.3 for large (50, 60 nodes)
        edge_prob = 0.2 if n_nodes <= 30 else 0.3
        
        print("\n" + "="*80)
        print(f"LINEAR SYNTHETIC DAG - {n_nodes} NODES (held-out OOD test, edge_prob={edge_prob})")
        print("="*80)

        data, true_graph = generate_synthetic_dag(n_nodes, edge_prob=edge_prob)
        print(f"Generated DAG: {n_nodes} nodes, {np.sum(true_graph)} edges")

        dataset_name = f'synthetic_{n_nodes}'
        run_all_algorithms_on_dataset(analyzer, data, true_graph, dataset_name, all_results, algorithms_to_run)

    return all_results


def run_new_datasets_experiments(analyzer: VarianceAnalyzer):
    """Run ALL algorithms on datasets: Alarm, Insurance (for benchmarking)."""

    if not NEW_DATASETS_AVAILABLE:
        print("\nSkipping new datasets (import failed)")
        return {}

    all_results = {}

    # ========================================================================
    # Alarm Network (Medical, 37 nodes)
    # ========================================================================
    print("\n" + "="*80)
    print("ALARM NETWORK - Medical ICU Monitoring (37 nodes)")
    print("="*80)

    try:
        data, true_graph, node_names = load_alarm(n_samples=5000)
        print(f"Loaded Alarm: {len(node_names)} nodes, {np.sum(true_graph)} edges")
        run_all_algorithms_on_dataset(analyzer, data, true_graph, 'alarm', all_results)
    except Exception as e:
        print(f"Error with Alarm network: {e}")

    # ========================================================================
    # Stock Market (Finance, 10 nodes)
    # ========================================================================
    print("\n" + "="*80)
    print("STOCK MARKET - Financial Causal Relationships (10 nodes)")
    print("="*80)

    try:
        data, true_graph, var_names = load_stock_market(n_samples=1000)
        print(f"Loaded Stock Market: {len(var_names)} variables, {np.sum(true_graph)} edges")
        run_all_algorithms_on_dataset(analyzer, data, true_graph, 'stock_market', all_results)
    except Exception as e:
        print(f"Error with Stock Market dataset: {e}")

    # ========================================================================
    # Insurance Network (Insurance, 27 nodes)
    # ========================================================================
    print("\n" + "="*80)
    print("INSURANCE NETWORK - Risk Assessment (27 nodes)")
    print("="*80)

    try:
        data, true_graph, node_names = load_insurance(n_samples=2000)
        print(f"Loaded Insurance: {len(node_names)} nodes, {np.sum(true_graph)} edges")
        run_all_algorithms_on_dataset(analyzer, data, true_graph, 'insurance', all_results)
    except Exception as e:
        print(f"Error with Insurance network: {e}")

    return all_results


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run all causal discovery experiments")
    parser.add_argument('--runs', type=int, default=100,
                       help='Number of runs per algorithm')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory')
    parser.add_argument('--experiments', nargs='+',
                       choices=['asia', 'alarm', 'sachs', 'survey', 'child', 'cancer', 'hepar2', 'earthquake', 'insurance', 'synthetic_12', 'synthetic_30', 'synthetic_50', 'synthetic_60', 'all'],
                       default=['all'],
                       help='Individual datasets to run (or "all" for everything)')
    parser.add_argument('--algorithms', nargs='+',
                       choices=['pc', 'lingam', 'fci', 'notears'],
                       default=None,
                       help='Algorithms to run (e.g., fci notears). If not specified, runs all algorithms')

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = VarianceAnalyzer(n_runs=args.runs, output_dir=args.output)

    print(f"Runs per algorithm: {args.runs}")
    print(f"Output directory: {args.output}")
    results = {}

    # Map of individual datasets to their runner functions
    benchmark_datasets = {'asia', 'alarm', 'sachs', 'survey', 'child', 'cancer', 'hepar2', 'earthquake', 'insurance'}
    synthetic_datasets = {'synthetic_12', 'synthetic_30', 'synthetic_50', 'synthetic_60'}
    
    # Determine which datasets to run
    if 'all' in args.experiments:
        datasets_to_run = benchmark_datasets | synthetic_datasets
    else:
        datasets_to_run = set(args.experiments)
    
    # Run benchmark datasets
    if benchmark_datasets & datasets_to_run:
        benchmark_results = run_benchmark_experiments(analyzer, datasets_to_run, args.algorithms)
        results.update(benchmark_results)
    
    # Run synthetic datasets
    if synthetic_datasets & datasets_to_run:
        synthetic_results = run_synthetic_experiments(analyzer, datasets_to_run, args.algorithms)
        results.update(synthetic_results)

    print(f"Results saved to: {args.output}/")
    print(f"\nTotal experiments run: {sum(len(v) if isinstance(v, dict) else 1 for v in results.values())}")
    return results


if __name__ == "__main__":
    main()
