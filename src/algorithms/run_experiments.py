#!/usr/bin/env python3
"""
Run all experiments from the paper with proper variance analysis.

This script reproduces all experiments but with 100 runs per algorithm
to establish proper confidence intervals.

"""

import warnings
warnings.filterwarnings('ignore', category=SyntaxWarning)

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import networkx as nx
import json
import time
import subprocess

# LLM and causal discovery libraries
import bnlearn as bn
from sklearn.preprocessing import LabelEncoder
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ConstraintBased.FCI import fci
from lingam import DirectLiNGAM
from notears_pytorch import notears_linear

# Add parent directory to path so imports work when run as: python run_experiments.py
sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================================
# Metrics Functions
# ============================================================================

def compute_metrics(true_graph, learned_graph):
    """Compute precision, recall, F1, and SHD between true and learned graphs."""
    # Make sure graphs are 2D numpy arrays
    true_graph = np.array(true_graph, dtype=int)
    learned_graph = np.array(learned_graph, dtype=int)
    
    # Ensure same dimensions
    n = max(true_graph.shape[0], learned_graph.shape[0])
    true_padded = np.zeros((n, n))
    learned_padded = np.zeros((n, n))
    
    true_padded[:true_graph.shape[0], :true_graph.shape[1]] = true_graph
    learned_padded[:learned_graph.shape[0], :learned_graph.shape[1]] = learned_graph
    
    true_edges = set(zip(*np.where(true_padded > 0)))
    learned_edges = set(zip(*np.where(learned_padded > 0)))
    
    # True positives, false positives, false negatives
    tp = len(true_edges & learned_edges)
    fp = len(learned_edges - true_edges)
    fn = len(true_edges - learned_edges)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Structural Hamming Distance
    shd = fp + fn
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'shd': shd,
        'n_learned_edges': len(learned_edges),
        'n_true_edges': len(true_edges)
    }


# ============================================================================
# Dataset Loaders
# ============================================================================

def load_bnlearn_network(name: str):
    """Load a benchmark network from local .bif files, no downloads."""

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
    
    # Generate synthetic data directly from DAG
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
# Algorithm Runners
# ============================================================================

def run_pc_multiple(data, true_graph, n_runs: int = 100):
    """Run PC algorithm multiple times and collect metrics."""
    results = []
    for run in range(n_runs):
        try:
            cg = pc(data.values)
            learned_graph = cg.G.graph
            metrics = compute_metrics(true_graph, learned_graph)
            metrics['run'] = run + 1
            results.append(metrics)
            if (run + 1) % 10 == 0:
                print(f"  Run {run+1}/{n_runs}: PC - F1={metrics['f1']:.4f}, SHD={metrics['shd']}")
        except Exception as e:
            print(f"  Run {run+1}/{n_runs}: PC failed - {str(e)[:50]}")
    
    return results


def run_lingam_multiple(data, true_graph, n_runs: int = 100):
    """Run LiNGAM algorithm multiple times and collect metrics."""
    results = []
    for run in range(n_runs):
        try:
            model = DirectLiNGAM()
            model.fit(data.values)
            learned_graph = (model.adjacency_matrix_ != 0).astype(int)
            metrics = compute_metrics(true_graph, learned_graph)
            metrics['run'] = run + 1
            results.append(metrics)
            if (run + 1) % 10 == 0:
                print(f"  Run {run+1}/{n_runs}: LiNGAM - F1={metrics['f1']:.4f}, SHD={metrics['shd']}")
        except Exception as e:
            print(f"  Run {run+1}/{n_runs}: LiNGAM failed - {str(e)[:50]}")
    
    return results


def run_fci_multiple(data, true_graph, n_runs: int = 100):
    """Run FCI algorithm multiple times and collect metrics."""
    results = []
    for run in range(n_runs):
        try:
            fg, _ = fci(data.values)
            learned_graph = fg.graph
            metrics = compute_metrics(true_graph, learned_graph)
            metrics['run'] = run + 1
            results.append(metrics)
            if (run + 1) % 10 == 0:
                print(f"  Run {run+1}/{n_runs}: FCI - F1={metrics['f1']:.4f}, SHD={metrics['shd']}")
        except Exception as e:
            print(f"  Run {run+1}/{n_runs}: FCI failed - {str(e)[:50]}")
    
    return results


def run_notears_multiple(data, true_graph, n_runs: int = 100):
    """Run NOTEARS algorithm multiple times and collect metrics."""
    results = []
    for run in range(n_runs):
        try:
            W_est = notears_linear(data.values, lambda1=0.1)
            learned_graph = (W_est != 0).astype(int)
            metrics = compute_metrics(true_graph, learned_graph)
            metrics['run'] = run + 1
            results.append(metrics)
            if (run + 1) % 10 == 0:
                print(f"  Run {run+1}/{n_runs}: NOTEARS - F1={metrics['f1']:.4f}, SHD={metrics['shd']}")
        except Exception as e:
            print(f"  Run {run+1}/{n_runs}: NOTEARS failed - {str(e)[:50]}")
    
    return results


# ============================================================================
# Experiment Runners
# ============================================================================

def run_all_algorithms_on_dataset(data, true_graph, dataset_name: str, output_dir: str, n_runs: int = 100, algorithms_to_run=None):
    """Run PC, LiNGAM, FCI, and NOTEARS algorithms on a single dataset."""
    
    all_algorithms = [
        ('pc', run_pc_multiple),
        ('lingam', run_lingam_multiple),
        ('fci', run_fci_multiple),
        ('notears', run_notears_multiple),
    ]
    
    # Filter algorithms if specified
    if algorithms_to_run is not None:
        algorithms = [(name, fn) for name, fn in all_algorithms if name in algorithms_to_run]
    else:
        algorithms = all_algorithms

    for algo_name, algo_fn in algorithms:
        print(f"\n--- Running {algo_name.upper()} Algorithm ---")
        try:
            results = algo_fn(data, true_graph, n_runs)
            
            if results:
                # Save results to JSON
                output_file = Path(output_dir) / f"{dataset_name}_{algo_name}_runs_{n_runs}_variance.json"
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)
                
                # Compute statistics
                precisions = [r['precision'] for r in results]
                f1s = [r['f1'] for r in results]
                shds = [r['shd'] for r in results]
                
                print(f"✓ {algo_name.upper()} completed on {dataset_name}")
                print(f"  Precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
                print(f"  F1:        {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
                print(f"  SHD:       {np.mean(shds):.1f} ± {np.std(shds):.1f}")
                print(f"  Saved to {output_file.name}")
            else:
                print(f"✗ {algo_name.upper()} produced no results on {dataset_name}")
        except Exception as e:
            print(f"✗ {algo_name.upper()} failed on {dataset_name}: {e}")


def run_benchmark_experiments(output_dir: str, n_runs: int = 100, datasets_to_run=None, algorithms_to_run=None):
    """Run benchmarks for specified datasets."""
    benchmarks = ['asia', 'alarm', 'sachs', 'survey', 'child', 'cancer', 'hepar2', 'earthquake', 'insurance']
    
    # If no specific datasets requested, run all benchmarks
    if datasets_to_run is None:
        datasets_to_run = set(benchmarks)
    
    # Filter to only requested benchmarks
    benchmarks_to_run = [b for b in benchmarks if b in datasets_to_run]

    for bench_name in benchmarks_to_run:
        print("\n" + "="*80)
        print(f"BENCHMARK: {bench_name.upper()}")
        print("="*80)

        try:
            data, true_graph, nodes = load_bnlearn_network(bench_name)
            print(f"Loaded {bench_name}: {len(nodes)} nodes, {data.shape[0]} samples")
            run_all_algorithms_on_dataset(data, true_graph, bench_name, output_dir, n_runs, algorithms_to_run)
        except Exception as e:
            print(f"Error processing {bench_name}: {e}")
            continue


def run_synthetic_experiments(output_dir: str, n_runs: int = 100, datasets_to_run=None, algorithms_to_run=None):
    """Run specified algorithms on synthetic DAGs."""
    synthetic_datasets = [('synthetic_12', 12), ('synthetic_30', 30), ('synthetic_50', 50), ('synthetic_60', 60)]
    
    if datasets_to_run:
        synthetic_datasets = [(name, nodes) for name, nodes in synthetic_datasets 
                             if name in datasets_to_run]
    
    if not synthetic_datasets:
        print("\nNo synthetic datasets requested, skipping synthetic experiments")
        return

    for dataset_name, n_nodes in synthetic_datasets:
        # Use edge_prob=0.2 for small/medium (12, 30 nodes), 0.3 for large (50, 60 nodes)
        edge_prob = 0.2 if n_nodes <= 30 else 0.3
        
        print("\n" + "="*80)
        print(f"LINEAR SYNTHETIC DAG - {n_nodes} NODES (edge_prob={edge_prob})")
        print("="*80)

        try:
            data, true_graph = generate_synthetic_dag(n_nodes, edge_prob=edge_prob)
            print(f"Generated DAG: {n_nodes} nodes, {int(np.sum(true_graph))} edges")
            run_all_algorithms_on_dataset(data, true_graph, dataset_name, output_dir, n_runs, algorithms_to_run)
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
            continue


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run all causal discovery experiments")
    parser.add_argument('--runs', type=int, default=100,
                       help='Number of runs per algorithm')
    parser.add_argument('--output', type=str, default=str(Path(__file__).parent.parent / 'results' / 'algorithms'),
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

    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory created/verified: {output_path}")
    
    print(f"Runs per algorithm: {args.runs}")
    print(f"Output directory: {args.output}")

    benchmark_datasets = {'asia', 'alarm', 'sachs', 'survey', 'child', 'cancer', 'hepar2', 'earthquake', 'insurance'}
    synthetic_datasets = {'synthetic_12', 'synthetic_30', 'synthetic_50', 'synthetic_60'}
    
    # Determine which datasets to run
    if 'all' in args.experiments:
        datasets_to_run = benchmark_datasets | synthetic_datasets
    else:
        datasets_to_run = set(args.experiments)
    
    # Run benchmark datasets
    if benchmark_datasets & datasets_to_run:
        run_benchmark_experiments(args.output, args.runs, datasets_to_run, args.algorithms)
    
    # Run synthetic datasets
    if synthetic_datasets & datasets_to_run:
        run_synthetic_experiments(args.output, args.runs, datasets_to_run, args.algorithms)

    print(f"\n{'='*80}")
    print(f"Results saved to: {args.output}/")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
