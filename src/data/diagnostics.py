# per-domain instance properties, exposed by the richer prompt levels.

import logging
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats

from src.conditions import Condition
from src.data.instances import adjacency, build_array, build_graph
from src.domains import CAUSAL, SHORTEST_PATH, SORTING

LOGGER = logging.getLogger(__name__)

# LiNGAM identifies structure from departure from gaussianity, so near-zero means its assumption is unmet
GAUSSIAN_KURTOSIS_TOLERANCE = 0.3

# sampled rather than counted exactly; enough to tell a sorted array from a random one
INVERSION_SAMPLE = 20000


@dataclass
class DatasetDiagnostics:
    """Properties of the data itself, computable before any algorithm is run."""

    n_samples: int
    n_variables: int
    samples_per_variable: float
    mean_abs_excess_kurtosis: float
    mean_abs_skew: float
    non_gaussianity_verdict: str
    mean_abs_correlation: float
    max_abs_correlation: float
    condition_number: float
    n_discrete_variables: int
    data_type: str


@dataclass
class GraphDiagnostics:
    """Properties of the true graph, for the causal domain."""

    n_edges: int
    edge_density: float
    mean_degree: float
    max_in_degree: float
    max_out_degree: float


def non_gaussianity(excess_kurtosis: float, skew: float) -> str:
    """Plain-language verdict on whether LiNGAM's assumption holds, not just the raw statistic."""
    if excess_kurtosis < GAUSSIAN_KURTOSIS_TOLERANCE and skew < GAUSSIAN_KURTOSIS_TOLERANCE:
        return "approximately Gaussian"
    if excess_kurtosis < 1.0 and skew < 1.0:
        return "mildly non-Gaussian"
    return "clearly non-Gaussian"


def compute_dataset_diagnostics(data: pd.DataFrame) -> DatasetDiagnostics:
    """Sample adequacy, non-Gaussianity and collinearity, computed from the data alone."""
    values = data.to_numpy(dtype=float)
    n_rows, n_cols = values.shape

    excess_kurtosis = float(np.mean(np.abs(stats.kurtosis(values, axis=0, fisher=True, bias=False))))
    skew = float(np.mean(np.abs(stats.skew(values, axis=0, bias=False))))

    correlation = np.corrcoef(values, rowvar=False)
    off_diagonal = correlation[~np.eye(n_cols, dtype=bool)]
    # a singular design matrix makes every conditional independence test unstable
    condition_number = float(np.linalg.cond(correlation)) if n_cols > 1 else 1.0

    n_discrete = int(sum(data[col].nunique() <= 10 for col in data.columns))
    data_type = "discrete" if n_discrete == n_cols else "continuous" if n_discrete == 0 else "mixed"

    return DatasetDiagnostics(
        n_samples=n_rows,
        n_variables=n_cols,
        samples_per_variable=n_rows / n_cols,
        mean_abs_excess_kurtosis=excess_kurtosis,
        mean_abs_skew=skew,
        non_gaussianity_verdict=non_gaussianity(excess_kurtosis, skew),
        mean_abs_correlation=float(np.mean(np.abs(off_diagonal))) if n_cols > 1 else 0.0,
        max_abs_correlation=float(np.max(np.abs(off_diagonal))) if n_cols > 1 else 0.0,
        condition_number=condition_number,
        n_discrete_variables=n_discrete,
        data_type=data_type,
    )


def compute_graph_diagnostics(true_graph) -> GraphDiagnostics:
    """Edge count and degree structure of the ground-truth DAG."""
    graph = np.asarray(true_graph)
    n_nodes = graph.shape[0]
    n_edges = int(graph.sum())
    max_possible = n_nodes * (n_nodes - 1) // 2

    return GraphDiagnostics(
        n_edges=n_edges,
        edge_density=n_edges / max_possible if max_possible else 0.0,
        mean_degree=2 * n_edges / n_nodes if n_nodes else 0.0,
        max_in_degree=float(graph.sum(axis=0).max()) if n_nodes else 0.0,
        max_out_degree=float(graph.sum(axis=1).max()) if n_nodes else 0.0,
    )


def sorting_diagnostics(values: list[int]) -> dict:
    """Sortedness and duplication — what decides whether quicksort degenerates.

    `sortedness` is the share of adjacent pairs already in order: 1.0 for a sorted array, 0.0 for a
    reversed one, about 0.5 for random. It is the single most predictive property in this domain and
    the original prompt would not have carried it at all.
    """
    array = np.asarray(values)
    n = len(array)
    ordered_pairs = int(np.sum(array[:-1] <= array[1:])) if n > 1 else 0

    sample = array[:INVERSION_SAMPLE]
    # inversions among a prefix, as a share of the pairs in that prefix
    inversions = int(
        sum(np.sum(sample[i + 1 :] < sample[i]) for i in range(0, len(sample), max(1, len(sample) // 200)))
    )
    sampled_positions = len(range(0, len(sample), max(1, len(sample) // 200)))

    n_unique = len(np.unique(array))
    return {
        "n_elements": n,
        "n_unique_values": n_unique,
        "duplicate_rate": 1.0 - n_unique / n if n else 0.0,
        "sortedness": ordered_pairs / (n - 1) if n > 1 else 1.0,
        "mean_inversions_per_element": inversions / sampled_positions if sampled_positions else 0.0,
        "value_range": int(array.max() - array.min()) if n else 0,
    }


def shortest_path_diagnostics(n_nodes: int, edges: list[tuple[int, int, float]]) -> dict:
    """Size and shape of the graph — all either algorithm's operation count depends on.

    `max_shortest_path_hops` is the one that decides Bellman-Ford's pass count, and therefore the
    gap between the two algorithms. Without it the gap is not predictable from V and E alone.
    """
    from src.algorithms.shortest_path import max_shortest_path_hops

    out_degrees = np.zeros(n_nodes, dtype=int)
    for source, _, _ in edges:
        out_degrees[source] += 1
    weights = np.array([w for _, _, w in edges], dtype=float)

    return {
        "n_nodes": n_nodes,
        "n_edges": len(edges),
        "mean_out_degree": float(out_degrees.mean()),
        "max_out_degree": int(out_degrees.max()) if n_nodes else 0,
        "n_isolated_nodes": int((out_degrees == 0).sum()),
        "edge_density": len(edges) / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0.0,
        "mean_edge_weight": float(weights.mean()) if len(weights) else 0.0,
        "min_edge_weight": float(weights.min()) if len(weights) else 0.0,
        "max_shortest_path_hops": max_shortest_path_hops(n_nodes, adjacency(n_nodes, edges)),
    }


def diagnostics_for(condition: Condition, data_dir: str = "data/raw", seed: int = 42) -> dict:
    """One flat record per (domain, instance, variant), written once and read by the prompt builder."""
    base = {
        "id": f"{condition.domain}__{condition.instance}__{condition.variant}",
        "domain": condition.domain,
        "instance": condition.instance,
        "variant": condition.variant,
    }

    if condition.domain == CAUSAL:
        from src.algorithms.runner import load_causal_data

        data, true_graph = load_causal_data(condition, data_dir, seed)
        return {**base, **asdict(compute_dataset_diagnostics(data)), **asdict(compute_graph_diagnostics(true_graph))}

    if condition.domain == SORTING:
        return {**base, **sorting_diagnostics(build_array(condition.instance, condition.variant, seed + 1))}

    if condition.domain == SHORTEST_PATH:
        n_nodes, edges = build_graph(condition.instance, condition.variant, seed + 1)
        return {**base, **shortest_path_diagnostics(n_nodes, edges)}

    raise ValueError(f"no diagnostics for domain {condition.domain}")
