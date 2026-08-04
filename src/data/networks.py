# generates the synthetic causal DAGs, one noise family at a time.

import logging

import networkx as nx
import numpy as np
import pandas as pd

from src.domains import SYNTHETIC_NODES, n_samples

LOGGER = logging.getLogger(__name__)

# all scaled to unit variance, so only the distribution shape varies and the scale cannot confound
NOISE_SAMPLERS = {
    "gaussian": lambda rng, n: rng.standard_normal(n),
    "uniform": lambda rng, n: rng.uniform(-np.sqrt(3), np.sqrt(3), n),
    "laplace": lambda rng, n: rng.laplace(0, 1 / np.sqrt(2), n),
    "exponential": lambda rng, n: rng.exponential(1.0, n) - 1.0,
}


def synthetic_edge_prob(n_nodes: int) -> float:
    """Sparser wiring for large graphs so edge density stays comparable across sizes."""
    return 0.2 if n_nodes <= 30 else 0.3


def random_dag(n_nodes: int, edge_prob: float, rng: np.random.Generator) -> nx.DiGraph:
    """Erdos-Renyi over a fixed topological order, so the result is acyclic by construction."""
    graph = nx.DiGraph()
    graph.add_nodes_from(range(n_nodes))
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if rng.random() < edge_prob:
                graph.add_edge(i, j)
    return graph


def generate_synthetic_dag(
    n_nodes: int,
    noise: str = "gaussian",
    edge_prob: float | None = None,
    n_rows: int = 1000,
    seed: int = 42,
):
    """Random DAG plus linear structural equations with the named noise family.

    Returns (data, true_graph). The graph depends only on (n_nodes, edge_prob, seed), so the same
    structure is reused across noise families and the noise is the only thing that varies.
    """
    assert noise in NOISE_SAMPLERS, f"unknown noise family {noise}, expected one of {sorted(NOISE_SAMPLERS)}"
    edge_prob = synthetic_edge_prob(n_nodes) if edge_prob is None else edge_prob

    # the graph is drawn from its own stream so it stays identical across noise families
    graph = random_dag(n_nodes, edge_prob, np.random.default_rng(seed))
    rng = np.random.default_rng(seed + 1)
    sample = NOISE_SAMPLERS[noise]

    data = np.zeros((n_rows, n_nodes))
    for node in nx.topological_sort(graph):
        parents = list(graph.predecessors(node))
        disturbance = sample(rng, n_rows)
        if not parents:
            data[:, node] = disturbance
        else:
            # weights bounded away from zero so a parent's influence is detectable at all
            weights = rng.uniform(0.5, 1.5, len(parents)) * rng.choice([-1, 1], len(parents))
            data[:, node] = data[:, parents] @ weights + disturbance

    true_graph = nx.to_numpy_array(graph, dtype=int)
    df = pd.DataFrame(data, columns=[f"X{i}" for i in range(n_nodes)])
    LOGGER.info(f"synthetic dag: {n_nodes} nodes, {int(true_graph.sum())} edges, {noise} noise, edge_prob={edge_prob}")
    return df, true_graph


def load_synthetic(dataset: str, noise: str, seed: int = 42):
    """Build the synthetic dataset named in the condition grid."""
    assert dataset in SYNTHETIC_NODES, f"{dataset} is not a synthetic dataset"
    return generate_synthetic_dag(SYNTHETIC_NODES[dataset], noise=noise, n_rows=n_samples(dataset), seed=seed)
