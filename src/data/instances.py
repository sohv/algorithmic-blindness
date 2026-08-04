# builds the arrays and weighted graphs the sorting and shortest-path domains run on.

import logging

import numpy as np

from src.domains import ARRAY_SIZES, GRAPH_DENSITIES, GRAPH_SIZES

LOGGER = logging.getLogger(__name__)

# few enough distinct values that a naive quicksort partition degrades badly
N_UNIQUE_VALUES = 10


def build_array(instance: str, variant: str, seed: int = 42) -> list[int]:
    """An array of integers whose order is set by the variant.

    `sorted` and `reverse` are the classic quicksort worst cases under a naive pivot; `few_unique`
    is the duplicate-heavy case. Mergesort is near-indifferent to all three, which is exactly the
    contrast a predictor has to get right.
    """
    assert instance in ARRAY_SIZES, f"{instance} is not a sorting instance"
    n = ARRAY_SIZES[instance]
    rng = np.random.default_rng(seed)

    if variant == "random":
        values = rng.integers(0, 10 * n, size=n)
    elif variant == "sorted":
        values = np.sort(rng.integers(0, 10 * n, size=n))
    elif variant == "reverse":
        values = np.sort(rng.integers(0, 10 * n, size=n))[::-1]
    elif variant == "few_unique":
        values = rng.integers(0, N_UNIQUE_VALUES, size=n)
    else:
        raise ValueError(f"unknown array distribution {variant}")

    LOGGER.info(f"built {instance}/{variant}: {n} integers")
    return [int(v) for v in values]


def build_graph(instance: str, variant: str, seed: int = 42) -> tuple[int, list[tuple[int, int, float]]]:
    """A weighted directed graph as (n_nodes, edges), each edge (source, target, weight).

    Weights are strictly positive so Dijkstra is valid; the two algorithms then solve exactly the
    same problem and differ only in how much work they do to get there.
    """
    assert instance in GRAPH_SIZES, f"{instance} is not a shortest-path instance"
    assert variant in GRAPH_DENSITIES, f"unknown graph density {variant}"

    n = GRAPH_SIZES[instance]
    mean_out_degree = GRAPH_DENSITIES[variant]
    rng = np.random.default_rng(seed)

    n_edges = int(n * mean_out_degree)
    sources = rng.integers(0, n, size=n_edges)
    targets = rng.integers(0, n, size=n_edges)
    weights = rng.uniform(1.0, 100.0, size=n_edges)

    edges = [(int(s), int(t), float(w)) for s, t, w in zip(sources, targets, weights) if s != t]

    # a spanning path keeps every node reachable, so no node is silently free to visit
    order = rng.permutation(n)
    edges.extend((int(order[i]), int(order[i + 1]), float(rng.uniform(1.0, 100.0))) for i in range(n - 1))

    LOGGER.info(f"built {instance}/{variant}: {n} nodes, {len(edges)} edges")
    return n, edges


def adjacency(n: int, edges: list[tuple[int, int, float]]) -> list[list[tuple[int, float]]]:
    """Adjacency list, built once and shared by both shortest-path algorithms."""
    graph: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for source, target, weight in edges:
        graph[source].append((target, weight))
    return graph
