# compares a learned adjacency matrix against the ground-truth DAG.

import logging

import numpy as np

LOGGER = logging.getLogger(__name__)


def compute_metrics(true_graph, learned_graph) -> dict:
    """Precision, recall, F1 and SHD between a true and a learned adjacency matrix.

    Both are padded to the larger side so an algorithm that returns a different node count
    (FCI's PAG, for one) still compares against the full ground truth.
    """
    true_graph = np.array(true_graph, dtype=int)
    learned_graph = np.array(learned_graph, dtype=int)

    n = max(true_graph.shape[0], learned_graph.shape[0])
    true_padded = np.zeros((n, n))
    learned_padded = np.zeros((n, n))
    true_padded[: true_graph.shape[0], : true_graph.shape[1]] = true_graph
    learned_padded[: learned_graph.shape[0], : learned_graph.shape[1]] = learned_graph

    true_edges = set(zip(*np.where(true_padded > 0)))
    learned_edges = set(zip(*np.where(learned_padded > 0)))

    tp = len(true_edges & learned_edges)
    fp = len(learned_edges - true_edges)
    fn = len(true_edges - learned_edges)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "shd": fp + fn,
        "n_learned_edges": len(learned_edges),
        "n_true_edges": len(true_edges),
    }
