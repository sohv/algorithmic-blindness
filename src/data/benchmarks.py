# samples data from the benchmark bayesian networks shipped as .bif files in data/raw/.

import logging
from pathlib import Path

import bnlearn as bn
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.conditions import BENCHMARK_DATASETS

LOGGER = logging.getLogger(__name__)


def load_benchmark(name: str, data_dir: Path | str = "data/raw", n_samples: int = 10000, seed: int = 42):
    """Load a benchmark bayesian network from a local .bif file and sample data from it.

    Returns (data, true_graph, nodes): a numeric DataFrame, the ground-truth adjacency matrix
    with nodes in sorted order, and that node order.
    """
    if name.lower() not in BENCHMARK_DATASETS:
        raise ValueError(f"Unknown network: {name}. Known: {BENCHMARK_DATASETS}")

    bif_path = Path(data_dir) / f"{name.lower()}.bif"
    if not bif_path.exists():
        raise FileNotFoundError(f"BIF file not found: {bif_path}")

    LOGGER.info(f"loading {name} from {bif_path}")
    dag = bn.import_DAG(str(bif_path))
    nodes = sorted(dag["model"].nodes())

    data = bn.sampling(dag, n=n_samples, verbose=0)
    if hasattr(data, "__getitem__") and "df" in data:
        data = data["df"]
    elif not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data, columns=nodes)
    if list(data.columns) != nodes:
        data = data[nodes]

    for col in data.columns:
        if data[col].dtype == "object" or str(data[col].dtype) == "category":
            data[col] = LabelEncoder().fit_transform(data[col].astype(str))
    data = data.apply(pd.to_numeric, errors="coerce").dropna().astype(np.float64)

    node_to_idx = {node: i for i, node in enumerate(nodes)}
    true_graph = np.zeros((len(nodes), len(nodes)))
    for source, target in dag["model"].edges():
        true_graph[node_to_idx[source], node_to_idx[target]] = 1

    LOGGER.info(f"{name}: {len(nodes)} nodes, {data.shape[0]} samples, {int(true_graph.sum())} edges")
    return data, true_graph, nodes
