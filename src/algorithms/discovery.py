# runs the causal discovery algorithms on bootstrap resamples of the data.

import logging
from collections.abc import Callable

import numpy as np
import pandas as pd
from causallearn.search.ConstraintBased.PC import pc
from lingam import DirectLiNGAM
from notears_pytorch import notears_linear

from src.metrics.graph import compute_metrics

LOGGER = logging.getLogger(__name__)


def fit_pc(data: pd.DataFrame):
    # progress bars off: one bar per bootstrap run would be thousands of lines in run.log
    return pc(data.values, show_progress=False).G.graph


def fit_lingam(data: pd.DataFrame):
    model = DirectLiNGAM()
    model.fit(data.values)
    return (model.adjacency_matrix_ != 0).astype(int)


def fit_notears(data: pd.DataFrame):
    return (notears_linear(data.values, lambda1=0.1) != 0).astype(int)


ALGORITHMS: dict[str, Callable] = {"pc": fit_pc, "lingam": fit_lingam, "notears": fit_notears}


def bootstrap_sample(data: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Resample rows with replacement, keeping the row count fixed."""
    indices = rng.integers(0, len(data), size=len(data))
    return data.iloc[indices].reset_index(drop=True)


def run_algorithm(
    data: pd.DataFrame,
    true_graph,
    algorithm: str,
    n_runs: int,
    seed: int = 42,
    bootstrap: bool = True,
) -> list[dict]:
    """Fit one algorithm n_runs times on bootstrap resamples, returning per-run metrics.

    A failed fit is the per-item boundary of a batch: logged and recorded as a failed run, so one
    singular resample never discards the runs that already succeeded.

    bootstrap=False refits the identical sample every time, reproducing the pre-rebuild code path.
    It is kept only so the difference can be measured, and it will report zero variance.
    """
    fit = ALGORITHMS[algorithm]
    rng = np.random.default_rng(seed)
    results = []

    for run in range(1, n_runs + 1):
        sample = bootstrap_sample(data, rng) if bootstrap else data
        try:
            metrics = compute_metrics(true_graph, fit(sample))
        except Exception:
            LOGGER.exception(f"{algorithm} run {run}/{n_runs} failed")
            results.append({"run": run, "failed": True})
            continue
        results.append({"run": run, "failed": False, **metrics})
        if run % 10 == 0:
            LOGGER.info(f"{algorithm} run {run}/{n_runs}: f1={metrics['f1']:.4f} shd={metrics['shd']}")

    return results
