# runs any condition in any domain and returns per-run records.

import logging

from src.algorithms import shortest_path, sorting
from src.conditions import Condition
from src.data.instances import adjacency, build_array, build_graph
from src.domains import CAUSAL, SHORTEST_PATH, SORTING, n_samples

LOGGER = logging.getLogger(__name__)

# domains whose only variation is the instance draw, so each run needs a fresh instance
REDRAWN_PER_RUN = (SORTING,)
# domains that are exactly deterministic given the instance, so more runs add nothing
DETERMINISTIC = (SHORTEST_PATH,)


# causal imports are deferred so the other two domains run without the causal stack installed


def load_causal_data(condition: Condition, data_dir: str = "data/raw", seed: int = 42):
    from src.data.benchmarks import load_benchmark
    from src.data.networks import load_synthetic

    if condition.is_synthetic:
        return load_synthetic(condition.instance, condition.variant, seed=seed)
    data, true_graph, _ = load_benchmark(condition.instance, data_dir, n_samples(condition.instance), seed=seed)
    return data, true_graph


def run_causal(condition: Condition, n_runs: int, data_dir: str, seed: int, bootstrap: bool) -> list[dict]:
    from src.algorithms.discovery import run_algorithm

    data, true_graph = load_causal_data(condition, data_dir, seed)
    return run_algorithm(data, true_graph, condition.algorithm, n_runs, seed, bootstrap)


def run_sorting(condition: Condition, n_runs: int, seed: int) -> list[dict]:
    """One sort per run, each on a freshly drawn array from the same distribution."""
    sort = sorting.ALGORITHMS[condition.algorithm]
    results = []
    for run in range(1, n_runs + 1):
        values = build_array(condition.instance, condition.variant, seed=seed + run)
        try:
            metrics = sort(values)
        except Exception:
            LOGGER.exception(f"{condition.key} run {run}/{n_runs} failed")
            results.append({"run": run, "failed": True})
            continue
        results.append({"run": run, "failed": False, **metrics})
    return results


def run_shortest_path(condition: Condition, n_runs: int, seed: int) -> list[dict]:
    """One graph, one run per draw. Both algorithms see the identical graph for a given seed, so
    their counts are directly comparable rather than being two samples of different problems."""
    solve = shortest_path.ALGORITHMS[condition.algorithm]
    results = []
    for run in range(1, n_runs + 1):
        n, edges = build_graph(condition.instance, condition.variant, seed=seed + run)
        graph = adjacency(n, edges)
        try:
            metrics = solve(n, graph)
        except Exception:
            LOGGER.exception(f"{condition.key} run {run}/{n_runs} failed")
            results.append({"run": run, "failed": True})
            continue
        results.append({"run": run, "failed": False, **metrics})
    return results


def run_condition(
    condition: Condition,
    n_runs: int,
    data_dir: str = "data/raw",
    seed: int = 42,
    bootstrap: bool = True,
) -> list[dict]:
    """Per-run records for one condition, tagged with the condition's axis values."""
    if condition.domain == CAUSAL:
        runs = run_causal(condition, n_runs, data_dir, seed, bootstrap)
    elif condition.domain == SORTING:
        runs = run_sorting(condition, n_runs, seed)
    elif condition.domain == SHORTEST_PATH:
        runs = run_shortest_path(condition, n_runs, seed)
    else:
        raise ValueError(f"no runner for domain {condition.domain}")

    return [
        {
            "id": f"{condition.key}__run{run['run']:04d}",
            "domain": condition.domain,
            "instance": condition.instance,
            "variant": condition.variant,
            "algorithm": condition.algorithm,
            **run,
        }
        for run in runs
    ]


def run_conditions(
    conditions: list[Condition],
    n_runs: int,
    data_dir: str = "data/raw",
    seed: int = 42,
    bootstrap: bool = True,
) -> tuple[list[dict], list[dict]]:
    """Run every condition, returning (per-run records, diagnostics records)."""
    from src.data.diagnostics import diagnostics_for

    runs, diagnostics = [], {}
    for condition in conditions:
        LOGGER.info(f"running {condition.key} for {n_runs} runs")
        runs.extend(run_condition(condition, n_runs, data_dir, seed, bootstrap))

        key = f"{condition.domain}__{condition.instance}__{condition.variant}"
        if key not in diagnostics:
            diagnostics[key] = diagnostics_for(condition, data_dir, seed)

    return runs, list(diagnostics.values())
