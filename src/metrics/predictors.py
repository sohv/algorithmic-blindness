# non-LLM predictors, emitted in the model schema so everything is scored by the same code.

import csv
import logging
from pathlib import Path
from typing import ClassVar

import numpy as np

from src.algorithms import shortest_path, sorting
from src.conditions import Condition
from src.domains import (
    ARRAY_SIZES,
    CAUSAL,
    GRAPH_SIZES,
    SHORTEST_PATH,
    SORTING,
    expected_edges,
    max_edges,
    metric_scale,
    metrics_of,
    n_nodes,
    n_samples,
)

LOGGER = logging.getLogger(__name__)

BOUNDED_METRICS = ("precision", "recall", "f1")


def metric_max(domain: str, metric: str, instance: str, variant: str = "") -> float:
    """The upper end of a metric's plausible domain, for drawing a uniform interval in it.

    A proportion is bounded at 1. A count has no hard bound, so the natural scale is used with
    generous headroom — a random baseline drawing from an implausibly narrow range would be an
    artificially strong floor.
    """
    if metric in BOUNDED_METRICS:
        return 1.0
    if domain == CAUSAL:
        return float(max_edges(instance))
    return 3.0 * metric_scale(domain, metric, instance, variant)


# two readings of "uniformly at random", with E[width] max/3 and max/4; the choice is recorded, not implicit
RANDOM_SCHEMES = ("sorted_pair", "nested")


class UniformRandom:
    """Uninformed intervals drawn within each metric's plausible domain."""

    kind = "baseline"

    def __init__(self, seed: int = 42, scheme: str = "sorted_pair"):
        assert scheme in RANDOM_SCHEMES, f"unknown scheme {scheme}, expected one of {RANDOM_SCHEMES}"
        self.name = f"uniform_{scheme}"
        self.scheme = scheme
        self.rng = np.random.default_rng(seed)

    def predict(self, condition: Condition, metric: str) -> tuple[float, float]:
        ceiling = metric_max(condition.domain, metric, condition.instance, condition.variant)
        if self.scheme == "sorted_pair":
            lower, upper = sorted(self.rng.uniform(0, ceiling, 2))
        else:
            lower = self.rng.uniform(0, ceiling)
            upper = self.rng.uniform(lower, ceiling)
        return float(lower), float(upper)


class Heuristic:
    """Simple rules on the instance's size, with no knowledge of which assumptions it satisfies.

    For causal discovery: more samples helps, more variables hurts. For the count domains: the
    asymptotic scale with a wide band around it — the "I know the big-O and nothing else" predictor.
    """

    kind = "baseline"
    name = "heuristic"

    ALGORITHM_BASE: ClassVar[dict[str, float]] = {"pc": 0.55, "lingam": 0.60, "notears": 0.58}
    WIDTH = 0.20
    COUNT_BAND = (0.3, 3.0)

    def predict(self, condition: Condition, metric: str) -> tuple[float, float]:
        if condition.domain != CAUSAL:
            scale = metric_scale(condition.domain, metric, condition.instance, condition.variant)
            return float(scale * self.COUNT_BAND[0]), float(scale * self.COUNT_BAND[1])

        base = self.ALGORITHM_BASE.get(condition.algorithm, 0.55)
        d = n_nodes(condition.instance)
        n = n_samples(condition.instance)

        sample_adjustment = float(np.clip(0.1 * np.log10(n / 1000), -0.15, 0.15))
        dimension_adjustment = float(np.clip(-0.015 * d, -0.20, 0.0))
        centre = float(np.clip(base + sample_adjustment + dimension_adjustment, 0.2, 0.9))

        if metric == "shd":
            ceiling = float(max_edges(condition.instance))
            expected = (1 - centre) * ceiling * 0.5
            return float(max(0.0, expected - ceiling * 0.1)), float(min(ceiling, expected + ceiling * 0.1))
        return float(max(0.0, centre - self.WIDTH / 2)), float(min(1.0, centre + self.WIDTH / 2))


class Marginal:
    """The empirical spread of this metric across every *other* condition in the same domain.

    Leave-one-out, so the interval for a condition never sees that condition's own truth. Pooled
    within a domain and on the normalised scale, because a raw comparison count on 1k elements and
    on 100k are not the same quantity.
    """

    kind = "baseline"
    name = "marginal"

    def __init__(self, ground_truth: list[dict], lower_q: float = 2.5, upper_q: float = 97.5):
        self.lower_q = lower_q
        self.upper_q = upper_q
        self.pooled: dict[tuple[str, str], list[tuple[tuple, float]]] = {}
        for row in ground_truth:
            scale = metric_scale(row["domain"], row["metric"], row["instance"], row["variant"])
            key = (row["domain"], row["metric"])
            entry = ((row["instance"], row["variant"], row["algorithm"]), row["mean"] / scale)
            self.pooled.setdefault(key, []).append(entry)

    def predict(self, condition: Condition, metric: str) -> tuple[float, float] | None:
        held_out = (condition.instance, condition.variant, condition.algorithm)
        others = [value for key, value in self.pooled.get((condition.domain, metric), []) if key != held_out]
        if len(others) < 2:
            LOGGER.warning(f"marginal has {len(others)} other conditions for {condition.domain}/{metric}, skipping")
            return None

        scale = metric_scale(condition.domain, metric, condition.instance, condition.variant)
        return float(np.percentile(others, self.lower_q)) * scale, float(np.percentile(others, self.upper_q)) * scale


class Analytic:
    """The textbook closed form, with a band around it for the constants the analysis omits.

    Only defined where a published expected count exists: sorting on distinct keys, and the
    relaxation counts for Dijkstra and Bellman-Ford. Causal discovery has no such formula, which is
    precisely the difference between the domains this predictor exists to expose.
    """

    kind = "baseline"
    name = "analytic"

    def __init__(self, diagnostics: list[dict] | None = None):
        # Bellman-Ford's pass count is the longest shortest path in edges, a measured graph property
        self.diagnostics = {row["id"]: row for row in (diagnostics or [])}

    def predict(self, condition: Condition, metric: str) -> tuple[float, float] | None:
        if condition.domain == SORTING:
            prediction = sorting.analytic_prediction(
                condition.algorithm, ARRAY_SIZES[condition.instance], condition.variant
            )
        elif condition.domain == SHORTEST_PATH:
            diagnostic = self.diagnostics.get(f"{condition.domain}__{condition.instance}__{condition.variant}")
            prediction = shortest_path.analytic_prediction(
                condition.algorithm,
                GRAPH_SIZES[condition.instance],
                diagnostic["n_edges"] if diagnostic else int(expected_edges(condition.instance, condition.variant)),
                diagnostic["max_shortest_path_hops"] if diagnostic else None,
            )
        else:
            return None

        if prediction is None or metric not in prediction:
            return None
        lower, upper = prediction[metric]
        return float(lower), float(upper)


class Oracle:
    """The true empirical interval. Scores perfectly on coverage by construction; its value is as
    the width reference and as a check that the scoring code rewards what it should."""

    kind = "baseline"
    name = "oracle"

    def __init__(self, ground_truth: list[dict]):
        self.truth = {(r["domain"], r["instance"], r["variant"], r["algorithm"], r["metric"]): r for r in ground_truth}

    def predict(self, condition: Condition, metric: str) -> tuple[float, float] | None:
        key = (condition.domain, condition.instance, condition.variant, condition.algorithm, metric)
        row = self.truth.get(key)
        return (row["ci_lower"], row["ci_upper"]) if row else None


def build_baselines(
    ground_truth: list[dict],
    seed: int = 42,
    random_scheme: str = "sorted_pair",
    diagnostics: list[dict] | None = None,
) -> list:
    return [
        UniformRandom(seed, random_scheme),
        Heuristic(),
        Marginal(ground_truth),
        Analytic(diagnostics),
        Oracle(ground_truth),
    ]


def baseline_predictions(
    conditions: list[Condition],
    ground_truth: list[dict],
    seed: int = 42,
    random_scheme: str = "sorted_pair",
    diagnostics: list[dict] | None = None,
) -> list[dict]:
    """One prediction row per (baseline, condition, metric), in the model schema.

    Baselines have no naming, metadata level or formulation: they never see a prompt. Those axes are
    recorded as "na" so the rows still join and group cleanly alongside the model rows. `analytic`
    declines on conditions with no closed form, recorded as an unparsed row rather than silently
    dropped — its coverage is then over the conditions it can actually speak to.
    """
    rows = []
    for baseline in build_baselines(ground_truth, seed, random_scheme, diagnostics):
        for condition in conditions:
            for metric in metrics_of(condition.domain):
                interval = baseline.predict(condition, metric)
                lower, upper = interval if interval else (None, None)
                rows.append(
                    {
                        "id": f"{baseline.name}__{condition.key}__{metric}",
                        "predictor": baseline.name,
                        "predictor_kind": baseline.kind,
                        "tier": "baseline",
                        "domain": condition.domain,
                        "instance": condition.instance,
                        "variant": condition.variant,
                        "algorithm": condition.algorithm,
                        "metric": metric,
                        "naming": "na",
                        "metadata_level": "na",
                        "formulation": 0,
                        "lower": lower,
                        "upper": upper,
                        "parsed": interval is not None,
                    }
                )
    LOGGER.info(f"built {len(rows)} baseline predictions over {len(conditions)} conditions")
    return rows


REQUIRED_HUMAN_COLUMNS = {"predictor", "domain", "instance", "variant", "algorithm", "metric", "lower", "upper"}


def load_human_predictions(path: Path | str) -> list[dict]:
    """Read an expert elicitation sheet into prediction rows.

    Reviewer KsDD's ask: without a human baseline, a low LLM score cannot be separated from the task
    simply being hard. The sheet is a csv so it can be filled in by hand, and it is validated on read
    rather than silently producing unscorable rows.
    """
    path = Path(path)
    with path.open() as handle:
        raw_rows = list(csv.DictReader(handle))
    assert raw_rows, f"{path} has no rows"

    missing = REQUIRED_HUMAN_COLUMNS - set(raw_rows[0])
    assert not missing, f"{path} is missing columns: {sorted(missing)}"

    rows = []
    for index, raw in enumerate(raw_rows):
        valid = metrics_of(raw["domain"])
        assert raw["metric"] in valid, f"{path} row {index}: {raw['metric']} is not a {raw['domain']} metric"
        lower, upper = float(raw["lower"]), float(raw["upper"])
        assert lower <= upper, f"{path} row {index}: inverted interval [{lower}, {upper}]"
        rows.append(
            {
                "id": f"{raw['predictor']}__{raw['domain']}__{raw['instance']}__{raw['variant']}__{raw['algorithm']}__{raw['metric']}",
                "predictor": raw["predictor"],
                "predictor_kind": "human",
                "tier": "human",
                "domain": raw["domain"],
                "instance": raw["instance"],
                "variant": raw["variant"],
                "algorithm": raw["algorithm"],
                "metric": raw["metric"],
                "naming": raw.get("naming", "real"),
                "metadata_level": raw.get("metadata_level", "sparse"),
                "formulation": int(raw.get("formulation", 0)),
                "lower": lower,
                "upper": upper,
                "parsed": True,
            }
        )
    LOGGER.info(f"loaded {len(rows)} human predictions from {path}")
    return rows


def write_elicitation_sheet(conditions: list[Condition], path: Path | str, predictor: str = "expert_1") -> Path:
    """Write the blank csv an expert fills in, so their rows land in the schema by construction."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "predictor",
        "domain",
        "instance",
        "variant",
        "algorithm",
        "metric",
        "naming",
        "metadata_level",
        "lower",
        "upper",
    ]

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for condition in conditions:
            for metric in metrics_of(condition.domain):
                writer.writerow(
                    {
                        "predictor": predictor,
                        "domain": condition.domain,
                        "instance": condition.instance,
                        "variant": condition.variant,
                        "algorithm": condition.algorithm,
                        "metric": metric,
                        "naming": "real",
                        "metadata_level": "sparse",
                        "lower": "",
                        "upper": "",
                    }
                )
    LOGGER.info(f"wrote elicitation sheet for {len(conditions)} conditions to {path}")
    return path
