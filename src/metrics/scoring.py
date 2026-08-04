# scores one predicted interval against one ground-truth distribution.

import logging
from dataclasses import asdict, dataclass

import numpy as np

from src.domains import lower_is_better
from src.domains import metric_scale as domain_metric_scale

LOGGER = logging.getLogger(__name__)

# the nominal level the interval is asked to hold. the Winkler score's miss penalty scales with 2/ALPHA
ALPHA = 0.05

BOUNDED_METRICS = ("precision", "recall", "f1")


def metric_scale(domain: str, metric: str, instance: str, variant: str = "") -> float:
    """Divisor putting a metric's widths and errors on a comparable footing across instances.

    Delegates to the domain registry: a comparison count is scaled by n log2 n, a relaxation count by
    the edge count, SHD by the possible edge count, and a proportion not at all.
    """
    return domain_metric_scale(domain, metric, instance, variant)


def interval_score(lower: float, upper: float, truth: float, alpha: float = ALPHA) -> float:
    """Winkler interval score. Lower is better.

    width + (2/alpha) * (distance outside the interval). A proper scoring rule for central
    intervals: unlike coverage it cannot be gamed by widening, because width is charged directly.
    """
    width = upper - lower
    penalty = 0.0
    if truth < lower:
        penalty = (2 / alpha) * (lower - truth)
    elif truth > upper:
        penalty = (2 / alpha) * (truth - upper)
    return width + penalty


def coverage_score(lower: float, upper: float, truth: float) -> float:
    """Continuous 0-1 companion to binary coverage: 1.0 at the interval centre, 0.5 at an edge,
    decaying to 0 one half-width outside. Carried over so v1 numbers stay comparable."""
    centre = (lower + upper) / 2
    half_width = (upper - lower) / 2
    if half_width == 0:
        return 1.0 if truth == centre else 0.0

    distance = abs(truth - centre)
    if distance <= half_width:
        return float(round(1.0 - (distance / half_width) * 0.5, 4))
    return float(round(max(0.0, 1.0 - (distance - half_width) / half_width), 4))


@dataclass
class Score:
    """Every quantity derived from one (interval, truth) pair."""

    covers_mean: bool
    covers_ci: bool
    coverage_score: float
    interval_score: float
    interval_score_normalised: float
    width: float
    width_normalised: float
    truth_ci_width: float
    width_ratio: float
    bias: float
    bias_normalised: float
    abs_error: float
    abs_error_normalised: float


def score_prediction(
    domain: str,
    metric: str,
    instance: str,
    lower: float,
    upper: float,
    truth_mean: float,
    truth_ci_lower: float,
    truth_ci_upper: float,
    variant: str = "",
    alpha: float = ALPHA,
) -> Score:
    """Score one interval. `bias` is signed on the metric's own orientation.

    For a metric where higher is better a positive bias means the prediction sits above the truth,
    which is optimistic. For a cost metric - SHD, comparisons, relaxations - lower is better, so the
    sign is flipped to keep "positive means optimistic" true in every domain.
    """
    assert lower <= upper, f"interval is inverted: [{lower}, {upper}]"
    scale = metric_scale(domain, metric, instance, variant)

    midpoint = (lower + upper) / 2
    raw_bias = midpoint - truth_mean
    optimism = -raw_bias if lower_is_better(domain, metric) else raw_bias

    width = upper - lower
    truth_ci_width = truth_ci_upper - truth_ci_lower
    raw_interval_score = interval_score(lower, upper, truth_mean, alpha)

    return Score(
        covers_mean=bool(lower <= truth_mean <= upper),
        # the stricter target: contains the whole empirical interval, not just its centre
        covers_ci=bool(lower <= truth_ci_lower and upper >= truth_ci_upper),
        coverage_score=coverage_score(lower, upper, truth_mean),
        interval_score=raw_interval_score,
        interval_score_normalised=raw_interval_score / scale,
        width=width,
        width_normalised=width / scale,
        truth_ci_width=truth_ci_width,
        # how many times wider the prediction is than the truth. inf when the truth has zero spread
        width_ratio=width / truth_ci_width if truth_ci_width > 0 else float("inf"),
        bias=optimism,
        bias_normalised=optimism / scale,
        abs_error=abs(raw_bias),
        abs_error_normalised=abs(raw_bias) / scale,
    )


def score_row(prediction: dict, truth: dict, alpha: float = ALPHA) -> dict:
    """Join one prediction row to its ground-truth row and return the scored row.

    An unparsed prediction stays in the output with null scores. Dropping it would shrink the
    denominator and quietly flatter whichever model refused to answer.
    """
    scored = {
        **prediction,
        "truth_mean": truth["mean"],
        "truth_ci_lower": truth["ci_lower"],
        "truth_ci_upper": truth["ci_upper"],
        "truth_std": truth["std"],
    }
    if not prediction.get("parsed") or prediction.get("lower") is None:
        return {**scored, **{field: None for field in Score.__dataclass_fields__}, "scored": False}

    score = score_prediction(
        domain=prediction["domain"],
        metric=prediction["metric"],
        instance=prediction["instance"],
        variant=prediction["variant"],
        lower=prediction["lower"],
        upper=prediction["upper"],
        truth_mean=truth["mean"],
        truth_ci_lower=truth["ci_lower"],
        truth_ci_upper=truth["ci_upper"],
        alpha=alpha,
    )
    return {**scored, **asdict(score), "scored": True}


def truth_key(row: dict) -> tuple[str, str, str, str, str]:
    return (row["domain"], row["instance"], row["variant"], row["algorithm"], row["metric"])


def score_predictions(predictions: list[dict], ground_truth: list[dict], alpha: float = ALPHA) -> list[dict]:
    """Score every prediction that has ground truth. A prediction with none is an error, not a skip:
    it means the two halves of the experiment were run over different grids."""
    truth_by_key = {truth_key(row): row for row in ground_truth}

    scored, missing = [], set()
    for prediction in predictions:
        truth = truth_by_key.get(truth_key(prediction))
        if truth is None:
            missing.add(truth_key(prediction))
            continue
        scored.append(score_row(prediction, truth, alpha))

    assert not missing, f"{len(missing)} predictions have no ground truth, e.g. {sorted(missing)[:3]}"
    return scored


def summarize(scored: list[dict]) -> dict:
    """Headline numbers over a set of scored rows, ignoring the unscored ones but counting them."""
    usable = [row for row in scored if row["scored"]]
    if not usable:
        return {"n": len(scored), "n_scored": 0}

    finite_ratios = [r["width_ratio"] for r in usable if np.isfinite(r["width_ratio"])]
    return {
        "n": len(scored),
        "n_scored": len(usable),
        "n_unparsed": len(scored) - len(usable),
        "coverage": float(np.mean([r["covers_mean"] for r in usable])),
        "coverage_ci_target": float(np.mean([r["covers_ci"] for r in usable])),
        "mean_coverage_score": float(np.mean([r["coverage_score"] for r in usable])),
        "mean_interval_score": float(np.mean([r["interval_score_normalised"] for r in usable])),
        "median_interval_score": float(np.median([r["interval_score_normalised"] for r in usable])),
        "mean_width": float(np.mean([r["width_normalised"] for r in usable])),
        "mean_width_ratio": float(np.mean(finite_ratios)) if finite_ratios else float("nan"),
        "mean_bias": float(np.mean([r["bias_normalised"] for r in usable])),
        "share_optimistic": float(np.mean([r["bias"] > 0 for r in usable])),
        "mean_abs_error": float(np.mean([r["abs_error_normalised"] for r in usable])),
    }
