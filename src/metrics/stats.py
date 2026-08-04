# intervals and paired tests, so no reported proportion or contrast is a bare number.

import logging
from dataclasses import dataclass

import numpy as np
from scipy import stats

LOGGER = logging.getLogger(__name__)

DEFAULT_N_BOOTSTRAP = 10000


@dataclass
class Proportion:
    """A coverage rate with the interval that says how much to trust it."""

    n: int
    n_success: int
    rate: float
    ci_lower: float
    ci_upper: float
    ci_method: str = "wilson"


def wilson_interval(n_success: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval. Preferred over the normal approximation because coverage rates here
    sit near 0.1, where the normal interval runs below zero and understates the uncertainty."""
    if n == 0:
        return (0.0, 0.0)

    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    phat = n_success / n
    denominator = 1 + z**2 / n
    centre = (phat + z**2 / (2 * n)) / denominator
    margin = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denominator
    return (float(max(0.0, centre - margin)), float(min(1.0, centre + margin)))


def proportion(successes: list[bool], confidence: float = 0.95) -> Proportion:
    n = len(successes)
    n_success = int(sum(successes))
    lower, upper = wilson_interval(n_success, n, confidence)
    return Proportion(n=n, n_success=n_success, rate=n_success / n if n else 0.0, ci_lower=lower, ci_upper=upper)


def binomial_test_against(successes: list[bool], null_rate: float, alternative: str = "two-sided") -> dict:
    """Is this coverage rate distinguishable from the baseline's? The original submission asserted
    a 2.9-point gap was 'not meaningfully distinguishable from chance' without testing it."""
    n = len(successes)
    n_success = int(sum(successes))
    if n == 0:
        return {"n": 0, "error": "no observations"}

    result = stats.binomtest(n_success, n, null_rate, alternative=alternative)
    return {
        "n": n,
        "n_success": n_success,
        "rate": n_success / n,
        "null_rate": null_rate,
        "alternative": alternative,
        "p_value": float(result.pvalue),
        "significant_at_0_05": bool(result.pvalue < 0.05),
    }


def mcnemar_test(a_success: list[bool], b_success: list[bool]) -> dict:
    """Paired comparison of two predictors on the same items.

    The predictors are scored on identical conditions, so the observations are paired and an
    unpaired test would throw away that structure and overstate the uncertainty.
    """
    assert len(a_success) == len(b_success), "paired test needs equal-length aligned inputs"
    n = len(a_success)
    if n == 0:
        return {"n": 0, "error": "no observations"}

    a_only = sum(1 for a, b in zip(a_success, b_success) if a and not b)
    b_only = sum(1 for a, b in zip(a_success, b_success) if b and not a)
    discordant = a_only + b_only
    if discordant == 0:
        return {"n": n, "n_discordant": 0, "p_value": 1.0, "significant_at_0_05": False}

    # exact binomial on the discordant pairs, which is the exact form of McNemar's test
    p_value = float(stats.binomtest(a_only, discordant, 0.5).pvalue)
    return {
        "n": n,
        "n_discordant": discordant,
        "a_only": a_only,
        "b_only": b_only,
        "rate_difference": (a_only - b_only) / n,
        "p_value": p_value,
        "significant_at_0_05": bool(p_value < 0.05),
    }


def bootstrap_ci(
    values: list[float],
    statistic=np.mean,
    confidence: float = 0.95,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = 42,
) -> dict:
    """Percentile bootstrap for a statistic with no closed-form interval, e.g. mean interval score."""
    array = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if array.size == 0:
        return {"n": 0, "error": "no finite values"}

    rng = np.random.default_rng(seed)
    draws = statistic(rng.choice(array, size=(n_bootstrap, array.size), replace=True), axis=1)
    tail = (1 - confidence) / 2
    return {
        "n": int(array.size),
        "estimate": float(statistic(array)),
        "ci_lower": float(np.percentile(draws, 100 * tail)),
        "ci_upper": float(np.percentile(draws, 100 * (1 - tail))),
        "n_bootstrap": n_bootstrap,
    }


def paired_difference(a: list[float], b: list[float], alternative: str = "two-sided") -> dict:
    """Paired t-test plus Cohen's d, for contrasts like real vs anonymized on the same conditions."""
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None and np.isfinite(x) and np.isfinite(y)]
    if len(pairs) < 2:
        return {"n": len(pairs), "error": "need at least 2 complete pairs"}

    left = np.array([x for x, _ in pairs])
    right = np.array([y for _, y in pairs])
    differences = left - right
    std = float(np.std(differences, ddof=1))

    result = stats.ttest_rel(left, right, alternative=alternative)
    return {
        "n": len(pairs),
        "mean_a": float(left.mean()),
        "mean_b": float(right.mean()),
        "mean_difference": float(differences.mean()),
        "std_difference": std,
        "cohens_d": float(differences.mean() / std) if std > 0 else 0.0,
        "t_statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "alternative": alternative,
        "significant_at_0_05": bool(result.pvalue < 0.05),
    }


def holm_bonferroni(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Step-down correction over a family of tests.

    This project runs one test per model per contrast, so an uncorrected 0.05 would be expected to
    produce a false positive somewhere in every family. Holm is uniformly more powerful than
    Bonferroni at the same error rate.
    """
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    n = len(ordered)
    corrected = {}
    running_max = 0.0
    for rank, (name, p) in enumerate(ordered):
        adjusted = min(1.0, (n - rank) * p)
        running_max = max(running_max, adjusted)
        corrected[name] = {
            "p_value": p,
            "p_adjusted": running_max,
            "significant_at_0_05": bool(running_max < alpha),
        }
    return corrected
