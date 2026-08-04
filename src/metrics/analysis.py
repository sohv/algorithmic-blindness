# every analysis in the paper, as a groupby over one scored table.

import logging
from itertools import combinations

import numpy as np
import pandas as pd

from src.domains import BENCHMARK_DATASETS, CAUSAL, NATIVE_VARIANT
from src.metrics.stats import binomial_test_against, holm_bonferroni, mcnemar_test, paired_difference, proportion

LOGGER = logging.getLogger(__name__)

GROUPING_COLUMNS = [
    "predictor",
    "predictor_kind",
    "instance",
    "variant",
    "algorithm",
    "metric",
    "naming",
    "metadata_level",
    "formulation",
]


def to_frame(scored: list[dict]) -> pd.DataFrame:
    """Scored rows as a frame, with the unscorable ones marked but retained."""
    frame = pd.DataFrame(scored)
    assert not frame.empty, "no scored rows"
    # only causal benchmark networks have memorable published statistics; the rest are generated
    frame["is_synthetic"] = ~frame["instance"].isin(BENCHMARK_DATASETS)
    frame["instance_kind"] = np.where(frame["is_synthetic"], "synthetic", "benchmark")
    return frame


def scorable(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows that actually produced a score. Kept as a separate step so the count that was dropped
    is always visible rather than implied."""
    return frame[frame["scored"]].copy()


def coverage_table(frame: pd.DataFrame, by: list[str] | str = "predictor") -> pd.DataFrame:
    """Coverage with a Wilson interval for each group, sorted best first.

    Reporting a bare rate invites reading a 3-point gap as real. The interval is what says whether
    the gap survives the sample size.
    """
    by = [by] if isinstance(by, str) else by
    usable = scorable(frame)

    # a cut can legitimately be empty, and one inapplicable cut must not take down the whole report
    if usable.empty:
        return pd.DataFrame(columns=[*by, "n", "n_covered", "coverage"])

    rows = []
    for key, group in usable.groupby(by, dropna=False):
        stats = proportion(group["covers_mean"].tolist())
        finite_ratio = group["width_ratio"].replace([np.inf, -np.inf], np.nan).dropna()
        rows.append(
            dict(
                zip(by, key if isinstance(key, tuple) else (key,)),
                n=stats.n,
                n_covered=stats.n_success,
                coverage=stats.rate,
                coverage_ci_lower=stats.ci_lower,
                coverage_ci_upper=stats.ci_upper,
                coverage_ci_target=group["covers_ci"].mean(),
                mean_interval_score=group["interval_score_normalised"].mean(),
                mean_width=group["width_normalised"].mean(),
                mean_width_ratio=finite_ratio.mean() if len(finite_ratio) else np.nan,
                mean_bias=group["bias_normalised"].mean(),
                share_optimistic=(group["bias"] > 0).mean(),
                mean_abs_error=group["abs_error_normalised"].mean(),
            )
        )
    return pd.DataFrame(rows).sort_values("coverage", ascending=False).reset_index(drop=True)


def parse_rates(frame: pd.DataFrame) -> pd.DataFrame:
    """Share of rows each predictor actually produced a usable interval for.

    A model answering 60% of the time is not comparable to one answering every time, so this has to
    be read alongside any coverage number rather than folded into it.
    """
    rows = [
        {
            "predictor": predictor,
            "n": len(group),
            "n_scored": int(group["scored"].sum()),
            "parse_rate": group["scored"].mean(),
        }
        for predictor, group in frame.groupby("predictor", dropna=False)
    ]
    return pd.DataFrame(rows).sort_values("parse_rate", ascending=False).reset_index(drop=True)


def paired_on(frame: pd.DataFrame, column: str, left: str, right: str, keys: list[str] | None = None) -> pd.DataFrame:
    """Align two levels of one axis on the conditions they share, one row per (predictor, condition).

    Every contrast in this project is within-predictor and within-condition, so the pairing has to
    be explicit. An unpaired comparison would mix in conditions only one arm was run on.
    """
    keys = keys or ["predictor", "domain", "instance", "variant", "algorithm", "metric"]
    usable = scorable(frame)
    left_rows = usable[usable[column] == left]
    right_rows = usable[usable[column] == right]
    if left_rows.empty or right_rows.empty:
        LOGGER.warning(f"{column}: no rows for {left!r} or {right!r}, contrast unavailable")
        return pd.DataFrame()

    value_columns = [
        "covers_mean",
        "interval_score_normalised",
        "width_normalised",
        "bias_normalised",
        "abs_error_normalised",
    ]
    # average over any axis not in `keys`, so e.g. three formulations collapse to one paired value
    left_grouped = left_rows.groupby(keys, dropna=False)[value_columns].mean().reset_index()
    right_grouped = right_rows.groupby(keys, dropna=False)[value_columns].mean().reset_index()
    return left_grouped.merge(right_grouped, on=keys, suffixes=("_left", "_right"))


def contrast(
    frame: pd.DataFrame,
    column: str,
    left: str,
    right: str,
    by_predictor: bool = True,
    keys: list[str] | None = None,
) -> dict:
    """Paired effect of moving one axis from `right` to `left`, overall and per predictor.

    `keys` defaults to pairing within a predictor. Pass condition-only keys for an axis whose two
    arms are different predictors by construction, such as the frontier-vs-mid tier contrast.
    """
    paired = paired_on(frame, column, left, right, keys)
    if paired.empty:
        return {"axis": column, "left": left, "right": right, "available": False}

    def one(rows: pd.DataFrame) -> dict:
        return {
            "n_pairs": len(rows),
            "coverage_left": float(rows["covers_mean_left"].mean()),
            "coverage_right": float(rows["covers_mean_right"].mean()),
            "coverage_difference": float((rows["covers_mean_left"] - rows["covers_mean_right"]).mean()),
            "coverage_test": mcnemar_test(
                (rows["covers_mean_left"] > 0.5).tolist(), (rows["covers_mean_right"] > 0.5).tolist()
            ),
            "interval_score_test": paired_difference(
                rows["interval_score_normalised_left"].tolist(), rows["interval_score_normalised_right"].tolist()
            ),
            "width_left": float(rows["width_normalised_left"].mean()),
            "width_right": float(rows["width_normalised_right"].mean()),
        }

    result = {"axis": column, "left": left, "right": right, "available": True, "overall": one(paired)}
    if by_predictor:
        per_predictor = {str(name): one(group) for name, group in paired.groupby("predictor", dropna=False)}
        result["by_predictor"] = per_predictor
        result["holm"] = holm_bonferroni(
            {name: stats["coverage_test"].get("p_value", 1.0) for name, stats in per_predictor.items()}
        )
    return result


def metadata_effect(frame: pd.DataFrame) -> dict:
    """Does telling the model more fix it?

    The central new experiment. Reviewers argued the original prompt withheld what was needed, so a
    failure under it says nothing. If coverage does not improve from `sparse` to `full` - where the
    model is handed the non-Gaussianity verdict, the assumption list, the implementation and the
    true graph structure - then underspecification is not the explanation.
    """
    return {
        "by_level": coverage_table(frame, ["predictor", "metadata_level"]).to_dict("records"),
        "overall_by_level": coverage_table(frame, "metadata_level").to_dict("records"),
        "diagnostic_vs_sparse": contrast(frame, "metadata_level", "diagnostic", "sparse"),
        "full_vs_sparse": contrast(frame, "metadata_level", "full", "sparse"),
        "full_vs_diagnostic": contrast(frame, "metadata_level", "full", "diagnostic"),
    }


def naming_effect(frame: pd.DataFrame) -> dict:
    """The anonymized-benchmark experiment.

    Only benchmark datasets carry a memorable name, so the contrast is restricted to them. A
    synthetic dataset has nothing to anonymize and would dilute the effect toward zero.
    """
    benchmarks = frame[~frame["is_synthetic"]]
    return {
        "by_naming": coverage_table(benchmarks, ["predictor", "naming"]).to_dict("records"),
        "real_vs_anonymized": contrast(benchmarks, "naming", "real", "anonymized"),
        "note": "restricted to benchmark datasets; synthetic names carry no cue to remove",
    }


def noise_effect(frame: pd.DataFrame) -> dict:
    """Does LiNGAM's synthetic collapse survive giving it the noise it requires?

    Reviewer KsDD's confound. Under Gaussian noise LiNGAM is not identifiable at all, so its poor
    synthetic result had an explanation that had nothing to do with memorization. Non-Gaussian
    noise satisfies the assumption while holding the graph fixed, which separates the two.
    """
    synthetic = frame[(frame["domain"] == CAUSAL) & frame["is_synthetic"]]
    non_gaussian = synthetic[synthetic["variant"] != "gaussian"]
    if non_gaussian.empty:
        return {"available": False, "note": "no non-Gaussian synthetic conditions in this run"}

    per_algorithm = {}
    for algorithm, group in synthetic.groupby("algorithm", dropna=False):
        gaussian_rows = group[group["variant"] == "gaussian"]
        other_rows = group[group["variant"] != "gaussian"]
        if gaussian_rows.empty or other_rows.empty:
            continue
        per_algorithm[str(algorithm)] = {
            "coverage_gaussian": float(scorable(gaussian_rows)["covers_mean"].mean()),
            "coverage_non_gaussian": float(scorable(other_rows)["covers_mean"].mean()),
            "n_gaussian": int(scorable(gaussian_rows).shape[0]),
            "n_non_gaussian": int(scorable(other_rows).shape[0]),
        }

    return {
        "available": True,
        "by_noise": coverage_table(synthetic, ["variant"]).to_dict("records"),
        "by_algorithm_and_noise": coverage_table(synthetic, ["algorithm", "variant"]).to_dict("records"),
        "per_algorithm_gaussian_vs_not": per_algorithm,
    }


def dataset_kind_effect(frame: pd.DataFrame) -> dict:
    """Benchmark versus synthetic coverage and width, the original memorization signal.

    Held to gaussian synthetic only, so it measures the same contrast the original submission did
    rather than silently averaging in the new noise families.
    """
    comparable = frame[(frame["domain"] == CAUSAL) & frame["variant"].isin([NATIVE_VARIANT, "gaussian"])]
    return {
        "by_kind": coverage_table(comparable, ["predictor", "instance_kind"]).to_dict("records"),
        "overall_by_kind": coverage_table(comparable, "instance_kind").to_dict("records"),
        "benchmark_vs_synthetic": contrast(comparable, "instance_kind", "benchmark", "synthetic", by_predictor=True),
    }


def against_baseline(frame: pd.DataFrame, baseline: str = "uniform_sorted_pair") -> dict:
    """Test every predictor against the baseline's rate, unpaired then paired.

    The original submission called a 2.9-point gap 'not meaningfully distinguishable from chance'
    without a test. Both are reported: the binomial against the baseline's rate, and McNemar on the
    conditions the two share, which is the stronger of the two because the items are the same.
    """
    usable = scorable(frame)
    baseline_rows = usable[usable["predictor"] == baseline]
    if baseline_rows.empty:
        return {"available": False, "note": f"baseline {baseline} not present"}

    baseline_rate = float(baseline_rows["covers_mean"].mean())
    keys = ["domain", "instance", "variant", "algorithm", "metric"]
    baseline_by_condition = baseline_rows.groupby(keys, dropna=False)["covers_mean"].mean()

    results, p_values = {}, {}
    for predictor, group in usable.groupby("predictor", dropna=False):
        if predictor == baseline:
            continue
        binomial = binomial_test_against(group["covers_mean"].tolist(), baseline_rate)

        aligned = group.groupby(keys, dropna=False)["covers_mean"].mean()
        shared = aligned.index.intersection(baseline_by_condition.index)
        paired = mcnemar_test((aligned.loc[shared] > 0.5).tolist(), (baseline_by_condition.loc[shared] > 0.5).tolist())

        results[str(predictor)] = {
            "coverage": float(group["covers_mean"].mean()),
            "difference": float(group["covers_mean"].mean()) - baseline_rate,
            "binomial_vs_baseline_rate": binomial,
            "mcnemar_paired": paired,
        }
        p_values[str(predictor)] = paired.get("p_value", 1.0)

    return {
        "available": True,
        "baseline": baseline,
        "baseline_rate": baseline_rate,
        "by_predictor": results,
        "holm": holm_bonferroni(p_values),
    }


def prompt_robustness(frame: pd.DataFrame) -> dict:
    """Coefficient of variation of the predicted midpoint across formulations.

    High CV means the number is a fact about the wording, not the model, and invalidates reading
    that model's coverage as a property of the model.
    """
    usable = scorable(frame)
    usable = usable[usable["formulation"] > 0]
    if usable.empty:
        return {"available": False}

    keys = ["predictor", "instance", "variant", "algorithm", "metric", "naming", "metadata_level"]
    usable = usable.assign(midpoint=(usable["lower"] + usable["upper"]) / 2)
    grouped = usable.groupby(keys, dropna=False)["midpoint"].agg(["mean", "std", "count"]).reset_index()
    grouped = grouped[grouped["count"] >= 2]
    if grouped.empty:
        return {"available": False, "note": "fewer than 2 formulations per condition"}

    grouped["cv_percent"] = np.where(grouped["mean"] != 0, 100 * grouped["std"] / grouped["mean"].abs(), 0.0)
    by_predictor = grouped.groupby("predictor", dropna=False)["cv_percent"].agg(["mean", "median", "max", "count"])
    return {
        "available": True,
        "by_predictor": by_predictor.reset_index().to_dict("records"),
        "overall_mean_cv_percent": float(grouped["cv_percent"].mean()),
    }


def cross_model_agreement(frame: pd.DataFrame) -> dict:
    """Mean pairwise distance between model midpoints on the same condition.

    Convergence on benchmarks and divergence on synthetic data is the original memorization signal.
    Distances are computed on normalised midpoints so SHD does not dominate the average.
    """
    usable = scorable(frame)
    usable = usable[usable["predictor_kind"] == "llm"]
    if usable.empty:
        return {"available": False}

    keys = ["domain", "instance", "variant", "algorithm", "metric", "instance_kind"]
    usable = usable.assign(midpoint=(usable["lower"] + usable["upper"]) / 2)
    # normalise before pooling, otherwise a 70-node SHD swamps every proportion metric
    scale = np.where(usable["metric"] == "shd", usable["truth_ci_upper"].abs().clip(lower=1.0), 1.0)
    usable = usable.assign(midpoint_normalised=usable["midpoint"] / scale)

    rows = []
    for key, group in usable.groupby(keys, dropna=False):
        values = group.groupby("predictor", dropna=False)["midpoint_normalised"].mean().to_numpy()
        if len(values) < 2:
            continue
        distances = [abs(a - b) for a, b in combinations(values, 2)]
        rows.append(dict(zip(keys, key), n_models=len(values), mean_pairwise_distance=float(np.mean(distances))))

    if not rows:
        return {"available": False, "note": "need at least 2 models per condition"}

    distances = pd.DataFrame(rows)
    return {
        "available": True,
        "by_dataset_kind": distances.groupby("instance_kind")["mean_pairwise_distance"].mean().to_dict(),
        "by_dataset": distances.groupby("instance")["mean_pairwise_distance"].mean().sort_values().to_dict(),
        "overall": float(distances["mean_pairwise_distance"].mean()),
    }


def bias_direction(frame: pd.DataFrame) -> dict:
    """Are predictions systematically optimistic? Reviewer j7Mn asked and the rebuttal promised it.

    Bias is signed so that positive always means optimistic, including for SHD where lower is
    better, so the sign is comparable across metrics.
    """
    usable = scorable(frame)
    by = usable.groupby("predictor", dropna=False).agg(
        mean_bias=("bias_normalised", "mean"),
        median_bias=("bias_normalised", "median"),
        share_optimistic=("bias", lambda s: float((s > 0).mean())),
        n=("bias", "size"),
    )
    by_metric = usable.groupby("metric", dropna=False).agg(
        mean_bias=("bias_normalised", "mean"), share_optimistic=("bias", lambda s: float((s > 0).mean()))
    )
    return {
        "by_predictor": by.reset_index().to_dict("records"),
        "by_metric": by_metric.reset_index().to_dict("records"),
        "overall_share_optimistic": float((usable["bias"] > 0).mean()),
    }


# baselines never see a prompt and already carry formulation 0, so this is one row per predictor
HEADLINE_FORMULATION = 0


def headline_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["formulation"] == HEADLINE_FORMULATION]


def analyze(scored: list[dict], baseline: str = "uniform_sorted_pair") -> dict:
    """Every analysis, over one scored table."""
    frame = to_frame(scored)
    llm_and_baselines = frame[frame["predictor_kind"].isin(["llm", "baseline", "human"])]

    return {
        "parse_rates": parse_rates(frame).to_dict("records"),
        "coverage_by_predictor": coverage_table(headline_frame(llm_and_baselines), ["predictor", "tier"]).to_dict(
            "records"
        ),
        "coverage_by_domain": coverage_table(headline_frame(llm_and_baselines), ["domain"]).to_dict("records"),
        "coverage_by_domain_and_predictor": coverage_table(
            headline_frame(llm_and_baselines), ["domain", "predictor"]
        ).to_dict("records"),
        "coverage_by_tier": coverage_table(llm_and_baselines, "tier").to_dict("records"),
        "frontier_vs_mid": contrast(
            frame[frame["predictor_kind"] == "llm"],
            "tier",
            "frontier",
            "mid",
            by_predictor=False,
            keys=["domain", "instance", "variant", "algorithm", "metric"],
        ),
        "coverage_by_algorithm": coverage_table(frame, ["domain", "algorithm"]).to_dict("records"),
        "coverage_by_metric": coverage_table(frame, "metric").to_dict("records"),
        "coverage_by_dataset": coverage_table(frame, "instance").to_dict("records"),
        "against_baseline": against_baseline(frame, baseline),
        "metadata_effect": metadata_effect(frame),
        "naming_effect": naming_effect(frame),
        "noise_effect": noise_effect(frame),
        "dataset_kind_effect": dataset_kind_effect(frame),
        "prompt_robustness": prompt_robustness(frame),
        "cross_model_agreement": cross_model_agreement(frame),
        "bias_direction": bias_direction(frame),
    }
