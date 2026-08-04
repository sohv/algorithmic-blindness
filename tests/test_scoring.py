import numpy as np
import pytest

from src.metrics.scoring import (
    coverage_score,
    interval_score,
    metric_scale,
    score_prediction,
    score_predictions,
    score_row,
    summarize,
)


def truth_row(instance="asia", variant="native", algorithm="pc", metric="f1", mean=0.6, lo=0.55, hi=0.65):
    return {
        "domain": "causal_discovery",
        "instance": instance,
        "variant": variant,
        "algorithm": algorithm,
        "metric": metric,
        "mean": mean,
        "std": 0.03,
        "ci_lower": lo,
        "ci_upper": hi,
    }


def prediction_row(lower=0.5, upper=0.7, metric="f1", parsed=True, predictor="claude"):
    return {
        "id": f"{predictor}__asia__native__pc__{metric}",
        "predictor": predictor,
        "predictor_kind": "llm",
        "domain": "causal_discovery",
        "instance": "asia",
        "variant": "native",
        "algorithm": "pc",
        "metric": metric,
        "naming": "real",
        "metadata_level": "sparse",
        "formulation": 0,
        "lower": lower,
        "upper": upper,
        "parsed": parsed,
    }


def test_interval_score_is_just_width_when_the_truth_is_inside():
    assert interval_score(0.4, 0.8, 0.6) == pytest.approx(0.4)


def test_interval_score_charges_for_missing():
    inside = interval_score(0.4, 0.8, 0.6)
    outside = interval_score(0.4, 0.8, 0.9)
    assert outside > inside
    # width 0.4 plus (2/0.05) * 0.1 overshoot
    assert outside == pytest.approx(0.4 + 40 * 0.1)


def test_interval_score_cannot_be_gamed_by_widening():
    # a proper rule: [0, 1] covers everything and still loses to a tight interval that also covers
    assert interval_score(0.55, 0.65, 0.6) < interval_score(0.0, 1.0, 0.6)


def test_coverage_score_peaks_at_the_centre_and_halves_at_the_edge():
    assert coverage_score(0.4, 0.8, 0.6) == 1.0
    assert coverage_score(0.4, 0.8, 0.8) == pytest.approx(0.5)
    assert coverage_score(0.4, 0.8, 1.5) == 0.0


def test_shd_is_scaled_by_the_possible_edge_count():
    assert metric_scale("causal_discovery", "f1", "asia") == 1.0
    assert metric_scale("causal_discovery", "shd", "asia") == 28.0
    assert metric_scale("causal_discovery", "shd", "hepar2") == 70 * 69 // 2


def test_covers_mean_and_covers_ci_are_different_targets():
    # contains the mean but not the whole empirical interval
    score = score_prediction("causal_discovery", "f1", "asia", 0.58, 0.62, 0.6, 0.55, 0.65)
    assert score.covers_mean
    assert not score.covers_ci

    wider = score_prediction("causal_discovery", "f1", "asia", 0.5, 0.7, 0.6, 0.55, 0.65)
    assert wider.covers_mean and wider.covers_ci


def test_bias_is_positive_when_a_proportion_is_overpredicted():
    score = score_prediction("causal_discovery", "f1", "asia", 0.7, 0.9, 0.6, 0.55, 0.65)
    assert score.bias > 0


def test_bias_sign_is_flipped_for_shd_so_positive_always_means_optimistic():
    # predicting fewer errors than the truth is optimistic, and must read as positive
    score = score_prediction("causal_discovery", "shd", "asia", 2.0, 4.0, 10.0, 9.0, 11.0)
    assert score.bias > 0

    pessimistic = score_prediction("causal_discovery", "shd", "asia", 15.0, 20.0, 10.0, 9.0, 11.0)
    assert pessimistic.bias < 0


def test_width_ratio_compares_against_the_true_interval():
    score = score_prediction("causal_discovery", "f1", "asia", 0.3, 0.9, 0.6, 0.55, 0.65)
    assert score.width == pytest.approx(0.6)
    assert score.truth_ci_width == pytest.approx(0.1)
    assert score.width_ratio == pytest.approx(6.0)


def test_a_degenerate_truth_interval_gives_an_infinite_width_ratio():
    # the signature of a deterministic algorithm run on unresampled data
    score = score_prediction("causal_discovery", "f1", "asia", 0.3, 0.9, 0.6, 0.6, 0.6)
    assert np.isinf(score.width_ratio)


def test_shd_quantities_are_reported_normalised():
    score = score_prediction("causal_discovery", "shd", "asia", 0.0, 28.0, 10.0, 9.0, 11.0)
    assert score.width == pytest.approx(28.0)
    assert score.width_normalised == pytest.approx(1.0)


def test_an_inverted_interval_is_rejected():
    with pytest.raises(AssertionError):
        score_prediction("causal_discovery", "f1", "asia", 0.9, 0.1, 0.6, 0.55, 0.65)


def test_an_unparsed_prediction_is_kept_with_null_scores():
    row = score_row(prediction_row(lower=None, upper=None, parsed=False), truth_row())
    assert row["scored"] is False
    assert row["covers_mean"] is None
    assert row["truth_mean"] == 0.6


def test_score_predictions_joins_on_every_axis():
    predictions = [prediction_row(metric="f1"), prediction_row(metric="precision")]
    ground_truth = [truth_row(metric="f1"), truth_row(metric="precision", mean=0.9)]
    scored = score_predictions(predictions, ground_truth)
    assert len(scored) == 2
    by_metric = {row["metric"]: row for row in scored}
    assert by_metric["f1"]["covers_mean"]
    assert not by_metric["precision"]["covers_mean"]


def test_a_prediction_without_ground_truth_is_an_error_not_a_skip():
    # a silent skip would mean the two halves ran over different grids and nobody noticed
    with pytest.raises(AssertionError):
        score_predictions([prediction_row(metric="recall")], [truth_row(metric="f1")])


def test_summarize_counts_unparsed_rows_in_the_denominator():
    predictions = [
        prediction_row(metric="f1"),
        prediction_row(metric="precision", lower=None, upper=None, parsed=False),
    ]
    ground_truth = [truth_row(metric="f1"), truth_row(metric="precision")]
    summary = summarize(score_predictions(predictions, ground_truth))
    assert summary["n"] == 2
    assert summary["n_scored"] == 1
    assert summary["n_unparsed"] == 1
    assert summary["coverage"] == 1.0
