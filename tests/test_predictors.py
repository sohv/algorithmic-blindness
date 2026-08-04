import csv

import pytest

from src.conditions import Condition
from src.domains import CAUSAL, metrics_of
from src.metrics.predictors import (
    Heuristic,
    Marginal,
    Oracle,
    UniformRandom,
    baseline_predictions,
    load_human_predictions,
    metric_max,
    write_elicitation_sheet,
)

CONDITIONS = [
    Condition("causal_discovery", "asia", "native", "pc"),
    Condition("causal_discovery", "asia", "native", "fci"),
    Condition("causal_discovery", "synthetic_12", "gaussian", "pc"),
]


def ground_truth_rows():
    means = {"precision": 0.6, "recall": 0.5, "f1": 0.55, "shd": 8.0}
    return [
        {
            "domain": condition.domain,
            "instance": condition.instance,
            "variant": condition.variant,
            "algorithm": condition.algorithm,
            "metric": metric,
            "mean": mean,
            "std": 0.05,
            "ci_lower": mean * 0.9,
            "ci_upper": mean * 1.1,
        }
        for condition in CONDITIONS
        for metric, mean in means.items()
    ]


def test_metric_max_bounds_proportions_at_one_and_shd_at_the_edge_count():
    assert metric_max(CAUSAL, "f1", "asia") == 1.0
    assert metric_max(CAUSAL, "shd", "asia") == 28.0


def test_uniform_random_stays_inside_the_valid_domain():
    baseline = UniformRandom(seed=42)
    for condition in CONDITIONS:
        for metric in metrics_of(CAUSAL):
            lower, upper = baseline.predict(condition, metric)
            assert 0 <= lower <= upper <= metric_max(condition.domain, metric, condition.instance, condition.variant)


def test_uniform_random_is_reproducible_under_a_seed():
    first = [UniformRandom(seed=7).predict(CONDITIONS[0], m) for m in metrics_of(CAUSAL)]
    second = [UniformRandom(seed=7).predict(CONDITIONS[0], m) for m in metrics_of(CAUSAL)]
    assert first == second


def test_the_two_random_schemes_are_different_baselines():
    # the rebuttal described `nested` while v1 ran `sorted_pair`, and their widths differ
    sorted_pair, nested = UniformRandom(1, "sorted_pair"), UniformRandom(1, "nested")
    sorted_widths = [b - a for a, b in (sorted_pair.predict(CONDITIONS[0], "f1") for _ in range(2000))]
    nested_widths = [b - a for a, b in (nested.predict(CONDITIONS[0], "f1") for _ in range(2000))]

    # E[width] is 1/3 for the sorted pair and 1/4 for the nested draw
    assert sum(sorted_widths) / len(sorted_widths) == pytest.approx(1 / 3, abs=0.03)
    assert sum(nested_widths) / len(nested_widths) == pytest.approx(1 / 4, abs=0.03)


def test_an_unknown_random_scheme_is_rejected():
    with pytest.raises(AssertionError):
        UniformRandom(scheme="whatever")


def test_heuristic_produces_a_valid_interval_for_every_metric():
    baseline = Heuristic()
    for condition in CONDITIONS:
        for metric in metrics_of(CAUSAL):
            lower, upper = baseline.predict(condition, metric)
            assert 0 <= lower <= upper <= metric_max(condition.domain, metric, condition.instance, condition.variant)


def test_marginal_never_sees_the_condition_it_is_predicting():
    ground_truth = ground_truth_rows()
    baseline = Marginal(ground_truth)
    held_out = CONDITIONS[0]

    # give the held-out condition an extreme f1 and confirm the interval ignores it
    for row in ground_truth:
        if (row["instance"], row["algorithm"], row["metric"]) == (held_out.instance, held_out.algorithm, "f1"):
            row["mean"] = 0.99
    baseline = Marginal(ground_truth)
    _lower, upper = baseline.predict(held_out, "f1")
    assert upper < 0.99


def test_marginal_declines_when_there_is_nothing_to_pool():
    single = [r for r in ground_truth_rows() if r["metric"] == "f1"][:1]
    assert Marginal(single).predict(CONDITIONS[0], "f1") is None


def test_oracle_reproduces_the_true_interval_and_covers_by_construction():
    ground_truth = ground_truth_rows()
    interval = Oracle(ground_truth).predict(CONDITIONS[0], "f1")
    truth = next(r for r in ground_truth if r["instance"] == "asia" and r["algorithm"] == "pc" and r["metric"] == "f1")
    assert interval == (truth["ci_lower"], truth["ci_upper"])


def test_oracle_returns_nothing_for_a_condition_it_has_no_truth_for():
    assert Oracle(ground_truth_rows()).predict(Condition("causal_discovery", "alarm", "native", "pc"), "f1") is None


def test_baseline_predictions_cover_every_condition_and_metric():
    rows = baseline_predictions(CONDITIONS, ground_truth_rows(), seed=42)
    assert len(rows) == 5 * len(CONDITIONS) * len(metrics_of(CAUSAL))
    assert {r["predictor"] for r in rows} == {"uniform_sorted_pair", "heuristic", "marginal", "analytic", "oracle"}


def test_baseline_rows_use_the_shared_schema():
    row = baseline_predictions(CONDITIONS, ground_truth_rows())[0]
    assert set(row) >= {
        "id",
        "predictor",
        "predictor_kind",
        "instance",
        "variant",
        "algorithm",
        "metric",
        "naming",
        "metadata_level",
        "formulation",
        "lower",
        "upper",
        "parsed",
    }
    # baselines never see a prompt, so the prompt axes are marked rather than faked
    assert row["naming"] == "na"
    assert row["metadata_level"] == "na"
    assert row["formulation"] == 0
    assert row["predictor_kind"] == "baseline"


def test_baselines_score_against_the_same_ground_truth():
    from src.metrics.scoring import score_predictions

    ground_truth = ground_truth_rows()
    rows = baseline_predictions(CONDITIONS, ground_truth)
    scored = score_predictions([r for r in rows if r["parsed"]], ground_truth)
    oracle = [r for r in scored if r["predictor"] == "oracle"]
    assert oracle and all(r["covers_mean"] for r in oracle)


def write_sheet(tmp_path, rows):
    path = tmp_path / "expert.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_human_predictions_load_into_the_shared_schema(tmp_path):
    path = write_sheet(
        tmp_path,
        [
            {
                "predictor": "expert_1",
                "domain": "causal_discovery",
                "instance": "asia",
                "variant": "native",
                "algorithm": "pc",
                "metric": "f1",
                "lower": "0.3",
                "upper": "0.6",
            }
        ],
    )
    rows = load_human_predictions(path)
    assert len(rows) == 1
    assert rows[0]["predictor_kind"] == "human"
    assert rows[0]["lower"] == 0.3
    assert rows[0]["parsed"]


def test_a_sheet_missing_a_column_is_rejected(tmp_path):
    path = write_sheet(
        tmp_path,
        [
            {
                "predictor": "expert_1",
                "domain": "causal_discovery",
                "instance": "asia",
                "metric": "f1",
                "lower": "0.3",
                "upper": "0.6",
            }
        ],
    )
    with pytest.raises(AssertionError, match="missing columns"):
        load_human_predictions(path)


def test_an_inverted_human_interval_is_rejected(tmp_path):
    path = write_sheet(
        tmp_path,
        [
            {
                "predictor": "expert_1",
                "domain": "causal_discovery",
                "instance": "asia",
                "variant": "native",
                "algorithm": "pc",
                "metric": "f1",
                "lower": "0.9",
                "upper": "0.2",
            }
        ],
    )
    with pytest.raises(AssertionError, match="inverted"):
        load_human_predictions(path)


def test_an_unknown_metric_is_rejected(tmp_path):
    path = write_sheet(
        tmp_path,
        [
            {
                "predictor": "expert_1",
                "domain": "causal_discovery",
                "instance": "asia",
                "variant": "native",
                "algorithm": "pc",
                "metric": "accuracy",
                "lower": "0.3",
                "upper": "0.6",
            }
        ],
    )
    with pytest.raises(AssertionError, match="is not a causal_discovery metric"):
        load_human_predictions(path)


def test_the_elicitation_sheet_round_trips_through_the_loader(tmp_path):
    path = write_elicitation_sheet(CONDITIONS, tmp_path / "sheet.csv")
    rows = list(csv.DictReader(path.open()))
    assert len(rows) == len(CONDITIONS) * len(metrics_of(CAUSAL))

    for row in rows:
        row["lower"], row["upper"] = "0.2", "0.8"
    filled = write_sheet(tmp_path, rows)
    loaded = load_human_predictions(filled)
    assert len(loaded) == len(rows)
    assert all(r["predictor_kind"] == "human" for r in loaded)
