import json

import pytest

from src.metrics.extraction import (
    aggregate_formulations,
    extract,
    extract_records,
    extraction_report,
    parse_confidence,
    parse_metric,
    parse_ranges,
)

CANONICAL = "Precision: [0.60, 0.90]\nRecall: [0.70, 0.95]\nF1: [0.65, 0.90]\nSHD: [2, 8]"


def test_parses_the_format_the_prompt_asks_for():
    ranges = parse_ranges(CANONICAL)
    assert set(ranges) == {"precision", "recall", "f1", "shd"}
    assert (ranges["precision"].lower, ranges["precision"].upper) == (0.60, 0.90)
    assert (ranges["shd"].lower, ranges["shd"].upper) == (2.0, 8.0)


@pytest.mark.parametrize(
    "response",
    [
        "**Precision**: [0.60, 0.90]",
        "- Precision: [0.60, 0.90]",
        "1. Precision: [0.60, 0.90]",
        "> Precision: [0.60, 0.90]",
        "Precision: (0.60, 0.90)",
        "Precision: [0.60 - 0.90]",
        "Precision: 0.60 - 0.90",
        "Precision: 0.60 to 0.90",
        "Precision: 0.60 – 0.90",
        "Precision range: [0.60, 0.90]",
        "precision = [0.60, 0.90]",
    ],
)
def test_parses_the_shapes_models_actually_emit(response):
    estimate = parse_metric(response, "precision")
    assert (estimate.lower, estimate.upper) == (0.60, 0.90)


def test_f1_label_digit_is_never_read_as_a_value():
    # the "1" in the F1 label sits right before the colon, so a bare-number parser reads it
    estimate = parse_metric("F1-Score: [0.65, 0.90]", "f1")
    assert (estimate.lower, estimate.upper) == (0.65, 0.90)


def test_shd_matches_its_spelled_out_name():
    estimate = parse_metric("Structural Hamming Distance: [4, 12]", "shd")
    assert (estimate.lower, estimate.upper) == (4.0, 12.0)


def test_reversed_bounds_are_sorted():
    estimate = parse_metric("Precision: [0.90, 0.60]", "precision")
    assert (estimate.lower, estimate.upper) == (0.60, 0.90)


def test_a_reasoning_model_answer_wins_over_its_working():
    response = "First I might guess Precision: [0.10, 0.20].\nOn reflection:\nPrecision: [0.60, 0.90]"
    assert parse_metric(response, "precision").lower == 0.60


def test_out_of_range_proportions_are_rejected():
    assert parse_metric("Precision: [0.6, 1.4]", "precision") is None
    assert parse_metric("Recall: [-0.2, 0.5]", "recall") is None


def test_shd_is_not_bounded_at_one():
    assert parse_metric("SHD: [40, 120]", "shd").upper == 120.0


def test_negative_shd_is_rejected():
    assert parse_metric("SHD: [-4, 12]", "shd") is None


def test_a_refusal_parses_to_nothing():
    assert parse_ranges("I cannot reliably estimate this without the data.") == {}
    assert parse_ranges("") == {}
    assert parse_ranges(None) == {}


def test_a_partial_response_keeps_only_what_parsed():
    assert set(parse_ranges("Precision: [0.6, 0.9]\nRecall: unknown")) == {"precision"}


def test_the_template_placeholder_is_not_a_match():
    # models sometimes echo the requested format before answering
    assert parse_ranges("Precision: [X.XX, X.XX]\nRecall: [X.XX, X.XX]") == {}


def test_confidence_defaults_to_medium_and_reads_a_stated_one():
    assert parse_confidence(CANONICAL) == "medium"
    assert parse_confidence(CANONICAL + "\nConfidence: high") == "high"
    assert parse_confidence(CANONICAL + "\nconfidence: moderate") == "medium"


def write_responses(tmp_path, responses, naming="real", metadata_level="sparse", model="claude"):
    records = [
        {
            "id": f"asia__native__pc__{naming}__{metadata_level}__f{i}__{model}",
            "domain": "causal_discovery",
            "instance": "asia",
            "variant": "native",
            "algorithm": "pc",
            "naming": naming,
            "metadata_level": metadata_level,
            "formulation": i,
            "model": model,
            "tier": "mid",
            "response": response,
        }
        for i, response in enumerate(responses, start=1)
    ]
    path = tmp_path / "responses.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return path


def test_extract_records_emits_one_row_per_metric_including_failures(tmp_path):
    records = extract_records(write_responses(tmp_path, [CANONICAL, "no idea"]))
    assert len(records) == 8
    assert sum(r["parsed"] for r in records) == 4
    unparsed = [r for r in records if not r["parsed"]]
    assert all(r["lower"] is None and r["confidence"] is None for r in unparsed)


def test_extract_records_carries_every_axis_onto_the_prediction(tmp_path):
    record = extract_records(write_responses(tmp_path, [CANONICAL], naming="anonymized", metadata_level="full"))[0]
    assert record["predictor"] == "claude"
    assert record["predictor_kind"] == "llm"
    assert record["variant"] == "native"
    assert record["naming"] == "anonymized"
    assert record["metadata_level"] == "full"
    assert record["formulation"] == 1


def test_a_null_response_is_recorded_not_dropped(tmp_path):
    records = extract_records(write_responses(tmp_path, [None]))
    assert len(records) == 4
    assert not any(r["parsed"] for r in records)


def test_extraction_report_breaks_the_parse_rate_down(tmp_path):
    records = extract_records(write_responses(tmp_path, [CANONICAL, "Precision: [0.6, 0.9]"]))
    report = extraction_report(records)
    assert report["overall"] == {"n": 8, "n_parsed": 5, "parse_rate": 0.625}
    assert report["by_predictor"]["claude"]["n_parsed"] == 5
    assert report["by_metric"]["precision"]["parse_rate"] == 1.0
    assert report["by_metric"]["recall"]["parse_rate"] == 0.5
    assert report["by_metadata_level"]["sparse"]["n"] == 8


def test_aggregation_averages_across_wordings(tmp_path):
    responses = [
        "Precision: [0.3, 0.5]\nRecall: [0.3, 0.5]\nF1: [0.3, 0.5]\nSHD: [1, 3]",
        "Precision: [0.5, 0.7]\nRecall: [0.5, 0.7]\nF1: [0.5, 0.7]\nSHD: [3, 5]",
    ]
    records = extract_records(write_responses(tmp_path, responses))
    aggregated = aggregate_formulations(records)

    assert len(aggregated) == 4
    precision = next(r for r in aggregated if r["metric"] == "precision")
    assert precision["formulation"] == 0
    assert precision["lower"] == pytest.approx(0.4)
    assert precision["upper"] == pytest.approx(0.6)
    assert precision["n_formulations"] == 2


def test_aggregation_needs_at_least_two_wordings(tmp_path):
    records = extract_records(write_responses(tmp_path, [CANONICAL]))
    assert aggregate_formulations(records) == []
    assert len(aggregate_formulations(records, min_formulations=1)) == 4


def test_aggregation_keeps_naming_arms_apart(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    real = extract_records(write_responses(tmp_path / "a", [CANONICAL, CANONICAL], naming="real"))
    anonymized = extract_records(write_responses(tmp_path / "b", [CANONICAL, CANONICAL], naming="anonymized"))
    aggregated = aggregate_formulations(real + anonymized)
    assert {r["naming"] for r in aggregated} == {"real", "anonymized"}
    assert len(aggregated) == 8


def test_extract_writes_predictions_with_the_aggregated_rows(tmp_path):
    records, report = extract(write_responses(tmp_path, [CANONICAL] * 3), tmp_path / "out")

    assert report["overall"]["parse_rate"] == 1.0
    assert (tmp_path / "out" / "predictions.jsonl").exists()
    per_formulation = [r for r in records if r["formulation"] > 0]
    aggregated = [r for r in records if r["formulation"] == 0]
    assert len(per_formulation) == 12
    assert len(aggregated) == 4
    assert all(r["predictor_kind"] == "llm" for r in records)


def test_extracted_predictions_score_against_ground_truth(tmp_path):
    from src.metrics.scoring import score_predictions

    records, _ = extract(write_responses(tmp_path, [CANONICAL] * 3), tmp_path / "out")
    ground_truth = [
        {
            "domain": "causal_discovery",
            "instance": "asia",
            "variant": "native",
            "algorithm": "pc",
            "metric": metric,
            "mean": mean,
            "std": 0.01,
            "ci_lower": mean - 0.02,
            "ci_upper": mean + 0.02,
        }
        for metric, mean in [("precision", 0.7), ("recall", 0.8), ("f1", 0.75), ("shd", 5.0)]
    ]
    scored = score_predictions(records, ground_truth)
    assert len(scored) == len(records)
    assert all(row["covers_mean"] for row in scored)
