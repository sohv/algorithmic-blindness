import os

import pytest

from src.generation.models import (
    FRONTIER_MODELS,
    MID_TIER_MODELS,
    MODELS,
    ModelSpec,
    check_against_catalog,
    openrouter_id,
    resolve,
    tier_of,
    tier_separation,
)
from src.generation.openrouter import fetch_catalog
from src.metrics.report import append_run, contrast_line, main_table, run_section

MAX_MID_OUTPUT_USD_PER_MTOK = 8.0
MAX_FRONTIER_MODELS = 1


def test_the_registry_is_one_frontier_model_and_six_mid_tier_across_six_labs():
    assert len(FRONTIER_MODELS) == MAX_FRONTIER_MODELS
    assert len(MID_TIER_MODELS) == 6
    assert len({m.openrouter_id.split("/")[0] for m in MODELS}) == 6


@pytest.mark.skipif(not os.getenv("OPENROUTER_API_KEY"), reason="reads the live openrouter catalog")
def test_only_the_one_frontier_model_is_expensive():
    """The run is cost-capped. Exactly one model may be dear, and it must be the declared frontier."""
    catalog = fetch_catalog()
    dear = [
        f"{m.key} ({m.tier}): ${float(catalog[m.openrouter_id]['pricing']['completion']) * 1e6:.2f}/Mtok out"
        for m in MID_TIER_MODELS
        if m.openrouter_id in catalog
        and float(catalog[m.openrouter_id]["pricing"]["completion"]) * 1e6 > MAX_MID_OUTPUT_USD_PER_MTOK
    ]
    assert not dear, f"mid-tier over ${MAX_MID_OUTPUT_USD_PER_MTOK:.0f}/Mtok output:\n" + "\n".join(dear)
    assert len(FRONTIER_MODELS) <= MAX_FRONTIER_MODELS, "only one model may be expensive"


def test_model_keys_and_openrouter_ids_are_unique():
    assert len({m.key for m in MODELS}) == len(MODELS)
    assert len({m.openrouter_id for m in MODELS}) == len(MODELS)


def test_every_openrouter_id_is_provider_prefixed():
    # a bare slug without its "<provider>/" prefix only surfaces at the first paid call
    for model in MODELS:
        assert "/" in model.openrouter_id, model.openrouter_id


def test_the_registry_spans_more_than_one_lab():
    # a shared failure across one lab's models is a lineage effect, not a finding about LLMs
    providers = {m.openrouter_id.split("/")[0] for m in MODELS}
    assert len(providers) >= 4


def test_resolve_round_trips_and_rejects_unknown_keys():
    assert resolve("gemini").tier == "mid"
    assert tier_of("gemini") == "mid"
    assert openrouter_id("haiku").startswith("anthropic/")
    with pytest.raises(AssertionError, match="unknown model"):
        resolve("not_a_model")


def test_an_unknown_tier_is_rejected_at_construction():
    with pytest.raises(AssertionError, match="unknown tier"):
        ModelSpec("x", "provider/x", "premium", "X")


def test_tier_separation_flags_a_frontier_model_that_is_no_longer_dearer_than_the_mid_tier():
    ordered = {
        "opus": {"declared_tier": "frontier", "output_usd_per_mtok": 25.0},
        "gpt": {"declared_tier": "frontier", "output_usd_per_mtok": 14.0},
        "gemini": {"declared_tier": "mid", "output_usd_per_mtok": 12.0},
    }
    assert tier_separation(ordered) == []

    repriced = {**ordered, "gemini": {"declared_tier": "mid", "output_usd_per_mtok": 14.0}}
    assert len(tier_separation(repriced)) == 1
    assert "gpt" in tier_separation(repriced)[0]


def test_tier_separation_says_nothing_when_only_one_tier_is_present():
    assert tier_separation({"opus": {"declared_tier": "frontier", "output_usd_per_mtok": 25.0}}) == []


def catalog_entry(prompt: float, completion: float, context: int = 200000, controls=("temperature", "seed")) -> dict:
    # openrouter quotes usd per token, as strings
    return {
        "pricing": {"prompt": str(prompt / 1e6), "completion": str(completion / 1e6)},
        "context_length": context,
        "supported_parameters": [*controls, "max_tokens"],
    }


def controls_of(model) -> tuple[str, ...]:
    return tuple(c for c in ("temperature", "seed") if getattr(model, f"supports_{c}"))


def full_catalog() -> dict:
    return {
        m.openrouter_id: catalog_entry(5.0, 25.0 if m.tier == "frontier" else 15.0, controls=controls_of(m))
        for m in MODELS
    }


def test_a_matching_catalog_passes():
    report = check_against_catalog(full_catalog())
    assert report["ok"]
    assert report["n_found"] == len(MODELS)
    assert not report["missing_ids"] and not report["tier_mismatches"]
    assert report["by_model"][MID_TIER_MODELS[0].key]["output_usd_per_mtok"] == pytest.approx(15.0)
    assert report["by_model"][FRONTIER_MODELS[0].key]["output_usd_per_mtok"] == pytest.approx(25.0)


def test_a_missing_slug_fails_the_check():
    catalog = full_catalog()
    catalog.pop(MODELS[0].openrouter_id)
    report = check_against_catalog(catalog)
    assert not report["ok"]
    assert MODELS[0].openrouter_id in report["missing_ids"]


def test_a_repriced_frontier_model_fails_the_tier_check():
    # a frontier model repriced below the mid tier silently invalidates the tier contrast
    catalog = full_catalog()
    frontier = FRONTIER_MODELS[0]
    catalog[frontier.openrouter_id] = catalog_entry(1.0, 4.0, controls=controls_of(frontier))
    report = check_against_catalog(catalog)
    assert not report["ok"]
    assert any(frontier.key in m for m in report["tier_mismatches"])


def coverage_rows() -> list[dict]:
    return [
        {
            "predictor": "opus",
            "tier": "frontier",
            "n": 208,
            "coverage": 0.394,
            "coverage_ci_lower": 0.33,
            "coverage_ci_upper": 0.46,
            "mean_interval_score": 0.51,
            "mean_width_ratio": 8.2,
            "share_optimistic": 0.71,
        },
        {
            "predictor": "uniform_sorted_pair",
            "tier": "baseline",
            "n": 208,
            "coverage": 0.365,
            "coverage_ci_lower": 0.30,
            "coverage_ci_upper": 0.43,
            "mean_interval_score": 1.9,
            "mean_width_ratio": float("nan"),
            "share_optimistic": 0.5,
        },
    ]


def analysis_fixture() -> dict:
    return {
        "coverage_by_predictor": coverage_rows(),
        "parse_rates": [
            {"predictor": "opus", "parse_rate": 1.0},
            {"predictor": "qwen", "parse_rate": 0.62},
        ],
        "metadata_effect": {
            "full_vs_sparse": {
                "available": True,
                "overall": {
                    "n_pairs": 96,
                    "coverage_left": 0.41,
                    "coverage_right": 0.39,
                    "coverage_difference": 0.02,
                    "coverage_test": {"p_value": 0.42},
                },
            },
            "diagnostic_vs_sparse": {"available": False},
        },
        "naming_effect": {"real_vs_anonymized": {"available": False}},
        "dataset_kind_effect": {"benchmark_vs_synthetic": {"available": False}},
        "frontier_vs_mid": {"available": False},
        "against_baseline": {
            "available": True,
            "baseline": "uniform_sorted_pair",
            "baseline_rate": 0.365,
            "by_predictor": {"opus": {"difference": 0.029}},
            "holm": {"opus": {"significant_at_0_05": False}},
        },
        "noise_effect": {"available": False},
    }


def test_main_table_renders_a_markdown_table():
    table = main_table(coverage_rows())
    assert "| Predictor" in table and "| Tier" in table
    assert "opus" in table and "frontier" in table
    assert "0.394" in table


def test_a_non_finite_width_ratio_renders_as_a_dash():
    assert "—" in main_table(coverage_rows())


def test_main_table_handles_no_predictions():
    assert "no scored predictions" in main_table([])


def test_a_contrast_reports_its_paired_test():
    line = contrast_line("metadata: full vs sparse", analysis_fixture()["metadata_effect"]["full_vs_sparse"])
    assert "0.390 → 0.410" in line
    assert "p=0.4200" in line
    assert "96 pairs" in line


def test_an_unavailable_contrast_says_so_rather_than_inventing_a_number():
    assert "not run in this grid" in contrast_line("naming", {"available": False})


def test_a_significant_contrast_is_starred():
    contrast = {
        "available": True,
        "overall": {
            "n_pairs": 50,
            "coverage_left": 0.6,
            "coverage_right": 0.3,
            "coverage_difference": 0.3,
            "coverage_test": {"p_value": 0.001},
        },
    }
    assert "*" in contrast_line("x", contrast)


def test_a_run_section_records_scope_parse_warnings_and_contrasts():
    section = run_section(
        "260804_scored_v1",
        analysis_fixture(),
        {"n": 1000, "n_scored": 900},
        {"n_models": 6, "n_conditions": 52, "n_requests": 5616, "namings": ["real", "anonymized"]},
    )
    assert "## 260804_scored_v1" in section
    assert "6 models" in section and "900/1000 rows scored" in section
    assert "git `" in section
    # a model that answered 62% of the time must be flagged next to its coverage number
    assert "Low parse rate" in section and "qwen 62%" in section
    assert "no predictor beat it after Holm correction" in section


def test_append_run_creates_the_file_then_appends_without_rewriting(tmp_path):
    path = tmp_path / "RESULTS.md"

    append_run(path, "run_one", analysis_fixture(), {"n": 10, "n_scored": 10})
    first = path.read_text()
    assert first.startswith("# Results")
    assert "## run_one" in first

    append_run(path, "run_two", analysis_fixture(), {"n": 10, "n_scored": 10})
    second = path.read_text()
    assert second.startswith(first)
    assert second.count("# Results") == 1
    assert second.index("## run_one") < second.index("## run_two")


def test_a_model_that_stops_honouring_temperature_is_caught():
    """openrouter drops an unsupported control silently, so a change here would go unnoticed."""
    catalog = full_catalog()
    honouring = next(m for m in MODELS if m.supports_temperature)
    catalog[honouring.openrouter_id] = catalog_entry(5.0, 25.0, controls=("seed",))
    report = check_against_catalog(catalog)
    assert not report["ok"]
    assert any(honouring.key in m and "temperature" in m for m in report["control_mismatches"])
