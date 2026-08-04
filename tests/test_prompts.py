import pytest

from src.conditions import FORMULATIONS, METADATA_LEVELS, NAMINGS, Condition, PromptSpec, anonymous_name
from src.domains import ALL_DOMAINS, domain
from src.generation.prompts import ALGORITHM_SPECS, TARGET_STATEMENT, build_prompt, dataset_label

CONDITION = Condition("causal_discovery", "asia", "native", "lingam")

DIAGNOSTICS = {
    "samples_per_variable": 1250.0,
    "mean_abs_excess_kurtosis": 0.04,
    "mean_abs_skew": 0.02,
    "non_gaussianity_verdict": "approximately Gaussian",
    "mean_abs_correlation": 0.18,
    "max_abs_correlation": 0.62,
    "condition_number": 4.3,
    "data_type": "discrete",
    "n_edges": 8,
    "edge_density": 0.286,
    "mean_degree": 2.0,
    "max_in_degree": 2,
    "max_out_degree": 2,
}


def test_every_algorithm_in_every_domain_has_a_spec():
    declared = {a for name in ALL_DOMAINS for a in domain(name).algorithms}
    assert set(ALGORITHM_SPECS) == declared


def test_every_axis_combination_builds_a_prompt():
    for naming in NAMINGS:
        for level in METADATA_LEVELS:
            for formulation in FORMULATIONS:
                prompt = build_prompt(CONDITION, PromptSpec(naming, level, formulation), DIAGNOSTICS)
                assert prompt
                assert "{" not in prompt


def test_the_real_name_appears_and_the_anonymized_one_does_not():
    real = build_prompt(CONDITION, PromptSpec("real", "sparse", 1), DIAGNOSTICS)
    anonymized = build_prompt(CONDITION, PromptSpec("anonymized", "sparse", 1), DIAGNOSTICS)

    assert "Asia" in real
    assert "Asia" not in anonymized
    assert anonymous_name("asia") in anonymized


def test_anonymizing_changes_only_the_name():
    real = build_prompt(CONDITION, PromptSpec("real", "full", 1), DIAGNOSTICS)
    anonymized = build_prompt(CONDITION, PromptSpec("anonymized", "full", 1), DIAGNOSTICS)
    assert real.replace("Asia", anonymous_name("asia")) == anonymized


def test_metadata_levels_are_strictly_nested():
    prompts = {level: build_prompt(CONDITION, PromptSpec("real", level, 1), DIAGNOSTICS) for level in METADATA_LEVELS}
    assert len(prompts["sparse"]) < len(prompts["diagnostic"]) < len(prompts["full"])


def test_only_the_richer_levels_carry_the_non_gaussianity_verdict():
    sparse = build_prompt(CONDITION, PromptSpec("real", "sparse", 1), DIAGNOSTICS)
    diagnostic = build_prompt(CONDITION, PromptSpec("real", "diagnostic", 1), DIAGNOSTICS)

    assert "kurtosis" not in sparse.lower()
    assert "kurtosis" in diagnostic.lower()
    assert DIAGNOSTICS["non_gaussianity_verdict"] in diagnostic


def test_the_full_level_states_the_assumption_lingam_depends_on():
    prompt = build_prompt(CONDITION, PromptSpec("real", "full", 1), DIAGNOSTICS)
    assert "NON-GAUSSIAN disturbances" in prompt
    assert "DirectLiNGAM" in prompt
    assert "Structural Hamming Distance" in prompt


def test_the_full_level_exposes_the_true_graph_structure():
    prompt = build_prompt(CONDITION, PromptSpec("real", "full", 1), DIAGNOSTICS)
    assert "True graph structure" in prompt
    assert "Edges: 8" in prompt


def test_a_richer_level_without_diagnostics_is_an_error_not_a_silent_downgrade():
    for level in ["diagnostic", "full"]:
        with pytest.raises(AssertionError, match="requires diagnostics"):
            build_prompt(CONDITION, PromptSpec("real", level, 1), None)


def test_sparse_needs_no_diagnostics():
    assert build_prompt(CONDITION, PromptSpec("real", "sparse", 1), None)


def test_every_formulation_states_the_same_prediction_target():
    # reviewers could not tell whether the interval was for run spread or for the mean
    for formulation in FORMULATIONS:
        prompt = build_prompt(CONDITION, PromptSpec("real", "sparse", formulation), DIAGNOSTICS)
        assert TARGET_STATEMENT.format(n_runs=100) in prompt
        assert "MEAN" in prompt


def test_the_formulations_differ_only_in_the_question():
    prompts = {f: build_prompt(CONDITION, PromptSpec("real", "sparse", f), DIAGNOSTICS) for f in FORMULATIONS}
    assert len(set(prompts.values())) == len(FORMULATIONS)


def test_an_unknown_formulation_is_rejected():
    with pytest.raises(AssertionError, match="unknown formulation"):
        build_prompt(CONDITION, PromptSpec("real", "sparse", 99), DIAGNOSTICS)


def test_an_unknown_naming_is_rejected():
    with pytest.raises(AssertionError, match="unknown naming"):
        dataset_label(CONDITION, "pseudonymized")


def test_every_prompt_asks_for_the_parseable_format():
    from src.metrics.extraction import parse_ranges

    prompt = build_prompt(CONDITION, PromptSpec("real", "sparse", 1), DIAGNOSTICS)
    assert "Precision: [low, high]" in prompt
    # the requested shape must be one the parser accepts, filled with real numbers
    assert set(parse_ranges("Precision: [0.1, 0.2]\nRecall: [0.1, 0.2]\nF1: [0.1, 0.2]\nSHD: [1, 2]")) == {
        "precision",
        "recall",
        "f1",
        "shd",
    }
