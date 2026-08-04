import pytest

from src.metrics.stats import (
    binomial_test_against,
    bootstrap_ci,
    holm_bonferroni,
    mcnemar_test,
    paired_difference,
    proportion,
    wilson_interval,
)


def test_wilson_interval_stays_inside_zero_one_near_the_boundary():
    # the normal approximation runs below zero here, which is why Wilson is used
    lower, upper = wilson_interval(2, 200)
    assert 0.0 <= lower < upper <= 1.0
    assert lower > 0.0


def test_wilson_interval_narrows_as_n_grows():
    small = wilson_interval(10, 50)
    large = wilson_interval(200, 1000)
    assert (large[1] - large[0]) < (small[1] - small[0])


def test_wilson_interval_handles_no_observations():
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_proportion_brackets_its_own_rate():
    result = proportion([True] * 30 + [False] * 70)
    assert result.n == 100
    assert result.rate == 0.3
    assert result.ci_lower < 0.3 < result.ci_upper


def test_binomial_test_detects_a_rate_far_from_the_baseline():
    result = binomial_test_against([True] * 5 + [False] * 195, null_rate=0.365)
    assert result["rate"] == pytest.approx(0.025)
    assert result["p_value"] < 0.001
    assert result["significant_at_0_05"]


def test_binomial_test_does_not_flag_a_rate_at_the_baseline():
    result = binomial_test_against([True] * 37 + [False] * 63, null_rate=0.365)
    assert not result["significant_at_0_05"]


def test_mcnemar_uses_only_the_discordant_pairs():
    # 20 pairs where a wins, 2 where b wins, and a long tail of agreement that carries no signal
    a = [True] * 20 + [False] * 2 + [True] * 50
    b = [False] * 20 + [True] * 2 + [True] * 50
    result = mcnemar_test(a, b)
    assert result["n_discordant"] == 22
    assert result["a_only"] == 20
    assert result["p_value"] < 0.001
    assert result["significant_at_0_05"]


def test_mcnemar_reports_no_effect_when_the_predictors_agree():
    shared = [True, False, True, False]
    result = mcnemar_test(shared, shared)
    assert result["n_discordant"] == 0
    assert result["p_value"] == 1.0


def test_mcnemar_requires_aligned_inputs():
    with pytest.raises(AssertionError):
        mcnemar_test([True, False], [True])


def test_bootstrap_ci_brackets_the_estimate():
    values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    result = bootstrap_ci(values, n_bootstrap=2000, seed=42)
    assert result["ci_lower"] < result["estimate"] < result["ci_upper"]


def test_bootstrap_ci_is_reproducible_under_a_seed():
    values = [0.1, 0.4, 0.9, 0.2]
    assert bootstrap_ci(values, seed=7, n_bootstrap=500) == bootstrap_ci(values, seed=7, n_bootstrap=500)


def test_bootstrap_ci_drops_non_finite_values():
    result = bootstrap_ci([0.5, float("inf"), None, 0.5], n_bootstrap=200)
    assert result["n"] == 2


def test_paired_difference_reports_effect_size_and_direction():
    result = paired_difference([0.8, 0.9, 0.85, 0.75], [0.5, 0.6, 0.55, 0.45])
    assert result["mean_difference"] == pytest.approx(0.3)
    assert result["cohens_d"] > 1
    assert result["significant_at_0_05"]


def test_paired_difference_needs_two_complete_pairs():
    assert "error" in paired_difference([0.5], [0.4])
    assert "error" in paired_difference([0.5, None], [0.4, 0.3])


def test_holm_is_monotone_and_never_below_the_raw_p_value():
    corrected = holm_bonferroni({"a": 0.001, "b": 0.02, "c": 0.04, "d": 0.9})
    for values in corrected.values():
        assert values["p_adjusted"] >= values["p_value"]
    ordered = [corrected[name]["p_adjusted"] for name in ["a", "b", "c", "d"]]
    assert ordered == sorted(ordered)


def test_holm_is_stricter_than_an_uncorrected_threshold():
    corrected = holm_bonferroni({f"model_{i}": 0.04 for i in range(5)})
    # every raw p is under 0.05, none survives the correction
    assert not any(values["significant_at_0_05"] for values in corrected.values())
