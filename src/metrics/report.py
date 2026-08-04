# appends one section per run to results/RESULTS.md.

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.utils.git import get_git_hash

LOGGER = logging.getLogger(__name__)

HEADER = """# Results

One section per run, appended in order. Never edit or reorder a past section — a result that turned
out to be wrong gets a new section saying so, not a rewrite.

Every section records the git hash and the grid it ran under. Two sections are only comparable if
those match; where they don't, the difference in the grid is usually the explanation.

Read the parse rate before any coverage number: a model that answered 70% of the time is not
comparable to one that answered every time.
"""

MAIN_COLUMNS = {
    "predictor": "Predictor",
    "tier": "Tier",
    "n": "n",
    "coverage": "Coverage",
    "coverage_ci": "95% CI",
    "mean_interval_score": "Interval score",
    "mean_width_ratio": "Width ratio",
    "share_optimistic": "Optimistic",
}


def format_float(value, places: int = 3) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    if isinstance(value, float) and not pd.notna(value):
        return "—"
    return f"{value:.{places}f}"


def main_table(coverage: list[dict]) -> str:
    """The headline table: one row per predictor, coverage with its interval, sharpness and bias."""
    if not coverage:
        return "_no scored predictions_\n"

    rows = []
    for entry in coverage:
        rows.append(
            {
                "Predictor": entry.get("predictor", "—"),
                "Tier": entry.get("tier", "—"),
                "n": entry.get("n", 0),
                "Coverage": format_float(entry.get("coverage")),
                "95% CI": f"[{format_float(entry.get('coverage_ci_lower'))}, {format_float(entry.get('coverage_ci_upper'))}]",
                "Interval score": format_float(entry.get("mean_interval_score")),
                "Width ratio": format_float(entry.get("mean_width_ratio"), 1),
                "Optimistic": format_float(entry.get("share_optimistic"), 2),
            }
        )
    return pd.DataFrame(rows).to_markdown(index=False) + "\n"


def domain_table(coverage: list[dict]) -> str:
    """Coverage per domain. Sorting is the one to read first: its expected counts are published, so
    a miss there cannot be explained by the task being unreasonable."""
    if not coverage:
        return "_no scored predictions_\n"

    rows = [
        {
            "Domain": entry.get("domain", "—"),
            "n": entry.get("n", 0),
            "Coverage": format_float(entry.get("coverage")),
            "95% CI": f"[{format_float(entry.get('coverage_ci_lower'))}, {format_float(entry.get('coverage_ci_upper'))}]",
            "Interval score": format_float(entry.get("mean_interval_score")),
            "Width ratio": format_float(entry.get("mean_width_ratio"), 1),
        }
        for entry in coverage
    ]
    return pd.DataFrame(rows).to_markdown(index=False) + "\n"


def contrast_line(name: str, contrast: dict) -> str:
    """One line per experimental contrast, with its paired test rather than a bare difference."""
    if not contrast.get("available"):
        return f"- **{name}**: not run in this grid"

    overall = contrast["overall"]
    test = overall.get("coverage_test", {})
    p_value = test.get("p_value")
    significance = "" if p_value is None else f", p={p_value:.4f}{' *' if p_value < 0.05 else ''}"
    return (
        f"- **{name}**: coverage {format_float(overall['coverage_right'])} → "
        f"{format_float(overall['coverage_left'])} "
        f"({overall['coverage_difference']:+.3f} over {overall['n_pairs']} pairs{significance})"
    )


def run_section(run_name: str, analysis: dict, summary: dict, grid: dict | None = None) -> str:
    """One markdown section for one run."""
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"\n## {run_name} — {stamp}", ""]

    scope = [f"git `{get_git_hash()}`"]
    if grid:
        scope.append(f"{grid.get('n_models', '?')} models")
        scope.append(f"{grid.get('n_conditions', '?')} conditions")
        scope.append(f"{grid.get('n_requests', '?')} requests")
        for axis in ("domains", "variants", "namings", "metadata_levels", "models"):
            if grid.get(axis):
                scope.append(f"{axis}={'/'.join(str(v) for v in grid[axis])}")
    scope.append(f"{summary.get('n_scored', 0)}/{summary.get('n', 0)} rows scored")
    lines.append("**Scope:** " + " | ".join(scope))
    lines.append("")

    lines.append(main_table(analysis.get("coverage_by_predictor", [])))

    by_domain = analysis.get("coverage_by_domain", [])
    if len(by_domain) > 1:
        lines.append("**By domain** — the whole point of three domains is that a failure confined to")
        lines.append("one of them is a fact about that domain, not about algorithmic reasoning.")
        lines.append("")
        lines.append(domain_table(by_domain))

    parse_rates = analysis.get("parse_rates", [])
    poor = [row for row in parse_rates if row.get("parse_rate", 1.0) < 0.9]
    if poor:
        joined = ", ".join(f"{row['predictor']} {row['parse_rate']:.0%}" for row in poor)
        lines.append(f"**Low parse rate — coverage below is not comparable for:** {joined}")
        lines.append("")

    lines.append("**Contrasts**")
    metadata = analysis.get("metadata_effect", {})
    lines.append(contrast_line("metadata: full vs sparse", metadata.get("full_vs_sparse", {})))
    lines.append(contrast_line("metadata: diagnostic vs sparse", metadata.get("diagnostic_vs_sparse", {})))
    lines.append(
        contrast_line("naming: real vs anonymized", analysis.get("naming_effect", {}).get("real_vs_anonymized", {}))
    )
    lines.append(
        contrast_line(
            "dataset: benchmark vs synthetic", analysis.get("dataset_kind_effect", {}).get("benchmark_vs_synthetic", {})
        )
    )
    lines.append(contrast_line("tier: frontier vs mid", analysis.get("frontier_vs_mid", {})))
    lines.append("")

    baseline = analysis.get("against_baseline", {})
    if baseline.get("available"):
        beat = [
            name
            for name, stats in baseline["by_predictor"].items()
            if stats["difference"] > 0 and baseline["holm"].get(name, {}).get("significant_at_0_05")
        ]
        lines.append(
            f"**Against `{baseline['baseline']}`** (rate {format_float(baseline['baseline_rate'])}): "
            + (
                f"{', '.join(beat)} beat it after Holm correction"
                if beat
                else "no predictor beat it after Holm correction"
            )
        )
        lines.append("")

    noise = analysis.get("noise_effect", {})
    if noise.get("available"):
        for algorithm, stats in noise.get("per_algorithm_gaussian_vs_not", {}).items():
            lines.append(
                f"- noise, {algorithm}: gaussian {format_float(stats['coverage_gaussian'])} vs "
                f"non-gaussian {format_float(stats['coverage_non_gaussian'])}"
            )
        lines.append("")

    return "\n".join(lines)


def append_run(
    results_path: Path | str,
    run_name: str,
    analysis: dict,
    summary: dict,
    grid: dict | None = None,
) -> Path:
    """Append this run's section. Creates the file with its header if it does not exist yet."""
    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    if not results_path.exists():
        results_path.write_text(HEADER)
    elif f"\n## {run_name} " in results_path.read_text():
        LOGGER.warning(f"{run_name} already has a section in {results_path}; appending a second one")

    with results_path.open("a") as handle:
        handle.write(run_section(run_name, analysis, summary, grid))

    LOGGER.info(f"appended run {run_name} to {results_path}")
    return results_path
