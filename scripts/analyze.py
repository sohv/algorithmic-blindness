# runs every analysis over the scored table and writes the paper's tables.
# uv run -m scripts.analyze --scored results/raw/260804_scored_v1/scored.jsonl --output_dir results/raw/260804_scored_v1

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import simple_parsing

from src.data.io import load_jsonl
from src.metrics.analysis import analyze, coverage_table, to_frame
from src.metrics.io import write_json
from src.metrics.report import append_run
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging

LOGGER = logging.getLogger(__name__)

TABLES = {
    "coverage_by_predictor": ["predictor", "tier"],
    "coverage_by_algorithm": "algorithm",
    "coverage_by_metric": "metric",
    "coverage_by_dataset": "instance",
    "coverage_by_metadata_level": ["predictor", "metadata_level"],
    "coverage_by_naming": ["predictor", "naming"],
    "coverage_by_noise": ["algorithm", "variant"],
}


@dataclass
class AnalyzeConfig(Config):
    scored: str = ""
    baseline: str = "uniform_sorted_pair"
    # the headline table reads the formulation-averaged rows; 1-3 are the individual wordings
    headline_formulation: int = 0
    # every run appends one section here; the file is the chronological record of the project
    results_md: str = "results/RESULTS.md"
    # run.json / grid.json from the prediction stage, so the section records the grid it ran under
    grid_path: str = ""


def main():
    config = simple_parsing.parse(AnalyzeConfig, add_config_path_arg=True)
    assert config.scored, "--scored is required, point it at a scored.jsonl"

    output_dir = Path(config.output_dir)
    setup_logging(output_dir)
    write_config_json(config, output_dir)

    scored = load_jsonl(config.scored)
    report = analyze(scored, config.baseline)
    write_json(report, output_dir / "analysis.json")

    summary_path = Path(config.scored).parent / "score_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    grid = json.loads(Path(config.grid_path).read_text()) if config.grid_path else None
    results_md = append_run(config.results_md, output_dir.name, report, summary, grid)

    frame = to_frame(scored)
    headline = frame[frame["formulation"] == config.headline_formulation]
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    for name, by in TABLES.items():
        table = coverage_table(headline if name.startswith("coverage_by_predictor") else frame, by)
        table.to_csv(tables_dir / f"{name}.csv", index=False)

    main_table = coverage_table(headline, ["predictor", "tier"])
    (tables_dir / "coverage_by_predictor.tex").write_text(main_table.to_latex(index=False, float_format="%.3f"))

    print(f"Analysis saved to {output_dir / 'analysis.json'}")
    print(f"Run appended to {results_md}")
    print(f"Tables saved to {tables_dir}")
    print()
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(
            main_table[
                [
                    "predictor",
                    "n",
                    "coverage",
                    "coverage_ci_lower",
                    "coverage_ci_upper",
                    "mean_interval_score",
                    "mean_width_ratio",
                    "share_optimistic",
                ]
            ].to_string(index=False)
        )

    metadata = report["metadata_effect"]["full_vs_sparse"]
    if metadata.get("available"):
        overall = metadata["overall"]
        print()
        print(
            f"metadata full vs sparse: coverage {overall['coverage_right']:.3f} -> {overall['coverage_left']:.3f} "
            f"(p={overall['coverage_test'].get('p_value', float('nan')):.4f}, n={overall['n_pairs']})"
        )

    naming = report["naming_effect"]["real_vs_anonymized"]
    if naming.get("available"):
        overall = naming["overall"]
        print(
            f"naming real vs anonymized: coverage {overall['coverage_right']:.3f} -> {overall['coverage_left']:.3f} "
            f"(p={overall['coverage_test'].get('p_value', float('nan')):.4f}, n={overall['n_pairs']})"
        )

    print(f"Plot with: uv run -m scripts.plot --scored {config.scored} --output_dir {output_dir}")


main()
