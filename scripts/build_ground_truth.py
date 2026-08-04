# runs each algorithm over bootstrap resamples to build the ground truth predictions are scored against.
# uv run -m scripts.build_ground_truth --output_dir results/raw/{date}_ground_truth --domains sorting --instances array_1k --variants random sorted --n_runs 100 --seed 42

import logging
from dataclasses import dataclass, field
from pathlib import Path

import simple_parsing

from src.algorithms.ground_truth import build_ground_truth, ground_truth_report
from src.algorithms.runner import run_conditions
from src.conditions import Grid
from src.domains import ALL_DOMAINS
from src.metrics.io import write_json, write_jsonl
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging
from src.utils.run_dir import resolve_run_dir, run_tag
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


@dataclass
class GroundTruthConfig(Config):
    # empty means "everything the selected domains offer"
    domains: list[str] = field(default_factory=lambda: list(ALL_DOMAINS))
    instances: list[str] = field(default_factory=list)
    algorithms: list[str] = field(default_factory=list)
    variants: list[str] = field(default_factory=list)
    n_runs: int = 100
    data_dir: str = "data/raw"
    # only turn this off to reproduce the pre-rebuild code path, which reports zero variance
    bootstrap: bool = True


def main():
    config = simple_parsing.parse(GroundTruthConfig, add_config_path_arg=True)

    output_dir = resolve_run_dir(config.output_dir, run_tag((config.domains, ALL_DOMAINS)))
    config.output_dir = str(output_dir)
    setup_logging(output_dir)
    set_seed(config.seed)
    write_config_json(config, output_dir)

    grid = Grid(
        domains=config.domains,
        instances=config.instances,
        algorithms=config.algorithms,
        variants=config.variants,
    )
    conditions = grid.conditions()
    assert conditions, f"no conditions from domains={config.domains} instances={config.instances}"
    if not config.bootstrap:
        LOGGER.error("bootstrap is off: every run refits identical data, so the intervals will be degenerate")

    LOGGER.info(f"{len(conditions)} conditions x {config.n_runs} runs: {grid.describe()['n_conditions_by_domain']}")
    runs, diagnostics = run_conditions(conditions, config.n_runs, config.data_dir, config.seed, config.bootstrap)

    ground_truth = build_ground_truth(runs)
    report = ground_truth_report(ground_truth)

    write_jsonl(runs, output_dir / "runs.jsonl")
    write_jsonl(ground_truth, output_dir / "ground_truth.jsonl")
    write_jsonl(diagnostics, output_dir / "diagnostics.jsonl")
    write_json(report, output_dir / "ground_truth_report.json")

    print(f"Ground truth saved to {output_dir / 'ground_truth.jsonl'}")
    print(
        f"{report['n_rows']} rows over {report['n_conditions']} conditions, mean CI width {report['mean_ci_width']:.4f}"
    )
    if report["n_degenerate_causal"]:
        print(
            f"WARNING: {report['n_degenerate_causal']} causal_discovery intervals have zero width. "
            "Bootstrap resampling is off; the run is invalid."
        )
    elif report["n_degenerate_intervals"]:
        print(
            f"{report['n_degenerate_intervals']} intervals have zero width outside causal discovery, "
            "which is expected where the count does not depend on the draw."
        )
    print(
        f"Predict with: uv run -m scripts.run_predictions --ground_truth_dir {output_dir} --output_dir {output_dir.parent / 'predictions_v1'}"
    )


main()
