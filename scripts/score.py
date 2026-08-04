# adds the baseline and human predictors, joins every prediction to its ground truth, and scores them all.
# uv run -m scripts.score --predictions results/raw/260804_predictions_v1/predictions.jsonl --ground_truth_dir results/raw/260804_ground_truth_v1 --output_dir results/raw/260804_scored_v1 --seed 42

import logging
from dataclasses import dataclass
from pathlib import Path

import simple_parsing

from src.conditions import Condition
from src.data.io import load_jsonl
from src.metrics.io import write_json, write_jsonl
from src.metrics.predictors import RANDOM_SCHEMES, baseline_predictions, load_human_predictions
from src.metrics.scoring import ALPHA, score_predictions, summarize
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


@dataclass
class ScoreConfig(Config):
    predictions: str = ""
    ground_truth_dir: str = ""
    human_predictions: str = ""
    random_scheme: str = RANDOM_SCHEMES[0]
    alpha: float = ALPHA
    include_baselines: bool = True


def main():
    config = simple_parsing.parse(ScoreConfig, add_config_path_arg=True)
    assert config.predictions, "--predictions is required, there is no default"
    assert config.ground_truth_dir, "--ground_truth_dir is required, there is no default"

    output_dir = Path(config.output_dir)
    setup_logging(output_dir)
    set_seed(config.seed)
    write_config_json(config, output_dir)

    ground_truth = load_jsonl(Path(config.ground_truth_dir) / "ground_truth.jsonl")
    predictions = load_jsonl(config.predictions)
    LOGGER.info(f"{len(predictions)} model predictions against {len(ground_truth)} ground-truth rows")

    if config.include_baselines:
        # built over exactly the conditions ground truth exists for, so all predictors share rows
        conditions = sorted({Condition(r["domain"], r["instance"], r["variant"], r["algorithm"]) for r in ground_truth})
        diagnostics_path = Path(config.ground_truth_dir) / "diagnostics.jsonl"
        diagnostics = load_jsonl(diagnostics_path) if diagnostics_path.exists() else []
        predictions += baseline_predictions(conditions, ground_truth, config.seed, config.random_scheme, diagnostics)

    if config.human_predictions:
        predictions += load_human_predictions(config.human_predictions)

    scored = score_predictions(predictions, ground_truth, config.alpha)
    summary = summarize(scored)

    write_jsonl(scored, output_dir / "scored.jsonl")
    write_json(summary, output_dir / "score_summary.json")

    print(f"Scored rows saved to {output_dir / 'scored.jsonl'}")
    for key, value in summary.items():
        print(f"{key}: {value}")
    grid_path = Path(config.predictions).parent / "grid.json"
    print(
        f"Analyse with: uv run -m scripts.analyze --scored {output_dir / 'scored.jsonl'} "
        f"--output_dir {output_dir}" + (f" --grid_path {grid_path}" if grid_path.exists() else "")
    )


main()
