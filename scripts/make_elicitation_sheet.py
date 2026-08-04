# writes the blank csv a human expert fills in, so their predictions land in the scoring schema by construction.
# uv run -m scripts.make_elicitation_sheet --ground_truth_dir results/raw/260804_ground_truth_v1 --output_dir results/raw/260804_human_v1 --n_conditions 20 --seed 42

import logging
import random
from dataclasses import dataclass
from pathlib import Path

import simple_parsing

from src.conditions import Condition
from src.data.io import load_jsonl
from src.metrics.predictors import write_elicitation_sheet
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


@dataclass
class ElicitationConfig(Config):
    ground_truth_dir: str = ""
    predictor: str = "expert_1"
    # a full grid is too long to fill in by hand, so a seeded subset is drawn and recorded
    n_conditions: int = 20


def main():
    config = simple_parsing.parse(ElicitationConfig, add_config_path_arg=True)
    assert config.ground_truth_dir, "--ground_truth_dir is required, there is no default"

    output_dir = Path(config.output_dir)
    setup_logging(output_dir)
    set_seed(config.seed)
    write_config_json(config, output_dir)

    ground_truth = load_jsonl(Path(config.ground_truth_dir) / "ground_truth.jsonl")
    conditions = sorted({Condition(r["domain"], r["instance"], r["variant"], r["algorithm"]) for r in ground_truth})

    if config.n_conditions and config.n_conditions < len(conditions):
        conditions = sorted(random.Random(config.seed).sample(conditions, config.n_conditions))

    path = write_elicitation_sheet(conditions, output_dir / f"{config.predictor}_sheet.csv", config.predictor)
    print(f"Elicitation sheet saved to {path}")
    print(f"{len(conditions)} conditions x 4 metrics = {len(conditions) * 4} rows to fill in")
    print(
        f"Score it with: uv run -m scripts.score --human_predictions {path} --predictions ... --ground_truth_dir {config.ground_truth_dir} --output_dir ..."
    )


main()
