# asks every model to predict each algorithm's performance, across the naming, metadata and wording axes.
# uv run -m scripts.run_predictions --ground_truth_dir results/raw/260804_ground_truth_v1 --output_dir results/raw/{date}_predictions --models gpt grok --metadata_levels sparse full --num_tasks 10 --seed 42

import logging
from dataclasses import dataclass, field
from pathlib import Path

import simple_parsing
from dotenv import load_dotenv

from src.conditions import FORMULATIONS, METADATA_LEVELS, NAMINGS, Grid
from src.data.io import load_jsonl
from src.domains import ALL_DOMAINS
from src.generation.models import ALL_MODELS
from src.generation.query import run_requests
from src.metrics.io import write_json
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging
from src.utils.run_dir import resolve_run_dir, run_tag
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


@dataclass
class PredictionConfig(Config):
    ground_truth_dir: str = ""
    models: list[str] = field(default_factory=lambda: ["claude"])
    # empty means "everything the selected domains offer"
    domains: list[str] = field(default_factory=lambda: list(ALL_DOMAINS))
    instances: list[str] = field(default_factory=list)
    algorithms: list[str] = field(default_factory=list)
    variants: list[str] = field(default_factory=list)
    namings: list[str] = field(default_factory=lambda: list(NAMINGS))
    metadata_levels: list[str] = field(default_factory=lambda: list(METADATA_LEVELS))
    formulations: list[int] = field(default_factory=lambda: list(FORMULATIONS))
    sleep_seconds: float = 1.0


def main():
    load_dotenv()
    config = simple_parsing.parse(PredictionConfig, add_config_path_arg=True)
    assert config.ground_truth_dir, "--ground_truth_dir is required: the prompts need the dataset diagnostics"

    models = ALL_MODELS if "all" in config.models else config.models
    output_dir = resolve_run_dir(config.output_dir, run_tag((config.domains, ALL_DOMAINS), (models, ALL_MODELS)))
    config.output_dir = str(output_dir)
    setup_logging(output_dir)
    set_seed(config.seed)
    write_config_json(config, output_dir)

    diagnostics = {row["id"]: row for row in load_jsonl(Path(config.ground_truth_dir) / "diagnostics.jsonl")}
    LOGGER.info(f"loaded diagnostics for {len(diagnostics)} (dataset, noise) pairs")

    grid = Grid(
        domains=config.domains,
        instances=config.instances,
        algorithms=config.algorithms,
        variants=config.variants,
        namings=config.namings,
        metadata_levels=config.metadata_levels,
        formulations=config.formulations,
        models=models,
    )
    write_json(grid.describe(), output_dir / "grid.json")

    requests = grid.requests()
    if config.num_tasks:
        requests = requests[: config.num_tasks]
    LOGGER.info(f"{len(requests)} requests: {grid.describe()}")

    records = run_requests(requests, diagnostics, output_dir, config.sleep_seconds)
    n_failed = sum(1 for r in records if r["response"] is None)

    print(f"Responses saved to {output_dir / 'responses.jsonl'}")
    if n_failed:
        print(f"{n_failed}/{len(records)} calls failed and were written with a null response")
    print(
        f"Parse with: uv run -m scripts.parse_predictions --dataset_path {output_dir / 'responses.jsonl'} --output_dir {output_dir}"
    )


main()
