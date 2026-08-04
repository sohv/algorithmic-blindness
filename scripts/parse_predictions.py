# parses predicted intervals out of the raw responses and averages them across prompt wordings.
# uv run -m scripts.parse_predictions --dataset_path results/raw/260804_predictions_v1/responses.jsonl --output_dir results/raw/260804_predictions_v1

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import simple_parsing

from src.metrics.extraction import extract
from src.metrics.io import write_json
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging

LOGGER = logging.getLogger(__name__)


@dataclass
class ParseConfig(Config):
    # a predictor parsing below this cannot be compared fairly against one that answers every time
    min_parse_rate: float = 0.9
    min_formulations: int = 2


def main():
    config = simple_parsing.parse(ParseConfig, add_config_path_arg=True)
    assert config.dataset_path, "--dataset_path is required, point it at a responses.jsonl"

    output_dir = Path(config.output_dir)
    setup_logging(output_dir)
    write_config_json(config, output_dir)

    _records, report = extract(config.dataset_path, output_dir, config.min_formulations)
    path = write_json(report, output_dir / "parse_report.json")

    print(f"Predictions saved to {output_dir / 'predictions.jsonl'}")
    print(f"Parse report saved to {path}")
    print(f"overall: {report['overall']['n_parsed']}/{report['overall']['n']} ({report['overall']['parse_rate']:.1%})")
    for predictor, stats in report["by_predictor"].items():
        flag = "" if stats["parse_rate"] >= config.min_parse_rate else "  <- below --min_parse_rate"
        print(f"{predictor}: {stats['parse_rate']:.1%}{flag}")
    print(
        "by metadata level: " + ", ".join(f"{k}={v['parse_rate']:.1%}" for k, v in report["by_metadata_level"].items())
    )
    source_config = Path(config.dataset_path).parent / "config.json"
    ground_truth_dir = (
        json.loads(source_config.read_text()).get("ground_truth_dir", "") if source_config.exists() else ""
    )
    print(
        f"Score with: uv run -m scripts.score --predictions {output_dir / 'predictions.jsonl'} "
        f"--ground_truth_dir {ground_truth_dir} --output_dir {output_dir}"
    )


main()
