# draws every figure the scored table supports. a figure needing an axis this run did not vary is skipped, loudly.
# uv run -m scripts.plot --scored results/raw/260804_scored_v1/scored.jsonl --output_dir results/raw/260804_scored_v1

import logging
from dataclasses import dataclass
from pathlib import Path

import simple_parsing

from src.data.io import load_jsonl
from src.metrics.analysis import to_frame
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging
from src.visualize.figures import FIGURES, build_figures

LOGGER = logging.getLogger(__name__)


@dataclass
class PlotConfig(Config):
    scored: str = ""


def main():
    config = simple_parsing.parse(PlotConfig, add_config_path_arg=True)
    assert config.scored, "--scored is required, point it at a scored.jsonl"

    output_dir = Path(config.output_dir)
    setup_logging(output_dir)
    write_config_json(config, output_dir)

    frame = to_frame(load_jsonl(config.scored))
    written = build_figures(frame, output_dir / "figures")

    print(f"Wrote {len(written)}/{len(FIGURES)} figures to {output_dir / 'figures'}")
    for path in written:
        print(path)


main()
