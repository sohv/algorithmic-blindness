# prints the rendered prompt for a condition at each metadata level, so wording can be checked before spending calls.
# uv run -m scripts.show_prompts --domain sorting --instance array_10k --variant reverse --algorithm quicksort --metadata_level full

from dataclasses import dataclass

import simple_parsing

from src.conditions import Condition, PromptSpec
from src.data.diagnostics import diagnostics_for
from src.generation.prompts import build_prompt


@dataclass
class ShowPromptsConfig:
    domain: str = "causal_discovery"
    instance: str = "asia"
    variant: str = "native"
    algorithm: str = "lingam"
    naming: str = "real"
    metadata_level: str = "sparse"
    formulation: int = 1
    data_dir: str = "data/raw"
    seed: int = 42


def main():
    config = simple_parsing.parse(ShowPromptsConfig)
    condition = Condition(config.domain, config.instance, config.variant, config.algorithm)
    spec = PromptSpec(config.naming, config.metadata_level, config.formulation)
    # computed for real rather than stubbed, so what is previewed is what a run would send
    diagnostics = None if spec.metadata_level == "sparse" else diagnostics_for(condition, config.data_dir, config.seed)
    print(build_prompt(condition, spec, diagnostics))


main()
