# validates the model registry against openrouter's live catalog and prints what each model costs.
# uv run -m scripts.check_models --probe true

import logging
from dataclasses import dataclass

import simple_parsing
from dotenv import load_dotenv

from src.generation.models import MIN_TIER_SEPARATION_USD_PER_MTOK, MODELS, check_against_catalog
from src.generation.openrouter import fetch_catalog, test_connection

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


@dataclass
class CheckConfig:
    # send one trivial billable call per model, to confirm the key works end to end
    probe: bool = False


def main():
    load_dotenv()
    config = simple_parsing.parse(CheckConfig)

    report = check_against_catalog(fetch_catalog())
    print(f"registry: {report['n_found']}/{report['n_models']} models found in the openrouter catalog")
    print(f"every frontier model must clear the mid-tier top by ${MIN_TIER_SEPARATION_USD_PER_MTOK:.0f}/Mtok output")
    print()
    print(f"{'key':<10} {'tier':<9} {'in $/Mtok':>10} {'out $/Mtok':>11} {'temp':>5} {'seed':>5}  openrouter id")
    for key, entry in report["by_model"].items():
        print(
            f"{key:<10} {entry['declared_tier']:<9} {entry['input_usd_per_mtok']:>10.2f} "
            f"{entry['output_usd_per_mtok']:>11.2f} {'yes' if entry['supports_temperature'] else 'NO':>5} "
            f"{'yes' if entry['supports_seed'] else 'no':>5}  {entry['openrouter_id']}"
        )

    for missing in report["missing_ids"]:
        print(f"MISSING: {missing} is not in the catalog. update src/generation/models.py")
    for mismatch in report["tier_mismatches"]:
        print(f"TIER MISMATCH: {mismatch}")
    for mismatch in report["control_mismatches"]:
        print(f"CONTROL MISMATCH: {mismatch}")

    if config.probe:
        print()
        for model in MODELS:
            print(f"{model.key}: {'ok' if test_connection(model.key) else 'FAILED'}")

    assert report["ok"], "registry does not match the live catalog, see the errors above"
    print()
    print("registry matches the catalog")


main()
