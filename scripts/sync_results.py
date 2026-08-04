# mirrors results/ to the hugging face dataset repo, the citable copy of a run.
# uv run -m scripts.sync_results --repo algorithmic-blindness-results --private true

import logging
from dataclasses import dataclass

import simple_parsing
from dotenv import load_dotenv

from src.data.hf import DEFAULT_REPO, sync_results

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


@dataclass
class SyncConfig:
    results_dir: str = "results"
    repo: str = DEFAULT_REPO
    # taken from the token's account when left blank
    owner: str = ""
    # a published run is skipped by default; set this only to deliberately replace one
    overwrite: bool = False
    private: bool = True
    dry_run: bool = False


def main():
    load_dotenv()
    config = simple_parsing.parse(SyncConfig)

    plan = sync_results(
        results_dir=config.results_dir,
        repo=config.repo,
        owner=config.owner or None,
        overwrite=config.overwrite,
        private=config.private,
        dry_run=config.dry_run,
    )

    print(f"repo: {plan['repo_id']}")
    print(f"runs found: {plan['n_runs_found']}, to upload: {plan['n_to_upload']}")
    for name in plan["uploading"]:
        print(f"  upload {name}")
    for name in plan["skipped_already_published"]:
        print(f"  skip   {name} (already published; pass --overwrite true to replace)")
    if plan["dry_run"]:
        print("dry run, nothing uploaded")
    else:
        print(plan["url"])


main()
