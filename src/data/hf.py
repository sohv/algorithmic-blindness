# mirrors results/ to a hugging face dataset repo.

import logging
import os
from pathlib import Path

LOGGER = logging.getLogger(__name__)

DEFAULT_REPO = "algorithmic-blindness-results"
REPO_TYPE = "dataset"


def hf_token() -> str:
    token = os.environ.get("HF_TOKEN")
    assert token, "HF_TOKEN is not set. copy .env.example to .env and fill it in"
    return token


def repo_id(repo: str, owner: str | None = None) -> str:
    """`owner/name`, taking the owner from the token's account when not given explicitly."""
    if "/" in repo:
        return repo
    assert owner, f"repo {repo!r} has no owner and none was resolved; pass --owner"
    return f"{owner}/{repo}"


def resolve_owner(token: str) -> str:
    from huggingface_hub import HfApi

    return HfApi(token=token).whoami()["name"]


def existing_run_dirs(api, target: str) -> set[str]:
    """Top-level run directories already published, so a re-sync can skip them."""
    from huggingface_hub.utils import RepositoryNotFoundError

    try:
        files = api.list_repo_files(repo_id=target, repo_type=REPO_TYPE)
    except RepositoryNotFoundError:
        return set()
    return {path.split("/")[1] for path in files if path.startswith("raw/") and "/" in path[4:]}


def sync_results(
    results_dir: Path | str = "results",
    repo: str = DEFAULT_REPO,
    owner: str | None = None,
    overwrite: bool = False,
    private: bool = True,
    dry_run: bool = False,
) -> dict:
    """Upload each run directory under results/raw/, plus RESULTS.md, to the dataset repo."""
    from huggingface_hub import HfApi

    results_dir = Path(results_dir)
    assert results_dir.exists(), f"{results_dir} does not exist"

    token = hf_token()
    api = HfApi(token=token)
    target = repo_id(repo, owner or resolve_owner(token))

    run_dirs = sorted(p for p in (results_dir / "raw").glob("*") if p.is_dir())
    published = set() if overwrite else existing_run_dirs(api, target)
    to_upload = [p for p in run_dirs if p.name not in published]
    skipped = [p.name for p in run_dirs if p.name in published]

    plan = {
        "repo_id": target,
        "n_runs_found": len(run_dirs),
        "n_to_upload": len(to_upload),
        "uploading": [p.name for p in to_upload],
        "skipped_already_published": skipped,
        "dry_run": dry_run,
    }
    if dry_run:
        return plan

    api.create_repo(repo_id=target, repo_type=REPO_TYPE, private=private, exist_ok=True)

    for run_dir in to_upload:
        LOGGER.info(f"uploading {run_dir} to {target}")
        api.upload_folder(
            repo_id=target,
            repo_type=REPO_TYPE,
            folder_path=str(run_dir),
            path_in_repo=f"raw/{run_dir.name}",
            commit_message=f"add run {run_dir.name}",
        )

    results_md = results_dir / "RESULTS.md"
    if results_md.exists():
        # always re-uploaded: it is the cumulative record, so the upstream copy must not go stale
        api.upload_file(
            repo_id=target,
            repo_type=REPO_TYPE,
            path_or_fileobj=str(results_md),
            path_in_repo="RESULTS.md",
            commit_message="update results table",
        )
        plan["results_md_uploaded"] = True

    plan["url"] = f"https://huggingface.co/datasets/{target}"
    return plan
