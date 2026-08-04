# resolves a run's output directory so a rerun is never written on top of its predecessor.

import logging
import re
from datetime import datetime, timezone
from pathlib import Path

LOGGER = logging.getLogger(__name__)

DATE_TOKEN = "{date}"
VERSION_SUFFIX = re.compile(r"_v(\d+)$")


def normalize(name: str) -> str:
    """Filename-safe form of a model id or instance name: `/` and whitespace become underscores."""
    return re.sub(r"[\s/]+", "_", str(name).strip().lower())


def run_tag(*selections: tuple[list[str], tuple[str, ...] | list[str]]) -> str:
    """Names the narrowed axes of a run, so the directory says what it covered.

    An axis that ran in full contributes nothing: `predictions_v1` is the whole grid, and
    `predictions_sorting_opus_v1` is the run that was restricted to one domain and one model.
    """
    parts = []
    for selected, available in selections:
        if selected and set(selected) < set(available):
            parts.extend(normalize(item) for item in selected)
    return "_".join(parts)


def resolve_run_dir(output_dir: str | Path, tag: str = "") -> Path:
    """A run directory that does not yet hold a run: date-stamped, tagged, and version-bumped.

    `results/raw/` is append-only, so a second run under the same name takes the next `_vN` rather
    than overwriting the first. Reruns are the common case here, and a silently merged directory
    would leave two runs' artifacts indistinguishable.
    """
    stamped = str(output_dir).replace(DATE_TOKEN, datetime.now(timezone.utc).strftime("%y%m%d"))
    path = Path(stamped)

    match = VERSION_SUFFIX.search(path.name)
    version = int(match.group(1)) if match else 1
    stem = VERSION_SUFFIX.sub("", path.name)
    if tag:
        stem = f"{stem}_{normalize(tag)}"

    candidate = path.parent / f"{stem}_v{version}"
    while candidate.exists() and any(candidate.iterdir()):
        version += 1
        candidate = path.parent / f"{stem}_v{version}"

    if candidate.name != path.name:
        LOGGER.info(f"run directory resolved to {candidate}")
    return candidate
