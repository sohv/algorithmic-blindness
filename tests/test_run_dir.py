import json
import re
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.domains import ALL_DOMAINS
from src.generation.models import ALL_MODELS
from src.utils.run_dir import normalize, resolve_run_dir, run_tag

REPO_ROOT = Path(__file__).parent.parent


def test_a_rerun_never_lands_on_a_directory_that_already_holds_a_run(tmp_path):
    first = resolve_run_dir(tmp_path / "260804_ground_truth")
    first.mkdir(parents=True)
    (first / "ground_truth.jsonl").write_text("{}\n")

    second = resolve_run_dir(tmp_path / "260804_ground_truth")
    assert first.name == "260804_ground_truth_v1"
    assert second.name == "260804_ground_truth_v2"
    assert second != first


def test_an_empty_directory_is_reused_rather_than_bumped(tmp_path):
    first = resolve_run_dir(tmp_path / "260804_predictions")
    first.mkdir(parents=True)
    assert resolve_run_dir(tmp_path / "260804_predictions") == first


def test_an_explicit_version_is_the_starting_point_not_a_ceiling(tmp_path):
    (tmp_path / "260804_scored_v3").mkdir(parents=True)
    (tmp_path / "260804_scored_v3" / "scored.jsonl").write_text("{}\n")
    assert resolve_run_dir(tmp_path / "260804_scored_v3").name == "260804_scored_v4"


def test_the_date_token_expands_to_today(tmp_path):
    resolved = resolve_run_dir(tmp_path / "{date}_ground_truth")
    today = datetime.now(timezone.utc).strftime("%y%m%d")
    assert resolved.name == f"{today}_ground_truth_v1"


def test_the_tag_names_only_the_axes_that_were_narrowed():
    assert run_tag((list(ALL_DOMAINS), ALL_DOMAINS), (list(ALL_MODELS), ALL_MODELS)) == ""
    assert run_tag((["sorting"], ALL_DOMAINS), (["gpt", "grok"], ALL_MODELS)) == "sorting_gpt_grok"
    assert run_tag(([], ALL_DOMAINS)) == ""


def test_a_narrowed_run_is_named_for_what_it_covered(tmp_path):
    tag = run_tag((["sorting"], ALL_DOMAINS), (["gpt"], ALL_MODELS))
    assert resolve_run_dir(tmp_path / "260804_predictions", tag).name == "260804_predictions_sorting_gpt_v1"


def test_model_ids_are_filename_safe():
    assert normalize("anthropic/claude-opus-5") == "anthropic_claude-opus-5"
    assert normalize(" Gemini 3 Pro ") == "gemini_3_pro"


def test_configs_date_their_output_dir_rather_than_hardcoding_one():
    """A hardcoded date silently reuses an old run's directory on the next day's run."""
    offenders = []
    for path in sorted((REPO_ROOT / "configs").glob("*.yaml")):
        output_dir = yaml.safe_load(path.read_text()).get("output_dir", "")
        if output_dir and "{date}" not in output_dir:
            offenders.append(f"{path.name}: {output_dir}")
    assert not offenders, "configs with a hardcoded date:\n" + "\n".join(offenders)


def test_configs_never_default_an_input_path():
    # "Never use default paths for input data. All input paths must be explicit CLI args."
    offenders = []
    for path in sorted((REPO_ROOT / "configs").glob("*.yaml")):
        config = yaml.safe_load(path.read_text())
        for field in ("ground_truth_dir", "dataset_path", "predictions", "scored", "human_predictions"):
            if config.get(field):
                offenders.append(f"{path.name}: {field}={config[field]}")
    assert not offenders, "input paths defaulted in a config:\n" + "\n".join(offenders)


def test_the_results_section_records_every_axis_the_grid_reports():
    """The scope line is the only record of what a run covered, so a renamed key silently empties it."""
    from src.conditions import Grid

    described = Grid(models=["gpt"]).describe()
    report_source = (REPO_ROOT / "src/metrics/report.py").read_text()
    axes = re.search(r"for axis in \(([^)]*)\)", report_source).group(1)
    for axis in re.findall(r'"([a-z_]+)"', axes):
        assert axis in described, f"report.py reads grid key {axis!r}, which Grid.describe() does not write"


def test_every_documented_model_key_exists():
    """A run command naming a model that was renamed fails at the first paid call."""
    offenders = []
    sources = list((REPO_ROOT / "scripts").glob("*.py")) + list((REPO_ROOT / "configs").glob("*.yaml"))
    for path in sources:
        for line in path.read_text().splitlines():
            if "--models" not in line:
                continue
            keys = re.search(r"--models ([a-z0-9 ]+)", line)
            unknown = set(keys.group(1).split()) - set(ALL_MODELS) - {"all"} if keys else set()
            if unknown:
                offenders.append(f"{path.name}: {sorted(unknown)}")
    for path in (REPO_ROOT / "configs").glob("*.yaml"):
        declared = yaml.safe_load(path.read_text()).get("models", [])
        unknown = set(declared) - set(ALL_MODELS) - {"all"}
        if unknown:
            offenders.append(f"{path.name}: {sorted(unknown)}")
    assert not offenders, "unknown model keys:\n" + "\n".join(offenders)


def test_analyze_finds_the_summary_beside_the_scored_file(tmp_path):
    """analyze.py derives score_summary.json from the scored path, so the two must stay siblings."""
    score_source = (REPO_ROOT / "scripts/score.py").read_text()
    analyze_source = (REPO_ROOT / "scripts/analyze.py").read_text()
    assert 'output_dir / "score_summary.json"' in score_source
    assert 'Path(config.scored).parent / "score_summary.json"' in analyze_source

    scored_dir = tmp_path / "260804_scored_v1"
    scored_dir.mkdir()
    (scored_dir / "score_summary.json").write_text(json.dumps({"n": 1}))
    assert (Path(scored_dir / "scored.jsonl").parent / "score_summary.json").exists()
