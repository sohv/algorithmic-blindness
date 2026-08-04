import re
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent

# proper nouns and identifiers that legitimately start a comment with a capital
PROPER_NOUNS = {
    "Anthropic",
    "Bellman",
    "CLRS",
    "CPDAG",
    "Dijkstra",
    "Efron",
    "Erdos",
    "FCI",
    "Fisher",
    "Gaussian",
    "Holm",
    "Hugging",
    "Kruskal",
    "LLM",
    "LiNGAM",
    "Lomuto",
    "McNemar",
    "NOTEARS",
    "OpenRouter",
    "PAG",
    "PC",
    "Prim",
    "SHD",
    "Spearman",
    "Wilson",
    "Winkler",
}

SEPARATOR = re.compile(r"([-=*#_~])\1{3,}")


def source_files() -> list[Path]:
    return sorted(
        path
        for directory in ("src", "scripts", "tests")
        for path in (REPO_ROOT / directory).rglob("*.py")
        if "__pycache__" not in str(path)
    )


def comment_runs(lines: list[str]) -> list[tuple[int, list[str]]]:
    """Runs of consecutive standalone comment lines, as (first line number, lines)."""
    runs, current, start = [], [], 0
    for index, line in enumerate(lines):
        if line.strip().startswith("#"):
            if not current:
                start = index
            current.append(line)
        elif current:
            runs.append((start, current))
            current = []
    if current:
        runs.append((start, current))
    return runs


def body_of(comment: str) -> str:
    return comment.strip().lstrip("#").strip()


def relative(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def test_file_headers_are_one_line_plus_an_optional_run_command():
    # a script gets one line plus its run command; a src/ module gets the one line only
    offenders = []
    for path in source_files():
        runs = comment_runs(path.read_text().splitlines())
        if not runs or runs[0][0] != 0:
            continue
        header = runs[0][1]
        limit = 2 if path.parts[-2] == "scripts" else 1
        if len(header) > limit:
            offenders.append(f"{relative(path)}: {len(header)}-line header, max {limit}")
    assert not offenders, "file headers over budget:\n" + "\n".join(offenders)


def test_inline_comments_are_one_line_each():
    # "Keep inline comments short, one line max"
    offenders = []
    for path in source_files():
        lines = path.read_text().splitlines()
        for start, run in comment_runs(lines):
            if start == 0:
                continue
            if len(run) > 1:
                offenders.append(f"{relative(path)}:{start + 1}: {len(run)}-line comment block")
    assert not offenders, "multi-line inline comments:\n" + "\n".join(offenders)


def test_no_decorative_separators():
    # "No lines of #, *, =, - or any other repeated character."
    offenders = []
    for path in source_files():
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            if line.strip().startswith("#") and SEPARATOR.search(line):
                offenders.append(f"{relative(path)}:{number}: {line.strip()[:60]}")
    assert not offenders, "decorative separators:\n" + "\n".join(offenders)


def test_comments_start_with_a_lowercase_letter():
    offenders = []
    for path in source_files():
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip().startswith("#"):
                continue
            body = body_of(line)
            if not body:
                continue
            words = [re.match(r"[A-Za-z]*", word).group() for word in body.split()]
            first = next((word for word in words if word), "")
            if not first or first.isupper() or first in PROPER_NOUNS:
                continue
            if first[0].isupper():
                offenders.append(f"{relative(path)}:{number}: {body[:60]}")
    assert not offenders, "comments starting with a capital:\n" + "\n".join(offenders)


def test_every_script_header_run_command_uses_flags_that_exist():
    """A header run command is documentation a reader will paste, so a stale flag is a broken doc."""
    base_fields = set(re.findall(r"^\s{4}([a-z_]+)\s*:", (REPO_ROOT / "src/utils/config.py").read_text(), re.MULTILINE))
    offenders = []
    for path in sorted((REPO_ROOT / "scripts").glob("*.py")):
        text = path.read_text()
        header = [line for line in text.splitlines()[:2] if line.startswith("#")]
        command = next((line for line in header if "uv run" in line), None)
        if command is None:
            continue
        declared = set(re.findall(r"^\s{4}([a-z_]+)\s*:", text[text.find("class ") :], re.MULTILINE))
        unknown = set(re.findall(r"--([a-z_]+)", command)) - declared - base_fields
        if unknown:
            offenders.append(f"{relative(path)}: {sorted(unknown)} not in its config")
    assert not offenders, "stale flags in header run commands:\n" + "\n".join(offenders)


def test_imports_are_not_commented():
    # "Do not comment imports"
    offenders = []
    for path in source_files():
        lines = path.read_text().splitlines()
        runs = comment_runs(lines)
        header_end = len(runs[0][1]) if runs and runs[0][0] == 0 else 0
        for number, line in enumerate(lines):
            if number < header_end or not line.strip().startswith("#"):
                continue
            following = next((lines[i] for i in range(number + 1, len(lines)) if lines[i].strip()), "")
            if following.startswith(("import ", "from ")):
                offenders.append(f"{relative(path)}:{number + 1}: {body_of(line)[:60]}")
    assert not offenders, "comments on imports:\n" + "\n".join(offenders)
