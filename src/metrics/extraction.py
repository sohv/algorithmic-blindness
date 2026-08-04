# the only parser for predicted ranges in raw llm responses.

import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from src.data.io import load_jsonl
from src.domains import ALL_DOMAINS, metrics_of
from src.generation.models import MODELS_BY_KEY
from src.metrics.io import write_jsonl

LOGGER = logging.getLogger(__name__)

# every metric across every domain, for a caller that parses without knowing the domain
ALL_METRICS = sorted({metric for name in ALL_DOMAINS for metric in metrics_of(name)})

# a metric label as it may appear in a response; the line-start anchor keeps one from matching inside another
METRIC_ALIASES = {
    "precision": r"precision",
    "recall": r"recall",
    "f1": r"f1(?:[\s\-_]?score)?",
    "shd": r"shd|structural\s+hamming(?:\s+distance)?",
    "comparisons": r"comparisons?|n[\s_]?comparisons",
    "moves": r"moves?|writes?|element[\s_]?moves?",
    "max_depth": r"max(?:imum)?[\s_-]?(?:recursion[\s_-]?)?depth",
    "relaxations": r"relaxations?|edge[\s_-]?relaxations?",
    "successful_relaxations": r"successful[\s_-]?relaxations?",
    "nodes_settled": r"nodes[\s_-]?settled|settled[\s_-]?nodes",
}

NUMBER = r"[-+]?\d*\.?\d+"

# bracketed pair, then bare pair separated by a dash, "to", or a comma
PATTERNS = [
    r"\[\s*({n})\s*(?:,|-|–|—|\bto\b)\s*({n})\s*\]",
    r"\(\s*({n})\s*(?:,|-|–|—|\bto\b)\s*({n})\s*\)",
    r"({n})\s*(?:–|—|\bto\b)\s*({n})",
    r"({n})\s*-\s*({n})",
    r"({n})\s*,\s*({n})",
]

CONFIDENCE_PATTERN = re.compile(r"confidence\s*[:=]?\s*(high|medium|moderate|low)", re.IGNORECASE)

# a metric line may be prefixed with a bullet, bold markers or numbering before the label
LINE_PREFIX = r"^[\s>*\-•\d.)]*\**\s*"


@dataclass
class MetricEstimate:
    lower: float
    upper: float

    def midpoint(self) -> float:
        return (self.lower + self.upper) / 2

    def width(self) -> float:
        return self.upper - self.lower


def metric_pattern(metric: str) -> re.Pattern:
    alias = METRIC_ALIASES[metric]
    body = "|".join(p.format(n=NUMBER) for p in PATTERNS)
    return re.compile(
        rf"{LINE_PREFIX}\**\s*(?:{alias})\**\s*(?:range)?\s*[:=]\s*\**\s*(?:{body})", re.IGNORECASE | re.MULTILINE
    )


PATTERNS_BY_METRIC = {metric: metric_pattern(metric) for metric in METRIC_ALIASES}


# metrics that are proportions; everything else is an unbounded non-negative count
BOUNDED_METRICS = ("precision", "recall", "f1")


def is_valid(metric: str, lower: float, upper: float) -> bool:
    if metric in BOUNDED_METRICS:
        return 0.0 <= lower <= 1.0 and 0.0 <= upper <= 1.0
    return lower >= 0 and upper >= 0


def parse_metric(text: str, metric: str) -> MetricEstimate | None:
    """Last valid match wins: a reasoning model states its answer after its working."""
    matches = []
    for match in PATTERNS_BY_METRIC[metric].finditer(text):
        pair = [g for g in match.groups() if g is not None]
        if len(pair) != 2:
            continue
        lower, upper = sorted(float(value) for value in pair)
        if is_valid(metric, lower, upper):
            matches.append(MetricEstimate(lower, upper))

    if not matches:
        return None
    if len({(m.lower, m.upper) for m in matches}) > 1:
        LOGGER.warning(f"{metric}: {len(matches)} differing matches in one response, taking the last")
    return matches[-1]


def parse_ranges(text: str, metrics: tuple[str, ...] | list[str] | None = None) -> dict[str, MetricEstimate]:
    """Every requested metric that parses. A missing metric is absent rather than guessed."""
    if not text:
        return {}
    estimates = {}
    for metric in metrics or ALL_METRICS:
        estimate = parse_metric(text, metric)
        if estimate is not None:
            estimates[metric] = estimate
    return estimates


def parse_confidence(text: str) -> str:
    """The prompts do not ask for a confidence, so medium unless the model volunteers one."""
    match = CONFIDENCE_PATTERN.search(text or "")
    if not match:
        return "medium"
    stated = match.group(1).lower()
    return "medium" if stated == "moderate" else stated


def extract_records(responses_path: Path | str) -> list[dict]:
    """One prediction row per (response, metric), in the schema every predictor shares.

    A response that parses no metric still produces rows with a null interval, so a model that
    ignores the format shows up as a parse failure rather than as a smaller experiment.
    """
    records = []
    for response in load_jsonl(responses_path):
        text = response.get("response")
        metrics = metrics_of(response["domain"])
        estimates = parse_ranges(text, metrics)
        confidence = parse_confidence(text)
        for metric in metrics:
            estimate = estimates.get(metric)
            records.append(
                {
                    "id": f"{response['id']}__{metric}",
                    "predictor": response["model"],
                    "predictor_kind": "llm",
                    "domain": response["domain"],
                    "tier": response.get("tier")
                    or (MODELS_BY_KEY[response["model"]].tier if response["model"] in MODELS_BY_KEY else "unknown"),
                    "instance": response["instance"],
                    "variant": response["variant"],
                    "algorithm": response["algorithm"],
                    "metric": metric,
                    "naming": response["naming"],
                    "metadata_level": response["metadata_level"],
                    "formulation": response["formulation"],
                    "lower": estimate.lower if estimate else None,
                    "upper": estimate.upper if estimate else None,
                    "confidence": confidence if estimate else None,
                    "parsed": estimate is not None,
                }
            )
    return records


AGGREGATION_KEYS = [
    "predictor",
    "predictor_kind",
    "tier",
    "domain",
    "instance",
    "variant",
    "algorithm",
    "metric",
    "naming",
    "metadata_level",
]


def aggregate_formulations(records: list[dict], min_formulations: int = 2) -> list[dict]:
    """Average each predictor's intervals across prompt wordings into one row per condition.

    Emitted as formulation 0, alongside the per-formulation rows rather than replacing them, so the
    headline number and the wording-sensitivity analysis read from the same file. Requiring two
    wordings stops a single parse from being reported as if it were robust to phrasing.
    """
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for record in records:
        if record["parsed"] and record["formulation"] > 0:
            grouped[tuple(record[key] for key in AGGREGATION_KEYS)].append(record)

    aggregated = []
    for key, rows in sorted(grouped.items()):
        if len(rows) < min_formulations:
            continue
        fields = dict(zip(AGGREGATION_KEYS, key))
        confidences = [r["confidence"] for r in rows]
        aggregated.append(
            {
                "id": "__".join(str(v) for v in key) + "__aggregated",
                **fields,
                "formulation": 0,
                "lower": sum(r["lower"] for r in rows) / len(rows),
                "upper": sum(r["upper"] for r in rows) / len(rows),
                # the least confident wording sets the aggregate's confidence
                "confidence": "low" if "low" in confidences else "medium" if "medium" in confidences else "high",
                "parsed": True,
                "n_formulations": len(rows),
            }
        )
    LOGGER.info(f"aggregated {len(aggregated)} predictions from {len(records)} per-formulation rows")
    return aggregated


def extraction_report(records: list[dict]) -> dict:
    """Parse rate overall, per predictor, per metric and per metadata level.

    The per-level breakdown matters for the metadata experiment: if a richer prompt merely made
    models more likely to answer in prose, a coverage change would be a parsing artefact.
    """

    def rates(key: str) -> dict:
        buckets: dict[str, list[bool]] = defaultdict(list)
        for record in records:
            buckets[str(record[key])].append(record["parsed"])
        return {name: rate(flags) for name, flags in sorted(buckets.items())}

    def rate(flags: list[bool]) -> dict:
        return {"n": len(flags), "n_parsed": sum(flags), "parse_rate": sum(flags) / len(flags) if flags else 0.0}

    return {
        "overall": rate([r["parsed"] for r in records]),
        "by_predictor": rates("predictor"),
        "by_tier": rates("tier"),
        "by_metric": rates("metric"),
        "by_domain": rates("domain"),
        "by_metadata_level": rates("metadata_level"),
        "by_naming": rates("naming"),
        "by_formulation": rates("formulation"),
    }


def extract(responses_path: Path | str, output_dir: Path | str, min_formulations: int = 2) -> tuple[list[dict], dict]:
    """Parse a responses.jsonl into prediction rows plus a parse report."""
    records = extract_records(responses_path)
    records += aggregate_formulations(records, min_formulations)
    write_jsonl(records, Path(output_dir) / "predictions.jsonl")
    report = extraction_report(records)
    LOGGER.info(f"parsed {report['overall']['n_parsed']}/{report['overall']['n']} prediction rows")
    return records, report
