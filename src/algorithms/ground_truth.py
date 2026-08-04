# turns per-run records into the one interval per condition and metric that predictions are scored against.

import logging

import numpy as np

from src.domains import CAUSAL, metrics_of

LOGGER = logging.getLogger(__name__)

# below this the interval is a point and any width ratio against it is meaningless
DEGENERATE_CI_WIDTH = 1e-9


def summarize_metric(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "ci_lower": float(np.percentile(array, 2.5)),
        "ci_upper": float(np.percentile(array, 97.5)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def build_ground_truth(runs: list[dict]) -> list[dict]:
    """One row per condition and metric. Degenerate intervals are flagged, loudly.

    A zero-width interval means every run returned an identical result, so nothing about run-to-run
    variation was measured. In the causal domain that is a bug (bootstrap resampling is off); in the
    others it can be legitimate, which is why the report separates the two by domain.
    """
    grouped: dict[tuple[str, str, str, str], list[dict]] = {}
    for run in runs:
        key = (run["domain"], run["instance"], run["variant"], run["algorithm"])
        grouped.setdefault(key, []).append(run)

    rows, degenerate = [], []
    for (domain, instance, variant, algorithm), condition_runs in sorted(grouped.items()):
        succeeded = [r for r in condition_runs if not r.get("failed", False)]
        if not succeeded:
            LOGGER.error(f"{domain}/{instance}/{variant}/{algorithm}: every run failed, no ground truth produced")
            continue

        for metric in metrics_of(domain):
            summary = summarize_metric([r[metric] for r in succeeded])
            width = summary["ci_upper"] - summary["ci_lower"]
            if width < DEGENERATE_CI_WIDTH:
                degenerate.append(f"{domain}/{instance}/{variant}/{algorithm}/{metric}")
            rows.append(
                {
                    "id": f"{domain}__{instance}__{variant}__{algorithm}__{metric}",
                    "domain": domain,
                    "instance": instance,
                    "variant": variant,
                    "algorithm": algorithm,
                    "metric": metric,
                    **summary,
                    "ci_width": width,
                    "n_runs": len(condition_runs),
                    "n_failed": len(condition_runs) - len(succeeded),
                }
            )

    # in causal discovery a zero width means resampling is off; elsewhere it can be the true answer
    causal_degenerate = [d for d in degenerate if d.startswith(CAUSAL)]
    if causal_degenerate:
        LOGGER.error(
            f"{len(causal_degenerate)} causal_discovery intervals have zero width, e.g. "
            f"{causal_degenerate[:3]}. the algorithms are deterministic, so this means bootstrap "
            "resampling is off and no sampling variation was measured."
        )
    other_degenerate = [d for d in degenerate if not d.startswith(CAUSAL)]
    if other_degenerate:
        LOGGER.info(
            f"{len(other_degenerate)} intervals have zero width outside causal discovery, e.g. "
            f"{other_degenerate[:3]}. this is expected where the count does not depend on the draw."
        )
    LOGGER.info(f"built {len(rows)} ground-truth rows from {len(runs)} runs")
    return rows


def ground_truth_report(ground_truth: list[dict]) -> dict:
    """Health of the ground truth, so a degenerate run is caught before anything is scored."""
    widths = np.array([row["ci_width"] for row in ground_truth], dtype=float)
    n_degenerate = int((widths < DEGENERATE_CI_WIDTH).sum())

    by_domain = {}
    for domain in sorted({row["domain"] for row in ground_truth}):
        domain_rows = [r for r in ground_truth if r["domain"] == domain]
        domain_widths = [r["ci_width"] for r in domain_rows]
        by_domain[domain] = {
            "n_rows": len(domain_rows),
            "n_conditions": len({(r["instance"], r["variant"], r["algorithm"]) for r in domain_rows}),
            "mean_ci_width": float(np.mean(domain_widths)),
            "n_degenerate": int(sum(w < DEGENERATE_CI_WIDTH for w in domain_widths)),
            "by_metric": {
                metric: {
                    "n": len([r for r in domain_rows if r["metric"] == metric]),
                    "mean": float(np.mean([r["mean"] for r in domain_rows if r["metric"] == metric])),
                    "mean_ci_width": float(np.mean([r["ci_width"] for r in domain_rows if r["metric"] == metric])),
                }
                for metric in sorted({r["metric"] for r in domain_rows})
            },
        }

    return {
        "n_rows": len(ground_truth),
        "n_conditions": len({(r["domain"], r["instance"], r["variant"], r["algorithm"]) for r in ground_truth}),
        "n_degenerate_intervals": n_degenerate,
        # only the causal count is a defect signal; elsewhere a deterministic count is legitimate
        "n_degenerate_causal": sum(
            1 for r in ground_truth if r["domain"] == CAUSAL and r["ci_width"] < DEGENERATE_CI_WIDTH
        ),
        "share_degenerate": n_degenerate / len(widths) if widths.size else 0.0,
        "mean_ci_width": float(widths.mean()) if widths.size else 0.0,
        "by_domain": by_domain,
    }
