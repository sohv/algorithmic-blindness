# the paper's figures, all from the scored table.

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.metrics.analysis import coverage_table, paired_on, scorable
from src.visualize.style import PALETTE, apply_style, save_figure

LOGGER = logging.getLogger(__name__)


def coverage_by_predictor(frame: pd.DataFrame, output_dir: Path) -> Path:
    """Coverage per predictor with Wilson intervals, baselines marked.

    The error bars are the point: without them a 3-point gap between two models reads as a ranking.
    """
    table = coverage_table(frame[frame["formulation"] == 0], "predictor")
    if table.empty:
        return None

    colours = [
        PALETTE[2] if kind else PALETTE[1]
        for kind in table["predictor"].str.startswith(("uniform", "heuristic", "marginal", "oracle"))
    ]
    # clipped at zero: a Wilson bound at a rate of exactly 1.0 lands a float epsilon the wrong side
    errors = np.vstack(
        [
            (table["coverage"] - table["coverage_ci_lower"]).clip(lower=0),
            (table["coverage_ci_upper"] - table["coverage"]).clip(lower=0),
        ]
    )

    fig, ax = plt.subplots(figsize=(5.5, max(2.5, 0.32 * len(table))))
    ax.barh(
        table["predictor"], table["coverage"], xerr=errors, color=colours, edgecolor="#333333", capsize=3, alpha=0.9
    )
    ax.invert_yaxis()
    ax.set_xlabel("Calibrated coverage")
    ax.set_title("Coverage by predictor, with 95% Wilson intervals")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    path = save_figure(fig, output_dir, "coverage_by_predictor")
    plt.close(fig)
    return path


def metadata_effect(frame: pd.DataFrame, output_dir: Path) -> Path:
    """Coverage against how much the model was told. A flat line is the finding.

    If coverage does not rise from sparse to full, the failure cannot be blamed on the prompt
    withholding what was needed - which is the criticism this experiment exists to answer.
    """
    llm = frame[(frame["predictor_kind"] == "llm") & (frame["metadata_level"] != "na")]
    if llm.empty:
        return None

    table = coverage_table(llm, ["predictor", "metadata_level"])
    order = [level for level in ["sparse", "diagnostic", "full"] if level in set(table["metadata_level"])]
    if len(order) < 2:
        return None

    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for index, (predictor, group) in enumerate(table.groupby("predictor")):
        group = group.set_index("metadata_level").reindex(order)
        ax.plot(order, group["coverage"], "o-", color=PALETTE[index % len(PALETTE)], label=predictor)

    overall = coverage_table(llm, "metadata_level").set_index("metadata_level").reindex(order)
    ax.plot(order, overall["coverage"], "s--", color="#333333", linewidth=2.5, label="mean", zorder=5)

    ax.set_xlabel("Information given to the model")
    ax.set_ylabel("Calibrated coverage")
    ax.set_title("Does more information fix it?")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    path = save_figure(fig, output_dir, "metadata_effect")
    plt.close(fig)
    return path


def naming_effect(frame: pd.DataFrame, output_dir: Path) -> Path:
    """Benchmark coverage under the real name versus a neutral id, per model.

    Points below the diagonal are models that did better when they could recognise the dataset.
    """
    benchmarks = frame[(~frame["is_synthetic"]) & (frame["predictor_kind"] == "llm")]
    paired = paired_on(benchmarks, "naming", "real", "anonymized", keys=["predictor"])
    if paired.empty:
        return None

    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    ax.plot([0, 1], [0, 1], "--", color="#888888", linewidth=1)
    for index, row in paired.reset_index(drop=True).iterrows():
        ax.scatter(
            row["covers_mean_right"],
            row["covers_mean_left"],
            color=PALETTE[index % len(PALETTE)],
            s=45,
            edgecolor="#333333",
            zorder=3,
        )
        ax.annotate(
            row["predictor"],
            (row["covers_mean_right"], row["covers_mean_left"]),
            fontsize=7,
            xytext=(4, 3),
            textcoords="offset points",
        )

    ax.set_xlabel("Coverage, anonymized name")
    ax.set_ylabel("Coverage, real name")
    ax.set_title("Does the name carry the performance?")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    path = save_figure(fig, output_dir, "naming_effect")
    plt.close(fig)
    return path


def noise_effect(frame: pd.DataFrame, output_dir: Path) -> Path:
    """Coverage per algorithm under Gaussian versus non-Gaussian synthetic noise.

    LiNGAM is the one to read: under Gaussian noise its identifying assumption is violated, so its
    performance there says nothing about whether models understand it.
    """
    synthetic = scorable(frame[frame["is_synthetic"] & (frame["predictor_kind"] == "llm")])
    if synthetic.empty or synthetic["variant"].nunique() < 2:
        return None

    synthetic = synthetic.assign(noise_kind=np.where(synthetic["variant"] == "gaussian", "Gaussian", "non-Gaussian"))
    table = synthetic.groupby(["algorithm", "noise_kind"])["covers_mean"].mean().unstack()
    if table.shape[1] < 2:
        return None

    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    x = np.arange(len(table))
    width = 0.35
    for index, column in enumerate(table.columns):
        ax.bar(
            x + index * width - width / 2,
            table[column],
            width,
            label=column,
            color=PALETTE[index],
            edgecolor="#333333",
            alpha=0.9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(table.index)
    ax.set_ylabel("Calibrated coverage")
    ax.set_title("Synthetic coverage by noise family")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = save_figure(fig, output_dir, "noise_effect")
    plt.close(fig)
    return path


def width_versus_coverage(frame: pd.DataFrame, output_dir: Path) -> Path:
    """Predicted width against coverage. Wide and still missing is the paper's core observation,
    and this is the plot where a predictor that widened its way to coverage would be visible."""
    table = coverage_table(frame[frame["formulation"] == 0], "predictor")
    table = table[np.isfinite(table["mean_width_ratio"])]
    if table.empty:
        return None

    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    for index, row in table.reset_index(drop=True).iterrows():
        ax.scatter(
            row["mean_width_ratio"],
            row["coverage"],
            s=50,
            color=PALETTE[index % len(PALETTE)],
            edgecolor="#333333",
            zorder=3,
        )
        ax.annotate(
            row["predictor"],
            (row["mean_width_ratio"], row["coverage"]),
            fontsize=7,
            xytext=(4, 3),
            textcoords="offset points",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Predicted width / true CI width")
    ax.set_ylabel("Calibrated coverage")
    ax.set_title("Wider intervals did not buy coverage")
    ax.axvline(1.0, linestyle="--", color="#888888", linewidth=1)
    fig.tight_layout()
    path = save_figure(fig, output_dir, "width_versus_coverage")
    plt.close(fig)
    return path


def bias_direction(frame: pd.DataFrame, output_dir: Path) -> Path:
    """Signed bias per predictor. Positive is optimistic on every metric, including SHD."""
    usable = scorable(frame[frame["formulation"] == 0])
    if usable.empty:
        return None

    order = usable.groupby("predictor")["bias_normalised"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(5.0, max(2.4, 0.3 * len(order))))
    ax.barh(
        order.index,
        order.to_numpy(),
        color=[PALETTE[i % len(PALETTE)] for i in range(len(order))],
        edgecolor="#333333",
        alpha=0.9,
    )
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_xlabel("Mean bias (positive = optimistic)")
    ax.set_title("Direction of error by predictor")
    fig.tight_layout()
    path = save_figure(fig, output_dir, "bias_direction")
    plt.close(fig)
    return path


FIGURES = [
    coverage_by_predictor,
    metadata_effect,
    naming_effect,
    noise_effect,
    width_versus_coverage,
    bias_direction,
]


def build_figures(frame: pd.DataFrame, output_dir: Path | str) -> list[Path]:
    """Every figure the data supports. One that needs an axis this run did not vary is skipped."""
    apply_style()
    output_dir = Path(output_dir)
    written = []
    for figure in FIGURES:
        path = figure(frame, output_dir)
        if path is None:
            LOGGER.warning(f"{figure.__name__}: not enough data in this run, skipped")
        else:
            written.append(path)
    return written
