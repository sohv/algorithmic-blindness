# one place for the figure style the paper uses.

import logging
from pathlib import Path

import matplotlib.pyplot as plt

LOGGER = logging.getLogger(__name__)

# okabe-ito, colourblind-safe
PALETTE = ["#E69F00", "#56B4E9", "#009E73", "#D55E00", "#0072B2", "#CC79A7", "#F0E442"]

# acl one-column format
RCPARAMS = {
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "lines.linewidth": 1.5,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Liberation Sans", "Helvetica", "Arial"],
    "text.antialiased": True,
    "figure.dpi": 100,
    "text.color": "#000000",
    "axes.labelcolor": "#000000",
    "xtick.color": "#000000",
    "ytick.color": "#000000",
    "axes.edgecolor": "#1a1a1a",
    "grid.color": "#4a4a4a",
}


def apply_style() -> None:
    plt.rcParams.update(RCPARAMS)


def save_figure(fig, output_dir: Path | str, name: str) -> Path:
    """Write pdf for the paper and png for previewing, both at 300dpi. Returns the pdf path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{name}.pdf"
    for path in [pdf_path, output_dir / f"{name}.png"]:
        fig.savefig(str(path), dpi=300, bbox_inches="tight", pad_inches=0.08)
    LOGGER.info(f"saved {name}.pdf and {name}.png to {output_dir}")
    return pdf_path
