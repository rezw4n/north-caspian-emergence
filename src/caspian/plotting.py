"""Shared figure style: sans-serif 5-7 pt text, thin recessive axes, embedded
TrueType fonts, vector PDF plus 600 dpi TIFF output.

Default column widths, used by the QC figures; the ERL figures take their widths
from figures/_erl.py.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

MM = 1 / 25.4  # inches per millimetre
WIDTH_SINGLE_MM = 88.0     # Nature Portfolio (read 2026-10-21)
WIDTH_DOUBLE_MM = 180.0

INK = "#1f1f1f"       # primary text
INK_MUTED = "#6b6b6b"  # secondary text, axes, ticks
GRID = "#e3e3e3"       # recessive grid


def use_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": 6.5,
        "axes.titlesize": 7,
        "axes.labelsize": 6.5,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": INK_MUTED,
        "axes.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.color": GRID,
        "grid.linewidth": 0.4,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 0.9,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.dpi": 600,
        "figure.dpi": 150,
    })


def panel_label(ax, letter: str, x: float = -0.12, y: float = 1.04) -> None:
    """Lowercase bold panel letter in the top-left corner."""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=8, fontweight="bold",
            va="bottom", ha="left", color=INK)


def save(fig, stem: str | Path, tiff: bool = True) -> list[Path]:
    """Write ``stem``.pdf (vector) and, optionally, a 600 dpi LZW TIFF."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = [stem.with_suffix(".pdf")]
    fig.savefig(paths[0], bbox_inches="tight", pad_inches=0.02)
    if tiff:
        paths.append(stem.with_suffix(".tif"))
        fig.savefig(paths[1], dpi=600, bbox_inches="tight", pad_inches=0.02,
                    pil_kwargs={"compression": "tiff_lzw"})
    png = stem.with_suffix(".png")
    fig.savefig(png, dpi=200, bbox_inches="tight", pad_inches=0.02)
    paths.append(png)
    return paths
