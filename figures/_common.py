"""Shared settings for the display-item scripts in figures/.

Widths follow the Nature Communications author guidelines read on 2026-10-21
(docs/journal_specs.md): single column 88 mm, double column 180 mm. They differ from
the placeholders in caspian.plotting (89 and 183 mm), which are not used here. The
venue is chosen at Gate G3; Communications Earth & Environment gives no width.

Text follows CLAUDE.md and the Nature Portfolio figure guide: sans-serif 5-7 pt,
lowercase bold 8 pt panel letters (caspian.plotting.use_style, panel_label). Files are
saved at the exact figure size (no tight cropping, which would change the width):
vector PDF with TrueType fonts, a 600 dpi LZW TIFF and a 200 dpi PNG preview, in
figures/out/.
"""

from __future__ import annotations

import json
from pathlib import Path

import caspian  # noqa: F401  (before any GDAL-linked import; D-004)
import matplotlib as mpl

from caspian import plotting

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "out"
MM = 1 / 25.4
SINGLE_MM = 88.0
DOUBLE_MM = 180.0

# Okabe-Ito hues, checked with the dataviz palette validator (light surface): CVD and
# normal-vision separation pass; orange is below 3:1 contrast, so every use carries a
# legend entry or a direct label.
SPLIT_COLOURS = {"train": "#0072B2", "calib": "#E69F00", "test": "#009E73"}
SPLIT_NAMES = {"train": "Training", "calib": "Calibration", "test": "Test"}
SECTOR_SHORT = {
    "west": "Kizlyar Bay–Dagestan–Kalmyk coast",
    "volga": "Volga delta",
    "interfluve": "Volga–Ural interfluve coast",
    "ural": "Ural delta",
    "northeast": "NE Kazakh coast (Tengiz–Kashagan)",
    "komsomolets": "Komsomolets Bay and Kaidak",
    "open_shelf": "Open shelf",
}


def style() -> None:
    plotting.use_style()
    mpl.rcParams.update({"lines.linewidth": 0.75, "patch.linewidth": 0.5, "hatch.linewidth": 0.5})


def metrics() -> dict:
    """results/metrics.json, so that dates and counts drawn on a figure are the ones
    the text quotes."""
    return json.loads((ROOT / "results" / "metrics.json").read_text(encoding="utf-8"))


def save(fig, name: str) -> list[Path]:
    """Write figures/out/<name>.pdf, .tif (600 dpi) and .png (preview) at the figure's
    own size."""
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / name
    paths = [stem.with_suffix(".pdf"), stem.with_suffix(".tif"), stem.with_suffix(".png")]
    fig.savefig(paths[0])
    fig.savefig(paths[1], dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(paths[2], dpi=200)
    w, h = fig.get_size_inches() / MM
    print(f"{name}: {w:.1f} x {h:.1f} mm ->", ", ".join(p.relative_to(ROOT).as_posix() for p in paths))
    return paths


def scale_bar(ax, x0: float, y0: float, length_m: float, label: str, transform=None, lw: float = 1.2) -> None:
    """A plain bar of ``length_m`` (map units in metres) from (x0, y0), labelled above."""
    from matplotlib import patheffects

    kw = {} if transform is None else {"transform": transform}
    halo = [patheffects.withStroke(linewidth=1.6, foreground="white")]
    ax.plot([x0, x0 + length_m], [y0, y0], color="white", lw=lw + 1.2, solid_capstyle="butt", zorder=8, **kw)
    ax.plot([x0, x0 + length_m], [y0, y0], color=plotting.INK, lw=lw, solid_capstyle="butt", zorder=9, **kw)
    ax.annotate(label, xy=(x0 + length_m / 2, y0), xycoords=transform or "data", xytext=(0, 1.5),
                textcoords="offset points", ha="center", va="bottom", fontsize=5.5, color=plotting.INK,
                path_effects=halo, zorder=9)
