"""Shared settings for the Remote Sensing of Environment figures (figures/rse_fig_*.py).

Elsevier artwork rules: single column 90 mm, 1.5 column 140 mm, double column 190 mm,
height at most about 240 mm; fonts Arial, Courier, Times New Roman or Symbol; vector
PDF, TIFF of at least 300 dpi for halftones and 500-1000 dpi for line art. The project
rules add: sans-serif text, 7 pt for labels and never below 6 pt at final
size; lowercase bold panel letters; colour-blind-safe perceptually uniform colour maps
(cmocean or Crameri); units on every axis; scale bars on maps. Files: vector PDF
(TrueType Arial embedded), 600 dpi LZW TIFF and a 200 dpi PNG preview in figures/out/, at
the exact figure size (_common.save).

Every number drawn comes from results/metrics.json (``num``, formatted as the text's
macros) or from a results/ table read by the figure script.
"""

from __future__ import annotations

import _common as C
import _erl as E
import matplotlib as mpl

from caspian import plotting

SINGLE_MM = 90.0
ONEHALF_MM = 140.0
DOUBLE_MM = 190.0
MAX_H_MM = 240.0
FS = 7.0            # labels, ticks, legends
FS_SMALL = 6.0      # the floor: map annotations, dense keys
FS_LABEL = 8.0      # panel letters
MIN_FS = 6.0
INK, MUTED, GRID = plotting.INK, plotting.INK_MUTED, plotting.GRID

COLOURS = dict(C.PRODUCT_COLOURS)
MARKERS = dict(C.PRODUCT_MARKERS)
NAMES = dict(E.NAMES)                 # SRTM15+ without the version
DASHES = dict(E.DASHES)
PK = dict(E.PK)
PRODUCTS = ("gebco_2026", "gebco_2022", "etopo_2022", "srtm15p_v2.7", "etopo1")
OBS = "#000000"
SEA = "#d7e6f2"
LAND = "#ececec"
VALIDATED = ("west", "interfluve", "northeast", "open_shelf")      # PREREG Amendment 18 item 1
NOT_VALIDATED = ("volga", "ural", "komsomolets")
SECTORS = VALIDATED + NOT_VALIDATED
# One name per sector, used on every RSE figure (review C18).
NAME = {"west": "Kizlyar–Dagestan coast", "volga": "Volga delta", "interfluve": "Volga–Ural interfluve coast",
        "ural": "Ural delta", "northeast": "NE Kazakh coast", "komsomolets": "Komsomolets Bay",
        "open_shelf": "Open shelf"}
NAME2 = {"west": "Kizlyar–Dagestan\ncoast", "volga": "Volga delta", "interfluve": "Volga–Ural\ninterfluve coast",
         "ural": "Ural delta", "northeast": "NE Kazakh\ncoast", "komsomolets": "Komsomolets\nBay",
         "open_shelf": "Open shelf"}

num = E.num
value = E.value
THIN = " "     # thousands separator, as the text's macros print it (_erl.num)


def fmt_int(n) -> str:
    return f"{int(n):,}".replace(",", THIN)


def fmt_num(v: float, nd: int = 1) -> str:
    """A number drawn from a results table, with a true minus sign."""
    return f"{v:.{nd}f}".replace("-", "−")
mm_axes = E.mm_axes
map_axes_clean = E.map_axes_clean


def style() -> None:
    C.style()
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                         "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": FS, "axes.titlesize": FS,
                         "axes.labelsize": FS, "xtick.labelsize": FS, "ytick.labelsize": FS, "legend.fontsize": FS,
                         "legend.title_fontsize": FS, "axes.linewidth": 0.5, "xtick.major.width": 0.5,
                         "ytick.major.width": 0.5, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
                         "xtick.minor.size": 1.5, "ytick.minor.size": 1.5, "lines.linewidth": 0.9,
                         "hatch.linewidth": 0.4, "axes.labelpad": 2.0, "xtick.major.pad": 1.8,
                         "ytick.major.pad": 1.8})


def label(ax, letter: str, x: float = 0.0, y: float = 1.0, inside: bool = False, **kw) -> None:
    """Lowercase bold panel letter at axes fraction (x, y): above the top-left corner, or
    inside it on a white box (maps)."""
    if inside:
        ax.text(x + 0.012, y - 0.015, letter, transform=ax.transAxes, fontsize=FS_LABEL, fontweight="bold",
                ha="left", va="top", color=INK, zorder=40,
                bbox={"boxstyle": "square,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.9}, **kw)
    else:
        ax.text(x, y, letter, transform=ax.transAxes, fontsize=FS_LABEL, fontweight="bold", ha="left",
                va="bottom", color=INK, **kw)


def fig_label(fig, W: float, H: float, letter: str, x_mm: float, y_top_mm: float) -> None:
    """Panel letter placed in millimetres from the figure's top-left corner."""
    fig.text(x_mm / W, 1 - y_top_mm / H, letter, fontsize=FS_LABEL, fontweight="bold", ha="left", va="top", color=INK)


def scale_bar(ax, x0: float, y0: float, length_m: float, text: str, transform=None, lw: float = 1.3,
              box: bool = False, size: float = FS_SMALL) -> None:
    """A bar of ``length_m`` metres (map units) from (x0, y0), labelled above it."""
    from matplotlib import patheffects

    kw = {} if transform is None else {"transform": transform}
    halo = [patheffects.withStroke(linewidth=1.8, foreground="white")]
    ax.plot([x0, x0 + length_m], [y0, y0], color="white", lw=lw + 2.0, solid_capstyle="butt", zorder=38, **kw)
    ax.plot([x0, x0 + length_m], [y0, y0], color=INK, lw=lw, solid_capstyle="butt", zorder=39, **kw)
    ax.annotate(text, xy=(x0 + length_m / 2, y0), xycoords=transform or "data", xytext=(0, 1.5),
                textcoords="offset points", ha="center", va="bottom", fontsize=size, color=INK,
                path_effects=None if box else halo, zorder=39,
                bbox={"boxstyle": "square,pad=0.1", "fc": "white", "ec": "none"} if box else None)


def laea_graticule(ax, fwd, extent, lons, lats, **kw) -> None:
    kw.setdefault("size", FS_SMALL)
    kw.setdefault("lw", 0.35)
    E.laea_graticule(ax, fwd, extent, lons, lats, **kw)


def check_text(fig, min_size: float = MIN_FS) -> None:
    """Stop if any visible text of the figure is smaller than ``min_size`` points."""
    from matplotlib.text import Text

    small = []
    for t in fig.findobj(Text):
        if t.get_visible() and t.get_text().strip() and t.get_fontsize() < min_size - 1e-6:
            small.append((t.get_text()[:30], t.get_fontsize()))
    if small:
        raise SystemExit(f"text below {min_size} pt: {small[:10]}")


def check_size(W: float, H: float) -> None:
    if H > MAX_H_MM + 1e-6:
        raise SystemExit(f"figure height {H:.1f} mm exceeds {MAX_H_MM} mm")


def save(fig, name: str, W: float, H: float):
    check_text(fig)
    check_size(W, H)
    return C.save(fig, name)
