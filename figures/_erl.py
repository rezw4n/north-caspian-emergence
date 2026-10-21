"""Shared settings for the ERL main display items (figures/erl_fig*.py, tab_report_card.py;
D-100, D-102).

Environmental Research Letters figure rules (docs/journal_specs.md): text 8-12 pt at final
size, fonts restricted to Times, Helvetica, Courier or Symbol (here TeX Gyre Heros, a
Helvetica clone; see FONT_FAMILY), widths 8.5 cm (single) and 15 cm
(double column), vector PDF preferred, colour never the only means of conveying
information, panels labelled (a), (b), ..., scale bars on maps. These take precedence over
the project's 5-7 pt rule (CLAUDE.md, Figures; D-099). Text here is 8 pt or more, panel
labels bold "(a)". Files: vector PDF, 600 dpi TIFF and a PNG preview in figures/out/,
written at the exact figure size by _common.save.

Every number drawn comes from results/metrics.json through ``num``, formatted as
tools/make_macros.py formats it for the text (same decimals), with LaTeX spacing and minus
signs turned into their Unicode characters.
"""

from __future__ import annotations

import importlib.util

import _common as C
import matplotlib as mpl

from caspian import plotting

SINGLE_MM = 85.0
DOUBLE_MM = 150.0
# Display constants quoted in the captions (exported by tools/export_params.py as params.erl_fig_*)
SCALE_MAP_KM = 100           # scale bar of the whole-domain maps
SCALE_INSET_KM = 20          # scale bar of the 80 x 70 km zooms
SCALE_GSL_KM = 10            # scale bar of the Great Salt Lake map
GRATICULE_MAP_DEG = 1.0      # graticule of the whole-domain maps
GRATICULE_ZOOM_DEG = 0.5     # graticule of the zooms and of the Great Salt Lake map
NEAR_KM = 10                 # exposure features drawn within this distance of the 2 m band
JPEG_QUALITY = 85            # photographic layers in the PDF (then-and-now composites), D-102 addendum
FS = 8.0            # every text element
FS_LABEL = 9.0      # panel labels
INK, MUTED, GRID = plotting.INK, plotting.INK_MUTED, plotting.GRID

# Product identity: the Okabe-Ito hues of _common (validated, CVD-checked) plus a line
# style and a marker per product, so that colour is never the only cue.
COLOURS = dict(C.PRODUCT_COLOURS)
MARKERS = dict(C.PRODUCT_MARKERS)
NAMES = dict(C.PRODUCT_NAMES)
NAMES["srtm15p_v2.7"] = "SRTM15+"
DASHES = {"gebco_2026": (0, (5.0, 1.8)), "etopo_2022": (0, (5.0, 1.5, 1.2, 1.5)), "srtm15p_v2.7": (0, (1.2, 1.4)),
          "gebco_2022": (0, (8.0, 2.0)), "etopo1": (0, (5.0, 1.2, 1.2, 1.2, 1.2, 1.2))}
PK = {"gebco_2026": "gebco2026", "gebco_2022": "gebco2022", "etopo_2022": "etopo2022", "srtm15p_v2.7": "srtm15p",
      "etopo1": "etopo1"}
CONTROL = "#4d4d4d"          # 3DEP lidar control at Great Salt Lake: ink grey, its own dash
CONTROL_DASH = (0, (3.0, 1.0, 1.0, 1.0, 1.0, 1.0))
OBS = "#000000"              # observed shoreline: solid black

_mm = None


def _macros():
    global _mm
    if _mm is None:
        spec = importlib.util.spec_from_file_location("make_macros", C.ROOT / "tools" / "make_macros.py")
        _mm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_mm)
    return _mm


def num(m: dict, key: str, unit: str = "") -> str:
    """The metric ``key`` as the text prints it (tools/make_macros.fmt), in Unicode: thin
    spaces as thousands separators, a true minus sign. Stops on a missing key."""
    if key not in m:
        raise SystemExit(f"metric {key} missing from results/metrics.json")
    r = m[key]
    s = _macros().fmt(key, r["value"], r.get("decimals"), r.get("unit"))
    bs = chr(92)
    s = s.replace(bs + "textminus{}", "−").replace(bs + ",", " ").replace(bs + "_", "_")
    return s + (unit and " " + unit)


def value(m: dict, key: str) -> float:
    if key not in m:
        raise SystemExit(f"metric {key} missing from results/metrics.json")
    return float(m[key]["value"])


# Typeface (D-102 addendum). IOP's figure guidelines: "Fonts used should be restricted to the
# standard font families (Times, Helvetica, Courier or Symbol)" (docs/journal_specs.md). Arial is
# not one of them, and Helvetica itself cannot be embedded here, so the figures use TeX Gyre
# Heros, the GUST Foundry's version of the URW Helvetica clone (GUST Font License), which
# matplotlib embeds as a CID font. Its OpenType files come from tectonic's bundle cache (the
# manuscript build fetches them); ``font_files`` fetches them with tectonic if they are missing.
FONT_FAMILY = "TeX Gyre Heros"
FONT_FILES = ("texgyreheros-regular.otf", "texgyreheros-bold.otf", "texgyreheros-italic.otf",
              "texgyreheros-bolditalic.otf")


def _font_dirs():
    import os
    from pathlib import Path

    env = os.environ.get("CASPIAN_FONT_DIR")
    roots = [Path(env)] if env else []
    local = os.environ.get("LOCALAPPDATA")
    if local:
        roots.append(Path(local) / "TectonicProject" / "Tectonic" / "cache" / "bundles" / "data")
    roots += [Path.home() / ".cache" / "Tectonic" / "bundles" / "data",
              Path.home() / "Library" / "Caches" / "Tectonic" / "bundles" / "data"]
    return roots


def _find_fonts() -> dict:
    found = {}
    for root in _font_dirs():
        if not root.exists():
            continue
        for name in FONT_FILES:
            if name not in found:
                hits = sorted(root.glob(name)) + sorted(root.glob(f"*/{name}"))
                if hits:
                    found[name] = hits[0]
    return found


def font_files() -> list:
    """The four TeX Gyre Heros files; if tectonic's cache lacks them, compile a one-line
    document with tectonic (which downloads them into its cache) and look again."""
    import shutil
    import subprocess
    import tempfile
    from pathlib import Path

    found = _find_fonts()
    if len(found) < len(FONT_FILES) and shutil.which("tectonic"):
        bs = chr(92)
        doc = (bs + "documentclass{article}" + chr(10) + bs + "usepackage{fontspec}" + chr(10) + bs
               + "setmainfont{texgyreheros}[Extension=.otf,UprightFont=*-regular,BoldFont=*-bold,"
               "ItalicFont=*-italic,BoldItalicFont=*-bolditalic]" + chr(10) + bs + "begin{document}x " + bs
               + "textbf{x} " + bs + "textit{x} " + bs + "textbf{" + bs + "textit{x}}" + bs + "end{document}" + chr(10))
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "f.tex").write_text(doc, encoding="utf-8")
            subprocess.run(["tectonic", "-X", "compile", "f.tex"], cwd=d, capture_output=True, check=False)
        found = _find_fonts()
    missing = [n for n in FONT_FILES if n not in found]
    if missing:
        raise SystemExit(f"TeX Gyre Heros files not found ({missing}): run tectonic once with network access, or set "
                         "CASPIAN_FONT_DIR to a folder holding them")
    return [found[n] for n in FONT_FILES]


def style() -> None:
    from matplotlib import font_manager

    C.style()
    for f in font_files():
        font_manager.fontManager.addfont(str(f))
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": [FONT_FAMILY], "pdf.fonttype": 42,
                         "font.size": FS, "axes.titlesize": FS, "axes.labelsize": FS, "xtick.labelsize": FS,
                         "ytick.labelsize": FS, "legend.fontsize": FS, "legend.title_fontsize": FS,
                         "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                         "xtick.major.size": 3.0, "ytick.major.size": 3.0, "lines.linewidth": 1.0,
                         "hatch.linewidth": 0.5})


def jpeg_images(pdf, quality: int = JPEG_QUALITY) -> tuple[int, int]:
    """Re-encode the raster images inside a figure PDF as JPEG (PyMuPDF rewrite_images) at
    their own resolution, leaving text, lines and fonts as they are; for photographic layers
    (satellite composites), whose lossless encoding makes the PDF too large for the Letter.
    Returns the file size before and after."""
    import os
    from pathlib import Path

    import fitz

    pdf = Path(pdf)
    before = pdf.stat().st_size
    doc = fitz.open(pdf)
    doc.rewrite_images(dpi_threshold=None, dpi_target=0, quality=quality, lossy=True, lossless=True, bitonal=False,
                       color=True, gray=True)
    tmp = pdf.with_suffix(".tmp.pdf")
    doc.save(tmp, garbage=3, deflate=True)
    doc.close()
    os.replace(tmp, pdf)
    return before, pdf.stat().st_size


def label(ax, letter: str, x: float = 0.0, y: float = 1.0, inside: bool = False, **kw) -> None:
    """Panel label "(a)" in bold, at axes fraction (x, y): above the top-left corner, or
    inside it on a white box (maps)."""
    if inside:
        ax.text(x + 0.012, y - 0.015, f"({letter})", transform=ax.transAxes, fontsize=FS_LABEL, fontweight="bold",
                ha="left", va="top", color=INK, zorder=30,
                bbox={"boxstyle": "square,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.9}, **kw)
    else:
        ax.text(x, y, f"({letter})", transform=ax.transAxes, fontsize=FS_LABEL, fontweight="bold", ha="left",
                va="bottom", color=INK, **kw)


def scale_bar(ax, x0: float, y0: float, length_m: float, label_: str, transform=None, lw: float = 1.6,
              box: bool = False) -> None:
    """A bar of ``length_m`` metres (map units) from (x0, y0), labelled above it; with ``box``
    the label sits on a white box, so that lines under it do not show through."""
    from matplotlib import patheffects

    kw = {} if transform is None else {"transform": transform}
    halo = [patheffects.withStroke(linewidth=2.0, foreground="white")]
    ax.plot([x0, x0 + length_m], [y0, y0], color="white", lw=lw + 2.4, solid_capstyle="butt", zorder=28, **kw)
    ax.plot([x0, x0 + length_m], [y0, y0], color=INK, lw=lw, solid_capstyle="butt", zorder=29, **kw)
    ax.annotate(label_, xy=(x0 + length_m / 2, y0), xycoords=transform or "data", xytext=(0, 2),
                textcoords="offset points", ha="center", va="bottom", fontsize=FS, color=INK,
                path_effects=None if box else halo, zorder=29,
                bbox={"boxstyle": "square,pad=0.12", "fc": "white", "ec": "none"} if box else None)


def graticule(ax, xlocs, ylocs, labels=("left", "bottom"), size: float = FS, colour: str | None = None, inline=False):
    """Graticule in degrees on a cartopy map; labels on the named sides only."""
    import cartopy.crs as ccrs

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=bool(labels), x_inline=inline, y_inline=inline,
                      xlocs=xlocs, ylocs=ylocs, color=colour or GRID, lw=0.45, zorder=3)
    if labels:
        gl.top_labels = "top" in labels
        gl.bottom_labels = "bottom" in labels
        gl.left_labels = "left" in labels
        gl.right_labels = "right" in labels
        gl.xlabel_style = gl.ylabel_style = {"size": size, "color": MUTED}
        gl.xpadding = gl.ypadding = 2
        gl.rotate_labels = False
    return gl


def mm_axes(fig, W: float, H: float, x0: float, y_top: float, w: float, h: float, **kw):
    """Axes placed in millimetres from the figure's top-left corner."""
    return fig.add_axes([x0 / W, (H - y_top - h) / H, w / W, h / H], **kw)


def laea_graticule(ax, fwd, extent, lons, lats, *, inside: bool = False, label_lon: bool = True,
                   label_lat: bool = True, size: float = FS, lw: float = 0.45, zorder: int = 3) -> None:
    """Graticule on a plain axes in equal-area metres (``fwd``: lon/lat -> x/y). Meridians and
    parallels are drawn as dense curves; each is labelled where it crosses the bottom (lon) or
    left (lat) edge, outside the frame or, with ``inside``, just inside it on a white box."""
    import numpy as np
    from matplotlib import patheffects

    x0, x1, y0, y1 = extent
    halo = [patheffects.withStroke(linewidth=1.8, foreground="white")]
    for lon in lons:
        la = np.linspace(min(lats) - 3, max(lats) + 3, 400)
        x, y = fwd(np.full_like(la, lon), la)
        ax.plot(x, y, color=GRID, lw=lw, zorder=zorder)
        if label_lon:
            k = np.argsort(y)
            xb = float(np.interp(y0, np.asarray(y)[k], np.asarray(x)[k]))
            if x0 < xb < x1:
                t = f"{abs(lon):g}°{'E' if lon >= 0 else 'W'}"
                if inside:
                    ax.text(xb, y0 + 0.015 * (y1 - y0), t, ha="center", va="bottom", fontsize=size, color=MUTED,
                            zorder=zorder + 20, path_effects=halo)
                else:
                    ax.text(xb, y0 - 0.012 * (y1 - y0), t, ha="center", va="top", fontsize=size, color=MUTED,
                            clip_on=False)
    for lat in lats:
        lo = np.linspace(min(lons) - 4, max(lons) + 4, 400)
        x, y = fwd(lo, np.full_like(lo, lat))
        ax.plot(x, y, color=GRID, lw=lw, zorder=zorder)
        if label_lat:
            yl = float(np.interp(x1 if inside else x0, np.asarray(x), np.asarray(y)))
            top = y1 - (0.2 if inside else 0.0) * (y1 - y0)     # inside: keep clear of the title and scale bar
            if y0 + (0.12 if inside else 0.0) * (y1 - y0) < yl < top:
                t = f"{abs(lat):g}°{'N' if lat >= 0 else 'S'}"
                if inside:
                    ax.text(x1 - 0.012 * (x1 - x0), yl, t, ha="right", va="bottom", fontsize=size, color=MUTED,
                            zorder=zorder + 20, path_effects=halo)
                else:
                    ax.text(x0 - 0.008 * (x1 - x0), yl, t, ha="right", va="center", fontsize=size, color=MUTED,
                            clip_on=False)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)


def map_axes_clean(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(MUTED)
        s.set_linewidth(0.6)
    ax.set_aspect("equal")
