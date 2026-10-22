"""ERL figure "Then and now" (15 cm wide).

(a) Landsat 5 TM, summer of the 1995-96 high stand, and (b) Landsat 8 and 9 OLI, summer of
    the 2025-26 low stand: true-colour median composites of Collection 2 Level-2 surface
    reflectance at 200 m on the analysis grid (scripts/f2_erl_landsat.py; cloud, shadow and
    saturated pixels masked; the adjacent summer fills pixels with no clear value). Both
    use the same linear stretch of reflectance (STRETCH, gamma GAMMA), so their colours
    compare. On (b): the observed shoreline at four E3 level bins 0.6 m apart: the
    0.5 contour of the share
    of sea-connected water per 240 m block of each bin's observed extent, built with the
    hindcast's functions (scripts/f3_erl_waterlines.py, data/interim/figures/erl/
    waterlines_main.npz), pieces shorter than MIN_LEN_KM dropped; coloured by level
    (Crameri batlow). Earlier versions drew contours of the waterline DEM's emergence
    dates, which failed validation at the Volga delta front; they are no longer used.
(c) The daily basin level (level_series_daily.csv: DAHITI, with Hydroweb shifted onto the
    DAHITI reference in its gaps), the summers of the two composites (shaded) and, for
    each level in (b), the period of its Landsat acquisitions (bar at the bin level, from
    the first to the last acquisition, same colours).

Numbers: results/metrics.json (erl.landsat.*, rev.fig.f1.*, params.*).
Output: figures/out/erl_fig_then_now.{pdf,tif,png}
Usage: python figures/erl_fig_then_now.py
"""

from __future__ import annotations

import sys

import json

import _common as C
import _erl as E
import numpy as np
import pandas as pd

from caspian.config import DATA, RAW

EXTENT = (-276_000.0, 322_000.0, -104_000.0, 200_000.0)
STRETCH = (0.0, 0.40)        # reflectance mapped to 0..1
GAMMA = 0.85
LEVEL_COLOUR = (0.08, 0.36, 0.62, 0.86)   # positions on Crameri batlow, high level to low
BLOCK = 4                          # display block: 4 x 60 m cells, whose water share is contoured at 0.5
MIN_LEN_KM = 6                     # contour pieces shorter than this are not drawn
WATERLINES = C.ROOT / "results" / "display" / "erl_waterlines.json"
WATERLINE_FIELDS = C.ROOT / "data" / "interim" / "figures" / "erl" / "waterlines_main.npz"


def rgb(path):
    z = np.load(path)
    bands = [z[b].astype(np.float32) for b in ("red", "green", "blue")]
    bad = np.any([b == -32768 for b in bands], axis=0)
    img = np.stack([np.clip((b / 10000.0 - STRETCH[0]) / (STRETCH[1] - STRETCH[0]), 0, 1) ** GAMMA for b in bands], -1)
    img[bad] = 1.0
    t = z["transform"]
    ny, nx = bands[0].shape
    return img, [t[2], t[2] + t[0] * nx, t[5] + t[4] * ny, t[5]]


def iso_lines(field, ext, level: float, min_len_m: float = MIN_LEN_KM * 1000.0):
    """Contour pieces of ``field`` at ``level`` (skimage.measure.find_contours, NaN masked) in
    map metres, without pieces shorter than ``min_len_m``."""
    from skimage import measure

    px = (ext[1] - ext[0]) / field.shape[1]
    ok = np.isfinite(field)
    out = []
    for c in measure.find_contours(np.where(ok, field, level - 50.0), level, mask=ok):
        xy = np.c_[ext[0] + (c[:, 1] + 0.5) * px, ext[3] - (c[:, 0] + 0.5) * px]
        if np.sum(np.hypot(*np.diff(xy, axis=0).T)) >= min_len_m:
            out.append(xy)
    return out


def main() -> int:
    import cmcrameri.cm as cmc
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib import patheffects
    from matplotlib.lines import Line2D

    m = C.metrics()
    _, fwd = C.projection()
    domain, _ = C.study_geometries()
    img96, ext96 = rgb(RAW / "erl_landsat" / "tm_1996.npz")
    img25, ext25 = rgb(RAW / "erl_landsat" / "oli_2025.npz")
    wl = json.loads(WATERLINES.read_text(encoding="utf-8"))
    wz = np.load(WATERLINE_FIELDS)
    wext = [float(v) for v in wz["extent"]]
    if int(wl["display_res_m"]) != BLOCK * 60:        # 60 m: the hindcast grid of the observed extents
        raise SystemExit("erl_waterlines.json: display blocks differ from BLOCK")
    bins = [b["bin"] for b in wl["bins"]]
    if len(bins) != len(LEVEL_COLOUR) or int(m["rev.fig.f1.n_levels"]["value"]) != len(bins):
        raise SystemExit("erl_waterlines.json and metrics.json disagree on the levels drawn")
    bcol = {b: cmc.batlow(v) for b, v in zip(bins, LEVEL_COLOUR)}
    blab = {}
    for i, b in enumerate(bins, start=1):
        k = f"rev.fig.f1.level{i}"
        if abs(E.value(m, f"{k}.level_m") - wl["bins"][i - 1]["level_m"]) > 1e-9:
            raise SystemExit(f"bin {b}: level differs from metrics.json")
        blab[b] = f"{E.num(m, f'{k}.level_m')} m ({m[f'{k}.first_year']['value']}–{m[f'{k}.last_year']['value']})"
    lv = pd.read_csv(DATA / "processed" / "dem" / "main" / "emergence" / "level_series_daily.csv", parse_dates=["date"])

    E.style()
    W = E.DOUBLE_MM
    L, R, TOP = 9.0, 1.5, 2.0
    mw = W - L - R
    mh = mw * (EXTENT[3] - EXTENT[2]) / (EXTENT[1] - EXTENT[0])
    GAP, HC, BOT = 4.0, 32.0, 9.0
    H = TOP + 2 * mh + GAP + 19.0 + HC + BOT
    fig = plt.figure(figsize=(W * C.MM, H * C.MM))
    halo = [patheffects.withStroke(linewidth=2.2, foreground="white")]
    for i, (img, ext, key) in enumerate(((img96, ext96, "tm_1996"), (img25, ext25, "oli_2025"))):
        ax = E.mm_axes(fig, W, H, L, TOP + i * (mh + GAP), mw, mh)
        ax.imshow(img, extent=ext, origin="upper", interpolation="nearest", zorder=1, rasterized=True)
        E.laea_graticule(ax, fwd, EXTENT, lons=np.arange(47, 55, E.GRATICULE_MAP_DEG),
                         lats=np.arange(44, 48, E.GRATICULE_MAP_DEG), label_lon=(i == 1), lw=0.35)
        dx, dy = domain.exterior.xy
        ax.plot(dx, dy, color="white", lw=0.6, ls=(0, (3, 2)), zorder=4)
        if i == 1:
            for b in bins:
                f = wz[f"obs_b{b:03d}"].astype(np.float32)
                for seg in iso_lines(f, wext, 0.5):
                    ax.plot(seg[:, 0], seg[:, 1], color=bcol[b], lw=1.3, zorder=6, solid_capstyle="round",
                            path_effects=[patheffects.withStroke(linewidth=2.6, foreground="white")])
        E.map_axes_clean(ax)
        ax.set_xlim(EXTENT[0], EXTENT[1])
        ax.set_ylim(EXTENT[2], EXTENT[3])
        E.scale_bar(ax, EXTENT[0] + 18_000, EXTENT[2] + 14_000, E.SCALE_MAP_KM * 1000.0,
                    E.num(m, "params.erl_fig_scale_map_km", "km"), box=True)
        E.label(ax, "ab"[i], inside=True)
        sensor = "Landsat 5 TM" if key == "tm_1996" else "Landsat 8 and 9 OLI"
        ax.text(0.985, 0.975, f"{sensor}, summer {m[f'erl.landsat.{key}.year']['value']}", transform=ax.transAxes,
                ha="right", va="top", fontsize=E.FS, color=E.INK, zorder=20,
                bbox={"boxstyle": "square,pad=0.25", "fc": "white", "ec": "none", "alpha": 0.9})
    # shoreline key (level and years of the acquisitions), under the maps
    yk = TOP + 2 * mh + GAP + 7.5
    handles = [Line2D([], [], color=bcol[b], lw=2.4, label=blab[b]) for b in bins]
    fig.text(L / W, 1 - (yk - 2.4) / H, "Observed shoreline at the level\n(years of the acquisitions; b, c):",
             ha="left", va="top", fontsize=E.FS, color=E.INK, linespacing=1.1)
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=((L + 49.0) / W, 1 - (yk - 3.2) / H), ncol=2,
               fontsize=E.FS, frameon=False, borderaxespad=0, handlelength=2.0, columnspacing=1.4, handletextpad=0.5,
               labelspacing=0.3)

    # (c) level record ----------------------------------------------------------------------------
    axc = E.mm_axes(fig, W, H, L + 6.0, yk + 11.0, mw - 6.0, HC)
    t0, t1 = pd.Timestamp("1992-07-01"), pd.Timestamp("2027-01-01")
    for key in ("tm_1996", "oli_2025"):
        y = int(m[f"erl.landsat.{key}.year"]["value"])
        axc.axvspan(pd.Timestamp(f"{y}-06-01"), pd.Timestamp(f"{y}-10-01"), color="#e3e3e3", lw=0, zorder=0)
        axc.text(pd.Timestamp(f"{y}-08-01"), 0.97, "(a)" if key == "tm_1996" else "(b)", transform=axc.get_xaxis_transform(),
                 ha="center", va="top", fontsize=E.FS, color=E.INK)
    for src, col, lab in (("dahiti", E.INK, "DAHITI"), ("hydroweb_shifted", "#8c8c8c", "Hydroweb, shifted (DAHITI gaps)")):
        yv = np.where(lv["source"] == src, lv["level_m"], np.nan)
        axc.plot(lv["date"], yv, color=col, lw=0.9, zorder=3, solid_capstyle="butt", label=lab)
    for b, rec in zip(bins, wl["bins"]):
        t0_, t1_ = pd.Timestamp(rec["first_date"]), pd.Timestamp(rec["last_date"])
        axc.plot([t0_, t1_], [rec["level_m"]] * 2, color=bcol[b], lw=3.0, solid_capstyle="butt", zorder=2,
                 path_effects=[patheffects.withStroke(linewidth=4.6, foreground="white")])
    axc.set_xlim(t0, t1)
    axc.xaxis.set_major_locator(mdates.YearLocator(5))
    axc.xaxis.set_minor_locator(mdates.YearLocator(1))
    axc.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axc.set_ylabel("Basin level (m)")
    axc.set_xlabel("Year")
    axc.grid(axis="y", color=E.GRID, lw=0.4)
    axc.set_axisbelow(True)
    axc.legend(loc="lower left", fontsize=E.FS, frameon=False, handlelength=1.6, borderaxespad=0.2)
    E.label(axc, "c", x=-0.075, y=1.0)
    del halo
    paths = C.save(fig, "erl_fig_then_now")
    # the composites are photographs: JPEG inside the PDF keeps the Letter small; text, lines and
    # fonts stay vector; the 600 dpi TIFF is written by C.save and not touched
    before, after = E.jpeg_images(paths[0])
    print(f"erl_fig_then_now.pdf: imagery as JPEG (quality {E.JPEG_QUALITY}), {before / 1e6:.1f} -> {after / 1e6:.1f} MB")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
