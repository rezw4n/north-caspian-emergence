"""ERL figure "Then and now" (D-100, D-102; 15 cm wide).

(a) Landsat 5 TM, summer of the 1995-96 high stand, and (b) Landsat 8 and 9 OLI, summer of
    the 2025-26 low stand: true-colour median composites of Collection 2 Level-2 surface
    reflectance at 200 m on the analysis grid (scripts/f2_erl_landsat.py; cloud, shadow and
    saturated pixels masked; the adjacent summer fills pixels with no clear value). Both
    use the same linear stretch of reflectance (STRETCH, gamma GAMMA), so their colours
    compare. On (b): the shoreline at successive dates, drawn as contours of the emergence
    date of the emerged band (data/processed/dem/main/emergence/, scripts/a6_emergence.py:
    the first day after the last day on which the basin level stood at or above the pixel's
    waterline-DEM height), on a 150 m display grid inside the D-071 emerged band, coloured
    by year (Crameri batlow); display smoothing as in ``smoothed`` and ``iso_lines``.
(c) The daily basin level used to date emergence (level_series_daily.csv: DAHITI, with
    Hydroweb shifted onto the DAHITI reference in its gaps), the summers of the two
    composites (shaded) and the level on 1 January of each contour year (dots, same
    colours).

The emergence dates inherit the waterline DEM's errors (Supplementary DEM figure); the
contours are a view of the record, not an independent observation.

Numbers: results/metrics.json (erl.landsat.*, params.*).
Output: figures/out/erl_fig_then_now.{pdf,tif,png}
Usage: python figures/erl_fig_then_now.py
"""

from __future__ import annotations

import sys

import _common as C
import _data as D
import _erl as E
import numpy as np
import pandas as pd

from caspian.config import DATA, RAW

EXTENT = (-276_000.0, 322_000.0, -104_000.0, 200_000.0)
STRETCH = (0.0, 0.40)        # reflectance mapped to 0..1
GAMMA = 0.85
YEARS = (2010, 2015, 2020, 2025)  # years of contrasting level (the level hovered near -26.5 m before 2006)
YEAR_COLOUR = (0.2, 0.45, 0.7, 0.92)   # positions on Crameri batlow, spread for distinct hues
BLOCK = 4                          # display smoothing: 4 x 150 m = 600 m block medians ...
SIGMA = 1.0                        # ... then a Gaussian of one block
MIN_LEN_KM = 6                     # contour pieces shorter than this are not drawn


def rgb(path):
    z = np.load(path)
    bands = [z[b].astype(np.float32) for b in ("red", "green", "blue")]
    bad = np.any([b == -32768 for b in bands], axis=0)
    img = np.stack([np.clip((b / 10000.0 - STRETCH[0]) / (STRETCH[1] - STRETCH[0]), 0, 1) ** GAMMA for b in bands], -1)
    img[bad] = 1.0
    t = z["transform"]
    ny, nx = bands[0].shape
    return img, [t[2], t[2] + t[0] * nx, t[5] + t[4] * ny, t[5]]


def smoothed(yr, ext, block: int = BLOCK, sigma: float = SIGMA):
    """Display smoothing of the emergence-year field: the median of each block x block group
    of 150 m pixels (NaN-aware; NaN where none is dated), then a Gaussian of ``sigma``
    blocks normalised by the dated share (NaN stays NaN). Returns the field and its extent."""
    from scipy import ndimage

    ny, nx = (yr.shape[0] // block) * block, (yr.shape[1] // block) * block
    a = yr[:ny, :nx].reshape(ny // block, block, nx // block, block).transpose(0, 2, 1, 3).reshape(
        ny // block, nx // block, block * block)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(a, axis=2)
    ok = np.isfinite(med)
    num = ndimage.gaussian_filter(np.where(ok, med, 0.0), sigma)
    den = ndimage.gaussian_filter(ok.astype(float), sigma)
    out = np.where(ok, num / np.maximum(den, 1e-9), np.nan)
    px = (ext[1] - ext[0]) / yr.shape[1] * block
    return out, [ext[0], ext[0] + px * out.shape[1], ext[3] - px * out.shape[0], ext[3]]


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
    em = D.mosaic("emergence", float(C.MAP_RES_M))
    band = D.emerged_band(em["extent"], em["z"].shape)
    yr = np.where(band, 1970.0 + em["z"] / 365.2425, np.nan)
    sm, sext = smoothed(yr, em["extent"])
    ycol = {y: cmc.batlow(v) for y, v in zip(YEARS, YEAR_COLOUR)}
    lv = pd.read_csv(DATA / "processed" / "dem" / "main" / "emergence" / "level_series_daily.csv", parse_dates=["date"])

    E.style()
    W = E.DOUBLE_MM
    L, R, TOP = 9.0, 1.5, 2.0
    mw = W - L - R
    mh = mw * (EXTENT[3] - EXTENT[2]) / (EXTENT[1] - EXTENT[0])
    GAP, HC, BOT = 4.0, 32.0, 9.0
    H = TOP + 2 * mh + GAP + 13.0 + HC + BOT
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
            for y in YEARS:
                for seg in iso_lines(sm, sext, float(y)):
                    ax.plot(seg[:, 0], seg[:, 1], color=ycol[y], lw=1.1, zorder=6, solid_capstyle="round",
                            path_effects=[patheffects.withStroke(linewidth=2.1, foreground="white")])
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
    # contour key (years), under the maps
    yk = TOP + 2 * mh + GAP + 7.5
    handles = [Line2D([], [], color=ycol[y], lw=1.6, label=str(y),
                      path_effects=[patheffects.withStroke(linewidth=2.6, foreground="white")]) for y in YEARS]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=((L + 58.0) / W, 1 - (yk - 3.2) / H), ncol=len(YEARS),
               fontsize=E.FS, frameon=False, borderaxespad=0, handlelength=1.8, columnspacing=1.0, handletextpad=0.4)
    fig.text(L / W, 1 - (yk - 3.2) / H, "Shoreline on 1 January (b, c):", ha="left", va="top", fontsize=E.FS,
             color=E.INK)

    # (c) level record ----------------------------------------------------------------------------
    axc = E.mm_axes(fig, W, H, L + 6.0, yk + 5.0, mw - 6.0, HC)
    t0, t1 = pd.Timestamp("1992-07-01"), pd.Timestamp("2027-01-01")
    for key in ("tm_1996", "oli_2025"):
        y = int(m[f"erl.landsat.{key}.year"]["value"])
        axc.axvspan(pd.Timestamp(f"{y}-06-01"), pd.Timestamp(f"{y}-10-01"), color="#e3e3e3", lw=0, zorder=0)
        axc.text(pd.Timestamp(f"{y}-08-01"), 0.97, "(a)" if key == "tm_1996" else "(b)", transform=axc.get_xaxis_transform(),
                 ha="center", va="top", fontsize=E.FS, color=E.INK)
    for src, col, lab in (("dahiti", E.INK, "DAHITI"), ("hydroweb_shifted", "#8c8c8c", "Hydroweb, shifted (DAHITI gaps)")):
        yv = np.where(lv["source"] == src, lv["level_m"], np.nan)
        axc.plot(lv["date"], yv, color=col, lw=0.9, zorder=3, solid_capstyle="butt", label=lab)
    s = lv.set_index("date")["level_m"]
    for y in YEARS:
        d = pd.Timestamp(f"{y}-01-01")
        v = float(np.interp(mdates.date2num(d), mdates.date2num(s.index), s.to_numpy()))
        axc.plot(d, v, "o", ms=5.0, mfc=ycol[y], mec="white", mew=0.8, zorder=5)
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
    C.save(fig, "erl_fig_then_now")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
