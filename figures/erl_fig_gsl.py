"""ERL figure "The same error at Great Salt Lake" (generality test A9; D-094, D-100, D-102;
15 cm wide).

(a) Both arms of Great Salt Lake at one representative hindcast unit (the unit whose
    per-arm median displacements are closest to the audit's summary medians, chosen by
    scripts/f1_erl_display.py gsl; level in metrics erl.gsl.level_navd88_m, NAVD 88): the
    observed state of every 60 m cell (sea-connected open water, land, not determined),
    the observed shoreline (solid black), the shape-variant shorelines of GEBCO_2026, ETOPO
    2022 and SRTM15+ (each with its per-arm D-070 constant), and that of the USGS 3DEP 2016
    lidar control (grey). Static layers, envelope, heights and flood fills are those of
    scripts/g1_hindcast.py; every per-arm median of the unit equals the arm's
    displacement.csv.
(b) Per arm, the median over evaluable levels of each surface's per-level median shoreline
    displacement (shape variant; h1_decision.json), against the arm's observation noise
    floor (median over the same levels of the 95th percentile of the displacement between
    independent halves of the observations). Filled: the surface mispredicts the arm (the
    H1 rule of PREREG section 9, Amendment 16); open: it does not.

Numbers: results/metrics.json (generality.great_salt_lake.*, erl.gsl.*).
Output: figures/out/erl_fig_gsl.{pdf,tif,png}
Usage: python figures/erl_fig_gsl.py
"""

from __future__ import annotations

import json
import sys

import _common as C
import _erl as E
import numpy as np

SEA, LAND, UNKNOWN = "#cfe3f1", "#e9e6df", "#ffffff"
MAP_PRODUCTS = ("gebco_2026", "etopo_2022", "srtm15p_v2.7")
CHART = ("gebco_2026", "gebco_2022", "etopo_2022", "etopo1", "srtm15p_v2.7", "dep3")
ARMS = (("north", "North arm"), ("south", "South arm"))
LW = 1.1


def main() -> int:
    import matplotlib.pyplot as plt
    import pyproj
    from matplotlib import patheffects
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from shapely.geometry import shape as as_shape

    m = C.metrics()
    z = np.load(C.ROOT / "data" / "interim" / "figures" / "erl" / "gsl.npz")
    crs = str(z["crs"])
    ext = z["extent"]
    fwd = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform
    gj = json.loads((C.ROOT / "results" / "generality" / "great_salt_lake" / "arms_laea.geojson").read_text(encoding="utf-8"))
    arms = {f["properties"]["arm"]: as_shape(f["geometry"]) for f in gj["features"] if f["properties"]["role"] == "sector"}
    xb0 = min(g.bounds[0] for g in arms.values())
    yb0 = min(g.bounds[1] for g in arms.values())
    xb1 = max(g.bounds[2] for g in arms.values())
    yb1 = max(g.bounds[3] for g in arms.values())
    pad = 3000.0
    box = (xb0 - pad, xb1 + pad, yb0 - pad, yb1 + pad)
    ny, nx = z["obs"].shape
    xs = ext[0] + (np.arange(nx) + 0.5) * (ext[1] - ext[0]) / nx
    ys = ext[3] - (np.arange(ny) + 0.5) * (ext[3] - ext[2]) / ny

    def cls(a):
        out = np.zeros(a.shape, np.int8)
        out[a == 0] = 1
        out[a == 1] = 2
        return out

    def contour(ax, a, colour, dash, lw=LW, zorder=8):
        f = a.astype(float)
        f[a < 0] = np.nan
        cs = ax.contour(xs, ys, f, [0.5], colors=[colour], linewidths=lw, zorder=zorder)
        cs.set_linestyle(dash)
        return cs

    E.style()
    W = E.DOUBLE_MM
    L, TOP = 9.0, 2.0
    mw = 66.0
    mh = mw * (box[3] - box[2]) / (box[1] - box[0])
    H = TOP + mh + 27.0
    fig = plt.figure(figsize=(W * C.MM, H * C.MM))
    ax = E.mm_axes(fig, W, H, L, TOP, mw, mh)
    ax.set_facecolor("#f4f4f4")
    ax.imshow(cls(z["obs"]), extent=ext, origin="upper", cmap=ListedColormap([UNKNOWN, LAND, SEA]), vmin=0, vmax=2,
              interpolation="nearest", zorder=1, rasterized=True)
    E.laea_graticule(ax, fwd, box, lons=np.arange(-113.5, -111.5, E.GRATICULE_ZOOM_DEG),
                     lats=np.arange(40.5, 42.0, E.GRATICULE_ZOOM_DEG))
    for g in arms.values():
        for part in getattr(g, "geoms", [g]):
            gx, gy = part.exterior.xy
            ax.plot(gx, gy, color=E.MUTED, lw=0.5, zorder=4)
    contour(ax, z["pred_dep3"], E.CONTROL, E.CONTROL_DASH, zorder=7)
    for p in MAP_PRODUCTS:
        contour(ax, z[f"pred_{p}"], E.COLOURS[p], E.DASHES[p])
    contour(ax, z["obs"], E.OBS, "solid", zorder=9)
    E.map_axes_clean(ax)
    ax.set_xlim(box[0], box[1])
    ax.set_ylim(box[2], box[3])
    halo = [patheffects.withStroke(linewidth=2.2, foreground="white")]
    for arm, name in ARMS:
        c = arms[arm].representative_point()
        ax.text(c.x, c.y, name, ha="center", va="center", fontsize=E.FS, fontstyle="italic", color=E.INK, zorder=20,
                path_effects=halo)
    E.scale_bar(ax, box[1] - 4000 - E.SCALE_GSL_KM * 1000.0, box[2] + 6000, E.SCALE_GSL_KM * 1000.0,
                E.num(m, "params.erl_fig_scale_gsl_km", "km"), box=True)
    E.label(ax, "a", inside=True)
    ax.text(0.015, 0.015, f"Lake level {E.num(m, 'erl.gsl.level_navd88_m')} m (NAVD 88)", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=E.FS, color=E.INK, zorder=20, linespacing=1.15,
            bbox={"boxstyle": "square,pad=0.25", "fc": "white", "ec": "none", "alpha": 0.9})
    lines = [Line2D([], [], color=E.OBS, lw=LW, label="Observed")]
    lines += [Line2D([], [], color=E.COLOURS[p], lw=LW, ls=E.DASHES[p], label=E.NAMES[p]) for p in MAP_PRODUCTS]
    lines += [Line2D([], [], color=E.CONTROL, lw=LW, ls=E.CONTROL_DASH, label="3DEP lidar (control)")]
    states = [Patch(fc=SEA, ec=E.MUTED, lw=0.4, label="Open water"), Patch(fc=LAND, ec=E.MUTED, lw=0.4, label="Land"),
              Patch(fc=UNKNOWN, ec=E.MUTED, lw=0.4, label="Not determined")]
    kw = {"fontsize": E.FS, "frameon": False, "borderaxespad": 0, "handletextpad": 0.5, "labelspacing": 0.3,
          "columnspacing": 1.0}
    fig.legend(handles=lines + states, loc="upper left", bbox_to_anchor=(L / W, 1 - (TOP + mh + 6.5) / H), ncol=2,
               handlelength=2.6, **kw)

    # (b) dot chart --------------------------------------------------------------------------------
    cx0 = L + mw + 22.0
    cw = W - cx0 - 2.0
    ch = (mh - 8.0) / 2
    g = "generality.great_salt_lake"
    names = {**E.NAMES, "dep3": "3DEP lidar (control)"}
    for i, (arm, name) in enumerate(ARMS):
        axc = E.mm_axes(fig, W, H, cx0, TOP + 5.0 + i * (ch + 5.0), cw, ch - 2.0)
        rows = list(CHART)[::-1]
        for j, p in enumerate(rows):
            if p == "dep3":
                med = E.value(m, f"{g}.control.{arm}.median_displacement_m") / 1000
                floor = E.value(m, f"{g}.control.{arm}.median_noise_p95_m") / 1000
                mis = m[f"{g}.control.{arm}.mispredicts"]["value"] == "true"
                col, mk = E.CONTROL, "o"
            else:
                med = E.value(m, f"{g}.{p}.shape.{arm}.median_displacement_m") / 1000
                floor = E.value(m, f"{g}.{p}.shape.{arm}.median_noise_p95_m") / 1000
                mis = m[f"{g}.{p}.shape.{arm}.mispredicts"]["value"] == "true"
                col, mk = E.COLOURS[p], E.MARKERS[p]
            axc.plot([0.01, floor], [j, j], color="#d9d9d9", lw=5.0, solid_capstyle="butt", zorder=1)
            axc.plot([floor, floor], [j - 0.32, j + 0.32], color=E.MUTED, lw=0.8, zorder=2)
            axc.plot([floor, med], [j, j], color=col, lw=0.8, zorder=2)
            axc.plot(med, j, mk, ms=5.5, mfc=col if mis else "white", mec=col, mew=1.0, zorder=3)
        axc.set_yticks(range(len(rows)), [names[p] for p in rows])
        axc.tick_params(axis="y", length=0)
        axc.set_xscale("log")
        axc.set_xlim(0.03, 5.0)
        axc.set_ylim(-0.6, len(rows) - 0.4)
        axc.set_xticks([0.05, 0.1, 0.2, 0.5, 1, 2, 5], ["0.05", "0.1", "0.2", "0.5", "1", "2", "5"])
        axc.xaxis.set_minor_locator(plt.NullLocator())
        axc.grid(axis="x", color=E.GRID, lw=0.4)
        axc.set_axisbelow(True)
        axc.text(0.0, 1.02, name, transform=axc.transAxes, ha="left", va="bottom", fontsize=E.FS, color=E.INK)
        if i == 0:
            fig.text((L + mw + 3.0) / W, 1 - (TOP + 0.5) / H, "(b)", fontsize=E.FS_LABEL, fontweight="bold", ha="left",
                     va="top", color=E.INK)
            axc.tick_params(labelbottom=False)
        else:
            axc.set_xlabel("Median shoreline displacement (km)")
    fig.legend(handles=[Patch(fc="#d9d9d9", ec="none", label="Below the noise floor"),
                        Line2D([], [], ls="none", marker="o", ms=5.5, mfc=E.INK, mec=E.INK, label="Mispredicts the arm"),
                        Line2D([], [], ls="none", marker="o", ms=5.5, mfc="white", mec=E.INK, mew=1.0,
                               label="Does not")],
               loc="upper left", bbox_to_anchor=((cx0 - 4.0) / W, 1 - (TOP + mh + 9.5) / H), ncol=1, handlelength=1.4,
               **kw)
    C.save(fig, "erl_fig_gsl")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
