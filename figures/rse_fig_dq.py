"""RSE figure: grid error against observation error, per sector and grid (double column).

For every sector (the four validated sectors in a, the three not validated in b) and every
grid, the per-level ratio D/Q at each evaluable level (results/hindcast/main/
h1_mispredicts_detail.csv, shape variant, evaluable == True): D is the grid's median
shoreline displacement at the level (the median distance between predicted and observed
shoreline cells, both directions pooled), Q the observation noise floor at the level (the
95th percentile of the distances between the shorelines of the two interleaved halves of
the acquisitions). Dots: one level; box: 25th-75th percentile over levels; line in the box:
the median, which the script checks against results/revision/s/item2_dq.csv. Marker at the
median filled where the grid fails the shoreline test in the sector (D > Q at a majority of
the evaluable levels; results/hindcast/main/h1_mispredicts.csv), open where it does not.

Grey, right of each grid: the degraded-DEM control of that grid (PREREG Amendment 18 item
6; results/revision/r/control/levels.csv, the rounded control, at the levels in the DEM's
contour range at which both the grid and the control are evaluable, as the excess rule
counts them): the waterline DEM put into the grid's vertical
frame, rounded to whole metres where the grid stores whole metres, averaged to the grid's
native cell and audited as the grid is; its D/Q at each level is D_control / Q.

Ratios below the axis (zero displacement) are drawn on it. The shaded band below
D/Q = 1 marks displacements within the noise floor.

Output: figures/out/rse_fig_dq.{pdf,tif,png}
Usage: python figures/rse_fig_dq.py
"""

from __future__ import annotations

import sys

import _common as C
import _rse as R
import numpy as np
import pandas as pd

from caspian.config import RESULTS

HC = RESULTS / "hindcast" / "main"
DQ = RESULTS / "revision" / "s" / "item2_dq.csv"
CTRL = RESULTS / "revision" / "r" / "control" / "levels.csv"
YMIN, YMAX = 0.01, 100.0
FLOOR = "#ececec"
CTRL_C = "#7a7a7a"
SEED = 20261023


def main() -> int:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle

    m = C.metrics()
    det = pd.read_csv(HC / "h1_mispredicts_detail.csv")
    det = det[(det["variant"] == "shape") & (det["evaluable"] == True)].copy()  # noqa: E712
    det["dq"] = det["median"] / det["p95_noise"]
    ref = pd.read_csv(DQ)
    ref = ref[ref["variant"] == "shape"].set_index(["product", "sector"])
    for (p, s), g in det.groupby(["product", "sector"]):
        r = ref.loc[(p, s)]
        if abs(float(g["dq"].median()) - float(r["dq_median"])) > 1e-9 or len(g) != int(r["dq_n"]):
            raise SystemExit(f"{p} {s}: per-level D/Q differs from item2_dq.csv")
    mis = pd.read_csv(HC / "h1_mispredicts.csv")
    mis = mis[mis["variant"] == "shape"].set_index(["product", "sector"])["mispredicts"]
    ctrl = pd.read_csv(CTRL)
    ctrl = ctrl[(ctrl["control_kind"] == "rounded") & ctrl["in_dem_range"] & ctrl["evaluable_control"]
                & ctrl["evaluable_product"]].copy()        # the levels of the excess rule (evaluable for both)
    ctrl["dq"] = ctrl["D_control"] / ctrl["Q"]
    # the control's median displacement per (grid, sector) must equal the metrics
    for (p, s), g in ctrl.groupby(["product", "sector"]):
        key = f"rev.r.control.rounded.{R.PK[p]}.{s.replace('open_shelf', 'openshelf')}.D_control_median_km"
        if abs(g["D_control"].median() / 1000.0 - R.value(m, key)) > 1e-6:
            raise SystemExit(f"{p} {s}: control median D differs from {key}")

    R.style()
    W = R.DOUBLE_MM
    L, RGT, TOP = 13.0, 2.0, 11.0
    HA, GAP, BOT = 52.0, 17.0, 11.0
    H = TOP + HA + GAP + HA + BOT
    fig = plt.figure(figsize=(W * C.MM, H * C.MM))
    rng = np.random.default_rng(SEED)
    width = W - L - RGT
    groups = (("a", R.VALIDATED, "Checked sectors (waterline DEM mean and median error within ±0.5 m of ICESat-2)"),
              ("b", R.NOT_VALIDATED, "Other sectors (observations biased against ICESat-2, or no test block)"))
    # the second row uses the same group width as the first, left-aligned
    gw = width / len(R.VALIDATED)
    offs = np.linspace(-0.36, 0.36, len(R.PRODUCTS))
    dx_c = 0.075
    for row, (letter, secs, title) in enumerate(groups):
        y_top = TOP + row * (HA + GAP)
        ax = R.mm_axes(fig, W, H, L, y_top, gw * len(secs), HA)
        ax.axhspan(YMIN, 1.0, color=FLOOR, lw=0, zorder=0)
        ax.axhline(1.0, color=R.MUTED, lw=0.7, zorder=1)
        for i, s in enumerate(secs):
            for p, dx in zip(R.PRODUCTS, offs):
                col = R.COLOURS[p]
                g = det[(det["product"] == p) & (det["sector"] == s)]
                v = np.maximum(g["dq"].to_numpy(float), YMIN)
                xp = i + dx - dx_c / 2
                jit = rng.uniform(-0.022, 0.022, v.size)
                ax.plot(xp + jit, v, "o", ms=1.3, mfc=col, mec="none", alpha=0.6, zorder=2, clip_on=False)
                q1, q2, q3 = np.percentile(g["dq"].to_numpy(float), [25, 50, 75])
                ax.add_patch(Rectangle((xp - 0.03, max(q1, YMIN)), 0.06, q3 - max(q1, YMIN), fc="none", ec=col,
                                       lw=0.8, zorder=3))
                filled = bool(mis.get((p, s), False))
                ax.plot(xp, q2, R.MARKERS[p], ms=4.2, mfc=col if filled else "white", mec=col, mew=0.8, zorder=4)
                c = ctrl[(ctrl["product"] == p) & (ctrl["sector"] == s)]
                if len(c):
                    vc = np.maximum(c["dq"].to_numpy(float), YMIN)
                    xc = i + dx + dx_c / 2
                    ax.plot(xc + rng.uniform(-0.012, 0.012, vc.size), vc, "o", ms=1.0, mfc=CTRL_C, mec="none",
                            alpha=0.55, zorder=2, clip_on=False)
                    c1, c2, c3 = np.percentile(c["dq"].to_numpy(float), [25, 50, 75])
                    ax.add_patch(Rectangle((xc - 0.018, max(c1, YMIN)), 0.036, c3 - max(c1, YMIN), fc="#d9d9d9",
                                           ec=CTRL_C, lw=0.5, zorder=3))
                    ax.plot([xc - 0.026, xc + 0.026], [c2, c2], color=R.INK, lw=0.9, zorder=4,
                            solid_capstyle="butt")
            if i:
                ax.axvline(i - 0.5, color=R.GRID, lw=0.6, zorder=0)
        ax.set_yscale("log")
        ax.set_ylim(YMIN, YMAX)
        ax.set_yticks([0.01, 0.1, 1, 10, 100], ["0.01", "0.1", "1", "10", "100"])
        ax.yaxis.set_minor_locator(plt.NullLocator())
        ax.set_xlim(-0.5, len(secs) - 0.5)
        ax.set_xticks(range(len(secs)), [R.NAME[s] for s in secs])
        ax.tick_params(axis="x", length=0, pad=3)
        ax.set_ylabel("D/Q (displacement / noise floor)")
        ax.grid(axis="y", color=R.GRID, lw=0.4)
        ax.set_axisbelow(True)
        ax.text(0.0, 1.03, title, transform=ax.transAxes, ha="left", va="bottom", fontsize=R.FS, color=R.INK)
        R.label(ax, letter, x=-0.075 * (len(R.VALIDATED) / len(secs)), y=1.03)

    # key, right of b ------------------------------------------------------------------------------
    handles = [Line2D([], [], ls="none", marker=R.MARKERS[p], ms=4.2, mfc=R.COLOURS[p], mec=R.COLOURS[p],
                      label=R.NAMES[p]) for p in R.PRODUCTS]
    kw = {"frameon": False, "borderaxespad": 0, "fontsize": R.FS, "handletextpad": 0.4, "labelspacing": 0.4,
          "alignment": "left"}
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(L / W, 1 - 1.0 / H), ncol=5, columnspacing=1.4,
               handlelength=1.0, **kw)
    kx = (L + gw * len(R.NOT_VALIDATED) + 5.0) / W
    ky = 1 - (TOP + HA + GAP + 1.0) / H
    sym = [Line2D([], [], ls="none", marker="o", ms=1.6, mfc=R.MUTED, mec="none", label="One level"),
           Patch(fc="none", ec=R.INK, lw=0.8, label="Box: 25th–75th percentile"),
           Line2D([], [], ls="none", marker="o", ms=4.2, mfc=R.INK, mec=R.INK, label="Median: grid fails the test"),
           Line2D([], [], ls="none", marker="o", ms=4.2, mfc="white", mec=R.INK, mew=0.8,
                  label="Median: grid passes the test"),
           Patch(fc="#d9d9d9", ec=CTRL_C, lw=0.5, label="Degraded-DEM control\n(grey, right of each grid)"),
           Patch(fc=FLOOR, ec="none", label="D/Q < 1: within the\nnoise floor")]
    fig.legend(handles=sym, loc="upper left", bbox_to_anchor=(kx, ky), handlelength=1.4, **kw)
    R.save(fig, "rse_fig_dq", W, H)
    plt.close(fig)
    n = det.groupby(["product", "sector"]).size()
    nc = ctrl.groupby(["product", "sector"]).size()
    print("levels per grid and sector:", int(n.min()), "-", int(n.max()), "| control levels:", int(nc.min()), "-",
          int(nc.max()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
