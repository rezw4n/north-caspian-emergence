"""RSE figure: the same audit at Great Salt Lake, with a sounding-based lake-bed control
(double column).

a, b  Per arm (north, south), at every hindcast level bin: the shoreline displacement D of
      each surface after its area-fitted vertical offset (shape variant; median distance
      between predicted and observed shoreline cells, both directions pooled) and the
      observation noise floor Q (95th percentile of the distances between the shorelines of
      two interleaved halves of the acquisitions; grey area). Surfaces: the five global grids,
      the USGS 3DEP control (mostly 2016 lidar) and the lake-bed control built from USGS
      single-beam echo soundings (2002-04 south, 2006 north) and digitised map contours
      (PREREG Amendment 19 item 13, D-150). All values: results/revision2/g/
      gsl_lakebed_levels.csv. The lake-bed control's marker is filled where D > Q at the
      level. Shaded columns: the level bins whose lake-bed heights come from soundings
      (source class "sounded"). D is drawn only at evaluable levels; a grid with no predicted
      shoreline at the level (D infinite) is drawn on the top edge.
c     Per arm, the median over evaluable levels of each surface's D, against the median Q
      over the same levels (grey bar), as figure 4b of the ERL version, with the lake-bed
      control over all its levels and over the sounded levels added. Filled: the surface
      fails the shoreline test in the arm (D > Q at a majority of the evaluable levels).
      Grids and 3DEP: results/metrics.json generality.great_salt_lake.*; lake-bed control:
      rev2.g.lakebed.*.
d     Per arm, the heights of the arm's emerged band in each grid (share of the band's 250 m
      cells per 0.5 m bin; results/revision/g/gsl_band_height_histograms.csv); dotted line,
      the gauge level of February 2000 (rev.g.gsl.<arm>.gauge_feb2000_navd88_m).

Output: figures/out/rse_fig_gsl.{pdf,tif,png}
Usage: python figures/rse_fig_gsl.py
"""

from __future__ import annotations

import sys

import _common as C
import _rse as R
import numpy as np
import pandas as pd

from caspian.config import RESULTS

LEV = RESULTS / "revision2" / "g" / "gsl_lakebed_levels.csv"
HIST = RESULTS / "revision" / "g" / "gsl_band_height_histograms.csv"
ARMS = (("north", "North arm"), ("south", "South arm"))
SOUNDED_BINS = {"north": (76, 91), "south": (76, 89)}
CTRL3, LAKE = "#7a7a7a", "#000000"
YMIN, YMAX = 20.0, 60_000.0
HIST_RANGE = (1276.0, 1296.0)
SOUND_C = "#e9f1f7"


def main() -> int:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    m = C.metrics()
    lev = pd.read_csv(LEV)
    for arm, (b0, b1) in SOUNDED_BINS.items():
        sb = sorted(lev[(lev["arm"] == arm) & (lev["source_class"] == "sounded")]["bin"])
        if sb != list(range(b0, b1 + 1)):
            raise SystemExit(f"{arm}: sounded bins {sb[0]}-{sb[-1]} differ from {b0}-{b1}")
        n_ev = int(lev[(lev["arm"] == arm) & lev["lakebed_shape_evaluable"]].shape[0])
        if n_ev != int(R.value(m, f"rev2.g.lakebed.all.{arm}.shape.n_evaluable")):
            raise SystemExit(f"{arm}: {n_ev} evaluable lake-bed levels, metrics differ")
    hist = pd.read_csv(HIST)

    R.style()
    W = R.DOUBLE_MM
    L, RGT, TOP = 14.0, 2.0, 9.0
    HA, GAP, HC, BOT = 52.0, 16.0, 38.0, 10.0
    H = TOP + HA + GAP + HC + BOT
    fig = plt.figure(figsize=(W * C.MM, H * C.MM))
    gap = 6.0
    aw = (W - L - RGT - gap) / 2

    # a, b: per-level D and Q ----------------------------------------------------------------------
    for i, (arm, name) in enumerate(ARMS):
        ax = R.mm_axes(fig, W, H, L + i * (aw + gap), TOP, aw, HA)
        d = lev[lev["arm"] == arm].sort_values("level_navd88_m")
        x = d["level_navd88_m"].to_numpy(float)
        sb = d[d["source_class"] == "sounded"]["level_navd88_m"]
        ax.axvspan(sb.min() - 0.05, sb.max() + 0.05, color=SOUND_C, lw=0, zorder=0)
        ax.text(sb.min() - 0.03, YMAX * 0.7, "sounded", ha="left", va="top", fontsize=R.FS_SMALL, color="#2f5d80")
        q = np.where(d["floor_p95_m"].to_numpy(float) > 0, d["floor_p95_m"].to_numpy(float), np.nan)
        ax.fill_between(x, YMIN, np.where(np.isfinite(q), q, YMIN), step="mid", color="#d9d9d9", lw=0, zorder=1)
        for p in R.PRODUCTS:
            v = d[f"{p}_median_m"].to_numpy(float)
            fin = np.isfinite(v)
            ax.plot(x, np.where(fin, v, np.nan), color=R.COLOURS[p], lw=0.6, zorder=2)
            ax.plot(x[fin], v[fin], R.MARKERS[p], ms=2.2, mfc=R.COLOURS[p], mec="none", zorder=3)
            inf = np.isinf(v)
            ax.plot(x[inf], np.full(inf.sum(), YMAX), R.MARKERS[p], ms=2.6, mfc="white", mec=R.COLOURS[p], mew=0.6,
                    zorder=3, clip_on=False)
        v3 = np.where(d["dep3_evaluable"], d["dep3_median_m"], np.nan).astype(float)
        ax.plot(x, v3, color=CTRL3, lw=0.9, ls=(0, (3, 1.5)), marker="o", ms=2.2, mfc=CTRL3, mec="none", zorder=4)
        ev = d["lakebed_shape_evaluable"].to_numpy(bool)
        vl = np.where(ev, d["lakebed_shape_median_m"], np.nan).astype(float)
        ex = d["lakebed_shape_exceeds"].to_numpy(bool) & ev
        ax.plot(x, vl, color=LAKE, lw=1.0, zorder=5)
        ax.plot(x[ex], vl[ex], "o", ms=3.0, mfc=LAKE, mec=LAKE, zorder=6)
        ax.plot(x[ev & ~ex], vl[ev & ~ex], "o", ms=3.0, mfc="white", mec=LAKE, mew=0.7, zorder=6)
        ax.set_yscale("log")
        ax.set_ylim(YMIN, YMAX)
        ax.set_yticks([30, 100, 300, 1000, 3000, 10_000, 30_000], ["30", "100", "300", "1,000", "3,000", "10,000",
                                                                    "30,000"])
        ax.yaxis.set_minor_locator(plt.NullLocator())
        ax.set_xlim(x.min() - 0.1, x.max() + 0.1)
        ax.set_xlabel("Level (m, NAVD 88)")
        ax.grid(axis="y", color=R.GRID, lw=0.4)
        ax.set_axisbelow(True)
        ax.set_title(name, fontsize=R.FS, loc="right", pad=2)
        if i == 0:
            ax.set_ylabel("Shoreline displacement D (m)")
        else:
            ax.tick_params(labelleft=False)
        R.label(ax, "ab"[i], x=-0.17 if i == 0 else -0.04, y=1.03)
    handles = [Line2D([], [], color=R.COLOURS[p], lw=0.6, marker=R.MARKERS[p], ms=2.6, mfc=R.COLOURS[p], mec="none",
                      label=R.NAMES[p]) for p in R.PRODUCTS]
    handles += [Line2D([], [], color=CTRL3, lw=0.9, ls=(0, (3, 1.5)), marker="o", ms=2.4, mec="none",
                       label="3DEP control"),
                Line2D([], [], color=LAKE, lw=1.0, marker="o", ms=3.0, label="Lake-bed control (D > Q filled)"),
                Patch(fc="#d9d9d9", ec="none", label="Noise floor Q"),
                Patch(fc=SOUND_C, ec="none", label="Sounded levels")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(L / W, 1 - 0.8 / H), ncol=5, frameon=False,
               borderaxespad=0, fontsize=R.FS_SMALL, handlelength=1.8, columnspacing=1.0, labelspacing=0.25,
               handletextpad=0.4)

    # c: medians against floors; d: band heights ----------------------------------------------------
    y2 = TOP + HA + GAP
    g = "generality.great_salt_lake"
    rows = [*R.PRODUCTS, "dep3", "lake_all", "lake_snd"]
    names = {**R.NAMES, "dep3": "3DEP control", "lake_all": "Lake-bed, all levels",
             "lake_snd": "Lake-bed, sounded levels"}
    cw = 33.0
    for i, (arm, name) in enumerate(ARMS):
        axc = R.mm_axes(fig, W, H, L + 22.0 + i * (cw + 3.0), y2, cw, HC)
        for j, p in enumerate(rows[::-1]):
            if p in R.PRODUCTS:
                k = f"{g}.{p}.shape.{arm}"
                med, fl = R.value(m, f"{k}.median_displacement_m"), R.value(m, f"{k}.median_noise_p95_m")
                mis = m[f"{k}.mispredicts"]["value"] == "true"
                col, mk = R.COLOURS[p], R.MARKERS[p]
            elif p == "dep3":
                k = f"{g}.control.{arm}"
                med, fl = R.value(m, f"{k}.median_displacement_m"), R.value(m, f"{k}.median_noise_p95_m")
                mis = m[f"{k}.mispredicts"]["value"] == "true"
                col, mk = CTRL3, "o"
            else:
                k = f"rev2.g.lakebed.{'all' if p == 'lake_all' else 'sounded'}.{arm}.shape"
                med, fl = R.value(m, f"{k}.median_displacement_m"), R.value(m, f"{k}.median_floor_p95_m")
                mis = m[f"{k}.mispredicts"]["value"] == "yes"
                col, mk = LAKE, "o" if p == "lake_all" else "s"
            axc.plot([YMIN, fl], [j, j], color="#d9d9d9", lw=4.0, solid_capstyle="butt", zorder=1)
            axc.plot([fl, fl], [j - 0.3, j + 0.3], color=R.MUTED, lw=0.7, zorder=2)
            axc.plot(med, j, mk, ms=4.0, mfc=col if mis else "white", mec=col, mew=0.9, zorder=3)
        axc.set_xscale("log")
        axc.set_xlim(YMIN, 5000)
        axc.set_xticks([30, 100, 300, 1000, 3000], ["30", "100", "300", "1k", "3k"])
        axc.xaxis.set_minor_locator(plt.NullLocator())
        axc.set_ylim(-0.6, len(rows) - 0.4)
        axc.set_yticks(range(len(rows)), [names[p] for p in rows[::-1]] if i == 0 else [""] * len(rows),
                       fontsize=R.FS_SMALL)
        axc.tick_params(axis="y", length=0)
        axc.grid(axis="x", color=R.GRID, lw=0.4)
        axc.set_axisbelow(True)
        axc.set_title(name, fontsize=R.FS_SMALL, loc="left", pad=2)
        axc.set_xlabel("Median D (m)", fontsize=R.FS_SMALL)
        axc.tick_params(axis="x", labelsize=R.FS_SMALL)
        if i == 0:
            R.fig_label(fig, W, H, "c", 1.0, y2 - 4.5)
    hx0 = L + 22.0 + 2 * (cw + 3.0) + 14.0
    hw = (W - hx0 - RGT - 3.0) / 2
    edges = np.arange(HIST_RANGE[0], HIST_RANGE[1] + 0.25, 0.5)
    for i, (arm, name) in enumerate(ARMS):
        axh = R.mm_axes(fig, W, H, hx0 + i * (hw + 3.0), y2, hw, HC)
        for p in R.PRODUCTS:
            h = hist[(hist["arm"] == arm) & (hist["product"] == p)]
            share = np.zeros(edges.size - 1)
            k = np.round((h["bin_low_navd88_m"].to_numpy(float) - HIST_RANGE[0]) / 0.5).astype(int)
            ok = (k >= 0) & (k < share.size)
            share[k[ok]] = 100.0 * h["share"].to_numpy(float)[ok]
            axh.stairs(share, edges, color=R.COLOURS[p], lw=0.8, ls=R.DASHES[p], zorder=3)
        gauge = R.value(m, f"rev.g.gsl.{arm}.gauge_feb2000_navd88_m")
        axh.axvline(gauge, color=R.INK, lw=0.7, ls=(0, (1.0, 1.5)), zorder=2)
        axh.set_xlim(*HIST_RANGE)
        axh.set_xticks([1280, 1288])
        axh.set_ylim(0, 70)
        axh.tick_params(labelsize=R.FS_SMALL)
        axh.grid(axis="y", color=R.GRID, lw=0.4)
        axh.set_axisbelow(True)
        axh.set_title(name, fontsize=R.FS_SMALL, loc="left", pad=2)
        axh.set_xlabel("Grid height (m, NAVD 88)", fontsize=R.FS_SMALL)
        if i == 0:
            axh.set_ylabel("Share of band (%)", fontsize=R.FS_SMALL)
            R.fig_label(fig, W, H, "d", hx0 - 11.0, y2 - 4.5)
            axh.text(gauge - 0.3, 66, "gauge,\nFeb. 2000", fontsize=R.FS_SMALL, ha="right", va="top", linespacing=1.0)
        else:
            axh.tick_params(labelleft=False)
    R.save(fig, "rse_fig_gsl", W, H)
    plt.close(fig)
    for arm, _ in ARMS:
        d = lev[(lev["arm"] == arm) & (lev["source_class"] == "sounded") & lev["lakebed_shape_evaluable"]]
        print(arm, "sounded evaluable", len(d), "exceeding", int(d["lakebed_shape_exceeds"].sum()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
