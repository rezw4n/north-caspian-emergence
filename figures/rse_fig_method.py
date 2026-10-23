"""RSE figure: the hindcast audit, step by step, on real data (double column).

One example: the Volga-Ural interfluve coast, GEBCO_2026, at the late-fall (E3) level bin
chosen by the rule of scripts/f5_rse_display.py (the level of rse_fig_shorelines).

a, Left: the daily basin level from altimetry (DAHITI, Hydroweb in DAHITI's gaps;
   data/processed/dem/main/emergence/level_series_daily.csv) and the Landsat acquisitions
   whose level falls in the example bin (0.1 m wide, shaded), split into the two
   interleaved halves by acquisition parity (data/interim/stage2/plan_main_tau9900cm_
   landsat.csv). Right: the water frequency of those acquisitions per 60 m cell (share
   of the valid acquisitions classified water; blank where fewer than 5 are valid), and
   the observed shoreline (black): the edge of the cells with frequency of at least 0.5
   that connect to the open sea.
b, GEBCO_2026 height relative to the example level after the grid's area-fitted offset
   (product height plus the datum correction minus c), the sea flooded from the open
   sea at that level (minimax flood fill; outlined, its edge the predicted shoreline),
   and the cells below the level that the flood does not reach (hatched).
c, Left: the predicted shoreline cells coloured by their distance to the nearest observed
   shoreline cell, the observed shoreline (black) and the shorelines of the two halves
   (grey). Right: cumulative distributions, over the whole sector at this level, of the
   distances between the predicted and observed shorelines (both directions pooled; D is
   their median) and between the shorelines of the two halves (Q, the noise floor, is their
   95th percentile). Distances of zero are drawn at the axis minimum.
d, D/Q at every evaluable level of the sector for GEBCO_2026 (results/hindcast/main/
   h1_mispredicts_detail.csv, shape variant); the rule: the grid fails the shoreline test in
   the sector when D > Q at a majority of the evaluable levels (h1_mispredicts.csv).

Data: data/interim/figures/rse/shorelines_main.npz (scripts/f5_rse_display.py, which checks
D and Q of the example against displacement.csv and noise_floor.csv), results/hindcast/main/.

Output: figures/out/rse_fig_method.{pdf,tif,png}
Usage: python figures/rse_fig_method.py
"""

from __future__ import annotations

import json
import sys

import _common as C
import _rse as R
import numpy as np
import pandas as pd

from caspian.config import DATA, INTERIM, RESULTS

CACHE = INTERIM / "figures" / "rse"
SECTOR, PRODUCT = "interfluve", "gebco_2026"
BOX = (22_000.0, 136_000.0, 62_000.0, 170_000.0)        # x0, y0, x1, y1 (equal-area metres)
D_EDGES = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 1e9]
D_LABELS = ["<0.5", "0.5–1", "1–2", "2–5", "5–10", ">10"]
HALF_A, HALF_B = "#1f1f1f", "#9a9a9a"


def coords(ext, shape):
    ny, nx = shape
    xs = ext[0] + (np.arange(nx) + 0.5) * (ext[1] - ext[0]) / nx
    ys = ext[3] - (np.arange(ny) + 0.5) * (ext[3] - ext[2]) / ny
    return xs, ys


def crop(arr, ext, box):
    xs, ys = coords(ext, arr.shape)
    ci = np.flatnonzero((xs >= box[0]) & (xs < box[2]))
    ri = np.flatnonzero((ys >= box[1]) & (ys < box[3]))
    sub = arr[ri[0]:ri[-1] + 1, ci[0]:ci[-1] + 1]
    a = (ext[1] - ext[0]) / arr.shape[1]
    return sub, [xs[ci[0]] - a / 2, xs[ci[-1]] + a / 2, ys[ri[-1]] - a / 2, ys[ri[0]] + a / 2]


def contour(ax, field, ext, colour, lw, ls="solid", zorder=10):
    f = np.asarray(field, float).copy()
    f[f < 0] = np.nan
    xs, ys = coords(ext, f.shape)
    cs = ax.contour(xs, ys, f, [0.5], colors=[colour], linewidths=lw, zorder=zorder)
    cs.set_linestyle(ls)
    return cs


def ecdf(v):
    v = np.sort(np.asarray(v, float))
    return v, np.arange(1, v.size + 1) / v.size


def main() -> int:
    import cmocean
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    m = C.metrics()
    rec = json.loads((RESULTS / "display" / "rse_shorelines.json").read_text(encoding="utf-8"))
    if not all(c.get("equal", False) for c in rec["checks"]):
        raise SystemExit("rse_shorelines.json holds a failed check")
    z = np.load(CACHE / "shorelines_main.npz")
    pick = rec["rule_units"][SECTOR]
    ep, b = pick["epoch"], pick["bin"]
    u = f"{ep}_b{b:03d}"
    unit = next(r for r in rec["units"] if r["epoch"] == ep and r["bin"] == b)
    level = float(unit["level_m"])
    c_off = float(rec["shape_offsets_m"][PRODUCT])
    disp = pd.read_csv(RESULTS / "hindcast" / "main" / "displacement.csv")
    D_m = float(disp[(disp["product"] == PRODUCT) & (disp["variant"] == "shape") & (disp["epoch"] == ep)
                     & (disp["bin"] == b) & (disp["sector"] == SECTOR)]["median_m"].iloc[0])
    nf = pd.read_csv(RESULTS / "hindcast" / "main" / "noise_floor.csv")
    Q_m = float(nf[(nf["kind"] == "pooled_level") & (nf["bin"] == b) & (nf["sector"] == SECTOR)]["p95_m"].iloc[0])
    dist = np.abs(z[f"dist_{u}_{PRODUCT}_{SECTOR}"].astype(float))
    hdist = np.abs(z[f"halfdist_{u}_{SECTOR}"].astype(float))
    if abs(np.median(dist) - D_m) > 1e-3 or abs(np.percentile(hdist, 95) - Q_m) > 1e-3:
        raise SystemExit("cached distances do not give D and Q of the results tables")
    det = pd.read_csv(RESULTS / "hindcast" / "main" / "h1_mispredicts_detail.csv")
    det = det[(det["product"] == PRODUCT) & (det["variant"] == "shape") & (det["sector"] == SECTOR)
              & (det["evaluable"] == True)].copy()  # noqa: E712
    det["dq"] = det["median"] / det["p95_noise"]
    mis = pd.read_csv(RESULTS / "hindcast" / "main" / "h1_mispredicts.csv")
    mrow = mis[(mis["product"] == PRODUCT) & (mis["variant"] == "shape") & (mis["sector"] == SECTOR)].iloc[0]
    if int(mrow["n_exceeding"]) != int((det["dq"] > 1).sum()) or int(mrow["n_evaluable"]) != len(det):
        raise SystemExit("the per-level D/Q do not reproduce h1_mispredicts.csv")
    lv = pd.read_csv(DATA / "processed" / "dem" / "main" / "emergence" / "level_series_daily.csv", parse_dates=["date"])
    t_acq = pd.to_datetime(z[f"acq_time_{u}"], unit="s")
    l_acq = z[f"acq_level_{u}"]
    par = z[f"acq_parity_{u}"]
    ext = z[f"ext_{SECTOR}"]

    R.style()
    W = R.DOUBLE_MM
    TOP, TT, ROW, GAP, BOT = 1.0, 6.0, 44.0, 9.0, 3.0
    H = TOP + 2 * (TT + ROW) + GAP + BOT
    fig = plt.figure(figsize=(W * C.MM, H * C.MM))
    bw, bh = 40.0, 40.0 * (BOX[3] - BOX[1]) / (BOX[2] - BOX[0])
    halo_kw = {"boxstyle": "square,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.85}

    def title(x, y, letter, text):
        R.fig_label(fig, W, H, letter, x, y)
        fig.text((x + 4.0) / W, 1 - y / H, text, ha="left", va="top", fontsize=R.FS, color=R.INK)

    def map_axes(x, y):
        a_ = R.mm_axes(fig, W, H, x, y, bw, bh)
        R.map_axes_clean(a_)
        a_.set_xlim(BOX[0], BOX[2])
        a_.set_ylim(BOX[1], BOX[3])
        return a_

    y1 = TOP + TT
    # a, levels and observed sea ----------------------------------------------------------------------
    title(1.0, TOP + 0.5, "a", "Landsat acquisitions levelled by altimetry give the observed sea at one level")
    axl = R.mm_axes(fig, W, H, 12.0, y1 + 1.0, 36.0, bh - 1.0)
    t0, t1 = t_acq.min() - pd.Timedelta(days=200), t_acq.max() + pd.Timedelta(days=200)
    s = lv[(lv["date"] >= t0) & (lv["date"] <= t1)]
    axl.plot(s["date"], s["level_m"], color=R.MUTED, lw=0.6, zorder=2)
    axl.axhspan(level - 0.05, level + 0.05, color="#dbe7f1", lw=0, zorder=1)
    for pv, mfc in ((0, HALF_A), (1, "white")):
        k = par == pv
        axl.plot(t_acq[k], l_acq[k], "o", ms=2.0, mfc=mfc, mec=HALF_A, mew=0.5, zorder=3)
    axl.set_xlim(t0, t1)
    axl.set_ylim(level - 0.8, level + 0.8)
    axl.xaxis.set_major_locator(mdates.YearLocator(2))
    axl.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axl.set_ylabel("Basin level (m, DAHITI)")
    axl.set_xlabel("Year")
    axl.legend(handles=[Line2D([], [], ls="none", marker="o", ms=2.6, mfc=HALF_A, mec=HALF_A, label="half A"),
                        Line2D([], [], ls="none", marker="o", ms=2.6, mfc="white", mec=HALF_A, label="half B"),
                        Patch(fc="#dbe7f1", ec="none", label=f"bin {R.fmt_num(level - 0.05, 2)} to "
                                                              f"{R.fmt_num(level + 0.05, 2)} m")],
               loc="upper right", fontsize=R.FS_SMALL, handlelength=1.0, borderaxespad=0.2, labelspacing=0.2,
               frameon=True, framealpha=0.9, edgecolor="none")
    axf = map_axes(52.0, y1)
    fq, fext = crop(z[f"freq_{u}_{SECTOR}"], z[f"freqext_{SECTOR}"], BOX)
    imf = axf.imshow(np.ma.masked_invalid(fq), extent=fext, origin="upper", cmap=ListedColormap(cmocean.cm.ice_r(np.linspace(0.0, 0.72, 256))), vmin=0, vmax=1,
                     interpolation="nearest", zorder=1, rasterized=True)
    obs, oext = crop(z[f"obs_{u}_{SECTOR}"], ext, BOX)
    contour(axf, obs, oext, R.OBS, 0.7, zorder=8)
    axf.text(0.03, 0.03, f"{unit['n_acquisitions']} acquisitions,\n{unit['first_year']}–{unit['last_year']}",
             transform=axf.transAxes, ha="left", va="bottom", fontsize=R.FS_SMALL, bbox=halo_kw, zorder=20)
    R.scale_bar(axf, BOX[2] - 13_000, BOX[3] - 5_000, 10_000.0, "10 km", box=True)
    cax = R.mm_axes(fig, W, H, 52.0 + bw + 1.5, y1, 1.5, bh)
    cb = fig.colorbar(imf, cax=cax)
    cb.set_label("Water frequency", fontsize=R.FS_SMALL, labelpad=1.0)
    cb.ax.tick_params(labelsize=R.FS_SMALL, length=2, width=0.4)
    cb.outline.set_linewidth(0.4)

    # b, flood fill ---------------------------------------------------------------------------------
    title(108.0, TOP + 0.5, "b", "A grid flooded from the open sea at the same level")
    axh = map_axes(112.0, y1)
    zz, zext = crop(z[f"z_{PRODUCT}_{SECTOR}"], ext, BOX)
    ff, _ = crop(z[f"F_{PRODUCT}_{SECTOR}"], ext, BOX)
    rel = zz - level
    imh = axh.imshow(np.ma.masked_invalid(rel), extent=zext, origin="upper", cmap=cmocean.cm.diff,
                     norm=Normalize(-2.0, 2.0), interpolation="nearest", zorder=1, rasterized=True)
    flooded = (ff < level).astype(float)
    pocket = ((zz < level) & ~(ff < level)).astype(float)
    xs, ys = coords(zext, zz.shape)
    if pocket.any():
        axh.contourf(xs, ys, pocket, levels=[0.5, 1.5], colors="none", hatches=["//////"], zorder=5)
    axh.contour(xs, ys, flooded, [0.5], colors=[R.INK], linewidths=0.9, zorder=8)
    axh.text(0.03, 0.03, f"GEBCO_2026, offset c = {R.fmt_num(c_off, 2)} m", transform=axh.transAxes, ha="left",
             va="bottom", fontsize=R.FS_SMALL, bbox=halo_kw, zorder=20)
    R.scale_bar(axh, BOX[2] - 13_000, BOX[3] - 5_000, 10_000.0, "10 km", box=True)
    cax = R.mm_axes(fig, W, H, 112.0 + bw + 1.5, y1, 1.5, bh)
    cb = fig.colorbar(imh, cax=cax, extend="both")
    cb.set_label("Grid height − level (m)", fontsize=R.FS_SMALL, labelpad=1.0)
    cb.ax.tick_params(labelsize=R.FS_SMALL, length=2, width=0.4)
    cb.outline.set_linewidth(0.4)
    bk = [Line2D([], [], color=R.INK, lw=0.9, label="Predicted shoreline\n(edge of the flood\nfrom the open sea)"),
          Patch(fc="white", ec=R.MUTED, hatch="//////", lw=0.3, label="Below the level,\nnot reached")]
    if not pocket.any():
        bk = bk[:1]
    fig.legend(handles=bk, loc="upper left", bbox_to_anchor=((112.0 + bw + 13.0) / W, 1 - (y1 + 1.0) / H),
               fontsize=R.FS_SMALL, frameon=False, borderaxespad=0, handlelength=1.4, labelspacing=0.8)

    # c, D and Q ----------------------------------------------------------------------------------------
    y2 = y1 + ROW + GAP
    title(1.0, y2 - TT + 0.5, "c", "Displacement D between the shorelines, and the noise floor Q")
    axc = map_axes(8.0, y2)
    dcols = cmocean.cm.matter(np.linspace(0.18, 0.95, len(D_LABELS)))
    dcmap = ListedColormap(dcols)
    dnorm = BoundaryNorm(D_EDGES, dcmap.N)
    axc.imshow(np.where(obs == 1, 1.0, np.nan), extent=oext, origin="upper",
               cmap=ListedColormap(["#d4e5f2"]), interpolation="nearest", zorder=1, rasterized=True)
    for k_, col, ls in ((0, HALF_B, (0, (2.5, 1.2))), (1, HALF_B, (0, (0.8, 0.8)))):
        hs, hext = crop(z[f"half{k_}_{u}_{SECTOR}"], ext, BOX)
        contour(axc, hs, hext, col, 0.7, ls=ls, zorder=7)
    contour(axc, obs, oext, R.OBS, 0.6, zorder=8)
    key = f"predcells_{u}_{PRODUCT}"
    cx, cy, cd, kn = z[f"{key}_x"], z[f"{key}_y"], z[f"{key}_d"], z[f"{key}_known"]
    inb = (cx >= BOX[0]) & (cx < BOX[2]) & (cy >= BOX[1]) & (cy < BOX[3]) & kn
    axc.scatter(cx[inb], cy[inb], s=0.9, marker="s", c=cd[inb] / 1000.0, cmap=dcmap, norm=dnorm, linewidths=0,
                zorder=10, rasterized=True)
    R.scale_bar(axc, BOX[2] - 13_000, BOX[3] - 5_000, 10_000.0, "10 km", box=True)
    cax = R.mm_axes(fig, W, H, 8.0, y2 + bh + 7.0, bw, 1.6)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=BoundaryNorm(np.arange(len(D_LABELS) + 1), dcmap.N), cmap=dcmap),
                      cax=cax, orientation="horizontal")
    cb.set_ticks(np.arange(len(D_LABELS)) + 0.5, labels=D_LABELS)
    cb.ax.tick_params(labelsize=R.FS_SMALL, length=0, pad=1.2)
    cb.outline.set_linewidth(0.4)
    cb.set_label("Predicted cell to observed shoreline (km)", fontsize=R.FS_SMALL, labelpad=1.0)
    cb.ax.xaxis.set_label_position("top")
    axe = R.mm_axes(fig, W, H, 60.0, y2 + 1.0, 36.0, bh - 1.0)
    lo = 0.03
    for v, col, lab in ((dist, R.COLOURS[PRODUCT], "GEBCO_2026 to observed"), (hdist, HALF_B, "half A to half B")):
        xv, yv = ecdf(np.maximum(v / 1000.0, lo))
        axe.step(xv, yv, where="post", color=col, lw=1.0, label=lab, zorder=3)
    axe.axvline(D_m / 1000.0, color=R.COLOURS[PRODUCT], lw=0.7, ls=(0, (3, 1.5)), zorder=2)
    axe.axhline(0.5, color=R.GRID, lw=0.6, zorder=1)
    axe.axhline(0.95, color=R.GRID, lw=0.6, zorder=1)
    axe.axvline(Q_m / 1000.0, color=HALF_B, lw=0.7, ls=(0, (3, 1.5)), zorder=2)
    axe.text(D_m / 1000.0 * 1.12, 0.42, f"D = {D_m / 1000:.2f} km", color=R.COLOURS[PRODUCT], fontsize=R.FS_SMALL,
             ha="left", va="center")
    axe.text(Q_m / 1000.0 / 1.12, 0.25, f"Q = {Q_m / 1000:.2f} km", color="#6b6b6b", fontsize=R.FS_SMALL,
             ha="right", va="center")
    axe.set_xscale("log")
    axe.set_xlim(lo, 30)
    axe.set_xticks([0.1, 1, 10], ["0.1", "1", "10"])
    axe.set_ylim(0, 1.0)
    axe.set_yticks([0, 0.5, 0.95], ["0", "0.5", "0.95"])
    axe.set_xlabel("Distance (km)")
    axe.set_ylabel("Cumulative share")
    axe.text(4.0, 0.2, "GEBCO_2026\nto observed", color=R.COLOURS[PRODUCT], fontsize=R.FS_SMALL,
             ha="left", va="center", linespacing=1.0)
    axe.text(0.035, 0.84, "half A to\nhalf B", color="#6b6b6b", fontsize=R.FS_SMALL, ha="left",
             va="center", linespacing=1.0)
    ck = [Line2D([], [], color=R.OBS, lw=0.8, label="Observed shoreline"),
          Line2D([], [], color=HALF_B, lw=0.8, ls=(0, (2.5, 1.2)), label="Half A"),
          Line2D([], [], color=HALF_B, lw=0.8, ls=(0, (0.8, 0.8)), label="Half B")]
    fig.legend(handles=ck, loc="upper left", bbox_to_anchor=(60.0 / W, 1 - (y2 + bh + 10.0) / H), ncol=3,
               fontsize=R.FS_SMALL, frameon=False, borderaxespad=0, handlelength=1.6, columnspacing=1.0)

    # d, decision rule ----------------------------------------------------------------------------------
    title(108.0, y2 - TT + 0.5, "d", "Decision: the grid fails where D > Q at most levels")
    axd = R.mm_axes(fig, W, H, 120.0, y2 + 1.0, W - 120.0 - 3.0, bh - 1.0)
    above = det["dq"] > 1
    axd.axhspan(0.01, 1.0, color="#ececec", lw=0, zorder=0)
    axd.axhline(1.0, color=R.MUTED, lw=0.7, zorder=1)
    col = R.COLOURS[PRODUCT]
    axd.plot(det["level_m"][above], det["dq"][above], "o", ms=3.0, mfc=col, mec=col, zorder=3)
    axd.plot(det["level_m"][~above], np.maximum(det["dq"][~above], 0.01), "o", ms=3.0, mfc="white", mec=col, mew=0.7,
             zorder=3)
    ex = det[det["bin"] == b]
    axd.plot(ex["level_m"], ex["dq"], "o", ms=6.5, mfc="none", mec=R.INK, mew=0.8, zorder=4)
    axd.annotate("example level\n(a–c)", (float(ex["level_m"].iloc[0]), float(ex["dq"].iloc[0])), xytext=(-30, 16),
                 textcoords="offset points", fontsize=R.FS_SMALL, color=R.INK, ha="right", va="bottom",
                 arrowprops={"arrowstyle": "-", "color": R.INK, "lw": 0.5})
    axd.set_yscale("log")
    axd.set_ylim(0.1, 50)
    axd.set_yticks([0.1, 1, 10], ["0.1", "1", "10"])
    axd.yaxis.set_minor_locator(plt.NullLocator())
    axd.set_xlabel("Level (m, DAHITI reference)")
    axd.set_ylabel("D/Q")
    axd.grid(axis="y", color=R.GRID, lw=0.4)
    axd.text(0.98, 0.04, f"D > Q at {int(mrow['n_exceeding'])} of {int(mrow['n_evaluable'])} levels: "
             f"{'fails' if bool(mrow['mispredicts']) else 'passes'} the test", transform=axd.transAxes,
             ha="right", va="bottom", fontsize=R.FS_SMALL, color=R.INK, zorder=5)
    R.save(fig, "rse_fig_method", W, H)
    plt.close(fig)
    print("example", u, level, "| D", round(D_m), "Q", round(Q_m), "| levels", len(det), "above", int(above.sum()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
