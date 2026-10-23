"""RSE figure: agreement in flooded area is not agreement in shoreline position (double column).

a, Area bias of each grid's predicted sea against the observed sea, per hindcast unit
   (epoch, level bin) inside the study domain, after each grid's area-fitted vertical
   offset (the offset-corrected or "shape" variant): 100 x (predicted - observed) /
   observed, counting the errors on known states outside the valid mask
   (results/hindcast/main/extent.csv, sector "domain", area_bias_rel_total; the rows that
   PREREG Amendment 18 item 4 summarises, results/revision/s/item4_area_bias.csv, whose
   medians the script checks). One small panel per epoch: rising limb (E1), early fall
   (E2), late fall (E3). Units beyond the axis are drawn on its edge (open arrowhead) and
   labelled with their value.
b, Out of sample: the late-fall (E3) area bias in the calibration and test blocks
   (all sectors) with the offset fitted on all epochs' training units (c_all, open) and
   with the offset fitted on the rising-limb and early-fall training units only (c_E12,
   filled): median and 25th-75th percentile over the E3 levels (results/revision/s/
   rasters/oos_area.csv, summarised in item4_oos.csv, which the script checks).
c, Three ways of fitting the vertical datum of a grid, fitted to flooded area on the
   training blocks (PREREG Amendment 18 item 9; results/revision/v/planar_comparison.json):
   one offset for the domain (the main analysis), one offset per sector, and a planar
   offset field c0 + a x + b y. Left: the summed absolute area error over the training-
   block units (the fitting objective). Right: the median over the seven sectors of each
   sector's median shoreline displacement. The planar field was audited for GEBCO_2026 and
   ETOPO1 only.

Output: figures/out/rse_fig_area.{pdf,tif,png}
Usage: python figures/rse_fig_area.py
"""

from __future__ import annotations

import json
import sys

import _common as C
import _rse as R
import numpy as np
import pandas as pd

from caspian.config import RESULTS

HC = RESULTS / "hindcast" / "main"
REV = RESULTS / "revision"
EPOCHS = (("E1", "Rising limb (E1)"), ("E2", "Early fall (E2)"), ("E3", "Late fall (E3)"))
YLIM = (-65.0, 65.0)
DESIGNS = (("domain_constant", "One offset"), ("sector_constants_R2", "Per sector"), ("planar", "Planar field"))


def main() -> int:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    m = C.metrics()
    ext = pd.read_csv(HC / "extent.csv")
    d = ext[(ext["sector"] == "domain") & (ext["area_obs_km2"] > 0) & (ext["variant"] == "shape")
            & ext["epoch"].isin(["E1", "E2", "E3"])].copy()
    d["bias"] = 100.0 * d["area_bias_rel_total"]
    ab = pd.read_csv(REV / "s" / "item4_area_bias.csv")
    ab = ab[(ab["variant"] == "shape") & (ab["scope"] == "domain_all_sectors")].set_index("product")
    for p, g in d.groupby("product"):
        if abs(g["bias"].median() - float(ab.loc[p, "bias_pct_median"])) > 1e-9 or len(g) != int(ab.loc[p, "bias_pct_n"]):
            raise SystemExit(f"{p}: domain area bias differs from item4_area_bias.csv")
    oa = pd.read_csv(REV / "s" / "rasters" / "oos_area.csv")
    oa = oa[oa["region"].isin(["calibtest_validated", "calibtest_other"])]
    g0 = oa.groupby(["product", "bin", "level_m"])[["cells_pred_c_all", "cells_pred_c_e12", "cells_water"]].sum()
    g0 = g0[g0["cells_water"] > 0].reset_index()
    for w in ("c_all", "c_e12"):
        g0[w] = 100.0 * (g0[f"cells_pred_{w}"] - g0["cells_water"]) / g0["cells_water"]
    oos = pd.read_csv(REV / "s" / "item4_oos.csv")
    oos = oos[oos["region"] == "calibtest"].set_index(["product", "constant"])
    for p, g in g0.groupby("product"):
        for w in ("c_all", "c_e12"):
            if abs(g[w].median() - float(oos.loc[(p, w), "bias_pct_median"])) > 1e-9:
                raise SystemExit(f"{p} {w}: out-of-sample bias differs from item4_oos.csv")
    pc = json.loads((REV / "v" / "planar_comparison.json").read_text(encoding="utf-8"))
    obj = pc["objective_eq_s11_km2_extent_domain_train"]
    ver = pc["verdicts"]

    R.style()
    W = R.DOUBLE_MM
    L, RGT, TOP = 14.0, 2.0, 10.0
    HA, GAP, HB, BOT = 42.0, 19.0, 46.0, 10.0
    H = TOP + HA + GAP + HB + BOT
    fig = plt.figure(figsize=(W * C.MM, H * C.MM))
    kw = {"frameon": False, "borderaxespad": 0, "fontsize": R.FS, "handletextpad": 0.4, "alignment": "left"}
    fig.legend(handles=[Line2D([], [], color=R.COLOURS[p], marker=R.MARKERS[p], ms=3.5, mec="white", mew=0.4, lw=1.1,
                               label=R.NAMES[p]) for p in R.PRODUCTS],
               loc="upper left", bbox_to_anchor=(L / W, 1 - 1.0 / H), ncol=5, columnspacing=1.6, handlelength=2.0, **kw)

    # (a) per-level area bias, one panel per epoch, widths proportional to the level range ----------------
    spans = {ep: (d[d["epoch"] == ep]["level_m"].min() - 0.1, d[d["epoch"] == ep]["level_m"].max() + 0.1)
             for ep, _ in EPOCHS}
    total = sum(hi - lo for lo, hi in spans.values())
    gap = 4.0
    aw = W - L - RGT - 2 * gap
    x0 = L
    for k, (ep, title) in enumerate(EPOCHS):
        lo, hi = spans[ep]
        w = aw * (hi - lo) / total
        ax = R.mm_axes(fig, W, H, x0, TOP, w, HA)
        x0 += w + gap
        ax.axhline(0, color=R.MUTED, lw=0.6, zorder=1)
        for p in R.PRODUCTS:
            g = d[(d["product"] == p) & (d["epoch"] == ep)].sort_values("level_m")
            y = g["bias"].to_numpy(float)
            yc = np.clip(y, *YLIM)
            ax.plot(g["level_m"], yc, color=R.COLOURS[p], lw=1.0, marker=R.MARKERS[p], ms=2.8, mec="white", mew=0.3,
                    zorder=3, clip_on=False)
            for xv, yv in zip(g["level_m"][y > YLIM[1]], y[y > YLIM[1]]):
                ax.plot(xv, YLIM[1], marker="^", ms=4.0, mfc="white", mec=R.COLOURS[p], mew=0.8, zorder=4,
                        clip_on=False)
        over = d[(d["epoch"] == ep) & (d["bias"] > YLIM[1])].sort_values("bias", ascending=False)
        if len(over):
            lv = float(over["level_m"].iloc[0])
            parts = [f"{R.NAMES[r['product']]} {r['bias']:.0f}%" for _, r in over.iterrows()]
            rows = [", ".join(parts[i:i + 2]) for i in range(0, len(parts), 2)]
            head = f"Off scale at {lv:.2f} m:".replace("-", "−")
            ax.text(0.02, 0.98, head + "\n" + ",\n".join(rows), transform=ax.transAxes, ha="left", va="top", fontsize=R.FS_SMALL, color=R.INK, linespacing=1.05)
        ax.set_xlim(lo, hi)
        ax.set_ylim(*YLIM)
        ax.set_yticks([-60, -40, -20, 0, 20, 40, 60])
        ax.set_xticks({"E1": [-26.2, -26.0, -25.8], "E2": [-26.6, -26.2, -25.8]}.get(ep, [-28.5, -28.0, -27.5, -27.0]))
        ax.grid(axis="y", color=R.GRID, lw=0.4)
        ax.set_axisbelow(True)
        ax.set_title(title, fontsize=R.FS, loc="left", pad=3)
        if k == 0:
            ax.set_ylabel("Area bias (%)")
            R.fig_label(fig, W, H, "a", 1.0, TOP - 1.0)
        else:
            ax.tick_params(labelleft=False)
    fig.text((L + (W - L - RGT) / 2) / W, 1 - (TOP + HA + 6.5) / H, "Level (m, DAHITI reference)", ha="center",
             va="top", fontsize=R.FS, color=R.INK)

    # (b) out of sample ------------------------------------------------------------------------------
    yb = TOP + HA + GAP
    axb = R.mm_axes(fig, W, H, L, yb, 58.0, HB)
    xs = np.arange(len(R.PRODUCTS))
    for i, p in enumerate(R.PRODUCTS):
        g = g0[g0["product"] == p]
        for w, dx, filled in (("c_all", -0.15, False), ("c_e12", 0.15, True)):
            q1, q2, q3 = np.percentile(g[w], [25, 50, 75])
            axb.plot([i + dx] * 2, [q1, q3], color=R.COLOURS[p], lw=1.1, solid_capstyle="butt", zorder=2)
            axb.plot(i + dx, q2, R.MARKERS[p], ms=4.5, mfc=R.COLOURS[p] if filled else "white", mec=R.COLOURS[p],
                     mew=0.9, zorder=3)
    axb.axhline(0, color=R.MUTED, lw=0.6, zorder=1)
    axb.set_xticks(xs, [R.NAMES[p] for p in R.PRODUCTS], rotation=30, ha="right", rotation_mode="anchor")
    axb.tick_params(axis="x", length=0)
    axb.set_xlim(-0.6, len(R.PRODUCTS) - 0.4)
    axb.set_ylim(-40, 15)
    axb.set_ylabel("Late-fall area bias,\ncalibration and test blocks (%)")
    axb.grid(axis="y", color=R.GRID, lw=0.4)
    axb.set_axisbelow(True)
    R.fig_label(fig, W, H, "b", 1.0, yb - 5.0)
    axb.legend(handles=[Line2D([], [], ls="none", marker="o", ms=4.5, mfc="white", mec=R.INK, mew=0.9,
                               label="Offset fitted on all epochs"),
                        Line2D([], [], ls="none", marker="o", ms=4.5, mfc=R.INK, mec=R.INK,
                               label="Offset fitted on E1 + E2 only")],
               loc="lower left", bbox_to_anchor=(0.0, 0.0), handlelength=0.8, labelspacing=0.3, **kw)

    # (c) area objective against shoreline displacement for three datum designs ----------------------------
    cx0 = L + 58.0 + 22.0
    cw = (W - cx0 - RGT - 14.0) / 2
    axo = R.mm_axes(fig, W, H, cx0, yb, cw, HB)
    axd = R.mm_axes(fig, W, H, cx0 + cw + 14.0, yb, cw, HB)
    for p in R.PRODUCTS:
        xo, yo, yd = [], [], []
        for j, (des, _) in enumerate(DESIGNS):
            if p in obj.get(des, {}) and p in ver and des in ver[p]:
                xo.append(j)
                yo.append(obj[des][p] / 1000.0)
                yd.append(ver[p][des]["median_of_sector_medians_m"] / 1000.0)
        for a_, yy in ((axo, yo), (axd, yd)):
            a_.plot(xo, yy, color=R.COLOURS[p], lw=1.0, marker=R.MARKERS[p], ms=4.0, mec="white", mew=0.4, zorder=3)
    for a_, lab in ((axo, "Summed |area error|,\ntraining blocks (10³ km²)"),
                    (axd, "Median shoreline\ndisplacement (km)")):
        a_.set_xticks(range(len(DESIGNS)), [t for _, t in DESIGNS], rotation=30, ha="right", rotation_mode="anchor")
        a_.tick_params(axis="x", length=0)
        a_.set_xlim(-0.35, len(DESIGNS) - 0.65)
        a_.set_ylim(0, None)
        a_.set_ylabel(lab)
        a_.grid(axis="y", color=R.GRID, lw=0.4)
        a_.set_axisbelow(True)
    axo.text(0.0, 1.04, "Fitted to area", transform=axo.transAxes, fontsize=R.FS, ha="left", va="bottom")
    axd.text(0.0, 1.04, "Shoreline position", transform=axd.transAxes, fontsize=R.FS, ha="left", va="bottom")
    R.fig_label(fig, W, H, "c", cx0 - 14.0, yb - 5.0)
    # checks against the metrics
    for p, key in (("gebco_2026", "rev.w.v.planar.gebco2026.after_km"), ("etopo1", "rev.w.v.planar.etopo1.after_km")):
        if abs(ver[p]["planar"]["median_of_sector_medians_m"] / 1000.0 - R.value(m, key)) > 0.006:
            raise SystemExit(f"{p}: planar displacement differs from {key}")
    R.save(fig, "rse_fig_area", W, H)
    plt.close(fig)
    print("units per epoch:", d.groupby("epoch")["level_m"].nunique().to_dict(), "| E3 calib/test levels:",
          g0.groupby("product").size().to_dict())
    return 0


if __name__ == "__main__":
    sys.exit(main())
