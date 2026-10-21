"""Fig. 2: how a falling sea surveys its own seabed.

a, schematic of the waterline-level pairing (no data): each cloud-free scene gives a
   waterline, the altimetry series gives the level at that date, and the waterline is
   a contour of the seabed at that level (PREREG section 4).
b, the waterlines of one illustrative 25 km block (data/interim/waterlines/main/
   points.csv.gz; scripts/a6_waterlines.py): E2 and E3 points used by the DEM,
   coloured by their level bin, and the points removed by the static-contour rule
   (D-034, D-075) in grey. The block is chosen by a fixed rule, among blocks that
   are training blocks at both 25 km (blocks_25km.csv) and 50 km (blocks_50km.csv)
   in a sector labelled neither deltaic nor vegetated (results/levels/
   sector_labels.csv): the most distinct level bins, then the most points.
c, the surge-screening sensitivity of D-077 (results/levels/screening_sensitivity.json):
   the pre-registered statistic S(tau) with its bootstrap SE, and, descriptive only,
   the mean and 90th percentile of the block-bin displacements.
d, rising- against falling-limb waterlines at the levels both limbs observed
   (results/hindcast/main/noise_floor.csv): the median distance between the E1 and
   the E2 + E3 shorelines per sector and level ("e1_vs_e23"), against the median
   distance between the two acquisition halves within E1 or E2 at the same levels
   ("halves", the noise floor).

Output: figures/out/fig2_method.{pdf,tif,png}
Usage: python figures/fig2_method.py
"""

from __future__ import annotations

import json
import sys

import _common as C
import _data as D
import numpy as np
import pandas as pd

from caspian import plotting
from caspian.config import RESULTS
from caspian.geography import SECTOR_ORDER

LEVEL_RANGE = (-29.0, -25.6)   # colour range of levels and heights (m, DAHITI reference), as in Fig. 3a
BLOCK_M = 25_000.0
ZERO_KM = 0.02                 # where a zero median is drawn on the log axis of d


def pick_block() -> tuple[int, int, pd.Series]:
    w = D.waterlines_by_block()
    b25 = pd.read_csv(RESULTS / "study_area" / "blocks_25km.csv")
    b50 = pd.read_csv(RESULTS / "study_area" / "blocks_50km.csv")
    lab = pd.read_csv(RESULTS / "levels" / "sector_labels.csv").set_index("sector")
    w = w.merge(b25[["i", "j", "sector", "split"]], on=["i", "j"], how="inner")
    w["I"] = np.floor(w["i"] * BLOCK_M / 50_000.0).astype(int)
    w["J"] = np.floor(w["j"] * BLOCK_M / 50_000.0).astype(int)
    w = w.merge(b50[["i", "j", "split"]].rename(columns={"i": "I", "j": "J", "split": "split50"}), on=["I", "J"])
    plain = [s for s in lab.index if not (lab.loc[s, "deltaic"] or lab.loc[s, "vegetated"])]
    c = w[(w["split"] == "train") & (w["split50"] == "train") & w["sector"].isin(plain)]
    best = c.sort_values(["n_bins", "n"], ascending=False).iloc[0]
    return int(best["i"]), int(best["j"]), best


def schematic(ax, cmap, norm, bin_width_m: float) -> None:
    """Panel a, drawn in unit axes coordinates."""
    from matplotlib.patches import FancyArrowPatch, Polygon

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ink, muted = plotting.INK, plotting.INK_MUTED
    levels = [-26.0, -27.2, -28.6]                 # illustrative only
    lab_t = ["t1", "t2", "t3"]                     # plain text: no mathtext (fonts, sizes)
    # cross-section (lower half): seabed profile falling to the right
    x = np.linspace(0.02, 0.62, 200)
    zb = 0.42 - 0.30 * ((x - 0.02) / 0.60) ** 0.7
    y_of = {L: 0.40 - 0.26 * ((levels[0] - L) / 2.6) for L in levels}
    ax.fill_between(x, 0.04, zb, color="#d9cbb3", lw=0)
    ax.plot(x, zb, color=ink, lw=0.8)
    xs = []
    for L, t in zip(levels, lab_t):
        yl = y_of[L]
        k = np.argmin(np.abs(zb - yl))
        xs.append(x[k])
        ax.plot([x[k], 0.62], [yl, yl], color=cmap(norm(L)), lw=0.9, ls=(0, (3, 1.5)))
        ax.plot(x[k], yl, "o", ms=3.2, mfc=cmap(norm(L)), mec="white", mew=0.5, zorder=5)
        ax.text(0.635, yl, f"level at {t}", fontsize=5.5, va="center", ha="left", color=ink)
    ax.text(0.03, 0.06, "Seabed (cross-section)", fontsize=5.5, color=ink, ha="left", va="bottom")
    # plan view (upper half): the same three waterlines seen from above
    yy = np.linspace(0.55, 0.95, 100)
    ax.add_patch(Polygon([[0.02, 0.55], [0.62, 0.55], [0.62, 0.95], [0.02, 0.95]], closed=True, fc=C.LAND, ec="none"))
    for xk, L, t in zip(xs, levels, lab_t):
        xx = xk + 0.018 * np.sin((yy - 0.55) * 14 + 3 * xk)
        ax.plot(xx, yy, color=cmap(norm(L)), lw=1.1, solid_capstyle="round")
        ax.text(xk + 0.02, 0.965, f"waterline\non {t}", fontsize=5.2, ha="center", va="bottom", color=ink,
                linespacing=0.95)
        ax.add_patch(FancyArrowPatch((xk, 0.545), (xk, y_of[L] + 0.03), arrowstyle="-|>", mutation_scale=4,
                                     lw=0.45, color=muted, shrinkA=0, shrinkB=0))
    ax.fill_betweenx(yy, xs[-1] + 0.018 * np.sin((yy - 0.55) * 14 + 3 * xs[-1]), 0.62, color=C.SEA, lw=0)
    ax.text(0.27, 0.57, "Plan view", fontsize=5.5, color=ink, ha="left", va="bottom")
    # altimetry series (right): level falling through t1..t3
    x0, x1, y0, y1 = 0.74, 0.99, 0.60, 0.92
    tt = np.linspace(0, 1, 200)
    lev = 0.95 - 0.75 * tt + 0.05 * np.sin(tt * 30)
    ax.plot(x0 + (x1 - x0) * tt, y0 + (y1 - y0) * lev, color=ink, lw=0.7)
    ax.plot([x0, x0], [y0 - 0.02, y1 + 0.02], color=muted, lw=0.5)
    ax.plot([x0, x1], [y0 - 0.02, y0 - 0.02], color=muted, lw=0.5)
    for f, L, t in zip((0.08, 0.5, 0.93), levels, lab_t):
        k = int(f * 199)
        ax.plot(x0 + (x1 - x0) * tt[k], y0 + (y1 - y0) * lev[k], "o", ms=3.2, mfc=cmap(norm(L)), mec="white", mew=0.5)
        ax.text(x0 + (x1 - x0) * tt[k], y0 - 0.045, t, fontsize=5.5, ha="center", va="top", color=ink)
    ax.text(x0 + 0.5 * (x1 - x0), y1 + 0.04, "Altimetry level", fontsize=5.5, ha="center", va="bottom", color=ink)
    ax.text(0.865, 0.30, "Each waterline is a\nseabed contour at the\nlevel of its date.\n"
            f"Scenes are pooled in\n{bin_width_m:.1f} m level bins;\neach bin gives one\ncontour per epoch.",
            fontsize=5.5, ha="center", va="center", color=ink, linespacing=1.15)


def main() -> int:
    import cmocean
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D

    m = C.metrics()
    MIN_N = int(m["params.h1_min_block_n"]["value"])   # distances below which a median in d is drawn open
    cmap = cmocean.cm.deep_r
    norm = Normalize(*LEVEL_RANGE)
    bi, bj, best = pick_block()
    x0, y0 = bi * BLOCK_M, bj * BLOCK_M
    wl = D.waterlines_in_box(x0, y0, x0 + BLOCK_M, y0 + BLOCK_M)
    wl = wl[wl["epoch"].isin(["E2", "E3"])]
    used, static = wl[~wl["static"]], wl[wl["static"]]
    if len(used) != int(best["n"]):
        print(f"FAIL block point count {len(used)} != {int(best['n'])}")
        return 1

    ss = json.loads((RESULTS / "levels" / "screening_sensitivity.json").read_text(encoding="utf-8"))
    nf = pd.read_csv(RESULTS / "hindcast" / "main" / "noise_floor.csv")
    ev = nf[(nf["kind"] == "e1_vs_e23") & (nf["n"] > 0) & (nf["sector"] != "all") & np.isfinite(nf["median_m"])]
    shared_bins = sorted(ev["bin"].unique())
    hv = nf[(nf["kind"] == "halves") & (nf["n"] > 0) & (nf["sector"] != "all") & nf["epoch"].isin(["E1", "E2"])
            & nf["bin"].isin(shared_bins) & np.isfinite(nf["median_m"])]
    n_inf = int(((nf["kind"] == "e1_vs_e23") & (nf["sector"] != "all") & ~np.isfinite(nf["median_m"])
                 & (nf["n"] > 0)).sum())

    C.style()
    W, H = C.DOUBLE_MM, 122.0
    fig = plt.figure(figsize=(W * C.MM, H * C.MM))
    ax_a = fig.add_axes([0.035, 0.535, 0.47, 0.43])
    SB = 50.0   # side of the square map b (mm)
    ax_b = fig.add_axes([112.0 / W, 66.0 / H, SB / W, SB / H])
    cax_b = fig.add_axes([(112.0 + SB + 2.5) / W, 66.0 / H, 2.0 / W, SB / H])
    ax_c = fig.add_axes([0.075, 0.09, 0.33, 0.33])
    ax_d = fig.add_axes([0.575, 0.09, 0.405, 0.33])

    # a
    schematic(ax_a, cmap, norm, float(m["params.bin_width_m"]["value"]))
    plotting.panel_label(ax_a, "a", x=-0.07, y=0.985)

    # b
    ax_b.scatter((static["x"] - x0) / 1000, (static["y"] - y0) / 1000, s=0.15, c="#c8c8c8", lw=0, rasterized=True,
                 zorder=1)
    order = np.argsort(used["z_m"].to_numpy())[::-1]
    u = used.iloc[order]
    sc = ax_b.scatter((u["x"] - x0) / 1000, (u["y"] - y0) / 1000, s=0.25, c=u["z_m"], cmap=cmap, norm=norm, lw=0,
                      rasterized=True, zorder=2)
    ax_b.set_xlim(0, BLOCK_M / 1000)
    ax_b.set_ylim(0, BLOCK_M / 1000)
    ax_b.set_aspect("equal")
    ax_b.set_xlabel("Easting in block (km)")
    ax_b.set_ylabel("Northing in block (km)")
    for sp in ("top", "right"):
        ax_b.spines[sp].set_visible(True)
    cb = fig.colorbar(sc, cax=cax_b)
    cb.set_label("Waterline level (m)", fontsize=6)
    cb.ax.tick_params(labelsize=5.5, length=2, width=0.4)
    cb.outline.set_linewidth(0.4)
    ax_b.text(0.02, 0.98, f"{C.SECTOR_SHORT[best['sector']]}, training block\n"
              f"n = {len(used):,} points in {int(best['n_bins'])} level bins (E2, E3)\n"
              f"grey: static contours, removed (n = {len(static):,})", transform=ax_b.transAxes,
              ha="left", va="top", fontsize=5.5, color=plotting.INK,
              bbox={"boxstyle": "square,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.85})
    plotting.panel_label(ax_b, "b", x=-0.28, y=1.0)

    # c
    taus = ss["taus"]
    xs = np.arange(len(taus))
    S = np.array([ss["results"][t]["S_m"] for t in taus])
    se = np.array([ss["results"][t]["se_m"] for t in taus])
    nb = [ss["results"][t]["n_block_bins"] for t in taus]
    desc = ss["descriptive_not_used_for_choice"]
    p90 = np.array([desc[t]["p90_m"] for t in taus])
    mean = np.array([desc[t]["mean_m"] for t in taus])
    ax_c.plot(xs, p90, color=plotting.INK_MUTED, lw=0.6, ls=(0, (2, 1.5)), marker="o", ms=3, mfc="white",
              mec=plotting.INK_MUTED, mew=0.6, label="90th percentile (descriptive)")
    ax_c.plot(xs, mean, color=plotting.INK_MUTED, lw=0.6, ls=(0, (2, 1.5)), marker="s", ms=2.6, mfc="white",
              mec=plotting.INK_MUTED, mew=0.6, label="Mean (descriptive)")
    ax_c.errorbar(xs, S, yerr=se, fmt="o", ms=3.2, color=plotting.INK, ecolor=plotting.INK, elinewidth=0.6, capsize=1.5,
                  capthick=0.6, label="S(τ), median ± bootstrap SE (rule)", zorder=4)
    ax_c.set_yscale("log")
    ax_c.set_ylim(20, 3000)
    ax_c.set_yticks([30, 100, 300, 1000, 3000], ["30", "100", "300", "1,000", "3,000"])
    ax_c.yaxis.set_minor_locator(plt.NullLocator())
    ax_c.set_xticks(xs, [("none" if t == "inf" else f"{float(t):.2f}") + f"\n{n}" for t, n in zip(taus, nb)])
    ax_c.text(-0.02, -0.105, "n", transform=ax_c.transAxes, ha="right", va="top", fontsize=5.5, color=plotting.INK_MUTED)
    ax_c.set_xlabel("Screening threshold τ on predicted setup (m)")
    ax_c.set_ylabel("Odd–even waterline distance (m)")
    cell = float(m["params.stage2_res_m"]["value"])
    ax_c.axhline(cell, color="#bbbbbb", lw=0.6, zorder=0, label=f"One {cell:.0f} m grid cell")
    ax_c.set_xlim(-0.5, len(taus) - 0.5)
    ax_c.legend(loc="upper left", bbox_to_anchor=(0.0, 0.64), fontsize=5.5, handlelength=1.6, borderaxespad=0.2)
    plotting.panel_label(ax_c, "c", x=-0.2, y=1.0)

    # d
    sectors = [s for s in SECTOR_ORDER if s in set(ev["sector"]) | set(hv["sector"])]
    ypos = {s: i for i, s in enumerate(sectors[::-1])}

    def xkm(v):
        return np.maximum(np.asarray(v, float) / 1000.0, ZERO_KM)

    for s in sectors:
        h = hv[hv["sector"] == s]
        ax_d.scatter(xkm(h["median_m"]), ypos[s] - 0.17 + 0.0 * h["bin"], s=7, marker="|", c=plotting.INK_MUTED,
                     lw=0.7, zorder=2)
        e = ev[ev["sector"] == s]
        for r in e.itertuples():
            filled = r.n >= MIN_N
            ax_d.scatter(xkm(r.median_m), ypos[s] + 0.12, s=13, marker="o", c=plotting.INK if filled else "white",
                         edgecolors="white" if filled else plotting.INK, lw=0.5 if filled else 0.7, zorder=3)
    ax_d.set_xscale("log")
    ax_d.set_xlim(ZERO_KM * 0.8, 400)
    ticks = [ZERO_KM, 0.06, 0.6, 6, 60]
    ax_d.set_xticks(ticks, ["0", "0.06", "0.6", "6", "60"])
    ax_d.xaxis.set_minor_locator(plt.NullLocator())
    ax_d.set_yticks(list(ypos.values()), [C.SECTOR_TINY[s] for s in ypos])
    ax_d.set_ylim(-0.6, len(sectors) + 0.55)
    ax_d.tick_params(axis="y", length=0)
    ax_d.grid(axis="x", color=plotting.GRID, lw=0.4)
    ax_d.set_axisbelow(True)
    ax_d.set_xlabel("Median shoreline distance (km)")
    lv_lo, lv_hi = ev["level_m"].min(), ev["level_m"].max()
    ax_d.legend(handles=[Line2D([], [], ls="none", marker="o", ms=3.4, mfc=plotting.INK, mec="white",
                                label="E1 (rising limb) vs E2 + E3, per level"),
                         Line2D([], [], ls="none", marker="o", ms=3.4, mfc="white", mec=plotting.INK,
                                label=f"same, < {MIN_N} distances"),
                         Line2D([], [], ls="none", marker="|", ms=4, mec=plotting.INK_MUTED, mew=0.7,
                                label="Halves within E1 or E2 (noise floor)")],
                loc="upper left", ncol=2, fontsize=5.5, handlelength=1.0, borderaxespad=0.15, columnspacing=1.0,
                handletextpad=0.3, frameon=False)
    ax_d.text(1.0, 1.02, f"levels {lv_lo:.2f} to {lv_hi:.2f} m".replace("-", "−"), transform=ax_d.transAxes,
              ha="right", va="bottom",
              fontsize=5.5, color=plotting.INK_MUTED)
    plotting.panel_label(ax_d, "d", x=-0.25, y=1.0)

    C.save(fig, "fig2_method")
    plt.close(fig)
    print("block", (bi, bj), best["sector"], "| used", len(used), "static", len(static), "bins", int(best["n_bins"]),
          "| e1_vs_e23 rows", len(ev), "bins", shared_bins, "| infinite medians (no E2+E3 shoreline)", n_inf,
          "| e1_end", m["levels.epoch_E1_end"]["value"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
