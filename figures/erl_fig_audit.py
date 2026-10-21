"""ERL figure "Auditing the bathymetry grids" (the Letter's audit figure; D-100, D-102
addendum; 15 cm wide, text 8 pt or more). The ERL version of figures/fig4_audit.py, which
stays the Supplementary/Phase A figure: the per-block maps (fig4_audit c) are left out here
(they are in the Supplementary), the other panels are kept and enlarged.

(a) Area bias of each product's predicted sea against the observed sea, per (epoch, level)
    unit, inside the domain and within the valid mask (results/hindcast/main/extent.csv,
    sector "domain", area_bias_rel; units with observed sea): left, after each product's
    best-fitting vertical constant (the shape variant, D-070, which decides; D-072); right,
    on the documented datum chain without a constant (the total variant).
(b) Shoreline displacement by level band, shape variant (displacement_by_level_band.csv,
    scripts/a7_level_bands.py): per band, the median over sectors and levels of the
    per-level median displacement; the dashed line marks h1.disp_band.edge_2_m.
(c) Per sector, the median over evaluable levels of each product's per-level median
    displacement (h1_mispredicts_detail.csv, shape variant), against the sector's noise
    floor (the median over the same levels of the noise-floor 95th percentile); filled
    where the product mispredicts the sector (h1_mispredicts.csv).

Panel letters (a) area bias, (b) bands, (c) sectors keep the Letter's existing references to
(a) and (b) valid. Nothing is recomputed: the values are those of fig4_audit.py, read from the
same tables.
Output: figures/out/erl_fig_audit.{pdf,tif,png}
Usage: python figures/erl_fig_audit.py
"""

from __future__ import annotations

import sys

import _common as C
import _erl as E
import numpy as np
import pandas as pd

from caspian.config import RESULTS
from caspian.geography import SECTOR_ORDER

HC = RESULTS / "hindcast" / "main"
PRODUCTS = ("gebco_2026", "gebco_2022", "etopo_2022", "etopo1", "srtm15p_v2.7")
EPOCH_STYLE = {"E3": ("-", 1.1), "E2": ((0, (3.5, 1.6)), 1.0), "E1": ((0, (1.0, 1.3)), 1.0)}
SECTOR_LABEL = {"west": "Kizlyar–\nDagestan", "volga": "Volga\ndelta", "interfluve": "Volga–Ural\ninterfluve",
                "ural": "Ural\ndelta", "northeast": "NE Kazakh\ncoast", "komsomolets": "Komsomolets\nBay",
                "open_shelf": "Open\nshelf"}
FLOOR = "#e3e3e3"


def band_edges(labels) -> list[tuple[float, float]]:
    return [tuple(float(v) for v in b.strip("(]").split(",")) for b in labels]


def main() -> int:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle

    m = C.metrics()
    ext = pd.read_csv(HC / "extent.csv")
    dom = ext[(ext["sector"] == "domain") & (ext["area_obs_km2"] > 0)].copy()
    dom["bias_pct"] = 100.0 * dom["area_bias_rel"]
    bands = pd.read_csv(HC / "displacement_by_level_band.csv")
    bands = bands[bands["variant"] == "shape"]
    det = pd.read_csv(HC / "h1_mispredicts_detail.csv")
    det = det[(det["variant"] == "shape") & (det["evaluable"] == True)]  # noqa: E712
    mis = pd.read_csv(HC / "h1_mispredicts.csv")
    mis = mis[mis["variant"] == "shape"].set_index(["product", "sector"])["mispredicts"]

    E.style()
    W = E.DOUBLE_MM
    L, R = 15.0, 2.0
    TOP, HA, G1, HC_, G2, HD, BOT = 6.0, 42.0, 15.0, 44.0, 14.0, 46.0, 13.0
    H = TOP + HA + G1 + HC_ + G2 + HD + BOT
    fig = plt.figure(figsize=(W * C.MM, H * C.MM))
    gap_a = 14.0
    aw = (W - L - R - gap_a) / 2

    # (a) area bias, two variants ---------------------------------------------------------------------------
    titles = {"shape": "After the best-fitting vertical shift", "total": "On the datum chain, no shift"}
    axs = []
    for i, (variant, letter) in enumerate((("shape", "a"), ("total", None))):
        ax = E.mm_axes(fig, W, H, L + i * (aw + gap_a), TOP, aw, HA)
        d = dom[dom["variant"] == variant]
        for p in PRODUCTS:
            for ep, (ls, lw) in EPOCH_STYLE.items():
                g = d[(d["product"] == p) & (d["epoch"] == ep)].sort_values("level_m")
                ax.plot(g["level_m"], g["bias_pct"], ls=ls, lw=lw, color=E.COLOURS[p], marker=E.MARKERS[p], ms=3.0,
                        mew=0, markevery=2, zorder=3)
        ax.axhline(0, color=E.MUTED, lw=0.6, zorder=1)
        ax.set_xlabel("Level (m, DAHITI reference)")
        ax.set_ylabel("Area bias (%)")
        ax.grid(axis="y", color=E.GRID, lw=0.4)
        ax.set_axisbelow(True)
        ax.set_xticks([-28.5, -28.0, -27.5, -27.0, -26.5, -26.0])
        ax.text(0.02, 0.97, titles[variant], transform=ax.transAxes, ha="left", va="top", fontsize=E.FS,
                color=E.INK, bbox={"boxstyle": "square,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.85})
        if letter:
            E.label(ax, letter, x=-0.2, y=1.02)
        axs.append(ax)
    axs[0].set_ylim(-55, 45)
    axs[1].set_ylim(-20, 75)

    # (b) displacement by level band ---------------------------------------------------------------
    cw = 66.0
    yc = TOP + HA + G1
    axc = E.mm_axes(fig, W, H, L, yc, cw, HC_)
    edges = band_edges(bands["band"])
    order = np.argsort([lo for lo, _ in edges])
    for p in PRODUCTS:
        vals = bands[p].to_numpy(float) / 1000.0
        for k in order:
            lo, hi = edges[k]
            axc.plot([lo, hi], [vals[k]] * 2, color=E.COLOURS[p], lw=1.6, solid_capstyle="butt", zorder=3)
            axc.plot((lo + hi) / 2, vals[k], E.MARKERS[p], ms=4.0, mfc=E.COLOURS[p], mec="white", mew=0.5, zorder=4)
    split = float(m["h1.disp_band.edge_2_m"]["value"])
    axc.axvline(split, color=E.MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)
    axc.text(split + 0.05, 0.17, E.num(m, "h1.disp_band.edge_2_m", "m"), fontsize=E.FS, color=E.MUTED, ha="left",
             va="bottom")
    for lo, _ in edges:
        axc.axvline(lo, color=E.GRID, lw=0.4, zorder=0)
    axc.set_yscale("log")
    axc.set_ylim(0.15, 30)
    axc.set_yticks([0.3, 1, 3, 10, 30], ["0.3", "1", "3", "10", "30"])
    axc.yaxis.set_minor_locator(plt.NullLocator())
    axc.set_xlim(min(e[0] for e in edges), max(e[1] for e in edges))
    axc.set_xlabel("Level band (m, DAHITI reference)")
    axc.set_ylabel("Displacement (km)")
    E.label(axc, "b", x=-0.2, y=1.02)

    # legend beside (b) ------------------------------------------------------------------------------
    handles = [Line2D([], [], color=E.COLOURS[p], marker=E.MARKERS[p], ms=4.5, mfc=E.COLOURS[p], mec="white", mew=0.5,
                      lw=1.4, label=E.NAMES[p]) for p in PRODUCTS]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=((L + cw + 10.0) / W, 1 - (yc - 1.0) / H),
               title="Product", alignment="left", fontsize=E.FS, frameon=False, borderaxespad=0, handlelength=2.2,
               labelspacing=0.35)
    ep_h = [Line2D([], [], color=E.INK, ls=EPOCH_STYLE[e][0], lw=EPOCH_STYLE[e][1], label=lab)
            for e, lab in (("E1", "E1"), ("E2", "E2"), ("E3", "E3"))]
    fig.legend(handles=ep_h, loc="upper left", bbox_to_anchor=((L + cw + 46.0) / W, 1 - (yc - 1.0) / H),
               title="Epoch (a)", alignment="left", fontsize=E.FS, frameon=False, borderaxespad=0,
               handlelength=2.6, labelspacing=0.35)
    fig.legend(handles=[Patch(fc=FLOOR, ec=E.MUTED, lw=0.5, label="Observation noise floor"),
                        Line2D([], [], ls="none", marker="o", ms=5, mfc=E.INK, mec=E.INK, label="Mispredicts the sector"),
                        Line2D([], [], ls="none", marker="o", ms=5, mfc="white", mec=E.INK, mew=1.0,
                               label="Does not")],
               loc="upper left", bbox_to_anchor=((L + cw + 10.0) / W, 1 - (yc + 25.0) / H), title="Sectors (c)",
               alignment="left", fontsize=E.FS, frameon=False, borderaxespad=0, handlelength=1.4, labelspacing=0.35)

    # (c) sector summary --------------------------------------------------------------------------------
    yd = yc + HC_ + G2
    axd = E.mm_axes(fig, W, H, L, yd, W - L - R, HD)
    offs = np.linspace(-0.3, 0.3, len(PRODUCTS))
    for i, s in enumerate(SECTOR_ORDER):
        g = det[det["sector"] == s]
        noise = float(g.groupby("product")["p95_noise"].median().median()) / 1000.0
        axd.add_patch(Rectangle((i - 0.42, 0.105), 0.84, noise - 0.105, fc=FLOOR, ec="none", zorder=1))
        axd.plot([i - 0.42, i + 0.42], [noise, noise], color=E.MUTED, lw=0.9, zorder=2)
        for p, dx in zip(PRODUCTS, offs):
            v = float(g.loc[g["product"] == p, "median"].median()) / 1000.0
            filled = bool(mis.get((p, s), False))
            axd.plot(i + dx, v, E.MARKERS[p], ms=5.5, mfc=E.COLOURS[p] if filled else "white", mec=E.COLOURS[p],
                     mew=1.0, zorder=4)
    axd.set_yscale("log")
    axd.set_ylim(0.1, 30)
    axd.set_yticks([0.1, 0.3, 1, 3, 10, 30], ["0.1", "0.3", "1", "3", "10", "30"])
    axd.yaxis.set_minor_locator(plt.NullLocator())
    axd.set_xlim(-0.55, len(SECTOR_ORDER) - 0.45)
    axd.set_xticks(range(len(SECTOR_ORDER)), [SECTOR_LABEL[s] for s in SECTOR_ORDER])
    axd.tick_params(axis="x", length=0)
    axd.set_ylabel("Median displacement (km)")
    axd.grid(axis="y", color=E.GRID, lw=0.4)
    axd.set_axisbelow(True)
    E.label(axd, "c", x=-0.09, y=1.02)
    C.save(fig, "erl_fig_audit")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
