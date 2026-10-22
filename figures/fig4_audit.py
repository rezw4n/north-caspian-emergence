"""Fig. 4: auditing the maps behind projections (H1).

a, area bias of each product's predicted sea against the observed sea, per (epoch,
   level) unit, inside the domain and within the valid mask (results/hindcast/main/
   extent.csv, sector "domain", area_bias_rel; units with observed sea), for the shape
   variant (the product shifted by its fitted constant; deciding variant)
   and the total variant (the product on the DAHITI reference by the per-cell geoid
   correction, no constant).
b, shoreline displacement by level band (results/hindcast/main/
   displacement_by_level_band.csv, shape variant; scripts/a7_level_bands.py): per band
   the median over sectors and levels of the per-level median displacement.
c, where the products err: the median absolute displacement per 25 km block, pooled
   over every unit (results/hindcast/main/block_h1.csv, shape variant), for GEBCO_2026
   and ETOPO 2022 (the other products: Supplementary Figure, figures/ed_h1_blocks.py).
   Blocks with fewer than params.h1_min_block_n distances are not drawn. Rows whose
   committed median is negative (not a distance: the decoded int32 minimum, a defect
   of block_h1.csv found while drawing this figure) are hatched as undefined, never
   coloured. Per-cell disagreement maps are not stored by the hindcast driver, so the
   map is per block.
d, sector summary: the median over evaluable levels of each product's per-level
   median displacement (h1_mispredicts_detail.csv, shape variant), against the
   sector's noise floor (the median over the same levels of the noise-floor 95th
   percentile); filled where the product "mispredicts" the sector (h1_mispredicts.csv).

Output: figures/out/fig4_audit.{pdf,tif,png}
Usage: python figures/fig4_audit.py
"""

from __future__ import annotations

import sys

import _common as C
import numpy as np
import pandas as pd

from caspian import plotting
from caspian.config import RESULTS
from caspian.geography import SECTOR_ORDER

HC = RESULTS / "hindcast" / "main"
MAP_PRODUCTS = ("gebco_2026", "etopo_2022")
DISP_RANGE_KM = (0.1, 30.0)
EPOCH_STYLE = {"E3": ("-", 1.0), "E2": ((0, (3, 1.5)), 0.8), "E1": ((0, (1, 1.2)), 0.8)}


def band_edges(labels) -> list[tuple[float, float]]:
    out = []
    for b in labels:
        lo, hi = (float(v) for v in b.strip("(]").split(","))
        out.append((lo, hi))
    return out


def main() -> int:
    import cmocean
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle

    m = C.metrics()
    min_n = int(m["params.h1_min_block_n"]["value"])
    ext = pd.read_csv(HC / "extent.csv")
    dom = ext[(ext["sector"] == "domain") & (ext["area_obs_km2"] > 0)].copy()
    dom["bias_pct"] = 100.0 * dom["area_bias_rel"]
    bands = pd.read_csv(HC / "displacement_by_level_band.csv")
    bands = bands[bands["variant"] == "shape"]
    blk = pd.read_csv(HC / "block_h1.csv")
    blk = blk[(blk["variant"] == "shape") & (blk["n"] >= min_n)]
    det = pd.read_csv(HC / "h1_mispredicts_detail.csv")
    det = det[(det["variant"] == "shape") & (det["evaluable"] == True)]  # noqa: E712
    mis = pd.read_csv(HC / "h1_mispredicts.csv")
    mis = mis[mis["variant"] == "shape"].set_index(["product", "sector"])["mispredicts"]

    proj, _ = C.projection()
    domain, sectors = C.study_geometries()
    extent = C.map_extent(domain)
    sea, sea_ext = C.sea_mask()

    C.style()
    W = C.DOUBLE_MM
    MAP_W = 80.0
    map_h = MAP_W * (extent[3] - extent[2]) / (extent[1] - extent[0])
    TOP, H_A, GAP1, GAP2, H_D, BOTTOM = 11.0, 50.0, 13.0, 15.0, 36.0, 12.0
    H = TOP + H_A + GAP1 + map_h + GAP2 + H_D + BOTTOM
    fig = plt.figure(figsize=(W * C.MM, H * C.MM))
    ya = (H - TOP - H_A) / H
    ax_a1 = fig.add_axes([12.0 / W, ya + (H_A * 0.53) / H, 72.0 / W, H_A * 0.47 / H])
    ax_a2 = fig.add_axes([12.0 / W, ya, 72.0 / W, H_A * 0.47 / H], sharex=ax_a1)
    ax_b = fig.add_axes([104.0 / W, ya, 73.0 / W, H_A / H])
    yc = (BOTTOM + H_D + GAP2) / H
    ax_c = [fig.add_axes([6.0 / W, yc, MAP_W / W, map_h / H], projection=proj),
            fig.add_axes([(6.0 + MAP_W + 4.0) / W, yc, MAP_W / W, map_h / H], projection=proj)]
    cax_c = fig.add_axes([(6.0 + 2 * MAP_W + 6.5) / W, yc + 0.1 * map_h / H, 1.8 / W, 0.8 * map_h / H])
    ax_d = fig.add_axes([24.0 / W, BOTTOM / H, 153.0 / W, H_D / H])

    # a
    for ax, variant, title in ((ax_a1, "shape", "Shape variant (fitted constant)"),
                               (ax_a2, "total", "Total variant (geoid conversion, no constant)")):
        d = dom[dom["variant"] == variant]
        for p in C.PRODUCTS:
            for ep, (ls, lw) in EPOCH_STYLE.items():
                g = d[(d["product"] == p) & (d["epoch"] == ep)].sort_values("level_m")
                ax.plot(g["level_m"], g["bias_pct"], ls=ls, lw=lw, color=C.PRODUCT_COLOURS[p], marker=C.PRODUCT_MARKERS[p],
                        ms=2.0, mew=0, zorder=3)
        ax.axhline(0, color=plotting.INK_MUTED, lw=0.5, zorder=1)
        ax.set_ylabel("Area bias (%)")
        ax.text(0.01, 0.97, title, transform=ax.transAxes, ha="left", va="top", fontsize=5.5, color=plotting.INK)
        ax.grid(axis="y", color=plotting.GRID, lw=0.4)
        ax.set_axisbelow(True)
    ax_a1.tick_params(labelbottom=False)
    ax_a1.set_ylim(-55, 40)
    ax_a2.set_ylim(-20, 70)
    ax_a2.set_xlabel("Level (m, DAHITI reference)")
    ax_a1.legend(handles=[Line2D([], [], color=C.PRODUCT_COLOURS[p], marker=C.PRODUCT_MARKERS[p], ms=2.5, mew=0, lw=0.8,
                                 label=C.PRODUCT_NAMES[p]) for p in C.PRODUCTS] +
                 [Line2D([], [], color=plotting.INK_MUTED, ls=EPOCH_STYLE[e][0], lw=0.8, label=e) for e in EPOCH_STYLE],
                 loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=4, fontsize=5.5, handlelength=1.6,
                 columnspacing=0.8, borderaxespad=0.1)
    plotting.panel_label(ax_a1, "a", x=-0.15, y=1.25)

    # b
    edges = band_edges(bands["band"])
    order = np.argsort([lo for lo, _ in edges])
    for p in C.PRODUCTS:
        vals = bands[p].to_numpy(float) / 1000.0
        for k in order:
            lo, hi = edges[k]
            ax_b.plot([lo, hi], [vals[k]] * 2, color=C.PRODUCT_COLOURS[p], lw=1.4, solid_capstyle="butt", zorder=3)
            ax_b.plot((lo + hi) / 2, vals[k], C.PRODUCT_MARKERS[p], ms=3, mfc=C.PRODUCT_COLOURS[p], mec="white", mew=0.4,
                      zorder=4)
    split = float(m["h1.disp_band.edge_2_m"]["value"])
    ax_b.axvline(split, color=plotting.INK_MUTED, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax_b.text(split + 0.04, 0.2, f"{split:.1f} m".replace("-", "−"), fontsize=5.5, color=plotting.INK_MUTED,
              ha="left", va="bottom")
    for lo, hi in edges:
        ax_b.axvline(lo, color=plotting.GRID, lw=0.4, zorder=0)
    ax_b.set_yscale("log")
    ax_b.set_ylim(0.15, 30)
    ax_b.set_yticks([0.3, 1, 3, 10], ["0.3", "1", "3", "10"])
    ax_b.yaxis.set_minor_locator(plt.NullLocator())
    ax_b.set_xlim(min(e[0] for e in edges), max(e[1] for e in edges))
    ax_b.set_xlabel("Level band (m, DAHITI reference)")
    ax_b.set_ylabel("Median shoreline displacement (km)")
    ax_b.text(0.99, 0.97, "Shape variant; lower levels emerged later", transform=ax_b.transAxes, ha="right", va="top",
              fontsize=5.5, color=plotting.INK)
    plotting.panel_label(ax_b, "b", x=-0.17, y=1.02)

    # c: block maps
    cmap = cmocean.cm.matter
    norm = LogNorm(*DISP_RANGE_KM)
    for ax, p in zip(ax_c, MAP_PRODUCTS):
        ax.set_extent(extent, crs=proj)
        ax.set_facecolor("white")
        ax.imshow(np.where(sea, 1.0, np.nan), extent=sea_ext, origin="upper", transform=proj,
                  cmap=plt.matplotlib.colors.ListedColormap([C.SEA]), interpolation="nearest", zorder=1,
                  rasterized=True)
        g = blk[blk["product"] == p]
        for r in g.itertuples():
            xy = (r.block_i * 25_000.0, r.block_j * 25_000.0)
            if r.median_abs_m < 0:     # not a distance (see the docstring): undefined, hatched
                ax.add_patch(Rectangle(xy, 25_000.0, 25_000.0, transform=proj, fc="white", ec=plotting.INK_MUTED,
                                       hatch="//////", lw=0.3, zorder=3))
                continue
            v = np.clip(r.median_abs_m / 1000.0, *DISP_RANGE_KM)
            ax.add_patch(Rectangle(xy, 25_000.0, 25_000.0, transform=proj, fc=cmap(norm(v)), ec="white", lw=0.3,
                                   zorder=3))
        C.draw_outlines(ax, proj, domain, sectors, lw_sector=0.35, lw_domain=0.7)
        C.graticule(ax, size=5)
        ax.text(0.015, 0.975, f"{C.PRODUCT_NAMES[p]}  ({len(g)} blocks)", transform=ax.transAxes, ha="left", va="top",
                fontsize=6, color=plotting.INK, fontweight="bold", zorder=10,
                bbox={"boxstyle": "square,pad=0.15", "fc": "white", "ec": "none"})
        C.scale_bar(ax, extent[0] + 14_000, extent[2] + 14_000, C.SCALE_KM * 1000, f"{C.SCALE_KM} km", transform=proj,
                    lw=1.0)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=cax_c, extend="both")
    cb.set_label("Block median displacement (km)", fontsize=6)
    cb.set_ticks([0.1, 0.3, 1, 3, 10, 30], labels=["0.1", "0.3", "1", "3", "10", "30"])
    cb.ax.yaxis.set_minor_locator(plt.NullLocator())
    cb.ax.tick_params(labelsize=5.5, length=2, width=0.4)
    cb.outline.set_linewidth(0.4)
    plotting.panel_label(ax_c[0], "c", x=-0.06, y=1.0)
    ax_c[1].legend(handles=[Patch(fc="white", ec=plotting.INK_MUTED, hatch="//////", lw=0.3,
                                  label="Undefined in block_h1.csv")], loc="lower right", bbox_to_anchor=(1.0, -0.005),
                   fontsize=5.5, handlelength=1.2, frameon=True, framealpha=1.0, edgecolor="none")

    # d: sector summary
    xs = np.arange(len(SECTOR_ORDER))
    offs = np.linspace(-0.28, 0.28, len(C.PRODUCTS))
    for i, s in enumerate(SECTOR_ORDER):
        g = det[det["sector"] == s]
        noise = float(g.groupby("product")["p95_noise"].median().median()) / 1000.0
        ax_d.add_patch(Rectangle((i - 0.4, 0.105), 0.8, noise - 0.105, fc="#e6e6e6", ec="none", zorder=1))
        ax_d.plot([i - 0.4, i + 0.4], [noise, noise], color=plotting.INK_MUTED, lw=0.8, zorder=2)
        for p, dx in zip(C.PRODUCTS, offs):
            v = float(g.loc[g["product"] == p, "median"].median()) / 1000.0
            filled = bool(mis.get((p, s), False))
            ax_d.plot(i + dx, v, C.PRODUCT_MARKERS[p], ms=4, mfc=C.PRODUCT_COLOURS[p] if filled else "white",
                      mec=C.PRODUCT_COLOURS[p], mew=0.8, zorder=4)
    ax_d.set_yscale("log")
    ax_d.set_ylim(0.1, 30)
    ax_d.set_yticks([0.1, 0.3, 1, 3, 10, 30], ["0.1", "0.3", "1", "3", "10", "30"])
    ax_d.yaxis.set_minor_locator(plt.NullLocator())
    ax_d.set_xlim(-0.5, len(SECTOR_ORDER) - 0.5)
    ax_d.set_xticks(xs, [C.SECTOR_TINY[s] for s in SECTOR_ORDER])
    ax_d.tick_params(axis="x", length=0)
    ax_d.set_ylabel("Median displacement (km)")
    ax_d.grid(axis="y", color=plotting.GRID, lw=0.4)
    ax_d.set_axisbelow(True)
    ax_d.legend(handles=[Line2D([], [], ls="none", marker=C.PRODUCT_MARKERS[p], ms=4, mfc=C.PRODUCT_COLOURS[p],
                                mec=C.PRODUCT_COLOURS[p], label=C.PRODUCT_NAMES[p]) for p in C.PRODUCTS] +
                [Line2D([], [], ls="none", marker="o", ms=4, mfc="white", mec=plotting.INK, mew=0.8,
                        label="open: does not mispredict"),
                 Patch(fc="#e6e6e6", ec=plotting.INK_MUTED, lw=0.5, label="Noise floor (95th percentile)")],
                loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=7, fontsize=5.5, handletextpad=0.2,
                columnspacing=0.8, borderaxespad=0.1)
    plotting.panel_label(ax_d, "d", x=-0.1, y=1.08)

    C.save(fig, "fig4_audit")
    plt.close(fig)
    print("units:", dom[["epoch", "bin"]].drop_duplicates().shape[0], "| blocks drawn:",
          {p: int((blk["product"] == p).sum()) for p in MAP_PRODUCTS},
          "| undefined blocks drawn:", {p: int(((blk["product"] == p) & (blk["median_abs_m"] < 0)).sum()) for p in MAP_PRODUCTS},
          "| defined median range km:", round(blk.loc[blk["median_abs_m"] >= 0, "median_abs_m"].min() / 1000, 3),
          round(blk["median_abs_m"].max() / 1000, 1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
