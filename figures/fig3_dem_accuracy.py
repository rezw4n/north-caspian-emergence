"""Fig. 3: seabed topography from waterlines and its accuracy (H2).

a, the waterline DEM (data/processed/dem/main/dem_*.tif band 1, height on the DAHITI
   reference; 30 m, averaged to the MAP_RES_M display grid), inside the emerged
   band only (as Fig. 1a), with the 50 km test blocks that hold ICESat-2 points of the
   H2 population (results/study_area/blocks_50km.csv).
b, DEM height against the ICESat-2 ATL06 reference height (H + c, c the training-block
   constant; results/validation/main/ATL06_datum_constant.json) for every test-block
   point of the H2 population (figures/_data.icesat2_test, checked against
   ATL06_accuracy.csv). The statistics printed are those of ATL06_accuracy.csv.
c, DEM - ICESat-2 (ATL06) against interpolation distance (DEM band 2 at the point), in
   factor-of-two bins from 30 m (median, 25-75% and 5-95%); and, per sector, the NMAD
   and RMSE of the DEM and of the five products (matched variant) on the same test
   points (results/validation/main/ATL06_accuracy.csv).
d, empirical coverage of the 90% conformal intervals on the test blocks, overall and
   per sector, for ATL06 and ATL08 (results/validation/main/<source>_coverage.csv),
   with the test-block bootstrap 95% interval (boot_low, boot_high).

Ural has no test block at 50 km and so no sector value; its row says so.

Output: figures/out/fig3_dem_accuracy.{pdf,tif,png}
Usage: python figures/fig3_dem_accuracy.py
"""

from __future__ import annotations

import sys

import _common as C
import _data as D
import numpy as np
import pandas as pd

from caspian import plotting
from caspian.config import RESULTS
from caspian.geography import SECTOR_ORDER

LEVEL_RANGE = (-29.0, -25.6)        # as Fig. 2b
HEX_LIM = (-30.5, -24.5)            # axes of b (m)
DIST_EDGES_M = np.r_[0.0, 30.0 * 2.0 ** np.arange(0, 10)]   # 0, 30, 60, ..., 15,360 m
MIN_BIN_N = 1000                    # distance bins with fewer points are not drawn
VAL = RESULTS / "validation" / "main"


def main() -> int:
    import cmcrameri.cm as cmc
    import cmocean
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle

    from caspian import validation

    m = C.metrics()
    proj, _ = C.projection()
    domain, sectors = C.study_geometries()
    extent = C.map_extent(domain)
    dem = D.mosaic("dem", float(C.MAP_RES_M))
    band = D.emerged_band(dem["extent"], dem["z"].shape)
    z = np.where(band, dem["z"], np.nan)
    sea, sea_ext = C.sea_mask()
    blocks50 = pd.read_csv(RESULTS / "study_area" / "blocks_50km.csv")
    pts = D.icesat2_test("ATL06")
    e = pts["dem_h"].to_numpy(float) - pts["y_ref"].to_numpy(float)
    test_blocks = sorted(pts["block"].unique())
    acc = {s: pd.read_csv(VAL / f"{s}_accuracy.csv") for s in ("ATL06", "ATL08")}
    cov = {s: pd.read_csv(VAL / f"{s}_coverage.csv").set_index("scope") for s in ("ATL06", "ATL08")}

    C.style()
    W = C.DOUBLE_MM
    # row 1: map a (left) and hexbin b (right); row 2: c (distance; sector NMAD and RMSE) and d
    MAP_W = 108.0
    map_h = MAP_W * (extent[3] - extent[2]) / (extent[1] - extent[0])
    TOP, GAP, ROW2, BOTTOM = 4.0, 13.0, 47.0, 9.0
    H = TOP + map_h + GAP + ROW2 + BOTTOM
    fig = plt.figure(figsize=(W * C.MM, H * C.MM))
    y1 = (BOTTOM + ROW2 + GAP) / H
    ax_a = fig.add_axes([8.0 / W, y1, MAP_W / W, map_h / H], projection=proj)
    HB, XB = 38.0, 127.0
    yb = y1 + (map_h - HB - 1.0) / H
    ax_b = fig.add_axes([XB / W, yb, HB / W, HB / H])
    cax_b = fig.add_axes([(XB + HB + 1.5) / W, yb, 1.6 / W, HB / H])
    y2, h2 = BOTTOM / H, ROW2 / H
    ax_c1 = fig.add_axes([10.0 / W, y2, 40.0 / W, h2])
    ax_c2 = fig.add_axes([83.0 / W, y2, 28.0 / W, h2])
    ax_c3 = fig.add_axes([113.0 / W, y2, 28.0 / W, h2], sharey=ax_c2)
    ax_d = fig.add_axes([146.0 / W, y2, 32.0 / W, h2], sharey=ax_c2)

    # a: DEM map
    ax_a.set_extent(extent, crs=proj)
    ax_a.set_facecolor(C.LAND)
    ax_a.imshow(np.where(sea, 1.0, np.nan), extent=sea_ext, origin="upper", transform=proj,
                cmap=plt.matplotlib.colors.ListedColormap([C.SEA]), interpolation="nearest", zorder=1, rasterized=True)
    cmap = cmocean.cm.deep_r
    ex = dem["extent"]
    im = ax_a.imshow(np.ma.masked_invalid(z), extent=[ex[0], ex[1], ex[2], ex[3]], origin="upper", transform=proj,
                     cmap=cmap, norm=Normalize(*LEVEL_RANGE), interpolation="nearest", zorder=2, rasterized=True)
    C.graticule(ax_a)
    C.draw_outlines(ax_a, proj, domain, sectors, lw_sector=0.4, lw_domain=0.8)
    size = 50_000.0
    for b in test_blocks:
        i, j = (int(v) for v in b.split("_"))
        ax_a.add_patch(Rectangle((i * size, j * size), size, size, transform=proj, fc="none", ec="#D55E00", lw=0.8,
                                 zorder=7))
    C.scale_bar(ax_a, extent[0] + 18_000, extent[2] + 16_000, C.SCALE_KM * 1000, f"{C.SCALE_KM} km", transform=proj)
    cax_a = fig.add_axes([(XB - 3.0) / W, y1 + 11.0 / H, 40.0 / W, 1.6 / H])
    cb = fig.colorbar(im, cax=cax_a, orientation="horizontal")
    cb.set_label("a: DEM height (m, DAHITI reference)", fontsize=5.5, labelpad=1.5)
    cb.ax.xaxis.set_label_position("top")
    cb.ax.tick_params(labelsize=5.5, length=2, width=0.4, pad=1.2)
    cb.outline.set_linewidth(0.4)
    fig.legend(handles=[Patch(fc="none", ec="#D55E00", lw=0.8, label=f"a: test block ({size / 1000:.0f} km) with ICESat-2"),
                        Patch(fc=C.SEA, ec="none", label="a: open sea, 2025–26")],
               loc="lower left", bbox_to_anchor=((XB - 4.0) / W, y1 - 0.5 / H), fontsize=5.5, handlelength=1.2,
               frameon=False, ncol=1)
    plotting.panel_label(ax_a, "a", x=-0.06, y=0.985)

    # b: hexbin DEM vs ATL06
    x, y = pts["y_ref"].to_numpy(float), pts["dem_h"].to_numpy(float)
    inside = (x >= HEX_LIM[0]) & (x <= HEX_LIM[1]) & (y >= HEX_LIM[0]) & (y <= HEX_LIM[1])
    hb = ax_b.hexbin(x[inside], y[inside], gridsize=70, extent=(*HEX_LIM, *HEX_LIM), cmap=cmc.devon_r,
                     norm=LogNorm(vmin=1), mincnt=1, linewidths=0, rasterized=True)
    ax_b.plot(HEX_LIM, HEX_LIM, color=plotting.INK, lw=0.6, ls=(0, (3, 2)), zorder=3)
    ax_b.set_xlim(HEX_LIM)
    ax_b.set_ylim(HEX_LIM)
    ax_b.set_aspect("equal")
    ax_b.set_xlabel("ICESat-2 ATL06 height (m)")
    ax_b.set_ylabel("Waterline DEM height (m)")
    for sp in ("top", "right"):
        ax_b.spines[sp].set_visible(True)
    cb = fig.colorbar(hb, cax=cax_b)
    cb.set_label("Points per hexagon", fontsize=5.5)
    cb.set_ticks([1, 10, 100, 1000, 10000], labels=["1", "10", "100", "1,000", "10,000"])
    cb.ax.yaxis.set_minor_locator(plt.NullLocator())
    cb.ax.tick_params(labelsize=5.5, length=2, width=0.4)
    cb.outline.set_linewidth(0.4)
    n_out = int((~inside).sum())
    txt = (f"n = {m['h2.atl06.n_test_points']['value']:,} ({m['h2.atl06.n_test_blocks']['value']} test blocks)\n"
           f"bias {m['h2.atl06.dem.bias_m']['value']:+.2f} m, NMAD {m['h2.atl06.dem.nmad_m']['value']:.2f} m\n"
           f"RMSE {m['h2.atl06.dem.rmse_m']['value']:.2f} m\n"
           f"{n_out:,} points outside the axes").replace("-", "−")
    ax_b.text(0.97, 0.03, txt, transform=ax_b.transAxes, ha="right", va="bottom", fontsize=5.2, color=plotting.INK,
              linespacing=1.12)
    plotting.panel_label(ax_b, "b", x=-0.3, y=1.02)

    # c1: error vs interpolation distance
    dist = pts["interp_dist"].to_numpy(float)
    rows = []
    for lo, hi in zip(DIST_EDGES_M[:-1], DIST_EDGES_M[1:]):
        k = (dist > lo) & (dist <= hi) if lo > 0 else (dist <= hi)
        if k.sum() >= MIN_BIN_N:
            q = np.percentile(e[k], [5, 25, 50, 75, 95])
            rows.append((np.sqrt(max(lo, 15.0) * hi), *q, int(k.sum())))
    r = np.array(rows)
    ax_c1.fill_between(r[:, 0], r[:, 1], r[:, 5], color="#d9d9d9", lw=0, label="5–95%")
    ax_c1.fill_between(r[:, 0], r[:, 2], r[:, 4], color="#a6a6a6", lw=0, label="25–75%")
    ax_c1.plot(r[:, 0], r[:, 3], color=plotting.INK, lw=0.8, marker="o", ms=2.2, label="Median")
    ax_c1.axhline(0, color=plotting.INK_MUTED, lw=0.5, ls=(0, (2, 2)))
    ax_c1.set_xscale("log")
    ax_c1.set_xticks([30, 100, 300, 1000, 3000], ["30", "100", "300", "1,000", "3,000"])
    ax_c1.xaxis.set_minor_locator(plt.NullLocator())
    ax_c1.set_xlabel("Interpolation distance (m)")
    ax_c1.set_ylabel("DEM − ICESat-2 ATL06 (m)")
    ax_c1.legend(loc="upper left", fontsize=5.5, handlelength=1.2, ncol=3, columnspacing=0.8, borderaxespad=0.2)
    ax_c1.set_ylim(-3, 4)
    plotting.panel_label(ax_c1, "c", x=-0.25, y=1.02)

    # c2, c3, d: per-sector rows (shared y)
    scopes = ["overall"] + list(SECTOR_ORDER)
    ypos = {s: i for i, s in enumerate(scopes[::-1])}
    a6 = acc["ATL06"]
    nb = a6[a6.model == "dem"].set_index("scope")["n_blocks"]
    labels = {s: ("All test blocks" if s == "overall" else C.SECTOR_TINY[s]) + f" ({int(nb.get(s, 0))})" for s in scopes}
    models = [("dem", plotting.INK, "o", "Waterline DEM")] + \
             [(f"{p}|matched", C.PRODUCT_COLOURS[p], C.PRODUCT_MARKERS[p], C.PRODUCT_NAMES[p]) for p in C.PRODUCTS]
    offs = np.linspace(0.27, -0.27, len(models))
    for ax, metric, xl in ((ax_c2, "nmad", "NMAD (m)"), (ax_c3, "rmse", "RMSE (m)")):
        t = a6.set_index(["scope", "model"])
        for (mod, col, mk, _), dy in zip(models, offs):
            for s in scopes:
                if (s, mod) in t.index and t.loc[(s, mod), "n"] > 0:
                    ax.plot(t.loc[(s, mod), metric], ypos[s] + dy, mk, ms=2.8 if mod != "dem" else 3.4,
                            mfc=col, mec="white" if mod == "dem" else col, mew=0.3, zorder=4 if mod == "dem" else 3)
        ax.set_xscale("log")
        ax.set_xlim(0.08, 6)
        ax.set_xticks([0.1, 0.3, 1, 3], ["0.1", "0.3", "1", "3"])
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.set_xlabel(xl)
        ax.grid(axis="x", color=plotting.GRID, lw=0.4)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
    ax_c2.set_yticks([ypos[s] for s in scopes], [labels[s] for s in scopes])
    ax_c2.set_ylim(-0.6, len(scopes) - 0.4)
    plt.setp(ax_c3.get_yticklabels(), visible=False)
    plt.setp(ax_d.get_yticklabels(), visible=False)
    for ax in (ax_c2, ax_c3, ax_d):
        ax.axhline(len(scopes) - 1.5, color=plotting.GRID, lw=0.5)
        ax.text(0.5, ypos["ural"], "no test block", transform=ax.get_yaxis_transform(), ha="center", va="center",
                fontsize=5, color=plotting.INK_MUTED)
    ax_c2.legend(handles=[Line2D([], [], ls="none", marker=mk, ms=3, mfc=col, mec=col, label=lab)
                          for _, col, mk, lab in models], loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=3,
                 fontsize=5.5, handletextpad=0.2, columnspacing=0.6, borderaxespad=0.1)

    # d: coverage
    nominal = float(m["params.conformal_level"]["value"])
    for src, dy, mfc in (("ATL06", 0.12, plotting.INK), ("ATL08", -0.12, "white")):
        c = cov[src]
        for s in scopes:
            if s in c.index and c.loc[s, "n"] > 0:
                ax_d.plot([c.loc[s, "boot_low"], c.loc[s, "boot_high"]], [ypos[s] + dy] * 2, color=plotting.INK, lw=0.6,
                          solid_capstyle="butt", zorder=2)
                ax_d.plot(c.loc[s, "coverage"], ypos[s] + dy, "o", ms=3, mfc=mfc, mec=plotting.INK, mew=0.6, zorder=3)
    ax_d.axvline(nominal, color="#D55E00", lw=0.7, ls=(0, (3, 1.5)), zorder=1)
    ax_d.set_xlim(0, 1.02)
    ax_d.set_xticks([0, 0.5, nominal], ["0", "0.5", f"{nominal:.1f}"])
    ax_d.set_xlabel("Empirical coverage")
    ax_d.grid(axis="x", color=plotting.GRID, lw=0.4)
    ax_d.set_axisbelow(True)
    ax_d.tick_params(axis="y", length=0)
    ax_d.legend(handles=[Line2D([], [], ls="none", marker="o", ms=3, mfc=plotting.INK, mec=plotting.INK, label="ATL06"),
                         Line2D([], [], ls="none", marker="o", ms=3, mfc="white", mec=plotting.INK, label="ATL08"),
                         Line2D([], [], color="#D55E00", lw=0.7, ls=(0, (3, 1.5)), label="Nominal")],
                loc="lower left", bbox_to_anchor=(-0.04, 1.0), ncol=3, fontsize=5.5, handletextpad=0.2,
                columnspacing=0.5, handlelength=1.0, borderaxespad=0.1)
    plotting.panel_label(ax_d, "d", x=-0.12, y=1.11)

    C.save(fig, "fig3_dem_accuracy")
    plt.close(fig)
    print("test blocks", test_blocks, "| outside hexbin axes", n_out, "| distance bins", r[:, -1].astype(int).tolist(),
          "| check NMAD", round(validation.nmad(e), 4))
    return 0


if __name__ == "__main__":
    sys.exit(main())
