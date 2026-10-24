"""RSE figure: accuracy of the waterline DEM against ICESat-2, by sector (double column).

a, The waterline DEM (data/processed/dem/main/dem_*.tif band 1, height on the DAHITI
   reference; 30 m, averaged to a 150 m display grid) inside the emerged band (water in
   the 1995-96 and not water in the 2025-26 zone-v6 extent), the 50 km test blocks that hold
   ICESat-2 points of the H2 population (results/validation/main/ATL08_accuracy_blocks.csv),
   sector outlines and names.
b, DEM height against the ICESat-2 ATL08 reference height (orthometric height plus the
   training-block datum constant, results/validation/main/ATL08_datum_constant.json) for
   every test-block point of the H2 population (figures/_data.icesat2_test, checked
   against ATL08_accuracy.csv); statistics from results/metrics.json.
c, Per sector (validated first) and over all test blocks: the DEM's mean and median error
   against ATL08 and ATL06 (DEM minus ICESat-2), the quantities of the validated-sector
   rule (PREREG Amendment 18 item 1: all four within +-0.5 m, shaded), and the mean error
   of each grid against ATL08 (grid heights with the training-block offset of
   ATL08_product_offsets.json, the "matched" variant). Values beyond the axis are drawn on
   its edge and labelled.
d, NMAD (1.4826 x median absolute deviation from the median) of the DEM and of each grid
   against ATL08, same points; logarithmic axis. Right: the number of ATL08 points and of
   50 km test blocks per row (ATL08_accuracy.csv).

Output: figures/out/rse_fig_dem_accuracy.{pdf,tif,png}
Usage: python figures/rse_fig_dem_accuracy.py
"""

from __future__ import annotations

import sys

import _common as C
import _data as D
import _rse as R
import numpy as np
import pandas as pd

from caspian.config import RESULTS

VAL = RESULTS / "validation" / "main"
LEVEL_RANGE = (-29.0, -25.6)
HEX_LIM = (-30.5, -24.5)
BIAS_LIM = (-2.3, 2.3)
TESTC = "#D55E00"
CRIT = "#e6eef5"
LABELS = {"west": (46.95, 45.17), "volga": (47.75, 46.62), "interfluve": (50.66, 46.72),
          "ural": (51.72, 46.6), "northeast": (53.45, 46.62), "komsomolets": (53.95, 44.72),
          "open_shelf": (49.55, 45.05)}


def main() -> int:
    import cmcrameri.cm as cmc
    import cmocean
    import matplotlib.pyplot as plt
    from matplotlib import patheffects
    from matplotlib.colors import ListedColormap, LogNorm, Normalize
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle

    from caspian import validation

    m = C.metrics()
    proj, fwd = C.projection()
    domain, sectors = C.study_geometries()
    extent = C.map_extent(domain)
    dem = D.mosaic("dem", float(C.MAP_RES_M))
    band = D.emerged_band(dem["extent"], dem["z"].shape)
    z = np.where(band, dem["z"], np.nan)
    sea, sea_ext = C.sea_mask()
    pts = D.icesat2_test("ATL08")
    e = pts["dem_h"].to_numpy(float) - pts["y_ref"].to_numpy(float)
    acc = {s: pd.read_csv(VAL / f"{s}_accuracy.csv") for s in ("ATL08", "ATL06")}
    blocks = pd.read_csv(VAL / "ATL08_accuracy_blocks.csv")
    blocks = blocks[blocks["model"] == "dem"]
    if sorted(blocks["block"]) != sorted(pts["block"].unique()):
        raise SystemExit("test blocks of the points differ from ATL08_accuracy_blocks.csv")

    R.style()
    W = R.DOUBLE_MM
    L, RGT, TOP = 8.5, 2.0, 2.0
    MAP_W = 118.0
    map_h = MAP_W * (extent[3] - extent[2]) / (extent[1] - extent[0])
    G1, ROW2, BOT = 13.0, 62.0, 10.0
    H = TOP + map_h + G1 + ROW2 + BOT
    fig = plt.figure(figsize=(W * C.MM, H * C.MM))

    # a, DEM map ----------------------------------------------------------------------------------
    ax = fig.add_axes([L / W, (H - TOP - map_h) / H, MAP_W / W, map_h / H], projection=proj)
    ax.set_extent(extent, crs=proj)
    ax.set_facecolor(R.LAND)
    ax.imshow(np.where(sea, 1.0, np.nan), extent=sea_ext, origin="upper", transform=proj,
              cmap=ListedColormap([R.SEA]), interpolation="nearest", zorder=1, rasterized=True)
    ex = dem["extent"]
    im = ax.imshow(np.ma.masked_invalid(z), extent=[ex[0], ex[1], ex[2], ex[3]], origin="upper", transform=proj,
                   cmap=cmocean.cm.deep_r, norm=Normalize(*LEVEL_RANGE), interpolation="nearest", zorder=2,
                   rasterized=True)
    gl = C.graticule(ax, size=R.FS_SMALL)
    gl.xlabel_style = gl.ylabel_style = {"size": R.FS_SMALL, "color": R.MUTED}
    C.draw_outlines(ax, proj, domain, sectors, lw_sector=0.45, lw_domain=0.8)
    for b in blocks["block"]:
        i, j = (int(v) for v in b.split("_"))
        ax.add_patch(Rectangle((i * 50_000.0, j * 50_000.0), 50_000.0, 50_000.0, transform=proj, fc="none", ec=TESTC,
                               lw=0.9, zorder=7))
    for sid, (lon, lat) in LABELS.items():
        x, y = fwd(lon, lat)
        ax.text(x, y, R.NAME2[sid], transform=proj, ha="center", va="center", fontsize=R.FS_SMALL,
                fontweight="bold" if sid in R.VALIDATED else "normal", style="normal" if sid in R.VALIDATED else "italic",
                color=R.INK, zorder=12, linespacing=1.0, multialignment="center",
                bbox={"boxstyle": "round,pad=0.2,rounding_size=0.12", "fc": "white", "ec": "none", "alpha": 0.85})
    R.scale_bar(ax, extent[0] + 18_000, extent[2] + 15_000, C.SCALE_KM * 1000.0,
                R.num(m, "params.fig_scale_bar_km", "km"), transform=proj)
    R.label(ax, "a", inside=True)
    # colour bar and key under the hexbin column
    xb0 = L + MAP_W + 14.0
    cax = fig.add_axes([xb0 / W, (H - TOP - map_h + 2.0) / H, 44.0 / W, 1.8 / H])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("DEM height in a (m, DAHITI reference)", fontsize=R.FS_SMALL, labelpad=1.5)
    cb.ax.xaxis.set_label_position("top")
    cb.ax.tick_params(labelsize=R.FS_SMALL, length=2, width=0.4, pad=1.2)
    cb.outline.set_linewidth(0.4)

    # b, hexbin -------------------------------------------------------------------------------------
    HB = 36.0
    ax_b = R.mm_axes(fig, W, H, xb0, TOP + 3.0, HB, HB)
    cax_b = R.mm_axes(fig, W, H, xb0 + HB + 1.5, TOP + 3.0, 1.6, HB)
    x, y = pts["y_ref"].to_numpy(float), pts["dem_h"].to_numpy(float)
    inside = (x >= HEX_LIM[0]) & (x <= HEX_LIM[1]) & (y >= HEX_LIM[0]) & (y <= HEX_LIM[1])
    hb = ax_b.hexbin(x[inside], y[inside], gridsize=60, extent=(*HEX_LIM, *HEX_LIM), cmap=cmc.devon_r,
                     norm=LogNorm(vmin=1), mincnt=1, linewidths=0, rasterized=True)
    ax_b.plot(HEX_LIM, HEX_LIM, color=R.INK, lw=0.6, ls=(0, (3, 2)), zorder=3)
    ax_b.set_xlim(HEX_LIM)
    ax_b.set_ylim(HEX_LIM)
    ax_b.set_aspect("equal")
    ax_b.set_xticks([-30, -28, -26])
    ax_b.set_yticks([-30, -28, -26])
    ax_b.set_xlabel("ICESat-2 ATL08 height (m)")
    ax_b.set_ylabel("Waterline DEM height (m)")
    for sp in ("top", "right"):
        ax_b.spines[sp].set_visible(True)
    cb = fig.colorbar(hb, cax=cax_b)
    cb.set_label("Points per hexagon", fontsize=R.FS_SMALL, labelpad=1.5)
    cb.set_ticks([1, 10, 100, 1000], labels=["1", "10", "100", "1,000"])
    cb.ax.yaxis.set_minor_locator(plt.NullLocator())
    cb.ax.tick_params(labelsize=R.FS_SMALL, length=2, width=0.4)
    cb.outline.set_linewidth(0.4)
    n_out = int((~inside).sum())
    txt = (f"n = {R.num(m, 'h2.atl08.n_test_points')} points, {R.num(m, 'h2.atl08.n_test_blocks')} blocks\n"
           f"mean error {R.num(m, 'h2.atl08.dem.bias_m', 'm')}\nNMAD {R.num(m, 'h2.atl08.dem.nmad_m', 'm')}, "
           f"RMSE {R.num(m, 'h2.atl08.dem.rmse_m', 'm')}\n{R.fmt_int(n_out)} points beyond the axes")
    ax_b.text(0.97, 0.03, txt, transform=ax_b.transAxes, ha="right", va="bottom", fontsize=R.FS_SMALL, color=R.INK,
              linespacing=1.1)
    R.label(ax_b, "b", x=-0.36, y=1.0)
    if abs(validation.nmad(e) - R.value(m, "h2.atl08.dem.nmad_m")) > 0.0006:
        raise SystemExit("NMAD of the drawn points differs from h2.atl08.dem.nmad_m")

    # c, d: per-sector rows ---------------------------------------------------------------------------
    rows = list(R.VALIDATED) + list(R.NOT_VALIDATED) + ["overall"]
    ypos = {s: len(rows) - 1 - k for k, s in enumerate(rows)}
    a8 = acc["ATL08"].set_index(["scope", "model"])
    a6 = acc["ATL06"].set_index(["scope", "model"])
    y2 = TOP + map_h + G1
    LAB_W = 33.0
    cw, dw, tw = 62.0, 50.0, 22.0
    ax_c = R.mm_axes(fig, W, H, L + LAB_W, y2, cw, ROW2)
    ax_d = R.mm_axes(fig, W, H, L + LAB_W + cw + 5.0, y2, dw, ROW2)
    ax_c.axvspan(-0.5, 0.5, color=CRIT, lw=0, zorder=0)
    ax_c.axvline(0, color=R.MUTED, lw=0.5, zorder=1)
    goffs = np.linspace(-0.10, -0.36, len(R.PRODUCTS))
    for s in rows:
        yy = ypos[s]
        if s != "overall" and int(a8.loc[(s, "dem"), "n"]) == 0:
            for a_ in (ax_c, ax_d):
                a_.text(0.5, yy, "no test block", transform=a_.get_yaxis_transform(), ha="center", va="center",
                        fontsize=R.FS_SMALL, color=R.MUTED)
            continue
        # DEM: ATL08 mean (filled) and median (open) circles; ATL06 squares, grey
        for tab, dy, mk, col in ((a8, 0.30, "o", R.INK), (a6, 0.13, "s", "#6b6b6b")):
            r = tab.loc[(s, "dem")]
            ax_c.plot(r["bias"], yy + dy, mk, ms=3.6, mfc=col, mec=col, mew=0.7, zorder=4)
            ax_c.plot(r["median"], yy + dy, mk, ms=3.6, mfc="white", mec=col, mew=0.7, zorder=4)
        for p, dy in zip(R.PRODUCTS, goffs):
            r = a8.loc[(s, f"{p}|matched")]
            v = float(r["bias"])
            vc = float(np.clip(v, *BIAS_LIM))
            ax_c.plot(vc, yy + dy, R.MARKERS[p], ms=2.8, mfc=R.COLOURS[p], mec="none", zorder=3, clip_on=False)
            if v != vc:
                ax_c.text(vc + (0.12 if v > 0 else -0.12) * -1, yy + dy, f"{v:.1f}".replace("-", "−"),
                          ha="left" if v < 0 else "right", va="center", fontsize=R.FS_SMALL, color=R.COLOURS[p])
        # NMAD
        ax_d.plot(float(a8.loc[(s, "dem"), "nmad"]), yy + 0.22, "o", ms=3.6, mfc=R.INK, mec=R.INK, zorder=4)
        for p, dy in zip(R.PRODUCTS, goffs):
            ax_d.plot(float(a8.loc[(s, f"{p}|matched"), "nmad"]), yy + dy + 0.04, R.MARKERS[p], ms=2.8,
                      mfc=R.COLOURS[p], mec="none", zorder=3)
        r = a8.loc[(s, "dem")]
        ax_d.text(1.04, yy - 0.05, R.fmt_int(r["n"]), transform=ax_d.get_yaxis_transform(), ha="left",
                  va="center", fontsize=R.FS_SMALL, color=R.INK)
        ax_d.text(1.04 + tw * 0.62 / dw, yy - 0.05, f"{int(r['n_blocks'])}", transform=ax_d.get_yaxis_transform(),
                  ha="right", va="center", fontsize=R.FS_SMALL, color=R.INK)
    ax_d.text(1.04, len(rows) - 0.35, "ATL08\npoints", transform=ax_d.get_yaxis_transform(), ha="left", va="bottom",
              fontsize=R.FS_SMALL, color=R.INK, linespacing=1.0)
    ax_d.text(1.04 + tw * 0.62 / dw, len(rows) - 0.35, "test\nblocks", transform=ax_d.get_yaxis_transform(),
              ha="right", va="bottom", fontsize=R.FS_SMALL, color=R.INK, linespacing=1.0, multialignment="right")
    names = {**R.NAME, "overall": "All test blocks"}
    for a_ in (ax_c, ax_d):
        a_.set_ylim(-0.6, len(rows) - 0.45)
        a_.axhline(0.5, color=R.MUTED, lw=0.5, zorder=1)
        a_.axhline(len(R.NOT_VALIDATED) + 0.5, color=R.GRID, lw=0.8, zorder=1)
        a_.grid(axis="x", color=R.GRID, lw=0.4)
        a_.set_axisbelow(True)
        a_.tick_params(axis="y", length=0)
    ax_c.set_yticks([ypos[s] for s in rows], [names[s] for s in rows])
    ax_d.set_yticks([ypos[s] for s in rows], [""] * len(rows))
    for s in rows:
        if s in R.VALIDATED:
            ax_c.get_yticklabels()[rows.index(s)].set_fontweight("bold")
    ax_c.tick_params(axis="y", labelcolor=R.INK)
    ax_c.set_xlim(*BIAS_LIM)
    ax_c.set_xticks([-2, -1, -0.5, 0, 0.5, 1, 2], ["−2", "−1", "−0.5", "0", "0.5", "1", "2"])
    ax_c.set_xlabel("Error, model − ICESat-2 (m)")
    ax_d.set_xscale("log")
    ax_d.set_xlim(0.07, 3.0)
    ax_d.set_xticks([0.1, 0.3, 1, 3], ["0.1", "0.3", "1", "3"])
    ax_d.xaxis.set_minor_locator(plt.NullLocator())
    ax_d.set_xlabel("NMAD against ATL08 (m)")
    R.label(ax_c, "c", x=-0.58, y=1.0)
    R.label(ax_d, "d", x=-0.06, y=1.0)
    # key for c and d, above the rows
    handles = [Line2D([], [], ls="none", marker="o", ms=3.6, mfc=R.INK, mec=R.INK, label="DEM, ATL08 mean"),
               Line2D([], [], ls="none", marker="o", ms=3.6, mfc="white", mec=R.INK, label="DEM, ATL08 median"),
               Line2D([], [], ls="none", marker="s", ms=3.6, mfc="#6b6b6b", mec="#6b6b6b", label="DEM, ATL06 mean"),
               Line2D([], [], ls="none", marker="s", ms=3.6, mfc="white", mec="#6b6b6b", label="DEM, ATL06 median"),
               Patch(fc=CRIT, ec="none", label="±0.5 m (checked-sector rule)")]
    handles += [Line2D([], [], ls="none", marker=R.MARKERS[p], ms=3.0, mfc=R.COLOURS[p], mec="none",
                       label=f"{R.NAMES[p]}, ATL08") for p in R.PRODUCTS]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=((L + 0.5) / W, 1 - (y2 - 10.5) / H), ncol=5,
               frameon=False, borderaxespad=0, fontsize=R.FS_SMALL, handletextpad=0.3, columnspacing=1.0,
               labelspacing=0.3, handlelength=1.0)
    del patheffects
    R.save(fig, "rse_fig_dem_accuracy", W, H)
    plt.close(fig)
    print("test blocks", len(blocks), "| points beyond the hexbin axes", n_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
