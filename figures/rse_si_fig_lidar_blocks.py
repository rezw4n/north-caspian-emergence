"""Supplementary figure of the journal revision (PREREG Amendment 20 item 7; D-162): where the
ICESat-2 support of the accuracy assessment lies.

One map (double column, 190 mm): every 50 km block that holds ICESat-2 ATL06 points on bare
emerged seabed (results/revision2/d/obsbias/obs_bias_blocks.csv, the all-block table of
Amendment 19 item 10; no value is computed here), drawn as a square
  - outlined by its split (training, calibration, test),
  - filled with the mean error of the waterline model in the block (model minus ICESat-2;
    cmocean balance, clipped at +-1.5 m),
  - labelled with its number of points, in thousands;
the sector outlines and names; the open sea of the 2025-26 low stand; graticule and scale bar.

Checked against results/metrics.json: the number of blocks, their split and the number of
points equal the ICESat-2 support table of Supplementary Section S11 (rev3.support.atl06.*).

Output: figures/out/rse_si_fig_lidar_blocks.{pdf,tif,png}
Usage: python figures/rse_si_fig_lidar_blocks.py
"""

from __future__ import annotations

import sys

import _common as C
import _rse as R
import numpy as np
import pandas as pd

from caspian.config import RESULTS

SOURCE = "ATL06"
SPLIT_COLOUR = {"train": "#0072B2", "calib": "#E69F00", "test": "#009E73"}       # Okabe-Ito
SPLIT_NAME = {"train": "training", "calib": "calibration", "test": "test"}
LIM_M = 1.5
BLOCK_M = 50_000.0
# Sector names sit on the edges between rows of blocks, clear of the numbers at the block centres
# (cartographic placement only; figures/_rse.check_overlap stops the script if a name meets a number).
LABELS = {"west": (47.12, 44.53, "Kizlyar–Dagestan\ncoast"), "volga": (47.3, 46.84, "Volga delta"),
          "interfluve": (50.66, 47.24, "Volga–Ural interfluve coast"), "ural": (51.78, 46.84, "Ural delta"),
          "northeast": (53.9, 47.2, "NE Kazakh coast"), "komsomolets": (54.1, 45.43, "Komsomolets Bay"),
          "open_shelf": (50.5, 44.53, "Open shelf")}


def thousands(n: int) -> str:
    """Points of a block in thousands: whole numbers from ten thousand, one decimal below."""
    k = n / 1000.0
    return f"{k:.0f}" if k >= 10 else (f"{k:.1f}" if k >= 0.05 else "<0.1")


def main() -> int:
    import cmocean
    import matplotlib.pyplot as plt
    from matplotlib import patheffects
    from matplotlib.colors import ListedColormap, Normalize
    from matplotlib.patches import Patch, Rectangle

    m = C.metrics()
    t = pd.read_csv(RESULTS / "revision2" / "d" / "obsbias" / "obs_bias_blocks.csv")
    t = t[t["source"] == SOURCE].reset_index(drop=True)
    want = {"n_blocks": R.value(m, "rev3.support.atl06.overall.n_blocks"),
            "n_points": R.value(m, "rev3.support.atl06.overall.n_points"),
            "train": R.value(m, "rev3.support.atl06.overall.n_blocks_train"),
            "calib": R.value(m, "rev3.support.atl06.overall.n_blocks_calib"),
            "test": R.value(m, "rev3.support.atl06.overall.n_blocks_test")}
    got = {"n_blocks": len(t), "n_points": int(t["n_points"].sum()),
           **{k: int((t["split"] == k).sum()) for k in SPLIT_COLOUR}}
    if any(int(want[k]) != got[k] for k in want):
        raise SystemExit(f"the block table {got} differs from the support table of Section S11 {want}")

    proj, fwd = C.projection()
    domain, sectors = C.study_geometries()
    extent = C.map_extent(domain)
    sea, sea_ext = C.sea_mask()

    R.style()
    W = R.DOUBLE_MM
    L, RGT, TOP, BOT = 8.5, 1.5, 2.0, 15.0
    map_w = W - L - RGT
    map_h = map_w * (extent[3] - extent[2]) / (extent[1] - extent[0])
    H = TOP + map_h + BOT
    fig = plt.figure(figsize=(W * C.MM, H * C.MM))
    ax = fig.add_axes([L / W, BOT / H, map_w / W, map_h / H], projection=proj)
    ax.set_extent(extent, crs=proj)
    ax.set_facecolor(R.LAND)
    ax.imshow(np.where(sea, 1.0, np.nan), extent=sea_ext, origin="upper", transform=proj,
              cmap=ListedColormap([R.SEA]), interpolation="nearest", zorder=1, rasterized=True)
    gl = C.graticule(ax, size=R.FS_SMALL)
    gl.xlabel_style = gl.ylabel_style = {"size": R.FS_SMALL, "color": R.MUTED}
    C.draw_outlines(ax, proj, domain, sectors, lw_sector=0.45, lw_domain=0.9, zorder=6)
    cmap, norm = cmocean.cm.balance, Normalize(-LIM_M, LIM_M)
    halo = [patheffects.withStroke(linewidth=1.6, foreground="white")]
    for r in t.itertuples():
        i, j = (int(v) for v in r.block.split("_"))
        x0, y0 = i * BLOCK_M, j * BLOCK_M
        ax.add_patch(Rectangle((x0, y0), BLOCK_M, BLOCK_M, transform=proj, fc=cmap(norm(r.mean)), ec="none", alpha=0.85,
                               zorder=3))
        # the outline is drawn inside the square, so that neighbours of different splits both show
        d = 2_200.0
        ax.add_patch(Rectangle((x0 + d, y0 + d), BLOCK_M - 2 * d, BLOCK_M - 2 * d, transform=proj, fc="none",
                               ec=SPLIT_COLOUR[r.split], lw=1.5, zorder=7))
        ax.text(x0 + BLOCK_M / 2, y0 + BLOCK_M / 2, thousands(int(r.n_points)), transform=proj, ha="center",
                va="center", fontsize=R.FS_SMALL, color=R.INK, path_effects=halo, zorder=9)
    for sid, (lon, lat, name) in LABELS.items():
        x, y = fwd(lon, lat)
        ax.text(x, y, name, transform=proj, ha="center", va="center", fontsize=R.FS_SMALL,
                fontweight="bold" if sid in R.VALIDATED else "normal", style="normal" if sid in R.VALIDATED else "italic",
                color=R.INK, zorder=12, linespacing=1.0, multialignment="center",
                bbox={"boxstyle": "round,pad=0.2,rounding_size=0.12", "fc": "white", "ec": "none", "alpha": 0.9})
    R.scale_bar(ax, extent[0] + 18_000, extent[2] + 15_000, C.SCALE_KM * 1000.0,
                R.num(m, "params.fig_scale_bar_km", "km"), transform=proj)

    # key under the map: colour bar on the left, the splits on the right
    cax = fig.add_axes([L / W, 5.2 / H, 62.0 / W, 2.0 / H])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", extend="both")
    cb.set_label("Mean error of the waterline model in the block (model minus ICESat-2, m)", fontsize=R.FS_SMALL,
                 labelpad=1.5)
    cb.ax.xaxis.set_label_position("top")
    cb.ax.tick_params(labelsize=R.FS_SMALL, length=2, width=0.4, pad=1.2)
    cb.outline.set_linewidth(0.4)
    handles = [Patch(fc="white", ec=SPLIT_COLOUR[k], lw=1.5,
                     label=f"{SPLIT_NAME[k].capitalize()} block ({got[k]})") for k in ("train", "calib", "test")]
    handles.append(Patch(fc=R.SEA, ec="none", label="Open sea, 2025–26 low stand"))
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=((L + 92.0) / W, 1.0 / H), ncol=2, frameon=False,
               fontsize=R.FS_SMALL, handlelength=1.5, handletextpad=0.6, columnspacing=2.0, labelspacing=0.5,
               borderaxespad=0, title="Number in a block: ICESat-2 ATL06 points, in thousands", title_fontsize=R.FS_SMALL,
               alignment="left")
    R.save(fig, "rse_si_fig_lidar_blocks", W, H)
    plt.close(fig)
    print(f"{SOURCE}: {got['n_blocks']} blocks ({got['train']} training, {got['calib']} calibration, {got['test']} test), "
          f"{got['n_points']} points | mean error {t['mean'].min():.2f} to {t['mean'].max():.2f} m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
