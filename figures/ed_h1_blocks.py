"""Supplementary Figures: where each product misplaces the historic shorelines (H1).

Per product, the median absolute shoreline displacement per 25 km block pooled over
every (epoch, level) unit (results/hindcast/main/block_h1.csv), for the shape
variant (ed_h1_blocks_shape) and the total variant (ed_h1_blocks_total). Blocks with
fewer than params.h1_min_block_n distances are not drawn; rows whose committed median
is negative (not a distance; see figures/fig4_audit.py) are hatched as undefined.
Colours as in Fig. 4c.

Output: figures/out/ed_h1_blocks_{shape,total}.{pdf,tif,png}
Usage: python figures/ed_h1_blocks.py
"""

from __future__ import annotations

import sys

import _common as C
import numpy as np
import pandas as pd

from caspian import plotting
from caspian.config import RESULTS

DISP_RANGE_KM = (0.1, 30.0)   # as Fig. 4c


def draw(variant: str, blk: pd.DataFrame, min_n: int) -> None:
    import cmocean
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Patch, Rectangle

    proj, _ = C.projection()
    domain, sectors = C.study_geometries()
    extent = C.map_extent(domain)
    sea, sea_ext = C.sea_mask()
    cmap, norm = cmocean.cm.matter, LogNorm(*DISP_RANGE_KM)
    W = C.DOUBLE_MM
    MW, GX, GY, L, B, T = 56.0, 3.5, 9.0, 5.0, 4.0, 3.0
    mh = MW * (extent[3] - extent[2]) / (extent[1] - extent[0])
    H = B + 2 * mh + GY + T
    fig = plt.figure(figsize=(W * C.MM, H * C.MM))
    pos = [(0, 1), (1, 1), (2, 1), (0, 0), (1, 0)]
    for (cx, cy), p, letter in zip(pos, C.PRODUCTS, "abcde"):
        ax = fig.add_axes([(L + cx * (MW + GX)) / W, (B + cy * (mh + GY)) / H, MW / W, mh / H], projection=proj)
        ax.set_extent(extent, crs=proj)
        ax.imshow(np.where(sea, 1.0, np.nan), extent=sea_ext, origin="upper", transform=proj,
                  cmap=plt.matplotlib.colors.ListedColormap([C.SEA]), interpolation="nearest", zorder=1, rasterized=True)
        g = blk[(blk["product"] == p) & (blk["variant"] == variant) & (blk["n"] >= min_n)]
        for r in g.itertuples():
            xy = (r.block_i * 25_000.0, r.block_j * 25_000.0)
            if r.median_abs_m < 0:
                ax.add_patch(Rectangle(xy, 25_000.0, 25_000.0, transform=proj, fc="white", ec=plotting.INK_MUTED,
                                       hatch="//////", lw=0.3, zorder=3))
                continue
            v = np.clip(r.median_abs_m / 1000.0, *DISP_RANGE_KM)
            ax.add_patch(Rectangle(xy, 25_000.0, 25_000.0, transform=proj, fc=cmap(norm(v)), ec="white", lw=0.25,
                                   zorder=3))
        C.draw_outlines(ax, proj, domain, sectors, lw_sector=0.3, lw_domain=0.6)
        C.graticule(ax, labels=False)
        ax.text(0.02, 0.97, f"{C.PRODUCT_NAMES[p]} ({len(g)} blocks)", transform=ax.transAxes, ha="left", va="top",
                fontsize=5.5, fontweight="bold", zorder=10, bbox={"boxstyle": "square,pad=0.12", "fc": "white", "ec": "none"})
        C.scale_bar(ax, extent[0] + 12_000, extent[2] + 12_000, C.SCALE_KM * 1000, f"{C.SCALE_KM} km", transform=proj,
                    lw=0.9)
        plotting.panel_label(ax, letter, x=-0.06, y=1.0)
    cax = fig.add_axes([(L + 2 * (MW + GX) + 6.0) / W, (B + 0.55 * mh) / H, MW * 0.6 / W, 1.8 / H])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", extend="both")
    cb.set_ticks([0.1, 0.3, 1, 3, 10, 30], labels=["0.1", "0.3", "1", "3", "10", "30"])
    cb.ax.xaxis.set_minor_locator(plt.NullLocator())
    cb.set_label(f"Block median displacement (km), {variant} variant", fontsize=6)
    cb.ax.xaxis.set_label_position("top")
    cb.ax.tick_params(labelsize=5.5, length=2, width=0.4)
    cb.outline.set_linewidth(0.4)
    fig.legend(handles=[Patch(fc="white", ec=plotting.INK_MUTED, hatch="//////", lw=0.3, label="Undefined in block_h1.csv"),
                        Patch(fc=C.SEA, ec="none", label="Open sea, 2025–26")],
               loc="upper left", bbox_to_anchor=((L + 2 * (MW + GX) + 5.0) / W, (B + 0.4 * mh) / H), fontsize=5.5,
               handlelength=1.2, frameon=False)
    C.save(fig, f"ed_h1_blocks_{variant}")
    plt.close(fig)


def main() -> int:
    m = C.metrics()
    min_n = int(m["params.h1_min_block_n"]["value"])
    blk = pd.read_csv(RESULTS / "hindcast" / "main" / "block_h1.csv")
    C.style()
    for variant in ("shape", "total"):
        draw(variant, blk, min_n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
