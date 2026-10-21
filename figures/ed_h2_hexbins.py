"""Supplementary Figure: every model against ICESat-2 on the test blocks (H2).

Model height against the ICESat-2 reference height (H + c, the training-block constant
of results/validation/main/<source>_datum_constant.json) for every test-block point of
the H2 population (figures/_data.icesat2_test, checked against <source>_accuracy.csv):
the waterline DEM and the five products in the matched variant (product + its
training-block offset, <source>_product_offsets.json). Rows: ATL06, ATL08. The NMAD and
RMSE printed are those of results/validation/main/<source>_accuracy.csv.

Output: figures/out/ed_h2_hexbins.{pdf,tif,png}
Usage: python figures/ed_h2_hexbins.py
"""

from __future__ import annotations

import sys

import _common as C
import _data as D
import numpy as np
import pandas as pd

from caspian import plotting
from caspian.config import RESULTS

LIM = (-30.5, -24.5)   # as Fig. 3b


def main() -> int:
    import cmcrameri.cm as cmc
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    models = [("dem_h", "dem", "Waterline DEM")] + [(f"{p}|matched", f"{p}|matched", C.PRODUCT_NAMES[p]) for p in C.PRODUCTS]
    C.style()
    W, H = C.DOUBLE_MM, 76.0
    fig, axs = plt.subplots(2, 6, figsize=(W * C.MM, H * C.MM), sharex=True, sharey=True,
                            gridspec_kw={"wspace": 0.08, "hspace": 0.12, "left": 0.065, "right": 0.925,
                                         "bottom": 0.13, "top": 0.93})
    hb = None
    for row, src in enumerate(("ATL06", "ATL08")):
        d = D.icesat2_test(src)
        acc = pd.read_csv(RESULTS / "validation" / "main" / f"{src}_accuracy.csv")
        o = acc[acc["scope"] == "overall"].set_index("model")
        x = d["y_ref"].to_numpy(float)
        for col, (column, key, name) in enumerate(models):
            ax = axs[row, col]
            y = d[column].to_numpy(float)
            k = np.isfinite(x) & np.isfinite(y)
            hb = ax.hexbin(x[k], y[k], gridsize=45, extent=(*LIM, *LIM), cmap=cmc.devon_r, norm=LogNorm(vmin=1, vmax=1e5),
                           mincnt=1, linewidths=0, rasterized=True)
            ax.plot(LIM, LIM, color=plotting.INK, lw=0.5, ls=(0, (3, 2)))
            ax.set_xlim(LIM)
            ax.set_ylim(LIM)
            ax.set_aspect("equal")
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(True)
            ax.text(0.04, 0.96, f"NMAD {o.loc[key, 'nmad']:.2f} m\nRMSE {o.loc[key, 'rmse']:.2f} m", transform=ax.transAxes,
                    ha="left", va="top", fontsize=5, color=plotting.INK)
            if row == 0:
                ax.set_title(name, fontsize=6, pad=3)
            if col == 0:
                ax.set_ylabel(f"{src}, n = {int(o.loc['dem', 'n']):,}\nModel height (m)", fontsize=6)
            if row == 1:
                ax.set_xlabel(f"ICESat-2 (m)", fontsize=6)
            ax.set_xticks([-30, -28, -26])
            ax.set_yticks([-30, -28, -26])
    cax = fig.add_axes([0.94, 0.13, 0.01, 0.8])
    cb = fig.colorbar(hb, cax=cax)
    cb.set_label("Points per hexagon", fontsize=6)
    cb.set_ticks([1, 10, 100, 1000, 10000, 100000], labels=["1", "10", "100", "1,000", "10,000", "100,000"])
    cb.ax.yaxis.set_minor_locator(plt.NullLocator())
    cb.ax.tick_params(labelsize=5.5, length=2, width=0.4)
    cb.outline.set_linewidth(0.4)
    for ax, letter in zip(axs[:, 0], "ab"):
        plotting.panel_label(ax, letter, x=-0.47, y=1.03)
    C.save(fig, "ed_h2_hexbins")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
