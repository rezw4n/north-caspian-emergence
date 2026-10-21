"""Supplementary Figure (brief section 10 check 9): the waterline DEM and the products
on bare against vegetated emerged seabed (D-084, D-088 addendum).

Test-block accuracy (scope "overall", matched variant for the products) of every model
on the H2 population (bare emerged seabed: E3 emergent-vegetation share below 0.5;
results/validation/main/<source>_accuracy.csv) and on the vegetated population (share
of at least 0.5; results/validation/main_vegetated/<source>_accuracy.csv), for ATL06
and ATL08: a, bias; b, NMAD; c, RMSE. ICESat-2 heights over dense reeds mix canopy and
ground, so the vegetated reference is itself uncertain.

Output: figures/out/ed_vegetation.{pdf,tif,png}
Usage: python figures/ed_vegetation.py
"""

from __future__ import annotations

import sys

import _common as C
import numpy as np
import pandas as pd

from caspian import plotting
from caspian.config import RESULTS

POPS = {"bare": RESULTS / "validation" / "main", "vegetated": RESULTS / "validation" / "main_vegetated"}


def main() -> int:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    models = [("dem", "Waterline DEM", plotting.INK, "o")] + \
             [(f"{p}|matched", C.PRODUCT_NAMES[p], C.PRODUCT_COLOURS[p], C.PRODUCT_MARKERS[p]) for p in C.PRODUCTS]
    C.style()
    W, H = C.DOUBLE_MM, 62.0
    fig, axs = plt.subplots(1, 3, figsize=(W * C.MM, H * C.MM), sharey=True,
                            gridspec_kw={"wspace": 0.12, "left": 0.1, "right": 0.99, "bottom": 0.18, "top": 0.78})
    ys = np.arange(len(models))[::-1]
    counts = {}
    for (src, dy) in (("ATL06", 0.14), ("ATL08", -0.14)):
        for pop, fill in (("bare", True), ("vegetated", False)):
            a = pd.read_csv(POPS[pop] / f"{src}_accuracy.csv")
            o = a[a["scope"] == "overall"].set_index("model")
            counts[(src, pop)] = (int(o.loc["dem", "n"]), int(o.loc["dem", "n_blocks"]))
            for ax, metric in zip(axs, ("bias", "nmad", "rmse")):
                for (mod, _, col, mk), y in zip(models, ys):
                    ax.plot(o.loc[mod, metric], y + dy + (0.0 if fill else 0.0), mk, ms=3.4,
                            mfc=col if fill else "white", mec=col, mew=0.7, zorder=3 if fill else 4,
                            alpha=1.0 if src == "ATL06" else 0.75)
    for ax, lab in zip(axs, ("Bias, model − ICESat-2 (m)", "NMAD (m)", "RMSE (m)")):
        ax.set_xlabel(lab)
        ax.grid(axis="x", color=plotting.GRID, lw=0.4)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
    axs[0].axvline(0, color=plotting.INK_MUTED, lw=0.5)
    axs[0].set_yticks(ys, [lab for _, lab, _, _ in models])
    axs[0].set_ylim(-0.6, len(models) - 0.4)
    for ax, letter in zip(axs, "abc"):
        plotting.panel_label(ax, letter, x=-0.04 if letter != "a" else -0.36, y=1.03)
    legend = [Line2D([], [], ls="none", marker="o", ms=3.4, mfc=plotting.INK_MUTED, mec=plotting.INK_MUTED,
                     label="Bare emerged seabed (H2 population): " +
                           ", ".join(f"{s} n = {counts[(s, 'bare')][0]:,} in {counts[(s, 'bare')][1]} blocks"
                                     for s in ("ATL06", "ATL08"))),
              Line2D([], [], ls="none", marker="o", ms=3.4, mfc="white", mec=plotting.INK_MUTED, mew=0.7,
                     label="Vegetated emerged seabed: " +
                           ", ".join(f"{s} n = {counts[(s, 'vegetated')][0]:,} in {counts[(s, 'vegetated')][1]} blocks"
                                     for s in ("ATL06", "ATL08")))]
    fig.legend(handles=legend, loc="upper center", ncol=1, fontsize=5.5, bbox_to_anchor=(0.56, 1.0), handlelength=1.0,
               title="Each model: ATL06 (upper marker), ATL08 (lower marker)", title_fontproperties={"size": 5.5})
    C.save(fig, "ed_vegetation")
    plt.close(fig)
    print(counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
