"""Supplementary Figure (robustness check 8): H2 block-size sensitivity.

For the primary 50 km blocks (results/validation/main/) and the 25 km and 12.5 km
sensitivities (results/validation/main/sensitivity_25000m/, sensitivity_12500m/), per
ICESat-2 source (rows):
a, NMAD and b, RMSE of the DEM and of each product (matched variant) on the test
   blocks of that size (<source>_accuracy.csv, scope "overall");
c, d, the DEM minus product difference in NMAD and RMSE with its paired test-block
   bootstrap 95% interval (<source>_paired_bootstrap.csv, matched variant);
e, empirical coverage of the 90% conformal intervals with the test-block bootstrap
   interval (<source>_coverage.csv).
The primary verdict is the 50 km one; the others are reported, not used to
change it.

Output: figures/out/ed_block_size.{pdf,tif,png}
Usage: python figures/ed_block_size.py
"""

from __future__ import annotations

import sys

import _common as C
import numpy as np
import pandas as pd

from caspian import plotting
from caspian.config import RESULTS

VAL = RESULTS / "validation" / "main"
SIZES = {50.0: VAL, 25.0: VAL / "sensitivity_25000m", 12.5: VAL / "sensitivity_12500m"}


def _row(df: pd.DataFrame, **eq) -> pd.Series:
    k = np.ones(len(df), bool)
    for c, v in eq.items():
        k &= (df[c] == v).to_numpy()
    if k.sum() != 1:
        raise ValueError(f"{k.sum()} rows match {eq}")
    return df[k].iloc[0]


def main() -> int:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    m = C.metrics()
    nominal = float(m["params.conformal_level"]["value"])
    sizes = sorted(SIZES)
    xs = np.arange(len(sizes))
    C.style()
    W, H = C.DOUBLE_MM, 100.0
    fig, axs = plt.subplots(2, 5, figsize=(W * C.MM, H * C.MM), gridspec_kw={"wspace": 0.55, "hspace": 0.55,
                                                                            "left": 0.06, "right": 0.99,
                                                                            "bottom": 0.17, "top": 0.86})
    for row, src in enumerate(("ATL06", "ATL08")):
        acc = {s: pd.read_csv(SIZES[s] / f"{src}_accuracy.csv") for s in sizes}
        pb = {s: pd.read_csv(SIZES[s] / f"{src}_paired_bootstrap.csv") for s in sizes}
        cov = {s: pd.read_csv(SIZES[s] / f"{src}_coverage.csv").set_index("scope") for s in sizes}
        nblk = [int(_row(acc[s], scope="overall", model="dem")["n_blocks"]) for s in sizes]
        ticks = [f"{s:g}\n({n})" for s, n in zip(sizes, nblk)]
        for col, metric, lab in ((0, "nmad", "NMAD (m)"), (1, "rmse", "RMSE (m)")):
            ax = axs[row, col]
            for model, colour, mk in [("dem", plotting.INK, "o")] + \
                    [(f"{p}|matched", C.PRODUCT_COLOURS[p], C.PRODUCT_MARKERS[p]) for p in C.PRODUCTS]:
                v = [float(_row(acc[s], scope="overall", model=model)[metric]) for s in sizes]
                ax.plot(xs, v, color=colour, marker=mk, ms=3, lw=0.8 if model == "dem" else 0.6, mew=0,
                        zorder=4 if model == "dem" else 3)
            ax.set_ylabel(f"{src} {lab}")
            ax.set_ylim(0, None)
        for col, metric, lab in ((2, "nmad", "ΔNMAD (m)"), (3, "rmse", "ΔRMSE (m)")):
            ax = axs[row, col]
            offs = np.linspace(-0.24, 0.24, len(C.PRODUCTS))
            for p, dx in zip(C.PRODUCTS, offs):
                r = [_row(pb[s], scope="overall", variant="matched", product=p, metric=metric) for s in sizes]
                d = np.array([x["diff"] for x in r])
                lo = np.array([x["ci_low"] for x in r])
                hi = np.array([x["ci_high"] for x in r])
                ax.errorbar(xs + dx, d, yerr=[d - lo, hi - d], fmt=C.PRODUCT_MARKERS[p], ms=2.6, color=C.PRODUCT_COLOURS[p],
                            elinewidth=0.6, capsize=0, mew=0)
            ax.axhline(0, color=plotting.INK_MUTED, lw=0.5)
            ax.set_ylabel(f"{src} {lab}")
        ax = axs[row, 4]
        for s, x in zip(sizes, xs):
            c = cov[s].loc["overall"]
            ax.plot([x, x], [c["boot_low"], c["boot_high"]], color=plotting.INK, lw=0.6)
            ax.plot(x, c["coverage"], "o", ms=3, color=plotting.INK)
        ax.axhline(nominal, color="#D55E00", lw=0.7, ls=(0, (3, 1.5)))
        ax.set_ylim(0.4, 1.0)
        ax.set_ylabel(f"{src} coverage")
        for ax in axs[row]:
            ax.set_xticks(xs, ticks)
            ax.set_xlim(-0.5, len(sizes) - 0.5)
            ax.grid(axis="y", color=plotting.GRID, lw=0.4)
            ax.set_axisbelow(True)
            if row == 1:
                ax.set_xlabel("Block size (km)\n(test blocks)")
    for ax, letter in zip(axs[0], "abcde"):
        plotting.panel_label(ax, letter, x=-0.42, y=1.04)
    handles = [Line2D([], [], color=plotting.INK, marker="o", ms=3, mew=0, lw=0.8, label="Waterline DEM")] + \
              [Line2D([], [], color=C.PRODUCT_COLOURS[p], marker=C.PRODUCT_MARKERS[p], ms=3, mew=0, lw=0.6,
                      label=C.PRODUCT_NAMES[p]) for p in C.PRODUCTS] + \
              [Line2D([], [], color="#D55E00", lw=0.7, ls=(0, (3, 1.5)), label="Nominal coverage")]
    fig.legend(handles=handles, loc="upper center", ncol=7, fontsize=5.5, bbox_to_anchor=(0.5, 0.995),
               handlelength=1.6, columnspacing=1.0)
    C.save(fig, "ed_block_size")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
