"""Supplementary Figure (brief section 10 check 4): surge-screening sensitivity (A5;
D-064, D-077).

a, the pre-registered statistic S(tau) (median over block-bins of the median odd/even
   waterline displacement) with its bootstrap SE, and, descriptive only (not used for
   the choice), the mean, 75th and 90th percentiles of the block-bin displacements
   (results/levels/screening_sensitivity.json; 41 sampled training blocks, tag sensB);
b, the share of block-bins whose displacement exceeds one 60 m cell, and c, the number
   of block-bins left at each threshold (same file);
d, the share of the Landsat acquisitions of the main stage-2 plan whose predicted
   setup in a sector exceeds each candidate threshold, i.e. the acquisitions that
   screening at that threshold would remove there (data/interim/stage2/
   plan_main_tau9900cm_landsat.csv, setup_<sector>_m from scripts/a5_setup_model.py;
   the plan itself uses no screening, D-077). Komsomolets uses a borrowed model (D-066).

Output: figures/out/ed_screening.{pdf,tif,png}
Usage: python figures/ed_screening.py
"""

from __future__ import annotations

import json
import sys

import _common as C
import numpy as np
import pandas as pd

from caspian import plotting
from caspian.config import INTERIM, RESULTS
from caspian.geography import SECTOR_ORDER


def main() -> int:
    import cmocean
    import matplotlib.pyplot as plt

    m = C.metrics()
    cell = float(m["params.stage2_res_m"]["value"])
    ss = json.loads((RESULTS / "levels" / "screening_sensitivity.json").read_text(encoding="utf-8"))
    taus = ss["taus"]
    lab = ["none" if t == "inf" else f"{float(t):.2f}" for t in taus]
    xs = np.arange(len(taus))
    res, desc = ss["results"], ss["descriptive_not_used_for_choice"]
    plan = pd.read_csv(INTERIM / "stage2" / "plan_main_tau9900cm_landsat.csv")
    acq = plan.drop_duplicates("acq")
    cand = [float(t) for t in taus if t != "inf"]
    # a6_screening keeps the setup classes whose upper edge is at most tau: |setup| >= tau is removed
    share = np.array([[100.0 * float((acq[f"setup_{s}_m"].abs() >= t).mean()) for t in cand] for s in SECTOR_ORDER])

    C.style()
    W, H = C.DOUBLE_MM, 58.0
    fig = plt.figure(figsize=(W * C.MM, H * C.MM))
    ax_a = fig.add_axes([0.055, 0.2, 0.25, 0.7])
    ax_b = fig.add_axes([0.375, 0.2, 0.13, 0.7])
    ax_c = fig.add_axes([0.56, 0.2, 0.12, 0.7])
    ax_d = fig.add_axes([0.80, 0.2, 0.13, 0.7])
    cax = fig.add_axes([0.94, 0.2, 0.008, 0.7])

    for key, mk, name in (("p90_m", "o", "90th percentile"), ("p75_m", "^", "75th percentile"), ("mean_m", "s", "Mean")):
        ax_a.plot(xs, [desc[t][key] for t in taus], color=plotting.INK_MUTED, lw=0.6, ls=(0, (2, 1.5)), marker=mk, ms=2.8,
                  mfc="white", mec=plotting.INK_MUTED, mew=0.6, label=f"{name} (descriptive)")
    ax_a.errorbar(xs, [res[t]["S_m"] for t in taus], yerr=[res[t]["se_m"] for t in taus], fmt="o", ms=3.2,
                  color=plotting.INK, elinewidth=0.6, capsize=1.5, capthick=0.6, label="S(τ) ± SE (decision rule)")
    ax_a.axhline(cell, color="#bbbbbb", lw=0.6, zorder=0, label=f"One {cell:.0f} m cell")
    ax_a.set_yscale("log")
    ax_a.set_ylim(20, 30000)
    ax_a.set_yticks([30, 100, 300, 1000, 3000, 10000], ["30", "100", "300", "1,000", "3,000", "10,000"])
    ax_a.yaxis.set_minor_locator(plt.NullLocator())
    ax_a.set_ylabel("Odd–even waterline distance (m)")
    ax_a.legend(loc="upper left", ncol=2, fontsize=5.2, handlelength=1.5, borderaxespad=0.2, columnspacing=0.8)
    ax_b.plot(xs, [100.0 * desc[t]["share_above_1px"] for t in taus], "o-", color=plotting.INK, ms=3, lw=0.7)
    ax_b.set_ylim(0, 100)
    ax_b.set_ylabel(f"Block-bins above one cell (%)")
    ax_c.bar(xs, [res[t]["n_block_bins"] for t in taus], width=0.6, color=plotting.INK_MUTED, lw=0)
    ax_c.set_ylabel("Block-bins (n)")
    for ax in (ax_a, ax_b, ax_c):
        ax.set_xticks(xs, lab, rotation=0)
        ax.set_xlim(-0.5, len(taus) - 0.5)
        ax.set_xlabel("Threshold τ (m)")
        ax.grid(axis="y", color=plotting.GRID, lw=0.4)
        ax.set_axisbelow(True)
    for ax in (ax_b, ax_c):
        plt.setp(ax.get_xticklabels(), rotation=60, ha="right", rotation_mode="anchor", fontsize=5)

    im = ax_d.imshow(share, cmap=cmocean.cm.amp, vmin=0, vmax=max(50.0, float(share.max())), aspect="auto")
    for i in range(share.shape[0]):
        for j in range(share.shape[1]):
            v = share[i, j]
            ax_d.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=5,
                      color="white" if v > 0.6 * max(50.0, share.max()) else plotting.INK)
    ax_d.set_xticks(range(len(cand)), [f"{t:.2f}" for t in cand], fontsize=5)
    ax_d.set_yticks(range(len(SECTOR_ORDER)), [C.SECTOR_TINY[s] for s in SECTOR_ORDER])
    ax_d.tick_params(length=0)
    for sp in ax_d.spines.values():
        sp.set_visible(False)
    ax_d.set_xlabel("Threshold τ (m)")
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Acquisitions removed (%)", fontsize=5.5)
    cb.ax.tick_params(labelsize=5, length=2, width=0.4)
    cb.outline.set_linewidth(0.4)
    ax_d.text(0.5, 1.02, f"n = {len(acq):,} Landsat acquisitions", transform=ax_d.transAxes, ha="center", va="bottom",
              fontsize=5.5)
    for ax, letter, x in ((ax_a, "a", -0.2), (ax_b, "b", -0.45), (ax_c, "c", -0.5), (ax_d, "d", -0.75)):
        plotting.panel_label(ax, letter, x=x, y=1.02)
    C.save(fig, "ed_screening")
    plt.close(fig)
    print("acquisitions", len(acq), "| share removed (%):", dict(zip(SECTOR_ORDER, share.round(1).tolist())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
