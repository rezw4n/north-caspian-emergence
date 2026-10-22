"""Supplementary Figure: calibration of the wind-setup model.

From results/levels/setup_model.json (scripts/a5_setup_model.py), per sector:
a, the calibration R^2 of the fitted model, on calibration blocks that the fit never
   used, with the number of calibration and training passes;
b, the standard deviation of the de-meaned sector anomalies of SWOT water-surface
   elevation minus the basin level (no model), and of the residuals after the model,
   on calibration blocks (filled) and training blocks (open).
A sector without SWOT open-sea passes (Komsomolets) is shown with the sector whose
model it borrows. The figure shows the inputs of the surge screening, not any result
of the waterline, DEM or hindcast analyses.

Output: figures/out/supp_setup_model.{pdf,tif,png}

Usage: python figures/supp_setup_model.py
"""

from __future__ import annotations

import json
import sys

import _common as C
import numpy as np

from caspian import plotting
from caspian.config import RESULTS
from caspian.geography import SECTOR_ORDER

NO_MODEL = "#BBBBBB"
MODEL = "#0072B2"


def main() -> int:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    sm = json.loads((RESULTS / "levels" / "setup_model.json").read_text(encoding="utf-8"))
    sec = sm["sectors"]
    order = [s for s in SECTOR_ORDER if s in sec]
    y = np.arange(len(order))[::-1]
    names = []
    for s in order:
        names.append(C.SECTOR_SHORT[s] + ("\n(borrowed model)" if sec[s].get("proxy") else ""))

    C.style()
    fig = plt.figure(figsize=(C.DOUBLE_MM * C.MM, 66 * C.MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.25], wspace=0.08, left=0.215, right=0.985, bottom=0.25,
                          top=0.9)
    ax_r, ax_s = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    for yi, s in zip(y, order):
        v = sec[s]
        if v.get("fitted"):
            ax_r.plot([0, v["r2_calib"]], [yi, yi], color=plotting.GRID, lw=2.0, solid_capstyle="butt", zorder=1)
            ax_r.plot(v["r2_calib"], yi, "o", ms=4.5, color=MODEL, mec="white", mew=0.6, zorder=3)
            ax_r.annotate(f"{v['r2_calib']:.2f}  (n = {v['n_calib_passes']} / {v['n_train_passes']})",
                          xy=(v["r2_calib"], yi), xytext=(5, 0), textcoords="offset points", va="center",
                          fontsize=5.5, color=plotting.INK)
            for split, mk in (("calib", "o"), ("train", "o")):
                filled = split == "calib"
                a, r = v[f"anom_sd_{split}_m"], v[f"resid_sd_{split}_m"]
                dy = 0.12 if filled else -0.12
                ax_s.plot([r, a], [yi + dy, yi + dy], color=plotting.GRID, lw=1.2, zorder=1)
                ax_s.plot(a, yi + dy, mk, ms=4, mfc=NO_MODEL if filled else "white", mec=NO_MODEL, mew=0.8, zorder=3)
                ax_s.plot(r, yi + dy, mk, ms=4, mfc=MODEL if filled else "white", mec=MODEL, mew=0.8, zorder=3)
        else:
            for ax in (ax_r, ax_s):
                msg = ("no SWOT open-sea pass; not fitted" if ax is ax_r else
                       f"uses the {C.SECTOR_SHORT[v['proxy']]} model" if v.get("proxy") else "")
                ax.text(0.01, yi, msg, transform=ax.get_yaxis_transform(),
                        va="center", ha="left", fontsize=5.5, color=plotting.INK_MUTED)

    ax_r.set_yticks(y, names)
    ax_r.set_xlim(0, 1)
    ax_r.set_xlabel("Calibration R²")   # Unicode superscript keeps Arial (no mathtext font)
    ax_r.set_ylim(-0.6, len(order) - 0.4)
    ax_s.set_yticks(y, [""] * len(y))
    ax_s.set_ylim(ax_r.get_ylim())
    xmax = max(max(sec[s][f"anom_sd_{k}_m"] for k in ("calib", "train")) for s in order if sec[s].get("fitted"))
    ax_s.set_xlim(0, 0.05 * np.ceil(xmax / 0.05 + 0.5))
    ax_s.set_xlabel("Standard deviation of sector anomaly (m)")
    for ax in (ax_r, ax_s):
        ax.grid(axis="x", color=plotting.GRID, lw=0.4)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
    ax_s.legend(handles=[Line2D([], [], ls="none", marker="o", ms=4, mfc=NO_MODEL, mec=NO_MODEL, label="No model"),
                         Line2D([], [], ls="none", marker="o", ms=4, mfc=MODEL, mec=MODEL, label="Model residual"),
                         Line2D([], [], ls="none", marker="o", ms=4, mfc=plotting.INK_MUTED, mec=plotting.INK_MUTED,
                                label="Calibration blocks (upper)"),
                         Line2D([], [], ls="none", marker="o", ms=4, mfc="white", mec=plotting.INK_MUTED,
                                label="Training blocks (lower)")],
                loc="upper right", bbox_to_anchor=(1.0, -0.2), ncol=4, handletextpad=0.3, columnspacing=1.0)
    plotting.panel_label(ax_r, "a", x=-0.02, y=1.04)
    plotting.panel_label(ax_s, "b", x=-0.02, y=1.04)
    ax_r.text(1.0, 1.02, "n = calibration / training passes", transform=ax_r.transAxes, ha="right", va="bottom",
              fontsize=5.5, color=plotting.INK_MUTED)

    C.save(fig, "supp_setup_model")
    plt.close(fig)
    print({s: {k: sec[s].get(k) for k in ("r2_calib", "resid_sd_calib_m", "anom_sd_calib_m", "proxy")} for s in order})
    return 0


if __name__ == "__main__":
    sys.exit(main())
