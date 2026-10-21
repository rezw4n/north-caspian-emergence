"""ERL figure "When and how much" (D-100, D-102; 15 cm wide).

(a) When: the daily basin level 1992-2026 (data/processed/dem/main/emergence/
    level_series_daily.csv: DAHITI, with Hydroweb shifted onto the DAHITI reference in its
    gaps), joined at the 2026 level (h3.level_2026_m) to the seven primary published
    pathways to 2100 (scripts/a8_emergence.DATE_PATHWAYS, role "primary"; values from
    results/levels/projection_pathways.csv, verified against their sources, D-081): each
    pathway's own change after 2026, as piecewise-linear knots (caspian.exposure.
    pathway_curves), the construction that dates the scenarios (scenario_dates, anchor
    "base"). Horizontal lines: the five scenario depths below the 2026 level; the 1 m and
    2 m bands are shaded. Dots: where each pathway reaches 1 m and 2 m; their years are
    checked against results/metrics.json (h3.date.*). Pathways that leave the frame are
    labelled where they leave it.
(b) How much: the emerged area of the domain below each product's own 2026 shoreline at the
    five scenario depths, for the five products as users apply them (uncorrected), with
    their split-conformal 90% intervals (results/emergence/main/emergence.csv through
    metrics h3.emergence.*). Grey bars: the range of the five estimates (dark) and of their
    intervals (light) at each depth. Only the scenario depths are computed; nothing joins
    them. No error model is applied (D-100).

Output: figures/out/erl_fig_timeline.{pdf,tif,png}
Usage: python figures/erl_fig_timeline.py
"""

from __future__ import annotations

import importlib.util
import sys

import _common as C
import _erl as E
import numpy as np
import pandas as pd

from caspian import exposure as ex
from caspian.config import DATA

SCEN = (("0p5", 0.5), ("1", 1.0), ("2", 2.0), ("3", 3.0), ("5", 5.0))
YLIM = (-34.6, -25.4)
PATH_LABEL = {"SAM23-07": "Samant & Prange, SSP2-4.5", "SAM23-10": "Samant & Prange, SSP5-8.5",
              "SAM23-14": "Samant & Prange, trend",
              "HOS25-01": "Hoseini et al., low", "HOS25-04": "Hoseini et al., medium",
              "HOS25-07": "Hoseini et al., medium-high", "HOS25-10": "Hoseini et al., high"}
PATH_STYLE = {"SAM23-07": ("#3b3b3b", (0, (6, 2))), "SAM23-10": ("#3b3b3b", "solid"),
              "SAM23-14": ("#3b3b3b", (0, (1.2, 1.6))),
              "HOS25-01": ("#8a8a8a", (0, (1.2, 1.6))), "HOS25-04": ("#8a8a8a", (0, (6, 2))),
              "HOS25-07": ("#8a8a8a", (0, (6, 1.5, 1.2, 1.5))), "HOS25-10": ("#8a8a8a", "solid")}


def a8():
    spec = importlib.util.spec_from_file_location("a8_emergence", C.ROOT / "scripts" / "a8_emergence.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def metric_pid(pid: str) -> str:
    s = "".join(ch if ch.isalnum() else "_" for ch in pid.lower())
    return "_".join(t for t in s.split("_") if t)


def main() -> int:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    m = C.metrics()
    l26 = E.value(m, "h3.level_2026_m")
    mod = a8()
    pw = mod.read_pathways(mod.PATHWAYS)
    prim = [pid for pid, role, *_ in mod.DATE_PATHWAYS if role == "primary"]
    curves = {k: v for k, v in ex.pathway_curves(pw).items() if k in prim}
    if set(curves) != set(PATH_LABEL):
        raise SystemExit(f"primary pathways {sorted(curves)} differ from the labelled set")
    dates = ex.scenario_dates(curves, [1.0, 2.0], base_year=2026.0)
    for r in dates.itertuples():     # the dots must be the dates the text quotes
        key = f"h3.date.{metric_pid(r.pathway_id)}.{'1' if r.delta_m == 1.0 else '2'}"
        want = m[key]["value"]
        got = str(int(np.floor(r.date))) if r.reached else want
        if got != want:
            raise SystemExit(f"{key}: curve gives {got}, metrics {want}")
    lv = pd.read_csv(DATA / "processed" / "dem" / "main" / "emergence" / "level_series_daily.csv", parse_dates=["date"])
    tdec = lv["date"].dt.year + (lv["date"].dt.dayofyear - 0.5) / 365.2425

    E.style()
    W = E.DOUBLE_MM
    L, R, TOP = 15.0, 38.0, 4.0
    HA, GAP, HB, BOT = 70.0, 17.0, 46.0, 10.0
    H = TOP + HA + GAP + HB + BOT
    fig = plt.figure(figsize=(W * C.MM, H * C.MM))
    ax = E.mm_axes(fig, W, H, L, TOP, W - L - R, HA)
    for (k, d) in SCEN:
        y = l26 - d
        if d in (1.0, 2.0):
            ax.axhspan(l26 - d, l26 - d + 1.0, color="#eef3f8" if d == 1.0 else "#f6f6f2",
                       lw=0, zorder=0)
        ax.axhline(y, color=E.MUTED, lw=0.6, ls=(0, (2, 2)), zorder=1)
        ax.text(1993.0, y + 0.06, f"{E.num(m, f'h3.scenario.{k}_m')} m below", ha="left", va="bottom", fontsize=E.FS,
                color=E.MUTED)
    ax.axhline(l26, color=E.INK, lw=0.6, zorder=1)
    ax.text(1993.0, l26 + 0.06, "2026 level", ha="left", va="bottom", fontsize=E.FS, color=E.INK)
    ax.plot(tdec, lv["level_m"], color=E.INK, lw=0.9, zorder=4, solid_capstyle="butt")
    ax.axvline(2026.0, color=E.MUTED, lw=0.6, zorder=1)
    ax.text(2025.0, YLIM[1] - 0.15, "observed", ha="right", va="top", fontsize=E.FS, color=E.INK)
    ax.text(2027.0, YLIM[1] - 0.15, "published pathways, from the 2026 level", ha="left", va="top", fontsize=E.FS,
            color=E.INK)
    ends = []
    for pid in prim:
        yrs, ch = curves[pid]
        t = np.r_[2026.0, yrs[yrs > 2026.0]]
        c0 = float(np.interp(2026.0, yrs, ch))
        y = l26 + (np.interp(t, yrs, ch) - c0)
        col, ls = PATH_STYLE[pid]
        ax.plot(t, y, color=col, lw=1.1, ls=ls, zorder=3)
        for r in dates[(dates["pathway_id"] == pid) & dates["reached"]].itertuples():
            ax.plot(r.date, l26 - r.delta_m, "o", ms=4.0, mfc="white", mec=col, mew=1.0, zorder=5)
        if y[-1] >= YLIM[0]:
            ends.append([float(y[-1]), pid, "end"])
        else:
            tx = float(np.interp(YLIM[0], y[::-1], t[::-1]))
            ends.append([YLIM[0], pid, tx])
    # labels in the right margin, with leader lines from the line's end (2100) or from where it
    # leaves the bottom of the frame; positions spread to a minimum gap, kept inside the frame
    inner = [e for e in ends if e[2] != "end" and e[2] < 2075.0]     # leaves the frame early: labelled inside
    for _, pid, x in inner:
        ax.text(x - 1.5, YLIM[0] + 0.25, PATH_LABEL[pid], ha="right", va="center", fontsize=E.FS, color=E.INK, zorder=6,
                bbox={"boxstyle": "square,pad=0.1", "fc": "white", "ec": "none"})
    ends = [e for e in ends if e not in inner]
    lab = sorted([(e[0], e[1], 2100.0 if e[2] == "end" else e[2]) for e in ends], key=lambda e: -e[0])
    gap, lo = 0.52, YLIM[0] + 0.15
    ypos = []
    for y, _, _ in lab:
        ypos.append(min(y, ypos[-1] - gap) if ypos else y)
    for k in range(len(ypos) - 1, -1, -1):
        floor = lo if k == len(ypos) - 1 else ypos[k + 1] + gap
        ypos[k] = max(ypos[k], floor)
    for (y, pid, x), yl in zip(lab, ypos):
        ax.annotate(PATH_LABEL[pid], xy=(x, y), xytext=(2103.0, yl), textcoords="data", fontsize=E.FS, color=E.INK,
                    va="center", ha="left", annotation_clip=False,
                    arrowprops={"arrowstyle": "-", "color": E.MUTED, "lw": 0.5, "shrinkA": 0, "shrinkB": 0})
    ax.set_xlim(1992.0, 2100.0)
    ax.set_ylim(*YLIM)
    ax.set_xticks([2000, 2020, 2040, 2060, 2080, 2100])
    ax.set_ylabel("Basin level (m, DAHITI reference)")
    ax.set_xlabel("Year")
    E.label(ax, "a", x=-0.1, y=1.0)
    ax.legend(handles=[Line2D([], [], color=E.INK, lw=0.9, label="Observed (altimetry)"),
                       Line2D([], [], ls="none", marker="o", ms=4.0, mfc="white", mec=E.INK, mew=1.0,
                              label="Pathway reaches 1 m or 2 m")],
              loc="upper right", bbox_to_anchor=(1.0, 0.93), fontsize=E.FS, frameon=False, borderaxespad=0.3,
              handlelength=1.6)

    # (b) how much ---------------------------------------------------------------------------------
    axb = E.mm_axes(fig, W, H, L, TOP + HA + GAP, W - L - R, HB)
    offs = np.linspace(-0.15, 0.15, len(E.PK))
    prods = ("gebco_2026", "gebco_2022", "etopo_2022", "etopo1", "srtm15p_v2.7")
    for k, d in SCEN:
        est = [E.value(m, f"h3.emergence.{E.PK[p]}.uncorrected.{k}.estimate_km2") / 1000 for p in prods]
        lo = [E.value(m, f"h3.emergence.{E.PK[p]}.uncorrected.{k}.lo_km2") / 1000 for p in prods]
        hi = [E.value(m, f"h3.emergence.{E.PK[p]}.uncorrected.{k}.hi_km2") / 1000 for p in prods]
        axb.add_patch(plt.Rectangle((d - 0.21, min(lo)), 0.42, max(hi) - min(lo), fc="#ececec", ec="none", zorder=0))
        axb.add_patch(plt.Rectangle((d - 0.21, min(est)), 0.42, max(est) - min(est), fc="#c9c9c9", ec="none", zorder=1))
        for p, dx, e_, l_, h_ in zip(prods, offs, est, lo, hi):
            axb.plot([d + dx, d + dx], [l_, h_], color=E.COLOURS[p], lw=0.9, zorder=3)
            axb.plot(d + dx, e_, E.MARKERS[p], ms=4.5, mfc=E.COLOURS[p], mec="white", mew=0.5, zorder=4)
    for d in (1.0, 2.0):
        axb.axvspan(d - 0.25, d + 0.25, color="#eef3f8" if d == 1.0 else "#f6f6f2", lw=0, zorder=-1)
    axb.set_xlim(0, 5.5)
    axb.set_xticks([d for _, d in SCEN], [E.num(m, f"h3.scenario.{k}_m") for k, _ in SCEN])
    axb.set_ylim(0, None)
    axb.set_xlabel("Fall below the 2026 level (m)")
    axb.set_ylabel("Emerged area (10³ km²)")
    axb.grid(axis="y", color=E.GRID, lw=0.4)
    axb.set_axisbelow(True)
    E.label(axb, "b", x=-0.1, y=1.0)
    handles = [Line2D([], [], color=E.COLOURS[p], marker=E.MARKERS[p], ms=4.5, mfc=E.COLOURS[p], mec="white", lw=0.9,
                      label=E.NAMES[p]) for p in prods]
    handles += [Patch(fc="#c9c9c9", ec="none", label="Range of the estimates"),
                Patch(fc="#ececec", ec="none", label="Range of 90% intervals")]
    axb.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=E.FS, frameon=False,
               borderaxespad=0, handlelength=1.6, labelspacing=0.35)
    C.save(fig, "erl_fig_timeline")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
