"""ERL figure "When and how much" (15 cm wide).

(a) When: the daily basin level 1992-2026 (data/processed/dem/main/emergence/
    level_series_daily.csv: DAHITI, with Hydroweb shifted onto the DAHITI reference in its
    gaps), joined at the 2026 level (h3.level_2026_m) to the seven primary published
    pathways to 2100 (scripts/a8_emergence.DATE_PATHWAYS, role "primary"; values from
    results/levels/projection_pathways.csv, verified against their sources): each
    pathway's own change after 2026, as piecewise-linear knots (caspian.exposure.
    pathway_curves), the construction that dates the scenarios (scenario_dates, anchor
    "base"). Horizontal lines: the five scenario depths below the 2026 level; the 1 m and
    2 m bands are shaded. Dots: where each pathway reaches 1 m and 2 m; their years are
    checked against results/metrics.json (h3.date.*). Pathways that leave the frame are
    labelled where they leave it.
    Revised: the twelve range pathways
    (role "range": inter-model ranges and quartiles) span a grey band, the y-axis holds
    every drawn pathway to 2100 (no clipping), and the seven primary pathways are keyed in
    a legend instead of labelled in the frame.
(b) How much (near-term revision analysis): the emerged area of the
    domain E(D) at 0.1 m steps of decline D below the 2026 level
    (results/revision/n/), one colour per grid: solid, on the composite observed baseline
    (curves_composite.csv, surface "common_location"); dashed, on the grids shifted by
    their H1 shape constant (curves.csv, "shifted"); shading, the decline-normalised
    hindcast envelopes (envelopes.csv, all pairs, score "decline"), drawn up to the
    largest decline of the hindcast pairs (rev.n.env.max_pair_decline_m), beyond which the
    curves are uncalibrated; thin grey, the grids as distributed (curves.csv,
    "uncorrected"). The script checks that the envelopes' estimates equal the curves.

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

N_DIR = C.ROOT / "results" / "revision" / "n"
PRODS = ("gebco_2026", "gebco_2022", "etopo_2022", "etopo1", "srtm15p_v2.7")
ENV_PAIRS, ENV_SCORE = "all_pairs", "decline"     # N's decline-normalised envelopes, all hindcast pairs
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
    rng = [pid for pid, role, *_ in mod.DATE_PATHWAYS if role == "range"]
    allc = ex.pathway_curves(pw)
    curves = {k: v for k, v in allc.items() if k in prim}
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

    def path(pid):
        yrs, ch = allc[pid]
        t = np.r_[2026.0, yrs[yrs > 2026.0]]
        c0 = float(np.interp(2026.0, yrs, ch))
        return t, l26 + (np.interp(t, yrs, ch) - c0)

    paths = {pid: path(pid) for pid in prim + rng}
    lo_all = min(float(y.min()) for _, y in paths.values())
    hi_all = max(max(float(y.max()) for _, y in paths.values()), float(lv["level_m"].max()))
    ylim = (np.floor(lo_all) - 0.5, np.ceil(hi_all) + 1.5)

    # near-term revision curves (km2), domain, main reference level
    cv = pd.read_csv(N_DIR / "curves.csv")
    cc = pd.read_csv(N_DIR / "curves_composite.csv")

    def sel(d, surf):
        return d[(d["surface"] == surf) & (d["sensitivity"] == "main") & (d["region"] == "domain")]

    cur = {"composite": sel(cc, "common_location"), "shifted": sel(cv, "shifted"), "uncorrected": sel(cv, "uncorrected")}
    for k, d in cur.items():
        if not np.allclose(d["level_2026_m"], l26, atol=1e-4):
            raise SystemExit(f"{k}: N's 2026 level differs from h3.level_2026_m")
    calib = E.value(m, "rev.n.env.max_pair_decline_m")

    E.style()
    W = E.DOUBLE_MM
    L, R, TOP = 15.0, 56.0, 4.0
    HA, GAP, HB, BOT = 76.0, 16.0, 56.0, 10.0
    H = TOP + HA + GAP + HB + BOT
    fig = plt.figure(figsize=(W * C.MM, H * C.MM))
    ax = E.mm_axes(fig, W, H, L, TOP, W - L - R, HA)
    for d in (1.0, 2.0):
        ax.axhspan(l26 - d, l26 - d + 1.0, color="#eef3f8" if d == 1.0 else "#f6f6f2", lw=0, zorder=0)
        ax.axhline(l26 - d, color=E.MUTED, lw=0.6, ls=(0, (2, 2)), zorder=1)
    ax.axhline(l26, color=E.INK, lw=0.6, zorder=1)
    ax.text(1993.0, l26 + 0.15, "2026 level", ha="left", va="bottom", fontsize=E.FS, color=E.INK)
    ax.text(1993.0, l26 - 2.0 - 0.15, f"{E.num(m, 'h3.scenario.1_m')} m and {E.num(m, 'h3.scenario.2_m')} m below",
            ha="left", va="top", fontsize=E.FS, color=E.MUTED)
    ax.plot(tdec, lv["level_m"], color=E.INK, lw=0.9, zorder=4, solid_capstyle="butt")
    ax.axvline(2026.0, color=E.MUTED, lw=0.6, zorder=1)
    ax.text(2025.0, ylim[1] - 0.3, "observed", ha="right", va="top", fontsize=E.FS, color=E.INK)
    ax.text(2027.0, ylim[1] - 0.3, "published pathways, from the 2026 level", ha="left", va="top", fontsize=E.FS,
            color=E.INK)
    tt = np.linspace(2026.0, 2100.0, 297)
    yy = np.array([np.interp(tt, *paths[pid]) for pid in rng])
    ax.fill_between(tt, yy.min(axis=0), yy.max(axis=0), color="#d9d9d9", alpha=0.6, lw=0, zorder=0.5)
    for pid in prim:
        t, y = paths[pid]
        col, ls = PATH_STYLE[pid]
        ax.plot(t, y, color=col, lw=1.1, ls=ls, zorder=3)
        for r in dates[(dates["pathway_id"] == pid) & dates["reached"]].itertuples():
            ax.plot(r.date, l26 - r.delta_m, "o", ms=3.6, mfc="white", mec=col, mew=0.9, zorder=5)
    ax.set_xlim(1992.0, 2100.0)
    ax.set_ylim(*ylim)
    ax.set_xticks([2000, 2020, 2040, 2060, 2080, 2100])
    ax.set_ylabel("Basin level (m, DAHITI reference)")
    ax.set_xlabel("Year")
    E.label(ax, "a", x=-0.1, y=1.0)
    ph = [Line2D([], [], color=E.INK, lw=0.9, label="Observed (altimetry)")]
    ph += [Line2D([], [], color=PATH_STYLE[pid][0], ls=PATH_STYLE[pid][1], lw=1.1, label=PATH_LABEL[pid])
           for pid in sorted(prim, key=lambda q: (q[:3] != "SAM", q))]
    ph += [Patch(fc="#d9d9d9", alpha=0.6, ec="none", label="Range pathways"),
           Line2D([], [], ls="none", marker="o", ms=3.6, mfc="white", mec=E.INK, mew=0.9,
                  label="Reaches 1 m or 2 m")]
    ax.legend(handles=ph, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=E.FS, frameon=False, borderaxespad=0,
              handlelength=2.2, labelspacing=0.45)

    # (b) how much: E(D) curves with the decline-normalised envelopes ------------------------------
    env = pd.read_csv(N_DIR / "envelopes.csv")
    env = env[(env["region"] == "domain") & (env["pairs"] == ENV_PAIRS) & (env["score"] == ENV_SCORE)
              & env["calibrated"] & env["lo_km2"].notna()]
    envs = {"composite": env[env["surface"] == "common_location_composite"], "shifted": env[env["surface"] == "shifted"]}
    for key, e in envs.items():         # the envelopes' estimates are N's curves
        for p in PRODS:
            a = e[e["product"] == p].set_index("delta_m")["estimate_km2"]
            b = cur[key][cur[key]["product"] == p].set_index("delta_m")["E_km2"].reindex(a.index)
            if not len(a) or not np.allclose(a.to_numpy(), b.to_numpy(), atol=0.01):
                raise SystemExit(f"{key} {p}: envelopes.csv estimates differ from N's curves")
    if not np.isclose(env["delta_m"].max(), calib):
        raise SystemExit("the envelopes end at a decline other than rev.n.env.max_pair_decline_m")
    yb = TOP + HA + GAP
    axb = E.mm_axes(fig, W, H, L, yb, W - L - R, HB)
    for p in PRODS:
        u = cur["uncorrected"][cur["uncorrected"]["product"] == p].sort_values("delta_m")
        axb.plot(u["delta_m"], u["E_km2"] / 1000, color="#a6a6a6", lw=0.6, zorder=2)
    for key, ls in (("composite", "solid"), ("shifted", (0, (4.0, 1.6)))):
        for p in PRODS:
            e = envs[key][envs[key]["product"] == p].sort_values("delta_m")
            axb.fill_between(e["delta_m"], e["lo_km2"] / 1000, e["hi_km2"] / 1000, color=E.COLOURS[p], alpha=0.12,
                             lw=0, zorder=1)
            d = cur[key][cur[key]["product"] == p].sort_values("delta_m")
            axb.plot(d["delta_m"], d["E_km2"] / 1000, color=E.COLOURS[p], lw=1.2, ls=ls, zorder=3)
    axb.axvline(calib, color=E.MUTED, lw=0.6, ls=(0, (2, 2)), zorder=1)
    axb.text(calib + 0.06, 0.97, "uncalibrated beyond", transform=axb.get_xaxis_transform(), ha="left", va="top",
             fontsize=E.FS, color=E.MUTED)
    axb.set_xlim(0, 5.0)
    axb.set_ylim(0, max(float(cur[k]["E_km2"].max()) for k in cur) / 1000 * 1.04)
    axb.set_xticks([0, 1, 2, 3, 4, 5])
    axb.set_xlabel("Decline below the 2026 level (m)")
    axb.set_ylabel("Emerged area (10³ km²)")
    axb.grid(axis="y", color=E.GRID, lw=0.4)
    axb.set_axisbelow(True)
    E.label(axb, "b", x=-0.1, y=1.0)
    handles = [Patch(fc=E.COLOURS[p], ec="none", label=E.NAMES[p]) for p in PRODS]
    handles += [Line2D([], [], color=E.INK, lw=1.2, label="From the observed\n2026 shoreline"),
                Line2D([], [], color=E.INK, lw=1.2, ls=(0, (4.0, 1.6)), label="Shifted grids"),
                Patch(fc=E.INK, alpha=0.12, ec="none", label="Hindcast envelope"),
                Line2D([], [], color="#a6a6a6", lw=0.6, label="As distributed")]
    axb.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=E.FS, frameon=False,
               borderaxespad=0, handlelength=2.2, labelspacing=0.4)
    C.save(fig, "erl_fig_timeline")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
