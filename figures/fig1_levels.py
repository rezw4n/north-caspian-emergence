"""Fig. 1 inset (or Supplementary Figure): the basin water level, the scene record
and the three epochs.

a, the daily basin level on the DAHITI reference, as used to date emergence
   (scripts/a6_emergence.daily_levels: the scene-level code path of
   scripts/a4_scene_levels.py, DAHITI with Hydroweb shifted onto the DAHITI reference
   in its gaps), with the DAHITI records as points. Epochs E1-E3 are shaded.
b, c, levelled scenes per calendar month (data/interim/levels/scene_levels.csv,
   epochs E1-E3): Landsat (b) and Sentinel-2 granules (c), each on its own axis,
   split into the open-water months that can enter stage 2 and the ice season, which
   stage 2 excludes (scripts/a3_thresholds.ICE_MONTHS).

Epoch limits come from results/metrics.json (levels.epoch_E1_end, params.e2_end), so
the figure and the text use the same dates; the script checks them against the epoch
labels in scene_levels.csv. No result is shown: these are the inputs of the per-scene level step.

Output: figures/out/fig1_levels.{pdf,tif,png}

Usage: python figures/fig1_levels.py
"""

from __future__ import annotations

import importlib.util
import sys

import _common as C
import numpy as np
import pandas as pd

from caspian import plotting
from caspian.config import INTERIM

LANDSAT = ("TM_L5", "ETM_L7", "OLI_L8", "OLI_L9")
COLOURS = {"dahiti": plotting.INK, "hydroweb_shifted": "#D55E00", "open": "#0072B2", "ice": "#BBBBBB"}
EPOCH_SHADE = {"E1": "#f2f2f2", "E2": "#ffffff", "E3": "#f2f2f2"}


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, C.ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    m = C.metrics()
    e1_end = pd.Timestamp(m["levels.epoch_E1_end"]["value"])
    e2_end = pd.Timestamp(m["params.e2_end"]["value"]) + pd.Timedelta(days=1)   # E3 starts the next day

    ice_months = _script("a3_thresholds").ICE_MONTHS
    daily = _script("a6_emergence").daily_levels()
    a4 = _script("a4_scene_levels")
    dahiti = a4.read_dahiti(a4.RAW / "dahiti" / "dahiti_39_caspian_sea_water_level.csv").data

    sc = pd.read_csv(INTERIM / "levels" / "scene_levels.csv", usecols=["sensor", "time", "epoch"], low_memory=False)
    sc = sc[sc["epoch"].isin(["E1", "E2", "E3"])].copy()
    sc["t"] = pd.to_datetime(sc["time"], format="ISO8601")
    # the epoch labels of the scene table and the dates in metrics.json must agree
    if sc.loc[sc["epoch"] == "E1", "t"].max() > e1_end + pd.Timedelta(days=1) or \
            sc.loc[sc["epoch"] == "E2", "t"].min() < e1_end or \
            sc.loc[sc["epoch"] == "E3", "t"].min() < e2_end or sc.loc[sc["epoch"] == "E2", "t"].max() >= e2_end:
        print("FAIL epoch labels in scene_levels.csv disagree with results/metrics.json")
        return 1
    n_expected = sum(m[f"levels.n_scenes_by_epoch.{e}"]["value"] for e in ("E1", "E2", "E3"))
    if len(sc) != n_expected:
        print(f"FAIL {len(sc)} levelled scenes, metrics.json says {n_expected}")
        return 1

    sc["month"] = sc["t"].dt.to_period("M")
    sc["family"] = np.where(sc["sensor"].isin(LANDSAT), "landsat", "s2")
    sc["season"] = np.where(sc["t"].dt.month.isin(ice_months), "ice", "open")
    months = pd.period_range(sc["month"].min(), sc["month"].max(), freq="M")
    counts = {f: sc[sc["family"] == f].groupby(["month", "season"]).size().unstack(fill_value=0)
              .reindex(months, fill_value=0).reindex(columns=["open", "ice"], fill_value=0)
              for f in ("landsat", "s2")}

    C.style()
    fig = plt.figure(figsize=(C.DOUBLE_MM * C.MM, 100 * C.MM))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.5, 0.8, 0.8], hspace=0.14, left=0.075, right=0.985,
                          bottom=0.09, top=0.955)
    ax_l = fig.add_subplot(gs[0])
    ax_ls = fig.add_subplot(gs[1], sharex=ax_l)
    ax_s2 = fig.add_subplot(gs[2], sharex=ax_l)

    t0 = pd.Timestamp(daily["date"].min()) - pd.Timedelta(days=60)
    t1 = pd.Timestamp(daily["date"].max()) + pd.Timedelta(days=60)
    bounds = {"E1": (t0, e1_end), "E2": (e1_end, e2_end), "E3": (e2_end, t1)}
    for ax in (ax_l, ax_ls, ax_s2):
        for e, (a, b) in bounds.items():
            ax.axvspan(a, b, color=EPOCH_SHADE[e], zorder=0, lw=0)
        for x in (e1_end, e2_end):
            ax.axvline(x, color=plotting.INK_MUTED, lw=0.5, ls=(0, (2, 2)), zorder=1)
    for e, (a, b) in bounds.items():
        ax_l.text(a + (b - a) / 2, 1.0, e, transform=ax_l.get_xaxis_transform(), ha="center", va="bottom",
                  fontsize=6.5, color=plotting.INK)

    # a: daily level, one line per source, broken where a day has no level
    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"])
    for src in ("dahiti", "hydroweb_shifted"):
        y = np.where(d["source"] == src, d["level_m"], np.nan)
        ax_l.plot(d["date"], y, color=COLOURS[src], lw=0.75, zorder=3, solid_capstyle="butt")
    ax_l.plot(dahiti["time"], dahiti["level_m"], ls="none", marker="o", ms=1.2, mew=0, color=plotting.INK,
              zorder=4)
    ax_l.set_ylabel("Water level (m, DAHITI ref.)")
    ax_l.tick_params(labelbottom=False)
    ax_l.legend(handles=[Line2D([], [], color=COLOURS["dahiti"], lw=0.75, marker="o", ms=1.6, mew=0,
                                label="DAHITI records and their interpolation"),
                         Line2D([], [], color=COLOURS["hydroweb_shifted"], lw=0.75,
                                label=f"Hydroweb, shifted to DAHITI (DAHITI gaps > {a4.MAX_GAP_DAYS:.0f} d)")],
                loc="lower left", handlelength=1.6)
    plotting.panel_label(ax_l, "a", x=-0.068, y=1.0)

    # b, c: scenes per month, open-water months and ice season stacked
    x = months.to_timestamp(how="start") + pd.Timedelta(days=15)
    names = {"landsat": "Landsat scenes", "s2": "Sentinel-2 granules"}
    labels = {"open": "Open-water months (can enter stage 2)",
              "ice": f"Ice season, {m['params.ice_first_month']['value']}–{m['params.ice_last_month']['value']} "
                     "(excluded from stage 2)"}
    for ax, fam, letter in ((ax_ls, "landsat", "b"), (ax_s2, "s2", "c")):
        bottom = np.zeros(len(months))
        for season in ("open", "ice"):
            v = counts[fam][season].to_numpy()
            ax.bar(x, v, width=28, bottom=bottom, color=COLOURS[season], lw=0, zorder=3)
            bottom += v
        ax.set_ylim(0, None)
        ax.set_ylabel("Per month")
        ax.text(0.006, 0.97, names[fam], transform=ax.transAxes, ha="left", va="top", fontsize=6.5,
                color=plotting.INK, zorder=5)
        plotting.panel_label(ax, letter, x=-0.068, y=1.0)
    ax_ls.tick_params(labelbottom=False)
    ax_ls.legend(handles=[Patch(facecolor=COLOURS[s], edgecolor="none", label=labels[s]) for s in labels],
                 loc="upper left", bbox_to_anchor=(0.0, 0.84), handlelength=1.2)
    ax_s2.set_xlim(t0, t1)
    ax_s2.xaxis.set_major_locator(mdates.YearLocator(5))
    ax_s2.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax_s2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_s2.set_xlabel("Year")

    C.save(fig, "fig1_levels")
    plt.close(fig)
    print("levelled scenes:", len(sc), "|", {f: c.sum().to_dict() for f, c in counts.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
