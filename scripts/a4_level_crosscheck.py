"""Water-level cross-check / robustness check 5: DAHITI (target 39) versus Hydroweb (L_caspian).

For each DAHITI epoch, Hydroweb is linearly interpolated between its two nearest
records if both lie within +/-5 days. Differences (DAHITI - Hydroweb) give the
reference-surface offset, its drift and the robust scatter, overall and by period.
Also tabulates records per year in each series and the median difference per year.
Pairing is caspian.levels.pair_series. With a window argument (e.g. 20) the outputs
carry a _w<N>d suffix; the +/-20-day pair table is the input of the moving-median
gap-fill offset in a4_scene_levels.py.

Outputs (committed): results/qc/level_crosscheck[_w<N>d].json,
                     results/qc/level_crosscheck_pairs[_w<N>d].csv,
                     outputs/qc/level_crosscheck[_w<N>d].{pdf,png}
"""

from __future__ import annotations

import json
import sys

import caspian  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from caspian import levels, plotting
from caspian.config import OUTPUTS, RAW, RESULTS
from caspian.io import read_dahiti, read_hydroweb

MAX_HALF_GAP_DAYS = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
PERIODS = [(1992, 2002), (2002, 2012), (2012, 2022), (2022, 2027)]


def nmad(x):
    x = np.asarray(x, float)
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def pair(d: pd.DataFrame, h: pd.DataFrame, max_half_gap_days: float | None = None) -> pd.DataFrame:
    w = MAX_HALF_GAP_DAYS if max_half_gap_days is None else max_half_gap_days
    p = levels.pair_series(d["time"], d["level_m"], h["time"], h["level_m"], w)
    return p.rename(columns={"ref": "dahiti", "other": "hydroweb"})


def main() -> int:
    d = read_dahiti(RAW / "dahiti" / "dahiti_39_caspian_sea_water_level.csv").data
    hs = read_hydroweb(RAW / "hydroweb" / "extracted" / "HYDROWEB_LAKES_OPE" / "hydroprd_L_caspian.txt")
    h = hs.data
    p = pair(d, h)
    years = (p["time"] - pd.Timestamp("1992-01-01")) / pd.Timedelta(days=1) / 365.25  # fractional days
    slope, intercept = np.polyfit(years, p["diff"], 1)
    by_period = {}
    for a, b in PERIODS:
        q = p[(p["time"].dt.year >= a) & (p["time"].dt.year < b)]
        by_period[f"{a}-{b - 1}"] = {"n": int(len(q)), "median_diff_m": float(q["diff"].median()),
                                     "nmad_m": nmad(q["diff"])} if len(q) else {"n": 0}
    by_year = {int(y): {"n": int(len(g)), "median_diff_m": float(g["diff"].median())}
               for y, g in p.groupby(p["time"].dt.year)}
    per_year = pd.DataFrame({"dahiti": d.groupby(d["time"].dt.year).size(),
                             "hydroweb": h.groupby(h["time"].dt.year).size()}).fillna(0).astype(int)
    res = {
        "hydroweb_records": int(len(h)), "hydroweb_first": str(h["time"].min()), "hydroweb_last": str(h["time"].max()),
        "hydroweb_surface_of_reference": hs.meta.get("surface_of_reference"),
        "hydroweb_sigma_median_m": float(h["sigma_m"].median()),
        "n_pairs": int(len(p)), "max_half_gap_days": MAX_HALF_GAP_DAYS,
        "offset_median_m": float(p["diff"].median()), "offset_mean_m": float(p["diff"].mean()),
        "diff_nmad_m": nmad(p["diff"]), "diff_p05_m": float(p["diff"].quantile(0.05)),
        "diff_p95_m": float(p["diff"].quantile(0.95)),
        "drift_m_per_decade": float(slope * 10), "by_period": by_period, "by_year": by_year,
        "records_per_year": {int(k): v for k, v in per_year.to_dict(orient="index").items()},
    }
    sfx = "" if MAX_HALF_GAP_DAYS == 5.0 else f"_w{int(MAX_HALF_GAP_DAYS)}d"
    (RESULTS / "qc").mkdir(parents=True, exist_ok=True)
    (RESULTS / "qc" / f"level_crosscheck{sfx}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    p[["time", "dahiti", "hydroweb", "diff"]].to_csv(RESULTS / "qc" / f"level_crosscheck_pairs{sfx}.csv",
                                                      index=False, float_format="%.6f")

    plotting.use_style()
    fig, axs = plt.subplots(2, 1, figsize=(plotting.WIDTH_DOUBLE_MM * plotting.MM, 95 * plotting.MM),
                            sharex=True, gridspec_kw={"height_ratios": [2, 1]}, constrained_layout=True)
    axs[0].plot(h["time"], h["level_m"], color="#8a8a8a", lw=0.5, label=f"Hydroweb L_caspian (n={len(h)})")
    axs[0].plot(d["time"], d["level_m"], color="#1f4e8c", lw=0.7, label=f"DAHITI 39 (n={len(d)})")
    axs[0].set_ylabel("Level (m, each product's reference)")
    axs[0].legend(loc="lower left")
    axs[0].grid(axis="y")
    axs[1].scatter(p["time"], p["diff"], s=2, color="#1f4e8c", lw=0)
    axs[1].axhline(res["offset_median_m"], color="#c75b39", lw=0.7)
    axs[1].set_ylabel("DAHITI − Hydroweb (m)")
    axs[1].set_xlabel("Year")
    axs[1].grid(axis="y")
    plotting.panel_label(axs[0], "a")
    plotting.panel_label(axs[1], "b")
    plotting.save(fig, OUTPUTS / "qc" / f"level_crosscheck{sfx}", tiff=False)
    print(json.dumps({k: v for k, v in res.items() if k not in ("records_per_year", "by_year")}, indent=1))
    print("records per year (dahiti, hydroweb), all years:",
          {y: (v["dahiti"], v["hydroweb"]) for y, v in res["records_per_year"].items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
