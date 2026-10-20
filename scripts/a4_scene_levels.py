"""A4: water level and 1-sigma for every inventoried scene (D-018, D-020).

Primary: DAHITI target 39; per-record 1-sigma = sqrt(wse_u^2 + floor^2), with the
floor = empirical neighbour-interpolation noise of the series. Gap-fill: Hydroweb
L_caspian shifted by the median DAHITI - Hydroweb offset (+/-20-day pairs), 1-sigma
inflated by the offset NMAD; used only where DAHITI's bracketing gap exceeds 60 days
(PREREG section 4). Between epochs, the Brownian-bridge variance of caspian.levels.
Scenes before the altimetry record (1984 - Sep 1992) get no level here (PREREG
section 2 fallback).

Epochs (PREREG section 4): E1 start - date of the DAHITI maximum; E2 to 2010-12-31; E3 after.

Outputs: data/interim/levels/scene_levels.csv (git-ignored; one row per scene)
         results/levels/scene_levels_summary.json (committed)
"""

from __future__ import annotations

import json
import sys

import caspian  # noqa: F401
import numpy as np
import pandas as pd

from caspian import levels
from caspian.config import INTERIM, RAW, RESULTS
from caspian.io import read_dahiti, read_hydroweb

MAX_GAP_DAYS = 60.0


def main() -> int:
    d = read_dahiti(RAW / "dahiti" / "dahiti_39_caspian_sea_water_level.csv").data
    hw = read_hydroweb(RAW / "hydroweb" / "extracted" / "HYDROWEB_LAKES_OPE" / "hydroprd_L_caspian.txt").data
    xc = json.loads((RESULTS / "qc" / "level_crosscheck_w20d.json").read_text(encoding="utf-8"))
    offset, offset_nmad = xc["offset_median_m"], xc["diff_nmad_m"]

    floor_d = levels.empirical_sigma(d["time"], d["level_m"])
    sig_d = np.sqrt(np.nan_to_num(d["sigma_m"].to_numpy()) ** 2 + floor_d ** 2)
    q_d = levels.estimate_diffusion(d["time"], d["level_m"], sig_d, max_gap_days=MAX_GAP_DAYS)

    h_level = hw["level_m"].to_numpy() + offset
    floor_h = levels.empirical_sigma(hw["time"], hw["level_m"])
    noise_h = np.maximum(np.nan_to_num(hw["sigma_m"].to_numpy()), floor_h)  # per-record random noise
    # q describes the level process itself (the same sea), so DAHITI's estimate is used
    # for both series. Estimating it from Hydroweb gives 0 because its early monthly
    # records carry conservative SDs (0.07-0.11 m) that exceed the true month-to-month noise.
    q_h = q_d
    sig_h = np.sqrt(noise_h ** 2 + offset_nmad ** 2)

    scenes = pd.read_csv(INTERIM / "inventory" / "scenes.csv", parse_dates=["time"])
    ld = levels.interpolate(d["time"], d["level_m"], sig_d, scenes["time"], q=q_d, max_gap_days=MAX_GAP_DAYS)
    lh = levels.interpolate(hw["time"], h_level, sig_h, scenes["time"], q=q_h, max_gap_days=MAX_GAP_DAYS)
    use_d = ~ld["in_gap"].to_numpy()
    use_h = ~use_d & ~lh["in_gap"].to_numpy()
    scenes["level_m"] = np.where(use_d, ld["level_m"], np.where(use_h, lh["level_m"], np.nan))
    scenes["sigma_m"] = np.where(use_d, ld["sigma_m"], np.where(use_h, lh["sigma_m"], np.nan))
    scenes["gap_days"] = np.where(use_d, ld["gap_days"], np.where(use_h, lh["gap_days"], np.nan))
    scenes["level_source"] = np.select([use_d, use_h], ["dahiti", "hydroweb_shifted"], "none")

    t_max = d.loc[d["level_m"].idxmax(), "time"]
    scenes["epoch"] = np.select([scenes["time"] <= t_max, scenes["time"] <= pd.Timestamp("2010-12-31 23:59:59")],
                                ["E1", "E2"], "E3")
    scenes.loc[scenes["level_source"] == "none", "epoch"] = np.where(
        scenes.loc[scenes["level_source"] == "none", "time"] < d["time"].min(), "pre_altimetry", "no_level")
    out = INTERIM / "levels"
    out.mkdir(parents=True, exist_ok=True)
    scenes.to_csv(out / "scene_levels.csv", index=False)

    tab = scenes.pivot_table(index="sensor", columns="level_source", values="time", aggfunc="count", fill_value=0)
    ep = scenes.pivot_table(index="sensor", columns="epoch", values="time", aggfunc="count", fill_value=0)
    ok = scenes["level_source"] != "none"
    summary = {
        "dahiti_noise_floor_m": round(floor_d, 4), "hydroweb_noise_floor_m": round(floor_h, 4),
        "q_dahiti_m2_per_day": q_d, "q_hydroweb_m2_per_day": q_h,
        "hydroweb_offset_m": offset, "hydroweb_offset_nmad_m": offset_nmad,
        "max_gap_days": MAX_GAP_DAYS, "epoch_E1_end": str(t_max),
        "n_scenes": int(len(scenes)), "by_sensor_source": tab.to_dict(orient="index"),
        "by_sensor_epoch": ep.to_dict(orient="index"),
        "sigma_m_percentiles_p50_p90_p99": [float(scenes.loc[ok, "sigma_m"].quantile(q)) for q in (0.5, 0.9, 0.99)],
        "level_range_m": [float(scenes.loc[ok, "level_m"].min()), float(scenes.loc[ok, "level_m"].max())],
    }
    (RESULTS / "levels").mkdir(parents=True, exist_ok=True)
    (RESULTS / "levels" / "scene_levels_summary.json").write_text(json.dumps(summary, indent=1, default=int),
                                                                   encoding="utf-8")
    print(json.dumps(summary, indent=1, default=int))
    return 0


if __name__ == "__main__":
    sys.exit(main())
