"""A5: empirical wind-setup model, calibrated on SWOT (PREREG section 3).

1. SWOT per-pass, per-block water-surface elevation (scripts/a5_swot_sector_levels.py)
   minus the basin level at the pass time (DAHITI, or Hydroweb shifted onto the DAHITI
   reference: scripts/a4_scene_levels.levels_at, the scene-level code path).
2. Each block's long-term mean anomaly is removed. That mean holds the static
   difference between SWOT's geoid and the DAHITI reference, which varies across the
   shelf, plus the block's mean setup. What remains is the time-varying setup.
3. For each pass and sector, the de-meaned anomalies of the sector's blocks are
   averaged, separately for training and calibration blocks
   (results/study_area/blocks_25km.csv).
4. Per sector, the training anomalies are regressed on lagged ERA5 forcing
   (caspian.surge: |U|u and |U|v means over 0-6, 6-12, 12-24 and 24-48 h, and the
   0-6 h pressure anomaly). The fit is evaluated on calibration blocks; test blocks
   are never used.
5. The fitted model predicts setup for every levelled scene and sector.

Outputs:
- results/levels/setup_model.json (coefficients, fit and calibration statistics; committed);
- data/interim/levels/scene_setup.csv (scene x sector predicted setup; git-ignored).

Usage: python scripts/a5_setup_model.py
"""

from __future__ import annotations

import glob
import importlib.util
import json
import sys

import caspian  # noqa: F401
import numpy as np
import pandas as pd

from caspian import surge
from caspian.config import INTERIM, RESULTS

MIN_BLOCK_PASSES = 10      # a block needs this many passes for its mean to be removed
MIN_PASS_PX = 30           # a block-pass needs this many pixels
PASS_GAP_MIN = 10          # granules within this many minutes form one pass
RIDGE = 0.0


def _a4():
    spec = importlib.util.spec_from_file_location("a4", RESULTS.parent / "scripts" / "a4_scene_levels.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def sector_anomalies(sw: pd.DataFrame, blocks: pd.DataFrame, prep, a4) -> pd.DataFrame:
    sw = sw[sw["n_px"] >= MIN_PASS_PX].copy()
    sw["time"] = pd.to_datetime(sw["time_utc"])
    sw = sw.merge(blocks[["i", "j", "sector", "split"]], left_on=["block_i", "block_j"], right_on=["i", "j"])
    # passes: granules of one overpass are minutes apart
    t = np.sort(sw["time"].unique())
    pid = np.cumsum(np.r_[True, np.diff(t) > np.timedelta64(PASS_GAP_MIN, "m")])
    sw["pass_id"] = sw["time"].map(dict(zip(t, pid)))
    pass_t = sw.groupby("pass_id")["time"].median()
    lv, _ = a4.levels_at(prep, pass_t.values)
    sw["level_m"] = sw["pass_id"].map(dict(zip(pass_t.index, lv["level_m"].to_numpy())))
    sw = sw[np.isfinite(sw["level_m"])]
    # one value per block and pass (overlapping tiles: pixel-weighted mean of the medians)
    bp = (sw.assign(w=sw["n_px"], wm=sw["wse_median_m"] * sw["n_px"])
          .groupby(["pass_id", "block_i", "block_j", "sector", "split"], as_index=False)
          .agg(wm=("wm", "sum"), w=("w", "sum"), level_m=("level_m", "first")))
    bp["anom"] = bp["wm"] / bp["w"] - bp["level_m"]
    n_pass = bp.groupby(["block_i", "block_j"])["pass_id"].transform("nunique")
    bp = bp[n_pass >= MIN_BLOCK_PASSES].copy()
    bp["anom_dm"] = bp["anom"] - bp.groupby(["block_i", "block_j"])["anom"].transform("mean")
    sec = (bp.groupby(["pass_id", "sector", "split"], as_index=False)
           .agg(anom_m=("anom_dm", "mean"), n_blocks=("anom_dm", "size")))
    sec["time"] = sec["pass_id"].map(pass_t)
    return sec, bp


def main() -> int:
    a4 = _a4()
    prep = a4.prepare()
    sw = pd.read_csv(RESULTS / "levels" / "swot_block_levels.csv")
    blocks = pd.read_csv(RESULTS / "study_area" / "blocks_25km.csv")
    sec, bp = sector_anomalies(sw, blocks, prep, a4)
    era5 = surge.load_era5(sorted(glob.glob(str(INTERIM / "era5" / "era5_sector_winds_*.csv"))))
    out = {"inputs": {"min_block_passes": MIN_BLOCK_PASSES, "min_pass_px": MIN_PASS_PX, "ridge": RIDGE,
                      "windows_h": surge.WINDOWS_H, "n_passes": int(sec["pass_id"].nunique()),
                      "n_block_passes": int(len(bp))}, "sectors": {}}
    models = {}
    for s in sorted(sec["sector"].unique()):
        tr = sec[(sec["sector"] == s) & (sec["split"] == "train")].reset_index(drop=True)
        ca = sec[(sec["sector"] == s) & (sec["split"] == "calib")].reset_index(drop=True)
        if len(tr) < 30:
            out["sectors"][s] = {"n_train_passes": int(len(tr)), "fitted": False}
            continue
        Xtr = surge.forcing_features(era5, s, tr["time"])
        m = surge.fit_setup(Xtr, tr["anom_m"], ridge=RIDGE)
        res = {"n_train_passes": int(m["n"]), "fitted": True, "resid_sd_train_m": round(m["resid_sd"], 4),
               "anom_sd_train_m": round(float(np.nanstd(tr["anom_m"])), 4),
               "intercept": m["intercept"], "coef": m["coef"]}
        if len(ca):
            Xca = surge.forcing_features(era5, s, ca["time"])
            pred = surge.predict_setup(m, Xca)
            y = ca["anom_m"].to_numpy()
            ok = np.isfinite(pred) & np.isfinite(y)
            r = y[ok] - pred[ok]
            res.update(n_calib_passes=int(ok.sum()), resid_sd_calib_m=round(float(np.std(r)), 4),
                       anom_sd_calib_m=round(float(np.std(y[ok])), 4),
                       r2_calib=round(float(1 - np.var(r) / np.var(y[ok])), 3) if ok.sum() > 2 else None)
        out["sectors"][s] = res
        models[s] = m
    (RESULTS / "levels").mkdir(parents=True, exist_ok=True)
    (RESULTS / "levels" / "setup_model.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    # predicted setup (wind and pressure part, intercept removed) for every levelled scene
    sc = pd.read_csv(INTERIM / "levels" / "scene_levels.csv", usecols=["system:index", "sensor", "time", "epoch"],
                     low_memory=False)
    sc = sc[sc["epoch"].isin(["E1", "E2", "E3"])].reset_index(drop=True)
    rows = {"scene": sc["system:index"], "sensor": sc["sensor"], "time": sc["time"]}
    for s, m in models.items():
        X = surge.forcing_features(era5, s, pd.to_datetime(sc["time"]))
        rows[f"setup_{s}_m"] = np.round(surge.predict_setup(m, X) - m["intercept"], 4)
    pd.DataFrame(rows).to_csv(INTERIM / "levels" / "scene_setup.csv", index=False)
    print(json.dumps({k: {kk: v for kk, v in d.items() if kk != "coef"} for k, d in out["sectors"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
