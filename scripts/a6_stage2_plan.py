"""A6: the stage-2 scene plan (local).

One row per scene that enters stage 2:
- open-water months (D-062);
- touches the zone (stage-1 zone_px > 0);
- has a level (epochs E1-E3);
- has an acquisition threshold (D-061).

Columns: scene, sensor, acq, time, threshold, threshold_source, level_m, sigma_m,
epoch, bin, parity, the predicted setup in each sector (scripts/a5_setup_model.py),
and either
- `excluded_sectors`: the sector codes where |setup| exceeds the screening threshold
  tau (main export), or
- `class_<sector>`: the setup class of the scene in each sector (sensitivity export).

Level bins follow PREREG section 4 (caspian.design: 0.10 m from -30.00 m). Parity
splits each (epoch, bin) into two halves of acquisitions, alternating in time order,
for the noise floor (PREREG section 9; Amendment 3 item 3: acquisitions, not scenes).

Outputs: data/interim/stage2/plan_<tag>.csv

Usage: python scripts/a6_stage2_plan.py main <tau_m> | sensitivity
"""

from __future__ import annotations

import glob
import sys

import caspian  # noqa: F401
import numpy as np
import pandas as pd

from caspian import design, levels
from caspian.config import INTERIM
from caspian.geography import SECTOR_ORDER

ICE_MONTHS = (11, 12, 1, 2, 3)          # D-062
SETUP_CLASS_EDGES = (0.05, 0.10, 0.15, 0.20, 0.30)   # |setup| class edges, sensitivity export (m; D-064)
SECTOR_CODE = {s: i + 1 for i, s in enumerate(SECTOR_ORDER)}


def build() -> pd.DataFrame:
    s1 = pd.concat([pd.read_csv(p, usecols=["scene", "sensor", "acq", "time", "zone_px"])
                    for p in sorted(glob.glob(str(INTERIM / "stage1_v2" / "*.csv"))) if "thresholds" not in p],
                   ignore_index=True)
    s1["t"] = pd.to_datetime(s1["time"], unit="ms")
    s1 = s1[~s1["t"].dt.month.isin(ICE_MONTHS) & (pd.to_numeric(s1["zone_px"], errors="coerce") > 0)]
    thr = pd.read_csv(INTERIM / "stage1_v2" / "thresholds_by_acquisition.csv",
                      usecols=["acq", "threshold_used", "source"])
    lv = pd.read_csv(INTERIM / "levels" / "scene_levels.csv",
                     usecols=["system:index", "level_m", "sigma_m", "sigma_random_m", "sigma_offset_m", "epoch"],
                     low_memory=False).rename(columns={"system:index": "scene"})
    d = s1.merge(thr, on="acq", how="inner").merge(lv, on="scene", how="inner")
    setup_file = INTERIM / "levels" / "scene_setup.csv"
    if setup_file.exists():
        d = d.merge(pd.read_csv(setup_file).drop(columns=["sensor", "time"]), on="scene", how="left")
    else:  # only for pipeline tests before the setup model exists: no screening
        print("WARNING: no scene_setup.csv; plan has no setup screening (test use only)")
    d = d[d["epoch"].isin(["E1", "E2", "E3"]) & np.isfinite(d["level_m"]) & np.isfinite(d["threshold_used"])]
    d = d.rename(columns={"threshold_used": "threshold", "source": "threshold_source"})
    d["bin"] = levels.bin_index(d["level_m"].to_numpy(), design.BIN_WIDTH_M, design.BIN_ORIGIN_M)
    # parity per (epoch, bin): alternate acquisitions in time order
    acq_t = d.groupby(["epoch", "bin", "acq"], as_index=False)["t"].min().sort_values(["epoch", "bin", "t"])
    acq_t["parity"] = acq_t.groupby(["epoch", "bin"]).cumcount() % 2
    d = d.merge(acq_t[["epoch", "bin", "acq", "parity"]], on=["epoch", "bin", "acq"])
    return d.drop(columns=["zone_px"]).sort_values("t").reset_index(drop=True)


def main(argv) -> int:
    mode = argv[0] if argv else "main"
    d = build()
    setup_cols = [c for c in d.columns if c.startswith("setup_") and c.endswith("_m")]
    out = INTERIM / "stage2"
    out.mkdir(parents=True, exist_ok=True)
    base = d["epoch"] + "_b" + d["bin"].astype(int).map("{:03d}".format) + "_p" + d["parity"].astype(str)
    if mode == "main":
        tau = float(argv[1])
        ex = []
        for _, r in d[setup_cols].iterrows():
            ex.append(",".join(str(SECTOR_CODE[c[6:-2]]) for c in setup_cols
                               if np.isfinite(r[c]) and abs(r[c]) > tau))
        d["excluded_sectors"] = ex
        d["group"], d["sectors"], d["mask_mode"] = base, ex, "exclude"
        name = f"plan_main_tau{int(round(tau * 100)):03d}cm.csv"
    else:
        # one row per scene and setup class: the sectors where the scene falls in that class
        rows = []
        cls = {c[6:-2]: np.digitize(np.abs(d[c].to_numpy()), SETUP_CLASS_EDGES) for c in setup_cols}
        fin = {c[6:-2]: np.isfinite(d[c].to_numpy()) for c in setup_cols}
        for k in range(len(SETUP_CLASS_EDGES) + 1):
            sec_str = [",".join(str(SECTOR_CODE[s]) for s in cls if fin[s][i] and cls[s][i] == k) for i in range(len(d))]
            part = d.assign(group=base + f"_c{k}", sectors=sec_str, mask_mode="include")
            rows.append(part[part["sectors"] != ""])
        d = pd.concat(rows, ignore_index=True)
        name = "plan_sensitivity.csv"
    d.to_csv(out / name, index=False)
    print(out / name, len(d), "scenes |", d.groupby("epoch")["bin"].nunique().to_dict(), "bins per epoch |",
          d["sensor"].value_counts().to_dict())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
