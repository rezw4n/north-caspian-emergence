"""A3: per-acquisition MNDWI thresholds from stage-1 run 2 histograms (local; D-061).

For each acquisition (one overpass: Landsat sensor + path + date, Sentinel-2
datatake), the 256-bucket in-zone histograms of its scenes are summed and Otsu's
threshold is computed from the pooled sample (caspian.thresholds). A pooled threshold
is valid when all of:
- the pooled sample has at least MIN_N pixels (240 m);
- each class holds at least MIN_SHARE of it;
- eta (between-class over total variance) is at least MIN_ETA;
- the valley ratio is at most MAX_VALLEY;
- the threshold lies in THR_RANGE. Open water absorbs strongly in the shortwave
  infrared (SWIR1 below green), so its MNDWI is positive, and a threshold below zero
  would class pixels with SWIR1 above green as water. Run 2 showed 15.5% of pooled
  thresholds below zero, many passing the bimodality tests, in overpasses whose
  in-zone sample is mostly land: Otsu then splits dry from moist land (D-061).

Acquisitions that fail take the median of the valid thresholds of the same sensor
and calendar month (the fallback); the source of every threshold is recorded.

Outputs:
- data/interim/stage1_v2/thresholds_by_acquisition.csv (git-ignored);
- results/qc/threshold_rule_summary.json (committed).

Usage: python scripts/a3_thresholds.py [--diagnose]
  --diagnose : print the distributions used to fix the rule constants, and exit
"""

from __future__ import annotations

import glob
import json
import sys

import caspian  # noqa: F401
import numpy as np
import pandas as pd

from caspian import thresholds
from caspian.config import INTERIM, RESULTS

# Rule constants, fixed from the run-2 diagnostics before stage 2 (D-061)
MIN_N = 2000
MIN_SHARE = 0.05
MIN_ETA = 0.70
MAX_VALLEY = 0.50
THR_RANGE = (0.0, 0.5)
ICE_MONTHS = (11, 12, 1, 2, 3)   # ice-affected, excluded from stage 2 (D-062; results/qc/snow_rules_run2.json)


def load() -> pd.DataFrame:
    d = pd.concat([pd.read_csv(p, dtype={"hist": str}) for p in sorted(glob.glob(str(INTERIM / "stage1_v2" / "*.csv")))
                   if not p.endswith("thresholds_by_acquisition.csv")], ignore_index=True)
    d["t"] = pd.to_datetime(d["time"], unit="ms")
    d["month"] = d["t"].dt.month
    return d[~d["month"].isin(ICE_MONTHS)]


def per_acquisition(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for acq, g in d.groupby("acq"):
        h = thresholds.pooled([thresholds.parse_hist(s) for s in g["hist"]])
        diag = thresholds.diagnostics(h)
        rows.append({"acq": acq, "sensor": g["sensor"].iloc[0], "month": int(g["month"].iloc[0]),
                     "time": g["t"].min(), "n_scenes": len(g), **diag})
    return pd.DataFrame(rows)


def main(argv) -> int:
    d = load()
    a = per_acquisition(d)
    if "--diagnose" in argv:
        ow = a[~a["month"].isin(ICE_MONTHS) & (a["n"] >= MIN_N)]
        print("acquisitions:", len(a), "| open-water months with n >= MIN_N:", len(ow))
        for c in ("threshold", "eta", "valley_ratio", "share_low", "share_high"):
            print(c, ow[c].quantile([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).round(3).to_dict())
        print("negative thresholds:", int((ow["threshold"] < 0).sum()), "of", len(ow))
        for lo, hi in ((-1, 0), (0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 1)):
            m = ow["threshold"].between(lo, hi, inclusive="left")
            print(f"thr [{lo},{hi}): n={m.sum()} eta_med={ow.loc[m, 'eta'].median():.3f} "
                  f"valley_med={ow.loc[m, 'valley_ratio'].median():.3f}")
        return 0
    if MIN_ETA is None or MAX_VALLEY is None:
        raise SystemExit("rule constants not fixed yet (D-061); run with --diagnose")
    valid = ((a["n"] >= MIN_N) & (a[["share_low", "share_high"]].min(axis=1) >= MIN_SHARE)
             & (a["eta"] >= MIN_ETA) & (a["valley_ratio"] <= MAX_VALLEY)
             & a["threshold"].between(*THR_RANGE))
    a["source"] = np.where(valid, "pooled_otsu", "fallback")
    fb = a[valid].groupby(["sensor", "month"])["threshold"].median()
    a["threshold_used"] = np.where(valid, a["threshold"],
                                   [fb.get((s, m), np.nan) for s, m in zip(a["sensor"], a["month"])])
    out = INTERIM / "stage1_v2" / "thresholds_by_acquisition.csv"
    a.to_csv(out, index=False)
    ow = ~a["month"].isin(ICE_MONTHS)
    summ = {"rule": {"min_n": MIN_N, "min_share": MIN_SHARE, "min_eta": MIN_ETA, "max_valley": MAX_VALLEY,
                     "threshold_range": THR_RANGE, "ice_months_excluded": ICE_MONTHS,
                     "fallback": "median of valid thresholds, same sensor and calendar month"},
            "n_acquisitions": int(len(a)), "n_open_water_months": int(ow.sum()),
            "share_valid_open_water": round(float(valid[ow].mean()), 4),
            "by_sensor": {s: {"n": int(((a.sensor == s) & ow).sum()), "valid": int(((a.sensor == s) & ow & valid).sum())}
                          for s in sorted(a["sensor"].unique())},
            "fallback_table": {f"{s}_{m:02d}": round(float(v), 4) for (s, m), v in fb.items()}}
    (RESULTS / "qc").mkdir(parents=True, exist_ok=True)
    (RESULTS / "qc" / "threshold_rule_summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summ.items() if k != "fallback_table"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
