"""A9 generality test, local scene levels, thresholds and stage-2 plans (PREREG
Amendment 16 items 2, 5 and 6; D-094).

Commands:
  levels <basin>      the level of each arm on each scene's acquisition date (UTC;
                      caspian.generality.level_on_dates), the arm's epoch and bin.
                      Output: data/interim/generality/<basin>/scene_levels.csv
  thresholds <basin>  per-acquisition MNDWI thresholds from the stage-1 histograms with
                      the D-061 rule and fallback (scripts/a3_thresholds.py, unchanged
                      constants). Output: thresholds_by_acquisition.csv and
                      results/generality/<basin>/threshold_rule_summary.json
  plan <basin> <tag> [--placeholder THR]
                      the stage-2 plan: one row per scene and arm whose scene touches
                      the zone (stage 1 zone_px > 0) and has a level and a threshold; group
                      = <epoch>_b<bin>_p<parity> of that arm. With --placeholder, every
                      scene gets the constant threshold THR and the zone test is skipped
                      (cost pilot only; the counts are not used).
Usage: python scripts/g1_plan.py <command> <basin> [...]
"""

from __future__ import annotations

import argparse
import json
import sys

import caspian  # noqa: F401
import numpy as np
import pandas as pd

from caspian import design, levels, thresholds
from caspian import generality as G
from caspian.config import INTERIM, RAW, RESULTS

MIN_N, MIN_SHARE, MIN_ETA, MAX_VALLEY, THR_RANGE = 2000, 0.05, 0.70, 0.50, (0.0, 0.5)   # D-061


def d_int(basin):
    return INTERIM / "generality" / basin


def d_res(basin):
    d = RESULTS / "generality" / basin
    d.mkdir(parents=True, exist_ok=True)
    return d


def arm_levels(basin) -> pd.DataFrame:
    if basin == "great_salt_lake":
        lv = pd.read_csv(RAW / "generality" / basin / "levels" / "gsl_daily_levels.csv", parse_dates=["date"])
        return lv.rename(columns={"level_navd88_m": "level"})[["arm", "date", "level"]]
    raise NotImplementedError(basin)


def scene_levels(basin) -> int:
    sc = pd.read_csv(d_int(basin) / "scenes.csv")
    sc["t"] = pd.to_datetime(sc["time"], unit="ms")
    lv = arm_levels(basin)
    off = G.BASINS[basin]["level_offset_m"]
    rec = {"rule": "Amendment 16 items 2 and 5", "arms": {}}
    out = sc[["scene", "sensor", "acq", "time", "t", "path", "row", "cloud_scene_pct"]].copy()
    for arm, g in lv.groupby("arm"):
        g = g[g["date"] >= "1984-01-01"].set_index("date")["level"].sort_index()
        ex = levels.smoothed_extremes(g.dropna().index, g.dropna().to_numpy(), k=3)
        e1_end = pd.Timestamp(ex["time_of_max"])
        lvl = G.level_on_dates(g, out["t"].dt.normalize())
        ep = G.epochs(out["t"], e1_end)
        b = levels.bin_index(lvl - off, design.BIN_WIDTH_M, design.BIN_ORIGIN_M)
        out[f"level_{arm}"] = lvl
        out[f"epoch_{arm}"] = np.where(np.isfinite(lvl), ep, "")
        out[f"bin_{arm}"] = np.where(np.isfinite(lvl), b, -1)
        rec["arms"][arm] = {"e1_end": str(e1_end.date()), "max_m": round(float(ex["max_m"]), 3),
                            "n_scenes_with_level": int(np.isfinite(lvl).sum()), "n_scenes": len(out),
                            "level_range_m": [round(float(np.nanmin(lvl)), 3), round(float(np.nanmax(lvl)), 3)],
                            "n_by_epoch": pd.Series(ep[np.isfinite(lvl)]).value_counts().to_dict()}
    out["has_level"] = np.column_stack([np.isfinite(out[f"level_{a}"]) for a in rec["arms"]]).any(axis=1)
    out.to_csv(d_int(basin) / "scene_levels.csv", index=False)
    (d_res(basin) / "scene_levels.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(json.dumps(rec, indent=1))
    return 0


def thresholds_cmd(basin) -> int:
    d = pd.read_csv(d_int(basin) / "stage1.csv", dtype={"hist": str})
    d["t"] = pd.to_datetime(d["time"], unit="ms")
    d["month"] = d["t"].dt.month
    rows = []
    for acq, g in d.groupby("acq"):
        hs = [thresholds.parse_hist(s) for s in g["hist"]]
        hs = [h for h in hs if h is not None]
        if not hs:
            rows.append({"acq": acq, "sensor": g["sensor"].iloc[0], "month": int(g["month"].iloc[0]),
                         "time": g["t"].min(), "n_scenes": len(g), "n": 0})
            continue
        diag = thresholds.diagnostics(thresholds.pooled(hs))
        rows.append({"acq": acq, "sensor": g["sensor"].iloc[0], "month": int(g["month"].iloc[0]),
                     "time": g["t"].min(), "n_scenes": len(g), **diag})
    a = pd.DataFrame(rows)
    for c in ("threshold", "eta", "valley_ratio", "share_low", "share_high"):
        if c not in a:
            a[c] = np.nan
    valid = ((a["n"] >= MIN_N) & (a[["share_low", "share_high"]].min(axis=1) >= MIN_SHARE)
             & (a["eta"] >= MIN_ETA) & (a["valley_ratio"] <= MAX_VALLEY) & a["threshold"].between(*THR_RANGE))
    a["source"] = np.where(valid, "pooled_otsu", "fallback")
    fb = a[valid].groupby(["sensor", "month"])["threshold"].median()
    a["threshold_used"] = np.where(valid, a["threshold"], [fb.get((s, m), np.nan) for s, m in zip(a["sensor"], a["month"])])
    a.to_csv(d_int(basin) / "thresholds_by_acquisition.csv", index=False)
    summ = {"rule": {"min_n": MIN_N, "min_share": MIN_SHARE, "min_eta": MIN_ETA, "max_valley": MAX_VALLEY,
                     "threshold_range": THR_RANGE, "fallback": "median of valid thresholds, same sensor and calendar month",
                     "source": "D-061, unchanged"},
            "n_acquisitions": int(len(a)), "share_valid": round(float(valid.mean()), 4),
            "n_no_threshold": int(a["threshold_used"].isna().sum()),
            "by_sensor": {s: {"n": int((a.sensor == s).sum()), "valid": int(((a.sensor == s) & valid).sum())}
                          for s in sorted(a["sensor"].unique())},
            "valid_threshold_quantiles": {str(q): round(float(a.loc[valid, "threshold"].quantile(q)), 4)
                                          for q in (0.05, 0.5, 0.95)} if valid.any() else {}}
    (d_res(basin) / "threshold_rule_summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(json.dumps(summ, indent=1))
    return 0


def plan_cmd(basin, tag, placeholder=None) -> int:
    sl = pd.read_csv(d_int(basin) / "scene_levels.csv")
    sl["t"] = pd.to_datetime(sl["time"], unit="ms")
    if placeholder is None:
        s1 = pd.read_csv(d_int(basin) / "stage1.csv", usecols=["scene", "zone_px"])
        touch = set(s1.loc[pd.to_numeric(s1["zone_px"], errors="coerce").fillna(0) > 0, "scene"])
        thr = pd.read_csv(d_int(basin) / "thresholds_by_acquisition.csv", usecols=["acq", "threshold_used", "source"])
        sl = sl[sl["scene"].isin(touch)].merge(thr, on="acq", how="inner")
    else:
        sl = sl.assign(threshold_used=float(placeholder), source="placeholder (cost pilot)")
    sl = sl[np.isfinite(sl["threshold_used"])]
    parts = []
    for arm in G.BASINS[basin]["sectors"]:
        p = sl[np.isfinite(sl[f"level_{arm}"])].copy()
        p["arm"], p["level_m"], p["epoch"], p["bin"] = arm, p[f"level_{arm}"], p[f"epoch_{arm}"], p[f"bin_{arm}"].astype(int)
        p["parity"] = G.parity(p)
        p["group"] = p["epoch"] + "_b" + p["bin"].map("{:03d}".format) + "_p" + p["parity"].astype(str)
        parts.append(p[["scene", "sensor", "acq", "time", "t", "threshold_used", "source", "arm", "level_m", "epoch",
                        "bin", "parity", "group"]].rename(columns={"threshold_used": "threshold"}))
    plan = pd.concat(parts, ignore_index=True)
    plan.to_csv(d_int(basin) / f"plan_{tag}.csv", index=False)
    summ = {"tag": tag, "placeholder_threshold": placeholder,
            "arms": {a: {"n_rows": int(len(g)), "n_acquisitions": int(g["acq"].nunique()), "n_groups": int(g["group"].nunique()),
                         "n_bins_by_epoch": g.groupby("epoch")["bin"].nunique().to_dict(),
                         "parity0_share_median": round(float(g.groupby(["epoch", "bin"])["parity"].apply(lambda x: (x == 0).mean()).median()), 3)}
                     for a, g in plan.groupby("arm")}}
    (d_res(basin) / f"plan_{tag}.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(json.dumps(summ, indent=1))
    return 0


def main(argv) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("basin")
    ap.add_argument("tag", nargs="?", default="main")
    ap.add_argument("--placeholder", type=float, default=None)
    a = ap.parse_args(argv)
    if a.cmd == "levels":
        return scene_levels(a.basin)
    if a.cmd == "thresholds":
        return thresholds_cmd(a.basin)
    if a.cmd == "plan":
        return plan_cmd(a.basin, a.tag, a.placeholder)
    raise SystemExit(f"unknown command {a.cmd}")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
