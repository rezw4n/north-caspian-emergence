"""Revision analysis V: further level statistics. Every number comes from the level
files.

1. The fall quoted in the abstract: the range of the 3-record running median of DAHITI
   (the rule that sets the end of E1), against the range of the single records.
2. Annual means on the DAHITI datum and in BHS-77 (H_BHS = H_DAHITI + delta):
   (a) DAHITI only, the daily linear interpolant with gaps of at most 60 days, months
       and years complete (the offset's averaging);
   (b) DAHITI with no gap limit;
   (c) the scene-level path (DAHITI, Hydroweb shifted onto DAHITI in gaps over 60
       days; scripts/a4_scene_levels.levels_at) evaluated daily at 12:00.
3. The 2026 reference level (the mean of the 2026 DAHITI records, as
   scripts/a8_emergence.py) in BHS; the seasonal bias of a January-September mean
   against the calendar-year mean in the years with complete DAHITI months; the
   end-2024 level for the comparison with Ginzburg et al. (2025).
4. Level control at the lowest levels: DAHITI records per year, the longest gaps, and
   per E3 level bin the share of scenes levelled by shifted Hydroweb and their 1-sigma.
5. CASPCOM's own annual means where the bulletins print them (refs/caspcom_bulletin_
   levels.csv; and the 2025 post values of Bulletin 31, transcribed in
   results/revision/v/caspcom_bulletin31_2025.csv with the verbatim quote).

Outputs: results/revision/v/level_stats.json, annual_levels.csv, e3_bin_level_control.csv.
Usage: python scripts/rev_v_levels.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import caspian  # noqa: F401
import numpy as np
import pandas as pd

from caspian import design, levels
from caspian import gauges as G
from caspian.config import INTERIM, RAW, RESULTS, ROOT
from caspian.io import read_dahiti

SCRIPTS = Path(__file__).resolve().parent
OUT = RESULTS / "revision" / "v"
DAHITI = RAW / "dahiti" / "dahiti_39_caspian_sea_water_level.csv"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def running_median_extremes(t, h, k: int = 3) -> dict:
    """Extremes of the k-record centred running median (consecutive records), with
    dates; and of the single records."""
    s = pd.Series(np.asarray(h, float), index=pd.to_datetime(t))
    rm = s.rolling(k, center=True).median().dropna()
    return {"raw_max_m": float(s.max()), "raw_max_date": str(s.idxmax().date()),
            "raw_min_m": float(s.min()), "raw_min_date": str(s.idxmin().date()),
            "raw_range_m": float(s.max() - s.min()),
            "smoothed_max_m": float(rm.max()), "smoothed_max_date": str(rm.idxmax().date()),
            "smoothed_min_m": float(rm.min()), "smoothed_min_date": str(rm.idxmin().date()),
            "smoothed_range_m": float(rm.max() - rm.min()), "k": k}


def annual_from_monthly(m: pd.DataFrame) -> pd.DataFrame:
    """Calendar-year mean of 12 complete monthly means (NaN otherwise)."""
    g = m.groupby("year")["level_m"]
    out = g.agg(["mean", "count"]).reset_index()
    out.loc[out["count"] < 12, "mean"] = np.nan
    return out.rename(columns={"mean": "level_m", "count": "n_months"})


def window_mean(m: pd.DataFrame, months) -> pd.Series:
    """Per year, the mean of the given complete monthly means (NaN if one is missing)."""
    w = m[m["month"].isin(months)]
    g = w.groupby("year")["level_m"]
    s = g.mean()
    s[g.count() < len(months)] = np.nan
    return s


def main() -> int:
    d = read_dahiti(DAHITI).data
    off = json.loads((RESULTS / "levels" / "dahiti_bhs_offset.json").read_text(encoding="utf-8"))
    delta = float(off["primary"]["delta_m"])
    sig_bhs = float(off["posts_and_years_bootstrap"]["sigma_m"])
    summ = json.loads((RESULTS / "qc" / "dahiti_summary.json").read_text(encoding="utf-8"))
    rec = {"delta_m": delta, "delta_sigma_posts_years_m": sig_bhs,
           "bhs_rule": "H_BHS-77 = H_DAHITI + delta (D-093); a basin-wide level in BHS carries sigma 0.05 m"}

    # 1. the 3.05 m fall
    ex = running_median_extremes(d["time"], d["level_m"])
    sm = summ["smoothed"]
    if abs(ex["smoothed_range_m"] - sm["range_m"]) > 1e-6 or abs(ex["raw_range_m"] - (summ["level_max_m"] - summ["level_min_m"])) > 1e-6:
        raise SystemExit("running-median extremes differ from results/qc/dahiti_summary.json")
    rec["fall"] = {**ex, "smoothed_max_bhs_m": ex["smoothed_max_m"] + delta, "smoothed_min_bhs_m": ex["smoothed_min_m"] + delta,
                   "raw_max_bhs_m": ex["raw_max_m"] + delta, "raw_min_bhs_m": ex["raw_min_m"] + delta,
                   "note": "the 3.05 m is the range of the 3-record running median (D-021 rule), from its maximum "
                           "(the E1 end) to its minimum; single records give the raw range"}

    # 2. annual means
    a = annual_from_monthly(G.monthly_means_daily(d["time"], d["level_m"], 60.0)).rename(
        columns={"level_m": "dahiti_gap60_m", "n_months": "n_months_gap60"})
    b = annual_from_monthly(G.monthly_means_daily(d["time"], d["level_m"], None)).rename(
        columns={"level_m": "dahiti_nogap_m", "n_months": "n_months_nogap"})
    a4 = _load("a4_scene_levels")
    prep = a4.prepare()
    days = pd.date_range("1993-01-01", "2026-09-14", freq="D") + pd.Timedelta(hours=12)
    lv, _ = a4.levels_at(prep, days)
    daily = pd.DataFrame({"time": days, "level_m": lv["level_m"].to_numpy(), "src": lv["level_source"].to_numpy(),
                          "sigma_offset_m": lv["sigma_offset_m"].to_numpy()})
    daily["year"] = daily["time"].dt.year
    c = daily.groupby("year").agg(scene_path_m=("level_m", "mean"), n_days=("level_m", "count"),
                                  share_hydroweb_days=("src", lambda s: float((s == "hydroweb_shifted").mean())),
                                  n_days_no_level=("level_m", lambda s: int(s.isna().sum()))).reset_index()
    full_year = daily.groupby("year")["time"].count() >= 365
    c.loc[~c["year"].map(full_year).fillna(False).to_numpy() | (c["n_days_no_level"] > 0), "scene_path_m"] = np.nan
    ann = a.merge(b, on="year", how="outer").merge(c, on="year", how="outer").sort_values("year")
    for col in ("dahiti_gap60_m", "dahiti_nogap_m", "scene_path_m"):
        ann[col.replace("_m", "_bhs_m")] = ann[col] + delta
    ann["records"] = ann["year"].map(pd.Series(pd.to_datetime(d["time"]).dt.year).value_counts())
    OUT.mkdir(parents=True, exist_ok=True)
    ann.round(4).to_csv(OUT / "annual_levels.csv", index=False)

    def yr(y, col):
        v = ann.loc[ann["year"] == y, col]
        return None if v.empty or not np.isfinite(v.iloc[0]) else float(v.iloc[0])
    rec["annual"] = {str(y): {k: yr(y, k) for k in ("dahiti_gap60_bhs_m", "dahiti_nogap_bhs_m", "scene_path_bhs_m",
                                                     "share_hydroweb_days", "records")}
                     for y in (1995, 2015, 2021, 2023, 2024, 2025)}
    best = {y: next((yr(y, k) for k in ("dahiti_gap60_bhs_m", "dahiti_nogap_bhs_m", "scene_path_bhs_m")
                     if yr(y, k) is not None), None) for y in (1995, 2025)}
    rec["annual_fall_1995_2025"] = {
        "rule": "1995 from DAHITI with complete months (gap <= 60 d); 2025 from the first of: DAHITI gap <= 60 d, "
                "DAHITI no gap limit, the scene-level path, that has a complete year",
        "1995_bhs_m": best[1995], "2025_bhs_m": best[2025],
        "fall_m": None if None in best.values() else best[1995] - best[2025],
        "2025_alternatives_bhs_m": rec["annual"]["2025"]}

    # 3. the 2026 reference and the seasonal bias of a January-September mean
    t = pd.to_datetime(d["time"])
    y26 = d[t.dt.year == 2026]
    l26 = float(y26["level_m"].mean())
    mo = G.monthly_means_daily(d["time"], d["level_m"], 60.0)
    js = window_mean(mo, range(1, 10))
    an = annual_from_monthly(mo).set_index("year")["level_m"]
    bias = (js - an).dropna()
    nd = G.monthly_means_daily(d["time"], d["level_m"], None)
    js_n, an_n = window_mean(nd, range(1, 10)), annual_from_monthly(nd).set_index("year")["level_m"]
    dec = daily[(daily["time"] >= "2024-12-01") & (daily["time"] < "2025-01-01")]
    rec["reference_2026"] = {
        "level_m": l26, "level_bhs_m": l26 + delta, "n_records": int(len(y26)),
        "first": str(pd.to_datetime(y26["time"]).min().date()), "last": str(pd.to_datetime(y26["time"]).max().date()),
        "jan_sep_minus_annual_mean_m": {"median": float(bias.median()), "q25": float(bias.quantile(0.25)),
                                         "q75": float(bias.quantile(0.75)), "n_years": int(bias.size),
                                         "years": [int(bias.index.min()), int(bias.index.max())]},
        "jan_sep_2025": {"dahiti_nogap_bhs_m": float(js_n.get(2025, np.nan) + delta),
                         "scene_path_bhs_m": float(daily[(daily["year"] == 2025) & (daily["time"].dt.month <= 9)]
                                                   ["level_m"].mean() + delta)},
        "jan_sep_2026_scene_path_bhs_m": float(daily[(daily["year"] == 2026)]["level_m"].mean() + delta),
        "dec_2024_scene_path_bhs_m": float(dec["level_m"].mean() + delta),
        "dec_2024_share_hydroweb_days": float((dec["src"] == "hydroweb_shifted").mean()),
        "annual_2024_bhs_m": rec["annual"]["2024"],
        "ginzburg2025_end2024_bhs_m": -29.34,
        "ginzburg_source": "lit.ginzburg2025.level_end2024_bhs_m (abstract; its averaging is not stated there)"}
    rec["reference_2026"]["annual_equivalent_bhs_m"] = l26 + delta - float(bias.median())

    # 4. level control at the lowest levels
    rpy = summ["records_per_year"]
    rec["level_control"] = {
        "dahiti_records_per_year": {y: rpy[y] for y in ("2021", "2022", "2023", "2024", "2025", "2026")},
        "max_gap_days": summ["max_gap_days"], "time_of_max_gap_start": summ["time_of_max_gap_start"],
        "gaps_over_60_days_by_year": {}}
    tt = t.sort_values().reset_index(drop=True)
    gaps = pd.DataFrame({"start": tt[:-1].to_numpy(), "days": np.diff(tt.to_numpy()) / np.timedelta64(1, "D")})
    gg = gaps[gaps["days"] > 60]
    rec["level_control"]["gaps_over_60_days"] = [{"start": str(pd.Timestamp(r.start).date()), "days": round(float(r.days), 1)}
                                                 for r in gg.itertuples()]
    sc = pd.read_csv(INTERIM / "levels" / "scene_levels.csv", usecols=["time", "level_m", "sigma_m", "sigma_offset_m",
                                                                       "level_source", "epoch", "sensor"],
                     low_memory=False)
    sc = sc[sc["epoch"] == "E3"].copy()
    sc["bin"] = levels.bin_index(sc["level_m"].to_numpy(), design.BIN_WIDTH_M, design.BIN_ORIGIN_M)
    sc["month"] = pd.to_datetime(sc["time"], format="mixed").dt.month
    sc = sc[sc["month"].between(4, 10)]
    tab = (sc.groupby("bin").agg(n_scenes=("level_m", "size"),
                                 share_hydroweb=("level_source", lambda s: float((s == "hydroweb_shifted").mean())),
                                 sigma_p50_m=("sigma_m", "median"), sigma_offset_p50_m=("sigma_offset_m", "median"),
                                 first=("time", "min"), last=("time", "max")).reset_index())
    tab["level_m"] = [float(levels.bin_center(b, design.BIN_WIDTH_M, design.BIN_ORIGIN_M)) for b in tab["bin"]]
    tab = tab[tab["level_m"] < -27.95]
    tab.round(4).to_csv(OUT / "e3_bin_level_control.csv", index=False)
    rec["level_control"]["e3_bins_below_minus_28"] = {
        "note": "all April-October E3 scenes of the scene table (Landsat and Sentinel-2), per 0.10 m bin; "
                "the stage-2 plan of the audit is a subset",
        "share_hydroweb_scenes": float((sc[sc["level_m"] < -27.95]["level_source"] == "hydroweb_shifted").mean()),
        "n_scenes": int((sc["level_m"] < -27.95).sum())}
    sm5 = json.loads((RESULTS / "levels" / "scene_levels_summary.json").read_text(encoding="utf-8"))
    rec["level_control"]["hydroweb_shift"] = {k: sm5.get(k) for k in ("hydroweb_offset",) if k in sm5}

    # 5. CASPCOM printed values
    bl = pd.read_csv(ROOT / "refs" / "caspcom_bulletin_levels.csv")
    rec["caspcom_printed"] = {"sea_mean": {str(int(r.year)): float(r.value_m) for r in bl[bl["quantity"] == "sea_mean"].itertuples()},
                              "post_2024_mean_of_6_m": float(bl[(bl["quantity"] == "post") & (bl["year"] == 2024)]["value_m"].mean())}
    b31 = OUT / "caspcom_bulletin31_2025.csv"
    if b31.exists():
        p31 = pd.read_csv(b31)
        rec["caspcom_printed"]["post_2025_mean_of_6_m"] = float(p31["value_m"].mean())
        rec["caspcom_printed"]["post_2025_posts"] = p31["post"].tolist()
    (OUT / "level_stats.json").write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: rec[k] for k in ("fall", "annual_fall_1995_2025", "reference_2026", "caspcom_printed")},
                     indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
