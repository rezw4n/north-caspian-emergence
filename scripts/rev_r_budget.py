"""Revision R, A18 item 7: shelf gradients, vertical equivalents, the vertical error budget
and the augmented floor.

1. Gradient g per sector: the median |grad z| (m per km) of the waterline DEM (main, 30 m
   tiles) over the cells in the 1995-2026 band (zone-v6 extents: water at the 1995-96
   high stand, not water at the 2025-26 low stand), the sector of each cell taken from
   the hindcast's sector raster. The mean, quartiles and the share of exact flats are
   reported beside it.
2. Vertical equivalents: D*g and Q*g (m) per product, sector and level (primary shape
   variant, epochs pooled; the primary floor).
3. Budget (1 sigma, m) per sector and level, in quadrature:
   - level sigma: the median scene level 1-sigma (random part, sigma_random_m) of the
     level bin's Landsat scenes (stage-2 plan of the main run, epochs pooled);
   - Hydroweb gap fill: the common offset 1-sigma of the gap-filled scenes
     (results/levels/scene_levels_summary.json, filled_scenes_offset_sigma_p50_m, 0.048 m)
     times the bin's share of gap-filled scenes (sigma_offset_m > 0 in the plan);
   - setup: the setup model's training residual sd of the sector (results/levels/
     setup_model.json; Komsomolets Bay uses its proxy, the NE coast, as the setup model
     does);
   - sector departure from DAHITI: |departure_rel_all_m| of results/revision/v/
     sector_departures.csv (the between-sector part; a common part cancels in the shape
     variant); absent where V reports none (Komsomolets Bay) or the file is missing.
4. Augmented floor (sensitivity): Q + 1.645 sigma / g, and the H1 verdicts under it.

Outputs: results/revision/r/budget/{gradients.csv, levels.csv, sigma_by_bin.csv,
verdicts.csv, compare.csv, summary.json}
Usage: python scripts/rev_r_budget.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import caspian  # noqa: F401
import numpy as np
import pandas as pd
import rasterio

from caspian import h1, rev_r
from caspian.config import INTERIM, PROCESSED, RESULTS
from caspian.geography import DOMAIN, SECTOR_ORDER

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("a7_hindcast", ROOT / "scripts" / "a7_hindcast.py")
a7 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(a7)
_spec = importlib.util.spec_from_file_location("rev_r_tables", ROOT / "scripts" / "rev_r_tables.py")
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)

OUT = RESULTS / "revision" / "r" / "budget"
PLAN = INTERIM / "stage2" / "plan_main_tau9900cm_landsat.csv"
DEPARTURES = RESULTS / "revision" / "v" / "sector_departures.csv"
Z = 1.645
NAME = {k + 1: s for k, s in enumerate(SECTOR_ORDER)}


def gradients(transform, shape, sector) -> pd.DataFrame:
    ext = np.load(RESULTS / "study_area" / "extents_v6_250m.npz")
    x0, y1 = float(transform[2]), float(transform[5])
    vals = {k: [] for k in NAME}
    for p in sorted((PROCESSED / "dem" / "main").glob("dem_*.tif")):
        with rasterio.open(p) as r:
            z = r.read(1, masked=True).astype(np.float32).filled(np.nan)
            t = r.transform
        if not np.isfinite(z).any():
            continue
        tr = [t.a, 0, t.c, 0, t.e, t.f]
        hi = a7.nearest(np.asarray(ext["high"]).astype(np.int8), ext["transform"], tr, z.shape, -1)
        lo = a7.nearest(np.asarray(ext["low"]).astype(np.int8), ext["transform"], tr, z.shape, -1)
        s = rev_r.slope_m_per_km(z, abs(t.a))
        band = np.isfinite(s) & (hi == 1) & (lo == 0)
        rr, cc = np.nonzero(band)
        r60 = np.floor((y1 - (t.f + t.e * (rr + 0.5))) / a7.RES).astype(np.int64)
        c60 = np.floor((t.c + t.a * (cc + 0.5) - x0) / a7.RES).astype(np.int64)
        ok = (r60 >= 0) & (r60 < shape[0]) & (c60 >= 0) & (c60 < shape[1])
        lab = np.zeros(rr.size, np.int64)
        lab[ok] = np.asarray(sector)[r60[ok], c60[ok]]
        sv = s[rr, cc]
        for k in NAME:
            vals[k].append(sv[lab == k].astype(np.float32))
    rows = []
    for k, parts in vals.items():
        v = np.concatenate(parts) if parts else np.empty(0, np.float32)
        rows.append({"sector": NAME[k], "n_cells_30m": int(v.size),
                     "g_median_m_per_km": float(np.median(v)) if v.size else np.nan,
                     "g_mean_m_per_km": float(np.mean(v)) if v.size else np.nan,
                     "g_p25_m_per_km": float(np.percentile(v, 25)) if v.size else np.nan,
                     "g_p75_m_per_km": float(np.percentile(v, 75)) if v.size else np.nan,
                     "share_flat": float(np.mean(v == 0)) if v.size else np.nan,
                     "km_per_0p1m_at_median": float(0.1 / np.median(v)) if v.size and np.median(v) > 0 else np.nan})
    return pd.DataFrame(rows)


def sigma_by_bin() -> tuple[pd.DataFrame, float]:
    plan = pd.read_csv(PLAN, usecols=["scene", "epoch", "bin", "sigma_random_m", "sigma_offset_m"]).drop_duplicates("scene")
    summ = json.loads((RESULTS / "levels" / "scene_levels_summary.json").read_text(encoding="utf-8"))
    common = float(summ["hydroweb_offset"]["filled_scenes_offset_sigma_p50_m"])
    g = plan.groupby("bin")
    out = pd.DataFrame({"n_scenes": g.size(), "sigma_level_m": g["sigma_random_m"].median(),
                        "share_gap_filled": g["sigma_offset_m"].apply(lambda x: float((x > 0).mean()))}).reset_index()
    out["sigma_gap_m"] = common * out["share_gap_filled"]
    return out, common


def main(argv) -> int:
    w, s, e, n = DOMAIN.bounds
    transform, shape = a7.grid((w - a7.MARGIN_DEG, s - a7.MARGIN_DEG, e + a7.MARGIN_DEG, n + a7.MARGIN_DEG))
    static = a7.production_static(transform, shape, a7.ht.TILE)
    grad = gradients(transform, shape, static["sector"])
    del static
    sig, common = sigma_by_bin()
    setup = json.loads((RESULTS / "levels" / "setup_model.json").read_text(encoding="utf-8"))["sectors"]
    sd_setup = {}
    for sct, v in setup.items():
        src = v.get("proxy") if not v.get("fitted") else sct
        sd_setup[sct] = float(setup[src]["resid_sd_train_m"])
    dep_note = "absent: results/revision/v/sector_departures.csv not found"
    dep = {}
    if DEPARTURES.exists():
        dv = pd.read_csv(DEPARTURES)
        dv = dv[dv["sector"].isin(SECTOR_ORDER)]
        dep = {r.sector: abs(float(r.departure_rel_all_m)) for r in dv.itertuples() if np.isfinite(r.departure_rel_all_m)}
        dep_note = f"|departure_rel_all_m| from {DEPARTURES.relative_to(ROOT).as_posix()}; absent for " + \
                   (", ".join(sorted(set(SECTOR_ORDER) - set(dep))) or "none")
    D, Q = T.main_levels()
    base, detail = h1.mispredicts(D.rename(columns={"median_m": "median"}),
                                  Q.rename(columns={"p95_m": "p95"})[["sector", "bin", "n", "p95"]])
    lv = detail[["product", "sector", "bin", "level_m", "median", "p95_noise", "evaluable", "exceeds"]].rename(
        columns={"median": "D_m", "p95_noise": "Q_m"})
    lv = lv.merge(grad[["sector", "g_median_m_per_km"]], on="sector", how="left").merge(
        sig[["bin", "sigma_level_m", "sigma_gap_m", "share_gap_filled"]], on="bin", how="left")
    lv["sigma_setup_m"] = lv["sector"].map(sd_setup)
    lv["departure_m"] = lv["sector"].map(dep)
    g = lv["g_median_m_per_km"]
    lv["D_vertical_m"] = lv["D_m"] / 1000.0 * g
    lv["Q_vertical_m"] = lv["Q_m"] / 1000.0 * g
    lv["sigma_budget_m"] = rev_r.budget_sigma(lv["sigma_level_m"], lv["sigma_gap_m"], lv["sigma_setup_m"],
                                              lv["departure_m"])
    lv["Q_aug_m"] = rev_r.augmented_floor(lv["Q_m"], lv["sigma_budget_m"], g, Z)
    lv["budget_horizontal_m"] = Z * lv["sigma_budget_m"] / g * 1000.0
    # verdicts under the augmented floor: the H1 rule with p95 replaced, per (sector, bin)
    qa = lv.drop_duplicates(["sector", "bin"])[["sector", "bin", "Q_aug_m"]]
    nz = Q.merge(qa, on=["sector", "bin"], how="left")
    nz["p95"] = nz["Q_aug_m"].where(np.isfinite(nz["Q_aug_m"]), nz["p95_m"])
    aug, aug_detail = h1.mispredicts(D.rename(columns={"median_m": "median"}), nz[["sector", "bin", "n", "p95"]])
    cmp = T.compare(base, aug)
    sector_tab = (lv.drop_duplicates(["sector", "bin"]).groupby("sector")
                  .agg(sigma_budget_median_m=("sigma_budget_m", "median"), sigma_budget_max_m=("sigma_budget_m", "max"),
                       budget_horizontal_median_m=("budget_horizontal_m", "median"),
                       Q_median_m=("Q_m", "median"), Q_aug_median_m=("Q_aug_m", "median")).reset_index())
    dq = (lv[lv["evaluable"].astype(bool)].groupby(["product", "sector"])
          .agg(D_vertical_median_m=("D_vertical_m", "median"), Q_vertical_median_m=("Q_vertical_m", "median"))
          .reset_index())
    OUT.mkdir(parents=True, exist_ok=True)
    grad.to_csv(OUT / "gradients.csv", index=False)
    sig.to_csv(OUT / "sigma_by_bin.csv", index=False)
    lv.to_csv(OUT / "levels.csv", index=False)
    aug.to_csv(OUT / "verdicts.csv", index=False)
    cmp.to_csv(OUT / "compare.csv", index=False)
    sector_tab.to_csv(OUT / "sector_budget.csv", index=False)
    dq.to_csv(OUT / "vertical_equivalents.csv", index=False)
    summary = {"z": Z, "common_gap_fill_sigma_m": common, "setup_sd_m": sd_setup, "departures": dep_note,
               "departure_m": dep,
               "gradients_m_per_km": grad.set_index("sector")[["g_median_m_per_km", "g_mean_m_per_km", "share_flat"]]
               .round(4).to_dict(orient="index"),
               "counts_primary": T.sector_counts(base), "counts_augmented": T.sector_counts(aug),
               "n_changed": int(cmp["changed"].sum()),
               "changed": cmp.loc[cmp["changed"].astype(bool), ["product", "sector", "mispredicts_primary",
                                                                 "mispredicts_new", "n_exceeding_new",
                                                                 "n_evaluable_new"]].to_dict(orient="records"),
               "sector_budget": sector_tab.round(4).to_dict(orient="records")}
    (OUT / "summary.json").write_text(T._json(summary), encoding="utf-8")
    print(T._json(summary)[:5000])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
