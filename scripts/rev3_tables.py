"""Journal revision, PREREG Amendment 20 items 1 (d: exposure split, e: band areas, f), 3, 6
and 8 (D-161): the analyses that only read tables already on disk. Exploratory, post-review.

tiers     Item 6. An overlay on the registered rule, defined after the displacements were
          known; it decides nothing. Per grid, sector and level
          (results/revision/r/control/levels.csv, the rounded degraded-DEM control of
          Amendment 18 item 6 at levels inside the model's contour range): the level is
          below the floor (D <= Q), resolution-scale, or material (D above the floor, the
          control's displacement, one native cell of the grid and the distance that 0.5 m
          of height spans at the sector's median gradient). Per grid and sector: pass,
          resolution-scale or material (caspian.rev3.tier_sector), the median over levels
          of D - Q with a moving-block bootstrap interval (blocks of 3 levels as Amendment
          18 item 2; B = 2,000), and the share of resamples with a material majority.
          Check: the levels above the floor reproduce results/hindcast/main/
          h1_mispredicts.csv.
setup     Item 8. Predicted wind setup of the Landsat acquisitions of the main processing
          plan (data/interim/stage2/plan_main_tau9900cm_landsat.csv, one value per
          acquisition and sector: the mean over its scenes): spread, shares above 0.10 and
          0.20 m, the calibration residual added in quadrature, shoreline equivalents at
          the sector's median gradient against the sector's median floor, the means per
          level bin, and the ERA5 wind speed at the acquisition hours against all hours
          of the same calendar months and years (and against the same hours of the day).
datum     Item 3. Two derived numbers from existing tables: the range of the checked
          sectors' all-block mean biases, and the largest between-sector SWOT departure
          among the checked sectors.
exposure  Item 1d. The first-metre exposure counts of the article (results/revision2/n/
          exposure2_*.csv, registered constants) split into the deltaic sectors and the
          marine domain.
band      Items 1e and 1f. On the 250 m study-area extents: the 1995-96 to 2025-26 band by
          sector, and in the deltaic sectors its part inside the Caspian's February 2000
          outline (GLWD v2 main class 2, "saline lake": the HydroLAKES polygon) and
          landward of it (caspian.rev3.band_display_parts, which the figures also call).
          Beside it, from existing tables, the SWOT departures of the Volga sector by month
          and the static-contour shares.

Outputs: results/revision3/{tiers,setup,datum,exposure,band}/...
Usage: python scripts/rev3_tables.py <tiers|setup|datum|exposure|band|all>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import caspian  # noqa: F401
import numpy as np
import pandas as pd

from caspian import rev3
from caspian.config import CRS_EQUAL_AREA, INTERIM, RAW, RESULTS
from caspian.geography import SECTOR_ORDER, sectors

OUT = RESULTS / "revision3"
SEED = 20261125
N_BOOT = 2000
MBB_BLOCK = 3                                   # levels per block, as Amendment 18 item 2
CHECKED = ("west", "interfluve", "northeast", "open_shelf")
FAILING = ("gebco_2026", "gebco_2022", "srtm15p_v2.7", "etopo1")           # the four grids that fail
ARC_SECONDS = {"gebco_2026": 15.0, "gebco_2022": 15.0, "srtm15p_v2.7": 15.0, "etopo_2022": 15.0, "etopo1": 60.0}
HEIGHT_M = 0.5                                  # the checked-sector tolerance (params.checked_tol_m)
GLWD_SALINE_LAKE = 2


def deltaic() -> list[str]:
    lab = pd.read_csv(RESULTS / "levels" / "sector_labels.csv").set_index("sector")
    return [s for s in SECTOR_ORDER if bool(lab.loc[s, "deltaic"])]


def gradients() -> dict:
    g = pd.read_csv(RESULTS / "revision" / "r" / "budget" / "gradients.csv")
    return {r.sector: float(r.g_median_m_per_km) for r in g.itertuples()}


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)),
                    encoding="utf-8")


# --- item 6 -----------------------------------------------------------------------------------

def cmd_tiers() -> int:
    lev = pd.read_csv(RESULTS / "revision" / "r" / "control" / "levels.csv")
    lev = lev[lev["control_kind"] == "rounded"].sort_values(["product", "sector", "bin"]).reset_index(drop=True)
    g = gradients()
    prim = pd.read_csv(RESULTS / "hindcast" / "main" / "h1_mispredicts.csv")
    prim = prim[prim["variant"] == "shape"].set_index(["product", "sector"])
    out_l, out_s, thr = [], [], []
    for (p, s), d in lev.groupby(["product", "sector"], sort=True):
        cell = rev3.native_cell_m(ARC_SECONDS[p])
        dist = rev3.half_metre_distance_m(g[s], HEIGHT_M)
        ctrl = np.where(d["evaluable_control"].to_numpy(bool) & d["in_dem_range"].to_numpy(bool),
                        d["D_control"].to_numpy(float), np.nan)
        ev = d["evaluable_product"].to_numpy(bool)
        D, Q = d["D"].to_numpy(float), d["Q"].to_numpy(float)
        tiers = rev3.tier_levels(D, Q, ctrl, ev, cell, dist)
        res = rev3.tier_sector(tiers)
        w = prim.loc[(p, s)]
        if (res["n_evaluable"], res["n_above_floor"]) != (int(w["n_evaluable"]), int(w["n_exceeding"])):
            raise SystemExit(f"tiers: {p} {s} does not reproduce the registered counts")
        if (res["class"] != "pass") != bool(w["mispredicts"]):
            raise SystemExit(f"tiers: {p} {s} does not reproduce the registered verdict")
        e = ev
        boot = rev3.mbb_median_interval((D - Q)[e], (tiers == "material")[e], N_BOOT, SEED, MBB_BLOCK)
        out_s.append({"product": p, "sector": s, "checked": s in CHECKED, **res,
                      "n_levels_without_control": int((e & ~np.isfinite(ctrl)).sum()),
                      "d_minus_q_median_m": boot["median"], "d_minus_q_lo_m": boot["lo"], "d_minus_q_hi_m": boot["hi"],
                      "p_material_majority": boot["p_material_majority"],
                      "d_median_m": float(np.median(D[e])) if e.any() else np.nan,
                      "threshold_cell_m": cell, "threshold_half_metre_m": dist})
        out_l.append(d[["product", "sector", "bin", "level_m", "D", "Q"]].assign(D_control=ctrl, tier=tiers))
        thr.append({"product": p, "sector": s, "cell_m": cell, "half_metre_distance_m": dist,
                    "gradient_m_per_km": g[s]})
    S = pd.DataFrame(out_s)
    d = OUT / "tiers"
    d.mkdir(parents=True, exist_ok=True)
    pd.concat(out_l, ignore_index=True).to_csv(d / "tier_levels.csv", index=False)
    S.to_csv(d / "tiers.csv", index=False)
    chk = S[S["checked"]]
    four = chk[chk["product"].isin(FAILING)]
    summ = {"amendment": "PREREG Amendment 20 item 6", "status": "post hoc overlay; decides nothing",
            "control": "rounded degraded-DEM control, levels in the model's contour range (D-116)",
            "height_m": HEIGHT_M, "native_cell_m": {p: rev3.native_cell_m(a) for p, a in ARC_SECONDS.items()},
            "half_metre_distance_m": {s: rev3.half_metre_distance_m(g[s], HEIGHT_M) for s in SECTOR_ORDER},
            "n_boot": N_BOOT, "seed": SEED, "mbb_block_levels": MBB_BLOCK,
            "registered_counts_reproduced": True,
            "checked_cases": int(len(chk)), "checked_by_class": chk["class"].value_counts().to_dict(),
            "four_failing_grids_checked_cases": int(len(four)),
            "four_failing_grids_by_class": four["class"].value_counts().to_dict(),
            "four_failing_grids_resolution": sorted(f"{r.product}|{r.sector}" for r in
                                                    four[four["class"] == "resolution"].itertuples())}
    _write(d / "summary.json", summ)
    print(json.dumps(summ, indent=1))
    return 0


# --- item 8 -----------------------------------------------------------------------------------

def cmd_setup() -> int:
    plan = pd.read_csv(INTERIM / "stage2" / "plan_main_tau9900cm_landsat.csv")
    cols = {c[6:-2]: c for c in plan.columns if c.startswith("setup_")}
    if set(cols) != set(SECTOR_ORDER):
        raise SystemExit("the plan's setup columns are not the seven sectors")
    acq = plan.groupby("acq", sort=True).agg(t=("t", "min"), epoch=("epoch", "first"), bin=("bin", "first"),
                                             **{s: (c, "mean") for s, c in cols.items()}).reset_index()
    if plan.groupby("acq")[["epoch", "bin"]].nunique().max().max() != 1:
        raise SystemExit("an acquisition lies in more than one unit")
    acq["t"] = pd.to_datetime(acq["t"])
    seas = pd.read_csv(RESULTS / "revision" / "r" / "diagnostics" / "season_shares.csv")
    u = acq.groupby(["epoch", "bin"]).size().rename("n").reset_index().merge(seas, on=["epoch", "bin"], how="outer")
    if not (u["n"] == u["n_acq"]).all():
        raise SystemExit("the acquisitions per unit differ from results/revision/r/diagnostics/season_shares.csv")
    sm = json.loads((RESULTS / "levels" / "setup_model.json").read_text(encoding="utf-8"))["sectors"]
    g = gradients()
    bud = pd.read_csv(RESULTS / "revision" / "r" / "budget" / "sector_budget.csv").set_index("sector")
    rows, bins = [], []
    for s in SECTOR_ORDER:
        x = acq[s].to_numpy(float)
        a = rev3.abs_summary(x)
        fitted = bool(sm[s].get("fitted"))
        resid = sm[s].get("resid_sd_calib_m") if fitted else sm["northeast"].get("resid_sd_calib_m")
        total = float(np.hypot(a["sd"], resid))
        q = float(bud.loc[s, "Q_median_m"])
        sig = float(bud.loc[s, "sigma_budget_median_m"])       # the systematic level-error budget (D-112)
        um = acq.groupby(["epoch", "bin"])[s].mean()
        rows.append({"sector": s, "n_acquisitions": a["n"], "sd_m": a["sd"], "abs_median_m": a["abs_median"],
                     "abs_p95_m": a["abs_p95"], "share_above_0p10": a["share_above_0.1"],
                     "share_above_0p20": a["share_above_0.2"], "model_fitted": fitted,
                     "r2_calib": sm[s].get("r2_calib") if fitted else np.nan, "resid_sd_calib_m": resid,
                     "total_sd_m": total, "gradient_m_per_km": g[s], "floor_median_m": q,
                     "sd_horizontal_m": a["sd"] / g[s] * 1000, "abs_p95_horizontal_m": a["abs_p95"] / g[s] * 1000,
                     "total_sd_horizontal_m": total / g[s] * 1000,
                     "abs_p95_over_floor": a["abs_p95"] / g[s] * 1000 / q,
                     "total_sd_over_floor": total / g[s] * 1000 / q,
                     "n_units": int(um.size), "unit_mean_sd_m": float(um.std(ddof=1)),
                     "unit_mean_abs_max_m": float(um.abs().max()), "unit_mean_abs_median_m": float(um.abs().median()),
                     "unit_mean_abs_max_horizontal_m": float(um.abs().max()) / g[s] * 1000,
                     "sigma_budget_m": sig, "unit_mean_abs_max_over_sigma": float(um.abs().max()) / sig,
                     "unit_mean_abs_median_over_sigma": float(um.abs().median()) / sig})
        for (ep, b), v in um.items():
            bins.append({"sector": s, "epoch": ep, "bin": int(b), "mean_setup_m": float(v)})
    # ERA5 wind speed at the acquisition hours against the reference hours
    era = pd.concat([pd.read_csv(f) for f in sorted((INTERIM / "era5").glob("era5_sector_winds_*.csv"))],
                    ignore_index=True)
    era["speed"] = np.hypot(era["u_component_of_wind_10m"].to_numpy(float), era["v_component_of_wind_10m"].to_numpy(float))
    era["t"] = pd.to_datetime(era["t"], unit="ms")
    era = era.drop_duplicates(["sector", "t"])
    hour = acq["t"].dt.round("h")
    ym = set(zip(acq["t"].dt.year, acq["t"].dt.month))
    hod = sorted(set(hour.dt.hour))
    wrows = []
    for s in SECTOR_ORDER:
        e = era[era["sector"] == s].set_index("t")["speed"]
        at = e.reindex(hour.to_numpy())
        n_missing = int(at.isna().sum())
        at = at.dropna().to_numpy(float)
        key = pd.MultiIndex.from_arrays([e.index.year, e.index.month])
        ref = e[key.isin(list(ym))]
        ref_h = ref[ref.index.hour.isin(hod)]
        wrows.append({"sector": s, "n_acquisitions_with_wind": int(at.size), "n_acquisitions_without_wind": n_missing,
                      "acq_median_m_s": float(np.median(at)), "acq_p90_m_s": float(np.percentile(at, 90)),
                      "ref_n_hours": int(ref.size), "ref_median_m_s": float(ref.median()),
                      "ref_p90_m_s": float(np.percentile(ref.to_numpy(float), 90)),
                      "ref_same_hours_n": int(ref_h.size), "ref_same_hours_median_m_s": float(ref_h.median()),
                      "ref_same_hours_p90_m_s": float(np.percentile(ref_h.to_numpy(float), 90)),
                      "median_ratio": float(np.median(at) / ref.median()),
                      "median_ratio_same_hours": float(np.median(at) / ref_h.median()),
                      "p90_ratio_same_hours": float(np.percentile(at, 90) / np.percentile(ref_h.to_numpy(float), 90))})
    d = OUT / "setup"
    d.mkdir(parents=True, exist_ok=True)
    R, W = pd.DataFrame(rows), pd.DataFrame(wrows)
    R.to_csv(d / "setup_by_sector.csv", index=False)
    pd.DataFrame(bins).to_csv(d / "setup_unit_means.csv", index=False)
    W.to_csv(d / "wind_sampling.csv", index=False)
    chk = R[R["sector"].isin(CHECKED)]
    summ = {"amendment": "PREREG Amendment 20 item 8", "n_acquisitions": int(len(acq)), "n_scenes": int(len(plan)),
            "n_units": int(len(u)), "acquisition_hours_utc": hod,
            "first_acquisition": str(acq["t"].min()), "last_acquisition": str(acq["t"].max()),
            "komsomolets_residual": "the northeastern coast's (its model is the northeastern one with its own forcing)",
            "checked": {"abs_p95_m": [float(chk["abs_p95_m"].min()), float(chk["abs_p95_m"].max())],
                        "total_sd_m": [float(chk["total_sd_m"].min()), float(chk["total_sd_m"].max())],
                        "total_sd_over_floor": [float(chk["total_sd_over_floor"].min()),
                                                float(chk["total_sd_over_floor"].max())],
                        "unit_mean_abs_max_m": [float(chk["unit_mean_abs_max_m"].min()),
                                                float(chk["unit_mean_abs_max_m"].max())]},
            "wind_median_ratio_same_hours": [float(W["median_ratio_same_hours"].min()),
                                             float(W["median_ratio_same_hours"].max())]}
    _write(d / "summary.json", summ)
    print(R.to_string(index=False))
    print(W.to_string(index=False))
    print(json.dumps(summ, indent=1))
    return 0


# --- item 3 -----------------------------------------------------------------------------------

def cmd_datum() -> int:
    ob = pd.read_csv(RESULTS / "revision2" / "d" / "obsbias" / "obs_bias_by_sector.csv")
    dep = pd.read_csv(RESULTS / "revision" / "v" / "sector_departures.csv").set_index("sector")
    out = {"amendment": "PREREG Amendment 20 item 3", "checked_sectors": list(CHECKED), "sources": {}}
    for src, gsrc in ob.groupby("source"):
        c = gsrc[gsrc["sector"].isin(CHECKED)].set_index("sector")
        out["sources"][src] = {"mean_bias_m": {s: float(c.loc[s, "mean"]) for s in CHECKED},
                               "mean_bias_min_m": float(c["mean"].min()), "mean_bias_max_m": float(c["mean"].max()),
                               "mean_bias_range_m": float(c["mean"].max() - c["mean"].min()),
                               "median_bias_range_m": float(c["median"].max() - c["median"].min()),
                               "largest_abs_mean_bias_m": float(c["mean"].abs().max())}
    rel = dep.loc[[s for s in CHECKED if s in dep.index], "departure_rel_all_m"].astype(float)
    out["swot"] = {"between_sector_departure_m": {s: float(v) for s, v in rel.items()},
                   "largest_abs_checked_m": float(rel.abs().max()), "sector_of_largest": str(rel.abs().idxmax()),
                   "all_sectors_rel_m": {s: (float(dep.loc[s, "departure_rel_all_m"])
                                             if s in dep.index and np.isfinite(dep.loc[s, "departure_rel_all_m"]) else None)
                                         for s in SECTOR_ORDER},
                   "common_departure_m": float(dep.loc["all", "departure_m"])}
    _write(OUT / "datum" / "datum.json", out)
    print(json.dumps(out, indent=1))
    return 0


# --- item 1d: exposure split ---------------------------------------------------------------------

def cmd_exposure() -> int:
    delt = deltaic()
    marine = [s for s in SECTOR_ORDER if s not in delt]
    fe = pd.read_csv(RESULTS / "revision2" / "n" / "exposure2_features.csv")
    pol = pd.read_csv(RESULTS / "revision2" / "n" / "exposure2_polygons.csv")
    rows = []
    key = ["layer", "rule", "product", "surface", "delta_m", "subset"]
    for k, gk in fe.groupby(key, sort=True):
        by = gk.set_index("region")["amount"]
        dom = float(by["domain"])
        parts = {"deltaic": float(by.reindex(delt).sum()), "marine": float(by.reindex(marine).sum())}
        if abs(parts["deltaic"] + parts["marine"] - dom) > 1e-6:
            raise SystemExit(f"exposure: the sectors do not add up to the domain for {k}")
        rows.append({**dict(zip(key, k)), "kind": "feature", "domain": dom, **parts})
    keyp = ["layer", "product", "surface", "delta_m"]
    for k, gk in pol.groupby(keyp, sort=True):
        a, e = gk.set_index("region")["emerged_in_layer_km2"], gk.set_index("region")["E_km2"]
        parts = {"deltaic": float(a.reindex(delt).sum()), "marine": float(a.reindex(marine).sum())}
        if abs(parts["deltaic"] + parts["marine"] - float(a["domain"])) > 1e-6:
            raise SystemExit(f"exposure: the sectors do not add up to the domain for {k}")
        em = float(e.reindex(marine).sum())
        rows.append({**dict(zip(keyp, k)), "rule": "area", "subset": "all", "kind": "polygon",
                     "domain": float(a["domain"]), **parts, "E_marine_km2": em, "E_domain_km2": float(e["domain"]),
                     "share_marine": parts["marine"] / em if em > 0 else np.nan})
    T = pd.DataFrame(rows)
    d = OUT / "exposure"
    d.mkdir(parents=True, exist_ok=True)
    T.to_csv(d / "exposure_split.csv", index=False)
    one = T[(T["surface"] == "observed_reref") & np.isclose(T["delta_m"], 1.0)]
    rng = (one.groupby(["layer", "subset", "kind"])[["domain", "deltaic", "marine"]].agg(["min", "max"]))
    rng.columns = [f"{a}_{b}" for a, b in rng.columns]
    rng = rng.reset_index()
    rng.to_csv(d / "exposure_split_first_metre.csv", index=False)
    print(rng.to_string(index=False))
    _write(d / "summary.json", {"amendment": "PREREG Amendment 20 item 1d (split of existing tables)",
                                "deltaic_sectors": delt, "marine_sectors": marine,
                                "source": "results/revision2/n/exposure2_{features,polygons}.csv (registered constants)"})
    return 0


# --- items 1e and 1f: band areas and the display mask ----------------------------------------------

def cmd_band() -> int:
    import pyproj

    ext_path = RESULTS / "study_area" / "extents_v6_250m.npz"
    if not pyproj.CRS(str(np.load(ext_path)["crs"])).equals(pyproj.CRS(CRS_EQUAL_AREA), ignore_axis_order=True):
        raise SystemExit("extents_v6_250m.npz: CRS differs from config.CRS_EQUAL_AREA")
    src = RAW / "habitat" / "glwd_v2" / "GLWD_v2_0_main_class.tif"
    outline = RESULTS / "study_area" / "caspian_feb2000_250m.npz"      # scripts/rev3_outline.py mask (D-163)
    if not outline.exists():
        raise SystemExit(f"{outline} is missing: run `python scripts/rev3_outline.py mask`")
    delt = deltaic()
    parts = rev3.band_display_parts(ext_path, src, {k: p for k, (_, p) in sectors().items()}, delt,
                                    saline_lake=GLWD_SALINE_LAKE, outline_npz=outline)
    tr, sec, lake2000 = [float(v) for v in parts["transform"]], parts["sector"], parts["lake2000"]
    raster = parts["lake2000_raster"]                      # the form the amendment named
    band = parts["band"] & (sec > 0)                       # the band inside the study domain
    cell = abs(tr[0] * tr[4]) / 1e6
    rows = []
    for k, sname in enumerate(SECTOR_ORDER, start=1):
        b = band & (sec == k)
        rows.append({"sector": sname, "deltaic": sname in delt, "band_km2": float(b.sum() * cell),
                     "band_inside_2000_outline_km2": float((b & lake2000).sum() * cell),
                     "band_landward_of_2000_outline_km2": float((b & ~lake2000).sum() * cell),
                     "share_landward": float((b & ~lake2000).sum() / b.sum()) if b.any() else np.nan,
                     "band_landward_glwd_raster_km2": float((b & ~raster).sum() * cell),
                     "share_landward_glwd_raster": float((b & ~raster).sum() / b.sum()) if b.any() else np.nan})
    T = pd.DataFrame(rows)
    d = OUT / "band"
    d.mkdir(parents=True, exist_ok=True)
    T.to_csv(d / "band_by_sector.csv", index=False)
    not_seabed = parts["not_seabed"]                       # not drawn as emerged seabed
    not_seabed_raster = parts["not_seabed_raster"]
    # what the raster adds to the lake in the band of the deltaic sectors, by GLWD class at the cell
    in_delta = band & np.isin(sec, [SECTOR_ORDER.index(s) + 1 for s in delt])
    dep = pd.read_csv(RESULTS / "revision" / "v" / "sector_departures.csv").set_index("sector")
    mon = pd.read_csv(RESULTS / "revision" / "v" / "sector_departures_by_month.csv")
    piv = mon.pivot(index="month", columns="sector", values="departure_m")
    volga_minus_all = (piv["volga"] - piv["all"]).round(4)
    st = pd.read_csv(RESULTS / "revision" / "r" / "static" / "audit_cells_static_share.csv")
    summ = {"amendment": "PREREG Amendment 20 items 1e and 1f", "deltaic_sectors": delt,
            "outline": "the HydroLAKES polygon of the Caspian Sea (lake 1; SRTM Water Body Data, February 2000), "
                       "cell centre inside the polygon (results/study_area/caspian_feb2000_250m.npz; D-163)",
            "outline_as_registered": "GLWD v2 main class 2 (saline lake) at the cell centre: the same polygon as "
                                     "rasterised at 15 arc-seconds, with every other saline lake of the class",
            "band_km2_not_drawn_as_seabed_glwd_raster": float(not_seabed_raster.sum() * cell),
            "share_of_domain_band_not_drawn_as_seabed_glwd_raster": float(not_seabed_raster.sum() / band.sum()),
            "deltaic_band_cells_lake_in_raster_not_in_polygon_km2": float((in_delta & raster & ~lake2000).sum() * cell),
            "deltaic_band_cells_lake_in_polygon_not_in_raster_km2": float((in_delta & lake2000 & ~raster).sum() * cell),
            "glwd_file": src.name, "extents": "results/study_area/extents_v6_250m.npz (250 m)",
            "band_definition": "water in the 1995-96 extent (high == 1) and not water in the 2025-26 extent "
                               "(low == 0), inside the study domain: the band the figures draw",
            "band_km2_domain": float(band.sum() * cell),
            "band_km2_deltaic": float(T.loc[T["deltaic"], "band_km2"].sum()),
            "band_km2_deltaic_landward": float(T.loc[T["deltaic"], "band_landward_of_2000_outline_km2"].sum()),
            "band_km2_not_drawn_as_seabed": float(not_seabed.sum() * cell),
            "share_of_domain_band_not_drawn_as_seabed": float(not_seabed.sum() / band.sum()),
            "swot_departure_rel_all_m": {sname: (float(dep.loc[sname, "departure_rel_all_m"])
                                                 if np.isfinite(dep.loc[sname, "departure_rel_all_m"]) else None)
                                         for sname in SECTOR_ORDER},
            "swot_n_passes": {sname: int(dep.loc[sname, "n_passes"]) for sname in SECTOR_ORDER},
            "volga_minus_all_by_month_m": {int(k): float(v) for k, v in volga_minus_all.items()},
            "volga_passes_by_month": {int(r.month): int(r.n_passes) for r in mon[mon["sector"] == "volga"].itertuples()},
            "static_share_E3": {r.sector: float(r.share_static) for r in st[st["epoch"] == "E3"].itertuples()}}
    _write(d / "summary.json", summ)
    print(T.to_string(index=False))
    print(json.dumps(summ, indent=1))
    return 0


def main(argv) -> int:
    cmds = {"tiers": cmd_tiers, "setup": cmd_setup, "datum": cmd_datum, "exposure": cmd_exposure, "band": cmd_band}
    which = argv[0] if argv else "all"
    if which == "all":
        for f in cmds.values():
            f()
        return 0
    if which not in cmds:
        raise SystemExit(__doc__)
    return cmds[which]()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
