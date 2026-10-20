"""Robustness: the Earth Engine cost arithmetic of docs/ROBUSTNESS_PLAN.md, from
committed inputs and input-level tables only (no stage-2 count, waterline, DEM or
metric is read). Every number in the plan's cost table comes from the JSON this writes.

Inputs
  data/interim/stage2/plan_sensitivity.csv   the scene set of stage 2 (every open-water,
        levelled scene with an acquisition threshold; the main plan at any tau has the
        same scenes, screening only masks sectors) with its setup predictions
  results/inventory/stage2_cost_model.json   zone area of each path/row or MGRS tile
        (the G2 scene weight) and the micro-pilot rates per scene per km2
  results/inventory/stage2_sensB.json        measured interactive cost of the D-064 block
        export (EECU-s and km2 per block, all Landsat scenes)
  results/inventory/stage1_v3_interactive_log.jsonl  measured stage-1 cost per scene
  results/study_area/blocks_25km.csv, zone_v6_cover_buffered.geojson  block areas
  data/interim/levels/scene_levels.csv       November scenes (D-062 variant)
  the lead's main-run estimates and the month's position (parameters below)

Output: results/robustness/cost_plan.json

Usage: python scripts/r17_cost_plan.py
"""

from __future__ import annotations

import json
import sys

import caspian  # noqa: F401
import numpy as np
import pandas as pd

from caspian import design, robustness as R
from caspian.config import INTERIM, RESULTS
from caspian.geography import SECTOR_ORDER

# the lead's estimates and quota position (2026-10-21; docs/gates/G2.md for the S2 range)
MAIN_L_H = 110.0
MAIN_S_H = {"low": 124.0, "central": 186.0, "high": 248.0}
CAP_H = 700.0
USED_BEFORE_MAIN_H = 140.0
# assumptions, stated in the plan
EDGE_FACTOR = 2.0            # edge-guided (Canny) pass over a plain histogram pass, to pilot
S1_60M_FACTOR = 16.0         # D-028: 60 m stage-1 requests cost about 16x the 240 m ones
TAUS = (0.30, 0.20, 0.15, 0.10, 0.05)    # D-064 candidates after no screening


def scene_units(df: pd.DataFrame, scene_col: str = "scene") -> list[str]:
    return [s.split("_")[-1] if se == "MSI_S2" else s.split("_")[1] for s, se in zip(df[scene_col], df["sensor"])]


def main() -> int:
    cm = json.loads((RESULTS / "inventory" / "stage2_cost_model.json").read_text(encoding="utf-8"))
    zone = {u["unit"]: u["zone_km2"] for u in cm["units"]}
    rate60 = cm["rates"]["LANDSAT"]["60m"]
    rate30 = cm["rates"]["LANDSAT"]["30m"]
    plan = pd.read_csv(INTERIM / "stage2" / "plan_sensitivity.csv", low_memory=False)
    base = plan.drop_duplicates("scene").reset_index(drop=True)
    base["km2"] = [zone[u] for u in scene_units(base)]
    base["fam"] = np.where(base["sensor"] == "MSI_S2", "S2", "LANDSAT")
    base["t"] = pd.to_datetime(base["t"], format="ISO8601")
    S = base.groupby("fam")["km2"].sum().to_dict()
    r_L = MAIN_L_H * 3600 / S["LANDSAT"]                         # EECU-s per scene per km2, batch, all overheads
    r_S = {k: v * 3600 / S["S2"] for k, v in MAIN_S_H.items()}
    rate = {"LANDSAT": r_L, "S2": r_S["central"]}
    out = {"inputs": {"main_landsat_h": MAIN_L_H, "main_s2_h": MAIN_S_H, "cap_h": CAP_H,
                      "used_before_main_h": USED_BEFORE_MAIN_H,
                      "scene_km2": {k: round(v) for k, v in S.items()},
                      "n_scenes": base["fam"].value_counts().to_dict(),
                      "effective_rate_eecu_s_per_scene_km2": {"LANDSAT": r_L, "S2": r_S},
                      "micropilot_rate_landsat_60m": rate60,
                      "implied_overhead_landsat": r_L / rate60}}
    # remaining budget
    rem = {k: CAP_H - USED_BEFORE_MAIN_H - MAIN_L_H - v for k, v in MAIN_S_H.items()}
    out["remaining_after_main_h"] = rem

    # measured interactive block cost
    st = json.loads((RESULTS / "inventory" / "stage2_sensB.json").read_text(encoding="utf-8"))
    it = [(v["eecu_s"], v["area_km2"]) for v in st["tiles"].values() if v.get("mode") == "interactive"]
    rho_L = sum(e for e, _ in it) / sum(a for _, a in it)
    rho_S = rho_L * MAIN_S_H["central"] / MAIN_L_H
    out["interactive_block_rate_eecu_s_per_km2"] = {"LANDSAT_measured": rho_L, "S2_scaled": rho_S,
                                                    "n_blocks": len(it), "km2": sum(a for _, a in it)}

    # block areas and the robustness sample R
    import pyproj
    import shapely
    from shapely.geometry import box
    from caspian.config import CRS_EQUAL_AREA
    fwd = pyproj.Transformer.from_crs("EPSG:4326", CRS_EQUAL_AREA, always_xy=True)
    cov = R.load_cover(RESULTS / "study_area" / "zone_v6_cover_buffered.geojson", CRS_EQUAL_AREA)
    blocks = pd.read_csv(RESULTS / "study_area" / "blocks_25km.csv")
    blocks["cover_km2"] = [cov.intersection(box(x, y, x + 25e3, y + 25e3)).area / 1e6
                           for x, y in zip(blocks["x0"], blocks["y0"])]
    Rs = R.block_sample(blocks, blocks["cover_km2"], n_per_sector=2)
    A_R = float(Rs["cover_km2"].sum())
    A_test = float(blocks.loc[blocks["split"] == "test", "cover_km2"].sum())
    out["areas_km2"] = {"cover": cov.area / 1e6, "test_blocks": A_test, "R_sample": A_R,
                        "R_blocks": [f"{i}_{j}" for i, j in zip(Rs["i"], Rs["j"])],
                        "R_by_sector": Rs.groupby("sector")["cover_km2"].sum().round(1).to_dict()}

    # stage 1 per scene
    log = pd.read_json(RESULTS / "inventory" / "stage1_v3_interactive_log.jsonl", lines=True)
    log["fam"] = np.where(log["sensor"] == "MSI_S2", "S2", "LANDSAT")
    s1 = (log.groupby("fam")["eecu_s"].sum() / log.groupby("fam")["n"].sum()).to_dict()
    out["stage1_eecu_s_per_scene"] = s1

    def h(x):
        return round(float(x) / 3600, 2)

    MAIN_H = {"LANDSAT": MAIN_L_H, "S2": MAIN_S_H["central"]}
    rows = {}

    def row(key, check, design_, eecu_h, arithmetic, tier="core"):
        rows[key] = {"check": check, "design": design_, "eecu_h": round(float(eecu_h), 2), "arithmetic": arithmetic,
                     "tier": tier}

    # pilot of the main-export design (split groups and extra bands): a 5 x 5 km box, all scenes, 3 variants
    row("pilot_main_design", "design", "5 x 5 km box, all Landsat scenes, unsplit / split / split + extra bands",
        h(rho_L * 25 * 3), f"{rho_L:.2f} EECU-s/km2 x 25 km2 x 3")
    # block-sample alternatives on R (Landsat only)
    blk = h(rho_L * A_R)
    row("fixed_mndwi0_R", "1", "R blocks, Landsat, MNDWI > 0 (or extra bands of the main export)", blk,
        f"{rho_L:.2f} x {A_R:.0f} km2")
    row("awei_R", "1", "R blocks, Landsat, AWEI_sh > 0 (two more bands read: x1.5, assumed)", blk * 1.5,
        f"{rho_L:.2f} x {A_R:.0f} x 1.5")
    row("snow_all_months_R", "3 (Amendment 9)", "R blocks, Landsat, both snow tests in every month (or extra bands)",
        blk, f"{rho_L:.2f} x {A_R:.0f}")
    row("ndvi_020_040_R", "9", "R blocks, Landsat, NDVI 0.20 and 0.40 (or extra bands)", 2 * blk,
        f"2 x {rho_L:.2f} x {A_R:.0f}", tier="optional")
    oli = base.loc[base["sensor"].isin(["OLI_L8", "OLI_L9"]), "km2"].sum() / S["LANDSAT"]
    a3b = 3 * A_R / len(Rs)
    row("resolution_30m", "1 (G2)", "3 R blocks, OLI scenes, 30 m", h(rho_L * a3b * (rate30 / rate60) * oli),
        f"{rho_L:.2f} x {a3b:.0f} km2 x {rate30 / rate60:.2f} (30/60 m) x {oli:.3f} (OLI share)")
    # SLC-off (only if the main export is not split by sensor)
    slc = (base["sensor"] == "ETM_L7") & (base["t"] >= pd.Timestamp(R.SLC_OFF))
    sh_slc = base.loc[slc, "km2"].sum() / S["LANDSAT"]
    row("slcoff_R", "2", "R blocks, SUB export of ETM+ SLC-off scenes (free with a sensor-split main)",
        h(rho_L * A_R * sh_slc), f"{rho_L:.2f} x {A_R:.0f} x {sh_slc:.3f} (SLC-off share)", tier="if_unsplit")
    row("slcoff_full", "2", "whole cover, SUB export of SLC-off scenes", h(base.loc[slc, "km2"].sum() * r_L),
        f"{base.loc[slc, 'km2'].sum():.0f} scene-km2 x {r_L:.5f}", tier="not_recommended")
    # Amendment 3: edge-guided sample (whole acquisitions)
    acq_s = R.acquisition_sample(base)
    samp = base[base["acq"].isin(acq_s["acq"])]
    n_f = samp["fam"].value_counts().to_dict()
    row("a3_edge_pass_240m", "1 (Amendment 3)", "edge-guided Otsu at 240 m on the sample (x2 for the Canny pass, assumed)",
        h(sum(n_f.get(f, 0) * s1[f] * EDGE_FACTOR for f in s1)),
        f"{n_f} scenes x stage-1 rate per scene x {EDGE_FACTOR:g}")
    row("a3_edge_60m_subsample", "1 (Amendment 3)", "50 Landsat sample scenes at 60 m (scale check of the edge threshold)",
        h(50 * s1["LANDSAT"] * S1_60M_FACTOR * EDGE_FACTOR), f"50 x {s1['LANDSAT']:.2f} x {S1_60M_FACTOR:g} x {EDGE_FACTOR:g}")
    a3_add = sum(samp.loc[samp["fam"] == f, "km2"].sum() * rate[f] for f in ("LANDSAT", "S2"))
    row("a3_add_export", "1 (Amendment 3)", "stage-2 export of the sample at the edge-guided thresholds", h(a3_add),
        "sample scene-km2 x effective rate, per family")
    row("a3_sub_export", "1 (Amendment 3)", "same, at the main thresholds (free if the main plan tags the sample)",
        h(a3_add), "as a3_add_export", tier="if_untagged")
    # D-028: 240 m vs 60 m thresholds, one Landsat acquisition per (epoch, bin)
    lb = base[base["fam"] == "LANDSAT"]
    n_units = lb.groupby(["epoch", "bin"]).ngroups
    spa = lb.groupby("acq").size().mean()
    row("d028_threshold_60m", "1 (D-028)", "60 m histograms of one Landsat acquisition per (epoch, bin)",
        h(n_units * spa * s1["LANDSAT"] * S1_60M_FACTOR),
        f"{n_units} units x {spa:.2f} scenes x {s1['LANDSAT']:.2f} x {S1_60M_FACTOR:g}")
    # number of main-export groups under each class split (robustness.split_groups)
    g0 = base.drop(columns=["group"])
    counts = {}
    for name, by in {"main": (), "sensor": ("sensor",), "sensor+half": ("sensor", "half"),
                     "sensor+half+season": ("sensor", "half", "season"),
                     "sensor+half+season+threshold": ("sensor", "half", "season", "threshold")}.items():
        sp = R.split_groups(g0, by=by)
        counts[name] = {f: int(sp.loc[sp["fam"] == f, "group"].nunique()) for f in ("LANDSAT", "S2")}
    out["groups_by_split"] = counts
    # fallbacks if the main export is not split (scripts/r16_split_plan.py)
    per_scene_L = r_L * S["LANDSAT"] / (base["fam"] == "LANDSAT").sum()
    win = base[(base["t"] >= pd.Timestamp("1995-05-28 01:37:03")) & (base["t"] <= pd.Timestamp("1995-07-16 15:29:42"))]
    row("e1_end_delta", "Amendment 4", "SUB from E1 and ADD to E2 of the E1-end window scenes, whole cover",
        h(2 * len(win) * per_scene_L), f"2 x {len(win)} window scenes x {per_scene_L:.1f} EECU-s per Landsat scene",
        tier="if_unsplit")
    fb = base[(base["fam"] == "LANDSAT") & (base["threshold_source"] != "pooled_otsu")]["km2"].sum() / S["LANDSAT"]
    row("fallback_sub_R", "1 (D-061)", "R blocks, SUB export of the fallback-threshold acquisitions",
        h(rho_L * A_R * fb), f"{rho_L:.2f} x {A_R:.0f} x {fb:.3f} (fallback share)", tier="if_unsplit")
    amj = lb.loc[lb["t"].dt.month <= 6, "km2"].sum() / S["LANDSAT"]
    row("season_sub_R", "3", "R blocks, SUB export of the April-June scenes (JASO-only variant)",
        h(rho_L * A_R * amj), f"{rho_L:.2f} x {A_R:.0f} x {amj:.3f} (AMJ share)", tier="if_unsplit")
    row("binwidth005_R", "7 (PREREG section 4)", "R blocks, Landsat, groups by 0.05 m bins", blk,
        f"{rho_L:.2f} x {A_R:.0f}", tier="if_unsplit")
    # cross-sensor acquisition pairs within 24 h in the plan (before the calm and level filters)
    aa = (base.groupby("acq").agg(sensor=("sensor", "first"), t=("t", "min"), level=("level_m", "mean"))
          .sort_values("t").reset_index())
    tt, ss, lvv = aa["t"].to_numpy(), aa["sensor"].to_numpy(), aa["level"].to_numpy()
    tally: dict = {}
    for i in range(len(aa)):
        j = i + 1
        while j < len(aa) and (tt[j] - tt[i]) <= np.timedelta64(24, "h"):
            if ss[i] != ss[j]:
                c = "|".join(sorted([ss[i], ss[j]]))
                e = tally.setdefault(c, {"within_24h": 0, "within_12h": 0, "dlevel_le_1cm": 0})
                e["within_24h"] += 1
                e["within_12h"] += int((tt[j] - tt[i]) <= np.timedelta64(12, "h"))
                e["dlevel_le_1cm"] += int(abs(lvv[i] - lvv[j]) <= 0.01)
            j += 1
    out["cross_sensor_pairs"] = tally
    # sensor pairs, overlap-clipped
    acq = base.groupby("acq").agg(fam=("fam", "first"), km2=("km2", "sum"))
    akm2 = float(acq.loc[acq["fam"] == "LANDSAT", "km2"].mean())
    side = 1 - 172 * np.cos(np.radians(46.0)) / 185
    ll = 4 * 20 * 2 * akm2 * r_L * side
    ls = 3 * 10 * akm2 * (r_L + r_S["central"])
    row("sensor_pairs", "2", "20 pairs per Landsat-Landsat combination (4), 10 per Landsat-MSI combination (3), "
        "each export clipped to the pair's overlap in the cover", h(ll + ls),
        f"4 x 20 x 2 x {akm2:.0f} km2 x {r_L:.5f} x {side:.2f} (WRS-2 sidelap at 46 N) + 3 x 10 x {akm2:.0f} x "
        f"({r_L:.5f} + {r_S['central']:.5f})")
    # November (D-062 5% bar)
    lv = pd.read_csv(INTERIM / "levels" / "scene_levels.csv", low_memory=False)
    nov = lv[lv["epoch"].isin(["E1", "E2", "E3"]) & (lv["month"] == 11) & (lv["cloud_scene_pct"] < 60)].copy()
    nov["unit"] = np.where(nov["sensor"] == "MSI_S2", "T" + nov["MGRS_TILE"].astype(str),
                           nov["WRS_PATH"].fillna(0).astype(int).astype(str).str.zfill(3)
                           + nov["WRS_ROW"].fillna(0).astype(int).astype(str).str.zfill(3))
    nov["km2"] = nov["unit"].map(zone).fillna(0.0)
    nov = nov[nov["km2"] > 0]
    nov["fam"] = np.where(nov["sensor"] == "MSI_S2", "S2", "LANDSAT")
    nv = {f: {"n": int((nov["fam"] == f).sum()), "km2": float(nov.loc[nov["fam"] == f, "km2"].sum())}
          for f in ("LANDSAT", "S2")}
    out["november"] = nv
    row("november_stage1", "3 (D-062)", "stage 1 for the November scenes (both families)",
        h(sum(nv[f]["n"] * s1[f] for f in nv)),
        f"{nv['LANDSAT']['n']} x {s1['LANDSAT']:.2f} + {nv['S2']['n']} x {s1['S2']:.2f}")
    row("november_add_landsat", "3 (D-062)", "ADD export of the November Landsat scenes, whole cover",
        h(nv["LANDSAT"]["km2"] * r_L), f"{nv['LANDSAT']['km2']:.0f} scene-km2 x {r_L:.5f} (upper bound)")
    row("november_add_s2", "3 (D-062)", "ADD export of the November Sentinel-2 scenes, whole cover",
        h(nv["S2"]["km2"] * r_S["central"]), f"{nv['S2']['km2']:.0f} x {r_S['central']:.5f} (upper bound)",
        tier="optional")
    # screening deltas, both families, both bounds
    w = {}
    for n, (_, p) in __import__("caspian.geography", fromlist=["sectors"]).sectors().items():
        w[n] = cov.intersection(shapely.transform(p.segmentize(0.01),
                                                  lambda xy: np.column_stack(fwd.transform(xy[:, 0], xy[:, 1])))).area
    tot = sum(w.values())
    w = {k: v / tot for k, v in w.items()}
    deltas = {}
    km2 = base["km2"]
    taus = (np.inf, *TAUS)
    pairs_ = [(x, y) for x, y in zip(taus[:-1], taus[1:])] + [(y, x) for x, y in zip(taus[:-1], taus[1:])]
    for fam in ("LANDSAT", "S2"):
        b = base[base["fam"] == fam]
        tot_f = float(b["km2"].sum())
        for a_, b_ in pairs_:
            d = R.screening_delta(b, R.exclusion(b, a_), R.exclusion(b, b_))
            for k in ("add", "sub"):
                if len(d[k]):
                    c = R.delta_cost_share(d[k], km2, tot_f, w)
                    deltas[f"{fam}|main{a_}|alt{b_}|{k}"] = {"upper_h": round(c["upper"] * MAIN_H[fam], 2),
                                                             "per_sector_h": round(c["per_sector"] * MAIN_H[fam], 2),
                                                             "n_scenes": c["n_scenes"],
                                                             "mixed_acq_sector": d["n_mixed_acq_sector"]}
        for t in TAUS:
            d = R.screening_delta(b, R.exclusion(b, t), R.exclusion(b, t, exempt_sectors=["komsomolets"]))
            c = R.delta_cost_share(d["add"], km2, tot_f, w)
            deltas[f"{fam}|komsomolets_unscreened|tau{t}|add"] = {"upper_h": round(c["upper"] * MAIN_H[fam], 2),
                                                                  "per_sector_h": round(c["per_sector"] * MAIN_H[fam], 2),
                                                                  "n_scenes": c["n_scenes"]}
    out["screening_deltas"] = deltas
    out["sector_cover_share"] = w
    by_tau = {}
    for i, t in enumerate(TAUS):
        looser = deltas.get(f"LANDSAT|main{t}|alt{taus[i]}|add", {}).get("per_sector_h", 0.0)
        stricter = (deltas.get(f"LANDSAT|main{t}|alt{TAUS[i + 1]}|sub", {}).get("per_sector_h", 0.0)
                    if i + 1 < len(TAUS) else 0.0)
        kom = deltas[f"LANDSAT|komsomolets_unscreened|tau{t}|add"]["per_sector_h"]
        by_tau[str(t)] = {"looser_add_h": looser, "stricter_sub_h": stricter, "komsomolets_add_h": kom,
                          "setup_form_placeholder_h": max(looser, stricter)}
    out["screening_by_tau_main_landsat"] = by_tau
    # generality (assumptions stated in the plan)
    unit_zone = sum(u["zone_km2"] for u in cm["units"] if u["family"] == "LANDSAT")
    zone_area = 67_011.1          # zone v6 area (D-029)
    years_caspian = 2026 - 1992 + 1
    dpr = S["LANDSAT"] / unit_zone / years_caspian
    dens = {"low": dpr, "high": S["LANDSAT"] / zone_area / years_caspian}
    gen = {}
    for name, band_km2, years, n_pr in (("great_salt_lake", 3400.0, 2026 - 1984 + 1, 4),
                                         ("lake_urmia", 5200.0, 2026 - 1992 + 1, 3)):
        g = {}
        s1h = n_pr * dpr * years * s1["LANDSAT"]
        for k, dpy in dens.items():
            g[k] = round(h(band_km2 * dpy * years * r_L) + h(s1h) + 0.5, 1)
        g["arithmetic"] = (f"{band_km2:.0f} km2 x {dens['low']:.1f}-{dens['high']:.1f} scenes per location-year "
                           f"(the Caspian plan per path/row, and per location with its path overlap) x {years} years "
                           f"x {r_L:.5f} EECU-s; + stage 1 ({n_pr} path/rows, assumed, x {dpr:.1f} x {years} scenes "
                           f"x {s1['LANDSAT']:.2f} EECU-s) + 0.5 h zone (assumed)")
        gen[name] = g
    out["generality"] = gen
    out["landsat_scene_density_per_location_year"] = dens
    out["rows"] = rows
    core = sum(v["eecu_h"] for v in rows.values() if v["tier"] == "core")
    optional = sum(v["eecu_h"] for v in rows.values() if v["tier"] == "optional")
    out["totals"] = {str(t): {"core_h": round(core + sum(by_tau[str(t)].values()), 1),
                              "optional_h": round(optional, 1),
                              "generality_h": [round(sum(g["low"] for g in gen.values()), 1),
                                               round(sum(g["high"] for g in gen.values()), 1)]}
                     for t in TAUS}
    (RESULTS / "robustness").mkdir(parents=True, exist_ok=True)
    (RESULTS / "robustness" / "cost_plan.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k not in ("screening_deltas", "areas_km2")}, indent=1,
                     default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
