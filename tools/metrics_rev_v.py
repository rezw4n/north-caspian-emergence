"""Workstream V metrics: water surface and datums (PREREG Amendment 18 item 9; R2.M3,
R2.M2, R1.M1(c), R1 minor 1 and 4-7, R2 minor 3 and 5, R3 minor 8; D-120 to D-122).

Loaded by tools/build_metrics.revision_metrics, which calls add(m). Every key starts
with "rev.v."; each record is {"value", "unit", "source", "note"?} as put() in
tools/build_metrics.py, with "decimals" where the default would mislead. Direct reads of
results/revision/v/ (scripts/rev_v_*.py); a file that does not exist yet adds no keys.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
V = ROOT / "results" / "revision" / "v"
REL = "results/revision/v"
PK = {"gebco_2026": "gebco2026", "gebco_2022": "gebco2022", "etopo_2022": "etopo2022",
      "srtm15p_v2.7": "srtm15p", "etopo1": "etopo1"}
VALIDATED = ["west", "interfluve", "northeast", "open_shelf"]


def _put(m, key, value, unit, source, note=None, decimals=None):
    if value is None:
        return
    if isinstance(value, float) and math.isnan(value):
        return
    if hasattr(value, "item"):
        value = value.item()
    rec = {"value": value, "unit": unit, "source": source}
    if note:
        rec["note"] = note
    if decimals is not None:
        rec["decimals"] = decimals
    m[f"rev.v.{key}"] = rec


def _json(name):
    p = V / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def departures(m):
    p = V / "sector_departures.csv"
    if not p.exists():
        return
    t = pd.read_csv(p).set_index("sector")
    src = f"{REL}/sector_departures.csv"
    for s, r in t.iterrows():
        _put(m, f"dep.{s}.m", r["departure_m"], "m", src, "median over Apr-Oct SWOT passes of block-mean water "
             "surface minus basin level, on the working datum (NGA realisation of SWOT's EGM2008; D-120)", 2)
        _put(m, f"dep.{s}.n_passes", int(r["n_passes"]), "passes", src)
        _put(m, f"dep.{s}.n_blocks", int(r["n_blocks"]), "blocks", src)
        _put(m, f"dep.{s}.iqr_m", r["iqr_m"], "m", src, decimals=2)
        _put(m, f"dep.{s}.rel_m", r.get("departure_rel_all_m"), "m", src,
             "departure minus the all-block departure (between-sector part)", 2)
    if "all" in t.index:
        _put(m, "dep.all.icgem_m", t.loc["all", "departure_icgem_realisation_m"], "m", src,
             "as dep.all.m with SWOT's EGM2008 taken without NGA's -0.41 m", 2)
        _put(m, "dep.all.swot_geoid_m", t.loc["all", "departure_swot_geoid_m"], "m", src, "no datum conversion", 2)
    rel = t.drop(index=[i for i in ("all",) if i in t.index])["departure_rel_all_m"].dropna()
    if len(rel):
        _put(m, "dep.rel_min_m", float(rel.min()), "m", src, decimals=2)
        _put(m, "dep.rel_max_m", float(rel.max()), "m", src, decimals=2)
        _put(m, "dep.n_sectors_measured", int(rel.size), "sectors", src)
    rec = _json("sector_departures.json")
    if rec:
        _put(m, "dep.n_passes_warm", rec["n_passes_warm"], "passes", f"{REL}/sector_departures.json")
        _put(m, "dep.first_pass", str(rec["pass_period"][0])[:10], "date", f"{REL}/sector_departures.json")
        _put(m, "dep.last_pass", str(rec["pass_period"][1])[:10], "date", f"{REL}/sector_departures.json")


def _verdicts(m, prefix, comp_csv, run_dir, variants):
    p = V / comp_csv
    if not p.exists():
        return
    t = pd.read_csv(p)
    src = f"{REL}/{comp_csv}"
    for (var, prod), g in t.groupby(["variant", "product"]):
        if var not in variants:
            continue
        pk = PK.get(prod, prod)
        md = g[g["mispredicts_departure"].astype(str) == "True"]["sector"]
        _put(m, f"{prefix}.{var}.{pk}.n_mispredicted", int(md.size), "sectors", src)
        _put(m, f"{prefix}.{var}.{pk}.n_mispredicted_validated", int(md.isin(VALIDATED).sum()), "sectors", src)
        _put(m, f"{prefix}.{var}.{pk}.n_changed", int(g["verdict_changed"].sum()), "sectors", src)
    so = run_dir / "shape_offsets.json"
    if so.exists():
        r = json.loads(so.read_text(encoding="utf-8"))
        for prod, v in r["products"].items():
            _put(m, f"{prefix}.c.{PK.get(prod, prod)}_m", v["offset_m"], "m", f"{REL}/{run_dir.name}/shape_offsets.json",
                 decimals=2)
            _put(m, f"{prefix}.objective.{PK.get(prod, prod)}_km2", round(v["objective_km2_at_offset"], 1), "km2",
                 f"{REL}/{run_dir.name}/shape_offsets.json", decimals=0)


def planar(m):
    rec = _json("planar_comparison.json")
    if not rec:
        return
    src = f"{REL}/planar_comparison.json"
    for prod, f in rec.get("fits", {}).items():
        pk = PK.get(prod, prod)
        ex = f["exact"]
        _put(m, f"planar.{pk}.a_m_per_100km", ex["a_m_per_km"] * 100, "m per 100 km", src, "east-west gradient", 2)
        _put(m, f"planar.{pk}.b_m_per_100km", ex["b_m_per_km"] * 100, "m per 100 km", src, "north-south gradient", 2)
        _put(m, f"planar.{pk}.c0_m", ex["c0_m"], "m", src, "value at 45.5 N, 50 E", 2)
        _put(m, f"planar.{pk}.objective_km2", round(ex["objective_km2"], 1), "km2", src, decimals=0)
        _put(m, f"planar.{pk}.objective_reduction_pct", f["objective_reduction_vs_primary_pct"], "%", src, decimals=1)
    for prod, runs in rec.get("verdicts", {}).items():
        pk = PK.get(prod, prod)
        for run, v in runs.items():
            _put(m, f"planar.verdicts.{run}.{pk}.n_mispredicted", v["n_mispredicted"], "sectors", src)
            _put(m, f"planar.verdicts.{run}.{pk}.n_mispredicted_validated", v["n_validated"], "sectors", src)
            _put(m, f"planar.verdicts.{run}.{pk}.median_sector_median_m", v["median_of_sector_medians_m"], "m", src,
                 decimals=0)
    for run, d in rec.get("objective_eq_s11_km2_extent_domain_train", {}).items():
        for prod, v in d.items():
            _put(m, f"planar.objective_s11.{run}.{PK.get(prod, prod)}_km2", v, "km2", src, decimals=0)


def posts(m):
    rec = _json("caspcom_posts.json")
    if not rec:
        return
    src = f"{REL}/caspcom_posts.json"
    _put(m, "posts.century_in_study_domain", rec["century_posts_in_study_domain"], "posts", src)
    _put(m, "posts.north_set_in_study_domain", rec["north_set_in_study_domain"], "posts", src)
    for r in rec["table"]:
        key = r["post"].split(" ")[0].lower()
        if not (r["century_post"] or r["north_caspian_set"]):
            continue
        _put(m, f"posts.{key}.offset_m", r["offset_m"], "m", src, f"{r['name']}: mean of annual post - DAHITI", 3)
        _put(m, f"posts.{key}.minus_delta_m", r["offset_minus_delta_m"], "m", src, decimals=2)
        _put(m, f"posts.{key}.warm_minus_delta_m", r["warm_minus_delta_m"], "m", src, "April-October months", 2)
        _put(m, f"posts.{key}.in_study_domain", bool(r["in_study_domain"]), "flag", src)


def datum_terms(m):
    rec = _json("datum_terms.json")
    if not rec:
        return
    src = f"{REL}/datum_terms.json"
    t = rec["terms"]
    _put(m, "cref.tp_term_min_m", t["tp_ellipsoid_if_unreduced_m"][0], "m", src, "h(T/P) - h(WGS84) over the domain", 3)
    _put(m, "cref.tp_term_max_m", t["tp_ellipsoid_if_unreduced_m"][1], "m", src, decimals=3)
    _put(m, "cref.tide_term_min_m", t["tide_if_dahiti_mean_tide_m"][0], "m", src, decimals=3)
    _put(m, "cref.tide_term_max_m", t["tide_if_dahiti_mean_tide_m"][1], "m", src, decimals=3)
    _put(m, "cref.gm_term_m", t["gm_zero_degree_if_dgfi_omits_m"], "m", src, decimals=4)
    _put(m, "cref.w0_term_iers2017_m", t["w0_term_iers2017_m"], "m", src, decimals=2)
    _put(m, "cref.w0_term_pre2015_m", t["w0_term_pre2015_m"], "m", src, decimals=2)
    for k, v in rec["residuals_m"].items():
        if k.startswith("SWOT"):
            continue
        kk = k.lower()
        _put(m, f"cref.{kk}.c_m", rec["constants"][k]["c_m"], "m", src, decimals=2)
        _put(m, f"cref.{kk}.resid_documented_min_m", v["documented_terms_only"]["residual_min_m"], "m", src,
             "c less the most the documented terms (GM, tide, frame) can explain", 2)
        _put(m, f"cref.{kk}.resid_tp_m", v["tp_ellipsoid"]["tide_free"], "m", src, "c less the T/P ellipsoid term", 2)
        _put(m, f"cref.{kk}.resid_w0_nga_m", v["w0_nga"], "m", src, decimals=2)
    _put(m, "cref.swot_nga_m", rec["swot_check"]["nga_realisation_m"], "m", src,
         "DAHITI minus the SWOT water surface on the working chain (no DEM)", 2)
    _put(m, "cref.swot_icgem_m", rec["swot_check"]["icgem_realisation_m"], "m", src, decimals=2)


def bhs(m):
    rec = _json("bhs_constants.json")
    if not rec:
        return
    src = f"{REL}/bhs_constants.json"
    for prod, r in rec["products"].items():
        pk = PK.get(prod, prod)
        _put(m, f"bhs.{pk}.k_m", r["k_bhs_m"], "m", src, "added to the heights as distributed gives BHS-77", 2)
        _put(m, f"bhs.{pk}.chart_zero_m", r["chart_zero_bhs_if_28m_shift_m"], "m BHS-77", src,
             "-28 + k: where the charts' zero would sit if all of k were a datum offset", 2)
        _put(m, f"bhs.{pk}.k_sector_min_m", r["k_bhs_range_over_sectors_m"][0], "m", src, "R2 sector constants", 2)
        _put(m, f"bhs.{pk}.k_sector_max_m", r["k_bhs_range_over_sectors_m"][1], "m", src, decimals=2)
        md = r.get("mean_depth_2026", {})
        for var in ("shape", "total"):
            if var in md:
                _put(m, f"depth2026.{pk}.{var}.mean_m", md[var]["mean_depth_m"], "m", src, decimals=1)
                _put(m, f"depth2026.{pk}.{var}.area_km2", round(md[var]["sea_area_km2"]), "km2", src, decimals=0)
    _put(m, "depth2026.shape_min_m", rec["mean_depth_2026_shape_range_m"][0], "m", src,
         "mean depth of the study domain's sea at the 2026 level, products with their shape constants", 1)
    _put(m, "depth2026.shape_max_m", rec["mean_depth_2026_shape_range_m"][1], "m", src, decimals=1)
    _put(m, "depth2026.level_bhs_m", rec["level_2026_bhs_m"], "m BHS-77", src, decimals=2)
    _put(m, "bhs.mean_dN_fit_m", rec["mean_dN_fit_m"], "m", src, decimals=3)


def level_stats(m):
    rec = _json("level_stats.json")
    if not rec:
        return
    src = f"{REL}/level_stats.json"
    f = rec["fall"]
    _put(m, "levels.fall_smoothed_m", f["smoothed_range_m"], "m", src, "3-record running median, max to min", 2)
    _put(m, "levels.fall_raw_m", f["raw_range_m"], "m", src, "single records, max to min", 2)
    _put(m, "levels.smoothed_max_date", f["smoothed_max_date"], "date", src)
    _put(m, "levels.smoothed_min_date", f["smoothed_min_date"], "date", src)
    _put(m, "levels.smoothed_max_bhs_m", f["smoothed_max_bhs_m"], "m BHS-77", src, decimals=2)
    _put(m, "levels.smoothed_min_bhs_m", f["smoothed_min_bhs_m"], "m BHS-77", src, decimals=2)
    af = rec["annual_fall_1995_2025"]
    _put(m, "levels.annual_1995_bhs_m", af["1995_bhs_m"], "m BHS-77", src, decimals=2)
    _put(m, "levels.annual_2025_bhs_m", af["2025_bhs_m"], "m BHS-77", src, decimals=2)
    _put(m, "levels.annual_fall_1995_2025_m", af["fall_m"], "m", src, decimals=2)
    a = rec["annual"]
    for y in ("2023", "2024"):
        v = a.get(y, {}).get("dahiti_nogap_bhs_m")
        _put(m, f"levels.annual_{y}_bhs_m", v, "m BHS-77", src, "DAHITI, no gap limit", 2)
    r = rec["reference_2026"]
    _put(m, "levels.l2026_bhs_m", r["level_bhs_m"], "m BHS-77", src, decimals=2)
    _put(m, "levels.l2026_n_records", r["n_records"], "records", src)
    _put(m, "levels.janSep_minus_annual_median_m", r["jan_sep_minus_annual_mean_m"]["median"], "m", src,
         "complete DAHITI years 1993-2021", 2)
    _put(m, "levels.l2026_annual_equivalent_bhs_m", r["annual_equivalent_bhs_m"], "m BHS-77", src, decimals=2)
    _put(m, "levels.dec2024_bhs_m", r["dec_2024_scene_path_bhs_m"], "m BHS-77", src,
         "December 2024 mean, scene-level path (Hydroweb shifted, DAHITI gap)", 2)
    _put(m, "levels.janSep2025_bhs_m", r["jan_sep_2025"]["scene_path_bhs_m"], "m BHS-77", src, decimals=2)
    lc = rec["level_control"]
    for y, v in lc["dahiti_records_per_year"].items():
        _put(m, f"levels.dahiti_records_{y}", v, "records", src)
    _put(m, "levels.e3_below_28_share_hydroweb_pct", 100 * lc["e3_bins_below_minus_28"]["share_hydroweb_scenes"], "%",
         src, "April-October E3 scenes below -27.95 m", 0)
    cp = rec["caspcom_printed"]
    _put(m, "levels.caspcom_2025_mean6_m", cp.get("post_2025_mean_of_6_m"), "m BHS-77", src,
         "mean of the six 2025 post values of CASPCOM Bulletin 31 (no Baku)", 2)
    _put(m, "levels.caspcom_2024_mean6_m", cp.get("post_2024_mean_of_6_m"), "m BHS-77", src, decimals=2)


def add(m: dict) -> None:
    departures(m)
    _verdicts(m, "depaudit", "departure_audit_comparison.csv", V / "hindcast_departure", ("shape", "total"))
    planar(m)
    posts(m)
    datum_terms(m)
    bhs(m)
    level_stats(m)
