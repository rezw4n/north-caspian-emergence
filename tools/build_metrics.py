"""Assemble results/metrics.json, the single source of every number in the manuscript
(CLAUDE.md integrity rule 2), from the committed result files.

Each metric is {"value", "unit", "source", "note"?}. Keys are dotted paths
(e.g. "levels.dahiti.n_records"). Rebuilt by `make metrics`; never edited by hand.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def load(rel: str) -> dict:
    p = RES / rel
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def put(metrics: dict, key: str, value, unit: str, source: str, note: str | None = None) -> None:
    if value is None:
        return
    rec = {"value": value, "unit": unit, "source": source}
    if note:
        rec["note"] = note
    metrics[key] = rec


def _half_up(x: float) -> int:
    import math

    return int(math.floor(x + 0.5))


def study_area_metrics(m: dict) -> None:
    """Domain and sector areas (scripts/a0_study_area.py) and the sector-layout check
    against the shorelines traced in the 15-arc-second grids (scripts/a0_sector_edges.py)."""
    sa = load("study_area/sector_areas.json")
    src = "results/study_area/sector_areas.json"
    put(m, "study_area.domain_km2", sa.get("domain_km2"), "km2", src,
        "edges straight in lon/lat, densified before the equal-area projection")
    for k, v in sa.get("sectors_km2", {}).items():
        put(m, f"study_area.sector_km2.{k}", v, "km2", src)
    geo = sa.get("geodesic_reading") or {}
    if geo:
        import math

        gnote = "the same polygons read with geodesic edges, as Earth Engine reads lon/lat GeoJSON"
        put(m, "study_area.geodesic.domain_km2", geo["domain_km2"], "km2", src, gnote)
        put(m, "study_area.geodesic.north_edge_offset_km", _half_up(geo["north_edge_offset_km"]), "km", src,
            "largest distance between the geodesic and the lon/lat reading of the domain's northern edge, "
            "rounded; " + gnote)
        put(m, "study_area.geodesic.other_edge_offset_max_km", math.ceil(geo["other_edges_max_offset_km"]), "km",
            src, "upper bound (rounded up) over every other edge of the domain and the sectors; " + gnote)

    se = load("study_area/sector_edges.json")
    grids = se.get("grids") or {}
    if not grids:
        return
    src = "results/study_area/sector_edges.json"
    note = (f"over the grids {', '.join(grids)}; shorelines at the DAHITI levels "
            f"{se['present_level_m']} m ({se['present_level_time'][:10]}) and {se['record_max_level_m']} m, no offset")
    put(m, "study_area.layout.n_grids", len(grids), "grids", src, note)
    shares = [100 * g["band_share_coastal_sectors"] for g in grids.values()]
    put(m, "study_area.layout.band_share_coastal_pct_min", _half_up(min(shares)), "%", src,
        "share of the seabed between the record-maximum and present shorelines inside the coastal sectors; " + note)
    put(m, "study_area.layout.band_share_coastal_pct_max", _half_up(max(shares)), "%", src, note)
    coastal = [k for k in next(iter(grids.values()))["sectors"] if k != "open_shelf"]
    reach = {k: [g["sectors"][k]["max_offshore_km"] for g in grids.values()] for k in coastal}
    edge = {k: [g["sectors"][k]["offshore_edge"]["p50_km"] for g in grids.values()] for k in coastal}
    put(m, "study_area.layout.max_offshore_km_min", _half_up(min(min(v) for v in reach.values())), "km", src,
        "farthest distance of a sector's sea cells from the present shoreline, smallest over sectors and grids; "
        + note)
    put(m, "study_area.layout.max_offshore_km_max", _half_up(max(max(v) for v in reach.values())), "km", src, note)
    for k in coastal:
        put(m, f"study_area.layout.max_offshore_km.{k}.min", _half_up(min(reach[k])), "km", src, note)
        put(m, f"study_area.layout.max_offshore_km.{k}.max", _half_up(max(reach[k])), "km", src, note)
        put(m, f"study_area.layout.edge_p50_km.{k}.min", _half_up(min(edge[k])), "km", src,
            "median signed distance of the boundary shared with the open shelf from the present shoreline "
            "(negative: on land); " + note)
        put(m, f"study_area.layout.edge_p50_km.{k}.max", _half_up(max(edge[k])), "km", src, note)
    readings = [g["edge_readings"] for g in grids.values() if "edge_readings" in g]
    if readings:
        import math

        pct = [100 * g["edge_readings"]["band_changed_km2"] / g["band_km2_total"]
               for g in grids.values() if "edge_readings" in g and g["band_km2_total"]]
        put(m, "study_area.geodesic.band_changed_pct_max", math.ceil(max(pct)), "%", src,
            "upper bound (rounded up), over the grids, of the share of the seabed between the record-maximum and "
            "present shorelines inside the domain that changes domain or sector between the lon/lat and the "
            "geodesic edge readings; " + note)
        put(m, "study_area.geodesic.band_changed_km2_max", max(r["band_changed_km2"] for r in readings), "km2",
            src, "largest over the grids; " + note)
        put(m, "study_area.geodesic.north_edge_sea_at_max_km2_max",
            max(r["north_edge_sea_at_max_km2"] for r in readings), "km2", src,
            "sea at the record-maximum level between the two readings of the northern edge, largest over the "
            "grids; " + note)


def main() -> int:
    m: dict = {}
    inv = load("inventory/inventory_meta.json")
    src = "results/inventory/inventory_meta.json"
    put(m, "inventory.n_scenes_total", inv.get("n_scenes_total"), "scenes", src)
    for k, v in inv.get("n_by_sensor", {}).items():
        put(m, f"inventory.n_scenes.{k}", v, "scenes", src)
    for k, v in inv.get("n_cloud_below_60_by_sensor", {}).items():
        put(m, f"inventory.n_scenes_cc_lt60.{k}", v, "scenes", src)
    if inv.get("landsat_path_rows"):
        put(m, "inventory.n_landsat_path_rows", len(inv["landsat_path_rows"]), "path/rows", src)
    put(m, "inventory.n_s2_tiles", inv.get("n_s2_tiles"), "tiles", src)
    for k, v in inv.get("first_by_sensor", {}).items():
        put(m, f"inventory.first.{k}", v[:10], "date", src)
    for k, v in inv.get("last_by_sensor", {}).items():
        put(m, f"inventory.last.{k}", v[:10], "date", src)

    study_area_metrics(m)

    dq = load("qc/dahiti_summary.json")
    src = "results/qc/dahiti_summary.json"
    put(m, "levels.dahiti.n_records", dq.get("n_records"), "records", src)
    put(m, "levels.dahiti.first", (dq.get("first") or "")[:10] or None, "date", src)
    put(m, "levels.dahiti.last", (dq.get("last") or "")[:10] or None, "date", src)
    put(m, "levels.dahiti.min_m", dq.get("level_min_m"), "m (DAHITI reference)", src)
    put(m, "levels.dahiti.max_m", dq.get("level_max_m"), "m (DAHITI reference)", src)
    put(m, "levels.dahiti.date_of_min", (dq.get("time_of_min") or "")[:10] or None, "date", src)
    put(m, "levels.dahiti.date_of_max", (dq.get("time_of_max") or "")[:10] or None, "date", src)
    if dq.get("level_max_m") is not None and dq.get("level_min_m") is not None:
        put(m, "levels.dahiti.range_m", round(dq["level_max_m"] - dq["level_min_m"], 3), "m", src,
            "highest single record minus lowest single record (each one noisy record; use "
            "levels.dahiti.smoothed.range_m for the fall of the sea)")
    sm = dq.get("smoothed") or {}
    if sm:
        note = f"{sm['rule']} (D-021 correction)"
        put(m, "levels.dahiti.smoothed.window_records", sm.get("k"), "records", src, note)
        put(m, "levels.dahiti.smoothed.max_m", sm.get("level_max_m"), "m (DAHITI reference)", src, note)
        put(m, "levels.dahiti.smoothed.min_m", sm.get("level_min_m"), "m (DAHITI reference)", src, note)
        put(m, "levels.dahiti.smoothed.date_of_max", (sm.get("time_of_max") or "")[:10] or None, "date", src, note)
        put(m, "levels.dahiti.smoothed.date_of_min", (sm.get("time_of_min") or "")[:10] or None, "date", src, note)
        put(m, "levels.dahiti.smoothed.range_m", sm.get("range_m"), "m", src,
            "maximum minus minimum of the running median; " + note)

    put(m, "levels.dahiti.sigma_formal_max_m", dq.get("sigma_max_m"), "m", src, "formal wse_u")
    rpy = {int(k): v for k, v in (dq.get("records_per_year") or {}).items()}
    # the two sampling periods compared in Methods (Water levels, gap filling); the years
    # are stored as text so that the macros print 1993, not 1\,993
    dense_years, sparse_years = (1993, 2021), (2022, 2025)
    sparse = [rpy[y] for y in range(sparse_years[0], sparse_years[1] + 1) if y in rpy]
    if sparse:
        put(m, "levels.dahiti.records_per_year_2022_2025_min", min(sparse), "records per year", src)
        put(m, "levels.dahiti.records_per_year_2022_2025_max", max(sparse), "records per year", src)
        put(m, "levels.dahiti.sparse_period", f"{sparse_years[0]}--{sparse_years[1]}", "years", src,
            "years of the records_per_year_2022_2025 metrics")
    dense = [rpy[y] for y in range(dense_years[0], dense_years[1] + 1) if y in rpy]
    if dense:
        put(m, "levels.dahiti.records_per_year_1993_2021_median", int(sorted(dense)[len(dense) // 2]),
            "records per year", src)
        put(m, "levels.dahiti.dense_period", f"{dense_years[0]}--{dense_years[1]}", "years", src,
            "years of the records_per_year_1993_2021 metric")
    put(m, "levels.dahiti.max_gap_days", dq.get("max_gap_days"), "days", src, "fractional days")
    put(m, "levels.dahiti.median_gap_days", dq.get("median_gap_days"), "days", src, "fractional days")
    put(m, "levels.dahiti.n_gaps_over_60_days", dq.get("n_gaps_over_60_days"), "gaps", src)

    xc = load("qc/level_crosscheck_w20d.json")
    src = "results/qc/level_crosscheck_w20d.json"
    put(m, "levels.hydroweb.n_records", xc.get("hydroweb_records"), "records", src)
    put(m, "levels.crosscheck.n_pairs", xc.get("n_pairs"), "pairs", src)
    put(m, "levels.crosscheck.window_days", xc.get("max_half_gap_days"), "days", src)
    put(m, "levels.hydroweb.last", (xc.get("hydroweb_last") or "")[:10] or None, "date", src)
    put(m, "levels.crosscheck.offset_median_m", xc.get("offset_median_m"), "m", src, "DAHITI minus Hydroweb")
    put(m, "levels.crosscheck.nmad_m", xc.get("diff_nmad_m"), "m", src)
    put(m, "levels.crosscheck.drift_m_per_decade", xc.get("drift_m_per_decade"), "m per decade", src)

    sl = load("levels/scene_levels_summary.json")
    src = "results/levels/scene_levels_summary.json"
    def put_dp(key, value, unit, source, note=None, dp=3):
        put(m, key, value, unit, source, note)
        if key in m:
            m[key]["decimals"] = dp

    joint = "joint estimate with q from cross-validated interpolation residuals (D-018 correction)"
    put(m, "levels.dahiti_noise_floor_m", sl.get("dahiti_noise_floor_m"), "m", src, joint)
    put(m, "levels.se.dahiti_noise_floor_m", sl.get("dahiti_noise_floor_se_m"), "m", src,
        "moving-block bootstrap SE of levels.dahiti_noise_floor_m")
    put(m, "levels.hydroweb_noise_floor_m", sl.get("hydroweb_noise_floor_m"), "m", src,
        "Hydroweb record noise with q fixed to the DAHITI estimate")
    put(m, "levels.q_m2_per_day", sl.get("q_dahiti_m2_per_day"), "m2 per day", src,
        "effective diffusion of the linear-interpolation error (absorbs sub-sampling variability); " + joint)
    put(m, "levels.se.q_m2_per_day", sl.get("q_dahiti_se_m2_per_day"), "m2 per day", src,
        "moving-block bootstrap SE of levels.q_m2_per_day")
    put(m, "levels.epoch_E1_end", (sl.get("epoch_E1_end") or "")[:10] or None, "date", src,
        sl.get("epoch_E1_end_rule"))
    put(m, "levels.max_gap_days", sl.get("max_gap_days"), "days", src,
        "gap rule: caspian.levels default fixed in commit aabacee before PREREG v1.0 and before the DAHITI "
        "data arrived; not in PREREG v1.0; stated for the record in PREREG Amendment 4 item 2 (D-038)")
    off = sl.get("hydroweb_offset") or {}
    if off:
        note = off.get("method")
        hw_ = off.get("half_window_years")
        put(m, "levels.gapfill.half_window_years", int(hw_) if hw_ is not None and float(hw_).is_integer() else hw_,
            "years", src, note)
        put(m, "levels.gapfill.offset_median_m", off.get("filled_scenes_offset_median_m"), "m", src,
            "median over gap-filled scenes of the moving-median offset applied; " + note)
        put_dp("levels.gapfill.offset_min_m", off.get("filled_scenes_offset_min_m"), "m", src, note)
        put_dp("levels.gapfill.offset_max_m", off.get("filled_scenes_offset_max_m"), "m", src, note)
        put(m, "levels.gapfill.offset_sigma_p50_m", off.get("filled_scenes_offset_sigma_p50_m"), "m", src,
            "median over gap-filled scenes of the common offset 1-sigma (SE of the moving median and the "
            "window-local tau); " + (off.get("tau_rule") or ""))
        put_dp("levels.gapfill.offset_sigma_p50_record_tau_m", off.get("filled_scenes_offset_sigma_p50_with_record_tau_m"),
               "m", src, "the same with the record-wide tau instead (superseded; first D-020 correction)")
        put(m, "levels.gapfill.resid_nmad_m", off.get("pair_residual_nmad_m"), "m", src,
            "NMAD of the pair differences about the moving median (per-record random term)")
        put_dp("levels.gapfill.tau_m", off.get("excess_sd_tau_m"), "m", src,
            "record-wide (all years pooled) year-to-year offset variability beyond sampling noise (method of "
            "moments); the floor of the applied tau, not the applied value (see levels.gapfill.tau_p50_m)")
        put_dp("levels.gapfill.tau_p50_m", off.get("filled_scenes_tau_p50_m"), "m", src,
               "median over gap-filled scenes of the applied window-local tau; " + (off.get("tau_rule") or ""))
        put_dp("levels.gapfill.tau_min_m", off.get("filled_scenes_tau_min_m"), "m", src, "applied tau, minimum")
        put_dp("levels.gapfill.tau_max_m", off.get("filled_scenes_tau_max_m"), "m", src, "applied tau, maximum")
        put(m, "levels.gapfill.tau_min_years", off.get("tau_min_years"), "years", src,
            "minimum number of years in the local tau window")
        put(m, "levels.gapfill.n_scenes_tau_at_floor", off.get("n_filled_scenes_tau_at_record_floor"), "scenes",
            src, "gap-filled scenes whose local tau is at or below the record-wide floor")
    dg = sl.get("diagnostics") or {}
    if dg:
        put(m, "levels.gapfill.n_scenes_other_bin_pooled_offset", dg.get("n_hydroweb_scenes_other_bin_with_pooled_offset"),
            "scenes", src, "gap-filled scenes whose 0.10 m bin (origin -30 m, DAHITI reference) differs if the pooled "
            "1992-2026 offset is used instead of the moving median")
        put(m, "levels.diag.n_scenes_above_smoothed_max", dg.get("n_scenes_above_smoothed_max"), "scenes", src,
            "scenes whose interpolated level exceeds the running-median maximum")
        put(m, "levels.diag.n_scenes_between_raw_and_smoothed_max", dg.get("n_scenes_between_raw_and_smoothed_max_dates"),
            "scenes", src, "scenes whose epoch (E1 or E2) depends on raw versus running-median maximum")
        nz = dg.get("dahiti_neighbour_z") or {}
        put_dp("levels.diag.neighbour_z_threshold", nz.get("z_threshold"), "sigma", src,
               "diagnostic threshold only; no record is removed", dp=1)
        put(m, "levels.diag.neighbour_z_n", nz.get("n"), "records", src)
        put(m, "levels.diag.neighbour_z_n_gt_2_5", nz.get("n_abs_gt_2_5"), "records", src,
            "records with |z| > 2.5 against their neighbours under the fitted noise and diffusion")
        put_dp("levels.diag.neighbour_z_expected", nz.get("expected_gaussian"), "records", src,
               "expected count of |z| > 2.5 under the Gaussian model", dp=1)
        put_dp("levels.diag.raw_max_record_z", nz.get("z_of_raw_max_record"), "sigma", src, dp=1)
        es = dg.get("e1_end_sensitivity") or {}
        for r in es.get("by_k", []):
            k_ = r["k"]
            nt = "single-record maximum" if k_ == 1 else f"maximum of the {k_}-record running median"
            put(m, f"levels.diag.e1_end.k{k_}.date", (r.get("time_of_max") or "")[:10] or None, "date", src,
                f"E1 end with the {nt} (sensitivity; applied k = {es.get('applied_k')})")
            put(m, f"levels.diag.e1_end.k{k_}.n_E1", r.get("n_E1_scenes"), "scenes", src,
                f"levelled E1 scenes with the {nt}")
        top = es.get("running_median_top_two_m")
        if top:
            put_dp("levels.diag.e1_end.top_two_margin_m", round(top[0] - top[1], 4), "m", src,
                   "margin between the two highest running-median values, which sets the applied E1 date")
        hq = dg.get("hydroweb_joint_free_q") or {}
        if hq:
            put(m, "levels.diag.hydroweb_free_q_m2_per_day", hq.get("q_m2_per_day"), "m2 per day", src,
                "Hydroweb's own joint (sigma, q) fit; not applied (DAHITI's q is used for the Hydroweb bridge)")
            put_dp("levels.diag.hydroweb_free_q_sigma_change_max_m", hq.get("filled_scenes_sigma_change_max_abs_m"),
                   "m", src, hq.get("note"))
        mdp = dg.get("offset_fill_year_max_departure") or {}
        if mdp:
            # calendar years are stored as text so that the macro prints 2023, not 2\,023
            put(m, "levels.gapfill.max_year_departure.year", str(mdp["year"]) if mdp.get("year") else None,
                "year", src, mdp.get("note"))
            put_dp("levels.gapfill.max_year_departure.m", mdp.get("departure_m"), "m", src, mdp.get("note"))
        c5 = dg.get("offset_check_5d_main_fill_year") or {}
        if c5:
            c5n = ("year with the most gap-filled scenes: median DAHITI - shifted Hydroweb at the +/-5-day epochs "
                   "(in-sample: a subset of the pairs that set the offset)")
            put(m, "levels.gapfill.check5.year", str(c5["year"]) if c5.get("year") else None, "year", src, c5n)
            put(m, "levels.gapfill.check5.n_filled_scenes", c5.get("n_filled_scenes"), "scenes", src, c5n)
            put(m, "levels.gapfill.check5.n_pairs", c5.get("n_pairs_5d"), "pairs", src, c5n)
            put_dp("levels.gapfill.check5.median_resid_m", c5.get("median_resid_m"), "m", src, c5n)
            put_dp("levels.gapfill.check5.offset_sigma_p50_m", c5.get("offset_sigma_p50_m"), "m", src, c5n)
    for e, rng_ in (sl.get("level_range_by_epoch_m") or {}).items():
        if rng_:
            put(m, f"levels.epoch_level_range.{e}.min_m", rng_[0], "m (DAHITI reference)", src)
            put(m, f"levels.epoch_level_range.{e}.max_m", rng_[1], "m (DAHITI reference)", src)
    for s, v in (sl.get("sigma_m_p50_by_source") or {}).items():
        put_dp(f"levels.scene_sigma_p50_by_source.{s}", v, "m", src)
    if sl.get("by_sensor_source"):
        src_tot = {}
        for sensor, d in sl["by_sensor_source"].items():
            for s, n in d.items():
                src_tot[s] = src_tot.get(s, 0) + int(n)
        for s, n in src_tot.items():
            put(m, f"levels.n_scenes_by_source.{s}", n, "scenes", src)
    if sl.get("by_sensor_epoch"):
        ep_tot = {}
        for sensor, d in sl["by_sensor_epoch"].items():
            for e, n in d.items():
                ep_tot[e] = ep_tot.get(e, 0) + int(n)
        for e, n in ep_tot.items():
            put(m, f"levels.n_scenes_by_epoch.{e}", n, "scenes", src)
    if sl.get("sigma_m_percentiles_p50_p90_p99"):
        p50, p90, p99 = sl["sigma_m_percentiles_p50_p90_p99"]
        put(m, "levels.scene_sigma_p50_m", p50, "m", src)
        put(m, "levels.scene_sigma_p90_m", p90, "m", src)
        put(m, "levels.scene_sigma_p99_m", p99, "m", src)

    # Values quoted from the literature (refs/literature_values.csv: value, unit, DOI,
    # location, verbatim quote); they enter the text through the same macros.
    import csv
    lit = ROOT / "refs" / "literature_values.csv"
    if lit.exists():
        for r in csv.DictReader(lit.open(encoding="utf-8")):
            raw = r["value"].strip()
            dp = len(raw.split(".")[1]) if "." in raw else 0
            v = float(raw) if dp else int(raw)
            put(m, r["key"], v, r["unit"], f"doi:{r['doi']} ({r['location']})")
            m[r["key"]]["decimals"] = dp  # keep the precision written in the source

    # Analysis parameters exported from the code constants (tools/export_params.py).
    # Identifiers that are integers but not quantities (the random seed) are stored as
    # text, so that the macro prints 20261020, not 20\,261\,020: make_macros puts
    # thousands separators on every number and has no per-key rule for integers.
    as_text = {"params.seed"}
    for k, rec in load("params.json").items():
        v = str(rec["value"]) if k in as_text and rec["value"] is not None else rec["value"]
        put(m, k, v, rec["unit"], rec["source"])
        if k in m and "decimals" in rec:
            m[k]["decimals"] = rec["decimals"]

    out = RES / "metrics.json"
    out.write_text(json.dumps(dict(sorted(m.items())), indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(m)} metrics to {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
