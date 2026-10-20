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

    sa = load("study_area/sector_areas.json")
    src = "results/study_area/sector_areas.json"
    put(m, "study_area.domain_km2", sa.get("domain_km2"), "km2", src)
    for k, v in sa.get("sectors_km2", {}).items():
        put(m, f"study_area.sector_km2.{k}", v, "km2", src)

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
            "record maximum minus minimum")

    put(m, "levels.dahiti.sigma_formal_max_m", dq.get("sigma_max_m"), "m", src, "formal wse_u")
    rpy = {int(k): v for k, v in (dq.get("records_per_year") or {}).items()}
    sparse = [rpy[y] for y in range(2022, 2026) if y in rpy]
    if sparse:
        put(m, "levels.dahiti.records_per_year_2022_2025_min", min(sparse), "records per year", src)
        put(m, "levels.dahiti.records_per_year_2022_2025_max", max(sparse), "records per year", src)
    dense = [rpy[y] for y in range(1993, 2022) if y in rpy]
    if dense:
        put(m, "levels.dahiti.records_per_year_1993_2021_median", int(sorted(dense)[len(dense) // 2]),
            "records per year", src)
    put(m, "levels.dahiti.max_gap_days", dq.get("max_gap_days"), "days", src)

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
    put(m, "levels.dahiti_noise_floor_m", sl.get("dahiti_noise_floor_m"), "m", src)
    put(m, "levels.hydroweb_noise_floor_m", sl.get("hydroweb_noise_floor_m"), "m", src)
    put(m, "levels.q_m2_per_day", sl.get("q_dahiti_m2_per_day"), "m2 per day", src)
    put(m, "levels.epoch_E1_end", (sl.get("epoch_E1_end") or "")[:10] or None, "date", src)
    put(m, "levels.max_gap_days", sl.get("max_gap_days"), "days", src, "PREREG section 4 gap rule")
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
    for k, rec in load("params.json").items():
        put(m, k, rec["value"], rec["unit"], rec["source"])

    out = RES / "metrics.json"
    out.write_text(json.dumps(dict(sorted(m.items())), indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(m)} metrics to {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
