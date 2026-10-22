"""Metrics of revision analysis R (PREREG Amendment 18 items 3, 6 and 7).

Loaded by tools/build_metrics.revision_metrics, which calls add(m). Every key starts with
"rev.r."; each record is {"value", "unit", "source", "note"?} as put() in
tools/build_metrics.py, with "decimals" where the default would mislead. Direct reads of
results/revision/r/ (scripts/rev_r_control.py, rev_r_tables.py, rev_r_budget.py,
rev_r_diagnostics.py); nothing is recomputed here except counts over those tables. A
missing input adds no key.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results" / "revision" / "r"
REL = "results/revision/r"
PK = {"gebco_2026": "gebco2026", "gebco_2022": "gebco2022", "etopo_2022": "etopo2022",
      "srtm15p_v2.7": "srtm15p", "etopo1": "etopo1"}
SK = {"west": "west", "volga": "volga", "interfluve": "interfluve", "ural": "ural", "northeast": "northeast",
      "komsomolets": "komsomolets", "open_shelf": "openshelf"}
VALIDATED = ["west", "interfluve", "northeast", "open_shelf"]


def _put(m, key, value, unit, source, note=None, decimals=None):
    if value is None:
        return
    if isinstance(value, float) and not math.isfinite(value):
        return
    if hasattr(value, "item"):
        value = value.item()
    rec = {"value": value, "unit": unit, "source": source}
    if note:
        rec["note"] = note
    if decimals is not None:
        rec["decimals"] = decimals
    m[f"rev.r.{key}"] = rec


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _csv(path: Path):
    if not path.exists():
        return None
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _b(x) -> bool:
    return str(x).strip().lower() in ("true", "1")


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _counts(m, prefix, counts: dict, src, note):
    for p, c in (counts or {}).items():
        if p not in PK:
            continue
        _put(m, f"{prefix}.{PK[p]}.n_all7", c.get("all7"), "sectors", src, note)
        _put(m, f"{prefix}.{PK[p]}.n_validated4", c.get("validated4"), "sectors", src, note + "; validated sectors")
        _put(m, f"{prefix}.{PK[p]}.n_six", c.get("six_without_komsomolets"), "sectors", src,
             note + "; the six sectors other than Komsomolets Bay")


def control(m):
    src = f"{REL}/control/stop_rule.json"
    d = _json(R / "control" / "stop_rule.json")
    if not d:
        return
    note = "A18 item 6 stop rule (D-116): 4 mispredicting grids x 4 validated sectors; rounded control, DEM level range"
    dec = d["deciding"]
    _put(m, "control.stop.n_cases", dec["n_cases"], "cases", src, note)
    _put(m, "control.stop.n_excess_holds", dec["n_excess_holds"], "cases", src, note)
    _put(m, "control.stop.n_excess_fails", dec["n_excess_fails"], "cases", src, note)
    _put(m, "control.stop.fires", "yes" if dec["fires"] else "no", "", src, note)
    for k, v in d["all"].items():
        kind, lev = k.split("|")
        tag = f"{kind}_{'demrange' if lev == 'dem_range' else 'alllevels'}"
        _put(m, f"control.stop.{tag}.n_excess_holds", v["n_excess_holds"], "cases", src,
             f"excess rule holds ({kind} control, {lev})")
    for c, v in (d.get("control_constants_m") or {}).items():
        _put(m, f"control.const.{c.replace('ctrl_', '')}_m", v, "m", src, "the control's own D-070 area constant")
    _put(m, "control.floor_reproduced_max_abs_diff_m", d.get("floor_reproduced_max_abs_diff_m"), "m", src,
         "the control run's floor against the primary's (must be 0)", decimals=6)
    ex = _csv(R / "control" / "excess.csv")
    for r in ex or []:
        if r["levels"] != "dem_range":
            continue
        key = f"control.{r['control_kind']}.{PK[r['product']]}.{SK[r['sector']]}"
        _put(m, f"{key}.n_exceeding", int(r["n_exceeding"]), "levels", f"{REL}/control/excess.csv",
             "levels with D > max(Q, D_control)")
        _put(m, f"{key}.n_evaluable", int(r["n_evaluable"]), "levels", f"{REL}/control/excess.csv")
    lv = _csv(R / "control" / "levels.csv")
    if lv:
        med = {}
        for r in lv:
            if not (_b(r["evaluable_product"]) and _b(r["evaluable_control"]) and _b(r["in_dem_range"])):
                continue
            dc = _f(r["D_control"])
            if dc is None:
                continue
            med.setdefault((r["control_kind"], r["product"], r["sector"]), []).append(dc)
        for (kind, p, s), v in med.items():
            v = sorted(v)
            n = len(v)
            q = v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])
            _put(m, f"control.{kind}.{PK[p]}.{SK[s]}.D_control_median_km", q / 1000, "km", f"{REL}/control/levels.csv",
                 "median over evaluable levels of the control's per-level median displacement")


def tables(m):
    for name, note in (("static", "static-contour filter on the audit waterlines (D-117)"),
                       ("res", "resolution-matched floor (D-117)")):
        d = _json(R / name / "summary.json")
        if not d:
            continue
        src = f"{REL}/{name}/summary.json"
        _counts(m, f"{name}.mispredicts", d.get(f"counts_{'static' if name == 'static' else 'resolution_matched'}"),
                src, note)
        _put(m, f"{name}.n_changed", d.get("n_changed"), "product-sectors", src, note)
    d = _json(R / "static" / "summary.json")
    if d:
        src = f"{REL}/static/summary.json"
        for s, v in (d.get("audit_cells_static_share_by_sector") or {}).items():
            _put(m, f"static.share_audit_cells.{SK[s]}_pct", 100 * v, "%", src,
                 "observed audit waterline cells flagged static (all epochs)", decimals=1)
        for s, v in (d.get("dem_points_static_share_by_sector") or {}).items():
            if s in SK:
                _put(m, f"static.share_dem_points.{SK[s]}_pct", 100 * v, "%", src,
                     "DEM contour points flagged static (E2 + E3, D-075)", decimals=1)
        for s, v in (d.get("floor_median_by_sector_m") or {}).items():
            _put(m, f"static.floor_primary.{SK[s]}_km", _f(v.get("Q_median_primary_m")) / 1000 if _f(
                v.get("Q_median_primary_m")) is not None else None, "km", src, "median over levels of Q")
            _put(m, f"static.floor_static.{SK[s]}_km", _f(v.get("Q_median_static_m")) / 1000 if _f(
                v.get("Q_median_static_m")) is not None else None, "km", src, "median over levels of Q, filtered")
    d = _json(R / "constants" / "summary.json")
    if d:
        src = f"{REL}/constants/summary.json"
        for p, f in (d.get("fits") or {}).items():
            _put(m, f"constants.{PK[p]}.c_D_m", f["c_D_m"], "m", src, "displacement-fitted constant (D-118)")
            _put(m, f"constants.{PK[p]}.c_D_minus_c_area_m", f["c_D_minus_c_area_m"], "m", src)
            _put(m, f"constants.{PK[p]}.at_edge", "yes" if f["at_search_edge"] else "no", "", src)
        _counts(m, "constants.mispredicts_at_c_D", d.get("counts_at_c_D"), src, "verdicts under c_D (D-118)")
        _put(m, "constants.n_changed_at_c_D", d.get("n_changed_at_c_D"), "product-sectors", src)
        for p, b in (d.get("best_case") or {}).items():
            _put(m, f"constants.bestcase.{PK[p]}.n_no_single_constant", b["n_no_single_constant"], "sectors", src,
                 "sectors where even the best constant leaves a majority of levels above the floor")
            _put(m, f"constants.bestcase.{PK[p]}.n_no_single_constant_validated", b["n_no_single_constant_validated"],
                 "sectors", src, "validated sectors only")
    d = _json(R / "budget" / "summary.json")
    if d:
        src = f"{REL}/budget/summary.json"
        for s, g in (d.get("gradients_m_per_km") or {}).items():
            _put(m, f"budget.g_median.{SK[s]}", g.get("g_median_m_per_km"), "m per km", src,
                 "median |grad z| of the waterline DEM in the 1995-2026 band", decimals=3)
            _put(m, f"budget.g_mean.{SK[s]}", g.get("g_mean_m_per_km"), "m per km", src, "mean, R2.M4(b)", decimals=3)
        for r in d.get("sector_budget") or []:
            _put(m, f"budget.sigma_median.{SK[r['sector']]}_m", r.get("sigma_budget_median_m"), "m", src,
                 "median over levels of the 1-sigma vertical budget", decimals=3)
            _put(m, f"budget.horizontal_median.{SK[r['sector']]}_km",
                 _f(r.get("budget_horizontal_median_m")) / 1000 if _f(r.get("budget_horizontal_median_m")) else None,
                 "km", src, "1.645 sigma / g, median over levels")
        _counts(m, "budget.mispredicts_augmented", d.get("counts_augmented"), src, "under the augmented floor (A18 item 7)")
        _put(m, "budget.n_changed", d.get("n_changed"), "product-sectors", src)
    vdir = R / "variant"
    for name in ("veg_as_dry", "veg_as_water", "rR_awei", "rR_fixed0"):
        d = _json(vdir / f"summary_{name}.json")
        if not d:
            continue
        src = f"{REL}/variant/summary_{name}.json"
        key = name.replace("_", "")
        _counts(m, f"variant.{key}.mispredicts", d.get("counts_variant"), src, f"observation variant {name} (D-119)")
        _counts(m, f"variant.{key}.reference", d.get("counts_reference"), src, f"reference for {name}")
        _put(m, f"variant.{key}.n_changed", d.get("n_changed"), "product-sectors", src)
        _put(m, f"variant.{key}.n_changed_validated", d.get("n_changed_validated"), "product-sectors", src)
    d = _json(R / "diagnostics" / "season.json")
    if d:
        src = f"{REL}/diagnostics/season.json"
        _put(m, "season.rho_min", d["rho_range"][0], "", src, "Spearman, May-July share vs unit D residual, deltas")
        _put(m, "season.rho_max", d["rho_range"][1], "", src)
        _put(m, "season.n_p05", d["n_p_below_0_05"], "tests", src, "of 10, unadjusted")
        _put(m, "season.n_holm05", d["n_holm_below_0_05"], "tests", src, "of 10, Holm")
    d = _json(R / "diagnostics" / "komsomolets.json")
    if d:
        src = f"{REL}/diagnostics/komsomolets.json"
        for r in d.get("cells") or []:
            if r["cells"] != "band_-32_to_-24":
                continue
            pk = {"gebco_2026|gebco_2022": "g26_g22", "gebco_2026|srtm15p_v2.7": "g26_srtm",
                  "gebco_2022|srtm15p_v2.7": "g22_srtm", "all three identical": "all3"}[r["pair"]]
            _put(m, f"kom.identical.{pk}_pct", 100 * r["share_identical"], "%", src,
                 "Komsomolets native cells, band -32 to -24 m", decimals=1)
        _put(m, "kom.km_per_0p2m", d.get("km_per_0p2m_at_median_gradient"), "km", src,
             "0.2 m at the sector's median gradient")
    d = _json(R / "diagnostics" / "nmad.json")
    if d:
        src = f"{REL}/diagnostics/nmad.json"
        for r in d.get("rank_correlations") or []:
            if r["sector"] == "overall":
                _put(m, f"nmad.rho_overall.{r['source'].lower()}", r["rho_nmad_vs_D"], "", src,
                     "Spearman across 5 grids: NMAD vs median displacement")


def replica(m):
    d = _json(R / "replica_main" / "summary.json")
    if not d:
        return
    src = f"{REL}/replica_main/summary.json"
    for k in ("levels_D_max_abs_diff_m", "units_max_abs_diff_m", "floor_p95_max_abs_diff_m"):
        v = d.get(k)
        _put(m, f"replica.{k}", v[0] if isinstance(v, list) else v, "m", src,
             "the revision pass against results/hindcast/main (must be 0)", decimals=6)


def add(m: dict) -> None:
    replica(m)
    control(m)
    tables(m)
