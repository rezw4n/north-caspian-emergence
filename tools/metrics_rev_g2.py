"""Metrics of revision-2 analysis G (PREREG Amendment 19 item 13; D-150): the Great Salt
Lake lake-bed control. Loaded by the metrics build, which calls add(m). Every key starts
with ``rev2.g.``; each record is {"value", "unit", "source", "note"?}, as _put() in
tools/metrics_rev_g.py. A missing input file adds nothing.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
G2 = ROOT / "results" / "revision2" / "g"
PRODUCT_KEY = {"gebco_2026": "gebco2026", "gebco_2022": "gebco2022", "etopo_2022": "etopo2022", "etopo1": "etopo1",
               "srtm15p_v2.7": "srtm15p", "dep3": "dep3"}
NOTE = "exploratory, unregistered additional control (A19 item 13; D-150); the registered verdict stands"


def _put(m: dict, key: str, value, unit: str, source: str, note: str | None = None) -> None:
    if value is None:
        return
    if isinstance(value, float) and not math.isfinite(value):
        return
    if hasattr(value, "item"):
        value = value.item()
    rec = {"value": value, "unit": unit, "source": source}
    if note:
        rec["note"] = note
    m[f"rev2.g.{key}"] = rec


def _json(rel: str):
    p = G2 / rel
    return (json.loads(p.read_text(encoding="utf-8")), f"results/revision2/g/{rel}") if p.exists() else (None, None)


def _r(x, d=3):
    if x is None or isinstance(x, str) or (isinstance(x, float) and not math.isfinite(x)):
        return None
    return round(float(x), d)


def _yn(x):
    return None if x is None else ("yes" if x else "no")


def download(m: dict) -> None:
    j, src = _json("gsl_lakebed_download.json")
    if not j:
        return
    for r in j["requests"]:
        if r["what"] == "GSLDEM.tif":
            _put(m, "lakebed.download.bytes", r["bytes"], "bytes", src, "GSLDEM.tif, HydroShare 582060f00f6b443bb26e896426d9f62a")
            _put(m, "lakebed.download.md5_matches", _yn(r["md5_matches_listing"]), "text", src)
            _put(m, "lakebed.download.date", r["requested_utc"][:10], "date", src)


def prepare(m: dict) -> None:
    j, src = _json("gsl_lakebed_prepare.json")
    if not j:
        return
    _put(m, "lakebed.native_res_m", _r(j["source"]["native_res_m"][0], 2), "m", src, "NAD83 / UTM 12N")
    for arm, d in j["gauge_delta_m"].items():
        _put(m, f"lakebed.{arm}.navd88_minus_ngvd29_m", _r(d, 3), "m", src, "VERTCON 3.0 via NCAT at the arm's gauge (D-094)")
    for arm, b in j["boundaries"].items():
        _put(m, f"lakebed.{arm}.sounded_top_navd88_m", _r(b["sounded_top_navd88_m"], 2), "m", src,
             "top of the soundings (4,193 ft south, 4,194 ft north, NGVD 29; Baskin 2005, 2006)")
        _put(m, f"lakebed.{arm}.contours_top_navd88_m", _r(b["contours_top_navd88_m"], 2), "m", src,
             "top of the bathymetric model (4,200 ft NGVD 29); NED above")
    for arm, a in j["arms"].items():
        _put(m, f"lakebed.{arm}.share_with_height_pct", _r(100 * a["share_with_height"], 2), "%", src, "arm cells at 60 m")
        _put(m, f"lakebed.{arm}.min_ngvd29_m", _r(a["min_ngvd29_m"], 2), "m", src)
        v = a["vertcon_cell_minus_gauge_m"]
        _put(m, f"lakebed.{arm}.vertcon_cell_minus_gauge_max_abs_m", _r(v["max_abs"], 3), "m", src,
             "per-cell VERTCON 3.0 (0.2 deg NCAT lattice) minus the gauge constant")
        _put(m, f"lakebed.{arm}.vertcon_rerun_required", _yn(a["vertcon_rerun_required"]), "text", src, "D-150 item 1 (> 0.05 m)")


def audit(m: dict) -> None:
    j, src = _json("gsl_lakebed.json")
    if not j:
        return
    for name, rec in j["sets"].items():
        k0 = f"lakebed.{name}"
        _put(m, f"{k0}.control_result", rec["control_result"], "text", src, NOTE)
        _put(m, f"{k0}.mispredicts_in_basin", _yn(rec["in_basin_shape"]), "text", src, NOTE)
        _put(m, f"{k0}.mispredicts_in_basin_total", _yn(rec["in_basin_total"]), "text", src, NOTE)
        v = rec["verdict_with_this_control"]
        _put(m, f"{k0}.verdict_with_this_control", v["verdict"], "text", src,
             "Amendment 16 item 9 with this control in place of 3DEP; descriptive")
        for arm, a in rec["arms"].items():
            _put(m, f"{k0}.{arm}.shape_offset_m", _r(rec["shape_offsets"][arm]["offset_m"], 2), "m", src)
            for var in ("shape", "total"):
                x = a.get(var)
                if not x:
                    continue
                k = f"{k0}.{arm}.{var}"
                _put(m, f"{k}.n_exceeding", x["n_exceeding"], "levels", src, NOTE)
                _put(m, f"{k}.n_evaluable", x["n_evaluable"], "levels", src, NOTE)
                _put(m, f"{k}.median_displacement_m", _r(x["median_displacement_m"], 0), "m", src, NOTE)
                _put(m, f"{k}.median_floor_p95_m", _r(x["median_noise_p95_m"], 0), "m", src)
                _put(m, f"{k}.mispredicts", _yn(x["mispredicts"]), "text", src, NOTE)
                _put(m, f"{k}.mispredicts_pooled_floor", _yn(x["mispredicts_pooled_floor"]), "text", src)
                _put(m, f"{k}.bins", x["bins"], "text", src)
            for var, x in (a.get("at_3dep_levels") or {}).items():
                k = f"{k0}.{arm}.at_3dep_levels.{var}"
                _put(m, f"{k}.n_exceeding", x["n_exceeding"], "levels", src, "rows of the all-levels run at 3DEP's bins")
                _put(m, f"{k}.n_evaluable", x["n_evaluable"], "levels", src)
                _put(m, f"{k}.median_displacement_m", _r(x["median_displacement_m"], 0), "m", src)
            for c, vv in (a.get("by_source_class") or {}).items():
                x = vv["shape"]
                if x["n_evaluable"]:
                    k = f"{k0}.{arm}.class.{c}"
                    _put(m, f"{k}.n_exceeding", x["n_exceeding"], "levels", src, "rows of the all-levels run (shape)")
                    _put(m, f"{k}.n_evaluable", x["n_evaluable"], "levels", src)
                    _put(m, f"{k}.median_displacement_m", _r(x["median_displacement_m"], 0), "m", src)
            if a.get("dep3_registered"):
                d = a["dep3_registered"]
                _put(m, f"{k0}.{arm}.dep3.n_exceeding", d["n_exceeding"], "levels", src, "registered 3DEP control (D-094)")
                _put(m, f"{k0}.{arm}.dep3.n_evaluable", d["n_evaluable"], "levels", src)
                _put(m, f"{k0}.{arm}.dep3.median_displacement_m", _r(d["median_displacement_m"], 0), "m", src)
            if a.get("survey"):
                s = a["survey"]
                _put(m, f"lakebed.survey.{arm}.start", s["window"][0], "date", src, s["wording"])
                _put(m, f"lakebed.survey.{arm}.end", s["window"][1], "date", src, s["wording"])
                _put(m, f"lakebed.survey.{arm}.gauge_min_navd88_m", _r(s["gauge_min_navd88_m"], 2), "m", src)
                _put(m, f"lakebed.survey.{arm}.gauge_max_navd88_m", _r(s["gauge_max_navd88_m"], 2), "m", src)
    for arm, cc in (j.get("coverage_by_source_class") or {}).items():
        for c, x in cc.items():
            k = f"lakebed.coverage.{arm}.{c}"
            _put(m, f"{k}.n_bins", x["n_bins"], "bins", src)
            _put(m, f"{k}.level_low_navd88_m", _r(x["levels_navd88_m"][0], 2), "m", src, "bin centre")
            _put(m, f"{k}.level_high_navd88_m", _r(x["levels_navd88_m"][1], 2), "m", src, "bin centre")
            _put(m, f"{k}.n_first_reached_after_survey", x["n_first_reached_after_survey"], "bins", src)
            _put(m, f"{k}.n_under_water_in_survey", x["n_under_water_in_survey"], "bins", src)
    for p, vv in (j.get("grids_at_sounded_levels") or {}).items():
        x = vv.get("shape")
        if not x:
            continue
        k = f"lakebed.grids_at_sounded.{PRODUCT_KEY.get(p, p)}"
        _put(m, f"{k}.mispredicts_in_basin", _yn(x["in_basin"]), "text", src, "five grids at the lake-bed control's sounded levels")
        for arm, y in x["arms"].items():
            _put(m, f"{k}.{arm}.n_exceeding", y["n_exceeding"], "levels", src)
            _put(m, f"{k}.{arm}.n_evaluable", y["n_evaluable"], "levels", src)
            _put(m, f"{k}.{arm}.median_displacement_m", _r(y["median_displacement_m"], 0), "m", src)
    v = j.get("grids_at_sounded_levels_verdict")
    if v:
        _put(m, "lakebed.grids_at_sounded.verdict", v["verdict"], "text", src, "descriptive")
        _put(m, "lakebed.grids_at_sounded.n_products_mispredict", v["n_mispredict"], "products", src)
    for ref, aa in (j.get("per_level_ratio_to_lakebed") or {}).items():
        for arm, x in aa.items():
            _put(m, f"lakebed.ratio.{PRODUCT_KEY.get(ref, ref)}.{arm}.median", _r(x["median_ratio_to_lakebed"], 2), "ratio",
                 src, "per level: reference median displacement / lake-bed median (shape; all-levels run)")
            _put(m, f"lakebed.ratio.{PRODUCT_KEY.get(ref, ref)}.{arm}.n_levels", x["n_levels"], "levels", src)


def citation(m: dict) -> None:
    j, src = _json("citation.json")
    if j:
        _put(m, "lakebed.citation.year", j["dataset"]["year"], "year", src, "HydroShare's suggested citation")
        _put(m, "lakebed.citation.licence", j["dataset"]["licence"], "text", src)


def add(m: dict) -> None:
    download(m)
    prepare(m)
    audit(m)
    citation(m)
