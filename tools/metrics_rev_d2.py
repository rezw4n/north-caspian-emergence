"""Metrics of second-review analysis D (PREREG Amendment 19 items 4, 6, 10, 12 and 14; D-149).

Loaded by tools/build_metrics.revision_metrics, which calls add(m). Every key starts with
"rev2.d."; each record is {"value", "unit", "source", "note"?, "decimals"?} as _put in
tools/metrics_rev_r.py. Direct reads of results/revision2/d/ (scripts/rev2_d_tables.py,
rev2_d_constants.py, rev2_d_obsbias.py, rev2_d_sources.py); nothing is recomputed here. A
missing input adds no key.
"""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "results" / "revision2" / "d"
REL = "results/revision2/d"
PK = {"gebco_2026": "gebco2026", "gebco_2022": "gebco2022", "etopo_2022": "etopo2022",
      "srtm15p_v2.7": "srtm15p", "etopo1": "etopo1"}
SK = {"west": "west", "volga": "volga", "interfluve": "interfluve", "ural": "ural", "northeast": "northeast",
      "komsomolets": "komsomolets", "open_shelf": "openshelf", "overall": "overall", "all": "all"}
TID = {"0": "land", "40": "predicted", "44": "sounding", "70": "pregenerated"}


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
    m[f"rev2.d.{key}"] = rec


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _csv(path: Path):
    if not path.exists():
        return None
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _b(x) -> bool:
    return str(x).strip().lower() in ("true", "1")


def _num(s: str) -> str:
    """-28.5 -> m28p5 (a key-safe level)."""
    v = float(s)
    return ("m" if v < 0 else "") + f"{abs(v):.1f}".replace(".", "p")


# --- item 4 ---------------------------------------------------------------------------------

def vertical(m):
    src = f"{REL}/vertical_equivalents_table.csv"
    note = "A19 item 4: median over evaluable levels (A18 item 7, shape variant)"
    for r in _csv(D / "vertical_equivalents_table.csv") or []:
        s, p = SK[r["sector"]], PK[r["product"]]
        _put(m, f"vert.{s}.{p}.Dg_m", _f(r["D_g_median_m"]), "m", src, note + "; D*g")
        _put(m, f"vert.{s}.{p}.ratio", _f(r["ratio_D_to_Q"]), "", src, "median D*g / median Q*g", decimals=1)
        _put(m, f"vert.{s}.Qg_m", _f(r["Q_g_median_m"]), "m", src, note + "; Q*g")
        _put(m, f"vert.{s}.n_levels", int(r["n_levels"]), "levels", src)
    d = _json(D / "vertical_equivalents_summary.json")
    if d:
        src = f"{REL}/vertical_equivalents_summary.json"
        wi, va = d["west_interfluve"], d["validated_four_sectors"]
        n4 = "the four grids that mispredict in the validated sectors"
        _put(m, "vert.westinterfluve.four_Dg_min_m", wi["mispredicting_four_D_g_min_m"], "m", src, n4)
        _put(m, "vert.westinterfluve.four_Dg_max_m", wi["mispredicting_four_D_g_max_m"], "m", src, n4)
        _put(m, "vert.westinterfluve.Qg_min_m", wi["Q_g_min_m"], "m", src)
        _put(m, "vert.westinterfluve.Qg_max_m", wi["Q_g_max_m"], "m", src)
        _put(m, "vert.westinterfluve.etopo2022_Dg_min_m", wi["etopo2022_D_g_min_m"], "m", src)
        _put(m, "vert.westinterfluve.etopo2022_Dg_max_m", wi["etopo2022_D_g_max_m"], "m", src)
        _put(m, "vert.validated.four_Dg_min_m", va["mispredicting_four_D_g_min_m"], "m", src, n4)
        _put(m, "vert.validated.four_Dg_max_m", va["mispredicting_four_D_g_max_m"], "m", src, n4)
        _put(m, "vert.validated.Qg_min_m", va["Q_g_min_m"], "m", src)
        _put(m, "vert.validated.Qg_max_m", va["Q_g_max_m"], "m", src)


# --- item 6 ---------------------------------------------------------------------------------

def _subset_key(kind: str, label: str) -> str | None:
    if kind == "epoch":
        return label
    if kind == "band":
        a, b = re.findall(r"-?\d+\.\d+", label)
        return f"band_{_num(a)}_{_num(b)}"
    if kind == "epoch_shared_levels":
        e = label.split()[0]
        lo, hi = re.search(r"bins (\d+)-(\d+)", label).groups()
        return f"{e}_bins{lo}to{hi}"
    return None


def constants(m):
    rows = _csv(D / "constants" / "constants.csv")
    src = f"{REL}/constants/constants.csv"
    note = "A19 item 6a: D-070 area constant refitted on a subset of the training-block units"
    for r in rows or []:
        p = PK[r["product"]]
        if r["kind"] == "all":
            _put(m, f"const.{p}.all_m", _f(r["c_m"]), "m", src, "all 43 units (reproduces the primary constant)")
            continue
        k = _subset_key(r["kind"], r["subset"])
        if k is None:
            continue
        _put(m, f"const.{p}.{k}_m", _f(r["c_m"]), "m", src, f"{note}: {r['subset']}, {r['n_units']} units")
        if _b(r["at_search_edge"]):
            _put(m, f"const.{p}.{k}_at_edge", "yes", "", src)
    d = _json(D / "constants" / "summary.json")
    if d:
        src = f"{REL}/constants/summary.json"
        for prod, v in d["by_product"].items():
            p = PK[prod]
            _put(m, f"const.{p}.epoch_spread_m", v["epoch_spread_m"], "m", src, "max - min over E1, E2, E3")
            _put(m, f"const.{p}.band_spread_m", v["band_spread_m"], "m", src, "max - min over the 0.5 m bands")
            _put(m, f"const.{p}.band_slope", v["band_slope_m_per_m"], "m/m", src,
                 "least-squares slope of the band constant on the band mid-level", decimals=2)
        for k, v in d["across_products"].items():
            kind, label = k.split("|", 1)
            key = "all" if kind == "all" else _subset_key(kind, label)
            if key:
                _put(m, f"const.across.{key}.spread_m", v["spread_m"], "m", src, "max - min over the five grids")
                _put(m, f"const.across.{key}.min_m", v["min_m"], "m", src)
                _put(m, f"const.across.{key}.max_m", v["max_m"], "m", src)
        rec = d.get("record") or {}
        for prod, v in rec.items():
            _put(m, f"const.{PK[prod]}.primary_check_m", v.get("c_m"), "m", src, "all-unit refit = primary constant")


def correlations(m):
    d = _json(D / "signed_displacement_correlations.json")
    if not d:
        return
    src = f"{REL}/signed_displacement_correlations.json"
    note = "A19 item 6b: correlation over evaluable levels of the per-level signed median displacement"
    for sec, v in d["by_sector"].items():
        s = SK[sec]
        for k in ("pearson_median", "pearson_min", "pearson_max", "spearman_median", "spearman_min", "spearman_max",
                  "pearson_cross_lineage_median", "spearman_cross_lineage_median",
                  "pearson_mispredicting_pairs_median", "spearman_mispredicting_pairs_median"):
            _put(m, f"corr.{s}.{k}", v[k], "", src, note, decimals=2)
        _put(m, f"corr.{s}.n_levels", v["n_levels"], "levels", src)
        _put(m, f"corr.{s}.spread_sd_median_km", v["spread_sd_median_m"] / 1000, "km", src,
             "median over levels of the across-grid SD of the signed median displacement")
        _put(m, f"corr.{s}.spread_sd_four_median_km", v["spread_sd_mispredicting_four_median_m"] / 1000, "km", src,
             "as above, the four mispredicting grids")
        _put(m, f"corr.{s}.same_sign_pct", 100 * v["share_levels_all_same_sign"], "%", src,
             "levels where all five signed medians share a sign", decimals=0)
    c = _csv(D / "signed_displacement_correlations.csv")
    for r in c or []:
        s = SK[r["sector"]]
        _put(m, f"corr.{s}.{PK[r['a']]}_{PK[r['b']]}.pearson", _f(r["pearson_r"]), "", f"{REL}/signed_displacement_correlations.csv",
             decimals=2)
        _put(m, f"corr.{s}.{PK[r['a']]}_{PK[r['b']]}.spearman", _f(r["spearman_rho"]), "",
             f"{REL}/signed_displacement_correlations.csv", decimals=2)


# --- item 10 --------------------------------------------------------------------------------

def obsbias(m):
    rows = _csv(D / "obsbias" / "obs_bias_by_sector.csv")
    src = f"{REL}/obsbias/obs_bias_by_sector.csv"
    note = ("A19 item 10: waterline DEM minus ICESat-2 on the H2 point set over all 50 km blocks; block bootstrap "
            "95% interval (B = 2,000, seed 20261023); descriptive")
    for r in rows or []:
        if int(float(r["n_points"] or 0)) == 0:
            continue
        b = f"obs.{r['source'].lower()}.{SK[r['sector']]}"
        _put(m, f"{b}.n_points", int(float(r["n_points"])), "points", src)
        _put(m, f"{b}.n_blocks", int(float(r["n_blocks"])), "blocks", src)
        for sp in ("train", "calib", "test"):
            if r.get(f"n_blocks_{sp}") not in (None, ""):
                _put(m, f"{b}.n_blocks_{sp}", int(float(r[f"n_blocks_{sp}"])), "blocks", src)
        for k in ("mean", "median", "nmad", "rmse"):
            _put(m, f"{b}.{k}_m", _f(r[k]), "m", src, note)
            _put(m, f"{b}.{k}_lo_m", _f(r[f"{k}_lo"]), "m", src, "2.5th percentile of the block bootstrap")
            _put(m, f"{b}.{k}_hi_m", _f(r[f"{k}_hi"]), "m", src, "97.5th percentile of the block bootstrap")
        for k in ("mean", "median"):
            _put(m, f"{b}.{k}_ci_inside_0p5", "yes" if _b(r[f"{k}_ci_inside_0p5"]) else "no", "", src,
                 "the 95% interval lies within +-0.5 m")
            _put(m, f"{b}.{k}_ci_outside_0p5", "yes" if _b(r[f"{k}_ci_excludes_0p5_band"]) else "no", "", src,
                 "the 95% interval lies wholly outside +-0.5 m")


# --- item 12 --------------------------------------------------------------------------------

def changepoint(m):
    d = _json(D / "etopo2022_changepoint.json")
    if not d:
        return
    src = f"{REL}/etopo2022_changepoint.json"
    for label, key in (("all_sectors", "cp.etopo2022"), ("validated_sectors", "cp.etopo2022.validated")):
        v = d.get(label)
        if not v:
            continue
        note = f"A19 item 12: one mean shift in ln(D/Q), {label.replace('_', ' ')} pooled"
        _put(m, f"{key}.edge_m", v["edge_m"], "m", src, note, decimals=1)
        _put(m, f"{key}.dq_geomean_below", v["DQ_geomean_below"], "", src, "exp(mean ln D/Q) below the edge", decimals=1)
        _put(m, f"{key}.dq_geomean_above", v["DQ_geomean_above"], "", src, "exp(mean ln D/Q) above the edge", decimals=2)
        _put(m, f"{key}.dq_median_below", v["DQ_median_below"], "", src, decimals=1)
        _put(m, f"{key}.dq_median_above", v["DQ_median_above"], "", src, decimals=2)
        _put(m, f"{key}.share_d_gt_q_below_pct", 100 * v["share_D_gt_Q_below"], "%", src, decimals=0)
        _put(m, f"{key}.share_d_gt_q_above_pct", 100 * v["share_D_gt_Q_above"], "%", src, decimals=0)
        _put(m, f"{key}.r2", v["r2"], "", src, "1 - RSS / RSS without a break", decimals=2)
        _put(m, f"{key}.n_levels_below", v["n_levels_below"], "levels", src)
        _put(m, f"{key}.n_levels_above", v["n_levels_above"], "levels", src)
        _put(m, f"{key}.n_rows", v["n"], "sector-levels", src)
        _put(m, f"{key}.rss", v["rss"], "", src, decimals=1)
        _put(m, f"{key}.rss_at_m27p7", v.get("rss_at_m27p7"), "", src, "RSS with the break at -27.7 m", decimals=1)


# --- item 14 --------------------------------------------------------------------------------

def sources(m):
    rows = _csv(D / "sources" / "tid_shares.csv")
    src = f"{REL}/sources/tid_shares.csv"
    for r in rows or []:
        if r["grid"] != "gebco_2026":
            continue
        code = r["tid"].split()[0]
        if code not in TID:
            continue
        reg = "domain" if r["region"] == "domain" else "band"
        sec = SK.get(r["sector"])
        if sec is None:
            continue
        key = f"tid.{reg}.{TID[code]}_pct" if sec == "all" else f"tid.{reg}.{sec}.{TID[code]}_pct"
        _put(m, key, 100 * float(r["share"]), "%", src, f"A19 item 14: GEBCO_2026 TID {r['tid']} share ({r['unit']})",
             decimals=1)
    d = _json(D / "sources" / "identical_cells.json")
    if d:
        src = f"{REL}/sources/identical_cells.json"
        g = d["gebco2026_eq_gebco2022"]
        _put(m, "same.g26_g22.domain_pct", 100 * g["domain"]["share_equal"], "%", src,
             "domain cells where GEBCO_2026 equals GEBCO_2022", decimals=1)
        _put(m, "same.g26_g22.band_pct", 100 * g["emerged_band"]["share_equal"], "%", src,
             "emerged-band cells (250 m) where GEBCO_2026 equals GEBCO_2022", decimals=1)
        for s, v in g["emerged_band"]["by_sector"].items():
            _put(m, f"same.g26_g22.band.{SK[s]}_pct", 100 * v, "%", src, decimals=1)
        _put(m, "same.g26_g22.tid_domain_pct", 100 * g["tid_identical_domain_share"], "%", src,
             "domain cells with the same TID in both", decimals=1)
        e = d["etopo2022_eq_gebco2022"]
        _put(m, "same.etopo_g22.lake_domain_pct", 100 * e["lake_depth_cells_domain"]["share_equal"], "%", src,
             "ETOPO 2022 lake-depth cells (source 11) in the domain equal to GEBCO_2022", decimals=1)
        _put(m, "same.etopo_g22.lake_domain_n", e["lake_depth_cells_domain"]["n_cells"], "cells", src)
        _put(m, "same.etopo_g22.lake_band_pct", 100 * e["lake_depth_cells_emerged_band"]["share_equal"], "%", src,
             "as above, emerged-band cells", decimals=1)
        _put(m, "same.etopo_g22.fabdem_domain_pct", 100 * e["fabdem_cells_domain"]["share_equal"], "%", src, decimals=1)
        _put(m, "same.etopo_g22.all_domain_pct", 100 * e["all_domain_cells"]["share_equal"], "%", src, decimals=1)
    s = _json(D / "sources" / "summary.json")
    if s:
        src = f"{REL}/sources/summary.json"
        gs = s["great_salt_lake"]["subset_tid_shares"]
        _put(m, "tid.gsl.land_pct", 100 * float(gs.get("0 Land", 0.0)), "%", src,
             "GEBCO_2026 TID 0 share over the Great Salt Lake subset (hence its emerged band)", decimals=0)
        _put(m, "tid.domain_cells", s["domain_cells"], "cells", src)
        _put(m, "tid.band_cells", s["band_cells_250m"], "cells", src)
    lin = _json(D / "verdicts_by_lineage.json")
    if lin:
        src = f"{REL}/verdicts_by_lineage.json"
        for v in lin["lineages"]:
            k = v["lineage"]
            _put(m, f"lineage.{k}.n_products", v["n_products"], "grids", src)
            _put(m, f"lineage.{k}.validated_all_mispredict", v["validated_sectors_all_members_mispredict"], "sectors",
                 src, "validated sectors where every grid of the lineage mispredicts")
            _put(m, f"lineage.{k}.validated_any_mispredicts", v["validated_sectors_any_member_mispredicts"], "sectors",
                 src, "validated sectors where at least one grid of the lineage mispredicts")
            _put(m, f"lineage.{k}.members_block_significant", v["members_block_test_significant"], "grids", src)


def add(m: dict) -> None:
    vertical(m)
    constants(m)
    correlations(m)
    obsbias(m)
    changepoint(m)
    sources(m)
