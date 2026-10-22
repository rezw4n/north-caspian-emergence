"""Metrics of revision workstream G (PREREG Amendment 18 item 11; D-128 to D-130): Great Salt
Lake, H2 and ETOPO 2022. Loaded by tools/build_metrics.revision_metrics, which calls
add(m). Every key starts with ``rev.g.``; each record is {"value", "unit", "source",
"note"?}, as put() in tools/build_metrics.py. A missing input file adds nothing.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
G = RES / "revision" / "g"
PRODUCT_KEY = {"gebco_2026": "gebco2026", "gebco_2022": "gebco2022", "etopo_2022": "etopo2022", "etopo1": "etopo1",
               "srtm15p_v2.7": "srtm15p"}


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
    m[f"rev.g.{key}"] = rec


def _json(rel: str):
    p = G / rel
    return (json.loads(p.read_text(encoding="utf-8")), f"results/revision/g/{rel}") if p.exists() else (None, None)


def _csv(rel: str):
    p = G / rel
    return (pd.read_csv(p), f"results/revision/g/{rel}") if p.exists() else (None, None)


def _r(x, d=3):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else round(float(x), d)


# --- Great Salt Lake ---------------------------------------------------------------------

def gsl(m: dict) -> None:
    s, src = _json("gsl_sources.json")
    if s:
        note = "exploratory (A18 item 11; D-128); emerged band of the arm, nearest native cell"
        for arm, r in s["arms"].items():
            _put(m, f"gsl.{arm}.band_km2", r["band_km2"], "km2", src, note)
            for p, x in r["products"].items():
                k = f"gsl.{arm}.{PRODUCT_KEY[p]}"
                _put(m, f"{k}.modal_share_pct", _r(100 * x["modal_share"], 1), "%", src,
                     "share of band cells at the product's modal height (a flat surface)")
                _put(m, f"{k}.modal_height_navd88_m", _r(x["modal_height_navd88_m"], 2), "m", src)
                _put(m, f"{k}.share_below_record_low_pct", _r(100 * x["share_below_record_low_gauge"], 1), "%", src,
                     "band cells below the arm's record-low gauge level")
                gg = x["gauge_during_land_source_navd88"]
                _put(m, f"{k}.gauge_during_land_source_mean_m", _r(gg["mean_m"], 2), "m", src, x["land_source"])
                cs = x.get("code_shares")
                if isinstance(cs, dict):
                    for code, share in cs.items():
                        _put(m, f"{k}.code_{code}_pct", _r(100 * share, 1), "%", src, x["code_meaning"].get(code))
                    _put(m, f"{k}.any_sounding_code", "yes" if x.get("any_sounding_code") else "no", "text", src)
            _put(m, f"gsl.{arm}.gauge_feb2000_navd88_m",
                 _r(r["products"]["gebco_2026"]["gauge_during_land_source_navd88"]["mean_m"], 2), "m", src,
                 "mean gauge level 11-22 February 2000 (SRTM acquisition)")
            _put(m, f"gsl.{arm}.gauge_2011_2015_navd88_min_m",
                 _r(r["products"]["etopo_2022"]["gauge_during_land_source_navd88"]["min_m"], 2), "m", src)
            _put(m, f"gsl.{arm}.gauge_2011_2015_navd88_max_m",
                 _r(r["products"]["etopo_2022"]["gauge_during_land_source_navd88"]["max_m"], 2), "m", src)
    t, src = _csv("gsl_control_levels.csv")
    if t is not None:
        note = "registered rule restricted to the levels at which the 3DEP substitute is evaluable (A18 item 11; D-128)"
        for _, r in t.iterrows():
            p = "control" if r["product"].startswith("dep3") else PRODUCT_KEY[r["product"]]
            k = f"gsl.control_levels.{p}.{r['variant']}.{r['arm']}"
            _put(m, f"{k}.n_exceeding", int(r["n_exceeding"]), "levels", src, note)
            _put(m, f"{k}.n_evaluable", int(r["n_evaluable"]), "levels", src, note)
            _put(m, f"{k}.median_displacement_m", _r(r["median_displacement_m"], 0), "m", src, note)
            _put(m, f"{k}.mispredicts", "yes" if r["mispredicts_at_control_levels"] else "no", "text", src, note)
            if isinstance(r["mispredicts_pooled_floor_at_control_levels"], (bool,)) or str(
                    r["mispredicts_pooled_floor_at_control_levels"]) in ("True", "False"):
                _put(m, f"{k}.mispredicts_pooled_floor",
                     "yes" if str(r["mispredicts_pooled_floor_at_control_levels"]) == "True" else "no", "text", src, note)
    j, src = _json("gsl_control_levels.json")
    if j:
        v = j["verdict_at_control_levels"]
        _put(m, "gsl.control_levels.verdict", v["verdict"], "text", src, "exploratory; the registered verdict stands")
        _put(m, "gsl.control_levels.n_products_mispredict", v["n_mispredict"], "products", src)
        for p, arms in j.get("per_level_ratio_product_to_control", {}).items():
            for arm, x in arms.items():
                _put(m, f"gsl.control_levels.ratio.{PRODUCT_KEY[p]}.{arm}.median", x["median"], "ratio", src,
                     "per-level product median displacement over the control's, shape variant")
    r, src = _json("root2023_retrieval.json")
    if r:
        st = sorted({x["status"] for x in r["requests"] if "sciencebase" in x["url"]})
        _put(m, "gsl.root2023.sciencebase_status", ",".join(str(s) for s in st), "HTTP status", src,
             "every scripted request met a Cloudflare browser challenge (2026-10-22)")
        _put(m, "gsl.root2023.served_to_script", "yes" if r["any_file_served_to_a_script"] else "no", "text", src)
    d, src = _json("dep3_provenance.json")
    if d:
        note = "3DEP 1/3 arc-second source footprints over the arm's emerged band (D-128 addendum)"
        for snap, sk in (("current", "current"), ("as_of_20220504", "asof2022")):
            v = d.get("band_shares", {}).get("snapshots", {}).get(snap)
            if not v:
                continue
            for arm in ("north", "south"):
                a = v[arm]
                y2016 = sum(r["share"] for r in a["by_source"]
                            if "2016" in r["source"] and ("GreatSaltLake" in r["source"] or "CentralUT_Southern" in r["source"]))
                _put(m, f"gsl.dep3.{sk}.{arm}.lidar2016_pct", _r(100 * y2016, 1), "%", src, note)
                _put(m, f"gsl.dep3.{sk}.{arm}.lidar_pct", _r(100 * a["share_lidar_source"], 1), "%", src, note)
                _put(m, f"gsl.dep3.{sk}.{arm}.non_lidar_pct", _r(100 * a["share_non_lidar_source"], 1), "%", src, note)
        _put(m, "gsl.dep3.lidar2016.collect_start", "2016-09-03", "date", src, "project report, PDF p. 5")
        _put(m, "gsl.dep3.lidar2016.collect_end", "2016-11-30", "date", src, "project report, PDF p. 5 (WESM: 2016-11-18)")


# --- H2 --------------------------------------------------------------------------------

def h2(m: dict) -> None:
    j, src = _json("h2/split_draws.json")
    if j:
        note = "D-104 reassignments of the 50 km blocks, ATL08 only; H2 rule re-applied (D-129)"
        _put(m, "h2.split.n_seeds", j["n_seeds"], "draws", src, note)
        _put(m, "h2.split.n_not_supported", j["n_not_supported_matched"], "draws", src, note)
        _put(m, "h2.split.n_supported", j["n_supported_matched"], "draws", src, note)
        _put(m, "h2.split.n_lower_dem_nmad", j["n_seeds_lower_dem_nmad_than_registered"], "draws", src, note)
        _put(m, "h2.split.n_higher_coverage", j["n_seeds_higher_coverage_than_registered"], "draws", src, note)
    for source in ("ATL06", "ATL08"):
        s = source.lower()
        t, src = _csv(f"h2/influence_summary_{source}.csv")
        if t is not None:
            note = "leave-one-out paired difference DEM - product, matched variant, registered fits (D-129)"
            for _, r in t[t.variant == "matched"].iterrows():
                k = f"h2.influence.{s}.{r['left_out_unit']}.{PRODUCT_KEY[r['product']]}.{r['metric']}"
                _put(m, f"{k}.diff_full_m", _r(r["diff_full"]), "m", src, note)
                _put(m, f"{k}.diff_min_m", _r(r["diff_min"]), "m", src, note)
                _put(m, f"{k}.diff_max_m", _r(r["diff_max"]), "m", src, note)
                _put(m, f"{k}.sign_change", "yes" if r["sign_change"] else "no", "text", src, note)
                _put(m, f"{k}.most_influential", str(r["most_influential"]), "text", src, note)
                if r["left_out_unit"] == "block":
                    _put(m, f"{k}.jackknife_se_m", _r(r["jackknife_se"]), "m", src, "jackknife over test blocks")
            for unit in ("block", "sector"):
                for mt in ("nmad", "rmse"):
                    q = t[(t.variant == "matched") & (t.left_out_unit == unit) & (t.metric == mt)]
                    if len(q):
                        _put(m, f"h2.influence.{s}.{unit}.{mt}.n_products_sign_change", int(q["sign_change"].sum()),
                             "products", src)
                        _put(m, f"h2.influence.{s}.{unit}.{mt}.min_m", _r(q["diff_min"].min()), "m", src)
                        _put(m, f"h2.influence.{s}.{unit}.{mt}.max_m", _r(q["diff_max"].max()), "m", src)
        t, src = _csv(f"h2/mondrian_{source}.csv")
        if t is not None:
            note = "registered calibration (Amendment 5); blocks counted here (D-129)"
            for _, r in t.iterrows():
                k = f"h2.mondrian.{s}.{r['sector']}"
                _put(m, f"{k}.q_hat", _r(r["q_hat"]), "scale units", src, note)
                _put(m, f"{k}.n_points", int(r["n_points"]), "points", src, note)
                _put(m, f"{k}.n_blocks", int(r["n_blocks"]), "blocks", src, note)
                _put(m, f"{k}.used_pooled", "yes" if bool(r["used_pooled"]) else "no", "text", src, note)
        t, src = _csv(f"h2/blocksizes_common_accuracy_{source}.csv")
        if t is not None:
            note = "points held out of fitting at 12.5, 25 and 50 km; each size's own fits (D-129)"
            for size, g in t.groupby("block_size_m"):
                sk = {12500.0: "12p5km", 25000.0: "25km", 50000.0: "50km"}[float(size)]
                d = g[g.model == "dem"].iloc[0]
                _put(m, f"h2.blocksizes.{s}.{sk}.dem.nmad_m", _r(d["nmad"]), "m", src, note)
                _put(m, f"h2.blocksizes.{s}.{sk}.dem.rmse_m", _r(d["rmse"]), "m", src, note)
                pm = g[g.model.str.endswith("|matched")]
                _put(m, f"h2.blocksizes.{s}.{sk}.products.rmse_min_m", _r(pm["rmse"].min()), "m", src, note)
                _put(m, f"h2.blocksizes.{s}.{sk}.products.rmse_max_m", _r(pm["rmse"].max()), "m", src, note)
                _put(m, f"h2.blocksizes.{s}.{sk}.products.nmad_min_m", _r(pm["nmad"].min()), "m", src, note)
                _put(m, f"h2.blocksizes.{s}.{sk}.products.nmad_max_m", _r(pm["nmad"].max()), "m", src, note)
                _put(m, f"h2.blocksizes.{s}.{sk}.n_blocks", int(d["n_blocks"]), "blocks", src, note)
            _put(m, f"h2.blocksizes.{s}.n_points", int(t["n"].iloc[0]), "points", src, note)
        j, src = _json(f"h2/blocksizes_common_{source}.json")
        if j:
            for size, v in j["rule_outcome_descriptive"].get(source, {}).items():
                sk = {"12500": "12p5km", "25000": "25km", "50000": "50km"}[size]
                _put(m, f"h2.blocksizes.{s}.{sk}.rule_matched", v["matched"], "text", src, "descriptive")
    t, src = _csv("h2/rmse_beside_nmad.csv")
    if t is not None:
        note = "registered accuracy tables, test blocks (R1 minor 9)"
        o = t[(t.scope == "overall")]
        for _, r in o.iterrows():
            mk = "dem" if r["model"] == "dem" else PRODUCT_KEY[r["model"].split("|")[0]] + "." + r["model"].split("|")[1]
            _put(m, f"h2.rmse.{r['source'].lower()}.{mk}.rmse_m", _r(r["rmse_m"], 2), "m", src, note)
            _put(m, f"h2.rmse.{r['source'].lower()}.{mk}.nmad_m", _r(r["nmad_m"], 2), "m", src, note)
        for source in ("ATL06", "ATL08"):
            q = o[(o.source == source) & o.model.str.endswith("|matched")]
            _put(m, f"h2.rmse.{source.lower()}.products_matched.rmse_min_m", _r(q["rmse_m"].min(), 2), "m", src, note)
            _put(m, f"h2.rmse.{source.lower()}.products_matched.rmse_max_m", _r(q["rmse_m"].max(), 2), "m", src, note)
    j, src = _json("h2/intervals.json")
    if j:
        for source, r in j.items():
            s = source.lower()
            c = r["coverage_overall"]
            for k in ("coverage", "ci_low", "ci_high", "boot_low", "boot_high"):
                _put(m, f"h2.coverage.{s}.{k}_pct", _r(100 * c[k], 1), "%", src,
                     "ci = Clopper-Pearson (points as independent); boot = test-block bootstrap")
            _put(m, f"h2.intervals.{s}.not_estimable_scopes", ", ".join(r["paired_scopes_not_estimable"]) or "none",
                 "text", src, "scopes resting on one test block")
    for source in ("ATL06", "ATL08"):
        s = source.lower()
        j, src = _json(f"h2_dryness/{source}_prepare.json")
        if j:
            note = "independent dryness test, observed E3 extent of the acquisition's level bin (D-129)"
            for k, v in j["counts"].items():
                _put(m, f"h2.dryness.{s}.{k}", int(v), "count", src, note)
            for k, v in j["agreement_in_band_testable"].items():
                _put(m, f"h2.dryness.{s}.agree.{k}", int(v), "points", src, note)
            a = j["agreement_in_band_testable"]
            tot = sum(a.values())
            if tot:
                _put(m, f"h2.dryness.{s}.agree.share_same_pct",
                     _r(100 * (a["dem_dry_no__observed_dry_no"] + a["dem_dry_yes__observed_dry_yes"]) / tot, 1), "%", src)
        t, src = _csv(f"h2_dryness/{source}_accuracy.csv")
        if t is not None:
            o = t[t.scope == "overall"].set_index("model")
            note = "independent dryness test (D-129), test blocks"
            _put(m, f"h2.dryness.{s}.dem.nmad_m", _r(o.loc["dem", "nmad"]), "m", src, note)
            _put(m, f"h2.dryness.{s}.dem.rmse_m", _r(o.loc["dem", "rmse"]), "m", src, note)
            _put(m, f"h2.dryness.{s}.dem.bias_m", _r(o.loc["dem", "bias"]), "m", src, note)
            pm = o[o.index.str.endswith("|matched")]
            _put(m, f"h2.dryness.{s}.products.nmad_min_m", _r(pm["nmad"].min()), "m", src, note)
            _put(m, f"h2.dryness.{s}.products.nmad_max_m", _r(pm["nmad"].max()), "m", src, note)
            _put(m, f"h2.dryness.{s}.products.rmse_min_m", _r(pm["rmse"].min()), "m", src, note)
            _put(m, f"h2.dryness.{s}.products.rmse_max_m", _r(pm["rmse"].max()), "m", src, note)
            _put(m, f"h2.dryness.{s}.n_test_points", int(o.loc["dem", "n"]), "points", src, note)
            _put(m, f"h2.dryness.{s}.n_test_blocks", int(o.loc["dem", "n_blocks"]), "blocks", src, note)
        t, src = _csv(f"h2_dryness/{source}_coverage.csv")
        if t is not None:
            c = t[t.scope == "overall"].iloc[0]
            _put(m, f"h2.dryness.{s}.coverage_pct", _r(100 * c["coverage"], 1), "%", src)
    j, src = _json("h2_dryness/h2_decision.json")
    if j:
        dec = j["decision_both_sources"]
        if isinstance(dec, dict):
            for v, x in dec.items():
                _put(m, f"h2.dryness.decision.{v}", x, "text", src, "descriptive; the registered verdict stands")
        for source, vv in j["by_source"].items():
            for v, x in vv.items():
                _put(m, f"h2.dryness.{source.lower()}.rule.{v}", x["overall"], "text", src, "descriptive")


# --- ETOPO 2022 ------------------------------------------------------------------------

def etopo(m: dict) -> None:
    t, src = _csv("etopo/etopo_minus_gebco2022_by_band_source.csv")
    if t is not None:
        note = "ETOPO 2022 minus GEBCO_2022 at the same 15-arcsec cell, emerged-band cells (D-130)"
        bk = {"(-26.3, -25.5]": "above_m26p3", "(-27.0, -26.3]": "m27p0_m26p3", "(-27.7, -27.0]": "m27p7_m27p0",
              "(-28.3, -27.7]": "m28p3_m27p7", "(-29.0, -28.3]": "below_m28p3"}
        for _, r in t.iterrows():
            if r["band"] not in bk:
                continue
            k = f"etopo.heights.{bk[r['band']]}.{r['source']}"
            _put(m, f"{k}.share_equal_pct", _r(100 * r["share_equal_0p01m"], 1), "%", src, note)
            _put(m, f"{k}.median_diff_m", _r(r["diff_median_m"], 2), "m", src, note)
            _put(m, f"{k}.n_cells", int(r["n_cells_250m"]), "cells", src, note)
    j, src = _json("etopo/lake_clamp.json")
    if j:
        for k, v in j.get("metrics", {}).items():
            _put(m, f"etopo.lake_clamp.{k}", v["value"], v["unit"], src, v.get("note"))
    j, src = _json("etopo/compare.json")
    if j:
        note = "distances split by the ETOPO 2022 source of the predicted shoreline cell; matched levels (D-130)"
        for p, r in j["products"].items():
            pk = PRODUCT_KEY[p]
            _put(m, f"etopo.by_source.{pk}.n_levels_matched", r["n_levels_matched"], "sector-levels", src, note)
            a = r.get("all_matched")
            if a:
                _put(m, f"etopo.by_source.{pk}.median_fabdem_m", _r(a["median_fabdem_m"], 0), "m", src, note)
                _put(m, f"etopo.by_source.{pk}.median_lake_depth_m", _r(a["median_lake_depth_m"], 0), "m", src, note)
                _put(m, f"etopo.by_source.{pk}.share_levels_fabdem_smaller_pct",
                     _r(100 * a["share_levels_fabdem_smaller"], 0), "%", src, note)
            for sec, x in r["by_sector"].items():
                _put(m, f"etopo.by_source.{pk}.{sec}.median_fabdem_m", _r(x["median_fabdem_m"], 0), "m", src, note)
                _put(m, f"etopo.by_source.{pk}.{sec}.median_lake_depth_m", _r(x["median_lake_depth_m"], 0), "m", src, note)
                _put(m, f"etopo.by_source.{pk}.{sec}.n_matched_levels", x["n_matched_levels"], "levels", src, note)
        for p, bands in j.get("shoreline_fabdem_share_by_band", {}).items():
            for b, v in bands.items():
                bk = b.replace("(", "").replace("]", "").replace(", ", "_").replace("-", "m").replace(".", "p")
                _put(m, f"etopo.shoreline_fabdem_share.{PRODUCT_KEY[p]}.{bk}_pct", _r(100 * v, 1), "%", src,
                     "share of predicted waterline cells in ETOPO 2022 FABDEM cells")
    t, src = _csv("etopo/etopo_shoreline_edge_classes.csv")
    if t is not None:
        note = "ETOPO 2022 predicted waterline cells: FABDEM, lake depth at the lake-mask edge, lake depth inside (D-130 addendum)"
        for _, r in t[t.sector == "all"].iterrows():
            bk = r["band"].replace("(", "").replace("]", "").replace(", ", "_").replace("-", "m").replace(".", "p")
            _put(m, f"etopo.edge.{bk}.{r['class']}.share_pct", _r(100 * r["share"], 1), "%", src, note)
            _put(m, f"etopo.edge.{bk}.{r['class']}.median_abs_m", _r(r["median_abs_m"], 0), "m", src, note)
    j, src = _json("etopo/bands.json")
    if j:
        note = "D-087 level bands; registered per-level floor and D (shape variant) (D-130)"
        for p, x in j["below_-27.7_sectors_majority_exceeds"].items():
            _put(m, f"etopo.bands.below_m27p7.{PRODUCT_KEY[p]}.n_sectors_majority", x["n_sectors"], "sectors", src, note)
            _put(m, f"etopo.bands.below_m27p7.{PRODUCT_KEY[p]}.not_majority", ", ".join(x["not_majority"]) or "none",
                 "text", src, note)
        for p, x in j["above_-27.7_sectors_majority_exceeds"].items():
            _put(m, f"etopo.bands.above_m27p7.{PRODUCT_KEY[p]}.n_sectors_majority", x["n_sectors"], "sectors", src, note)
        for b, v in j["floor_by_band_median_over_sectors_m"].items():
            bk = b.replace("(", "").replace("]", "").replace(", ", "_").replace("-", "m").replace(".", "p")
            _put(m, f"etopo.bands.floor_median_m.{bk}", _r(v, 0), "m", src, note)
    t, src = _csv("etopo/above_below_by_sector.csv")
    if t is not None:
        note = "levels above and below -27.7 m within each sector, shape variant (D-130)"
        for _, r in t.iterrows():
            k = f"etopo.bands.{r['part'].replace('-27.7', 'm27p7')}.{PRODUCT_KEY[r['product']]}.{r['sector']}"
            _put(m, f"{k}.n_exceeding", int(r["n_exceeding"]), "levels", src, note)
            _put(m, f"{k}.n_evaluable", int(r["n_evaluable"]), "levels", src, note)
            _put(m, f"{k}.median_D_km", _r(r["median_D_m"] / 1000, 2), "km", src, note)
            _put(m, f"{k}.median_DQ", _r(r["median_DQ"], 2), "ratio", src, note)
    t, src = _csv("etopo/icesat2_by_etopo_source.csv")
    if t is not None:
        note = "test points by ETOPO 2022 source ID (nearest cell); matched variants (D-130)"
        for _, r in t.iterrows():
            k = f"etopo.is2.{r['source'].lower()}.{r['scope']}.{r['etopo_source']}"
            _put(m, f"{k}.share_pct", _r(100 * r["share_of_scope"], 1), "%", src, note)
            _put(m, f"{k}.n_points", int(r["n_points"]), "points", src, note)
            for mod in ("dem", "etopo_2022", "gebco_2026"):
                _put(m, f"{k}.{PRODUCT_KEY.get(mod, mod)}.nmad_m", _r(r[f"{mod}_nmad"]), "m", src, note)
    j, src = _json("etopo/gedi.json")
    if j:
        note = "GEDI test points by ETOPO 2022 source; block bootstrap B = 2000, seed 20261022 (D-130)"
        for r in j["bootstrap"]:
            k = f"etopo.gedi.{PRODUCT_KEY.get(r['series'], r['series'].replace(' minus ', '_minus_'))}.{r['statistic']}"
            _put(m, f"{k}_m", _r(r["value"]), "m", src, note)
            _put(m, f"{k}_ci_low_m", _r(r["ci_low"]), "m", src, note)
            _put(m, f"{k}_ci_high_m", _r(r["ci_high"]), "m", src, note)
        _put(m, "etopo.gedi.n_blocks", j["n_blocks"], "blocks", src, note)
        wb = j["within_block"]
        _put(m, "etopo.gedi.within_block.n_blocks", wb["n_blocks"], "blocks", src, "blocks with >= 50 points per source")
        for k in ("etopo_median_diff_lake_minus_fabdem_m", "dem_median_diff_lake_minus_fabdem_m"):
            if k in wb:
                _put(m, f"etopo.gedi.within_block.{k}", _r(wb[k]), "m", src)
        for k in ("n_blocks_etopo_fabdem_lower", "n_blocks_dem_fabdem_lower"):
            if k in wb:
                _put(m, f"etopo.gedi.within_block.{k}", wb[k], "blocks", src)


def add(m: dict) -> None:
    gsl(m)
    h2(m)
    etopo(m)
