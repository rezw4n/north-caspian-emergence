"""Metrics of the second-round near-term analyses (PREREG Amendment 19 items 1-3; D-146),
loaded by tools/build_metrics.revision_metrics. Keys start with ``rev2.n.``; each record
is {"value", "unit", "source", "note"?} as build_metrics.put writes them.

Inputs: results/revision2/n/*.csv and *.json written by scripts/rev2_n_reref.py,
rev2_n_exposure.py, rev2_n_pathways.py and rev2_n_simulate.py. A missing table is
skipped (its keys are then absent).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "results" / "revision2" / "n"
SRC = "results/revision2/n/"
PKEY = {"gebco_2026": "gebco2026", "etopo_2022": "etopo2022", "srtm15p_v2.7": "srtm15p",
        "gebco_2022": "gebco2022", "etopo1": "etopo1"}
VAR = {"area_match": "area", "bins": "bins"}
SENS = {"main": "main", "L26+0.3": "Lp0p3", "L26-0.3": "Lm0p3"}
GRID = {"products": "products", "lineages_mean": "lineages", "lineages_gebco2022": "lineages_g22"}
NOTE_REREF = ("A19 item 1 (exploratory): observed composite 2026 sea S_obs; product re-referenced by its shape "
              "constant and c26 (area match at L2026 unless 'bins')")


def _put(m, key, value, unit, source, note=None):
    if value is None:
        return
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if math.isnan(value):
            return
        if math.isinf(value):
            value = "inf" if value > 0 else "-inf"
    elif isinstance(value, np.integer):
        value = int(value)
    elif isinstance(value, np.bool_):
        value = bool(value)
    rec = {"value": value, "unit": unit, "source": source}
    if note:
        rec["note"] = note
    m[f"rev2.n.{key}"] = rec


def dk(x: float) -> str:
    return f"{x:g}".replace(".", "p")


def _read(name):
    p = D / name
    if not p.exists():
        return None
    if p.suffix in (".csv", ".gz"):
        try:
            return pd.read_csv(p)
        except pd.errors.EmptyDataError:
            return None
    return json.loads(p.read_text(encoding="utf-8"))


def reref(m):
    k = _read("reref_constants.csv")
    cv = _read("reref_curves.csv")
    run = _read("run_reref.json")
    if run:
        src = SRC + "run_reref.json"
        _put(m, "reref.S_obs_km2", run["S_obs_domain_km2"], "km2", src,
             "composite observed 2026 sea in the domain (second D-123 addendum)")
        ch = run["checks"]
        _put(m, "reref.check.refit_at_shifted_level_same_T", all(ch["refit_at_shifted_level_same_T"].values()), "",
             src, "area-match c26 refitted at L2026 +- 0.3 m returns the same threshold for every grid")
        _put(m, "reref.check.bisection_equals_full_evaluation", all(ch["bisection_equals_full_evaluation"].values()),
             "", src)
        _put(m, "reref.check.band_rasters_vs_counts_max_abs_diff_km2", ch["band_rasters_vs_counts_max_abs_diff_km2"],
             "km2", src)
    if k is not None:
        src = SRC + "reref_constants.csv"
        for r in k[k["sensitivity"] == "main"].itertuples():
            v = VAR[r.variant]
            _put(m, f"reref.{v}.{PKEY[r.product]}.c26_m", r.c26_m, "m", src,
                 "second constant: area match to S_obs at L2026" if v == "area" else
                 "second constant: fitted to the three lowest audited bins (E3 12-14), summed |area error|")
            _put(m, f"reref.{v}.{PKEY[r.product]}.c_total_m", r.c_total_m, "m", src, "shape constant + c26")
            _put(m, f"reref.{v}.{PKEY[r.product]}.area_error_km2", r.area_error_domain_km2, "km2", src,
                 "product domain sea at L2026 minus |S_obs|")
        for v_, g in k[k["sensitivity"] == "main"].groupby("variant"):
            v = VAR[v_]
            for col, unit in (("c26_m", "m"), ("c_total_m", "m")):
                _put(m, f"reref.{v}.{col[:-2]}.min_m", g[col].min(), unit, src)
                _put(m, f"reref.{v}.{col[:-2]}.max_m", g[col].max(), unit, src)
    if cv is None:
        return
    src = SRC + "reref_curves.csv"
    S = float(cv[(cv["quantity"] == "S_obs") & (cv["region"] == "domain")]["km2"].iloc[0])
    for (var_, sens), g in cv.groupby(["variant", "sensitivity"]):
        v, s = VAR[var_], SENS[sens]
        e = g[(g["quantity"] == "E") & g["region"].isin(["domain", "validated"]) & (g["delta_m"] > 0)]
        for r in e.itertuples():
            if (v == "area" and s == "main") or r.region == "domain":
                _put(m, f"reref.{v}.{s}.{PKEY[r.product]}.E.{dk(r.delta_m)}.{r.region}_km2", r.km2, "km2", src,
                     NOTE_REREF if (v == "area" and s == "main") else None)
        for (reg, d), gg in e.groupby(["region", "delta_m"]):
            _put(m, f"reref.{v}.{s}.E.{dk(d)}.{reg}.min_km2", gg["km2"].min(), "km2", src)
            _put(m, f"reref.{v}.{s}.E.{dk(d)}.{reg}.max_km2", gg["km2"].max(), "km2", src)
            _put(m, f"reref.{v}.{s}.E.{dk(d)}.{reg}.ratio_max_min", gg["km2"].max() / gg["km2"].min()
                 if gg["km2"].min() > 0 else np.nan, "", src)
        mm = g[(g["quantity"] == "misfit") & (g["region"] == "domain")]
        for r in mm.itertuples():
            _put(m, f"reref.{v}.{s}.{PKEY[r.product]}.M_km2", r.km2, "km2", src,
                 "misfit: observed 2026 sea the re-referenced grid holds dry at L2026 (domain)")
            _put(m, f"reref.{v}.{s}.{PKEY[r.product]}.M_pct_of_S_obs", 100 * r.km2 / S, "%", src)
        _put(m, f"reref.{v}.{s}.M.min_km2", mm["km2"].min(), "km2", src)
        _put(m, f"reref.{v}.{s}.M.max_km2", mm["km2"].max(), "km2", src)
        _put(m, f"reref.{v}.{s}.M.min_pct_of_S_obs", 100 * mm["km2"].min() / S, "%", src)
        _put(m, f"reref.{v}.{s}.M.max_pct_of_S_obs", 100 * mm["km2"].max() / S, "%", src)
    e = cv[(cv["quantity"] == "E") & (cv["region"] == "domain") & (cv["delta_m"] > 0)]
    for d, g in e.groupby("delta_m"):
        _put(m, f"reref.allsens.E.{dk(d)}.domain.min_km2", g["km2"].min(), "km2", src,
             "over the five grids, both c26 fits and L2026 +- 0.3 m (c26 held)")
        _put(m, f"reref.allsens.E.{dk(d)}.domain.max_km2", g["km2"].max(), "km2", src,
             "over the five grids, both c26 fits and L2026 +- 0.3 m (c26 held)")
        a = g[g["variant"] == "area_match"]
        _put(m, f"reref.area.levelsens.E.{dk(d)}.domain.min_km2", a["km2"].min(), "km2", src,
             "area-match c26, L2026 and +- 0.3 m (c26 held), five grids")
        _put(m, f"reref.area.levelsens.E.{dk(d)}.domain.max_km2", a["km2"].max(), "km2", src)


def agreement60(m):
    a = _read("agreement_60m.csv")
    if a is None:
        return
    src = SRC + "agreement_60m.csv"
    for r in a.itertuples():
        k = f"agree60.{r.surface}.{dk(r.delta_m)}"
        _put(m, f"{k}.any_km2", r.any_km2, "km2", src, "area any of the five grids places in the first metre (60 m)")
        _put(m, f"{k}.all_km2", r.all_km2, "km2", src, "area all five grids place in the first metre (60 m)")
        _put(m, f"{k}.majority_km2", r.majority_3of5_km2, "km2", src, "three or more of five grids")
        _put(m, f"{k}.all_pct_of_any", 100 * r.ratio_all_over_any, "%", src)


EXP_LAYERS_SHARE = ("wdpa_protected_areas", "wdpa_protected_areas_noksrz", "wdpa_ramsar", "osm_wetlands")


def exposure(m):
    pol = _read("exposure2_polygons.csv")
    if pol is not None:
        src = SRC + "exposure2_polygons.csv"
        t = pol[pol["region"] == "domain"]
        for r in t[t["delta_m"].isin([1.0, 2.0])].itertuples():
            k = f"exp.{r.layer}.{r.surface}.{PKEY[r.product]}.{dk(r.delta_m)}"
            _put(m, f"{k}.km2", r.emerged_in_layer_km2, "km2", src, "emerged area inside the layer, domain cells only")
            _put(m, f"{k}.share", r.share, "", src, "share of the re-referenced increment E inside the layer")
        for (lay, surf, d), g in t.groupby(["layer", "surface", "delta_m"]):
            k = f"exp.{lay}.{surf}.{dk(d)}"
            _put(m, f"{k}.min_km2", g["emerged_in_layer_km2"].min(), "km2", src)
            _put(m, f"{k}.max_km2", g["emerged_in_layer_km2"].max(), "km2", src)
            _put(m, f"{k}.min_share", g["share"].min(), "", src)
            _put(m, f"{k}.max_share", g["share"].max(), "", src)
    fe = _read("exposure2_features.csv")
    if fe is not None:
        src = SRC + "exposure2_features.csv"
        t = fe[fe["region"] == "domain"]
        for r in t[t["delta_m"].isin([1.0, 2.0])].itertuples():
            sub = "" if r.subset == "all" else f".{r.subset}"
            k = f"exp.{r.layer}{sub}.{r.surface}.{PKEY[r.product]}.{dk(r.delta_m)}"
            unit = {"length": "km", "area": "km2"}.get(r.rule, "features")
            _put(m, f"{k}.amount", r.amount, unit, src)
        for (lay, sub, surf, d), g in t.groupby(["layer", "subset", "surface", "delta_m"]):
            s_ = "" if sub == "all" else f".{sub}"
            _put(m, f"exp.{lay}{s_}.{surf}.{dk(d)}.min", g["amount"].min(), "", src)
            _put(m, f"exp.{lay}{s_}.{surf}.{dk(d)}.max", g["amount"].max(), "", src)


def pathways(m):
    v = _read("pathways2_variance.csv")
    if v is not None:
        src = SRC + "pathways2_variance.csv"
        for r in v.itertuples():
            if not isinstance(getattr(r, "share_pathway", None), float) or not np.isfinite(r.share_pathway):
                continue
            sc = "area" if r.scale == "area_km2" else "log"
            k = f"var.{r.baseline}.{GRID[r.grid_factor]}.{r.region}.{r.year}.{sc}"
            note = "superseded A18 observed-shoreline estimate (Supplementary deviation)" \
                if r.baseline == "observed_a18_superseded" else None
            _put(m, f"{k}.pathway", r.share_pathway, "", src, note)
            _put(m, f"{k}.grid", r.share_grid, "", src, note)
            _put(m, f"{k}.interaction", r.share_interaction, "", src, note)
    rt = _read("product_ratios.csv")
    if rt is not None:
        src = SRC + "product_ratios.csv"
        for r in rt[rt["region"] == "domain"].itertuples():
            k = f"ratio.{r.baseline}.{dk(r.delta_m)}"
            _put(m, f"{k}.products", r.ratio_max_min_products, "", src, "largest over smallest domain E across the five grids")
            _put(m, f"{k}.lineages", r.ratio_max_min_lineages, "", src, "largest over smallest across the three lineage means")
            _put(m, f"{k}.max_product", r.max_product, "text", src)
            _put(m, f"{k}.min_product", r.min_product, "text", src)
    mt = _read("pathways2_matrix.csv")
    if mt is not None:
        src = SRC + "pathways2_matrix.csv"
        t = mt[(mt["region"] == "domain") & (mt["baseline"] == "observed_reref")]
        for r in t.itertuples():
            k = f"path.{r.pathway_id.lower().replace('-', '_')}.{r.year}.observed_reref.{PKEY[r.product]}_km2"
            _put(m, k, r.E_km2, "km2", src, r.note if isinstance(r.note, str) and r.note else None)
        for y, g in t.groupby("year"):
            _put(m, f"path.{y}.observed_reref.min_km2", g["E_km2"].min(), "km2", src)
            _put(m, f"path.{y}.observed_reref.max_km2", g["E_km2"].max(), "km2", src)


def simulation(m):
    s = _read("sim2_summary.csv")
    run = _read("run_simulate2.json")
    if run:
        src = SRC + "run_simulate2.json"
        ch = run["checks"]
        _put(m, "sim.check.d125_groups_max_abs_diff", ch["groups_max_abs_diff"], "", src,
             "D-125 group ratios (uncorrected, shifted) reproduced from the regenerated realisations")
        _put(m, "sim.check.d125_realisations_max_abs_diff_km2", ch["realisations_max_abs_diff_km2"], "km2", src)
        _put(m, "sim.n_fields", run["n_fields"], "fields", src)
        _put(m, "sim.flood_grid_m", run["flood_grid_m"], "m", src)
    if s is None:
        return
    src = SRC + "sim2_summary.csv"
    for r in s.itertuples():
        k = f"sim.{PKEY[r.product]}.{r.quantity}"
        if r.quantity.startswith("agreement"):
            for f in ("p05", "p50", "p95", "mean"):
                _put(m, f"{k}.{f}", getattr(r, f), "", src)
            _put(m, f"{k}.observed_480m", r.observed_five_products_480m, "", src)
            _put(m, f"{k}.share_groups_at_or_below_observed", r.share_groups_at_or_below_observed, "", src)
        else:
            unit = "m" if r.quantity == "c26_m" else "km2"
            for f in ("mean", "sd", "p05", "p50", "p95", "unperturbed_480m"):
                _put(m, f"{k}.{f}", getattr(r, f), unit, src)
            _put(m, f"{k}.sd_rel", r.sd_rel, "", src)
            if isinstance(getattr(r, "value_60m", None), float):
                _put(m, f"{k}.value_60m_km2", r.value_60m, "km2", src)
    ag = s[s["quantity"].str.startswith("agreement")]
    for q, g in ag.groupby("quantity"):
        _put(m, f"sim.{q}.observed_480m", float(g["observed_five_products_480m"].iloc[0]), "", src,
             "observed five-grid all/any ratio at 480 m")
        _put(m, f"sim.{q}.p50.min", g["p50"].min(), "", src, "median of the reference groups, smallest over grids")
        _put(m, f"sim.{q}.p50.max", g["p50"].max(), "", src)
        _put(m, f"sim.{q}.p05.min", g["p05"].min(), "", src)
        _put(m, f"sim.{q}.p95.max", g["p95"].max(), "", src)
        _put(m, f"sim.{q}.share_at_or_below.min", g["share_groups_at_or_below_observed"].min(), "", src)
        _put(m, f"sim.{q}.share_at_or_below.max", g["share_groups_at_or_below_observed"].max(), "", src)


def add(m: dict) -> None:
    for f in (reref, agreement60, exposure, pathways, simulation):
        f(m)
