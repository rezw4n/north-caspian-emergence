"""Metrics of workstream N (near-term emergence; PREREG Amendment 18 item 10; D-123 to
D-127), loaded by tools/build_metrics.revision_metrics. Keys start with ``rev.n.``; each
record is {"value", "unit", "source", "note"?} as build_metrics.put writes them.

Inputs: results/revision/n/*.csv and *.json written by scripts/rev_n_curves.py,
rev_n_envelopes.py, rev_n_simulate.py, rev_n_exposure.py and rev_n_pathways.py. A
missing table is skipped (its keys are then absent).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "results" / "revision" / "n"
SRC = "results/revision/n/"
PKEY = {"gebco_2026": "gebco2026", "etopo_2022": "etopo2022", "srtm15p_v2.7": "srtm15p",
        "gebco_2022": "gebco2022", "etopo1": "etopo1"}
PLABEL = {"gebco_2026": "GEBCO_2026", "etopo_2022": "ETOPO 2022", "srtm15p_v2.7": "SRTM15+",
          "gebco_2022": "GEBCO_2022", "etopo1": "ETOPO1"}
SCEN = (0.5, 1.0, 2.0, 3.0, 5.0)
SURF = ("uncorrected", "common_area", "common_location", "shifted", "shifted_common_location")
KEEP_SURF = ("uncorrected", "common_area", "common_location", "shifted")       # per-product keys
NEAR = (0.5, 1.0, 2.0)


def _put(m, key, value, unit, source, note=None):
    if value is None:
        return
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if math.isnan(value):
            return                                  # undefined: no key
        if math.isinf(value):
            value = "inf" if value > 0 else "-inf"
    elif isinstance(value, (np.integer,)):
        value = int(value)
    elif isinstance(value, (np.bool_,)):
        value = bool(value)
    rec = {"value": value, "unit": unit, "source": source}
    if note:
        rec["note"] = note
    m[f"rev.n.{key}"] = rec


def dk(x: float) -> str:
    return f"{x:g}".replace(".", "p")


def _read(name):
    p = D / name
    if not p.exists():
        return None
    if p.suffix in (".csv", ".gz"):
        try:
            return pd.read_csv(p)
        except pd.errors.EmptyDataError:          # a table written without rows (a step cut, D-126 addendum)
            return None
    return json.loads(p.read_text(encoding="utf-8"))


BASELINES = (("curves.csv", "", None),
             ("curves_composite.csv", "composite.", "composite observed baseline (D-123 addendum 2)"),
             ("curves_constfused.csv", "constfused.", "constant-form fused baseline (D-123 addendum)"))


def curves(m):
    for fname, tag, note in BASELINES:
        _curves(m, fname, tag, note)


def _curves(m, fname, tag, bnote):
    cv = _read(fname)
    run = _read("run_curves.json") if not tag else None
    if cv is None:
        return
    if tag == "composite.":
        cv = cv[cv["surface"].isin(["common_area", "common_location", "shifted_common_location"])]
    elif tag:
        cv = cv[cv["surface"].isin(["common_location"]) & (cv["region"] == "domain")]
    src = SRC + fname
    if run:
        _put(m, "level_2026_m", run["level_2026_m"], "m", SRC + "run_curves.json", "registered: mean of the 2026 DAHITI records (DAHITI working datum)")
        _put(m, "level_2026_running_median_m", run["level_2026_running_median"]["rounded_m"], "m",
             SRC + "run_curves.json", "registered sensitivity: last value of the 3-record running median (D-081)")
        _put(m, "check.common_area_vs_registered_max_abs_diff_km2",
             run["checks"]["common_area_vs_emergence_csv_sea_area"]["max_abs_diff_km2"], "km2", SRC + "run_curves.json",
             "common baseline (area) recomputed against the sea_area rows of results/emergence/main/emergence.csv")
        dn = run["dN_over_domain"]
        _put(m, "dN_min_m", dn["dN_min_m"], "m", SRC + "run_curves.json", "N(EGM2008, NGA) - N(EIGEN-6C4) over the domain cells")
        _put(m, "dN_max_m", dn["dN_max_m"], "m", SRC + "run_curves.json")
        for p, c in run["shape_constants_m"].items():
            _put(m, f"shape_constant.{PKEY[p]}_m", c, "m", SRC + "run_curves.json", "H1 shape constant (D-070)")
    main = cv[cv["sensitivity"] == "main"]
    keep = KEEP_SURF if not tag else ("common_area", "common_location")
    for r in main[main["region"].isin(["domain", "validated"]) & main["surface"].isin(keep)].itertuples():
        if not any(np.isclose(r.delta_m, s) for s in (0.0,) + SCEN):
            continue
        if np.isclose(r.delta_m, 0.0) and not r.surface.startswith("common"):
            continue
        k = f"{tag}E.{r.surface}.{PKEY[r.product]}.{dk(r.delta_m)}.{r.region}_km2"
        _put(m, k, r.E_km2, "km2", src, bnote)
        if (isinstance(getattr(r, "isolated_km2", None), float) and np.isfinite(r.isolated_km2)
                and r.region == "domain" and any(np.isclose(r.delta_m, s) for s in (1.0, 2.0))):
            _put(m, f"{tag}isolated.{r.surface}.{PKEY[r.product]}.{dk(r.delta_m)}.{r.region}_km2", r.isolated_km2, "km2", src,
                 "isolated water: below the scenario level but not connected to the sea (z < L <= F)")
    for surf in sorted(main["surface"].unique()):
        for s in SCEN:
            g = main[(main["surface"] == surf) & (main["region"] == "domain") & np.isclose(main["delta_m"], s)]
            if g.empty:
                continue
            g = g.set_index("product")["E_km2"]
            k = f"{tag}summary.{surf}.{dk(s)}"
            _put(m, f"{k}.min_km2", g.min(), "km2", src)
            _put(m, f"{k}.max_km2", g.max(), "km2", src)
            _put(m, f"{k}.min_product", PLABEL[g.idxmin()], "", src)
            _put(m, f"{k}.max_product", PLABEL[g.idxmax()], "", src)
            if g.min() > 0:
                _put(m, f"{k}.ratio", g.max() / g.min(), "", src)
            iso = main[(main["surface"] == surf) & (main["region"] == "domain") & np.isclose(main["delta_m"], s)]
            if "isolated_km2" in iso and iso["isolated_km2"].notna().all():
                sh = (iso["isolated_km2"] / iso["E_km2"]).replace([np.inf, -np.inf], np.nan)
                _put(m, f"{k}.isolated_share_min", sh.min(), "", src)
                _put(m, f"{k}.isolated_share_max", sh.max(), "", src)
    if tag == "constfused.":
        return
    sens = cv[(cv["region"] == "domain") & cv["surface"].isin(["uncorrected", "common_location", "shifted"])]
    for (surf, sn), g in sens.groupby(["surface", "sensitivity"]):
        for s in (0.5, 1.0, 2.0):
            gg = g[np.isclose(g["delta_m"], s)].set_index("product")["E_km2"]
            if gg.empty:
                continue
            _put(m, f"{tag}sens.{sn}.{surf}.{dk(s)}.min_km2", gg.min(), "km2", src)
            _put(m, f"{tag}sens.{sn}.{surf}.{dk(s)}.max_km2", gg.max(), "km2", src)
            for p, v in gg.items():
                if s == 2.0 or sn == "main":
                    continue
                _put(m, f"{tag}sens.{sn}.{surf}.{PKEY[p]}.{dk(s)}.domain_km2", v, "km2", src)
            if {"etopo_2022", "gebco_2022"} <= set(gg.index) and gg["gebco_2022"] > 0:
                _put(m, f"{tag}sens.{sn}.{surf}.{dk(s)}.ratio_etopo2022_gebco2022", gg["etopo_2022"] / gg["gebco_2022"], "", src)
            if gg.min() > 0:
                _put(m, f"{tag}sens.{sn}.{surf}.{dk(s)}.ratio_max_min", gg.max() / gg.min(), "", src)


def already_dry(m):
    for fname, tag in (("already_dry.csv", ""), ("already_dry_composite.csv", "composite.")):
        _already_dry(m, fname, tag)


def _already_dry(m, fname, tag):
    t = _read(fname)
    if t is None:
        return
    src = SRC + fname
    t = t[t["region"] == "domain"]
    for r in t[t["surface"].isin(["uncorrected", "shifted"]) & t["delta_m"].isin([1.0, 2.0])].itertuples():
        k = f"{tag}already_dry.{r.surface}.{PKEY[r.product]}.{dk(r.delta_m)}"
        _put(m, f"{k}.share_zv6", r.share_dry_zv6, "", src,
             "share of the band not water in the zone-v6 2025-26 near-minimum extent (D-123 (c))")
        _put(m, f"{k}.share_outside_obs_baseline", r.share_outside_obs_baseline, "", src,
             "share of the band outside the observed 2026 baseline (60 m)")
        _put(m, f"{k}.band_km2", r.band_km2, "km2", src)
    for (surf, s), g in t.groupby(["surface", "delta_m"]):
        for col, nm in (("share_dry_zv6", "share_zv6"), ("share_outside_obs_baseline", "share_outside_obs_baseline")):
            _put(m, f"{tag}already_dry.{surf}.{dk(s)}.{nm}_min", g[col].min(), "", src)
            _put(m, f"{tag}already_dry.{surf}.{dk(s)}.{nm}_max", g[col].max(), "", src)


def observed_band(m):
    t = _read("observed_band.csv")
    if t is None:
        return
    src = SRC + "observed_band.csv"
    o = t[(t["product"] == "observed")].set_index(["quantity", "region"])["km2"]
    for q in ("band_13_12", "band_14_12", "unknown12_sea13", "unknown12_sea14", "both_12_13", "sea12", "sea13"):
        for reg in ("domain", "validated"):
            if (q, reg) in o.index:
                _put(m, f"observed_band.{q}.{reg}_km2", o[(q, reg)], "km2", src)
    e2 = t[(t["quantity"] == "e2_baseline_less_sea12") & (t["region"] == "domain")]["km2"]
    if len(e2):
        _put(m, "observed_band.e2.domain_min_km2", e2.min(), "km2", src, "observed 2026 baseline less the sea at E3 bin 12, over products")
        _put(m, "observed_band.e2.domain_max_km2", e2.max(), "km2", src)
    b = t[(t["quantity"] == "baseline_sea_L2026") & (t["region"] == "domain")]["km2"]
    if len(b):
        _put(m, "baseline.sea_L2026.domain_min_km2", b.min(), "km2", src, "registered fused baseline at L2026, over products")
        _put(m, "baseline.sea_L2026.domain_max_km2", b.max(), "km2", src)


def observed_band_v2(m):
    t = _read("observed_band_v2.csv")
    if t is not None:
        src = SRC + "observed_band_v2.csv"
        for r in t[t["region"] == "domain"].itertuples():
            _put(m, f"observed_band_v2.{r.variant}.{r.quantity}.{r.region}_km2", r.km2, "km2", src,
                 "E3 bins 12 (-28.75 m), 13 (-28.65 m), 14 (-28.55 m); cells known at both units (D-123 addendum)")
    b = _read("baseline_composition.csv")
    if b is not None:
        src = SRC + "baseline_composition.csv"
        b = b[b["region"] == "domain"]
        for r in b.itertuples():
            who = "composite" if r.product == "composite" else f"{r.variant}.{PKEY[r.product]}"
            _put(m, f"baseline.{who}.{r.quantity}.{r.region}_km2", r.km2, "km2", src)
    s = _read("band_states.csv")
    if s is not None:
        src = SRC + "band_states.csv"
        s = s[s["region"] == "domain"]
        for (p, surf, d), g in s[s["delta_m"] == 1.0].groupby(["product", "surface", "delta_m"]):
            tot = g["km2"].sum()
            if tot <= 0:
                continue
            k = f"band_states.{surf}.{PKEY[p]}.{dk(d)}"
            for st in ("sea12", "land12", "unobserved12"):
                _put(m, f"{k}.share_{st}", g[g["state_bin12"] == st]["km2"].sum() / tot, "", src,
                     "share of the band by its observed state at E3 bin 12")
            _put(m, f"{k}.share_unobserved_outside_baseline_zv6_water",
                 g[(g["state_bin12"] == "unobserved12") & (g["baseline"] == "outside_baseline")
                   & (g["zone_v6"] == "zv6_water")]["km2"].sum() / tot, "", src,
                 "unobserved at bin 12, outside the registered fused baseline, water in zone v6 2025-26")


def contours(m):
    ss = _read("shared_soundings.json")
    if ss:
        src = SRC + "shared_soundings.json"
        for p, rec in ss["products"].items():
            for scope in ("domain", "domain_-34_to_-26m"):
                if scope not in rec:
                    continue
                sk = "all" if scope == "domain" else "band34_26"
                for f in ("share_equal", "modal_difference_m", "share_within_0p01_of_mode", "median_difference_m",
                          "share_product_integer", "share_reference_integer", "n_cells"):
                    _put(m, f"shared.{PKEY[p]}.{sk}.{f}", rec[scope][f], "m" if f.endswith("_m") else "", src,
                         "native 15-arc-second cells against GEBCO_2022")
    cc = _read("contour_classes.csv")
    if cc is not None:
        src = SRC + "contour_classes.csv"
        for (p, band, d), g in cc.groupby(["product", "band", "delta_m"]):
            vs = g[g["class"] == "vs_gebco_2022"]
            if len(vs) and band != "domain":
                k = f"contours.{band}.{PKEY[p]}.{dk(d)}"
                _put(m, f"{k}.share_equal_gebco2022", float(vs["share_equal_gebco_2022"].iloc[0]), "", src,
                     "share of the band's 60 m cells whose bilinear height equals GEBCO_2022's")
                _put(m, f"{k}.share_within_0p5m_gebco2022", float(vs["share_within_0p5m_gebco_2022"].iloc[0]), "", src)
                _put(m, f"{k}.median_difference_gebco2022_m", float(vs["median_difference_gebco_2022_m"].iloc[0]), "m", src)
            g = g[g["class"].isin(["terrace", "slope", "other"])]
            tot = g["area_km2"].sum()
            if band == "domain" or tot <= 0:
                continue
            ter = g[g["class"] == "terrace"]["area_km2"].sum()
            _put(m, f"contours.{band}.{PKEY[p]}.{dk(d)}.terrace_share", ter / tot, "", src,
                 "share of the band on cells whose bilinear height is an exact whole metre (a terrace)")
            top = g[g["class"] == "terrace"].sort_values("area_km2", ascending=False).head(1)
            if len(top):
                _put(m, f"contours.{band}.{PKEY[p]}.{dk(d)}.main_terrace_m", int(top["k"].iloc[0]), "m", src)
                _put(m, f"contours.{band}.{PKEY[p]}.{dk(d)}.main_terrace_km2", float(top["area_km2"].iloc[0]), "km2", src)


def envelopes(m):
    q = _read("envelope_q.csv")
    e = _read("envelopes.csv")
    run = _read("run_envelopes.json")
    if run:
        src = SRC + "run_envelopes.json"
        for ep, n in run["n_scored_pairs_by_epoch_domain"].items():
            _put(m, f"env.n_pairs.{ep}", n, "pairs", src)
        _put(m, "env.n_pairs.total", sum(run["n_scored_pairs_by_epoch_domain"].values()), "pairs", src)
        _put(m, "env.max_pair_decline_m", run["max_pair_decline_m"], "m", src)
        _put(m, "env.check_errors_max_abs_diff_km2", run["checks"]["uncorrected_errors_vs_loeo_errors_csv"]["max_abs_diff_km2"],
             "km2", src)
    if q is not None:
        src = SRC + "envelope_q.csv"
        for r in q[(q["region"] == "domain") & q["surface"].isin(KEEP_SURF + ("common_location_composite",))].itertuples():
            k = f"env.q.{r.surface}.{PKEY[r.product]}.{r.region}.{r.pairs}.{r.score}"
            unit = {"scale": "", "decline": "km2 per m", "abs": "km2"}[r.score]
            _put(m, k, r.q_hat, unit, src)
            if r.score == "scale":
                _put(m, f"env.n_infinite.{r.surface}.{PKEY[r.product]}.{r.pairs}", r.n_infinite, "scores", src,
                     "pairs whose observed area change over the common cells is zero")
                _put(m, f"env.n_scores.{r.surface}.{PKEY[r.product]}.{r.pairs}", r.n_scores, "scores", src)
    if e is not None:
        src = SRC + "envelopes.csv"
        e = e[(e["region"] == "domain") & e["delta_m"].isin(NEAR) & e["surface"].isin(KEEP_SURF + ("common_location_composite",))
              & (((e["score"] == "decline")) | ((e["score"] == "abs") & (e["pairs"] == "E3_pairs")))]
        for r in e.itertuples():
            k = f"env.{r.surface}.{PKEY[r.product]}.{r.pairs}.{r.score}.{dk(r.delta_m)}"
            if not r.calibrated:
                _put(m, f"{k}.calibrated", False, "", src, "decline beyond the largest hindcast pair decline")
                continue
            _put(m, f"{k}.lo_km2", r.lo_km2, "km2", src)
            _put(m, f"{k}.hi_km2", r.hi_km2, "km2", src)
    pdiff = _read("paired_difference.csv")
    if pdiff is not None:
        src = SRC + "paired_difference.csv"
        for r in pdiff.itertuples():
            k = f"paired.{r.form}.{PKEY[r.product]}.{dk(r.delta_m)}"
            _put(m, f"{k}.difference_km2", r.difference_km2, "km2", src)
            _put(m, f"{k}.q_hat_km2", r.q_hat_d_km2, "km2", src)
            _put(m, f"{k}.outside", bool(r.outside_paired_envelope), "", src)
            _put(m, f"{k}.registered_relevant", bool(r.registered_relevant), "", src)
        for form, g in pdiff.groupby("form"):
            _put(m, f"paired.{form}.n_outside", int(g["outside_paired_envelope"].sum()), "cases", src)
            _put(m, f"paired.{form}.n_registered_relevant", int(g["registered_relevant"].sum()), "cases", src)
            _put(m, f"paired.{form}.n_cases", int(len(g)), "cases", src)


def simulation(m):
    _simulation(m, "", None)
    _simulation(m, "_range50", 50000.0)          # D-125 addendum 2: the 50 km range ("sim.50km.*" keys)


def _simulation(m, sfx, only_range):
    s = _read(f"sim_summary{sfx}.csv")
    run = _read(f"run_simulate{sfx}.json")
    if s is None:
        return
    if only_range is not None:
        s = s[s["range_m"] == only_range]
    src = SRC + f"sim_summary{sfx}.csv"
    if run:
        if not sfx:
            _put(m, "sim.n_fields", run["n_fields"], "fields", SRC + "run_simulate.json")
        for k, v in run["negative_eigenvalue_share"].items():
            r_, p = k.split("|")
            if only_range is not None and float(r_) != only_range:
                continue
            _put(m, f"sim.neg_eigen_share.{int(float(r_) / 1000)}km.{PKEY[p]}", v, "", SRC + f"run_simulate{sfx}.json")
        for p, v in run["nmad_m"].items():
            _put(m, f"sim.nmad.{PKEY[p]}_m", v, "m", SRC + "run_simulate.json", "ATL08 test-block NMAD, overall (the field sd)")
    for r in s.itertuples():
        rk = f"{int(r.range_m / 1000)}km"
        k = f"sim.{rk}.{PKEY[r.product]}.{r.quantity}"
        if r.quantity.startswith("agreement"):
            for f in ("p05", "p50", "p95", "mean"):
                _put(m, f"{k}.{f}", getattr(r, f), "", src)
            _put(m, f"{k}.observed_240m", r.observed_five_products_240m, "", src)
            _put(m, f"{k}.share_groups_at_or_below_observed", r.share_groups_at_or_below_observed, "", src)
        elif r.quantity.endswith("domain"):
            for f in ("mean", "sd", "p05", "p50", "p95", "unperturbed_240m"):
                _put(m, f"{k}.{f}_km2", getattr(r, f), "km2", src)
            _put(m, f"{k}.sd_rel", r.sd_rel, "", src)
            if hasattr(r, "value_60m"):
                _put(m, f"{k}.value_60m_km2", r.value_60m, "km2", src)
    o = _read(f"sim_observed_240m{sfx}.csv") if not sfx else None
    if o is not None:
        a = o[o["product"] == "all_five_agreement"]
        for r in a.itertuples():
            _put(m, f"sim.observed_agreement_240m.{r.surface}.{dk(r.delta_m)}", r.ratio, "", SRC + "sim_observed_240m.csv")


EXP_SURF = ("uncorrected", "shifted", "common_location", "common_location_composite")


def exposure(m):
    pol = _read("exposure_polygons.csv")
    if pol is not None:
        src = SRC + "exposure_polygons.csv"
        t = pol[pol["region"] == "domain"]
        for r in t[t["delta_m"].isin([1.0, 2.0]) & t["surface"].isin(EXP_SURF)].itertuples():
            k = f"exp.{r.layer}.{r.surface}.{PKEY[r.product]}.{dk(r.delta_m)}"
            _put(m, f"{k}.km2", r.emerged_in_layer_km2, "km2", src, "emerged area inside the layer, domain cells only")
            _put(m, f"{k}.share", r.share, "", src, "share of the surface's domain emergence inside the layer")
        for (lay, surf, d), g in t[t["delta_m"].isin([0.5, 1.0, 2.0])].groupby(["layer", "surface", "delta_m"]):
            k = f"exp.{lay}.{surf}.{dk(d)}"
            _put(m, f"{k}.min_km2", g["emerged_in_layer_km2"].min(), "km2", src)
            _put(m, f"{k}.max_km2", g["emerged_in_layer_km2"].max(), "km2", src)
            _put(m, f"{k}.min_share", g["share"].min(), "", src)
            _put(m, f"{k}.max_share", g["share"].max(), "", src)
    fe = _read("exposure_features.csv")
    if fe is not None:
        src = SRC + "exposure_features.csv"
        t = fe[(fe["region"] == "domain") & fe["delta_m"].isin([1.0, 2.0]) & fe["surface"].isin(EXP_SURF)]
        for r in t.itertuples():
            sub = "" if r.subset == "all" else f".{r.subset}"
            k = f"exp.{r.layer}{sub}.{r.surface}.{PKEY[r.product]}.{dk(r.delta_m)}"
            unit = {"length": "km", "area": "km2"}.get(r.rule, "features")
            _put(m, f"{k}.amount", r.amount, unit, src)
        for (lay, sub, surf, d), g in t.groupby(["layer", "subset", "surface", "delta_m"]):
            s_ = "" if sub == "all" else f".{sub}"
            _put(m, f"exp.{lay}{s_}.{surf}.{dk(d)}.min", g["amount"].min(), "", src)
            _put(m, f"exp.{lay}{s_}.{surf}.{dk(d)}.max", g["amount"].max(), "", src)
    en = _read("exposure_envelope.csv")
    if en is not None:
        src = SRC + "exposure_envelope.csv"
        t = en[(en["region"] == "domain")]
        for (lay, surf, p, d), g in t.groupby(["layer", "surface", "product", "delta_m"]):
            b = g.set_index("bound")["amount"]
            k = f"exp_env.{lay}.{surf}.{PKEY[p]}.{dk(d)}"
            for tag in ("lo", "est", "hi"):
                if tag in b.index:
                    _put(m, f"{k}.{tag}", b[tag], "", src)
            _put(m, f"{k}.radius_m", float(g["radius_m"].iloc[0]), "m", src,
                 "validated-sector median displacement (shape variant for shifted, total for common_location)")
    ps = _read("ports_summary.csv")
    if ps is not None:
        src = SRC + "ports_summary.csv"
        for r in ps.itertuples():
            k = f"ports.{r.by}.{r.value}"
            for f in ("n", "n_in_domain", "n_within_10km", "n_within_20km", "n_within_50km", "n_far_up_delta"):
                _put(m, f"{k}.{f}", getattr(r, f), "features", src)


def pathways(m):
    v = _read("pathways_variance.csv")
    if v is not None:
        src = SRC + "pathways_variance.csv"
        for r in v[v["region"] == "domain"].itertuples():
            if not isinstance(getattr(r, "share_pathway", None), float) or not np.isfinite(r.share_pathway):
                continue
            sc = "area" if r.scale == "area_km2" else "log"
            k = f"var.{r.surface}.{r.region}.{r.year}.{sc}"
            _put(m, f"{k}.pathway", r.share_pathway, "", src)
            _put(m, f"{k}.product", r.share_product, "", src)
            _put(m, f"{k}.interaction", r.share_interaction, "", src)
    mt = _read("pathways_matrix.csv")
    if mt is not None:
        src = SRC + "pathways_matrix.csv"
        t = mt[(mt["region"] == "domain") & mt["surface"].isin(["common_location_composite", "shifted", "uncorrected"])]
        for r in t[t["role"] == "primary"].itertuples():
            k = f"path.{r.pathway_id.lower().replace('-', '_')}.{r.year}.{r.surface}.{PKEY[r.product]}_km2"
            _put(m, k, r.E_km2, "km2", src, r.note if isinstance(r.note, str) and r.note else None)
        for r in t.drop_duplicates(["pathway_id", "year"]).itertuples():
            _put(m, f"path.{r.pathway_id.lower().replace('-', '_')}.{r.year}.fall_m", r.fall_m, "m", src,
                 "fall below the 2026 level (negative: a rise), anchor base (D-127)")
        for (role, surf, y), g in t.groupby(["role", "surface", "year"]):
            _put(m, f"path.{role}.{y}.{surf}.min_km2", g["E_km2"].min(), "km2", src)
            _put(m, f"path.{role}.{y}.{surf}.max_km2", g["E_km2"].max(), "km2", src)
    an = _read("pathways_anchoring.csv")
    if an is not None:
        src = SRC + "pathways_anchoring.csv"
        for r in an.itertuples():
            k = f"anchor.{r.pathway_id.lower().replace('-', '_')}"
            _put(m, f"{k}.fall_2021_2026_m", r.pathway_fall_2021_2026_m, "m", src)
        _put(m, "anchor.observed_fall_2021_2026_m", float(an["observed_fall_2021_2026_m"].iloc[0]), "m", src,
             "mean of the DAHITI 2021 records minus the 2026 level")
        _put(m, "anchor.observed_fall_same_window_m", float(an["observed_fall_same_window_m"].iloc[0]), "m", src,
             "2021 records in the calendar window of the 2026 records")
        pr = an[an["role"] == "primary"]
        _put(m, "anchor.primary.max_fall_2021_2026_m", pr["pathway_fall_2021_2026_m"].max(), "m", src)
        _put(m, "anchor.primary.min_fall_2021_2026_m", pr["pathway_fall_2021_2026_m"].min(), "m", src)
    tw = _read("trend_windows.csv")
    if tw is not None:
        src = SRC + "trend_windows.csv"
        for r in tw.itertuples():
            k = f"trend.{r.start_year}_{r.end_year}"
            _put(m, f"{k}.rate_cm_per_yr", 100 * r.rate_m_per_yr, "cm/yr", src)
            _put(m, f"{k}.se_cm_per_yr", 100 * r.se_m_per_yr, "cm/yr", src)
            _put(m, f"{k}.n_records", r.n, "records", src)
            for d_ in ("1", "2"):
                y = getattr(r, f"year_reaches_{d_}m")
                _put(m, f"{k}.year_reaches_{d_}m", str(int(math.floor(y))) if np.isfinite(y) else "not by 2100", "year",
                     src, "applied from the 2026 level (anchor base); the calendar year in which the fall is reached, "
                          "as h3.date.*")


def add(m: dict) -> None:
    for f in (curves, already_dry, observed_band, observed_band_v2, contours, envelopes, simulation, exposure,
              pathways):
        f(m)
