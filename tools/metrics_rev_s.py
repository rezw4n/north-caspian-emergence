"""Metrics of revision analysis S (PREREG Amendment 18 items 1, 2, 4, 5 and 8; the H1
statistics).

Loaded by tools/build_metrics.revision_metrics, which calls add(m). Every key starts
with "rev.s."; each record is {"value", "unit", "source", "note"?} as put() in
tools/build_metrics.py, with "decimals" where the default 2 dp would mislead. Direct
reads of results/revision/s/ (scripts/rev_s_tables.py, scripts/rev_s_rasters.py);
nothing is recomputed here except counts and medians over those tables.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S = ROOT / "results" / "revision" / "s"
REL = "results/revision/s"
PK = {"gebco_2026": "gebco2026", "gebco_2022": "gebco2022", "etopo_2022": "etopo2022",
      "srtm15p_v2.7": "srtm15p", "etopo1": "etopo1"}
MAJORITY = ["gebco_2026", "gebco_2022", "etopo1", "srtm15p_v2.7"]      # the four products that mispredict
NAMES = {"gebco_2026": "GEBCO_2026", "gebco_2022": "GEBCO_2022", "etopo_2022": "ETOPO 2022",
         "srtm15p_v2.7": "SRTM15+", "etopo1": "ETOPO1"}


def _put(m, key, value, unit, source, note=None, decimals=None):
    if value is None:
        return
    if isinstance(value, float) and (math.isnan(value)):
        return
    if hasattr(value, "item"):
        value = value.item()
    rec = {"value": value, "unit": unit, "source": source}
    if note:
        rec["note"] = note
    if decimals is not None:
        rec["decimals"] = decimals
    m[f"rev.s.{key}"] = rec


def _read(name):
    import pandas as pd
    p = S / name
    return pd.read_csv(p) if p.exists() else None


def _r(x, nd):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), nd)


def _km(x, nd=2):
    return None if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))) else round(float(x) / 1000, nd)


def item1(m):
    d = _read("item1_counts.csv")
    if d is None:
        return
    src = f"{REL}/item1_counts.csv"
    for r in d.to_dict("records"):
        run = "" if r["run"] == "primary" else f"{r['run']}."
        variant = "sector_shape" if r["run"] == "sector_constants" else r["variant"]
        _put(m, f"item1.{run}{variant}.{PK[r['product']]}.{r['scope']}.n_mispredict", int(r["n_mispredict"]), "sectors",
             src, f"of {r['n_sectors']} sectors; A18 item 1" + ("" if not run else f"; registered red-team run {r['run']}"))
    d = d[d["run"] == "primary"]
    for v in ("shape", "total"):
        for scope in ("validated4", "six_no_komsomolets"):
            g = d[(d["variant"] == v) & (d["scope"] == scope) & d["product"].isin(MAJORITY)]
            _put(m, f"item1.{v}.{scope}.majority_products_min", int(g["n_mispredict"].min()), "sectors", src,
                 "smallest count over GEBCO_2026, GEBCO_2022, ETOPO1 and SRTM15+")
            _put(m, f"item1.{v}.{scope}.n_products_all", int((g["n_mispredict"] == g["n_sectors"]).sum()), "products", src,
                 "products among the four that mispredict in every sector of the scope")


def item2(m):
    import numpy as np
    d = _read("item2_p_mispredict.csv")
    if d is not None:
        src = f"{REL}/item2_p_mispredict.csv"
        for r in d.to_dict("records"):
            _put(m, f"item2.p_mispredict.{r['variant']}.{PK[r['product']]}.{r['sector']}", _r(r["p_mispredict"], 3),
                 "probability", src, "moving-block bootstrap over levels, blocks of 3, B = 2,000, seed 20261022 (D-113)",
                 3)
    c = _read("item2_count_distribution.csv")
    if c is not None:
        src = f"{REL}/item2_count_distribution.csv"
        for r in c.to_dict("records"):
            b = f"item2.count.{r['variant']}.{PK[r['product']]}.{r['scope']}"
            _put(m, b + ".registered", int(r["registered_count"]), "sectors", src)
            _put(m, b + ".p_at_least_registered", _r(r["p_at_least_registered"], 3), "probability", src,
                 "share of joint resamples with at least the registered number of mispredicted sectors", 3)
            _put(m, b + ".p_equal_registered", _r(r["p_equal_registered"], 3), "probability", src, None, 3)
            _put(m, b + ".q025", int(r["q025"]), "sectors", src, "2.5th percentile of the resampled count")
            _put(m, b + ".q975", int(r["q975"]), "sectors", src, "97.5th percentile of the resampled count")
            n = int(r["n_sectors"])
            if n >= 2:
                _put(m, b + ".p_at_least_n_minus_1", _r(r[f"p_count_{n - 1}"] + r[f"p_count_{n}"], 3), "probability", src,
                     "share of resamples with all sectors of the scope, or all but one, mispredicted", 3)
        for v in ("shape", "total"):
            for scope in ("validated4", "six_no_komsomolets", "all7"):
                g = c[(c["variant"] == v) & (c["scope"] == scope) & c["product"].isin(MAJORITY)]
                n = int(g["n_sectors"].iloc[0])
                p_all = g[f"p_count_{n}"]
                p_nm1 = g[f"p_count_{n}"] + g[f"p_count_{n - 1}"]
                _put(m, f"item2.count.{v}.{scope}.majority.p_all_min", _r(p_all.min(), 2), "probability", f"{REL}/item2_count_distribution.csv",
                     "over the four mispredicting products: share of resamples with every sector of the scope mispredicted", 2)
                _put(m, f"item2.count.{v}.{scope}.majority.p_all_max", _r(p_all.max(), 2), "probability", f"{REL}/item2_count_distribution.csv", None, 2)
                _put(m, f"item2.count.{v}.{scope}.majority.p_all_but_one_min", _r(p_nm1.min(), 2), "probability",
                     f"{REL}/item2_count_distribution.csv", "smallest share, over the four, of resamples with at most one sector not mispredicted", 2)
    q = _read("item2_dq.csv")
    if q is not None:
        src = f"{REL}/item2_dq.csv"
        for r in q.to_dict("records"):
            b = f"item2.dq.{r['variant']}.{PK[r['product']]}.{r['sector']}"
            for k in ("median", "p25", "p75"):
                v = r[f"dq_{k}"]
                _put(m, f"{b}.{k}", None if not np.isfinite(v) else _r(v, 2), "ratio", src,
                     "per-level D (median displacement) / Q (floor 95th percentile) over evaluable levels" if k == "median" else None, 2)
            _put(m, f"{b}.n_inf", int(r["dq_n_inf"]), "levels", src, "levels with no predicted waterline (D/Q infinite)")
            _put(m, f"{b}.d_km", _km(r["d_median_m"]), "km", src, "median over evaluable levels of D (Table S8 sector median)")
            _put(m, f"{b}.q_km", _km(r["q_median_m"]), "km", src, "median over the same levels of Q")
            _put(m, f"{b}.n_levels_q_over_10km", int(r["n_levels_q_over_10km"]), "levels", src,
                 "evaluable levels whose floor exceeds 10 km")


def item4(m):
    a = _read("item4_area_bias.csv")
    if a is not None:
        src = f"{REL}/item4_area_bias.csv"
        scopes = {"domain_all_sectors": "domain", "domain_validated_sectors": "validated"}
        for r in a.to_dict("records"):
            if r["scope"] not in scopes:
                continue
            b = f"item4.area_bias.{r['variant']}.{PK[r['product']]}.{scopes[r['scope']]}"
            for k in ("min", "p25", "median", "p75", "max", "iqr"):
                _put(m, f"{b}.{k}_pct", _r(r[f"bias_pct_{k}"], 1), "%", src,
                     "relative total area bias per (epoch, level) unit" if k == "median" else None, 1)
            _put(m, f"{b}.n_units", int(r["bias_pct_n"]), "units", src)
            for k, col in (("theil_sen_pct_per_m", "theil_sen_pct_per_m"), ("theil_sen_lo", "theil_sen_lo"),
                           ("theil_sen_hi", "theil_sen_hi")):
                _put(m, f"{b}.{k}", _r(r[col], 1), "% per m", src,
                     "Theil-Sen slope of the per-unit bias against level, 95% interval" if k == "theil_sen_pct_per_m" else None, 1)
        g = a[(a["variant"] == "shape") & (a["scope"] == "domain_all_sectors") & a["product"].isin(PK)]
        if len(g):
            _put(m, "item4.area_bias.shape.domain.iqr_min_pct", _r(g["bias_pct_iqr"].min(), 0), "%", src, "over the five products", 0)
            _put(m, "item4.area_bias.shape.domain.iqr_max_pct", _r(g["bias_pct_iqr"].max(), 0), "%", src, None, 0)
            _put(m, "item4.area_bias.shape.domain.slope_min", _r(g["theil_sen_pct_per_m"].min(), 0), "% per m", src, None, 0)
            _put(m, "item4.area_bias.shape.domain.slope_max", _r(g["theil_sen_pct_per_m"].max(), 0), "% per m", src, None, 0)
    o = _read("item4_oos.csv")
    if o is not None:
        src = f"{REL}/item4_oos.csv"
        for r in o.to_dict("records"):
            b = f"item4.oos.{PK[r['product']]}.{r['region']}.{r['constant']}"
            for k in ("min", "p25", "median", "p75", "max"):
                _put(m, f"{b}.{k}_pct", _r(r[f"bias_pct_{k}"], 1), "%", src,
                     "relative area bias of E3 units (D-114)" if k == "median" else None, 1)
            _put(m, f"{b}.abs_median_pct", _r(r["abs_bias_pct_median"], 1), "%", src, "median |relative area bias|", 1)
            _put(m, f"{b}.theil_sen_pct_per_m", _r(r["theil_sen_pct_per_m"], 1), "% per m", src, None, 1)
            _put(m, f"{b}.n_units", int(r["n_units"]), "units", src)
        for p, g in o.groupby("product"):
            _put(m, f"item4.oos.{PK[p]}.c_e12_m", _r(g["c_e12_m"].iloc[0], 2), "m", src,
                 "shape constant fitted on E1 + E2 training-block units (shape_offsets.json leave_one_epoch_out.E3)", 2)
            _put(m, f"item4.oos.{PK[p]}.c_all_m", _r(g["c_all_m"].iloc[0], 2), "m", src, "registered shape constant", 2)
        g = o[(o["region"] == "calibtest") & (o["constant"] == "c_e12")]
        if len(g):
            _put(m, "item4.oos.calibtest.c_e12.median_min_pct", _r(g["bias_pct_median"].min(), 1), "%", src, "over the five products", 1)
            _put(m, "item4.oos.calibtest.c_e12.median_max_pct", _r(g["bias_pct_median"].max(), 1), "%", src, None, 1)
    v = _read("item4_overlap.csv")
    if v is not None:
        src = f"{REL}/item4_overlap.csv"
        for r in v.to_dict("records"):
            sc = r["scope"]
            full = sc in ("domain", "band_domain", "band_validated")
            b = f"item4.overlap.{r['variant']}.{PK[r['product']]}.{sc}.{r['metric'].replace('_total', '')}"
            for k in (("min", "p25", "median", "p75", "max") if full else ("median",)):
                _put(m, f"{b}.{k}", _r(r[k], 3), "fraction", src,
                     "per (epoch, level) unit, with the known outside states" if k == "median" else None, 3)
        for sc in ("domain", "band_domain", "band_validated"):
            g = v[(v["variant"] == "shape") & (v["scope"] == sc) & (v["metric"] == "iou_total")]
            if len(g):
                _put(m, f"item4.overlap.shape.{sc}.iou.median_min", _r(g["median"].min(), 2), "fraction", src, "over the five products", 2)
                _put(m, f"item4.overlap.shape.{sc}.iou.median_max", _r(g["median"].max(), 2), "fraction", src, None, 2)


def item5(m):
    t = _read("item5_tests.csv")
    if t is None:
        return
    src = f"{REL}/item5_tests.csv"
    for r in t.to_dict("records"):
        b = f"item5.{r['variant']}.{PK[r['product']]}"
        _put(m, f"{b}.sector.stat_km", _km(r["sector_stat_m"]), "km", src,
             "Eq. S12 on sector medians: group median minus rest median (D-113)")
        _put(m, f"{b}.sector.p", _r(r["sector_p"], 3), "p", src, "exact one-sided, 35 labellings", 3)
        _put(m, f"{b}.sector.p_holm", _r(r["sector_p_holm"], 3), "p", src, "Holm over 5 products x 2 variants", 3)
        _put(m, f"{b}.sector.group_ranks", str(r["group_ranks_from_top"]).replace("+", ", "), "text", src,
             "ranks (1 = largest sector median) of ural, volga and west")
        _put(m, f"{b}.ranksum.p", _r(r["ranksum_p"], 3), "p", src, "D-115 sensitivity: exact rank-sum, 35 labellings", 3)
        _put(m, f"{b}.ranksum.p_holm", _r(r["ranksum_p_holm"], 3), "p", src, None, 3)
        _put(m, f"{b}.block.p", _r(r["block_p"], 4), "p", src, "registered block permutation test (redrawn, identical)", 4)
        _put(m, f"{b}.block.p_holm", _r(r["block_p_holm"], 4), "p", src, "Holm over 5 products x 2 variants", 4)
        _put(m, f"{b}.block.p_holm_within_variant", _r(r["block_p_holm_within_variant"], 4), "p", src,
             "Holm over the 5 products of this variant only (context, not A18; D-115 addendum)", 4)
        _put(m, f"{b}.block.n_group", int(r["n_group_blocks"]), "blocks", src, "deltaic or vegetated sectors")
        _put(m, f"{b}.block.n_rest", int(r["n_rest_blocks"]), "blocks", src)
        _put(m, f"{b}.block.null_q95_km", _km(r["block_null_q95_m"]), "km", src, "95th percentile of the 10,000 permuted statistics")
    for v in ("shape", "total"):
        g = t[t["variant"] == v]
        _put(m, f"item5.{v}.block.n_significant_holm", int(g["block_p_holm_significant"].astype(bool).sum()), "products", src,
             "block tests with Holm-adjusted p <= 0.05 (10 tests)")
        _put(m, f"item5.{v}.sector.n_significant_holm", int(g["sector_p_holm_significant"].astype(bool).sum()), "products", src)
        _put(m, f"item5.{v}.sector.p_min", _r(g["sector_p"].min(), 3), "p", src, None, 3)
        _put(m, f"item5.{v}.sector.p_max", _r(g["sector_p"].max(), 3), "p", src, None, 3)
        # Holm within one variant (5 tests), reported as context
        _put(m, f"item5.{v}.block.n_significant_holm_within_variant",
             int((g["block_p_holm_within_variant"] <= 0.05).sum()), "products", src,
             "block tests with Holm-adjusted p <= 0.05 over the 5 products of this variant only (context, not A18)")
    _put(m, "item5.sector.n_labellings", int(t["sector_n_labellings"].iloc[0]), "labellings", src)
    _put(m, "item5.sector.min_attainable_p", _r(2 / 35, 3), "p", src,
         "Eq. S12 with 3 against 4 sectors: the top labelling always ties with one other (D-115)", 3)
    _put(m, "item5.sector.n_group_sectors", int(t["n_group_sectors"].iloc[0]), "sectors", src)
    _put(m, "item5.sector.n_rest_sectors", int(t["n_rest_sectors"].iloc[0]), "sectors", src)
    gr = _read("item5_block_groups.csv")
    if gr is not None:
        s2 = f"{REL}/item5_block_groups.csv"
        for r in gr.to_dict("records"):
            b = f"item5.{r['variant']}.{PK[r['product']]}.block.{r['group']}"
            _put(m, f"{b}.distances_per_block_median", int(round(r["distances_per_block_median"])), "distances", s2)
            _put(m, f"{b}.distances_per_block_min", int(r["distances_per_block_min"]), "distances", s2)
            _put(m, f"{b}.distances_per_block_max", int(r["distances_per_block_max"]), "distances", s2)


def item8(m):
    b = _read("item8_binomial.csv")
    if b is not None:
        src = f"{REL}/item8_binomial.csv"
        for r in b.to_dict("records"):
            sc = r["scope"]
            if sc.startswith("all products"):
                k = "item8.all"
            elif sc.startswith("product "):
                k = f"item8.product.{PK[sc.split(' ', 1)[1]]}"
            elif sc.startswith("level bin "):
                k = f"item8.bin{int(sc.split()[2])}"
            else:
                k = "item8.votes"
            _put(m, f"{k}.n", int(r["n"]), "comparisons", src)
            _put(m, f"{k}.k_e1_better", int(r["k_e1_better"]), "comparisons", src)
            _put(m, f"{k}.share_pct", _r(100 * r["share"], 0), "%", src, None, 0)
            _put(m, f"{k}.p", _r(r["p_two_sided"], 6), "p", src, "two-sided exact binomial test against 0.5", 4)
    h = _read("item8_hysteresis.csv")
    if h is not None:
        src = f"{REL}/item8_hysteresis.csv"
        ok = h[~h["flag_few"].astype(bool)]
        _put(m, "item8.hysteresis.level_min_m", _r(h["level_m"].min(), 2), "m (DAHITI reference)", src, None, 2)
        _put(m, "item8.hysteresis.level_max_m", _r(h["level_m"].max(), 2), "m (DAHITI reference)", src, None, 2)
        _put(m, "item8.hysteresis.n_comparisons", int(len(ok)), "sector-levels", src, "with at least 20 distances")
        for s, g in ok.groupby("sector"):
            fin = g[g["e1_vs_later_median_m"] != float("inf")]
            _put(m, f"item8.hysteresis.{s}.median_km", _km(fin["e1_vs_later_median_m"].median()), "km", src,
                 "median over shared levels of the median E1-vs-(E2 + E3) distance")
            _put(m, f"item8.hysteresis.{s}.n_levels", int(len(g)), "levels", src)
            _put(m, f"item8.hysteresis.{s}.e2_halves_median_km", _km(g["halves_E2_median_m"].median()), "km", src,
                 "median over the same levels of the E2 within-epoch (halves) median distance")
            _put(m, f"item8.hysteresis.{s}.e2_halves_p95_km", _km(g["halves_E2_p95_m"].median()), "km", src)
            if "halves_pooled_p95_m" in g:
                _put(m, f"item8.hysteresis.{s}.halves_pooled_p95_km", _km(g["halves_pooled_p95_m"].median()), "km", src,
                     "median over the same levels of the pooled halves-only floor (D-114)")
        _put(m, "item8.hysteresis.e1_halves_n_measurable", int((h["halves_E1_n"] > 0).sum()), "sector-levels", src,
             "E1 within-epoch halves with any measured distance: none, so E1 has no within-epoch floor of its own")


def r3m4(m):
    t = _read("r3m4_test_blocks.csv")
    _put(m, "r3m4.floor_includes_between_epoch", "yes", "text", "scripts/a7_hindcast.py (run: nf_bin.add for kinds halves and e1_vs_e23)",
         "at levels crossed by E1 the registered per-level floor pools the E1-vs-(E2 + E3) distances with the halves")
    if t is not None:
        src = f"{REL}/r3m4_test_blocks.csv"
        for (p, v), g in t.groupby(["product", "variant"]):
            b = f"r3m4.test.{v}.{PK[p]}"
            mis = g["mispredicts_test"].astype(str).str.lower().eq("true")
            _put(m, f"{b}.n_sectors", int(mis.sum()), "sectors", src, "of 7; displacement from test-block cells only, registered floor")
            _put(m, f"{b}.n_validated", int(mis[g["sector"].isin(["west", "interfluve", "northeast", "open_shelf"])].sum()),
                 "sectors", src, "of the 4 validated")
            for r in g.to_dict("records"):
                _put(m, f"{b}.{r['sector']}.d_km", _km(r["d_test_sector_median_m"]), "km", src)
                _put(m, f"{b}.{r['sector']}.n_exceeding", int(r["n_exceeding_test"]), "levels", src)
                _put(m, f"{b}.{r['sector']}.n_evaluable", int(r["n_evaluable_test"]), "levels", src)
    f = _read("r3m4_floor_halves.csv")
    if f is not None:
        src = f"{REL}/r3m4_floor_halves.csv"
        for (p, v), g in f.groupby(["product", "variant"]):
            b = f"r3m4.halves_floor.{v}.{PK[p]}"
            a = g["mispredicts_halves_only"].astype(str).str.lower().eq("true")
            r = g["mispredicts_registered"].astype(str).str.lower().eq("true")
            _put(m, f"{b}.n_sectors", int(a.sum()), "sectors", src, "of 7; floor without the between-epoch distances (D-114)")
            _put(m, f"{b}.n_sectors_registered", int(r.sum()), "sectors", src)
            ch = g.loc[a != r, "sector"].tolist()
            _put(m, f"{b}.changed", ", ".join(ch) if ch else "none", "text", src)


def other(m):
    import pandas as pd
    f = _read("r3minor1_floor.csv")
    if f is not None:
        src = f"{REL}/r3minor1_floor.csv"
        for r in f.to_dict("records"):
            b = f"r3minor1.{r['sector']}"
            _put(m, f"{b}.floor_primary_km", _km(r["floor_primary_median_m"]), "km", src, "median over levels of the per-level floor")
            _put(m, f"{b}.floor_noenv_km", _km(r["floor_noenv_median_m"]), "km", src, "the same without the record-low rule")
            _put(m, f"{b}.ratio", _r(r["ratio_of_medians"], 2), "ratio", src, None, 2)
        hc = ROOT / "results" / "hindcast"
        for run in ("main", "main_noenv"):
            mp = pd.read_csv(hc / run / "h1_mispredicts.csv")
            for v, g in mp.groupby("variant"):
                g = g[g["sector"] == "open_shelf"]
                _put(m, f"r3minor1.open_shelf.{run}.{v}.n_products", int(g["mispredicts"].astype(str).str.lower().eq("true").sum()),
                     "products", f"results/hindcast/{run}/h1_mispredicts.csv", "products that mispredict on the open shelf (of 5)")
    d = _read("r1minor11_medians.csv")
    if d is not None:
        src = f"{REL}/r1minor11_medians.csv"
        for r in d[d["sector"] == "open_shelf"].to_dict("records"):
            b = f"r1minor11.{r['variant']}.{PK[r['product']]}.open_shelf"
            _put(m, f"{b}.median_observed_km", _km(r["median_observed_levels_m"]), "km", src, "over every observed level (SI red-team table)")
            _put(m, f"{b}.n_observed", int(r["n_observed_levels"]), "levels", src)
            _put(m, f"{b}.median_evaluable_km", _km(r["median_evaluable_levels_m"]), "km", src, "over evaluable levels (Table S8)")
            _put(m, f"{b}.n_evaluable", int(r["n_evaluable_levels"]), "levels", src)
    rl = _read("r2minor8_record_low.csv")
    if rl is not None:
        src = f"{REL}/r2minor8_record_low.csv"
        b = rl[rl["what"] == "record-low bin"]
        _put(m, "r2minor8.bins_text", ", ".join(str(int(x)) for x in b["bin"]), "text", src)
        _put(m, "r2minor8.level_low_m", _r(b["bin_low_m"].min(), 2), "m (DAHITI reference)", src, "lower edge of the lowest bin", 2)
        _put(m, "r2minor8.level_high_m", _r(b["bin_high_m"].max(), 2), "m (DAHITI reference)", src, "upper edge of the third bin", 2)
        _put(m, "r2minor8.n_acquisitions", int(b["n_acquisitions"].sum()), "acquisitions", src)
        _put(m, "r2minor8.first", str(b["first"].min()), "date", src)
        _put(m, "r2minor8.last", str(b["last"].max()), "date", src)
        months = sorted({int(x) for s in b["months"] for x in str(s).split("+")})
        _put(m, "r2minor8.months_text", ", ".join(str(x) for x in months), "text", src, "calendar months of the acquisitions")
        e = rl[rl["what"] == "lowest evaluable level"]
        _put(m, "r2minor8.lowest_evaluable_level_m", _r(e["level_center_m"].min(), 2), "m (DAHITI reference)", src,
             "lowest evaluable bin centre over sectors", 2)
        _put(m, "r2minor8.lowest_evaluable_level_max_m", _r(e["level_center_m"].max(), 2), "m (DAHITI reference)", src,
             "highest of the sectors' lowest evaluable bin centres", 2)
    s = S / "summary.json"
    if s.exists():
        js = json.loads(s.read_text(encoding="utf-8"))
        _put(m, "r2minor18.domain_km2_grid", js.get("domain_km2_from_grid"), "km2", f"{REL}/summary.json",
             "domain cells of the 60 m hindcast grid x 0.0036 km2 (lon/lat edges densified, D-039)")


def headline(m):
    """Ranges for the validated-sector headline (shape variant, the four products that
    mispredict): D and Q sector medians, the per-level D/Q medians and P(mispredict)."""
    q = _read("item2_dq.csv")
    p = _read("item2_p_mispredict.csv")
    if q is None or p is None:
        return
    for scope, secs in (("validated4", ["west", "interfluve", "northeast", "open_shelf"]),
                        ("six_no_komsomolets", ["west", "volga", "interfluve", "ural", "northeast", "open_shelf"])):
        g = q[(q["variant"] == "shape") & q["product"].isin(MAJORITY) & q["sector"].isin(secs)]
        src = f"{REL}/item2_dq.csv"
        note = f"over GEBCO_2026, GEBCO_2022, ETOPO1 and SRTM15+ and the sectors {', '.join(secs)} (shape variant)"
        _put(m, f"headline.{scope}.d_km_min", _km(g["d_median_m"].min()), "km", src, note)
        _put(m, f"headline.{scope}.d_km_max", _km(g["d_median_m"].max()), "km", src, note)
        _put(m, f"headline.{scope}.q_km_min", _km(g["q_median_m"].min()), "km", src, note)
        _put(m, f"headline.{scope}.q_km_max", _km(g["q_median_m"].max()), "km", src, note)
        _put(m, f"headline.{scope}.dq_median_min", _r(g["dq_median"].min(), 1), "ratio", src, note, 1)
        _put(m, f"headline.{scope}.dq_median_max", _r(g["dq_median"].max(), 1), "ratio", src, note, 1)
        h = p[(p["variant"] == "shape") & p["product"].isin(MAJORITY) & p["sector"].isin(secs)]
        _put(m, f"headline.{scope}.p_mispredict_min", _r(h["p_mispredict"].min(), 2), "probability",
             f"{REL}/item2_p_mispredict.csv", note, 2)
        for s in secs:
            gs = g[g["sector"] == s]
            _put(m, f"headline.{s}.d_km_min", _km(gs["d_median_m"].min()), "km", src, "over the four mispredicting products, shape")
            _put(m, f"headline.{s}.d_km_max", _km(gs["d_median_m"].max()), "km", src)
            _put(m, f"headline.{s}.q_km", _km(gs["q_median_m"].iloc[0]), "km", src, "the sector floor (median over evaluable levels)")
            _put(m, f"headline.{s}.dq_median_min", _r(gs["dq_median"].min(), 1), "ratio", src, None, 1)
            _put(m, f"headline.{s}.dq_median_max", _r(gs["dq_median"].max(), 1), "ratio", src, None, 1)
            hs = h[h["sector"] == s]
            _put(m, f"headline.{s}.p_mispredict_min", _r(hs["p_mispredict"].min(), 2), "probability",
                 f"{REL}/item2_p_mispredict.csv", None, 2)
            _put(m, f"headline.{s}.p_mispredict_max", _r(hs["p_mispredict"].max(), 2), "probability",
                 f"{REL}/item2_p_mispredict.csv", None, 2)


def add(m: dict) -> None:
    headline(m)
    item1(m)
    item2(m)
    item4(m)
    item5(m)
    item8(m)
    r3m4(m)
    other(m)
