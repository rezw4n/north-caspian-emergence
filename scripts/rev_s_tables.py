"""Revision analysis S: statistics from the frozen H1 tables (PREREG Amendment 18
items 1, 2, 4, 5 and 8). Reads the results of scripts/rev_s_rasters.py where an item needs them,
and skips those parts when that run is absent. No Earth Engine.

Outputs (results/revision/s/): item1_counts.csv; item2_p_mispredict.csv,
item2_count_distribution.csv, item2_dq.csv; item4_area_bias.csv, item4_oos.csv,
item4_overlap.csv; item5_sector_values.csv, item5_tests.csv, item5_null.csv,
item5_block_groups.csv; item8_binomial.csv, item8_hysteresis.csv; r3m4_test_blocks.csv,
r3m4_floor_halves.csv; r3minor1_floor.csv; r1minor11_medians.csv; r2minor8_record_low.csv;
summary.json.
Usage: python scripts/rev_s_tables.py [--tag main]
"""

from __future__ import annotations

import argparse
import json
import math
import sys

import caspian  # noqa: F401
import numpy as np
import pandas as pd
from scipy.stats import binomtest

from caspian import design, h1, rev_s
from caspian.config import INTERIM, RESULTS
from caspian.geography import SECTOR_ORDER

OUT = RESULTS / "revision" / "s"
RAST = OUT / "rasters"
VALIDATED = ["west", "interfluve", "northeast", "open_shelf"]           # A18 item 1
SIX = [s for s in SECTOR_ORDER if s != "komsomolets"]
SEED = 20261022                                                         # A18: every new resampling
B = 2000
BLOCK = 3
SCOPES = {"all7": list(SECTOR_ORDER), "validated4": VALIDATED, "six_no_komsomolets": SIX}


def _bool(s):
    return s.astype(str).str.lower().eq("true")


# --- item 1 ---------------------------------------------------------------------------------

def item1(mp: pd.DataFrame, others: dict | None = None) -> pd.DataFrame:
    """Verdict counts per scope; ``others`` = {run name: h1_mispredicts table} adds the
    registered red-team runs counted in the same scopes (their variants as named there)."""
    rows = []
    for run, tab in {"primary": mp, **(others or {})}.items():
        for (p, v), g in tab.groupby(["product", "variant"]):
            g = g.set_index("sector")
            for scope, secs in SCOPES.items():
                m = [s for s in secs if str(g.loc[s, "mispredicts"]).lower() == "true"]
                rows.append({"run": run, "product": p, "variant": v, "scope": scope, "n_sectors": len(secs),
                             "n_mispredict": len(m), "sectors_mispredicted": "+".join(m),
                             "sectors_not": "+".join(s for s in secs if s not in m)})
    return pd.DataFrame(rows)


# --- item 2 ---------------------------------------------------------------------------------

def evaluable(det: pd.DataFrame) -> pd.DataFrame:
    d = det[_bool(det["evaluable"])].copy()
    d["exceeds"] = _bool(d["exceeds"])
    return d.sort_values(["product", "variant", "sector", "level_m"])


def item2(det: pd.DataFrame, mp: pd.DataFrame):
    ev = evaluable(det)
    cells = {(p, v, s): g["exceeds"].to_numpy() for (p, v, s), g in ev.groupby(["product", "variant", "sector"], sort=True)}
    summ, flags, keys = rev_s.mbb_mispredict(cells, B, SEED, BLOCK)
    summ[["product", "variant", "sector"]] = pd.DataFrame(summ["key"].tolist(), index=summ.index)
    reg = mp.set_index(["product", "variant", "sector"])["mispredicts"]
    summ["registered_mispredicts"] = [bool(reg.loc[k]) for k in summ["key"]]
    if not (summ["registered_mispredicts"] == summ["observed_mispredicts"]).all():
        raise SystemExit("the evaluable-level flags do not reproduce h1_mispredicts.csv")
    summ = summ.drop(columns="key")
    dist_rows = []
    col = {k: j for j, k in enumerate(keys)}
    for (p, v), _ in summ.groupby(["product", "variant"]):
        for scope, secs in SCOPES.items():
            f = flags[:, [col[(p, v, s)] for s in secs]]
            dist = rev_s.count_distribution(f, len(secs))
            n_reg = int(sum(bool(reg.loc[(p, v, s)]) for s in secs))
            c = f.sum(axis=1)
            row = {"product": p, "variant": v, "scope": scope, "n_sectors": len(secs), "registered_count": n_reg,
                   "p_at_least_registered": float((c >= n_reg).mean()), "p_equal_registered": float((c == n_reg).mean()),
                   "mean_count": float(c.mean()), "q025": float(np.percentile(c, 2.5)), "q975": float(np.percentile(c, 97.5)),
                   "mode": int(np.argmax(dist))}
            row.update({f"p_count_{k}": float(x) for k, x in enumerate(dist)})
            dist_rows.append(row)
    dq = []
    for (p, v, s), g in ev.groupby(["product", "variant", "sector"]):
        r = rev_s.ratio_summary(g["median"].astype(float), g["p95_noise"].astype(float))
        dq.append({"product": p, "variant": v, "sector": s, **{f"dq_{k}": x for k, x in r.items()},
                   "d_median_m": float(g["median"].astype(float).median()),
                   "q_median_m": float(g["p95_noise"].astype(float).median()),
                   "n_levels_q_over_10km": int((g["p95_noise"].astype(float) > 10_000).sum())})
    return summ, pd.DataFrame(dist_rows), pd.DataFrame(dq)


# --- item 4 ---------------------------------------------------------------------------------

def _bias_summary(df, x, y, keys) -> dict:
    q = rev_s.quantile_summary(df[y])
    t = rev_s.theil_sen(df[x], df[y])
    return {**keys, **{f"bias_pct_{k}": v for k, v in q.items()},
            "theil_sen_pct_per_m": t["slope"], "theil_sen_lo": t["lo"], "theil_sen_hi": t["hi"]}


def item4(ext: pd.DataFrame):
    ext = ext[ext["epoch"].isin(["E1", "E2", "E3"])]
    rows, ov = [], []
    dom = ext[(ext["sector"] == "domain") & (ext["area_obs_km2"] > 0)].copy()
    dom["bias_pct"] = 100 * dom["area_bias_rel_total"].astype(float)
    for (p, v), g in dom.groupby(["product", "variant"]):
        rows.append(_bias_summary(g, "level_m", "bias_pct", {"product": p, "variant": v, "scope": "domain_all_sectors",
                                                              "source": "extent.csv domain rows"}))
        for metric in ("iou_total", "f1_total"):
            ov.append({"product": p, "variant": v, "scope": "domain", "metric": metric, "source": "extent.csv",
                       **rev_s.quantile_summary(g[metric].astype(float))})
    # cross-check: the validated sectors' rows summed (their cells reach beyond the domain)
    sec = ext[ext["sector"].isin(VALIDATED)]
    agg = sec.groupby(["product", "variant", "epoch", "bin", "level_m"])[
        ["area_pred_km2", "area_obs_km2", "area_fp_outside_km2", "area_fn_outside_km2"]].sum().reset_index()
    agg = agg[agg["area_obs_km2"] > 0]
    agg["bias_pct"] = 100 * (agg["area_pred_km2"] - agg["area_obs_km2"] + agg["area_fp_outside_km2"]
                             - agg["area_fn_outside_km2"]) / agg["area_obs_km2"]
    for (p, v), g in agg.groupby(["product", "variant"]):
        rows.append(_bias_summary(g, "level_m", "bias_pct", {"product": p, "variant": v,
                                                              "scope": "validated_sector_rows_check",
                                                              "source": "extent.csv validated sector rows summed"}))
    oos = None
    if (RAST / "band_overlap.csv").exists():
        bo = pd.read_csv(RAST / "band_overlap.csv")
        dv = bo[(bo["scope"] == "domain_validated") & (bo["area_obs_km2"] > 0)].copy()
        dv["bias_pct"] = 100 * dv["area_bias_rel_total"].astype(float)
        for (p, v), g in dv.groupby(["product", "variant"]):
            rows.append(_bias_summary(g, "level_m", "bias_pct", {"product": p, "variant": v,
                                                                  "scope": "domain_validated_sectors",
                                                                  "source": "rasters/band_overlap.csv"}))
        for scope in ["band_domain", "band_validated"] + [f"band_{s}" for s in SECTOR_ORDER]:
            g0 = bo[(bo["scope"] == scope) & (bo["area_obs_km2"] > 0)]     # units with observed water there
            for (p, v), g in g0.groupby(["product", "variant"]):
                for metric in ("iou_total", "f1_total"):
                    ov.append({"product": p, "variant": v, "scope": scope, "metric": metric,
                               "source": "rasters/band_overlap.csv", **rev_s.quantile_summary(g[metric].astype(float))})
    if (RAST / "oos_area.csv").exists():
        oa = pd.read_csv(RAST / "oos_area.csv")
        oa = oa[oa["region"].isin(["train_validated", "train_other", "calibtest_validated", "calibtest_other"])]
        regs = {"calibtest": ["calibtest_validated", "calibtest_other"], "calibtest_validated": ["calibtest_validated"],
                "train": ["train_validated", "train_other"]}
        orow = []
        for scope, rr in regs.items():
            g0 = (oa[oa["region"].isin(rr)].groupby(["product", "bin", "level_m"])
                  [["cells_pred_c_all", "cells_pred_c_e12", "cells_water"]].sum().reset_index())
            g0 = g0[g0["cells_water"] > 0]
            for which in ("c_e12", "c_all"):
                g0[f"bias_pct_{which}"] = 100 * (g0[f"cells_pred_{which}"] - g0["cells_water"]) / g0["cells_water"]
            for p, g in g0.groupby("product"):
                for which in ("c_e12", "c_all"):
                    s = _bias_summary(g, "level_m", f"bias_pct_{which}", {"product": p, "region": scope, "constant": which})
                    s["abs_bias_pct_median"] = float(g[f"bias_pct_{which}"].abs().median())
                    s["n_units"] = int(len(g))
                    orow.append(s)
        oos = pd.DataFrame(orow)
        consts = oa.groupby("product")[["c_all_m", "c_e12_m"]].first().reset_index()
        oos = oos.merge(consts, on="product", how="left")
    return pd.DataFrame(rows), oos, pd.DataFrame(ov)


# --- item 5 ---------------------------------------------------------------------------------

def item5(det: pd.DataFrame, blocks: pd.DataFrame, bt: pd.DataFrame, labels: pd.DataFrame):
    lab = labels.set_index("sector")
    group = sorted(lab.index[_bool(lab["deltaic"]) | _bool(lab["vegetated"])])
    ev = evaluable(det)
    vals = ev.groupby(["product", "variant", "sector"])["median"].median().rename("sector_median_m").reset_index()
    tests, nulls, groups = [], [], []
    for (p, v), g in vals.groupby(["product", "variant"]):
        values = dict(zip(g["sector"], g["sector_median_m"].astype(float)))
        ex = rev_s.exact_group_test(values, group)
        rk = rev_s.exact_rank_sum_test(values, group)
        b = blocks[(blocks["product"] == p) & (blocks["variant"] == v)]
        few, no_sector = b["n"] < h1.MIN_BLOCK_N, b["block_sector"].isna()
        keep = b[~few & ~no_sector]
        in_g = keep["block_sector"].isin(group)
        bn = rev_s.block_permutation_null(keep["median_abs_m"].to_numpy(float), in_g.to_numpy(), design.PERMUTATIONS_N,
                                          design.SEED)
        reg = bt[(bt["product"] == p) & (bt["variant"] == v)].iloc[0]
        if (abs(bn["p_value"] - float(reg["p_value"])) > 1e-12 or bn["n_group"] != int(reg["n_group"])
                or bn["n_rest"] != int(reg["n_rest"])):     # the CSV keeps 15 significant digits
            raise SystemExit(f"{p} {v}: the block-test redraw does not reproduce h1_block_test.csv")
        rank = {s: int(r) for s, r in zip(g.sort_values("sector_median_m", ascending=False)["sector"], range(1, 8))}
        tests.append({"product": p, "variant": v, "group": "+".join(group), "n_group_sectors": ex["group_size"],
                      "n_rest_sectors": ex["rest_size"], "sector_stat_m": ex["statistic"], "sector_p": ex["p_value"],
                      "sector_n_at_least": ex["n_at_least"], "sector_n_labellings": ex["n_labellings"],
                      "group_ranks_from_top": "+".join(str(rank[s]) for s in group),
                      "ranksum_stat": rk["statistic"], "ranksum_expected": rk["expected"], "ranksum_p": rk["p_value"],
                      "block_stat_m": bn["statistic"], "block_p": bn["p_value"], "n_group_blocks": bn["n_group"],
                      "n_rest_blocks": bn["n_rest"],
                      "block_null_q50_m": float(np.percentile(bn["null"], 50)),
                      "block_null_q95_m": float(np.percentile(bn["null"], 95)),
                      "block_null_q99_m": float(np.percentile(bn["null"], 99)),
                      "block_null_max_m": float(bn["null"].max())})
        nulls += [{"product": p, "variant": v, "test": "sector_exact", "labelling": l, "statistic_m": float(t)}
                  for l, t in zip(ex["labellings"], ex["null"])]
        qs = [0, 1, 2.5, 5, 10, 25, 50, 75, 90, 95, 97.5, 99, 100]
        nulls += [{"product": p, "variant": v, "test": "block_permutation", "labelling": f"q{q:g}",
                   "statistic_m": float(np.percentile(bn["null"], q))} for q in qs]
        for gname, sub in (("group", keep[in_g]), ("rest", keep[~in_g])):
            q = rev_s.quantile_summary(sub["n"].astype(float))
            groups.append({"product": p, "variant": v, "group": gname, "n_blocks": int(len(sub)),
                           "sectors": "+".join(sorted(sub["block_sector"].unique())),
                           "n_excluded_few": int((few & ~no_sector).sum()), "n_excluded_no_sector": int(no_sector.sum()),
                           **{f"distances_per_block_{k}": x for k, x in q.items()},
                           "block_value_median_m": float(sub["median_abs_m"].median())})
    t = pd.DataFrame(tests)
    for c in ("sector_p", "ranksum_p", "block_p"):
        t[f"{c}_holm"] = rev_s.holm(t[c].to_numpy())
        t[f"{c}_holm_significant"] = t[f"{c}_holm"] <= design.ALPHA
    # context only: Holm over the 5 products of one variant
    t["block_p_holm_within_variant"] = np.nan
    for v, idx in t.groupby("variant").groups.items():
        t.loc[idx, "block_p_holm_within_variant"] = rev_s.holm(t.loc[idx, "block_p"].to_numpy())
    return vals, t, pd.DataFrame(nulls), pd.DataFrame(groups)


# --- item 8 ---------------------------------------------------------------------------------

def item8(nf: pd.DataFrame):
    s = json.loads((RESULTS / "robustness" / "r10_morphology" / "summary.json").read_text(encoding="utf-8"))
    sm = pd.DataFrame(s["summary_shape"])
    rows = []
    n, k = int(sm["n_bins"].sum()), int(sm["n_e1_better"].sum())
    rows.append({"scope": "all products, sectors and levels (the 35% claim)", "n": n, "k_e1_better": k,
                 "share": k / n, "p_two_sided": binomtest(k, n, 0.5).pvalue})
    for p, g in sm.groupby("product"):
        n_, k_ = int(g["n_bins"].sum()), int(g["n_e1_better"].sum())
        rows.append({"scope": f"product {p}", "n": n_, "k_e1_better": k_, "share": k_ / n_,
                     "p_two_sided": binomtest(k_, n_, 0.5).pvalue})
    md = pd.read_csv(RESULTS / "robustness" / "r10_morphology" / "morphology_detail.csv")
    md = md[md["variant"] == "shape"]
    md = md[np.isfinite(md["E1"]) & np.isfinite(md["E23"]) & (md["E1"] != md["E23"])].copy()
    md["e1_better"] = _bool(md["e1_better"])
    if len(md) != n:
        raise SystemExit(f"morphology detail gives {len(md)} comparisons, the summary {n}")
    for b, g in md.groupby("bin"):
        rows.append({"scope": f"level bin {int(b)} ({-30 + (int(b) + 0.5) * 0.1:.2f} m)", "n": len(g),
                     "k_e1_better": int(g["e1_better"].sum()),
                     "share": float(g["e1_better"].mean()), "p_two_sided": binomtest(int(g["e1_better"].sum()), len(g), 0.5).pvalue})
    votes = rev_s.majority_votes(md, ["sector", "bin"], "e1_better")
    rows.append({"scope": "one vote per (sector, level): majority over products, ties dropped", "n": int(votes.size),
                 "k_e1_better": int(votes.sum()), "share": float(votes.mean()),
                 "p_two_sided": binomtest(int(votes.sum()), int(votes.size), 0.5).pvalue})
    binom = pd.DataFrame(rows)
    # hysteresis
    ev = nf[(nf["kind"] == "e1_vs_e23") & (nf["sector"] != "all") & (nf["n"] > 0)].copy()
    hv = nf[(nf["kind"] == "halves") & (nf["sector"] != "all")]
    hyst = []
    fl = pd.read_csv(RAST / "floor_e1_bins.csv") if (RAST / "floor_e1_bins.csv").exists() else None
    for r in ev.itertuples():
        row = {"sector": r.sector, "bin": int(r.bin), "level_m": r.level_m, "e1_vs_later_median_m": r.median_m,
               "e1_vs_later_p95_m": r.p95_m, "e1_vs_later_n": int(r.n), "flag_few": bool(r.n < 20)}
        for ep in ("E1", "E2"):
            h = hv[(hv["epoch"] == ep) & (hv["bin"] == r.bin) & (hv["sector"] == r.sector)]
            ok = len(h) and float(h["n"].iloc[0]) >= 20          # the 20-distance flag
            row[f"halves_{ep}_median_m"] = float(h["median_m"].iloc[0]) if ok else np.nan
            row[f"halves_{ep}_p95_m"] = float(h["p95_m"].iloc[0]) if ok else np.nan
            row[f"halves_{ep}_n"] = int(h["n"].iloc[0]) if len(h) and float(h["n"].iloc[0]) > 0 else 0
        if fl is not None:
            f = fl[(fl["bin"] == int(r.bin)) & (fl["sector"] == r.sector)]
            okf = len(f) and float(f["halves_n"].iloc[0]) >= 20
            row["halves_pooled_median_m"] = float(f["halves_median_m"].iloc[0]) if okf else np.nan
            row["halves_pooled_p95_m"] = float(f["halves_p95_m"].iloc[0]) if okf else np.nan
            row["halves_pooled_n"] = int(f["halves_n"].iloc[0]) if len(f) else 0
        hyst.append(row)
    hyst = pd.DataFrame(hyst)
    return binom, hyst


# --- held-out displacement and the floor without between-epoch distances ----------------------

def r3m4(det: pd.DataFrame):
    out_t, out_f = None, None
    reg = det.copy()
    reg["observed_level"] = _bool(reg["observed_level"])
    if (RAST / "test_displacement.csv").exists():
        td = pd.read_csv(RAST / "test_displacement.csv")
        td = td[td["epoch"] == "pooled"][["product", "variant", "sector", "bin", "median_m", "n"]]
        td = td.rename(columns={"median_m": "d_test_m", "n": "n_test"})
        m = reg.merge(td, on=["product", "variant", "sector", "bin"], how="inner")
        m["evaluable_test"] = m["observed_level"] & (m["n_test"] > 0) & (m["n_noise"] > 0)
        d = m["d_test_m"].astype(float)
        m["exceeds_test"] = m["evaluable_test"] & ((d == np.inf) | (d > m["p95_noise"].astype(float)))
        rows = []
        for (p, v, s), g in m.groupby(["product", "variant", "sector"]):
            e = g[g["evaluable_test"]]
            ne, nx = int(len(e)), int(e["exceeds_test"].sum())
            ra = g[_bool(g["evaluable"])]
            rows.append({"product": p, "variant": v, "sector": s, "n_evaluable_test": ne, "n_exceeding_test": nx,
                         "mispredicts_test": (nx > ne / 2) if ne else None,
                         "d_test_sector_median_m": float(e["d_test_m"].astype(float).median()) if ne else np.nan,
                         "n_evaluable_registered": int(len(ra)), "n_exceeding_registered": int(_bool(ra["exceeds"]).sum()),
                         "d_registered_sector_median_m": float(ra["median"].astype(float).median()),
                         "test_distances_total": int(e["n_test"].sum())})
        out_t = pd.DataFrame(rows)
    if (RAST / "floor_e1_bins.csv").exists():
        fl = pd.read_csv(RAST / "floor_e1_bins.csv")[["bin", "sector", "halves_p95_m", "halves_n"]]
        nz = reg.merge(fl, on=["bin", "sector"], how="left")
        alt = nz["halves_n"].notna()
        nz["p95_alt"] = np.where(alt, nz["halves_p95_m"], nz["p95_noise"])
        nz["n_alt"] = np.where(alt, nz["halves_n"].fillna(0), nz["n_noise"])
        prod = nz[["product", "variant", "sector", "bin", "observed_level", "n", "median"]]
        noise = nz[["sector", "bin", "n_alt", "p95_alt"]].drop_duplicates(["sector", "bin"]).rename(
            columns={"n_alt": "n", "p95_alt": "p95"})
        summ, _ = h1.mispredicts(prod, noise)
        regs = h1.mispredicts(reg[["product", "variant", "sector", "bin", "observed_level", "n", "median"]],
                              reg[["sector", "bin", "n_noise", "p95_noise"]].drop_duplicates(["sector", "bin"]).rename(
                                  columns={"n_noise": "n", "p95_noise": "p95"}))[0]
        out_f = summ.merge(regs, on=["product", "variant", "sector"], suffixes=("_halves_only", "_registered"))
        fl2 = fl.merge(reg[["sector", "bin", "p95_noise", "n_noise"]].drop_duplicates(["sector", "bin"]),
                       on=["sector", "bin"], how="left")
        out_f.attrs["levels"] = fl2
    return out_t, out_f


# --- further H1 table checks -------------------------------------------------------------------

def r3minor1(nf: pd.DataFrame, nf0: pd.DataFrame, mp: pd.DataFrame, mp0: pd.DataFrame) -> pd.DataFrame:
    def pl(x):
        x = x[(x["kind"] == "pooled_level") & (x["sector"] != "all") & (x["n"] > 0)]
        return x[["sector", "bin", "p95_m", "n"]]
    a, b = pl(nf), pl(nf0)
    m = a.merge(b, on=["sector", "bin"], suffixes=("_primary", "_noenv"))
    rows = []
    for s, g in m.groupby("sector"):
        rows.append({"sector": s, "floor_primary_median_m": float(pl(nf).query("sector == @s")["p95_m"].median()),
                     "floor_noenv_median_m": float(pl(nf0).query("sector == @s")["p95_m"].median()),
                     "paired_ratio_median": float((g["p95_m_primary"] / g["p95_m_noenv"]).median()),
                     "n_levels_paired": int(len(g)),
                     "share_levels_primary_lower": float((g["p95_m_primary"] < g["p95_m_noenv"]).mean())})
    r = pd.DataFrame(rows)
    r["ratio_of_medians"] = r["floor_primary_median_m"] / r["floor_noenv_median_m"]
    for name, x in (("primary", mp), ("noenv", mp0)):
        c = x[_bool(x["mispredicts"].astype(str))].groupby(["sector"]).size()
        r[f"n_product_variant_mispredicting_{name}"] = r["sector"].map(c).fillna(0).astype(int)
    return r


def r1minor11(disp: pd.DataFrame, det: pd.DataFrame) -> pd.DataFrame:
    q = disp[(disp["epoch"] == "pooled") & (disp["sector"] != "all")]
    q = q[_bool(q["observed_level"].astype(str))]
    a = q.groupby(["product", "variant", "sector"])["median_m"].agg(["median", "size"]).rename(
        columns={"median": "median_observed_levels_m", "size": "n_observed_levels"})
    ev = evaluable(det)
    b = ev.groupby(["product", "variant", "sector"])["median"].agg(["median", "size"]).rename(
        columns={"median": "median_evaluable_levels_m", "size": "n_evaluable_levels"})
    return a.join(b).reset_index()


def r2minor8(det: pd.DataFrame) -> pd.DataFrame:
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("a7_hindcast", Path(__file__).resolve().parent / "a7_hindcast.py")
    a7 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(a7)
    units = a7.stage2_index(INTERIM / "stage2" / "main")
    e3 = sorted(b for b, eps in units.items() if "E3" in eps)
    low = e3[:a7.ENVELOPE_LOW_BINS]
    plan = pd.read_csv(INTERIM / "stage2" / "plan_main_tau9900cm_landsat.csv")
    plan["time"] = pd.to_datetime(plan["time"], unit="ms")         # acquisition time, ms since 1970 (UTC)
    rows = []
    for b in low:
        g = plan[(plan["epoch"] == "E3") & (plan["bin"] == b)]
        lo = design.BIN_ORIGIN_M + b * design.BIN_WIDTH_M
        rows.append({"what": "record-low bin", "bin": b, "level_center_m": round(a7._bin_level(b), 2),
                     "bin_low_m": round(lo, 2), "bin_high_m": round(lo + design.BIN_WIDTH_M, 2),
                     "n_scenes": int(len(g)), "n_acquisitions": int(g["acq"].nunique()),
                     "first": str(g["time"].min().date()), "last": str(g["time"].max().date()),
                     "months": "+".join(str(m) for m in sorted(g["time"].dt.month.unique())),
                     "years": "+".join(str(y) for y in sorted(g["time"].dt.year.unique()))})
    ev = evaluable(det)
    for s, g in ev[(ev["product"] == "gebco_2026") & (ev["variant"] == "shape")].groupby("sector"):
        rows.append({"what": "lowest evaluable level", "sector": s, "bin": int(g["bin"].min()),
                     "level_center_m": round(float(g["level_m"].min()), 2)})
    obs = det[_bool(det["observed_level"].astype(str))]
    rows.append({"what": "lowest observed level (any sector)", "bin": int(obs["bin"].min()),
                 "level_center_m": round(float(obs["level_m"].min()), 2)})
    return pd.DataFrame(rows)


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default="main")
    a = ap.parse_args(argv)
    hc, hc0 = RESULTS / "hindcast" / a.tag, RESULTS / "hindcast" / f"{a.tag}_noenv"
    OUT.mkdir(parents=True, exist_ok=True)
    mp = pd.read_csv(hc / "h1_mispredicts.csv")
    det = pd.read_csv(hc / "h1_mispredicts_detail.csv")
    nf, nf0 = pd.read_csv(hc / "noise_floor.csv"), pd.read_csv(hc0 / "noise_floor.csv")
    others = {"no_record_low_rule": pd.read_csv(hc0 / "h1_mispredicts.csv"),
              "sector_constants": pd.read_csv(RESULTS / "hindcast" / f"{a.tag}_sectorshape" / "h1_mispredicts.csv")}
    item1(mp, others).to_csv(OUT / "item1_counts.csv", index=False)
    pm, cd, dq = item2(det, mp)
    pm.to_csv(OUT / "item2_p_mispredict.csv", index=False)
    cd.to_csv(OUT / "item2_count_distribution.csv", index=False)
    dq.to_csv(OUT / "item2_dq.csv", index=False)
    ab, oos, ov = item4(pd.read_csv(hc / "extent.csv"))
    ab.to_csv(OUT / "item4_area_bias.csv", index=False)
    ov.to_csv(OUT / "item4_overlap.csv", index=False)
    if oos is not None:
        oos.to_csv(OUT / "item4_oos.csv", index=False)
    vals, tests, nulls, groups = item5(det, pd.read_csv(hc / "block_h1.csv"), pd.read_csv(hc / "h1_block_test.csv"),
                                       pd.read_csv(RESULTS / "levels" / "sector_labels.csv"))
    vals.to_csv(OUT / "item5_sector_values.csv", index=False)
    tests.to_csv(OUT / "item5_tests.csv", index=False)
    nulls.to_csv(OUT / "item5_null.csv", index=False)
    groups.to_csv(OUT / "item5_block_groups.csv", index=False)
    binom, hyst = item8(nf)
    binom.to_csv(OUT / "item8_binomial.csv", index=False)
    hyst.to_csv(OUT / "item8_hysteresis.csv", index=False)
    t, f = r3m4(det)
    if t is not None:
        t.to_csv(OUT / "r3m4_test_blocks.csv", index=False)
    if f is not None:
        f.to_csv(OUT / "r3m4_floor_halves.csv", index=False)
        f.attrs["levels"].to_csv(OUT / "r3m4_floor_halves_levels.csv", index=False)
    r3minor1(nf, nf0, mp, pd.read_csv(hc0 / "h1_mispredicts.csv")).to_csv(OUT / "r3minor1_floor.csv", index=False)
    r1minor11(pd.read_csv(hc / "displacement.csv"), det).to_csv(OUT / "r1minor11_medians.csv", index=False)
    r2minor8(det).to_csv(OUT / "r2minor8_record_low.csv", index=False)
    run = json.loads((hc / "run.json").read_text(encoding="utf-8"))
    summary = {"decisions": ["D-113", "D-114", "D-115"], "seed": SEED, "n_boot": B, "block_levels": BLOCK,
               "validated_sectors": VALIDATED, "rasters_used": RAST.exists(),
               "domain_cells": run["static_counts"]["domain"],
               "domain_km2_from_grid": round(run["static_counts"]["domain"] * 0.0036, 1)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
