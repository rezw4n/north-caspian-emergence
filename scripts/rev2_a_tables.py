"""Second-round revision analysis A: H1 verdicts for PREREG Amendment 19 items 7, 8 and 11
(D-147), from the pass tables of scripts/rev2_a_audit.py and scripts/rev2_a_variants.py.

Commands (outputs in results/revision2/a/):
  replica  the pass's shape and total (c = 0) variants and its registered floor against
           results/hindcast/main: per-level and per-unit medians, n and the floor must
           agree exactly (largest differences reported); replica.json
  item7    (a) the total variant with c_ref (and the ATL06 and SWOT sensitivities) against
           the registered total and shape variants; (b) the shape variant without the geoid
           field against the primary shape variant; item7/*.csv, item7.json
  item8    the share of predicted waterline cells left out as unknown (from the frozen
           displacement table, checked against the pass), and the verdicts under (i) and
           (ii) with the registered floor and with the matched floor; item8/*.csv, item8.json
  item11   the observation variants against their reference (the primary, or the primary
           on the same footprint for the R-block variants); item11/*.csv, item11.json
  all      every command above, then summary.json (validated sectors first, and the
           verdict of every product in the west and the interfluve under every variant)

Usage: python scripts/rev2_a_tables.py replica|item7|item8|item11|all [--name main]
"""

from __future__ import annotations

import argparse
import json
import sys

import caspian  # noqa: F401
import numpy as np
import pandas as pd

from caspian import h1, rev2_a
from caspian.config import RESULTS

A = RESULTS / "revision2" / "a"
AUDIT = A / "audit"
MAIN = RESULTS / "hindcast" / "main"
VALIDATED = rev2_a.VALIDATED
ORDER = rev2_a.SECTOR_REPORT_ORDER
PRODUCTS = ["gebco_2026", "gebco_2022", "etopo_2022", "etopo1", "srtm15p_v2.7"]
VARIANTS_11 = ["veg_as_dry", "veg_as_water", "rR_awei", "rR_fixed0"]


def _json(obj) -> str:
    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if o is pd.NA:
            return None
        return str(o)
    return json.dumps(obj, indent=1, default=conv)


def _write(sub: str | None, tables: dict, name: str, summary: dict):
    d = A / sub if sub else A
    d.mkdir(parents=True, exist_ok=True)
    for k, t in tables.items():
        t.to_csv(d / f"{k}.csv", index=False)
    (A / f"{name}.json").write_text(_json(summary), encoding="utf-8")
    print(_json(summary)[:3000])


def main_tables():
    disp = pd.read_csv(MAIN / "displacement.csv")
    noise = pd.read_csv(MAIN / "noise_floor.csv")
    pooled = disp[(disp["epoch"] == "pooled") & (disp["sector"] != "all")]
    Q = noise[(noise["kind"] == "pooled_level") & (noise["sector"] != "all")]
    return disp, pooled, Q


def verdicts(D: pd.DataFrame, Q: pd.DataFrame, variant: str):
    """caspian.h1.mispredicts on per-level tables (D: product, sector, bin, level_m,
    observed_level, n, median_m; Q: sector, bin, n, p95_m)."""
    prod = D.assign(variant=variant).rename(columns={"median_m": "median"})
    nz = Q.rename(columns={"p95_m": "p95"})[["sector", "bin", "n", "p95"]]
    return h1.mispredicts(prod[["product", "variant", "sector", "bin", "level_m", "observed_level", "n", "median"]], nz)


def frozen_verdicts(variant: str) -> pd.DataFrame:
    s = pd.read_csv(MAIN / "h1_mispredicts.csv")
    s = s[s["variant"] == variant].copy()
    s["mispredicts"] = s["mispredicts"].map({True: True, False: False, "True": True, "False": False}).astype("boolean")
    return s


def _ordered(t: pd.DataFrame) -> pd.DataFrame:
    t = t.copy()
    t["_o"] = t["sector"].map({s: i for i, s in enumerate(ORDER)})
    return t.sort_values(["product", "_o"]).drop(columns="_o").reset_index(drop=True)


def _verdict_map(summ: pd.DataFrame) -> dict:
    out = {}
    for r in summ.itertuples():
        v = r.mispredicts
        out.setdefault(r.product, {})[r.sector] = (None if v is pd.NA or (isinstance(v, float) and np.isnan(v))
                                                   else bool(v))
    return {p: {s: d[s] for s in ORDER if s in d} for p, d in out.items()}


def _changed(cmp: pd.DataFrame) -> list:
    c = cmp[cmp["changed"].astype(bool)]
    return c[["product", "sector", "validated", "mispredicts_reference", "mispredicts_new", "n_exceeding_reference",
              "n_exceeding_new", "n_evaluable_new"]].to_dict(orient="records")


def _levels(name):
    d = AUDIT / name
    return pd.read_csv(d / "levels_D.csv.gz"), pd.read_csv(d / "levels_Q.csv"), pd.read_csv(d / "units_D.csv.gz")


def _sector_medians(D: pd.DataFrame, label: str) -> pd.DataFrame:
    return D.groupby(["product", "sector"])["median_m"].median().rename(f"D_median_{label}_m").reset_index()


# --- commands ------------------------------------------------------------------------------------

def maxdiff(x, y):
    x, y = x.astype(float).to_numpy(), y.astype(float).to_numpy()
    both = np.isnan(x) & np.isnan(y)
    same_inf = np.isinf(x) & np.isinf(y) & (np.sign(x) == np.sign(y))
    dd = np.abs(x - y)
    dd[both | same_inf] = 0.0
    return (float(np.nanmax(dd)) if dd.size else float("nan")), int(np.sum(np.isnan(x) != np.isnan(y)))


def cmd_replica(a) -> dict:
    LD, LQ, U = _levels(a.name)
    disp, pooled, Q = main_tables()
    out = {}
    for cfg, var in (("shape", "shape"), ("total0", "total")):
        dm = pooled[pooled["variant"] == var]
        rep = LD[LD["config"] == cfg].merge(dm, on=["product", "sector", "bin"], suffixes=("", "_main"))
        du = disp[(disp["variant"] == var) & disp["epoch"].isin(["E1", "E2", "E3"]) & (disp["sector"] != "all")]
        uu = U[U["config"] == cfg].merge(du, on=["product", "epoch", "bin", "sector"], suffixes=("", "_main"))
        out[cfg] = {"levels_rows": len(rep), "levels_rows_main": len(dm),
                    "levels_median_max_abs_diff_m": maxdiff(rep["median_m"], rep["median_m_main"]),
                    "levels_n_max_abs_diff": maxdiff(rep["n"], rep["n_main"]),
                    "units_rows": len(uu), "units_median_max_abs_diff_m": maxdiff(uu["median_m"], uu["median_m_main"]),
                    "units_n_pred_unknown_max_abs_diff": maxdiff(uu["n_pred_unknown"], uu["n_pred_unknown_main"]),
                    "units_n_pred_max_abs_diff": maxdiff(uu["n_pred"], uu["n_pred_main"])}
        s, _ = verdicts(LD[LD["config"] == cfg], LQ[(LQ["config"] == "registered") & (LQ["kind"] == "pooled_level")], var)
        f = frozen_verdicts(var)
        cmp = rev2_a.compare_verdicts(f, s)
        out[cfg]["verdicts_equal_frozen"] = bool((~cmp["changed"].astype(bool)).all()) and len(cmp) == len(f)
    q = LQ[(LQ["config"] == "registered") & (LQ["kind"] == "pooled_level")].merge(Q, on=["sector", "bin"],
                                                                                    suffixes=("", "_main"))
    out["floor"] = {"rows": len(q), "rows_main": len(Q), "p95_max_abs_diff_m": maxdiff(q["p95_m"], q["p95_m_main"]),
                    "n_max_abs_diff": maxdiff(q["n"], q["n_main"])}
    exact = all(v[0] == 0 and v[1] == 0 for c in ("shape", "total0") for k, v in out[c].items()
                if k.endswith("max_abs_diff_m") or k.endswith("max_abs_diff")) and \
        out["floor"]["p95_max_abs_diff_m"] == (0.0, 0) and out["floor"]["n_max_abs_diff"] == (0.0, 0) and \
        all(out[c]["levels_rows"] == out[c]["levels_rows_main"] for c in ("shape", "total0")) and \
        out["floor"]["rows"] == out["floor"]["rows_main"] and \
        all(out[c]["verdicts_equal_frozen"] for c in ("shape", "total0"))
    out["exact"] = bool(exact)
    (A / "replica.json").write_text(_json(out), encoding="utf-8")
    print(_json(out))
    return out


def _require_replica():
    r = json.loads((A / "replica.json").read_text(encoding="utf-8"))
    if not r.get("exact"):
        raise SystemExit("the replica check failed: no item 7 or 8 result is reported (D-147)")


def cmd_item7(a) -> dict:
    _require_replica()
    LD, LQ, _ = _levels(a.name)
    run = json.loads((AUDIT / a.name / "run.json").read_text(encoding="utf-8"))
    Qr = LQ[(LQ["config"] == "registered") & (LQ["kind"] == "pooled_level")]
    ref_total, ref_shape = frozen_verdicts("total"), frozen_verdicts("shape")
    tabs, summ = {}, {"c_ref_m": run["c_ref_m"], "counts": {}, "changed": {}, "verdicts": {}}
    summ["counts"]["registered_total"] = rev2_a.sector_counts(ref_total)
    summ["counts"]["registered_shape"] = rev2_a.sector_counts(ref_shape)
    for cfg in ("total_cref", "total_cref06", "total_swot"):
        s, det = verdicts(LD[LD["config"] == cfg], Qr, cfg)
        tabs[f"verdicts_{cfg}"] = _ordered(s)
        cmp = rev2_a.compare_verdicts(ref_total, s)
        tabs[f"compare_{cfg}_vs_total"] = cmp
        summ["counts"][cfg] = rev2_a.sector_counts(s)
        summ["changed"][f"{cfg}_vs_registered_total"] = _changed(cmp)
        summ["verdicts"][cfg] = _verdict_map(s)
    s, _ = verdicts(LD[LD["config"] == "nogeoid"], Qr, "nogeoid")
    tabs["verdicts_nogeoid"] = _ordered(s)
    cmp = rev2_a.compare_verdicts(ref_shape[ref_shape["product"].isin(s["product"].unique())], s)
    tabs["compare_nogeoid_vs_shape"] = cmp
    summ["counts"]["nogeoid"] = rev2_a.sector_counts(s)
    summ["changed"]["nogeoid_vs_shape"] = _changed(cmp)
    summ["verdicts"]["nogeoid"] = _verdict_map(s)
    ng = json.loads((A / "nogeoid_constants.json").read_text(encoding="utf-8"))
    summ["nogeoid_constants"] = ng["products"]
    summ["nogeoid_registered_products"] = ng["registered_products"]
    summ["nogeoid_extra_products"] = ng["extra_products"]
    med = _sector_medians(LD[LD["config"] == "shape"], "shape")
    for cfg in ("total0", "total_cref", "nogeoid"):
        med = med.merge(_sector_medians(LD[LD["config"] == cfg], cfg), on=["product", "sector"], how="left")
    tabs["sector_medians"] = _ordered(med)
    _write("item7", tabs, "item7", summ)
    return summ


def cmd_item8(a) -> dict:
    _require_replica()
    LD, LQ, U = _levels(a.name)
    disp, _, _ = main_tables()
    share = rev2_a.unknown_share(disp)
    share = share.assign(validated=share["sector"].isin(VALIDATED))
    share["_o"] = share["sector"].map({s: i for i, s in enumerate(ORDER)})
    share = share.sort_values(["variant", "product", "_o"], ascending=[False, True, True]).drop(columns="_o")
    # the pass's own counts must equal the frozen ones
    us = U[U["config"] == "shape"].assign(variant="shape")
    share_pass = rev2_a.unknown_share(us)
    chk = share[share["variant"] == "shape"].merge(share_pass, on=["product", "variant", "sector"], suffixes=("", "_pass"))
    counts_equal = bool((chk["n_pred_unknown"] == chk["n_pred_unknown_pass"]).all()
                        and (chk["n_pred_measured"] == chk["n_pred_measured_pass"]).all())
    ref_shape = frozen_verdicts("shape")
    Qf = {f: LQ[(LQ["config"] == f) & (LQ["kind"] == "pooled_level")] for f in ("registered", "matched_i", "matched_ii")}
    tabs = {"unknown_share": share}
    summ = {"unknown_share_counts_equal_pass": counts_equal, "counts": {"registered_shape": rev2_a.sector_counts(ref_shape)},
            "changed": {}, "verdicts": {}}
    for rule, matched in (("shape_i", "matched_i"), ("shape_ii", "matched_ii")):
        for floor in ("registered", matched):
            key = f"{rule}__floor_{'registered' if floor == 'registered' else 'matched'}"
            s, _ = verdicts(LD[LD["config"] == rule], Qf[floor], key)
            tabs[f"verdicts_{key}"] = _ordered(s)
            cmp = rev2_a.compare_verdicts(ref_shape, s)
            tabs[f"compare_{key}"] = cmp
            summ["counts"][key] = rev2_a.sector_counts(s)
            summ["changed"][key] = _changed(cmp)
            summ["verdicts"][key] = _verdict_map(s)
    med = _sector_medians(LD[LD["config"] == "shape"], "registered")
    for cfg in ("shape_i", "shape_ii"):
        med = med.merge(_sector_medians(LD[LD["config"] == cfg], cfg), on=["product", "sector"], how="left")
    tabs["sector_medians_D"] = _ordered(med)
    qm = None
    for f, q in Qf.items():
        t = q.groupby("sector")["p95_m"].median().rename(f"Q_median_{f}_m").reset_index()
        qm = t if qm is None else qm.merge(t, on="sector", how="outer")
    tabs["sector_medians_Q"] = qm
    sh = share[share["variant"] == "shape"]
    summ["unknown_share_shape"] = {p: {r.sector: {"pooled_pct": round(100 * r.pooled_share, 2),
                                                  "unit_median_pct": round(100 * r.unit_share_median, 2),
                                                  "unit_q25_pct": round(100 * r.unit_share_q25, 2),
                                                  "unit_q75_pct": round(100 * r.unit_share_q75, 2)}
                                       for r in g.itertuples()} for p, g in sh.groupby("product")}
    _write("item8", tabs, "item8", summ)
    return summ


def cmd_item11(a) -> dict:
    disp, pooled, Q = main_tables()
    summ = {"runs": {}}
    tabs = {}
    for name in VARIANTS_11:
        d = AUDIT / name
        if not (d / "run.json").exists():
            summ["runs"][name] = {"status": "not run"}
            continue
        run = json.loads((d / "run.json").read_text(encoding="utf-8"))
        LD, LQ = pd.read_csv(d / "levels_D.csv.gz"), pd.read_csv(d / "levels_Q.csv")
        Dv = LD[LD["config"] == "variant"]
        Qv = LQ[(LQ["config"] == "variant") & (LQ["kind"] == "pooled_level")]
        sv, _ = verdicts(Dv, Qv, "variant")
        if run["hybrid"]:
            Dp = LD[LD["config"] == "primary_fp"]
            Qp = LQ[(LQ["config"] == "primary_fp") & (LQ["kind"] == "pooled_level")]
            sp, _ = verdicts(Dp, Qp, "primary_fp")
            ref = "the primary on the same footprint (the 14 R-block tiles), same pass"
        else:
            Dp = pooled[pooled["variant"] == "shape"]
            sp = frozen_verdicts("shape")
            ref = "the primary (results/hindcast/main, shape variant)"
        cmp = rev2_a.compare_verdicts(sp, sv)
        med = (_sector_medians(Dv, "variant").merge(_sector_medians(Dp, "reference"), on=["product", "sector"],
                                                    how="outer"))
        qv = Qv.groupby("sector")["p95_m"].median().rename("Q_median_variant_m").reset_index()
        qr = (Qp if run["hybrid"] else Q).groupby("sector")["p95_m"].median().rename("Q_median_reference_m").reset_index()
        tabs[f"verdicts_{name}"] = _ordered(sv)
        tabs[f"reference_{name}"] = _ordered(sp)
        tabs[f"compare_{name}"] = cmp
        tabs[f"medians_{name}"] = _ordered(med).merge(qv.merge(qr, on="sector", how="outer"), on="sector", how="left")
        summ["runs"][name] = {"stage2": run["stage2"], "hybrid": run["hybrid"], "reference": ref,
                              "seconds": run.get("seconds"), "footprint_cells": run.get("footprint_cells"),
                              "counts_reference": rev2_a.sector_counts(sp), "counts_variant": rev2_a.sector_counts(sv),
                              "n_changed": int(cmp["changed"].sum()),
                              "n_changed_validated": int((cmp["changed"].astype(bool) & cmp["validated"]).sum()),
                              "changed": _changed(cmp), "verdicts_variant": _verdict_map(sv),
                              "verdicts_reference": _verdict_map(sp)}
    _write("item11", tabs, "item11", summ)
    return summ


def cmd_all(a) -> int:
    rep = cmd_replica(a)
    s7 = cmd_item7(a) if rep.get("exact") else None
    s8 = cmd_item8(a) if rep.get("exact") else None
    s11 = cmd_item11(a)
    # every product's verdict in each validated sector under every variant (validated sectors first)
    runs = {"primary_shape": _verdict_map(frozen_verdicts("shape")),
            "primary_total": _verdict_map(frozen_verdicts("total"))}
    if s7:
        for k in ("total_cref", "total_cref06", "total_swot", "nogeoid"):
            runs[f"7_{k}"] = s7["verdicts"][k]
    if s8:
        for k, v in s8["verdicts"].items():
            runs[f"8_{k}"] = v
    for name, r in s11["runs"].items():
        if "verdicts_variant" in r:
            runs[f"11_{name}"] = r["verdicts_variant"]
            if r["hybrid"]:
                runs[f"11_{name}__reference_footprint"] = r["verdicts_reference"]
    matrix = {}
    for run, v in runs.items():
        for p, d in v.items():
            for s in VALIDATED:
                matrix.setdefault(p, {}).setdefault(s, {})[run] = d.get(s)
    survive = {}
    for p, d in matrix.items():
        for s in VALIDATED:
            vals = {k: x for k, x in d.get(s, {}).items() if not k.endswith("__reference_footprint")}
            shape_based = {k: x for k, x in vals.items() if k != "primary_total" and not k.startswith("7_total")}
            survive.setdefault(p, {})[s] = {
                "mispredicted_in_every_run": all(x is True for x in vals.values()),
                "mispredicted_in_every_shape_based_run": all(x is True for x in shape_based.values()),
                "not_mispredicted_or_na_in": sorted(k for k, x in vals.items() if x is not True)}
    out = {"replica_exact": rep.get("exact"), "validated_sectors": VALIDATED, "validated_verdicts": matrix,
           "validated_survival": survive,
           "note": "shape-based runs: the deciding variant and its recomputations (items 7b, 8, 11); the total-variant "
                   "runs (registered total, item 7a) are included only in mispredicted_in_every_run; the R-block "
                   "variants are judged on their 14 tiles only"}
    (A / "summary.json").write_text(_json(out), encoding="utf-8")
    print(_json({p: {s: v for s, v in d.items() if s in ("west", "interfluve")} for p, d in survive.items()}))
    return 0


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["replica", "item7", "item8", "item11", "all"])
    ap.add_argument("--name", default="main")
    a = ap.parse_args(argv)
    f = {"replica": cmd_replica, "item7": cmd_item7, "item8": cmd_item8, "item11": cmd_item11, "all": cmd_all}
    f[a.command](a)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
