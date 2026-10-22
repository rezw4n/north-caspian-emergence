"""Revision analysis G (PREREG Amendment 18 item 11, H2). Exploratory, post-review analyses
of the frozen H2 validation (results/validation/main/); the registered verdict ("not
supported") stands. Seed 20261022 for every new resampling, B = 2,000 (design.BOOTSTRAP_N).

Stages (outputs under results/revision/g/h2/):
  influence   Leave-one-test-block-out and leave-one-sector-out paired differences
              DEM - product (NMAD, RMSE), per source and variant, with the registered
              fits (caspian.rev_g.leave_one_out); the full-sample values are checked
              against the registered paired-bootstrap table.
  intervals   Block-bootstrap intervals resting on fewer than 2 blocks marked "not
              estimable" (caspian.rev_g.mark_not_estimable) in the paired-bootstrap and
              coverage tables of ICESat-2, GLO-30 and GEDI.
  mondrian    q-hat, usable calibration points, the pooled fallback, and calibration
              blocks per sector (with >= 1 and >= 50 usable points).
  blocksizes  12.5, 25 and 50 km blocks on a common point set: points held out of fitting
              at all three sizes (calibration or test blocks at each size). Each
              size's own registered fits; accuracy and paired differences with block-
              bootstrap intervals over that size's blocks.
  rmse        RMSE beside NMAD (Table 1, ICESat-2 column), overall and per sector, from the
              registered accuracy and paired-bootstrap tables.
  splits      The H2 rule on each of the 20 seeded split reassignments (ATL08)
              and on the calibration/test swap.

Usage: python scripts/rev_g_h2.py <stage> [--sources ATL08,ATL06]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import caspian  # noqa: F401
import numpy as np
import pandas as pd

from caspian import conformal, rev_g, validation
from caspian.config import RESULTS
from caspian.geography import SECTOR_ORDER

ROOT = Path(__file__).resolve().parents[1]
VAL = RESULTS / "validation" / "main"
OUT = RESULTS / "revision" / "g" / "h2"
SEED = 20261022
N_BOOT = 2000
PRODUCTS = ("gebco_2026", "etopo_2022", "srtm15p_v2.7", "gebco_2022", "etopo1")
SIZES = {12500.0: VAL / "sensitivity_12500m", 25000.0: VAL / "sensitivity_25000m", 50000.0: VAL}


def _drv():
    spec = importlib.util.spec_from_file_location("a6_validate_dem", ROOT / "scripts" / "a6_validate_dem.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fits(folder: Path, source: str) -> tuple[float, dict]:
    c = json.loads((folder / f"{source}_datum_constant.json").read_text(encoding="utf-8"))["constant"]["constant_m"]
    off = json.loads((folder / f"{source}_product_offsets.json").read_text(encoding="utf-8"))["product_offsets"]
    return float(c), {p: float(v["offset_m"]) for p, v in off.items()}


def _errors(pts: pd.DataFrame, c: float, off: dict) -> tuple[np.ndarray, dict]:
    """DEM error and product errors per variant ("matched": + offset; "total": + c for
    the products documented on EGM2008, otherwise = matched), as validation.run."""
    y = pts["h_orth"].to_numpy(float) + c
    e_dem = pts["dem_h"].to_numpy(float) - y
    e = {}
    for p in PRODUCTS:
        z = pts[f"prod:{p}"].to_numpy(float)
        e[(p, "matched")] = z + off[p] - y
        e[(p, "total")] = (z + c - y) if p in validation.DOCUMENTED_EGM2008 else e[(p, "matched")]
    return e_dem, e


def stage_influence(a) -> int:
    drv = _drv()
    rows_s, rows_d, checks = [], [], []
    for source in a.sources:
        pts = validation.attach_blocks(drv._load_points("main", source, "bare"), drv._blocks(50000.0), 50000.0)
        te = validation.split_rows(pts, "test").reset_index(drop=True)
        del pts
        c, off = _fits(VAL, source)
        e_dem, e = _errors(te, c, off)
        reg = pd.read_csv(VAL / f"{source}_paired_bootstrap.csv")
        reg = reg[reg.scope == "overall"]
        for var in ("matched", "total"):
            ep = {p: e[(p, var)] for p in PRODUCTS}
            ok = np.isfinite(e_dem)
            for v in ep.values():
                ok &= np.isfinite(v)
            full = rev_g.paired_diffs(e_dem[ok], {p: v[ok] for p, v in ep.items()})
            for (p, mt), v in full.items():
                r = reg[(reg["product"] == p) & (reg["metric"] == mt) & (reg["variant"] == var)]
                checks.append({"source": source, "variant": var, "product": p, "metric": mt, "diff": v,
                               "diff_registered": float(r["diff"].iloc[0]), "n": int(ok.sum()),
                               "n_registered": int(r["n_points"].iloc[0])})
            for unit, col in (("block", "block"), ("sector", "sector")):
                g = te[col].to_numpy(object)[ok]
                loo = rev_g.leave_one_out(e_dem[ok], {p: v[ok] for p, v in ep.items()}, g)
                rows_d.append(loo.assign(source=source, variant=var, left_out_unit=unit))
                s = rev_g.summarise_influence(loo)
                rows_s.append(s.assign(source=source, variant=var, left_out_unit=unit))
    OUT.mkdir(parents=True, exist_ok=True)
    tag = "_".join(a.sources)
    s = pd.concat(rows_s, ignore_index=True)
    d = pd.concat(rows_d, ignore_index=True)
    ck = pd.DataFrame(checks)
    s.to_csv(OUT / f"influence_summary_{tag}.csv", index=False, float_format="%.4f")
    d.to_csv(OUT / f"influence_detail_{tag}.csv", index=False, float_format="%.4f")
    ck.to_csv(OUT / f"influence_check_{tag}.csv", index=False, float_format="%.6f")
    print("max |diff - registered|", float((ck["diff"] - ck["diff_registered"]).abs().max()),
          "n mismatch", int((ck["n"] != ck["n_registered"]).sum()))
    print(s[s.variant == "matched"].round(3).to_string(index=False))
    return 0


def stage_intervals(a) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    out = {}
    for folder, src in ((VAL, "ATL06"), (VAL, "ATL08"), (RESULTS / "validation" / "main_glo30", "GLO30"),
                        (RESULTS / "validation" / "main_gedi", "GEDI")):
        pb = pd.read_csv(folder / f"{src}_paired_bootstrap.csv")
        m = rev_g.mark_not_estimable(pb, ["ci_low", "ci_high"])
        m.to_csv(OUT / f"intervals_paired_{src}.csv", index=False, float_format="%.4f")
        cv = pd.read_csv(folder / f"{src}_coverage.csv")
        mc = rev_g.mark_not_estimable(cv, ["boot_low", "boot_high"])
        mc["clopper_pearson_note"] = "treats dependent points as independent; not an interval for this design"
        mc.to_csv(OUT / f"intervals_coverage_{src}.csv", index=False, float_format="%.4f")
        out[src] = {"paired_scopes_not_estimable": sorted(m.loc[m.interval_status.str.startswith("not"), "scope"].unique().tolist()),
                    "coverage_scopes_not_estimable": sorted(mc.loc[mc.interval_status.str.startswith("not"), "scope"].unique().tolist()),
                    "coverage_overall": {k: float(mc.loc[mc.scope == "overall", k].iloc[0])
                                         for k in ("coverage", "ci_low", "ci_high", "boot_low", "boot_high")}}
    (OUT / "intervals.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


def stage_mondrian(a) -> int:
    drv = _drv()
    rows = []
    for source in a.sources:
        pts = validation.attach_blocks(drv._load_points("main", source, "bare"), drv._blocks(50000.0), 50000.0)
        cal = validation.split_rows(pts, "calib").reset_index(drop=True)
        del pts
        c, _ = _fits(VAL, source)
        cal = validation.apply_datum_constant(cal, {"constant_m": c})
        model = json.loads((VAL / f"{source}_scale_model.json").read_text(encoding="utf-8"))["scale_model"]
        s = validation.scale_values(cal, model)
        use = conformal.usable(cal["y_ref"].to_numpy(float), cal["dem_h"].to_numpy(float), s)
        rec = json.loads((VAL / f"{source}_calibration.json").read_text(encoding="utf-8"))["calibration"]
        sec = cal["sector"].to_numpy(object)
        blk = cal["block"].to_numpy(object)
        for g in SECTOR_ORDER:
            m = (sec == g) & use
            per_block = pd.Series(blk[m]).value_counts()
            gr = rec["groups"].get(g, {})
            rows.append({"source": source, "sector": g, "q_hat": gr.get("q"), "n_points": gr.get("n"),
                         "n_points_recount": int(m.sum()), "used_pooled": gr.get("used_pooled"),
                         "n_blocks": int(per_block.size), "n_blocks_ge_50_points": int((per_block >= 50).sum()),
                         "points_per_block_min": int(per_block.min()) if per_block.size else 0,
                         "points_per_block_max": int(per_block.max()) if per_block.size else 0,
                         "pooled_q_hat": rec["pooled"]["q"], "pooled_n_points": rec["pooled"]["n"],
                         "min_n_points_rule": rec.get("min_n")})
    OUT.mkdir(parents=True, exist_ok=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / f"mondrian_{'_'.join(a.sources)}.csv", index=False, float_format="%.4f")
    print(t.to_string(index=False))
    return 0


def stage_blocksizes(a) -> int:
    drv = _drv()
    acc_rows, pb_rows, meta = [], [], {}
    for source in a.sources:
        pts = drv._load_points("main", source, "bare")
        lab = {}
        for size in SIZES:
            q = validation.attach_blocks(pts[["x", "y"]], drv._blocks(size), size)
            lab[size] = q[["block", "sector", "split"]].reset_index(drop=True)
        held = np.ones(len(pts), bool)
        for size in SIZES:
            held &= lab[size]["split"].isin(["calib", "test"]).to_numpy()
        common = pts.loc[held].reset_index(drop=True)
        meta[source] = {"n_common_points": int(held.sum()),
                        "n_blocks": {f"{int(s)}": int(lab[s].loc[held, "block"].nunique()) for s in SIZES},
                        "n_test_all_three": int(np.logical_and.reduce([lab[s]["split"].eq("test").to_numpy() for s in SIZES]).sum())}
        for size, folder in SIZES.items():
            c, off = _fits(folder, source)
            l = lab[size].loc[held].reset_index(drop=True)
            e_dem, e = _errors(common, c, off)
            df = pd.DataFrame({"block": l["block"], "sector": l["sector"], "split": "test",
                               "y_ref": common["h_orth"].to_numpy(float) + c,
                               "dem_h": common["dem_h"].to_numpy(float)})
            for (p, var), v in e.items():
                df[f"{p}|{var}"] = v + df["y_ref"].to_numpy()
            ok = np.isfinite(e_dem)
            for v in e.values():
                ok &= np.isfinite(v)
            for model, err in [("dem", e_dem)] + [(f"{p}|{var}", v) for (p, var), v in e.items()]:
                acc_rows.append({"source": source, "block_size_m": size, "model": model, "c_m": c,
                                 **validation.error_metrics(err[ok]), "n_blocks": int(l.loc[ok, "block"].nunique())})
            for var in ("matched", "total"):
                b = validation.paired_block_bootstrap(df.loc[ok].reset_index(drop=True), "dem_h",
                                                      {p: f"{p}|{var}" for p in PRODUCTS}, n_boot=N_BOOT, seed=SEED)
                pb_rows.append(b.assign(variant=var, source=source, block_size_m=size))
        del pts, common
    OUT.mkdir(parents=True, exist_ok=True)
    tag = "_".join(a.sources)
    acc = pd.DataFrame(acc_rows)
    pb = pd.concat(pb_rows, ignore_index=True)
    acc.to_csv(OUT / f"blocksizes_common_accuracy_{tag}.csv", index=False, float_format="%.4f")
    pb.to_csv(OUT / f"blocksizes_common_paired_{tag}.csv", index=False, float_format="%.4f")
    dec = {}
    for (src, size, var), g in pb.groupby(["source", "block_size_m", "variant"]):
        dec.setdefault(src, {}).setdefault(f"{int(size)}", {})[var] = validation.h2_decision(g, PRODUCTS)["overall"]
    (OUT / f"blocksizes_common_{tag}.json").write_text(json.dumps({"rule": "D-129: points held out of fitting at "
                                                                   "12.5, 25 and 50 km; each size's own fits",
                                                                   "meta": meta, "rule_outcome_descriptive": dec},
                                                                  indent=1), encoding="utf-8")
    print(json.dumps(meta, indent=1), json.dumps(dec, indent=1))
    show = acc[acc.model.isin(["dem", "gebco_2026|matched", "etopo_2022|matched", "etopo1|matched"])]
    print(show[["source", "block_size_m", "model", "n", "nmad", "rmse", "bias", "n_blocks"]].round(3).to_string(index=False))
    return 0


def stage_rmse(a) -> int:
    rows = []
    for source in ("ATL06", "ATL08"):
        acc = pd.read_csv(VAL / f"{source}_accuracy.csv")
        pb = pd.read_csv(VAL / f"{source}_paired_bootstrap.csv")
        pb = rev_g.mark_not_estimable(pb, ["ci_low", "ci_high"])
        for scope, g in acc.groupby("scope", sort=False):
            for _, r in g.iterrows():
                model = r["model"]
                p, _, var = model.partition("|")
                row = {"source": source, "scope": scope, "model": model, "n": int(r["n"]), "n_blocks": int(r["n_blocks"]),
                       "nmad_m": r["nmad"], "rmse_m": r["rmse"], "bias_m": r["bias"]}
                if model != "dem":
                    for mt in ("nmad", "rmse"):
                        x = pb[(pb.scope == scope) & (pb["product"] == p) & (pb.metric == mt) & (pb.variant == var)]
                        if len(x):
                            row.update({f"diff_{mt}": float(x["diff"].iloc[0]), f"diff_{mt}_ci_low": x["ci_low"].iloc[0],
                                        f"diff_{mt}_ci_high": x["ci_high"].iloc[0],
                                        "interval_status": x["interval_status"].iloc[0]})
                rows.append(row)
    t = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "rmse_beside_nmad.csv", index=False, float_format="%.4f")
    o = t[(t.scope == "overall") & ~t.model.str.endswith("|total")]
    print(o[["source", "model", "n", "nmad_m", "rmse_m", "bias_m", "diff_rmse", "diff_rmse_ci_low", "diff_rmse_ci_high"]]
          .round(3).to_string(index=False))
    return 0


def stage_splits(a) -> int:
    base = RESULTS / "robustness" / "r08_blocking"
    rows = []
    for d in sorted(base.glob("split_*")):
        f = d / "validation" / "ATL08_alt_paired_bootstrap.csv"
        if not f.exists():
            continue
        pb = pd.read_csv(f)
        h = pd.read_csv(d / "validation" / "ATL08_headline_alt.csv")
        get = lambda t, s, i, q: float(h[(h.table == t) & (h.scope == s) & (h["item"] == i) & (h.quantity == q)]["value"].iloc[0])
        r = {"variant": d.name, "n_boot": int(pb["n_boot"].iloc[0]), "seed": int(pb["seed"].iloc[0]),
             "dem_nmad_m": get("accuracy", "overall", "dem", "nmad"), "coverage": get("coverage", "overall", "dem", "coverage"),
             "n_test_blocks": int(pb[pb.scope == "overall"]["n_blocks"].iloc[0])}
        for var in ("matched", "total"):
            r[f"h2_{var}"] = validation.h2_decision(pb[pb.variant == var], PRODUCTS)["overall"]
        rows.append(r)
    t = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "split_draws.csv", index=False, float_format="%.4f")
    seeds = t[t.variant.str.startswith("split_seed")]
    main_nmad = float(pd.read_csv(VAL / "ATL08_accuracy.csv").query("scope == 'overall' and model == 'dem'")["nmad"].iloc[0])
    main_cov = float(pd.read_csv(VAL / "ATL08_coverage.csv").query("scope == 'overall'")["coverage"].iloc[0])
    summ = {"source": "ATL08 only (the D-104 reassignments did not run ATL06)", "n_seeds": int(len(seeds)),
            "n_supported_matched": int((seeds.h2_matched == "supported").sum()),
            "n_supported_total": int((seeds.h2_total == "supported").sum()),
            "n_not_supported_matched": int((seeds.h2_matched == "not supported").sum()),
            "supported_seeds": seeds.loc[seeds.h2_matched == "supported", "variant"].tolist(),
            "registered_draw_dem_nmad_m": main_nmad, "registered_draw_coverage": main_cov,
            "n_seeds_lower_dem_nmad_than_registered": int((seeds.dem_nmad_m < main_nmad).sum()),
            "n_seeds_higher_coverage_than_registered": int((seeds.coverage > main_cov).sum()),
            "swap": t[t.variant == "split_swap_calib_test"].to_dict(orient="records"),
            "note": "the registered H2 decision needs both ATL06 and ATL08 (D-069); with ATL08 alone 'not supported' "
                    "in a draw implies 'not supported' for that draw, while an ATL08 'supported' is at most 'partial "
                    "support' until ATL06 is run"}
    (OUT / "split_draws.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(t.round(3).to_string(index=False))
    print(json.dumps(summ, indent=1))
    return 0


def main(argv) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["influence", "intervals", "mondrian", "blocksizes", "rmse", "splits"])
    ap.add_argument("--sources", default="ATL08,ATL06")
    a = ap.parse_args(argv)
    a.sources = a.sources.split(",")
    return {"influence": stage_influence, "intervals": stage_intervals, "mondrian": stage_mondrian,
            "blocksizes": stage_blocksizes, "rmse": stage_rmse, "splits": stage_splits}[a.stage](a)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
