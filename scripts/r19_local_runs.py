"""Robustness: the local checks of docs/ROBUSTNESS_PLAN.md on the primary outputs
(tag main; D-077 to D-088), one command per check. No Earth Engine. Every command
reads the primary outputs, never writes into them, and writes its tables and a short
summary.json to results/robustness/<check>/ (git-tracked folder; large intermediates
go to data/, git-ignored).

Primary inputs: data/interim/stage2/main (Landsat counts, D-079), data/interim/
waterlines/main/points.csv.gz (with `static`), data/processed/dem/main (+ emergence),
data/interim/validation/main (prepared ICESat-2 points and their D-084 annotations),
results/validation/main (H2 at 50 km, D-085/D-086), results/hindcast/main (H1, D-087).

Commands (in the coordinator's priority order)
  h1-e2e3      H1 hindcast on E2 + E3 only: E1 dropped from every unit, product side
               and noise floor alike; shape constants = the primary's leave-one-epoch-out
               refit without E1 (equal to a refit on the E2 + E3 units); envelope D-082
               and outside states D-083 as the primary
  nostatic     DEM without the static-contour exclusion (D-034, D-075), and H2 at the
               final 50 km blocks, paired on the primary's points (bare population)
  lag365       the ICESat-2 dryness test with emergence + 365 days (D-069)
  nearest      products resampled by nearest neighbour (D-070): H2 (product heights at
               the points) and, with --hindcast, H1 (new heights and flood maps)
  dn-total     H1 total variant with N(EGM2008) without NGA's -0.41 m offset (D-074);
               the shape variant cannot change and is not rerun
  gp           TIN vs ordinary kriging (check 7) on a seeded subsample of 50 km
               training and test blocks
  bw020        0.20 m bins by merging the 0.10 m counts (check 7): waterlines, DEM, H2
  levels       level series and interpolation, first order (check 5): Hydroweb with one
               constant offset, DAHITI with PCHIP, the +-1 year gap-fill window
  setupcorr    setup correction instead of screening, first order (check 4)
  komsomolets  Komsomolets waterline statistics with and without screening, from the
               D-064 sensitivity blocks (local part of the D-066 check)
  morphology   check 10: morphology table and the rising/falling-limb difference
  prior        check 11: the area-level slope against Akbari et al. (2020)
  rblock       an R-block alternative exported by scripts/r18_rblock_export.py: hybrid
               TIN on R and the R-local paired comparison (cmd_rblock)
  datums       check 6, validation side: c and the product offsets at +-1 sigma, and
               the two mean-tide recipes of D-073
  vegetation   check 9 bounds (--mode as_dry | as_water) from the o band: waterlines,
               DEM, H2
  validate     add a source to an existing variant's validation and summary

Usage: python scripts/r19_local_runs.py <command> [--n-boot N] [--hindcast]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import caspian  # noqa: F401
import numpy as np
import pandas as pd

from caspian import design, levels, robustness as R, validation
from caspian.config import INTERIM, PROCESSED, RESULTS
from caspian.geography import SECTOR_ORDER

MAIN = "main"
ROB = RESULTS / "robustness"
CORR_PRIMARY = RESULTS / "study_area" / "geoid_correction_primary.npz"
CORR_NO_CONSTANT = RESULTS / "study_area" / "geoid_correction_no_constant.npz"
LABELS = RESULTS / "levels" / "sector_labels.csv"
PLAN = INTERIM / "stage2" / "plan_main_tau9900cm_landsat.csv"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


P = _load("r00_pipeline")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _json(obj) -> str:
    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (pd.Timestamp,)):
            return str(o)
        return str(o)
    return json.dumps(obj, indent=1, default=conv)


def write_summary(check: str, rec: dict) -> None:
    d = ROB / check
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(_json({"check": check, "created": _now(), **rec}), encoding="utf-8")
    print(_json(rec)[:3000])


# --- lean I/O (memory is shared with two large jobs) ---------------------------------------

SLIM = ["x", "y", "epoch", "bin", "z_m", "static", "level_sigma_p50_m"]


def read_points_slim(tag: str = MAIN, cols=SLIM, epochs=None, chunk: int = 2_000_000) -> pd.DataFrame:
    """The contour table with only the columns the DEM and its variants need, read in
    chunks (scripts/a6_dem.py reads x, y, epoch, z_m and static only)."""
    path = INTERIM / "waterlines" / tag / "points.csv.gz"
    head = pd.read_csv(path, nrows=0).columns
    use = [c for c in cols if c in head]
    parts = []
    for d in pd.read_csv(path, usecols=use, chunksize=chunk):
        parts.append(d if epochs is None else d[d["epoch"].isin(epochs)])
    return pd.concat(parts, ignore_index=True)


def write_points(tag: str, pts: pd.DataFrame) -> Path:
    d = INTERIM / "waterlines" / tag
    d.mkdir(parents=True, exist_ok=True)
    pts.to_csv(d / "points.csv.gz", index=False, float_format="%.6f")
    return d / "points.csv.gz"


# --- summaries ---------------------------------------------------------------------------------

def h1_summary(variant_dir: Path) -> dict:
    """Per product and variant: sectors where it mispredicts (primary vs variant), the
    sectors that changed, the block permutation test (statistic, p) of both runs, and
    the median over levels of the per-level median displacement per sector."""
    c = pd.read_csv(variant_dir / "h1_compare.csv")
    out = {}
    for (p, v), g in c.groupby(["product", "variant"]):
        g = g[g["in_both"]]
        main_in = sorted(g.loc[g["mispredicts_main"].astype(str) == "True", "sector"])
        alt_in = sorted(g.loc[g["mispredicts_alt"].astype(str) == "True", "sector"])
        out[f"{p}|{v}"] = {"mispredicts_in_primary": main_in, "mispredicts_in_variant": alt_in,
                           "changed_sectors": sorted(g.loc[g["changed"], "sector"]),
                           "n_exceeding/n_evaluable_variant": {r.sector: f"{int(r.n_exceeding_alt)}/{int(r.n_evaluable_alt)}"
                                                              for r in g.itertuples() if np.isfinite(r.n_evaluable_alt)}}
    bt = variant_dir / "block_test_compare.csv"
    if bt.exists():
        b = pd.read_csv(bt)
        fixed = RESULTS / "hindcast" / MAIN
        if "main_fixed" not in set(b["run"]) and (fixed / "d089_rerun_comparison.json").exists():
            from caspian import h1 as _h1   # primary block test from the D-089 rerun's block table
            b = pd.concat([b, _h1.block_test(pd.read_csv(fixed / "block_h1.csv"), pd.read_csv(LABELS))
                           .assign(run="main_fixed")], ignore_index=True)
            b.to_csv(bt, index=False)
        for (p, v), g in b.groupby(["product", "variant"]):
            k = f"{p}|{v}"
            out.setdefault(k, {})
            for run, gg in g.groupby("run"):
                out[k][f"block_test_{run}"] = {"statistic_m": float(gg["statistic"].iloc[0]),
                                               "p": float(gg["p_value"].iloc[0]),
                                               "significant": bool(gg["significant"].iloc[0])}
    disp_alt = pd.read_csv(variant_dir / "displacement.csv")
    disp_main = pd.read_csv(RESULTS / "hindcast" / MAIN / "displacement.csv")
    rows = []
    for name, dd in (("median_m_primary", disp_main), ("median_m_variant", disp_alt)):
        q = dd[(dd["epoch"] == "pooled") & dd["sector"].isin(SECTOR_ORDER) & dd["observed_level"].astype(str).eq("True")]
        rows.append(q.groupby(["product", "variant", "sector"])["median_m"].median().rename(name))
    med = pd.concat(rows, axis=1).reset_index()
    med.to_csv(variant_dir / "median_displacement_by_sector.csv", index=False)
    return {"by_product": out, "median_displacement_table": str(variant_dir / "median_displacement_by_sector.csv")}


def h2_summary(check: str, variant: str) -> dict:
    """Per source: DEM test-block bias, RMSE, NMAD (overall and per sector), coverage
    and the H2 decision (validation.h2_decision, both variants) of primary and variant."""
    vd = ROB / check / variant / "validation"
    out = {}
    for source in ("ATL06", "ATL08"):
        f = vd / f"{source}_compare.csv"
        if not f.exists():
            continue
        c = pd.read_csv(f)
        acc = c[(c["table"] == "accuracy") & (c["item"] == "dem") & c["quantity"].isin(["n", "bias", "rmse", "nmad"])]
        cov = c[(c["table"] == "coverage") & (c["quantity"] == "coverage")]
        rec = {"dem": {f"{r.scope}|{r.quantity}": [r.value_main, r.value_alt] for r in acc.itertuples()},
               "coverage": {r.scope: [r.value_main, r.value_alt] for r in cov.itertuples()}}
        alt_pb = pd.read_csv(vd / f"{source}_alt_paired_bootstrap.csv")
        main_pb = pd.read_csv(RESULTS / "validation" / MAIN / f"{source}_paired_bootstrap.csv")
        for v in ("matched", "total"):
            prods = sorted(main_pb.loc[main_pb["variant"] == v, "product"].unique())
            dm = validation.h2_decision(main_pb[main_pb["variant"] == v], prods)
            da = validation.h2_decision(alt_pb[alt_pb["variant"] == v], prods)
            rec[f"h2_{v}"] = {"primary": dm["overall"], "variant": da["overall"],
                              "sectors_supported_primary": dm["sectors_supported"],
                              "sectors_supported_variant": da["sectors_supported"]}
        out[source] = rec
    return out


# --- commands ----------------------------------------------------------------------------------

def cmd_h1_e2e3(a) -> int:
    out = ROB / "h1_e2e3" / "main__e2e3" / "hindcast"
    if not a.summary_only:
        out = P.hindcast_step("h1_e2e3", "main__e2e3", MAIN, stage2_tag=MAIN, vertical_correction=str(CORR_PRIMARY),
                              shape="loeo:E1", sector_labels=str(LABELS), drop_epochs=("E1",),
                              note="H1 on E2 + E3 only: E1 dropped from the product side and the noise floor")
    write_summary("h1_e2e3", {"variant_dir": str(out), **h1_summary(out),
                              "design": "E1 units and the E1 noise-floor rows (halves, E1 vs E2/E3) dropped; shape "
                                        "constants = primary leave-one-epoch-out refit without E1"})
    return 0


def cmd_nostatic(a) -> int:
    p = read_points_slim(cols=["x", "y", "epoch", "z_m", "static"])
    variant = f"{MAIN}__nostatic"
    e23 = p["epoch"].isin(["E2", "E3"])
    n_static = int(p.loc[e23, "static"].astype(bool).sum())
    if not a.skip_dem:
        write_points(variant, p.drop(columns=["static"]))
    del p
    if not a.skip_dem:
        P.dem_step(variant)
    diff = P.dem_difference("r12_static_rule", variant, MAIN)
    P.compare_validation("r12_static_rule", variant, MAIN, alt_points=P.variant_dem_points(variant), n_boot=a.n_boot, sources=a.sources,
                         note="DEM without the static-contour exclusion; points and dryness of the primary")
    write_summary("r12_static_rule", {"variant": variant, "n_points_e23": int(e23.sum()), "n_static_e23": n_static,
                                      "dem_difference_overall": diff.set_index("sector").loc["overall"].to_dict(),
                                      "h2": h2_summary("r12_static_rule", variant), "n_boot": a.n_boot})
    return 0


# --- results the lead ran, folded into the D-092 verdicts (read-only on them) ----------------------

def h2_from_folders(main_dir: Path, alt_dir: Path) -> dict:
    """h2_summary's structure from two validation folders as written by
    scripts/a6_validate_dem.py (e.g. the D-088 block-size sensitivities)."""
    out = {}
    for source in ("ATL06", "ATL08"):
        if not (alt_dir / f"{source}_accuracy.csv").exists():
            continue
        rec = {"dem": {}, "coverage": {}}
        acc = {k: pd.read_csv(d / f"{source}_accuracy.csv") for k, d in (("m", main_dir), ("a", alt_dir))}
        for q in ("n", "bias", "rmse", "nmad"):
            vals = [float(t.loc[(t["scope"] == "overall") & (t["model"] == "dem"), q].iloc[0]) for t in acc.values()]
            rec["dem"][f"overall|{q}"] = vals
        cov = [float(pd.read_csv(d / f"{source}_coverage.csv").set_index("scope").loc["overall", "coverage"])
               for d in (main_dir, alt_dir)]
        rec["coverage"]["overall"] = cov
        main_pb = pd.read_csv(main_dir / f"{source}_paired_bootstrap.csv")
        alt_pb = pd.read_csv(alt_dir / f"{source}_paired_bootstrap.csv")
        for v in ("matched", "total"):
            prods = sorted(main_pb.loc[main_pb["variant"] == v, "product"].unique())
            dm = validation.h2_decision(main_pb[main_pb["variant"] == v], prods)
            da = validation.h2_decision(alt_pb[alt_pb["variant"] == v], prods)
            rec[f"h2_{v}"] = {"primary": dm["overall"], "variant": da["overall"],
                              "sectors_supported_primary": dm["sectors_supported"],
                              "sectors_supported_variant": da["sectors_supported"]}
        out[source] = rec
    return out


def h1_from_folders(main_dir: Path, alt_dir: Path, main_variant: str, alt_variant: str) -> dict:
    """h1_summary's by_product structure from two hindcast folders (h1_mispredicts.csv,
    h1_block_test.csv), comparing the primary's ``main_variant`` rows with the other
    run's ``alt_variant`` rows (e.g. the red-team sector-shape run writes its variant as
    "total")."""
    mm, ma = (pd.read_csv(d / "h1_mispredicts.csv") for d in (main_dir, alt_dir))
    bm, ba = (pd.read_csv(d / "h1_block_test.csv") for d in (main_dir, alt_dir))
    out = {}
    for p in sorted(set(mm["product"]) & set(ma["product"])):
        gm = mm[(mm["product"] == p) & (mm["variant"] == main_variant)].set_index("sector")["mispredicts"]
        ga = ma[(ma["product"] == p) & (ma["variant"] == alt_variant)].set_index("sector")["mispredicts"]
        both = sorted(set(gm.index) & set(ga.index))
        t = lambda s: s.astype(str) == "True"   # noqa: E731
        rec = {"mispredicts_in_primary": sorted(s for s in both if t(gm[[s]]).iloc[0]),
               "mispredicts_in_variant": sorted(s for s in both if t(ga[[s]]).iloc[0]),
               "changed_sectors": sorted(s for s in both if t(gm[[s]]).iloc[0] != t(ga[[s]]).iloc[0])}
        for run, b, var in (("main_fixed", bm, main_variant), ("alt", ba, alt_variant)):
            r = b[(b["product"] == p) & (b["variant"] == var)]
            if len(r):
                rec[f"block_test_{run}"] = {"statistic_m": float(r["statistic"].iloc[0]),
                                            "p": float(r["p_value"].iloc[0]),
                                            "significant": bool(str(r["significant"].iloc[0]) == "True")}
        out[f"{p}|{main_variant}"] = rec
    return {"by_product": out}


def cmd_fold(a) -> int:
    """Fold the lead's runs into robustness summaries, read-only on them:
    check 8 from the D-088 block-size sensitivities (results/validation/main/
    sensitivity_25000m, sensitivity_12500m; their test blocks differ from the 50 km
    ones, so the magnitude criterion does not apply); check 6 from the red-team
    per-sector datum run (results/hindcast/main_sectorshape; its "total" rows are the
    sector-shape variant, compared with the primary's shape variant, D-072)."""
    vmain = RESULTS / "validation" / MAIN
    dirs = {"block_25km": "sensitivity_25000m", "block_12_5km": "sensitivity_12500m"}
    h2 = {k: h2_from_folders(vmain, vmain / d) for k, d in dirs.items() if (vmain / d).exists()}
    # the calibration-split check (scripts/r08_blocking.py splits: seeded reassignments of the
    # final-size blocks and the calibration/test swap), where run; other test points again
    splits = {d.name: h2_summary("r08_blocking", d.name) for d in sorted((ROB / "r08_blocking").glob("split_*"))
              if (d / "validation" / "run.json").exists()}
    write_summary("r08_blocking", {"h2": h2, "same_points": False,
                                   "sources": {k: f"results/validation/main/{dirs[k]}" for k in h2},
                                   "split_reassignment": ({"variants": sorted(splits), "design": "scripts/r08_blocking.py "
                                                           "splits: seeds 1..N of caspian.robustness.reassign_splits "
                                                           "and the calibration/test swap; post-hoc sensitivity that "
                                                           "reuses test blocks for fitting"} if splits else "not run"),
                                   "splits": splits})
    hm, hs = RESULTS / "hindcast" / MAIN, RESULTS / "hindcast" / "main_sectorshape"
    if (hs / "h1_mispredicts.csv").exists():
        rec = h1_from_folders(hm, hs, "shape", "total")
        write_summary("r06_datums_sectorshape", {**rec, "source": "results/hindcast/main_sectorshape",
                                                 "note": "per-sector D-070 constants (red team); its 'total' rows "
                                                         "against the primary's shape rows"})
    return 0


def cmd_lag365(a) -> int:
    r13 = _load("r13_emergence_lag")
    em = validation.TileSet(PROCESSED / "dem" / MAIN / "emergence", "emergence")
    variant = "lag365d"
    rec = P.compare_validation("r13_emergence_lag", variant, MAIN,
                               alt_points=lambda pts, source: r13.stricter(pts, em, source, 365.0), n_boot=a.n_boot,
                               sources=a.sources,
                               note="dryness test with emergence + 365 days")
    write_summary("r13_emergence_lag", {"variant": variant, "points": rec["sources"],
                                        "h2": h2_summary("r13_emergence_lag", variant), "n_boot": a.n_boot})
    return 0


def cmd_nearest(a) -> int:
    r14 = _load("r14_product_resampling")
    if not a.skip_validation:
        P.compare_validation("r14_product_resampling", "nearest", MAIN, alt_points=r14.nearest_products,
                             n_boot=a.n_boot, sources=a.sources, note="product heights at the points by nearest neighbour")
    rec = {"h2": h2_summary("r14_product_resampling", "nearest"), "n_boot": a.n_boot}
    if a.hindcast:
        out = P.hindcast_step("r14_product_resampling", "nearest", MAIN, stage2_tag=MAIN,
                              vertical_correction=str(CORR_PRIMARY), resampling="nearest", shape="fit",
                              sector_labels=str(LABELS), note="products resampled by nearest neighbour")
        rec["h1"] = h1_summary(out)
    write_summary("r14_product_resampling", rec)
    return 0


def cmd_dn_total(a) -> int:
    with np.load(CORR_PRIMARY) as z1, np.load(CORR_NO_CONSTANT) as z2:
        d = z1["dN"].astype(float) - z2["dN"].astype(float)
    out = ROB / "r06_datums" / "total_dN_m" / "hindcast"
    if not a.summary_only:
        out = P.hindcast_step("r06_datums", "total_dN_m", MAIN, stage2_tag=MAIN,
                              vertical_correction=str(CORR_NO_CONSTANT), correction_is_variant=True, shape="none",
                              sector_labels=str(LABELS), note="total error without NGA's -0.41 m offset (D-074)")
    write_summary("r06_datums", {"variant_dir": str(out),
                                 "primary_minus_variant_correction_m": {"mean": float(np.nanmean(d)),
                                                                        "sd": float(np.nanstd(d))},
                                 "shape_variant": "unchanged by construction: the correction files differ by a "
                                                  "near-constant (sd below the 0.01 m grid of the shape constant)",
                                 **h1_summary(out)})
    return 0


def _alt_fits(check: str, variant: str, source: str) -> dict:
    """The datum constant and product offsets a variant's validation used."""
    f = ROB / check / variant / "validation" / f"{source}_alt_fits.json"
    if not f.exists():
        return {}
    r = json.loads(f.read_text(encoding="utf-8"))
    out = {"constant_m": r["constant"]["constant_m"], "constant_sigma_boot_m": r["constant"].get("sigma_boot_m")}
    for k, v in (r.get("product_offsets") or {}).items():
        out[f"offset_m|{k}"] = v["offset_m"]
    return out


def cmd_datums(a) -> int:
    """Check 6, validation side (the design of scripts/r06_datums.py validation): the
    ICESat-2 -> DEM constant c at +-1 bootstrap sigma; the product offsets at +-1 sigma;
    the ICESat-2 heights in a mean-tide system by either D-073 recipe. Paired on the
    primary's points (bare population, final blocks); the primary from its files. Writes
    results/robustness/r06_datums_validation/summary.json (dn-total keeps
    r06_datums/summary.json); the variant tables go to r06_datums/<variant>/validation."""
    check = "r06_datums"
    path = ROB / "r06_datums_validation" / "summary.json"
    rec = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"variants": {}, "fits": {}}
    runs = {}
    for n_sigma in (-1.0, 1.0):
        def run_c(p, pr, n_sigma=n_sigma):
            c = R.shifted_constant(validation.fit_datum_constant(p, a.n_boot), n_sigma)
            return R.run_fixed(p, pr, constant=c, n_boot=a.n_boot)

        def run_o(p, pr, n_sigma=n_sigma):
            c = validation.fit_datum_constant(p, a.n_boot)
            off = validation.fit_product_offsets(validation.apply_datum_constant(p, c), pr, a.n_boot)
            return R.run_fixed(p, pr, constant=c, product_offsets=R.shifted_offsets(off, n_sigma), n_boot=a.n_boot)
        runs[f"c_{n_sigma:+g}sigma"] = {"alt_run": run_c, "note": f"ICESat-2 -> DEM constant c {n_sigma:+g} sigma"}
        runs[f"offsets_{n_sigma:+g}sigma"] = {"alt_run": run_o, "note": f"product offsets {n_sigma:+g} sigma"}
    for recipe in ("both_mean", "geoid_mean"):
        def tide(p, source, recipe=recipe):
            return p.assign(h_orth=p["h_orth"].to_numpy(float) + R.tide_system_shift(p["lat"], recipe))
        runs[f"tide_{recipe}"] = {"alt_points": tide, "note": f"ICESat-2 heights in a mean-tide system ({recipe}, D-073)"}
    for name, kw in runs.items():
        note = kw.pop("note")
        P.compare_validation(check, name, MAIN, sources=a.sources, n_boot=a.n_boot, note=note, **kw)
        rec["variants"][name] = h2_summary(check, name)
        rec["fits"][name] = {s: _alt_fits(check, name, s) for s in ("ATL06", "ATL08")
                             if (ROB / check / name / "validation" / f"{s}_alt_fits.json").exists()}
        rec["n_boot"] = a.n_boot
        write_summary("r06_datums_validation", {k: v for k, v in rec.items() if k not in ("check", "created")})
    return 0


# --- GP / kriging on a block subsample ------------------------------------------------------------

def cmd_gp(a) -> int:
    import rasterio
    from caspian import blocking
    r07 = _load("r07_interpolation")
    a6v = P.script("a6_validate_dem")
    size = P.final_block_size(MAIN)
    blocks = a6v._blocks(size)
    bi_all, bj_all = [], []
    for d_ in pd.read_csv(a6v._cache(MAIN, "ATL06"), usecols=["x", "y"], chunksize=2_000_000):
        bi_all.append(np.floor(d_["x"].to_numpy() / size).astype(int))
        bj_all.append(np.floor(d_["y"].to_numpy() / size).astype(int))
    ann = np.load(INTERIM / "validation" / MAIN / "ATL06_annotations.npz")
    bare = ann["in_band"] & (ann["veg_share"] < a6v.VEG_CELL_SHARE)
    bk = pd.DataFrame({"i": np.concatenate(bi_all)[bare], "j": np.concatenate(bj_all)[bare]})
    counts = bk.value_counts().rename("n").reset_index().merge(blocks[["i", "j", "split"]], on=["i", "j"])
    counts["block"] = counts["i"].astype(str) + "_" + counts["j"].astype(str)
    del bi_all, bj_all, bk
    rng = np.random.default_rng(design.SEED)
    pick = []
    for split, k in (("test", a.gp_test), ("train", a.gp_train)):
        g = counts[(counts["split"] == split) & (counts["n"] >= 1000)].sort_values("block")
        pick += g.iloc[rng.permutation(len(g))[:k]]["block"].tolist()
    p = read_points_slim(epochs=("E2", "E3"))
    p = p[~p["static"].astype(bool)].reset_index(drop=True)
    split = blocking.split_of_points(p["x"], p["y"], blocks, blocks["split"], size)
    tr = r07.thinned(p[split == "train"], a.thin_m)
    cov = R.fit_covariance(tr[["x", "y"]].to_numpy(float), tr["z_m"].to_numpy(float))
    sig = float(np.nanmedian(p["level_sigma_p50_m"])) if "level_sigma_p50_m" in p.columns else 0.05
    nugget = sig ** 2 + design.BIN_WIDTH_M ** 2 / 12
    t = r07.thinned(p, a.thin_m)
    xy, z = t[["x", "y"]].to_numpy(float), t["z_m"].to_numpy(float)
    del p, tr
    variant = f"{MAIN}__gp_sample"
    src, out = PROCESSED / "dem" / MAIN, PROCESSED / "dem" / variant
    out.mkdir(parents=True, exist_ok=True)
    sel_codes = np.array([int(b.split("_")[0]) * 100_000 + int(b.split("_")[1]) for b in pick], np.int64)
    n_cells = 0
    for f in sorted(src.glob("dem_*.tif")):
        with rasterio.open(f) as r:
            h, dist, prof, trf = r.read(1), r.read(2), r.profile, r.transform
        rows, cols = np.nonzero(np.isfinite(h))
        qx, qy = trf.c + trf.a * (cols + 0.5), trf.f + trf.e * (rows + 0.5)
        code = np.floor(qx / size).astype(np.int64) * 100_000 + np.floor(qy / size).astype(np.int64)
        keep = np.isin(code, sel_codes)
        if not keep.any():
            continue
        zk, var = R.ordinary_kriging(xy, z, np.column_stack([qx[keep], qy[keep]]), sill=cov["sill"], a=cov["a"],
                                     nugget=nugget, k=32, per_quadrant=8, chunk=5000)
        g, sd = np.full(h.shape, np.nan, np.float32), np.full(h.shape, np.nan, np.float32)
        g[rows[keep], cols[keep]], sd[rows[keep], cols[keep]] = zk, np.sqrt(np.maximum(var, 0))
        prof.update(count=3, dtype="float32", nodata=np.nan)
        with rasterio.open(out / f.name, "w", **prof) as w:
            w.write(g, 1)
            w.write(dist.astype(np.float32), 2)
            w.write(sd, 3)
        n_cells += int(keep.sum())
        print(f.name, int(keep.sum()), "cells kriged", flush=True)
    write_points(variant, read_points_slim(cols=["x", "y", "epoch", "static"]))   # the same contours (density)
    # paired comparison on the sampled blocks only, with the primary's datum constant for both
    rows = []
    for source in ("ATL06", "ATL08"):
        keep_fn = (lambda d_: np.isin(np.floor(d_["x"].to_numpy() / size).astype(np.int64) * 100_000
                                      + np.floor(d_["y"].to_numpy() / size).astype(np.int64), sel_codes))
        base = validation.attach_blocks(P.load_points_light(MAIN, source, row_filter=keep_fn), blocks, size)
        base = base[base["block"].isin(pick)].reset_index(drop=True)
        gpp = P.variant_dem_points(variant)(base, source)
        c = json.loads((RESULTS / "validation" / MAIN / f"{source}_datum_constant.json").read_text(encoding="utf-8"))
        y = base["h_orth"].to_numpy(float) + c["constant"]["constant_m"]
        e_tin, e_gp = base["dem_h"].to_numpy(float) - y, gpp["dem_h"].to_numpy(float) - y
        offp = json.loads((RESULTS / "validation" / MAIN / f"{source}_product_offsets.json").read_text(
            encoding="utf-8"))["product_offsets"]
        e_prod = {pn: base[f"prod:{pn}"].to_numpy(float) + o["offset_m"] - y for pn, o in offp.items()}
        ok = np.isfinite(e_tin) & np.isfinite(e_gp)
        for pv in e_prod.values():
            ok &= np.isfinite(pv)
        for scope, m in [("all_sampled", ok)] + [(f"split:{s}", ok & (base["split"] == s).to_numpy())
                                                  for s in ("train", "test")] + \
                        [(f"block:{b}", ok & (base["block"] == b).to_numpy()) for b in pick]:
            if not m.any():
                continue
            mt, mg = validation.error_metrics(e_tin[m]), validation.error_metrics(e_gp[m])
            rows.append({"source": source, "scope": scope, "n": mt["n"],
                         **{f"tin_{k}": mt[k] for k in ("bias", "rmse", "nmad")},
                         **{f"gp_{k}": mg[k] for k in ("bias", "rmse", "nmad")},
                         "min_product_nmad": min(validation.nmad(v[m]) for v in e_prod.values())})
        blk = base["block"].to_numpy()[ok]
        dd = (e_gp[ok], e_tin[ok])
        boot = validation.block_bootstrap(blk, lambda ix: [validation.nmad(dd[0][ix]) - validation.nmad(dd[1][ix]),
                                                           validation.rmse(dd[0][ix]) - validation.rmse(dd[1][ix])],
                                          a.n_boot)
        rows.append({"source": source, "scope": "gp_minus_tin_ci95",
                     "nmad_diff_ci": list(validation.percentile_interval(boot[:, 0])),
                     "rmse_diff_ci": list(validation.percentile_interval(boot[:, 1]))})
    tab = pd.DataFrame(rows)
    d = ROB / "r07_interpolation"
    d.mkdir(parents=True, exist_ok=True)
    tab.to_csv(d / "gp_vs_tin.csv", index=False)
    write_summary("r07_interpolation", {"gp_blocks": pick, "n_cells_kriged": n_cells, "covariance": cov,
                                        "nugget_m2": nugget, "thin_m": a.thin_m, "block_size_m": size,
                                        "datum_constant": "primary c for both DEMs (no refit on the subsample)",
                                        "table": str(d / "gp_vs_tin.csv"),
                                        "overall": tab[tab["scope"].isin(["all_sampled", "split:test",
                                                                          "gp_minus_tin_ci95"])].to_dict(orient="records")})
    return 0


def rebin_streaming(src_tag: str, variant: str, factor: int = 2) -> dict:
    """robustness.rebin_mapper(factor) applied tile by tile and band by band, writing
    each summed band straight into the output .npz, so a 255-band 100 km tile never sits
    in memory (r00_counts.build holds every band as int64). Same result as
    r00_counts.build(variant, [src_tag], bin_factor=factor)."""
    import zipfile
    out = INTERIM / "stage2" / variant
    out.mkdir(parents=True, exist_ok=True)
    rec = {"variant": variant, "sources": [src_tag], "bin_factor": factor,
           "bin_width_m": round(design.BIN_WIDTH_M * factor, 6), "bin_origin_m": design.BIN_ORIGIN_M, "tiles": {}}
    for f in sorted((INTERIM / "stage2" / src_tag).glob("tile_*.npz")):
        with np.load(f) as z:
            targets: dict = {}
            for k in z.files:
                p_ = R.parse_band(k)
                if p_ is None:
                    continue
                tgt = R.group_name(p_["epoch"], p_["bin"] // factor, p_["parity"], p_["suffix"])
                targets.setdefault(f"{p_['kind']}_{tgt}", []).append(k)
            tmp = out / (f.name + ".tmp")
            with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                with zf.open("transform.npy", "w") as fh:
                    np.lib.format.write_array(fh, np.asarray(z["transform"]))
                for band, srcs in sorted(targets.items()):
                    acc = np.zeros(z[srcs[0]].shape, np.uint32)
                    for k in srcs:
                        acc += z[k]
                    if acc.max(initial=0) > np.iinfo(np.uint16).max:
                        raise ValueError(f"{band}: count overflows uint16")
                    with zf.open(band + ".npy", "w", force_zip64=True) as fh:
                        np.lib.format.write_array(fh, acc.astype(np.uint16))
            tmp.replace(out / f.name)
        rec["tiles"][f.stem.replace("tile_", "")] = {"n_bands": len(targets)}
        print(variant, f.name, len(targets), "bands", flush=True)
    (out / "variant.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return rec


def cmd_bw020(a) -> int:
    variant = f"{MAIN}__bw020"
    if not (INTERIM / "stage2" / variant / "variant.json").exists():
        rebin_streaming(MAIN, variant, 2)
    P.points_step(variant, MAIN)                     # static flag from the frozen primary rho*
    P.dem_step(variant)
    diff = P.dem_difference("r07_interpolation_bw020", variant, MAIN)
    P.compare_validation("r07_interpolation_bw020", variant, MAIN, alt_points=P.variant_dem_points(variant),
                         n_boot=a.n_boot, sources=a.sources, note="0.20 m bins from the 0.10 m counts")
    pts = pd.read_csv(INTERIM / "waterlines" / variant / "points.csv.gz", usecols=["epoch", "static"])
    write_summary("r07_interpolation_bw020", {"variant": variant, "n_points": len(pts),
                                              "static_share": float(pts["static"].astype(bool).mean()),
                                              "dem_difference_overall": diff.set_index("sector").loc["overall"].to_dict(),
                                              "h2": h2_summary("r07_interpolation_bw020", variant), "n_boot": a.n_boot})
    return 0


# --- check 9: reed beds as dry or as water (bounds from the o band) -----------------------------------

def vegetation_streaming(src_tag: str, variant: str, mode: str) -> dict:
    """robustness.vegetation_tile(mode) applied group by group and written band by band
    (the rebin_streaming pattern), so a 100 km tile never sits in memory. Writes only w
    and v bands. Also counts the acquisitions that saw vegetation (o - v) against all
    observed (o), over every cell and group."""
    import zipfile
    out = INTERIM / "stage2" / variant
    out.mkdir(parents=True, exist_ok=True)
    rec = {"variant": variant, "sources": [src_tag], "vegetation": mode,
           "bin_width_m": P.script("r00_counts").bin_width_of(src_tag, INTERIM / "stage2"),
           "bin_origin_m": design.BIN_ORIGIN_M, "tiles": {}}
    n_veg = n_obs = 0
    for f in sorted((INTERIM / "stage2" / src_tag).glob("tile_*.npz")):
        with np.load(f) as z:
            groups: dict = {}
            for k in z.files:
                p_ = R.parse_band(k)
                if p_ is not None:
                    groups.setdefault(R.group_name(p_["epoch"], p_["bin"], p_["parity"], p_["suffix"]), set()).add(p_["kind"])
            tmp = out / (f.name + ".tmp")
            with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                with zf.open("transform.npy", "w") as fh:
                    np.lib.format.write_array(fh, np.asarray(z["transform"]))
                for g, ks in sorted(groups.items()):
                    if not {"w", "v", "o"} <= ks:
                        raise ValueError(f"{f.name} group {g}: the vegetation variants need w, v and o (D-071)")
                    o = z[f"o_{g}"]
                    v = z[f"v_{g}"]
                    n_veg += int(o.sum(dtype=np.int64) - v.sum(dtype=np.int64))
                    n_obs += int(o.sum(dtype=np.int64))
                    w2, v2 = R.vegetation_variant(z[f"w_{g}"], v, o, mode)
                    for kind, arr in (("w", w2), ("v", v2)):
                        if arr.max(initial=0) > np.iinfo(np.uint16).max:
                            raise ValueError(f"{kind}_{g}: count overflows uint16")
                        with zf.open(f"{kind}_{g}.npy", "w", force_zip64=True) as fh:
                            np.lib.format.write_array(fh, arr.astype(np.uint16))
                    del o, v, w2, v2
            tmp.replace(out / f.name)
        rec["tiles"][f.stem.replace("tile_", "")] = {"n_groups": len(groups)}
        print(variant, f.name, len(groups), "groups", flush=True)
    rec["share_observations_vegetation"] = n_veg / n_obs if n_obs else None
    (out / "variant.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return rec


def cmd_vegetation(a) -> int:
    """Check 9 bounds: emergent vegetation counted as land (as_dry: v = o) or as water
    (as_water: w + o - v, v = o) instead of not observed (PREREG section 3); waterlines,
    DEM and the paired H2 on the primary's points (bare population). The H1 side would
    need a full hindcast on the variant counts and is not run here."""
    if a.mode not in ("as_dry", "as_water"):
        raise SystemExit("--mode as_dry | as_water")
    variant = f"{MAIN}__veg_{a.mode}"
    vj = INTERIM / "stage2" / variant / "variant.json"
    vrec = json.loads(vj.read_text(encoding="utf-8")) if vj.exists() else vegetation_streaming(MAIN, variant, a.mode)
    if not a.skip_dem:
        P.points_step(variant, MAIN)                 # static flag from the frozen primary rho*
        P.dem_step(variant)
    diff = P.dem_difference("r09_vegetation", variant, MAIN)
    P.compare_validation("r09_vegetation", variant, MAIN, alt_points=P.variant_dem_points(variant),
                         n_boot=a.n_boot, sources=a.sources, note=f"reed beds {a.mode} (check 9)")
    pts = pd.read_csv(INTERIM / "waterlines" / variant / "points.csv.gz", usecols=["static"])
    path = ROB / "r09_vegetation" / "summary.json"
    rec = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"alternatives": {}}
    rec["alternatives"][a.mode] = {"variant": variant, "n_points": len(pts),
                                   "static_share": float(pts["static"].astype(bool).mean()),
                                   "share_observations_vegetation": vrec.get("share_observations_vegetation"),
                                   "dem_difference_overall": diff.set_index("sector").loc["overall"].to_dict(),
                                   "h2": h2_summary("r09_vegetation", variant), "n_boot": a.n_boot}
    write_summary("r09_vegetation", {k: v for k, v in rec.items() if k not in ("check", "created")})
    return 0


def _shifted_dem(check: str, variant: str, shifted: pd.DataFrame, a) -> dict:
    write_points(variant, shifted[[c for c in ("x", "y", "epoch", "z_m", "static") if c in shifted.columns]])
    P.dem_step(variant)
    diff = P.dem_difference(check, variant, MAIN)
    P.compare_validation(check, variant, MAIN, alt_points=P.variant_dem_points(variant), n_boot=a.n_boot,
                         sources=a.sources, note=check)
    return {"dem_difference_overall": diff.set_index("sector").loc["overall"].to_dict(),
            "h2": h2_summary(check, variant)}


def cmd_levels(a) -> int:
    r05 = _load("r05_levels")
    plan = pd.read_csv(PLAN, usecols=["scene", "acq", "t", "level_m", "epoch", "bin"]).drop_duplicates("scene")
    plan = plan.reset_index(drop=True)
    times = pd.to_datetime(plan["t"], format="ISO8601")
    pts = read_points_slim(cols=["x", "y", "epoch", "bin", "z_m", "static"]) if a.dem else None
    out = {}
    for name in a.level_names.split(","):
        alt, info = r05.alt_levels(name, times, None)
        alt = np.where(np.isfinite(alt), alt, plan["level_m"].to_numpy(float))
        sh = R.group_level_shift(plan, alt)
        d = ROB / "r05_levels"
        d.mkdir(parents=True, exist_ok=True)
        sh.to_csv(d / f"{name}_shifts.csv", index=False)
        rec = {**info, "shift_m": {q: float(np.percentile(sh["shift_m"], v)) for q, v in
                                   (("p05", 5), ("p50", 50), ("p95", 95))},
               "shift_range_m": [float(sh["shift_m"].min()), float(sh["shift_m"].max())],
               "share_acq_rebin": float(sh["n_acq_rebin"].sum() / sh["n_acq"].sum())}
        if a.dem:
            missing = pts.merge(sh[["epoch", "bin"]], on=["epoch", "bin"], how="left", indicator=True)
            keep = (missing["_merge"] == "both").to_numpy()
            rec["n_points_without_group_shift_kept"] = int((~keep).sum())
            shifted = pd.concat([R.shift_points(pts[keep], sh), pts[~keep]], ignore_index=True)
            rec.update(_shifted_dem("r05_levels", f"{MAIN}__lv_{name}", shifted, a))
        out[name] = rec
    prev = ROB / "r05_levels" / "summary.json"
    if prev.exists():                      # keep alternatives computed in earlier calls
        out = {**json.loads(prev.read_text(encoding="utf-8")).get("alternatives", {}), **out}
    write_summary("r05_levels", {"alternatives": out, "plan": str(PLAN), "n_boot": a.n_boot,
                                 "design": "first order: contour heights shifted by the group mean level change; "
                                           "the hindcast is not rerun (see share_acq_rebin and the shift range)"})
    return 0


def cmd_setupcorr(a) -> int:
    r04 = _load("r04_screening")
    plan = pd.read_csv(PLAN, low_memory=False)
    excl = R.exclusion(plan, np.inf)            # tau = infinity (D-077): only unknown setup excluded
    sh = R.setup_shift(plan, excl)
    d = ROB / "r04_screening"
    d.mkdir(parents=True, exist_ok=True)
    sh.to_csv(d / "setup_correction_shifts.csv", index=False)
    rec = {"shift_m": {q: float(np.nanpercentile(sh["shift_m"], v)) for q, v in (("p05", 5), ("p50", 50), ("p95", 95))},
           "shift_range_m": [float(sh["shift_m"].min()), float(sh["shift_m"].max())],
           "by_sector_median_m": sh.groupby("sector")["shift_m"].median().to_dict(),
           "by_sector_p05_p95_m": {k: [float(np.nanpercentile(g, 5)), float(np.nanpercentile(g, 95))]
                                   for k, g in sh.groupby("sector")["shift_m"]}}
    if not a.dem:
        write_summary("r04_screening_setupcorr", {**rec, "dem": "not rebuilt in this call (--dem)",
                                                  "design": "contour height + mean predicted setup of the group's "
                                                            "acquisitions in the point's sector (first order)"})
        return 0
    pts = read_points_slim(cols=["x", "y", "epoch", "bin", "z_m", "static"])
    pts["sector_pt"] = r04.point_sectors(pts["x"].to_numpy(), pts["y"].to_numpy())
    s = sh.rename(columns={"sector": "sector_pt"})
    full = (pts[["epoch", "bin", "sector_pt"]].drop_duplicates()
            .merge(s[["epoch", "bin", "sector_pt", "shift_m"]], on=["epoch", "bin", "sector_pt"], how="left"))
    n_missing = int(full["shift_m"].isna().sum())
    full["shift_m"] = full["shift_m"].fillna(0.0)
    shifted = R.shift_points(pts, full, on=("epoch", "bin", "sector_pt")).drop(columns=["sector_pt"])
    rec["n_groups_sector_without_shift_set_to_0"] = n_missing
    rec.update(_shifted_dem("r04_screening_setupcorr", f"{MAIN}__setupcorr", shifted, a))
    write_summary("r04_screening_setupcorr", {**rec, "n_boot": a.n_boot,
                                              "design": "contour height + mean predicted setup of the group's "
                                                        "acquisitions in the point's sector (first order)"})
    return 0


def cmd_komsomolets(a) -> int:
    r04 = _load("r04_screening")
    rows = []
    for tau in (0.30, 0.20, 0.15, 0.10, 0.05):
        r04.komsomolets("sensB", tau)
        t = pd.read_csv(ROB / "r04_screening" / "komsomolets_sensB.csv")
        rows.append(t.assign(tau_compared_m=tau))
    t = pd.concat(rows, ignore_index=True)
    t.to_csv(ROB / "r04_screening" / "komsomolets_sensB_all_taus.csv", index=False)
    s = (t.groupby(["tau_compared_m", "screening"])
         .agg(n_block_bins=("bin", "size"), S_median_m=("displacement_median_m", "median"),
              p90_m=("displacement_median_m", lambda x: float(np.nanpercentile(x, 90))),
              valid_km2=("valid_km2", "sum"), waterline_cells=("waterline_cells", "sum")).reset_index())
    s.to_csv(ROB / "r04_screening" / "komsomolets_sensB_summary_all_taus.csv", index=False)
    write_summary("r04_screening_komsomolets", {"table": s.to_dict(orient="records"),
                                                "blocks": "the 6 Komsomolets training blocks of the D-064 sample",
                                                "note": "local part of the D-066 check; the H1 headline in Komsomolets "
                                                        "needs no export, since the primary is unscreened (tau = inf)"})
    return 0


TWO_LIMB_BAND_M = (-26.65, -25.82)   # levels crossed on both limbs, DAHITI reference (D-037)


def cmd_morphology(a) -> int:
    r10 = _load("r10_morphology")
    r10.main(["--main", MAIN, "--no-map"])
    d = ROB / "r10_morphology"
    summ = pd.read_csv(d / "morphology_summary.csv")
    rec = {"summary_shape": summ[summ["variant"] == "shape"].to_dict(orient="records")}
    if a.map:   # E1 TIN DEM (slim points) minus the primary E2 + E3 DEM, on the cells both have
        e1 = f"{MAIN}__e1"
        write_points(e1, read_points_slim(cols=["x", "y", "epoch", "z_m", "static"], epochs=("E1",)))
        P.dem_step(e1, epochs="E1")
        lim = P.dem_difference("r10_morphology", e1, MAIN)
        lim.to_csv(d / "limb_difference_by_sector.csv", index=False)
        rec["limb_difference_E1_minus_E2E3"] = lim.to_dict(orient="records")
        # inside the two-limb band only (cells whose E2 + E3 height lies in it), where both
        # limbs have contours; elsewhere the E1 TIN is unconstrained (plan, check 10)
        band = P.dem_difference("r10_morphology", e1, MAIN, main_range=TWO_LIMB_BAND_M,
                                name="dem_difference_two_limb_band.csv")
        band.to_csv(d / "limb_difference_two_limb_band_by_sector.csv", index=False)
        rec["two_limb_band_m"] = list(TWO_LIMB_BAND_M)
        rec["limb_difference_two_limb_band"] = band.to_dict(orient="records")
    write_summary("r10_morphology", rec)
    return 0


def cmd_prior(a) -> int:
    r11 = _load("r11_prior_work")
    r11.main(["--main", MAIN])
    rec = json.loads((ROB / "r11_prior_work" / "akbari2020_area_slope.json").read_text(encoding="utf-8"))
    obs = pd.read_csv(RESULTS / "hindcast" / MAIN / "observed.csv")
    by_epoch = {}
    for ep in ("E1", "E2", "E3"):
        o = obs[obs["epoch"] == ep]
        by_epoch[ep] = R.area_level_slope(o, lo=-29.0, hi=-26.0)
    sea_unknown = R.area_level_slope(obs.assign(area_sea_plus_unknown_km2=obs["area_sea_km2"]
                                                + obs["area_unconnected_unknown_km2"]),
                                     area_col="area_sea_plus_unknown_km2")
    no_filter = R.area_level_slope(obs, min_valid_frac=0.0)
    # coverage-complete alternative: the hypsometry of the waterline DEM (area of the DEM
    # between two heights per metre), product-free; DEM heights are on the DAHITI reference
    import rasterio
    ex = np.load(RESULTS / "study_area" / "extents_v6_250m.npz")
    band, et = (ex["high"] == 1) & (ex["low"] == 0), ex["transform"]
    edges = np.round(np.arange(-30.0, -25.0 + 1e-9, 0.1), 2)
    hist, hist_band = np.zeros(edges.size - 1, np.int64), np.zeros(edges.size - 1, np.int64)
    n_band = 0
    for f in sorted((PROCESSED / "dem" / MAIN).glob("dem_*.tif")):
        with rasterio.open(f) as r:
            z, trf = r.read(1), r.transform
        rows, cols = np.nonzero(np.isfinite(z))
        v = z[rows, cols]
        hist += np.histogram(v, bins=edges)[0]
        x, y = trf.c + trf.a * (cols + 0.5), trf.f + trf.e * (rows + 0.5)
        er, ec = np.floor((y - et[5]) / et[4]).astype(int), np.floor((x - et[2]) / et[0]).astype(int)
        ok = (er >= 0) & (er < band.shape[0]) & (ec >= 0) & (ec < band.shape[1])
        inb = np.zeros(v.size, bool)
        inb[ok] = band[er[ok], ec[ok]]
        hist_band += np.histogram(v[inb], bins=edges)[0]
        n_band += int(inb.sum())
    cell_km2 = (design.DEM_RES_M / 1000.0) ** 2
    hyp = pd.DataFrame({"z_lo_m": edges[:-1], "z_hi_m": edges[1:], "area_km2": hist * cell_km2,
                        "area_band_km2": hist_band * cell_km2})
    hyp.to_csv(ROB / "r11_prior_work" / "dem_hypsometry.csv", index=False)
    dem_slopes = {}
    for shift in (0.0, -0.5, 0.5):
        lo, hi = -29.0 + shift, -26.0 + shift
        m = (hyp["z_lo_m"] >= lo - 1e-9) & (hyp["z_hi_m"] <= hi + 1e-9)
        dem_slopes[f"range_{lo:.1f}_{hi:.1f}"] = {
            "area_all_dem_km2": float(hyp.loc[m, "area_km2"].sum()),
            "area_emerged_band_km2": float(hyp.loc[m, "area_band_km2"].sum()),
            "km2_per_m_emerged_band": float(hyp.loc[m, "area_band_km2"].sum() / (hi - lo))}
    dem_slopes["emerged_band_dem_cells_km2"] = n_band * cell_km2
    write_summary("r11_prior_work", {**rec, "ours_by_epoch": by_epoch, "ours_sea_plus_unknown": sea_unknown,
                                     "ours_no_valid_filter": no_filter,
                                     "dem_hypsometry_km2_per_m": dem_slopes,
                                     "dem_hypsometry_note": "DEM area between the heights divided by the height range; "
                                                            "DAHITI reference, so the range is also shifted by +-0.5 m "
                                                            "for an unknown DAHITI - BHS offset; the DEM covers the "
                                                            "emerged band only (TIN hull within the stage-2 cover)",
                                     "sources": "refs/literature_values.csv (verbatim quote, Table 1 of Akbari et al. "
                                                "2020); no new citation"})
    return 0


# --- R-block alternatives (scripts/r18_rblock_export.py outputs) -----------------------------------

RBLOCK_MARGIN_M = 6_000.0        # = scripts/a6_dem.py OVERLAP_M: points kept around R for the local TINs


def main_window(block_tr, dims, bands) -> dict:
    """The primary counts of ``bands`` on a block grid (same 60 m alignment), from the
    primary stage-2 tiles. Where tiles overlap (shared edge cells, identical where both
    computed them) the larger count is kept, since a tile holds 0 outside its region."""
    cols, rows = dims
    out = {b: np.zeros((rows, cols), np.uint16) for b in bands}
    c2, f2 = float(block_tr[2]), float(block_tr[5])
    for f in sorted((INTERIM / "stage2" / MAIN).glob("tile_*.npz")):
        with np.load(f) as z:
            tr = z["transform"]
            first = next(k for k in z.files if k != "transform")
            shape = z[first].shape
            dc, dr = int(round((c2 - float(tr[2])) / float(tr[0]))), int(round((f2 - float(tr[5])) / float(tr[4])))
            r0, c0 = max(dr, 0), max(dc, 0)
            r1, c1 = min(dr + rows, shape[0]), min(dc + cols, shape[1])
            if r1 <= r0 or c1 <= c0:
                continue
            for b in bands:
                if b in z.files:   # adjacent tiles share edge cells; outside its region a tile holds 0
                    dst = out[b][r0 - dr:r1 - dr, c0 - dc:c1 - dc]
                    np.maximum(dst, z[b][r0:r1, c0:c1], out=dst)
    return out


def cmd_rblock(a) -> int:
    """Analysis of one R-block alternative: its contours inside R, a hybrid TIN DEM (the
    alternative's contours inside R, the primary's elsewhere) against a TIN of the
    primary contours with the same point set around R, the R-local paired H2 comparison
    (all R points, the primary's datum constant and product offsets; nothing refitted)
    and the contour offsets against the primary inside R."""
    import rasterio  # noqa: F401
    C = P.script("r00_counts")
    a6w = P.script("a6_waterlines")
    tag, alt, kind = a.tag, a.alt, a.kind
    width = 0.05 if alt == "bins005" else design.BIN_WIDTH_M
    st = json.loads((RESULTS / "inventory" / f"stage2_{tag}.json").read_text(encoding="utf-8"))
    regions = C.block_regions(tag)
    rho_star = P.main_static_rule(MAIN)["rho_star"]
    work = INTERIM / "robustness" / "rblock" / alt
    work.mkdir(parents=True, exist_ok=True)
    pts_alt, eecu = [], 0.0
    main_bands = set()
    for f in sorted((INTERIM / "stage2" / MAIN).glob("tile_*.npz")):
        with np.load(f) as z:
            main_bands |= {k for k in z.files if k[:2] in ("w_", "v_")}
    seam_cells = {}
    for key, t in sorted(st["tiles"].items()):
        tr, arr = R.load_tile(INTERIM / "stage2" / tag / f"tile_{key}.npz")
        eecu += float(t.get("eecu_s") or 0)
        mw = main_window(tr, t["dims"], sorted(main_bands))
        # cells the primary never observed (seams between its 100 km tiles, where the
        # lon/lat-planar export region bows inside both tiles) stay unobserved here too
        seen = np.zeros(next(iter(mw.values())).shape, bool)
        for b, x in mw.items():
            if b.startswith("v_"):
                seen |= x > 0
        drop = ~seen & np.any([x > 0 for b, x in arr.items() if b.startswith("v_")], axis=0)
        seam_cells[key] = int(drop.sum())
        arr = {b: np.where(seen, x, 0).astype(np.uint16) for b, x in arr.items()}
        if kind == "sub":            # every primary group in the window, less the SUB counts
            arr = R.apply_delta(mw, sub=arr)
        elif kind == "add":          # every primary group in the window, plus the ADD counts
            arr = R.apply_delta({b: x for b, x in mw.items()}, add={b: x for b, x in arr.items() if b in mw})
        R.write_tile(work / f"tile_{key}.npz", tr, arr)
        p = a6w.tile_points(work / f"tile_{key}.npz", spacing=abs(float(tr[0])))
        if len(p):
            pts_alt.append(p.assign(block=key))
    pa = pd.concat(pts_alt, ignore_index=True)
    pa["z_m"] = levels.bin_center(pa["bin"].to_numpy(), width, design.BIN_ORIGIN_M)
    pa["static"] = pa["rho_level"].to_numpy(float) < rho_star
    # primary contours around R, and which of them lie inside an R region
    pm = read_points_slim(cols=["x", "y", "epoch", "bin", "z_m", "static"])
    size = float(design.BLOCK_SIZE_M)
    near = np.zeros(len(pm), bool)
    inside = np.zeros(len(pm), bool)
    for key, (btr, reg) in regions.items():
        i, j = (int(v) for v in key.split("_"))
        x0, y0 = i * size, j * size
        near |= ((pm["x"] > x0 - RBLOCK_MARGIN_M) & (pm["x"] < x0 + size + RBLOCK_MARGIN_M)
                 & (pm["y"] > y0 - RBLOCK_MARGIN_M) & (pm["y"] < y0 + size + RBLOCK_MARGIN_M)).to_numpy()
        col = np.floor((pm["x"].to_numpy() - btr[2]) / btr[0]).astype(int)
        row = np.floor((pm["y"].to_numpy() - btr[5]) / btr[4]).astype(int)
        ok = (row >= 0) & (row < reg.shape[0]) & (col >= 0) & (col < reg.shape[1])
        hit = np.zeros(len(pm), bool)
        hit[ok] = reg[row[ok], col[ok]]
        inside |= hit
    ref_tag, hyb_tag = f"{MAIN}__Rnear", f"{MAIN}__rR_{alt}"
    cols = ["x", "y", "epoch", "z_m", "static"]
    if not (PROCESSED / "dem" / ref_tag).exists():
        write_points(ref_tag, pm.loc[near, cols])
        P.dem_step(ref_tag)
    hyb = pd.concat([pm.loc[near & ~inside, cols], pa[cols]], ignore_index=True)
    write_points(hyb_tag, hyb)
    P.dem_step(hyb_tag)
    # contour offsets inside R (same bin width only)
    offs = None
    if width == design.BIN_WIDTH_M:
        e23 = lambda d: d[d["epoch"].isin(["E2", "E3"]) & ~d["static"].astype(bool)]  # noqa: E731
        offs = R.contour_offsets(e23(pa), e23(pm.loc[inside]))
        nf = pd.read_csv(RESULTS / "hindcast" / MAIN / "noise_floor.csv")
        nf = nf[(nf["kind"] == "halves") & (nf["sector"] == "all")][["epoch", "bin", "median_m"]]
        offs = offs.merge(nf.rename(columns={"median_m": "noise_halves_median_m"}), on=["epoch", "bin"], how="left")
    out = ROB / f"rblock_{alt}"
    out.mkdir(parents=True, exist_ok=True)
    if offs is not None:
        offs.to_csv(out / "contour_offsets.csv", index=False)
    del pm
    # R-local paired H2
    rb = pd.read_csv(RESULTS / "study_area" / "blocks_25km.csv")
    rkeys = set(regions)
    code = np.array([int(k.split("_")[0]) * 100_000 + int(k.split("_")[1]) for k in rkeys], np.int64)
    keep_fn = (lambda d_: np.isin(np.floor(d_["x"].to_numpy() / size).astype(np.int64) * 100_000
                                  + np.floor(d_["y"].to_numpy() / size).astype(np.int64), code))
    ref_dem, hyb_dem = (validation.TileSet(PROCESSED / "dem" / t_, "dem") for t_ in (ref_tag, hyb_tag))
    full_dem = validation.TileSet(PROCESSED / "dem" / MAIN, "dem")
    h2 = {}
    rows_out = []
    for source in a.sources:
        pts = P.load_points_light(MAIN, source, row_filter=keep_fn)
        bi, bj = np.floor(pts["x"] / size).astype(int), np.floor(pts["y"] / size).astype(int)
        pts["block"] = [f"{i}_{j}" for i, j in zip(bi, bj)]
        sec = rb.set_index(rb["i"].astype(str) + "_" + rb["j"].astype(str))["sector"]
        pts["sector"] = pts["block"].map(sec)
        x, y = pts["x"].to_numpy(float), pts["y"].to_numpy(float)
        cst = json.loads((RESULTS / "validation" / MAIN / f"{source}_datum_constant.json").read_text(encoding="utf-8"))
        offp = json.loads((RESULTS / "validation" / MAIN / f"{source}_product_offsets.json").read_text(
            encoding="utf-8"))["product_offsets"]
        yref = pts["h_orth"].to_numpy(float) + cst["constant"]["constant_m"]
        e_ref, e_hyb, e_full = (t_.sample(x, y, 1) - yref for t_ in (ref_dem, hyb_dem, full_dem))
        e_prod = {pname: pts[f"prod:{pname}"].to_numpy(float) + o["offset_m"] - yref for pname, o in offp.items()}
        ok = np.isfinite(e_ref) & np.isfinite(e_hyb)
        for pv in e_prod.values():
            ok &= np.isfinite(pv)
        rec = {}
        for scope, m_ in [("overall", ok)] + [(s_, ok & (pts["sector"] == s_).to_numpy()) for s_ in SECTOR_ORDER]:
            if m_.sum() < 50:
                continue
            r_ = {"n": int(m_.sum()), "dem_primary_nmad_m": validation.nmad(e_ref[m_]),
                  "dem_variant_nmad_m": validation.nmad(e_hyb[m_]), "dem_primary_rmse_m": validation.rmse(e_ref[m_]),
                  "dem_variant_rmse_m": validation.rmse(e_hyb[m_]), "dem_primary_bias_m": validation.bias(e_ref[m_]),
                  "dem_variant_bias_m": validation.bias(e_hyb[m_]),
                  "min_product_nmad_m": min(validation.nmad(v[m_]) for v in e_prod.values()),
                  "truncation_check_nmad_m": validation.nmad(e_full[m_ & np.isfinite(e_full)])}
            r_["ordering_primary"] = bool(r_["dem_primary_nmad_m"] < r_["min_product_nmad_m"])
            r_["ordering_variant"] = bool(r_["dem_variant_nmad_m"] < r_["min_product_nmad_m"])
            rec[scope] = r_
            rows_out.append({"source": source, "scope": scope, **r_})
        blk = pts["block"].to_numpy()[ok]
        boot = validation.block_bootstrap(blk, lambda ix: validation.nmad(e_hyb[ok][ix]) - validation.nmad(e_ref[ok][ix]),
                                          a.n_boot)
        rec["nmad_diff_ci95_m"] = list(validation.percentile_interval(boot))
        h2[source] = rec
    pd.DataFrame(rows_out).to_csv(out / "h2_rlocal.csv", index=False)
    summ = {"alt": alt, "tag": tag, "kind": kind, "bin_width_m": width, "eecu_s_export": eecu,
            "cells_unobserved_in_primary_dropped": seam_cells,
            "n_blocks": len(st["tiles"]), "n_points_alt": int(len(pa)), "static_share_alt": float(pa["static"].mean()),
            "h2_rlocal": h2, "design": "R-local paired comparison: TIN of the primary contours around R vs the hybrid "
                                       "TIN; primary datum constant and matched product offsets; all R points"}
    if offs is not None and len(offs):
        summ["contour_offset_median_m"] = float(np.nanmedian(offs["median_m"]))
        summ["share_bins_offset_above_noise"] = float(np.nanmean(offs["median_m"] > offs["noise_halves_median_m"]))
    write_summary(f"rblock_{alt}", summ)
    return 0


def cmd_validate(a) -> int:
    """Validate an existing variant DEM for more sources (e.g. ATL06 after ATL08) and
    refresh the H2 block of that check's summary.json (--summary-key, dotted path)."""
    P.compare_validation(a.check, a.variant, MAIN, alt_points=P.variant_dem_points(a.variant), n_boot=a.n_boot,
                         sources=a.sources, note=f"{a.check} (added source)")
    path = ROB / a.summary_check / "summary.json"
    rec = json.loads(path.read_text(encoding="utf-8"))
    node = rec
    keys = a.summary_key.split(".")
    for k in keys[:-1]:
        node = node[k]
    node[keys[-1]] = h2_summary(a.check, a.variant)
    path.write_text(_json(rec), encoding="utf-8")
    print(_json(node[keys[-1]])[:1500])
    return 0


def cmd_validate_alt(a) -> int:
    """Validation-side variants without a DEM change (lag365, nearest) for more sources."""
    return {"lag365": cmd_lag365, "nearest": cmd_nearest}[a.variant](a)


COMMANDS = {"validate": cmd_validate, "h1-e2e3": cmd_h1_e2e3, "nostatic": cmd_nostatic, "lag365": cmd_lag365, "nearest": cmd_nearest,
            "dn-total": cmd_dn_total, "gp": cmd_gp, "bw020": cmd_bw020, "levels": cmd_levels,
            "setupcorr": cmd_setupcorr, "komsomolets": cmd_komsomolets, "morphology": cmd_morphology,
            "prior": cmd_prior, "rblock": cmd_rblock, "datums": cmd_datums, "vegetation": cmd_vegetation, "fold": cmd_fold}


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=sorted(COMMANDS))
    ap.add_argument("--n-boot", type=int, default=design.BOOTSTRAP_N)
    ap.add_argument("--hindcast", action="store_true", help="nearest: also the H1 hindcast")
    ap.add_argument("--skip-validation", action="store_true", help="nearest: hindcast only")
    ap.add_argument("--summary-only", action="store_true", help="h1-e2e3, dn-total: summarise an existing run")
    ap.add_argument("--sources", default="ATL06,ATL08", type=lambda v: tuple(v.split(",")))
    ap.add_argument("--skip-dem", action="store_true", help="nostatic/bw020: the variant DEM exists already")
    ap.add_argument("--gp-test", type=int, default=4)
    ap.add_argument("--gp-train", type=int, default=4)
    ap.add_argument("--thin-m", type=float, default=90.0)
    ap.add_argument("--level-names", default="hydroweb_const,dahiti_pchip,window1y")
    ap.add_argument("--dem", action="store_true", help="levels: also rebuild the DEM and validate")
    ap.add_argument("--map", action="store_true", help="morphology: also the limb difference map")
    ap.add_argument("--alt", help="rblock: the alternative (scripts/r18_rblock_export.py)")
    ap.add_argument("--check", help="validate: the results/robustness/<check> folder of the variant")
    ap.add_argument("--variant", help="validate: the variant DEM tag")
    ap.add_argument("--summary-check", help="validate: the folder whose summary.json is refreshed")
    ap.add_argument("--summary-key", default="h2", help="validate: dotted key of the H2 block in summary.json")
    ap.add_argument("--tag", help="rblock: its stage-2 tag")
    ap.add_argument("--kind", default="full", choices=["full", "sub", "add"],
                    help="rblock: full alternative, SUB delta or ADD delta")
    ap.add_argument("--mode", help="vegetation: as_dry | as_water")
    a = ap.parse_args(argv)
    return COMMANDS[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
