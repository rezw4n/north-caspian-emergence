"""Journal revision, PREREG Amendment 20 items 4, 5 and 7 (D-161): what the ICESat-2 points
say about the level of the Landsat shorelines, by surface type; whether the test points are
independent of the training blocks; and how much lidar support each sector has.
Descriptive, post-review; H2 and the checked-sector rule are unchanged, nothing is refitted.

The model height at a point is the level at which the Landsat shoreline passed it, so
e = model - (lidar + constant) is the error of that shoreline's level measured by another
sensor, and e divided by the sector's median shelf gradient is a shoreline position error.

Points: the prepared ATL06 and ATL08 tables of the registered evaluation
(data/interim/validation/main/<source>_points.csv.gz: quality-filtered, dry when
measured), inside the emerged band, with the 50 km blocks of the validation
(results/study_area/blocks_50km.csv), the registered training constant
(results/validation/main/<source>_datum_constant.json) and, for the grids, their registered
training offsets (<source>_product_offsets.json). A point is used when the reference, the
model and all five grids have a value (the paired set of the registered evaluation).
Check first: on the test blocks of the registered bare population (vegetation share below
0.5) the model's sector statistics reproduce results/validation/main/<source>_accuracy.csv.

Item 4  strata: surface (bare < 0.1, mixed 0.1 to < 0.5, vegetated >= 0.5, by the E3
        emergent-vegetation share of the cell) and, for the registered bare population,
        0.5 m bands of the model height from the lowest audited level upward. Mean, median,
        NMAD, points and blocks; block-bootstrap 95% intervals (B = 2,000, seed 20261125;
        caspian.rev3.HistBoot, a 1 mm histogram of the errors); the horizontal equivalents
        of the median and the NMAD (errors divided by the point's sector gradient).
Item 5  the registered test-block statistics of the model and the grids on the test points
        at least 10 km and at least 25 km from every training or calibration block; and
        the mean absolute error of the model within 10 km of such a block minus that
        farther away, with a block-bootstrap interval.
Item 7  per sector (registered bare population, all blocks): points, blocks by split,
        track segments (granule and beam), acquisition dates and their span; the
        intra-cluster correlation of the error, the design effect and the effective sample
        size, with blocks and with track segments as clusters.

Outputs (results/revision3/lidar/): strata.csv, independence.csv, independence_near_far.csv,
support.csv, summary.json.
Usage: python scripts/rev3_lidar.py [ATL08,ATL06]
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import caspian  # noqa: F401
import numpy as np
import pandas as pd

from caspian import rev3, validation
from caspian.config import INTERIM, RESULTS
from caspian.geography import SECTOR_ORDER

SCRIPTS = Path(__file__).resolve().parent
OUT = RESULTS / "revision3" / "lidar"
VAL = RESULTS / "validation" / "main"
SEED = 20261125
N_BOOT = 2000
BLOCK_M = 50_000.0
CHECKED = ("west", "interfluve", "northeast", "open_shelf")        # Amendment 18 item 1
BUFFERS_M = (10_000.0, 25_000.0)
NEAR_M = 10_000.0
BAND_STEP_M = 0.5
N_BANDS = 6


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def lowest_audited_level() -> float:
    run = json.loads((RESULTS / "hindcast" / "main" / "run.json").read_text(encoding="utf-8"))
    return float(min(u["level_m"] for u in run["units"]))


def gradients() -> dict:
    g = pd.read_csv(RESULTS / "revision" / "r" / "budget" / "gradients.csv")
    return {r.sector: float(r.g_median_m_per_km) for r in g.itertuples()}


def points(source: str) -> tuple[pd.DataFrame, dict]:
    """Every paired point of the emerged band inside the 50 km block table, with its
    error, the grids' errors (grid + registered offset - reference), block, sector, split,
    vegetation share, model height, granule, beam and date."""
    path = INTERIM / "validation" / "main" / f"{source}_points.csv.gz"
    cols = list(pd.read_csv(path, nrows=0).columns)
    prods = [c for c in cols if c.startswith("prod:")]
    d = pd.read_csv(path, usecols=["x", "y", "h_orth", "dem_h", "beam", "granule", *prods],
                    dtype={"beam": "category", "granule": "category"})
    ann = np.load(INTERIM / "validation" / "main" / f"{source}_annotations.npz")
    if len(ann["in_band"]) != len(d):
        raise SystemExit(f"{source}: the annotations do not match the point table")
    n_prepared = len(d)
    d["veg_share"] = ann["veg_share"]
    d = d.loc[ann["in_band"]].reset_index(drop=True)
    n_band = len(d)
    d = validation.attach_blocks(d, pd.read_csv(RESULTS / "study_area" / "blocks_50km.csv"), BLOCK_M)
    n_outside = int(d["split"].isna().sum())
    d = d[d["split"].notna()].reset_index(drop=True)
    c = json.loads((VAL / f"{source}_datum_constant.json").read_text(encoding="utf-8"))["constant"]["constant_m"]
    off = json.loads((VAL / f"{source}_product_offsets.json").read_text(encoding="utf-8"))["product_offsets"]
    y = d["h_orth"].to_numpy(float) + c
    ok = np.isfinite(y) & np.isfinite(d["dem_h"].to_numpy(float)) & np.isfinite(d["veg_share"].to_numpy(float))
    for p in prods:
        ok &= np.isfinite(d[p].to_numpy(float))
    out = pd.DataFrame({"x": d["x"].to_numpy(float)[ok], "y": d["y"].to_numpy(float)[ok],
                        "block": d["block"].to_numpy()[ok], "sector": d["sector"].to_numpy(object)[ok],
                        "split": d["split"].to_numpy(object)[ok], "veg_share": d["veg_share"].to_numpy(float)[ok],
                        "dem_h": d["dem_h"].to_numpy(float)[ok], "e": (d["dem_h"].to_numpy(float) - y)[ok]})
    for p in prods:
        out[f"e:{p[5:]}"] = (d[p].to_numpy(float) + float(off[p[5:]]["offset_m"]) - y)[ok]
    gran = d["granule"].astype(str).to_numpy()[ok]
    out["segment"] = pd.Categorical(np.char.add(np.char.add(gran, "|"), d["beam"].astype(str).to_numpy()[ok]))
    out["date"] = pd.Categorical(pd.Series(gran).str.extract(r"_(\d{8})\d{6}_", expand=False).to_numpy())
    if out["date"].isna().any():
        raise SystemExit(f"{source}: a granule name without a date")
    info = {"n_prepared": n_prepared, "n_in_band": n_band, "n_outside_block_table": n_outside,
            "n_not_paired": int((~ok).sum()), "n_used": len(out), "datum_constant_m": c,
            "grids": [p[5:] for p in prods]}
    return out, info


def check_test_blocks(d: pd.DataFrame, source: str, veg_max: float) -> float:
    """The registered bare population on test blocks reproduces the registered accuracy
    table (model rows and the grids' matched rows)."""
    acc = pd.read_csv(VAL / f"{source}_accuracy.csv").set_index(["scope", "model"])
    te = d[(d["split"] == "test") & (d["veg_share"] < veg_max)]
    worst = 0.0
    models = {"dem": "e", **{f"{c[2:]}|matched": c for c in d.columns if c.startswith("e:")}}
    for scope in ["overall"] + SECTOR_ORDER:
        g = te if scope == "overall" else te[te["sector"] == scope]
        for model, col in models.items():
            m = validation.error_metrics(g[col].to_numpy())
            w = acc.loc[(scope, model)]
            if int(w["n"]) != m["n"]:
                raise SystemExit(f"{source} {scope} {model}: {m['n']} test points, registered {int(w['n'])}")
            if m["n"] == 0:
                continue
            for k in ("bias", "median", "rmse", "nmad"):
                dev = abs(m[k] - float(w[k]))
                worst = max(worst, dev)
                if dev > 1e-9:
                    raise SystemExit(f"{source} {scope} {model} {k}: {m[k]} against registered {w[k]}")
    return worst


def stat_row(e, blocks, eh) -> dict:
    """Exact point estimates; block-bootstrap intervals of the mean, median and NMAD."""
    m = validation.error_metrics(e)
    row = {"n_points": m["n"], "n_blocks": int(np.unique(np.asarray(blocks).astype(str)).size), "mean": m["bias"],
           "median": m["median"], "nmad": m["nmad"], "rmse": m["rmse"],
           "median_horizontal_km": float(np.median(eh)), "nmad_horizontal_km": validation.nmad(eh)}
    boot = rev3.HistBoot(e, blocks).replicates(N_BOOT, SEED)
    for k, name in enumerate(("mean", "median", "nmad")):
        row[f"{name}_lo"], row[f"{name}_hi"] = rev3.interval(boot[:, k])
    return row


def scopes(d: pd.DataFrame):
    yield "checked4", d["sector"].isin(CHECKED).to_numpy()
    for s in SECTOR_ORDER:
        yield s, (d["sector"] == s).to_numpy()


def item4(d: pd.DataFrame, source: str, veg_max: float, lowest: float, log) -> list[dict]:
    g = gradients()
    eh = d["e"].to_numpy() / d["sector"].map(g).to_numpy(float)          # km
    surf = rev3.surface_stratum(d["veg_share"].to_numpy())
    band = rev3.height_band(d["dem_h"].to_numpy(), lowest, BAND_STEP_M, N_BANDS)
    bare_pop = d["veg_share"].to_numpy() < veg_max
    e, blk = d["e"].to_numpy(), d["block"].to_numpy()
    rows = []
    for scope, in_scope in scopes(d):
        strata = [("surface", n, in_scope & (surf == n)) for n in rev3.SURFACE_NAMES]
        strata.append(("surface", "registered_bare", in_scope & bare_pop))
        for k in range(N_BANDS):
            lo = lowest + k * BAND_STEP_M
            name = f"{lo:.2f}_to_{lo + BAND_STEP_M:.2f}" if k < N_BANDS - 1 else f"{lo:.2f}_and_above"
            strata.append(("height_band", name, in_scope & bare_pop & (band == k)))
        for kind, name, m in strata:
            base = {"source": source, "scope": scope, "kind": kind, "stratum": name}
            if not m.any():
                rows.append({**base, "n_points": 0, "n_blocks": 0})
                continue
            rows.append({**base, **stat_row(e[m], blk[m], eh[m])})
        log(f"  item 4 {source} {scope}")
    return rows


def item5(d: pd.DataFrame, source: str, veg_max: float) -> tuple[list[dict], list[dict]]:
    blocks = pd.read_csv(RESULTS / "study_area" / "blocks_50km.csv")
    fit = blocks[blocks["split"].isin(["train", "calib"])]
    te = d[(d["split"] == "test") & (d["veg_share"] < veg_max)].reset_index(drop=True)
    dist = rev3.distance_to_blocks(te["x"].to_numpy(), te["y"].to_numpy(), fit["x0"].to_numpy(float),
                                   fit["y0"].to_numpy(float), BLOCK_M)
    models = {"dem": "e", **{c[2:]: c for c in te.columns if c.startswith("e:")}}
    rows = []
    for name, buf in [("all_test_points", 0.0)] + [(f"at_least_{int(b / 1000)}km", b) for b in BUFFERS_M]:
        m = dist >= buf
        for model, col in models.items():
            em = validation.error_metrics(te.loc[m, col].to_numpy())
            rows.append({"source": source, "subset": name, "buffer_km": buf / 1000, "model": model, "n_points": em["n"],
                         "n_blocks": int(te.loc[m, "block"].nunique()), "mean": em["bias"], "median": em["median"],
                         "nmad": em["nmad"], "rmse": em["rmse"]})
    # near against far: mean |e| of the model, with a block bootstrap over the test blocks
    a = np.abs(te["e"].to_numpy())
    near = dist < NEAR_M
    lab = te["block"].to_numpy().astype(str)
    uniq, inv = np.unique(lab, return_inverse=True)
    s_near = np.bincount(inv, weights=a * near, minlength=uniq.size)
    n_near = np.bincount(inv, weights=near.astype(float), minlength=uniq.size)
    s_far = np.bincount(inv, weights=a * ~near, minlength=uniq.size)
    n_far = np.bincount(inv, weights=(~near).astype(float), minlength=uniq.size)

    def diff(w):
        if (w * n_near).sum() == 0 or (w * n_far).sum() == 0:
            return np.nan
        return (w * s_near).sum() / (w * n_near).sum() - (w * s_far).sum() / (w * n_far).sum()

    rng = np.random.default_rng(SEED)
    boot = np.array([diff(np.bincount(rng.integers(0, uniq.size, uniq.size), minlength=uniq.size))
                     for _ in range(N_BOOT)])
    lo, hi = rev3.interval(boot)
    nf = [{"source": source, "near_km": NEAR_M / 1000, "n_near": int(near.sum()), "n_far": int((~near).sum()),
           "n_blocks_with_near": int((n_near > 0).sum()), "n_blocks_with_far": int((n_far > 0).sum()),
           "mean_abs_near_m": float(a[near].mean()) if near.any() else np.nan,
           "mean_abs_far_m": float(a[~near].mean()) if (~near).any() else np.nan,
           "difference_m": float(diff(np.ones(uniq.size))), "difference_lo_m": lo, "difference_hi_m": hi,
           "n_boot_undefined": int(np.isnan(boot).sum()),
           "max_distance_km": float(dist.max() / 1000), "median_distance_km": float(np.median(dist) / 1000)}]
    return rows, nf


def item7(d: pd.DataFrame, source: str, veg_max: float) -> list[dict]:
    b = d[d["veg_share"] < veg_max]
    rows = []
    for scope in ["overall"] + SECTOR_ORDER:
        g = b if scope == "overall" else b[b["sector"] == scope]
        row = {"source": source, "sector": scope, "n_points": int(len(g))}
        if g.empty:
            rows.append(row)
            continue
        dates = np.array(sorted(g["date"].astype(str).unique()))
        row.update({"n_blocks": int(g["block"].nunique()),
                    **{f"n_blocks_{s}": int(g.loc[g["split"] == s, "block"].nunique()) for s in validation.SPLITS},
                    **{f"n_points_{s}": int((g["split"] == s).sum()) for s in validation.SPLITS},
                    "n_segments": int(g["segment"].astype(str).nunique()), "n_dates": int(dates.size),
                    "first_date": f"{dates[0][:4]}-{dates[0][4:6]}-{dates[0][6:]}",
                    "last_date": f"{dates[-1][:4]}-{dates[-1][4:6]}-{dates[-1][6:]}"})
        e = g["e"].to_numpy()
        for name, lab in (("block", g["block"].to_numpy().astype(str)), ("segment", g["segment"].astype(str).to_numpy())):
            icc = rev3.icc_oneway(e, lab)
            sizes = np.unique(lab, return_counts=True)[1]
            k = rev3.kish_effective_n(sizes, icc["rho"])
            row.update({f"icc_{name}": icc["rho"], f"deff_{name}": k["deff"], f"n_eff_{name}": k["n_eff"]})
        rows.append(row)
    return rows


def main(argv) -> int:
    sources = argv[0].split(",") if argv else ["ATL08", "ATL06"]
    a6v = _load("a6_validate_dem")
    veg_max = float(a6v.VEG_CELL_SHARE)
    lowest = lowest_audited_level()
    r4, r5, r5b, r7, summ = [], [], [], [], {}
    for source in sources:
        t0 = time.time()

        def log(msg):
            print(f"{msg} ({time.time() - t0:.0f} s)", flush=True)

        d, info = points(source)
        info["test_block_check_max_abs_dev_m"] = check_test_blocks(d, source, veg_max)
        log(f"{source}: {info['n_used']} points in the band; the registered test-block statistics are reproduced "
            f"(max dev {info['test_block_check_max_abs_dev_m']:.1e} m)")
        r4 += item4(d, source, veg_max, lowest, log)
        a, b = item5(d, source, veg_max)
        r5 += a
        r5b += b
        log(f"  item 5 {source}")
        r7 += item7(d, source, veg_max)
        log(f"  item 7 {source}")
        summ[source] = info
        del d
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(r4).to_csv(OUT / "strata.csv", index=False, float_format="%.6g")
    pd.DataFrame(r5).to_csv(OUT / "independence.csv", index=False, float_format="%.6g")
    pd.DataFrame(r5b).to_csv(OUT / "independence_near_far.csv", index=False, float_format="%.6g")
    pd.DataFrame(r7).to_csv(OUT / "support.csv", index=False, float_format="%.6g")
    (OUT / "summary.json").write_text(json.dumps({
        "amendment": "PREREG Amendment 20 items 4, 5 and 7", "decision": "D-161",
        "rule": f"model - (lidar + registered constant) on the paired points of the emerged band, 50 km blocks; "
                f"block bootstrap B = {N_BOOT}, seed {SEED}, 95% percentile intervals from a 1 mm histogram; descriptive",
        "registered_bare_population_veg_share_below": veg_max, "surface_edges": list(rev3.SURFACE_EDGES),
        "lowest_audited_level_m": lowest, "height_band_step_m": BAND_STEP_M, "n_height_bands": N_BANDS,
        "buffers_km": [b / 1000 for b in BUFFERS_M], "near_km": NEAR_M / 1000, "checked_sectors": list(CHECKED),
        "gradients_m_per_km": gradients(), "sources": summ}, indent=1, default=float), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
