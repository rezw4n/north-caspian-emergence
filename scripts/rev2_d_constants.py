"""Second-review analysis D, area constants by subset (PREREG Amendment 19 item 6a).
Exploratory, post-review; the registered constants and verdicts stand. Defaults in D-149.

The D-070 area constant (scripts/a7_hindcast.fit_offset: the c on the 0.01 m grid over
+-5 m that minimises the sum over units of |area bias| on the domain's training-block
cells; ties to the smallest |c|, then the smaller c) is refitted on subsets of the 43
(epoch, level) units of the primary fit:
  - per epoch (E1, E2, E3);
  - per 0.5 m level band (lo, hi], edges at multiples of 0.5 m, epochs pooled
    (caspian.rev2_d.level_band);
  - per epoch at the levels two epochs share (E2 and E3 at the bins both hold; E1 and E2
    likewise), to separate epoch from level.
Each unit's signed area bias at every c is counted exactly as in the primary fit from the
cached fit masks (data/interim/revision_v/fit_masks.npz, scripts/rev_v_planar.py masks) and
the primary's cached flood-level maps (data/interim/hindcast/flood/; never recomputed here:
the run stops if a map is missing), by scripts/rev_v_planar.exact_bias.

Checks (the run stops if one fails): the all-unit fit reproduces each product's primary
constant and objective (results/hindcast/main/shape_offsets.json); the per-epoch objective
equals the primary objective less the leave-one-epoch-out objective stored there.

Stages:
  bias  HEAVY (run inside tools/heavy_slot.py): per product, the [unit, c] bias array ->
        data/interim/revision2/d/unit_bias.npz
  fit   the subset fits -> results/revision2/d/constants/{constants.csv, summary.json}
Usage: python scripts/rev2_d_constants.py bias|fit|all
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import caspian  # noqa: F401
import numpy as np
import pandas as pd

from caspian import products, rev2_d
from caspian.config import INTERIM, RESULTS

SCRIPTS = Path(__file__).resolve().parent
OUT = RESULTS / "revision2" / "d" / "constants"
CACHE = INTERIM / "revision2" / "d"
BIAS = CACHE / "unit_bias.npz"
PRIMARY = RESULTS / "hindcast" / "main" / "shape_offsets.json"
PRODUCTS = ["gebco_2026", "gebco_2022", "etopo_2022", "srtm15p_v2.7", "etopo1"]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


a7 = _load("a7_hindcast")


def flood_path(name: str, raw, corr_sha: str, seed, cache: Path = INTERIM / "hindcast" / "flood") -> Path:
    """The cache path scripts/a7_hindcast.flood_map_cached uses for these inputs."""
    key_meta = {"heights": Path(raw.filename).name, "correction": corr_sha}
    key = hashlib.sha256(json.dumps({**key_meta, "seed": hashlib.sha256(np.packbits(seed)).hexdigest(),
                                     "connectivity": 8}, sort_keys=True).encode()).hexdigest()[:12]
    return cache / f"{name}_{key}.npy"


def stage_bias() -> int:
    rvp = _load("rev_v_planar")
    run_rec, tile, transform, shape, static, blocks, fit_region = rvp.setup()
    del fit_region
    fd = rvp.FitData()
    prim = json.loads(PRIMARY.read_text(encoding="utf-8"))
    if fd.n != prim["fit_region_cells"]:
        raise SystemExit("the cached fit masks are not on the primary fit region")
    if [(int(b), str(e)) for b, e in zip(fd.bins, fd.epochs)] != [(u["bin"], u["epoch"]) for u in prim["units"]]:
        raise SystemExit("the cached fit masks are not the primary's units")
    corr, corr_rec = products.correction(rvp.CORRECTION, transform, shape)
    if corr_rec["sha256"] != run_rec["vertical_correction"]["sha256"]:
        raise SystemExit("the correction is not the primary run's")
    out, rec = {}, {}
    for p in PRODUCTS:
        t0 = time.time()
        raw = products.heights(p, transform, shape, run_rec["resampling"])
        path = flood_path(p, raw, corr_rec["sha256"], static["sea_seed"])
        if not path.exists():
            raise SystemExit(f"{p}: the primary flood-level map {path.name} is not cached; not recomputed here")
        F = np.load(path, mmap_mode="r")
        Fv = np.asarray(F).reshape(-1)[fd.idx]
        finite = np.zeros(fd.n, bool)       # as a7_hindcast: the corrected height (float32 sum) is finite
        flat_h, flat_c = np.asarray(raw).reshape(-1), np.asarray(corr).reshape(-1)
        for s0 in range(0, fd.n, 1 << 24):
            j = fd.idx[s0:s0 + (1 << 24)]
            finite[s0:s0 + j.size] = np.isfinite(flat_h[j].astype(np.float32) + flat_c[j].astype(np.float32))
        b = rvp.exact_bias(fd, Fv, finite)
        r0 = a7.fit_offset(b, fd.epochs)
        want = prim["products"][p]
        if r0["offset_m"] != want["offset_m"] or abs(r0["objective_km2_at_offset"] - want["objective_km2_at_offset"]) > 1e-3:
            raise SystemExit(f"{p}: exact counts give c={r0['offset_m']} obj={r0['objective_km2_at_offset']}, "
                             f"primary c={want['offset_m']} obj={want['objective_km2_at_offset']}")
        out[p] = b
        rec[p] = {"flood_map": path.name, "c_m": r0["offset_m"], "objective_km2": r0["objective_km2_at_offset"]}
        print(f"{p}: primary fit reproduced (c={r0['offset_m']}, {r0['objective_km2_at_offset']:.1f} km2), "
              f"{time.time() - t0:.0f} s", flush=True)
        del Fv, finite, F
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(BIAS, **{f"bias__{p}": v for p, v in out.items()}, levels=np.asarray(fd.levels, float),
             epochs=np.asarray(fd.epochs), bins=np.asarray(fd.bins), c_grid=a7.C_GRID,
             record=json.dumps(rec))
    print("written", BIAS, flush=True)
    return 0


def subsets(levels, epochs, bins) -> list[tuple[str, str, np.ndarray]]:
    levels, epochs, bins = np.asarray(levels, float), np.asarray(epochs).astype(str), np.asarray(bins, int)
    out = [("all", "all", np.ones(levels.size, bool))]
    for e in sorted(set(epochs)):
        out.append(("epoch", e, epochs == e))
    lo, hi = rev2_d.level_band(levels)
    for a, b in sorted(set(zip(lo.tolist(), hi.tolist()))):
        out.append(("band", f"({a:.1f}, {b:.1f}]", (lo == a) & (hi == b)))
    eps = sorted(set(epochs))
    for i, e1 in enumerate(eps):
        for e2 in eps[i + 1:]:
            shared = sorted(set(bins[epochs == e1]) & set(bins[epochs == e2]))
            if len(shared) < 2:
                continue
            tag = f"{e1}&{e2} bins {shared[0]}-{shared[-1]}"
            for e in (e1, e2):
                out.append(("epoch_shared_levels", f"{e} at {tag}", (epochs == e) & np.isin(bins, shared)))
    return out


def stage_fit() -> int:
    z = np.load(BIAS)
    levels, epochs, bins = z["levels"], [str(e) for e in z["epochs"]], z["bins"]
    if not np.array_equal(z["c_grid"], a7.C_GRID):
        raise SystemExit("cached bias is on another c grid")
    prim = json.loads(PRIMARY.read_text(encoding="utf-8"))
    rows = []
    for p in PRODUCTS:
        b = z[f"bias__{p}"]
        allobj = np.abs(b).sum(axis=0) * a7.CELL_KM2
        # check: per-epoch objective = primary objective - leave-one-epoch-out objective
        po = np.asarray(prim["products"][p]["objective_km2"])
        for e, rr in prim["products"][p]["leave_one_epoch_out"].items():
            mine = np.abs(b[np.array(epochs) == e]).sum(axis=0) * a7.CELL_KM2
            dev = np.max(np.abs(mine - (po - np.asarray(rr["objective_km2"]))))
            if dev > 5e-4:
                raise SystemExit(f"{p} {e}: per-epoch objective differs from the stored curves by {dev} km2")
        if np.max(np.abs(allobj - po)) > 5e-4:
            raise SystemExit(f"{p}: all-unit objective differs from the stored curve")
        for kind, label, m in subsets(levels, epochs, bins):
            r = a7.fit_offset(b[m], [e for e, k in zip(epochs, m) if k])
            lv = levels[m]
            rows.append({"product": p, "kind": kind, "subset": label, "n_units": int(m.sum()),
                         "level_min_m": float(lv.min()), "level_max_m": float(lv.max()),
                         "level_mean_m": float(lv.mean()), "epochs": ",".join(sorted(set(np.array(epochs)[m]))),
                         "c_m": r["offset_m"], "objective_km2": r["objective_km2_at_offset"],
                         "n_tied_minima": r["n_tied_minima"], "at_search_edge": abs(r["offset_m"]) >= 5.0})
    t = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "constants.csv", index=False)
    summ = {"rule": "A19 item 6a (D-149): D-070 area constant (scripts/a7_hindcast.fit_offset) refitted on subsets "
                    "of the 43 training-block units of results/hindcast/main/shape_offsets.json",
            "record": json.loads(str(z["record"])), "by_product": {}, "across_products": {}}
    for p, g in t.groupby("product", sort=False):
        ep = g[g["kind"] == "epoch"].set_index("subset")["c_m"]
        bd = g[g["kind"] == "band"]
        mid = (bd["level_min_m"] + bd["level_max_m"]) / 2
        slope = float(np.polyfit(mid, bd["c_m"], 1)[0]) if len(bd) >= 3 else None
        sh = g[g["kind"] == "epoch_shared_levels"].set_index("subset")["c_m"].to_dict()
        summ["by_product"][p] = {
            "c_all_m": float(g.loc[g["kind"] == "all", "c_m"].iloc[0]),
            "c_epoch_m": ep.to_dict(), "epoch_spread_m": float(ep.max() - ep.min()),
            "c_band_m": bd.set_index("subset")["c_m"].to_dict(), "band_spread_m": float(bd["c_m"].max() - bd["c_m"].min()),
            "band_slope_m_per_m": slope, "c_epoch_shared_levels_m": sh}
    for (kind, label), g in t.groupby(["kind", "subset"], sort=False):
        summ["across_products"][f"{kind}|{label}"] = {"min_m": float(g["c_m"].min()), "max_m": float(g["c_m"].max()),
                                                      "spread_m": float(g["c_m"].max() - g["c_m"].min()),
                                                      "n_units": int(g["n_units"].iloc[0])}
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    piv = t.pivot_table(index=["kind", "subset", "n_units"], columns="product", values="c_m", sort=False)[PRODUCTS]
    print(piv.to_string())
    print(json.dumps({p: {k: v for k, v in d.items() if k in ("epoch_spread_m", "band_spread_m", "band_slope_m_per_m")}
                      for p, d in summ["by_product"].items()}, indent=1))
    return 0


def main(argv) -> int:
    stage = argv[0] if argv else "all"
    if stage == "bias":
        return stage_bias()
    if stage == "fit":
        return stage_fit()
    if stage == "all":
        return stage_bias() or stage_fit()
    raise SystemExit(__doc__)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
