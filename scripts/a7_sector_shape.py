"""A7 red-team sensitivity: is the H1 error a spatially varying datum and not seabed
shape? The D-070 shape constant is fitted per sector instead of once for the domain,
and the hindcast is rerun with it (G3 memo, red team).

1. Per product and sector k: c_k is the D-070 fit (scripts/a7_hindcast.fit_offset) on
   the training cells of sector k. It uses the same (epoch, level) units, record-low
   envelope and flood-level maps as the primary; within one sector a constant is a
   threshold on the primary's flood-level map, as in D-070. The per-sector bias tables
   sum to the domain table, and the domain fit must reproduce the primary's constants
   exactly (checked).
2. The offset map is c_k in sector k and the domain constant outside the sectors. It is
   smoothed by a Gaussian of SMOOTH_M on a COARSE-cell grid, so that no step at a sector
   edge acts as a dam or a breach, and interpolated bilinearly to the hindcast grid.
3. The products, corrected as in the primary (the D-074 correction) and less the offset
   map, are flood-filled again and audited with scripts/a7_hindcast.run, total variant
   only. The run's "total" rows are therefore the sector-shape variant.

Outputs: results/hindcast/<tag>_sectorshape/ (the run's tables and sector_offsets.json);
the H1 rule then runs with scripts/a7_h1_decision.py <tag>_sectorshape.
Usage: python scripts/a7_sector_shape.py main --vertical-correction <npz>
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import caspian  # noqa: F401
import numpy as np
import pandas as pd
from scipy import ndimage

from caspian import hindcast_tiled as ht
from caspian import products
from caspian.config import INTERIM, RESULTS
from caspian.geography import DOMAIN, SECTOR_ORDER

_spec = importlib.util.spec_from_file_location("a7_hindcast", Path(__file__).resolve().parent / "a7_hindcast.py")
a7 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(a7)

SMOOTH_M = 10_000.0     # Gaussian sigma of the offset map (m)
COARSE = 16             # coarse cells per side for the smoothing (16 x 60 m = 960 m)
SUFFIX = "_sectorshape"


def unit_counts_by_sector(F, code, outside, finite, region, sector, n_sectors: int, thresholds, *,
                          tile: int = ht.TILE):
    """a7_hindcast.unit_counts split by sector id (0 = no sector, 1..n_sectors): per
    sector, the known cells of ``region`` with F below each threshold, and those observed
    as water. Rows sum to unit_counts over the same region."""
    T = np.asarray(thresholds, float)
    if np.any(np.diff(T) <= 0):
        raise ValueError("thresholds must increase strictly")
    nb = T.size + 1
    hist = np.zeros((n_sectors + 1, nb), np.int64)
    water = np.zeros(n_sectors + 1, np.int64)
    for r0, r1, c0, c1 in ht.tiles(code.shape, tile):
        c, st = code[r0:r1, c0:c1], outside[r0:r1, c0:c1]
        ok = np.asarray(ht._get(finite, r0, r1, c0, c1), bool) & np.asarray(region[r0:r1, c0:c1], bool)
        k_in, k_out = (c >= 0) & ok, (c == -2) & (st >= 0) & ok
        sec = np.asarray(sector[r0:r1, c0:c1]).astype(np.int64)
        sec = np.where((sec >= 1) & (sec <= n_sectors), sec, 0)
        wet = ((c == 1) & k_in) | ((st == 1) & k_out)
        water += np.bincount(sec[wet], minlength=n_sectors + 1)
        known = k_in | k_out
        idx = np.searchsorted(T, np.asarray(F[r0:r1, c0:c1], float)[known], side="right")
        hist += np.bincount(sec[known] * nb + idx, minlength=(n_sectors + 1) * nb).reshape(n_sectors + 1, nb)
    return np.cumsum(hist, axis=1)[:, :T.size], water


def offset_map(sector, offsets: dict, default: float, out, *, coarse: int = COARSE, sigma_m: float = SMOOTH_M,
               pixel: float = a7.RES, tile: int = ht.TILE) -> None:
    """Write into ``out`` (the hindcast grid) the smoothed per-cell offset: ``offsets[k]``
    in sector k, ``default`` elsewhere, Gaussian-smoothed (sigma ``sigma_m``) on a grid of
    ``coarse`` cells and interpolated bilinearly back."""
    H, W = sector.shape
    rs = np.arange(coarse // 2, H, coarse)
    cs = np.arange(coarse // 2, W, coarse)
    sc = np.asarray(sector)[np.ix_(rs, cs)]
    cm = np.full(sc.shape, float(default))
    for k, v in offsets.items():
        cm[sc == k] = float(v)
    cm = ndimage.gaussian_filter(cm, sigma=sigma_m / (pixel * coarse), mode="nearest")
    for r0, r1, c0, c1 in ht.tiles((H, W), tile):
        rr, cc = np.meshgrid((np.arange(r0, r1) - coarse // 2) / coarse, (np.arange(c0, c1) - coarse // 2) / coarse,
                             indexing="ij")
        out[r0:r1, c0:c1] = ndimage.map_coordinates(cm, [rr, cc], order=1, mode="nearest").astype(np.float32)


def fit_sector_offsets(units, transform, static, flood_maps, finite, fit_region, sector_names, *, tile: int,
                       primary: dict, log=print) -> dict:
    sector = static["sector"]
    shape = sector.shape
    K = len(sector_names)
    ulist = [(b, ep) for b in sorted(units) for ep in sorted(units[b])]
    bias = {p: np.zeros((K + 1, len(ulist), a7.C_GRID.size), np.int64) for p in flood_maps}
    t0 = time.time()
    for i, (b, ep) in enumerate(ulist):
        raw, _ = a7.observed_raw(a7.groups_for(units, b, [ep]), transform, shape)
        code = a7.classify(raw, static, tile)
        del raw
        T = products.flood_level(a7._bin_level(b), 0.0, 0.0, a7.C_GRID)
        for p, F in flood_maps.items():
            cnt, water = unit_counts_by_sector(F, code, static["outside"], finite[p], fit_region, sector, K, T,
                                               tile=tile)
            bias[p][:, i] = cnt - water[:, None]
        log(f"sector fit: unit {ep} bin {b} ({i + 1}/{len(ulist)}), {time.time() - t0:.0f} s", flush=True)
    epochs = [ep for _, ep in ulist]
    cells = np.zeros(K + 1, np.int64)
    for r0, r1, c0, c1 in ht.tiles(shape, tile):
        s = np.asarray(sector[r0:r1, c0:c1]).astype(np.int64)
        s = np.where((s >= 1) & (s <= K), s, 0)
        cells += np.bincount(s[np.asarray(fit_region[r0:r1, c0:c1], bool)], minlength=K + 1)
    rec = {"rule": "D-070 fit per sector on its training cells; domain constant where a sector has no "
                   "training cell; offsets Gaussian-smoothed and subtracted from the corrected heights",
           "smooth_sigma_m": SMOOTH_M, "coarse_cells": COARSE,
           "training_cells_by_sector": {(sector_names[k - 1] if k else "none"): int(cells[k]) for k in range(K + 1)},
           "products": {}}
    for p in flood_maps:
        dom = a7.fit_offset(bias[p].sum(axis=0), epochs)
        want = float(primary[p]["offset_m"])
        if abs(dom["offset_m"] - want) > 1e-9:
            raise SystemExit(f"{p}: the summed sector tables give c = {dom['offset_m']}, the primary {want}")
        by = {}
        for k in range(1, K + 1):
            if cells[k] == 0:
                by[sector_names[k - 1]] = {"offset_m": want, "source": "domain constant (no training cells)"}
                continue
            r = a7.fit_offset(bias[p][k], epochs)
            by[sector_names[k - 1]] = {"offset_m": r["offset_m"], "objective_km2_at_offset": r["objective_km2_at_offset"],
                                       "n_tied_minima": r["n_tied_minima"], "source": "fitted"}
        rec["products"][p] = {"domain_offset_m": want, "sectors": by}
    return rec


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("tag")
    ap.add_argument("--vertical-correction", required=True)
    a = ap.parse_args(argv)
    main_dir = RESULTS / "hindcast" / a.tag
    run_rec = json.loads((main_dir / "run.json").read_text(encoding="utf-8"))
    primary = json.loads((main_dir / "shape_offsets.json").read_text(encoding="utf-8"))["used_in_this_run"]
    names, tile, resampling = list(run_rec["products"]), int(run_rec["tile"]), run_rec["resampling"]
    w, s, e, n = DOMAIN.bounds
    transform, shape = a7.grid((w - a7.MARGIN_DEG, s - a7.MARGIN_DEG, e + a7.MARGIN_DEG, n + a7.MARGIN_DEG))
    static = a7.production_static(transform, shape, tile)
    for k in ("zone", "land_seed"):
        static.pop(k)
    blocks = pd.read_csv(a7.BLOCKS)
    fit_region = static["domain"] & a7.train_mask(blocks, transform, shape)
    corr, corr_rec = products.correction(a.vertical_correction, transform, shape)
    if corr_rec["sha256"] != run_rec["vertical_correction"]["sha256"]:
        raise SystemExit("--vertical-correction is not the primary run's correction (sha256 differs)")
    heights, flood_maps, raw_h = {}, {}, {}
    for p in names:
        raw_h[p] = products.heights(p, transform, shape, resampling)
        heights[p] = products.Corrected(raw_h[p], corr)
        flood_maps[p] = a7.flood_map_cached(p, heights[p], static["sea_seed"],
                                            {"heights": Path(raw_h[p].filename).name, "correction": corr_rec["sha256"]},
                                            tile)
    finite = {p: (lambda h: lambda r0, r1, c0, c1: np.isfinite(np.asarray(h[r0:r1, c0:c1], float)))(h)
              for p, h in heights.items()}
    stage2 = INTERIM / "stage2" / a.tag
    units = a7.stage2_index(stage2)
    a7.ENVELOPE = a7.record_low_water(units, transform, shape)
    out = RESULTS / "hindcast" / (a.tag + SUFFIX)
    out.mkdir(parents=True, exist_ok=True)
    rec = fit_sector_offsets(units, transform, static, flood_maps, finite, fit_region, SECTOR_ORDER, tile=tile,
                             primary=primary)
    (out / "sector_offsets.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(json.dumps({p: {k: v["offset_m"] for k, v in r["sectors"].items()} for p, r in rec["products"].items()}),
          flush=True)
    cache = INTERIM / "hindcast" / "sector_shape"
    cache.mkdir(parents=True, exist_ok=True)
    s_heights, s_flood = {}, {}
    for p in names:
        offs = {k + 1: rec["products"][p]["sectors"][nm]["offset_m"] for k, nm in enumerate(SECTOR_ORDER)}
        key = hashlib.sha256(json.dumps({"offsets": offs, "default": rec["products"][p]["domain_offset_m"],
                                         "sigma": SMOOTH_M, "coarse": COARSE, "corr": corr_rec["sha256"],
                                         "shape": list(shape)}, sort_keys=True).encode()).hexdigest()[:12]
        path = cache / f"corr_{p}_{key}.npy"
        if not path.exists():
            tmp = path.with_suffix(".tmp.npy")
            mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=tuple(shape))
            offset_map(static["sector"], offs, rec["products"][p]["domain_offset_m"], mm, tile=tile)
            for r0, r1, c0, c1 in ht.tiles(shape, tile):          # corrected heights less the offset
                mm[r0:r1, c0:c1] = np.asarray(corr[r0:r1, c0:c1], np.float32) - mm[r0:r1, c0:c1]
            mm.flush()
            del mm
            tmp.replace(path)
        total = np.load(path, mmap_mode="r")
        s_heights[p] = products.Corrected(raw_h[p], total)
        s_flood[p] = a7.flood_map_cached(f"{p}{SUFFIX}", s_heights[p], static["sea_seed"],
                                         {"heights": Path(raw_h[p].filename).name, "correction": path.name}, tile)
        print("flood-level map", p, flush=True)
    meta = {"tag": a.tag, "variant": "sector shape (red team)", "resampling": resampling, "tile": tile,
            "envelope": run_rec.get("envelope"), "vertical_correction": corr_rec,
            "sector_offsets": "sector_offsets.json", "references": run_rec.get("references"),
            "note": "total rows = the sector-shape variant: products corrected as the primary, less the smoothed "
                    "per-sector D-070 constants; no further constant"}
    a7.run(stage2, out, static, s_heights, transform, product_offsets={}, shape_offsets=None, fit_region=fit_region,
           flood_maps=s_flood, blocks=blocks, tile=tile, meta=meta)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
