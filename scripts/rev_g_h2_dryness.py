"""Revision workstream G (PREREG Amendment 18 item 11, H2; D-129; R1 minor 8): H2 re-evaluated
with a dryness test that does not use the waterline DEM. Exploratory and descriptive; the
registered H2 verdict ("not supported", D-086) stands.

The registered test (validation.dry_at_acquisition) keeps an ICESat-2 point only if the
DEM's emergence dates say its pixels were dry at acquisition, so the reference set
depends on the DEM under test. Here a point is dry when its 60 m stage-2 cell is valid
and observed dry in the E3 observed extent of the level bin of its acquisition date:

1. every recorded granule of the source, through the registered quality steps
   (validation.prepare_icesat2 with the EIGEN-6C4 geoid, D-078), as
   scripts/a6_validate_dem.py prepare does, but without the DEM-based dry step;
2. the basin level at the granule's start time (file name) from the scene-level function
   of scripts/a4_scene_levels.py (DAHITI; Hydroweb shifted onto DAHITI in DAHITI's gaps),
   and its 0.10 m bin (PREREG section 4 origin); epoch E3 (every granule is after 2011);
3. the E3 observed extent of that bin, both parity halves summed, with the registered
   water rule and no sea connection or envelope (scripts/a7_hindcast.observed_raw):
   dry = valid and not water (caspian.rev_g.independent_dry). Points whose bin has no E3
   extent, or whose cell is not valid, are left out and counted;
4. then as registered: in the emerged band and with an E3 vegetation share below 0.5
   (the bare population, D-084), DEM and product heights, 50 km blocks, validation.run
   (B = 2,000, seed 20261022), and the H2 rule (validation.h2_decision; both sources,
   D-069).

The DEM-based flag is computed for the same in-band points, for a 2 x 2 agreement table.

Stages:
  prepare   writes data/interim/revision_g/h2_dryness/<SOURCE>_points.csv.gz and
            results/revision/g/h2_dryness/<SOURCE>_prepare.json (counts, agreement)
  evaluate  writes results/revision/g/h2_dryness/<SOURCE>_{accuracy,accuracy_blocks,paired_bootstrap,
            coverage}.csv, <SOURCE>_counts.json and the fits
  decide    writes results/revision/g/h2_dryness/h2_decision.json (the rule on both sources)
Usage: python scripts/rev_g_h2_dryness.py <prepare|evaluate|decide> --sources ATL08[,ATL06]
       [--max-granules N]  (pilot)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import zlib
from pathlib import Path

import importlib.util

import caspian  # noqa: F401
import numpy as np
import pandas as pd
import pyproj

from caspian import design, frequency, levels, rev_g, validation
from caspian.config import CRS_EQUAL_AREA, INTERIM, PROCESSED, RESULTS
from caspian.geography import DOMAIN

ROOT = Path(__file__).resolve().parents[1]
OUT = RESULTS / "revision" / "g" / "h2_dryness"
CACHE = INTERIM / "revision_g" / "h2_dryness"
SEED = 20261022
PRODUCTS = ("gebco_2026", "etopo_2022", "srtm15p_v2.7", "gebco_2022", "etopo1")
VEG = re.compile(r"([ov])_E3_b\d{3}_p[01]$")


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def granule_time(name: str) -> pd.Timestamp:
    return pd.Timestamp(pd.to_datetime(Path(name).name.split("_")[1], format="%Y%m%d%H%M%S"))


def veg_share_at(x, y, stage2_dir: Path) -> np.ndarray:
    """E3 emergent-vegetation share (o - v) / o of each point's 60 m stage-2 cell (NaN with
    fewer than design.MIN_VALID_OBS classified acquisitions); the rule of
    scripts/a6_validate_dem._annotations."""
    veg = np.full(len(x), np.nan)
    for f in sorted(Path(stage2_dir).glob("tile_*.npz")):
        z = np.load(f)
        o = v = None
        for k in z.files:
            m = VEG.match(k)
            if m:
                arr = z[k].astype(np.int32)
                if m.group(1) == "o":
                    o = arr if o is None else o + arr
                else:
                    v = arr if v is None else v + arr
        if o is None:
            continue
        share = frequency.vegetation_share(o, v, design.MIN_VALID_OBS)
        t = z["transform"]
        r, c = np.floor((y - t[5]) / t[4]).astype(int), np.floor((x - t[2]) / t[0]).astype(int)
        hit = (r >= 0) & (r < share.shape[0]) & (c >= 0) & (c < share.shape[1]) & np.isnan(veg)
        veg[hit] = share[r[hit], c[hit]]
    return veg


def stage_prepare(a) -> int:
    A4 = _load("a4_scene_levels", "a4_scene_levels.py")
    A7 = _load("a7_hindcast", "a7_hindcast.py")
    drv = _load("a6_validate_dem", "a6_validate_dem.py")
    t0 = time.time()
    w, s, e, n = DOMAIN.bounds
    transform, shape = A7.grid((w - A7.MARGIN_DEG, s - A7.MARGIN_DEG, e + A7.MARGIN_DEG, n + A7.MARGIN_DEG))
    units = A7.stage2_index(INTERIM / "stage2" / "main")
    e3_bins = sorted(b for b in units if "E3" in units[b])
    ex = np.load(RESULTS / "study_area" / "extents_v6_250m.npz")
    band = (ex["high"] == 1) & (ex["low"] == 0)
    et = ex["transform"]
    fwd = pyproj.Transformer.from_crs("EPSG:4326", CRS_EQUAL_AREA, always_xy=True)
    geoid = validation.EigenGeoid()
    em = validation.TileSet(PROCESSED / "dem" / "main" / "emergence", "emergence")
    prep = A4.prepare()
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for source in a.sources:
        rec = drv._recorded_granules(source)
        names = sorted(rec)
        if a.max_granules:
            names = names[:: max(1, len(names) // a.max_granules)][: a.max_granules]
        times = [granule_time(k) for k in names]
        lv, _ = A4.levels_at(prep, times)
        gb = pd.DataFrame({"granule": names, "time": times, "level_m": lv["level_m"].to_numpy(),
                           "level_source": lv["level_source"].to_numpy()})
        gb["bin"] = np.where(np.isfinite(gb["level_m"]),
                             levels.bin_index(gb["level_m"].fillna(0), design.BIN_WIDTH_M, design.BIN_ORIGIN_M), -999)
        gb["has_e3_extent"] = gb["bin"].isin(e3_bins)
        counts = {"granules": len(names), "granules_without_level": int((gb["bin"] == -999).sum()),
                  "granules_bin_without_e3_extent": int((~gb["has_e3_extent"] & (gb["bin"] != -999)).sum()),
                  "points_quality": 0, "points_in_band": 0, "points_in_band_bin_without_extent": 0,
                  "points_in_band_cell_not_valid": 0, "points_in_band_observed_water": 0,
                  "points_in_band_observed_dry": 0, "files_missing_or_incomplete": 0}
        kept = []
        r_dem = validation.DRY_RADIUS_PX[source]
        for b, gg in gb.groupby("bin"):
            code = None
            if b in e3_bins:
                code, _ = A7.observed_raw(A7.groups_for(units, int(b), ["E3"]), transform, shape, envelope=False)
            for name in gg["granule"]:
                path = INTERIM / "icesat2" / source / f"{name}.csv.gz"
                try:
                    d = validation.read_icesat2_csv(path, source)
                except (FileNotFoundError, EOFError, OSError, ValueError, UnicodeDecodeError, zlib.error,
                        pd.errors.ParserError):
                    counts["files_missing_or_incomplete"] += 1
                    continue
                if len(d) != rec[name]:
                    counts["files_missing_or_incomplete"] += 1
                    continue
                d, _ = validation.prepare_icesat2(d, source, geoid=geoid)
                counts["points_quality"] += len(d)
                x, y = fwd.transform(d["lon"].to_numpy(float), d["lat"].to_numpy(float))
                er, ec = np.floor((y - et[5]) / et[4]).astype(int), np.floor((x - et[2]) / et[0]).astype(int)
                ok = (er >= 0) & (er < band.shape[0]) & (ec >= 0) & (ec < band.shape[1])
                inb = np.zeros(len(d), bool)
                inb[ok] = band[er[ok], ec[ok]]
                counts["points_in_band"] += int(inb.sum())
                if not inb.any():
                    continue
                d = d.loc[inb].reset_index(drop=True)
                x, y = x[inb], y[inb]
                if code is None:
                    counts["points_in_band_bin_without_extent"] += len(d)
                    continue
                st = rev_g.cell_states(x, y, code, transform)
                dry = rev_g.independent_dry(st)
                counts["points_in_band_cell_not_valid"] += int((st < 0).sum())
                counts["points_in_band_observed_water"] += int((st == 1).sum())
                counts["points_in_band_observed_dry"] += int(dry.sum())
                testable = st >= 0
                if testable.any():
                    kept.append(pd.DataFrame({"lon": d["lon"].to_numpy(float)[testable],
                                              "lat": d["lat"].to_numpy(float)[testable],
                                              "time": d["time"].to_numpy()[testable],
                                              "h_orth": d["h_orth"].to_numpy(float)[testable],
                                              "x": x[testable], "y": y[testable], "obs_dry": dry[testable],
                                              "bin": int(b), "granule": name}))
            del code
            print(source, "bin", b, len(gg), "granules", f"{time.time() - t0:.0f} s", flush=True)
        pts = pd.concat(kept, ignore_index=True)
        del kept
        dem_dry = np.zeros(len(pts), bool)
        for k in range(0, len(pts), 500_000):          # DEM-based flag, for the agreement table only
            sl = slice(k, k + 500_000)
            xx, yy = pts["x"].to_numpy(float)[sl], pts["y"].to_numpy(float)[sl]
            dem_dry[sl] = validation.dry_at_acquisition(pts["time"].to_numpy()[sl],
                                                        em.sample_window(xx, yy, 1, r_dem).reshape(xx.size, -1))
        od = pts["obs_dry"].to_numpy(bool)
        agree = np.array([[np.sum(~dem_dry & ~od), np.sum(~dem_dry & od)], [np.sum(dem_dry & ~od), np.sum(dem_dry & od)]])
        pts = pts.loc[od].assign(dem_dry=dem_dry[od]).reset_index(drop=True)
        print(source, "flags done", f"{time.time() - t0:.0f} s", flush=True)
        pts["veg_share"] = veg_share_at(pts["x"].to_numpy(float), pts["y"].to_numpy(float), INTERIM / "stage2" / "main")
        bare = pts["veg_share"].to_numpy(float) < drv.VEG_CELL_SHARE
        counts["points_observed_dry_bare"] = int(bare.sum())
        pts = pts.loc[bare].reset_index(drop=True)
        names_p, sample = drv._product_sampler(None, False)
        dem = validation.TileSet(PROCESSED / "dem" / "main", "dem")
        cx, cy = drv._contour_points("main", ["E2", "E3"])
        pts = drv.attach_model_columns(pts, dem, cx, cy, names_p, sample)
        counts["points_observed_dry_bare_with_dem"] = int(np.isfinite(pts["dem_h"]).sum())
        pts.to_csv(CACHE / f"{source}_points.csv.gz", index=False, float_format="%.9g")
        gb.to_csv(OUT / f"{source}_granule_levels.csv", index=False)
        res = {"source": source, "rule": "D-129: observed dry in the E3 extent of the level bin of the granule date "
                                         "(no DEM); bare population; 50 km blocks",
               "counts": counts, "seconds": round(time.time() - t0, 1),
               "agreement_in_band_testable": {"dem_dry_no__observed_dry_no": int(agree[0, 0]),
                                              "dem_dry_no__observed_dry_yes": int(agree[0, 1]),
                                              "dem_dry_yes__observed_dry_no": int(agree[1, 0]),
                                              "dem_dry_yes__observed_dry_yes": int(agree[1, 1])},
               "bare_points_also_dem_dry": int(pts["dem_dry"].sum()), "bare_points": int(len(pts)),
               "max_granules": a.max_granules}
        (OUT / f"{source}_prepare.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(json.dumps(res, indent=1), flush=True)
    return 0


def stage_evaluate(a) -> int:
    drv = _load("a6_validate_dem", "a6_validate_dem.py")
    for source in a.sources:
        d = pd.read_csv(CACHE / f"{source}_points.csv.gz")
        d["time"] = pd.to_datetime(d["time"], format="ISO8601").to_numpy().astype("datetime64[ns]")
        pts = validation.attach_blocks(d, drv._blocks(50000.0), 50000.0)
        prods = {c.split(":", 1)[1]: c for c in pts.columns if c.startswith("prod:")}
        r = validation.run(pts, prods, n_boot=design.BOOTSTRAP_N, seed=SEED)
        for key in ("constant", "product_offsets", "scale_model", "calibration"):
            drv._write(OUT / f"{source}_{key}.json", {"source": source, key: r[key]})
        for key in ("coverage", "accuracy", "accuracy_blocks", "paired_bootstrap"):
            r[key].assign(source=source).to_csv(OUT / f"{source}_{key}.csv", index=False)
        sp = pts["split"]
        cnt = {"n_points": int(len(pts)), "n_outside_block_table": int(sp.isna().sum()),
               "n_test_points": int((sp == "test").sum()),
               "n_test_blocks": int(pts.loc[sp == "test", "block"].nunique()),
               "test_by_sector": pts.loc[sp == "test"].groupby("sector").size().to_dict()}
        (OUT / f"{source}_counts.json").write_text(json.dumps(cnt, indent=1), encoding="utf-8")
        print(source, cnt, flush=True)
    return 0


def stage_decide(a) -> int:
    out = {}
    for src in ("ATL06", "ATL08"):
        f = OUT / f"{src}_paired_bootstrap.csv"
        if f.exists():
            pb = pd.read_csv(f)
            out[src] = {v: validation.h2_decision(pb[pb["variant"] == v], PRODUCTS) for v in sorted(pb["variant"].unique())}
    dec = {}
    if len(out) == 2:
        for v in out["ATL06"]:
            st = [out[s][v]["overall"] for s in ("ATL06", "ATL08")]
            dec[v] = ("supported" if all(x == "supported" for x in st) else
                      "partial support" if any(x == "supported" for x in st) else "not supported")
    res = {"status": "exploratory, descriptive (D-129); the registered H2 verdict (D-086) stands",
           "decision_both_sources": dec or "needs both sources",
           "by_source": {s: {v: {"overall": r["overall"], "sectors_supported": r["sectors_supported"]}
                             for v, r in vv.items()} for s, vv in out.items()}}
    (OUT / "h2_decision.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))
    return 0


def main(argv) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "evaluate", "decide"])
    ap.add_argument("--sources", default="ATL08")
    ap.add_argument("--max-granules", type=int, default=None)
    a = ap.parse_args(argv)
    a.sources = a.sources.split(",")
    return {"prepare": stage_prepare, "evaluate": stage_evaluate, "decide": stage_decide}[a.stage](a)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
