"""A5 input: SWOT water-surface elevation per pass and 25 km block over the North
Caspian, for calibrating the wind-setup model (brief section 8 A5; PREREG section 3).
Statistics are kept per block (results/study_area/blocks_25km.csv, PREREG section 6),
so that the model is fitted on training blocks only and can be refitted if the
blocks are enlarged (25 km blocks nest in larger aligned blocks).

The hydroweb.next zip packaging of the SWOT_L2_HR_RASTER_250M collection ran at about
4% per hour, so granules are listed from the hydroweb.next STAC API and downloaded
directly, in parallel. For each pixel, the latest processing version of each granule
is kept: baseline D over C, forward (G) over interim (I), then the version counters.

Per granule (UTM tile, pass, time) and 25 km block, pixels pass all of:
- wse_qual <= 1 (good or suspect; on a tested open-sea tile, good-only kept 4,329
  pixels and good-or-suspect 14,516, with medians -27.732 and -27.715 m);
- water_frac >= MIN_WATER_FRAC (interior water);
- ice_clim_flag and ice_dyn_flag not 1 or 2 (both flags are mostly fill, i.e. no
  ice information, which counts as ice-free);
- a finite wse;
- inside the open sea (results/study_area/sea_mask_1km.npz, scripts/a5_sea_mask.py):
  "suspect" pixels over desert and salt flats gave sector medians near -13 m on a
  test tile, so SWOT is used over the open sea only, where the setup is measured.

For each block the script records the pixel counts, the median, mean and NMAD of wse,
and the median illumination time. The granule file is deleted after reduction, keeping only
the granule ID, size and checksum.

wse is "water surface elevation above geoid ... after using models to subtract the
effects of tides (solid_earth_tide, load_tide_fes, pole_tide)" (file attributes; ATBD
JPL D-105507 Eq. 11). The geoid is referred to the mean tide system; its model is not
named in the file attributes (docs/DATUMS.md).

Outputs:
- results/levels/swot_block_levels.csv (committed);
- results/provenance/swot_granules.json (granule IDs, sizes, checksums; committed).

Usage: python scripts/a5_swot_sector_levels.py [--workers 8]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import caspian  # noqa: F401
import numpy as np
import pyproj
import requests
import xarray as xr
from shapely import contains_xy

from caspian import credentials
from caspian.config import CRS_EQUAL_AREA, RESULTS
from caspian.geography import DOMAIN, sectors

STAC = "https://hydroweb.next.theia-land.fr/api/v1/rs-catalog/stac/search"
COLLECTION = "SWOT_L2_HR_RASTER_250M"
MIN_WATER_FRAC = 0.9
MIN_PIXELS = 10
BLOCK_M = 25_000
OUT = RESULTS / "levels" / "swot_block_levels.csv"
PROV = RESULTS / "provenance" / "swot_granules.json"
FIELDS = ["granule", "crid", "block_i", "block_j", "time_utc", "n_px", "n_px_good", "wse_median_m", "wse_mean_m",
          "wse_nmad_m", "wse_uncert_median_m"]
NAME = re.compile(r"(SWOT_L2_HR_Raster_250m_\w+?_N_x_x_x_\d+_\d+_\d+F_\d{8}T\d{6}_\d{8}T\d{6})_(P[GI][A-Z])(\d)_(\d+)\.nc")
_lock = Lock()
_SEA = np.load(RESULTS / "study_area" / "sea_mask_1km.npz")
SEA_MASK, SEA_T = _SEA["mask"].astype(bool), _SEA["transform"]
TO_LAEA = pyproj.Transformer.from_crs("EPSG:4326", CRS_EQUAL_AREA, always_xy=True)


def in_sea(lon, lat):
    x, y = TO_LAEA.transform(lon, lat)
    a, _, c, _, e, f = SEA_T
    col = np.floor((x - c) / a).astype(int)
    row = np.floor((y - f) / e).astype(int)
    ok = (row >= 0) & (row < SEA_MASK.shape[0]) & (col >= 0) & (col < SEA_MASK.shape[1])
    out = np.zeros(lon.shape, bool)
    out[ok] = SEA_MASK[row[ok], col[ok]]
    return out


def list_granules(headers) -> list[dict]:
    w, s, e, n = DOMAIN.bounds
    params = {"collections": COLLECTION, "bbox": f"{w},{s},{e},{n}", "limit": 200}
    items, nxt = [], None
    while True:
        r = requests.get(nxt or STAC, headers=headers, params=None if nxt else params, timeout=120)
        r.raise_for_status()
        j = r.json()
        items += j["features"]
        nxt = next((ln["href"] for ln in j.get("links", []) if ln.get("rel") == "next"), None)
        if not nxt or not j["features"]:
            break
    best = {}
    for it in items:
        for key, a in it["assets"].items():
            m = NAME.match(key)
            if not m:
                continue
            base, crid, cnum, prod = m.groups()
            rec = {"granule": key, "base": base, "crid": f"{crid}{cnum}", "href": a["href"],
                   "size": a.get("file:size"), "checksum": a.get("file:checksum"),
                   "rank": (crid[2], crid[1] == "G", int(cnum), int(prod))}
            if base not in best or rec["rank"] > best[base]["rank"]:
                best[base] = rec
    return sorted(best.values(), key=lambda r: r["granule"])


def reduce_granule(path: Path, granule: str, crid: str, secs) -> list[dict]:
    ds = xr.open_dataset(path)
    ok = ((ds["wse_qual"] <= 1) & (ds["water_frac"] >= MIN_WATER_FRAC) & (ds["ice_clim_flag"].fillna(0) == 0)
          & (ds["ice_dyn_flag"].fillna(0) == 0) & np.isfinite(ds["wse"])).values
    if not ok.any():
        return []
    zone = int(ds.attrs["utm_zone_num"])
    xx, yy = np.meshgrid(ds["x"].values, ds["y"].values)
    lon, lat = pyproj.Transformer.from_crs(f"EPSG:{32600 + zone}", "EPSG:4326", always_xy=True).transform(xx[ok], yy[ok])
    sea = in_sea(lon, lat)
    lon, lat = lon[sea], lat[sea]
    ok_idx = np.flatnonzero(ok)[sea]
    ok = np.zeros(ok.shape, bool)
    ok.ravel()[ok_idx] = True
    wse = ds["wse"].values[ok]
    unc = ds["wse_uncert"].values[ok]
    tim = ds["illumination_time"].values[ok]
    good = (ds["wse_qual"].values[ok] == 0)
    x, y = TO_LAEA.transform(lon, lat)
    bi, bj = np.floor(x / BLOCK_M).astype(int), np.floor(y / BLOCK_M).astype(int)
    rows = []
    for (a_, b_) in sorted(set(zip(bi.tolist(), bj.tolist()))):
        inside = (bi == a_) & (bj == b_)
        if inside.sum() < MIN_PIXELS:
            continue
        w = wse[inside]
        med = float(np.median(w))
        t = np.sort(tim[inside])[inside.sum() // 2]
        rows.append({"granule": granule, "crid": crid, "block_i": int(a_), "block_j": int(b_),
                     "time_utc": str(np.datetime_as_string(t, unit="s")), "n_px": int(inside.sum()),
                     "n_px_good": int((inside & good).sum()), "wse_median_m": round(med, 4),
                     "wse_mean_m": round(float(w.mean()), 4),
                     "wse_nmad_m": round(float(1.4826 * np.median(np.abs(w - med))), 4),
                     "wse_uncert_median_m": round(float(np.nanmedian(unc[inside])), 4)})
    ds.close()
    return rows


def work(g, headers, secs, tmpdir) -> tuple[dict, list[dict]]:
    path = Path(tmpdir) / g["granule"]
    for attempt in range(4):
        try:
            with requests.get(g["href"], headers=headers, timeout=900, stream=True) as r:
                r.raise_for_status()
                with path.open("wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            break
        except Exception:
            if attempt == 3:
                raise
            time.sleep(10 * (attempt + 1))
    try:
        rows = reduce_granule(path, g["granule"], g["crid"], secs)
    finally:
        path.unlink(missing_ok=True)
    return {k: g[k] for k in ("granule", "crid", "size", "checksum")}, rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    key = credentials.get_secret("HYDROWEB_API_KEY")
    if not key:
        print("FAIL HYDROWEB_API_KEY is not set")
        return 1
    headers = {"X-Api-Key": key}
    grans = list_granules(headers)
    done = set()
    if OUT.exists():
        with OUT.open(encoding="utf-8") as f:
            done = {r["granule"] for r in csv.DictReader(f)}
    prov = json.loads(PROV.read_text(encoding="utf-8")) if PROV.exists() else {"granules": []}
    done |= {g["granule"] for g in prov["granules"]}
    todo = [g for g in grans if g["granule"] not in done]
    print(f"{len(grans)} granules (latest versions); {len(todo)} to process", flush=True)
    secs = sectors()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    new = not OUT.exists()
    t0, n_done = time.time(), 0
    with tempfile.TemporaryDirectory() as tmpdir, OUT.open("a", newline="", encoding="utf-8") as f, \
            ThreadPoolExecutor(a.workers) as pool:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        futs = [pool.submit(work, g, headers, secs, tmpdir) for g in todo]
        for fut in as_completed(futs):
            try:
                meta, rows = fut.result()
            except Exception as e:  # leave the granule for a rerun
                print("error:", str(e)[:200], flush=True)
                continue
            with _lock:
                w.writerows(rows)
                f.flush()
                prov["granules"].append(meta)
                n_done += 1
                if n_done % 50 == 0:
                    PROV.write_text(json.dumps(prov, indent=0), encoding="utf-8")
                    rate = n_done / (time.time() - t0)
                    print(f"{n_done}/{len(todo)} granules, {rate * 60:.1f}/min", flush=True)
    prov.update(collection=COLLECTION, min_water_frac=MIN_WATER_FRAC, min_pixels=MIN_PIXELS,
                selection="latest processing version per granule (baseline D > C, G > I, counters)")
    PROV.write_text(json.dumps(prov, indent=0), encoding="utf-8")
    print("done", n_done, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
