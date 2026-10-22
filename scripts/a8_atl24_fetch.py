"""Independent check (PREREG section 2 "ATL24 bathymetry where available"; Amendment
17): ICESat-2 ATL24 v002 bathymetry photons over the North Caspian domain.

NASA Earthdata (CMR search and HTTPS range reads through earthaccess and h5py; the token
comes from caspian.credentials and is never printed). The 995 granules that CMR lists
for the domain's bounding box total about 65,600 MB (CMR sizes), above a 60 GB download
limit, and each covers a whole ATL03 region, so most of a granule lies outside the
domain (the Persian Gulf, the Middle and South Caspian). Each granule is therefore read
in place and subset by space:
1. ``metadata/extent`` (the granule's data-extent polygon) is read; a granule whose
   extent misses the domain's bounding box is recorded with no photons and not read
   further;
2. otherwise, per beam, the latitude array is read in full, and only the index range
   inside the domain's latitude band is read for the other variables;
3. the class-40 (bathymetry) photons inside the bounding box are saved unchanged, with
   every variable of caspian.atl24.PHOTON_VARS and the beam, to
   data/raw/icesat2/atl24/<granule>.npz (git-ignored). Photons in the box are counted
   per class.

Provenance (committed): results/provenance/icesat2_atl24.json, one record per granule:
the extent test, the photons in the box per class, the number saved, the saved file's
size and sha256, the ATL03 resource, the release and the spacecraft orientation.
A spot check reads the latitude arrays of a random sample of granules whose extent
missed the box, and records how many photons they hold in the box (expected 0).

Usage: python scripts/a8_atl24_fetch.py [--workers 6] [--limit N] [--spot-check 20]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import caspian  # noqa: F401
import numpy as np

from caspian import atl24
from caspian.config import RAW, RESULTS
from caspian.geography import DOMAIN

OUT = RAW / "icesat2" / "atl24"
PROV = RESULTS / "provenance" / "icesat2_atl24.json"
TEMPORAL = ("2018-10-01", "2026-12-31")
OPEN_KW = {"cache_type": "blockcache", "block_size": 2 * 1024 * 1024}     # photon arrays
EXTENT_KW = {"cache_type": "blockcache", "block_size": 64 * 1024}        # metadata/extent only


def _init():
    from caspian import earthdata
    earthdata.login()


def _open(url: str, kw: dict):
    import earthaccess
    return earthaccess.open([url], provider="NSIDC_CPRD", open_kwargs=kw, show_progress=False)[0]


def read_one(url: str, spot_check: bool = False) -> dict:
    """Read one granule (module docstring). With ``spot_check``, the latitude arrays of
    every beam are read whatever the extent says, and nothing is saved."""
    import h5py
    from shapely.geometry import box

    bounds = DOMAIN.bounds
    name = Path(url).name
    rec = {"url": url}
    # the extent test reads a few small metadata objects: small blocks (latency-bound)
    f = _open(url, EXTENT_KW)
    try:
        with h5py.File(f, "r") as h:
            ext = h["metadata/extent"][()] if "metadata/extent" in h else None
    finally:
        f.close()
    poly = atl24.extent_polygon(ext) if ext is not None else None
    rec["extent_found"] = poly is not None
    rec["extent_intersects_box"] = bool(poly is None or poly.intersects(box(*bounds)))
    rec["extent_bounds_wsen"] = [round(v, 4) for v in poly.bounds] if poly is not None else None
    if not spot_check and not rec["extent_intersects_box"]:
        rec.update(n_box_by_class={}, n_saved=0)
        return rec
    f = _open(url, OPEN_KW)
    try:
        with h5py.File(f, "r") as h:
            if spot_check:
                n = 0
                for gt in atl24.BEAMS:
                    if gt in h and "lat_ph" in h[gt]:
                        lat = h[f"{gt}/lat_ph"][:]
                        band = (lat >= bounds[1]) & (lat <= bounds[3])
                        if band.any():
                            idx = np.flatnonzero(band)
                            lon = h[f"{gt}/lon_ph"][idx[0]:idx[-1] + 1]
                            n += int(atl24.box_mask(lat[idx[0]:idx[-1] + 1], lon, bounds).sum())
                rec["spot_check_n_box"] = n
                return rec
            arrays, info = atl24.read_granule(h, bounds)
    finally:
        f.close()
    rec.update(info)
    if info["n_saved"]:
        OUT.mkdir(parents=True, exist_ok=True)
        buf = io.BytesIO()
        np.savez_compressed(buf, **arrays, sc_orient=np.int8(info["sc_orient"]),
                            granule=np.array(name), atl03_resource=np.array(info["atl03_resource"] or ""))
        data = buf.getvalue()
        path = OUT / f"{Path(name).stem}.npz"
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)
        rec.update(file=path.name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    return rec


def search() -> list[dict]:
    import earthaccess

    from caspian import earthdata
    earthdata.login()
    w, s, e, n = DOMAIN.bounds
    res = earthaccess.search_data(short_name="ATL24", version="002", bounding_box=(w, s, e, n),
                                  temporal=TEMPORAL, count=-1)
    out = []
    for g in res:
        umm = g["umm"]
        links = [u["URL"] for u in umm.get("RelatedUrls", []) if u.get("Type") == "GET DATA"
                 and u["URL"].startswith("https")]
        info = (umm.get("DataGranule", {}).get("ArchiveAndDistributionInformation") or [{}])[0]
        out.append({"granule_ur": umm["GranuleUR"], "url": links[0] if links else None,
                    "size_mib_cmr": float(g.size()), "checksum_cmr": info.get("Checksum")})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--spot-check", type=int, default=20)
    a = ap.parse_args()
    found = search()
    prov = json.loads(PROV.read_text(encoding="utf-8")) if PROV.exists() else {"granules": {}}
    prov.update(product="ATL24", version="002", doi="10.5067/ATLAS/ATL24.002",
                search_bbox_wsen=list(DOMAIN.bounds), temporal=list(TEMPORAL), n_granules_found=len(found),
                size_mib_cmr_total=round(sum(g["size_mib_cmr"] or 0 for g in found), 1),
                method=("HTTPS range reads (earthaccess.open, fsspec blockcache: 64 KiB for the extent test, 2 MiB for photons); subset by space: class-40 "
                        "photons inside the domain bounding box, all variables of caspian.atl24.PHOTON_VARS; "
                        "granules whose metadata/extent polygon misses the box are not read further (D-095)"),
                output_dir="data/raw/icesat2/atl24")
    meta = {Path(g["url"]).name: g for g in found if g["url"]}
    todo = [g["url"] for g in found if g["url"] and Path(g["url"]).name not in prov["granules"]]
    if a.limit:
        todo = todo[:a.limit]
    print(f"ATL24 v002: {len(found)} granules, {len(todo)} to read", flush=True)
    t0, k = time.time(), 0
    with ProcessPoolExecutor(a.workers, initializer=_init) as pool:
        futs = {pool.submit(read_one, u): u for u in todo}
        for fut in as_completed(futs):
            u = futs[fut]
            try:
                rec = fut.result()
            except Exception as ex:  # left for a rerun
                print("error", Path(u).name, str(ex)[:160], flush=True)
                continue
            g = meta[Path(u).name]
            rec.update(size_mib_cmr=g["size_mib_cmr"], checksum_cmr=g["checksum_cmr"])
            prov["granules"][Path(u).name] = rec
            k += 1
            if k % 25 == 0:
                PROV.write_text(json.dumps(prov, indent=0), encoding="utf-8")
                print(f"{k}/{len(todo)} granules, {k / (time.time() - t0) * 60:.1f}/min", flush=True)
    recs = prov["granules"].values()
    prov["summary"] = {"n_read": len(prov["granules"]),
                       "n_extent_intersects_box": sum(bool(r.get("extent_intersects_box")) for r in recs),
                       "n_with_saved_photons": sum(int(r.get("n_saved", 0)) > 0 for r in recs),
                       "n_photons_saved": sum(int(r.get("n_saved", 0)) for r in recs),
                       "bytes_saved": sum(int(r.get("bytes", 0)) for r in recs),
                       "n_box_by_class": {c: sum(int(r.get("n_box_by_class", {}).get(c, 0)) for r in recs)
                                          for c in ("0", "1", "40", "41")}}
    # spot check of granules whose extent missed the box
    if a.spot_check and "spot_check" not in prov:
        miss = sorted(n for n, r in prov["granules"].items() if r.get("extent_intersects_box") is False)
        rng = np.random.default_rng(20261020)
        pick = [miss[i] for i in sorted(rng.choice(len(miss), min(a.spot_check, len(miss)), replace=False))] if miss else []
        res = {}
        with ProcessPoolExecutor(a.workers, initializer=_init) as pool:
            futs = {pool.submit(read_one, prov["granules"][n]["url"], True): n for n in pick}
            for fut in as_completed(futs):
                try:
                    res[futs[fut]] = fut.result()["spot_check_n_box"]
                except Exception as ex:
                    res[futs[fut]] = f"error: {str(ex)[:120]}"
        prov["spot_check"] = {"rule": "latitude arrays of every beam read for a random sample (seed 20261020) of "
                                      "granules whose extent missed the box; photons in the box counted",
                              "n_sampled": len(pick), "n_box_by_granule": res}
    prov["updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    PROV.write_text(json.dumps(prov, indent=0), encoding="utf-8")
    print("done", k, json.dumps(prov["summary"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
