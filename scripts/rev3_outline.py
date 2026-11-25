"""Journal revision (D-163): the Caspian's February 2000 shoreline as a vector polygon, for the
display rule of PREREG Amendment 20 item 1e.

The amendment named the HydroLAKES lake polygon "as rasterised in GLWD v2" (15 arc-second
cells, about 325 by 461 m here). On the delta fronts that raster is too coarse: moving the
outline by one cell changes the area of the band landward of it by -10 to +16 per cent (a
planning look made before the download, not a reported result). Its
main-class layer also gives a cell the wetland class with the largest share of the cell, so
a cell that the lake only touches can carry the lake class; and GLWD v2 rasterised version
1.1 of HydroLAKES (Lehner et al. 2025, section 3.2.1). The polygon itself is used instead:
HydroLAKES v1.0, lake 1 (Caspian Sea), whose polygon source is the SRTM Water Body Data,
"delineated as they appeared at the time of the data collection in February 2000"
(HydroLAKES Technical Documentation v1.0). The rasterised version stays in the tables as
the registered form, beside the polygon.

fetch   Download HydroLAKES_polys_v10_shp.zip (782 MB; CC BY 4.0) from data.hydrosheds.org to
        data/raw/hydrolakes/ (git-ignored) with <file>.meta.json (URL, time, size, SHA-256),
        and write the acquisition record results/provenance/hydrolakes_polys.json.
        A network step, outside `make all`. The archive is data: it is read in place through
        GDAL (/vsizip/), never extracted and never executed.
mask    Check the archive against its acquisition record (size and SHA-256), read the polygon
        of lake 1, check its name and source (also against the HydroLAKES points table, where
        that is on disk), clip it to the study domain's bounding box, and mark every cell of
        the 250 m study-area grid whose centre lies inside it (rasterio.features.rasterize in
        the equal-area projection). Writes results/study_area/caspian_feb2000_250m.npz (the
        mask, with the grid's transform and CRS) and caspian_feb2000.json (provenance, vertex
        and area counts, and the agreement with the GLWD v2 raster).

Usage: python scripts/rev3_outline.py <fetch|mask>
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import caspian  # noqa: F401
import numpy as np

from caspian import rev3
from caspian.config import CRS_EQUAL_AREA, RAW, RESULTS

URL ="https://data.hydrosheds.org/file/hydrolakes/HydroLAKES_polys_v10_shp.zip"
ZIP = RAW / "hydrolakes" / "HydroLAKES_polys_v10_shp.zip"
POINTS = RAW / "revision_g" / "etopo" / "HydroLAKES_points_v10_shp.zip"
EXTENTS = RESULTS / "study_area" / "extents_v6_250m.npz"
GLWD = RAW / "habitat" / "glwd_v2" / "GLWD_v2_0_main_class.tif"
OUT_NPZ = RESULTS / "study_area" / "caspian_feb2000_250m.npz"
OUT_JSON = RESULTS / "study_area" / "caspian_feb2000.json"
PROV = RESULTS / "provenance" / "hydrolakes_polys.json"
HYLAK_ID = 1
UA = "generic-client"                      # nothing about the project or its authors is sent
ROOT = Path(__file__).resolve().parents[1]


def utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> tuple[int, str]:
    h, n = hashlib.sha256(), 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
            n += len(chunk)
    return n, h.hexdigest()


def write_record(m: dict) -> None:
    """The committed acquisition record (the convention of results/provenance/): what was
    retrieved, from where and when, with its size and SHA-256. `mask` checks the archive against it."""
    rec = {"dataset": "HydroLAKES v1.0, lake polygons (shapefile archive)",
           "reference": "refs/references.bib: messager2016 (doi:10.1038/ncomms13603); documentation: hydrolakes_doc",
           "licence": m["licence"], "url": m["url"], "final_url": m["final_url"], "file": m["file"],
           "retrieved_utc": m["retrieved_utc"], "http_status": m["http_status"], "bytes": m["bytes"],
           "sha256": m["sha256"], "members": m["members"]}
    PROV.parent.mkdir(parents=True, exist_ok=True)
    PROV.write_text(json.dumps(rec, indent=1), encoding="utf-8")


def cmd_fetch() -> int:
    import requests

    meta = ZIP.with_name(ZIP.name + ".meta.json")
    if ZIP.exists() and meta.exists() and json.loads(meta.read_text(encoding="utf-8")).get("ok"):
        print("already on disk:", ZIP)
        if not PROV.exists():
            write_record(json.loads(meta.read_text(encoding="utf-8")))
            print("wrote", PROV.relative_to(ROOT).as_posix())
        return 0
    import time

    ZIP.parent.mkdir(parents=True, exist_ok=True)
    part = ZIP.with_name(ZIP.name + ".part")
    t = utcnow()
    head = requests.head(URL, timeout=60, headers={"User-Agent": UA}, allow_redirects=True)
    want = int(head.headers.get("Content-Length") or 0)
    if head.status_code != 200 or not want:
        raise SystemExit(f"HTTP {head.status_code} from {URL} (length {want})")
    status, final, ctype = head.status_code, head.url, head.headers.get("Content-Type")
    # the server can drop a long transfer and does not honour byte ranges, so a broken transfer
    # starts again from the first byte
    for attempt in range(1, 9):
        n, step = 0, -1
        try:
            with requests.get(URL, stream=True, timeout=180, headers={"User-Agent": UA}) as r:
                if r.status_code != 200:
                    raise SystemExit(f"HTTP {r.status_code} from {URL}")
                with open(part, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                        n += len(chunk)
                        if n // (100 << 20) > step:
                            step = n // (100 << 20)
                            print(f"  attempt {attempt}: {n / 1e6:.0f} MB of {want / 1e6:.0f}", flush=True)
            if n == want:
                break
            print(f"  attempt {attempt}: stream ended at {n / 1e6:.0f} MB; starting again", flush=True)
        except (requests.RequestException, OSError) as e:
            print(f"  attempt {attempt}: connection broke at {n / 1e6:.0f} MB ({type(e).__name__}); starting again", flush=True)
        time.sleep(5 * attempt)
    else:
        raise SystemExit("eight attempts failed; the file was not downloaded")
    n, digest = sha256_file(part)
    err = None if n == want else f"{n} bytes received, {want} announced"
    if err is None:
        with zipfile.ZipFile(part) as z:
            bad = z.testzip()
            names = z.namelist()
        if bad is not None:
            err = f"damaged archive member {bad}"
    if err is None:
        part.replace(ZIP)
    m = {"file": ZIP.relative_to(ROOT).as_posix(), "url": URL, "final_url": final, "retrieved_utc": t,
         "http_status": status, "content_type": ctype, "bytes": n, "sha256": digest, "ok": err is None,
         "error": err, "members": names if err is None else None,
         "licence": "CC BY 4.0 (HydroLAKES v1.0; Messager et al. 2016)"}
    meta.write_text(json.dumps(m, indent=1), encoding="utf-8")
    if m["ok"]:
        write_record(m)
    print(("ok  " if m["ok"] else "FAIL"), status, n, ZIP.name, err or "")
    return 0 if m["ok"] else 1


def cmd_mask() -> int:
    import pyogrio
    import pyproj
    import rasterio
    import shapely
    from rasterio.windows import from_bounds

    if not ZIP.exists() or not PROV.exists():
        raise SystemExit(f"{ZIP} or its record {PROV} is missing: run `python scripts/rev3_outline.py fetch` "
                         "(a 782 MB download)")
    meta = json.loads(PROV.read_text(encoding="utf-8"))
    n, digest = sha256_file(ZIP)
    if (n, digest) != (meta["bytes"], meta["sha256"]):
        raise SystemExit(f"{ZIP} is not the file of its acquisition record ({n} bytes, sha256 {digest[:12]}; "
                         f"recorded {meta['bytes']} bytes, {meta['sha256'][:12]})")
    shp = next(n for n in meta["members"] if n.lower().endswith(".shp"))
    src = f"/vsizip/{ZIP.as_posix()}/{shp}"
    gdf = pyogrio.read_dataframe(src, where=f"Hylak_id = {HYLAK_ID}")
    if len(gdf) != 1:
        raise SystemExit(f"{len(gdf)} records with Hylak_id = {HYLAK_ID}")
    rec = gdf.iloc[0]
    if (rec["Lake_name"], rec["Poly_src"]) != ("Caspian Sea", "SWBD"):
        raise SystemExit(f"lake {HYLAK_ID} is {rec['Lake_name']!r} from {rec['Poly_src']!r}, not the Caspian Sea from SWBD")
    if POINTS.exists():                    # the points table of the same release, used earlier for ETOPO 2022
        with zipfile.ZipFile(POINTS) as z:
            pshp = next(n for n in z.namelist() if n.lower().endswith(".shp"))
        pts = pyogrio.read_dataframe(f"/vsizip/{POINTS.as_posix()}/{pshp}", where=f"Hylak_id = {HYLAK_ID}",
                                     read_geometry=False).iloc[0]
        if pts["Poly_src"] != rec["Poly_src"]:
            raise SystemExit(f"lake {HYLAK_ID}: the points table gives the source {pts['Poly_src']!r}, "
                             f"the polygons {rec['Poly_src']!r}")
    crs_src = pyproj.CRS(gdf.crs)
    if not crs_src.equals(pyproj.CRS("EPSG:4326"), ignore_axis_order=True):
        raise SystemExit(f"the polygons are in {crs_src.to_string()}, not WGS84 longitude and latitude")
    poly = rec.geometry
    ext = np.load(EXTENTS)
    if not pyproj.CRS(str(ext["crs"])).equals(pyproj.CRS(CRS_EQUAL_AREA), ignore_axis_order=True):
        raise SystemExit("extents_v6_250m.npz: CRS differs from config.CRS_EQUAL_AREA")
    tr = [float(v) for v in ext["transform"][:6]]
    shape = np.asarray(ext["high"]).shape
    rows, cols = shape
    X, Y = np.meshgrid(tr[2] + tr[0] * (np.arange(cols) + 0.5), tr[5] + tr[4] * (np.arange(rows) + 0.5))
    lon, lat = pyproj.Transformer.from_crs(pyproj.CRS(str(ext["crs"])), "EPSG:4326", always_xy=True).transform(X, Y)
    box = shapely.box(float(lon.min()) - 0.2, float(lat.min()) - 0.2, float(lon.max()) + 0.2, float(lat.max()) + 0.2)
    part = shapely.intersection(poly, box)
    lake = rev3.lake_mask_from_polygon(part, tr, shape, str(ext["crs"]))
    # the registered form, for the record: GLWD v2 main class 2 at the cell centre
    with rasterio.open(GLWD) as ds:
        win = from_bounds(float(lon.min()) - 0.1, float(lat.min()) - 0.1, float(lon.max()) + 0.1,
                          float(lat.max()) + 0.1, ds.transform).round_offsets(op="floor").round_lengths(op="ceil")
        g = ds.read(1, window=win)
        gt = ds.window_transform(win)
    raster = g[np.floor((lat - gt.f) / gt.e).astype(int), np.floor((lon - gt.c) / gt.a).astype(int)] == 2
    cell = abs(tr[0] * tr[4]) / 1e6
    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_NPZ, lake2000=lake, transform=np.asarray(ext["transform"]), crs=str(ext["crs"]))
    info = {"decision": "D-163", "amendment": "PREREG Amendment 20 item 1e (the display rule), polygon form",
            "source": {"file": meta["file"], "url": meta["url"], "retrieved_utc": meta["retrieved_utc"],
                       "bytes": meta["bytes"], "sha256": meta["sha256"], "licence": meta["licence"], "layer": shp},
            "lake": {"Hylak_id": int(rec["Hylak_id"]), "Lake_name": rec["Lake_name"], "Poly_src": rec["Poly_src"],
                     "Lake_area_km2": float(rec["Lake_area"]), "Elevation_m": float(rec["Elevation"])},
            "shoreline_date": "February 2000 (SRTM Water Body Data; HydroLAKES Technical Documentation v1.0)",
            "polygon": {"n_vertices_whole_lake": int(shapely.get_num_coordinates(poly)),
                        "n_vertices_in_window": int(shapely.get_num_coordinates(part)),
                        "n_interior_rings_in_window": int(sum(shapely.get_num_interior_rings(p)
                                                              for p in shapely.get_parts(part)))},
            "grid": {"file": "results/study_area/extents_v6_250m.npz", "cell_m": abs(tr[0]), "shape": list(shape),
                     "rule": "cell centre inside the polygon (rasterize, all_touched = False), equal-area projection"},
            "lake_cells": int(lake.sum()), "lake_km2_in_grid": float(lake.sum() * cell),
            "glwd_raster": {"file": GLWD.name, "cell_arcsec": 15.0, "lake_km2_in_grid": float(raster.sum() * cell),
                            "cells_lake_in_both": int((lake & raster).sum()),
                            "cells_lake_in_polygon_only": int((lake & ~raster).sum()),
                            "cells_lake_in_raster_only": int((~lake & raster).sum())}}
    OUT_JSON.write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(json.dumps(info, indent=1))
    return 0


def main(argv) -> int:
    cmds = {"fetch": cmd_fetch, "mask": cmd_mask}
    if not argv or argv[0] not in cmds:
        raise SystemExit(__doc__)
    return cmds[argv[0]]()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
