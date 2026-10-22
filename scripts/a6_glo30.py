"""Copernicus DEM GLO-30 as an independent height source for the waterline DEM (PREREG
section 2: "GLO-30 over pixels dry throughout its 2011-2015 acquisition window").
Builds a point table in the format of scripts/a6_validate_dem.py, so that its
``blocksize`` and ``evaluate`` stages run on it with ``--sources GLO30``.

Stages
  download  The 1-degree tiles that hold emerged-band cells (the emerged band of
            results/study_area/extents_v6_250m.npz), less N44E048 and N44E049
            (Caspian geocells not acquired by TanDEM-X), from the public AWS Open Data
            bucket over HTTPS (no login, no click-through): the DEM, the editing,
            filling and water-body masks (EDM, FLM, WBM), the XML metadata and the
            Source Data Layer (SRC, kml), into data/raw/glo30/<tile>/. Every file is
            checked against the bucket listing (size, and MD5 against the ETag) and
            recorded with its sha256 in results/provenance/copernicus_glo30.json, one
            record per tile, with the acquisition dates read from its SRC layer and the
            licence statement (INFO/eula_F.pdf, read once).
  points    Per tile, the GLO-30 pixels on a thinned grid (every THIN_ROWS-th row and
            THIN_COLS-th column: about 93 m x 87 m), then ordered steps, each counted:
              dry_centre       the 30 m DEM pixel under the GLO-30 pixel emerged
                               (emergence raster of a6_emergence.py) more than 30 days
                               before the window start, 2010-12-01;
              height           finite GLO-30 height;
              wbm_no_water     WBM = 0;
              edm_not_edited   EDM = 1 (not edited: no infill, interpolation,
                               smoothing, flattening, water or shoreline editing);
              flm_not_filled   FLM = 2 (not edited, not filled from another DEM);
              geoid            finite N(EGM2008) - N(EIGEN-6C4) at the pixel;
              dry_throughout_window
                               the same test for every 30 m DEM pixel within one pixel
                               (the GEDI neighbourhood of Amendment 2), with
                               validation.dry_at_acquisition at t = 2010-12-01.
            Heights: h (EGM2008, NGA undulations, as distributed) is brought to the
            DAHITI (EIGEN-6C4) reference cell by cell as h_orth = h + dN, with dN =
            N_EGM2008(NGA) - N_EIGEN-6C4 bilinear from
            results/study_area/geoid_correction_primary.npz (the products' primary
            correction). The constant parts (NGA's -0.41 m, zero-degree and
            tide-system terms) are absorbed by the training-block constant c of
            scripts/a6_validate_dem.py evaluate (PREREG section 5).
            Then the DEM height, interpolation distance, slope, contour density and
            the five product heights at each point, exactly as in
            a6_validate_dem.prepare. ``time`` is the window start (the date the
            dryness test is made at).

Outputs
  data/raw/glo30/<tile>/...                         (git-ignored)
  data/interim/validation/<tag>/GLO30_points.csv.gz (git-ignored; the point table)
  results/provenance/copernicus_glo30.json
  results/validation/<tag>_glo30/prepare.json, GLO30_filter_cascade.csv

Usage
  python scripts/a6_glo30.py download
  python scripts/a6_glo30.py points <tag> --emergence DIR
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import caspian  # noqa: F401
import numpy as np
import pandas as pd

from caspian import validation
from caspian.config import CRS_EQUAL_AREA, RAW, RESULTS

BUCKET = "https://copernicus-dem-30m.s3.amazonaws.com/"
RAW_DIR = RAW / "glo30"
PROV = RESULTS / "provenance" / "copernicus_glo30.json"
SOURCE = "GLO30"
THIN_ROWS, THIN_COLS = 3, 4        # 1" grid: 3 x 30.9 m = 93 m north-south; 4 x 20.9-22.2 m = 84-89 m east-west
ROOT = Path(__file__).resolve().parents[1]


def _drv():
    """scripts/a6_validate_dem.py as a module (its helpers: contour points, product
    sampler, cache and output paths)."""
    spec = importlib.util.spec_from_file_location("a6_validate_dem", ROOT / "scripts" / "a6_validate_dem.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def band_tiles() -> list[str]:
    """Names (N44E050 form) of the 1-degree cells holding at least one emerged-band cell
    (emerged band on the 250 m zone-v6 extents), less validation.GLO30_EXCLUDED_TILES."""
    import pyproj

    ex = np.load(RESULTS / "study_area" / "extents_v6_250m.npz")
    band = (ex["high"] == 1) & (ex["low"] == 0)
    t = ex["transform"]
    r, c = np.nonzero(band)
    inv = pyproj.Transformer.from_crs(CRS_EQUAL_AREA, "EPSG:4326", always_xy=True)
    lon, lat = inv.transform(t[2] + (c + 0.5) * t[0], t[5] + (r + 0.5) * t[4])
    cells = set(zip(np.floor(lat).astype(int).tolist(), np.floor(lon).astype(int).tolist()))
    names = sorted(validation.glo30_cell_name(la, lo) for la, lo in cells)
    return [n for n in names if n not in validation.GLO30_EXCLUDED_TILES]


def _listing(prefix: str, session) -> dict[str, dict]:
    """Keys under a bucket prefix with size and ETag (S3 ListObjectsV2 over HTTPS)."""
    r = session.get(BUCKET, params={"list-type": "2", "prefix": prefix}, timeout=60)
    r.raise_for_status()
    out = {}
    for m in re.finditer(r"<Key>([^<]+)</Key><LastModified>([^<]+)</LastModified><ETag>([^<]+)</ETag>"
                         r"<Size>(\d+)</Size>", r.text):
        out[m.group(1)] = {"last_modified": m.group(2), "etag": m.group(3).replace("&quot;", "").strip('"'),
                           "size": int(m.group(4))}
    return out


def _fetch(url: str, dest: Path, size: int, etag: str, session, tries: int = 4) -> dict:
    """Download to dest (via a .part file), checking the size and, for single-part
    uploads, the MD5 against the ETag. Returns sha256 and checks."""
    for k in range(tries):
        try:
            md5, sha = hashlib.md5(), hashlib.sha256()
            part = dest.with_suffix(dest.suffix + ".part")
            with session.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with part.open("wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                        md5.update(chunk)
                        sha.update(chunk)
            n = part.stat().st_size
            if n != size:
                raise OSError(f"size {n} != listed {size}")
            md5_ok = None if "-" in etag else (md5.hexdigest() == etag)
            if md5_ok is False:
                raise OSError("MD5 does not match the ETag")
            part.replace(dest)
            return {"sha256": sha.hexdigest(), "bytes": n, "md5_matches_etag": md5_ok}
        except Exception as ex:  # noqa: BLE001 - retried, then raised
            if k == tries - 1:
                raise
            print("retry", url, str(ex)[:120], flush=True)
            time.sleep(5 * (k + 1))
    raise RuntimeError("unreachable")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def src_dates(kml_text: str) -> dict:
    """Acquisition dates named in a Source Data Layer kml: every ISO date (YYYY-MM-DD)
    in the file, as first, last and count of distinct dates."""
    dates = sorted(set(re.findall(r"(?<!\d)(20[01]\d-[01]\d-[0-3]\d)", kml_text)))
    return {"first": dates[0] if dates else None, "last": dates[-1] if dates else None,
            "n_distinct_dates": len(dates)}


def download(a) -> int:
    import requests

    tiles = band_tiles() if not a.tiles else a.tiles.split(",")
    prov = json.loads(PROV.read_text(encoding="utf-8")) if PROV.exists() else {}
    prov.setdefault("tiles", {})
    s = requests.Session()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name in tiles:
        tid = validation.glo30_tile_id(name)
        if name in prov["tiles"] and all((RAW_DIR / name / Path(f["key"]).name).exists()
                                         for f in prov["tiles"][name]["files"].values()):
            print(name, "already recorded", flush=True)
            continue
        lst = _listing(tid + "/", s)
        if not lst:
            prov["tiles"][name] = {"tile_id": tid, "status": "not in the bucket", "checked": _now()}
            print(name, "not in the bucket", flush=True)
            continue
        base = tid.replace("_COG", "").replace("_DEM", "")
        want = {"dem": f"{tid}/{tid}.tif",
                **{m.lower(): f"{tid}/AUXFILES/{tid.replace('_DEM', '')}_{m}.tif" for m in ("EDM", "FLM", "WBM")},
                "xml": f"{tid}/{base}.xml", "src_kml": f"{tid}/PREVIEW/{base}_SRC.kml"}
        rec = {"tile_id": tid, "files": {}, "listing_keys": sorted(lst), "downloaded": _now()}
        for role, key in want.items():
            if key not in lst:
                rec["files"][role] = {"key": key, "status": "not in the listing"}
                continue
            dest = RAW_DIR / name / Path(key).name
            dest.parent.mkdir(parents=True, exist_ok=True)
            chk = _fetch(BUCKET + key, dest, lst[key]["size"], lst[key]["etag"], s)
            rec["files"][role] = {"key": key, "url": BUCKET + key, **lst[key], **chk}
        eula = f"{tid}/INFO/eula_F.pdf"
        rec["eula_etag"] = lst.get(eula, {}).get("etag")
        kml = RAW_DIR / name / Path(want["src_kml"]).name
        if kml.exists():
            rec["src_acquisition_dates"] = src_dates(kml.read_text(encoding="utf-8", errors="replace"))
        prov["tiles"][name] = rec
        PROV.write_text(json.dumps(prov, indent=1), encoding="utf-8")
        print(name, {k: v.get("bytes") for k, v in rec["files"].items()}, rec.get("src_acquisition_dates"), flush=True)
    # licence: INFO/eula_F.pdf of the first tile, read once (all tiles' ETags compared)
    first = next((v for v in prov["tiles"].values() if v.get("files")), None)
    if first is not None:
        key = f"{first['tile_id']}/INFO/eula_F.pdf"
        dest = RAW_DIR / "eula_F.pdf"
        if not dest.exists():
            lst = _listing(key, s)
            _fetch(BUCKET + key, dest, lst[key]["size"], lst[key]["etag"], s)
        etags = {v.get("eula_etag") for v in prov["tiles"].values() if v.get("files")}
        prov["licence"] = {
            "file": str(dest.relative_to(ROOT)).replace("\\", "/"), "url": BUCKET + key, "sha256": _sha256(dest),
            "title": "Licence for Copernicus DEM instance COP-DEM-GLO-30-F Global 30m Full, Free & Open "
                     "(Licence for the use of the Copernicus WorldDEM-30)",
            "same_eula_in_every_tile": len(etags) == 1,
            "access": "public HTTPS from the AWS Open Data bucket; no login and no click-through",
            "statements": {
                "Article 1": "The user must accept the terms and conditions of this licence to benefit from the "
                             "rights of use granted in this licence.",
                "Article 4": "non-exclusive rights of reproduction; distribution; communication to the General "
                             "Public; adaptation, modification and combination with other data and information",
                "Article 5": "The use rights granted under this licence are free of charge to the User.",
                "Article 6(b) notice for modified data": "produced using Copernicus WorldDEM-30 (c) DLR e.V. "
                    "2010-2014 and (c) Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the "
                    "European Union and ESA; all rights reserved",
                "Article 6(c)": "The organisations in charge of the Copernicus programme by law or by delegation do "
                                "not incur any liability for any use of the Copernicus WorldDEM-30",
            },
            "aws_readme": "GLO-30 Public and GLO-90 are available on a free basis for the general public under the "
                          "terms and conditions of the Licence (https://copernicus-dem-30m.s3.amazonaws.com/readme.html)",
        }
    prov.update(product="Copernicus DEM GLO-30 Public (COP-DEM_GLO-30-DGED, AWS Open Data COG distribution)",
                bucket=BUCKET, rule="1-degree cells holding emerged-band cells (D-071 band, extents_v6_250m.npz), "
                "less N44E048 and N44E049 (D-073)", excluded_tiles=list(validation.GLO30_EXCLUDED_TILES),
                vertical_datum="EGM2008 (EPSG 3855), NGA undulations w.r.t. WGS 84 (Product Handbook v5.0 §1.2.1)",
                acquisition_window=["2010-12-01", "2015-01-31"],
                auxiliary_layers="EDM, FLM, WBM (AUXFILES) are in the AWS distribution; HEM is not downloaded")
    PROV.write_text(json.dumps(prov, indent=1), encoding="utf-8")
    print("tiles recorded:", len(prov["tiles"]))
    return 0


def _read_tile(name: str, rec: dict) -> dict[str, np.ndarray]:
    """The DEM and the three masks of one tile, thinned (validation.thin_index), with
    the lon/lat of the kept pixel centres (rasterio's transform already places pixel
    (0, 0) at the tile's whole-degree corner; RasterPixelIsPoint)."""
    import rasterio

    rows = cols = None
    out = {}
    for role in ("dem", "wbm", "edm", "flm"):
        f = RAW_DIR / name / Path(rec["files"][role]["key"]).name
        with rasterio.open(f) as src:
            if rows is None:
                rows, cols = validation.thin_index(src.height, THIN_ROWS), validation.thin_index(src.width, THIN_COLS)
                shape, tr = (src.height, src.width), src.transform
                cc, rr = np.meshgrid(cols, rows)
                c2, r2 = cc.ravel() + 0.5, rr.ravel() + 0.5
                lon, lat = tr.c + tr.a * c2 + tr.b * r2, tr.f + tr.d * c2 + tr.e * r2
                out["lon"], out["lat"] = np.asarray(lon, float), np.asarray(lat, float)
                out["n_pixels"] = shape[0] * shape[1]
            elif (src.height, src.width) != shape or src.transform != tr:
                raise SystemExit(f"{name}: {role} is not on the DEM grid")
            a = src.read(1)
            out[role] = a[np.ix_(rows, cols)].ravel()
            del a
    out["dem"] = out["dem"].astype(float)
    return out


def points(a) -> int:
    import pyproj

    drv = _drv()
    if a.emergence is None or not any(Path(a.emergence).glob("emergence_*.tif")):
        raise SystemExit("PREREG section 2 needs the emergence map of a6_emergence.py: pass --emergence DIR")
    prov = json.loads(PROV.read_text(encoding="utf-8"))
    tiles = [n for n, r in sorted(prov["tiles"].items()) if r.get("files") and n not in validation.GLO30_EXCLUDED_TILES]
    names, sample = drv._product_sampler(a.products, a.no_products)
    dem = validation.TileSet(drv.PROCESSED / "dem" / a.tag, "dem")
    em = validation.TileSet(Path(a.emergence), "emergence")
    dN = validation.GridLookup(RESULTS / "study_area" / "geoid_correction_primary.npz", "dN")
    fwd = pyproj.Transformer.from_crs("EPSG:4326", CRS_EQUAL_AREA, always_xy=True)
    t0 = validation.GLO30_WINDOW[0]
    frames, cascades, per_tile = [], [], {}
    for name in tiles:
        r = _read_tile(name, prov["tiles"][name])
        x, y = fwd.transform(r["lon"], r["lat"])
        x, y = np.asarray(x), np.asarray(y)
        n = x.size
        t = np.full(n, t0)
        e = em.sample(x, y, 1)
        g = dN(r["lon"], r["lat"])
        steps = {"dry_centre": validation.dry_at_acquisition(t, e),
                 **validation.glo30_steps(r["dem"], r["wbm"], r["edm"], r["flm"], g)}
        keep, casc = validation.cascade(steps, n)
        cascades.append(casc.set_index("step"))
        per_tile[name] = {"n_pixels": int(r["n_pixels"]), "n_thinned": int(n), "n_kept": int(keep.sum())}
        if keep.any():
            frames.append(pd.DataFrame({"lat": r["lat"][keep], "lon": r["lon"][keep], "h": r["dem"][keep],
                                        "wbm": r["wbm"][keep], "edm": r["edm"][keep], "flm": r["flm"][keep],
                                        "dN_geoid": g[keep], "x": x[keep], "y": y[keep], "tile": name}))
        print(name, per_tile[name], flush=True)
        del r, x, y, e, g, steps, keep
    casc = sum(cascades[1:], cascades[0]).reset_index()
    d = pd.concat(frames, ignore_index=True)
    del frames
    # PREREG section 2 / Amendment 2 neighbourhood: every 30 m DEM pixel within one pixel
    rad = validation.DRY_RADIUS_PX[SOURCE]
    dry = np.zeros(len(d), bool)
    tt = np.full(len(d), t0)
    for k in range(0, len(d), drv.CHUNK):
        s = slice(k, k + drv.CHUNK)
        dry[s] = validation.dry_at_acquisition(tt[s], em.sample_window(d["x"].to_numpy()[s], d["y"].to_numpy()[s], 1, rad)
                                               .reshape(len(tt[s]), -1))
    casc = pd.concat([casc, pd.DataFrame([{"step": "dry_throughout_window", "n_remaining": int(dry.sum()),
                                           "n_fail_alone": int((~dry).sum())}])], ignore_index=True)
    d = d.loc[dry].reset_index(drop=True)
    d["h_orth"] = d["h"] + d["dN_geoid"]                      # EGM2008 (NGA) -> EIGEN-6C4 reference
    d["time"] = pd.Series(np.full(len(d), t0)).dt.strftime("%Y-%m-%d")
    d["source"] = SOURCE
    cx, cy = drv._contour_points(a.tag, a.epochs.split(","))
    d = drv.attach_model_columns(d, dem, cx, cy, names, sample)
    path = drv._cache(a.tag, SOURCE)
    path.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(path, index=False, float_format="%.9g")
    res = RESULTS / "validation" / f"{a.tag}_{SOURCE.lower()}"
    res.mkdir(parents=True, exist_ok=True)
    casc.to_csv(res / f"{SOURCE}_filter_cascade.csv", index=False)
    meta = {"created": _now(), "source": SOURCE, "dem_tag": a.tag, "emergence_dir": str(a.emergence),
            "epochs": a.epochs.split(","), "products": names,
            "product_interpolation": "bilinear, as distributed (caspian.products.sample)" if names else None,
            "tiles": tiles, "excluded_tiles": list(validation.GLO30_EXCLUDED_TILES),
            "thinning": {"rows": THIN_ROWS, "cols": THIN_COLS, "offset": "centred (validation.thin_index)",
                         "spacing_m": "about 93 m north-south, 84-89 m east-west"},
            "window": [str(v)[:10] for v in validation.GLO30_WINDOW],
            "dryness": "emergence + 30 days before the window start, for every 30 m DEM pixel within "
                       f"{rad} pixel (validation.dry_at_acquisition at t = window start)",
            "quality_steps": {"wbm": validation.GLO30_WBM_NO_WATER, "edm": validation.GLO30_EDM_NOT_EDITED,
                              "flm": validation.GLO30_FLM_NOT_FILLED},
            "datum": "h_orth = h_GLO30 (EGM2008, NGA) + dN, dN = N_EGM2008(NGA) - N_EIGEN-6C4 bilinear from "
                     "results/study_area/geoid_correction_primary.npz (D-074); constants absorbed by the training "
                     "constant c",
            "emergence_lag_days": int(validation.design.EMERGENCE_LAG_DAYS), "per_tile": per_tile,
            "n_points": len(d), "n_with_dem": int(np.isfinite(d["dem_h"]).sum())}
    (res / "prepare.json").write_text(drv._json(meta), encoding="utf-8")
    print(SOURCE, "->", len(d), "dry points,", meta["n_with_dem"], "with a DEM height", flush=True)
    return 0


def main(argv) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["download", "points"])
    ap.add_argument("tag", nargs="?", default="main")
    ap.add_argument("--tiles", default=None, help="download: comma-separated cell names (default: the band rule)")
    ap.add_argument("--emergence", default=None)
    ap.add_argument("--products", default=None)
    ap.add_argument("--no-products", action="store_true")
    ap.add_argument("--epochs", default="E2,E3")
    a = ap.parse_args(argv)
    return {"download": download, "points": points}[a.stage](a)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
