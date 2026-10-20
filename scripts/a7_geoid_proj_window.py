"""Read a window of the PROJ CDN EGM2008 2.5' grid (us_nga_egm08_25.tif) via HTTP range requests.

Usage: python scripts/a7_geoid_proj_window.py OUT_TIF
Saves the window unchanged (float32 values, same georeferencing) and OUT_TIF + '.request.json'.
The full 80.6 MB file is not downloaded; its sha256 therefore cannot be computed here.
"""
import caspian  # noqa: F401
import datetime as dt
import hashlib
import json
import pathlib
import sys

import numpy as np
import rasterio
import requests
from rasterio.windows import Window

URL = "https://cdn.proj.org/us_nga_egm08_25.tif"
W, E, S, N = 46.20, 54.80, 43.75, 47.55  # region 46.3-54.7 E, 43.85-47.45 N plus 0.1 deg margin


def main():
    out = pathlib.Path(sys.argv[1])
    if out.exists():
        sys.exit(f"refusing to overwrite {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    rec = {"url": URL, "t_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    h = requests.head(URL, timeout=60)
    rec["http_head"] = {k: h.headers.get(k) for k in ("Content-Length", "Last-Modified", "ETag")}
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif"):
        with rasterio.open("/vsicurl/" + URL) as src:
            rec["source_profile"] = {"width": src.width, "height": src.height, "dtype": src.dtypes[0],
                                     "crs": src.crs.to_string() if src.crs else None,
                                     "transform": list(src.transform)[:6], "nodata": src.nodata}
            rec["source_tags"] = src.tags()
            rec["source_band_tags"] = src.tags(1)
            rec["source_image_structure"] = src.tags(ns="IMAGE_STRUCTURE")
            rec["source_band_description"] = src.descriptions[0]
            rec["source_units"] = src.units[0] if src.units else None
            inv = ~src.transform
            c0, r0 = inv * (W, N)
            c1, r1 = inv * (E, S)
            c0, r0 = int(np.floor(c0)), int(np.floor(r0))
            c1, r1 = int(np.ceil(c1)), int(np.ceil(r1))
            win = Window(c0, r0, c1 - c0, r1 - r0)
            a = src.read(1, window=win)
            tr = src.window_transform(win)
            prof = src.profile.copy()
            prof.update(width=a.shape[1], height=a.shape[0], transform=tr, driver="GTiff",
                        compress="deflate", tiled=False, blockxsize=None, blockysize=None)
            for k in ("blockxsize", "blockysize"):
                prof.pop(k, None)
            with rasterio.open(out, "w", **prof) as dst:
                dst.write(a, 1)
                dst.update_tags(**src.tags())
                dst.update_tags(1, **src.tags(1))
    rec["window"] = {"col_off": c0, "row_off": r0, "width": a.shape[1], "height": a.shape[0],
                     "transform": list(tr)[:6]}
    rec["array_sha256_float32_C"] = hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
    rec["saved_file_sha256"] = hashlib.sha256(out.read_bytes()).hexdigest()
    rec["value_range_m"] = [float(a.min()), float(a.max())]
    pathlib.Path(str(out) + ".request.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
