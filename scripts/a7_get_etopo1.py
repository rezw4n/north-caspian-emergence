"""Subset ETOPO1 (Amante & Eakins 2009, doi:10.7289/V5C8276M) over the study
domain from the Earth Engine catalogue copy (NOAA/NGDC/ETOPO1, bands bedrock and
ice_surface; 1 arc-minute, grid-registered), on the asset's native grid.

The first version used getDownloadURL, which ignored the crsTransform key and
returned a 1-degree grid (9 x 5 pixels; review of 2026-10-20). This version reads
the native grid window with ee.data.computePixels (caspian.ee_io) and checks the
resolution before writing.

Outputs (git-ignored): data/raw/bathymetry/etopo1_ncaspian.tif
Provenance (committed): results/provenance/etopo1.json
"""

import hashlib
import json
import math
import sys

import caspian  # noqa: F401
import ee
import numpy as np
import rasterio
from rasterio.transform import Affine

from caspian import ee_io
from caspian.config import EE_PROJECT, RAW, RESULTS
from caspian.geography import DOMAIN

ASSET = "NOAA/NGDC/ETOPO1"
MARGIN_DEG = 0.1


def main() -> int:
    ee.Initialize(project=EE_PROJECT)
    band_grid = ee.data.getAsset(ASSET)["bands"][0]["grid"]["affineTransform"]
    sx, x0g = band_grid["scaleX"], band_grid["translateX"]
    sy, y0g = band_grid["scaleY"], band_grid["translateY"]
    w, s, e, n = DOMAIN.bounds
    c0 = math.floor((w - MARGIN_DEG - x0g) / sx)
    c1 = math.ceil((e + MARGIN_DEG - x0g) / sx)
    r0 = math.floor((n + MARGIN_DEG - y0g) / sy)
    r1 = math.ceil((s - MARGIN_DEG - y0g) / sy)
    transform = [sx, 0, x0g + c0 * sx, 0, sy, y0g + r0 * sy]
    dims = (c1 - c0, r1 - r0)
    arr = ee_io.fetch(ASSET, ["bedrock", "ice_surface"], transform, dims, None, block=512)
    assert abs(sx * 60 - 1) < 1e-9 and dims[0] > 400, f"unexpected grid {sx}, {dims}"
    out = RAW / "bathymetry" / "etopo1_ncaspian.tif"
    out.parent.mkdir(parents=True, exist_ok=True)
    aff = Affine(transform[0], 0, transform[2], 0, transform[4], transform[5])
    with rasterio.open(out, "w", driver="GTiff", width=dims[0], height=dims[1], count=2, dtype="int16",
                       crs="EPSG:4326", transform=aff, compress="deflate") as dst:
        dst.write(arr["bedrock"].astype(np.int16), 1)
        dst.write(arr["ice_surface"].astype(np.int16), 2)
        dst.set_band_description(1, "bedrock")
        dst.set_band_description(2, "ice_surface")
    data = out.read_bytes()
    prov = {"product": "ETOPO1 (bedrock, ice_surface), 1 arc-minute, grid-registered", "doi": "10.7289/V5C8276M",
            "source": f"Earth Engine catalogue {ASSET}", "native_affine": band_grid,
            "subset_transform": transform, "subset_dims": dims,
            "bedrock_min_m": int(arr["bedrock"].min()), "bedrock_max_m": int(arr["bedrock"].max()),
            "subset_sha256": hashlib.sha256(data).hexdigest()}
    (RESULTS / "provenance").mkdir(parents=True, exist_ok=True)
    (RESULTS / "provenance" / "etopo1.json").write_text(json.dumps(prov, indent=1, default=str), encoding="utf-8")
    print(json.dumps(prov, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
