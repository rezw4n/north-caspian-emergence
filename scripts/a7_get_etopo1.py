"""Subset ETOPO1 (Amante & Eakins 2009, doi:10.7289/V5C8276M) over the study
domain from the Earth Engine catalogue copy (NOAA/NGDC/ETOPO1, band ice_surface
and bedrock; 1 arc-minute), using a direct GeoTIFF download (small).

Outputs (git-ignored): data/raw/bathymetry/etopo1_ncaspian.tif
Provenance (committed): results/provenance/etopo1.json
"""

import hashlib
import io
import json
import sys
import zipfile

import caspian  # noqa: F401
import ee
import requests

from caspian.config import EE_PROJECT, RAW, RESULTS
from caspian.geography import DOMAIN

ASSET = "NOAA/NGDC/ETOPO1"


def main() -> int:
    ee.Initialize(project=EE_PROJECT)
    w, s, e, n = DOMAIN.bounds
    region = ee.Geometry.Rectangle([w - 0.1, s - 0.1, e + 0.1, n + 0.1], None, False)
    img = ee.Image(ASSET).select(["bedrock", "ice_surface"])
    info = ee.Image(ASSET).getInfo()
    proj = img.select("bedrock").projection().getInfo()
    url = img.getDownloadURL({"region": region, "crs": proj["crs"], "crsTransform": proj["transform"],
                              "format": "GEO_TIFF", "filePerBand": False})
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    data = r.content
    if data[:2] == b"PK":
        data = zipfile.ZipFile(io.BytesIO(data)).read(zipfile.ZipFile(io.BytesIO(data)).namelist()[0])
    out = RAW / "bathymetry" / "etopo1_ncaspian.tif"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    prov = {"product": "ETOPO1 (bedrock, ice_surface), 1 arc-minute", "doi": "10.7289/V5C8276M",
            "source": f"Earth Engine catalogue {ASSET}", "asset_properties": info.get("properties", {}),
            "projection": proj, "subset_sha256": hashlib.sha256(data).hexdigest()}
    (RESULTS / "provenance").mkdir(parents=True, exist_ok=True)
    (RESULTS / "provenance" / "etopo1.json").write_text(json.dumps(prov, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in prov.items() if k != "asset_properties"}, indent=1))
    print("asset properties keys:", list(info.get("properties", {}))[:20])
    return 0


if __name__ == "__main__":
    sys.exit(main())
