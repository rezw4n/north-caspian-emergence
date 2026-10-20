"""Subset GEBCO_2026 (elevation and Type Identifier grids) over the study domain by
HTTP range requests on CEDA's netCDF4 files (no full download).

Outputs (git-ignored): data/raw/bathymetry/gebco_2026_ncaspian.nc
Provenance (committed): results/provenance/gebco_2026.json
"""

from __future__ import annotations

import hashlib
import json
import sys

import caspian  # noqa: F401
from caspian import netcdf3
import fsspec
import h5py
import numpy as np
import requests
import xarray as xr

from caspian.config import RAW, RESULTS
from caspian.geography import DOMAIN

BASE = "https://dap.ceda.ac.uk/bodc/gebco/global/gebco_2026"
FILES = {
    "elevation": f"{BASE}/ice_surface_elevation/netcdf/GEBCO_2026.nc",
    "tid": f"{BASE}/type_identifier_grid/netcdf/gebco_2026_tid.nc",
}
LISTING = "https://data.ceda.ac.uk/bodc/gebco/global/gebco_2026/{}/netcdf?json"
DOI = "10.5285/4f68d5c7-45eb-f999-e063-7086abc036fa"
MARGIN = 0.1  # degrees around the domain bounds


def ceda_listing(sub: str) -> list[dict]:
    r = requests.get(LISTING.format(sub), timeout=60)
    r.raise_for_status()
    return r.json()["items"]


def read_window(url: str, var: str, bounds):
    w, s, e, n = bounds
    with fsspec.open(url, "rb", block_size=4 * 2**20) as f, h5py.File(f, "r") as h:
        lat = h["lat"][:]
        lon = h["lon"][:]
        i0, i1 = np.searchsorted(lat, s - MARGIN), np.searchsorted(lat, n + MARGIN)
        j0, j1 = np.searchsorted(lon, w - MARGIN), np.searchsorted(lon, e + MARGIN)
        data = h[var][i0:i1, j0:j1]
        attrs = {k: (v.decode() if isinstance(v, bytes) else str(v)) for k, v in h[var].attrs.items()
                 if k not in ("DIMENSION_LIST",)}
        chunks = h[var].chunks
    return lat[i0:i1], lon[j0:j1], data, attrs, chunks


def main() -> int:
    bounds = DOMAIN.bounds
    out_dir = RAW / "bathymetry"
    out_dir.mkdir(parents=True, exist_ok=True)
    items = {it["name"]: it for sub in ("ice_surface_elevation", "type_identifier_grid")
             for it in ceda_listing(sub) if it["name"].endswith(".nc")}
    for key, url in FILES.items():
        name = url.rsplit("/", 1)[1]
        if name not in items:
            print(f"FAIL {name} not in CEDA listing: {sorted(items)}")
            return 1
    lat, lon, elev, attrs_e, chunks_e = read_window(FILES["elevation"], "elevation", bounds)
    hdr = netcdf3.header(FILES["tid"])
    tlat = netcdf3.read_1d(FILES["tid"], hdr, "lat")
    tlon = netcdf3.read_1d(FILES["tid"], hdr, "lon")
    i0, i1 = np.searchsorted(tlat, lat[0]), np.searchsorted(tlat, lat[-1]) + 1
    j0, j1 = np.searchsorted(tlon, lon[0]), np.searchsorted(tlon, lon[-1]) + 1
    if not (np.allclose(tlat[i0:i1], lat) and np.allclose(tlon[j0:j1], lon)):
        print("FAIL TID grid is not aligned with the elevation grid")
        return 1
    tid = netcdf3.read_rows(FILES["tid"], hdr, "tid", i0, i1, j0, j1)
    attrs_t = {k: str(v) for k, v in hdr["variables"]["tid"]["atts"].items()}
    ds = xr.Dataset(
        {"elevation": (("lat", "lon"), elev.astype("int16"), attrs_e),
         "tid": (("lat", "lon"), tid.astype("int8"), attrs_t)},
        coords={"lat": lat, "lon": lon},
        attrs={"source": "GEBCO_2026 Grid via CEDA (HTTP range subset)", "doi": DOI,
               "urls": json.dumps(FILES)})
    out = out_dir / "gebco_2026_ncaspian.nc"
    ds.to_netcdf(out)
    prov = {
        "product": "GEBCO_2026 Grid", "doi": DOI, "urls": FILES,
        "ceda_md5_full_files": {items[u.rsplit('/', 1)[1]]["name"]: items[u.rsplit('/', 1)[1]].get("md5")
                                for u in FILES.values()},
        "ceda_last_modified": {items[u.rsplit('/', 1)[1]]["name"]: items[u.rsplit('/', 1)[1]].get("last_modified")
                               for u in FILES.values()},
        "subset_bounds_wsen": [round(float(lon[0]), 5), round(float(lat[0]), 5),
                               round(float(lon[-1]), 5), round(float(lat[-1]), 5)],
        "shape": list(elev.shape), "hdf5_chunks": list(chunks_e) if chunks_e else None,
        "elevation_attrs": attrs_e,
        "elevation_range_m": [int(elev.min()), int(elev.max())],
        "subset_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "tid_values_present": sorted(int(v) for v in np.unique(tid)),
    }
    (RESULTS / "provenance").mkdir(parents=True, exist_ok=True)
    (RESULTS / "provenance" / "gebco_2026.json").write_text(json.dumps(prov, indent=1), encoding="utf-8")
    print(json.dumps(prov, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
