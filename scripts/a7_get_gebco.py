"""Subset a GEBCO grid (default GEBCO_2026; argument "2022" for GEBCO_2022) (elevation and Type Identifier grids) over the study domain by
HTTP range requests on CEDA's files, without a full download (D-016). Both files
store the grid contiguously, so each row of the window is one small range request.

Outputs (git-ignored): data/raw/bathymetry/gebco_2026_ncaspian.nc
Provenance (committed): results/provenance/gebco_2026.json
"""

from __future__ import annotations

import hashlib
import json
import sys

import caspian  # noqa: F401
import fsspec
import h5py
import numpy as np
import requests
import xarray as xr

from caspian import netcdf3
from caspian.config import RAW, RESULTS
from caspian.geography import DOMAIN
from caspian.remote import contiguous_window

VERSIONS = {
    "2026": {"elevation": "ice_surface_elevation/netcdf/GEBCO_2026.nc",
             "tid": "type_identifier_grid/netcdf/gebco_2026_tid.nc",
             "doi": "10.5285/4f68d5c7-45eb-f999-e063-7086abc036fa"},
    "2022": {"elevation": "ice_surface_elevation/netcdf/GEBCO_2022_CF.nc",
             "tid": "type_identifier_grid/netcdf/GEBCO_2022_TID.nc",
             "doi": "10.5285/e0f0bb80-ab44-2739-e053-6c86abc0289c"},
}
VERSION = sys.argv[1] if len(sys.argv) > 1 else "2026"
BASE = f"https://dap.ceda.ac.uk/bodc/gebco/global/gebco_{VERSION}"
FILES = {k: f"{BASE}/{VERSIONS[VERSION][k]}" for k in ("elevation", "tid")}
LISTING = f"https://data.ceda.ac.uk/bodc/gebco/global/gebco_{VERSION}/{{}}/netcdf?json"
DOI = VERSIONS[VERSION]["doi"]
MARGIN = 0.1


def listing() -> dict:
    out = {}
    for sub in ("ice_surface_elevation", "type_identifier_grid"):
        r = requests.get(LISTING.format(sub), timeout=60)
        r.raise_for_status()
        out.update({it["name"]: it for it in r.json()["items"] if it["name"].endswith(".nc")})
    return out


def main() -> int:
    w, s, e, n = DOMAIN.bounds
    items = listing()
    # elevation (HDF5, contiguous int16)
    with fsspec.open(FILES["elevation"], "rb", block_size=2**20) as f, h5py.File(f, "r") as h:
        lat, lon = h["lat"][:], h["lon"][:]
        v = h["elevation"]
        if v.chunks is not None:
            print("FAIL elevation is chunked; use caspian.remote.h5_window instead")
            return 1
        offset, ncols, dtype = v.id.get_offset(), v.shape[1], v.dtype
        attrs_e = {k: (x.decode() if isinstance(x, bytes) else str(x)) for k, x in v.attrs.items()
                   if k != "DIMENSION_LIST" and not k.startswith("_")}
    i0, i1 = np.searchsorted(lat, s - MARGIN), np.searchsorted(lat, n + MARGIN)
    j0, j1 = np.searchsorted(lon, w - MARGIN), np.searchsorted(lon, e + MARGIN)
    elev = contiguous_window(FILES["elevation"], offset, ncols, dtype, i0, i1, j0, j1)
    # TID (netCDF-3, contiguous int8)
    hdr = netcdf3.header(FILES["tid"])
    tlat = netcdf3.read_1d(FILES["tid"], hdr, "lat")
    tlon = netcdf3.read_1d(FILES["tid"], hdr, "lon")
    if not (np.allclose(tlat[i0:i1], lat[i0:i1]) and np.allclose(tlon[j0:j1], lon[j0:j1])):
        print("FAIL TID grid not aligned with elevation grid")
        return 1
    tv = hdr["variables"]["tid"]
    tid = contiguous_window(FILES["tid"], tv["begin"], tv["dims"][1][1], "i1", i0, i1, j0, j1)
    attrs_t = {k: str(x) for k, x in tv["atts"].items() if not k.startswith("_")}
    ds = xr.Dataset({"elevation": (("lat", "lon"), elev.astype("int16"), attrs_e),
                     "tid": (("lat", "lon"), tid.astype("int8"), attrs_t)},
                    coords={"lat": lat[i0:i1], "lon": lon[j0:j1]},
                    attrs={"source": f"GEBCO_{VERSION} Grid via CEDA (HTTP range subset)", "doi": DOI})
    out_dir = RAW / "bathymetry"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"gebco_{VERSION}_ncaspian.nc"
    ds.to_netcdf(out)
    names = [u.rsplit("/", 1)[1] for u in FILES.values()]
    prov = {"product": f"GEBCO_{VERSION} Grid", "doi": DOI, "urls": FILES,
            "ceda_md5_full_files": {nm: items.get(nm, {}).get("md5") for nm in names},
            "ceda_last_modified": {nm: items.get(nm, {}).get("last_modified") for nm in names},
            "window_index": [int(i0), int(i1), int(j0), int(j1)],
            "subset_bounds_wsen": [float(lon[j0]), float(lat[i0]), float(lon[j1 - 1]), float(lat[i1 - 1])],
            "shape": list(elev.shape), "elevation_attrs": attrs_e, "tid_attrs": attrs_t,
            "elevation_range_m": [int(elev.min()), int(elev.max())],
            "tid_values_present": sorted(int(x) for x in np.unique(tid)),
            "subset_sha256": hashlib.sha256(out.read_bytes()).hexdigest()}
    (RESULTS / "provenance").mkdir(parents=True, exist_ok=True)
    (RESULTS / "provenance" / f"gebco_{VERSION}.json").write_text(json.dumps(prov, indent=1), encoding="utf-8")
    print(json.dumps(prov, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
