"""Subset SRTM15+ V2.7 (elevation z and source ID) over the study domain by HTTP
range requests on the Scripps netCDF-4 files.

Outputs (git-ignored): data/raw/bathymetry/srtm15p_v2.7_ncaspian.nc
Provenance (committed): results/provenance/srtm15p_v2.7.json
"""

from __future__ import annotations

import hashlib
import json
import sys

import caspian  # noqa: F401
import numpy as np
import xarray as xr

from caspian.config import RAW, RESULTS
from caspian.geography import DOMAIN
from caspian.remote import h5_window

BASE = "https://topex.ucsd.edu/pub/srtm15_plus"
FILES = {"elevation": f"{BASE}/SRTM15_V2.7.nc", "source_id": f"{BASE}/SID_V2.7.nc",
         "readme": f"{BASE}/README_V2.7.txt"}
CITATION_DOI = "10.1029/2019ea000658"  # Tozer et al. 2019, SRTM15+ description


def main() -> int:
    b = DOMAIN.bounds
    lat, lon, z, attrs_z, chunks = h5_window(FILES["elevation"], "z", b)
    try:
        lat2, lon2, sid, attrs_s, _ = h5_window(FILES["source_id"], "z", b)
    except KeyError:
        lat2, lon2, sid, attrs_s, _ = h5_window(FILES["source_id"], "sid", b)
    if not (np.allclose(lat, lat2) and np.allclose(lon, lon2)):
        print("FAIL SID grid not aligned with elevation grid")
        return 1
    ds = xr.Dataset({"elevation": (("lat", "lon"), z.astype("float32"), attrs_z),
                     "source_id": (("lat", "lon"), sid.astype("int32"), attrs_s)},
                    coords={"lat": lat, "lon": lon},
                    attrs={"source": "SRTM15+ V2.7 (Scripps), HTTP range subset", "urls": json.dumps(FILES)})
    out_dir = RAW / "bathymetry"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "srtm15p_v2.7_ncaspian.nc"
    ds.to_netcdf(out)
    prov = {"product": "SRTM15+ V2.7", "citation_doi": CITATION_DOI, "urls": FILES,
            "shape": list(z.shape), "hdf5_chunks": list(chunks) if chunks else None,
            "elevation_attrs": attrs_z, "elevation_range_m": [float(z.min()), float(z.max())],
            "n_source_ids": int(np.unique(sid).size),
            "subset_sha256": hashlib.sha256(out.read_bytes()).hexdigest()}
    (RESULTS / "provenance").mkdir(parents=True, exist_ok=True)
    (RESULTS / "provenance" / "srtm15p_v2.7.json").write_text(json.dumps(prov, indent=1), encoding="utf-8")
    print(json.dumps(prov, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
