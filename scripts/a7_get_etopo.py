"""Subset ETOPO 2022 v1 15 arc-second (surface elevation, source ID, geoid height)
over the study domain via NOAA NCEI THREDDS OPeNDAP.

Outputs (git-ignored): data/raw/bathymetry/etopo_2022_ncaspian.nc
Provenance (committed): results/provenance/etopo_2022.json
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

DODS = "https://www.ngdc.noaa.gov/thredds/dodsC/global/ETOPO2022/15s"
LAYERS = {"elevation": ("15s_surface_elev_netcdf", "surface"),
          "source_id": ("15s_surface_sid_netcdf", "surface_sid"),
          "geoid_height": ("15s_geoid_netcdf", "geoid")}
TILES = ["N45E045", "N60E045"]
DOI = "10.25921/fd45-gt74"
MARGIN = 0.1


def main() -> int:
    w, s, e, n = DOMAIN.bounds
    parts = {}
    urls = {}
    attrs = {}
    for name, (folder, suffix) in LAYERS.items():
        pieces = []
        for tile in TILES:
            url = f"{DODS}/{folder}/ETOPO_2022_v1_15s_{tile}_{suffix}.nc"
            urls.setdefault(name, []).append(url)
            ds = xr.open_dataset(url, engine="netcdf4")
            var = [v for v in ds.data_vars if ds[v].ndim == 2][0]
            sub = ds[var].sel(lat=slice(s - MARGIN, n + MARGIN), lon=slice(w - MARGIN, e + MARGIN))
            if sub.size:
                pieces.append(sub.load())
            attrs[name] = {k: str(v) for k, v in ds[var].attrs.items()}
            attrs[name]["_var"] = var
        da = xr.concat(pieces, dim="lat").sortby("lat")
        da = da.isel(lat=~da.get_index("lat").duplicated())
        parts[name] = da.rename(name)
    out_ds = xr.Dataset(parts)
    out_ds.attrs = {"source": "ETOPO 2022 v1 15s via NCEI THREDDS OPeNDAP", "doi": DOI}
    out_dir = RAW / "bathymetry"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "etopo_2022_ncaspian.nc"
    out_ds.to_netcdf(out)
    el = out_ds["elevation"].values
    prov = {"product": "ETOPO 2022 v1 15 arc-second", "doi": DOI, "urls": urls, "attrs": attrs,
            "shape": list(el.shape), "elevation_range_m": [float(np.nanmin(el)), float(np.nanmax(el))],
            "geoid_height_range_m": [float(out_ds["geoid_height"].min()), float(out_ds["geoid_height"].max())],
            "source_ids_present": sorted(int(v) for v in np.unique(out_ds["source_id"].values)
                                         if np.isfinite(v)),
            "subset_sha256": hashlib.sha256(out.read_bytes()).hexdigest()}
    (RESULTS / "provenance").mkdir(parents=True, exist_ok=True)
    (RESULTS / "provenance" / "etopo_2022.json").write_text(json.dumps(prov, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in prov.items() if k != "attrs"}, indent=1))
    print(json.dumps(attrs, indent=1)[:2000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
