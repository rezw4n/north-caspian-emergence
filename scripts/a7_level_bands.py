"""A7 (interpretation, D-087): product shoreline displacement by level band, and the
ETOPO 2022 source datasets over the emerged band.

ETOPO 2022 (MacFerrin et al. 2025, Table 1) fills the emerged band from FABDEM (source
ID 10, land topography from the Copernicus DEM, acquired 2010-2015) and GEBCO Lake
Depths (ID 11). Product errors are tabulated by level band to show where each
product's skill comes from.

Outputs (committed): results/hindcast/<tag>/displacement_by_level_band.csv,
results/hindcast/<tag>/etopo2022_sources_emerged_band.json
Usage: python scripts/a7_level_bands.py <tag>
"""

import json
import sys

import caspian  # noqa: F401
import numpy as np
import pandas as pd
import pyproj
import xarray as xr

from caspian.config import CRS_EQUAL_AREA, RAW, RESULTS

BANDS = [-29.0, -28.3, -27.7, -27.0, -26.3, -25.5]
ETOPO_SOURCES = {10: "FABDEM (land topography)", 11: "GEBCO Lake Depths"}   # MacFerrin et al. 2025, Table 1


def main(tag: str) -> int:
    d = pd.read_csv(RESULTS / "hindcast" / tag / "displacement.csv")
    d = d[(d.epoch == "pooled") & (d.sector != "all") & d.observed_level & (d.n > 0)].copy()
    d["band"] = pd.cut(d["level_m"], BANDS).astype(str)
    t = d.groupby(["variant", "band", "product"])["median_m"].median().unstack().reset_index()
    t.to_csv(RESULTS / "hindcast" / tag / "displacement_by_level_band.csv", index=False, float_format="%.1f")
    print(t[t.variant == "shape"].round(0).to_string(index=False))
    ds = xr.open_dataset(RAW / "bathymetry" / "etopo_2022_ncaspian.nc")
    ex = np.load(RESULTS / "study_area" / "extents_v6_250m.npz")
    band = (ex["high"] == 1) & (ex["low"] == 0)
    tr = ex["transform"]
    r, c = np.nonzero(band)
    lon, lat = pyproj.Transformer.from_crs(CRS_EQUAL_AREA, "EPSG:4326", always_xy=True).transform(
        tr[2] + (c + 0.5) * tr[0], tr[5] + (r + 0.5) * tr[4])
    v = ds["source_id"].sel(lon=xr.DataArray(lon), lat=xr.DataArray(lat), method="nearest").values
    u, n = np.unique(v, return_counts=True)
    share = {f"{int(k)} {ETOPO_SOURCES.get(int(k), 'other')}": round(float(m / n.sum()), 4) for k, m in zip(u, n)}
    (RESULTS / "hindcast" / tag / "etopo2022_sources_emerged_band.json").write_text(
        json.dumps({"cells_250m": int(n.sum()), "share": share, "source": "MacFerrin et al. 2025, Table 1"}, indent=1),
        encoding="utf-8")
    print(share)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
