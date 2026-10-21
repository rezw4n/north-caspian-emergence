"""A9 generality test, descriptive product lineage over each arm's emerged band
(PREREG Amendment 16 item 9, descriptive, not deciding; D-094).

Emerged band: near-maximum but not near-minimum water (scripts/g1_static.py, 250 m),
inside the arm. At each band cell centre (nearest native cell): the ETOPO 2022 source
ID (MacFerrin et al. 2025, Table 1: 10 FABDEM, 11 GEBCO Lake Depths; other codes
reported as numbers), the GEBCO_2026 type identifier and the SRTM15+ source ID, and
for every product the share of band cells holding its modal height (a flat lake
surface shows as a large share) and the height range. No observed extent is used.

Output: results/generality/<basin>/lineage.json
Usage: python scripts/g1_lineage.py <basin>
"""

from __future__ import annotations

import json
import sys

import caspian  # noqa: F401
import numpy as np
import pyproj
import xarray as xr

from caspian import generality as G
from caspian.config import INTERIM, RAW, RESULTS

ETOPO_SOURCES = {10: "FABDEM (land topography)", 11: "GEBCO Lake Depths"}   # MacFerrin et al. 2025, Table 1


def main(basin: str) -> int:
    B = G.BASINS[basin]
    st = np.load(INTERIM / "generality" / basin / "static_250m.npz")
    tr = [float(v) for v in st["transform"]]
    x, y = G.cell_centres(tr, st["high"].shape)
    X, Y = np.meshgrid(x, y)
    lon, lat = pyproj.Transformer.from_crs(B["crs"], "EPSG:4326", always_xy=True).transform(X, Y)
    raw = RAW / "generality" / basin / "bathymetry"
    out = {"rule": "Amendment 16 item 9 (descriptive): band = near-maximum and not near-minimum water, per arm, 250 m",
           "arms": {}}
    for k, arm in enumerate(B["sectors"], start=1):
        band = st["high"] & ~st["low"] & (st["armlab"] == k)
        lo, la = lon[band], lat[band]
        rec = {"band_cells_250m": int(band.sum()), "band_km2": round(float(band.sum()) * 0.0625, 1)}
        for name, f, layer in (("etopo_2022", "etopo_2022.nc", "source_id"), ("gebco_2026", "gebco_2026.nc", "tid"),
                               ("srtm15p_v2.7", "srtm15p_v2.7.nc", "source_id")):
            with xr.open_dataset(raw / f) as ds:
                v = ds[layer].sel(lon=xr.DataArray(lo), lat=xr.DataArray(la), method="nearest").values
            vals, cnt = np.unique(v[np.isfinite(v)].astype(int), return_counts=True)
            shares = {str(int(a)): round(float(c) / band.sum(), 4) for a, c in zip(vals, cnt) if c / band.sum() >= 0.005}
            rec[f"{name}_{layer}_shares"] = shares
        for name, f, layer in (("gebco_2026", "gebco_2026.nc", "elevation"), ("gebco_2022", "gebco_2022.nc", "elevation"),
                               ("etopo_2022", "etopo_2022.nc", "elevation"), ("srtm15p_v2.7", "srtm15p_v2.7.nc", "elevation")):
            with xr.open_dataset(raw / f) as ds:
                v = ds[layer].sel(lon=xr.DataArray(lo), lat=xr.DataArray(la), method="nearest").values.astype(float)
            vals, cnt = np.unique(np.round(v[np.isfinite(v)], 2), return_counts=True)
            rec[f"{name}_heights"] = {"modal_m": float(vals[np.argmax(cnt)]), "modal_share": round(float(cnt.max() / v.size), 4),
                                      "p05_m": round(float(np.nanpercentile(v, 5)), 2), "p95_m": round(float(np.nanpercentile(v, 95)), 2),
                                      "whole_metre_share": round(float(np.mean(np.isclose(v, np.round(v)))), 4)}
        dep = INTERIM / "generality" / basin / "dep3_60m.npy"
        if dep.exists():   # the positive-control substitute: flat (hydro-flattened) patches in the band
            z = np.load(dep)
            tr60, shp60 = G.grid(B["bbox"], B["crs"], 60)
            x60, y60 = G.cell_centres(tr60, shp60)
            ci = np.floor((x60 - tr[2]) / tr[0]).astype(int)
            ri = np.floor((y60 - tr[5]) / tr[4]).astype(int)
            ok_c, ok_r = (ci >= 0) & (ci < band.shape[1]), (ri >= 0) & (ri < band.shape[0])
            b60 = np.zeros(shp60, bool)
            b60[np.ix_(ok_r, ok_c)] = band[np.ix_(ri[ok_r], ci[ok_c])]
            v = z[b60]
            v = v[np.isfinite(v)]
            vals, cnt = np.unique(np.round(v, 2), return_counts=True)
            top = np.argsort(cnt)[::-1][:3]
            rec["dep3_band"] = {"cells_60m": int(v.size),
                                "plateaus": [{"height_m": float(vals[i]), "share": round(float(cnt[i] / v.size), 4)} for i in top],
                                "p05_m": round(float(np.percentile(v, 5)), 2), "p95_m": round(float(np.percentile(v, 95)), 2)}
        out["arms"][arm] = rec
    out["etopo_source_names"] = {str(k): v for k, v in ETOPO_SOURCES.items()}
    (RESULTS / "generality" / basin / "lineage.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
