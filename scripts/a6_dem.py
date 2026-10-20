"""A6: waterline DEM (PREREG section 4): TIN interpolation of the E2 + E3 waterline
points to design.DEM_RES_M on the equal-area grid, with the interpolation distance
(distance to the nearest point) stored for every pixel (the conformal scale
variable, PREREG section 8).

Processed in TILE_M tiles with an OVERLAP_M margin of points, so that triangles near
a tile edge see their neighbours. Pixels outside the stage-2 cover are NaN. Points
flagged as static contours (column `static`, D-034) are excluded when the column
exists.

Outputs: data/processed/dem/<tag>/dem_<i>_<j>.tif (band 1: height in m on the DAHITI
reference; band 2: interpolation distance in m)

Usage: python scripts/a6_dem.py <tag> [--epochs E2,E3]
"""

from __future__ import annotations

import argparse
import json
import math
import sys

import caspian  # noqa: F401
import numpy as np
import pandas as pd
import pyproj
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import Affine
from shapely.geometry import box, mapping, shape
from shapely.ops import transform as shp_transform

from caspian import dem, design
from caspian.config import CRS_EQUAL_AREA, DATA, INTERIM, RESULTS

TILE_M = 50_000
OVERLAP_M = 5_000


def main(argv) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--epochs", default="E2,E3")
    a = ap.parse_args(argv)
    res = design.DEM_RES_M
    p = pd.read_csv(INTERIM / "waterlines" / a.tag / "points.csv.gz")
    p = p[p["epoch"].isin(a.epochs.split(","))]
    if "static" in p.columns:
        p = p[~p["static"].astype(bool)]
    fwd = pyproj.Transformer.from_crs("EPSG:4326", CRS_EQUAL_AREA, always_xy=True).transform
    cov = shp_transform(fwd, shape(json.loads((RESULTS / "study_area" / "zone_v6_cover_buffered.geojson")
                                              .read_text(encoding="utf-8"))["features"][0]["geometry"]))
    out = DATA / "processed" / "dem" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    xy = p[["x", "y"]].to_numpy()
    z = p["z_m"].to_numpy()
    x0, y0, x1, y1 = cov.bounds
    n_tiles = 0
    for i in range(math.floor(x0 / TILE_M), math.ceil(x1 / TILE_M)):
        for j in range(math.floor(y0 / TILE_M), math.ceil(y1 / TILE_M)):
            tb = box(i * TILE_M, j * TILE_M, (i + 1) * TILE_M, (j + 1) * TILE_M)
            part = cov.intersection(tb)
            if part.is_empty:
                continue
            m = ((xy[:, 0] > tb.bounds[0] - OVERLAP_M) & (xy[:, 0] < tb.bounds[2] + OVERLAP_M)
                 & (xy[:, 1] > tb.bounds[1] - OVERLAP_M) & (xy[:, 1] < tb.bounds[3] + OVERLAP_M))
            if m.sum() < 3:
                continue
            nx = ny = int(TILE_M / res)
            gx, gy = dem.grid_centers(i * TILE_M, (j + 1) * TILE_M, nx, ny, res)
            h = dem.tin(xy[m], z[m], gx, gy).astype(np.float32)
            d = dem.interpolation_distance(xy[m], gx, gy).astype(np.float32)
            aff = Affine(res, 0, i * TILE_M, 0, -res, (j + 1) * TILE_M)
            outside = geometry_mask([mapping(part)], out_shape=(ny, nx), transform=aff, invert=False)
            h[outside] = np.nan
            d[outside | ~np.isfinite(h)] = np.nan
            if not np.isfinite(h).any():
                continue
            with rasterio.open(out / f"dem_{i}_{j}.tif", "w", driver="GTiff", width=nx, height=ny, count=2,
                               dtype="float32", crs=pyproj.CRS(CRS_EQUAL_AREA).to_wkt(), transform=aff,
                               nodata=np.nan, compress="deflate", tiled=True) as dst:
                dst.write(h, 1)
                dst.write(d, 2)
                dst.set_band_description(1, "height_m_dahiti")
                dst.set_band_description(2, "interpolation_distance_m")
            n_tiles += 1
            print(f"tile {i}_{j}: {int(m.sum())} points, {int(np.isfinite(h).sum())} cells", flush=True)
    print("DEM tiles written:", n_tiles, "->", out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
