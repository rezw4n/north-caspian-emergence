"""Open-sea mask at 1 km on the equal-area grid: 2025-26
near-minimum water whose 20 km surroundings are at least 80% water. It restricts
SWOT pixels to the open sea in scripts/a5_swot_sector_levels.py.

Output (committed, small): results/study_area/sea_mask_1km.npz (mask, transform, crs).
"""

import sys

import caspian  # noqa: F401
import ee
import numpy as np

from caspian import ee_io, pipeline_ee
from caspian.config import EE_PROJECT, RESULTS
from caspian.geography import DOMAIN

EXTENTS = f"projects/{EE_PROJECT}/assets/caspian/extents_v6"
SCALE = 1000


def main() -> int:
    ee.Initialize(project=EE_PROJECT)
    proj = pipeline_ee.laea(SCALE)
    low = ee.Image(EXTENTS).select("low")
    lo_c = low.reduceResolution(ee.Reducer.mean(), maxPixels=1024).reproject(proj)
    r = int(round(pipeline_ee.SEA_RADIUS_M / SCALE))
    sea = lo_c.gte(0.5).And(lo_c.reduceNeighborhood(ee.Reducer.mean(), ee.Kernel.circle(r, "pixels"))
                            .reproject(proj).gte(pipeline_ee.SEA_MIN_FRAC)).rename("sea").uint8()
    transform, dims = pipeline_ee.laea_transform(SCALE, DOMAIN.bounds)
    arr = ee_io.fetch(sea, ["sea"], transform, dims, pipeline_ee.LAEA_WKT, block=1024)["sea"]
    out = RESULTS / "study_area" / "sea_mask_1km.npz"
    np.savez_compressed(out, mask=arr.astype(np.uint8), transform=np.array(transform), crs=pipeline_ee.LAEA_WKT)
    print("sea cells:", int(arr.sum()), "of", arr.size, "->", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
