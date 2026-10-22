"""PREREG section 6: the spatial block assignment, frozen before any evaluation.

Square blocks on the equal-area grid, aligned to multiples of the block size;
each block takes the sector that contains its centre; blocks are randomly assigned to
train / calib / test (0.6 / 0.2 / 0.2), stratified by sector, with seed 20261020
(caspian.blocking, caspian.design). The block-size check of section 6 (semivariogram of
DEM residuals in training blocks) may later enlarge the blocks; this table is the
initial 25 km assignment, and any reassignment uses the same code and seed.

Output (committed): results/study_area/blocks_<size>km.csv

Usage: python scripts/a6_blocks.py [block_size_m]
"""

import sys

import caspian  # noqa: F401
import pyproj
from shapely.ops import transform

from caspian import blocking, design
from caspian.config import CRS_EQUAL_AREA, RESULTS
from caspian.geography import DOMAIN, sectors


def main(size_m: float = design.BLOCK_SIZE_M) -> int:
    t = pyproj.Transformer.from_crs("EPSG:4326", CRS_EQUAL_AREA, always_xy=True).transform
    secs = {name: transform(t, poly) for name, (_, poly) in sectors().items()}
    dom = transform(t, DOMAIN)
    blocks = blocking.assign_sectors(blocking.block_grid(dom.bounds, size_m), secs)
    blocks["split"] = blocking.assign_splits(blocks)
    out = RESULTS / "study_area" / f"blocks_{int(size_m / 1000)}km.csv"
    blocks.to_csv(out, index=False)
    print(out, len(blocks), "blocks")
    print(blocks.groupby(["sector", "split"]).size().unstack(fill_value=0).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main(float(sys.argv[1]) if len(sys.argv) > 1 else design.BLOCK_SIZE_M))
