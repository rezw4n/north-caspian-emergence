"""A7: the beyond-domain state for the hindcast (D-050, D-067; PREREG Amendment 10).

Cells outside the domain (and in its rim) are unknown in the hindcast unless an
independent state is supplied. This script supplies land only:
1. Region: the product bounding box (DOMAIN bounds plus MARGIN_DEG, as the product
   downloads in scripts/a7_get_*.py) minus the domain eroded by INNER_M, a ring that
   holds every cell outside the domain and every cell within INNER_M inside it.
2. Observed near-maximum water: the 1995-96 extent of zone v6
   (pipeline_ee.extents_v6, band `high`: water in at least HIGH_MIN_FRAC of clear
   May-September TM scenes, GSW max_extent where clear observations are too few),
   exported over the ring at ZONE_SCALE on the equal-area grid.
3. State: land (0) where a ring cell lies farther than ZONE_DILATE_M (the zone
   dilation) from that water; unknown (NaN) everywhere else. Water is never
   assigned, so that open sea beyond the domain (the Middle Caspian) cannot add
   agreement to the products' scores.

Land beyond the 1995-96 near-maximum extent is land at every evaluated level, since
no levelled scene lies above the 1995-96 high stand. The state comes only from
Landsat and GSW observations, independent of the audited products.

Outputs (committed, small): results/study_area/beyond_state_250m.npz with `state`
(int8: 0 land, -1 unknown), `observed` (1 where the ring was observed), transform,
crs. Use: hindcast.outside_state_from_zone(..., beyond_state=...) after nearest-
neighbour resampling to the hindcast grid (-1 becomes NaN).

Usage: python scripts/a7_beyond_state.py export | build
"""

from __future__ import annotations

import json
import sys

import caspian  # noqa: F401
import numpy as np
import pyproj
from shapely.geometry import box
from shapely.ops import transform

from caspian.config import CRS_EQUAL_AREA, EE_PROJECT, RESULTS
from caspian.geography import DOMAIN

MARGIN_DEG = 0.1          # as MARGIN in scripts/a7_get_gebco.py and a7_get_etopo.py
INNER_M = 5000
ROOT = f"projects/{EE_PROJECT}/assets/caspian"
ASSET = f"{ROOT}/beyond_high_v1"
OUT = RESULTS / "study_area" / "beyond_state_250m.npz"
FWD = pyproj.Transformer.from_crs("EPSG:4326", CRS_EQUAL_AREA, always_xy=True).transform
INV = pyproj.Transformer.from_crs(CRS_EQUAL_AREA, "EPSG:4326", always_xy=True).transform


def bbox_lonlat():
    w, s, e, n = DOMAIN.bounds
    return (w - MARGIN_DEG, s - MARGIN_DEG, e + MARGIN_DEG, n + MARGIN_DEG)


def ring_laea():
    dom = transform(FWD, DOMAIN.segmentize(0.01))
    bb = transform(FWD, box(*bbox_lonlat()).segmentize(0.01))
    return bb.difference(dom.buffer(-INNER_M))


def export() -> int:
    import ee

    from caspian import pipeline_ee
    sys.path.insert(0, str(RESULTS.parent / "scripts"))
    from a6_stage2_export import ee_planar
    ee.Initialize(project=EE_PROJECT)
    region = ee_planar(transform(INV, ring_laea()))
    tr, dims = pipeline_ee.laea_transform(pipeline_ee.ZONE_SCALE, bbox_lonlat())
    img = pipeline_ee.extents_v6(region).select(["high", "filled"]).uint8()
    t = ee.batch.Export.image.toAsset(image=img.clip(region), description="beyond_high_v1", assetId=ASSET,
                                      region=region, crs=pipeline_ee.LAEA_WKT, crsTransform=tr, maxPixels=1e10)
    t.start()
    rec = {"task": t.id, "asset": ASSET, "transform": tr, "dims": dims, "bbox_lonlat": bbox_lonlat(),
           "ring_km2": round(ring_laea().area / 1e6, 1)}
    (RESULTS / "inventory" / "beyond_high_task.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print("submitted", t.id, rec["ring_km2"], "km2")
    return 0


def build() -> int:
    import ee
    from rasterio.features import geometry_mask
    from rasterio.transform import Affine
    from shapely.geometry import mapping

    from caspian import ee_io, hindcast, pipeline_ee
    ee.Initialize(project=EE_PROJECT)
    rec = json.loads((RESULTS / "inventory" / "beyond_high_task.json").read_text(encoding="utf-8"))
    tr, dims = rec["transform"], rec["dims"]
    sentinel = 255
    img = ee.Image(ASSET).select(["high"]).unmask(sentinel)
    arr = ee_io.fetch(img, ["high"], tr, dims, pipeline_ee.LAEA_WKT, block=1024, bytes_per_px=1)["high"]
    ring = ~geometry_mask([mapping(ring_laea())], out_shape=(dims[1], dims[0]), transform=Affine(*tr), invert=False)
    observed = (arr != sentinel) & ring
    b = hindcast.beyond_land_state(arr == 1, observed, float(tr[0]), pipeline_ee.ZONE_DILATE_M)
    state = np.where(b == 0, 0, -1).astype(np.int8)
    np.savez_compressed(OUT, state=state, observed=observed.astype(np.uint8), transform=np.array(tr),
                        crs=pipeline_ee.LAEA_WKT)
    cell_km2 = (tr[0] / 1000) ** 2
    print(f"ring observed {observed.sum() * cell_km2:.0f} km2 | land {np.sum(state == 0) * cell_km2:.0f} km2 "
          f"| unknown in ring {np.sum((state == -1) & ring) * cell_km2:.0f} km2 -> {OUT}")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "export":
        sys.exit(export())
    if cmd == "build":
        sys.exit(build())
    raise SystemExit(__doc__)
