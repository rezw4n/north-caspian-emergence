"""QC finding of the robustness runs: cells inside the stage-2 cover that no
primary stage-2 tile observed, by distance to the seams of the 100 km export tiles.
The export regions are lon/lat polygons with planar edges (a6_stage2_export.ee_planar),
so a tile edge that is straight in the equal-area grid bows in the export, and the
cells between the bow and the seam belong to neither tile. Read-only on the primary;
writes results/robustness/r21_seam_gaps/summary.json.

Usage: python scripts/r21_seam_gaps.py
"""

from __future__ import annotations

import json
import sys

import caspian  # noqa: F401
import numpy as np
from affine import Affine
from rasterio.features import rasterize
from shapely.geometry import mapping

from caspian import robustness as R
from caspian.config import CRS_EQUAL_AREA, INTERIM, RESULTS

RES, TILE = 60.0, 100_000.0


def main() -> int:
    cov = R.load_cover(RESULTS / "study_area" / "zone_v6_cover_buffered.geojson", CRS_EQUAL_AREA)
    x0, y0, x1, y1 = cov.bounds
    x0, y1 = np.floor(x0 / RES) * RES, np.ceil(y1 / RES) * RES
    cols, rows = int(np.ceil((x1 - x0) / RES)), int(np.ceil((y1 - y0) / RES))
    tr = Affine(RES, 0, x0, 0, -RES, y1)
    inside = rasterize([(mapping(cov), 1)], out_shape=(rows, cols), transform=tr, fill=0, dtype="uint8") > 0
    seen = np.zeros((rows, cols), bool)
    for f in sorted((INTERIM / "stage2" / "main").glob("tile_*.npz")):
        with np.load(f) as z:
            t = z["transform"]
            acc = None
            for k in z.files:
                if k.startswith("v_"):
                    a = z[k] > 0
                    acc = a if acc is None else (acc | a)
        c0, r0 = int(round((float(t[2]) - x0) / RES)), int(round((y1 - float(t[5])) / RES))
        h, w = acc.shape
        rr0, cc0 = max(r0, 0), max(c0, 0)
        rr1, cc1 = min(r0 + h, rows), min(c0 + w, cols)
        seen[rr0:rr1, cc0:cc1] |= acc[rr0 - r0:rr1 - r0, cc0 - c0:cc1 - c0]
    xs = x0 + (np.arange(cols) + 0.5) * RES
    ys = y1 - (np.arange(rows) + 0.5) * RES
    dx = np.abs(xs - np.round(xs / TILE) * TILE)
    dy = np.abs(ys - np.round(ys / TILE) * TILE)
    d = np.minimum(dx[None, :], dy[:, None])
    unseen = inside & ~seen
    rec = {"cover_cells": int(inside.sum()), "unseen_cells": int(unseen.sum()), "by_distance_to_seam_m": {}}
    for lo, hi in ((0, 120), (120, 300), (300, 1000), (1000, 1e9)):
        band = inside & (d >= lo) & (d < hi)
        rec["by_distance_to_seam_m"][f"{lo:g}-{hi:g}"] = {
            "cells": int(band.sum()), "unseen": int((band & unseen).sum()),
            "unseen_share": float((band & unseen).sum() / band.sum()) if band.any() else None}
    rec["unseen_within_120m_of_seam_km2"] = rec["by_distance_to_seam_m"]["0-120"]["unseen"] * RES * RES / 1e6
    rec["note"] = ("cells never valid in any group (e.g. permanently land or water-masked) are also 'unseen'; the "
                   "seam excess over the background rate is the export gap")
    out = RESULTS / "robustness" / "r21_seam_gaps"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
