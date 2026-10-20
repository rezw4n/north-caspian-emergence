"""A6/A7: vegetation fraction of the emerged band per sector, and the vegetated-sector
labels for the H1 permutation test (PREREG section 2; D-071).

Emerged band: cells that are water in the 1995-96 near-maximum extent and not water
in the 2025-26 near-minimum extent (zone v6 extents, Earth Engine asset extents_v6,
fetched once at 250 m to results/study_area/extents_v6_250m.npz), inside the domain.
For each 60 m stage-2 cell of the band, the emergent-vegetation share is
caspian.frequency.vegetation_share of the observed (o) and valid (v) counts summed
over all E3 groups; a cell is vegetated when the share is at least VEG_CELL_SHARE.
A sector is vegetated when vegetated cells make up at least VEG_SECTOR_FRACTION of
its band cells with o >= MIN_VALID_OBS. Deltaic sectors: Volga and Ural (PREREG §2).

Outputs: results/levels/vegetation_fraction.json; results/levels/sector_labels.csv
(sector, deltaic, vegetated), the input of h1.block_test.

Usage: python scripts/a6_vegetation.py fetch-extents | <stage-2 tag>
"""

from __future__ import annotations

import json
import re
import sys

import caspian  # noqa: F401
import numpy as np
import pandas as pd
import pyproj
import shapely
from shapely.ops import transform

from caspian import design, frequency
from caspian.config import CRS_EQUAL_AREA, EE_PROJECT, INTERIM, RESULTS
from caspian.geography import DOMAIN, SECTOR_ORDER, sectors

VEG_CELL_SHARE = 0.5
VEG_SECTOR_FRACTION = 0.10
DELTAIC = ("volga", "ural")
EXTENTS = RESULTS / "study_area" / "extents_v6_250m.npz"
BAND = re.compile(r"([ov])_(E3)_b(\d{3})_p([01])$")
FWD = pyproj.Transformer.from_crs("EPSG:4326", CRS_EQUAL_AREA, always_xy=True).transform


def fetch_extents() -> int:
    import ee

    from caspian import ee_io, pipeline_ee
    ee.Initialize(project=EE_PROJECT)
    tr, dims = pipeline_ee.laea_transform(pipeline_ee.ZONE_SCALE, DOMAIN.bounds)
    img = ee.Image(f"projects/{EE_PROJECT}/assets/caspian/extents_v6").select(["high", "low"]).unmask(255)
    arr = ee_io.fetch(img, ["high", "low"], tr, dims, pipeline_ee.LAEA_WKT, block=1024, bytes_per_px=1)
    np.savez_compressed(EXTENTS, high=arr["high"].astype(np.uint8), low=arr["low"].astype(np.uint8),
                        transform=np.array(tr), crs=pipeline_ee.LAEA_WKT)
    print("extents", dims, "->", EXTENTS)
    return 0


def main(tag: str) -> int:
    ex = np.load(EXTENTS)
    et = ex["transform"]
    high, low = ex["high"], ex["low"]
    band250 = (high == 1) & (low == 0)
    polys = {k: transform(FWD, p.segmentize(0.01)) for k, (_, p) in sectors().items()}
    for p in polys.values():
        shapely.prepare(p)
    n_band = {s: 0 for s in SECTOR_ORDER}
    n_veg = {s: 0 for s in SECTOR_ORDER}
    for f in sorted((INTERIM / "stage2" / tag).glob("tile_*.npz")):
        z = np.load(f)
        o = v = None
        for k in z.files:
            m = BAND.match(k)
            if not m:
                continue
            a = z[k].astype(np.int64)
            if m.group(1) == "o":
                o = a if o is None else o + a
            else:
                v = a if v is None else v + a
        if o is None:
            raise SystemExit(f"{f.name}: no observed-count bands (D-071 needs the main export)")
        t = z["transform"]
        rows, cols = np.indices(o.shape)
        x = t[2] + (cols + 0.5) * t[0]
        y = t[5] + (rows + 0.5) * t[4]
        er = np.floor((y - et[5]) / et[4]).astype(int)
        ec = np.floor((x - et[2]) / et[0]).astype(int)
        inside = (er >= 0) & (er < band250.shape[0]) & (ec >= 0) & (ec < band250.shape[1])
        in_band = np.zeros(o.shape, bool)
        in_band[inside] = band250[er[inside], ec[inside]]
        share = frequency.vegetation_share(o, v, design.MIN_VALID_OBS)
        use = in_band & np.isfinite(share)
        if not use.any():
            continue
        xs, ys, sh = x[use], y[use], share[use]
        for s, p in polys.items():
            hit = shapely.contains_xy(p, xs, ys)
            n_band[s] += int(hit.sum())
            n_veg[s] += int((hit & (sh >= VEG_CELL_SHARE)).sum())
        print(f.name, int(use.sum()), "band cells", flush=True)
    frac = {s: (n_veg[s] / n_band[s] if n_band[s] else None) for s in SECTOR_ORDER}
    lab = pd.DataFrame({"sector": SECTOR_ORDER,
                        "deltaic": [s in DELTAIC for s in SECTOR_ORDER],
                        "vegetated": [bool(frac[s] is not None and frac[s] >= VEG_SECTOR_FRACTION) for s in SECTOR_ORDER]})
    out = {"tag": tag, "veg_cell_share": VEG_CELL_SHARE, "veg_sector_fraction": VEG_SECTOR_FRACTION,
           "min_valid_obs": design.MIN_VALID_OBS, "n_band_cells": n_band, "n_vegetated_cells": n_veg,
           "fraction": frac}
    (RESULTS / "levels" / "vegetation_fraction.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    lab.to_csv(RESULTS / "levels" / "sector_labels.csv", index=False)
    print(json.dumps(frac, indent=1))
    print(lab.to_string(index=False))
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "fetch-extents":
        sys.exit(fetch_extents())
    if a:
        sys.exit(main(a[0]))
    raise SystemExit(__doc__)
