"""Robustness (docs/ROBUSTNESS_PLAN.md): build a variant stage-2 tag from existing
stage-2 tiles by count algebra alone (caspian.robustness), so that the downstream
steps (scripts/r00_pipeline.py, scripts/a6_*.py, scripts/a7_hindcast.py) run on it
unchanged. No Earth Engine.

Order of operations, per tile key:
1. sum the tiles of every --from tag band by band (e.g. Landsat + Sentinel-2, D-064
   step 3; tiles must share the transform);
2. --add / --sub: screening deltas (main + ADD - SUB, robustness.apply_delta;
   scripts/r04_screening.py builds their plans);
3. --substitute <tag>: inside each block of a block-sample Earth Engine alternative
   (tiles tile_<i>_<j>.npz of 25 km blocks, export record results/inventory/
   stage2_<tag>.json), the main counts are replaced by the alternative's (the hybrid
   mosaic of the robustness block sample R; robustness.substitute_window). The block
   region is the stage-2 cover within the block square, and within the block's sector
   when the record names one, rasterised by cell centre on the block tile's grid;
4. --vegetation as_dry | as_water (check 9; needs the observed count o, D-071);
5. group mapping (robustness.merge_groups): --e1-end with --e1-ends (Amendment 4),
   --drop / --keep-only class tokens (sensor, season, threshold source; split plans,
   scripts/r16_split_plan.py), --fine-from-halves (0.05 m bins) and --bin-factor N
   (0.10 m -> N x 0.10 m bins). Class tokens are always summed out at the end, so the
   variant carries the standard main-export band names.

Outputs: data/interim/stage2/<variant>/tile_<key>.npz and variant.json (sources and
their tile checksums, operations, bin width of the result).

Usage:
  python scripts/r00_counts.py <variant> --from <tag>[,<tag>] [--add <tag>] [--sub <tag>]
      [--substitute <tag>] [--vegetation as_dry|as_water] [--drop T1,T2] [--keep-only T1,T2]
      [--e1-end ISO --e1-ends ISO,ISO,ISO] [--fine-from-halves] [--bin-factor N]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import caspian  # noqa: F401
import numpy as np

from caspian import design, robustness as R
from caspian.config import CRS_EQUAL_AREA, INTERIM, RESULTS

STAGE2 = INTERIM / "stage2"
COVER = RESULTS / "study_area" / "zone_v6_cover_buffered.geojson"


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _tiles(tag: str, root: Path) -> dict:
    files = {f.stem.replace("tile_", ""): f for f in sorted((root / tag).glob("tile_*.npz"))}
    if not files:
        raise SystemExit(f"no stage-2 tiles in {root / tag}")
    return files


def block_regions(tag: str, block_m: float = design.BLOCK_SIZE_M) -> dict:
    """{block key: (transform, region mask)} of a block-sample export: the stage-2 cover
    within the block square (and within the record's sector, if any), rasterised by
    cell centre on the block tile's grid (export record dims = (columns, rows))."""
    import pyproj
    import shapely
    from affine import Affine
    from rasterio.features import geometry_mask
    from shapely.geometry import box, mapping
    from caspian.geography import SECTOR_ORDER, sectors
    fwd = pyproj.Transformer.from_crs("EPSG:4326", CRS_EQUAL_AREA, always_xy=True)

    def laea(g, densify=None):
        g = g.segmentize(densify) if densify else g
        return shapely.transform(g, lambda xy: np.column_stack(fwd.transform(xy[:, 0], xy[:, 1])))
    cover = R.load_cover(COVER, CRS_EQUAL_AREA)
    secs = {k + 1: laea(sectors()[name][1], 0.01) for k, name in enumerate(SECTOR_ORDER)}
    st = json.loads((RESULTS / "inventory" / f"stage2_{tag}.json").read_text(encoding="utf-8"))
    out = {}
    for key, t in st["tiles"].items():
        i, j = (int(v) for v in key.split("_"))
        part = cover.intersection(box(i * block_m, j * block_m, (i + 1) * block_m, (j + 1) * block_m))
        if t.get("sector_code"):
            part = part.intersection(secs[int(t["sector_code"])])
        cols, rows = t["dims"]
        tr = t["transform"]
        out[key] = (tr, geometry_mask([mapping(part)], out_shape=(rows, cols), transform=Affine(*tr[:6]),
                                      invert=True) if not part.is_empty else np.zeros((rows, cols), bool))
    return out


def bin_width_of(tag: str, root: Path | None = None) -> float:
    """Bin width (m) of a stage-2 tag: from its variant.json, else the PREREG width."""
    p = Path(STAGE2 if root is None else root) / tag / "variant.json"
    if p.exists():
        return float(json.loads(p.read_text(encoding="utf-8"))["bin_width_m"])
    return float(design.BIN_WIDTH_M)


def build(variant: str, sources: list[str], *, add: str | None = None, sub: str | None = None,
          substitute: str | None = None, vegetation: str | None = None, drop=(), keep_only=None, e1_end=None,
          e1_ends=None,
          fine_from_halves: bool = False, bin_factor: int = 1, root: Path | None = None) -> dict:
    """Write the variant tag (see the module docstring). Returns its record."""
    root = Path(root) if root is not None else STAGE2
    if (root / variant).exists() and any((root / variant).glob("tile_*.npz")):
        raise SystemExit(f"{root / variant} already holds tiles; choose a new variant name")
    widths = {bin_width_of(t, root) for t in sources}
    if len(widths) != 1:
        raise SystemExit(f"sources have different bin widths {sorted(widths)}")
    width = widths.pop()
    src = {t: _tiles(t, root) for t in sources}
    keys = sorted(set().union(*[set(v) for v in src.values()]))
    extra = {k: (_tiles(t, root) if t else {}) for k, t in (("add", add), ("sub", sub))}
    stray = sorted((set(extra["add"]) | set(extra["sub"])) - set(keys))
    if stray:
        raise SystemExit(f"delta tiles {stray} have no main tile")
    mappers = []
    if e1_end is not None:
        if not e1_ends:
            raise SystemExit("--e1-end needs --e1-ends (the ends the split plan was tagged with)")
        mappers.append(R.e1_end_mapper(sorted(e1_ends), e1_end))
    if drop or keep_only is not None:
        mappers.append(R.select_mapper(drop=drop, keep_only=keep_only, strip=[]))
    if fine_from_halves:
        mappers.append(R.halves_to_fine_mapper)
        width = width / 2
    if bin_factor > 1:
        mappers.append(R.rebin_mapper(bin_factor))
        width = width * bin_factor
    mappers.append(R.select_mapper())            # sum out the remaining class tokens
    mapper = R.compose(*mappers)
    subst = {}
    if substitute:
        regions = block_regions(substitute)
        subst = {k: (f, regions[k]) for k, f in _tiles(substitute, root).items()}
    out = root / variant
    out.mkdir(parents=True, exist_ok=True)
    rec = {"variant": variant, "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "sources": {t: {k: _sha(f) for k, f in v.items()} for t, v in src.items()},
           "add": add, "sub": sub, "substitute": substitute, "vegetation": vegetation, "drop": list(drop),
           "keep_only": None if keep_only is None else list(keep_only), "e1_end": None if e1_end is None else str(e1_end),
           "e1_ends": None if not e1_ends else [str(x) for x in e1_ends], "fine_from_halves": fine_from_halves,
           "bin_factor": bin_factor, "bin_width_m": round(width, 6), "bin_origin_m": design.BIN_ORIGIN_M, "tiles": {}}
    for key in keys:
        parts, tr = [], None
        for t, files in src.items():
            if key in files:
                t_tr, arr = R.load_tile(files[key])
                if tr is not None and not np.allclose(t_tr, tr):
                    raise SystemExit(f"tile {key}: transforms differ between sources")
                tr = t_tr
                parts.append(arr)
        arrays = R.sum_tiles(parts)
        d_add = R.load_tile(extra["add"][key])[1] if key in extra["add"] else None
        d_sub = R.load_tile(extra["sub"][key])[1] if key in extra["sub"] else None
        arrays = R.apply_delta(arrays, d_add, d_sub) if (d_add or d_sub) else {k: a.astype(np.uint16) for k, a in
                                                                               arrays.items()}
        n_sub = 0
        for bkey, (bf, (btr, reg)) in subst.items():
            arrays, n = R.substitute_window(arrays, tr, R.load_tile(bf)[1], btr, reg)
            n_sub += n
        if vegetation:
            arrays = R.vegetation_tile(arrays, vegetation)
        arrays = R.merge_groups(arrays, mapper)
        R.write_tile(out / f"tile_{key}.npz", tr, arrays)
        rec["tiles"][key] = {"n_bands": len(arrays), "cells_substituted": n_sub}
        print(f"{variant}: tile {key}, {len(arrays)} bands", flush=True)
    (out / "variant.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return rec


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("variant")
    ap.add_argument("--from", dest="sources", required=True)
    ap.add_argument("--add")
    ap.add_argument("--sub")
    ap.add_argument("--substitute")
    ap.add_argument("--vegetation", choices=["as_dry", "as_water"])
    ap.add_argument("--drop", default="")
    ap.add_argument("--keep-only", default=None)
    ap.add_argument("--e1-end", default=None)
    ap.add_argument("--e1-ends", default=None)
    ap.add_argument("--fine-from-halves", action="store_true")
    ap.add_argument("--bin-factor", type=int, default=1)
    a = ap.parse_args(argv)
    build(a.variant, a.sources.split(","), add=a.add, sub=a.sub, substitute=a.substitute, vegetation=a.vegetation,
          drop=[t for t in a.drop.split(",") if t], keep_only=a.keep_only.split(",") if a.keep_only else None,
          e1_end=a.e1_end, e1_ends=a.e1_ends.split(",") if a.e1_ends else None,
          fine_from_halves=a.fine_from_halves, bin_factor=a.bin_factor)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
