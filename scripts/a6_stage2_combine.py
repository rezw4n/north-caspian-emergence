"""Sum stage-2 counts of several exports into one tag.

The main export runs per sensor family (tags mainL for Landsat, mainS for
Sentinel-2) from one plan, whose level bins and acquisition parities were assigned
over all sensors together (scripts/a6_stage2_plan.py). Counts of the same tile and
band are summed, never averaged (caspian.frequency); a band missing from a source
counts as zero there. Bands are read one at a time to keep memory low.

A tile is written only when every source tag has it (``--allow-missing`` writes the
tiles of the first tag alone, for engineering checks only; such a combination is
recorded as partial in the provenance file).

Outputs: data/interim/stage2/<out>/tile_<key>.npz (uint16; same transform);
results/provenance/stage2_<out>_combine.json (sources, tiles, partial flag).

Usage: python scripts/a6_stage2_combine.py <out> <tag> [<tag> ...] [--allow-missing]
"""

from __future__ import annotations

import json
import sys

import caspian  # noqa: F401
import numpy as np

from caspian.config import INTERIM, RESULTS


def combine_tile(paths, out_path) -> int:
    zs = [np.load(p) for p in paths]
    tr = zs[0]["transform"]
    for z in zs[1:]:
        if not np.allclose(z["transform"], tr):
            raise ValueError(f"transform mismatch for {out_path.name}")
    bands = sorted({k for z in zs for k in z.files if k != "transform"})
    out = {}
    for b in bands:
        acc = None
        for z in zs:
            if b in z.files:
                a = z[b].astype(np.uint32)
                acc = a if acc is None else acc + a
        if acc.max() > np.iinfo(np.uint16).max:
            raise ValueError(f"{b}: count exceeds uint16")
        out[b] = acc.astype(np.uint16)
    np.savez_compressed(out_path, transform=tr, **out)
    return len(bands)


def main(argv) -> int:
    allow = "--allow-missing" in argv
    args = [a for a in argv if a != "--allow-missing"]
    out_tag, tags = args[0], args[1:]
    src = [INTERIM / "stage2" / t for t in tags]
    keys = [sorted(p.name for p in d.glob("tile_*.npz")) for d in src]
    common = sorted(set(keys[0]).intersection(*keys[1:])) if not allow else keys[0]
    out = INTERIM / "stage2" / out_tag
    out.mkdir(parents=True, exist_ok=True)
    rec = {"out": out_tag, "sources": tags, "partial": allow, "tiles": {}}
    for name in common:
        paths = [d / name for d in src if (d / name).exists()]
        n = combine_tile(paths, out / name)
        rec["tiles"][name] = {"sources": [p.parent.name for p in paths], "bands": n}
        print(name, "from", [p.parent.name for p in paths], n, "bands", flush=True)
    missing = sorted(set().union(*map(set, keys)) - set(common))
    rec["tiles_not_combined"] = missing
    (RESULTS / "provenance").mkdir(parents=True, exist_ok=True)
    (RESULTS / "provenance" / f"stage2_{out_tag}_combine.json").write_text(json.dumps(rec, indent=1),
                                                                           encoding="utf-8")
    print(f"combined {len(common)} tiles; not combined: {len(missing)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
