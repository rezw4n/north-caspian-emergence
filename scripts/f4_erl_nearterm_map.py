"""Display rasters for the revised ERL figure 5 (where the first metre emerges).

For every 60 m cell of the study domain, the number of the five grids that place it in the
seabed that emerges in the first metre of fall below the 2026 level L26, on the two
surfaces that the near-term emergence revision analysis headlines (scripts/rev_n_curves.py):
  (a) composite observed baseline (data/interim/revision_n/
      composite_baseline.npy): the baseline sea (Fb < L26) that grid p puts above the
      water at L26 - 1 m, i.e. Fb < L26 and not F_p < L26 - 1 (rev_n's "common_location");
  (b) shifted grids: L26 + c_p - 1 <= F_p < L26 + c_p, with c_p the H1 shape constant
      (rev_n's "shifted").
F_p is the grid's flood-level map from the hindcast cache, read through
scripts/rev_n_common.setup, so the inputs are rev_n's. Each grid's band area must equal rev_n's
domain E(1 m) (curves_composite.csv and curves.csv) to within 0.1% or the run stops. The
outline of the composite baseline (the observed 2025-26 sea) is kept as the water share
per 240 m block. Nothing here is a new result.

Outputs
  data/interim/figures/erl/nearterm_map.npz   per surface, the most frequent count per 240 m
                                              block (255 outside the domain), and the
                                              baseline's water share per block (git-ignored)
  results/display/erl_nearterm_map.json       areas by count, any and majority (3 of 5 or
                                              more), the per-grid checks
Usage: python scripts/f4_erl_nearterm_map.py [--tile 2048]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rev_n_common as N  # noqa: E402

from caspian import hindcast_tiled as ht  # noqa: E402
from caspian.config import INTERIM, RESULTS  # noqa: E402

DELTA = 1.0
FACTOR = 4                       # 4 x 60 m = 240 m display blocks
CACHE = INTERIM / "figures" / "erl"
OUT = RESULTS / "display" / "erl_nearterm_map.json"
TOL = 0.001                      # relative tolerance of the per-grid check against rev_n's curves


def mode_blocks(count: np.ndarray, inside: np.ndarray, n: int) -> np.ndarray:
    """Per FACTOR x FACTOR block, the most frequent count among its domain cells (ties to the
    higher count), 255 where the block holds no domain cell (as f1_erl_display.agreement)."""
    r_, c_ = (count.shape[0] // FACTOR) * FACTOR, (count.shape[1] // FACTOR) * FACTOR
    best = np.full((r_ // FACTOR, c_ // FACTOR), 255, np.uint8)
    top = np.zeros(best.shape, np.int32)
    step = 256 * FACTOR
    for k in range(n + 1):
        s = np.zeros(best.shape, np.int32)
        for r0 in range(0, r_, step):
            r1 = min(r0 + step, r_)
            blk = (count[r0:r1, :c_] == k) & inside[r0:r1, :c_]
            s[r0 // FACTOR:r1 // FACTOR] = blk.reshape(blk.shape[0] // FACTOR, FACTOR, c_ // FACTOR,
                                                       FACTOR).sum(axis=(1, 3))
        take = (s > 0) & (s >= top)
        best[take], top[take] = k, s[take]
    return best


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tile", type=int, default=2048)
    a = ap.parse_args(argv)
    t0 = time.time()
    S = N.setup(tile=a.tile, fused=True, fused_form="composite")
    f1, shape, L26, cell = S["f1"], S["shape"], S["L26"], S["a7"].CELL_KM2
    inside = S["lab_in"] > 0
    names = S["names"]
    Fb = S["Ffused"][names[0]]           # the same composite map for every grid
    ref_a = pd.read_csv(RESULTS / "revision" / "n" / "curves_composite.csv")
    ref_b = pd.read_csv(RESULTS / "revision" / "n" / "curves.csv")

    def ref(df, surf, p):
        r = df[(df["surface"] == surf) & (df["sensitivity"] == "main") & (df["product"] == p)
               & (df["region"] == "domain") & np.isclose(df["delta_m"], DELTA)]
        return float(r["E_km2"].iloc[0])

    counts = {"composite": np.zeros(shape, np.uint8), "shifted": np.zeros(shape, np.uint8)}
    base_state = np.full(shape, -1, np.int8)
    rec = {"level_2026_m": L26, "delta_m": DELTA, "products": names, "shape_constants_m": S["c"],
           "composite_baseline": "data/interim/revision_n/composite_baseline.npy", "checks": [], "surfaces": {}}
    for r0, r1, c0, c1 in ht.tiles(shape, a.tile):
        b = np.asarray(Fb[r0:r1, c0:c1], np.float64) < L26
        dm = inside[r0:r1, c0:c1]
        st = base_state[r0:r1, c0:c1]
        st[dm & b] = 1
        st[dm & ~b] = 0
    for p in names:
        F, c = S["F"][p], S["c"][p]
        n_a = n_b = 0
        for r0, r1, c0, c1 in ht.tiles(shape, a.tile):
            f = np.asarray(F[r0:r1, c0:c1], np.float64)
            dm = inside[r0:r1, c0:c1]
            ma = dm & (base_state[r0:r1, c0:c1] == 1) & ~(f < L26 - DELTA)
            mb = dm & (f < L26 + c) & ~(f < L26 + c - DELTA)
            counts["composite"][r0:r1, c0:c1] += ma.astype(np.uint8)
            counts["shifted"][r0:r1, c0:c1] += mb.astype(np.uint8)
            n_a += int(ma.sum())
            n_b += int(mb.sum())
        for surf, n, want in (("composite", n_a, ref(ref_a, "common_location", p)),
                              ("shifted", n_b, ref(ref_b, "shifted", p))):
            got = n * cell
            ok = abs(got - want) <= TOL * max(abs(want), 1.0)
            rec["checks"].append({"surface": surf, "product": p, "km2": round(got, 3), "curves_km2": round(want, 3),
                                  "rel_diff": round((got - want) / want, 6) if want else None, "equal": bool(ok)})
            if not ok:
                raise SystemExit(f"{surf} {p}: {got:.1f} km2 differs from rev_n's curves {want:.1f} km2")
        N.log(p, "done")
    out = {}
    for surf, cnt in counts.items():
        hist = np.bincount(cnt[inside], minlength=len(names) + 1) * cell
        rec["surfaces"][surf] = {"area_km2_by_count": {str(k): float(v) for k, v in enumerate(hist)},
                                 "area_km2_any": float(hist[1:].sum()),
                                 "area_km2_majority": float(hist[len(names) // 2 + 1:].sum()),
                                 "area_km2_all": float(hist[len(names)])}
        out[f"count_{surf}"] = mode_blocks(cnt, inside, len(names))
    out["baseline"] = f1.block_fraction(base_state, FACTOR)
    tr = S["transform"]
    ta, _, tc, _, te, tf = (float(v) for v in tr[:6])
    r_, c_ = (shape[0] // FACTOR) * FACTOR, (shape[1] // FACTOR) * FACTOR
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE / "nearterm_map.npz", extent=np.array([tc, tc + ta * c_, tf + te * r_, tf]), **out)
    rec["seconds"] = round(time.time() - t0, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    N.log("done", rec["surfaces"])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
