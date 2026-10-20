"""A5/A6: choose the surge-screening threshold tau (PREREG section 3; D-064).

Input: the screening-sensitivity export (scripts/a6_stage2_export.py, tag `sens`):
counts per (epoch, bin, parity, setup class) over a stratified sample of training
blocks (Landsat). For each candidate tau, the classes whose upper edge is at most tau
are summed. For every block and (epoch, bin), the water extents of the odd and even
halves (frequency >= 0.5, at least design.MIN_VALID_OBS valid) are compared with
caspian.hindcast.noise_floor_displacement. The within-bin contour scatter S(tau) is
the median, over block-bins with waterlines in both halves, of the median
displacement.

Bootstrap SE: design.BOOTSTRAP_N resamples of the sampled blocks. Choice: from no
screening towards stricter tau, stop at the first tau where the next stricter step
reduces S by less than its bootstrap SE.

Output (committed): results/levels/screening_sensitivity.json

Usage: python scripts/a6_screening.py <tag>
"""

from __future__ import annotations

import json
import re
import sys

import caspian  # noqa: F401
import numpy as np

from caspian import design, frequency, hindcast
from caspian.config import INTERIM, RESULTS

BAND = re.compile(r"([wv])_(E[123])_b(\d{3})_p([01])_c(\d)$")
EDGES = (0.05, 0.10, 0.15, 0.20, 0.30)          # class k covers [EDGES[k-1], EDGES[k]); last class >= 0.30
TAUS = (np.inf, 0.30, 0.20, 0.15, 0.10, 0.05)   # loosest to strictest


def classes_for(tau: float) -> list[int]:
    """Setup classes whose members all have |setup| <= tau."""
    if not np.isfinite(tau):
        return list(range(len(EDGES) + 1))
    return [k for k in range(len(EDGES) + 1) if (EDGES[k] if k < len(EDGES) else np.inf) <= tau + 1e-9]


def block_scatter(npz, tau: float, pixel: float) -> list[float]:
    """Median odd-even displacement for each (epoch, bin) of one block at tau."""
    z = np.load(npz)
    g = {}
    for k in z.files:
        m = BAND.match(k)
        if m:
            kind, ep, b, p, c = m.groups()
            g.setdefault((ep, int(b), int(p)), {}).setdefault(int(c), {})[kind] = z[k]
    keep = set(classes_for(tau))
    out = []
    for ep, b in sorted({(e, b) for e, b, _ in g}):
        halves = []
        for p in (0, 1):
            cl = {c: (h["w"], h["v"]) for c, h in g.get((ep, b, p), {}).items() if c in keep and "w" in h}
            if not cl:
                break
            w, v = frequency.combine(cl, list(cl))
            f = frequency.water_frequency(w, v)
            halves.append((f >= design.CONTOUR_LEVEL, np.isfinite(f)))
        if len(halves) < 2:
            continue
        (wa, va), (wb, vb) = halves
        r = hindcast.noise_floor_displacement(wa, va, wb, vb, pixel)
        if np.isfinite(r.get("median", np.nan)) and r.get("n", 0) > 0:
            out.append(float(r["median"]))
    return out


def main(argv) -> int:
    tag = argv[0]
    files = sorted((INTERIM / "stage2" / tag).glob("tile_*.npz"))
    pixel = abs(float(np.load(files[0])["transform"][0]))
    per_block = {tau: [block_scatter(f, tau, pixel) for f in files] for tau in TAUS}
    rng = np.random.default_rng(design.SEED)
    res = {}
    for tau in TAUS:
        vals = per_block[tau]
        s = float(np.median(np.concatenate([v for v in vals if v]))) if any(vals) else np.nan
        boot = []
        for _ in range(design.BOOTSTRAP_N):
            pick = rng.integers(0, len(vals), len(vals))
            pool = np.concatenate([vals[i] for i in pick if vals[i]]) if any(vals[i] for i in pick) else np.array([])
            boot.append(np.median(pool) if pool.size else np.nan)
        res[str(tau)] = {"S_m": s, "se_m": float(np.nanstd(boot)), "n_block_bins": int(sum(len(v) for v in vals))}
    chosen = None
    for a, b in zip(TAUS[:-1], TAUS[1:]):
        if res[str(a)]["S_m"] - res[str(b)]["S_m"] < res[str(b)]["se_m"]:
            chosen = a
            break
    if chosen is None:
        chosen = TAUS[-1]
    out = {"tag": tag, "n_blocks": len(files), "taus": [str(t) for t in TAUS], "results": res,
           "chosen_tau_m": None if not np.isfinite(chosen) else chosen, "chosen": str(chosen)}
    (RESULTS / "levels").mkdir(parents=True, exist_ok=True)
    (RESULTS / "levels" / "screening_sensitivity.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
