"""A6: level-binned waterlines from stage-2 counts (PREREG section 4).

For each fetched stage-2 tile (scripts/a6_stage2_export.py fetch) and each
(epoch, level bin):
- the two acquisition halves are summed (caspian.frequency.combine);
- water frequency is computed where at least design.MIN_VALID_OBS acquisitions are
  valid;
- the design.CONTOUR_LEVEL contour is traced at sub-pixel precision
  (caspian.waterlines, marching squares);
- the contour is sampled at the grid spacing.

Each point takes the bin's centre level as its height. The per-bin level statistics
of the scenes behind it (from the stage-2 plan) are kept for the contour error
budget.

Output: data/interim/waterlines/<tag>/points.csv.gz with x, y (equal-area metres),
z_m, epoch, bin, tile, n_scenes_bin, level_sigma_p50_m (bin median of the scenes'
level 1-sigma), and rho_level: the Spearman correlation, at the point's pixel, between
water frequency and bin level over the epoch's valid bins. Sea-controlled shorelines
get wetter as the level rises (rho near 1); river and pond banks do not (rho near 0).
The static-contour rule of D-034 is a threshold on rho_level, fixed on training blocks.

Usage: python scripts/a6_waterlines.py <tag> <plan.csv>
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import caspian  # noqa: F401
import numpy as np
import pandas as pd

from caspian import design, frequency, levels, waterlines
from caspian.config import INTERIM

BAND = re.compile(r"([wv])_(E[123])_b(\d{3})_p([01])$")


def _rho(stack) -> np.ndarray:
    """Spearman correlation between frequency and bin index, per pixel, over the bins
    where the pixel has a frequency (at least 4 bins, else NaN)."""
    bins = np.array([b for b, _ in stack], float)
    F = np.stack([f for _, f in stack])                       # (n_bins, rows, cols)
    ok = np.isfinite(F)
    # ranks of frequency along the bin axis (ties averaged), restricted to valid bins
    rk_f = np.where(ok, pd.DataFrame(F.reshape(len(bins), -1)).rank(axis=0).to_numpy().reshape(F.shape), np.nan)
    rk_b = np.where(ok, np.broadcast_to(bins[:, None, None], F.shape), np.nan)
    rk_b = np.where(ok, pd.DataFrame(rk_b.reshape(len(bins), -1)).rank(axis=0).to_numpy().reshape(F.shape), np.nan)
    n = ok.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mf, mb = np.nanmean(rk_f, 0), np.nanmean(rk_b, 0)
        cov = np.nansum((rk_f - mf) * (rk_b - mb), 0)
        r = cov / np.sqrt(np.nansum((rk_f - mf) ** 2, 0) * np.nansum((rk_b - mb) ** 2, 0))
    r[n < 4] = np.nan
    return r


def tile_points(npz_path: Path, spacing: float) -> pd.DataFrame:
    z = np.load(npz_path)
    transform = z["transform"]
    groups = {}
    for k in z.files:
        m = BAND.match(k)
        if m:
            kind, ep, b, p = m.groups()
            groups.setdefault((ep, int(b)), {}).setdefault(int(p), {})[kind] = z[k]
    rows = []
    # per-pixel frequency stack per epoch, for the rho_level statistic
    stacks = {}
    for (ep, b), halves in sorted(groups.items()):
        counts = {p: (h["w"], h["v"]) for p, h in halves.items() if "w" in h and "v" in h}
        if counts:
            w, v = frequency.combine(counts, list(counts))
            stacks.setdefault(ep, []).append((b, frequency.water_frequency(w, v)))
    rho = {ep: _rho(st) for ep, st in stacks.items()}
    a_, b_, c_, d_, e_, f_ = transform[:6]
    for (ep, b), halves in sorted(groups.items()):
        counts = {p: (h["w"], h["v"]) for p, h in halves.items() if "w" in h and "v" in h}
        if not counts:
            continue
        w, v = frequency.combine(counts, list(counts))
        f = frequency.water_frequency(w, v)
        valid = v >= design.MIN_VALID_OBS
        if valid.sum() < 4 or np.nanmin(f) > design.CONTOUR_LEVEL or np.nanmax(f) < design.CONTOUR_LEVEL:
            continue
        lines = waterlines.extract(f, transform, design.CONTOUR_LEVEL, valid=valid)
        pts = waterlines.contour_points(lines, spacing)
        if len(pts):
            col = np.clip(((pts[:, 0] - c_) / a_).astype(int), 0, f.shape[1] - 1)
            row = np.clip(((pts[:, 1] - f_) / e_).astype(int), 0, f.shape[0] - 1)
            rows.append(pd.DataFrame({"x": pts[:, 0], "y": pts[:, 1], "epoch": ep, "bin": b,
                                      "rho_level": rho[ep][row, col]}))
    if not rows:
        return pd.DataFrame(columns=["x", "y", "epoch", "bin"])
    out = pd.concat(rows, ignore_index=True)
    out["tile"] = npz_path.stem.replace("tile_", "")
    return out


def main(argv) -> int:
    tag, plan_csv = argv[0], argv[1]
    src = INTERIM / "stage2" / tag
    plan = pd.read_csv(plan_csv, usecols=["scene", "epoch", "bin", "sigma_m"]).drop_duplicates("scene")
    stats = plan.groupby(["epoch", "bin"]).agg(n_scenes_bin=("scene", "size"), level_sigma_p50_m=("sigma_m", "median"))
    pts = []
    for f in sorted(src.glob("tile_*.npz")):
        t = np.load(f)["transform"]
        pts.append(tile_points(f, spacing=abs(float(t[0]))))
        print(f.name, len(pts[-1]), "points", flush=True)
    d = pd.concat(pts, ignore_index=True)
    d["z_m"] = levels.bin_center(d["bin"].to_numpy(), design.BIN_WIDTH_M, design.BIN_ORIGIN_M)
    d = d.merge(stats, left_on=["epoch", "bin"], right_index=True, how="left")
    out = INTERIM / "waterlines" / tag
    out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out / "points.csv.gz", index=False, float_format="%.6f")
    print(out / "points.csv.gz", len(d), "points |", d.groupby("epoch")["bin"].nunique().to_dict(), "bins per epoch")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
