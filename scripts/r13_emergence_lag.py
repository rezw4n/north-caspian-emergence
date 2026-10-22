"""Robustness: the ICESat-2 dryness test with emergence + 365 days instead of
+ 30 days (PREREG section 2). No Earth Engine.

A point that passes the 365-day test passes the 30-day test, so the variant's points
are a subset of the main run's prepared points: the main table is re-tested with
validation.dry_at_acquisition(lag_days=365) on the main emergence tiles (the same
neighbourhood radius per source, validation.DRY_RADIUS_PX; 60 km tiles), and the
headline is recomputed on the points that remain (scripts/r00_pipeline.py
compare_validation). The DEM is unchanged.

Usage:
  python scripts/r13_emergence_lag.py --main <tag> [--lag-days 365] [--sources ATL06,ATL08]
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import caspian  # noqa: F401
import numpy as np

from caspian import robustness as R, validation
from caspian.config import PROCESSED

CHECK = "r13_emergence_lag"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def stricter(points, em: validation.TileSet, source: str, lag_days: float, chunk: int = 250_000):
    """The points that pass the dryness test with ``lag_days``."""
    r = validation.DRY_RADIUS_PX[source]
    x, y, t = points["x"].to_numpy(float), points["y"].to_numpy(float), points["time"].to_numpy()
    keep = np.zeros(len(points), bool)
    for k in range(0, len(points), chunk):
        s = slice(k, k + chunk)
        w = em.sample_window(x[s], y[s], 1, r).reshape(len(x[s]), -1)
        keep[s] = validation.dry_at_acquisition(t[s], w, lag_days=lag_days)
    return points.loc[keep].reset_index(drop=True)


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lag-days", type=float, default=365.0)
    P = _load("r00_pipeline")
    P.add_common(ap)
    a = ap.parse_args(argv)
    em = validation.TileSet(PROCESSED / "dem" / a.main / "emergence", "emergence", tile_m=R.DEM_TILE_M)
    P.compare_validation(CHECK, f"lag{int(a.lag_days)}d", a.main, sources=a.sources.split(","),
                         alt_points=lambda p, source: stricter(p, em, source, a.lag_days), n_boot=a.n_boot,
                         note=f"dryness test with emergence + {a.lag_days:g} days")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
