"""Emergence-date map on the DEM tiling (PREREG section 2), for the
dryness test of the ICESat-2 validation (scripts/a6_validate_dem.py --emergence).

The basin level is evaluated daily over the whole levelled record with the scene-
level code path (scripts/a4_scene_levels.levels_at: DAHITI, Hydroweb shifted onto the
DAHITI reference in gaps); days without a level are dropped. Each DEM pixel's
emergence date is caspian.dem.emergence_days of its height: the first day after the
last day on which the level was at or above the pixel (NaN where the pixel was never
sea in the record or has not emerged).

Outputs: data/processed/dem/<tag>/emergence/emergence_<i>_<j>.tif (float32, days since
1970-01-01, NaN as above) next to each dem_<i>_<j>.tif; the daily series used,
data/processed/dem/<tag>/emergence/level_series_daily.csv.

Usage: python scripts/a6_emergence.py <tag>
"""

from __future__ import annotations

import importlib.util
import sys

import caspian  # noqa: F401
import numpy as np
import pandas as pd
import rasterio

from caspian import dem
from caspian.config import DATA, RESULTS


def daily_levels() -> pd.DataFrame:
    spec = importlib.util.spec_from_file_location("a4", RESULTS.parent / "scripts" / "a4_scene_levels.py")
    a4 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(a4)
    prep = a4.prepare()
    t0 = min(prep["d"]["time"].min(), prep["hw"]["time"].min())
    t1 = max(prep["d"]["time"].max(), prep["hw"]["time"].max())
    days = pd.date_range(pd.Timestamp(t0).normalize(), pd.Timestamp(t1).normalize(), freq="D")
    lv, _ = a4.levels_at(prep, days)
    out = pd.DataFrame({"date": days, "level_m": lv["level_m"].to_numpy(), "source": lv["level_source"].to_numpy()})
    return out[np.isfinite(out["level_m"])].reset_index(drop=True)


def main(argv) -> int:
    tag = argv[0]
    src = DATA / "processed" / "dem" / tag
    out = src / "emergence"
    out.mkdir(parents=True, exist_ok=True)
    s = daily_levels()
    s.to_csv(out / "level_series_daily.csv", index=False, float_format="%.4f")
    t = (s["date"] - pd.Timestamp("1970-01-01")).dt.days.to_numpy(float)
    lv = s["level_m"].to_numpy(float)
    print(f"daily levels {s['date'].min().date()} .. {s['date'].max().date()}: {len(s)} days, "
          f"{lv.min():.3f} .. {lv.max():.3f} m", flush=True)
    n = 0
    for f in sorted(src.glob("dem_*.tif")):
        with rasterio.open(f) as r:
            z = r.read(1).astype(float)
            prof = r.profile
        e = dem.emergence_days(z, t, lv).astype(np.float32)
        prof.update(count=1, dtype="float32", nodata=np.nan)
        with rasterio.open(out / f.name.replace("dem_", "emergence_"), "w", **prof) as w:
            w.write(e, 1)
            w.set_band_description(1, "emergence_days_since_1970")
        n += 1
    print("emergence tiles written:", n, "->", out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
