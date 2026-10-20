"""Phase 0: parse the DAHITI Caspian series, summarise it and plot it for QC.

Gaps are in fractional days. Extremes are given twice: the single raw records and
the extremes of a 3-point running median (caspian.levels.smoothed_extremes), which a
single noisy record cannot set (D-021 correction).

Inputs : the single series file in data/raw/dahiti/
Outputs: results/qc/dahiti_summary.json (committed)
         outputs/qc/dahiti_level_series.{pdf,png}
"""

from __future__ import annotations

import hashlib
import json
import sys

import caspian  # noqa: F401  (Windows DLL-path fix before rasterio-dependent imports)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from caspian import levels, plotting
from caspian.config import OUTPUTS, RAW, RESULTS
from caspian.io import find_dahiti_file, read_dahiti

SMOOTH_K = 3  # running-median length for the robust extremes (D-021 correction)


def main() -> int:
    path = find_dahiti_file(RAW / "dahiti")
    if path is None:
        print(f"FAIL no DAHITI series file in {RAW / 'dahiti'}")
        return 1
    series = read_dahiti(path)
    d = series.data
    years = (d["time"].iloc[-1] - d["time"].iloc[0]) / pd.Timedelta(days=1) / 365.25
    # Fractional days (``.dt.days`` would floor each gap to whole days). d is time-sorted.
    gaps = pd.Series(levels.gap_days(d["time"]))
    sm = levels.smoothed_extremes(d["time"], d["level_m"], k=SMOOTH_K)
    summary = {
        "file": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "layout": series.layout,
        "header_metadata": series.meta,
        "n_records": int(series.n),
        "first": d["time"].iloc[0].isoformat(),
        "last": d["time"].iloc[-1].isoformat(),
        "span_years": round(years, 2),
        "median_gap_days": round(float(np.median(gaps)), 3) if len(gaps) else None,
        "max_gap_days": round(float(gaps.max()), 3) if len(gaps) else None,
        "time_of_max_gap_start": d["time"].iloc[int(np.argmax(gaps.to_numpy()))].isoformat() if len(gaps) else None,
        "n_gaps_over_60_days": int((gaps > 60).sum()),
        # single-record extremes (raw records)
        "level_min_m": float(d["level_m"].min()),
        "level_max_m": float(d["level_m"].max()),
        "time_of_min": d.loc[d["level_m"].idxmin(), "time"].isoformat(),
        "time_of_max": d.loc[d["level_m"].idxmax(), "time"].isoformat(),
        # extremes of the k-point running median (D-021 correction): robust to one noisy record
        "smoothed": {"rule": f"{SMOOTH_K}-point running median of consecutive records",
                     "k": sm["k"], "level_max_m": round(sm["max_m"], 4),
                     "time_of_max": sm["time_of_max"].isoformat(), "level_min_m": round(sm["min_m"], 4),
                     "time_of_min": sm["time_of_min"].isoformat(), "range_m": round(sm["range_m"], 4)},
        "sigma_median_m": float(d["sigma_m"].median()) if d["sigma_m"].notna().any() else None,
        "sigma_max_m": float(d["sigma_m"].max()) if d["sigma_m"].notna().any() else None,
        "records_per_year": {int(k): int(v) for k, v in d.groupby(d["time"].dt.year).size().items()},
        "extra_columns": [c for c in d.columns if c not in ("time", "level_m", "sigma_m")],
    }
    out_json = RESULTS / "qc" / "dahiti_summary.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plotting.use_style()
    fig, ax = plt.subplots(figsize=(plotting.WIDTH_DOUBLE_MM * plotting.MM, 55 * plotting.MM))
    if d["sigma_m"].notna().any():
        ax.fill_between(d["time"], d["level_m"] - d["sigma_m"], d["level_m"] + d["sigma_m"],
                        color="#9fb7d6", lw=0, alpha=0.6, label="±1σ (DAHITI error)")
    ax.plot(d["time"], d["level_m"], color="#1f4e8c", lw=0.7, label="DAHITI water level")
    ax.set_ylabel("Water level (m, DAHITI height reference)")
    ax.set_xlabel("Year")
    ax.grid(axis="y")
    ax.set_title(f"DAHITI Caspian Sea series - QC ({series.n} records, {summary['first'][:10]} "
                 f"to {summary['last'][:10]})", loc="left")
    if d["sigma_m"].notna().any():
        ax.legend(loc="lower left")
    plotting.save(fig, OUTPUTS / "qc" / "dahiti_level_series", tiff=False)
    plt.close(fig)
    print(json.dumps({k: v for k, v in summary.items() if k != "header_metadata"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
