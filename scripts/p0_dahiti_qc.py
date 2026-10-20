"""Phase 0: parse the DAHITI Caspian series, summarise it and plot it for QC.

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

from caspian import plotting
from caspian.config import OUTPUTS, RAW, RESULTS
from caspian.io import find_dahiti_file, read_dahiti


def main() -> int:
    path = find_dahiti_file(RAW / "dahiti")
    if path is None:
        print(f"FAIL no DAHITI series file in {RAW / 'dahiti'}")
        return 1
    series = read_dahiti(path)
    d = series.data
    years = (d["time"].iloc[-1] - d["time"].iloc[0]).days / 365.25
    gaps = d["time"].diff().dt.days.dropna()
    summary = {
        "file": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "layout": series.layout,
        "header_metadata": series.meta,
        "n_records": int(series.n),
        "first": d["time"].iloc[0].isoformat(),
        "last": d["time"].iloc[-1].isoformat(),
        "span_years": round(years, 2),
        "median_gap_days": float(np.median(gaps)) if len(gaps) else None,
        "max_gap_days": float(gaps.max()) if len(gaps) else None,
        "level_min_m": float(d["level_m"].min()),
        "level_max_m": float(d["level_m"].max()),
        "time_of_min": d.loc[d["level_m"].idxmin(), "time"].isoformat(),
        "time_of_max": d.loc[d["level_m"].idxmax(), "time"].isoformat(),
        "sigma_median_m": float(d["sigma_m"].median()) if d["sigma_m"].notna().any() else None,
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
