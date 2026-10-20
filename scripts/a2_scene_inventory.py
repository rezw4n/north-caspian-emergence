"""A2: inventory of every Landsat C2 L2 (Tier 1) and Sentinel-2 L2A scene that
intersects the study domain, 1984-2026, from scene metadata only (no pixels).

Outputs
  data/interim/inventory/scenes.csv                    one row per scene (git-ignored)
  results/inventory/summary_sensor_year.csv            committed
  results/inventory/summary_sensor_month.csv           committed
  results/inventory/inventory_meta.json                committed

Cloud cover here is the scene-level property (whole scene), not the fraction
over the domain; the domain-level clear fraction comes from the A2 pilot.
"""

from __future__ import annotations

import json
import sys
import time

import caspian  # noqa: F401
import ee
import pandas as pd
from shapely.geometry import mapping

from caspian.config import EE_PROJECT, INTERIM, RESULTS
from caspian.geography import DOMAIN
from caspian.masks import LANDSAT, S2

START_YEAR, END_YEAR = 1984, 2026
END_DATE = "2026-10-21"


def fetch(cid: str, region, year: int, props: list[str]) -> pd.DataFrame:
    col = ee.ImageCollection(cid).filterBounds(region).filterDate(f"{year}-01-01",
                                                                   min(f"{year + 1}-01-01", END_DATE))
    lists = ee.Dictionary({p: col.aggregate_array(p) for p in props + ["system:index"]}).getInfo()
    df = pd.DataFrame(lists)
    return df


def main() -> int:
    ee.Initialize(project=EE_PROJECT)
    region = ee.Geometry(mapping(DOMAIN))
    frames = []
    jobs = [(k, cid, ["system:time_start", "CLOUD_COVER", "WRS_PATH", "WRS_ROW"])
            for k, (cid, _, _) in LANDSAT.items()]
    jobs.append(("MSI_S2", S2[0], ["system:time_start", "CLOUDY_PIXEL_PERCENTAGE", "MGRS_TILE"]))
    for sensor, cid, props in jobs:
        for year in range(START_YEAR, END_YEAR + 1):
            for attempt in range(4):
                try:
                    df = fetch(cid, region, year, props)
                    break
                except Exception as exc:  # transient EE errors
                    if attempt == 3:
                        raise
                    time.sleep(10 * (attempt + 1))
            if df.empty:
                continue
            df["sensor"] = sensor
            df["collection"] = cid
            frames.append(df)
            print(f"{sensor} {year}: {len(df)}", flush=True)
    scenes = pd.concat(frames, ignore_index=True)
    scenes["time"] = pd.to_datetime(scenes["system:time_start"], unit="ms", utc=True).dt.tz_convert(None)
    scenes["cloud_scene_pct"] = scenes["CLOUD_COVER"].fillna(scenes.get("CLOUDY_PIXEL_PERCENTAGE"))
    scenes["year"] = scenes["time"].dt.year
    scenes["month"] = scenes["time"].dt.month
    keep = ["sensor", "collection", "system:index", "time", "year", "month", "cloud_scene_pct",
            "WRS_PATH", "WRS_ROW", "MGRS_TILE"]
    scenes = scenes[[c for c in keep if c in scenes.columns]].sort_values(["sensor", "time"])
    (INTERIM / "inventory").mkdir(parents=True, exist_ok=True)
    scenes.to_csv(INTERIM / "inventory" / "scenes.csv", index=False)

    out = RESULTS / "inventory"
    out.mkdir(parents=True, exist_ok=True)
    bins = [-1, 20, 40, 60, 80, 100.1]
    labels = ["cc0_20", "cc20_40", "cc40_60", "cc60_80", "cc80_100"]
    scenes["cc_class"] = pd.cut(scenes["cloud_scene_pct"], bins=bins, labels=labels)
    by_year = scenes.pivot_table(index=["sensor", "year"], columns="cc_class", values="time",
                                 aggfunc="count", fill_value=0, observed=False)
    by_year["total"] = by_year.sum(axis=1)
    by_year.to_csv(out / "summary_sensor_year.csv")
    by_month = scenes.pivot_table(index=["sensor", "month"], columns="cc_class", values="time",
                                  aggfunc="count", fill_value=0, observed=False)
    by_month["total"] = by_month.sum(axis=1)
    by_month.to_csv(out / "summary_sensor_month.csv")
    meta = {
        "generated": pd.Timestamp.now(tz="UTC").isoformat(),
        "end_date_exclusive": END_DATE,
        "n_scenes_total": int(len(scenes)),
        "n_by_sensor": scenes["sensor"].value_counts().sort_index().to_dict(),
        "n_cloud_below_60_by_sensor": scenes[scenes["cloud_scene_pct"] < 60]["sensor"]
        .value_counts().sort_index().to_dict(),
        "first_by_sensor": scenes.groupby("sensor")["time"].min().astype(str).to_dict(),
        "last_by_sensor": scenes.groupby("sensor")["time"].max().astype(str).to_dict(),
        "landsat_path_rows": sorted({f"{int(p):03d}/{int(r):03d}" for p, r in
                                     scenes[["WRS_PATH", "WRS_ROW"]].dropna().itertuples(index=False)}),
        "n_s2_tiles": int(scenes["MGRS_TILE"].nunique()) if "MGRS_TILE" in scenes else 0,
    }
    (out / "inventory_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in meta.items() if k != "landsat_path_rows"}, indent=1))
    print("path/rows:", len(meta["landsat_path_rows"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
