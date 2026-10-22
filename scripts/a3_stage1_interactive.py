"""Water classification stage 1 run as profiled interactive requests.

Batch table exports report almost no EECU (0.01 EECU-s for 20 scenes), while the
Earth Engine profiler measures about 1-5 EECU-s per scene for the same computation.
The budget guard cannot see costs that are not reported, so stage 1 is computed in
batches of interactive requests instead, each wrapped in ee.profilePrinting. Every
batch's EECU-seconds are parsed from the profile, logged and summed, and the run
stops at a hard budget.

Scenes: every levelled scene (epochs E1-E3) with scene cloud cover below
MAX_CLOUD, from data/interim/levels/scene_levels.csv.
  run 1 : all months, no histogram column (data/interim/stage1/, log stage1_interactive_log.jsonl)
  run 2 : snow/ice diagnostics cloud-masked, NDSI test in the pixel mask only in
          November-April, histogram kept, all months, scenes that touch the zone in run 1
          (data/interim/stage1_v2/, log stage1_v2_interactive_log.jsonl).
Output (git-ignored): data/interim/stage1[_v2]/<sensor>.csv, one row per scene.

Usage: python scripts/a3_stage1_interactive.py <budget_eecu_s> [sensor ...] [--batch N] [--workers K]
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import caspian  # noqa: F401
import ee
import pandas as pd

from caspian import masks, pipeline_ee
from caspian.config import EE_PROJECT, INTERIM, RESULTS, STAGE1

ZONE = f"projects/{EE_PROJECT}/assets/caspian/active_zone_v6"
COVER = RESULTS / "study_area" / "zone_v6_tiles.geojson"
MAX_CLOUD = 60
SENSORS = ["TM_L5", "ETM_L7", "OLI_L8", "OLI_L9", "MSI_S2"]
FIELDS = ["scene", "sensor", "acq", "time", "otsu_threshold", "otsu_n", "zone_px", "zone_clear",
          "zone_snow", "zone_snow_ndsi", "zone_snow_qa", "cloud_scene", "hist"]
RUN1 = INTERIM / "stage1"
OUT = STAGE1
LOG = RESULTS / "inventory" / "stage1_v3_interactive_log.jsonl"
MONTHS = range(4, 11)   # open-water months (run 3; ice months Nov-Mar are excluded)
_lock = Lock()


def profile_eecu(text: str) -> float:
    """Total EECU-seconds of an ee.profilePrinting table (first column summed)."""
    tot = 0.0
    for line in text.splitlines()[1:]:
        m = re.match(r"\s*([\d.]+)\s", line)
        if m:
            tot += float(m.group(1))
    return tot


def scenes_todo(sensor: str) -> pd.DataFrame:
    lv = pd.read_csv(INTERIM / "levels" / "scene_levels.csv", low_memory=False)
    d = lv[(lv["sensor"] == sensor) & lv["epoch"].isin(["E1", "E2", "E3"]) & (lv["cloud_scene_pct"] < MAX_CLOUD)]
    d = d.assign(time=pd.to_datetime(d["time"])).sort_values("time")
    d = d[d["time"].dt.month.isin(list(MONTHS))]
    r1 = RUN1 / f"{sensor}.csv"
    if r1.exists():  # skip scenes that do not touch the zone (run 1: zone_px == 0)
        z = pd.read_csv(r1, usecols=["scene", "zone_px"])
        d = d[d["system:index"].isin(set(z.loc[pd.to_numeric(z["zone_px"], errors="coerce").fillna(0) > 0, "scene"]))]
    done = OUT / f"{sensor}.csv"
    if done.exists():
        have = set(pd.read_csv(done, usecols=["scene"])["scene"])
        d = d[~d["system:index"].isin(have)]
    return d


def run_batch(sensor: str, batch: pd.DataFrame, zone, cover) -> tuple[list[dict], float]:
    ids = batch["system:index"].tolist()
    start = (batch["time"].min() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    end = (batch["time"].max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    dom = pipeline_ee.domain_geometry()
    col = (masks.sentinel2(dom, start, end, bands=masks.CLASSIFY_BANDS, ids=ids) if sensor == "MSI_S2"
           else masks.landsat(sensor, dom, start, end, bands=masks.CLASSIFY_BANDS, ids=ids))
    table = pipeline_ee.scene_table(col, zone, clip=cover)
    buf = io.StringIO()
    with ee.profilePrinting(destination=buf):
        res = table.getInfo()
    rows = [{k: f["properties"].get(k) for k in FIELDS} for f in res["features"]]
    for r in rows:
        if r["hist"] is not None:
            r["hist"] = " ".join(str(int(c)) for c in r["hist"])
    return rows, profile_eecu(buf.getvalue())


def append_rows(sensor: str, rows: list[dict]) -> None:
    path = OUT / f"{sensor}.csv"
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("budget", type=float, help="hard EECU-s budget for this invocation")
    ap.add_argument("sensors", nargs="*", default=SENSORS)
    ap.add_argument("--batch", type=int, default=40)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    ee.Initialize(project=EE_PROJECT)
    OUT.mkdir(parents=True, exist_ok=True)
    zone = ee.Image(ZONE).select("zone").eq(1)
    cover = pipeline_ee.zone_cover(COVER)
    spent, stop = 0.0, False
    for sensor in a.sensors:
        todo = scenes_todo(sensor)
        batches = [todo.iloc[i:i + a.batch] for i in range(0, len(todo), a.batch)]
        print(f"{sensor}: {len(todo)} scenes in {len(batches)} batches", flush=True)
        with ThreadPoolExecutor(a.workers) as pool:
            pending = {}
            it = iter(enumerate(batches))
            for _ in range(a.workers):
                nxt = next(it, None)
                if nxt:
                    pending[pool.submit(run_batch, sensor, nxt[1], zone, cover)] = nxt
            while pending:
                for fut in as_completed(list(pending)):
                    i, b = pending.pop(fut)
                    try:
                        rows, eecu = fut.result()
                    except Exception as e:  # transient errors: log and leave the batch for a rerun
                        rec = {"sensor": sensor, "batch": i, "n": len(b), "error": str(e)[:300]}
                        rows, eecu = [], 0.0
                    else:
                        rec = {"sensor": sensor, "batch": i, "n": len(b), "rows": len(rows), "eecu_s": round(eecu, 2)}
                    with _lock:
                        spent += eecu
                        rec.update(cum_eecu_s=round(spent, 1), time=time.strftime("%Y-%m-%dT%H:%M:%S"))
                        if rows:
                            append_rows(sensor, rows)
                        with LOG.open("a", encoding="utf-8") as f:
                            f.write(json.dumps(rec) + "\n")
                    print(json.dumps(rec), flush=True)
                    if spent >= a.budget:
                        stop = True
                    if not stop:
                        nxt = next(it, None)
                        if nxt:
                            pending[pool.submit(run_batch, sensor, nxt[1], zone, cover)] = nxt
                    break
        if stop:
            print(f"budget reached: {spent:.1f} EECU-s", flush=True)
            break
    print(f"total EECU-s this run: {spent:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
