"""Generality test, Earth Engine steps (PREREG Amendment 16 items 6 and 10).

Every request is an interactive computePixels or getInfo wrapped in
ee.profilePrinting; its EECU-seconds are parsed, appended to the ledger
results/inventory/generality_ee_ledger.jsonl, and checked against two limits before
each step starts and while it runs:
- the generality cap of Amendment 16 item 10 (30 EECU-h for both basins), and
- the monthly rule: October-to-date use (the baseline recorded in the
  ledger header, plus the reservation for other analyses) plus this step's planned spend
  must stay under 640 EECU-h.

Commands (basin = great_salt_lake | lake_urmia):
  extents <basin>   near-maximum and near-minimum extents (zone v6 rules) at
                    250 m on the basin grid, by computePixels; GSW max_extent at 250 m.
  scenes <basin>    Landsat scene inventory (metadata only) for the open-water months.
  stage1 <basin> --budget S [--max-scenes N]
                    per-scene Otsu tables (pipeline_ee.scene_table) in batches.
  pilot <basin> --box w,s,e,n
                    stage-2 cost pilot on a 0.3 deg box: all groups of the plan,
                    512 px x 16 groups per request; writes the pilot record.
  stage2 <basin> --tag T --budget S
                    stage-2 counts (w, v, o) per plan group over the cover tiles.
  dep3 <basin>      3DEP 1/3" DEM averaged to the 60 m grid (positive-control substitute).
Usage: python scripts/g1_ee.py <command> <basin> [options]
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import caspian  # noqa: F401
import ee
import numpy as np
import pandas as pd
import pyproj

from caspian import ee_io, masks, pipeline_ee
from caspian import generality as G
from caspian.config import EE_PROJECT, INTERIM, RESULTS

INVENTORY = RESULTS / "inventory"
LEDGER = INVENTORY / "generality_ee_ledger.jsonl"
CAP_EECU_S = 60 * 3600.0                 # 60 EECU-h for GSL, spent included
MONTH_LIMIT_H = 640.0                    # monthly rule: month-to-date + planned < 640 EECU-h
MONTH = "2026-10"
SENSORS = ["TM_L5", "ETM_L7", "OLI_L8", "OLI_L9"]
MAX_CLOUD = 60
MONTHS = (4, 5, 6, 7, 8, 9, 10)          # November-March excluded
_lock = threading.Lock()


def wkt(basin):
    return pyproj.CRS(G.BASINS[basin]["crs"]).to_wkt(version="WKT1_GDAL")


def interim(basin) -> Path:
    d = INTERIM / "generality" / basin
    d.mkdir(parents=True, exist_ok=True)
    return d


def out_dir(basin) -> Path:
    d = RESULTS / "generality" / basin
    d.mkdir(parents=True, exist_ok=True)
    return d


# --- metering -------------------------------------------------------------------------------

def profile_eecu(text: str) -> float:
    tot = 0.0
    for line in text.splitlines()[1:]:
        m = re.match(r"\s*([\d.]+)\s", line)
        if m:
            tot += float(m.group(1))
    return tot


def spent_s() -> float:
    if not LEDGER.exists():
        return 0.0
    return sum(json.loads(l).get("eecu_s", 0.0) for l in LEDGER.read_text(encoding="utf-8").splitlines() if l.strip())


def log(basin, step, eecu_s, **kw):
    with _lock, LEDGER.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "basin": basin, "step": step,
                            "eecu_s": round(float(eecu_s), 3), **kw}) + "\n")


def interactive_logs_s(inventory: Path = INVENTORY) -> dict:
    """EECU-s per interactive log or ledger in results/inventory/ (every *.jsonl except the
    guard's cancellation log, whose tasks are batch tasks): the sum of the per-entry
    ``eecu_s``. Entries without it count 0."""
    out = {}
    for f in sorted(Path(inventory).glob("*.jsonl")):
        if f.name == "ee_guard_log.jsonl":
            continue
        tot = 0.0
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    tot += float(json.loads(line).get("eecu_s") or 0.0)
                except (ValueError, AttributeError):
                    pass
        out[f.name] = round(tot, 1)
    return out


def batch_s(month: str = MONTH) -> tuple[float, int, str]:
    """EECU-s of this project's batch tasks created in ``month`` (ee.data.listOperations,
    batchEecuUsageSeconds); falls back to results/inventory/ee_batch_tasks_<month>.json."""
    try:
        ops = ee.data.listOperations()
        tot, n = 0.0, 0
        for op in ops:
            md = op.get("metadata", {})
            if str(md.get("createTime", "")).startswith(month):
                tot += float(md.get("batchEecuUsageSeconds") or 0.0)
                n += 1
        return tot, n, "ee.data.listOperations"
    except Exception:
        recs = json.loads((INVENTORY / f"ee_batch_tasks_{month}.json").read_text(encoding="utf-8"))
        return sum(float(r.get("eecu_s") or 0.0) for r in recs), len(recs), f"ee_batch_tasks_{month}.json"


def month_to_date(write: bool = True) -> dict:
    """October-to-date use: every interactive log and ledger in
    results/inventory/ plus every batch task of the month. Interactive requests that no log
    recorded are not counted (the Earth Engine console figure stays authoritative)."""
    inter = interactive_logs_s()
    b, n, src = batch_s()
    rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "month": MONTH, "interactive_s": inter,
           "interactive_total_h": round(sum(inter.values()) / 3600, 2), "batch_s": round(b, 1), "batch_tasks": n,
           "batch_source": src, "total_h": round((sum(inter.values()) + b) / 3600, 2)}
    if write:
        (INVENTORY / "generality_month_to_date.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return rec


def check(planned_s: float, what: str) -> None:
    """Stop before a step that would break the GSL cap (60 EECU-h, spend included) or the
    month rule (month-to-date from every log and batch task, plus this step, under 640)."""
    used = spent_s()
    mtd = month_to_date()
    month = mtd["total_h"] + planned_s / 3600
    if used + planned_s > CAP_EECU_S:
        raise SystemExit(f"{what}: planned {planned_s:.0f} EECU-s + spent {used:.0f} exceeds the "
                         f"{CAP_EECU_S / 3600:.0f} EECU-h cap")
    if month >= MONTH_LIMIT_H:
        raise SystemExit(f"{what}: month-to-date {mtd['total_h']:.1f} + planned {planned_s / 3600:.1f} EECU-h "
                         f">= {MONTH_LIMIT_H}")
    print(f"budget check {what}: generality spent {used / 3600:.2f} h, planned {planned_s / 3600:.2f} h; "
          f"month to date {mtd['total_h']:.1f} h (interactive {mtd['interactive_total_h']:.1f}, batch "
          f"{mtd['batch_s'] / 3600:.1f}) + planned = {month:.1f} h (< {MONTH_LIMIT_H})", flush=True)


def metered(fn, *a, **k):
    buf = io.StringIO()
    with ee.profilePrinting(destination=buf):
        r = fn(*a, **k)
    return r, profile_eecu(buf.getvalue())


def pixels(img, transform, dims, crs_wkt):
    return ee.data.computePixels({"expression": img, "fileFormat": "NUMPY_NDARRAY",
                                  "grid": ee_io.grid(transform, dims, crs_wkt)})


def region(basin):
    w, s, e, n = G.BASINS[basin]["bbox"]
    return ee.Geometry.Rectangle([w, s, e, n], None, False)


# --- extents ----------------------------------------------------------------------------------

def windows(basin) -> dict:
    """Amendment 16 item 6: per arm, the April-October season of the year of the record
    maximum (k = 3 running median) and of the year of the lowest April-October level."""
    from caspian import levels
    lv = load_levels(basin)
    out = {}
    for arm, g in lv.groupby("arm"):
        g = g[(g["date"] >= "1984-01-01") & g["level"].notna()].sort_values("date")
        ex = levels.smoothed_extremes(g["date"], g["level"], k=3)
        ow = g[g["date"].dt.month.isin(MONTHS)]
        low_t = ow.loc[ow["level"].idxmin(), "date"]
        out[arm] = {"max_date": str(pd.Timestamp(ex["time_of_max"]).date()), "max_m": round(float(ex["max_m"]), 3),
                    "high_year": int(pd.Timestamp(ex["time_of_max"]).year),
                    "low_date": str(low_t.date()), "low_m": round(float(ow["level"].min()), 3), "low_year": int(low_t.year)}
    return out


def load_levels(basin) -> pd.DataFrame:
    from caspian.config import RAW
    if basin == "great_salt_lake":
        lv = pd.read_csv(RAW / "generality" / basin / "levels" / "gsl_daily_levels.csv", parse_dates=["date"])
        return lv.rename(columns={"level_navd88_m": "level"})[["arm", "date", "level"]]
    raise NotImplementedError(basin)


def extents(basin) -> int:
    ee.Initialize(project=EE_PROJECT)
    check(1500, "extents")
    win = windows(basin)
    tr, shape = G.grid(G.BASINS[basin]["bbox"], G.BASINS[basin]["crs"], 250)
    dims = (shape[1], shape[0])
    reg = region(basin)
    bands = ["green", "nir", "swir1"]
    gsw = ee.Image(pipeline_ee.GSW).select("max_extent").unmask(0)
    imgs = {}
    for year in sorted({w["high_year"] for w in win.values()}):
        col = masks.landsat("TM_L5", reg, f"{year}-04-01", f"{year}-11-01", bands=bands).filter(
            ee.Filter.calendarRange(4, 10, "month"))
        n = col.map(pipeline_ee.zone_water).count().unmask(0)
        frac = col.map(pipeline_ee.zone_water).mean()
        filled = n.lt(pipeline_ee.MIN_OBS_HIGH)
        hi = frac.gte(pipeline_ee.HIGH_MIN_FRAC).unmask(0).where(filled, gsw)
        imgs[f"high_{year}"] = hi
        imgs[f"filled_{year}"] = filled
        imgs[f"nhigh_{year}"] = n
    for year in sorted({w["low_year"] for w in win.values()}):
        cols = [masks.landsat(s, reg, f"{year}-04-01", f"{year}-11-01", bands=bands) for s in ("OLI_L8", "OLI_L9")]
        col = cols[0].merge(cols[1]).filter(ee.Filter.calendarRange(4, 10, "month"))
        imgs[f"low_{year}"] = col.map(pipeline_ee.zone_water).mean().gte(pipeline_ee.LOW_MIN_FRAC).unmask(0)
        imgs[f"nlow_{year}"] = col.map(pipeline_ee.zone_water).count().unmask(0)
    # GSW max_extent at 250 m: any 30 m pixel ever water (so "never water" is strict)
    imgs["gsw_max"] = gsw.reduceResolution(ee.Reducer.max(), maxPixels=1024)
    img = ee.Image.cat([v.rename(k).toInt16() for k, v in imgs.items()])
    arr, eecu = metered(pixels, img, tr, dims, wkt(basin))
    log(basin, "extents", eecu, dims=dims)
    np.savez_compressed(interim(basin) / "extents_250m.npz", transform=np.array(tr),
                        **{k: np.asarray(arr[k]) for k in imgs}, windows=json.dumps(win))
    rec = {"windows": win, "transform": tr, "shape": list(shape), "eecu_s": round(eecu, 2),
           "rules": "zone v6 (D-029): high = water (zone_water) in >= 10% of clear TM April-October observations "
                    "of the year of the arm's record maximum, GSW max_extent where < 3 observations; low = water in "
                    ">= 90% of clear OLI April-October observations of the year of the arm's lowest April-October level",
           "counts": {k: int((np.asarray(arr[k]) > 0).sum()) for k in imgs}}
    (out_dir(basin) / "extents.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(json.dumps(rec, indent=1))
    return 0


# --- scene inventory ----------------------------------------------------------------------------

def scenes(basin) -> int:
    ee.Initialize(project=EE_PROJECT)
    check(200, "scenes")
    reg = region(basin)
    rows, tot = [], 0.0
    for s in SENSORS:
        cid = masks.LANDSAT[s][0]
        col = (ee.ImageCollection(cid).filterBounds(reg).filterDate("1984-01-01", "2027-01-01")
               .filter(ee.Filter.lt("CLOUD_COVER", MAX_CLOUD)).filter(ee.Filter.calendarRange(4, 10, "month")))
        props = ["system:index", "system:time_start", "WRS_PATH", "WRS_ROW", "CLOUD_COVER"]
        d, eecu = metered(lambda: col.reduceColumns(ee.Reducer.toList(len(props)), props).get("list").getInfo())
        tot += eecu
        for r in d:
            rows.append({"scene": r[0], "sensor": s, "time": int(r[1]), "path": int(r[2]), "row": int(r[3]),
                         "cloud_scene_pct": float(r[4])})
        print(s, len(d), flush=True)
    log(basin, "scenes", tot, n=len(rows))
    df = pd.DataFrame(rows)
    df["t"] = pd.to_datetime(df["time"], unit="ms")
    df["acq"] = df["sensor"] + "_" + df["path"].map("{:03d}".format) + "_" + df["t"].dt.strftime("%Y%m%d")
    df.to_csv(interim(basin) / "scenes.csv", index=False)
    summ = df.groupby(["sensor", "path", "row"]).size().rename("n").reset_index()
    (out_dir(basin) / "scene_inventory.json").write_text(json.dumps(
        {"n_scenes": len(df), "n_acquisitions": int(df["acq"].nunique()), "eecu_s": round(tot, 2),
         "by_path_row": summ.to_dict("records"), "rule": f"cloud < {MAX_CLOUD}%, April-October, 1984-2026"}, indent=1),
        encoding="utf-8")
    print(len(df), "scenes", df["acq"].nunique(), "acquisitions", round(tot, 1), "EECU-s")
    return 0


# --- zone geometry in Earth Engine -------------------------------------------------------------

def zone_fc(basin):
    """The local zone (scripts/g1_static.py) as an ee.FeatureCollection of lon/lat polygons,
    and the cover tiles likewise."""
    d = out_dir(basin)
    z = json.loads((d / "zone_lonlat.geojson").read_text(encoding="utf-8"))
    c = json.loads((d / "cover_lonlat.geojson").read_text(encoding="utf-8"))
    zf = ee.FeatureCollection([ee.Feature(ee.Geometry(f["geometry"], None, False), {}) for f in z["features"]])
    return zf, c


def zone_image(zf):
    return ee.Image(0).byte().paint(zf, 1).rename("zone")


# --- stage 1 ----------------------------------------------------------------------------------

FIELDS = ["scene", "sensor", "acq", "time", "otsu_threshold", "otsu_n", "zone_px", "zone_clear",
          "zone_snow", "zone_snow_ndsi", "zone_snow_qa", "cloud_scene", "hist"]


def stage1(basin, budget: float, max_scenes: int | None = None, batch: int = 40, workers: int = 4,
           log_name: str = "stage1") -> int:
    ee.Initialize(project=EE_PROJECT)
    check(budget, log_name)
    sc = pd.read_csv(interim(basin) / "scenes.csv")
    lvl = pd.read_csv(interim(basin) / "scene_levels.csv")
    sc = sc[sc["scene"].isin(set(lvl.loc[lvl["has_level"], "scene"]))]
    out = interim(basin) / "stage1.csv"
    done = set(pd.read_csv(out, usecols=["scene"])["scene"]) if out.exists() else set()
    todo = sc[~sc["scene"].isin(done)].sort_values("time")
    if max_scenes:
        todo = todo.iloc[:max_scenes]
    zf, cover = zone_fc(basin)
    zone = zone_image(zf)
    cov = ee.FeatureCollection([ee.Feature(ee.Geometry(f["geometry"], None, False), {}) for f in cover["features"]]).geometry()
    reg = region(basin)
    jobs = []
    for s, g in todo.groupby("sensor"):
        g = g.sort_values("time")
        for k in range(0, len(g), batch):
            jobs.append((s, g.iloc[k:k + batch]))
    print(f"{len(todo)} scenes in {len(jobs)} batches", flush=True)
    spent = [0.0]
    stop = [False]

    def run(job):
        s, b = job
        t = pd.to_datetime(b["time"], unit="ms")
        start = (t.min() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        end = (t.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        col = masks.landsat(s, reg, start, end, bands=masks.CLASSIFY_BANDS, ids=b["scene"].tolist())
        table = pipeline_ee.scene_table(col, zone, clip=cov)
        res, eecu = metered(table.getInfo)
        rows = [{k: f["properties"].get(k) for k in FIELDS} for f in res["features"]]
        for r in rows:
            if r["hist"] is not None:
                r["hist"] = " ".join(str(int(c)) for c in r["hist"])
        return rows, eecu

    with ThreadPoolExecutor(workers) as pool:
        it = iter(jobs)
        running = {}
        for _ in range(workers):
            j = next(it, None)
            if j is not None:
                running[pool.submit(run, j)] = j
        while running:
            dn, _ = wait(running, return_when=FIRST_COMPLETED)
            for fut in dn:
                j = running.pop(fut)
                try:
                    rows, eecu = fut.result()
                except Exception as ex:
                    print("batch failed", j[0], len(j[1]), str(ex)[:200], flush=True)
                    rows, eecu = [], 0.0
                with _lock:
                    spent[0] += eecu
                    if rows:
                        df = pd.DataFrame(rows, columns=FIELDS)
                        df.to_csv(out, mode="a", header=not out.exists(), index=False)
                log(basin, log_name, eecu, n=len(j[1]), rows=len(rows))
                print(f"{j[0]} {len(rows)} rows {eecu:.1f} EECU-s | {spent[0]:.0f}", flush=True)
                if spent[0] >= budget:
                    stop[0] = True
                if not stop[0]:
                    nj = next(it, None)
                    if nj is not None:
                        running[pool.submit(run, nj)] = nj
    print(f"stage 1 spent {spent[0]:.1f} EECU-s", flush=True)
    return 0


# --- stage 2 ----------------------------------------------------------------------------------

def build_counts(plan: pd.DataFrame, groups, region_ll, observed: bool = True):
    """Counts (w, v[, o]) per group over ``region_ll``: the main-export image of
    scripts/a6_stage2_export.build_image without the sector masks (no screening,
    Amendment 16 item 6): masks.landsat -> pipeline_ee.water_valid with the
    acquisition's threshold -> pipeline_ee.group_counts (each acquisition once)."""
    kinds = ("w", "v", "o") if observed else ("w", "v")
    by_group = dict(tuple(plan[plan["group"].isin(set(groups))].groupby("group")))
    bands = []
    for g in groups:
        pg = by_group.get(g)
        if pg is None or pg.empty:
            bands.append(ee.Image.constant([0] * len(kinds)).rename([f"{k}_{g}" for k in kinds]).uint16())
            continue
        thr = ee.Dictionary(dict(zip(pg["scene"], pg["threshold"].astype(float))))
        cols = []
        for s in SENSORS:
            ids = pg.loc[pg["sensor"] == s, "scene"].tolist()
            if ids:
                cols.append(masks.landsat(s, region_ll, "1984-01-01", "2027-01-01", bands=masks.CLASSIFY_BANDS, ids=ids))
        col = cols[0]
        for c in cols[1:]:
            col = col.merge(c)

        def wv(im, thr=thr):
            sid = im.get("system:index")
            key = ee.String(ee.Algorithms.If(thr.contains(sid), sid, ee.String(sid).split("_").slice(-3).join("_")))
            return pipeline_ee.water_valid(im, thr.get(key), observed=observed).copyProperties(im, ["system:time_start", "acq"])
        bands.append(pipeline_ee.group_counts(col.map(wv), g, kinds))
    names = [f"{p}_{g}" for g in groups for p in kinds]
    return ee.ImageCollection.fromImages(bands).toBands().rename(names).clip(region_ll)


KINDS = ("w", "v", "o")


def _partial_dir(out: Path, key: str) -> Path:
    return Path(out) / "partial" / key


def load_partials(out: Path, key: str, dims) -> tuple[dict, dict]:
    """Checkpointed requests of one target: ({band: array of the target},
    {(c0, r0): set of finished groups})."""
    arr, done = {}, {}
    d = _partial_dir(out, key)
    if not d.exists():
        return arr, done
    for f in sorted(d.glob("*.npz")):
        with np.load(f) as z:
            c0, r0 = int(z["c0"]), int(z["r0"])
            groups = [str(g) for g in z["groups"]]
            for g in groups:
                for kind in KINDS:
                    band = f"{kind}_{g}"
                    sub = z[band]
                    if band not in arr:
                        arr[band] = np.zeros((dims[1], dims[0]), np.uint16)
                    arr[band][r0:r0 + sub.shape[0], c0:c0 + sub.shape[1]] = sub
        done.setdefault((c0, r0), set()).update(groups)
    return arr, done


def save_partial(out: Path, key: str, c0: int, r0: int, chunk, arr) -> Path:
    """One finished request, written atomically, so that a budget stop or a crash loses at
    most the requests in flight."""
    d = _partial_dir(out, key)
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"r{r0:05d}_c{c0:05d}_{chunk[0]}_{len(chunk):02d}.npz"
    tmp = f.with_name(f.stem + ".tmp.npz")
    np.savez_compressed(tmp, c0=c0, r0=r0, groups=np.array(list(chunk)),
                        **{f"{k}_{g}": np.asarray(arr[f"{k}_{g}"], np.uint16) for g in chunk for k in KINDS})
    tmp.replace(f)
    return f


def run_counts(basin, plan, targets: dict, budget: float, step: str, px: int = 512, groups_per_req: int = 16,
               workers: int = 8, out=None, request_fn=None, guard=None, guard_every: int = 10,
               first_guess: float = 250.0) -> dict:
    """Interactive computePixels over ``targets`` = {key: {"region": ee.Geometry, "transform",
    "dims" (w, h), "groups"}} (512 px x 16 groups per request), metered against a hard
    ``budget`` (EECU-s).

    Checkpointing: every finished request is saved under
    ``out``/partial/<key>/; a rerun loads those and requests only the missing groups of each
    sub-block. A target is written (tile_<key>.npz) when all its groups are in, and its
    partial files are then removed.

    Stops: predictively, before a request that with those in flight would pass ``budget``
    (mean cost of the finished requests, ``first_guess`` before the first); and when
    ``guard(spent)`` returns False (checked every ``guard_every`` finished requests).
    ``request_fn(job, target)`` -> (arrays, eecu_s, seconds) replaces the Earth Engine
    request (tests). Returns {key: eecu_s of this invocation} for the targets written."""
    import shutil
    crs = wkt(basin)
    jobs, state = [], {}
    result = {}
    for key, t in targets.items():
        if out is not None and (out / f"tile_{key}.npz").exists():
            continue
        arr, done = load_partials(out, key, t["dims"]) if out is not None else ({}, {})
        state[key] = {"pending": 0, "failed": False, "eecu": 0.0, "arr": arr, **t}
        for c0, r0, w, h in ee_io.tiles(t["dims"], px):
            todo = [g for g in t["groups"] if g not in done.get((c0, r0), set())]
            for k in range(0, len(todo), groups_per_req):
                jobs.append((key, c0, r0, w, h, todo[k:k + groups_per_req]))
                state[key]["pending"] += 1
        if state[key]["pending"] == 0 and out is not None:       # every request was checkpointed
            np.savez_compressed(out / f"tile_{key}.npz", transform=np.array(t["transform"]), **arr)
            shutil.rmtree(_partial_dir(out, key), ignore_errors=True)
            result[key] = 0.0
            print(f"wrote {key} from checkpoints", flush=True)
    print(f"{len(state)} targets, {len(jobs)} requests", flush=True)
    spent, stop = 0.0, False
    tries: dict = {}

    def run(job):
        key, c0, r0, w, h, chunk = job
        t = state[key]
        if request_fn is not None:
            return request_fn(job, t)
        a, b_, c, d, e, f = t["transform"]
        sub = [a, b_, c + a * c0, d, e, f + e * r0]
        img = build_counts(plan, chunk, t["region"])
        t0 = time.time()
        arr, eecu = metered(pixels, img, sub, (w, h), crs)
        return arr, eecu, time.time() - t0

    queue = list(jobs)
    done_n = 0
    with ThreadPoolExecutor(workers) as pool:
        running = {}
        while queue or running:
            avg = spent / done_n if done_n else first_guess
            while queue and len(running) < workers and not stop:
                if spent + (len(running) + 1) * avg > budget:
                    stop = True
                    queue.clear()
                    print(f"budget stop: spent {spent:.0f} + {len(running) + 1} x {avg:.0f} > {budget:.0f}", flush=True)
                    break
                job = queue.pop(0)
                running[pool.submit(run, job)] = job
            if not running:
                break
            dn, _ = wait(running, return_when=FIRST_COMPLETED)
            for fut in dn:
                job = running.pop(fut)
                key, c0, r0, w, h, chunk = job
                t = state[key]
                try:
                    arr, eecu, secs = fut.result()
                except Exception as ex:
                    msg = str(ex).lower()
                    jid = (key, c0, r0, tuple(chunk))
                    if ("profile" in msg or "429" in msg or "unavailable" in msg) and tries.get(jid, 0) < 3:
                        tries[jid] = tries.get(jid, 0) + 1
                        queue.append(job)
                        continue
                    if len(chunk) > 1 and ("memory" in msg or "timed out" in msg or "deadline" in msg):
                        half = len(chunk) // 2
                        queue.extend([(key, c0, r0, w, h, chunk[:half]), (key, c0, r0, w, h, chunk[half:])])
                        t["pending"] += 1
                        print("retry split", key, len(chunk), str(ex)[:100], flush=True)
                    else:
                        t["failed"] = True
                        t["pending"] -= 1
                        print("FAILED", key, r0, c0, chunk[0], str(ex)[:200], flush=True)
                    continue
                spent += eecu
                done_n += 1
                t["eecu"] += eecu
                for g in chunk:
                    for kind in KINDS:
                        band = f"{kind}_{g}"
                        if band not in t["arr"]:
                            t["arr"][band] = np.zeros((t["dims"][1], t["dims"][0]), np.uint16)
                        t["arr"][band][r0:r0 + h, c0:c0 + w] = arr[band]
                if out is not None:
                    save_partial(out, key, c0, r0, chunk, arr)
                t["pending"] -= 1
                log(basin, step, eecu, key=key, n_groups=len(chunk), seconds=round(secs, 1))
                if t["pending"] == 0 and not t["failed"]:
                    if out is not None:
                        np.savez_compressed(out / f"tile_{key}.npz", transform=np.array(t["transform"]), **t["arr"])
                        shutil.rmtree(_partial_dir(out, key), ignore_errors=True)
                    result[key] = round(t["eecu"], 2)
                    print(f"wrote {key}: {t['eecu']:.0f} EECU-s | {spent:.0f}", flush=True)
                    t["arr"] = {}
                if guard is not None and done_n % guard_every == 0 and not stop and not guard(spent):
                    stop = True
                    queue.clear()
                    print(f"guard stop after {done_n} requests ({spent:.0f} EECU-s)", flush=True)
                if spent >= budget and not stop:
                    stop = True
                    queue.clear()
                    print(f"budget reached: {spent:.0f} EECU-s", flush=True)
    print(f"{step}: {spent:.1f} EECU-s", flush=True)
    return result


BLOCK_PX = 512      # one request = one 512 px block x 16 groups


def tile_targets(basin, plan_by_arm: dict, res: float = 60.0, only=None):
    """Stage-2 targets: the 512 px blocks of the basin's 60 m grid that meet the
    cover, one per block and arm, each clipped to cover and arm (Earth Engine reads whole
    scene tiles per request, so many small targets cost far more per km2; measured in the
    first pilot). Groups are the arm's plan groups. ``only``: keys to keep."""
    from shapely.geometry import box, shape
    from shapely.ops import transform as stransform
    from shapely.ops import unary_union
    d = out_dir(basin)
    arms = json.loads((d / "arms_laea.geojson").read_text(encoding="utf-8"))
    cover = unary_union([shape(f["geometry"]) for f in json.loads((d / "cover_laea.geojson").read_text(encoding="utf-8"))["features"]])
    inv = pyproj.Transformer.from_crs(G.BASINS[basin]["crs"], "EPSG:4326", always_xy=True).transform
    tr, shp = G.grid(G.BASINS[basin]["bbox"], G.BASINS[basin]["crs"], res)
    out = {}
    for f in arms["features"]:
        arm = f["properties"]["arm"]
        if arm not in plan_by_arm:
            continue
        dom = shape(f["geometry"]).intersection(cover)
        groups = sorted(plan_by_arm[arm]["group"].unique())
        for bj, r0 in enumerate(range(0, shp[0], BLOCK_PX)):
            for bi, c0 in enumerate(range(0, shp[1], BLOCK_PX)):
                w, h = min(BLOCK_PX, shp[1] - c0), min(BLOCK_PX, shp[0] - r0)
                x0, y1 = tr[2] + res * c0, tr[5] - res * r0
                part = dom.intersection(box(x0, y1 - res * h, x0 + res * w, y1))
                if part.is_empty or part.area < 1e6:
                    continue
                key = f"{arm}_{bi:02d}_{bj:02d}"
                if only and key not in only:
                    continue
                ll = stransform(inv, part.segmentize(1000))
                out[key] = {"region": ee.Geometry(json.loads(json.dumps(ll.__geo_interface__)), None, False),
                            "transform": [res, 0.0, float(x0), 0.0, -res, float(y1)], "dims": (w, h), "groups": groups,
                            "arm": arm, "area_km2": round(part.area / 1e6, 2)}
    return out


def load_plan(basin, tag):
    p = pd.read_csv(interim(basin) / f"plan_{tag}.csv")
    return {arm: g for arm, g in p.groupby("arm")}


def pilot(basin, box_ll, tag: str = "main", budget: float = 3000.0) -> int:
    """Stage-2 cost pilot (Amendment 16 item 10): the 512 px block of the main run (real
    thresholds, so its tile is kept for the main run) that overlaps the 0.3 deg box most,
    with all its arm's groups; the projection for the whole cover is the larger of the
    per-km2 and the per-target scalings."""
    from shapely.geometry import box, shape
    from shapely.ops import transform as stransform
    ee.Initialize(project=EE_PROJECT)
    check(budget, "pilot")
    plans = load_plan(basin, tag)
    tg = tile_targets(basin, plans)
    fwd = pyproj.Transformer.from_crs("EPSG:4326", G.BASINS[basin]["crs"], always_xy=True).transform
    bx = stransform(fwd, box(*box_ll).segmentize(0.01))

    def overlap(t):
        a, _, c, _, e, f = t["transform"]
        w, h = t["dims"]
        return box(c, f + e * h, c + a * w, f).intersection(bx).area
    key = max(tg, key=lambda k: overlap(tg[k]))
    sdir = interim(basin) / f"stage2_{tag}"
    sdir.mkdir(exist_ok=True)
    t0 = time.time()
    res = run_counts(basin, pd.concat(plans.values()), {key: tg[key]}, budget, "stage2_pilot_block", out=sdir, workers=4)
    if key not in res:
        print("pilot block not completed")
        return 1
    eecu = res[key]
    area = tg[key]["area_km2"]
    tot_area = sum(t["area_km2"] for t in tg.values())
    n_req = {k: -(-len(t["groups"]) // 16) for k, t in tg.items()}
    proj_area = eecu / area * tot_area
    proj_target = eecu / n_req[key] * sum(n_req.values())
    rec = {"box": list(box_ll), "tag": tag, "pilot_target": key, "pilot_area_km2": area,
           "pilot_eecu_s": round(eecu, 1), "pilot_eecu_s_per_km2": round(eecu / area, 3),
           "pilot_seconds": round(time.time() - t0, 1), "px": BLOCK_PX, "groups_per_request": 16,
           "n_targets": len(tg), "cover_area_km2": round(tot_area, 1), "n_requests": int(sum(n_req.values())),
           "projection_stage2_eecu_s": {"per_km2": round(proj_area), "per_request": round(proj_target),
                                        "used": round(max(proj_area, proj_target))},
           "note": "first pilot (5 km tiles, placeholder thresholds) spent 7,436 EECU-s for 17.9 km2 (366 EECU-s/km2): "
                   "the per-request cost did not shrink with the target, so the targets became 512 px blocks (D-094)"}
    (RESULTS / "inventory" / f"generality_{basin}_pilot.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(json.dumps(rec, indent=1))
    return 0


def stage2(basin, tag: str, budget: float) -> int:
    ee.Initialize(project=EE_PROJECT)
    check(budget, "stage2")
    plans = load_plan(basin, tag)
    tg = tile_targets(basin, plans)
    sdir = interim(basin) / f"stage2_{tag}"
    sdir.mkdir(exist_ok=True)
    st_path = RESULTS / "inventory" / f"generality_{basin}_stage2_{tag}.json"
    st = json.loads(st_path.read_text(encoding="utf-8")) if st_path.exists() else {"tiles": {}}
    def guard(spent):
        """Month rule during the run: logged month-to-date (which already holds this run's
        logged requests) plus what this run may still spend must stay under 640 EECU-h."""
        mtd = month_to_date()
        ok = mtd["total_h"] + max(budget - spent, 0.0) / 3600 < MONTH_LIMIT_H
        if not ok:
            print(f"month guard: {mtd['total_h']:.1f} + {(budget - spent) / 3600:.1f} h >= {MONTH_LIMIT_H}", flush=True)
        return ok

    res = run_counts(basin, pd.concat(plans.values()), tg, budget, f"stage2_{tag}", out=sdir, guard=guard,
                     guard_every=20)
    for k, v in res.items():
        st["tiles"][k] = {"eecu_s": v, "area_km2": tg[k]["area_km2"], "arm": tg[k]["arm"], "dims": tg[k]["dims"],
                          "transform": tg[k]["transform"], "n_groups": len(tg[k]["groups"])}
    st["n_targets"] = len(tg)
    st["n_done"] = len(list(sdir.glob("tile_*.npz")))
    st_path.write_text(json.dumps(st, indent=1), encoding="utf-8")
    print("tiles done", st["n_done"], "of", len(tg))
    return 0


# --- 3DEP (positive-control substitute) ---------------------------------------------------------

def dep3(basin) -> int:
    ee.Initialize(project=EE_PROJECT)
    check(600, "dep3")
    tr, shape = G.grid(G.BASINS[basin]["bbox"], G.BASINS[basin]["crs"], 60)
    proj = ee.Projection(wkt(basin)).atScale(60)
    col = ee.ImageCollection("USGS/3DEP/10m_collection").filterBounds(region(basin))
    img = col.mosaic().select(0).rename("elev")
    native = col.first().projection()
    img = (img.setDefaultProjection(native).reduceResolution(ee.Reducer.mean(), maxPixels=256)
           .reproject(proj).unmask(-9999).toFloat())
    arr, eecu = {}, 0.0
    out = np.full(shape, np.nan, np.float32)
    for c0, r0, w, h in ee_io.tiles((shape[1], shape[0]), 1024):
        sub = [tr[0], 0.0, tr[2] + tr[0] * c0, 0.0, tr[4], tr[5] + tr[4] * r0]
        a, e = metered(pixels, img, sub, (w, h), wkt(basin))
        eecu += e
        v = np.asarray(a["elev"], np.float32)
        v[v <= -9000] = np.nan
        out[r0:r0 + h, c0:c0 + w] = v
    log(basin, "dep3", eecu)
    np.save(interim(basin) / "dep3_60m.npy", out)
    ids = col.aggregate_array("system:index").getInfo()
    rec = {"asset": "USGS/3DEP/10m_collection", "n_images": len(ids), "reduce": "mean of the 1/3 arc-second cells "
           "in each 60 m cell (reduceResolution)", "transform": tr, "shape": list(shape), "eecu_s": round(eecu, 2),
           "finite_cells": int(np.isfinite(out).sum()),
           "range_m": [float(np.nanmin(out)), float(np.nanmax(out))]}
    (RESULTS / "provenance" / f"generality_{basin}_dep3.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(json.dumps(rec, indent=1))
    return 0


def main(argv) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("basin")
    ap.add_argument("--budget", type=float, default=0.0)
    ap.add_argument("--max-scenes", type=int, default=None)
    ap.add_argument("--box", default=None)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--log-name", default="stage1")
    a = ap.parse_args(argv)
    if a.cmd == "extents":
        return extents(a.basin)
    if a.cmd == "scenes":
        return scenes(a.basin)
    if a.cmd == "stage1":
        return stage1(a.basin, a.budget, a.max_scenes, log_name=a.log_name)
    if a.cmd == "pilot":
        return pilot(a.basin, tuple(float(v) for v in a.box.split(",")), a.tag, a.budget or 3000.0)
    if a.cmd == "stage2":
        return stage2(a.basin, a.tag, a.budget)
    if a.cmd == "dep3":
        return dep3(a.basin)
    raise SystemExit(f"unknown command {a.cmd}")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
