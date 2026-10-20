"""Check that the raw inputs `make all` reads are the ones the committed acquisition
records describe (review finding 19).

`make all` never acquires data (stage 1 of the Makefile). It reuses git-ignored raw
files under data/ and committed acquisition records, so before any result is rebuilt
this tool verifies:

  * every raw file against the SHA-256 its acquisition script recorded:
      - DAHITI series            results/provenance/dahiti_39.json      sha256
      - Hydroweb series          results/provenance/hydroweb_l_caspian.json: the zip's
                                 zip_sha256, and the extracted series must be
                                 byte-identical to its member in that zip
      - relief grids             results/provenance/<product>.json      subset_sha256
                                 (GEBCO_2022, GEBCO_2026, ETOPO 2022, SRTM15+, ETOPO1)
  * the Earth Engine scene inventory (data/interim/inventory/scenes.csv) against
    results/inventory/scenes_csv.json (SHA-256, bytes, rows), and every field of
    results/inventory/inventory_meta.json and both summary_sensor_*.csv tables,
    from which tools/build_metrics.py and the Methods take the inventory numbers,
    recomputed from scenes.csv as scripts/a2_scene_inventory.py computes them; they
    must be equal.

scripts/a2_scene_inventory.py records no checksum of scenes.csv, so
`make ee-scene-inventory` runs `--record-scenes` after it. That writes
scenes_csv.json only when the summary recomputed from scenes.csv equals the
committed inventory_meta.json and summary tables, so the checksum can only be
recorded for the file those numbers came from.

Usage
  python tools/check_inputs.py [--dahiti P] [--hydroweb P] [--hydroweb-dir D]
                               [--scenes P] [--bathy D] [--stamp P]
  python tools/check_inputs.py --record-scenes [--scenes P]

Exit status 1 on any missing file or mismatch. --stamp P writes P (a git-ignored
marker the Makefile orders every result after) only when every check passes.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROV = ROOT / "results" / "provenance"
INV = ROOT / "results" / "inventory"
SCENES_RECORD = INV / "scenes_csv.json"

DEFAULTS = {
    "dahiti": ROOT / "data" / "raw" / "dahiti" / "dahiti_39_caspian_sea_water_level.csv",
    "hydroweb": ROOT / "data" / "raw" / "hydroweb" / "extracted" / "HYDROWEB_LAKES_OPE" / "hydroprd_L_caspian.txt",
    "hydroweb_dir": ROOT / "data" / "raw" / "hydroweb",
    "scenes": ROOT / "data" / "interim" / "inventory" / "scenes.csv",
    "bathy": ROOT / "data" / "raw" / "bathymetry",
}

# Provenance record -> relief subset it describes (file names as the a7_* scripts write them).
GRIDS = {
    "gebco_2022": "gebco_2022_ncaspian.nc",
    "gebco_2026": "gebco_2026_ncaspian.nc",
    "etopo_2022": "etopo_2022_ncaspian.nc",
    "srtm15p_v2.7": "srtm15p_v2.7_ncaspian.nc",
    "etopo1": "etopo1_ncaspian.tif",
}

# inventory_meta.json fields that do not describe scenes.csv.
META_NOT_FROM_SCENES = {"generated", "end_date_exclusive"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _load(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check_sha(label: str, path: Path, expected: str | None, target: str = "") -> list[str]:
    path = Path(path)
    if not path.exists():
        return [f"{label}: {path} is missing" + (f" (run `make {target}`)" if target else "")]
    if not expected:
        return [f"{label}: the acquisition record gives no checksum for {path.name}"]
    got = sha256(path)
    return [] if got == expected else [f"{label}: {path} has SHA-256 {got}, the acquisition record says {expected}"]


def check_dahiti(path: Path, record: Path = PROV / "dahiti_39.json") -> list[str]:
    rec = _load(record)
    errs = check_sha("DAHITI series", path, rec.get("sha256"), "dahiti-download")
    if not errs and rec.get("bytes") is not None and Path(path).stat().st_size != rec["bytes"]:
        errs.append(f"DAHITI series: {path} is {Path(path).stat().st_size} bytes, the record says {rec['bytes']}")
    return errs


def check_hydroweb(path: Path, zip_dir: Path, record: Path = PROV / "hydroweb_l_caspian.json") -> list[str]:
    rec = _load(record)
    zpath = Path(zip_dir) / rec["zip"]
    errs = check_sha("Hydroweb zip", zpath, rec.get("zip_sha256"), "hydroweb-download")
    if errs:
        return errs
    path = Path(path)
    if not path.exists():
        return [f"Hydroweb series: {path} is missing (run `make hydroweb-download`)"]
    with zipfile.ZipFile(zpath) as zf:
        members = [m for m in zf.namelist() if m.rsplit("/", 1)[-1] == path.name]
        if len(members) != 1:
            return [f"Hydroweb series: {path.name} is not a single member of {zpath.name} ({zf.namelist()})"]
        if zf.read(members[0]) != path.read_bytes():
            return [f"Hydroweb series: {path} differs from {members[0]} in {zpath.name}"]
    return []


def check_grids(bathy: Path, prov_dir: Path = PROV) -> list[str]:
    errs = []
    for stem, fname in GRIDS.items():
        rec = _load(Path(prov_dir) / f"{stem}.json")
        target = "ee-etopo1 EE_OK=1" if stem == "etopo1" else "bathymetry-download"
        errs += check_sha(f"relief grid {stem}", Path(bathy) / fname, rec.get("subset_sha256"), target)
    return errs


# ----------------------------------------------------------------------------- scene inventory

def read_scenes(path: Path):
    import pandas as pd

    return pd.read_csv(path, parse_dates=["time"], low_memory=False)


def summarise_scenes(scenes) -> tuple[dict, str, str]:
    """(inventory_meta fields, summary_sensor_year.csv text, summary_sensor_month.csv
    text) computed from the scene table exactly as scripts/a2_scene_inventory.py does."""
    import pandas as pd

    scenes = scenes.copy()
    meta = {
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
    meta = json.loads(json.dumps(meta))  # plain ints and str keys, as read back from JSON
    bins = [-1, 20, 40, 60, 80, 100.1]
    labels = ["cc0_20", "cc20_40", "cc40_60", "cc60_80", "cc80_100"]
    scenes["cc_class"] = pd.cut(scenes["cloud_scene_pct"], bins=bins, labels=labels)
    tables = []
    for by in ("year", "month"):
        t = scenes.pivot_table(index=["sensor", by], columns="cc_class", values="time",
                               aggfunc="count", fill_value=0, observed=False)
        t["total"] = t.sum(axis=1)
        buf = io.StringIO()
        t.to_csv(buf)
        tables.append(buf.getvalue())
    return meta, tables[0], tables[1]


def _lines(text: str) -> list[str]:
    return [ln.rstrip() for ln in text.splitlines() if ln.strip()]


def check_scene_summary(scenes_path: Path, inv_dir: Path = INV) -> list[str]:
    """The committed inventory numbers are those scenes.csv gives."""
    meta_c = _load(Path(inv_dir) / "inventory_meta.json")
    meta, year_csv, month_csv = summarise_scenes(read_scenes(scenes_path))
    errs = []
    for k in sorted(set(meta_c) - META_NOT_FROM_SCENES | set(meta)):
        if meta_c.get(k) != meta.get(k):
            errs.append(f"scene inventory: inventory_meta.json {k} = {meta_c.get(k)!r}, "
                        f"recomputed from {Path(scenes_path).name}: {meta.get(k)!r}")
    for name, text in (("summary_sensor_year.csv", year_csv), ("summary_sensor_month.csv", month_csv)):
        committed = (Path(inv_dir) / name).read_text(encoding="utf-8")
        if _lines(committed) != _lines(text):
            errs.append(f"scene inventory: {name} differs from the table recomputed from {Path(scenes_path).name}")
    return errs


def check_scenes(scenes_path: Path, inv_dir: Path = INV, record: Path = SCENES_RECORD) -> list[str]:
    scenes_path = Path(scenes_path)
    if not scenes_path.exists():
        return [f"scene inventory: {scenes_path} is missing (run `make ee-scene-inventory EE_OK=1`)"]
    if not Path(record).exists():
        return [f"scene inventory: no checksum record {record} (after `make ee-scene-inventory`, "
                f"run `python tools/check_inputs.py --record-scenes`)"]
    rec = _load(record)
    errs = check_sha("scene inventory", scenes_path, rec.get("sha256"), "ee-scene-inventory EE_OK=1")
    if not errs and scenes_path.stat().st_size != rec.get("bytes"):
        errs.append(f"scene inventory: {scenes_path} is {scenes_path.stat().st_size} bytes, the record says {rec.get('bytes')}")
    return errs + check_scene_summary(scenes_path, inv_dir)


def record_scenes(scenes_path: Path, inv_dir: Path = INV, record: Path = SCENES_RECORD) -> list[str]:
    """Write the checksum record of scenes.csv, only if its recomputed summary equals
    the committed inventory_meta.json and summary tables."""
    scenes_path = Path(scenes_path)
    if not scenes_path.exists():
        return [f"scene inventory: {scenes_path} is missing"]
    errs = check_scene_summary(scenes_path, inv_dir)
    if errs:
        return errs + ["checksum NOT recorded: scenes.csv is not the file the committed inventory numbers came from"]
    n_rows = len(read_scenes(scenes_path))
    try:
        shown = scenes_path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        shown = scenes_path.name
    rec = {
        "file": shown,
        "bytes": scenes_path.stat().st_size,
        "sha256": sha256(scenes_path),
        "n_rows": n_rows,
        "inventory_meta_generated": _load(Path(inv_dir) / "inventory_meta.json").get("generated"),
        "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "written_by": "tools/check_inputs.py --record-scenes",
        "basis": "the summary recomputed from this file equals results/inventory/inventory_meta.json, "
                 "summary_sensor_year.csv and summary_sensor_month.csv",
    }
    Path(record).write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    print(f"recorded {record.relative_to(ROOT) if Path(record).is_relative_to(ROOT) else record}: "
          f"{rec['n_rows']} rows, sha256 {rec['sha256']}")
    return []


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dahiti", type=Path, default=DEFAULTS["dahiti"])
    ap.add_argument("--hydroweb", type=Path, default=DEFAULTS["hydroweb"])
    ap.add_argument("--hydroweb-dir", type=Path, default=DEFAULTS["hydroweb_dir"])
    ap.add_argument("--scenes", type=Path, default=DEFAULTS["scenes"])
    ap.add_argument("--bathy", type=Path, default=DEFAULTS["bathy"])
    ap.add_argument("--stamp", type=Path)
    ap.add_argument("--record-scenes", action="store_true")
    a = ap.parse_args(argv)

    if a.record_scenes:
        errs = record_scenes(a.scenes)
    else:
        errs = (check_dahiti(a.dahiti) + check_hydroweb(a.hydroweb, a.hydroweb_dir) + check_grids(a.bathy)
                + check_scenes(a.scenes))
    for e in errs:
        print("FAIL", e)
    if errs:
        return 1
    if not a.record_scenes:
        print("raw inputs match their acquisition records: DAHITI, Hydroweb, "
              f"{len(GRIDS)} relief grids, scene inventory")
        if a.stamp:
            a.stamp.parent.mkdir(parents=True, exist_ok=True)
            a.stamp.write_text(datetime.now(timezone.utc).isoformat(timespec="seconds") + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
