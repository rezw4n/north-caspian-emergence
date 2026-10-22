"""tools/check_inputs.py: raw inputs are verified against their acquisition records
before `make all` builds on them, and the inventory numbers the Methods take from
inventory_meta.json are recomputed from scenes.csv."""

import hashlib
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import check_inputs as ci  # noqa: E402


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ----------------------------------------------------------------------------- checksums

def test_dahiti_checksum(tmp_path):
    f = tmp_path / "dahiti.csv"
    f.write_bytes(b"date,wse\n2020-01-01,-28.1\n")
    rec = tmp_path / "dahiti_39.json"
    rec.write_text(json.dumps({"sha256": sha(f), "bytes": f.stat().st_size}), encoding="utf-8")
    assert ci.check_dahiti(f, rec) == []
    f.write_bytes(b"date,wse\n2020-01-01,-28.2\n")
    errs = ci.check_dahiti(f, rec)
    assert errs and "SHA-256" in errs[0]
    errs = ci.check_dahiti(tmp_path / "absent.csv", rec)
    assert errs and "missing" in errs[0] and "dahiti-download" in errs[0]


def test_record_without_checksum_fails(tmp_path):
    f = tmp_path / "x.csv"
    f.write_text("x", encoding="utf-8")
    rec = tmp_path / "r.json"
    rec.write_text("{}", encoding="utf-8")
    assert ci.check_dahiti(f, rec)  # never passes for lack of a recorded checksum


def _hydroweb(tmp_path, member=b"level series\n"):
    zdir = tmp_path / "hydroweb"
    (zdir / "extracted" / "HYDROWEB_LAKES_OPE").mkdir(parents=True)
    z = zdir / "abc.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("HYDROWEB_LAKES_OPE/hydroprd_L_caspian.txt", member)
    txt = zdir / "extracted" / "HYDROWEB_LAKES_OPE" / "hydroprd_L_caspian.txt"
    txt.write_bytes(member)
    rec = tmp_path / "hydroweb.json"
    rec.write_text(json.dumps({"zip": z.name, "zip_sha256": sha(z)}), encoding="utf-8")
    return txt, zdir, rec


def test_hydroweb_zip_and_extracted_series(tmp_path):
    txt, zdir, rec = _hydroweb(tmp_path)
    assert ci.check_hydroweb(txt, zdir, rec) == []
    txt.write_bytes(b"edited series\n")  # extracted file no longer the zip member
    errs = ci.check_hydroweb(txt, zdir, rec)
    assert errs and "differs" in errs[0]


def test_hydroweb_zip_checksum(tmp_path):
    txt, zdir, rec = _hydroweb(tmp_path)
    with zipfile.ZipFile(zdir / "abc.zip", "a") as zf:
        zf.writestr("extra.txt", b"x")
    errs = ci.check_hydroweb(txt, zdir, rec)
    assert errs and "Hydroweb zip" in errs[0]


def test_grids_checksums(tmp_path):
    bathy, prov = tmp_path / "bathymetry", tmp_path / "prov"
    bathy.mkdir()
    prov.mkdir()
    for stem, fname in ci.GRIDS.items():
        (bathy / fname).write_bytes(stem.encode())
        (prov / f"{stem}.json").write_text(json.dumps({"subset_sha256": sha(bathy / fname)}), encoding="utf-8")
    assert ci.check_grids(bathy, prov) == []
    (bathy / "gebco_2026_ncaspian.nc").write_bytes(b"other")
    errs = ci.check_grids(bathy, prov)
    assert len(errs) == 1 and "gebco_2026" in errs[0]
    (bathy / "etopo1_ncaspian.tif").unlink()
    assert any("ee-etopo1" in e for e in ci.check_grids(bathy, prov))


# ----------------------------------------------------------------------------- scene inventory

def _scenes() -> pd.DataFrame:
    return pd.DataFrame({
        "sensor": ["TM_L5", "TM_L5", "MSI_S2"],
        "collection": ["LANDSAT/LT05/C02/T1_L2", "LANDSAT/LT05/C02/T1_L2", "COPERNICUS/S2_SR_HARMONIZED"],
        "system:index": ["a", "b", "c"],
        "time": pd.to_datetime(["1990-05-01 06:30:00.125", "1991-07-02 06:31:00.250", "2020-01-03 07:40:00.500"]),
        "year": [1990, 1991, 2020],
        "month": [5, 7, 1],
        "cloud_scene_pct": [10.0, 70.0, 50.0],
        "WRS_PATH": [164.0, 165.0, None],
        "WRS_ROW": [28.0, 29.0, None],
        "MGRS_TILE": [None, None, "39TVL"],
    })


def _inventory(tmp_path, scenes: pd.DataFrame):
    """Write scenes.csv as scripts/a2_scene_inventory.py does, and the committed
    records computed by summarise_scenes."""
    csv = tmp_path / "scenes.csv"
    scenes.to_csv(csv, index=False)
    inv = tmp_path / "inventory"
    inv.mkdir()
    meta, year_csv, month_csv = ci.summarise_scenes(ci.read_scenes(csv))
    meta = {"generated": "2026-10-20T00:00:00+00:00", "end_date_exclusive": "2026-10-21", **meta}
    (inv / "inventory_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    (inv / "summary_sensor_year.csv").write_text(year_csv, encoding="utf-8")
    (inv / "summary_sensor_month.csv").write_text(month_csv, encoding="utf-8")
    return csv, inv


def test_summary_matches_hand_count(tmp_path):
    csv = tmp_path / "scenes.csv"
    _scenes().to_csv(csv, index=False)
    meta, year_csv, _ = ci.summarise_scenes(ci.read_scenes(csv))
    assert meta["n_scenes_total"] == 3
    assert meta["n_by_sensor"] == {"MSI_S2": 1, "TM_L5": 2}
    assert meta["n_cloud_below_60_by_sensor"] == {"MSI_S2": 1, "TM_L5": 1}
    assert meta["landsat_path_rows"] == ["164/028", "165/029"]
    assert meta["n_s2_tiles"] == 1
    assert meta["first_by_sensor"]["TM_L5"].startswith("1990-05-01 06:30:00.125")
    assert meta["last_by_sensor"]["TM_L5"].startswith("1991-07-02 06:31:00.25")
    assert year_csv.splitlines()[0] == "sensor,year,cc0_20,cc20_40,cc40_60,cc60_80,cc80_100,total"


def test_scene_summary_detects_a_changed_scene_table(tmp_path):
    csv, inv = _inventory(tmp_path, _scenes())
    assert ci.check_scene_summary(csv, inv) == []
    s = _scenes()
    s.loc[1, "cloud_scene_pct"] = 30.0  # one scene moves below the 60 % cloud cut
    s.to_csv(csv, index=False)
    errs = ci.check_scene_summary(csv, inv)
    assert any("n_cloud_below_60_by_sensor" in e for e in errs)
    assert any("summary_sensor_year.csv" in e for e in errs)


def test_record_then_check_scenes(tmp_path):
    csv, inv = _inventory(tmp_path, _scenes())
    rec = tmp_path / "scenes_csv.json"
    assert ci.check_scenes(csv, inv, rec) and "no checksum record" in ci.check_scenes(csv, inv, rec)[0]
    assert ci.record_scenes(csv, inv, rec) == []
    r = json.loads(rec.read_text(encoding="utf-8"))
    assert r["sha256"] == sha(csv) and r["n_rows"] == 3
    assert ci.check_scenes(csv, inv, rec) == []
    csv.write_text(csv.read_text(encoding="utf-8").replace("TM_L5,LANDSAT", "TM_L5 ,LANDSAT", 1), encoding="utf-8")
    errs = ci.check_scenes(csv, inv, rec)
    assert errs and "SHA-256" in errs[0]


def test_record_refused_when_numbers_did_not_come_from_this_file(tmp_path):
    csv, inv = _inventory(tmp_path, _scenes())
    _scenes().iloc[:2].to_csv(csv, index=False)  # a different scene table
    rec = tmp_path / "scenes_csv.json"
    errs = ci.record_scenes(csv, inv, rec)
    assert errs and any("NOT recorded" in e for e in errs)
    assert not rec.exists()


# ----------------------------------------------------------------------------- the real inputs

HAVE_RAW = all(p.exists() for p in (ci.DEFAULTS["dahiti"], ci.DEFAULTS["hydroweb"], ci.DEFAULTS["scenes"],
                                     ci.DEFAULTS["bathy"]))


@pytest.mark.skipif(not HAVE_RAW, reason="raw inputs not acquired on this machine (fresh clone)")
def test_local_raw_inputs_match_their_records():
    # also checks that summarise_scenes reproduces what a2_scene_inventory.py wrote
    assert ci.main([]) == 0
