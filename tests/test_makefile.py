"""Makefile: `make all` rebuilds from raw inputs in dependency order, checks every raw
input against its acquisition record first, never runs a network, credential or Earth
Engine step, and stops with the acquiring target's name when a raw input is missing
(review finding 19). Dry runs only (make -n).

Every case supplies its own stand-in raw files, so the result does not depend on
whether data/ exists on this machine (it does not in a fresh clone)."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MAKE = shutil.which("make")
pytestmark = pytest.mark.skipif(MAKE is None, reason="GNU make not on PATH (it is in the caspian env)")

ACQUISITION = ["p0_dahiti_download.py", "a4_hydroweb_download.py", "a5_swot_download.py", "a2_scene_inventory.py",
               "a7_get_", "a8_osm_exposure.py", "a2_zone.py", "a3_stage1_thresholds.py", "a5_era5_winds.py",
               "a2_eecu_pilot", "--record-scenes"]
GRID_FILES = ("gebco_2022_ncaspian.nc", "gebco_2026_ncaspian.nc", "etopo_2022_ncaspian.nc",
              "srtm15p_v2.7_ncaspian.nc", "etopo1_ncaspian.tif")


def raw_inputs(tmp: Path, grids: bool = True) -> dict[str, str]:
    """Stand-in files for every raw input `make all` reads (contents irrelevant to make -n)."""
    tmp.mkdir(parents=True, exist_ok=True)
    over = {}
    for name in ("DAHITI", "HYDROWEB", "SCENES"):
        f = tmp / f"{name.lower()}.txt"
        f.write_text("x", encoding="utf-8")
        over[name] = f.as_posix()
    bathy = tmp / "bathymetry"
    bathy.mkdir(exist_ok=True)
    if grids:
        for fn in GRID_FILES:
            (bathy / fn).write_text("x", encoding="utf-8")
    over["BATHY"] = bathy.as_posix()
    over["VERIFIED"] = (tmp / "verified.stamp").as_posix()
    return over


def make(*args, over: dict[str, str] | None = None):
    cmd = [MAKE, "--no-print-directory", *args, *(f"{k}={v}" for k, v in (over or {}).items())]
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)


def all_commands(tmp: Path) -> list[str]:
    r = make("-n", "-B", "all", over=raw_inputs(tmp))
    assert r.returncode == 0, r.stderr
    text = re.sub(r"\\\n\s*", " ", r.stdout)  # join recipe continuation lines
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def test_all_runs_every_step_in_dependency_order(tmp_path):
    cmds = all_commands(tmp_path)

    def pos(pattern):
        hits = [i for i, c in enumerate(cmds) if re.search(pattern, c)]
        assert hits, f"{pattern} not run by make all:\n" + "\n".join(cmds)
        return hits[0]

    assert pos(r"a4_level_crosscheck\.py 20$") < pos(r"a4_scene_levels\.py")
    assert pos(r"a4_scene_levels\.py") < pos(r"build_metrics\.py")
    assert pos(r"export_params\.py") < pos(r"build_metrics\.py")
    assert pos(r"p0_dahiti_qc\.py") < pos(r"build_metrics\.py")
    assert pos(r"a0_study_area\.py") < pos(r"build_metrics\.py")
    assert pos(r"a0_sector_edges\.py") < pos(r"build_metrics\.py")
    assert pos(r"build_metrics\.py") < pos(r"make_macros\.py") < pos(r"tectonic .*main\.tex")
    assert pos(r"fetch_bibtex\.py --check") < pos(r"tectonic .*main\.tex")
    # raw inputs are checked against their acquisition records before anything reads them
    check = pos(r"check_inputs\.py .*--stamp")
    for step in (r"p0_dahiti_qc\.py", r"a0_sector_edges\.py", r"a4_level_crosscheck\.py", r"a4_scene_levels\.py",
                 r"build_metrics\.py", r"qc_gebco_tid\.py"):
        assert check < pos(step), step


def test_input_check_covers_every_raw_input(tmp_path):
    over = raw_inputs(tmp_path)
    cmd = next(c for c in all_commands(tmp_path) if "check_inputs.py" in c)
    for flag, var in (("--dahiti", "DAHITI"), ("--hydroweb", "HYDROWEB"), ("--scenes", "SCENES"), ("--bathy", "BATHY")):
        assert f"{flag} {over[var]}" in cmd, (flag, cmd)


def test_all_never_runs_an_acquisition_step(tmp_path):
    out = "\n".join(all_commands(tmp_path))
    for s in ACQUISITION:
        assert s not in out, s
    assert not re.search(r"fetch_bibtex\.py\s*$", out, re.M)  # the network refs build


def _writers(stem: str) -> list[str]:
    """Scripts (scripts/, figures/) whose source names this outputs/qc stem."""
    out = []
    for p in sorted([*(ROOT / "scripts").glob("*.py"), *(ROOT / "figures").glob("*.py")]):
        t = p.read_text(encoding="utf-8")
        if re.search(r"""["']""" + re.escape(stem) + r"""["']""", t) or any(
                stem.startswith(pre) and len(pre) >= 8 for pre in re.findall(r"""f["']([^"'{]+)\{""", t)):
            out.append(p.relative_to(ROOT).as_posix())
    return out


def test_every_committed_qc_figure_is_rebuilt_by_make_all(tmp_path):
    # outputs/qc/gebco_2026_tid.{pdf,png} was committed with no generating script
    cmds = "\n".join(all_commands(tmp_path))
    stems = sorted({p.stem for p in (ROOT / "outputs" / "qc").glob("*") if p.suffix in (".pdf", ".png", ".tif")})
    assert stems
    for stem in stems:
        w = _writers(stem)
        assert w, f"outputs/qc/{stem}: no script in scripts/ or figures/ writes it"
        assert any(s in cmds for s in w), f"outputs/qc/{stem}: written by {w}, none of which `make all` runs"


@pytest.mark.parametrize("var,target", [("SCENES", "ee-scene-inventory"), ("HYDROWEB", "hydroweb-download"),
                                        ("DAHITI", "dahiti-download")])
def test_missing_raw_input_names_its_target(tmp_path, var, target):
    over = raw_inputs(tmp_path)  # every other input present, so this one is the first to fail
    over[var] = (tmp_path / "absent.csv").as_posix()
    r = make("-n", "all", over=over)
    assert r.returncode != 0
    assert "missing raw input" in r.stderr and target in r.stderr and "absent.csv" in r.stderr, r.stderr


def test_missing_bathymetry_grid_names_its_target(tmp_path):
    over = raw_inputs(tmp_path, grids=False)
    r = make("-n", "all", over=over)
    assert r.returncode != 0
    assert "missing raw input" in r.stderr and over["BATHY"] in r.stderr, r.stderr
    assert "bathymetry-download" in r.stderr or "ee-etopo1" in r.stderr


@pytest.mark.parametrize("target", ["ee-scene-inventory", "ee-etopo1", "ee-inputs"])
def test_earth_engine_targets_refuse_without_ee_ok(target):
    r = make("-n", target)
    assert r.returncode != 0 and "EE_OK=1" in r.stderr


def test_scene_inventory_records_its_checksum():
    r = make("-n", "ee-scene-inventory", "EE_OK=1", "SCENES=x/scenes.csv")
    assert r.returncode == 0, r.stderr
    cmds = [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
    i = next(i for i, c in enumerate(cmds) if "a2_scene_inventory.py" in c)
    assert any("check_inputs.py --record-scenes --scenes x/scenes.csv" in c for c in cmds[i + 1:]), cmds


def test_help_runs():
    r = make("help")
    assert r.returncode == 0, r.stderr
    assert "check-inputs" in r.stdout and "qc-figures" in r.stdout


def test_batch_stage_needs_step():
    r = make("-n", "ee-stage1", "EE_OK=1")
    assert r.returncode != 0 and "STEP=" in r.stderr


def test_metrics_prerequisites_cover_every_build_metrics_input():
    src = (ROOT / "tools" / "build_metrics.py").read_text(encoding="utf-8")
    read = {f"results/{p}" for p in re.findall(r"""load\(\s*["']([^"']+)["']""", src)}
    read |= {"refs/" + p for p in re.findall(r"""ROOT\s*/\s*"refs"\s*/\s*"([^"]+)\"""", src)}
    assert read, "parsed no inputs from tools/build_metrics.py"
    db = make("-p", "-n", "-q", "help").stdout
    line = next(ln for ln in db.splitlines() if ln.startswith("results/metrics.json:"))
    prereqs = set(line.split(":", 1)[1].split("|")[0].split())
    assert read <= prereqs, f"not prerequisites of results/metrics.json: {sorted(read - prereqs)}"
