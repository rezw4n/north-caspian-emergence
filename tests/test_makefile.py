"""Makefile: `make all` rebuilds from raw inputs in dependency order, checks every raw
input against its acquisition record first, never runs a network, credential or Earth
Engine step, and stops with the acquiring target's name when a raw input is missing.
It rebuilds every committed result under results/ except the ones
declared in NOT_REBUILT with their reasons. Dry runs only (make -n).

Every case supplies its own stand-in raw files, so the result does not depend on
whether data/ exists on this machine (it does not in a fresh clone)."""

import os
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
               "a2_eecu_pilot", "--record-scenes", "a8_wdpa_exposure.py fetch",
               # stage 2b's inputs: Earth Engine exports and downloads
               "a6_stage2_export.py", "r18_rblock_export.py", "a6_icesat2.py", "a6_glo30.py download",
               "a6_gedi.py pilot", "a6_gedi.py export", "a8_atl24_fetch.py", "a6_vegetation.py fetch-extents",
               "a7_beyond_state.py", "g1_data.py", "g1_ee.py", "f2_erl_landsat.py", "a6_atl24_search.py"]
GRID_FILES = ("gebco_2022_ncaspian.nc", "gebco_2026_ncaspian.nc", "etopo_2022_ncaspian.nc",
              "srtm15p_v2.7_ncaspian.nc", "etopo1_ncaspian.tif")
# Raw-input variables with stand-ins of their own above, or read only by a recipe.
OWN_STANDINS = {"DAHITI", "HYDROWEB", "SCENES", "CASPCOM", "BATHY", "VERIFIED", "HYDROWEB_DIR"}


def raw_inputs(tmp: Path, grids: bool = True) -> dict[str, str]:
    """Stand-in files for every raw input `make all` reads (contents irrelevant to make -n).
    The inputs of stage 2b that acquisition steps write: every other `?= data/...`
    variable of the Makefile points into tmp, and each path `make inputs` then lists there
    gets a stand-in, a file if its name has a suffix, else a folder."""
    tmp.mkdir(parents=True, exist_ok=True)
    over = {}
    for name in ("DAHITI", "HYDROWEB", "SCENES", "CASPCOM"):
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
    mk = (ROOT / "Makefile").read_text(encoding="utf-8")
    roots = []
    for var in re.findall(r"^([A-Z0-9_]+)\s*\?=\s*data/\S+\s*$", mk, re.M):
        if var not in OWN_STANDINS:
            over[var] = (tmp / "acq" / var.lower()).as_posix()
            roots.append(over[var])
    r = make("-s", "inputs", over=over)
    assert r.returncode == 0, r.stderr
    for line in r.stdout.splitlines():
        p = line.strip()
        if not any(p == q or p.startswith(q + "/") for q in roots):
            continue
        if Path(p).suffix:
            Path(p).parent.mkdir(parents=True, exist_ok=True)
            Path(p).write_text("x", encoding="utf-8")
        else:
            Path(p).mkdir(parents=True, exist_ok=True)
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
    assert pos(r"build_metrics\.py") < pos(r"make_macros\.py") < pos(r"tectonic .*main_erl\.tex")
    assert pos(r"fetch_bibtex\.py --check") < pos(r"tectonic .*main_erl\.tex")
    # raw inputs are checked against their acquisition records before anything reads them
    check = pos(r"check_inputs\.py .*--stamp")
    for step in (r"p0_dahiti_qc\.py", r"a0_sector_edges\.py", r"a4_level_crosscheck\.py", r"a4_scene_levels\.py",
                 r"build_metrics\.py", r"qc_gebco_tid\.py"):
        assert check < pos(step), step


# Stage 2b: the main-analysis pipeline, each chain in the order its steps read each other.
PHASE_A_CHAINS = [
    [r"a6_stage2_combine\.py main mainL", r"a6_vegetation\.py main$", r"a7_h1_decision\.py main$"],
    [r"a6_stage2_combine\.py", r"a6_waterlines\.py main ", r"a6_static_rule\.py main", r"a6_dem\.py main",
     r"make_stamp\.py \S+stamp_dem_main", r"a6_emergence\.py main", r"a6_validate_dem\.py main prepare",
     r"a6_validate_dem\.py main blocksize$", r"a6_validate_dem\.py main evaluate$", r"a6_h2_decision\.py main$"],
    [r"a6_validate_dem\.py main blocksize$", r"evaluate --block-size 25000", r"r19_local_runs\.py fold"],
    [r"a6_validate_dem\.py main blocksize$", r"blocksize --population vegetated", r"evaluate --population vegetated"],
    [r"a6_emergence\.py main", r"a6_glo30\.py points main", r"blocksize --sources GLO30", r"evaluate --sources GLO30",
     r"--additional GLO30", r"a6_etopo_sources\.py main GLO30"],
    [r"a6_emergence\.py main", r"a6_gedi\.py points main", r"blocksize --sources GEDI", r"evaluate --sources GEDI$",
     r"--additional GEDI$", r"a6_etopo_sources\.py main GEDI"],
    [r"a6_stage2_combine\.py", r"a7_hindcast\.py main --vertical-correction \S+$", r"a7_h1_decision\.py main$",
     r"a7_sector_shape\.py main", r"a7_h1_decision\.py main_sectorshape", r"r19_local_runs\.py fold"],
    [r"a7_hindcast\.py main .*--no-envelope --out-suffix _noenv", r"a7_h1_decision\.py main_noenv"],
    [r"a6_dem\.py main", r"a7_hindcast\.py main --vertical-correction \S+$", r"a7_level_bands\.py main"],
    [r"a7_hindcast\.py main --vertical-correction \S+$", r"a8_emergence\.py main .*primary\.npz$",
     r"a8_h3_decision\.py main$", r"--sensitivity-form constant", r"a8_h3_decision\.py main_constant",
     r"a8_wdpa_exposure\.py exposure", r"build_metrics\.py"],
    [r"a8_wdpa_exposure\.py layers", r"a8_wdpa_exposure\.py exposure"],
    [r"a6_validate_dem\.py main evaluate$", r"a8_emergence\.py main .*primary\.npz$", r"a8_atl24_check\.py"],
    [r"g1_static\.py", r"g1_plan\.py levels", r"g1_plan\.py thresholds", r"g1_plan\.py plan great_salt_lake main",
     r"g1_hindcast\.py", r"g1_decide\.py", r"f1_erl_display\.py gsl", r"build_metrics\.py"],
    [r"g1_static\.py", r"g1_lineage\.py", r"build_metrics\.py"],
    [r"make_stamp\.py --clean \S+stage2_main_north_masked", r"g1_hindcast\.py"],
    [r"a7_hindcast\.py main --vertical-correction \S+$", r"r19_local_runs\.py h1-e2e3", r"r20_verdicts\.py"],
    [r"r08_blocking\.py splits", r"r19_local_runs\.py fold", r"r20_verdicts\.py", r"r22_report_table\.py"],
    [r"make_stamp\.py --clean \S+main__Rnear", r"rblock --alt fixed_mndwi0", r"rblock --alt awei_sh",
     r"r20_verdicts\.py"],
    [r"bw020 --sources ATL08", r"--check r07_interpolation_bw020", r"r20_verdicts\.py"],
    [r"vegetation --mode as_dry", r"--check r09_vegetation --variant main__veg_as_dry", r"r20_verdicts\.py"],
    [r"a6_stage2_combine\.py", r"r21_seam_gaps\.py", r"build_metrics\.py"],
    [r"r20_verdicts\.py", r"build_metrics\.py", r"make_macros\.py"],
    [r"a8_emergence\.py main .*primary\.npz$", r"f1_erl_display\.py caspian", r"build_metrics\.py"],
]


def test_all_runs_the_phase_a_pipeline_in_dependency_order(tmp_path):
    cmds = all_commands(tmp_path)

    def pos(pattern):
        hits = [i for i, c in enumerate(cmds) if re.search(pattern, c)]
        assert hits, f"{pattern} not run by make all:\n" + "\n".join(cmds)
        return hits[0]

    for chain in PHASE_A_CHAINS:
        at = [pos(p) for p in chain]
        assert at == sorted(at), [(p, i) for p, i in zip(chain, at)]
    check = pos(r"check_inputs\.py .*--stamp")
    assert check < pos(r"a6_stage2_combine\.py") and check < pos(r"g1_static\.py")
    # the heavy runs each once
    for heavy in (r"a7_hindcast\.py main", r"a7_sector_shape\.py", r"a8_emergence\.py main", r"h1-e2e3"):
        assert len([c for c in cmds if re.search(heavy, c)]) == {r"a7_hindcast\.py main": 2,
                                                                 r"a8_emergence\.py main": 2}.get(heavy, 1), heavy


def test_missing_phase_a_input_names_its_acquisition(tmp_path):
    over = raw_inputs(tmp_path)
    shutil.rmtree(Path(over["STAGE2_DIR"]) / "mainL")
    r = make("-n", "all", over=over)
    assert r.returncode != 0
    assert "missing input" in r.stderr and "mainL" in r.stderr and "a6_stage2_export.py" in r.stderr, r.stderr


# Committed files under results/ that `make all` does not rebuild, each with the reason.
# Everything else under results/ must be a target that `all` reaches, or a side output of one
# (SIDE_OUTPUTS).
NOT_REBUILT = {
    "results/inventory/*": "acquisition records: scene inventory, Earth Engine task records, ledgers and budgets",
    "results/provenance/*": "acquisition records (checksums of the raw files), except stage2_main_combine.json",
    "results/generality/great_salt_lake/extents.json": "Earth Engine record (scripts/g1_ee.py extents)",
    "results/generality/great_salt_lake/scene_inventory.json": "Earth Engine record (scripts/g1_ee.py scenes)",
    "results/generality/great_salt_lake/run.json": "hand-written run record of the generality test",
    "results/generality/great_salt_lake/status.json": "hand-written status record",
    "results/generality/lake_urmia/status.json": "hand-written status record: not run",
    "results/hindcast/main/d089_rerun_comparison.json": "one-off comparison with the superseded first run",
    "results/levels/projection_pathways.csv": "values transcribed from the literature (scripts/a8_check_pathways.py)",
    "results/levels/swot_block_levels.csv": "SWOT reduction during download (scripts/a5_swot_sector_levels.py)",
    "results/levels/setup_model.json": "design step before the stage-2 export (scripts/a5_setup_model.py)",
    "results/levels/screening_sensitivity.json": "design step before the stage-2 export (scripts/a6_screening.py)",
    "results/qc/threshold_rule_summary.json": "design step before the stage-2 export (scripts/a3_thresholds.py)",
    "results/qc/snow_rules_run2.json": "design step before the stage-2 export (scripts/a3_snow_rules.py)",
    "results/qc/stage1_run1_monthly.csv": "stage-1 QC record with no script in the repository",
    "results/qc/stage1_scale_check.json": "stage-1 QC record with no script in the repository",
    "results/qc/zone_v6_seaward_ring_check.csv": "zone QC record with no script in the repository",
    "results/study_area/extents_v6_250m.npz": "Earth Engine (scripts/a6_vegetation.py fetch-extents)",
    "results/study_area/sea_mask_1km.npz": "Earth Engine (scripts/a5_sea_mask.py)",
    "results/study_area/beyond_state_250m.npz": "Earth Engine (scripts/a7_beyond_state.py build)",
    "results/study_area/zone_v6_tiles.geojson": "Earth Engine (scripts/a2_zone.py)",
    "results/study_area/zone_v6_cover_buffered.geojson": "design step before the stage-2 export (a6_cover_buffer.py)",
    "results/study_area/blocks_*km.csv": "spatial blocks frozen before any evaluation (scripts/a6_blocks.py)",
    "results/study_area/geoid_*.npz": "geoid grids (scripts/a7_geoid_difference.py, a7_geoid_corrections.py)",
    "results/robustness/cost_plan.json": "Earth Engine cost plan (scripts/r17_cost_plan.py)",
    "results/robustness/r18_rblock_export/plan.json": "Earth Engine export plan (scripts/r18_rblock_export.py plan)",
    "results/revision/g/root2023_retrieval.json": "network snapshot: the ScienceBase retry of 2026-10-22 "
                                                  "(make revision-g-fetch)",
    "results/revision/g/dep3_provenance.json": "network snapshot: USGS 3DEP metadata read on 2026-10-22 "
                                               "(make revision-g-fetch)",
    "results/revision/g/etopo_lake_passages.json": "network snapshot: ETOPO 2022 and HydroLAKES documents read on "
                                                   "2026-10-22 (make revision-g-fetch)",
    "results/revision/g/citation_requests.md": "hand-written document: citation requests of revision analysis G",
    "results/revision/v/citation_requests.md": "hand-written document: citation requests of revision analysis V",
    "results/revision/v/caspcom_bulletin31_2025.csv": "values transcribed with their quote from CASPCOM Bulletin 31 "
                                                      "(input of scripts/rev_v_levels.py)",
}
# Files a rebuilt target's recipe also writes (prefix -> that target).
SIDE_OUTPUTS = {
    "results/study_area/sectors.geojson": "results/study_area/sector_areas.json",
    "results/robustness/r06_datums/": "results/robustness/r06_datums_validation/summary.json",
}


def _make_graph(over: dict[str, str]) -> tuple[dict[str, list[str]], set[str]]:
    """Prerequisites (normal and order-only) of every file in make's database, plus the
    other members of its grouped target (one recipe makes them all), and the files that
    have a recipe."""
    db = make("-p", "-n", "-q", "help", over=over).stdout
    db = db[db.index("\n# Files"):]
    deps, ruled = {}, set()
    for entry in db.split("\n\n"):
        lines = [ln for ln in entry.splitlines() if ln.strip()]
        if not lines or lines[0].startswith("# Not a target:"):
            continue
        m = re.match(r"^((?:[A-Za-z]:)?[^:#\s][^:]*?):{1,2}(?:\s|$)(.*)$", lines[0])
        if not m:
            continue
        also = [ln.split(":", 1)[1].split() for ln in lines if ln.startswith("#  Also makes:")]
        deps[m.group(1)] = m.group(2).replace("|", " ").split() + [f for a in also for f in a]
        recipe = [ln for ln in lines if ln.startswith("\t")]
        guard = recipe and all(re.match(r"\t\$\(call need(_acq)?,", ln) for ln in recipe)   # raw-input check
        if any("recipe to execute" in ln for ln in lines) and not guard:
            ruled.add(m.group(1))
    return deps, ruled


def test_every_committed_result_is_rebuilt_or_declared(tmp_path):
    from fnmatch import fnmatch

    deps, ruled = _make_graph(raw_inputs(tmp_path))
    seen, todo = set(), ["all"]
    while todo:
        t = todo.pop()
        if t not in seen:
            seen.add(t)
            todo += deps.get(t, [])
    rebuilt = seen & ruled
    tracked = subprocess.run(["git", "ls-files", "results"], cwd=ROOT, capture_output=True, text=True).stdout.split()
    assert tracked
    missing = []
    for f in tracked:
        if f in rebuilt or any(fnmatch(f, p) for p in NOT_REBUILT):
            continue
        side = next((t for p, t in SIDE_OUTPUTS.items() if f == p or (p.endswith("/") and f.startswith(p))), None)
        if side is None:
            m = re.match(r"(results/robustness/[^/]+)/", f)          # a check's tables beside its summary
            side = f"{m.group(1)}/summary.json" if m else None
        if side is None or side not in rebuilt:
            missing.append(f)
    assert not missing, (f"{len(missing)} committed results neither rebuilt by make all nor declared: {missing[:20]}. "
                         "Give each a rule in stage 2b of the Makefile, or declare it in NOT_REBUILT with the reason.")
    # the declared ones really are not rebuilt (except the rebuilt provenance record of the combine step)
    stale = [f for f in tracked if f in rebuilt and any(fnmatch(f, p) for p in NOT_REBUILT)]
    assert stale == ["results/provenance/stage2_main_combine.json"], stale


def test_make_stamp_lists_outputs_and_refuses_unsafe_cleans(tmp_path, monkeypatch):
    import importlib.util

    spec = importlib.util.spec_from_file_location("make_stamp", ROOT / "tools" / "make_stamp.py")
    ms = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ms)
    out = tmp_path / "out"
    out.mkdir()
    (out / "dem_0_0.tif").write_bytes(b"abc")
    (out / "notes.txt").write_text("x", encoding="utf-8")
    ms.stamp(tmp_path / ".stamp", [out], "dem_*.tif")
    assert (tmp_path / ".stamp").read_text(encoding="utf-8") == "dem_0_0.tif\t3\n"
    with pytest.raises(SystemExit):                       # the step wrote nothing
        ms.stamp(tmp_path / ".stamp2", [out], "emergence_*.tif")
    assert not (tmp_path / ".stamp2").exists()
    with pytest.raises(SystemExit):                       # not under data/interim or data/processed
        ms.clean([tmp_path / "out"])
    assert out.exists()
    interim = tmp_path / "data" / "interim"
    monkeypatch.setattr(ms, "CLEANABLE", (os.path.normcase(os.path.abspath(interim)),))
    victim = interim / "derived"
    (victim / "sub").mkdir(parents=True)
    (victim / "sub" / "tile.npz").write_bytes(b"x")
    ms.clean([victim, interim / "absent"])
    assert not victim.exists()
    with pytest.raises(SystemExit):                       # the root itself is never removed
        ms.clean([interim])


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
