"""Shared set-up of the rev_n revision scripts (PREREG Amendment 18 item 10). Not an
entry point.

Everything is read from the registered runs and their caches; nothing is refitted:
- the hindcast run (results/hindcast/main/run.json): grid, products, shape constants,
  the vertical correction;
- the static layers and the record-low envelope as scripts/f1_erl_display.py caches them;
- the uncorrected flood-level maps (hindcast cache) and the registered fused maps
  (`linear_hold`, H3 cache; scripts/a8_wdpa_exposure.fused_cache);
- the H3 run (results/emergence/main/run.json): the 2026 level and the scenarios.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import caspian  # noqa: F401
import numpy as np
import pandas as pd

from caspian import exposure as ex
from caspian import products
from caspian.config import INTERIM, RAW, RESULTS
from caspian.geography import SECTOR_ORDER

ROOT = Path(__file__).resolve().parents[1]
OUT = RESULTS / "revision" / "n"
WORK = INTERIM / "revision_n"
DAHITI = RAW / "dahiti" / "dahiti_39_caspian_sea_water_level.csv"
CORR_NO_CONSTANT = RESULTS / "study_area" / "geoid_correction_no_constant.npz"
VALIDATED = ("west", "interfluve", "northeast", "open_shelf")      # A18 item 1
REGIONS = list(SECTOR_ORDER) + ["domain", "validated"]
LABEL = {"gebco_2026": "GEBCO_2026", "etopo_2022": "ETOPO 2022", "srtm15p_v2.7": "SRTM15+",
         "gebco_2022": "GEBCO_2022", "etopo1": "ETOPO1"}
KEY = {"gebco_2026": "gebco2026", "etopo_2022": "etopo2022", "srtm15p_v2.7": "srtm15p",
       "gebco_2022": "gebco2022", "etopo1": "etopo1"}


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, m)
    spec.loader.exec_module(m)
    return m


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def sha_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def to_regions(arr) -> np.ndarray:
    """[7 sectors, ...] -> [7 sectors + domain + validated, ...] (sums)."""
    a = np.asarray(arr)
    dom = a.sum(axis=0, keepdims=True)
    idx = [SECTOR_ORDER.index(s) for s in VALIDATED]
    val = a[idx].sum(axis=0, keepdims=True)
    return np.concatenate([a, dom, val], axis=0)


def setup(tile: int = 2048, fused: bool = True, heights: bool = False, fused_form: str = "linear_hold") -> dict:
    """The registered inputs (module docstring) as a dict; ``fused_form`` "constant" reads the
    constant-form fused maps instead."""
    f1 = load_script("f1_erl_display")
    wd = load_script("a8_wdpa_exposure")
    a7 = f1.a7
    hrun = json.loads((RESULTS / "hindcast" / "main" / "run.json").read_text(encoding="utf-8"))
    erun = json.loads((RESULTS / "emergence" / "main" / "run.json").read_text(encoding="utf-8"))
    transform, shape = f1.caspian_grid(hrun)
    static = f1.static_main("main", transform, shape, tile)
    if int(static["envelope"].sum()) != int(hrun["envelope"]["cells"]):
        raise SystemExit("the record-low envelope differs from the hindcast run's")
    lab_in = np.where(np.asarray(static["domain"], bool), np.asarray(static["sector"]), 0).astype(np.uint8)
    names = list(hrun["products"])
    F, fin = f1.flood_maps(names, hrun, {"sea_seed": static["sea_seed"]}, transform, shape, tile)
    out = {"f1": f1, "a7": a7, "wd": wd, "hrun": hrun, "erun": erun, "transform": transform, "shape": shape,
           "static": static, "lab_in": lab_in, "names": names, "F": F, "finite": fin,
           "c": {p: float(hrun["shape_offsets"][p]["offset_m"]) for p in names},
           "L26": round(float(erun["level_2026"]["level_m"]), 4),
           "scenarios": [float(d) for d in erun["scenarios_m"]], "K": len(SECTOR_ORDER), "tile": tile}
    if fused:
        out["fused_form"] = fused_form
        if fused_form == "composite":           # composite observed baseline (scripts/rev_n_baseline.py)
            out["fused_files"] = {p: WORK / "composite_baseline.npy" for p in names}
        else:
            out["fused_files"] = {p: wd.fused_cache(fused_form, p) for p in names}
        out["Ffused"] = {p: np.load(out["fused_files"][p], mmap_mode="r") for p in names}
    if heights:
        corr, corr_rec = products.correction(f1.CORRECTION, transform, shape)
        out["corr"], out["corr_rec"] = corr, corr_rec
        out["raw"] = {p: products.heights(p, transform, shape, hrun.get("resampling", "bilinear")) for p in names}
        out["z"] = {p: products.Corrected(out["raw"][p], corr) for p in names}
    return out


def level_2026_running_median() -> dict:
    d = pd.read_csv(DAHITI, sep=";")
    return ex.reference_level(d["datetime"], d["wse"], 2026, "latest_running_median")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)),
                    encoding="utf-8")
