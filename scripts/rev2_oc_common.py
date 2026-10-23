"""Shared set-up of the Amendment 19 item 5 and item 9 scripts (D-148). Not an entry point.

- ``setup``: the hindcast grid and run record, the hindcast's own static layers and record-low
  envelope (scripts/f1_erl_display.static_main cache), the 43 (epoch, level) units of the
  primary area fit, and the primary floor Q per sector and level.
- ``prep_obs``: per unit, the observed extent exactly as scripts/a7_hindcast.py builds it
  (observed_raw with the record-low envelope, classify), cached as the observed state
  (int8: 1 water, 0 land, -1 unknown, outside states folded in, caspian.rev_r.obs_states) and
  its waterline cells (data/interim/revision2/oc/obs/).
- ``Obs``: one unit's cache, with a k-d tree of its whole-grid waterline.
- ``audit_levels``: per level bin and validated sector, D pooled over the bin's epochs from
  per-unit distance codes, the evaluable flag and Q; ``flags`` applies the H1 rule.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import caspian  # noqa: F401
import numpy as np
import pandas as pd

from caspian import products, rev2_oc as oc, rev_r
from caspian.config import INTERIM, RESULTS
from caspian.geography import SECTOR_ORDER

ROOT = Path(__file__).resolve().parents[1]
OUT = RESULTS / "revision2" / "oc"
WORK = INTERIM / "revision2" / "oc"
OBS = WORK / "obs"
TAG = "main"
TILE = 2048
SEED = 20261023
VALIDATED = ["west", "interfluve", "northeast", "open_shelf"]
VIDS = [SECTOR_ORDER.index(s) + 1 for s in VALIDATED]
NAME = {k + 1: s for k, s in enumerate(SECTOR_ORDER)}
PRODUCTS = ["gebco_2026", "gebco_2022", "etopo_2022", "etopo1", "srtm15p_v2.7"]
TOP_DEM_BIN = 41                    # D-116: highest E2 bin; the DEM holds no contour above


def load_script(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")


def seeds():
    """Item 5 and item 9 streams (D-148): SeedSequence(20261023).spawn(2)."""
    return np.random.SeedSequence(SEED).spawn(2)


# --- set-up -----------------------------------------------------------------------------------

def static_paths():
    return {k: WORK / f"static_{k}.npy" for k in ("sector", "sea_seed", "outside", "domain")}


def setup(load_static: bool = True) -> dict:
    """Grid, units, levels, floor; with ``load_static`` the static layers as memory maps
    (written on first use from the f1_erl_display cache)."""
    a7 = load_script("a7_hindcast")
    hrun = json.loads((RESULTS / "hindcast" / TAG / "run.json").read_text(encoding="utf-8"))
    so = json.loads((RESULTS / "hindcast" / TAG / "shape_offsets.json").read_text(encoding="utf-8"))
    units = [(int(u["bin"]), str(u["epoch"])) for u in so["units"]]
    S = {"a7": a7, "hrun": hrun, "units": units, "shape_offsets": so,
         "c": {p: float(v["offset_m"]) for p, v in so["used_in_this_run"].items()},
         "level": {b: a7._bin_level(b) for b, _ in units},
         "transform": hrun["grid"]["transform"], "shape": tuple(hrun["grid"]["shape"])}
    S["bins"] = sorted({b for b, _ in units})
    S["epochs_of"] = {b: sorted(ep for bb, ep in units if bb == b) for b in S["bins"]}
    S["Q"] = primary_floor()
    if load_static:
        paths = static_paths()
        if not all(p.exists() for p in paths.values()):
            f1 = load_script("f1_erl_display")
            transform, shape = f1.caspian_grid(hrun)
            st = f1.static_main(TAG, transform, shape, TILE)
            if int(st["envelope"].sum()) != int(hrun["envelope"]["cells"]):
                raise SystemExit("the record-low envelope differs from the hindcast run's")
            WORK.mkdir(parents=True, exist_ok=True)
            for k, p in paths.items():
                np.save(p, np.asarray(st[k]))
            np.save(WORK / "static_envelope.npy", np.asarray(st["envelope"], bool))
        S["static"] = {k: np.load(p, mmap_mode="r") for k, p in paths.items()}
    return S


def primary_floor() -> pd.DataFrame:
    """Q per (sector, bin): the primary floor, pooled per level (n_noise, p95)."""
    nf = pd.read_csv(RESULTS / "hindcast" / TAG / "noise_floor.csv")
    nz = nf[(nf["kind"] == "pooled_level") & (nf["sector"] != "all")]
    return nz[["sector", "bin", "n", "p95_m"]].rename(columns={"n": "n_noise", "p95_m": "Q_m"}).reset_index(drop=True)


# --- observed extents -------------------------------------------------------------------------

def obs_path(b: int, ep: str) -> Path:
    return OBS / f"unit_b{b:03d}_{ep}"


def prep_obs(S: dict) -> None:
    """Per unit, the observed state and waterline (module docstring); skips units cached."""
    a7 = S["a7"]
    todo = [(b, ep) for b, ep in S["units"] if not obs_path(b, ep).with_suffix(".npz").exists()]
    if not todo:
        return
    transform, shape = S["transform"], S["shape"]
    st = {"outside": np.asarray(S["static"]["outside"]), "sea_seed": np.asarray(S["static"]["sea_seed"])}
    a7.ENVELOPE = np.load(WORK / "static_envelope.npy")
    stage = a7.stage2_index(INTERIM / "stage2" / TAG)
    OBS.mkdir(parents=True, exist_ok=True)
    sector = np.asarray(S["static"]["sector"])
    t0 = time.time()
    for i, (b, ep) in enumerate(todo):
        raw, _ = a7.observed_raw(a7.groups_for(stage, b, [ep]), transform, shape)
        code = a7.classify(raw, st, TILE)
        del raw
        kw, kl = rev_r.obs_states(code, st["outside"])
        ostate = np.where(kw, 1, np.where(kl, 0, -1)).astype(np.int8)
        del kw, kl
        r, c = rev_r.obs_waterline(code, st["outside"])
        del code
        p = obs_path(b, ep)
        np.save(p.with_suffix(".ostate.npy"), ostate)
        np.savez(p.with_suffix(".npz"), rows=r, cols=c, lab=sector[r, c].astype(np.uint8))
        del ostate
        log(f"obs cache: unit {ep} bin {b} ({i + 1}/{len(todo)}), {len(r)} waterline cells, "
            f"{time.time() - t0:.0f} s")


class Obs:
    """One unit's observed state (memory map) and waterline, with a k-d tree."""

    def __init__(self, b: int, ep: str):
        from scipy.spatial import cKDTree
        p = obs_path(b, ep)
        z = np.load(p.with_suffix(".npz"))
        self.rows, self.cols, self.lab = z["rows"], z["cols"], z["lab"]
        self.ostate = np.load(p.with_suffix(".ostate.npy"), mmap_mode="r")
        self.tree = cKDTree(np.c_[self.rows, self.cols].astype(float)) if self.rows.size else None
        self.observed = {s: bool((self.lab == s).any()) for s in VIDS}

    def codes(self, cand: dict, T: float, sector, fin_obs) -> dict:
        """{sector id: |distance| codes} of the prediction at T (``cand``) against this unit,
        for the validated sectors. ``fin_obs``: the prediction's finite mask at the waterline."""
        pr, pc = oc.waterline_at(cand, T)
        ost = np.asarray(self.ostate[pr, pc]) if pr.size else np.empty(0, np.int8)
        lab_p = np.asarray(sector[pr, pc]) if pr.size else np.empty(0, np.uint8)
        return oc.displacement_codes((pr, pc), (self.rows, self.cols), ost, fin_obs, lab_p, self.lab, VIDS,
                                     obs_tree=self.tree)


def observed_levels(S: dict) -> dict:
    """{(bin, sector id): observed in any epoch of the bin}."""
    out = {}
    for b in S["bins"]:
        for ep in S["epochs_of"][b]:
            z = np.load(obs_path(b, ep).with_suffix(".npz"))
            for s in VIDS:
                out[(b, s)] = out.get((b, s), False) or bool((z["lab"] == s).any())
    return out


# --- level table and the rule -----------------------------------------------------------------

def level_frame(S: dict, obs_level: dict) -> pd.DataFrame:
    """One row per (validated sector, bin): observed, Q, n_noise, in the DEM's range."""
    Q = S["Q"].set_index(["sector", "bin"])
    rows = []
    for s in VIDS:
        for b in S["bins"]:
            q = Q.loc[(NAME[s], b)] if (NAME[s], b) in Q.index else None
            rows.append({"sector": NAME[s], "sid": s, "bin": b, "level_m": round(S["level"][b], 4),
                         "observed_level": bool(obs_level[(b, s)]),
                         "n_noise": int(q["n_noise"]) if q is not None else 0,
                         "Q_m": float(q["Q_m"]) if q is not None else np.nan,
                         "in_dem_range": b <= TOP_DEM_BIN})
    return pd.DataFrame(rows)


def flags(lv: pd.DataFrame, D, n, restrict_dem: bool = False):
    """The H1 rule per validated sector. ``D``, ``n``: arrays [..., sector, bin] in the order of
    VIDS and S["bins"] (lv's order). Returns (flagged, n_exceeding, n_evaluable) [..., sector]."""
    ns, nb = len(VIDS), len(lv) // len(VIDS)
    obs = lv["observed_level"].to_numpy(bool).reshape(ns, nb)
    nn = lv["n_noise"].to_numpy(int).reshape(ns, nb)
    Q = lv["Q_m"].to_numpy(float).reshape(ns, nb)
    ev = obs & (np.asarray(n) > 0) & (nn > 0)
    if restrict_dem:
        ev = ev & lv["in_dem_range"].to_numpy(bool).reshape(ns, nb)
    return oc.rule_flags(D, Q, ev)


def pooled_D(parts_by_bin: dict, S: dict):
    """{bin: {sid: [code arrays of the bin's epochs]}} -> D, n arrays [sector, bin]."""
    D = np.full((len(VIDS), len(S["bins"])), np.nan)
    n = np.zeros((len(VIDS), len(S["bins"])), np.int64)
    for j, b in enumerate(S["bins"]):
        for i, s in enumerate(VIDS):
            D[i, j], n[i, j] = oc.pooled_median(parts_by_bin[b][s])
    return D, n


def product_flood(S: dict, p: str):
    """The hindcast's cached flood-level map of product p (corrected heights) and the finite
    mask of its heights, as a boolean memory map cached under WORK."""
    f1 = load_script("f1_erl_display")
    transform, shape = f1.caspian_grid(S["hrun"])
    st = {"sea_seed": np.asarray(S["static"]["sea_seed"])}
    F, _ = f1.flood_maps([p], S["hrun"], st, transform, shape, TILE)
    path = WORK / f"finite_{p}.npy"
    if not path.exists():
        corr, _ = products.correction(f1.CORRECTION, transform, shape)
        hh = products.Corrected(products.heights(p, transform, shape, S["hrun"].get("resampling", "bilinear")), corr)
        tmp = path.with_suffix(".tmp.npy")
        mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=bool, shape=tuple(shape))
        for r0 in range(0, shape[0], 1024):
            mm[r0:r0 + 1024] = np.isfinite(hh[r0:r0 + 1024, :])
        mm.flush()
        del mm
        tmp.replace(path)
    return F[p], np.load(path, mmap_mode="r")
