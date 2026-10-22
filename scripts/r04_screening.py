"""Robustness check 4: surge screening threshold and setup-model form, plus the
Komsomolets proxy.

Commands
  delta-plan       Local plan builder for an Earth Engine delta export (the screening machinery,
                   robustness.screening_delta): alt = main + ADD - SUB, so only the
                   scene-sector pairs on which the two screening rules disagree are
                   exported. Alternatives (--alt):
                     tau:<m>                 another threshold (the next looser and
                                             stricter candidates)
                     komsomolets_unscreened  Komsomolets not screened
                     unknown_included        NaN setup counts as calm
                     setup:<csv>             another setup model's per-scene predictions
                                             (columns scene, setup_<sector>_m), same tau
                   Writes data/interim/stage2/plan_delta_<name>_{add,sub}.csv for
                   scripts/a6_stage2_export.py upload, and the cost shares of the two
                   exports (robustness.delta_cost_share) to results/robustness/
                   r04_screening/<name>_delta.json.
  variant          After the ADD/SUB exports: the variant tag (r00_counts --add/--sub),
                   then the downstream steps and the paired comparison.
  komsomolets      Local, from the screening-sensitivity export on training blocks (tag
                   sensB): for the Komsomolets blocks, per block, epoch and bin,
                   the odd/even waterline displacement, the valid area and the waterline
                   length with and without screening at the main tau.
  setup-correction Setup correction instead of screening, to first order (the planned
                   fallback for screening): each contour point's height + the mean predicted setup of
                   its group's included acquisitions in its sector (robustness.setup_shift),
                   then DEM, emergence and the paired validation.

Usage:
  python scripts/r04_screening.py delta-plan --plan <main plan csv> --tau-main <m> --alt tau:0.20 --name tau020
  python scripts/r04_screening.py variant --main <tag> --name tau020 --add-tag <tag> --sub-tag <tag>
  python scripts/r04_screening.py komsomolets --sens-tag sensB --tau-main <m>
  python scripts/r04_screening.py setup-correction --main <tag> --plan <main plan csv> --tau-main <m>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import caspian  # noqa: F401
import numpy as np
import pandas as pd

from caspian import design, frequency, hindcast, robustness as R
from caspian.config import CRS_EQUAL_AREA, INTERIM, RESULTS
from caspian.geography import SECTOR_ORDER

CHECK = "r04_screening"
OUT = RESULTS / "robustness" / CHECK


def _load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def scene_km2(plan: pd.DataFrame) -> pd.Series:
    """Zone area (km2) of each plan row's path/row or MGRS tile (results/inventory/
    stage2_cost_model.json), the scene weight of the stage-2 cost model."""
    cm = json.loads((RESULTS / "inventory" / "stage2_cost_model.json").read_text(encoding="utf-8"))
    zone = {u["unit"]: u["zone_km2"] for u in cm["units"]}
    unit = [s.split("_")[-1] if se == "MSI_S2" else s.split("_")[1] for s, se in zip(plan["scene"], plan["sensor"])]
    return pd.Series([zone.get(u, np.nan) for u in unit], index=plan.index, dtype=float)


def sector_cover_share() -> dict:
    """Share of the stage-2 cover in each sector (equal-area), for per-sector delta costs."""
    import pyproj
    import shapely
    from caspian.geography import sectors
    fwd = pyproj.Transformer.from_crs("EPSG:4326", CRS_EQUAL_AREA, always_xy=True)
    laea = lambda g: shapely.transform(g, lambda xy: np.column_stack(fwd.transform(xy[:, 0], xy[:, 1])))  # noqa: E731
    cov = R.load_cover(RESULTS / "study_area" / "zone_v6_cover_buffered.geojson", CRS_EQUAL_AREA)
    a = {n: cov.intersection(laea(p.segmentize(0.01))).area for n, (_, p) in sectors().items()}
    tot = sum(a.values())
    return {n: v / tot for n, v in a.items()}


def alt_exclusion(plan: pd.DataFrame, tau_main: float, alt: str) -> np.ndarray:
    if alt.startswith("tau:"):
        return R.exclusion(plan, float(alt[4:]))
    if alt == "komsomolets_unscreened":
        return R.exclusion(plan, tau_main, exempt_sectors=["komsomolets"])
    if alt == "unknown_included":
        return R.exclusion(plan, tau_main, unknown_excluded=False)
    if alt.startswith("setup:"):
        s = pd.read_csv(alt[6:])
        cols = R.setup_columns(s)
        m = plan[["scene"]].merge(s[["scene", *cols]], on="scene", how="left")
        return R.exclusion(m, tau_main, cols=cols)
    raise SystemExit(f"unknown alternative {alt}")


def delta_plan(plan_csv: str, tau_main: float, alt: str, name: str) -> int:
    plan = pd.read_csv(plan_csv, low_memory=False, dtype={"excluded_sectors": str, "sectors": str})
    if "excluded_sectors" in plan.columns:      # a main plan: its own rule must be the tau given
        ex = R.exclusion(plan, tau_main)
        got = [R._codes(r) for r in ex]
        if [str(x) if isinstance(x, str) else "" for x in plan["excluded_sectors"].fillna("")] != got:
            raise SystemExit("--tau-main does not reproduce the plan's excluded_sectors")
    d = R.screening_delta(plan, R.exclusion(plan, tau_main), alt_exclusion(plan, tau_main, alt))
    km2 = scene_km2(plan)
    total = float(km2.sum())
    w = sector_cover_share()
    rec = {"plan": plan_csv, "tau_main_m": tau_main, "alt": alt, "n_add_pairs": d["n_add_pairs"],
           "n_sub_pairs": d["n_sub_pairs"], "n_mixed_acq_sector": d["n_mixed_acq_sector"],
           "add_cost_share": R.delta_cost_share(d["add"], km2, total, w),
           "sub_cost_share": R.delta_cost_share(d["sub"], km2, total, w), "sector_cover_share": w,
           "note": "cost share = scene-weighted zone area of the delta scenes / that of the main plan; "
                   "'upper' for a full-cover export, 'per_sector' for sector-region exports"}
    for k in ("add", "sub"):
        d[k].drop(columns=[c for c in ("excluded_sectors",) if c in d[k].columns]).to_csv(
            INTERIM / "stage2" / f"plan_delta_{name}_{k}.csv", index=False)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}_delta.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(json.dumps(rec, indent=1))
    return 0


BAND = re.compile(r"([wv])_(E[123])_b(\d{3})_p([01])_c(\d)$")


def komsomolets(sens_tag: str, tau_main: float) -> int:
    a6s = _load("a6_screening")
    st = json.loads((RESULTS / "inventory" / f"stage2_{sens_tag}.json").read_text(encoding="utf-8"))
    code = SECTOR_ORDER.index("komsomolets") + 1
    rows = []
    for key, t in st["tiles"].items():
        if int(t.get("sector_code") or 0) != code:
            continue
        f = INTERIM / "stage2" / sens_tag / f"tile_{key}.npz"
        if not f.exists():
            continue
        tr, z = R.load_tile(f)
        pixel = abs(float(tr[0]))
        g = {}
        for k, arr in z.items():
            m = BAND.match(k)
            if m:
                kind, ep, b, p, c = m.groups()
                g.setdefault((ep, int(b), int(p)), {}).setdefault(int(c), {})[kind] = arr
        for label, tau in (("unscreened", np.inf), ("screened", tau_main)):
            keep = set(a6s.classes_for(tau))
            for ep, b in sorted({(e, bb) for e, bb, _ in g}):
                halves, wv = [], []
                for p in (0, 1):
                    cl = {c: (h["w"], h["v"]) for c, h in g.get((ep, b, p), {}).items() if c in keep and "w" in h}
                    if not cl:
                        break
                    w, v = frequency.combine(cl, list(cl))
                    f_ = frequency.water_frequency(w, v)
                    halves.append((f_ >= design.CONTOUR_LEVEL, np.isfinite(f_)))
                    wv.append((w, v))
                if len(halves) < 2:
                    continue
                (wa, va), (wb, vb) = halves
                r = hindcast.noise_floor_displacement(wa, va, wb, vb, pixel)
                v_all = wv[0][1] + wv[1][1]
                ff = frequency.water_frequency(wv[0][0] + wv[1][0], v_all)
                known = np.isfinite(ff)
                wl = hindcast.waterline(known & (ff >= design.CONTOUR_LEVEL), known & (ff < design.CONTOUR_LEVEL))
                rows.append({"block": key, "screening": label, "tau_m": tau, "epoch": ep, "bin": b,
                             "displacement_median_m": r.get("median"), "displacement_n": r.get("n"),
                             "valid_km2": float(known.sum()) * pixel ** 2 / 1e6,
                             "waterline_cells": int(np.asarray(wl).sum())})
    d = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / "komsomolets_sensB.csv", index=False)
    if len(d):
        s = d.groupby("screening").agg(n_block_bins=("bin", "size"), S_m=("displacement_median_m", "median"),
                                       valid_km2=("valid_km2", "sum"), waterline_cells=("waterline_cells", "sum"))
        s.to_csv(OUT / "komsomolets_sensB_summary.csv")
        print(s.to_string())
    return 0


def point_sectors(x, y) -> np.ndarray:
    """Sector of each point (equal-area x, y): the containing sector, else the nearest."""
    import pyproj
    import shapely
    from caspian.geography import sectors
    fwd = pyproj.Transformer.from_crs("EPSG:4326", CRS_EQUAL_AREA, always_xy=True)
    polys = {n: shapely.transform(p.segmentize(0.01), lambda xy: np.column_stack(fwd.transform(xy[:, 0], xy[:, 1])))
             for n, (_, p) in sectors().items()}
    x, y = np.asarray(x, float), np.asarray(y, float)
    out = np.full(x.shape, None, object)
    for n in SECTOR_ORDER:
        out[(out == None) & shapely.contains_xy(polys[n], x, y)] = n  # noqa: E711
    miss = out == None  # noqa: E711
    if miss.any():
        pts = shapely.points(x[miss], y[miss])
        dist = np.column_stack([shapely.distance(polys[n], pts) for n in SECTOR_ORDER])
        out[miss] = np.array(SECTOR_ORDER, object)[dist.argmin(axis=1)]
    return out


def setup_correction(main_tag: str, plan_csv: str, tau_main: float, P, a) -> int:
    plan = pd.read_csv(plan_csv, low_memory=False)
    sh = R.setup_shift(plan, R.exclusion(plan, tau_main))
    pts = pd.read_csv(INTERIM / "waterlines" / main_tag / "points.csv.gz")
    pts["sector_pt"] = point_sectors(pts["x"], pts["y"])
    s = sh.rename(columns={"sector": "sector_pt"})
    m = pts[["epoch", "bin", "sector_pt"]].merge(s, on=["epoch", "bin", "sector_pt"], how="left")
    n_missing = int((~np.isfinite(m["shift_m"].to_numpy(float))).sum())
    s = s.copy()
    s["shift_m"] = s["shift_m"].fillna(0.0)
    full = pts[["epoch", "bin", "sector_pt"]].drop_duplicates().merge(s, on=["epoch", "bin", "sector_pt"], how="left")
    full["shift_m"] = full["shift_m"].fillna(0.0)
    out = R.shift_points(pts, full[["epoch", "bin", "sector_pt", "shift_m"]], on=("epoch", "bin", "sector_pt"))
    variant = f"{main_tag}__setupcorr"
    d = INTERIM / "waterlines" / variant
    d.mkdir(parents=True, exist_ok=True)
    out.drop(columns=["sector_pt"]).to_csv(d / "points.csv.gz", index=False, float_format="%.6f")
    OUT.mkdir(parents=True, exist_ok=True)
    sh.to_csv(OUT / "setup_correction_shifts.csv", index=False)
    (OUT / "setup_correction.json").write_text(json.dumps(
        {"main": main_tag, "tau_main_m": tau_main, "n_points": len(pts), "n_points_without_shift_set_to_0": n_missing,
         "shift_p05_p50_p95_m": np.nanpercentile(m["shift_m"], [5, 50, 95]).tolist()}, indent=1), encoding="utf-8")
    steps = [s_ for s_ in a.steps.split(",") if s_ in ("dem", "emergence", "validate")]
    P.run_steps(CHECK, variant, main_tag, steps, sources=a.sources.split(","), n_boot=a.n_boot,
                note="setup correction, first order")
    return 0


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["delta-plan", "variant", "komsomolets", "setup-correction"])
    ap.add_argument("--plan")
    ap.add_argument("--tau-main", type=float)
    ap.add_argument("--alt")
    ap.add_argument("--name")
    ap.add_argument("--add-tag")
    ap.add_argument("--sub-tag")
    ap.add_argument("--sens-tag", default="sensB")
    if argv and argv[0] == "delta-plan":
        a, _ = ap.parse_known_args(argv)
        return delta_plan(a.plan, a.tau_main, a.alt, a.name)
    if argv and argv[0] == "komsomolets":
        a, _ = ap.parse_known_args(argv)
        return komsomolets(a.sens_tag, a.tau_main)
    P = _load("r00_pipeline")
    P.add_common(ap)
    a = ap.parse_args(argv)
    if a.cmd == "setup-correction":
        return setup_correction(a.main, a.plan, a.tau_main, P, a)
    variant = f"{a.main}__{a.name}"
    P.script("r00_counts").build(variant, [a.main], add=a.add_tag, sub=a.sub_tag)
    P.run_steps(CHECK, variant, a.main, a.steps.split(","), sources=a.sources.split(","),
                vertical_correction=a.vertical_correction, shape=a.shape, sector_labels=a.sector_labels,
                n_boot=a.n_boot, note=a.name)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
