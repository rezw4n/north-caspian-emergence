"""Revision 2, workstream G (PREREG Amendment 19 item 13; D-150): the sounding-based
Great Salt Lake lake-bed elevation model (HydroShare 582060f00f6b443bb26e896426d9f62a,
GSLDEM.tif; scripts/rev2_g_lakebed_data.py) audited as an additional control, in both
arms, exactly as the 3DEP substitute (scripts/g1_hindcast.py, scripts/g1_decide.py).

Commands:
  prepare    the model averaged onto the basin's 60 m grid (area-weighted mean of the
             native cells; NGVD 29, m) -> data/interim/revision2/g/lakebed_60m_ngvd29.npy;
             each arm's minimum (unit check); VERTCON 3.0 (NOAA NCAT) on a 0.2 deg lattice
             and its spread over each arm against the gauge constant (D-150 item 1);
             record results/revision2/g/gsl_lakebed_prepare.json
  hindcast   per arm and level set (D-150 item 5: "all", "sounded"), the Caspian hindcast
             of the lake-bed control on the g1_hindcast construction, heights on NAVD 88
             with the arm's gauge constant (or --vertcon cell) ->
             results/revision2/g/gsl_lakebed/<arm>/<set>[_vcell]/
  decide     the H1 rule per arm and set (shape decides, total reported), the control's
             basin flag, the Amendment 16 item 9 verdict with this control, the rows at
             3DEP's levels and by source class, survey timing per bin, the five grids at
             the sounded levels, per-level comparison with 3DEP and the grids ->
             results/revision2/g/gsl_lakebed.json, gsl_lakebed_levels.csv,
             gsl_lakebed_summary.csv, gsl_lakebed_grids_at_sounded.csv
Usage: python scripts/rev2_g_lakebed.py prepare | hindcast [--arms north,south]
       [--sets all,sounded] [--vertcon gauge|cell] | decide
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import caspian  # noqa: F401  (DLL path fix before rasterio)
import numpy as np
import pandas as pd
import pyproj
import requests
from shapely.geometry import shape as as_shape
from shapely.ops import unary_union

from caspian import design, hindcast
from caspian import generality as G
from caspian import hindcast_tiled as ht
from caspian import lakebed as LB
from caspian.config import INTERIM, RAW, RESULTS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.heavy_slot import heavy_slot  # noqa: E402


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


G1 = _load("g1_hindcast")          # its A7 (scripts/a7_hindcast.py) is the module patched below
DEC = _load("g1_decide")
A7 = G1.A7

BASIN = "great_salt_lake"
B = G.BASINS[BASIN]
RES = G1.RES
SRC = RAW / "revision2" / "gsl_biowest" / "GSLDEM.tif"
NCAT_CACHE = RAW / "revision2" / "gsl_biowest" / "vertcon_ncat_lattice.json"
IM = INTERIM / "revision2" / "g"
OUT = RESULTS / "revision2" / "g"
RUNS = OUT / "gsl_lakebed"
GEN = RESULTS / "generality" / BASIN
LEVELS = RAW / "generality" / BASIN / "levels" / "gsl_daily_levels.csv"
LEVELS_PROV = RESULTS / "provenance" / "generality_great_salt_lake_levels.json"
UA = {"User-Agent": "caspian-research/0.1 (rezwan490@gmail.com)"}

# D-150: sources by elevation (feet above NGVD 29), survey windows (Baskin 2005, 2006)
SOUNDED_TOP_FT = {"south": 4193.0, "north": 4194.0}
CONTOURS_TOP_FT = 4200.0
SURVEY = {"south": ("2002-09-01", "2004-06-30", "fall 2002, first half of 2003, spring 2004 (Baskin 2005)"),
          "north": ("2006-03-01", "2006-07-31", "spring and early summer 2006 (Baskin 2006)")}
VERTCON_RERUN_M = 0.05      # D-150 item 1
LATTICE_STEP = 0.2


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def gauge_delta() -> dict:
    """The arm's NAVD 88 - NGVD 29 at its gauge (VERTCON 3.0 via NCAT; D-094)."""
    prov = json.loads(LEVELS_PROV.read_text(encoding="utf-8"))
    return {arm: float(g["ngvd29_to_navd88"]["delta_m"]) for arm, g in prov["gauges"].items()}


def boundaries(delta: dict) -> dict:
    return {arm: {"sounded_top_navd88_m": round(LB.ft_ngvd29_to_navd88_m(SOUNDED_TOP_FT[arm], d), 4),
                  "contours_top_navd88_m": round(LB.ft_ngvd29_to_navd88_m(CONTOURS_TOP_FT, d), 4)}
            for arm, d in delta.items()}


def arms_geom():
    gj = json.loads((GEN / "arms_laea.geojson").read_text(encoding="utf-8"))
    return {f["properties"]["arm"]: as_shape(f["geometry"]) for f in gj["features"]
            if f["properties"]["role"] == "sector"}


# --- prepare ----------------------------------------------------------------------------------

def ncat_lattice() -> dict:
    """NAVD 88 - NGVD 29 from NOAA NCAT (VERTCON 3.0) on a 0.2 deg lattice over the basin box,
    the same request as the gauges' (scripts/g1_data.py ncat_delta); cached raw."""
    if NCAT_CACHE.exists():
        return json.loads(NCAT_CACHE.read_text(encoding="utf-8"))
    w, s, e, n = B["bbox"]
    lons = np.round(np.arange(np.floor(w / LATTICE_STEP) * LATTICE_STEP, e + LATTICE_STEP - 1e-9, LATTICE_STEP), 4)
    lats = np.round(np.arange(np.floor(s / LATTICE_STEP) * LATTICE_STEP, n + LATTICE_STEP - 1e-9, LATTICE_STEP), 4)
    pts = []
    for la in lats:
        for lo in lons:
            url = (f"https://geodesy.noaa.gov/api/ncat/llh?lat={la}&lon={lo}&orthoHt=1280.0&inDatum=NAD83(1986)"
                   f"&outDatum=NAD83(1986)&inVertDatum=NGVD29&outVertDatum=NAVD88")
            r = requests.get(url, timeout=120, headers=UA)
            r.raise_for_status()
            d = r.json()
            pts.append({"lat": float(la), "lon": float(lo), "url": url, "vertcon_version": d.get("vertconVersion"),
                        "delta_m": round(float(d["destOrthoht"]) - float(d["srcOrthoht"]), 4),
                        "sigma_m": float(d["sigOrthoht"])})
            time.sleep(0.3)
    rec = {"source": "NOAA NGS NCAT API (VERTCON 3.0), NGVD 29 -> NAVD 88 at orthometric height 1,280 m",
           "retrieved_utc": now(), "step_deg": LATTICE_STEP, "lons": lons.tolist(), "lats": lats.tolist(),
           "points": pts}
    NCAT_CACHE.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return rec


def vertcon_cells(transform, shape, lattice: dict) -> np.ndarray:
    """Per-cell VERTCON on the basin grid (bilinear in lon/lat from the NCAT lattice)."""
    lons, lats = np.array(lattice["lons"]), np.array(lattice["lats"])
    v = np.full((lats.size, lons.size), np.nan)
    for p in lattice["points"]:
        v[int(np.argmin(np.abs(lats - p["lat"]))), int(np.argmin(np.abs(lons - p["lon"])))] = p["delta_m"]
    f = LB.lattice_interpolator(lons, lats, v)
    x, y = G.cell_centres(transform, shape)
    X, Y = np.meshgrid(x, y)
    lon, lat = pyproj.Transformer.from_crs(B["crs"], "EPSG:4326", always_xy=True).transform(X, Y)
    return f(lon, lat).astype(np.float32)


def prepare() -> int:
    import rasterio
    IM.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    transform, shape = G.grid(B["bbox"], B["crs"], RES)
    out_npy = IM / "lakebed_60m_ngvd29.npy"
    with rasterio.open(SRC) as src:
        info = {"crs": src.crs.to_string(), "epsg": src.crs.to_epsg(), "native_res_m": [src.transform.a, -src.transform.e],
                "size": [src.width, src.height], "nodata": float(src.nodata), "dtype": src.dtypes[0],
                "units_tag": src.units[0]}
        t0 = time.time()
        with heavy_slot("G2: lake-bed model to 60 m"):
            z = LB.average_to_grid(src, transform, shape, B["crs"])
        info["resample_seconds"] = round(time.time() - t0, 1)
    np.save(out_npy, z)
    delta = gauge_delta()
    lat = ncat_lattice()
    vc = vertcon_cells(transform, shape, lat)
    np.save(IM / "vertcon_60m.npy", vc)
    arms = arms_geom()
    rec = {"rule": "D-150: area-weighted mean of the native cells in each 60 m cell (GDAL average via "
                   "rasterio.warp.reproject; no-data excluded); NAD83 taken as WGS84",
           "source_file": str(SRC.relative_to(ROOT)).replace("\\", "/"), "source": info,
           "rasterio_gdal": rasterio.__gdal_version__, "grid": {"transform": transform, "shape": list(shape),
                                                                 "crs": B["crs"]},
           "output": str(out_npy.relative_to(ROOT)).replace("\\", "/"), "finite_cells": int(np.isfinite(z).sum()),
           "gauge_delta_m": delta, "boundaries": boundaries(delta),
           "vertcon_lattice": {"n_points": len(lat["points"]), "step_deg": lat["step_deg"],
                               "versions": sorted({str(p["vertcon_version"]) for p in lat["points"]}),
                               "min_m": min(p["delta_m"] for p in lat["points"]),
                               "max_m": max(p["delta_m"] for p in lat["points"])},
           "arms": {}}
    for k, (arm, geom) in enumerate(arms.items(), start=1):
        dom = G1.burn(geom, transform, shape)
        zz = z[dom]
        ok = np.isfinite(zz)
        dv = vc[dom][ok] - delta[arm]
        rec["arms"][arm] = {
            "domain_cells": int(dom.sum()), "cells_with_height": int(ok.sum()),
            "share_with_height": round(float(ok.mean()), 5),
            "min_ngvd29_m": round(float(np.nanmin(zz)), 3), "min_ft": round(float(np.nanmin(zz)) / LB.FT_M, 2),
            "p01_ngvd29_m": round(float(np.nanpercentile(zz, 1)), 3),
            "median_ngvd29_m": round(float(np.nanmedian(zz)), 3),
            "max_ngvd29_m": round(float(np.nanmax(zz)), 3),
            "vertcon_cell_minus_gauge_m": {"min": round(float(dv.min()), 4), "max": round(float(dv.max()), 4),
                                           "median": round(float(np.median(dv)), 4),
                                           "max_abs": round(float(np.abs(dv).max()), 4)},
            "vertcon_rerun_required": bool(np.abs(dv).max() > VERTCON_RERUN_M)}
    rec["unit_check"] = ("metres: the documented deepest points are 'just below 4,167 feet (1,270 meters)' "
                         "(Baskin 2005, 2006); 4,167 ft = 1,270.10 m")
    (OUT / "gsl_lakebed_prepare.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(json.dumps(rec, indent=1))
    return 0


# --- hindcast ---------------------------------------------------------------------------------

def arm_static(arm, arms, transform, shape, st, cover, ponds):
    """g1_hindcast.main's per-arm construction, unchanged (its code, line by line)."""
    tr250 = [float(v) for v in st["transform"]]
    rim_cells = int(round(G1.RIM_M / RES))
    dom = G1.burn(arms[arm], transform, shape)
    k = 1 + list(arms).index(arm)
    armlab = np.asarray(st["armlab"]) == k
    hi250 = np.asarray(st["high"]) & armlab
    lo250 = np.asarray(st["low"]) & armlab
    zone = G1.burn(cover.intersection(arms[arm]), transform, shape) & dom
    zone |= A7.coast_band({"high": hi250.astype(np.int8), "low": lo250.astype(np.int8), "transform": tr250},
                          transform, shape) & dom
    rim_in = ht.eroded(dom, rim_cells)
    sea_seed = (G1.nearest250(np.asarray(st["seed"]) & armlab, tr250, transform, shape, False) > 0) & rim_in
    land250 = hindcast.beyond_land_state(np.asarray(st["high"]), np.ones(hi250.shape, bool), 250.0, 2000.0)
    b = G1.nearest250(np.nan_to_num(land250, nan=-1).astype(np.int8), tr250, transform, shape, -1)
    land_seed = (b == 0) & ~zone & rim_in
    if (sea_seed & ~zone).any():
        outside = ht.outside_state(zone, sea_seed & ~zone, land_seed, domain=dom, beyond_state=None,
                                   domain_margin=rim_cells)
    else:
        land = ht.components_any(~zone & rim_in, [land_seed])[0]
        outside = np.full(shape, -1, np.int8)
        outside[land] = 0
    pond = G1.burn(ponds, transform, shape)
    outside[pond] = -1
    sector = np.ones(shape, np.uint8)
    static = {"domain": dom, "sector": sector, "sea_seed": sea_seed, "outside": outside,
              "counts": {"domain": int(dom.sum()), "zone": int(zone.sum()), "sea_seed": int(sea_seed.sum()),
                         "land_seed": int(land_seed.sum()), "ponds_unknown": int((pond & dom).sum()),
                         "outside_water": int((outside == 1).sum()), "outside_land": int((outside == 0).sum()),
                         "outside_unknown": int((outside == -1).sum())}}
    return static, dom, pond


def hindcast_runs(arm_list, sets, vertcon: str) -> int:
    A7.CRS_EQUAL_AREA = B["crs"]
    transform, shape = G.grid(B["bbox"], B["crs"], RES)
    arms = arms_geom()
    cover = unary_union([as_shape(f["geometry"]) for f in
                         json.loads((GEN / "cover_laea.geojson").read_text(encoding="utf-8"))["features"]])
    ponds = unary_union([as_shape(f["geometry"]) for f in
                         json.loads((GEN / "ponds_laea.geojson").read_text(encoding="utf-8"))["features"]])
    st = np.load(INTERIM / "generality" / BASIN / "static_250m.npz")
    blocks = G1.blocks_table(BASIN, arms, transform, shape)
    reg = pd.read_csv(GEN / "blocks_25km.csv")
    cols = ["i", "j", "sector", "split"]
    if not blocks[cols].reset_index(drop=True).astype(str).equals(reg[cols].reset_index(drop=True).astype(str)):
        raise SystemExit("blocks differ from the registered blocks_25km.csv")
    z = np.load(IM / "lakebed_60m_ngvd29.npy")
    delta = gauge_delta()
    bnd = boundaries(delta)
    vc = np.load(IM / "vertcon_60m.npy") if vertcon == "cell" else None
    for arm in arm_list:
        with heavy_slot(f"G2: lake-bed hindcast {arm}"):
            arm_runs(arm, sets, arms, transform, shape, st, cover, ponds, blocks, z, delta, bnd, vc)
    return 0


def arm_runs(arm, sets, arms, transform, shape, st, cover, ponds, blocks, z, delta, bnd, vc) -> None:
    t0 = time.time()
    static, dom, pond = arm_static(arm, arms, transform, shape, st, cover, ponds)
    print(arm, "static", static["counts"], flush=True)
    if static["counts"] != json.loads((GEN / arm / "hindcast_dep3" / "run.json").read_text(
            encoding="utf-8"))["static_counts"]:
        raise SystemExit(f"{arm}: static counts differ from the registered 3DEP run")
    s2 = G1.masked_stage2(BASIN, "main", arm, dom & ~pond, transform)
    orig = A7.stage2_index
    units_all = orig(s2)
    A7.ENVELOPE = A7.record_low_water(units_all, transform, shape)
    env = {"rule": "D-082: cells water at the record low (3 lowest E3 bins) are water at every level",
           "cells": int(A7.ENVELOPE.sum())}
    fit_region = dom & A7.train_mask(blocks, transform, shape)
    h = LB.to_navd88(z, vc if vc is not None else delta[arm])
    h[~dom] = np.nan
    conv = ({"kind": "per-cell VERTCON 3.0 (NCAT 0.2 deg lattice, bilinear)", "file": "vertcon_60m.npy"}
            if vc is not None else {"kind": "arm gauge constant, VERTCON 3.0 via NCAT (D-094)",
                                    "delta_m": delta[arm]})
    fl = {"lakebed": ht.flood_level_map(h, static["sea_seed"])}
    for name in sets:
        if name == "all":
            keep, rule = None, "every audited bin"
        elif name == "sounded":
            bmax = LB.max_bin_at_or_below(bnd[arm]["sounded_top_navd88_m"], B["level_offset_m"])
            keep, rule = (lambda b, _m=bmax: b <= _m), (
                f"<= {bmax} (upper edge <= top of the soundings, {SOUNDED_TOP_FT[arm]:.0f} ft NGVD 29 = "
                f"{bnd[arm]['sounded_top_navd88_m']} m NAVD 88)")
        else:
            raise SystemExit(f"unknown set {name}")
        out = RUNS / arm / (name + ("_vcell" if vc is not None else ""))
        A7.stage2_index = (orig if keep is None else
                           (lambda src, _o=orig, _k=keep: {k: v for k, v in _o(src).items() if _k(k)}))
        try:
            meta = {"basin": BASIN, "arm": arm, "tag": "main", "resampling": "area-weighted mean (GDAL average)",
                    "envelope": env, "vertical_correction": conv,
                    "level_datum": "bins on NAVD 88 minus %.1f m (Amendment 16 item 5)" % B["level_offset_m"],
                    "product": "Great Salt Lake lake-bed model (HydroShare 582060f00f6b443bb26e896426d9f62a, "
                               "GSLDEM.tif; USGS soundings 2002-04 / 2006, map contours, NED), NGVD 29 -> NAVD 88",
                    "bins": rule, "decision": "D-150", "ponds": "OpenStreetMap salt ponds + 120 m: unknown"}
            A7.run(s2, out, static, {"lakebed": h}, transform, level_datum_offset=B["level_offset_m"],
                   level_datum_source=meta["level_datum"], shape_offsets="fit", fit_region=fit_region,
                   flood_maps=fl, sector_names=[arm], blocks=blocks, block_m=design.BLOCK_SIZE_M, meta=meta)
        finally:
            A7.stage2_index = orig
        print(arm, name, "done", round(time.time() - t0), "s", flush=True)


# --- decide -------------------------------------------------------------------------------------

def _tables(path: Path):
    return pd.read_csv(path / "displacement.csv"), pd.read_csv(path / "noise_floor.csv")


def _arm_rec(s, d, s_all, product, arm, variant):
    r = s[(s["product"] == product) & (s["variant"] == variant)]
    ra = s_all[(s_all["product"] == product) & (s_all["variant"] == variant)]
    dd = d[(d["product"] == product) & (d["variant"] == variant) & d["evaluable"]]
    if r.empty:
        return None
    r = r.iloc[0]
    return {"mispredicts": DEC._flag(r["mispredicts"]), "n_observed_levels": int(r["n_observed_levels"]),
            "n_evaluable": int(r["n_evaluable"]), "n_exceeding": int(r["n_exceeding"]),
            "mispredicts_pooled_floor": DEC._flag(ra.iloc[0]["mispredicts"]) if len(ra) else None,
            "median_displacement_m": float(np.median(dd["median"])) if len(dd) else None,
            "median_noise_p95_m": float(np.median(dd["p95_noise"])) if len(dd) else None,
            "bins": f"{int(dd['bin'].min())}-{int(dd['bin'].max())}" if len(dd) else None}


def _subset(d, variant, bins):
    dd = d[(d["variant"] == variant) & d["evaluable"] & d["bin"].astype(int).isin(bins)]
    n_ev, n_ex = int(len(dd)), int(dd["exceeds"].sum())
    return {"n_evaluable": n_ev, "n_exceeding": n_ex, "majority": (n_ex > n_ev / 2) if n_ev else None,
            "median_displacement_m": float(np.median(dd["median"])) if n_ev else None}


def decide() -> int:
    delta = gauge_delta()
    bnd = boundaries(delta)
    lv = pd.read_csv(LEVELS, parse_dates=["date"])
    reg = json.loads((GEN / "h1_decision.json").read_text(encoding="utf-8"))
    sets = sorted({p.name for a in B["sectors"] for p in (RUNS / a).iterdir() if (p / "displacement.csv").exists()})
    out = {"status": "exploratory, unregistered additional control (PREREG Amendment 19 item 13; D-150); the "
                     "registered verdict ('recurs', D-094 addendum) and the 3DEP substitute stand",
           "rule": "caspian.h1.mispredicts per arm (per-level floor; shape variant decides, total reported); the "
                   "control mispredicts in the basin in a majority of arms (caspian.generality.in_basin)",
           "boundaries_navd88_m": bnd, "gauge_delta_m": delta, "survey_windows": SURVEY, "sets": {}}
    level_rows, grid_rows = [], []
    for name in sets:
        rec = {"arms": {}, "shape_offsets": {}}
        flags = {"shape": [], "total": []}
        for arm in B["sectors"]:
            p = RUNS / arm / name
            disp, noise = _tables(p)
            s, d, s_all = DEC.arm_decision(disp, noise, arm)
            so = json.loads((p / "shape_offsets.json").read_text(encoding="utf-8"))["products"]["lakebed"]
            rec["shape_offsets"][arm] = {"offset_m": so["offset_m"],
                                         "loeo_m": {e: x["offset_m"] for e, x in so["leave_one_epoch_out"].items()}}
            for var in ("shape", "total"):
                a = _arm_rec(s, d, s_all, "lakebed", arm, var)
                rec["arms"].setdefault(arm, {})[var] = a
                flags[var].append(None if a is None else a["mispredicts"])
            # registered 3DEP control and the five grids, for comparison
            dc_s, dc_d, _ = DEC.arm_decision(*_tables(GEN / arm / "hindcast_dep3"), arm)
            gp_s, gp_d, gp_all = DEC.arm_decision(*_tables(GEN / arm / "hindcast"), arm)
            dep_bins = set(dc_d[(dc_d["product"] == "dep3") & (dc_d["variant"] == "shape") & dc_d["evaluable"]]
                           ["bin"].astype(int))
            lb = d[d["product"] == "lakebed"]
            lb_bins = set(lb[(lb["variant"] == "shape") & lb["evaluable"]]["bin"].astype(int))
            classes = dict(zip(sorted(lb["bin"].astype(int).unique()),
                               LB.source_class(sorted(lb["bin"].astype(int).unique()),
                                               bnd[arm]["sounded_top_navd88_m"], bnd[arm]["contours_top_navd88_m"],
                                               B["level_offset_m"])))
            if name == "all":
                rec["arms"][arm]["at_3dep_levels"] = {v: _subset(lb, v, dep_bins) for v in ("shape", "total")}
                rec["arms"][arm]["by_source_class"] = {
                    c: {v: _subset(lb, v, {b for b, k in classes.items() if k == c}) for v in ("shape", "total")}
                    for c in ("sounded", "map_contours", "ned", "mixed")}
                rec["arms"][arm]["dep3_registered"] = reg["control_arms"][arm]
                rec["arms"][arm]["dep3_shape_offset_m"] = reg["shape_offsets"][arm]["dep3"]["offset_m"]
                g = lv[lv["arm"] == arm].set_index("date")["level_navd88_m"]
                tim = LB.survey_timing(g, sorted(classes), SURVEY[arm][:2],
                                       level_offset_m=B["level_offset_m"]).set_index("bin")
                rec["arms"][arm]["survey"] = {"window": list(SURVEY[arm][:2]), "wording": SURVEY[arm][2],
                                              "gauge_min_navd88_m": round(float(tim["survey_min_m"].iloc[0]), 3),
                                              "gauge_max_navd88_m": round(float(tim["survey_max_m"].iloc[0]), 3)}
                for b in sorted(classes):
                    row = {"arm": arm, "bin": b, "level_navd88_m": round(float(np.mean(LB.bin_edges(b, B["level_offset_m"]))), 2),
                           "source_class": classes[b], **{k: tim.loc[b, k] for k in
                                                          ("under_water_in_survey", "within_sounding_depth",
                                                           "first_reached", "first_reached_after_survey")}}
                    for var in ("shape", "total"):
                        x = lb[(lb["variant"] == var) & (lb["bin"].astype(int) == b)]
                        if len(x):
                            x = x.iloc[0]
                            row[f"lakebed_{var}_evaluable"] = bool(x["evaluable"])
                            row[f"lakebed_{var}_median_m"] = float(x["median"])
                            row[f"lakebed_{var}_exceeds"] = bool(x["exceeds"])
                    row["floor_p95_m"] = float(x["p95_noise"]) if len(x) else np.nan
                    y = dc_d[(dc_d["product"] == "dep3") & (dc_d["variant"] == "shape") & (dc_d["bin"].astype(int) == b)]
                    row["dep3_evaluable"] = bool(y["evaluable"].iloc[0]) if len(y) else False
                    row["dep3_median_m"] = float(y["median"].iloc[0]) if row["dep3_evaluable"] else np.nan
                    for prod in G.GLOBAL_PRODUCTS:
                        z = gp_d[(gp_d["product"] == prod) & (gp_d["variant"] == "shape") & (gp_d["bin"].astype(int) == b)]
                        row[f"{prod}_median_m"] = float(z["median"].iloc[0]) if len(z) and bool(z["evaluable"].iloc[0]) else np.nan
                    level_rows.append(row)
            if name == "sounded":
                # the five grids restricted to the levels at which this control is evaluable (as rev_g_gsl
                # control_levels does for 3DEP); each grid keeps its own per-level floor
                for prod in G.GLOBAL_PRODUCTS:
                    for var in ("shape", "total"):
                        x = _subset(gp_d[gp_d["product"] == prod], var, lb_bins)
                        grid_rows.append({"arm": arm, "product": prod, "variant": var,
                                          "control_bins": f"{min(lb_bins)}-{max(lb_bins)}" if lb_bins else None,
                                          **x, "mispredicts_at_lakebed_levels": x["majority"]})
        for var in ("shape", "total"):
            rec[f"in_basin_{var}"] = G.in_basin(flags[var])
        rec["control_result"] = ("not evaluable" if rec["in_basin_shape"] is None else
                                 ("fail (mispredicts in the basin)" if rec["in_basin_shape"] else "pass"))
        gp = {p: reg["products"][p]["shape"]["in_basin"] for p in G.GLOBAL_PRODUCTS}
        rec["verdict_with_this_control"] = G.verdict(gp, rec["in_basin_shape"])
        rec["verdict_total_variant_with_this_control"] = G.verdict(
            {p: reg["products"][p]["total"]["in_basin"] for p in G.GLOBAL_PRODUCTS}, rec["in_basin_total"])
        out["sets"][name] = rec
    out["registered"] = {"verdict": reg["verdict"], "positive_control": reg["positive_control"],
                         "control_arms": reg["control_arms"]}
    L = pd.DataFrame(level_rows)
    if len(grid_rows):
        gr = pd.DataFrame(grid_rows)
        gr.to_csv(OUT / "gsl_lakebed_grids_at_sounded.csv", index=False)
        out["grids_at_sounded_levels"] = {
            p: {v: {"arms_mispredicted": int(sum(bool(x) for x in g["mispredicts_at_lakebed_levels"] if x is not None)),
                    "in_basin": G.in_basin([x if x is not None else None for x in g["mispredicts_at_lakebed_levels"]]),
                    "arms": {r["arm"]: {"n_exceeding": r["n_exceeding"], "n_evaluable": r["n_evaluable"],
                                        "median_displacement_m": r["median_displacement_m"]} for _, r in g.iterrows()}}
                for v, g in gp_.groupby("variant")} for p, gp_ in gr.groupby("product")}
        out["grids_at_sounded_levels_verdict"] = G.verdict(
            {p: out["grids_at_sounded_levels"][p]["shape"]["in_basin"] for p in G.GLOBAL_PRODUCTS},
            out["sets"].get("sounded", {}).get("in_basin_shape"))
    if len(L):
        L.to_csv(OUT / "gsl_lakebed_levels.csv", index=False, float_format="%.3f")
        ratio = {}
        for arm, g in L.groupby("arm"):
            ok = g["lakebed_shape_evaluable"].fillna(False).astype(bool)
            for ref in ["dep3"] + G.GLOBAL_PRODUCTS:
                x = g.loc[ok, f"{ref}_median_m"] / g.loc[ok, "lakebed_shape_median_m"]
                x = x[np.isfinite(x)]
                if len(x):
                    ratio.setdefault(ref, {})[arm] = {"median_ratio_to_lakebed": round(float(x.median()), 2),
                                                      "n_levels": int(len(x)),
                                                      "share_levels_ref_larger": round(float((x > 1).mean()), 3)}
        out["per_level_ratio_to_lakebed"] = ratio
        cov = {}
        for arm, g in L.groupby("arm"):
            cov[arm] = {c: {"bins": f"{int(x['bin'].min())}-{int(x['bin'].max())}", "n_bins": int(len(x)),
                            "levels_navd88_m": [float(x["level_navd88_m"].min()), float(x["level_navd88_m"].max())],
                            "n_under_water_in_survey": int(x["under_water_in_survey"].sum()),
                            "n_within_sounding_depth": int(x["within_sounding_depth"].sum()),
                            "n_first_reached_after_survey": int(x["first_reached_after_survey"].sum())}
                        for c, x in g.groupby("source_class")}
        out["coverage_by_source_class"] = cov
    summ = []
    for name, rec in out["sets"].items():
        for arm, a in rec["arms"].items():
            for var in ("shape", "total"):
                x = a[var]
                if x:
                    summ.append({"set": name, "arm": arm, "variant": var, **{k: x[k] for k in (
                        "n_evaluable", "n_exceeding", "mispredicts", "mispredicts_pooled_floor", "median_displacement_m",
                        "median_noise_p95_m", "bins")}, "shape_offset_m": rec["shape_offsets"][arm]["offset_m"]})
    pd.DataFrame(summ).to_csv(OUT / "gsl_lakebed_summary.csv", index=False, float_format="%.3f")
    (OUT / "gsl_lakebed.json").write_text(json.dumps(DEC.finite_json(_plain(out)), indent=1), encoding="utf-8")
    print(pd.DataFrame(summ).to_string(index=False))
    print(json.dumps(DEC.finite_json(_plain({k: out[k] for k in out if k not in ("sets",)})), indent=1)[:6000])
    for name, rec in out["sets"].items():
        print(name, rec["control_result"], rec["verdict_with_this_control"], rec["shape_offsets"])
    return 0


def _plain(x):
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, pd.Timestamp):
        return str(x.date())
    if x is pd.NaT:
        return None
    return x


def main(argv) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "hindcast", "decide"])
    ap.add_argument("--arms", default=",".join(B["sectors"]))
    ap.add_argument("--sets", default="all,sounded")
    ap.add_argument("--vertcon", default="gauge", choices=["gauge", "cell"])
    a = ap.parse_args(argv)
    if a.cmd == "prepare":
        return prepare()
    if a.cmd == "hindcast":
        return hindcast_runs(a.arms.split(","), a.sets.split(","), a.vertcon)
    return decide()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
