"""Revision analysis G (PREREG Amendment 18 item 11, ETOPO 2022). Exploratory, post-review
analyses; the registered H1 result stands.

Stages (each writes under results/revision/g/etopo/):
  sources   HEAVY. The main run's machinery
            (scripts/a7_hindcast.py, unchanged: static layers, record-low envelope, sea
            connection, the cached corrected heights and flood-level maps, the shape
            constants of results/hindcast/main/run.json) recomputes the waterline
            distances of ETOPO 2022 and GEBCO_2022 (shape variant) at every bin and epoch,
            and gives each distance the ETOPO 2022 source ID of its predicted waterline
            cell (caspian.rev_g.attribute_sources; nearest ETOPO cell to the 60 m cell
            centre). Per (product, sector, bin, source), epochs pooled: n and median
            |d|; the per-level medians over all sources are checked against the
            registered displacement.csv. --bins limits the run (pilot).
  compare   From the sources tables: matched levels (both sources >= 20 distances in
            the sector), per sector and H1 level band, and ETOPO 2022's shoreline
            cell shares by source.
  edge      ETOPO 2022's predicted waterline cells in three classes: FABDEM,
            lake depth at the edge of the lake-depth area (an ETOPO cell with a FABDEM
            8-neighbour), lake depth inside; share and median |d| per level band and sector.
  bands     The registered per-level floor (pooled-level rows) and D, Q, D/Q per H1
            level band and sector, for every product (shape variant), with within-sector
            comparisons.
  heights   ETOPO 2022 minus GEBCO_2022 at the same 15" cells of the emerged band, by
            ETOPO source ID and level band (the band of each 250 m cell from the waterline
            DEM, as scripts/a7_level_bands.py).
  clamp     Over every ETOPO 2022 lake-depth cell of the subset: ETOPO 2022 against
            GEBCO_2022 at the same cell (equal, lower, higher; the min(GEBCO_2022, -29 m) rule).
  is2       ICESat-2 (ATL06, ATL08) test points split by ETOPO 2022 source ID, per
            sector: NMAD, RMSE and bias of the DEM, ETOPO 2022 and GEBCO_2026 (matched).
  gedi      GEDI test points split by ETOPO 2022 source ID: NMAD by source with block-
            bootstrap intervals (B = 2,000, seed 20261022) for ETOPO 2022 and the DEM,
            and the within-block contrast (blocks with >= 50 points of each source).

Usage: python scripts/rev_g_etopo.py <stage> [--bins 30,31]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import caspian  # noqa: F401
import numpy as np
import pandas as pd
import pyproj
import xarray as xr

from caspian import products, rev_g, validation
from caspian import hindcast_tiled as ht
from caspian.config import CRS_EQUAL_AREA, DATA, INTERIM, RAW, RESULTS
from caspian.geography import DOMAIN, SECTOR_ORDER

ROOT = Path(__file__).resolve().parents[1]
OUT = RESULTS / "revision" / "g" / "etopo"
CACHE = INTERIM / "revision_g" / "etopo"
SEED = 20261022
N_BOOT = 2000
MIN_N = 20                     # registered minimum number of distances
MIN_N_BLOCK = 50               # within-block contrast: points of each source per block
BANDS = [-29.0, -28.3, -27.7, -27.0, -26.3, -25.5]   # H1 level bands (scripts/a7_level_bands.py)
SRC_NAME = {10: "fabdem", 11: "lake_depth"}
CORRECTION = RESULTS / "study_area" / "geoid_correction_primary.npz"
PRODS = ("etopo_2022", "gebco_2022")


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _band(level) -> pd.Series:
    return pd.cut(pd.Series(np.asarray(level, float)), BANDS).astype(str)


class EtopoSource:
    """ETOPO 2022 source ID of the nearest native cell, at lon/lat points or at the
    centres of hindcast-grid cells (``transform`` of that grid)."""

    def __init__(self, transform=None):
        with xr.open_dataset(RAW / "bathymetry" / "etopo_2022_ncaspian.nc") as ds:
            self.ilon, self.ilat = ds.indexes["lon"], ds.indexes["lat"]
            self.sid = ds["source_id"].transpose("lat", "lon").values
        self.tr = None if transform is None else [float(v) for v in transform[:6]]
        self.inv = pyproj.Transformer.from_crs(CRS_EQUAL_AREA, "EPSG:4326", always_xy=True)

    def at_cells(self, rows, cols) -> np.ndarray:
        a, _, c, _, e, f = self.tr
        lon, lat = self.inv.transform(c + a * (np.asarray(cols) + 0.5), f + e * (np.asarray(rows) + 0.5))
        return self.at_lonlat(lon, lat)

    def at_lonlat(self, lon, lat) -> np.ndarray:
        i = self.ilat.get_indexer(np.asarray(lat, float), method="nearest")
        j = self.ilon.get_indexer(np.asarray(lon, float), method="nearest")
        return self.sid[i, j].astype(int)


def stage_sources(a) -> int:
    A7 = _load("a7_hindcast", "a7_hindcast.py")
    run = json.loads((RESULTS / "hindcast" / "main" / "run.json").read_text(encoding="utf-8"))
    tile = int(run["tile"])
    w, s, e, n = DOMAIN.bounds
    transform, shape = A7.grid((w - A7.MARGIN_DEG, s - A7.MARGIN_DEG, e + A7.MARGIN_DEG, n + A7.MARGIN_DEG))
    t0 = time.time()
    static = A7.production_static(transform, shape, tile)
    for k in ("zone", "land_seed"):
        static.pop(k)
    print("static layers", round(time.time() - t0), "s", flush=True)
    corr, corr_rec = products.correction(CORRECTION, transform, shape)
    flood, finite = {}, {}
    for p in PRODS:
        h = products.heights(p, transform, shape, run["resampling"])
        H = products.Corrected(h, corr)
        flood[p] = A7.flood_map_cached(p, H, static["sea_seed"], {"heights": Path(h.filename).name,
                                                                  "correction": corr_rec["sha256"]}, tile)
        finite[p] = (lambda hh: lambda r0, r1, c0, c1: np.isfinite(np.asarray(hh[r0:r1, c0:c1], float)))(H)
    off = {p: float(run["shape_offsets"][p]["offset_m"]) for p in PRODS}
    units = A7.stage2_index(INTERIM / "stage2" / "main")
    A7.ENVELOPE = A7.record_low_water(units, transform, shape)
    print("envelope cells", int(A7.ENVELOPE.sum()), round(time.time() - t0), "s", flush=True)
    src = EtopoSource(transform)
    bins = sorted(units) if not a.bins else [int(b) for b in a.bins.split(",")]
    sector = static["sector"]
    name = {k + 1: sname for k, sname in enumerate(SECTOR_ORDER)}
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    per_level, shares, raw_parts = [], [], []
    for b in bins:
        level = A7._bin_level(b)
        epochs = sorted(units[b])
        pooled = {p: [] for p in PRODS}
        for ep in epochs:
            raw, _ = A7.observed_raw(A7.groups_for(units, b, [ep]), transform, shape)
            obs = A7.classify(raw, static, tile)
            del raw
            for p in PRODS:
                fl = float(products.flood_level(level, 0.0, 0.0, off[p]))
                pred = ht.flooded(flood[p], fl)
                cells = ht.displacement_cells(pred, obs, A7._valid(obs), sector, A7.RES, pred_valid=finite[p],
                                              outside_state=static["outside"], shape=shape, tile=tile)
                sid = src.at_cells(cells["pred"]["rows"], cells["pred"]["cols"])
                att = rev_g.attribute_sources(cells, sid).assign(product=p, epoch=ep, bin=b)
                pooled[p].append(att)
                pr = cells["pred"]
                kp = pr["known"]
                for lab in np.unique(pr["label"][kp]):
                    m = kp & (pr["label"] == lab)
                    u, c = np.unique(sid[m], return_counts=True)
                    for uu, cc in zip(u, c):
                        shares.append({"product": p, "epoch": ep, "bin": b, "level_m": round(level, 4),
                                       "sector": name.get(int(lab), str(lab)), "source": int(uu), "n_pred_cells": int(cc)})
                del cells, pred
            del obs
        for p in PRODS:
            d = pd.concat(pooled[p], ignore_index=True)
            raw_parts.append(d)
            for lab, g in d.groupby("label"):
                if int(lab) not in name:
                    continue
                sec = name[int(lab)]
                per_level.append({"product": p, "sector": sec, "bin": b, "level_m": round(level, 4), "source": "all",
                                  "n": int(len(g)), "median_m": float(np.median(g["abs_m"]))})
                for sv, gg in g.groupby("source"):
                    per_level.append({"product": p, "sector": sec, "bin": b, "level_m": round(level, 4),
                                      "source": SRC_NAME.get(int(sv), f"other_{int(sv)}"), "n": int(len(gg)),
                                      "median_m": float(np.median(gg["abs_m"]))})
        print(f"bin {b} ({level:.2f} m) {epochs} done, {time.time() - t0:.0f} s", flush=True)
    tag = "" if not a.bins else "_pilot"
    pl = pd.DataFrame(per_level)
    pl.to_csv(OUT / f"source_per_level{tag}.csv", index=False, float_format="%.1f")
    pd.DataFrame(shares).to_csv(OUT / f"shoreline_cell_sources{tag}.csv", index=False)
    pd.concat(raw_parts, ignore_index=True).to_csv(CACHE / f"distances{tag}.csv.gz", index=False, float_format="%.1f")
    # consistency with the registered per-level medians (pooled epochs, shape variant)
    reg = pd.read_csv(RESULTS / "hindcast" / "main" / "displacement.csv")
    reg = reg[(reg.epoch == "pooled") & (reg.variant == "shape") & reg["product"].isin(PRODS)]
    chk = pl[pl.source == "all"].merge(reg[["product", "sector", "bin", "median_m", "n"]], on=["product", "sector", "bin"],
                                       suffixes=("", "_registered"))
    diff = (chk["median_m"] - chk["median_m_registered"]).abs()
    rec = {"n_compared": int(len(chk)), "max_abs_median_diff_m": float(diff.max()) if len(chk) else None,
           "n_count_mismatch": int((chk["n"] != chk["n_registered"]).sum()), "seconds": round(time.time() - t0, 1),
           "bins": bins, "shape_offsets_m": off, "correction_sha256": corr_rec["sha256"],
           "source_sampling": "nearest ETOPO 2022 cell to the 60 m cell centre (heights are bilinear)"}
    (OUT / f"sources_run{tag}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(json.dumps(rec, indent=1))
    return 0


def stage_compare(a) -> int:
    pl = pd.read_csv(OUT / "source_per_level.csv")
    out = {"rule": "D-130: per (product, sector, level), epochs pooled; matched levels have >= 20 distances from "
                   "both sources; each distance takes the ETOPO 2022 source ID of its predicted waterline cell",
           "products": {}}
    rows = []
    for p, gp in pl[pl.source != "all"].groupby("product"):
        m = rev_g.matched_levels(gp, ["fabdem", "lake_depth"], MIN_N)
        m["band"] = _band(m["level_m"]).to_numpy()
        m["product"] = p
        rows.append(m)
        mm = m[m.matched]
        rec = {"n_levels_matched": int(len(mm)), "n_levels_total": int(len(m)), "by_sector": {}, "by_band": {}}
        for key, grp in (("by_sector", "sector"), ("by_band", "band")):
            for k, g in mm.groupby(grp):
                rec[key][k] = {"n_matched_levels": int(len(g)),
                               "median_fabdem_m": round(float(g["median_fabdem"].median()), 1),
                               "median_lake_depth_m": round(float(g["median_lake_depth"].median()), 1),
                               "share_levels_fabdem_smaller": round(float((g["median_fabdem"] < g["median_lake_depth"]).mean()), 3)}
        if len(mm):
            rec["all_matched"] = {"median_fabdem_m": round(float(mm["median_fabdem"].median()), 1),
                                  "median_lake_depth_m": round(float(mm["median_lake_depth"].median()), 1),
                                  "share_levels_fabdem_smaller": round(float((mm["median_fabdem"] < mm["median_lake_depth"]).mean()), 3)}
        out["products"][p] = rec
    pd.concat(rows, ignore_index=True).to_csv(OUT / "matched_levels.csv", index=False, float_format="%.1f")
    sh = pd.read_csv(OUT / "shoreline_cell_sources.csv")
    sh["band"] = _band(sh["level_m"]).to_numpy()
    sh["source_name"] = sh["source"].map(SRC_NAME).fillna("other")
    t = sh.groupby(["product", "sector", "band", "source_name"])["n_pred_cells"].sum().unstack(fill_value=0)
    t["fabdem_share"] = t.get("fabdem", 0) / t.sum(axis=1)
    t.reset_index().to_csv(OUT / "shoreline_cell_sources_by_band.csv", index=False, float_format="%.4f")
    tb = sh.groupby(["product", "band", "source_name"])["n_pred_cells"].sum().unstack(fill_value=0)
    tb["fabdem_share"] = tb.get("fabdem", 0) / tb.sum(axis=1)
    out["shoreline_fabdem_share_by_band"] = {p: {b: round(float(v), 4) for b, v in g.droplevel(0)["fabdem_share"].items()}
                                             for p, g in tb.groupby(level=0)}
    # registered per-level medians (all sources), by band, for reference
    allm = pl[pl.source == "all"].copy()
    allm["band"] = _band(allm["level_m"]).to_numpy()
    out["median_all_sources_by_band_m"] = {p: {b: round(float(g2["median_m"].median()), 1) for b, g2 in g.groupby("band")}
                                           for p, g in allm.groupby("product")}
    (OUT / "compare.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


def stage_bands(a) -> int:
    det = pd.read_csv(RESULTS / "hindcast" / "main" / "h1_mispredicts_detail.csv")
    det = det[(det.variant == "shape") & det.evaluable].copy()
    det["band"] = _band(det["level_m"]).to_numpy()
    det["dq"] = det["median"].astype(float) / det["p95_noise"].astype(float)
    nf = pd.read_csv(RESULTS / "hindcast" / "main" / "noise_floor.csv")
    nf = nf[(nf.kind == "pooled_level") & (nf.n > 0)].copy()
    nf["band"] = _band(nf["level_m"]).to_numpy()
    floor = (nf.groupby(["sector", "band"]).agg(n_levels=("p95_m", "size"), floor_p95_median_m=("p95_m", "median"),
                                                 floor_p95_min_m=("p95_m", "min"), floor_p95_max_m=("p95_m", "max"))
             .reset_index())
    floor.to_csv(OUT / "floor_by_band_sector.csv", index=False, float_format="%.1f")
    g = det.groupby(["product", "sector", "band"])
    t = pd.DataFrame({"n_evaluable": g.size(), "n_exceeding": g["exceeds"].sum().astype(int),
                      "median_D_m": g["median"].median(), "median_Q_m": g["p95_noise"].median(),
                      "median_DQ": g["dq"].median()}).reset_index()
    t["majority_exceeds"] = t["n_exceeding"] > t["n_evaluable"] / 2
    t.to_csv(OUT / "dq_by_band_sector.csv", index=False, float_format="%.3f")
    # within-sector: ETOPO 2022 against the others in the same band; ETOPO above vs below -27.7 m
    et = t[t["product"] == "etopo_2022"].set_index(["sector", "band"])
    comp = []
    for p in sorted(set(t["product"]) - {"etopo_2022"}):
        o = t[t["product"] == p].set_index(["sector", "band"])
        j = et.join(o, lsuffix="_etopo", rsuffix="_other", how="inner")
        for (sec, bd), r in j.iterrows():
            comp.append({"other": p, "sector": sec, "band": bd, "DQ_etopo": r["median_DQ_etopo"],
                         "DQ_other": r["median_DQ_other"], "etopo_lower": bool(r["median_DQ_etopo"] < r["median_DQ_other"])})
    comp = pd.DataFrame(comp)
    comp.to_csv(OUT / "dq_within_sector_etopo_vs_others.csv", index=False, float_format="%.3f")
    ed = det.copy()
    ed["part"] = np.where(ed["level_m"] > -27.7, "above_-27.7", "below_-27.7")
    ab = (ed.groupby(["product", "sector", "part"]).agg(n_evaluable=("dq", "size"), n_exceeding=("exceeds", "sum"),
                                                         median_D_m=("median", "median"), median_Q_m=("p95_noise", "median"),
                                                         median_DQ=("dq", "median"))
          .reset_index())
    ab["majority_exceeds"] = ab["n_exceeding"] > ab["n_evaluable"] / 2
    ab.to_csv(OUT / "above_below_by_sector.csv", index=False, float_format="%.3f")
    summ = {"rule": "D-130: D-087 bands; shape variant; evaluable levels of h1_mispredicts_detail.csv; floor = "
                    "pooled-level 95th percentile of noise_floor.csv",
            "below_-27.7_majority_exceeds": {}, "etopo_lower_DQ_share_by_band": {}}
    low = t[t["band"].isin(["(-29.0, -28.3]", "(-28.3, -27.7]"])]
    for p, gp in low.groupby("product"):
        summ["below_-27.7_majority_exceeds"][p] = {f"{r.sector} {r.band}": f"{int(r.n_exceeding)}/{int(r.n_evaluable)}"
                                                   for r in gp.itertuples()}
    for bd, gb in comp.groupby("band"):
        summ["etopo_lower_DQ_share_by_band"][bd] = {"n_pairs": int(len(gb)), "share_etopo_lower": round(float(gb["etopo_lower"].mean()), 3)}
    summ["below_-27.7_sectors_majority_exceeds"] = {
        p: {"n_sectors": int(g["majority_exceeds"].sum()), "of": int(len(g)),
            "not_majority": sorted(g.loc[~g["majority_exceeds"], "sector"].tolist())}
        for p, g in ab[ab["part"] == "below_-27.7"].groupby("product")}
    summ["above_-27.7_sectors_majority_exceeds"] = {
        p: {"n_sectors": int(g["majority_exceeds"].sum()), "of": int(len(g))}
        for p, g in ab[ab["part"] == "above_-27.7"].groupby("product")}
    summ["floor_by_band_median_over_sectors_m"] = {bd: round(float(gb["floor_p95_median_m"].median()), 1)
                                                   for bd, gb in floor.groupby("band")}
    (OUT / "bands.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(json.dumps(summ, indent=1))
    return 0


def stage_heights(a) -> int:
    ex = np.load(RESULTS / "study_area" / "extents_v6_250m.npz")
    band = (ex["high"] == 1) & (ex["low"] == 0)
    tr = ex["transform"]
    r, c = np.nonzero(band)
    xs, ys = tr[2] + (c + 0.5) * tr[0], tr[5] + (r + 0.5) * tr[4]
    lon, lat = pyproj.Transformer.from_crs(CRS_EQUAL_AREA, "EPSG:4326", always_xy=True).transform(xs, ys)
    with xr.open_dataset(RAW / "bathymetry" / "etopo_2022_ncaspian.nc") as ds:
        e_lon, e_lat = ds["lon"].values, ds["lat"].values
        sel = dict(lon=xr.DataArray(lon), lat=xr.DataArray(lat), method="nearest")
        ez = ds["elevation"].sel(**sel).values.astype(float)
        sid = ds["source_id"].sel(**sel).values.astype(int)
        elon = ds["lon"].sel(lon=xr.DataArray(lon), method="nearest").values
        elat = ds["lat"].sel(lat=xr.DataArray(lat), method="nearest").values
    with xr.open_dataset(RAW / "bathymetry" / "gebco_2022_ncaspian.nc") as ds:
        g_lon, g_lat = ds["lon"].values, ds["lat"].values
        gz = ds["elevation"].sel(lon=xr.DataArray(elon), lat=xr.DataArray(elat), method="nearest").values.astype(float)
    align = {"etopo_lon_step": float(np.median(np.diff(e_lon))), "gebco_lon_step": float(np.median(np.diff(g_lon)))}
    # centre alignment: distance from each sampled ETOPO cell centre to the nearest GEBCO centre
    gi = np.clip(np.searchsorted(g_lon, elon), 1, g_lon.size - 1)
    dlon = np.minimum(np.abs(g_lon[gi] - elon), np.abs(g_lon[gi - 1] - elon))
    glat = np.sort(g_lat)
    gj = np.clip(np.searchsorted(glat, elat), 1, glat.size - 1)
    dlat = np.minimum(np.abs(glat[gj] - elat), np.abs(glat[gj - 1] - elat))
    align["max_centre_offset_deg"] = float(max(dlon.max(), dlat.max()))
    z = validation.TileSet(DATA / "processed" / "dem" / "main").sample(xs, ys)
    lab = _band(z).where(np.isfinite(z), "no DEM height").to_numpy()
    d = ez - gz
    df = pd.DataFrame({"band": lab, "source": pd.Series(sid).map(SRC_NAME).fillna("other").to_numpy(), "diff": d,
                       "etopo": ez, "gebco_2022": gz})
    rows = []
    for (bd, sv), g in df.groupby(["band", "source"]):
        v = g["diff"].to_numpy()
        v = v[np.isfinite(v)]
        rows.append({"band": bd, "source": sv, "n_cells_250m": int(len(g)),
                     "share_equal_0p01m": round(float(np.mean(np.abs(v) < 0.01)), 4) if v.size else np.nan,
                     "diff_p05_m": float(np.percentile(v, 5)) if v.size else np.nan,
                     "diff_median_m": float(np.median(v)) if v.size else np.nan,
                     "diff_p95_m": float(np.percentile(v, 95)) if v.size else np.nan,
                     "etopo_median_m": float(np.nanmedian(g["etopo"])), "gebco_2022_median_m": float(np.nanmedian(g["gebco_2022"])),
                     "etopo_whole_metre_share": round(float(np.mean(np.isclose(g["etopo"], np.round(g["etopo"])))), 4)})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "etopo_minus_gebco2022_by_band_source.csv", index=False, float_format="%.3f")
    (OUT / "heights.json").write_text(json.dumps({"rule": "D-130: emerged-band 250 m cells (extents_v6_250m.npz), nearest "
                                                           "ETOPO 2022 cell and the GEBCO_2022 cell nearest its centre; "
                                                           "band from the waterline DEM height",
                                                   "grid_alignment": align}, indent=1), encoding="utf-8")
    print(align)
    print(t.to_string(index=False))
    return 0


def _points(drv, source: str, size: float = 50_000.0):
    pts = validation.attach_blocks(drv._load_points("main", source, "bare"), drv._blocks(size), size)
    return validation.split_rows(pts, "test").reset_index(drop=True)


def _errors(te: pd.DataFrame, res: Path, source: str, prods) -> dict:
    c = json.loads((res / f"{source}_datum_constant.json").read_text(encoding="utf-8"))["constant"]["constant_m"]
    off = json.loads((res / f"{source}_product_offsets.json").read_text(encoding="utf-8"))["product_offsets"]
    y = te["h_orth"].to_numpy(float) + c
    e = {"dem": te["dem_h"].to_numpy(float) - y}
    for p in prods:
        e[p] = te[f"prod:{p}"].to_numpy(float) + off[p]["offset_m"] - y
    return e


def stage_is2(a) -> int:
    drv = _load("a6_validate_dem", "a6_validate_dem.py")
    rows = []
    for source in a.sources.split(","):
        te = _points(drv, source)
        e = _errors(te, RESULTS / "validation" / "main", source, ("etopo_2022", "gebco_2026", "gebco_2022"))
        sid = EtopoSource().at_lonlat(te["lon"].to_numpy(float), te["lat"].to_numpy(float))
        sname = pd.Series(sid).map(SRC_NAME).fillna("other").to_numpy()
        ok = np.ones(len(te), bool)
        for v in e.values():
            ok &= np.isfinite(v)
        sec = te["sector"].to_numpy(object)
        for scope in ["overall", *SECTOR_ORDER]:
            ms = ok if scope == "overall" else ok & (sec == scope)
            for sv in ("all", "fabdem", "lake_depth"):
                m = ms if sv == "all" else ms & (sname == sv)
                if not m.any():
                    continue
                r = {"source": source, "scope": scope, "etopo_source": sv, "n_points": int(m.sum()),
                     "share_of_scope": round(float(m.sum() / max(ms.sum(), 1)), 4),
                     "n_blocks": int(te.loc[m, "block"].nunique())}
                for k, v in e.items():
                    r[f"{k}_nmad"] = validation.nmad(v[m])
                    r[f"{k}_rmse"] = validation.rmse(v[m])
                    r[f"{k}_bias"] = validation.bias(v[m])
                rows.append(r)
    t = pd.DataFrame(rows)
    name = "icesat2_by_etopo_source.csv" if a.sources == "ATL06,ATL08" else f"icesat2_by_etopo_source_{a.sources}.csv"
    t.to_csv(OUT / name, index=False, float_format="%.4f")
    print(t[t.scope.isin(["overall", "komsomolets"])].round(3).to_string(index=False))
    return 0


def stage_gedi(a) -> int:
    drv = _load("a6_validate_dem", "a6_validate_dem.py")
    res = RESULTS / "validation" / "main_gedi"
    size = float(json.loads((res / "block_size_check.json").read_text(encoding="utf-8"))["final_block_size_m"])
    te = _points(drv, "GEDI", size)
    e = _errors(te, res, "GEDI", ("etopo_2022",))
    sid = EtopoSource().at_lonlat(te["lon"].to_numpy(float), te["lat"].to_numpy(float))
    grp = pd.Series(sid).map(SRC_NAME).fillna("other").to_numpy()
    keep = np.isin(grp, ["fabdem", "lake_depth"])
    errs = {"dem": e["dem"][keep], "etopo_2022": e["etopo_2022"][keep]}
    blk = te["block"].to_numpy()[keep]
    t = rev_g.group_nmad_bootstrap(errs, grp[keep], blk, ("fabdem", "lake_depth"), N_BOOT, SEED)
    t.to_csv(OUT / "gedi_nmad_by_source_bootstrap.csv", index=False, float_format="%.4f")
    w = rev_g.within_block_contrast(errs, grp[keep], blk, ("fabdem", "lake_depth"), MIN_N_BLOCK)
    w.to_csv(OUT / "gedi_within_block.csv", index=False, float_format="%.4f")
    summ = {"rule": "D-130: GEDI test points (bare, %g km blocks), ETOPO 2022 source ID of the nearest cell; block "
                    "bootstrap B = %d, seed %d; within-block contrast where each source has >= %d points"
                    % (size / 1000, N_BOOT, SEED, MIN_N_BLOCK),
            "n_points": int(keep.sum()), "n_blocks": int(np.unique(blk).size), "n_other_source_dropped": int((~keep).sum()),
            "bootstrap": t.to_dict(orient="records"),
            "within_block": {"n_blocks": int(len(w))}}
    if len(w):
        summ["within_block"].update(
            etopo_median_diff_lake_minus_fabdem_m=float(w["etopo_2022_diff"].median()),
            dem_median_diff_lake_minus_fabdem_m=float(w["dem_diff"].median()),
            n_blocks_etopo_fabdem_lower=int((w["etopo_2022_diff"] > 0).sum()),
            n_blocks_dem_fabdem_lower=int((w["dem_diff"] > 0).sum()))
    (OUT / "gedi.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(t.round(3).to_string(index=False))
    print(w.round(3).to_string(index=False))
    return 0


def stage_clamp(a) -> int:
    """Over every ETOPO 2022 lake-depth cell (source ID 11) of the North Caspian subset:
    how ETOPO 2022's height relates to GEBCO_2022's at the same 15" cell (the grids share
    cell centres; checked in the heights stage). Context for the heights stage."""
    e = xr.open_dataset(RAW / "bathymetry" / "etopo_2022_ncaspian.nc")
    g = xr.open_dataset(RAW / "bathymetry" / "gebco_2022_ncaspian.nc")
    lon = np.intersect1d(np.round(e["lon"].values, 6), np.round(g["lon"].values, 6))
    lat = np.intersect1d(np.round(e["lat"].values, 6), np.round(g["lat"].values, 6))
    E = e["elevation"].sel(lon=lon, lat=lat, method="nearest").transpose("lat", "lon").values.astype(float)
    S = e["source_id"].sel(lon=lon, lat=lat, method="nearest").transpose("lat", "lon").values
    Gz = g["elevation"].sel(lon=lon, lat=lat, method="nearest").transpose("lat", "lon").values.astype(float)
    m = (S == 11) & np.isfinite(E) & np.isfinite(Gz)
    ev, gv = E[m], Gz[m]
    rule = np.minimum(gv, -29.0)
    above = gv > -29.0
    src = "MacFerrin et al. 2025 describe GEBCO Lake Depths as GEBCO 2022 lake bathymetry included unchanged (MSL)"
    met = {
        "n_lake_cells": {"value": int(m.sum()), "unit": "cells", "note": "ETOPO 2022 source ID 11 in the subset"},
        "share_equal_gebco2022_pct": {"value": round(100 * float(np.mean(np.abs(ev - gv) < 0.01)), 2), "unit": "%"},
        "share_etopo_lower_pct": {"value": round(100 * float(np.mean(ev < gv - 0.01)), 2), "unit": "%"},
        "share_etopo_higher_pct": {"value": round(100 * float(np.mean(ev > gv + 0.01)), 2), "unit": "%"},
        "max_etopo_lake_m": {"value": float(ev.max()), "unit": "m"},
        "share_gebco_above_m29_pct": {"value": round(100 * float(np.mean(above)), 2), "unit": "%",
                                      "note": "GEBCO_2022 higher than -29 m at ETOPO lake-depth cells"},
        "share_matching_min_rule_pct": {"value": round(100 * float(np.mean(np.abs(ev - rule) < 0.01)), 2), "unit": "%",
                                        "note": "ETOPO 2022 = min(GEBCO_2022, -29 m)"},
        "share_matching_min_rule_where_gebco_above_m29_pct": {
            "value": round(100 * float(np.mean(np.abs(ev[above] - rule[above]) < 0.01)), 2), "unit": "%"},
    }
    vals, cnt = np.unique(np.round(ev - gv, 2), return_counts=True)
    out = {"rule": "D-130 heights stage, whole lake-depth area of the subset (context)", "documentation": src,
           "metrics": met, "difference_counts": {str(v): int(c) for v, c in zip(vals, cnt) if c >= 10}}
    (OUT / "lake_clamp.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


def stage_edge(a) -> int:
    """ETOPO 2022's predicted waterline cells of the full sources run in three
    classes (FABDEM; lake depth at the edge of the lake-depth area, i.e. an ETOPO cell with a
    FABDEM 8-neighbour; lake depth inside), per H1 level band and sector: share of cells and median
    |d| of their predicted-to-observed distances."""
    from scipy import ndimage
    A7 = _load("a7_hindcast", "a7_hindcast.py")
    w, s_, e, n = DOMAIN.bounds
    transform, shape = A7.grid((w - A7.MARGIN_DEG, s_ - A7.MARGIN_DEG, e + A7.MARGIN_DEG, n + A7.MARGIN_DEG))
    src = EtopoSource(transform)
    fab = src.sid == 10
    edge = (src.sid == 11) & ndimage.binary_dilation(fab, structure=np.ones((3, 3), bool))
    parts = []
    for ch in pd.read_csv(CACHE / "distances.csv.gz", chunksize=2_000_000,
                          usecols=["kind", "row", "col", "label", "abs_m", "product", "bin"]):
        parts.append(ch[(ch["kind"] == "pred") & (ch["product"] == "etopo_2022")])
    d = pd.concat(parts, ignore_index=True)
    a_, _, c_, _, e_, f_ = (float(v) for v in transform[:6])
    lon, lat = src.inv.transform(c_ + a_ * (d["col"].to_numpy() + 0.5), f_ + e_ * (d["row"].to_numpy() + 0.5))
    i = src.ilat.get_indexer(np.asarray(lat, float), method="nearest")
    j = src.ilon.get_indexer(np.asarray(lon, float), method="nearest")
    cls = np.where(src.sid[i, j] == 10, "fabdem", np.where(edge[i, j], "lake_depth_edge",
                   np.where(src.sid[i, j] == 11, "lake_depth_inside", "other")))
    name = {k + 1: sname for k, sname in enumerate(SECTOR_ORDER)}
    d = d.assign(cls=cls, sector=d["label"].map(name), level_m=[A7._bin_level(int(b)) for b in d["bin"]])
    d["band"] = _band(d["level_m"]).to_numpy()
    rows = []
    for keys, g in [(("all", b), gb) for b, gb in d.groupby("band")] + list(d.groupby(["sector", "band"])):
        tot = len(g)
        for c, gc in g.groupby("cls"):
            rows.append({"sector": keys[0], "band": keys[1], "class": c, "n_cells": int(len(gc)),
                         "share": round(len(gc) / tot, 4), "median_abs_m": float(np.median(gc["abs_m"]))})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "etopo_shoreline_edge_classes.csv", index=False, float_format="%.4f")
    print(t[t.sector == "all"].to_string(index=False))
    return 0


def main(argv) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["sources", "compare", "edge", "bands", "heights", "clamp", "is2", "gedi"])
    ap.add_argument("--bins", default=None)
    ap.add_argument("--sources", default="ATL06,ATL08", help="is2 stage: ICESat-2 sources")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    return {"sources": stage_sources, "compare": stage_compare, "edge": stage_edge, "bands": stage_bands, "heights": stage_heights, "clamp": stage_clamp,
            "is2": stage_is2, "gedi": stage_gedi}[a.stage](a)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
