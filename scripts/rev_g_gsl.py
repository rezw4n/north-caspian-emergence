"""Revision analysis G (PREREG Amendment 18 item 11, Great Salt Lake): two exploratory,
post-review analyses of the frozen Great Salt Lake (GSL) outputs. No hindcast is rerun
and no registered verdict is replaced.

1. sources  Where each global product's heights come from in each arm's emerged band
            (near-maximum but not near-minimum water inside the arm, the 250 m band of
            scripts/g1_lineage.py), read from the products' own source grids at the
            band cell centres (nearest native cell):
            - GEBCO_2026 and GEBCO_2022 type identifier (TID; GEBCO documentation,
              Table 2: 0 "Land");
            - ETOPO 2022 source ID (MacFerrin et al. 2025, Table 1: 10 FABDEM, 11 GEBCO
              Lake Depths);
            - SRTM15+ V2.7 source ID (Tozer et al. 2019, section 5.1: a cell
              constrained by a sounding holds the sounding's index; 0 otherwise);
            - ETOPO1 has no source grid; only its heights are described.
            For every product the heights are described as well: the modal height and
            the share of band cells at it (a flat water surface shows as one height over
            much of the band), converted to NAVD 88 with the GSL datum grid
            (H(NAVD 88) - H(EGM2008), results/provenance/generality_great_salt_lake_
            vdatum.json), and compared with the arm's gauge level during the land
            source's acquisition: SRTM, 11-22 February 2000 (GEBCO, SRTM15+ and ETOPO1
            land), and the Copernicus DEM, 2011-2015 (FABDEM in ETOPO 2022). The share
            of band cells whose height lies below the arm's record-low gauge level
            (cells that could hold lake-bed information at the levels audited) is also
            given.
2. control_levels  The registered H1 rule (caspian.h1.mispredicts; shape variant
            decides) for each global product restricted to the levels at which the
            positive-control substitute (3DEP, scripts/g1_hindcast.py) is evaluable in
            the same arm, with the product's own per-level floor (the same observed
            extents as the control's); the pooled-floor sensitivity likewise. Per level,
            the product's median displacement is paired with the control's.

3. above_flat  The registered per-level rows of each product split at the
            level where its flood fill passes its flat surface (level + c above the flat
            height on NAVD 88): exceeding levels and median displacement in each part.

Outputs: results/revision/g/gsl_sources.json, gsl_above_flat.json, gsl_band_height_histograms.csv (per arm and
product, band heights on NAVD 88 in 0.5 m bins from 1,270 to 1,300 m, for a figure panel), gsl_control_levels.csv (per product,
variant and arm), gsl_control_levels_detail.csv (per level), gsl_control_levels.json.
Usage: python scripts/rev_g_gsl.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import caspian  # noqa: F401
import numpy as np
import pandas as pd
import pyproj
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

from caspian import generality as G
from caspian import h1
from caspian.config import INTERIM, RAW, RESULTS

ROOT = Path(__file__).resolve().parents[1]
OUT = RESULTS / "revision" / "g"
BASIN = "great_salt_lake"
_spec = importlib.util.spec_from_file_location("g1_decide", ROOT / "scripts" / "g1_decide.py")
DEC = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(DEC)

CODES = {
    "gebco_2026": ("gebco_2026.nc", "tid", "GEBCO type identifier; 0 = 'Land' (GEBCO grid documentation, Table 2)"),
    "gebco_2022": ("gebco_2022.nc", "tid", "GEBCO type identifier; 0 = 'Land' (GEBCO grid documentation, Table 2)"),
    "etopo_2022": ("etopo_2022.nc", "source_id", "ETOPO 2022 source ID; 10 = FABDEM, 11 = GEBCO Lake Depths "
                                                 "(MacFerrin et al. 2025, Table 1)"),
    "srtm15p_v2.7": ("srtm15p_v2.7.nc", "source_id", "SRTM15+ source ID; a cell constrained by a sounding holds the "
                                                     "sounding's index, other cells 0 (Tozer et al. 2019, section 5.1)"),
}
CODE_MEANING = {"gebco_2026": {0: "Land"}, "gebco_2022": {0: "Land"},
                "etopo_2022": {10: "FABDEM (land topography)", 11: "GEBCO Lake Depths"},
                "srtm15p_v2.7": {0: "no sounding (land or predicted)"}}
HEIGHTS = {"gebco_2026": ("gebco_2026.nc", "elevation"), "gebco_2022": ("gebco_2022.nc", "elevation"),
           "etopo_2022": ("etopo_2022.nc", "elevation"), "srtm15p_v2.7": ("srtm15p_v2.7.nc", "elevation"),
           "etopo1": ("etopo1.tif", "bedrock")}
# acquisition window of each product's land source over GSL, for the gauge comparison
LAND_SOURCE = {"gebco_2026": ("SRTM (via SRTM15+)", "2000-02-11", "2000-02-22"),
               "gebco_2022": ("SRTM (via SRTM15+)", "2000-02-11", "2000-02-22"),
               "srtm15p_v2.7": ("SRTM CGIAR-CSI V4.1", "2000-02-11", "2000-02-22"),
               "etopo1": ("not documented in the sources read (SRTM era assumed for the comparison only)",
                          "2000-02-11", "2000-02-22"),
               "etopo_2022": ("FABDEM (Copernicus DEM, TanDEM-X)", "2011-01-01", "2015-12-31")}


def band_points(B):
    st = np.load(INTERIM / "generality" / BASIN / "static_250m.npz")
    tr = [float(v) for v in st["transform"]]
    x, y = G.cell_centres(tr, st["high"].shape)
    X, Y = np.meshgrid(x, y)
    lon, lat = pyproj.Transformer.from_crs(B["crs"], "EPSG:4326", always_xy=True).transform(X, Y)
    out = {}
    for k, arm in enumerate(B["sectors"], start=1):
        band = st["high"] & ~st["low"] & (st["armlab"] == k)
        out[arm] = (lon[band], lat[band])
    return out


def read_layer(f: str, layer: str, lon, lat) -> np.ndarray:
    raw = RAW / "generality" / BASIN / "bathymetry" / f
    if raw.suffix == ".tif":
        import rasterio
        with rasterio.open(raw) as r:
            band = list(r.descriptions).index(layer) + 1
            z = r.read(band, masked=True).astype(float).filled(np.nan)
            t = r.transform
        cols = np.floor((np.asarray(lon) - t.c) / t.a).astype(int)
        rows = np.floor((np.asarray(lat) - t.f) / t.e).astype(int)
        rows, cols = np.clip(rows, 0, z.shape[0] - 1), np.clip(cols, 0, z.shape[1] - 1)
        return z[rows, cols]
    with xr.open_dataset(raw) as ds:
        return ds[layer].sel(lon=xr.DataArray(lon), lat=xr.DataArray(lat), method="nearest").values.astype(float)


def vdatum(lon, lat) -> np.ndarray:
    z = np.load(RAW / "generality" / BASIN / "vdatum_navd88_minus_egm2008.npz")
    f = RegularGridInterpolator((z["lat"], z["lon"]), z["dN"], bounds_error=False, fill_value=None)
    return f(np.c_[lat, lon])


def gauge(arm: str, start: str, end: str) -> dict:
    lv = pd.read_csv(RAW / "generality" / BASIN / "levels" / "gsl_daily_levels.csv", parse_dates=["date"])
    g = lv[(lv["arm"] == arm)].set_index("date").sort_index()
    w = g.loc[start:end, "level_navd88_m"]
    return {"window": [start, end], "n_values": int(w.size), "mean_m": round(float(w.mean()), 3),
            "min_m": round(float(w.min()), 3), "max_m": round(float(w.max()), 3),
            "record_low_m": round(float(g["level_navd88_m"].min()), 3),
            "record_low_date": str(g["level_navd88_m"].idxmin().date())}


def sources() -> dict:
    B = G.BASINS[BASIN]
    pts = band_points(B)
    out = {"rule": "emerged band = near-maximum and not near-minimum water inside the arm (250 m; "
                   "scripts/g1_static.py, as scripts/g1_lineage.py); nearest native cell at each band cell centre",
           "status": "exploratory, post-review (PREREG Amendment 18 item 11); descriptive",
           "code_grids": {p: v[2] for p, v in CODES.items()}, "arms": {}}
    hist_rows = []
    for arm, (lon, lat) in pts.items():
        n = lon.size
        dn = vdatum(lon, lat)
        rec = {"band_cells_250m": int(n), "band_km2": round(n * 0.0625, 1),
               "navd88_minus_egm2008_m": {"median": round(float(np.median(dn)), 3),
                                          "min": round(float(dn.min()), 3), "max": round(float(dn.max()), 3)},
               "products": {}}
        for p, (f, layer) in HEIGHTS.items():
            z = read_layer(f, layer, lon, lat)
            ok = np.isfinite(z)
            vals, cnt = np.unique(np.round(z[ok], 2), return_counts=True)
            mode = float(vals[np.argmax(cnt)])
            pr = {"heights_reference": "EGM2008 as read (D-070)", "n_finite": int(ok.sum()),
                  "modal_height_egm2008_m": mode, "modal_share": round(float(cnt.max() / n), 4),
                  "modal_height_navd88_m": round(mode + float(np.median(dn)), 2),
                  "share_within_0p5m_of_mode": round(float(np.mean(np.abs(z[ok] - mode) <= 0.5)), 4),
                  "whole_metre_share": round(float(np.mean(np.isclose(z[ok], np.round(z[ok])))), 4),
                  "n_distinct_heights": int(vals.size),
                  "p05_navd88_m": round(float(np.percentile(z[ok] + dn[ok], 5)), 2),
                  "p50_navd88_m": round(float(np.percentile(z[ok] + dn[ok], 50)), 2),
                  "p95_navd88_m": round(float(np.percentile(z[ok] + dn[ok], 95)), 2)}
            src, s0, s1 = LAND_SOURCE[p]
            gg = gauge(arm, s0, s1)
            pr["land_source"] = src
            pr["gauge_during_land_source_navd88"] = gg
            pr["share_below_record_low_gauge"] = round(float(np.mean((z[ok] + dn[ok]) < gg["record_low_m"])), 4)
            if p in CODES:
                f2, layer2, _ = CODES[p]
                c = read_layer(f2, layer2, lon, lat)
                cv, cc = np.unique(c[np.isfinite(c)].astype(int), return_counts=True)
                pr["code_layer"] = layer2
                pr["code_shares"] = {str(int(a)): round(float(b / n), 4) for a, b in zip(cv, cc)}
                pr["code_meaning"] = {str(int(a)): CODE_MEANING[p].get(int(a), "not documented here") for a in cv}
                pr["any_sounding_code"] = bool(
                    (p.startswith("gebco") and np.any((cv >= 10) & (cv < 40))) or
                    (p == "srtm15p_v2.7" and np.any(cv > 0)) or (p == "etopo_2022" and np.any(cv == 11)))
            else:
                pr["code_layer"] = None
                pr["code_shares"] = "no source grid distributed"
            rec["products"][p] = pr
        out["arms"][arm] = rec
        for p, (f, layer) in HEIGHTS.items():      # height histogram (NAVD 88, 0.5 m) for a figure panel
            z = read_layer(f, layer, lon, lat) + dn
            z = z[np.isfinite(z)]
            edges = np.arange(1270.0, 1300.5, 0.5)        # the band's levels and their surroundings
            cnt, _ = np.histogram(z, edges)
            hist_rows += [{"arm": arm, "product": p, "bin_low_navd88_m": float(a), "bin_high_navd88_m": float(b),
                           "n_cells_250m": int(c), "share": round(float(c) / z.size, 5)}
                          for a, b, c in zip(edges[:-1], edges[1:], cnt) if c]
    pd.DataFrame(hist_rows).to_csv(OUT / "gsl_band_height_histograms.csv", index=False)
    return out


def control_levels():
    B = G.BASINS[BASIN]
    d = RESULTS / "generality" / BASIN
    rows, det_all, per_level = [], [], []
    for arm in B["sectors"]:
        h, c = d / arm / "hindcast", d / arm / "hindcast_dep3"
        disp, noise = pd.read_csv(h / "displacement.csv"), pd.read_csv(h / "noise_floor.csv")
        dc, nc = pd.read_csv(c / "displacement.csv"), pd.read_csv(c / "noise_floor.csv")
        _, det, _ = DEC.arm_decision(disp, noise, arm)
        _, cdet, _ = DEC.arm_decision(dc, nc, arm)
        cev = cdet[(cdet["product"] == "dep3") & (cdet["variant"] == "shape") & cdet["evaluable"]]
        cbins = set(cev["bin"].astype(int))
        cmed = cev.set_index("bin")["median"]
        cq = cev.set_index("bin")["p95_noise"]
        prod = disp[(disp["epoch"] == "pooled") & (disp["sector"] == arm)].rename(columns={"median_m": "median"})
        prod = prod[prod["bin"].astype(int).isin(cbins)]
        nz = noise[(noise["kind"] == "pooled_level") & (noise["sector"] == arm)].rename(columns={"p95_m": "p95"})
        nz_all = noise[(noise["kind"] == "pooled_all_levels") & (noise["sector"] == arm)].rename(columns={"p95_m": "p95"})
        s, dd = h1.mispredicts(prod, nz[["sector", "bin", "n", "p95"]])
        s_all, _ = h1.mispredicts(prod, nz_all[["sector", "n", "p95"]])
        full = det[det["evaluable"]].groupby(["product", "variant"]).agg(n_ev_all=("evaluable", "size"),
                                                                        n_ex_all=("exceeds", "sum"))
        for _, r in s.iterrows():
            ra = s_all[(s_all["product"] == r["product"]) & (s_all["variant"] == r["variant"])].iloc[0]
            e = dd[(dd["product"] == r["product"]) & (dd["variant"] == r["variant"]) & dd["evaluable"]]
            fa = full.loc[(r["product"], r["variant"])]
            rows.append({"arm": arm, "product": r["product"], "variant": r["variant"],
                         "control_bins": f"{min(cbins)}-{max(cbins)}", "n_control_levels": len(cbins),
                         "n_evaluable": int(r["n_evaluable"]), "n_exceeding": int(r["n_exceeding"]),
                         "mispredicts_at_control_levels": bool(r["mispredicts"]),
                         "mispredicts_pooled_floor_at_control_levels": bool(ra["mispredicts"]),
                         "median_displacement_m": float(np.median(e["median"])) if len(e) else np.nan,
                         "median_floor_p95_m": float(np.median(e["p95_noise"])) if len(e) else np.nan,
                         "n_evaluable_all_levels": int(fa["n_ev_all"]), "n_exceeding_all_levels": int(fa["n_ex_all"])})
            if r["variant"] == "shape":
                for _, x in e.iterrows():
                    b = int(x["bin"])
                    per_level.append({"arm": arm, "product": r["product"], "bin": b, "level_navd88_m":
                                      round(float(x["level_m"]) + B["level_offset_m"], 2),
                                      "product_median_m": float(x["median"]), "control_median_m": float(cmed.get(b, np.nan)),
                                      "floor_p95_m": float(x["p95_noise"]), "control_floor_p95_m": float(cq.get(b, np.nan)),
                                      "exceeds": bool(x["exceeds"])})
        ctl = cdet[(cdet["product"] == "dep3") & (cdet["variant"] == "shape")]
        rows.append({"arm": arm, "product": "dep3 (control substitute)", "variant": "shape",
                     "control_bins": f"{min(cbins)}-{max(cbins)}", "n_control_levels": len(cbins),
                     "n_evaluable": int(ctl["evaluable"].sum()), "n_exceeding": int(ctl["exceeds"].sum()),
                     "mispredicts_at_control_levels": bool(ctl["exceeds"].sum() > ctl["evaluable"].sum() / 2),
                     "mispredicts_pooled_floor_at_control_levels": None,
                     "median_displacement_m": float(np.median(cev["median"])),
                     "median_floor_p95_m": float(np.median(cev["p95_noise"])),
                     "n_evaluable_all_levels": int(ctl["evaluable"].sum()), "n_exceeding_all_levels": int(ctl["exceeds"].sum())})
    t = pd.DataFrame(rows)
    pl = pd.DataFrame(per_level)
    t.to_csv(OUT / "gsl_control_levels.csv", index=False, float_format="%.1f")
    pl.to_csv(OUT / "gsl_control_levels_detail.csv", index=False, float_format="%.1f")
    summ = {"rule": "caspian.h1.mispredicts (shape variant decides) on each product's pooled-per-level rows "
                    "restricted to the bins at which the 3DEP substitute is evaluable in the same arm; the "
                    "product's own per-level floor; pooled-floor sensitivity reported",
            "status": "exploratory, post-review (PREREG Amendment 18 item 11); the registered verdict "
                      "('recurs', D-094 addendum) stands",
            "by_product": {}}
    for p in G.GLOBAL_PRODUCTS:
        for v in ("shape", "total"):
            q = t[(t["product"] == p) & (t["variant"] == v)]
            flags = q["mispredicts_at_control_levels"].tolist()
            summ["by_product"].setdefault(p, {})[v] = {
                "arms_mispredicted": int(sum(flags)), "arms_evaluable": len(flags),
                "in_basin": G.in_basin(flags),
                "arms": {r["arm"]: {"n_exceeding": int(r["n_exceeding"]), "n_evaluable": int(r["n_evaluable"]),
                                     "median_displacement_m": round(float(r["median_displacement_m"]), 1),
                                     "pooled_floor_mispredicts": bool(r["mispredicts_pooled_floor_at_control_levels"])}
                         for _, r in q.iterrows()}}
    summ["verdict_at_control_levels"] = G.verdict({p: summ["by_product"][p]["shape"]["in_basin"]
                                                   for p in G.GLOBAL_PRODUCTS}, False)
    if len(pl):
        pl["ratio"] = pl["product_median_m"] / pl["control_median_m"]
        summ["per_level_ratio_product_to_control"] = {
            p: {a: {"median": round(float(g["ratio"].median()), 2), "n_levels": int(len(g)),
                    "share_levels_product_larger": round(float((g["ratio"] > 1).mean()), 3)}
                for a, g in gp.groupby("arm")} for p, gp in pl.groupby("product")}
    (OUT / "gsl_control_levels.json").write_text(json.dumps(DEC.finite_json(summ), indent=1), encoding="utf-8")
    return t, summ


def above_flat() -> dict:
    """The registered per-level rows (shape variant) split at the level at
    which each product's flood fill passes its flat surface (flood level = level + c above
    the flat height on NAVD 88)."""
    B = G.BASINS[BASIN]
    src = json.loads((OUT / "gsl_sources.json").read_text(encoding="utf-8"))
    dec = json.loads((RESULTS / "generality" / BASIN / "h1_decision.json").read_text(encoding="utf-8"))
    det = pd.read_csv(RESULTS / "generality" / BASIN / "h1_mispredicts_detail.csv")
    det = det[(det["variant"] == "shape") & det["evaluable"]]
    out = {"rule": "D-128 addendum: a level is above the flat surface when level + c exceeds the product's modal "
                   "height on NAVD 88 (modal EGM2008 height + the arm's median NAVD 88 - EGM2008)",
           "status": "exploratory, descriptive", "arms": {}}
    for arm in B["sectors"]:
        for p in G.GLOBAL_PRODUCTS:
            flat = src["arms"][arm]["products"][p]["modal_height_navd88_m"]
            c = dec["shape_offsets"][arm][p]["offset_m"]
            d = det[(det["product"] == p) & (det["arm"] == arm)]
            lv = d["level_m"].astype(float) + B["level_offset_m"]
            up = (lv + c) > flat
            rec = {"flat_navd88_m": flat, "shape_c_m": c, "level_where_flat_floods_navd88_m": round(flat - c, 2)}
            for name, m in (("above_flat", up), ("below_flat", ~up)):
                g = d[m.to_numpy()]
                rec[name] = {"n_evaluable": int(len(g)), "n_exceeding": int(g["exceeds"].sum()),
                             "majority": bool(g["exceeds"].sum() > len(g) / 2) if len(g) else None,
                             "median_displacement_m": float(np.median(g["median"])) if len(g) else None}
            out["arms"].setdefault(arm, {})[p] = rec
    (OUT / "gsl_above_flat.json").write_text(json.dumps(DEC.finite_json(out), indent=1), encoding="utf-8")
    return out


def main(argv) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    what = argv[0] if argv else "all"
    if what in ("all", "sources"):
        s = sources()
        (OUT / "gsl_sources.json").write_text(json.dumps(s, indent=1), encoding="utf-8")
        for arm, r in s["arms"].items():
            for p, x in r["products"].items():
                print(arm, p, x.get("code_shares"), "mode", x["modal_height_navd88_m"], x["modal_share"],
                      "gauge", x["gauge_during_land_source_navd88"]["mean_m"], "below record low",
                      x["share_below_record_low_gauge"])
    if what in ("all", "above_flat"):
        print(json.dumps(DEC.finite_json(above_flat()), indent=1))
    if what in ("all", "control_levels"):
        t, summ = control_levels()
        print(t.to_string(index=False))
        print(json.dumps(DEC.finite_json(summ["verdict_at_control_levels"]), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
