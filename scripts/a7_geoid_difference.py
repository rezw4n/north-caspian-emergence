"""Build the geoid-difference map N(EGM2008) - N(EIGEN-6C4) for the North Caspian product box.

Inputs (raw, never modified):
  data/raw/geoid/icgem/*.gdf                    ICGEM calcgrid, functional 'geoid', WGS84, tide_free, zero-degree term on
  data/raw/geoid/icgem/checks/*.gdf             coarse (15') ICGEM check grids (mean/zero tide; height_anomaly_ell)
  data/raw/geoid/proj/us_nga_egm08_25_window_*.tif   window of PROJ CDN us_nga_egm08_25.tif (NGA/GeographicLib EGM2008)
  data/raw/bathymetry/etopo_2022_ncaspian.nc    ETOPO 2022 15" subset incl. its EGM2008 geoid-height layer
Outputs:
  results/study_area/geoid_egm2008_minus_eigen6c4.npz
  results/provenance/geoid_difference.json

Run: python scripts/a7_geoid_difference.py
"""
import caspian  # noqa: F401  (project rule)
from caspian.geography import DOMAIN

import datetime as dt
import glob
import hashlib
import json
import pathlib
import sys

import numpy as np
import rasterio
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

from caspian.config import RESULTS  # noqa: E402
ROOT = RESULTS.parent
RAW = ROOT / "data/raw/geoid"
OUT_NPZ = ROOT / "results/study_area/geoid_egm2008_minus_eigen6c4.npz"
OUT_PROV = ROOT / "results/provenance/geoid_difference.json"

LON0, LAT0, STEP = 46.3, 43.85, 1.0 / 60.0
NLON, NLAT = 505, 217
lon = LON0 + STEP * np.arange(NLON)          # cell centres (= ICGEM nodes), ascending
lat = LAT0 + STEP * np.arange(NLAT)          # cell centres, ascending (south -> north)
REGION = (46.3, 43.8677, 54.7, 47.45)        # DOMAIN bounds +/- 0.1 deg (W, S, E, N)


def sha256(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def rel(p):
    return pathlib.Path(p).resolve().relative_to(ROOT.resolve()).as_posix()


def read_gdf(path, lon_axis, lat_axis, step):
    """Parse an ICGEM .gdf (long_lat_value) onto the given ascending axes; check full, unique coverage."""
    head, rows = {}, []
    with open(path, encoding="latin-1") as f:
        for line in f:
            if line.startswith("end_of_head"):
                break
            parts = line.split(None, 1)
            if len(parts) == 2:
                head[parts[0]] = parts[1].strip()
        data = np.loadtxt(f)
    lo, la, v = data[:, 0], data[:, 1], data[:, 2]
    j = np.rint((lo - lon_axis[0]) / step).astype(int)
    i = np.rint((la - lat_axis[0]) / step).astype(int)
    # printed coordinates have 4 decimals: check they match the reconstructed nodes
    assert np.all(np.abs(lon_axis[j] - lo) < 6e-5) and np.all(np.abs(lat_axis[i] - la) < 6e-5), path
    grid = np.full((len(lat_axis), len(lon_axis)), np.nan)
    cnt = np.zeros_like(grid, dtype=int)
    np.add.at(cnt, (i, j), 1)
    grid[i, j] = v
    assert cnt.min() == 1 and cnt.max() == 1, f"coverage not unique/full in {path}"
    assert not np.any(v == float(head.get("gapvalue", "999.0"))), "gap values present"
    return grid, head


def stats(a, mask=None):
    x = a if mask is None else a[mask]
    x = x[np.isfinite(x)]
    return {"n": int(x.size), "min_m": float(x.min()), "max_m": float(x.max()),
            "mean_m": float(x.mean()), "sd_m": float(x.std(ddof=1)),
            "p01_m": float(np.percentile(x, 1)), "p99_m": float(np.percentile(x, 99))}


def main():
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    gdf = {m: RAW / f"icgem/icgem_{m}_geoid_tidefree_zdt_WGS84_1min_E46.3-54.7_N43.85-47.45.gdf"
           for m in ("EGM2008", "EIGEN-6C4")}
    N = {}
    heads = {}
    for m, p in gdf.items():
        N[m], heads[m] = read_gdf(p, lon, lat, STEP)
    for k in ("tide_system", "functional", "zero_degree_term", "refsysname", "gmrefpot", "radiusrefpot",
              "flatrefpot", "omegarefpot", "normal_potential", "max_used_degree", "gridstep"):
        assert heads["EGM2008"][k] == heads["EIGEN-6C4"][k] or k == "max_used_degree", k
    dN = N["EGM2008"] - N["EIGEN-6C4"]

    LON, LAT = np.meshgrid(lon, lat)
    region = (LON >= REGION[0] - 1e-9) & (LON <= REGION[2] + 1e-9) & (LAT >= REGION[1]) & (LAT <= REGION[3] + 1e-9)
    dw, ds, de, dn = DOMAIN.bounds
    domain_box = (LON >= dw) & (LON <= de) & (LAT >= ds) & (LAT <= dn)

    # --- NGA/PROJ EGM2008 (GeographicLib-built 2.5' grid), bilinear at the nodes (as PROJ interpolates) ---
    tif = sorted(glob.glob(str(RAW / "proj/us_nga_egm08_25_window_*.tif")))[0]
    with rasterio.open(tif) as src:
        a = src.read(1).astype(np.float64)
        tr = src.transform
        assert src.tags().get("AREA_OR_POINT") == "Point"
    plon = tr.c + tr.a * (np.arange(a.shape[1]) + 0.5)
    plat = tr.f + tr.e * (np.arange(a.shape[0]) + 0.5)
    assert np.allclose(plon * 24, np.rint(plon * 24), atol=1e-6) and np.allclose(plat * 24, np.rint(plat * 24), atol=1e-6)
    f = RegularGridInterpolator((plat[::-1], plon), a[::-1, :], method="linear", bounds_error=True)
    N_nga = f(np.column_stack([LAT.ravel(), LON.ravel()])).reshape(LAT.shape)
    # exact-node check: 2.5' nodes that coincide with 1' nodes (every 5 arc-min)
    on = (np.rint(LON * 60) % 5 == 0) & (np.rint(LAT * 60) % 5 == 0)
    d_nga = N_nga - N["EGM2008"]

    # --- ETOPO 2022 geoid-height layer (EGM2008, EPSG:3855), linear at the nodes (inside its extent) ---
    etopo = ROOT / "data/raw/bathymetry/etopo_2022_ncaspian.nc"
    with xr.open_dataset(etopo) as ds_:
        g = ds_["geoid_height"].load()
        elon, elat = g["lon"].values, g["lat"].values
        fe = RegularGridInterpolator((elat, elon), g.values.astype(np.float64), method="linear",
                                     bounds_error=False, fill_value=np.nan)
    N_etopo = fe(np.column_stack([LAT.ravel(), LON.ravel()])).reshape(LAT.shape)
    d_etopo_nga = N_etopo - N_nga
    d_etopo_icgem = N_etopo - N["EGM2008"]

    # --- coarse ICGEM checks (15' nodes, subset of the 1' nodes) ---
    clon = 46.3 + 0.25 * np.arange(34)
    clat = 43.85 + 0.25 * np.arange(15)
    ci = np.rint((clat - LAT0) / STEP).astype(int)
    cj = np.rint((clon - LON0) / STEP).astype(int)
    chk = {}
    chk_files = {}
    for p in sorted(glob.glob(str(RAW / "icgem/checks/*.gdf"))):
        name = pathlib.Path(p).name
        g_, h_ = read_gdf(p, clon, clat, 0.25)
        chk[name] = (g_, h_)
        chk_files[name] = p
    checks = {}

    def C(model, tide, func):
        n = f"icgem_{model}_{func}_{tide}_zdt_WGS84_15min_E46.3-54.55_N43.85-47.35.gdf"
        return chk[n][0] if n in chk else None

    full_free = {m: N[m][np.ix_(ci, cj)] for m in N}
    e_mean, g_mean = C("EIGEN-6C4", "mean_tide", "geoid"), C("EGM2008", "mean_tide", "geoid")
    if e_mean is not None and g_mean is not None:
        dmean = g_mean - e_mean
        dfree = full_free["EGM2008"] - full_free["EIGEN-6C4"]
        checks["dN_mean_tide_minus_dN_tide_free"] = stats(dmean - dfree)
        checks["EIGEN6C4_N_mean_tide_minus_tide_free"] = stats(e_mean - full_free["EIGEN-6C4"])
        checks["EGM2008_N_mean_tide_minus_tide_free"] = stats(g_mean - full_free["EGM2008"])
    e_zero = C("EIGEN-6C4", "zero_tide", "geoid")
    if e_zero is not None:
        checks["EIGEN6C4_N_zero_tide_minus_tide_free"] = stats(e_zero - full_free["EIGEN-6C4"])
    e_qg, g_qg = C("EIGEN-6C4", "tide_free", "height_anomaly_ell"), C("EGM2008", "tide_free", "height_anomaly_ell")
    if e_qg is not None and g_qg is not None:
        checks["EIGEN6C4_geoid_minus_height_anomaly_ell"] = stats(full_free["EIGEN-6C4"] - e_qg)
        checks["EGM2008_geoid_minus_height_anomaly_ell"] = stats(full_free["EGM2008"] - g_qg)
        checks["dN_geoid_minus_dzeta_ell"] = stats((full_free["EGM2008"] - full_free["EIGEN-6C4"]) - (g_qg - e_qg))
    # the coarse tide_free grids are a subset of the 1' nodes; repeatability is not requested (no tide_free coarse grid)

    dN_nga = N_nga - N["EIGEN-6C4"]
    summary = {
        "dN_m_region": stats(dN, region),
        "dN_m_domain_bounds": stats(dN, domain_box),
        "dN_m_full_grid": stats(dN),
        "dN_ngaEGM2008_m_region": stats(dN_nga, region),
        "N_EGM2008_NGA_minus_ICGEM_region": stats(d_nga, region),
        "N_EGM2008_NGA_minus_ICGEM_at_coincident_2.5min_nodes": stats(d_nga, region & on),
        "N_ETOPO2022geoid_minus_NGA_region": stats(d_etopo_nga, region),
        "N_ETOPO2022geoid_minus_ICGEM_EGM2008_region": stats(d_etopo_icgem, region),
        "N_EGM2008_ICGEM_region": stats(N["EGM2008"], region),
        "N_EIGEN6C4_ICGEM_region": stats(N["EIGEN-6C4"], region),
        "coarse_checks_15min": checks,
    }

    meta = {
        "title": "Geoid-height difference N(EGM2008) - N(EIGEN-6C4), North Caspian product box",
        "created_utc": now,
        "units": "m",
        "dN_m": "N_EGM2008 - N_EIGEN6C4, both from the ICGEM calculation service (GFZ), functional 'geoid', "
                "reference system WGS84 (a=6378137 m, 1/f=298.257223563, GM=3.986004418e14 m^3/s^2, "
                "omega=7.292115e-5 rad/s), tide system tide_free (the native system of both coefficient sets), "
                "zero-degree term 'included' (ICGEM: GM part only; (W0-U0)/gamma not applied), full degree 2190, "
                "no filter, height 0 on the ellipsoid. Point values at the nodes.",
        "conversion_convention": "A height above EGM2008 converts to the DAHITI (EIGEN-6C4) reference as "
                                 "H_D = H_EGM2008 + N_EGM2008 - N_EIGEN6C4 = H_EGM2008 + dN",
        "zero_degree_caveat": "dN_m is the pure model difference (identical GM in both models, so the ICGEM GM "
                              "zero-degree term cancels). The NGA/PROJ/GeographicLib EGM2008 grid (and ETOPO's "
                              "geoid layer) include the NGA constant HeightOffset -0.41 m and omit the GM term; "
                              "dN_ngaEGM2008_m uses that realisation of N_EGM2008 instead. Which one is right for "
                              "the DAHITI conversion depends on how DGFI evaluated N_EIGEN6C4 (zero-degree "
                              "handling), which is not documented on the pages read.",
        "dN_ngaEGM2008_m": "N_EGM2008(PROJ CDN us_nga_egm08_25.tif, bilinear at the nodes) - N_EIGEN6C4(ICGEM, as dN_m)",
        "N_egm2008_icgem_m": "ICGEM EGM2008 geoid (as above)",
        "N_eigen6c4_icgem_m": "ICGEM EIGEN-6C4 geoid (as above)",
        "N_egm2008_nga_m": "PROJ CDN us_nga_egm08_25.tif (GeographicLib egm2008, HeightOffset -0.41 m) bilinear at the nodes",
        "axes": "arrays are [lat, lon]; lat ascending (south to north), lon ascending; 1-D axes are cell centres "
                "of a regular 1 arc-minute lon/lat grid (WGS84), cells span centre +/- 0.5 arc-minute",
        "grid": {"lon0": LON0, "lat0": LAT0, "step_deg": STEP, "nlon": NLON, "nlat": NLAT,
                 "cell_edges_wsen": [LON0 - STEP / 2, LAT0 - STEP / 2, lon[-1] + STEP / 2, lat[-1] + STEP / 2]},
        "region_wsen": list(REGION),
        "domain_bounds_wsen": list(DOMAIN.bounds),
        "summary": summary,
        "sources": {
            "icgem_EGM2008": {"file": rel(gdf["EGM2008"]), "sha256": sha256(gdf["EGM2008"]), "gdf_header": heads["EGM2008"]},
            "icgem_EIGEN-6C4": {"file": rel(gdf["EIGEN-6C4"]), "sha256": sha256(gdf["EIGEN-6C4"]), "gdf_header": heads["EIGEN-6C4"]},
            "proj_window": {"file": rel(tif), "sha256": sha256(tif)},
            "etopo_2022": {"file": rel(etopo), "sha256": sha256(etopo)},
        },
        "provenance": "results/provenance/geoid_difference.json",
        "notes": "refs/notes/geoid_models.md",
    }
    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_NPZ, dN_m=dN.astype(np.float32), lon=lon, lat=lat,
                        dN_ngaEGM2008_m=dN_nga.astype(np.float32),
                        N_egm2008_icgem_m=N["EGM2008"].astype(np.float32),
                        N_eigen6c4_icgem_m=N["EIGEN-6C4"].astype(np.float32),
                        N_egm2008_nga_m=N_nga.astype(np.float32),
                        meta=np.array(json.dumps(meta)))
    # float32 round-trip check
    z = np.load(OUT_NPZ)
    assert np.nanmax(np.abs(z["dN_m"] - dN)) < 1e-5
    print(json.dumps(summary, indent=1))

    # --- provenance record ---
    def req(p):
        q = pathlib.Path(str(p) + ".request.json")
        return json.loads(q.read_text(encoding="utf-8")) if q.exists() else None

    raw = []
    for m, p in gdf.items():
        r = req(p)
        raw.append({"role": f"N_{m} on the 1' grid", "file": rel(p), "sha256": sha256(p),
                    "summary_page": rel(str(p) + ".summary.html"), "summary_page_sha256": sha256(str(p) + ".summary.html"),
                    "request": r})
    for name, p in chk_files.items():
        raw.append({"role": "coarse 15' check grid", "file": rel(p), "sha256": sha256(p), "request": req(p)})
    raw.append({"role": "N_EGM2008 (NGA/PROJ realisation), window", "file": rel(tif), "sha256": sha256(tif),
                "request": req(tif)})
    docs = []
    for p in sorted((RAW / "docs").glob("*")):
        if p.suffix in (".txt", ".html", ".pdf", ".json", ".sh") and not p.name.endswith(".headers.txt"):
            docs.append({"file": rel(p), "sha256": sha256(p)})
    prov = {
        "product": "geoid_egm2008_minus_eigen6c4",
        "created_utc": now,
        "output": {"file": rel(OUT_NPZ), "sha256": sha256(OUT_NPZ)},
        "builder_script": "scripts/a7_geoid_difference.py",
        "fetch_scripts": ["scripts/a7_geoid_icgem_fetch.py", "scripts/a7_geoid_proj_window.py"],
        "icgem_service": {
            "page": "https://icgem.gfz.de/calcgrid (http://icgem.gfz-potsdam.de/calcgrid redirects here)",
            "submit": "POST https://icgem.gfz.de/calcsrvgrid (multipart form, as the web form sends it)",
            "result": "GET https://icgem.gfz.de/calc_res_gdf/<calc_id> (same HTTP session)",
            "model_ids": {"EIGEN-6C4": "7fd8fe44aa1518cd79ca84300aef4b41ddb2364aef9e82b7cdaabdb60a9053f1",
                          "EGM2008": "c50128797a9cb62e936337c890e4425f03f0461d7329b09a8cc8561504465340"},
            "options_offered": {"tide_system": ["use model's system", "tide free", "zero tide", "mean tide"],
                                "zero_degree_term": "checkbox (default checked)",
                                "reference_systems": ["WGS84", "GRS80", "GRIM5", "EGM2008", "GRS67", "JGM3", "WGS72",
                                                      "NIMA96", "user defined"],
                                "filters": ["none", "half response", "half transfer", "6 sigma"]},
            "note_point_count": "The submission page reported 110308 grid points for the 1' request; the delivered "
                                "files contain 217 x 505 = 109585 nodes exactly on the requested axes (checked).",
        },
        "raw_downloads": raw,
        "documents_retrieved": docs,
        "summary": summary,
    }
    OUT_PROV.write_text(json.dumps(prov, indent=1), encoding="utf-8")
    print("wrote", OUT_NPZ, OUT_PROV)


if __name__ == "__main__":
    main()
