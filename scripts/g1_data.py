"""Generality test, inputs (PREREG Amendment 16 items 2, 4 and 7).

Commands (basin = great_salt_lake | lake_urmia):
  levels <basin>    GSL: USGS NWIS daily values (parameter 62614, ft above NGVD 29) of
                    both gauges, and NAVD 88 - NGVD 29 at each gauge from NOAA NCAT
                    (VERTCON 3.0); Urmia: DAHITI target 104 (API key from
                    caspian.credentials, never printed).
  products <basin>  the five global products over the basin box: GEBCO_2026 and
                    GEBCO_2022 (CEDA, HTTP range requests on the contiguous grids),
                    ETOPO 2022 15" (NCEI OPeNDAP), SRTM15+ V2.7 (Scripps, HDF5 range
                    reads), ETOPO1 (Earth Engine catalogue copy, native grid).
  osm <basin>       GSL: the Union Pacific Lakeside Subdivision, the West Davis County
                    Causeway and salt-evaporation ponds from OpenStreetMap (Overpass).
  vdatum <basin>    GSL: H(NAVD 88) - H(EGM2008) on a 0.05 deg grid over the box, by
                    PROJ (EGM2008 -> WGS84 -> NAD83(2011) -> GEOID18; grids from the
                    PROJ CDN). Urmia: not needed here (ICGEM grids).

Raw files (git-ignored): data/raw/generality/<basin>/...
Provenance (committed): results/provenance/generality_<basin>_<item>.json
Usage: python scripts/g1_data.py <command> <basin>
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import caspian  # noqa: F401
import numpy as np
import pandas as pd
import requests

from caspian import generality as G
from caspian.config import EE_PROJECT, RAW, RESULTS

UA = {"User-Agent": "caspian-research/0.1 (rezwan490@gmail.com)"}
NWIS = "https://waterservices.usgs.gov/nwis/dv/"
NCAT = "https://geodesy.noaa.gov/api/ncat/llh"
PROV = RESULTS / "provenance"


def raw_dir(basin: str) -> Path:
    d = RAW / "generality" / basin
    d.mkdir(parents=True, exist_ok=True)
    return d


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def write_prov(basin: str, item: str, rec: dict) -> None:
    PROV.mkdir(parents=True, exist_ok=True)
    (PROV / f"generality_{basin}_{item}.json").write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")


# --- levels --------------------------------------------------------------------------------

def parse_rdb(text: str) -> pd.DataFrame:
    rows = [l for l in text.splitlines() if l and not l.startswith("#")]
    hdr = rows[0].split("\t")
    data = [r.split("\t") for r in rows[2:]]
    df = pd.DataFrame(data, columns=hdr)
    val = [c for c in hdr if c.endswith("_62614_00003")]
    if len(val) != 1:
        raise ValueError(f"expected one 62614 mean column, got {val}")
    out = pd.DataFrame({"date": pd.to_datetime(df["datetime"]), "ft_ngvd29": pd.to_numeric(df[val[0]], errors="coerce"),
                        "qualifier": df[val[0] + "_cd"]})
    return out


def ncat_delta(lat: float, lon: float, h: float = 1280.0) -> dict:
    url = (f"{NCAT}?lat={lat}&lon={lon}&orthoHt={h}&inDatum=NAD83(1986)&outDatum=NAD83(1986)"
           f"&inVertDatum=NGVD29&outVertDatum=NAVD88")
    r = requests.get(url, timeout=120, headers=UA)
    r.raise_for_status()
    d = r.json()
    return {"url": url, "vertcon_version": d.get("vertconVersion"), "src_orthoht_m": float(d["srcOrthoht"]),
            "dest_orthoht_m": float(d["destOrthoht"]), "sigma_m": float(d["sigOrthoht"]),
            "delta_m": round(float(d["destOrthoht"]) - float(d["srcOrthoht"]), 4)}


def levels_gsl(basin: str) -> int:
    out = raw_dir(basin) / "levels"
    out.mkdir(exist_ok=True)
    rec = {"source": "USGS NWIS daily values, parameter 62614 (lake water-surface elevation above NGVD 1929, ft), "
                     "statistic 00003 (mean)", "retrieved_utc": now(), "gauges": {}}
    frames = []
    for arm, site in BASINS_GSL_GAUGES.items():
        url = f"{NWIS}?format=rdb&sites={site}&parameterCd=62614&statCd=00003&startDT=1983-01-01&endDT=2026-12-31"
        r = requests.get(url, timeout=300, headers=UA)
        r.raise_for_status()
        f = out / f"nwis_{site}_62614_dv.rdb"
        f.write_bytes(r.content)
        site_url = f"https://waterservices.usgs.gov/nwis/site/?format=rdb&sites={site}&siteOutput=expanded"
        st = requests.get(site_url, timeout=120, headers=UA).text
        rows = [l for l in st.splitlines() if not l.startswith("#")]
        meta = dict(zip(rows[0].split("\t"), rows[2].split("\t")))
        lat, lon = float(meta["dec_lat_va"]), float(meta["dec_long_va"])
        nc = ncat_delta(lat, lon)
        df = parse_rdb(r.text)
        df["arm"], df["site"] = arm, site
        df["level_navd88_m"] = G.ngvd29_ft_to_navd88_m(df["ft_ngvd29"], nc["delta_m"])
        frames.append(df)
        rec["gauges"][arm] = {"site": site, "station": meta.get("station_nm"), "lat": lat, "lon": lon,
                              "coord_datum": meta.get("dec_coord_datum_cd"), "url": url, "file": f.name,
                              "bytes": len(r.content), "sha256": sha(r.content), "n_days": int(df["ft_ngvd29"].notna().sum()),
                              "first": str(df["date"].min().date()), "last": str(df["date"].max().date()),
                              "qualifiers": df["qualifier"].value_counts().to_dict(),
                              "ngvd29_to_navd88": nc}
        print(arm, site, rec["gauges"][arm]["n_days"], rec["gauges"][arm]["first"], rec["gauges"][arm]["last"],
              "delta", nc["delta_m"], flush=True)
    lv = pd.concat(frames, ignore_index=True)
    lv.to_csv(out / "gsl_daily_levels.csv", index=False)
    rec["combined_file"] = "gsl_daily_levels.csv"
    rec["conversion"] = "level_navd88_m = 0.3048 * ft_ngvd29 + delta_m (VERTCON 3.0 via NOAA NCAT at the gauge)"
    write_prov(basin, "levels", rec)
    return 0


BASINS_GSL_GAUGES = G.BASINS["great_salt_lake"]["gauges"]


def levels_dahiti(basin: str) -> int:
    from caspian import credentials
    key = credentials.get_secret("DAHITI_API_KEY")
    if not key:
        print("FAIL DAHITI_API_KEY is not set")
        return 1
    api = "https://dahiti.dgfi.tum.de/api/v2"
    did = G.BASINS[basin]["dahiti_id"]
    r = requests.post(f"{api}/download-water-level/", data={"api_key": key, "dahiti_id": did, "format": "csv"},
                      timeout=300)
    if r.status_code != 200:
        print(f"FAIL download-water-level HTTP {r.status_code}")
        return 1
    out = raw_dir(basin) / "levels"
    out.mkdir(exist_ok=True)
    f = out / f"dahiti_{did}_water_level.csv"
    f.write_bytes(r.content)
    info = requests.post(f"{api}/get-target-info/", data={"api_key": key, "dahiti_id": did}, timeout=120)
    (out / f"dahiti_{did}_target_info.json").write_bytes(info.content if info.status_code == 200 else b"{}")
    write_prov(basin, "levels", {"dahiti_id": did, "endpoint": f"{api}/download-water-level/", "retrieved_utc": now(),
                                 "file": f.name, "bytes": len(r.content), "sha256": sha(r.content),
                                 "target_info_http_status": info.status_code})
    print("dahiti", did, len(r.content), "bytes")
    return 0


# --- products -------------------------------------------------------------------------------

def fetch_gebco(version: str, bounds, out: Path) -> dict:
    import fsspec
    import h5py
    import xarray as xr
    from caspian import netcdf3
    from caspian.remote import contiguous_window
    files = {"2026": ("ice_surface_elevation/netcdf/GEBCO_2026.nc", "type_identifier_grid/netcdf/gebco_2026_tid.nc",
                      "10.5285/4f68d5c7-45eb-f999-e063-7086abc036fa"),
             "2022": ("ice_surface_elevation/netcdf/GEBCO_2022_CF.nc", "type_identifier_grid/netcdf/GEBCO_2022_TID.nc",
                      "10.5285/e0f0bb80-ab44-2739-e053-6c86abc0289c")}[version]
    base = f"https://dap.ceda.ac.uk/bodc/gebco/global/gebco_{version}"
    eurl, turl = f"{base}/{files[0]}", f"{base}/{files[1]}"
    w, s, e, n = bounds
    with fsspec.open(eurl, "rb", block_size=2**20) as f, h5py.File(f, "r") as h:
        lat, lon = h["lat"][:], h["lon"][:]
        v = h["elevation"]
        if v.chunks is not None:
            raise RuntimeError("GEBCO elevation is chunked")
        offset, ncols, dtype = v.id.get_offset(), v.shape[1], v.dtype
    i0, i1 = np.searchsorted(lat, s), np.searchsorted(lat, n)
    j0, j1 = np.searchsorted(lon, w), np.searchsorted(lon, e)
    elev = contiguous_window(eurl, offset, ncols, dtype, i0, i1, j0, j1)
    hdr = netcdf3.header(turl)
    tv = hdr["variables"]["tid"]
    tid = contiguous_window(turl, tv["begin"], tv["dims"][1][1], "i1", i0, i1, j0, j1)
    ds = xr.Dataset({"elevation": (("lat", "lon"), elev.astype("int16")), "tid": (("lat", "lon"), tid.astype("int8"))},
                    coords={"lat": lat[i0:i1], "lon": lon[j0:j1]},
                    attrs={"source": f"GEBCO_{version} Grid via CEDA (HTTP range subset)", "doi": files[2]})
    ds.to_netcdf(out)
    return {"product": f"GEBCO_{version}", "doi": files[2], "urls": [eurl, turl], "window_index": [int(i0), int(i1), int(j0), int(j1)],
            "shape": list(elev.shape), "elevation_range_m": [int(elev.min()), int(elev.max())],
            "tid_values_present": sorted(int(x) for x in np.unique(tid))}


def fetch_etopo2022(tiles, bounds, out: Path) -> dict:
    import xarray as xr
    dods = "https://www.ngdc.noaa.gov/thredds/dodsC/global/ETOPO2022/15s"
    layers = {"elevation": ("15s_surface_elev_netcdf", "surface"), "source_id": ("15s_surface_sid_netcdf", "surface_sid")}
    w, s, e, n = bounds
    parts, urls, attrs = {}, {}, {}
    for name, (folder, suffix) in layers.items():
        pieces = []
        for tile in tiles:
            url = f"{dods}/{folder}/ETOPO_2022_v1_15s_{tile}_{suffix}.nc"
            urls.setdefault(name, []).append(url)
            ds = xr.open_dataset(url, engine="netcdf4")
            var = [v for v in ds.data_vars if ds[v].ndim == 2][0]
            sub = ds[var].sel(lat=slice(s, n), lon=slice(w, e))
            if sub.size:
                pieces.append(sub.load())
            attrs[name] = {k: str(v) for k, v in ds[var].attrs.items()}
        da = xr.combine_by_coords([p.to_dataset(name=name) for p in pieces])[name].sortby("lat").sortby("lon")
        parts[name] = da
    xr.Dataset(parts, attrs={"source": "ETOPO 2022 v1 15s via NCEI THREDDS OPeNDAP", "doi": "10.25921/fd45-gt74"}).to_netcdf(out)
    el = parts["elevation"].values
    sid = parts["source_id"].values
    return {"product": "ETOPO 2022 v1 15 arc-second", "doi": "10.25921/fd45-gt74", "urls": urls,
            "attrs": attrs, "shape": list(el.shape), "elevation_range_m": [float(np.nanmin(el)), float(np.nanmax(el))],
            "source_ids_present": sorted(int(v) for v in np.unique(sid[np.isfinite(sid)]))}


def fetch_srtm15(bounds, out: Path) -> dict:
    import xarray as xr
    from caspian.remote import h5_window
    base = "https://topex.ucsd.edu/pub/srtm15_plus"
    lat, lon, z, attrs_z, chunks = h5_window(f"{base}/SRTM15_V2.7.nc", "z", bounds, margin=0.0)
    try:
        lat2, lon2, sid, _, _ = h5_window(f"{base}/SID_V2.7.nc", "z", bounds, margin=0.0)
    except KeyError:
        lat2, lon2, sid, _, _ = h5_window(f"{base}/SID_V2.7.nc", "sid", bounds, margin=0.0)
    if not (np.allclose(lat, lat2) and np.allclose(lon, lon2)):
        raise RuntimeError("SRTM15+ SID grid not aligned")
    xr.Dataset({"elevation": (("lat", "lon"), z.astype("float32")), "source_id": (("lat", "lon"), sid.astype("int32"))},
               coords={"lat": lat, "lon": lon}, attrs={"source": "SRTM15+ V2.7 (Scripps), HTTP range subset"}).to_netcdf(out)
    return {"product": "SRTM15+ V2.7", "citation_doi": "10.1029/2019ea000658",
            "urls": [f"{base}/SRTM15_V2.7.nc", f"{base}/SID_V2.7.nc"], "shape": list(z.shape),
            "elevation_range_m": [float(z.min()), float(z.max())], "n_source_ids": int(np.unique(sid).size)}


def fetch_etopo1(bounds, out: Path) -> dict:
    import ee
    import rasterio
    from rasterio.transform import Affine
    from caspian import ee_io
    ee.Initialize(project=EE_PROJECT)
    asset = "NOAA/NGDC/ETOPO1"
    g = ee.data.getAsset(asset)["bands"][0]["grid"]["affineTransform"]
    sx, x0g, sy, y0g = g["scaleX"], g["translateX"], g["scaleY"], g["translateY"]
    w, s, e, n = bounds
    c0, c1 = math.floor((w - x0g) / sx), math.ceil((e - x0g) / sx)
    r0, r1 = math.floor((n - y0g) / sy), math.ceil((s - y0g) / sy)
    tr = [sx, 0, x0g + c0 * sx, 0, sy, y0g + r0 * sy]
    dims = (c1 - c0, r1 - r0)
    arr = ee_io.fetch(asset, ["bedrock", "ice_surface"], tr, dims, None, block=512)
    with rasterio.open(out, "w", driver="GTiff", width=dims[0], height=dims[1], count=2, dtype="int16",
                       crs="EPSG:4326", transform=Affine(tr[0], 0, tr[2], 0, tr[4], tr[5]), compress="deflate") as dst:
        dst.write(arr["bedrock"].astype(np.int16), 1)
        dst.write(arr["ice_surface"].astype(np.int16), 2)
        dst.set_band_description(1, "bedrock")
        dst.set_band_description(2, "ice_surface")
    return {"product": "ETOPO1 (bedrock, ice_surface), 1 arc-minute", "doi": "10.7289/V5C8276M",
            "source": f"Earth Engine catalogue {asset}", "native_affine": g, "subset_transform": tr, "subset_dims": dims,
            "bedrock_range_m": [int(arr["bedrock"].min()), int(arr["bedrock"].max())]}


FILES = {"gebco_2026": "gebco_2026.nc", "gebco_2022": "gebco_2022.nc", "etopo_2022": "etopo_2022.nc",
         "srtm15p_v2.7": "srtm15p_v2.7.nc", "etopo1": "etopo1.tif"}


def products_cmd(basin: str, only=None) -> int:
    b = G.BASINS[basin]
    w, s, e, n = b["bbox"]
    out = raw_dir(basin) / "bathymetry"
    out.mkdir(exist_ok=True)
    prov_path = PROV / f"generality_{basin}_products.json"
    rec = json.loads(prov_path.read_text(encoding="utf-8")) if prov_path.exists() else {}
    for name in (only or G.GLOBAL_PRODUCTS):
        f = out / FILES[name]
        t0 = time.time()
        if name == "gebco_2026":
            r = fetch_gebco("2026", b["bbox"], f)
        elif name == "gebco_2022":
            r = fetch_gebco("2022", b["bbox"], f)
        elif name == "etopo_2022":
            r = fetch_etopo2022(b["etopo_tiles"], b["bbox"], f)
        elif name == "srtm15p_v2.7":
            r = fetch_srtm15(b["bbox"], f)
        else:
            r = fetch_etopo1(b["bbox"], f)
        r.update(file=f.name, subset_sha256=sha(f.read_bytes()), retrieved_utc=now(), seconds=round(time.time() - t0, 1),
                 box_wsen=[w, s, e, n])
        rec[name] = r
        prov_path.write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
        print(name, r.get("shape") or r.get("subset_dims"), r.get("elevation_range_m") or r.get("bedrock_range_m"), flush=True)
    return 0


# --- OpenStreetMap (GSL) --------------------------------------------------------------------

def osm_cmd(basin: str) -> int:
    w, s, e, n = G.BASINS[basin]["bbox"]
    bb = f"{s},{w},{n},{e}"
    q = f"""[out:json][timeout:180];
(way["railway"="rail"]["name"="UP Lakeside Subdivision"]({bb});
 way["highway"]["name"="West Davis County Causeway"]({bb});
 way["landuse"="salt_pond"]({bb});
 relation["landuse"="salt_pond"]({bb});
);
out geom;"""
    r = requests.post("https://overpass-api.de/api/interpreter", data={"data": q}, headers=UA, timeout=300)
    r.raise_for_status()
    f = raw_dir(basin) / "osm_gsl.json"
    f.write_bytes(r.content)
    d = r.json()
    kinds = {}
    for el in d["elements"]:
        t = el.get("tags", {})
        k = t.get("name") if el["type"] == "way" and "railway" in t or "highway" in t else f"{el['type']}:salt_pond"
        kinds[k] = kinds.get(k, 0) + 1
    write_prov(basin, "osm", {"source": "OpenStreetMap via Overpass API (data (c) OpenStreetMap contributors, ODbL 1.0)",
                              "endpoint": "https://overpass-api.de/api/interpreter", "query": q, "retrieved_utc": now(),
                              "file": f.name, "bytes": len(r.content), "sha256": sha(r.content),
                              "osm_base": d.get("osm3s", {}).get("timestamp_osm_base"), "elements": kinds})
    print(kinds)
    return 0


# --- vertical datum grid (GSL) ---------------------------------------------------------------

def vdatum_cmd(basin: str) -> int:
    import pyproj
    pyproj.network.set_network_enabled(True)
    w, s, e, n = G.BASINS[basin]["bbox"]
    lon = np.arange(w - 0.05, e + 0.1, 0.05)
    lat = np.arange(s - 0.05, n + 0.1, 0.05)
    LON, LAT = np.meshgrid(lon, lat)
    # H_E (EGM2008) -> h on WGS84 (taken as ITRF2014, cm level) -> h on NAD83(2011) by the
    # 14-parameter ITRF2014 -> NAD83(2011) Helmert at epoch 2010.0 -> NAVD 88 by GEOID18.
    # (PROJ's direct EPSG:4326+3855 -> EPSG:6318+5703 route uses the null "NAD83(2011) to
    # WGS 84 (1)" step and drops the ~0.7 m ellipsoidal-height shift; not used.)
    EPOCH = 2010.0
    H0 = np.full(LON.shape, 1280.0)
    t1 = pyproj.Transformer.from_crs("EPSG:4326+3855", "EPSG:4979", always_xy=True)
    t2 = pyproj.Transformer.from_crs("EPSG:7912", "EPSG:6319", always_xy=True)
    t3 = pyproj.Transformer.from_crs("EPSG:6319", "EPSG:6318+5703", always_xy=True)
    x1, y1, h_i = t1.transform(LON, LAT, H0)
    x2, y2, h_n, _ = t2.transform(x1, y1, h_i, np.full(LON.shape, EPOCH))
    _, _, H88 = t3.transform(x2, y2, h_n)
    dN = (H88 - H0).astype(np.float32)
    if not np.all(np.isfinite(dN)) or np.nanmax(np.abs(dN)) > 5:
        print("FAIL implausible conversion", np.nanmin(dN), np.nanmax(dN))
        return 1
    out = raw_dir(basin) / "vdatum_navd88_minus_egm2008.npz"
    np.savez_compressed(out, lon=lon, lat=lat, dN=dN,
                        definition="H(NAVD 88, GEOID18) - H(EGM2008), m, for a point at 1280 m; add to EGM2008 heights")
    write_prov(basin, "vdatum", {"method": "pyproj: EPSG:4326+3855 -> EPSG:4979 (EGM2008 grid); EPSG:7912 -> EPSG:6319 "
                                           f"(ITRF2014 to NAD83(2011), epoch {EPOCH}); EPSG:6319 -> EPSG:6318+5703 (GEOID18)",
                                 "steps": [t1.description, t2.description, t3.description],
                                 "proj_version": pyproj.proj_version_str, "epoch": EPOCH,
                                 "grid_step_deg": 0.05, "file": out.name, "sha256": sha(out.read_bytes()),
                                 "dN_range_m": [float(dN.min()), float(dN.max())], "dN_mean_m": float(dN.mean()),
                                 "itrf_to_nad83_dh_m_range": [float((h_n - h_i).min()), float((h_n - h_i).max())],
                                 "retrieved_utc": now()})
    print([t1.description, t2.description, t3.description], "dN", float(dN.min()), float(dN.max()), float(dN.mean()))
    return 0


def main(argv) -> int:
    cmd, basin = argv[0], argv[1]
    if basin not in G.BASINS:
        raise SystemExit(f"unknown basin {basin}")
    if cmd == "levels":
        return levels_gsl(basin) if basin == "great_salt_lake" else levels_dahiti(basin)
    if cmd == "products":
        return products_cmd(basin, argv[2].split(",") if len(argv) > 2 else None)
    if cmd == "osm":
        return osm_cmd(basin)
    if cmd == "vdatum":
        return vdatum_cmd(basin)
    raise SystemExit(f"unknown command {cmd}")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
