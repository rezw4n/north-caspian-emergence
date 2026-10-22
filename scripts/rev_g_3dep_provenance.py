"""Provenance of the USGS 3DEP 1/3 arc-second seamless DEM over Great Salt Lake (revision G, task A).

The Great Salt Lake positive control used the Earth Engine collection USGS/3DEP/10m_collection
("USGS 3DEP 10m National Map Seamless (1/3 Arc-Second)"). This script establishes, from USGS
sources retrieved here, which source DEMs feed the seamless 1/3 arc-second DEM over the lake-bed
band, their collection dates and quality levels, and what the sources say about hydro-flattening.
It makes no Earth Engine API call. The only Google resource it reads is the static public STAC
record of the collection (a JSON file on storage.googleapis.com).

What it queries (all public, no credentials):
  1. TNM Access API (tnmaccess.nationalmap.gov/api/v1): the dataset list; products in the GSL box
     for the 1/3 arc-second DEM (all tile versions), the 1 m DEM (all pages), and the first page
     plus the total count for the lidar point cloud (LPC) and original-resolution (OPR) DEM.
  2. S3 bucket prd-tnm, StagedProducts/Elevation/13/TIFF: listings of the current/ and historical/
     folders of the six 1x1 degree tiles, and every tile-version FGDC XML and source-footprint
     GeoPackage (.gpkg). No elevation TIFF is downloaded.
  3. 3DEP Elevation Index ArcGIS MapServer (index.nationalmap.gov): service and layer descriptions;
     layer 21 (Query / 1/3 arc-second: national footprint of seamless-DEM sources), layer 24 (Query /
     Lidar Point Cloud: WESM work units with quality level) and layer 27 (Query / DEM - Source),
     attributes and geometries for the box. The service fails intermittently, so each query is
     retried and every attempt is logged.
  4. Work-unit metadata: for the 2016 Great Salt Lake QL1/QL2 work units, the folder listings, the
     project-level XML, the project report PDF and the Great Salt Lake breaklines shapefile; for the
     other work units that feed at least 1 % of a band, the folder listing and small project XMLs.
  5. Documentation: WESM data dictionary (general attributes), the USGS 3DEP data dictionary
     (spatial metadata glossary PDF), USGS TM 11-B9 (seamless DEM specification PDF) and the Earth
     Engine STAC record of USGS/3DEP/10m_collection.

What it computes:
  - lake-bed band (high & ~low & armlab == k, 250 m grid of data/interim/generality/great_salt_lake/
    static_250m.npz); shares of band cell centres inside each source polygon, per arm, for (a) the
    current tiles and (b) the tile versions current on 2022-05-04 (end of the temporal interval in
    the Earth Engine STAC record), plus per tile-version counts;
  - quality level (QL) per arm from the WESM work-unit footprints of the source project;
  - a consistency check of the paper's 60 m grid (dep3_60m.npy, produced earlier from Earth Engine)
    against the 2016 hydro-flattened water-surface elevations found in the breaklines;
  - the elevations of the hydro-flattened Lake Pond breaklines and the arm they fall in;
  - verbatim quotes (whitespace collapsed, soft hyphens at line ends removed) with file and location.

Writes:
  data/raw/revision_g/3dep/...                every raw response, each with <name>.meta.json holding
                                              URL, retrieval time (UTC), HTTP status, size, SHA-256
                                              and every attempt (git-ignored)
  results/revision/g/dep3_provenance.json     the summary

Run from the repository root:
  micromamba run -n caspian python -I scripts/rev_g_3dep_provenance.py [--refresh]
Responses already on disk are reused unless --refresh is given (failed ones are always retried).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
import re
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlencode

import fitz  # PyMuPDF, text extraction only
import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import requests
import shapely

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "revision_g" / "3dep"
OUT = ROOT / "results" / "revision" / "g" / "dep3_provenance.json"
GSL = ROOT / "data" / "interim" / "generality" / "great_salt_lake"
NPZ = GSL / "static_250m.npz"
DEP3 = GSL / "dep3_60m.npy"
DEP3_PROV = ROOT / "results" / "provenance" / "generality_great_salt_lake_dep3.json"

LAEA = "+proj=laea +lat_0=41.2 +lon_0=-112.5 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
BOX = (-113.17, 40.56, -111.82, 41.80)            # west, south, east, north
TILES = ["n41w114", "n41w113", "n41w112", "n42w114", "n42w113", "n42w112"]
EE_STAC_END = "20220504"                           # end of temporal interval in the EE STAC record
FLAT = {"north_1277.65": 1277.65, "south_1278.66": 1278.66}
FLAT_TOL = 0.01                                    # m, for the 60 m consistency check
ARMS = {1: "north", 2: "south"}

UA = "Mozilla/5.0 (compatible; caspian-provenance/1.0; research script)"
TNM = "https://tnmaccess.nationalmap.gov/api/v1"
S3 = "https://prd-tnm.s3.amazonaws.com"
IDX = "https://index.nationalmap.gov/arcgis/rest/services/3DEPElevationIndex/MapServer"
GSL_PROJECT = "UT_CentralUT_Southern_ID_2016_A16"
GSL_WU = {"UT_GreatSaltLakeQL1_2016": "QL1", "UT_GreatSaltLakeQL2_2016": "QL2"}
GSL_META = f"StagedProducts/Elevation/metadata/{GSL_PROJECT}"
GSL_XML = "best_use_xml/29083_Utah2016_GreatSaltLakeAOIs_ProjectLevel_Metadata.xml"
GSL_REPORT = "reports/29083_Utah2016_GreatSaltLake_ProjectReport.pdf"
GSL_BREAKLINES = "breaklines/GreatSaltLake_Breaklines"
DOCS = {
    "wesm_dictionary_general_attributes.html":
        "https://www.usgs.gov/ngp-standards-and-specifications/wesm-data-dictionary-general-attributes",
    "3DEP_Spatial_Metadata_Glossary_0.pdf":
        "https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/atoms/files/"
        "3DEP_Spatial_Metadata_Glossary_0.pdf",
    "tm11B9.pdf": "https://pubs.usgs.gov/tm/11b9/tm11B9.pdf",
    "ee_stac_USGS_3DEP_10m_collection.json":
        "https://storage.googleapis.com/earthengine-stac/catalog/USGS/USGS_3DEP_10m_collection.json",
}


def utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


# --- download with logging ---------------------------------------------------------------------

class Fetcher:
    """GET a URL, save the body under RAW, and write <name>.meta.json beside it."""

    def __init__(self, refresh: bool):
        self.s = requests.Session()
        self.s.headers["User-Agent"] = UA
        self.refresh = refresh
        self.log: list[dict] = []

    def get(self, name: str, url: str, *, arcgis: bool = False, retries: int = 1, timeout: int = 300) -> dict:
        path = RAW / name
        meta_path = path.with_name(path.name + ".meta.json")
        if not self.refresh and path.exists() and meta_path.exists():
            m = json.loads(meta_path.read_text(encoding="utf-8"))
            if m.get("ok"):
                m["reused_from_disk"] = True
                self.log.append(m)
                return m
        path.parent.mkdir(parents=True, exist_ok=True)
        attempts = []
        body, status, ctype, err = b"", None, None, "not attempted"
        for i in range(retries):
            t = utcnow()
            try:
                r = self.s.get(url, timeout=timeout)
                body, status, ctype = r.content, r.status_code, r.headers.get("Content-Type")
                err = None if status == 200 else f"HTTP {status}"
                if err is None and arcgis:
                    try:
                        j = json.loads(body)
                        if isinstance(j, dict) and "error" in j:
                            err = "ArcGIS error in body: " + json.dumps(j["error"])
                    except ValueError:
                        err = "response is not JSON"
            except requests.RequestException as e:
                body, status, ctype, err = b"", None, None, f"{type(e).__name__}: {e}"
            attempts.append({"retrieved_utc": t, "http_status": status, "error": err, "bytes": len(body),
                             "response_head": body[:300].decode("utf-8", "replace") if err else None})
            if err is None:
                break
            time.sleep(3 * (i + 1))
        path.write_bytes(body)
        m = {"file": rel(path), "url": url, "retrieved_utc": attempts[-1]["retrieved_utc"],
             "http_status": status, "content_type": ctype, "bytes": len(body),
             "sha256": hashlib.sha256(body).hexdigest(), "ok": err is None, "error": err,
             "attempts": attempts}
        meta_path.write_text(json.dumps(m, indent=1), encoding="utf-8")
        self.log.append(m)
        print(("ok  " if m["ok"] else "FAIL") + f" {status} {len(body):>10} {name}")
        return m


def jload(m: dict):
    return json.loads((ROOT / m["file"]).read_bytes()) if m["ok"] else None


def s3_list(f: Fetcher, name: str, prefix: str, delimiter: bool = False, max_pages: int = 5) -> tuple[list, dict]:
    """List an S3 prefix (ListObjectsV2). Returns ([{key, size, last_modified}], meta of first page)."""
    items, token, first = [], None, None
    for page in range(max_pages):
        q = {"list-type": "2", "prefix": prefix}
        if delimiter:
            q["delimiter"] = "/"
        if token:
            q["continuation-token"] = token
        m = f.get(f"{name}_p{page}.xml", f"{S3}/?{urlencode(q)}")
        first = first or m
        if not m["ok"]:
            break
        root = ET.fromstring((ROOT / m["file"]).read_bytes())
        ns = {"s": root.tag.split("}")[0].strip("{")}
        for c in root.findall("s:Contents", ns):
            items.append({"key": c.findtext("s:Key", namespaces=ns), "size": int(c.findtext("s:Size", namespaces=ns)),
                          "last_modified": c.findtext("s:LastModified", namespaces=ns)})
        if root.findtext("s:IsTruncated", namespaces=ns) != "true":
            break
        token = root.findtext("s:NextContinuationToken", namespaces=ns)
    return items, first


def esri_query(layer: int, extra: dict) -> str:
    q = {"geometry": ",".join(str(v) for v in BOX), "geometryType": "esriGeometryEnvelope", "inSR": "4326",
         "spatialRel": "esriSpatialRelIntersects", "where": "1=1", "f": "json"}
    q.update(extra)
    return f"{IDX}/{layer}/query?{urlencode(q)}"


def esri_date(v):
    if isinstance(v, (int, float)) and abs(v) > 1e10:
        return dt.datetime.fromtimestamp(v / 1000, dt.timezone.utc).strftime("%Y-%m-%d")
    return v


def esri_to_gdf(j: dict) -> gpd.GeoDataFrame:
    rows, geoms = [], []
    for ft in j.get("features", []):
        rows.append({k: esri_date(v) for k, v in ft["attributes"].items()})
        rings = (ft.get("geometry") or {}).get("rings", [])
        polys = []
        for ring in rings:   # ESRI: clockwise = outer, counter-clockwise = hole
            lr = shapely.LinearRing(ring)
            if not lr.is_ccw:
                polys.append([ring, []])
            elif polys:
                polys[-1][1].append(ring)
        geoms.append(shapely.make_valid(shapely.MultiPolygon([shapely.Polygon(o, h) for o, h in polys]))
                     if polys else None)
    return gpd.GeoDataFrame(rows, geometry=geoms, crs=4326)


# --- text and quotes ---------------------------------------------------------------------------

def norm(t: str) -> str:
    t = t.replace("­\n", "").replace("­", "").replace("‑", "-")
    return re.sub(r"\s+", " ", t).strip()


def pdf_pages(path: Path) -> list[str]:
    with fitz.open(path) as doc:
        return [norm(p.get_text()) for p in doc]


def html_text(path: Path) -> str:
    t = path.read_text(encoding="utf-8", errors="replace")
    t = re.sub(r"(?is)<(script|style).*?</\1>", " ", t)
    return norm(html.unescape(re.sub(r"<[^>]+>", " ", t)))


def find_quote(chunks: list[tuple[str, str]], start: str, end: str | None = None) -> tuple[str | None, str | None]:
    """chunks: [(location, normalized text)]. Returns (quote, location) from start through end."""
    for loc, txt in chunks:
        i = txt.find(start)
        if i < 0:
            continue
        if end is None:
            return txt[i:i + len(start)], loc
        j = txt.find(end, i)
        if j >= 0:
            return txt[i:j + len(end)], loc
    return None, None


# --- tile metadata -----------------------------------------------------------------------------

def parse_tile_xml(path: Path) -> dict:
    root = ET.fromstring(path.read_bytes())
    g = root.findtext
    return {"title": g("idinfo/citation/citeinfo/title"), "pubdate": g("idinfo/citation/citeinfo/pubdate"),
            "begdate": g("idinfo/timeperd/timeinfo/rngdates/begdate"),
            "enddate": g("idinfo/timeperd/timeinfo/rngdates/enddate"),
            "metadata_date": g("metainfo/metd"),
            "has_lineage_section": root.find("dataqual/lineage") is not None,
            "has_dataqual_section": root.find("dataqual") is not None,
            "keyword_Hydro-Flattened": any(k.text == "Hydro-Flattened" for k in root.iter("themekey")),
            "keyword_LIDAR": any(k.text == "LIDAR" for k in root.iter("themekey")),
            "vertical_datum": g("spref/vertdef/altsys/altdatum"),
            "work_unit_metadata_links": [o.text for o in root.findall("metainfo/metextns/onlink")
                                         if o.text and "metadata/" in o.text]}


def stamp_of(key: str) -> str | None:
    m = re.search(r"_(\d{8})\.(gpkg|xml)$", key)
    return m.group(1) if m else None


# --- band geometry -----------------------------------------------------------------------------

def band_points() -> dict:
    z = np.load(NPZ)
    a, _, c, _, e, f = z["transform"]
    ny, nx = z["high"].shape
    xs = c + a * (np.arange(nx) + 0.5)
    ys = f + e * (np.arange(ny) + 0.5)
    X, Y = np.meshgrid(xs, ys)
    tr = pyproj.Transformer.from_crs(LAEA, "EPSG:4326", always_xy=True)
    out = {}
    for k, arm in ARMS.items():
        band = z["high"] & ~z["low"] & (z["armlab"] == k)
        lon, lat = tr.transform(X[band], Y[band])
        out[arm] = pd.DataFrame({"lon": lon, "lat": lat, "tile": tile_of(lon, lat)})
    return out


def tile_of(lon, lat) -> np.ndarray:
    return np.array([f"n{int(np.floor(b)) + 1}w{int(np.floor(-a)) + 1}" for a, b in zip(lon, lat)])


def band_points_60m() -> dict:
    """60 m cells of dep3_60m.npy whose centre lies in a 250 m band cell; with the dep3 value."""
    z = np.load(NPZ)
    a, _, c, _, e, f = z["transform"]
    prov = json.loads(DEP3_PROV.read_text(encoding="utf-8"))
    t = prov["transform"]
    dep = np.load(DEP3)
    ny, nx = dep.shape
    X, Y = np.meshgrid(t[2] + t[0] * (np.arange(nx) + 0.5), t[5] + t[4] * (np.arange(ny) + 0.5))
    col = np.floor((X - c) / a).astype(int)
    row = np.floor((Y - f) / e).astype(int)
    H, W = z["high"].shape
    ok = (col >= 0) & (col < W) & (row >= 0) & (row < H)
    tr = pyproj.Transformer.from_crs(LAEA, "EPSG:4326", always_xy=True)
    out = {}
    for k, arm in ARMS.items():
        band = z["high"] & ~z["low"] & (z["armlab"] == k)
        sel = np.zeros_like(ok)
        sel[ok] = band[row[ok], col[ok]]
        lon, lat = tr.transform(X[sel], Y[sel])
        out[arm] = pd.DataFrame({"lon": lon, "lat": lat, "z": dep[sel], "tile": tile_of(lon, lat)})
    return out


# PMETHOD codes of the legacy spatial-metadata schema, as listed in the USGS 3DEP Data Dictionary
# (docs/3DEP_Spatial_Metadata_Glossary_0.pdf, PMETHOD).
PMETHOD = {0: "Unknown", 1: "Electronic Image Correlation (GPM II)", 2: "Manual Profiling", 3: "DLG2DEM",
           4: "DCASS", 5: "LT4X", 6: "Complex polynomial interpolation", 7: "Lidar",
           8: "Photogrammetric mass points and break lines", 9: "Digital camera correlation", 10: "Ifsar",
           11: "Other remote sensing technique", 12: "Topobathymetric model", 13: "Topobathymetric lidar",
           14: "Bathymetric lidar", 15: "Geiger mode lidar", 16: "Single photon lidar", 17: "Flash lidar",
           18: "Acoustic", 19: "Structure from motion"}
LIDAR_CODES = {7, 13, 14, 15, 16, 17}


def _val(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else x


def source_key(r) -> str:
    """One label per source. Two schemas occur in the tile GeoPackages: the current one (workpackage,
    p_method, collect_start, collect_end) and the legacy NED one (demname, pmethod code, s_date)."""
    if "workpackage" in r or "p_method" in r:
        wp = _val(r.get("workpackage"))
        wp = wp if isinstance(wp, str) and wp else "no work package"
        return f"{wp} | {r.get('p_method')} | {r.get('collect_start')}..{r.get('collect_end')}"
    code = _val(r.get("pmethod"))
    code = int(code) if code is not None else None
    name = PMETHOD.get(code, "unknown code")
    sdate = _val(r.get("s_date"))
    sdate = int(float(sdate)) if sdate not in (None, "") else None
    if code in LIDAR_CODES:
        return f"{r.get('demname')} | legacy schema pmethod {code} ({name}) | s_date {sdate}"
    return f"legacy quadrangle DEMs | legacy schema pmethod {code} ({name}) | s_date {sdate}"


def is_lidar_key(k: str) -> bool:
    parts = k.split(" | ")
    return len(parts) > 1 and "lidar" in parts[1].lower()


def family(k: str) -> str:
    """Coarse label: 2016 GSL lidar project, other lidar by first collection year, or non-lidar method."""
    parts = k.split(" | ")
    if len(parts) < 3:
        return k
    m = re.search(r"(\d{4})", parts[2])
    year = m.group(1) if m else "unknown year"
    if parts[0] in (GSL_PROJECT, *GSL_WU):
        return "lidar, 2016 Great Salt Lake QL1/QL2 project (UT_CentralUT_Southern_ID_2016_A16)"
    if is_lidar_key(k):
        return f"lidar, other project, collection starting {year}"
    meth = "LT4X" if "LT4X" in parts[1] else parts[1]
    return f"non-lidar ({meth}), source date {year}"


def classify(pts: pd.DataFrame, polys: dict[str, gpd.GeoDataFrame], keycol: str = "key") -> np.ndarray:
    """For each point, the key of the first polygon (in its tile's GeoDataFrame) containing it."""
    keys = np.full(len(pts), "NO SOURCE POLYGON", dtype=object)
    for tile, g in polys.items():
        idx = np.where(pts["tile"].values == tile)[0]
        if g is None or len(idx) == 0:
            continue
        pg = gpd.GeoDataFrame(geometry=gpd.points_from_xy(pts["lon"].values[idx], pts["lat"].values[idx]), crs=4326)
        j = gpd.sjoin(pg, g[[keycol, "geometry"]], how="left", predicate="within")
        first = j[~j.index.duplicated()][keycol]
        keys[idx] = first.fillna("NO SOURCE POLYGON").values
    return keys


def share_table(keys: np.ndarray, attrs: dict | None = None) -> list[dict]:
    n = len(keys)
    out = []
    for k, c in Counter(keys).most_common():
        row = {"source": k, "n_cells": int(c), "share": round(c / n, 4)}
        if attrs and k in attrs:
            row.update(attrs[k])
        out.append(row)
    return out


# --- main --------------------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-download responses already on disk")
    args = ap.parse_args()
    f = Fetcher(args.refresh)
    S: dict = {"script": "scripts/rev_g_3dep_provenance.py", "run_utc": utcnow(), "box_wgs84_wsen": BOX,
               "tiles": TILES, "earth_engine_api_used": False,
               "notes": ["Quotes are verbatim except that whitespace is collapsed to single spaces and soft "
                         "hyphens at line ends are removed (PDF text extraction).",
                         "Band = high & ~low & (armlab == k) on the 250 m grid of static_250m.npz; shares are "
                         "of band cell centres. Source polygons are in NAD83 (EPSG:4269) and are compared with "
                         "WGS84 points without a datum shift (difference of order 1 m)."]}

    # 1. TNM Access API ------------------------------------------------------------------------
    bbox = ",".join(str(v) for v in BOX)
    tnm: dict = {}
    m = f.get("tnm/datasets.json", f"{TNM}/datasets")
    tnm["datasets_query"] = {"url": m["url"], "ok": m["ok"], "http_status": m["http_status"]}
    if m["ok"]:
        def walk(items):
            for it in items:
                yield it
                yield from walk(it.get("tags") or [])
        tnm["elevation_datasets"] = [
            {"title": it.get("title"), "sbDatasetTag": it.get("sbDatasetTag"),
             "lastPublishedDate": it.get("lastPublishedDate"), "lastUpdatedDate": it.get("lastUpdatedDate")}
            for it in walk(jload(m)) if re.search(r"NED|DEM|Lidar|Elevation", str(it.get("sbDatasetTag")))]

    def products(name, ds, max_items, offset=0):
        q = {"datasets": ds, "bbox": bbox, "outputFormat": "JSON", "max": max_items, "offset": offset}
        return f.get(name, f"{TNM}/products?{urlencode(q)}")

    m = products("tnm/products_13arcsec.json", "National Elevation Dataset (NED) 1/3 arc-second", 500)
    j = jload(m) or {}
    tnm["products_13arcsec"] = {
        "url": m["url"], "ok": m["ok"], "http_status": m["http_status"], "total": j.get("total"),
        "items": sorted([{"title": it["title"], "publicationDate": it.get("publicationDate"),
                          "lastUpdated": it.get("lastUpdated"), "downloadURL": it.get("downloadURL"),
                          "metaUrl": it.get("metaUrl"), "vendorMetaUrl": it.get("vendorMetaUrl")}
                         for it in j.get("items", [])], key=lambda r: r["downloadURL"] or "")}
    one_m, off, total = [], 0, None
    while True:
        m = products(f"tnm/products_1m_offset{off}.json", "Digital Elevation Model (DEM) 1 meter", 100, off)
        j = jload(m) or {}
        total = j.get("total", total)
        one_m += j.get("items", [])
        off += 100
        if not m["ok"] or not j.get("items") or off >= (total or 0):
            break
    proj_1m = defaultdict(lambda: {"n_tiles": 0, "publicationDates": set()})
    for it in one_m:   # titles: "USGS 1 Meter 12 x31y450 UT_WestEast_B22" or "USGS one meter x31y450 UT BoxElder 2018"
        mt = re.search(r"x\d+y\d+\s+(.+)$", it["title"])
        p = mt.group(1).strip() if mt else it["title"]
        proj_1m[p]["n_tiles"] += 1
        proj_1m[p]["publicationDates"].add(it.get("publicationDate"))
    tnm["products_1m"] = {"total": total, "n_retrieved": len(one_m),
                          "by_project": {k: {"n_tiles": v["n_tiles"],
                                             "publicationDates": sorted(x for x in v["publicationDates"] if x)}
                                         for k, v in sorted(proj_1m.items())}}
    for key, ds in (("products_lpc_first_page", "Lidar Point Cloud (LPC)"),
                    ("products_opr_first_page", "Original Product Resolution (OPR) Digital Elevation Model (DEM)")):
        m = products(f"tnm/{key}.json", ds, 50)
        j = jload(m) or {}
        tnm[key] = {"url": m["url"], "ok": m["ok"], "total": j.get("total"),
                    "note": "first page of 50 only; WESM (index layer 24) is used for work units",
                    "titles_first_page": [it["title"] for it in j.get("items", [])][:10]}
    S["tnm_access_api"] = tnm

    # 2. Tiles: S3 listings, XML and GeoPackage of every version -------------------------------
    tiles: dict = {}
    gpkg_paths: dict = {}
    for t in TILES:
        rec = {"current": {}, "versions": []}
        cur, mc = s3_list(f, f"s3_listings/current_{t}", f"StagedProducts/Elevation/13/TIFF/current/{t}/")
        his, mh = s3_list(f, f"s3_listings/historical_{t}", f"StagedProducts/Elevation/13/TIFF/historical/{t}/")
        rec["listing_urls"] = [mc["url"], mh["url"]]
        for it in cur:
            if it["key"].endswith((".xml", ".gpkg")):
                fn = it["key"].split("/")[-1]
                mm = f.get(f"tiles/current/{t}/{fn}", f"{S3}/{it['key']}")
                rec["current"][fn] = {"url": mm["url"], "ok": mm["ok"], "last_modified": it["last_modified"],
                                      "bytes": it["size"]}
                if fn.endswith(".xml") and mm["ok"]:
                    rec["current"]["xml_parsed"] = parse_tile_xml(ROOT / mm["file"])
                if fn.endswith(".gpkg") and mm["ok"]:
                    gpkg_paths[(t, "current")] = ROOT / mm["file"]
            elif it["key"].endswith(".tif"):
                rec["current"]["tif_last_modified"] = it["last_modified"]
                rec["current"]["tif_bytes_not_downloaded"] = it["size"]
        stamps = sorted({stamp_of(it["key"]) for it in his if stamp_of(it["key"])})
        for st in stamps:
            v = {"stamp": st}
            for it in his:
                fn = it["key"].split("/")[-1]
                if stamp_of(it["key"]) != st:
                    continue
                if it["key"].endswith((".xml", ".gpkg")):
                    mm = f.get(f"tiles/historical/{t}/{fn}", f"{S3}/{it['key']}")
                    v[fn] = {"url": mm["url"], "ok": mm["ok"], "last_modified": it["last_modified"]}
                    if fn.endswith(".xml") and mm["ok"]:
                        v["xml_parsed"] = parse_tile_xml(ROOT / mm["file"])
                    if fn.endswith(".gpkg") and mm["ok"]:
                        gpkg_paths[(t, st)] = ROOT / mm["file"]
            rec["versions"].append(v)
        cx = rec["current"].get("xml_parsed", {})
        rec["current_matches_version"] = next((v["stamp"] for v in rec["versions"]
                                               if v.get("xml_parsed", {}).get("title") == cx.get("title")), None)
        tiles[t] = rec
    S["tiles"] = tiles

    def load_gpkg(p: Path | None) -> gpd.GeoDataFrame | None:
        if p is None:
            return None
        g = gpd.read_file(p).to_crs(4326)
        g.columns = [c.lower() if c != "geometry" else c for c in g.columns]
        g = g.set_geometry("geometry")
        for c in ("collect_start", "collect_end", "pub_date"):
            if c in g:
                g[c] = g[c].astype(str).str.slice(0, 10)
        g["key"] = [source_key(r) for r in g.drop(columns="geometry").to_dict("records")]
        return g

    gp = {k: load_gpkg(p) for k, p in gpkg_paths.items()}
    attrs: dict = {}
    for g in gp.values():
        if g is None:
            continue
        for r in g.drop(columns="geometry").to_dict("records"):
            a = attrs.setdefault(r["key"], {"schema": "current" if "p_method" in r else "legacy NED"})
            for c in ("workpackage", "p_method", "p_device", "collect_start", "collect_end", "metadata_link",
                      "demname", "pdevice", "psite", "s_date", "quaddate"):
                v = _val(r.get(c))
                if v is not None and c not in a:
                    a[c] = v if not isinstance(v, (np.integer, np.floating)) else v.item()
            for c in ("quadname", "freetext"):
                v = _val(r.get(c))
                if isinstance(v, str) and v.strip():
                    s = a.setdefault(c + "_examples", [])
                    if norm(v) not in s and len(s) < 12:
                        s.append(norm(v))

    # 3. 3DEP Elevation Index (WESM) ----------------------------------------------------------
    idx: dict = {}
    m = f.get("index/service.json", f"{IDX}?f=json")
    if m["ok"]:
        idx["layers"] = [{"id": l["id"], "name": l["name"], "parentLayerId": l.get("parentLayerId")}
                         for l in jload(m).get("layers", [])]
    for L in (21, 24, 27):
        f.get(f"index/layer{L}.json", f"{IDX}/{L}?f=json")
    q = {
        "layer27_dem_source_attrs": (27, {"outFields": "*", "returnGeometry": "false"}),
        "layer24_lpc_attrs": (24, {"outFields": "*", "returnGeometry": "false"}),
        "layer21_seamless13_attrs": (21, {"outFields": "*", "returnGeometry": "false"}),
        "layer24_lpc_geom": (24, {"outFields": "workunit,workunit_id,project,ql,collect_start,collect_end",
                                  "returnGeometry": "true", "outSR": "4326"}),
        "layer21_seamless13_geom": (21, {"outFields": "*", "returnGeometry": "true", "outSR": "4326"}),
        "layer27_dem_source_geom_generalized": (27, {"outFields": "workunit,project,ql", "returnGeometry": "true",
                                                     "outSR": "4326", "maxAllowableOffset": "0.0001",
                                                     "geometryPrecision": "6"}),
    }
    qres = {}
    for name, (L, extra) in q.items():
        m = f.get(f"index/{name}.json", esri_query(L, extra), arcgis=True, retries=6)
        qres[name] = m
        idx.setdefault("queries", {})[name] = {
            "url": m["url"], "ok": m["ok"], "http_status": m["http_status"], "error": m["error"],
            "n_attempts": len(m.get("attempts", [])),
            "failed_attempts": [a for a in m.get("attempts", []) if a["error"]]}
    wesm = []
    j = jload(qres["layer24_lpc_attrs"]) if qres["layer24_lpc_attrs"]["ok"] else None
    j27 = jload(qres["layer27_dem_source_attrs"]) if qres["layer27_dem_source_attrs"]["ok"] else None
    keep = ("workunit", "workunit_id", "project", "collect_start", "collect_end", "ql", "spec", "p_method",
            "dem_gsd_meters", "vert_crs", "geoid", "lpc_pub_date", "sourcedem_pub_date", "seamless_category",
            "seamless_reason", "lpc_link", "sourcedem_link", "metadata_link")
    in27 = {ft["attributes"]["workunit"] for ft in (j27 or {}).get("features", [])}
    for ft in (j or j27 or {}).get("features", []):
        a = {k: esri_date(v) for k, v in ft["attributes"].items() if k in keep}
        a["in_dem_source_layer27"] = a["workunit"] in in27
        wesm.append(a)
    idx["work_units_intersecting_box"] = sorted(wesm, key=lambda r: (r["collect_start"] or "", r["workunit"]))
    j21 = jload(qres["layer21_seamless13_attrs"]) if qres["layer21_seamless13_attrs"]["ok"] else None
    idx["seamless13_sources_intersecting_box"] = sorted(
        [{k: v for k, v in ft["attributes"].items() if k not in ("SHAPE_Area", "SHAPE_Length")}
         for ft in (j21 or {}).get("features", [])], key=lambda r: (str(r.get("collect_start")), str(r.get("project"))))
    S["index_service"] = idx

    # 4. Band shares by source ------------------------------------------------------------------
    pts = band_points()
    snap_def = {"current": {t: (t, "current") for t in TILES}}
    asof = {}
    for t in TILES:
        cands = [st for (tt, st) in gp if tt == t and st != "current" and st <= EE_STAC_END]
        asof[t] = (t, max(cands)) if cands else None
    snap_def[f"as_of_{EE_STAC_END}"] = asof
    shares: dict = {"snapshots": {}, "per_tile_version": {}}
    snap_keys: dict = {}
    for sname, sd in snap_def.items():
        polys = {t: (gp.get(v) if v else None) for t, v in sd.items()}
        rec = {"tile_versions_used": {t: (v[1] if v else None) for t, v in sd.items()}}
        for arm, df in pts.items():
            keys = classify(df, polys)
            snap_keys[(sname, arm)] = keys
            lidar = np.array([is_lidar_key(k) for k in keys])
            fam = np.array([family(k) for k in keys], dtype=object)
            rec[arm] = {"n_band_cells_250m": int(len(keys)),
                        "share_lidar_source": round(float(lidar.mean()), 4),
                        "share_non_lidar_source": round(float(1 - lidar.mean()), 4),
                        "by_method_and_collection_year": share_table(fam),
                        "by_source": share_table(keys, attrs),
                        "cells_by_tile": dict(Counter(df["tile"]))}
        shares["snapshots"][sname] = rec
    for (t, st), g in sorted(gp.items()):
        row = {}
        for arm, df in pts.items():
            sub = df[df["tile"] == t]
            if len(sub):
                row[arm] = dict(Counter(classify(sub.reset_index(drop=True), {t: g})))
        if row:
            shares["per_tile_version"][f"{t}_{st}"] = row

    # Quality level from WESM work units (layer 24 geometry) of the source project
    if qres["layer24_lpc_geom"]["ok"]:
        wu = esri_to_gdf(jload(qres["layer24_lpc_geom"]))
        wu["wu_key"] = wu["workunit"] + " (" + wu["ql"].fillna("?") + ")"
        for sname in snap_def:
            for arm, df in pts.items():
                keys = snap_keys[(sname, arm)]
                proj = np.array([k.split(" | ")[0] for k in keys])
                pg = gpd.GeoDataFrame({"proj": proj}, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs=4326)
                jn = gpd.sjoin(pg, wu[["project", "workunit", "wu_key", "ql", "geometry"]], how="left",
                               predicate="within")
                # current schema names the project; legacy schema names the work unit (demname), sometimes
                # with '-' where WESM has '_' (e.g. UT_Wasatch-L4_2013 vs UT_Wasatch_L4_2013)
                def nm(s):
                    return s.fillna("").str.replace("-", "_").str.lower()
                jn = jn[(nm(jn["proj"]) == nm(jn["project"])) | (nm(jn["proj"]) == nm(jn["workunit"]))]
                lab = jn.groupby(level=0)["wu_key"].agg(lambda s: " + ".join(sorted(set(s))))
                qlab = jn.groupby(level=0)["ql"].agg(lambda s: " + ".join(sorted(set(s.fillna("?")))))
                wk = np.array(["no WESM work unit of the source project (non-lidar source or none)"] * len(df),
                              dtype=object)
                ql = wk.copy()
                wk[lab.index.values] = lab.values
                ql[qlab.index.values] = qlab.values
                shares["snapshots"][sname][arm]["by_wesm_work_unit"] = share_table(wk)
                shares["snapshots"][sname][arm]["by_quality_level"] = share_table(ql)
    # Cross-check with the national seamless-source layer 21
    if qres["layer21_seamless13_geom"]["ok"]:
        g21 = esri_to_gdf(jload(qres["layer21_seamless13_geom"]))
        g21["key"] = (g21["project"].fillna("no project") + " | " + g21["p_method"].astype(str) + " | "
                      + g21["collect_start"].astype(str) + ".." + g21["collect_end"].astype(str))
        shares["index_layer21_crosscheck"] = {}
        for arm, df in pts.items():
            pg = gpd.GeoDataFrame(geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs=4326)
            jn = gpd.sjoin(pg, g21[["key", "pub_date", "geometry"]], how="left", predicate="within")
            n_multi = int(jn.index.duplicated().sum())
            jn = jn[~jn.index.duplicated()]
            shares["index_layer21_crosscheck"][arm] = {
                "by_source": share_table(jn["key"].fillna("NO SOURCE POLYGON").values),
                "n_cells_in_more_than_one_polygon": n_multi}
    S["band_shares"] = shares

    # 5. 2016 GSL work units: metadata, report, breaklines ---------------------------------------
    gslw: dict = {"work_units": {}}
    for wu_name in GSL_WU:
        items, mm = s3_list(f, f"workunits/{wu_name}/listing", f"{GSL_META}/{wu_name}/")
        gslw["work_units"][wu_name] = {"listing_url": mm["url"], "n_keys": len(items),
                                       "pdf_reports": [(i["key"].split("/")[-1], i["size"]) for i in items
                                                       if i["key"].endswith(".pdf")][:20]}
        m = f.get(f"workunits/{wu_name}/project_metadata.xml", f"{S3}/{GSL_META}/{wu_name}/{GSL_XML}")
        gslw["work_units"][wu_name]["project_xml"] = {"url": m["url"], "ok": m["ok"], "sha256": m["sha256"]}
    xml_path = RAW / "workunits" / "UT_GreatSaltLakeQL1_2016" / "project_metadata.xml"
    m = f.get("workunits/UT_GreatSaltLakeQL1_2016/project_report.pdf",
              f"{S3}/{GSL_META}/UT_GreatSaltLakeQL1_2016/{GSL_REPORT}")
    report_path = ROOT / m["file"] if m["ok"] else None
    gslw["project_report"] = {"url": m["url"], "ok": m["ok"], "bytes": m["bytes"]}
    for ext in ("shp", "shx", "dbf", "prj"):
        m = f.get(f"workunits/UT_GreatSaltLakeQL1_2016/breaklines/GreatSaltLake_Breaklines.{ext}",
                  f"{S3}/{GSL_META}/UT_GreatSaltLakeQL1_2016/{GSL_BREAKLINES}.{ext}")
        gslw.setdefault("breaklines_files", []).append({"url": m["url"], "ok": m["ok"], "bytes": m["bytes"]})
    if xml_path.exists():
        r = ET.fromstring(xml_path.read_bytes())
        gslw["project_xml_parsed"] = {
            "title": r.findtext("idinfo/citation/citeinfo/title"),
            "origin": r.findtext("idinfo/citation/citeinfo/origin"),
            "pubdate": r.findtext("idinfo/citation/citeinfo/pubdate"),
            "begdate": r.findtext("idinfo/timeperd/timeinfo/rngdates/begdate"),
            "enddate": r.findtext("idinfo/timeperd/timeinfo/rngdates/enddate"),
            "current": r.findtext("idinfo/timeperd/current"),
            "vertical_datum": r.findtext("spref/vertdef/altsys/altdatum"),
            "lidar_spec": r.findtext("idinfo/descript/lidar/ldrinfo/ldrspec"),
            "procstep_titles": [norm(p.findtext("procdesc") or "").split(":")[0]
                                for p in r.findall("dataqual/lineage/procstep")]}

    # Breaklines: elevations of the hydro-flattened Lake Pond features and their arm
    shp = RAW / "workunits" / "UT_GreatSaltLakeQL1_2016" / "breaklines" / "GreatSaltLake_Breaklines.shp"
    if shp.exists():
        bl = gpd.read_file(shp)
        zz = np.load(NPZ)
        a, _, c, _, e, f0 = zz["transform"]
        H, W = zz["armlab"].shape
        bll = bl.to_crs(LAEA)
        bl4 = bl.to_crs(4326)
        zc = [shapely.get_coordinates(gm, include_z=True)[:, 2] for gm in bl.geometry]
        bl["zmin"] = [float(v.min()) for v in zc]
        bl["zmax"] = [float(v.max()) for v in zc]
        rows = []
        lp = bl[(bl["B_LINE_TY"] == "Lake Pond") & ((bl["zmax"] - bl["zmin"]) < 0.005)].copy()
        lp["z01"] = lp["zmin"].round(2)
        for zval, grp in lp.groupby("z01"):
            length_km = float(bll.loc[grp.index].length.sum() / 1000)
            if length_km < 20:
                continue
            xy = np.vstack([shapely.get_coordinates(bll.geometry[i]) for i in grp.index])
            cc = np.floor((xy[:, 0] - c) / a).astype(int)
            rr = np.floor((xy[:, 1] - f0) / e).astype(int)
            ok = (cc >= 0) & (cc < W) & (rr >= 0) & (rr < H)
            lab = Counter(zz["armlab"][rr[ok], cc[ok]].tolist())
            ll = np.vstack([shapely.get_coordinates(bl4.geometry[i]) for i in grp.index])
            rows.append({"z_m_navd88_geoid12b": round(float(zval), 3), "n_features": int(len(grp)),
                         "total_length_km": round(length_km, 1),
                         "vertices_by_armlab": {ARMS.get(k, f"armlab_{k}"): int(v) for k, v in sorted(lab.items())},
                         "lon_range": [round(float(ll[:, 0].min()), 3), round(float(ll[:, 0].max()), 3)],
                         "lat_range": [round(float(ll[:, 1].min()), 3), round(float(ll[:, 1].max()), 3)]})
        gslw["breaklines_summary"] = {
            "file": rel(shp), "crs": str(bl.crs), "n_features": int(len(bl)),
            "feature_types": dict(Counter(bl["B_LINE_TY"])),
            "lake_pond_constant_elevation_groups_over_20km": sorted(rows, key=lambda r: -r["total_length_km"]),
            "note": "Z values read from the 3D polylines; armlab is the arm label of the 250 m grid at each vertex."}
    S["gsl_2016_lidar"] = gslw

    # 6. Consistency check: the paper's 60 m grid against the 2016 flattened elevations ----------
    if DEP3.exists() and DEP3_PROV.exists():
        p60 = band_points_60m()
        chk = {"file": rel(DEP3), "provenance": rel(DEP3_PROV), "tolerance_m": FLAT_TOL, "snapshots": {}}
        for sname, sd in snap_def.items():
            polys = {t: (gp.get(v) if v else None) for t, v in sd.items()}
            chk["snapshots"][sname] = {}
            for arm, df in p60.items():
                keys = classify(df, polys)
                rows = []
                for k in pd.unique(keys):
                    zv = df["z"].values[keys == k]
                    zv = zv[np.isfinite(zv)]
                    if len(zv) == 0:
                        continue
                    mode_v, mode_n = Counter(np.round(zv, 2).tolist()).most_common(1)[0]
                    rows.append({"source": k, "n_cells_60m": int(len(zv)), "median_z": round(float(np.median(zv)), 3),
                                 **{f"share_within_tol_of_{n}": round(float(np.mean(np.abs(zv - v) <= FLAT_TOL)), 4)
                                    for n, v in FLAT.items()},
                                 "modal_z_rounded_0.01": mode_v, "modal_share": round(mode_n / len(zv), 4)})
                chk["snapshots"][sname][arm] = sorted(rows, key=lambda r: -r["n_cells_60m"])
        S["dep3_60m_consistency_check"] = chk

    # 7. Metadata of the other work units that feed >= 1 % of a band -----------------------------
    other = set()
    for sname in snap_def:
        for arm in ARMS.values():
            for r in S["band_shares"]["snapshots"][sname][arm].get("by_wesm_work_unit", []):
                if r["share"] >= 0.01 and not r["source"].startswith("no WESM"):
                    for w in r["source"].split(" + "):
                        other.add(w.split(" (")[0])
    other -= set(GSL_WU)
    wl = {w["workunit"]: w for w in wesm}
    om: dict = {}
    for w in sorted(other):
        link = (wl.get(w) or {}).get("metadata_link") or ""
        pref = link.split("prefix=")[-1].rstrip("/") + "/" if "prefix=" in link else None
        rec = {"metadata_link": link}
        if pref:
            items, mm = s3_list(f, f"workunits/{w}/listing", pref, max_pages=3)
            rec["listing_url"], rec["n_keys_listed"] = mm["url"], len(items)
            pick = [i for i in items if i["key"].lower().endswith(".xml") and i["size"] < 300_000 and
                    re.search(r"best_use_xml|vendor_provided_xml|project", i["key"], re.I)][:3]
            rec["xml"] = []
            for i in pick:
                fn = i["key"].split("/")[-1]
                m = f.get(f"workunits/{w}/{fn}", f"{S3}/{i['key']}")
                x = {"url": m["url"], "ok": m["ok"]}
                if m["ok"]:
                    txt = (ROOT / m["file"]).read_text(encoding="utf-8", errors="replace")
                    try:
                        r = ET.fromstring((ROOT / m["file"]).read_bytes())
                        x["title"] = r.findtext("idinfo/citation/citeinfo/title")
                        x["begdate"] = r.findtext(".//timeperd/timeinfo/rngdates/begdate")
                        x["enddate"] = r.findtext(".//timeperd/timeinfo/rngdates/enddate")
                    except ET.ParseError as e:
                        x["parse_error"] = str(e)
                    sents = re.split(r"(?<=[.!?])\s+", norm(html.unescape(re.sub(r"<[^>]+>", " ", txt))))
                    x["hydro_flattening_sentences_verbatim"] = [s for s in sents if
                                                                re.search(r"hydro[\s-]*flatten", s, re.I)][:5]
                rec["xml"].append(x)
        om[w] = rec
    S["other_contributing_work_units_metadata"] = om

    # 8. Documentation --------------------------------------------------------------------------
    docs = {}
    for fn, url in DOCS.items():
        m = f.get(f"docs/{fn}", url)
        docs[fn] = {"url": url, "ok": m["ok"], "http_status": m["http_status"], "file": m["file"]}
    S["documentation_files"] = docs
    stac = jload(f.get("docs/ee_stac_USGS_3DEP_10m_collection.json", DOCS["ee_stac_USGS_3DEP_10m_collection.json"]))
    if stac:
        S["earth_engine_stac_record"] = {"id": stac.get("id"), "title": stac.get("title"),
                                         "temporal_interval": stac.get("extent", {}).get("temporal", {}).get("interval"),
                                         "gee:user_uploaded": stac.get("gee:user_uploaded"),
                                         "version": stac.get("version")}

    # 9. Verbatim quotes ------------------------------------------------------------------------
    quotes = []

    def add(claim, file: Path | None, chunks, start, end=None, source=None):
        q, loc = find_quote(chunks, start, end) if chunks else (None, None)
        quotes.append({"claim": claim, "file": rel(file) if file else None, "location": loc, "quote": q,
                       "found": q is not None, **(source or {})})

    def pdf_chunks(p: Path | None):
        return [(f"PDF page {i + 1}", t) for i, t in enumerate(pdf_pages(p))] if p and p.exists() else []

    src_tile = {"source": "USGS 1/3 Arc Second n41w113 tile metadata (FGDC XML), U.S. Geological Survey"}
    tx = RAW / "tiles" / "current" / "n41w113" / "USGS_13_n41w113.xml"
    if tx.exists():
        add("Seamless DEM is built from diverse sources", tx,
            [("idinfo/descript/abstract", norm(ET.fromstring(tx.read_bytes()).findtext("idinfo/descript/abstract")))],
            "Seamless 3DEP data are derived from diverse source data", "unit of vertical measure.", src_tile)
    src_x = {"source": "Utah 2016 LiDAR - Great Salt Lake AOIs, project-level FGDC metadata, Quantum Spatial, Inc. "
                       "(pubdate 20171024), distributed by USGS"}
    if xml_path.exists():
        r = ET.fromstring(xml_path.read_bytes())
        ch = [("idinfo/descript/abstract", norm(r.findtext("idinfo/descript/abstract"))),
              ("idinfo/descript/purpose", norm(r.findtext("idinfo/descript/purpose")))]
        ch += [(f"dataqual/lineage/procstep[{i + 1}]/procdesc", norm(p.findtext("procdesc")))
               for i, p in enumerate(r.findall("dataqual/lineage/procstep"))]
        add("2016 collection, ground conditions", xml_path, ch, "Ground Conditions: Lidar was partially collected",
            "normal levels.", src_x)
        add("Purpose: hydro-flattened DEMs", xml_path, ch, "The purpose of these lidar data was to produce",
            "cell sizes.", src_x)
        add("Lake elevations assigned (hydro-flattening)", xml_path, ch,
            "Elevation values were assigned to all Inland Ponds and Lakes", "using TerraModeler functionality.", src_x)
        add("Datum of the lidar", xml_path, ch, "vertical datum of NAVD88 (GEOID12B), meters.", None, src_x)
    src_r = {"source": "Utah 2016 - Great Salt Lake AOIs LiDAR Project Report, Quantum Spatial, Inc., submitted "
                       "August 14, 2017 (Contract # AV2408), distributed by USGS"}
    rc = pdf_chunks(report_path)
    add("Acquisition dates", report_path, rc, "LiDAR data was acquired on September 3, 2016", "48 total lifts.", src_r)
    add("Sub-AOI areas", report_path, rc, "Great Salt Lake QL1 1,537 km2", "Great Salt Lake QL2 2,147 km2", src_r)
    add("Hydro-flattened deliverables", report_path, rc, "Hydro-flattened breaklines in Esri shapefile format",
        "1-meter hydro-flattened bare-earth raster DEM, tiled, in ERDAS .IMG format - QL2", src_r)
    add("Datum of deliverables", report_path, rc, "All geospatial deliverables were produced in NAD83 UTM Zone 12",
        "NAVD88 (GEOID12B), meters.", src_r)
    add("Lift on Nov 30, 2016 in the GSL QL1/QL2 lift list", report_path, rc, "Nov 30, 2016-A (N280NR, SN8146)",
        None, src_r)
    add("Breakline elevations assigned to lakes", report_path, rc,
        "Elevation values were assigned to all Inland Ponds and Lakes", "proprietary software.", src_r)
    add("Hydro-flattened DEM processing", report_path, rc,
        "Class 2 (Ground) lidar points in conjunction with the hydro breaklines", "for the QL2 AOI.", src_r)
    src_tm = {"source": "Archuleta et al. 2017, The National Map seamless digital elevation model specifications, "
                        "USGS Techniques and Methods 11-B9, doi:10.3133/tm11B9"}
    tm = RAW / "docs" / "tm11B9.pdf"
    tc = pdf_chunks(tm)
    add("Seamless DEMs are hydro-flattened", tm, tc, "The seamless DEMs distributed by the USGS are",
        "any other surface objects.", src_tm)
    add("Bilinear interpolation from source DEMs", tm, tc, "Elevation values at each arc-second increment",
        "State Plane coordinate space.", src_tm)
    add("Source priority table", tm, tc, "Priority Data sources 1 High-resolution data derived from lidar",
        "1:63,360-scale cartographic contours.", src_tm)
    add("Surface treatment", tm, tc, "The seamless DEM product shall be a hydroflattened",
        "improved cartographic appearance only.", src_tm)
    add("Data source and tile dates", tm, tc, "The data source date for the seamless DEM product",
        "incorporated into the seamless DEM dataset.", src_tm)
    add("Legacy contour-generated source being replaced", tm, tc, "Under 3DEP, the legacy con",
        "lidar-equivalent accuracy in CONUS", src_tm)
    add("Example spatial metadata with LT4X", tm, tc, "Production Device: Windows version of LT4X",
        "Data Source Date: 1948", src_tm)
    src_gl = {"source": "3D Elevation Program (3DEP) Data Dictionary (3DEP Spatial Metadata Glossary), USGS, "
                        "last updated August 24, 2018"}
    gl = RAW / "docs" / "3DEP_Spatial_Metadata_Glossary_0.pdf"
    gc = pdf_chunks(gl)
    add("PMETHOD definition", gl, gc, "PMETHOD Production Method (short integer)", "Digital Elevation Models (USGS, 1993).",
        src_gl)
    add("PMETHOD codes 5 = LT4X, 7 = Lidar", gl, gc, "5 LT4X 6 Complex polynomial interpolation", "7 Lidar", src_gl)
    add("PDEVICE LT4X", gl, gc, "LT4X Either LT4X or LTPlus software", None, src_gl)
    add("S_DATE for map-derived DEMs", gl, gc, "For DEMs derived from standard USGS paper map series, this field is "
        "data element 21", "from which the DEM was compiled", src_gl)
    src_w = {"source": "WESM Data Dictionary - general attributes, USGS (web page)"}
    wh = RAW / "docs" / "wesm_dictionary_general_attributes.html"
    whc = [("web page text", html_text(wh))] if wh.exists() else []
    add("ql definition", wh, whc, "ql Description : Quality level", "QL 5 Other", src_w)
    add("collect_start definition", wh, whc, "collect_start Description : Collection start date",
        "Date data collection for the work unit was started.", src_w)
    src_ee = {"source": "Earth Engine STAC record USGS/3DEP/10m_collection (static JSON)"}
    ej = RAW / "docs" / "ee_stac_USGS_3DEP_10m_collection.json"
    if ej.exists() and stac:
        add("Collection uploaded by a third party", ej, [("description", norm(stac.get("description", "")))],
            "Dataset uploaded by", "(https://fbn.com).", src_ee)
    S["verbatim_quotes"] = quotes

    # 10. Request log ---------------------------------------------------------------------------
    S["requests"] = [{k: m.get(k) for k in ("file", "url", "retrieved_utc", "http_status", "bytes", "sha256", "ok",
                                            "error", "reused_from_disk")} | {"n_attempts": len(m.get("attempts", []))}
                     for m in f.log]
    S["failures"] = [{"url": m["url"], "http_status": m["http_status"], "error": m["error"],
                      "attempts": m.get("attempts")} for m in f.log if not m["ok"]]
    S["retried_then_succeeded"] = [{"url": m["url"], "failed_attempts": [a for a in m["attempts"] if a["error"]]}
                                   for m in f.log if m["ok"] and len(m.get("attempts", [])) > 1]
    S["download_bytes_total"] = int(sum(m["bytes"] or 0 for m in f.log))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(S, indent=1, default=str), encoding="utf-8")
    print("wrote", rel(OUT), "| requests", len(f.log), "| failures", len(S["failures"]),
          "| bytes", S["download_bytes_total"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
