"""Exposure layers from the WDPA: protected areas and Ramsar
sites as "area" layers of the H3 exposure framework (scripts/a8_emergence.py,
exposure_table), computed on the H3 runs' own cached surfaces. H3 is not rerun, and
nothing under results/emergence/<tag>/ is written.

Data: UNEP-WCMC and IUCN (2026), Protected Planet: The World Database on Protected Areas
(WDPA) and World Database on Other Effective Area-based Conservation Measures (WD-OECM)
[Online], October 2026, Cambridge, UK: UNEP-WCMC and IUCN. Available at:
www.protectedplanet.net. Used under its terms (non-commercial research): the
polygons and attributes stay in data/ (git-ignored) and are never redistributed
(docs/DATA_RELEASE.md); only derived counts and areas are committed.

Commands
  fetch     the public country downloads of the release (Kazakhstan, Russia: the only
            countries the domain reaches) from the Protected Planet file store, unless
            present; checks their size against the server's; extracts the shapefile
            parts and the English documentation (member names checked, no paths kept)
            into data/raw/wdpa/<ISO3>/; provenance (URL, HTTP headers, size, sha256,
            release month, terms page sha256) -> results/provenance/wdpa.json.
  layers    every polygon record of both countries (all shapefile parts), selected by
            select() below, made valid, kept when it intersects the domain (whole, as
            a8_emergence.polygon_layers keeps polygons); per layer the union of the
            kept polygons (overlapping designations are not counted twice), split
            into its connected parts, which are the layer's features (as the habitat
            rasters' 8-connected patches). Writes data/interim/wdpa/layers.gpkg, sites.csv
            and points_in_domain.csv (git-ignored) and the counts-only
            results/emergence/wdpa/layers.json.
  exposure  per H3 run (--tags; default main, the registered linear_hold form, and
            main_constant, the constant-form sensitivity): the uncorrected flood-
            level maps from the hindcast cache and the fused ones from the H3 cache
            (stops if one is missing: no map is built here), and the interval bounds
            (equivalent levels) from the run's emergence.csv. Two checks stop the run on
            any difference: (1) the emerged area of every product, surface, region and
            scenario recomputed from the cached maps equals emergence.csv; (2) the OSM
            wetland layer recomputed by this path equals the run's exposure.csv rows.
            Then a8_emergence.exposure_table on the WDPA layers ->
            results/emergence/wdpa/exposure_<tag>.csv and run.json.

            Sensitivity: --layers wdpa_protected_areas_noksrz --suffix _noksrz
            (the protected areas without the Kazakh northern-Caspian reserved zone) ->
            exposure_<tag>_noksrz.csv and run_noksrz.json.

Usage: python scripts/a8_wdpa_exposure.py fetch | layers | exposure [--tags main,main_constant]
           [--layers wdpa_protected_areas,wdpa_ramsar] [--suffix ""]
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.util
import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import caspian  # noqa: F401
import numpy as np
import pandas as pd

from caspian import exposure as ex
from caspian.config import CRS_EQUAL_AREA, INTERIM, RAW, RESULTS
from caspian.geography import DOMAIN, SECTOR_ORDER

ROOT = Path(__file__).resolve().parents[1]


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


RELEASE = "Oct2026"
RELEASE_TEXT = "October 2026"
COUNTRIES = ("KAZ", "RUS")              # the domain lies in Kazakhstan and Russia only
STORE = "https://d1gam3xoknrgr2.cloudfront.net/current"
TERMS_URL = "https://www.protectedplanet.net/en/legal"
RAWDIR = RAW / "wdpa"
WORK = INTERIM / "wdpa"
OUT = RESULTS / "emergence" / "wdpa"
PROV = RESULTS / "provenance" / "wdpa.json"
CITATION = ("UNEP-WCMC and IUCN (2026), Protected Planet: The World Database on Protected Areas (WDPA) and World "
            "Database on Other Effective Area-based Conservation Measures (WD-OECM) [Online], October 2026, "
            "Cambridge, UK: UNEP-WCMC and IUCN. Available at: www.protectedplanet.net.")
# Record selection. Status: records "Proposed" or "Not Reported" are left
# out (the WDPA manual 1.6, section on the STATUS field, describes this choice; "Established"
# sites are kept, as it advises for sites that meet the protected-area definition). Only
# protected areas (SITE_TYPE "PA"); OECMs are counted, not used (none in this release for
# KAZ and RUS). UNESCO-MAB Biosphere Reserve records are protected areas in the WDPA and are
# kept; they are counted separately. Ramsar sites: DESIG_ENG naming the Ramsar Convention
# ("Wetland of International Importance (Ramsar Site)" in this release). Point records have
# no boundary: counted (Ramsar points separately), not used.
STATUS_OUT = ("Proposed", "Not Reported")
MAB = "UNESCO-MAB Biosphere Reserve"
RAMSAR_KEY = "Ramsar"
LAYERS = {"wdpa_protected_areas": "protected areas (WDPA, every designated protected-area polygon; dissolved)",
          "wdpa_ramsar": "Ramsar sites (WDPA international designation; polygons; dissolved)",
          "wdpa_protected_areas_noksrz": "protected areas without the Kazakh State Reserved Zone in the northern "
                                         "Caspian (sensitivity, D-103 addendum; dissolved)"}
# Sensitivity: the Kazakh "State Reserved Zone in Northern part of Caspian Sea"
# is drawn as a 63,077 km2 polygon against a reported area of 6,626.3 km2 and dominates the
# protected-area layer; the sensitivity layer leaves out this one record. layers stops unless
# exactly one record in the domain has this id and its name names the reserved zone.
KSRZ_SITE_ID = 342651
KSRZ_NAME_KEY = "Reserved Zone"
# the exposure command's default layers: the two registered tables
PRIMARY_LAYERS = ("wdpa_protected_areas", "wdpa_ramsar")
DOC_FILES = ("Resources_in_English/WDPA_WDOECM_Manual_1_6.pdf", "Resources_in_English/WDPA_WDOECM_Metadata_1_6.pdf",
             "Resources_in_English/Summary_table_WDPA_WDOECM_attributes.pdf", "Shapefile_splitting_README.txt",
             f"WDPA_sources_{RELEASE}.csv")
SHP_EXT = (".shp", ".shx", ".dbf", ".prj", ".cpg")


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _json(obj) -> str:
    def conv(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        return str(o)
    return json.dumps(obj, indent=1, default=conv, ensure_ascii=False)


# --- fetch -----------------------------------------------------------------------------------

def country_url(iso3: str) -> str:
    return f"{STORE}/WDPA_WDOECM_{RELEASE}_Public_{iso3}_shp.zip"


def head(url: str) -> dict:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return {k: v for k, v in r.headers.items() if k in ("Content-Length", "Last-Modified", "ETag", "Date")}


def safe_extract(zf: zipfile.ZipFile, member: str, dest: Path) -> Path:
    """One member to dest/<basename>; refuses names with drive letters or parent steps."""
    name = Path(member).name
    if not name or ".." in Path(member).parts or ":" in member:
        raise ValueError(f"unsafe member name {member!r}")
    out = dest / name
    with zf.open(member) as src, open(out, "wb") as dst:
        dst.write(src.read())
    return out


def cmd_fetch(a) -> int:
    zdir = RAWDIR / "zips"
    zdir.mkdir(parents=True, exist_ok=True)
    rec = {"dataset": "WDPA and WD-OECM (Protected Planet), public country downloads", "release": RELEASE_TEXT,
           "citation": CITATION, "terms": TERMS_URL,
           "terms_conditions": ["non-commercial use only (D-101: academic research)",
                                "no redistribution of the data, in whole or in part, as downloadable files: the "
                                "polygons and attributes stay in data/ (git-ignored); only derived counts and areas",
                                "published material: the citation with the release month and year, and a link to "
                                "www.protectedplanet.net",
                                "UNEP-WCMC asks for two free copies of published materials and recommends review "
                                "before publication (the PI's action; contacting anyone is the PI's step)"],
           "files": {}, "retrieved_by": "scripts/a8_wdpa_exposure.py fetch"}
    terms = sorted(RAWDIR.glob("protectedplanet_legal_*.html"))
    if terms:
        rec["terms_page"] = {"file": terms[-1].name, "sha256": sha_file(terms[-1]),
                             "note": "text identical to the copy of 2026-10-21 01:45 (data/raw/terms) except the "
                                     "page's CSRF token and e-mail obfuscation"}
    for iso3 in COUNTRIES:
        url = country_url(iso3)
        z = zdir / Path(url).name
        hd = head(url)
        if not z.exists() or z.stat().st_size != int(hd["Content-Length"]):
            log("downloading", url)
            urllib.request.urlretrieve(url, z)
        if z.stat().st_size != int(hd["Content-Length"]):
            raise SystemExit(f"{z.name}: {z.stat().st_size} bytes, the server says {hd['Content-Length']}")
        dest = RAWDIR / iso3
        dest.mkdir(parents=True, exist_ok=True)
        parts = []
        with zipfile.ZipFile(z) as zf:
            names = zf.namelist()
            for n in names:
                if n.endswith(".zip"):
                    pdir = dest / Path(n).stem
                    pdir.mkdir(exist_ok=True)
                    with zipfile.ZipFile(zf.open(n)) as z2:
                        for m in z2.namelist():
                            if m.lower().endswith(SHP_EXT):
                                safe_extract(z2, m, pdir)
                    parts.append(pdir.name)
                elif n in DOC_FILES:
                    (dest / "docs").mkdir(exist_ok=True)
                    safe_extract(zf, n, dest / "docs")
        rec["files"][iso3] = {"url": url, "file": f"data/raw/wdpa/zips/{z.name}", "bytes": z.stat().st_size,
                              "sha256": sha_file(z), "http": hd, "shapefile_parts": parts,
                              "members": len(names)}
        log(iso3, rec["files"][iso3]["sha256"])
    PROV.write_text(_json(rec), encoding="utf-8")
    log("wrote", PROV)
    return 0


# --- layers ------------------------------------------------------------------------------------

def read_country(iso3: str, kind: str):
    import geopandas as gpd
    frames = []
    for shp in sorted((RAWDIR / iso3).glob(f"*/*_shp-{kind}.shp")):
        g = gpd.read_file(shp)
        frames.append(g.assign(_part=shp.parent.name))
    if not frames:
        raise SystemExit(f"no {kind} shapefiles under {RAWDIR / iso3}: run fetch")
    return pd.concat(frames, ignore_index=True)


def _col(g, *names):
    for n in names:
        if n in g.columns:
            return n
    raise KeyError(f"none of {names} in {list(g.columns)}")


def select(g) -> dict:
    """Boolean masks of the record-selection rules over a WDPA polygon table."""
    status, desig = g[_col(g, "STATUS")].astype(str), g[_col(g, "DESIG_ENG")].astype(str)
    site_type = g[_col(g, "SITE_TYPE", "PA_DEF")].astype(str)
    is_pa = site_type.isin(["PA", "1", "Protected Area", "protected area"])
    designated = ~status.isin(STATUS_OUT)
    ramsar = desig.str.contains(RAMSAR_KEY, case=False, regex=False)
    idc = next((c for c in ("SITE_ID", "WDPAID") if c in g.columns), None)
    ksrz = (pd.to_numeric(g[idc], errors="coerce") == KSRZ_SITE_ID) if idc else pd.Series(False, index=g.index)
    return {"wdpa_protected_areas": is_pa & designated, "wdpa_ramsar": is_pa & designated & ramsar,
            "wdpa_protected_areas_noksrz": is_pa & designated & ~ksrz,
            "_oecm": ~is_pa, "_mab": desig == MAB, "_not_designated": ~designated, "_ramsar_any": ramsar,
            "_ksrz": ksrz}


def dissolve(geoms) -> list:
    """The union of polygon geometries (made valid) as its connected polygon parts: a flat layer
    with no overlaps, so that an area covered by several designations counts once (WDPA
    manual 1.6, on overlapping records). Empty input gives no parts."""
    import shapely
    geoms = [g for g in geoms if g is not None and not g.is_empty]
    if not geoms:
        return []
    u = shapely.union_all(shapely.make_valid(np.asarray(geoms, dtype=object)))
    return [q for q in shapely.get_parts(u) if q.geom_type == "Polygon" and q.area > 0]


def cmd_layers(a) -> int:
    import geopandas as gpd
    a8 = _script("a8_emergence")
    dom = gpd.GeoSeries([DOMAIN], crs="EPSG:4326").to_crs(CRS_EQUAL_AREA).iloc[0]
    polys, pts = [], []
    for iso3 in COUNTRIES:
        polys.append(read_country(iso3, "polygons").assign(_iso3=iso3))
        pts.append(read_country(iso3, "points").assign(_iso3=iso3))
    g = gpd.GeoDataFrame(pd.concat(polys, ignore_index=True), crs=polys[0].crs).to_crs(CRS_EQUAL_AREA)
    p = gpd.GeoDataFrame(pd.concat(pts, ignore_index=True), crs=pts[0].crs).to_crs(CRS_EQUAL_AREA)
    g = a8._polygons_only(g)
    inside = g[g.intersects(dom)].copy()
    p_in = p[p.intersects(dom)]
    sel = select(inside)
    k = inside.loc[sel["_ksrz"].to_numpy()]
    if len(k) != 1 or KSRZ_NAME_KEY.lower() not in str(k[_col(k, "NAME_ENG", "NAME")].iloc[0]).lower():
        raise SystemExit(f"expected one record {KSRZ_SITE_ID} naming the {KSRZ_NAME_KEY!r} in the domain, found {len(k)}")
    idc = _col(inside, "SITE_ID", "WDPAID")
    keep_cols = [c for c in (idc, "SITE_PID", "WDPA_PID", "NAME_ENG", "NAME", "DESIG_ENG", "DESIG_TYPE", "SITE_TYPE",
                             "PA_DEF", "IUCN_CAT", "STATUS", "STATUS_YR", "REALM", "MARINE", "REP_AREA", "GIS_AREA",
                             "ISO3", "_iso3", "_part") if c in inside.columns]
    sites = inside[keep_cols].copy()
    for k, m in sel.items():
        sites[k.lstrip("_")] = m.to_numpy()
    sites["area_km2_laea"] = inside.geometry.area.to_numpy() / 1e6
    sites["area_in_domain_km2"] = inside.geometry.intersection(dom).area.to_numpy() / 1e6
    WORK.mkdir(parents=True, exist_ok=True)
    sites.to_csv(WORK / "sites.csv", index=False)
    p_in.drop(columns="geometry").to_csv(WORK / "points_in_domain.csv", index=False)
    out = {"release": RELEASE_TEXT, "citation": CITATION, "countries": list(COUNTRIES), "crs": CRS_EQUAL_AREA,
           "rules": {"status_excluded": list(STATUS_OUT), "site_type": "PA (OECMs counted, not used)",
                     "mab": "UNESCO-MAB Biosphere Reserve records kept in the protected-area layer, counted",
                     "ramsar": "DESIG_ENG contains 'Ramsar'", "points": "counted, not used",
                     "dissolve": "union per layer, split into connected parts (the features)",
                     "domain": "whole polygons that intersect the domain (as a8_emergence.polygon_layers)"},
           "n_polygon_records_countries": int(len(g)), "n_point_records_in_domain": int(len(p_in)),
           "n_ramsar_point_records_in_domain": int(p_in[_col(p_in, "DESIG_ENG")].astype(str)
                                                   .str.contains(RAMSAR_KEY, case=False, regex=False).sum()),
           "n_ramsar_point_records_countries": int(p[_col(p, "DESIG_ENG")].astype(str)
                                                   .str.contains(RAMSAR_KEY, case=False, regex=False).sum()),
           "n_ramsar_polygon_records_countries": int(g[_col(g, "DESIG_ENG")].astype(str)
                                                     .str.contains(RAMSAR_KEY, case=False, regex=False).sum()),
           "n_polygon_records_in_domain": int(len(inside)),
           "n_sites_in_domain": int(inside[idc].nunique()),
           "n_oecm_records_in_domain": int(sel["_oecm"].sum()), "n_mab_records_in_domain": int(sel["_mab"].sum()),
           "n_not_designated_records_in_domain": int(sel["_not_designated"].sum()),
           "by_designation_type": {str(k): int(v) for k, v in
                                   inside[_col(inside, "DESIG_TYPE")].astype(str).value_counts().items()},
           "layers": {}}
    frames = []
    for name, what in LAYERS.items():
        m = sel[name].to_numpy()
        parts = dissolve(inside.geometry[m].to_numpy())
        frames.append(gpd.GeoDataFrame({"layer": name, "part": np.arange(len(parts))}, geometry=parts,
                                       crs=CRS_EQUAL_AREA))
        out["layers"][name] = {
            "what": what, "n_records": int(m.sum()), "n_sites": int(inside.loc[m, idc].nunique()),
            "n_features": len(parts),
            "union_area_km2": float(sum(q.area for q in parts) / 1e6),
            "union_area_in_domain_km2": float(sum(q.intersection(dom).area for q in parts) / 1e6),
            "sum_record_area_km2": float(inside.geometry[m].area.sum() / 1e6)}
        log(name, out["layers"][name])
    gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=CRS_EQUAL_AREA).to_file(WORK / "layers.gpkg",
                                                                                       driver="GPKG")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "layers.json").write_text(_json(out), encoding="utf-8")
    log("wrote", OUT / "layers.json")
    return 0


# --- exposure ----------------------------------------------------------------------------------

def bounds_of(em: pd.DataFrame) -> dict:
    """{(product, surface, region, scenario): {level_lo_bound, lo_beyond_grid, ...}} from
    a run's emergence.csv ("emergence" functional), as a8_emergence.run passes them."""
    e = em[em["functional"] == "emergence"]
    out = {}
    for r in e.itertuples():
        out[(r.product, r.surface, r.region, float(r.scenario_m))] = {
            "level_lo_bound": float(r.level_lo_bound), "lo_beyond_grid": bool(r.lo_beyond_grid),
            "level_hi_bound": float(r.level_hi_bound), "hi_beyond_grid": bool(r.hi_beyond_grid)}
    return out


def fused_cache(form: str, p: str) -> Path:
    hits = sorted(glob.glob(str(INTERIM / "emergence" / "main" / "final" / f"{p}_{form}_fused_*.npy")))
    if len(hits) != 1:
        raise SystemExit(f"{p} {form}: expected one cached fused flood-level map, found {len(hits)}")
    return Path(hits[0])


def check_emergence(surfaces, em, lab_in, level_2026, deltas, lv_full, cell: float) -> dict:
    """Check (1): emerged area per product, surface, region and scenario from the cached maps
    against emergence.csv (the "emergence" functional, the surface's own 2026 baseline)."""
    K = len(SECTOR_ORDER)
    regions = list(SECTOR_ORDER) + ["domain"]
    k26 = int(np.flatnonzero(np.isclose(lv_full, level_2026, atol=1e-6))[0])
    ks = [k26] + [int(np.flatnonzero(np.isclose(lv_full, round(level_2026 - D, 6), atol=1e-6))[0]) for D in deltas]
    order = np.argsort(ks)
    lv = lv_full[np.array(ks)[order]]
    e = em[em["functional"] == "emergence"].set_index(["product", "surface", "region", "scenario_m"])["estimate_km2"]
    worst, n = 0.0, 0
    for p, sf in surfaces.items():
        for s in ("uncorrected", "fused"):
            cells = ex.sea_cells_by_region(sf[s], lab_in, K, lv)
            cells = cells[:, np.argsort(order)]            # back to [L26, L26 - D...]
            for k, reg in enumerate(regions):
                a = cells.sum(axis=0) if reg == "domain" else cells[k]
                for j, D in enumerate(deltas):
                    got = float(a[0] - a[j + 1]) * cell
                    want = float(e.loc[(p, s, reg, float(D))])
                    worst = max(worst, abs(got - want))
                    n += 1
            log("check 1:", p, s, "max |diff| so far", worst)
    if worst > 1e-6:
        raise SystemExit(f"check 1 failed: emerged area differs from emergence.csv by up to {worst} km2")
    return {"n_values": n, "max_abs_diff_km2": worst}


def compare_exposure(mine: pd.DataFrame, ref: pd.DataFrame) -> dict:
    """Check (2): the recomputed OSM wetland rows against the run's exposure.csv."""
    keys = ["layer", "product", "surface", "region", "scenario_m"]
    num = ["n_features", "n_considered", "amount", "amount_lo", "amount_hi", "n_changed", "n_changed_lo",
           "n_changed_hi", "level_lo_bound", "level_hi_bound"]
    r = ref[ref["layer"].isin(mine["layer"].unique())]
    m = mine.merge(r, on=keys, suffixes=("", "_ref"), how="outer", indicator=True)
    if (m["_merge"] != "both").any():
        raise SystemExit(f"check 2 failed: {int((m['_merge'] != 'both').sum())} rows do not match up")
    worst = 0.0
    for c in num:
        a, b = m[c].to_numpy(float), m[f"{c}_ref"].to_numpy(float)
        both_nan = np.isnan(a) & np.isnan(b)
        d = np.where(both_nan, 0.0, np.abs(a - b))
        if np.isnan(d).any():
            raise SystemExit(f"check 2 failed: {c} is NaN on one side only")
        worst = max(worst, float(d.max()) if d.size else 0.0)
    if worst > 1e-9:
        raise SystemExit(f"check 2 failed: OSM wetland exposure differs by up to {worst}")
    return {"n_rows": int(len(m)), "max_abs_diff": worst}


def cmd_exposure(a) -> int:
    import geopandas as gpd
    a8 = _script("a8_emergence")
    f1 = _script("f1_erl_display")
    hrun = json.loads((RESULTS / "hindcast" / "main" / "run.json").read_text(encoding="utf-8"))
    transform, shape = f1.caspian_grid(hrun)
    static = f1.static_main("main", transform, shape, a.tile)
    lab_in = np.where(np.asarray(static["domain"], bool), np.asarray(static["sector"]), 0).astype(np.uint8)
    names = list(hrun["products"])
    F_unc, _ = f1.flood_maps(names, hrun, {"sea_seed": static["sea_seed"]}, transform, shape, a.tile)
    del static
    lay = gpd.read_file(WORK / "layers.gpkg")
    if lay.crs is None or not lay.crs.equals(CRS_EQUAL_AREA):
        lay = lay.to_crs(CRS_EQUAL_AREA)
    names_l = [n for n in a.layers.split(",") if n]
    if not set(names_l) <= set(LAYERS):
        raise SystemExit(f"unknown layers: {sorted(set(names_l) - set(LAYERS))}")
    layers = {n: {"geoms": list(lay.loc[lay["layer"] == n].geometry), "rule": "area", "threshold_m": None,
                  "file": "data/interim/wdpa/layers.gpkg (from the WDPA October 2026 release)"} for n in names_l}
    if any(not v["geoms"] for v in layers.values()):
        raise SystemExit("a requested layer has no features in layers.gpkg: run layers")
    osm = {k: v for k, v in a8.osm_layers(RAW / "osm").items() if k == "osm_wetlands"}
    rec = {"script": "scripts/a8_wdpa_exposure.py exposure", "citation": CITATION, "release": RELEASE_TEXT,
           "layers_json": "results/emergence/wdpa/layers.json", "runs": {}}
    OUT.mkdir(parents=True, exist_ok=True)
    for tag in a.tags.split(","):
        run = json.loads((RESULTS / "emergence" / tag / "run.json").read_text(encoding="utf-8"))
        form = run["forms"][0]
        if sorted(run.get("products") or []) != sorted(names):
            raise SystemExit(f"{tag}: products differ from the hindcast run's")
        level_2026 = round(float(run["level_2026"]["level_m"]), 4)
        deltas = [float(d) for d in run["scenarios_m"]]
        em = pd.read_csv(RESULTS / "emergence" / tag / "emergence.csv")
        surfaces = {p: {"uncorrected": F_unc[p], "fused": np.load(fused_cache(form, p), mmap_mode="r")}
                    for p in names}
        lv_full = ex.level_grid(level_2026, max(deltas) + a8.LEVEL_MARGIN_M)
        c1 = check_emergence(surfaces, em, lab_in, level_2026, deltas, lv_full, a8.CELL_KM2)
        bounds = bounds_of(em)
        log(tag, "check 1 passed", c1)
        mine = a8.exposure_table(osm, surfaces, bounds, lab_in, SECTOR_ORDER, transform, shape, level_2026, deltas)
        c2 = compare_exposure(mine, pd.read_csv(RESULTS / "emergence" / tag / "exposure.csv"))
        log(tag, "check 2 passed", c2)
        t0 = time.time()
        x = a8.exposure_table(layers, surfaces, bounds, lab_in, SECTOR_ORDER, transform, shape, level_2026, deltas)
        x.insert(0, "form", form)
        x.to_csv(OUT / f"exposure_{tag}{a.suffix}.csv", index=False)
        rec["runs"][tag] = {"form": form, "emergence_csv": f"results/emergence/{tag}/emergence.csv",
                            "fused_maps": [fused_cache(form, p).name for p in names],
                            "level_2026_m": level_2026, "scenarios_m": deltas,
                            "check_emergence": c1, "check_osm_wetlands": c2,
                            "n_features": {n: len(v["geoms"]) for n, v in layers.items()},
                            "seconds": round(time.time() - t0, 1),
                            "table": f"results/emergence/wdpa/exposure_{tag}{a.suffix}.csv"}
        log(tag, "wrote", OUT / f"exposure_{tag}{a.suffix}.csv")
        del surfaces
    rec["layers"] = names_l
    (OUT / f"run{a.suffix}.json").write_text(_json(rec), encoding="utf-8")
    return 0


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["fetch", "layers", "exposure"])
    ap.add_argument("--tags", default="main,main_constant")
    ap.add_argument("--layers", default=",".join(PRIMARY_LAYERS), help="exposure: layers of layers.gpkg to compute")
    ap.add_argument("--suffix", default="", help="exposure: suffix of the output tables and run record")
    ap.add_argument("--tile", type=int, default=2048)
    a = ap.parse_args(argv)
    return {"fetch": cmd_fetch, "layers": cmd_layers, "exposure": cmd_exposure}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
