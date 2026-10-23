"""Build the Zenodo-ready data package of the derived tables: release/zenodo/.

    python tools/build_data_package.py [--out release/zenodo]

The package holds the small derived tables of `results/` that docs/DATA_RELEASE.md allows
to be released under CC BY 4.0, grouped into zip archives (Zenodo records
are flat and hold at most 100 files), plus:

- README.md: contents, provenance, datums, input licences, exclusions, how to cite;
- LICENSE: the CC BY 4.0 legal code, verbatim (tools/data_package/, sha256 checked);
- metadata.json: Zenodo deposit metadata (creators and ORCID iDs from CITATION.cff);
- metrics.json: results/metrics.json without the keys that may not be released, and
  metrics_excluded_keys.csv naming those keys (names only, no values);
- MANIFEST.csv: every archive member with its source path, size and SHA-256;
- SHA256SUMS: every other file of the package.

Never released (DATA_RELEASE.md): CASPCOM per-gauge values, and anything
derived from the WDPA, whose terms (non-commercial use, no redistribution as files) are
incompatible with CC BY 4.0. Both are excluded by explicit rules and by a content scan
of every candidate file; the build lists what it excluded and why. Large rasters stay in
data/ and are listed by name with the command that regenerates them.
"""
from __future__ import annotations

import argparse
import csv
import glob
import gzip
import hashlib
import io
import json
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RELEASE_DIR = ROOT / "release"
DEFAULT_OUT = RELEASE_DIR / "zenodo"
REPO_URL = "https://github.com/rezw4n/north-caspian-emergence"

LICENCE_SRC = ROOT / "tools" / "data_package" / "CC-BY-4.0-legalcode.txt"
LICENCE_URL = "https://creativecommons.org/licenses/by/4.0/legalcode.txt"
LICENCE_RETRIEVED = "2026-10-22"
LICENCE_SHA256 = "9ba9550ad48438d0836ddab3da480b3b69ffa0aac7b7878b5a0039e7ab429411"  # of the text with LF line ends

ZIP_DATE = (1980, 1, 1, 0, 0, 0)  # fixed member timestamps: the archives are byte-reproducible

# (archive, title, description, glob patterns relative to the repository root)
ARCHIVES = [
    ("study_area.zip", "Study area, sectors and spatial blocks",
     "The study domain and its seven frozen sectors (WGS84 GeoJSON), their areas and edges, and the "
     "12.5, 25 and 50 km spatial blocks of the training, calibration and test splits.",
     ["results/study_area/sectors.geojson", "results/study_area/sector_areas.json",
      "results/study_area/sector_edges.json", "results/study_area/blocks_*.csv"]),
    ("levels_and_waterlines.zip", "Scene levels and waterline summaries",
     "Summary of the altimetry level assigned to every scene (noise floor, epochs, level "
     "range), the static-contour rule applied to the level-binned waterlines, the vegetated "
     "share of the emerged band and the vegetated-sector labels.",
     ["results/levels/scene_levels_summary.json", "results/levels/static_contour_rule.json",
      "results/levels/vegetation_fraction.json", "results/levels/sector_labels.csv"]),
    ("hindcast_caspian.zip", "North Caspian hindcast (registered audit)",
     "Level-binned observed extents by epoch, bin and sector (observed.csv), the waterline "
     "noise floor (noise_floor.csv), and for each of the five bathymetry grids the predicted "
     "extents and their errors (extent.csv), waterline displacements (displacement.csv), "
     "block statistics and the registered H1 decision.",
     ["results/hindcast/main/*"]),
    ("hindcast_caspian_revision.zip", "North Caspian hindcast (revision analyses)",
     "Tables of the analyses added after the internal review: validated-sector counts, "
     "displacement over the noise floor, area bias in and out of sample, seabed-change tests "
     "and the interleaved-halves floor.",
     ["results/revision/s/*.csv", "results/revision/s/*.json",
      "results/revision/s/rasters/*.csv", "results/revision/s/rasters/*.json"]),
    ("hindcast_great_salt_lake.zip", "Great Salt Lake hindcast (generality test)",
     "The same audit for the north and south arms of Great Salt Lake, with the lidar-based "
     "control surface: observed extents, noise floor, displacements and decisions.",
     ["results/generality/great_salt_lake/h1_decision.json",
      "results/generality/great_salt_lake/h1_mispredicts*.csv",
      "results/generality/great_salt_lake/north/hindcast/*",
      "results/generality/great_salt_lake/north/hindcast_dep3/*",
      "results/generality/great_salt_lake/south/hindcast/*",
      "results/generality/great_salt_lake/south/hindcast_dep3/*"]),
    ("emergence.zip", "Near-term emergence and exposure (registered)",
     "Emergence 0.5-5 m below the 2026 level for each grid, uncorrected and corrected, with "
     "conformal intervals; exposure of infrastructure and wetland layers; the dates implied "
     "by published projection pathways; the H3 decision. main_constant is the registered "
     "constant-offset sensitivity.",
     ["results/emergence/main/*", "results/emergence/main_constant/*"]),
    ("emergence_revision.zip", "Near-term emergence (revision analyses)",
     "Emergence curves on the common observed baseline and on shifted grids, the "
     "already-dry seabed, hindcast error envelopes, pathway decomposition and the "
     "vertical-error simulation.",
     ["results/revision/n/*"]),
]
TOP_LEVEL = ["results/params.json"]   # copied as they are (after the content scan)
METRICS = "results/metrics.json"      # filtered, then written at the top level

# Explicitly never released (docs/DATA_RELEASE.md).
EXPLICIT_EXCLUDE = {
    "results/levels/dahiti_bhs_annual_diffs.csv": "CASPCOM per-gauge annual means (no licence; DATA_RELEASE.md)",
    "results/levels/dahiti_bhs_offset.json": "holds per-gauge CASPCOM values (per_post, all_baltic_posts); "
                                            "DATA_RELEASE.md item 1",
}
SKIP_SUFFIXES = (".log",)             # run logs are not data

WDPA_RE = re.compile(r"wdpa|ramsar|protected\s*planet|protectedplanet", re.I)
CASPCOM_RE = re.compile(r"caspcom", re.I)
REASON_WDPA = "derived from the WDPA (terms: non-commercial use, no redistribution as files; not CC BY 4.0)"
REASON_GAUGE = "CASPCOM per-gauge value (no licence; DATA_RELEASE.md)"


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def caspcom_posts() -> list[str]:
    """CASPCOM post codes, read from the repository's own records (never typed here)."""
    codes: set[str] = set()
    off = ROOT / "results" / "levels" / "dahiti_bhs_offset.json"
    if off.exists():
        d = json.loads(off.read_text(encoding="utf-8"))
        codes |= set(d.get("post_meta", {}))
        codes |= set(d.get("settings", {}).get("posts", []))
    posts = ROOT / "results" / "revision" / "v" / "caspcom_posts.csv"
    if posts.exists():
        with posts.open(encoding="utf-8") as f:
            codes |= {r["post"] for r in csv.DictReader(f) if r.get("post")}
    return sorted(c for c in codes if c)


def post_regex(codes: list[str]) -> re.Pattern | None:
    words = sorted({c for c in codes if re.fullmatch(r"[A-Z_]{3,}", c)}, key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(map(re.escape, words)) + r")\b") if words else None


def file_text(p: Path) -> str:
    raw = p.read_bytes()
    if p.suffix == ".gz":
        raw = gzip.decompress(raw)
    return raw.decode("utf-8", errors="replace")


def content_exclusion(rel: str, text: str, post_re: re.Pattern | None) -> str | None:
    """Why a candidate file may not be released, or None."""
    if rel in EXPLICIT_EXCLUDE:
        return EXPLICIT_EXCLUDE[rel]
    if WDPA_RE.search(text):
        return REASON_WDPA
    if CASPCOM_RE.search(text) or (post_re is not None and post_re.search(text)):
        return REASON_GAUGE
    return None


def metric_exclusion(key: str, entry, post_re: re.Pattern | None) -> str | None:
    """Why a metrics.json key may not be released, or None."""
    # Basin-wide summaries over the posts (delta, its sigma, the between-post SD) are released;
    # a value that belongs to one post is not. The build stops if a kept key still
    # names a post code or the WDPA (build(), "leftover").
    src = str(entry.get("source", "")) if isinstance(entry, dict) else ""
    if WDPA_RE.search(key) or WDPA_RE.search(src):
        return REASON_WDPA
    if key.startswith("rev.v.posts.") or "caspcom_posts" in src:
        return REASON_GAUGE
    if re.fullmatch(r"datum\.bhs\.post_(?:min|max)_m", key):      # one post's offset each
        return REASON_GAUGE
    if post_re is not None and post_re.search(key.upper()):
        return REASON_GAUGE
    return None


def filter_metrics(metrics: dict, post_re: re.Pattern | None) -> tuple[dict, list[tuple[str, str]]]:
    kept, dropped = {}, []
    for k, v in metrics.items():
        why = metric_exclusion(k, v, post_re)
        if why:
            dropped.append((k, why))
        else:
            kept[k] = v
    return kept, dropped


def collect(patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    for pat in patterns:
        for s in sorted(glob.glob(str(ROOT / pat))):
            p = Path(s)
            if p.is_file() and not p.name.endswith(SKIP_SUFFIXES) and p not in out:
                out.append(p)
    return out


def rel(p: Path) -> str:
    return p.resolve().relative_to(ROOT).as_posix()


def write_zip(path: Path, members: list[tuple[str, bytes]]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name, data in sorted(members):
            info = zipfile.ZipInfo(name, date_time=ZIP_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, data)


def manuscript_field(macro: str) -> str:
    tex = (ROOT / "manuscript" / "main_erl.tex").read_text(encoding="utf-8")
    body = tex[tex.find("\\begin{document}"):]
    m = re.search(r"\\" + macro + r"\{([^{}]*)\}", body)
    if not m:
        raise SystemExit(f"main_erl.tex: no \\{macro}{{...}}")
    return re.sub(r"\s+", " ", m.group(1)).strip()


def input_dois() -> list[tuple[str, str]]:
    """(DOI, provenance file) of the bathymetry grids and wetland layers behind the tables."""
    out = []
    for name in ("gebco_2026", "gebco_2022", "etopo_2022", "etopo1", "habitat_rasters"):
        p = ROOT / "results" / "provenance" / f"{name}.json"
        if not p.exists():
            continue
        found: list[str] = []

        def walk(x):
            if isinstance(x, dict):
                for k, v in x.items():
                    if k in ("doi", "data_doi") and isinstance(v, str):
                        found.append(v.split()[0])
                    walk(v)
            elif isinstance(x, list):
                for v in x:
                    walk(v)
        walk(json.loads(p.read_text(encoding="utf-8")))
        out += [(d, rel(p)) for d in found if (d, rel(p)) not in out]
    return out


def git_state() -> str:
    try:
        head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", "results"],
                               capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown (git not available)"
    return f"{head}" + (" plus uncommitted changes under results/" if dirty else "")


RASTERS = [
    ("data/interim/stage2/main/tile_*.npz",
     "Level-binned water-frequency counts (valid and water observations per epoch and level bin), tiled",
     "Earth Engine export `scripts/a6_stage2_export.py` (record `results/inventory/stage2_mainL.json`), combined by "
     "`scripts/a6_stage2_combine.py` (`make results/provenance/stage2_main_combine.json`)"),
    ("data/interim/waterlines/main/points.csv.gz",
     "Waterline cells of every (epoch, level bin), with the static-contour flag",
     "`scripts/a6_waterlines.py` and `scripts/a6_static_rule.py` (`make results/levels/static_contour_rule.json`)"),
    ("data/processed/dem/main/dem_*.tif", "Waterline elevation model of the emerged band, 30 m, 60 km tiles",
     "`scripts/a6_dem.py` (`make data/interim/.stamp_dem_main`)"),
    ("data/processed/dem/main/emergence/emergence_*.tif", "Emergence date of every elevation-model pixel",
     "`scripts/a6_emergence.py` (`make data/interim/.stamp_emergence_main`)"),
    ("data/interim/hindcast/", "Flood-level maps of the five grids at every observed level (cache)",
     "`scripts/a7_hindcast.py` (`make all`)"),
    ("data/interim/emergence/", "Flood-level maps at the near-term scenario levels (cache)",
     "`scripts/a8_emergence.py` (`make all`)"),
]

INPUT_TERMS = [
    ("Landsat Collection 2 Level-2 (USGS), via Google Earth Engine", "USGS data policy (no restrictions on use)"),
    ("DAHITI water levels (DGFI-TUM); Hydroweb levels (CNES/Theia)", "the providers' terms of use; cite as in the paper"),
    ("GEBCO_2026, GEBCO_2022 grids; ETOPO 2022, ETOPO1 (NOAA NCEI); SRTM15+ V2.7 (Scripps)",
     "the providers' terms of use; cite as in the paper (DOIs in `metadata.json`)"),
    ("Copernicus DEM GLO-30 (H2 metric keys `h2.glo30.*`)", "Copernicus WorldDEM-30 licence; notice below"),
    ("ICESat-2 (NSIDC) and GEDI (LP DAAC), NASA", "NASA Earth science data policy (open)"),
    ("ERA5 (ECMWF, Copernicus Climate Change Service)", "Licence to use Copernicus products; notice below"),
    ("OpenStreetMap (exposure counts and lengths only)", "ODbL 1.0; (c) OpenStreetMap contributors"),
    ("ESA WorldCover 2021, GLWD v2, GWL_FCS30 (wetland exposure areas only)", "CC BY 4.0 (recorded in "
     "`results/provenance/habitat_rasters.json`)"),
    ("USGS gauge levels and 3DEP elevation data (Great Salt Lake)", "USGS data policy (no restrictions on use)"),
    ("CASPCOM gauge levels", "no licence stated: not released (only the basin-wide offset and its uncertainty)"),
    ("World Database on Protected Areas (UNEP-WCMC and IUCN)", "not released in this package (see Exclusions)"),
]

GLO30_NOTICE = ("Values derived from the Copernicus DEM were produced using Copernicus WorldDEM-30 (c) DLR e.V. "
                "2010-2014 and (c) Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the "
                "European Union and ESA; all rights reserved. The organisations in charge of the Copernicus "
                "programme by law or by delegation do not incur any liability for any use of the Copernicus "
                "WorldDEM-30.")
ERA5_NOTICE = "Wind-setup values were generated using Copernicus Climate Change Service information (ERA5)."
OSM_NOTICE = ("Exposure counts and lengths are derived from OpenStreetMap data, (c) OpenStreetMap contributors, "
              "available under the Open Database License (ODbL 1.0).")


def readme(title: str, creators: list[dict], archives: list[dict], top: list[dict], excluded: list[tuple[str, str]],
           dropped: list[tuple[str, str]], dois: list[tuple[str, str]], state: str, metrics: dict) -> str:
    fall = f"{metrics['levels.dahiti.smoothed.range_m']['value']:.1f} m"
    d0 = metrics["levels.dahiti.smoothed.date_of_max"]["value"][:4]
    d1 = metrics["levels.dahiti.smoothed.date_of_min"]["value"][:4]
    authors = "; ".join(f"{c['name']} (ORCID {c['orcid']})" for c in creators)
    L = [f"# Derived tables for \"{title}\"", "",
         f"**Authors:** {authors}.", "",
         "**Licence:** Creative Commons Attribution 4.0 International (CC BY 4.0); the legal code is in `LICENSE`. "
         "Third-party notices below apply to the values derived from the inputs named there.", "",
         f"**Code and full audit trail:** {REPO_URL} (MIT licence for the code). Every table here is a file of the "
         "repository's `results/` folder, at the same path inside its archive; `MANIFEST.csv` gives each member's "
         "source path, size and SHA-256. Built by `tools/build_data_package.py` (`make data-package`) from "
         f"repository state {state}.", "",
         "## What the data are", "",
         f"The level of the North Caspian Sea fell by {fall} between {d0} and {d1} (`levels.dahiti.smoothed.*` "
         "in `metrics.json`). Landsat waterlines, each paired with the "
         "satellite-altimetry level of its acquisition date and aggregated into 0.1 m level bins per epoch, record "
         "where the shoreline was at each level. Five global bathymetry grids (GEBCO_2026, GEBCO_2022, ETOPO 2022, "
         "ETOPO1, SRTM15+ V2.7) were flooded from the open sea at each observed level and compared, sector by "
         "sector, with the observed extents and waterlines (the hindcast). The same test was run at Great Salt "
         "Lake. The near-term tables give the seabed that each grid places in the next 0.5-5 m of fall, with "
         "conformal intervals, and its exposure. Methods, hypotheses and decision rules: the paper, its "
         "Supplementary Information and the pre-registration (`docs/PREREG.md` in the repository).", "",
         "## Contents", "",
         "| File | Contents |", "|---|---|"]
    for t in top:
        L.append(f"| `{t['name']}` | {t['desc']} |")
    for a in archives:
        L.append(f"| `{a['name']}` | **{a['title']}.** {a['desc']} {a['n']} files. |")
    L += ["", "Formats: CSV (UTF-8, comma-separated, header row; `.csv.gz` gzip-compressed), JSON and GeoJSON (WGS84). "
          "Each `run.json` records the command, settings and inputs of the run that wrote its folder. "
          "`metrics.json` maps each key to its value, unit, source file and, where needed, a note; every number in "
          "the paper is one of these keys.", "",
          "## Datums and units", "",
          "- Heights and levels are in metres on the working datum of the analysis, the DAHITI reference (normal "
          "heights above the EIGEN-6C4 quasi-geoid), unless a column or key says otherwise. Levels quoted in the "
          "Baltic Height System 1977 add the constant offset delta (`datum.bhs.delta_m` in `metrics.json`).",
          "- The full vertical datum chain (DAHITI, Hydroweb, ICESat-2, GEDI, GLO-30, the five grids, the Baltic "
          "system) is documented in `docs/DATUMS.md` of the repository: "
          f"{REPO_URL}/blob/main/docs/DATUMS.md",
          "- Horizontal coordinates are WGS84 (EPSG:4326). Areas are computed on an equal-area Lambert azimuthal "
          "projection centred at 45.5 N, 50 E (`+proj=laea +lat_0=45.5 +lon_0=50`) and, for "
          "Great Salt Lake, a basin-centred one of the same kind (`run.json` of each arm); distances in metres.", "",
          "## Provenance", "",
          "- Pipeline: `make all` in the repository rebuilds every table from the raw inputs; acquisition steps "
          "(downloads and Earth Engine exports) run separately (`make fetch`, `make ee-inputs EE_OK=1`).",
          "- Input acquisition records (URLs, dates, SHA-256): `results/provenance/` of the repository.",
          "- The registered plan and its dated amendments: "
          "`docs/PREREG.md`.", "",
          "## Licences of the inputs", "",
          "| Input | Terms |", "|---|---|"]
    L += [f"| {a} | {b} |" for a, b in INPUT_TERMS]
    L += ["", "Re-check each provider's terms before reuse; they change without notice.", "",
          "### Third-party notices", "",
          f"- {GLO30_NOTICE}", f"- {ERA5_NOTICE}", f"- {OSM_NOTICE}", "",
          "## Exclusions", "",
          "Not in this package, and why:", "",
          "- **Large rasters** (git-ignored `data/` of the repository), regenerated from the raw inputs:", "",
          "| Path in the repository | Contents | Regenerated by |", "|---|---|---|"]
    L += [f"| `{p}` | {c} | {r} |" for p, c, r in RASTERS]
    L += ["", "- **Raw inputs:** never redistributed; each is acquired from its provider by the repository's "
          "download and Earth Engine scripts.",
          "- **CASPCOM gauge values:** the catalogue states no licence (\"All Rights Reserved\"), so no per-gauge "
          "value is released. The basin-wide offset to the Baltic system and its uncertainty are in `metrics.json`; "
          "`scripts/a7_dahiti_bhs_offset.py` recomputes everything from the catalogue.",
          "- **WDPA-derived values** (protected-area and Ramsar exposure): the database's terms allow non-commercial "
          "use only and forbid redistributing the data as files, which CC BY 4.0 cannot carry. The polygons are "
          "never redistributed; the derived counts and areas appear in the paper, citing UNEP-WCMC and IUCN (2026), "
          "Protected Planet: The World Database on Protected Areas (WDPA) and World Database on Other Effective "
          "Area-based Conservation Measures (WD-OECM) [Online], October 2026, Cambridge, UK: UNEP-WCMC and IUCN, "
          "www.protectedplanet.net.",
          "- **Other result folders** (robustness runs, H2 validation tables, the other revision analyses, "
          "Earth Engine inventories): in the repository's `results/`; outside the scope of this package.", "",
          "Files left out by the build's rules and content scan:", ""]
    L += [f"- `{f}`: {why}" for f, why in excluded] or ["- none"]
    by_reason: dict[str, int] = {}
    for _, why in dropped:
        by_reason[why] = by_reason.get(why, 0) + 1
    L += ["", f"Keys removed from `metrics.json` ({len(dropped)}; names in `metrics_excluded_keys.csv`, values "
          "not released):", ""]
    L += [f"- {n} keys: {why}" for why, n in sorted(by_reason.items())] or ["- none"]
    L += ["", "## Checksums", "",
          "`SHA256SUMS` covers every file of the package except itself (`sha256sum -c SHA256SUMS`). Archive members "
          "are covered by `MANIFEST.csv`. The archives are byte-reproducible (fixed member timestamps).", "",
          "## How to cite", "",
          "Cite the dataset by the DOI that Zenodo assigns on deposit, and the paper:", "",
          f"- {'; '.join(c['name'] for c in creators)}. *{title}* (data set, version 1.0). Zenodo. "
          "doi:[assigned on deposit].",
          f"- {'; '.join(c['name'] for c in creators)}. *{title}*. Remote Sensing of Environment (submitted).", "",
          "Also cite the input products you rely on (the grids' DOIs are in `metadata.json`; the full reference "
          "list is in the paper), and keep the third-party notices above with any redistribution.", ""]
    return "\n".join(L)


def zenodo_metadata(title: str, creators: list[dict], keywords: list[str], dois: list[tuple[str, str]]) -> dict:
    related = [{"identifier": REPO_URL, "relation": "isCompiledBy", "resource_type": "software", "scheme": "url"}]
    related += [{"identifier": d, "relation": "isDerivedFrom", "resource_type": "dataset", "scheme": "doi"}
                for d, _ in dois]
    desc = (f"<p>Derived tables for the paper \"{title}\" (Environmental Research Letters, submitted).</p>"
            "<p>Landsat waterlines of the North Caspian Sea (1992-2026), each paired with the satellite-altimetry "
            "level of its date and aggregated into level bins, are used to test five global bathymetry grids: at "
            "every observed level each grid is flooded from the open sea and its shoreline compared, sector by "
            "sector, with the observed one. The package holds the level-binned observed extents and waterline "
            "summaries, the hindcast tables for the North Caspian and Great Salt Lake, the near-term emergence "
            "and exposure tables, and metrics.json, the source of every number in the paper.</p>"
            f"<p>Code, pipeline and audit trail: <a href=\"{REPO_URL}\">{REPO_URL}</a>. See README.md for contents, "
            "datums, input licences and exclusions (CASPCOM per-gauge values and WDPA-derived values are not "
            "released).</p>")
    notes = " ".join([GLO30_NOTICE, ERA5_NOTICE, OSM_NOTICE])
    return {"metadata": {
        "upload_type": "dataset",
        "title": f"Derived tables for \"{title}\"",
        "creators": [{"name": c["name"], "affiliation": c["affiliation"], "orcid": c["orcid"]} for c in creators],
        "description": desc,
        "access_right": "open",
        "license": "cc-by-4.0",
        "keywords": keywords,
        "related_identifiers": related,
        "language": "eng",
        "version": "1.0",
        "notes": notes,
    }}


def creators_from_cff() -> list[dict]:
    cff = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    out = []
    for a in cff["authors"]:
        orcid = a.get("orcid", "").rsplit("/", 1)[-1]
        out.append({"name": f"{a['family-names']}, {a['given-names']}", "affiliation": a.get("affiliation", ""),
                    "orcid": orcid})
    return out


def build(out: Path) -> dict:
    out = out.resolve()
    if RELEASE_DIR.resolve() not in out.parents:
        raise SystemExit(f"refusing to write outside {RELEASE_DIR}: {out}")
    lic = LICENCE_SRC.read_bytes()
    if sha256_bytes(lic.replace(b"\r\n", b"\n")) != LICENCE_SHA256:
        raise SystemExit(f"{rel(LICENCE_SRC)} differs from the legal code retrieved from {LICENCE_URL}")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    post_re = post_regex(caspcom_posts())
    excluded: list[tuple[str, str]] = []
    manifest: list[dict] = []
    archives: list[dict] = []
    for name, title, desc, pats in ARCHIVES:
        members = []
        for p in collect(pats):
            r = rel(p)
            why = content_exclusion(r, file_text(p), post_re)
            if why:
                excluded.append((r, why))
                continue
            data = p.read_bytes()
            members.append((r, data))
            manifest.append({"archive": name, "member": r, "source": r, "bytes": len(data),
                             "sha256": sha256_bytes(data)})
        if not members:
            raise SystemExit(f"{name}: no files")
        write_zip(out / name, members)
        archives.append({"name": name, "title": title, "desc": desc, "n": len(members)})

    top = []
    for r in TOP_LEVEL:
        p = ROOT / r
        why = content_exclusion(r, file_text(p), post_re)
        if why:
            raise SystemExit(f"{r}: {why}")
        shutil.copyfile(p, out / p.name)
        top.append({"name": p.name, "desc": f"Copy of `{r}`: the parameters and display constants of the pipeline."})

    metrics = json.loads((ROOT / METRICS).read_text(encoding="utf-8"))
    kept, dropped = filter_metrics(metrics, post_re)
    leftover = [k for k, v in kept.items() if WDPA_RE.search(json.dumps(v)) or
                (post_re is not None and post_re.search(json.dumps(v)))]
    if leftover:
        raise SystemExit(f"metrics.json: keys still naming WDPA or CASPCOM posts after filtering: {leftover[:10]}")
    (out / "metrics.json").write_text(json.dumps(kept, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    with (out / "metrics_excluded_keys.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["key", "reason"])
        w.writerows(dropped)
    top = [{"name": "metrics.json", "desc": f"`{METRICS}` without the {len(dropped)} keys listed in "
            f"`metrics_excluded_keys.csv` ({len(kept)} keys): every number in the paper."},
           {"name": "metrics_excluded_keys.csv", "desc": "Names of the metric keys not released, with the reason."},
           *top,
           {"name": "MANIFEST.csv", "desc": "Every archive member: source path in the repository, bytes, SHA-256."},
           {"name": "metadata.json", "desc": "Zenodo deposit metadata."},
           {"name": "LICENSE", "desc": "CC BY 4.0 legal code."},
           {"name": "SHA256SUMS", "desc": "SHA-256 of every other file of the package."}]

    with (out / "MANIFEST.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, ["archive", "member", "source", "bytes", "sha256"], lineterminator="\n")
        w.writeheader()
        w.writerows(manifest)

    title = manuscript_field("title")
    keywords = [k.strip() for k in manuscript_field("keywords").split(",") if k.strip()]
    keywords += [k for k in ("North Caspian Sea", "seabed emergence", "hindcast", "waterlines")
                 if k not in keywords]
    creators = creators_from_cff()
    dois = input_dois()
    (out / "LICENSE").write_bytes(lic)
    (out / "metadata.json").write_text(json.dumps(zenodo_metadata(title, creators, keywords, dois), indent=2,
                                                  ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "README.md").write_text(readme(title, creators, archives, top, excluded, dropped, dois, git_state(), kept),
                                   encoding="utf-8")

    files = sorted(p for p in out.iterdir() if p.is_file() and p.name != "SHA256SUMS")
    (out / "SHA256SUMS").write_text("".join(f"{sha256_file(p)}  {p.name}\n" for p in files), encoding="utf-8")
    total = sum(p.stat().st_size for p in out.iterdir() if p.is_file())
    return {"out": str(out), "files": len(files) + 1, "members": len(manifest), "bytes": total,
            "excluded_files": excluded, "excluded_keys": len(dropped), "kept_keys": len(kept)}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = ap.parse_args(argv)
    s = build(a.out if a.out.is_absolute() else ROOT / a.out)
    print(f"data package: {s['out']}: {s['files']} files, {s['members']} archive members, "
          f"{s['bytes'] / 1e6:.1f} MB; metrics keys kept {s['kept_keys']}, removed {s['excluded_keys']}")
    for f, why in s["excluded_files"]:
        print(f"  excluded {f}: {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
