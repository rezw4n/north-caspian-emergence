"""ETOPO 2022 and HydroLAKES documentation on lakes (revision G, task B).

Downloads the NOAA NCEI ETOPO 2022 documentation and the HydroLAKES documentation, extracts their
text, and lists every passage that matches lake-related search terms, with page numbers, so the
passages can be read and quoted verbatim.

Sources (all public):
  - ETOPO 2022 User Guide (NCEI): https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO2022/docs/
    1.2 ETOPO 2022 User Guide.pdf (linked from https://www.ncei.noaa.gov/products/etopo-global-relief-model)
  - NCEI ETOPO pages "Data Sources", "Product Details", "Frequently Asked Questions" (PDF exports linked
    from the same product page)
  - ISO 19115 metadata record of ETOPO 2022 (doi:10.25921/fd45-gt74), XML from data.noaa.gov/waf
  - OPeNDAP attribute listing (.das) of one ETOPO 2022 source-ID tile (to see whether IDs are defined)
  - MacFerrin et al. 2025, ESSD 17:1835, doi:10.5194/essd-17-1835-2025 (open access PDF; to compare source IDs)
  - HydroLAKES technical documentation v1.0 (data.hydrosheds.org)
  - Messager et al. 2016, Nature Communications 7:13603, doi:10.1038/ncomms13603 (open-access full text
    XML from Europe PMC, PMC5171767) and the Crossref records of both DOIs
  - HydroLAKES_points_v10_shp.zip (79 MB): attributes of the Caspian Sea record (Poly_src etc.)

Writes:
  data/raw/revision_g/etopo/<file> and <file>.meta.json (URL, retrieval time UTC, HTTP status, size,
  SHA-256); <file>.txt with the extracted text, one "=== PAGE n ===" header per PDF page (git-ignored)
  results/revision/g/etopo_lake_passages.json  matches per search term, per document

Run from the repository root:
  micromamba run -n caspian python -I scripts/rev_g_etopo_docs.py [--refresh]
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
import re
import zipfile
from pathlib import Path

import fitz  # PyMuPDF, text extraction only
import pyogrio
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "revision_g" / "etopo"
OUT = ROOT / "results" / "revision" / "g" / "etopo_lake_passages.json"
UA = "Mozilla/5.0 (compatible; caspian-provenance/1.0; research script)"

DOCS = {
    "ETOPO_2022_User_Guide_v1.2.pdf":
        "https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO2022/docs/1.2%20ETOPO%202022%20User%20Guide.pdf",
    "NCEI_ETOPO_Data_Sources.pdf": "https://www.ncei.noaa.gov/sites/default/files/2023-01/Data%20Sources.pdf",
    "NCEI_ETOPO_Product_Details.pdf": "https://www.ncei.noaa.gov/sites/default/files/2023-01/Product%20Details.pdf",
    "NCEI_ETOPO_FAQ.pdf": "https://www.ncei.noaa.gov/sites/default/files/2023-01/Frequently%20Asked%20Questions.pdf",
    "etopo_2022_iso.xml": "https://data.noaa.gov/waf/NOAA/NESDIS/NGDC/MGG/DEM/iso/xml/etopo_2022.xml",
    "etopo_product_page.html": "https://www.ncei.noaa.gov/products/etopo-global-relief-model",
    "ETOPO_2022_v1_15s_N45E045_surface_sid.nc.das":
        "https://www.ngdc.noaa.gov/thredds/dodsC/global/ETOPO2022/15s/15s_surface_sid_netcdf/"
        "ETOPO_2022_v1_15s_N45E045_surface_sid.nc.das",
    "MacFerrin2025_essd-17-1835-2025.pdf":
        "https://essd.copernicus.org/articles/17/1835/2025/essd-17-1835-2025.pdf",
    "HydroLAKES_TechDoc_v10.pdf":
        "https://data.hydrosheds.org/file/technical-documentation/HydroLAKES_TechDoc_v10.pdf",
    # nature.com serves a cookie page to scripts; the open-access full text is taken from Europe PMC
    "Messager2016_ncomms13603_EuropePMC_PMC5171767.xml":
        "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC5171767/fullTextXML",
    "crossref_10.1038_ncomms13603.json": "https://api.crossref.org/works/10.1038/ncomms13603",
    "crossref_10.5194_essd-17-1835-2025.json": "https://api.crossref.org/works/10.5194/essd-17-1835-2025",
}
HYDROLAKES_POINTS = ("HydroLAKES_points_v10_shp.zip",
                     "https://data.hydrosheds.org/file/hydrolakes/HydroLAKES_points_v10_shp.zip")

TERMS = {
    "lake": r"\blakes?\b",
    "GEBCO lake depths / source 11": r"GEBCO[^.]{0,40}lake|lake depth|source[ _-]?id[^.]{0,30}\b11\b|\b11\b[^.]{0,20}GEBCO",
    "HydroLAKES": r"hydro\s*lakes",
    "flat surface": r"flat(ten)?(ed)?\s+(water\s+)?surface|flattened|\bflat\b",
    "vertical reference": r"\bMSL\b|mean sea level|EGM\s*2008|EGM2008|geoid|vertical datum|WGS\s*84 ellipsoid",
    "lake surface elevation": r"(lake|water)[ -](surface|level)[^.]{0,60}(elevation|height)|surface elevation",
    "Caspian": r"caspian",
    "FABDEM / Copernicus water": r"FABDEM|Copernicus|COPDEM|water\s*body|water\s*mask|WBM",
    "source ID grid": r"source[ _-]?id|\bSID\b|source identification",
    "SRTM water body data": r"SRTM\s*Water|SWBD|water\s*body\s*data",
}


def utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch(s: requests.Session, name: str, url: str, refresh: bool) -> dict:
    path = RAW / name
    meta = path.with_name(path.name + ".meta.json")
    if not refresh and path.exists() and meta.exists():
        m = json.loads(meta.read_text(encoding="utf-8"))
        if m.get("ok"):
            m["reused_from_disk"] = True
            return m
    path.parent.mkdir(parents=True, exist_ok=True)
    t = utcnow()
    try:
        r = s.get(url, timeout=300)
        body, status, final, ctype = r.content, r.status_code, r.url, r.headers.get("Content-Type")
        err = None if status == 200 else f"HTTP {status}"
        if err is None and name.endswith(".pdf") and not body.startswith(b"%PDF"):
            err = "response is not a PDF"
    except requests.RequestException as e:
        body, status, final, ctype, err = b"", None, None, None, f"{type(e).__name__}: {e}"
    path.write_bytes(body)
    m = {"file": path.relative_to(ROOT).as_posix(), "url": url, "final_url": final, "retrieved_utc": t,
         "http_status": status, "content_type": ctype, "bytes": len(body),
         "sha256": hashlib.sha256(body).hexdigest(), "ok": err is None, "error": err,
         "response_head": body[:300].decode("utf-8", "replace") if err else None}
    meta.write_text(json.dumps(m, indent=1), encoding="utf-8")
    print(("ok  " if m["ok"] else "FAIL"), status, len(body), name)
    return m


def extract(path: Path) -> list[tuple[str, str]]:
    """[(location, text)]; PDFs per page, XML/HTML as one chunk with tags removed."""
    if path.suffix.lower() == ".pdf":
        with fitz.open(path) as doc:
            meta = doc.metadata
            pages = [(f"PDF page {i + 1}", p.get_text()) for i, p in enumerate(doc)]
        return pages, meta
    t = path.read_text(encoding="utf-8", errors="replace")
    t = re.sub(r"(?is)<(script|style).*?</\1>", " ", t)
    return [("whole document (tags removed)", html.unescape(re.sub(r"<[^>]+>", " ", t)))], {}


def norm(t: str) -> str:
    t = t.replace("­\n", "").replace("­", "")
    return re.sub(r"\s+", " ", t).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()
    s = requests.Session()
    s.headers["User-Agent"] = UA
    out = {"script": "scripts/rev_g_etopo_docs.py", "run_utc": utcnow(),
           "note": "Matches are sentences (whitespace collapsed) containing a search term; read the .txt files "
                   "for context. Locations are 1-based PDF page indices.", "documents": {}}
    for name, url in DOCS.items():
        m = fetch(s, name, url, a.refresh)
        rec = {k: m.get(k) for k in ("url", "final_url", "retrieved_utc", "http_status", "bytes", "sha256", "ok",
                                     "error", "file")}
        if m["ok"]:
            chunks, pdfmeta = extract(ROOT / m["file"])
            rec["pdf_metadata"] = {k: v for k, v in pdfmeta.items() if v} if pdfmeta else None
            txt = ROOT / (m["file"] + ".txt")
            txt.write_text("".join(f"\n=== {loc} ===\n{t}" for loc, t in chunks), encoding="utf-8")
            rec["text_file"] = txt.relative_to(ROOT).as_posix()
            rec["n_chunks"] = len(chunks)
            hits = {}
            for term, rx in TERMS.items():
                found = []
                for loc, t in chunks:
                    for sent in re.split(r"(?<=[.!?])\s+(?=[A-Z(])", norm(t)):
                        if re.search(rx, sent, re.I):
                            found.append({"location": loc, "sentence": sent[:1200]})
                hits[term] = {"n": len(found), "matches": found[:60]}
            rec["matches"] = hits
        out["documents"][name] = rec

    # HydroLAKES attributes of the Caspian Sea (and the largest lakes) from the points shapefile.
    # The zip is only read (pyogrio /vsizip/), never extracted or executed.
    name, url = HYDROLAKES_POINTS
    m = fetch(s, name, url, a.refresh)
    hl = {k: m.get(k) for k in ("url", "retrieved_utc", "http_status", "bytes", "sha256", "ok", "error", "file")}
    if m["ok"]:
        zpath = ROOT / m["file"]
        with zipfile.ZipFile(zpath) as z:
            shp = [n for n in z.namelist() if n.lower().endswith(".shp")]
        hl["shapefile_in_zip"] = shp
        df = pyogrio.read_dataframe(f"/vsizip/{zpath.as_posix()}/{shp[0]}", read_geometry=False)
        cols = [c for c in ("Hylak_id", "Lake_name", "Country", "Continent", "Poly_src", "Lake_type", "Lake_area",
                            "Shore_len", "Vol_total", "Depth_avg", "Elevation") if c in df.columns]
        hl["n_records"] = int(len(df))
        hl["caspian_records"] = df[df["Lake_name"].fillna("").str.contains("Caspian", case=False)][cols] \
            .to_dict("records")
        hl["five_largest_by_Lake_area"] = df.nlargest(5, "Lake_area")[cols].to_dict("records")
        hl["poly_src_counts"] = df["Poly_src"].value_counts().to_dict()
    out["hydrolakes_points"] = hl
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", OUT.relative_to(ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
