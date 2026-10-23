"""Revision 2, workstream G (PREREG Amendment 19 item 13): the Great Salt Lake lake-bed
elevation model from HydroShare, downloaded by script with its documentation.

Resource: HydroShare 582060f00f6b443bb26e896426d9f62a, "Great Salt Lake Bathymetry"
(D. Tarboton, M. Merck). Files are taken from the resource's own file listing; each
download is checked against the MD5 that HydroShare lists for it.

Commands:
  docs          the file listing, system metadata (JSON), science metadata (XML), the
                resource landing page (HTML, for the licence statement), README.txt and
                GSLDEM.vrt
  file <name>   one file of the listing by its file_name (e.g. GSLDEM.tif), streamed
                to disk; MD5 and SHA-256 recorded
  usgs          the two USGS survey reports the README cites (Baskin 2005, 2006):
                Pubs Warehouse records and their PDFs, under docs/
  text          plain text of those PDFs (pypdf), saved next to them
  cite          results/revision2/g/citation.json: the dataset's BibTeX-ready record and
                the two survey reports (Crossref title check, BibTeX by DOI)

Everything lands in data/raw/revision2/gsl_biowest/ (git-ignored); the record of every
request (URL, UTC time, bytes, checksums, HTTP status) is appended to
data/raw/revision2/gsl_biowest/download_log.json and copied to
results/revision2/g/gsl_lakebed_download.json.
Usage: python scripts/rev2_g_lakebed_data.py docs | file GSLDEM.tif
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RID = "582060f00f6b443bb26e896426d9f62a"
API = f"https://www.hydroshare.org/hsapi/resource/{RID}"
LANDING = f"https://www.hydroshare.org/resource/{RID}/"
DST = ROOT / "data" / "raw" / "revision2" / "gsl_biowest"
LOG = DST / "download_log.json"
RES_COPY = ROOT / "results" / "revision2" / "g" / "gsl_lakebed_download.json"
UA = {"User-Agent": "caspian-research/0.1 (rezwan490@gmail.com)"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(entry: dict) -> None:
    rec = json.loads(LOG.read_text(encoding="utf-8")) if LOG.exists() else {"resource": RID, "requests": []}
    rec["requests"].append(entry)
    LOG.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    RES_COPY.parent.mkdir(parents=True, exist_ok=True)
    RES_COPY.write_text(json.dumps(rec, indent=1), encoding="utf-8")


def get_small(url: str, name: str) -> bytes:
    t = now()
    r = requests.get(url, timeout=120, headers=UA)
    r.raise_for_status()
    (DST / name).write_bytes(r.content)
    log({"what": name, "url": url, "requested_utc": t, "http_status": r.status_code, "bytes": len(r.content),
         "md5": hashlib.md5(r.content).hexdigest(), "sha256": hashlib.sha256(r.content).hexdigest(),
         "content_type": r.headers.get("Content-Type")})
    return r.content


def listing() -> list[dict]:
    p = DST / "files_listing.json"
    if not p.exists():
        get_small(f"{API}/files/", "files_listing.json")
    return json.loads(p.read_text(encoding="utf-8"))["results"]


def docs() -> int:
    get_small(f"{API}/files/", "files_listing.json")
    get_small(f"{API}/sysmeta/", "sysmeta.json")
    get_small(f"{API}/scimeta/", "scimeta.xml")
    get_small(LANDING, "landing_page.html")
    for f in listing():
        if f["file_name"] in ("README.txt", "GSLDEM.vrt"):
            fetch(f)
    return 0


def fetch(f: dict) -> dict:
    """Stream one listed file to disk; verify HydroShare's MD5."""
    url = f["url"].replace("http://", "https://", 1)
    out = DST / f["file_name"]
    t = now()
    md5, sha = hashlib.md5(), hashlib.sha256()
    n = 0
    with requests.get(url, timeout=600, headers=UA, stream=True) as r:
        r.raise_for_status()
        status = r.status_code
        with open(out.with_suffix(out.suffix + ".part"), "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                md5.update(chunk)
                sha.update(chunk)
                n += len(chunk)
    out.with_suffix(out.suffix + ".part").replace(out)
    entry = {"what": f["file_name"], "url": url, "requested_utc": t, "finished_utc": now(), "http_status": status,
             "bytes": n, "listed_bytes": f["size"], "md5": md5.hexdigest(), "listed_md5": f["checksum"],
             "md5_matches_listing": md5.hexdigest() == f["checksum"], "size_matches_listing": n == f["size"],
             "sha256": sha.hexdigest(), "listed_modified_time": f["modified_time"]}
    log(entry)
    print(json.dumps(entry, indent=1))
    if not (entry["md5_matches_listing"] and entry["size_matches_listing"]):
        raise SystemExit(f"{f['file_name']}: checksum or size differs from HydroShare's listing")
    return entry


USGS_REPORTS = ("ofr20051327", "ofr20061359")   # Baskin (2005, south part), Baskin (2006, north part)
PUBS = "https://pubs.usgs.gov/pubs-services/publication/"


def usgs() -> int:
    """The two USGS survey reports the README names: the Pubs Warehouse record (JSON)
    and every linked Document (PDF), saved under docs/."""
    (DST / "docs").mkdir(exist_ok=True)
    for rid in USGS_REPORTS:
        raw = get_small(f"{PUBS}?q={rid}", f"docs/{rid}_pubs_record.json")
        rec = [r for r in json.loads(raw)["records"] if r.get("indexId") == rid]
        if len(rec) != 1:
            raise SystemExit(f"{rid}: {len(rec)} matching records")
        for link in rec[0].get("links", []):
            if (link.get("type") or {}).get("text") == "Document":
                url = link["url"]
                get_small(url, f"docs/{rid}_{Path(url).name}")
    return 0


def pdf_text() -> int:
    """Plain text of each saved PDF (pypdf), next to it, for reading and quoting."""
    from pypdf import PdfReader
    for f in sorted((DST / "docs").glob("*.pdf")):
        pages = PdfReader(str(f)).pages
        txt = "\n".join(f"=== page {i + 1} ===\n{p.extract_text() or ''}" for i, p in enumerate(pages))
        f.with_suffix(".txt").write_text(txt, encoding="utf-8")
        print(f.name, len(pages), "pages", len(txt), "characters")
    return 0


def cite() -> int:
    """results/revision2/g/citation.json: a BibTeX-ready record of the dataset, from the
    resource's own science metadata (creators in their order, HydroShare's suggested
    citation, licence; the resource has no DOI), and the two USGS survey reports it
    cites, each resolved through Crossref by DOI (title checked against the Pubs
    Warehouse record) with BibTeX by DOI content negotiation."""
    import re
    import xml.etree.ElementTree as ET
    ns = {"dc": "http://purl.org/dc/elements/1.1/", "dcterms": "http://purl.org/dc/terms/",
          "hsterms": "https://www.hydroshare.org/terms/", "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#"}
    root = ET.parse(DST / "scimeta.xml").getroot()
    res = root.find("hsterms:CompositeResource", ns)
    creators = []
    for c in res.findall("dc:creator/rdf:Description", ns):
        creators.append((int(c.findtext("hsterms:creatorOrder", "99", ns)), c.findtext("hsterms:name", "", ns),
                         c.findtext("hsterms:organization", "", ns)))
    creators = [{"name": n, "organization": o} for _, n, o in sorted(creators)]
    contributors = [{"name": c.findtext("hsterms:name", "", ns), "organization": c.findtext("hsterms:organization", "", ns)}
                    for c in res.findall("dc:contributor/rdf:Description", ns)]
    title = res.findtext("dc:title", "", ns)
    suggested = res.findtext("dcterms:bibliographicCitation", "", ns)
    year = int(re.search(r"\((\d{4})\)", suggested).group(1))
    rights = res.find("dc:rights/rdf:Description", ns)
    lic_url = rights.find("hsterms:URL", ns).attrib[f"{{{ns['rdf']}}}resource"]
    created = res.findtext("dc:date/dcterms:created/rdf:value", "", ns)
    modified = res.findtext("dc:date/dcterms:modified/rdf:value", "", ns)
    sysm = json.loads((DST / "sysmeta.json").read_text(encoding="utf-8"))

    def last_first(name: str) -> str:
        return name if "," in name else f"{name.split()[-1]}, {' '.join(name.split()[:-1])}"
    authors = " and ".join(last_first(c["name"]) for c in creators)
    url = f"http://www.hydroshare.org/resource/{RID}"
    bib = (f"@misc{{tarboton_gsl_bathymetry,\n  author = {{{authors}}},\n  title = {{{title}}},\n"
           f"  year = {{{year}}},\n  publisher = {{HydroShare}},\n  howpublished = {{HydroShare resource {RID}}},\n"
           f"  url = {{{url}}},\n  note = {{Dataset (GSLDEM.tif); licence CC BY 4.0; no DOI; accessed 2026-10-23}}\n}}")
    rec = {"dataset": {"key": "tarboton_gsl_bathymetry", "type": "dataset", "title": title, "creators": creators,
                       "contributors": contributors, "year": year,
                       "year_note": "the year of HydroShare's suggested citation (last modified %s); created %s; "
                                    "GSLDEM.tif dated 2017-10-28 in the file listing" % (modified[:10], created[:10]),
                       "publisher": "HydroShare", "doi": sysm.get("doi"), "published_with_doi": bool(sysm.get("published")),
                       "resource_id": RID, "url": url, "licence": "CC BY 4.0", "licence_url": lic_url,
                       "file_used": "GSLDEM/GSLDEM.tif", "hydroshare_suggested_citation": suggested,
                       "accessed": "2026-10-23", "bibtex": bib,
                       "tier": "data citation of the provider's own record (Tier A: 'official data or product "
                               "documentation from ... the data providers'); not peer reviewed and without a DOI, so "
                               "listed for the PI's check before it enters refs/"},
           "survey_reports": []}
    for rid in USGS_REPORTS:
        pub = [r for r in json.loads((DST / "docs" / f"{rid}_pubs_record.json").read_text(encoding="utf-8"))["records"]
               if r.get("indexId") == rid][0]
        doi = pub["doi"]
        cr = requests.get(f"https://api.crossref.org/works/{doi}", timeout=60, headers=UA)
        cr_title = cr.json()["message"]["title"][0] if cr.ok else None
        bt = requests.get(f"https://doi.org/{doi}", timeout=60, headers={**UA, "Accept": "application/x-bibtex"})
        rec["survey_reports"].append({
            "doi": doi, "pubs_title": pub["title"], "crossref_status": cr.status_code, "crossref_title": cr_title,
            "title_matches": (cr_title or "").strip().lower() == pub["title"].strip().lower(),
            "bibtex_status": bt.status_code, "bibtex": bt.text.strip() if bt.ok else None,
            "why": "survey source of the lake-bed model (README.txt); dates, depth limit and map-contour range"})
    out = ROOT / "results" / "revision2" / "g" / "citation.json"
    out.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(json.dumps(rec, indent=1))
    return 0


def main(argv) -> int:
    DST.mkdir(parents=True, exist_ok=True)
    if argv and argv[0] == "cite":
        return cite()
    if not argv or argv[0] == "docs":
        return docs()
    if argv[0] == "usgs":
        return usgs()
    if argv[0] == "text":
        return pdf_text()
    if argv[0] == "file" and len(argv) == 2:
        f = [x for x in listing() if x["file_name"] == argv[1]]
        if len(f) != 1:
            raise SystemExit(f"{argv[1]} not in the listing")
        fetch(f[0])
        return 0
    raise SystemExit(__doc__)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
