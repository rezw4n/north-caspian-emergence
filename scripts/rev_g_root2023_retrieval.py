"""Revision analysis G (PREREG Amendment 18 item 11, Great Salt Lake): retry the
scripted retrieval of the registered positive control, the USGS half-metre
topobathymetric DEM of Great Salt Lake (Root 2023, doi:10.5066/P9DGG75W), from
ScienceBase.

Amendment 16 item 7 registered this DEM as the positive control. On 2026-10-21 every
scripted request to ScienceBase met a browser challenge (HTTP 403). This script asks
again, without trying to get round any challenge:

1. the DOI resolver (where doi.org sends the DOI);
2. the ScienceBase catalogue JSON of the release item and of the GeoTIFF item (the two
   item ids recorded at the first attempt), which lists every file with its size;
3. a HEAD request (no body) on each file's download URL, and on the S3 URL if the
   catalogue gives one, to see whether it would be served to a script;
4. two public routes outside ScienceBase's web front end: the DataCite record of the
   DOI (sizes and formats as registered) and a HEAD request on the public ScienceBase
   S3 publish bucket at the GeoTIFF's file name recorded at the first attempt
   (Great_Salt_Lake_TBDEM.tif). An AccessDenied answer is recorded, not worked round.

Nothing is downloaded beyond the JSON records and response headers. The download rule
of the revision (log over 2 GB, never over 15 GB) is applied to
the sizes found: a file over 15 GB is not downloaded whatever the response.

Output: results/revision/g/root2023_retrieval.json (every URL, the time, the HTTP
status, the content type, the first 300 characters of any non-JSON body, and the file
list with sizes when it was served).
Usage: python scripts/rev_g_root2023_retrieval.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "revision" / "g" / "root2023_retrieval.json"
DOI = "10.5066/P9DGG75W"
ITEMS = {"release": "64595c00d34ec179a8368788", "geotiff": "646d0ed2d34ee02593fb50a7"}   # recorded on 2026-10-21
LIMIT_GB = 15.0
UA = "caspian-revision-g/1.0 (research script; contact rezwan490@gmail.com)"
TIMEOUT = 60


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _record(method: str, url: str, r: requests.Response | None, err: str | None = None) -> dict:
    rec = {"method": method, "url": url, "time_utc": _now()}
    if r is None:
        rec.update(status=None, error=err)
        return rec
    rec.update(status=r.status_code, final_url=r.url, content_type=r.headers.get("Content-Type"),
               content_length=r.headers.get("Content-Length"), server=r.headers.get("Server"),
               cf_mitigated=r.headers.get("cf-mitigated"),
               redirects=[h.headers.get("Location") for h in r.history])
    if method == "GET" and "json" not in (r.headers.get("Content-Type") or ""):
        rec["body_head"] = r.text[:300]
    return rec


def _get(s: requests.Session, url: str, **kw):
    try:
        return s.get(url, timeout=TIMEOUT, **kw), None
    except requests.RequestException as e:
        return None, f"{type(e).__name__}: {e}"


def _head(s: requests.Session, url: str):
    try:
        return s.head(url, timeout=TIMEOUT, allow_redirects=True), None
    except requests.RequestException as e:
        return None, f"{type(e).__name__}: {e}"


def main() -> int:
    s = requests.Session()
    s.headers["User-Agent"] = UA
    log, files = [], []
    r, err = _get(s, f"https://doi.org/{DOI}", allow_redirects=True)
    log.append(_record("GET", f"https://doi.org/{DOI}", r, err))
    for name, item in ITEMS.items():
        for url in (f"https://www.sciencebase.gov/catalog/item/{item}?format=json",
                    f"https://www.sciencebase.gov/catalog/item/{item}?format=json&fields=files,title,distributionLinks"):
            r, err = _get(s, url, headers={"Accept": "application/json"})
            rec = _record("GET", url, r, err)
            if r is not None and r.status_code == 200 and "json" in (r.headers.get("Content-Type") or ""):
                j = r.json()
                rec["title"] = j.get("title")
                for f in j.get("files", []) or []:
                    files.append({"item": name, "name": f.get("name"), "size_bytes": f.get("size"),
                                  "size_gb": round(f["size"] / 1e9, 3) if f.get("size") else None,
                                  "url": f.get("url"), "downloadUri": f.get("downloadUri"),
                                  "s3": (f.get("cuid") or None)})
                rec["distributionLinks"] = j.get("distributionLinks")
            log.append(rec)
    url = f"https://api.datacite.org/dois/{DOI}"
    r, err = _get(s, url)
    rec = _record("GET", url, r, err)
    if r is not None and r.status_code == 200:
        a = r.json()["data"]["attributes"]
        rec.update(title=[t.get("title") for t in a.get("titles", [])], sizes=a.get("sizes"),
                   formats=a.get("formats"), landing_url=a.get("url"), publisher=a.get("publisher"),
                   published=a.get("publicationYear"))
    log.append(rec)
    heads = []
    for key in ITEMS.values():
        url = f"https://prod-is-usgs-sb-prod-publish.s3.amazonaws.com/{key}/Great_Salt_Lake_TBDEM.tif"
        r, err = _head(s, url)
        heads.append({"file": "Great_Salt_Lake_TBDEM.tif (S3 publish bucket)", **_record("HEAD", url, r, err)})
    for f in files:
        for key in ("url", "downloadUri"):
            if f.get(key):
                r, err = _head(s, f[key])
                heads.append({"file": f["name"], **_record("HEAD", f[key], r, err)})
    big = [f for f in files if (f.get("size_gb") or 0) > LIMIT_GB]
    served = [h for h in heads if h.get("status") == 200 and "html" not in (h.get("content_type") or "")]
    out = {"created": _now(), "doi": DOI, "items": ITEMS, "requests": log, "files": files, "heads": heads,
           "download_limit_gb": LIMIT_GB,
           "files_over_limit": [f["name"] for f in big],
           "any_file_served_to_a_script": bool(served),
           "note": "No challenge was circumvented; nothing beyond JSON records and headers was downloaded."}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("files_over_limit", "any_file_served_to_a_script")}, indent=1))
    for rec in log + heads:
        print(rec["method"], rec["status"], rec.get("content_type"), rec["url"])
    for f in files:
        print(f["item"], f["name"], f["size_gb"], "GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
