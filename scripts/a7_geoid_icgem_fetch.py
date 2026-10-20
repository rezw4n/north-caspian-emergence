"""Request one regular-grid computation from the ICGEM calculation service and save the raw result.

Usage:
  python scripts/a7_geoid_icgem_fetch.py OUT_FILE MODEL TIDESYS LEFT RIGHT BOTTOM TOP STEP [FUNC] [ZERO_TERM]

Writes OUT_FILE (refuses to overwrite) and OUT_FILE + '.request.json' with every form field sent,
UTC timestamps, the calculation id, HTTP status/headers of the result and the sha256 of the file.
Session cookies are used only in memory; they are never written to disk.
"""
import caspian  # noqa: F401  (project rule: import caspian first)
import datetime as dt
import hashlib
import json
import pathlib
import re
import sys
import time

import requests

BASE = "https://icgem.gfz.de"
UA = {"User-Agent": "Mozilla/5.0 (research script; North Caspian study)"}
MODELS = {  # ids from https://icgem.gfz.de/json_modelmap/name?modeltype=longtime&usefor=calc (2026-10-21)
    "EIGEN-6C4": ("7fd8fe44aa1518cd79ca84300aef4b41ddb2364aef9e82b7cdaabdb60a9053f1", 2190),
    "EGM2008": ("c50128797a9cb62e936337c890e4425f03f0461d7329b09a8cc8561504465340", 2190),
}


def utc():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main():
    out = pathlib.Path(sys.argv[1])
    model, tidesys = sys.argv[2], sys.argv[3]
    left, right, bottom, top, step = sys.argv[4:9]
    func = sys.argv[9] if len(sys.argv) > 9 else "geoid"
    zero_term = (sys.argv[10] if len(sys.argv) > 10 else "on") == "on"
    if out.exists():
        sys.exit(f"refusing to overwrite {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    mid, deg = MODELS[model]

    fields = {
        "sel_type": "longtime",
        "sel_model": mid,
        "grid_left": left, "grid_right": right, "grid_top": top, "grid_bottom": bottom,
        "grid_step": step,
        "sel_func": func,
        "sel_refsys": "WGS84",
        "sel_tidesys": tidesys,
        "trunc_gentle": str(deg), "trunc_max": str(deg),
        "filter": "none", "flength": "1.0", "funit": "degree",
    }
    if zero_term:
        fields["zero_term"] = "on"  # checkbox: sent only when checked (browser semantics)

    rec = {"service": f"{BASE}/calcgrid", "submit_url": f"{BASE}/calcsrvgrid",
           "model_name": model, "form_fields": fields, "t_start_utc": utc()}
    s = requests.Session()
    s.headers.update(UA)
    s.get(f"{BASE}/calcgrid", timeout=60).raise_for_status()
    s.post(f"{BASE}/api_key/{int(time.time())}", timeout=60).raise_for_status()
    rec["load_before_submit"] = s.get(f"{BASE}/calc_check_load", timeout=60).text.strip()
    r = s.post(f"{BASE}/calcsrvgrid", files={k: (None, v) for k, v in fields.items()}, timeout=600)
    r.raise_for_status()
    m = re.search(r'href="/calc_res/([0-9a-f]{64})"', r.text)
    if not m:
        sys.exit("no calculation id in response")
    cid = m.group(1)
    npts = re.search(r"([0-9]+) gridpoints \[([^\]]*)\]", r.text)
    rec["calc_id"] = cid
    rec["service_reported_points"] = npts.group(0).replace("&nbsp;", " ") if npts else None
    print("calc", cid, rec["service_reported_points"], flush=True)

    last, t0 = 0, time.time()
    while True:
        u = s.get(f"{BASE}/calc_updates/{last}", timeout=120).json()
        last = u.get("updated", last)
        c = u.get("calcs", {}).get(cid)
        if c:
            print(c.get("calc_progress"), c.get("calc_stage"), flush=True)
            if c.get("calc_end"):
                rec["calc_status"] = {k: c.get(k) for k in ("calc_start", "calc_end", "calc_stage", "calc_progress")}
                break
        if time.time() - t0 > 3600:
            sys.exit("timeout waiting for ICGEM")
        time.sleep(5)

    r = s.get(f"{BASE}/calc_res/{cid}", timeout=600)
    r.raise_for_status()
    summ = pathlib.Path(str(out) + ".summary.html")
    summ.write_bytes(r.content)
    rec["summary_page_url"] = f"{BASE}/calc_res/{cid}"
    rec["summary_page_sha256"] = hashlib.sha256(r.content).hexdigest()
    r = s.get(f"{BASE}/calc_res_gdf/{cid}", timeout=600)
    rec["result_url"] = f"{BASE}/calc_res_gdf/{cid}"
    rec["result_http_status"] = r.status_code
    rec["result_headers"] = {k: v for k, v in r.headers.items() if k.lower() != "set-cookie"}
    r.raise_for_status()
    out.write_bytes(r.content)
    rec["t_end_utc"] = utc()
    rec["bytes"] = len(r.content)
    rec["sha256"] = hashlib.sha256(r.content).hexdigest()
    pathlib.Path(str(out) + ".request.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print("saved", out, rec["bytes"], rec["sha256"])


if __name__ == "__main__":
    main()
