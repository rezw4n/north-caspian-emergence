"""Revision, workstream V: the CASPCOM posts behind the DAHITI -> BHS-77 offset
(R2.M3(a), R1 minor 6, R1.M1(c); D-122).

Per Baltic-system post of the CASPCOM General Catalogue: the position as printed in
the workbook header ("Position data: latitude ..., longitude ..."), whether that
position lies in the study domain and in which sector, whether the post is one of the
seven century posts behind delta (D-093), and derived offset statistics only:
- m: the post's offset, mean over its overlap years of (post annual mean - DAHITI
  annual mean), exactly as results/levels/dahiti_bhs_offset.json (D-093);
- m - delta: the post's departure from the seven-post offset;
- warm and cold season: the mean of the monthly differences (post monthly mean -
  DAHITI monthly mean, both complete months, 1993 onward) over April-October and over
  November-March, minus delta.
The per-year values are not written (D-101: the gauge annual means are not
redistributed).

Outputs: results/revision/v/caspcom_posts.csv and caspcom_posts.json.
Usage: python scripts/rev_v_posts.py
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import zipfile
from pathlib import Path

import caspian  # noqa: F401
import numpy as np
import pandas as pd
from shapely.geometry import Point

from caspian import gauges as G
from caspian.config import RESULTS
from caspian.geography import DOMAIN, sectors

SCRIPTS = Path(__file__).resolve().parent
OUT = RESULTS / "revision" / "v"
WARM = (4, 5, 6, 7, 8, 9, 10)
FIRST_YEAR = 1993
POS = re.compile(r"latitude\W*?(\d+)\s*[^\d\s,']\s*(\d+)?.*?longitude\W*?(\d+)\s*[^\d\s,']\s*(\d+)?", re.I)


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def parse_position(text: str) -> dict:
    """Degrees and minutes as printed; decimal degrees as deg + min/60. A minute field of
    60 or more is not a valid minute and is flagged (the value is still deg + min/60)."""
    m = POS.search(re.sub(r"_+", " ", text).replace("′", "'"))
    if not m:
        return {"lat": np.nan, "lon": np.nan, "position_flag": "no position parsed"}
    la, lam, lo, lom = (int(v) if v else 0 for v in m.groups())
    flag = "minute field >= 60 as printed" if (lam >= 60 or lom >= 60) else ""
    return {"lat": la + lam / 60.0, "lon": lo + lom / 60.0, "position_flag": flag}


def headers(zip_path) -> dict:
    out = {}
    with zipfile.ZipFile(zip_path) as z:
        for fn in z.namelist():
            if not fn.lower().endswith((".xls", ".xlsx")):
                continue
            cells = G.read_cells(fn, z.read(fn))
            rows = [G._row_text(cells, r) for r in range(0, 8)]
            pos = next((r for r in rows if "position" in r.lower()), "")
            out[re.sub(r"_av$", "", Path(fn).stem)] = {"position_text": re.sub(r"_+", " ", pos).strip(),
                                                      **parse_position(pos)}
    return out


def main() -> int:
    a7o = _load("a7_dahiti_bhs_offset")
    off = json.loads((RESULTS / "levels" / "dahiti_bhs_offset.json").read_text(encoding="utf-8"))
    delta = float(off["primary"]["delta_m"])
    per = {p["post"]: p for p in off["all_baltic_posts"]}
    monthly, meta = G.read_catalogue_zip(a7o.ZIP)
    hd = headers(a7o.ZIP)
    dm = a7o.dahiti_monthly(a7o.MAX_GAP_DAYS)
    md = monthly.merge(dm[["year", "month", "level_m"]], on=["year", "month"], suffixes=("", "_dahiti"))
    md = md[(md["year"] >= FIRST_YEAR) & md["level_m"].notna() & md["level_m_dahiti"].notna()]
    md["d"] = md["level_m"] - md["level_m_dahiti"]
    secs = sectors()
    rows = []
    for post, p in per.items():
        h = hd.get(post, {})
        pt = Point(h.get("lon", np.nan), h.get("lat", np.nan))
        inside = bool(np.isfinite(h.get("lat", np.nan)) and DOMAIN.contains(pt))
        sector = next((k for k, (_, poly) in secs.items() if poly.contains(pt)), "") if inside else ""
        g = md[md["post"] == post]
        w, c = g[g["month"].isin(WARM)], g[~g["month"].isin(WARM)]
        m = p.get("m")
        rows.append({"post": post, "name": p.get("name"), "century_post": post in a7o.CENTURY,
                     "north_caspian_set": post in a7o.NORTH,
                     "position_as_printed": h.get("position_text", ""),
                     "lat": round(h.get("lat", np.nan), 3), "lon": round(h.get("lon", np.nan), 3),
                     "position_flag": h.get("position_flag", ""), "in_study_domain": inside, "sector": sector,
                     "n_years": p.get("n_years"), "first_year": p.get("first"), "last_year": p.get("last"),
                     "offset_m": m, "offset_minus_delta_m": (m - delta) if m is not None and np.isfinite(m) else np.nan,
                     "trend_m_per_decade": p.get("trend_m_per_decade"),
                     "n_months_warm": int(len(w)), "warm_minus_delta_m": float(w["d"].mean() - delta) if len(w) else np.nan,
                     "n_months_cold": int(len(c)), "cold_minus_delta_m": float(c["d"].mean() - delta) if len(c) else np.nan})
    t = pd.DataFrame(rows).sort_values(["century_post", "north_caspian_set", "post"], ascending=[False, False, True])
    for col in t.columns:
        if t[col].dtype.kind == "f":
            t[col] = t[col].round(3)
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "caspcom_posts.csv", index=False)
    cen = t[t["century_post"]]
    rec = {"delta_m": delta, "delta_definition": off["definition"],
           "century_posts_in_study_domain": int(cen["in_study_domain"].sum()),
           "north_set_in_study_domain": int(t[t["north_caspian_set"]]["in_study_domain"].sum()),
           "century_offset_range_m": [float(cen["offset_m"].min()), float(cen["offset_m"].max())],
           "release_rule": "D-101/D-122: per-post offset statistics are reported; per-year values are not",
           "warm_months": list(WARM), "first_year": FIRST_YEAR,
           "table": json.loads(t.to_json(orient="records"))}
    (OUT / "caspcom_posts.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
    print(t[["post", "name", "lat", "lon", "position_flag", "in_study_domain", "sector", "century_post", "offset_m",
             "offset_minus_delta_m", "warm_minus_delta_m", "cold_minus_delta_m", "n_months_warm"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
