"""ATL24 (ICESat-2 near-shore bathymetry) availability over the North Caspian domain
(PREREG section 2, "ATL24 bathymetry where available"; D-091). Search only: NASA CMR
through earthaccess, the domain's bounding box, every ATL24 version CMR lists. Nothing
is downloaded here.

Output: results/inventory/atl24_search.json (granule count, date range and versions per
collection; the granule names).

Usage: python scripts/a6_atl24_search.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

import caspian  # noqa: F401

from caspian.config import RESULTS
from caspian.geography import DOMAIN

TEMPORAL = ("2018-10-01", "2026-12-31")


def _begin(g) -> str | None:
    rng = g["umm"].get("TemporalExtent", {}).get("RangeDateTime", {})
    return rng.get("BeginningDateTime")


def main() -> int:
    import earthaccess

    from caspian import earthdata

    earthdata.login()
    w, s, e, n = DOMAIN.bounds
    cols = earthaccess.search_datasets(short_name="ATL24")
    rec = {"created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "search_bbox_wsen": [w, s, e, n], "temporal": list(TEMPORAL),
           "note": "bounding-box search; granules intersect the box, not necessarily the domain polygon",
           "collections": []}
    for c in cols:
        umm = c["umm"]
        ver = umm.get("Version")
        res = earthaccess.search_data(short_name="ATL24", version=ver, bounding_box=(w, s, e, n),
                                      temporal=TEMPORAL, count=-1)
        begins = sorted(b for b in (_begin(g) for g in res) if b)
        rec["collections"].append({
            "short_name": umm.get("ShortName"), "version": ver, "concept_id": c["meta"].get("concept-id"),
            "doi": (umm.get("DOI") or {}).get("DOI"), "n_granules": len(res),
            "first_begin": begins[0] if begins else None, "last_begin": begins[-1] if begins else None,
            "granules": sorted(g["umm"].get("GranuleUR") for g in res)})
        print(f"ATL24 v{ver}: {len(res)} granules, {begins[0] if begins else '-'} .. "
              f"{begins[-1] if begins else '-'}", flush=True)
    out = RESULTS / "inventory" / "atl24_search.json"
    out.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
