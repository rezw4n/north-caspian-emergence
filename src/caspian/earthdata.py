"""NASA Earthdata access via earthaccess, authenticated with EARTHDATA_TOKEN.

The token is passed to earthaccess through this process's environment only; it
is never printed or written to disk.
"""

from __future__ import annotations

import os

from . import credentials
from .config import NC_BBOX


def login():
    """Log in to Earthdata with the token and return the earthaccess Auth object."""
    import earthaccess

    token = credentials.earthdata_token()
    if not token:
        raise RuntimeError("EARTHDATA_TOKEN is not set (process env or Windows user env)")
    os.environ["EARTHDATA_TOKEN"] = token
    auth = earthaccess.login(strategy="environment", persist=False)
    if not getattr(auth, "authenticated", False):
        raise RuntimeError("Earthdata login with EARTHDATA_TOKEN failed")
    return auth


def search_icesat2(short_name: str = "ATL06", temporal=("2023-07-01", "2023-08-31"),
                   bbox=NC_BBOX, count: int = 1):
    """Search CMR for ICESat-2 granules over the North Caspian (no download)."""
    import earthaccess

    return earthaccess.search_data(
        short_name=short_name,
        bounding_box=tuple(bbox),
        temporal=temporal,
        count=count,
    )


def granule_summary(granule) -> dict:
    umm = granule["umm"]
    links = [u["URL"] for u in umm.get("RelatedUrls", []) if u.get("Type") == "GET DATA"]
    return {
        "granule_ur": umm.get("GranuleUR"),
        "begin": umm.get("TemporalExtent", {}).get("RangeDateTime", {}).get("BeginningDateTime"),
        "size_mb": granule.size() if hasattr(granule, "size") else None,
        "data_links": links[:2],
    }
