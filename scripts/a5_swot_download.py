"""A5 input: SWOT L2 HR Raster (250 m) water-surface elevation granules over the
North Caspian from hydroweb.next (collection SWOT_L2_HR_RASTER_250M; latest
processing version only), for calibrating the wind-setup model (brief section 8 A5).

Outputs (git-ignored): data/raw/swot/raster250m/<zip + extracted .nc>
Provenance (committed): results/provenance/swot_raster250m.json
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import caspian  # noqa: F401
import py_hydroweb

from caspian import credentials
from caspian.config import RAW, RESULTS
from caspian.geography import DOMAIN

COLLECTION = "SWOT_L2_HR_RASTER_250M"


def main() -> int:
    key = credentials.get_secret("HYDROWEB_API_KEY")
    if not key:
        print("FAIL HYDROWEB_API_KEY is not set")
        return 1
    out = RAW / "swot" / "raster250m"
    out.mkdir(parents=True, exist_ok=True)
    w, s, e, n = DOMAIN.bounds
    client = py_hydroweb.Client(api_key=key)
    basket = py_hydroweb.DownloadBasket("caspian_swot_raster250m")
    basket.add_collection(COLLECTION, bbox=[w, s, e, n], query={"latest": {"eq": True}})
    cwd = os.getcwd()
    os.chdir(out)
    try:
        zip_name = client.submit_and_download_zip(basket)
    finally:
        os.chdir(cwd)
    zip_path = out / Path(str(zip_name)).name
    with zipfile.ZipFile(zip_path) as z:
        names = [nm for nm in z.namelist() if nm.endswith(".nc")]
        z.extractall(out / "extracted")
    prov = {"collection": COLLECTION, "bbox_wsen": [w, s, e, n], "query": {"latest": True},
            "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "zip": zip_path.name, "zip_bytes": zip_path.stat().st_size,
            "zip_sha256": hashlib.sha256(zip_path.read_bytes()).hexdigest(), "n_netcdf": len(names)}
    (RESULTS / "provenance").mkdir(parents=True, exist_ok=True)
    (RESULTS / "provenance" / "swot_raster250m.json").write_text(json.dumps(prov, indent=1), encoding="utf-8")
    print(json.dumps(prov, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
