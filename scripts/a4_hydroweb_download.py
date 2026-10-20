"""A4 cross-check and gap-fill source: Hydroweb operational lake water level for the
Caspian Sea (hydroweb.next, collection HYDROWEB_LAKES_OPE, item L_caspian), via the
official py_hydroweb client, authenticated with HYDROWEB_API_KEY (Windows user env
or process env; never printed).

Outputs (git-ignored): data/raw/hydroweb/<zip and extracted files>
Provenance (committed): results/provenance/hydroweb_l_caspian.json
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import zipfile
from datetime import datetime, timezone

import caspian  # noqa: F401
import py_hydroweb

from caspian import credentials
from caspian.config import RAW, RESULTS

COLLECTION = "HYDROWEB_LAKES_OPE"
ITEM_REF = "HYDROWEB_LAKES_OPE.L_caspian"


def main() -> int:
    key = credentials.get_secret("HYDROWEB_API_KEY")
    if not key:
        print("FAIL HYDROWEB_API_KEY is not set")
        return 1
    out_dir = RAW / "hydroweb"
    out_dir.mkdir(parents=True, exist_ok=True)
    client = py_hydroweb.Client(api_key=key)
    basket = py_hydroweb.DownloadBasket("caspian_l_caspian")
    basket.add_collection(COLLECTION, intersects_refs=[ITEM_REF])
    cwd = os.getcwd()
    os.chdir(out_dir)  # the client writes the zip to the current folder
    try:
        zip_name = client.submit_and_download_zip(basket)
    finally:
        os.chdir(cwd)
    zip_path = out_dir / zip_name if not os.path.isabs(str(zip_name)) else zip_name
    zip_path = type(out_dir)(zip_path)
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        z.extractall(out_dir / "extracted")
    prov = {"collection": COLLECTION, "item_ref": ITEM_REF,
            "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "zip": zip_path.name, "zip_sha256": hashlib.sha256(zip_path.read_bytes()).hexdigest(),
            "files": names, "client": f"py_hydroweb {py_hydroweb.__version__ if hasattr(py_hydroweb, '__version__') else ''}"}
    (RESULTS / "provenance").mkdir(parents=True, exist_ok=True)
    (RESULTS / "provenance" / "hydroweb_l_caspian.json").write_text(json.dumps(prov, indent=1), encoding="utf-8")
    print(json.dumps(prov, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
