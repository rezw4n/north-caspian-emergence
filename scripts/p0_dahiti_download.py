"""Download the DAHITI Caspian Sea water-level series (target 39) through the DAHITI
API v2, authenticated with DAHITI_API_KEY (Windows user env or process env; never
printed). The response is saved unchanged; target metadata are saved alongside.

Outputs (git-ignored): data/raw/dahiti/dahiti_39_caspian_sea_water_level.csv
                       data/raw/dahiti/meta/dahiti_39_target_info.json
Provenance (committed): results/provenance/dahiti_39.json
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone

import caspian  # noqa: F401
import requests

from caspian import credentials
from caspian.config import RAW, RESULTS

API = "https://dahiti.dgfi.tum.de/api/v2"
DAHITI_ID = 39  # Caspian Sea


def main() -> int:
    key = credentials.get_secret("DAHITI_API_KEY")
    if not key:
        print("FAIL DAHITI_API_KEY is not set (setx DAHITI_API_KEY \"...\")")
        return 1
    out_dir = RAW / "dahiti"
    (out_dir / "meta").mkdir(parents=True, exist_ok=True)
    # POST keeps the key out of URLs and server access logs.
    r = requests.post(f"{API}/download-water-level/",
                      data={"api_key": key, "dahiti_id": DAHITI_ID, "format": "csv"}, timeout=300)
    if r.status_code != 200:
        print(f"FAIL download-water-level returned HTTP {r.status_code}: {r.text[:200]}")
        return 1
    series = out_dir / "dahiti_39_caspian_sea_water_level.csv"
    series.write_bytes(r.content)
    info = requests.post(f"{API}/get-target-info/", data={"api_key": key, "dahiti_id": DAHITI_ID},
                         timeout=120)
    meta_path = out_dir / "meta" / "dahiti_39_target_info.json"
    meta_path.write_bytes(info.content if info.status_code == 200 else b"{}")
    prov = {"dahiti_id": DAHITI_ID, "endpoint": f"{API}/download-water-level/", "format": "csv",
            "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "file": series.name, "bytes": len(r.content),
            "sha256": hashlib.sha256(r.content).hexdigest(),
            "target_info_http_status": info.status_code}
    (RESULTS / "provenance").mkdir(parents=True, exist_ok=True)
    (RESULTS / "provenance" / "dahiti_39.json").write_text(json.dumps(prov, indent=1), encoding="utf-8")
    print(json.dumps(prov, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
