"""A6: the PREREG section 7 H2 decision from the frozen paired-bootstrap tables
(caspian.validation.h2_decision), per ICESat-2 source and variant, and the combined
verdict of D-069 (supported only if supported on both ATL06 and ATL08; partial if on
one). Applied once, after the evaluate stage, to the tables as written.

Output (committed): results/validation/<tag>/h2_decision.json
Usage: python scripts/a6_h2_decision.py <tag>
"""

import json
import sys

import caspian  # noqa: F401
import pandas as pd

from caspian import validation
from caspian.config import RESULTS

PRODUCTS = ("gebco_2026", "etopo_2022", "srtm15p_v2.7", "gebco_2022", "etopo1")


def main(tag: str) -> int:
    out = {}
    for src in ("ATL06", "ATL08"):
        pb = pd.read_csv(RESULTS / "validation" / tag / f"{src}_paired_bootstrap.csv")
        out[src] = {v: validation.h2_decision(pb[pb["variant"] == v], PRODUCTS) for v in sorted(pb["variant"].unique())}
    dec = {}
    for v in out["ATL06"]:
        st = [out[s][v]["overall"] for s in ("ATL06", "ATL08")]
        dec[v] = ("supported" if all(x == "supported" for x in st) else
                  "partial support" if any(x == "supported" for x in st) else "not supported")
    res = {"tag": tag, "rule": "PREREG section 7; D-069 (both sources, never pooled; 'matched' decides)",
           "decision": dec, "by_source": out}
    (RESULTS / "validation" / tag / "h2_decision.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print("H2:", dec)
    for s in ("ATL06", "ATL08"):
        for v in out[s]:
            sc = out[s][v]["scopes"]
            print(s, v, {k: d["status"] for k, d in sc.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
