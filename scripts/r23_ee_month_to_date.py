"""Earth Engine use, month to date, recomputed from every ledger (D-104): the EECU-seconds
of every profiled interactive request in results/inventory/*.jsonl (interactive logs and
the agents' ledgers; ee_guard_log.jsonl is left out, since it records cancelled batch
tasks, which the batch total covers), plus the batch operations created this month from
ee.data.listOperations (batchEecuUsageSeconds; metadata only, no computation). Writes
results/inventory/ee_month_to_date.json.

Usage: python scripts/r23_ee_month_to_date.py [--no-live]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

import caspian  # noqa: F401

from caspian.config import EE_PROJECT, RESULTS

INV = RESULTS / "inventory"
SKIP = {"ee_guard_log.jsonl"}


def ledgers(month: str) -> dict:
    """{file: EECU-hours} over the ledger lines dated in ``month`` (YYYY-MM), or undated
    (the stage-2 logs carry no time; every one of them is from this project's October
    runs, the month the project started)."""
    out = {}
    for f in sorted(INV.glob("*.jsonl")):
        if f.name in SKIP:
            continue
        s = 0.0
        for line in f.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            t = str(r.get("time") or "")
            if t and not t.startswith(month):
                continue
            s += float(r.get("eecu_s") or 0.0)
        out[f.name] = round(s / 3600.0, 3)
    return out


def batch_live(month: str) -> tuple[float, int]:
    import ee
    ee.Initialize(project=EE_PROJECT)
    s, n = 0.0, 0
    for op in ee.data.listOperations():
        md = op.get("metadata", {})
        if str(md.get("createTime", "")).startswith(month):
            s += float(md.get("batchEecuUsageSeconds") or 0.0)
            n += 1
    return s / 3600.0, n


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-live", action="store_true")
    a = ap.parse_args(argv)
    now = datetime.now(timezone.utc)
    month = now.strftime("%Y-%m")
    per = ledgers(month)
    rec = {"time_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "month": month, "interactive_by_ledger_h": per,
           "interactive_h": round(sum(per.values()), 2), "skipped": sorted(SKIP)}
    if not a.no_live:
        b, n = batch_live(month)
        rec["batch_h"], rec["batch_operations"] = round(b, 2), n
    rec["total_h"] = round(rec["interactive_h"] + rec.get("batch_h", 0.0), 2)
    rec["limit_h"] = 640.0
    (INV / "ee_month_to_date.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in rec.items() if k != "interactive_by_ledger_h"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
