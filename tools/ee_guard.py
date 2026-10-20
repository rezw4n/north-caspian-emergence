"""Budget guard for Earth Engine batch tasks (D-019).

Polls this project's running tasks every 60 s and cancels any task whose
consumed EECU-seconds exceed its budget. Budgets come from
results/inventory/ee_budgets.json ({"description prefix": max_eecu_s}); tasks
not listed there get DEFAULT_BUDGET. Exits when no task is pending or running.

Usage: python tools/ee_guard.py [default_budget_eecu_s]
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import caspian  # noqa: E402,F401
import ee  # noqa: E402

from caspian.config import EE_PROJECT  # noqa: E402

DEFAULT_BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 3600.0
BUDGETS = ROOT / "results" / "inventory" / "ee_budgets.json"
LOG = ROOT / "results" / "inventory" / "ee_guard_log.jsonl"


def budget_for(desc: str) -> float:
    budgets = json.loads(BUDGETS.read_text(encoding="utf-8")) if BUDGETS.exists() else {}
    matches = [v for k, v in budgets.items() if desc.startswith(k)]
    return float(max(matches)) if matches else DEFAULT_BUDGET


def main() -> int:
    ee.Initialize(project=EE_PROJECT)
    while True:
        active = 0
        for op in ee.data.listOperations():
            md = op["metadata"]
            state, desc = md.get("state"), md.get("description", "")
            if state in ("PENDING", "RUNNING"):
                active += 1
            if state != "RUNNING":
                continue
            used = float(md.get("batchEecuUsageSeconds") or 0)
            limit = budget_for(desc)
            if used > limit:
                ee.data.cancelOperation(op["name"])
                rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "cancelled": desc,
                       "eecu_s": used, "budget_eecu_s": limit}
                with LOG.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
                print("CANCELLED over budget:", rec, flush=True)
        if active == 0:
            print("no active tasks; guard exiting")
            return 0
        time.sleep(60)


if __name__ == "__main__":
    sys.exit(main())
