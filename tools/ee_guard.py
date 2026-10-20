"""Budget guard for Earth Engine batch tasks (D-019, D-026).

Polls this project's running tasks every POLL_S seconds and cancels any task whose
consumed EECU-seconds exceed its budget, or would exceed it before the next poll at
its current burn rate. The predictive rule exists because a stage-2 pilot burned
about 1,200 EECU-s per minute and overshot a 3,600 EECU-s budget by 2x between two
60 s polls (D-026). Budgets come from results/inventory/ee_budgets.json
({"description prefix": max_eecu_s}); tasks not listed there get DEFAULT_BUDGET.
Exits when no task is pending or running.

Usage: python tools/ee_guard.py [default_budget_eecu_s] [poll_seconds]
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
POLL_S = float(sys.argv[2]) if len(sys.argv) > 2 else 15.0
BUDGETS = ROOT / "results" / "inventory" / "ee_budgets.json"
LOG = ROOT / "results" / "inventory" / "ee_guard_log.jsonl"


def budget_for(desc: str) -> float:
    budgets = json.loads(BUDGETS.read_text(encoding="utf-8")) if BUDGETS.exists() else {}
    matches = [v for k, v in budgets.items() if desc.startswith(k)]
    return float(max(matches)) if matches else DEFAULT_BUDGET


def should_cancel(used: float, rate: float, limit: float, poll_s: float) -> bool:
    """Cancel when the budget is spent, or would be spent before the next poll."""
    return used > limit or used + max(rate, 0.0) * poll_s > limit


def main() -> int:
    ee.Initialize(project=EE_PROJECT)
    last = {}  # operation name -> (monotonic time, eecu_s)
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
            now = time.monotonic()
            t0, u0 = last.get(op["name"], (now, used))
            rate = (used - u0) / (now - t0) if now > t0 else 0.0
            last[op["name"]] = (now, used)
            limit = budget_for(desc)
            if should_cancel(used, rate, limit, POLL_S):
                ee.data.cancelOperation(op["name"])
                rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "cancelled": desc,
                       "eecu_s": used, "rate_eecu_s_per_s": round(rate, 2), "budget_eecu_s": limit}
                with LOG.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
                print("CANCELLED over budget:", rec, flush=True)
        if active == 0:
            print("no active tasks; guard exiting")
            return 0
        time.sleep(POLL_S)


if __name__ == "__main__":
    sys.exit(main())
