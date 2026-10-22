"""Metrics of the revised ERL figures 1, 3 and 4. Loaded by
tools/build_metrics.revision_metrics, which calls
add(m). Every key starts with ``rev.fig.``; each record is {"value", "unit", "source",
"note"?, "decimals"?}, as put() in tools/build_metrics.py. A missing input file adds
nothing. Years are stored as text so that the macros print them without a thousands
separator.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
DQ_FLOOR = 0.01          # bottom of the D/Q axis of figure 3(c) (figures/erl_fig_audit.py)


def _put(m: dict, key: str, value, unit: str, source: str, note: str | None = None, decimals=None) -> None:
    if value is None:
        return
    if isinstance(value, float) and not math.isfinite(value):
        return
    if hasattr(value, "item"):
        value = value.item()
    rec = {"value": value, "unit": unit, "source": source}
    if note:
        rec["note"] = note
    if decimals is not None:
        rec["decimals"] = decimals
    m[f"rev.fig.{key}"] = rec


def fig1(m: dict) -> None:
    """Figure 1(b): the observed level-binned waterlines (scripts/f3_erl_waterlines.py)."""
    p = RES / "display" / "erl_waterlines.json"
    if not p.exists():
        return
    src = "results/display/erl_waterlines.json"
    w = json.loads(p.read_text(encoding="utf-8"))
    bins = w["bins"]
    _put(m, "f1.n_levels", len(bins), "levels", src, "E3 level bins drawn as observed shorelines (D-140)")
    for i, b in enumerate(bins, start=1):
        k = f"f1.level{i}"
        _put(m, f"{k}.level_m", float(b["level_m"]), "m (DAHITI)", src, f"bin {b['bin']} centre", decimals=2)
        _put(m, f"{k}.first_year", str(b["first_year"]), "year", src, "first Landsat acquisition of the bin")
        _put(m, f"{k}.last_year", str(b["last_year"]), "year", src, "last Landsat acquisition of the bin")
        _put(m, f"{k}.n_acquisitions", int(b["n_acquisitions"]), "acquisitions", src)
    steps = {round(bins[i]["level_m"] - bins[i + 1]["level_m"], 4) for i in range(len(bins) - 1)}
    if len(steps) == 1:
        _put(m, "f1.level_step_m", float(steps.pop()), "m", src, "spacing of the drawn levels", decimals=1)


def fig3(m: dict) -> None:
    """Figure 3(c): per-level D/Q (h1_mispredicts_detail.csv; revision analysis S's item2_dq.csv)."""
    dq = RES / "revision" / "s" / "item2_dq.csv"
    det = RES / "hindcast" / "main" / "h1_mispredicts_detail.csv"
    if not (dq.exists() and det.exists()):
        return
    t = pd.read_csv(dq)
    t = t[t["variant"] == "shape"]
    src = "results/revision/s/item2_dq.csv"
    _put(m, "f3.dq_n_levels_min", int(t["dq_n"].min()), "levels", src, "evaluable levels per product and sector, min")
    _put(m, "f3.dq_n_levels_max", int(t["dq_n"].max()), "levels", src, "evaluable levels per product and sector, max")
    d = pd.read_csv(det)
    d = d[(d["variant"] == "shape") & (d["evaluable"] == True)]  # noqa: E712
    r = d["median"] / d["p95_noise"]
    src2 = "results/hindcast/main/h1_mispredicts_detail.csv"
    _put(m, "f3.dq_n_points", int(r.size), "product-sector-levels", src2, "dots in figure 3(c)")
    _put(m, "f3.dq_n_below_axis", int((r < DQ_FLOOR).sum()), "product-sector-levels", src2,
         f"per-level D/Q below {DQ_FLOOR}, drawn on the axis of figure 3(c)")
    _put(m, "f3.dq_axis_min", DQ_FLOOR, "ratio", "figures/erl_fig_audit.py", "bottom of the D/Q axis", decimals=2)


def fig5(m: dict) -> None:
    """Figure 5: where the first metre emerges (scripts/f4_erl_nearterm_map.py), and the
    already-dry shares of revision analysis N in percent (results/revision/n/already_dry.csv)."""
    p = RES / "display" / "erl_nearterm_map.json"
    if p.exists():
        src = "results/display/erl_nearterm_map.json"
        r = json.loads(p.read_text(encoding="utf-8"))
        for surf, s in r["surfaces"].items():
            for k in ("area_km2_any", "area_km2_majority", "area_km2_all"):
                _put(m, f"f5.{surf}.{k}", float(s[k]), "km2", src,
                     f"{surf} surface, first metre; any grid / three or more of five / all five")
        _put(m, "f5.n_checks_equal", sum(bool(c["equal"]) for c in r["checks"]), "checks", src,
             "per-grid band areas equal to workstream N's curves")
    d = RES / "revision" / "n" / "already_dry.csv"
    if d.exists():
        t = pd.read_csv(d)
        t = t[(t["region"] == "domain") & (t["delta_m"] == 1.0)]
        src = "results/revision/n/already_dry.csv"
        for surf in ("uncorrected", "shifted"):
            v = 100.0 * t.loc[t["surface"] == surf, "share_dry_zv6"]
            dp = 0 if surf == "uncorrected" else 1
            if len(v):
                _put(m, f"f5.dry_{surf}_pct_min", round(float(v.min()), 1), "%", src,
                     "share of a grid's 1 m band dry in the 2025-26 extent, min over grids", decimals=dp)
                _put(m, f"f5.dry_{surf}_pct_max", round(float(v.max()), 1), "%", src,
                     "share of a grid's 1 m band dry in the 2025-26 extent, max over grids", decimals=dp)


def add(m: dict) -> None:
    fig1(m)
    fig3(m)
    fig5(m)
