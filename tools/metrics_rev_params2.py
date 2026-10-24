"""Design constants of the RSE text that had no metric key (D-154).

Loaded by tools/build_metrics.revision_metrics, which calls add(m). Each record is
{"value", "unit", "source", "note"?, "decimals"?} as _put in tools/metrics_rev_r.py, except
that keys are written in full (they belong to several families). Every value is read from
the code or result file that applies it, never typed here:

- params.checked_tol_m: the checked-sector tolerance of PREREG Amendment 18 item 1, from
  scripts/rev2_d_obsbias.py (LIMIT_M), checked against the text of docs/PREREG.md;
- params.iqr_lo_pct, params.iqr_hi_pct: the box and bar percentiles drawn by
  figures/rse_fig_dq.py and figures/rse_fig_area.py (their np.percentile calls);
- params.rse_fig_method_scale_km: the scale bars of figures/rse_fig_method.py (scale_bar calls);
- rev2.g.lakebed.survey.<arm>.sounded_top_ngvd29_m: the top of the USGS soundings in the
  Great Salt Lake lake-bed model, from results/revision2/g/gsl_lakebed.json, checked against
  SOUNDED_TOP_FT of scripts/rev2_g_lakebed.py;
- rev2.d.const.band.<band>.lo_m, .hi_m: the edges of the half-metre level bands of A19 item 6a,
  from results/revision2/d/constants/constants.csv;
- rev.s.checked.n_sectors: the number of checked sectors, from results/revision/s/summary.json.

A missing input, or two sources that disagree, adds no key (the manuscript build then stops
at the unknown key and tools/verify.py VB.5 names it).
"""

from __future__ import annotations

import ast
import csv
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _put(m, key, value, unit, source, note=None, decimals=None):
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
    m[key] = rec


def _tree(rel: str):
    p = ROOT / rel
    return ast.parse(p.read_text(encoding="utf-8")) if p.exists() else None


def _const(rel: str, name: str):
    """A module-level constant (literal) of a Python file, or None."""
    tree = _tree(rel)
    for node in (tree.body if tree else []):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    return None


def _calls(rel: str, func: str) -> list[ast.Call]:
    """Every call of ``func`` (a bare name or an attribute name) in a Python file."""
    tree = _tree(rel)
    out = []
    for node in ast.walk(tree) if tree else []:
        if isinstance(node, ast.Call):
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else None
            if name == func:
                out.append(node)
    return out


def checked_tolerance(m):
    rel = "scripts/rev2_d_obsbias.py"
    v = _const(rel, "LIMIT_M")
    prereg = (ROOT / "docs" / "PREREG.md").read_text(encoding="utf-8")
    a18 = prereg.split("## Amendment 18", 1)[-1]
    t = re.search(r"\*\*1\. Observation-validated sectors\.\*\*.*?within ±(\d+(?:\.\d+)?) m", a18, re.S)
    if v is None or t is None or not math.isclose(float(v), float(t.group(1))):
        return
    _put(m, "params.checked_tol_m", float(v), "m", f"{rel} LIMIT_M = docs/PREREG.md Amendment 18 item 1",
         "a sector is checked when the waterline model's mean and median errors against ATL08 and ATL06 "
         "lie within +- this value in its test blocks (half the whole-metre step of the grids)", decimals=1)


def box_percentiles(m):
    """The quartile pair of every np.percentile(x, [lo, 50, hi]) call of the two figure scripts."""
    rels = ("figures/rse_fig_dq.py", "figures/rse_fig_area.py")
    pairs = set()
    for rel in rels:
        calls = _calls(rel, "percentile")
        if not calls:
            return
        for c in calls:
            q = ast.literal_eval(c.args[1])
            if len(q) != 3 or q[1] != 50:
                return
            pairs.add((q[0], q[2]))
    if len(pairs) != 1:
        return
    lo, hi = pairs.pop()
    src = " and ".join(rels) + " (np.percentile(..., [lo, 50, hi]): boxes and bars)"
    _put(m, "params.iqr_lo_pct", lo, "percentile", src, "lower edge of the boxes (rse_fig_dq) and bars (rse_fig_area)")
    _put(m, "params.iqr_hi_pct", hi, "percentile", src, "upper edge of the boxes (rse_fig_dq) and bars (rse_fig_area)")


def method_scale_bar(m):
    rel = "figures/rse_fig_method.py"
    vals = set()
    for c in _calls(rel, "scale_bar"):
        length_m, text = ast.literal_eval(c.args[3]), ast.literal_eval(c.args[4])
        km = length_m / 1000.0
        if text != f"{km:g} km":
            return
        vals.add(km)
    if len(vals) != 1:
        return
    km = vals.pop()
    _put(m, "params.rse_fig_method_scale_km", int(km) if km.is_integer() else km, "km",
         f"{rel} (every scale_bar call: length_m and label)")


def gsl_sounding_limit(m):
    rel = "results/revision2/g/gsl_lakebed.json"
    p = ROOT / rel
    if not p.exists():
        return
    j = json.loads(p.read_text(encoding="utf-8"))
    ft = _const("scripts/rev2_g_lakebed.py", "SOUNDED_TOP_FT") or {}
    ft_m = _const("src/caspian/lakebed.py", "FT_M")
    for arm, b in j["boundaries_navd88_m"].items():
        v = b["sounded_top_navd88_m"] - j["gauge_delta_m"][arm]
        if arm not in ft or ft_m is None or not math.isclose(v, ft[arm] * ft_m, abs_tol=1e-3):
            continue
        _put(m, f"rev2.g.lakebed.survey.{arm}.sounded_top_ngvd29_m", round(v, 4), "m (NGVD 29)",
             f"{rel} (sounded_top_navd88_m - gauge_delta_m)",
             f"top of the USGS sonar soundings in the lake-bed model, {ft[arm]:,.0f} ft NGVD 29 "
             "(scripts/rev2_g_lakebed.SOUNDED_TOP_FT; Baskin 2005, 2006)", decimals=1)


def band_edges(m):
    rel = "results/revision2/d/constants/constants.csv"
    p = ROOT / rel
    if not p.exists():
        return
    with p.open(encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["kind"] == "band"]

    def k(x: float) -> str:
        return ("m" if x < 0 else "") + f"{abs(x):.1f}".replace(".", "p")

    seen = {}
    for r in rows:
        lo, hi = (float(x) for x in re.findall(r"-?\d+\.\d+", r["subset"]))
        seen[(lo, hi)] = r["subset"]
    for (lo, hi), label in sorted(seen.items()):
        key = f"rev2.d.const.band.{k(lo)}_{k(hi)}"
        note = f"A19 item 6a: half-metre level band {label} on the working datum (lower edge open, upper closed)"
        _put(m, f"{key}.lo_m", lo, "m", rel, note, decimals=1)
        _put(m, f"{key}.hi_m", hi, "m", rel, note, decimals=1)


def checked_count(m):
    rel = "results/revision/s/summary.json"
    p = ROOT / rel
    if not p.exists():
        return
    v = json.loads(p.read_text(encoding="utf-8")).get("validated_sectors")
    if v:
        _put(m, "rev.s.checked.n_sectors", len(v), "sectors", f"{rel} (validated_sectors)",
             "sectors that pass the checked-sector criterion of PREREG Amendment 18 item 1: " + ", ".join(v))


def add(m: dict) -> None:
    checked_tolerance(m)
    box_percentiles(m)
    method_scale_bar(m)
    gsl_sounding_limit(m)
    band_edges(m)
    checked_count(m)
