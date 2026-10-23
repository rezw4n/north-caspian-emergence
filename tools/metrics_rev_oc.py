"""Metrics of PREREG Amendment 19 items 5 and 9 (D-148): the operating characteristic of the H1
decision rule and the systematic-budget Monte Carlo.

Loaded by tools/build_metrics.revision_metrics, which calls add(m). Every key starts with
"rev2.oc."; each record is {"value", "unit", "source", "note"?} as put() in tools/build_metrics.py,
with "decimals" where the default would mislead. Direct reads of results/revision2/oc/
(scripts/rev2_oc_item5.py, scripts/rev2_oc_item9.py); nothing is recomputed here. A missing input
adds no key.

Keys:
  rev2.oc.i5.range_km, rev2.oc.i5.R.<class>.s<sigma cm>         design and realisations run
  rev2.oc.i5.<class>.c_sigma0_m                                    the control's constant at sigma 0
  rev2.oc.i5.<class>.<sector>.flag_sigma0                          "yes"/"no", the unperturbed control
  rev2.oc.i5.<class>.<sector>.s<sigma cm>.p_flag_pct (+ n_flagged, ci95_lo_pct, ci95_hi_pct)
  rev2.oc.i5.<class>.<sector>.s<sigma cm>.median_D_km
  rev2.oc.i5.<class>.<sector>.first_sigma_m (or .first_sigma "none up to 1.00 m")
  rev2.oc.i9.B, rev2.oc.i9.sigma.<sector>.<epoch>_m, rev2.oc.i9.check.*
  rev2.oc.i9.<grid>.<sector>.p_mispredict_pct (+ n_flagged, ci95_lo_pct, ci95_hi_pct, primary)
with class c15s (15" control) or c1m (1' control), sigma in cm (s010, s025, s050, s100).
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
O = ROOT / "results" / "revision2" / "oc"
REL = "results/revision2/oc"
PK = {"gebco_2026": "gebco2026", "gebco_2022": "gebco2022", "etopo_2022": "etopo2022",
      "srtm15p_v2.7": "srtm15p", "etopo1": "etopo1"}
SK = {"west": "west", "interfluve": "interfluve", "northeast": "northeast", "open_shelf": "openshelf"}
CK = {"15s": "c15s", "1m": "c1m"}


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
    m[f"rev2.oc.{key}"] = rec


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _csv(path: Path):
    if not path.exists():
        return None
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _scode(s) -> str:
    return f"s{int(round(float(s) * 100)):03d}"


def item5(m):
    summ = _json(O / "item5_summary.json")
    pf = _csv(O / "item5_pflag.csv")
    if not summ or not pf:
        return
    src_s, src_p = f"{REL}/item5_summary.json", f"{REL}/item5_pflag.csv"
    note = ("A19 item 5 (D-148): the D-116 rounded control of the resolution class plus a correlated Gaussian "
            "vertical error field (exponential, 150 km range), own D-070 constant, H1 rule with the primary floor "
            "in the validated sectors")
    _put(m, "i5.range_km", summ["range_m"] / 1000.0, "km", src_s, "practical range of the error field (D-125)")
    for k, v in summ["R_per_class_sigma"].items():
        cls, s = k.split("|")
        _put(m, f"i5.R.{CK[cls]}.{_scode(s)}", int(v), "realisations", src_s, note)
    for cls, v in (summ.get("sigma0") or {}).items():
        _put(m, f"i5.{CK[cls]}.c_sigma0_m", v["c_m"], "m", src_s, "the unperturbed control's area constant", decimals=2)
        for i, sec in enumerate(v["sectors"]):
            _put(m, f"i5.{CK[cls]}.{SK[sec]}.flag_sigma0", "yes" if v["flag"][i] else "no", "", src_s,
                 "the H1 rule on the unperturbed control (sigma = 0)")
    for r in pf:
        s = _f(r["sigma_m"])
        if not s:
            continue
        key = f"i5.{CK[r['class']]}.{SK[r['sector']]}.{_scode(s)}"
        _put(m, f"{key}.p_flag_pct", 100 * float(r["p_flag"]), "%", src_p, note, decimals=0)
        _put(m, f"{key}.n_flagged", int(r["n_flagged"]), "realisations", src_p, f"of {r['R']}")
        _put(m, f"{key}.ci95_lo_pct", _f(r["ci95_lo"]) * 100 if _f(r["ci95_lo"]) is not None else None, "%", src_p,
             "Clopper-Pearson 95% interval", decimals=0)
        _put(m, f"{key}.ci95_hi_pct", _f(r["ci95_hi"]) * 100 if _f(r["ci95_hi"]) is not None else None, "%", src_p,
             "Clopper-Pearson 95% interval", decimals=0)
        d = _f(r["median_D_m"])
        _put(m, f"{key}.median_D_km", d / 1000 if d is not None else None, "km", src_p,
             "median over realisations of the sector's median level displacement", decimals=2)
    for k, v in (summ.get("first_sigma_p_above_0.5") or {}).items():
        cls, sec = k.split("|")
        if v is None:
            _put(m, f"i5.{CK[cls]}.{SK[sec]}.first_sigma", "none up to 1.00 m", "", src_s,
                 "P(flag) never exceeds 0.5 on the sigma grid")
        else:
            _put(m, f"i5.{CK[cls]}.{SK[sec]}.first_sigma_m", float(v), "m", src_s,
                 "smallest sigma of the grid with P(flag) > 0.5 (0: the unperturbed control is flagged)", decimals=2)


def item9(m):
    summ = _json(O / "item9_summary.json")
    pm = _csv(O / "item9_pmispredict.csv")
    if not summ or not pm:
        return
    src_s, src_p = f"{REL}/item9_summary.json", f"{REL}/item9_pmispredict.csv"
    note = ("A19 item 9 (D-148): 500 draws of sector-epoch offsets from the A18 item 7 budget, flood maps "
            "thresholded at L + c + delta, Q unchanged")
    _put(m, "i9.B", int(summ["B"]), "draws", src_s, note)
    _put(m, "i9.check.n_levels", int(summ["check_n_rows"]), "product-sector-levels", src_s,
         "delta = 0 against the primary pooled rows")
    _put(m, "i9.check.n_not_identical", int(summ["check_n_not_identical"]), "product-sector-levels", src_s,
         "must be 0")
    _put(m, "i9.check.verdicts_reproduced", "yes" if summ["check_verdicts_reproduced"] else "no", "", src_s)
    for r in summ.get("sigma") or []:
        _put(m, f"i9.sigma.{SK[r['sector']]}.{r['epoch']}_m", r["sigma_budget_m"], "m", f"{REL}/item9_sigma.csv",
             "sector-epoch median of the A18 item 7 budget sigma", decimals=3)
    for r in pm:
        key = f"i9.{PK[r['product']]}.{SK[r['sector']]}"
        _put(m, f"{key}.p_mispredict_pct", 100 * float(r["p_mispredict"]), "%", src_p, note, decimals=1)
        _put(m, f"{key}.n_flagged", int(r["n_flagged"]), "draws", src_p, f"of {r['B']}")
        _put(m, f"{key}.ci95_lo_pct", 100 * float(r["ci95_lo"]), "%", src_p, "Clopper-Pearson 95% interval", decimals=1)
        _put(m, f"{key}.ci95_hi_pct", 100 * float(r["ci95_hi"]), "%", src_p, "Clopper-Pearson 95% interval", decimals=1)
        _put(m, f"{key}.primary", "yes" if str(r["primary_verdict"]).lower() == "true" else "no", "", src_p,
             "the registered verdict (delta = 0)")


def add(m):
    item5(m)
    item9(m)
