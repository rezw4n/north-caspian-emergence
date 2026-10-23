"""Metrics of the second-round revision analysis A (PREREG Amendment 19 items 7, 8 and 11;
D-147).

Loaded by tools/build_metrics.revision_metrics, which calls add(m). Every key starts with
"rev2.a."; each record is {"value", "unit", "source", "note"?, "decimals"?} as _put() in
tools/metrics_rev_r.py. Direct reads of results/revision2/a/ (scripts/rev2_a_tables.py);
nothing is recomputed here except counts over those files. A missing input adds no key.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "results" / "revision2" / "a"
REL = "results/revision2/a"
PK = {"gebco_2026": "gebco2026", "gebco_2022": "gebco2022", "etopo_2022": "etopo2022",
      "srtm15p_v2.7": "srtm15p", "etopo1": "etopo1"}
SK = {"west": "west", "volga": "volga", "interfluve": "interfluve", "ural": "ural", "northeast": "northeast",
      "komsomolets": "komsomolets", "open_shelf": "openshelf"}
VALIDATED = ["west", "interfluve", "northeast", "open_shelf"]


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
    m[f"rev2.a.{key}"] = rec


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _yn(v):
    return None if v is None else ("yes" if v else "no")


def _counts(m, prefix, counts: dict, src, note):
    for p, c in (counts or {}).items():
        if p not in PK:
            continue
        _put(m, f"{prefix}.{PK[p]}.n_validated4", c.get("validated4"), "sectors", src, note + "; validated sectors")
        _put(m, f"{prefix}.{PK[p]}.n_all7", c.get("all7"), "sectors", src, note)
        _put(m, f"{prefix}.{PK[p]}.n_six", c.get("six_without_komsomolets"), "sectors", src,
             note + "; the six sectors other than Komsomolets Bay")


def _verdicts(m, prefix, verdicts: dict, src, note):
    for p, d in (verdicts or {}).items():
        if p not in PK:
            continue
        for s in VALIDATED:
            if s in d:
                v = d[s]
                _put(m, f"{prefix}.{PK[p]}.{SK[s]}", "na" if v is None else _yn(v), "", src,
                     note + "; mispredicts (yes/no), na = not evaluable")


def replica(m):
    d = _json(A / "replica.json")
    if not d:
        return
    src = f"{REL}/replica.json"
    _put(m, "replica.exact", _yn(d.get("exact")), "", src,
         "the pass's shape and total variants and floor reproduce results/hindcast/main exactly")
    for cfg in ("shape", "total0"):
        v = (d.get(cfg) or {}).get("levels_median_max_abs_diff_m")
        _put(m, f"replica.{cfg}.levels_max_abs_diff_m", v[0] if isinstance(v, list) else v, "m", src,
             "must be 0", decimals=6)
    v = (d.get("floor") or {}).get("p95_max_abs_diff_m")
    _put(m, "replica.floor_max_abs_diff_m", v[0] if isinstance(v, list) else v, "m", src, "must be 0", decimals=6)


def item7(m):
    d = _json(A / "item7.json")
    if not d:
        return
    src = f"{REL}/item7.json"
    cr = d.get("c_ref_m") or {}
    _put(m, "i7.cref_atl08_m", cr.get("total_cref"), "m", src, "c_ref applied in item 7(a) (ATL08, D-147)", decimals=3)
    _put(m, "i7.cref_atl06_m", cr.get("total_cref06"), "m", src, "sensitivity (ATL06)", decimals=3)
    _put(m, "i7.swot_m", cr.get("total_swot"), "m", src, "sensitivity (SWOT check)", decimals=3)
    notes = {"total_cref": "total variant with c_ref (A19 item 7a)",
             "total_cref06": "total variant with the ATL06 constant (sensitivity)",
             "total_swot": "total variant with the SWOT constant (sensitivity)",
             "nogeoid": "shape variant without the EGM2008 - EIGEN-6C4 field, constant refitted (A19 item 7b)"}
    for cfg, note in notes.items():
        _counts(m, f"i7.{cfg}", (d.get("counts") or {}).get(cfg), src, note)
        _verdicts(m, f"i7.{cfg}.verdict", (d.get("verdicts") or {}).get(cfg), src, note)
    for cfg in ("total_cref", "total_cref06", "total_swot"):
        ch = (d.get("changed") or {}).get(f"{cfg}_vs_registered_total")
        if ch is not None:
            _put(m, f"i7.{cfg}.n_changed_vs_total", len(ch), "product-sectors", src, "against the registered total variant")
    ch = (d.get("changed") or {}).get("nogeoid_vs_shape")
    if ch is not None:
        _put(m, "i7.nogeoid.n_changed_vs_shape", len(ch), "product-sectors", src, "against the primary shape variant")
    for p, r in (d.get("nogeoid_constants") or {}).items():
        if p not in PK:
            continue
        reg = "registered" if r.get("registered") else "labelled extra, not registered"
        _put(m, f"i7.nogeoid.{PK[p]}.c_m", r.get("c_nogeoid_m"), "m", src, f"refitted area constant ({reg})", decimals=2)
        _put(m, f"i7.nogeoid.{PK[p]}.c_change_m", r.get("c_change_m"), "m", src, "minus the primary c_area", decimals=2)


def item8(m):
    d = _json(A / "item8.json")
    if not d:
        return
    src = f"{REL}/item8.json"
    for p, secs in (d.get("unknown_share_shape") or {}).items():
        if p not in PK:
            continue
        for s, v in secs.items():
            if s not in SK:
                continue
            base = f"i8.share.{PK[p]}.{SK[s]}"
            note = "predicted waterline cells left out because the observed state is unknown (shape variant)"
            _put(m, f"{base}.pooled_pct", v.get("pooled_pct"), "%", src, note + "; pooled over units", decimals=1)
            _put(m, f"{base}.unit_median_pct", v.get("unit_median_pct"), "%", src, note + "; median over units",
                 decimals=1)
            _put(m, f"{base}.unit_q25_pct", v.get("unit_q25_pct"), "%", src, note + "; 25th percentile", decimals=1)
            _put(m, f"{base}.unit_q75_pct", v.get("unit_q75_pct"), "%", src, note + "; 75th percentile", decimals=1)
    for key, c in (d.get("counts") or {}).items():
        if key == "registered_shape":
            continue
        rule, floor = key.split("__floor_")
        tag = f"i8.{rule.replace('shape_', '')}.{floor}"
        note = (f"A19 item 8 ({'i: unknown predicted cells measured' if rule == 'shape_i' else 'ii: observed cells within 2 cells of unknown not measured'}), "
                f"{'registered floor' if floor == 'registered' else 'floor recomputed by the same rule'}")
        _counts(m, tag, c, src, note)
        _verdicts(m, f"{tag}.verdict", (d.get("verdicts") or {}).get(key), src, note)
        ch = (d.get("changed") or {}).get(key)
        if ch is not None:
            _put(m, f"{tag}.n_changed", len(ch), "product-sectors", src, note + "; against the primary shape variant")


def item11(m):
    d = _json(A / "item11.json")
    if not d:
        return
    src = f"{REL}/item11.json"
    for name, r in (d.get("runs") or {}).items():
        if "counts_variant" not in r:
            continue
        key = name.replace("_", "")
        note = f"observation variant {name} through the hindcast (A19 item 11; D-147)"
        if r.get("hybrid"):
            note += "; 14 R-block tiles only"
        _counts(m, f"i11.{key}", r.get("counts_variant"), src, note)
        _counts(m, f"i11.{key}.reference", r.get("counts_reference"), src, f"reference for {name}")
        _verdicts(m, f"i11.{key}.verdict", r.get("verdicts_variant"), src, note)
        if r.get("hybrid"):
            _verdicts(m, f"i11.{key}.reference.verdict", r.get("verdicts_reference"), src, f"reference for {name}")
        _put(m, f"i11.{key}.n_changed", r.get("n_changed"), "product-sectors", src, note)
        _put(m, f"i11.{key}.n_changed_validated", r.get("n_changed_validated"), "product-sectors", src,
             note + "; validated sectors")


def survival(m):
    d = _json(A / "summary.json")
    if not d:
        return
    src = f"{REL}/summary.json"
    for p, secs in (d.get("validated_survival") or {}).items():
        if p not in PK:
            continue
        for s, v in secs.items():
            if s not in SK:
                continue
            _put(m, f"survival.{PK[p]}.{SK[s]}.every_shape_run", _yn(v.get("mispredicted_in_every_shape_based_run")),
                 "", src, "mispredicted in the deciding variant and every A19 item 7b, 8 and 11 recomputation")
            _put(m, f"survival.{PK[p]}.{SK[s]}.every_run", _yn(v.get("mispredicted_in_every_run")), "", src,
                 "also in the registered total variant and the item 7a total variants")


def add(m: dict) -> None:
    replica(m)
    item7(m)
    item8(m)
    item11(m)
    survival(m)
