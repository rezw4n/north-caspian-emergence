"""Stage-2 Earth Engine cost model.

For each Landsat path/row and Sentinel-2 MGRS tile with usable scenes (levelled,
scene cloud < 60%), measure the area of the scene footprint that lies inside
active zone v6, multiply by the number of scenes, and convert to EECU with the
rates measured in the 0.3-degree micro-pilot (results/inventory/stage2_micropilot.json):
EECU-s per scene per km2 of computed area at each resolution. An overhead factor
(computed area / zone area, from tile granularity) is reported as a range.

Footprint areas are measured with profiled interactive requests on the 250 m zone
asset. Output: results/inventory/stage2_cost_model.json.

Usage: python scripts/a2_cost_model.py
"""

from __future__ import annotations

import io
import json
import re
import sys

import caspian  # noqa: F401
import ee
import pandas as pd

from caspian.config import EE_PROJECT, INTERIM, RESULTS

ZONE = f"projects/{EE_PROJECT}/assets/caspian/active_zone_v6"
MAX_CLOUD = 60
OVERHEAD = (1.0, 1.5, 2.0)   # computed area / zone area: low, central, high
COMMUNITY_EECU_H, CONTRIBUTOR_EECU_H, CAP = 150, 1000, 0.70


def profile_eecu(text: str) -> float:
    return sum(float(m.group(1)) for line in text.splitlines()[1:] if (m := re.match(r"\s*([\d.]+)\s", line)))


def main() -> int:
    ee.Initialize(project=EE_PROJECT)
    lv = pd.read_csv(INTERIM / "levels" / "scene_levels.csv", low_memory=False)
    d = lv[lv["epoch"].isin(["E1", "E2", "E3"]) & (lv["cloud_scene_pct"] < MAX_CLOUD)].copy()
    d["unit"] = d.apply(lambda r: f"T{r['MGRS_TILE']}" if r["sensor"] == "MSI_S2"
                        else f"{int(r['WRS_PATH']):03d}{int(r['WRS_ROW']):03d}", axis=1)
    d["family"] = d["sensor"].where(d["sensor"] == "MSI_S2", "LANDSAT")
    groups = d.groupby(["family", "unit"])
    reps = groups["system:index"].first()
    coll = groups["collection"].first()
    counts = groups.size()
    zone = ee.Image(ZONE).select("zone").eq(1).unmask(0)
    feats = []
    for (fam, unit), sid in reps.items():
        img = ee.Image(f"{coll[(fam, unit)]}/{sid}")
        feats.append(ee.Feature(img.geometry(), {"family": fam, "unit": unit}))
    fc = ee.FeatureCollection(feats)
    px = ee.Image.pixelArea().divide(1e6)
    meas = ee.Image.cat([zone.multiply(px).rename("zone_km2"), px.rename("fp_km2")]).reduceRegions(
        fc, ee.Reducer.sum(), scale=250, tileScale=2)
    buf = io.StringIO()
    with ee.profilePrinting(destination=buf):
        rows = [f["properties"] for f in meas.getInfo()["features"]]
    query_eecu = profile_eecu(buf.getvalue())
    rates = json.loads((RESULTS / "inventory" / "stage2_micropilot.json").read_text(encoding="utf-8"))
    out = {"inputs": {"max_cloud_pct": MAX_CLOUD, "overhead": OVERHEAD, "query_eecu_s": round(query_eecu, 1)},
           "units": [], "rates": {}, "totals": {}}
    # micro-pilot rate per scene per km2 of the 0.3-degree box
    def rate(v):
        st = rates[v]
        box_km2 = st["dims"][0] * st["dims"][1] * st["res_m"] ** 2 / 1e6
        return st["eecu_s"] / (st["n_scenes"] * box_km2)

    # Landsat rates are measured at 30, 60 and 90 m. Sentinel-2 is measured at 60 m
    # (variant s60) and scaled to 30 and 90 m by the Landsat resolution ratios.
    lan = {"30m": rate("a30"), "60m": rate("b60"), "90m": rate("d90")}
    s2_60 = rate("s60")
    out["rates"] = {"LANDSAT": lan, "MSI_S2": {k: v * s2_60 / lan["60m"] for k, v in lan.items()}}
    zone_scene_km2 = {"LANDSAT": 0.0, "MSI_S2": 0.0}
    for r in rows:
        n = int(counts[(r["family"], r["unit"])])
        out["units"].append({**r, "n_scenes": n})
        zone_scene_km2[r["family"]] += r["zone_km2"] * n
    for fam, zkm2 in zone_scene_km2.items():
        for res, r in out["rates"][fam].items():
            for k, f in zip(("low", "central", "high"), OVERHEAD):
                eecu_h = zkm2 * f * r / 3600
                out["totals"].setdefault(fam, {}).setdefault(res, {})[k] = round(eecu_h, 1)
        out["totals"][fam]["zone_scene_km2"] = round(zkm2)
    for tier, q in (("community", COMMUNITY_EECU_H), ("contributor", CONTRIBUTOR_EECU_H)):
        out["inputs"][f"{tier}_budget_eecu_h_per_month"] = q * CAP
    (RESULTS / "inventory" / "stage2_cost_model.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("inputs", "rates", "totals")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
