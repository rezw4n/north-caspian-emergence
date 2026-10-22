"""A18 item 10(e) and the composition of the common baseline.

(e1) The observed seabed between E3 bin 13 (-28.65 m, the bin that holds L2026) or bin 14
(-28.55 m) and bin 12 (-28.75 m, the lowest audited unit): cells observed at both units,
sea-connected water at the higher and land at the lower. Computed on the observed extents
without the record-low envelope (the envelope pools bins 12-14, so the enveloped extents cannot
differ there) and, for reference, with it.
(e2) The waterline-DEM strip: cells of the fused baseline S_base (sea at L2026) observed as
land at bin 12.
Composition of S_base per product and fused form: known sea at bin 12, known land at bin
12, and unobserved at bin 12 (its state from the waterline DEM or the corrected product,
by the fused surface's source code).

Band states: the uncorrected and shifted 1 m and 2 m bands of each product split by the
observed state at bin 12 (known sea, known land, unobserved), by S_base (registered fused
baseline) and by the zone-v6 2025-26 extent (water, not water, no value), to read the
already-dry shares (item 10(c)).

Composite observed baseline: known sea at bin 12 (enveloped, as
registered) plus the cells unobserved at bin 12 that are water in the zone-v6 2025-26
extent, kept where 8-connected to the open-sea seed; written as a flood-level-like map
(-1000 m on the set, +inf elsewhere) to data/interim/revision_n/composite_baseline.npy.

Outputs: results/revision/n/observed_band_v2.csv, baseline_composition.csv,
band_states.csv; the composite map.

Usage: python scripts/rev_n_baseline.py [--tile 2048]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rev_n_common as C  # noqa: E402

from caspian import exposure as ex  # noqa: E402
from caspian import hindcast_tiled as ht  # noqa: E402
from caspian.config import INTERIM, RESULTS  # noqa: E402


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tile", type=int, default=2048)
    a = ap.parse_args(argv)
    t0 = time.time()
    S = C.setup(tile=a.tile, fused=True)
    Sc = {p: np.load(C.load_script("a8_wdpa_exposure").fused_cache("constant", p), mmap_mode="r") for p in S["names"]}
    a7, st, tr, shape, lab, K, L26 = S["a7"], S["static"], S["transform"], S["shape"], S["lab_in"], S["K"], S["L26"]
    cell = a7.CELL_KM2
    units = a7.stage2_index(INTERIM / "stage2" / "main")
    codes = {}
    for env in (False, True):
        a7.ENVELOPE = st["envelope"] if env else None
        for b in (12, 13, 14):
            raw, _ = a7.observed_raw(a7.groups_for(units, b, ["E3"]), tr, shape)
            codes[(env, b)] = a7.classify(raw, st, a.tile)
            del raw
    a7.ENVELOPE = st["envelope"]
    nR = K + 1
    acc = {}

    def add(key, lb, m):
        acc.setdefault(key, np.zeros(nR, np.int64))
        acc[key] += np.bincount(lb[m], minlength=nR)

    zv = np.load(RESULTS / "study_area" / "extents_v6_250m.npz")
    low = a7.nearest(zv["low"], zv["transform"], tr, shape, 255)
    for r0, r1, c0, c1 in ht.tiles(shape, a.tile):
        o = np.asarray(st["outside"][r0:r1, c0:c1])
        lb = np.asarray(lab[r0:r1, c0:c1]).astype(np.int64)
        km = {k: ex.known_masks(np.asarray(v[r0:r1, c0:c1]), o) for k, v in codes.items()}
        for env in (False, True):
            tag = "enveloped" if env else "no_envelope"
            k12, w12 = km[(env, 12)]
            add(("observed", tag, "known12"), lb, k12)
            add(("observed", tag, "sea12"), lb, k12 & w12)
            for b in (13, 14):
                kb, wb = km[(env, b)]
                add(("observed", tag, f"known{b}"), lb, kb)
                add(("observed", tag, f"sea{b}"), lb, kb & wb)
                add(("observed", tag, f"both_12_{b}"), lb, k12 & kb)
                add(("observed", tag, f"band_{b}_12"), lb, k12 & kb & wb & ~w12)
                add(("observed", tag, f"band_reverse_12_{b}"), lb, k12 & kb & w12 & ~wb)
        k12, w12 = km[(True, 12)]
        z6 = np.asarray(low[r0:r1, c0:c1])
        st12 = {"sea12": k12 & w12, "land12": k12 & ~w12, "unobserved12": ~k12}
        sz = {"zv6_water": z6 == 1, "zv6_not_water": z6 == 0, "zv6_no_value": z6 == 255}
        for p in S["names"]:
            f = np.asarray(S["F"][p][r0:r1, c0:c1], np.float64)
            sb_ = np.asarray(S["Ffused"][p][r0:r1, c0:c1], np.float64) < L26
            for surf, top in (("uncorrected", L26), ("shifted", L26 + S["c"][p])):
                for D in (1.0, 2.0):
                    band = (f < top) & ~(f < top - D)
                    for s1, m1 in st12.items():
                        for s2, m2 in (("in_baseline", sb_), ("outside_baseline", ~sb_)):
                            for s3, m3 in sz.items():
                                add((p, f"{surf}|{D:g}", f"{s1}|{s2}|{s3}"), lb, band & m1 & m2 & m3)
        for p in S["names"]:
            for form, FF in (("linear_hold", S["Ffused"][p]), ("constant", Sc[p])):
                sb = np.asarray(FF[r0:r1, c0:c1], np.float64) < L26
                add((p, form, "baseline"), lb, sb)
                add((p, form, "baseline_known_sea12"), lb, sb & k12 & w12)
                add((p, form, "baseline_known_land12_e2"), lb, sb & k12 & ~w12)
                add((p, form, "baseline_unobserved12"), lb, sb & ~k12)
                add((p, form, "unobserved12_not_baseline"), lb, ~sb & ~k12)
    # composite observed baseline
    comp_in = np.zeros(shape, bool)
    for r0, r1, c0, c1 in ht.tiles(shape, a.tile):
        o = np.asarray(st["outside"][r0:r1, c0:c1])
        k12, w12 = ex.known_masks(np.asarray(codes[(True, 12)][r0:r1, c0:c1]), o)
        comp_in[r0:r1, c0:c1] = (k12 & w12) | (~k12 & (np.asarray(low[r0:r1, c0:c1]) == 1))
    comp = ht.component_any(comp_in, st["sea_seed"], tile=a.tile)
    del comp_in
    C.WORK.mkdir(parents=True, exist_ok=True)
    path = C.WORK / "composite_baseline.npy"
    tmp = C.WORK / "composite_baseline.tmp.npy"
    mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=tuple(shape))
    for r0, r1, c0, c1 in ht.tiles(shape, a.tile):
        mm[r0:r1, c0:c1] = np.where(comp[r0:r1, c0:c1], np.float32(-1000.0), np.float32(np.inf))
    mm.flush()
    del mm
    tmp.replace(path)
    for r0, r1, c0, c1 in ht.tiles(shape, a.tile):
        lb = np.asarray(lab[r0:r1, c0:c1]).astype(np.int64)
        o = np.asarray(st["outside"][r0:r1, c0:c1])
        k12, w12 = ex.known_masks(np.asarray(codes[(True, 12)][r0:r1, c0:c1]), o)
        cc = comp[r0:r1, c0:c1]
        add(("composite", "composite", "baseline"), lb, cc)
        add(("composite", "composite", "baseline_known_sea12"), lb, cc & k12 & w12)
        add(("composite", "composite", "baseline_unobserved12"), lb, cc & ~k12)
    del comp
    rows = []
    for (who, form, q), v in acc.items():
        rr = C.to_regions(v[1:])
        rows += [{"product": who, "variant": form, "quantity": q, "region": C.REGIONS[i], "km2": float(rr[i] * cell)}
                 for i in range(len(C.REGIONS))]
    t = pd.DataFrame(rows)
    C.OUT.mkdir(parents=True, exist_ok=True)
    t[t["product"] == "observed"].to_csv(C.OUT / "observed_band_v2.csv", index=False)
    bs = t[t["variant"].str.contains("|", regex=False)]
    t = t[~t["variant"].str.contains("|", regex=False)]
    t[t["product"] != "observed"].to_csv(C.OUT / "baseline_composition.csv", index=False)
    bs = bs.assign(surface=bs["variant"].str.split("|").str[0], delta_m=bs["variant"].str.split("|").str[1].astype(float),
                   state_bin12=bs["quantity"].str.split("|").str[0], baseline=bs["quantity"].str.split("|").str[1],
                   zone_v6=bs["quantity"].str.split("|").str[2]).drop(columns=["variant", "quantity"])
    bs.to_csv(C.OUT / "band_states.csv", index=False)
    C.log("done", round(time.time() - t0, 1), "s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
