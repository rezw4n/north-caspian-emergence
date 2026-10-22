"""A18 item 10(g): vertical-error simulation of the near-term emergence.
Exploratory.

Per product and range (150 km primary, 50 km sensitivity), 200 Gaussian random fields with
exponential covariance (sill = the product's ATL08 test-block NMAD squared; practical
range R), simulated on a 2 km grid by circulant embedding and interpolated bilinearly to
a 240 m flood grid (4 x 4 blocks of the 60 m hindcast grid, for run time). Each field
is added to the block-mean total-variant heights and the flood level is recomputed from
the open-sea seed (caspian.nearterm.minimax_flood). From each realisation:
- E(1 m) and E(2 m), self-referenced, on the uncorrected surface (F) and on the shifted
  product (F - c), and the location version of the common baseline on both;
- for the agreement reference, the bands at 1 m and 2 m of five consecutive realisations
  (1-5, 6-10, ...; 40 disjoint groups) give one all-five / any ratio per group.
The unperturbed 240 m values and the observed five-product agreement at 240 m are computed
the same way.

Outputs (results/revision/n/): sim_realisations.csv, sim_groups.csv, sim_summary.csv,
sim_observed_240m.csv, run_simulate.json.

Usage: python scripts/rev_n_simulate.py [--n 200] [--ranges 150000,50000] [--products ...]
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

from caspian import hindcast_tiled as ht  # noqa: E402
from caspian import nearterm as nt  # noqa: E402
from caspian.config import RESULTS  # noqa: E402
from caspian.geography import SECTOR_ORDER  # noqa: E402

SEED = 20261022
K = 4                      # 240 m flood grid
GRF_DX = 2000.0            # 2 km field grid
GROUP = 5
DELTAS = (1.0, 2.0)


def nmad_table() -> dict:
    a = pd.read_csv(RESULTS / "validation" / "main" / "ATL08_accuracy.csv")
    a = a[a["scope"] == "overall"]
    out = {}
    for m, v in zip(a["model"], a["nmad"]):
        if "|" in m:
            p, var = m.split("|")
            out.setdefault(p, {})[var] = float(v)
    return out


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--ranges", default="150000,50000")
    ap.add_argument("--products", default=None)
    ap.add_argument("--tile", type=int, default=2048)
    ap.add_argument("--suffix", default="", help="appended to every output name (e.g. _range50)")
    ap.add_argument("--block", type=int, default=K, help="flood-grid block factor (4: 240 m; 8: 480 m)")
    a = ap.parse_args(argv)
    k = a.block
    t0 = time.time()
    S = C.setup(tile=a.tile, fused=True, heights=True)
    names = a.products.split(",") if a.products else S["names"]
    L26, cell = S["L26"], S["a7"].CELL_KM2
    tr = S["transform"]
    x0, y0 = float(tr[2]), float(tr[5])
    nm = nmad_table()
    sill = {p: nm[p]["matched"] ** 2 for p in names}
    if any(abs(nm[p]["matched"] - nm[p]["total"]) > 1e-9 for p in names):
        raise SystemExit("ATL08 NMAD differs between the total and matched rows")
    # 240 m grid
    lab = S["lab_in"]
    Wr = nt.block_region_counts(lab, k, S["K"]).astype(np.uint8)
    w_dom = Wr.sum(axis=0).astype(np.float64)
    vidx = [SECTOR_ORDER.index(s) for s in C.VALIDATED]
    w_val = Wr[vidx].sum(axis=0).astype(np.float64)
    n_dom_full = int((np.asarray(lab) > 0).sum())
    crop = {"domain_cells_60m": n_dom_full, "domain_cells_in_blocks": int(w_dom.sum())}
    del Wr
    seed_c = nt.block_all(S["static"]["sea_seed"], k)
    R, Cc = seed_c.shape
    res = float(S["a7"].RES)
    ys = y0 - k * res * (np.arange(R) + 0.5)
    xs = x0 + k * res * (np.arange(Cc) + 0.5)
    ny = int(np.ceil((y0 - ys[-1]) / GRF_DX)) + 2
    nx = int(np.ceil((xs[-1] - x0) / GRF_DX)) + 2
    zc, wb_dom, wb_val, F0 = {}, {}, {}, {}
    for p in names:
        zc[p] = nt.block_mean(S["z"][p], k)
        base = np.zeros(S["shape"], bool)
        for r0, r1, c0, c1 in ht.tiles(S["shape"], a.tile):
            base[r0:r1, c0:c1] = np.asarray(S["Ffused"][p][r0:r1, c0:c1], np.float64) < L26
        Wb = nt.block_region_counts(lab, k, S["K"], mask=base).astype(np.uint8)
        wb_dom[p] = Wb.sum(axis=0).astype(np.float64)
        wb_val[p] = Wb[vidx].sum(axis=0).astype(np.float64)
        del Wb, base
        F0[p] = nt.minimax_flood(zc[p], seed_c)
        C.log("240 m set-up", p)

    def measures(F, p):
        """E (km2) for domain and validated, and the band masks for the agreement."""
        c = S["c"][p]
        out, bands = {}, {}
        for D in DELTAS:
            for surf, top in (("uncorrected", L26), ("shifted", L26 + c)):
                band = (F < top) & ~(F < top - D)
                bands[(surf, D)] = band
                out[f"E_{surf}_{D:g}m_domain"] = float((band * w_dom).sum() * cell)
                out[f"E_{surf}_{D:g}m_validated"] = float((band * w_val).sum() * cell)
                notsea = ~(F < top - D)
                out[f"E_common_location_{surf}_{D:g}m_domain"] = float((notsea * wb_dom[p]).sum() * cell)
                out[f"E_common_location_{surf}_{D:g}m_validated"] = float((notsea * wb_val[p]).sum() * cell)
        return out, bands

    obs_rows = []
    obs_bands = {}
    for p in names:
        m, b = measures(F0[p], p)
        obs_rows.append({"product": p, **m})
        obs_bands[p] = b
    for key in obs_bands[names[0]]:
        st = np.stack([obs_bands[p][key] for p in names])
        obs_rows.append({"product": "all_five_agreement", "surface": key[0], "delta_m": key[1],
                         "ratio": nt.agreement_ratio(st, w_dom)})
    del obs_bands
    obs = pd.DataFrame(obs_rows)
    C.OUT.mkdir(parents=True, exist_ok=True)
    obs.to_csv(C.OUT / f"sim_observed_240m{a.suffix}.csv", index=False)
    C.log("unperturbed 240 m done")
    ranges = [float(r) for r in a.ranges.split(",")]
    children = np.random.SeedSequence(SEED).spawn(len(ranges) * len(S["names"]))
    real_rows, group_rows, neg = [], [], {}
    for ir, Rr in enumerate(ranges):
        for ip, p in enumerate(S["names"]):
            if p not in names:
                continue
            rng = np.random.default_rng(children[ir * len(S["names"]) + ip])
            t1 = time.time()
            acc_all, acc_any, g, i_in = {}, {}, 0, 0
            fields, neg[(Rr, p)] = None, None
            for i in range(a.n):
                if fields is None or not len(fields):
                    fields, ng = nt.grf_exponential(ny, nx, GRF_DX, sill[p], Rr, rng, n=min(20, a.n - i))
                    fields = list(fields)
                    neg[(Rr, p)] = ng
                f = fields.pop(0)
                eps = nt.bilinear_to(f, y0, x0, GRF_DX, ys, xs)
                F = nt.minimax_flood(zc[p] + eps, seed_c)
                m, bands = measures(F, p)
                real_rows.append({"range_m": Rr, "product": p, "realisation": i + 1,
                                  "eps_mean_m": float(eps.mean()), "eps_sd_m": float(eps.std()), **m})
                for key, b in bands.items():
                    if i_in == 0:
                        acc_all[key], acc_any[key] = b.copy(), b.copy()
                    else:
                        acc_all[key] &= b
                        acc_any[key] |= b
                i_in += 1
                if i_in == GROUP:
                    g += 1
                    for key in acc_all:
                        an = float((acc_any[key] * w_dom).sum())
                        group_rows.append({"range_m": Rr, "product": p, "group": g, "surface": key[0],
                                           "delta_m": key[1], "ratio": float((acc_all[key] * w_dom).sum()) / an
                                           if an > 0 else np.nan})
                    i_in = 0
                del F, eps
            C.log("simulated", p, f"range {Rr / 1000:g} km", a.n, "fields", round(time.time() - t1, 1), "s")
    rz = pd.DataFrame(real_rows)
    rz.to_csv(C.OUT / f"sim_realisations{a.suffix}.csv", index=False)
    gz = pd.DataFrame(group_rows)
    gz.to_csv(C.OUT / f"sim_groups{a.suffix}.csv", index=False)
    # summary
    em = pd.read_csv(C.OUT / "curves.csv") if (C.OUT / "curves.csv").exists() else None
    srows = []
    qcols = [c for c in rz.columns if c.startswith("E_")]
    for (Rr, p), g in rz.groupby(["range_m", "product"]):
        o = obs[obs["product"] == p].iloc[0]
        for q in qcols:
            v = g[q].to_numpy(float)
            rec = {"range_m": Rr, "product": p, "quantity": q, "n": int(v.size), "mean": float(v.mean()),
                   "sd": float(v.std(ddof=1)), "p05": float(np.percentile(v, 5)), "p50": float(np.median(v)),
                   "p95": float(np.percentile(v, 95)), "unperturbed_240m": float(o[q])}
            rec["sd_rel"] = rec["sd"] / rec["unperturbed_240m"] if rec["unperturbed_240m"] > 0 else np.nan
            if em is not None:
                parts = q.split("_")
                surf = "_".join(parts[1:-2])
                surf = {"uncorrected": "uncorrected", "shifted": "shifted",
                        "common_location_uncorrected": "common_location",
                        "common_location_shifted": "shifted_common_location"}[surf]
                D, reg = float(parts[-2][:-1]), parts[-1]
                r60 = em[(em["sensitivity"] == "main") & (em["surface"] == surf) & (em["product"] == p)
                         & (em["region"] == reg) & np.isclose(em["delta_m"], D)]
                rec["value_60m"] = float(r60["E_km2"].iloc[0]) if len(r60) else np.nan
            srows.append(rec)
        for (surf, D), gg in gz[(gz["range_m"] == Rr) & (gz["product"] == p)].groupby(["surface", "delta_m"]):
            v = gg["ratio"].to_numpy(float)
            ob = float(obs[(obs["product"] == "all_five_agreement") & (obs["surface"] == surf)
                           & np.isclose(obs["delta_m"], D)]["ratio"].iloc[0])
            srows.append({"range_m": Rr, "product": p, "quantity": f"agreement_{surf}_{D:g}m", "n": int(v.size),
                          "mean": float(np.nanmean(v)), "sd": float(np.nanstd(v, ddof=1)),
                          "p05": float(np.nanpercentile(v, 5)), "p50": float(np.nanmedian(v)),
                          "p95": float(np.nanpercentile(v, 95)), "observed_five_products_240m": ob,
                          "share_groups_at_or_below_observed": float(np.mean(v <= ob))})
    pd.DataFrame(srows).to_csv(C.OUT / f"sim_summary{a.suffix}.csv", index=False)
    C.write_json(C.OUT / f"run_simulate{a.suffix}.json", {
        "script": "scripts/rev_n_simulate.py", "decision": "D-125", "seed": SEED, "n_fields": a.n,
        "ranges_m": ranges, "grf_dx_m": GRF_DX, "flood_grid_m": k * 60, "group_size": GROUP,
        "sill_m2": sill, "nmad_m": {p: nm[p]["matched"] for p in names},
        "nmad_source": "results/validation/main/ATL08_accuracy.csv, scope overall, test blocks",
        "negative_eigenvalue_share": {f"{int(r)}|{p}": v for (r, p), v in neg.items()},
        "grf_grid": {"ny": ny, "nx": nx}, "flood_grid_shape": [int(R), int(Cc)], "cropping": crop,
        "shape_constants_m": S["c"], "level_2026_m": L26, "seconds": round(time.time() - t0, 1)})
    C.log("done", round(time.time() - t0, 1), "s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
