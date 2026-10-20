"""scripts/a7_hindcast.py end to end on a synthetic world, and caspian.h1 (the H1
decision rule) on synthetic tables. No real stage-2 data or product is read.

The world: a 100 x 140 grid of 60 m cells (3 km "blocks", all training blocks
unless a test says otherwise); seabed heights rise eastward 0.05 m per
cell with a north-south wobble, plus an inland pit. The domain leaves a rim of
beyond-domain cells; the stage-2 cover is a north-south band split into two stage-2
tiles that share one column, as the export's tiles do. Open sea lies west of the
cover, observed land east of it. The odd half sees 0.1 m less water in rows 10-15, so
the half-sample noise floor is non-zero in the north sector only.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from caspian import design, h1, levels, products
from caspian import hindcast_tiled as ht

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("a7_hindcast", ROOT / "scripts" / "a7_hindcast.py")
a7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a7)

H, W = 100, 140
TR = [60, 0, 0, 0, -60, 6000]
UNITS = {"E1": [20, 21, 22], "E2": [21, 22, 23], "E3": [10, 11]}
SECTORS = ["north", "south"]


def truth():
    r, c = np.mgrid[:H, :W]
    z = -32 + 0.05 * c + 0.3 * np.sin(r / 7.0 + 0.3)
    z[20:25, 95:100] = -40.0                   # an inland pit, below every level
    return z


def write_stage2(path: Path, z):
    path.mkdir(parents=True, exist_ok=True)
    # two tiles on rows 2-94: cols 30-70 and 70-109 (column 70 in both, identical counts)
    for name, c0, c1 in (("0_0", 30, 71), ("1_0", 70, 110)):
        zt = z[2:95, c0:c1]
        bands = {}
        for ep, bins in UNITS.items():
            for b in bins:
                level = float(levels.bin_center(b, design.BIN_WIDTH_M, design.BIN_ORIGIN_M))
                for p in (0, 1):
                    water = zt < level
                    if p == 1:
                        water[8:14] = zt[8:14] < level - 0.1      # global rows 10-15
                    v = np.full(zt.shape, 6, np.uint16)
                    if (ep, b, p) == ("E2", 22, 0) and name == "0_0":
                        v[40:44, 20:25] = 0                       # a cloud hole in one half
                    bands[f"w_{ep}_b{b:03d}_p{p}"] = (water * v).astype(np.uint16)
                    bands[f"v_{ep}_b{b:03d}_p{p}"] = v
        np.savez_compressed(path / f"tile_{name}.npz", transform=np.array([60, 0, 60 * c0, 0, -60, 6000 - 120]),
                            **bands)


def static(tile):
    sea = np.zeros((20, 28), np.uint8)
    sea[:, :5] = 1                                                # x < 1500 m
    beyond = np.full((25, 35), -1, np.int8)
    beyond[:, 29:] = 0                                            # x >= 6960 m: land
    return a7.static_layers(
        TR, (H, W), domain=box(120, 300, 8280, 5880),
        sector_polys={"north": box(120, 3000, 8280, 5880), "south": box(120, 300, 8280, 3000)},
        cover=box(1800, 300, 6600, 5880),
        sea={"mask": sea, "transform": [300, 0, 0, 0, -300, 6000]},
        beyond={"state": beyond, "transform": [240, 0, 0, 0, -240, 6000]}, tile=tile)


def heights():
    z = truth().astype(np.float32)
    z[:2, :10] = np.nan                                           # a product gap beyond the domain
    return {"perfect": z, "high": z + np.float32(0.3)}


def blocks(train_rows=("north", "south")):
    """3 km blocks: j = 1 north (rows 0-49), j = 0 south; training where listed."""
    return pd.DataFrame([{"i": i, "j": j, "sector": "north" if j == 1 else "south",
                          "split": "train" if ("north" if j == 1 else "south") in train_rows else "test"}
                         for i in range(3) for j in range(2)])


def fit_region(s, b):
    return s["domain"] & a7.train_mask(b, TR, (H, W), 3000)


def run(tmp: Path, tile: int, hts=None, **kw):
    write_stage2(tmp / "stage2", truth())
    s = static(tile)
    args = {"sector_names": SECTORS, "blocks": blocks(), "block_m": 3000, "tile": tile,
            "fit_region": fit_region(s, blocks()), "log": lambda *a, **k: None}
    args.update(kw)
    return a7.run(tmp / "stage2", tmp / "out", s, heights() if hts is None else hts, TR, **args)


@pytest.fixture(scope="module")
def tables(tmp_path_factory):
    return run(tmp_path_factory.mktemp("a7"), 16)


# --- static layers --------------------------------------------------------------------------

def test_static_layers():
    s = static(16)
    assert s["counts"]["sector_gaps_filled"] == 0
    assert set(np.unique(s["sector"])) == {1, 2}
    assert np.all(s["sector"][:50] == 1) and np.all(s["sector"][50:] == 2)     # extended beyond the domain
    out = s["outside"]
    assert np.all(out[3:94, 3:30] == 1)                           # open sea west of the cover
    assert np.all(out[3:94, 110:137] == 0)                        # hinterland, seeded from the beyond state
    assert np.all(out[s["zone"]] == -1)
    assert np.all(out[:2, 116:] == 0) and np.all(out[:2, :116] == -1)   # beyond the domain: land only
    assert s["sea_seed"].any() and not (s["sea_seed"] & s["zone"]).any()


def test_grid_matches_the_earth_engine_helper():
    pytest.importorskip("ee")
    from caspian import pipeline_ee
    bounds = (46.3, 43.875, 54.7, 47.45)
    tr, shape = a7.grid(bounds)
    tr_ee, dims = pipeline_ee.laea_transform(60, bounds)
    assert tr == tr_ee and shape == (dims[1], dims[0])


def test_nearest_resampling_and_misaligned_tiles(tmp_path):
    src = np.arange(12).reshape(3, 4)
    out = a7.nearest(src, [100, 0, 0, 0, -100, 300], [60, 0, 0, 0, -60, 300], (6, 8), -1)
    assert out[0, 0] == 0 and out[0, 1] == 0 and out[0, 2] == 1 and out[2, 0] == 4   # centres 30, 90, 150 m
    assert np.all(out[5] == -1) and np.all(out[:, 7] == -1)       # centres beyond the source
    np.savez_compressed(tmp_path / "tile_0_0.npz", transform=np.array([60, 0, 30, 0, -60, 6000]),
                        w_E1_b020_p0=np.zeros((2, 2), np.uint16), v_E1_b020_p0=np.full((2, 2), 6, np.uint16))
    with pytest.raises(ValueError, match="60 m grid"):
        a7.observed_raw({tmp_path / "tile_0_0.npz": ["E1_b020_p0"]}, TR, (H, W))


# --- end to end ------------------------------------------------------------------------------

def test_outputs_written_with_every_count(tables, tmp_path_factory):
    for k in ("extent", "displacement", "noise_floor", "observed", "blocks", "block_h1"):
        assert len(tables[k]) > 0
    assert {"north", "south", "all", "domain", "domain_train"} == set(tables["extent"]["sector"])
    for col in ("n_pred_unknown", "n_obs_unknown", "n_pred", "n_obs", "observed_level"):
        assert col in tables["displacement"].columns
    for col in ("n_tp", "n_fp", "n_fn", "n_fp_out", "n_fn_out", "area_bias_total_km2", "iou", "iou_total"):
        assert col in tables["extent"].columns
    units = {(ep, b) for ep, bins in UNITS.items() for b in bins}
    d = tables["displacement"]
    got = set(zip(d.loc[d["epoch"] != "pooled", "epoch"], d.loc[d["epoch"] != "pooled", "bin"]))
    assert got == units
    assert set(d["variant"]) == {"total", "shape"}
    assert set(d.loc[d["variant"] == "shape", "product"]) == {"high", "perfect"}


def test_perfect_product_scores_zero(tables):
    d, e = tables["displacement"], tables["extent"]
    p = d[(d["product"] == "perfect") & (d["variant"] == "total")]
    assert np.all(p["median_m"] == 0) and np.all(p["p95_m"] == 0)
    assert np.all(p.loc[p["sector"] != "all", "n_pred_unknown"] >= 0)
    assert p.loc[p["epoch"] != "pooled", "n_pred_unknown"].sum() > 0  # waterline beyond the domain: unmeasured
    pe = e[(e["product"] == "perfect") & (e["variant"] == "total")]
    assert np.all(pe["area_bias_total_km2"] == 0) and np.all(pe["iou_total"] == 1)
    assert np.all(pe["n_fp_out"] == 0) and np.all(pe["n_fn_out"] == 0)


def test_too_high_product_and_its_shape_variant(tables):
    d, e = tables["displacement"], tables["extent"]
    hi = d[(d["product"] == "high") & (d["variant"] == "total") & (d["sector"] != "all")]
    assert np.all(hi["signed_median_m"] < 0)                      # too little water
    assert np.all((hi["median_m"] > 0) & (hi["median_m"] <= 6 * 60 + 1e-9))
    he = e[(e["product"] == "high") & (e["variant"] == "total")]
    assert np.all(he["area_bias_total_km2"] < 0)
    sh = d[(d["product"] == "high") & (d["variant"] == "shape")]
    assert np.all(sh["median_m"] == 0)                            # the fitted shape offset removes it
    assert np.allclose(sh["flood_level_m"], sh["level_m"] + 0.3)
    ps = d[(d["product"] == "perfect") & (d["variant"] == "shape")]
    assert np.allclose(ps["flood_level_m"], ps["level_m"]) and np.all(ps["median_m"] == 0)


def test_level_datum_offset_is_applied_to_the_level(tmp_path, tables):
    t = run(tmp_path, 1000, level_datum_offset=-0.3, shape_offsets=None)
    cols = ["epoch", "bin", "sector", "median_m", "p95_m", "signed_median_m", "n", "n_pred", "n_obs"]
    a = t["displacement"].query("product == 'perfect'")[cols].reset_index(drop=True)
    b = tables["displacement"].query("product == 'high' and variant == 'total'")[cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)
    assert np.allclose(t["displacement"].query("product == 'perfect'")["flood_level_m"],
                       t["displacement"].query("product == 'perfect'")["level_m"] - 0.3)


def test_results_do_not_depend_on_the_tiling(tmp_path, tables):
    t = run(tmp_path, 1000)
    for k in ("extent", "displacement", "noise_floor", "observed", "blocks", "block_h1"):
        pd.testing.assert_frame_equal(t[k], tables[k], check_dtype=False)
    assert t["shape_offsets"] == tables["shape_offsets"]


def test_noise_floor_and_observed_classes(tables):
    nf = tables["noise_floor"]
    halves = nf[nf["kind"] == "halves"]
    assert np.all(halves.loc[halves["sector"] == "north", "p95_m"] > 0)
    assert np.all(halves.loc[halves["sector"] == "south", "p95_m"] == 0)
    e1 = nf[nf["kind"] == "e1_vs_e23"]
    assert set(e1["bin"]) == {21, 22} and np.all(e1["median_m"] == 0)
    pooled = nf[nf["kind"] == "pooled_level"]
    assert set(pooled["bin"]) == {10, 11, 20, 21, 22, 23}
    n21 = pooled.query("bin == 21 and sector == 'north'")["n"].item()
    parts = nf.query("bin == 21 and sector == 'north' and kind in ['halves', 'e1_vs_e23']")["n"].sum()
    assert n21 == parts                                          # halves of E1 and E2, and E1 vs E2
    assert set(nf.loc[nf["kind"] == "pooled_all_levels", "sector"]) == {"north", "south"}
    ob = tables["observed"]
    full = ob[(ob["sample"] == "full") & (ob["sector"] == "north") & ob["epoch"].isin(["E1", "E2", "E3"])]
    assert np.allclose(full["area_unconnected_land_km2"], 25 * 0.0036)    # the inland pit
    assert np.all(full["area_unconnected_unknown_km2"] == 0)
    # the cloud hole lowers the valid area of that half only
    v = ob.query("epoch == 'E2' and bin == 22 and sector == 'all'").set_index("sample")["area_valid_km2"]
    assert v["p0"] < v["p1"] == v["full"]


def test_pooled_product_rows_sum_the_epochs(tables):
    d = tables["displacement"].query("product == 'high' and variant == 'total' and sector == 'north' and bin == 22")
    pooled = d[d["epoch"] == "pooled"].iloc[0]
    per = d[d["epoch"] != "pooled"]
    assert pooled["epochs"] == "E1+E2" and pooled["n"] == per["n"].sum()
    assert pooled["observed_level"] == bool(per["observed_level"].any())


def test_block_table_and_run_record(tables, tmp_path):
    bl = tables["blocks"]
    assert {"block_i", "block_j", "n", "median_abs_m", "block_sector", "split"} <= set(bl.columns)
    assert set(bl["block_sector"]) <= {"north", "south"}
    t = run(tmp_path, 64, shape_offsets=None)
    import json
    rec = json.loads((tmp_path / "out" / "run.json").read_text(encoding="utf-8"))
    assert rec["level_datum"]["offset_m"] == 0.0 and rec["level_datum"]["source"] == a7.DATUM_SOURCE
    assert rec["shape_offsets"].startswith("none")
    over = [u["overlap_cells"] for u in rec["units"] if u["sample"] == "full" and u["epoch"] in UNITS]
    assert all(o == 93 for o in over)                            # the shared column, kept once
    assert all(u["overlap_disagree"] == 0 for u in rec["units"])
    assert set(t["displacement"]["variant"]) == {"total"}


def test_distance_codes_decode_to_the_same_floats():
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(3)
    a, b = rng.integers(0, 5000, (400, 2)), rng.integers(0, 5000, (300, 2))
    d = cKDTree(b.astype(float)).query(a.astype(float))[0] * 60.0      # as hindcast_tiled measures
    signed = np.r_[d * np.where(rng.random(d.size) < 0.5, -1, 1), np.inf, -np.inf, 0.0]
    assert np.array_equal(a7.decode(a7.encode(signed)), signed)
    pool = a7.Pool()
    for part in np.array_split(signed[:-3], 3):
        pool.add(1, {"distances": part, "n_pred": 1, "n_obs": 2, "n_pred_unknown": 0, "n_obs_unknown": 0,
                     "observed_level": False})
    s = pool.summary(1)
    assert s["median"] == np.median(np.abs(signed[:-3])) and s["n_obs"] == 6


def test_rule_applied_to_the_synthetic_driver_tables(tables):
    d = tables["displacement"]
    pl = d[d["epoch"] == "pooled"].rename(columns={"median_m": "median"})
    nf = tables["noise_floor"].query("kind == 'pooled_level'").rename(columns={"p95_m": "p95"})
    s, _ = h1.mispredicts(pl[["product", "variant", "sector", "bin", "observed_level", "n", "median"]],
                          nf[["sector", "bin", "n", "p95"]])
    s = s.set_index(["product", "variant", "sector"])["mispredicts"]
    assert not s.loc[("perfect", "total")].any() and not s.loc[("high", "shape")].any()
    assert s.loc[("high", "total")].all()


# --- the shape offset (D-070) ---------------------------------------------------------------

def test_shape_offset_fit_and_its_record(tables, tmp_path_factory):
    rec = tables["shape_offsets"]
    assert rec["products"]["high"]["offset_m"] == 0.3 and rec["products"]["perfect"]["offset_m"] == 0.0
    for r in rec["products"].values():
        assert len(r["objective_km2"]) == a7.C_GRID.size == 1001 and r["objective_km2_at_offset"] == 0
        assert set(r["leave_one_epoch_out"]) == {"E1", "E2", "E3"}
        assert r["n_units"] == len(rec["units"]) == 8
    assert rec["used_in_this_run"]["high"]["offset_m"] == 0.3
    # the objective at c = 0 is the sum over units of |total area bias| in the fit region
    e = tables["extent"].query("product == 'high' and variant == 'total' and sector == 'domain_train'")
    assert len(e) == 8
    assert rec["products"]["high"]["objective_km2"][500] == pytest.approx(e["area_bias_total_km2"].abs().sum())
    assert rec["products"]["high"]["objective_km2"][500] > 0
    # leave-one-epoch-out: the refit uses only the other epochs' units
    loeo = rec["products"]["high"]["leave_one_epoch_out"]["E3"]
    assert loeo["n_units"] == 6 and loeo["offset_m"] == 0.3


def test_shape_fit_uses_training_blocks_only(tmp_path):
    write_stage2(tmp_path / "stage2", truth())
    s = static(16)
    z = truth().astype(np.float32)
    z[50:] += np.float32(0.5)                                    # 0.5 m too high in the south only
    units = a7.stage2_index(tmp_path / "stage2")
    F = {"bump": ht.flood_level_map(z, s["sea_seed"], tile=16)}
    fin = {"bump": lambda r0, r1, c0, c1: np.isfinite(z[r0:r1, c0:c1])}
    got = {}
    for train in (("north",), ("south",)):
        rec, bias = a7.fit_shape(units, TR, s, F, fin, fit_region(s, blocks(train)), level_datum_offset=0.0,
                                 product_offsets={}, tile=16, log=lambda *a, **k: None)
        got[train] = rec["products"]["bump"]["offset_m"]
    assert got[("north",)] == 0.0 and got[("south",)] == 0.5


def test_fit_offset_ties_go_to_the_smallest_magnitude():
    bias = np.zeros((2, a7.C_GRID.size), np.int64)
    bias[0] = np.abs(np.round(a7.C_GRID * 100)).astype(int) - 30   # |bias| zero at c = +-0.3
    assert a7.fit_offset(bias, ["E1", "E2"])["offset_m"] == -0.3      # +-0.3 tie: the smaller c
    flat = np.zeros((1, a7.C_GRID.size), np.int64)
    r = a7.fit_offset(flat, ["E1"])
    assert r["offset_m"] == 0.0 and r["n_tied_minima"] == 1001
    with pytest.raises(ValueError, match="no unit"):
        a7.fit_offset(flat, ["E1"], exclude="E1")


def test_total_error_reads_corrected_heights(tmp_path, tables):
    hi = heights()["high"]
    corr = np.full(hi.shape, -0.3, np.float32)                   # a uniform geoid difference
    t = run(tmp_path, 32, hts={"high": products.Corrected(hi, corr)}, shape_offsets=None)
    got = t["displacement"].query("variant == 'total'")
    assert np.all(got["median_m"] == 0)                           # = the perfect product
    e = t["extent"].query("variant == 'total'")
    assert np.all(e["area_bias_total_km2"] == 0)


def test_block_values_pool_every_unit(tables):
    bl, hb = tables["blocks"], tables["block_h1"]
    n = bl.groupby(["product", "variant", "block_i", "block_j"])["n"].sum()
    got = hb.set_index(["product", "variant", "block_i", "block_j"])["n"]
    pd.testing.assert_series_equal(got.sort_index(), n.sort_index(), check_names=False, check_dtype=False)
    assert np.all(hb.query("product == 'perfect'")["median_abs_m"] == 0)
    assert np.all(hb.query("product == 'high' and variant == 'total'")["median_abs_m"] > 0)
    assert {"block_sector", "split"} <= set(hb.columns)


def test_block_median_from_counts_equals_np_median():
    rng = np.random.default_rng(4)
    cells = {"pred": {"rows": rng.integers(0, 100, 300), "cols": rng.integers(0, 140, 300),
                      "known": rng.random(300) < 0.9},
             "obs": {"rows": rng.integers(0, 100, 200), "cols": rng.integers(0, 140, 200),
                     "known": rng.random(200) < 0.9}}
    k = rng.integers(0, 400, 500)
    d = np.sqrt(k) * 60 * np.where(rng.random(500) < 0.5, -1, 1)
    d[:3] = np.inf
    cells["signed_p"], cells["signed_o"] = d[:300], d[300:]
    h = a7.BlockHist(TR, 3000)
    h.add(("p", "total"), cells)
    h.add(("p", "total"), cells)                                 # a second unit
    tab = h.table().set_index(["block_i", "block_j"])
    bi, bj, dd = a7._measured(cells, TR, 3000)
    for (i, j), r in tab.iterrows():
        sel = np.abs(np.r_[dd, dd][np.r_[(bi == i) & (bj == j), (bi == i) & (bj == j)]])
        assert r["n"] == sel.size and r["median_abs_m"] == np.median(sel)


# --- the H1 decision rule on hand-made tables -----------------------------------------------

def _levels(rows):
    return pd.DataFrame(rows, columns=["product", "variant", "sector", "bin", "observed_level", "n", "median"])


def test_mispredicts_majority_of_evaluable_observed_levels():
    P = [("g", "total", "volga", b, True, 10, m) for b, m in zip(range(5), [50, 50, 50, 1, 1])]   # 3 of 5
    P += [("g", "total", "west", b, True, 10, m) for b, m in zip(range(4), [50, 50, 1, 1])]       # 2 of 4
    P += [("g", "total", "west", 9, False, 10, 999.0)]           # not an observed level there: ignored
    P += [("g", "total", "ural", 0, True, 10, np.inf),           # no predicted waterline: exceeds
          ("g", "total", "ural", 1, True, 10, 1.0),
          ("g", "total", "ural", 2, True, 0, np.nan),            # nothing measured: not evaluable
          ("g", "total", "ural", 3, True, 10, 99.0)]             # no noise distance: not evaluable
    P += [("g", "total", "north", 0, True, 0, np.nan)]
    N = pd.DataFrame([(s, b, 20, 10.0) for s in ("volga", "west") for b in range(10)]
                     + [("ural", 0, 20, np.inf), ("ural", 1, 20, 10.0), ("ural", 2, 20, 10.0), ("ural", 3, 0, np.nan)],
                     columns=["sector", "bin", "n", "p95"])
    s, detail = h1.mispredicts(_levels(P), N)
    s = s.set_index("sector")
    assert bool(s.loc["volga", "mispredicts"]) is True and s.loc["volga", "n_exceeding"] == 3
    assert bool(s.loc["west", "mispredicts"]) is False                      # exactly half is not a majority
    assert s.loc["west", "n_observed_levels"] == 4
    assert s.loc["ural", "n_evaluable"] == 2 and s.loc["ural", "n_not_evaluable"] == 2
    assert bool(s.loc["ural", "mispredicts"]) is False                      # 1 of 2
    assert pd.isna(s.loc["north", "mispredicts"])                           # nothing evaluable
    assert detail.query("sector == 'ural' and bin == 0")["exceeds"].item()  # inf exceeds even an inf floor
    with pytest.raises(ValueError, match="one row"):
        h1.mispredicts(_levels(P), pd.concat([N, N]))


def test_mispredicts_with_the_pooled_floor_sensitivity():
    P = [("g", "total", "volga", b, True, 10, m) for b, m in zip(range(3), [5, 30, 30])]
    per_level = pd.DataFrame([("volga", 0, 9, 1.0), ("volga", 1, 9, 50.0), ("volga", 2, 9, 50.0)],
                             columns=["sector", "bin", "n", "p95"])
    pooled = pd.DataFrame([("volga", 27, 20.0)], columns=["sector", "n", "p95"])
    assert not h1.mispredicts(_levels(P), per_level)[0]["mispredicts"].item()
    assert h1.mispredicts(_levels(P), pooled)[0]["mispredicts"].item()


def test_permutation_test():
    rng = np.random.default_rng(1)
    b = pd.DataFrame({"sector": ["volga"] * 15 + ["ural"] * 15 + ["west"] * 30,
                      "value": np.r_[rng.normal(300, 30, 30), rng.normal(100, 30, 30)]})
    r = h1.permutation_test(b, ["volga", "ural"], n_perm=2000)
    assert r["significant"] and r["p_value"] == pytest.approx(1 / 2001) and r["statistic"] > 150
    assert r == h1.permutation_test(b, ["volga", "ural"], n_perm=2000)           # seeded
    flat = b.assign(value=rng.normal(100, 30, len(b)))
    assert not h1.permutation_test(flat, ["volga", "ural"], n_perm=2000)["significant"]
    rev = h1.permutation_test(b, ["west"], n_perm=2000)                          # one-sided
    assert rev["statistic"] < 0 and rev["p_value"] > 0.99
    nan = b.copy()
    nan.loc[0, "value"] = np.nan
    assert h1.permutation_test(nan, ["volga"], n_perm=10)["n_excluded_nan"] == 1
    assert np.isnan(h1.permutation_test(b, ["komsomolets"], n_perm=10)["p_value"])


def test_block_test_groups_and_exclusions():
    rng = np.random.default_rng(6)
    rows = []
    for prod in ("g", "e"):
        for k in range(40):
            sec = ["volga", "ural", "west", "northeast"][k % 4]
            big = prod == "g" and sec in ("volga", "west")
            rows.append({"product": prod, "variant": "total", "block_sector": sec,
                         "n": 25, "median_abs_m": rng.normal(400 if big else 100, 20)})
    rows += [{"product": "g", "variant": "total", "block_sector": "volga", "n": 19, "median_abs_m": 9e9},
             {"product": "g", "variant": "total", "block_sector": None, "n": 50, "median_abs_m": 9e9}]
    labels = pd.DataFrame({"sector": ["volga", "ural", "west", "northeast"], "deltaic": [True, True, False, False],
                           "vegetated": [False, False, True, False]})
    r = h1.block_test(pd.DataFrame(rows), labels, n_perm=500).set_index("product")
    assert r.loc["g", "n_excluded_few"] == 1 and r.loc["g", "n_excluded_no_sector"] == 1
    assert r.loc["g", "n_group"] == 30 and r.loc["g", "n_rest"] == 10        # deltaic or vegetated
    assert r.loc["g", "group_sectors"] == "ural+volga+west"
    assert r.loc["g", "statistic"] > 0 and r.loc["e", "n_excluded_few"] == 0
    with pytest.raises(ValueError, match="label"):
        h1.block_test(pd.DataFrame(rows), labels[labels["sector"] != "west"], n_perm=10)
    with pytest.raises(ValueError, match="missing"):
        h1.block_test(pd.DataFrame(rows), labels.assign(vegetated=[False, None, True, False]), n_perm=10)
    assert design.PERMUTATIONS_N == 10_000 and h1.MIN_BLOCK_N == 20


def test_record_low_envelope_sets_dark_water_back_to_water(tmp_path, monkeypatch):
    # one 4 x 4 tile; E3 bins 10-12 (record low) water everywhere; bin 20 has a noise-dry cell
    tr = np.array([60.0, 0, 0, 0, -60.0, 240.0])
    full = np.full((4, 4), 6, np.uint16)
    dry = full.copy()
    dry[1, 1] = 0                                     # observed dry six times out of six: noise
    bands = {}
    for b in (10, 11, 12):
        for par in (0, 1):
            bands[f"w_E3_b{b:03d}_p{par}"] = full
            bands[f"v_E3_b{b:03d}_p{par}"] = full
    for par in (0, 1):
        bands[f"w_E3_b020_p{par}"] = dry
        bands[f"v_E3_b020_p{par}"] = full
    np.savez(tmp_path / "tile_0_0.npz", transform=tr, **bands)
    units = a7.stage2_index(tmp_path)
    env = a7.record_low_water(units, list(tr), (4, 4))
    assert env.all()
    monkeypatch.setattr(a7, "ENVELOPE", None)
    raw, _ = a7.observed_raw(a7.groups_for(units, 20, ["E3"]), list(tr), (4, 4))
    assert raw[1, 1] == 0                              # without the envelope: the noise cell is dry
    monkeypatch.setattr(a7, "ENVELOPE", env)
    raw, info = a7.observed_raw(a7.groups_for(units, 20, ["E3"]), list(tr), (4, 4))
    assert raw[1, 1] == 1 and info["envelope_cells"] == 1
