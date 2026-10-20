"""caspian.robustness on synthetic data: stage-2 count algebra (bin width, class splits,
E1-end windows, vegetation, screening deltas), plan builders, level shifts, ordinary
kriging, the validation variants, tide-system shifts, hindcast headline tables, the
morphology table, the area-level slope and the robustness block sample. No real stage-2
count, DEM or metric is read."""

import numpy as np
import pandas as pd
import pytest

from caspian import design, levels, robustness as R, validation
from caspian.geography import SECTOR_ORDER

RNG = np.random.default_rng(7)


# --- synthetic acquisitions -------------------------------------------------------------------

def _acquisitions(n=60, shape=(20, 30), seed=1, lo=-27.0, hi=-26.6):
    """Acquisitions with a level, a time, a sensor and per-pixel water / valid / observed
    masks over a seabed rising eastward; parity alternates in time order within each
    (epoch, 0.10 m bin) as in scripts/a6_stage2_plan.py."""
    rng = np.random.default_rng(seed)
    r, c = np.mgrid[:shape[0], :shape[1]]
    z = -27.2 + 0.03 * c + 0.01 * np.sin(r / 3)
    t = pd.Timestamp("2014-04-01") + pd.to_timedelta(np.sort(rng.uniform(0, 3000, n)), unit="D")
    lev = rng.uniform(lo, hi, n)
    sensors = rng.choice(["ETM_L7", "OLI_L8", "MSI_S2"], n)
    acq = pd.DataFrame({"acq": [f"A{k:03d}" for k in range(n)], "t": t, "level_m": lev, "sensor": sensors,
                        "epoch": "E3"})
    acq["bin"] = levels.bin_index(lev, design.BIN_WIDTH_M, design.BIN_ORIGIN_M)
    acq = acq.sort_values("t").reset_index(drop=True)
    acq["parity"] = acq.groupby(["epoch", "bin"]).cumcount() % 2
    masks = {}
    for k, row in acq.iterrows():
        obs = rng.random(shape) > 0.15
        veg = obs & (rng.random(shape) < 0.1)
        water = obs & ~veg & (z < row["level_m"] + rng.normal(0, 0.02, shape))
        valid = obs & ~veg
        masks[row["acq"]] = (water, valid, obs)
    return acq, masks


def _counts(acq, masks, key):
    """Count bands {kind_group: array} with the group of each acquisition given by key(row)."""
    out = {}
    for _, row in acq.iterrows():
        g = key(row)
        if g is None:
            continue
        for kind, m in zip("wvo", masks[row["acq"]]):
            b = f"{kind}_{g}"
            out[b] = out.get(b, 0) + m.astype(np.int64)
    return out


def _main_key(row):
    return R.group_name(row["epoch"], row["bin"], row["parity"])


# --- count algebra --------------------------------------------------------------------------------

def test_band_names_round_trip():
    p = R.parse_band("w_E2_b043_p1_E7off_lo")
    assert p == {"kind": "w", "epoch": "E2", "bin": 43, "parity": 1, "suffix": "E7off_lo"}
    assert R.parse_band("v_E1_b007_p0")["suffix"] == ""
    assert R.parse_band("transform") is None and R.parse_band("w_E4_b001_p0") is None
    assert R.group_name("E3", 5, 0, "MSI") == "E3_b005_p0_MSI"
    with pytest.raises(ValueError):
        R.group_name("E3", 1000, 0)


def test_rebin_to_020_equals_direct_020_counts():
    acq, masks = _acquisitions()
    fine = _counts(acq, masks, _main_key)
    merged = R.merge_groups(fine, R.rebin_mapper(2))
    acq20 = acq.assign(bin=levels.bin_index(acq["level_m"], 0.20, design.BIN_ORIGIN_M))
    direct = _counts(acq20, masks, _main_key)
    assert set(merged) == set(direct)
    for k in direct:
        assert np.array_equal(merged[k], direct[k]) and merged[k].dtype == np.uint16
    # the 0.20 m bin k spans the 0.10 m bins 2k and 2k + 1 on the same origin
    assert np.allclose(levels.bin_center(np.arange(3), 0.20, -30.0),
                       (levels.bin_center(2 * np.arange(3), 0.10, -30.0)
                        + levels.bin_center(2 * np.arange(3) + 1, 0.10, -30.0)) / 2)


def test_merge_groups_checks():
    a = np.ones((2, 2), np.uint16)
    with pytest.raises(ValueError, match="exceeds valid"):
        R.merge_groups({"w_E2_b001_p0": 2 * a, "v_E2_b001_p0": a}, lambda g: g)
    with pytest.raises(ValueError, match="different kinds"):
        R.merge_groups({"w_E2_b002_p0": a, "v_E2_b002_p0": a, "o_E2_b002_p0": a,
                        "w_E2_b003_p0": a, "v_E2_b003_p0": a}, R.rebin_mapper(2))
    with pytest.raises(ValueError, match="needs both"):
        R.merge_groups({"w_E2_b002_p0": a}, lambda g: g)
    with pytest.raises(ValueError, match="overflow"):
        big = np.full((1, 1), 40_000, np.uint16)
        R.merge_groups({"w_E2_b002_p0": big, "v_E2_b002_p0": big, "w_E2_b003_p0": big, "v_E2_b003_p0": big},
                       R.rebin_mapper(2))
    with pytest.raises(ValueError, match="not a count band"):
        R.merge_groups({"transform": a}, lambda g: g)
    out = R.merge_groups({"w_E2_b002_p0": a, "v_E2_b002_p0": a}, lambda g: None)
    assert out == {}


def _plan_from_acq(acq, scenes_per_acq=2):
    rows = []
    for _, a in acq.iterrows():
        for s in range(scenes_per_acq):
            rows.append({"scene": f"{a['acq']}_{s}", "acq": a["acq"], "sensor": a["sensor"],
                         "t": a["t"] + pd.Timedelta(seconds=25 * s), "level_m": a["level_m"] + 1e-6 * s,
                         "epoch": a["epoch"], "bin": a["bin"], "parity": a["parity"],
                         "threshold_source": "pooled_otsu" if s == 0 or a["acq"] < "A030" else "pooled_otsu"})
    p = pd.DataFrame(rows)
    p["group"] = [R.group_name(e, b, q) for e, b, q in zip(p["epoch"], p["bin"], p["parity"])]
    return p


def test_split_groups_halves_give_005_bins_and_sum_back():
    acq, masks = _acquisitions(n=80)
    plan = _plan_from_acq(acq)
    sp = R.split_groups(plan, by=("sensor", "half"))
    assert (sp["base_group"] == plan["group"]).all()
    g_of = sp.groupby("acq")["group"].first()
    split_counts = _counts(acq, masks, lambda row: g_of[row["acq"]])
    # summing every class gives the unsplit main groups exactly
    back = R.merge_groups(split_counts, R.select_mapper())
    main = _counts(acq, masks, _main_key)
    assert set(back) == set(main) and all(np.array_equal(back[k], main[k]) for k in main)
    # 0.05 m bins from the half tokens equal the direct 0.05 m binning
    fine = R.merge_groups(split_counts, R.compose(R.halves_to_fine_mapper, R.select_mapper()))
    acq05 = acq.assign(bin=levels.bin_index(acq["level_m"], 0.05, design.BIN_ORIGIN_M))
    direct = _counts(acq05, masks, _main_key)
    assert set(fine) == set(direct) and all(np.array_equal(fine[k], direct[k]) for k in direct)


def test_select_mapper_drops_a_sensor_class():
    acq, masks = _acquisitions(n=60, seed=3)
    plan = _plan_from_acq(acq)
    sp = R.split_groups(plan, by=("sensor",))
    g_of = sp.groupby("acq")["group"].first()
    split_counts = _counts(acq, masks, lambda row: g_of[row["acq"]])
    landsat = R.merge_groups(split_counts, R.select_mapper(drop=["MSI"]))
    direct = _counts(acq[acq["sensor"] != "MSI_S2"], masks, _main_key)
    assert set(landsat) == set(direct) and all(np.array_equal(landsat[k], direct[k]) for k in direct)
    only_oli = R.merge_groups(split_counts, R.select_mapper(keep_only=["OLI"]))
    direct = _counts(acq[acq["sensor"] == "OLI_L8"], masks, _main_key)
    assert all(np.array_equal(only_oli[k], direct[k]) for k in direct)


def test_sensor_class_and_slc_date():
    from caspian import masks
    assert R.SLC_OFF == masks.L7_SLC_OFF
    c = R.sensor_class(["TM_L5", "ETM_L7", "ETM_L7", "OLI_L9", "MSI_S2"],
                       ["1995-01-01", "2003-05-30", "2003-05-31", "2022-01-01", "2020-01-01"])
    assert list(c) == ["TM", "E7on", "E7off", "OLI", "MSI"]
    with pytest.raises(ValueError):
        R.sensor_class(["XX"], ["2000-01-01"])


def test_e1_end_windows():
    ends = [pd.Timestamp("1995-05-28 01:37:03"), pd.Timestamp("1995-07-06 17:31:11"),
            pd.Timestamp("1995-07-16 15:29:42")]
    times = ["1995-05-01", "1995-05-28 01:37:03", "1995-06-10", "1995-07-06 17:31:11", "1995-07-10", "1995-08-01"]
    j = R.e1_window_index(times, ends)
    assert list(j) == [0, 0, 1, 1, 2, 3]
    # under each end, E1 exactly when time <= end (the a4 rule)
    for a, end in enumerate(ends):
        m = R.e1_end_mapper(ends, end)
        for t, jj in zip(times, j):
            g = {"epoch": "E1", "bin": 40, "parity": 0, "suffix": f"x{jj}" if 0 < jj < 3 else ""}
            ep = m(g)["epoch"] if g["suffix"] else ("E1" if jj == 0 else "E2")
            assert ep == ("E1" if pd.Timestamp(t) <= end else "E2")
    with pytest.raises(ValueError):
        R.e1_window_index(times, ends[::-1])


def test_split_groups_tags_window_and_sample():
    acq, _ = _acquisitions(n=20, seed=5)
    plan = _plan_from_acq(acq)
    plan.loc[plan["acq"].isin(["A000", "A001"]), "epoch"] = "E1"
    t_mid = plan["t"].sort_values().iloc[len(plan) // 2]
    ends = [plan["t"].min() - pd.Timedelta(days=1), t_mid, plan["t"].max() + pd.Timedelta(days=1)]
    sp = R.split_groups(plan, by=(), e1_ends=ends, sample_acqs=["A003"])
    assert sp.loc[sp["acq"] == "A003", "group"].str.endswith("a3").all()
    assert sp["group"].str.contains("_x").any()
    assert sp.groupby("acq")["group"].nunique().max() == 1
    with pytest.raises(ValueError):
        R.split_groups(plan, by=("colour",))


def test_vegetation_variants():
    w, v, o = np.array([1, 0, 3]), np.array([2, 1, 3]), np.array([4, 1, 5])
    assert [x.tolist() for x in R.vegetation_variant(w, v, o, "not_observed")] == [[1, 0, 3], [2, 1, 3]]
    assert [x.tolist() for x in R.vegetation_variant(w, v, o, "as_dry")] == [[1, 0, 3], [4, 1, 5]]
    assert [x.tolist() for x in R.vegetation_variant(w, v, o, "as_water")] == [[3, 0, 5], [4, 1, 5]]
    with pytest.raises(ValueError):
        R.vegetation_variant(w, o, v, "as_dry")
    with pytest.raises(ValueError):
        R.vegetation_variant(w, v, o, "as_reeds")
    acq, masks = _acquisitions(n=10)
    tile = _counts(acq, masks, _main_key)
    out = R.vegetation_tile(tile, "as_dry")
    g = next(k for k in tile if k.startswith("o_"))[2:]
    assert np.array_equal(out[f"v_{g}"], tile[f"o_{g}"]) and set(k[0] for k in out) == {"w", "v"}
    with pytest.raises(ValueError, match="need w, v and o"):
        R.vegetation_tile({f"w_{g}": tile[f"w_{g}"], f"v_{g}": tile[f"v_{g}"]}, "as_dry")


# --- screening deltas ---------------------------------------------------------------------------

def _setup_plan(n_acq=40, seed=11):
    rng = np.random.default_rng(seed)
    acq, masks = _acquisitions(n=n_acq, seed=seed)
    plan = _plan_from_acq(acq, scenes_per_acq=1)
    for s in SECTOR_ORDER:
        plan[f"setup_{s}_m"] = rng.normal(0, 0.15, len(plan))
    plan.loc[3, "setup_west_m"] = np.nan
    return acq, masks, plan


def test_exclusion_rules():
    _, _, plan = _setup_plan()
    ex = R.exclusion(plan, 0.15)
    a = plan[[f"setup_{s}_m" for s in SECTOR_ORDER]].to_numpy()
    assert np.array_equal(ex, (np.abs(a) > 0.15) | np.isnan(a))
    assert ex[3, 0]
    assert not R.exclusion(plan, 0.15, unknown_excluded=False)[3, 0]
    assert not R.exclusion(plan, 0.15, exempt_sectors=["komsomolets"])[:, SECTOR_ORDER.index("komsomolets")].any()
    with pytest.raises(ValueError):
        R.exclusion(plan.drop(columns=["setup_ural_m"]), 0.1)


def test_screening_delta_reproduces_alternative_counts():
    """Two sectors (left and right halves of the grid): main = tau 0.15, alt = tau 0.20;
    main + ADD - SUB equals the alternative's counts, and the reverse direction too."""
    acq, masks, plan = _setup_plan()
    sector = np.zeros((20, 30), int)
    sector[:, 15:] = 1

    def counts(excl):
        out = {}
        for k, row in plan.iterrows():
            inc = np.zeros((20, 30), bool)
            for s in (0, 1):
                inc |= (sector == s) & ~excl[k, s]
            g = row["group"]
            for kind, m in zip("wv", masks[row["acq"]][:2]):
                out[f"{kind}_{g}"] = out.get(f"{kind}_{g}", 0) + (m & inc).astype(np.int64)
        return out

    def subset_counts(rows):
        out = {}
        for _, row in rows.iterrows():
            codes = [int(c) - 1 for c in str(row["sectors"]).split(",") if c]
            inc = np.isin(sector, codes)
            g = row["group"]
            for kind, m in zip("wv", masks[row["acq"]][:2]):
                out[f"{kind}_{g}"] = out.get(f"{kind}_{g}", 0) + (m & inc).astype(np.int64)
        return out

    for t_main, t_alt in ((0.15, 0.20), (0.20, 0.10)):
        ex_main, ex_alt = R.exclusion(plan, t_main), R.exclusion(plan, t_alt)
        d = R.screening_delta(plan, ex_main, ex_alt)
        assert (d["add"]["mask_mode"] == "include").all()
        main_c, alt_c = counts(ex_main), counts(ex_alt)
        got = R.apply_delta({k: v.astype(np.uint16) for k, v in main_c.items()}, subset_counts(d["add"]),
                            subset_counts(d["sub"]))
        for k in alt_c:
            assert np.array_equal(got[k], alt_c[k])
        assert d["n_mixed_acq_sector"] == 0          # one scene per acquisition here
    with pytest.raises(ValueError):
        R.screening_delta(plan, ex_main[:-1], ex_alt[:-1])


def test_apply_delta_refuses_negative_counts():
    a = np.ones((2, 2), np.uint16)
    with pytest.raises(ValueError, match="SUB removes"):
        R.apply_delta({"w_E2_b001_p0": a, "v_E2_b001_p0": a}, sub={"v_E2_b001_p0": 2 * a})


def test_delta_cost_share():
    rows = pd.DataFrame({"sectors": ["1", "1,7"]}, index=[0, 2])
    km2 = pd.Series([100.0, 50.0, 200.0])
    w = {s: (0.5 if s in ("west", "open_shelf") else 0.0) for s in SECTOR_ORDER}
    c = R.delta_cost_share(rows, km2, 350.0, w)
    assert c["upper"] == pytest.approx(300 / 350)
    assert c["per_sector"] == pytest.approx((100 * 0.5 + 200 * 1.0) / 350)


def test_acquisition_sample_whole_acquisitions():
    acq, _ = _acquisitions(n=60, seed=2)
    plan = _plan_from_acq(acq, scenes_per_acq=3)
    s1, s2 = R.acquisition_sample(plan, 10), R.acquisition_sample(plan, 10)
    pd.testing.assert_frame_equal(s1, s2)
    for _, g in s1.groupby("sensor"):
        assert g["n_scenes"].sum() >= 10 and g["n_scenes"].iloc[:-1].sum() < 10
    assert (s1["n_scenes"] == 3).all()


# --- level shifts --------------------------------------------------------------------------------

def test_group_level_shift_constant_offset_and_rebinning():
    acq, _ = _acquisitions(n=50)
    plan = _plan_from_acq(acq)
    sh = R.group_level_shift(plan, plan["level_m"] + 0.03)
    assert np.allclose(sh["shift_m"], 0.03)
    expected = (levels.bin_index(acq["level_m"] + 0.03, 0.1, -30) != acq["bin"]).sum()
    assert sh["n_acq_rebin"].sum() == expected
    pts = pd.DataFrame({"epoch": "E3", "bin": sh["bin"], "z_m": levels.bin_center(sh["bin"], 0.1, -30.0)})
    out = R.shift_points(pts, sh)
    assert np.allclose(out["z_m"] - pts["z_m"], 0.03)
    with pytest.raises(ValueError):
        R.shift_points(pts.assign(bin=999), sh)


def test_setup_shift_means_over_included_acquisitions():
    _, _, plan = _setup_plan()
    ex = R.exclusion(plan, 0.15)
    sh = R.setup_shift(plan, ex)
    k = SECTOR_ORDER.index("volga")
    g = plan.loc[~ex[:, k]].groupby(["epoch", "bin"])["setup_volga_m"].mean()
    got = sh[sh["sector"] == "volga"].set_index(["epoch", "bin"])["shift_m"]
    assert np.allclose(got.loc[g.index], g)
    assert (np.abs(sh["shift_m"].dropna()) <= 0.15).all()


# --- kriging -----------------------------------------------------------------------------------------

def _contours(n_lines=8, spacing=60.0):
    """Contour points of the plane z = -28 + 1e-4 x + small curvature, along iso-lines."""
    pts, zs = [], []
    for k in range(n_lines):
        x0 = 1000 + 1500 * k
        y = np.arange(0, 12_000, spacing)
        x = x0 + 200 * np.sin(y / 3000)
        pts.append(np.column_stack([x, y]))
        zs.append(-28 + 1e-4 * x + 2e-9 * (y - 6000) ** 2)
    return np.vstack(pts), np.concatenate(zs)


def test_kriging_exact_at_data_and_reproduces_constant():
    xy = RNG.uniform(0, 1000, (40, 2))
    z = RNG.normal(0, 1, 40)
    zh, var = R.ordinary_kriging(xy, z, xy[:10], sill=1.0, a=300.0, nugget=0.0, k=12, per_quadrant=None)
    assert np.allclose(zh, z[:10], atol=1e-6) and np.allclose(var, 0, atol=1e-6)
    q = RNG.uniform(0, 1000, (50, 2))
    zc, var = R.ordinary_kriging(xy, np.full(40, -26.5), q, sill=1.0, a=300.0, nugget=0.01, k=16)
    assert np.allclose(zc, -26.5) and (var > 0).all()


def test_kriging_between_contours_matches_tin():
    from caspian import dem
    xy, z = _contours()
    keep = R.thin_points(xy, 120.0)
    assert 0 < len(keep) < len(xy)
    gx, gy = dem.grid_centers(2000, 11_000, 60, 60, 150.0)
    q = np.column_stack([gx.ravel(), gy.ravel()])
    truth = -28 + 1e-4 * q[:, 0] + 2e-9 * (q[:, 1] - 6000) ** 2
    cov = R.fit_covariance(xy[keep], z[keep], bin_width=500.0, max_lag=8000.0)
    assert cov["sill"] > 0 and cov["a"] > 0
    zk, var = R.ordinary_kriging(xy[keep], z[keep], q, sill=cov["sill"], a=cov["a"], nugget=1e-4, k=32,
                                 per_quadrant=8, chunk=1000)
    zt = dem.tin(xy, z, gx, gy).ravel()
    ok = np.isfinite(zt)
    rmse_k = np.sqrt(np.mean((zk[ok] - truth[ok]) ** 2))
    rmse_t = np.sqrt(np.mean((zt[ok] - truth[ok]) ** 2))
    assert rmse_k < 0.02 and rmse_t < 0.02
    # the quadrant rule makes a cell between two contours use both: without it, all
    # neighbours come from the nearest line
    zn, _ = R.ordinary_kriging(xy, z, q, sill=cov["sill"], a=cov["a"], nugget=1e-4, k=16, per_quadrant=None,
                               chunk=1000)
    assert np.sqrt(np.mean((zn[ok] - truth[ok]) ** 2)) > rmse_k


# --- validation variants ----------------------------------------------------------------------

SIZE = 25_000


def _blocks(n=10, size=SIZE):
    from caspian import blocking
    b = blocking.block_grid((0, 0, n * size, n * size), size)
    b["sector"] = np.where(b["i"] < n // 2, "west", "volga")
    b["split"] = blocking.assign_splits(b)
    return b


def _points(n_pts=4000, seed=0, c_true=-1.25, n=10):
    rng = np.random.default_rng(seed)
    x, y = rng.uniform(0, n * SIZE, n_pts), rng.uniform(0, n * SIZE, n_pts)
    truth = -27.0 + 2e-5 * x - 1e-5 * y
    d = rng.uniform(0, 3000, n_pts)
    pts = pd.DataFrame({"x": x, "y": y, "h_orth": truth - c_true + rng.normal(0, 0.01, n_pts),
                        "dem_h": truth + rng.normal(0, 0.03 * np.exp(0.4 * np.log1p(d))), "interp_dist": d,
                        "contour_density": rng.uniform(0, 50, n_pts), "slope": rng.uniform(0, 0.01, n_pts),
                        "prod:etopo_2022": truth - c_true + rng.normal(0, 0.8, n_pts)})
    return validation.attach_blocks(pts, _blocks(n), SIZE)


PRODUCTS = {"etopo_2022": "prod:etopo_2022"}


def test_run_fixed_equals_validation_run_and_shifts_bias():
    pts = _points()
    a = validation.run(pts, PRODUCTS, n_boot=40)
    b = R.run_fixed(pts, PRODUCTS, n_boot=40)
    pd.testing.assert_frame_equal(a["accuracy"], b["accuracy"])
    pd.testing.assert_frame_equal(a["coverage"], b["coverage"])
    pd.testing.assert_frame_equal(a["paired_bootstrap"], b["paired_bootstrap"])
    c = R.shifted_constant(a["constant"], 1.0)
    s = R.run_fixed(pts, PRODUCTS, constant=c, n_boot=40)
    am = a["accuracy"].set_index(["scope", "model"])
    sm = s["accuracy"].set_index(["scope", "model"])
    delta = c["constant_m"] - a["constant"]["constant_m"]
    assert delta > 0
    assert np.allclose(sm.loc[("overall", "dem"), "bias"], am.loc[("overall", "dem"), "bias"] - delta)
    assert np.isclose(sm.loc[("overall", "dem"), "nmad"], am.loc[("overall", "dem"), "nmad"])
    off = R.shifted_offsets(a["product_offsets"], -1.0)
    assert off["etopo_2022"]["offset_m"] < a["product_offsets"]["etopo_2022"]["offset_m"]
    h = R.validation_headline(a)
    cmp = R.compare_headlines(h, R.validation_headline(s))
    row = cmp[(cmp["table"] == "accuracy") & (cmp["scope"] == "overall") & (cmp["item"] == "dem")
              & (cmp["quantity"] == "bias")]
    assert np.isclose(row["delta"].iloc[0], -delta)


def test_split_variants():
    b = _blocks()
    r = R.reassign_splits(b, seed=1)
    assert sorted(r["split"].value_counts().to_dict().items()) == sorted(b["split"].value_counts().to_dict().items())
    assert not (r["split"] == b["split"]).all()
    pts = _points(500)
    sw = R.swap_splits(pts)
    assert ((pts["split"] == "test") == (sw["split"] == "calib")).all()
    assert ((pts["split"] == "train") == (sw["split"] == "train")).all()
    big = _blocks(n=5, size=50_000)
    pts50 = validation.attach_blocks(pts[["x", "y"]].copy(), big, 50_000)
    kept = R.exclude_initial_training(pts50, b, SIZE)
    i, j = (np.floor(kept["x"] / SIZE).astype(int), np.floor(kept["y"] / SIZE).astype(int))
    tr = set(zip(b.loc[b["split"] == "train", "i"], b.loc[b["split"] == "train", "j"]))
    in_tr = np.array([(a, c) in tr for a, c in zip(i, j)])
    assert not (in_tr & kept["split"].isin(["calib", "test"]).to_numpy()).any()
    assert (kept["split"] == "train").sum() == (pts50["split"] == "train").sum()


def test_dem_columns_from_variant_tiles(tmp_path):
    import rasterio
    from rasterio.transform import Affine
    from caspian.config import CRS_EQUAL_AREA
    import pyproj
    tile, res = 600.0, 30.0
    n = int(tile / res)
    for i in (0, 1):
        h = np.full((n, n), -27.0 + i, np.float32)
        d = np.full((n, n), 100.0 * (i + 1), np.float32)
        with rasterio.open(tmp_path / f"dem_{i}_0.tif", "w", driver="GTiff", width=n, height=n, count=2,
                           dtype="float32", crs=pyproj.CRS(CRS_EQUAL_AREA).to_wkt(),
                           transform=Affine(res, 0, i * tile, 0, -res, tile), nodata=np.nan) as dst:
            dst.write(h, 1)
            dst.write(d, 2)
    pts = pd.DataFrame({"x": [100.0, 700.0, 5000.0], "y": [300.0, 300.0, 300.0], "dem_h": 0.0, "interp_dist": 0.0,
                        "slope": 1.0, "contour_density": 0.0, "h_orth": -26.0})
    out = R.dem_columns(pts, tmp_path, np.array([100.0, 110.0]), np.array([300.0, 300.0]), tile_m=tile)
    assert np.allclose(out["dem_h"][:2], [-27.0, -26.0]) and np.isnan(out["dem_h"][2])
    assert np.allclose(out["interp_dist"][:2], [100.0, 200.0])
    assert out["slope"][0] == pytest.approx(0.0)
    assert out["contour_density"][0] > 0 and out["contour_density"][1] == 0
    assert (out["h_orth"] == -26.0).all()


def test_tide_system_shift_matches_datums_numbers():
    lat = np.linspace(43.9, 47.5, 50)
    a, b = R.tide_system_shift(lat, "both_mean"), R.tide_system_shift(lat, "geoid_mean")
    # docs/DATUMS.md (D-073): +0.030 to +0.042 m (spread 0.013), +0.056 to +0.081 m (spread 0.024)
    assert a.min() > 0.029 and a.max() < 0.043 and np.ptp(a) < 0.014
    assert b.min() > 0.055 and b.max() < 0.082 and np.ptp(b) < 0.025
    with pytest.raises(ValueError):
        R.tide_system_shift(lat, "zero")


# --- hindcast -----------------------------------------------------------------------------------------

def test_write_correction_reads_back(tmp_path):
    from caspian import products
    lon, lat = np.linspace(46.3, 47.0, 8), np.linspace(44.0, 44.5, 6)
    a = RNG.normal(0, 0.1, (6, 8)).astype(np.float32)
    np.savez_compressed(tmp_path / "g.npz", lon=lon, lat=lat, dN_m=a, dN_ngaEGM2008_m=a - 0.405)
    rec = R.write_correction(tmp_path / "g.npz", "dN_m", tmp_path / "c.npz")
    nat, meta = products.load_correction(tmp_path / "c.npz")
    assert np.allclose(nat.z, a) and "dN_m" in meta["definition"] and rec["layer"] == "dN_m"
    with pytest.raises(KeyError):
        R.write_correction(tmp_path / "g.npz", "dN_x", tmp_path / "d.npz")


def _hindcast_tables(tmp_path, shift=0.0):
    disp, nf, ext = [], [], []
    for b in (40, 41, 42):
        for s in ("west", "volga"):
            for prod, med in (("gebco_2026", 900.0 + shift), ("etopo_2022", 100.0)):
                disp.append({"product": prod, "variant": "shape", "epoch": "pooled", "bin": b, "sector": s,
                             "median_m": med, "n": 50, "observed_level": True})
                for ep, m in (("E1", med * 0.5), ("E2", med), ("E3", med * 1.1)):
                    disp.append({"product": prod, "variant": "shape", "epoch": ep, "bin": b, "sector": s,
                                 "median_m": m, "n": 20, "observed_level": True})
            nf.append({"kind": "pooled_level", "bin": b, "sector": s, "p95_m": 300.0, "n": 40, "median_m": 100.0})
            nf.append({"kind": "e1_vs_e23", "bin": b, "sector": s, "p95_m": 500.0, "n": 40, "median_m": 450.0})
        for prod in ("gebco_2026", "etopo_2022"):
            for ep in ("E1", "E2"):
                ext.append({"product": prod, "variant": "shape", "epoch": ep, "bin": b, "sector": "domain",
                            "area_bias_total_km2": 10.0 if prod == "gebco_2026" else -1.0,
                            "area_bias_rel_total": 0.1, "iou_total": 0.9})
    for s in ("west", "volga"):
        nf.append({"kind": "pooled_all_levels", "bin": np.nan, "sector": s, "p95_m": 1000.0, "n": 120,
                   "median_m": 100.0})
    d = tmp_path
    pd.DataFrame(disp).to_csv(d / "displacement.csv", index=False)
    pd.DataFrame(nf).to_csv(d / "noise_floor.csv", index=False)
    pd.DataFrame(ext).to_csv(d / "extent.csv", index=False)
    return pd.DataFrame(disp), pd.DataFrame(nf)


def test_h1_headline_and_comparison(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    _hindcast_tables(a)
    _hindcast_tables(b, shift=-700.0)
    ha, hb = R.h1_headline(a), R.h1_headline(b)
    m = ha["mispredicts"].set_index(["product", "sector"])["mispredicts"]
    assert m.loc[("gebco_2026", "west")] and not m.loc[("etopo_2022", "west")]
    assert not ha["mispredicts_pooled_floor"]["mispredicts"].any()
    assert ha["area"].set_index("product").loc["gebco_2026", "sum_abs_area_bias_km2"] == pytest.approx(60.0)
    c = R.compare_h1(ha, hb)
    assert c.loc[c["product"] == "gebco_2026", "changed"].all()
    assert not c.loc[c["product"] == "etopo_2022", "changed"].any()
    only_main = {**hb, "mispredicts": hb["mispredicts"][hb["mispredicts"]["product"] == "etopo_2022"]}
    c2 = R.compare_h1(ha, only_main)
    assert not c2.loc[c2["product"] == "gebco_2026", "in_both"].any()
    assert not c2.loc[c2["product"] == "gebco_2026", "changed"].any()


def test_morphology_table(tmp_path):
    disp, nf = _hindcast_tables(tmp_path)
    det, summ = R.morphology_table(disp, nf)
    assert det["e1_better"].all()
    g = summ.set_index(["product", "sector"])
    assert (g["n_bins"] == 3).all() and (g["n_e1_better"] == 3).all()
    assert g.loc[("gebco_2026", "west"), "median_ratio"] == pytest.approx((900 + 990) / 2 / 450)


def test_area_level_slope():
    lev = np.linspace(-28.9, -26.1, 29)
    obs = pd.DataFrame({"epoch": "E3", "sample": "full", "sector": "all", "level_m": lev,
                        "area_sea_km2": 7000.0 * (lev + 30) + RNG.normal(0, 5, lev.size),
                        "area_valid_km2": 50_000.0})
    obs.loc[0, "area_valid_km2"] = 10_000.0       # a cloud-holed unit
    r = R.area_level_slope(obs, n_boot=200)
    assert r["slope_km2_per_m"] == pytest.approx(7000, rel=0.01) and r["n_left_out"] == 1
    assert r["ci_low"] < 7000 < r["ci_high"]
    assert R.area_level_slope(obs, lo=-20, hi=-19)["n_units"] == 0


# --- sampling and cost -----------------------------------------------------------------------------

def test_block_sample():
    b = pd.DataFrame({"i": np.arange(30), "j": 0, "sector": np.repeat(["west", "volga", "ural"], 10),
                      "split": np.tile(["test", "train"], 15)})
    cov = np.where(b["i"] % 4 == 0, 50.0, 300.0)
    s1 = R.block_sample(b, cov, n_per_sector=2)
    s2 = R.block_sample(b.sample(frac=1, random_state=3), pd.Series(cov, index=b.index).loc[
        b.sample(frac=1, random_state=3).index].to_numpy(), n_per_sector=2)
    assert set(zip(s1["i"], s1["j"])) == set(zip(s2["i"], s2["j"]))
    assert (s1["split"] == "test").all() and (s1["cover_km2"] >= 100).all()
    assert s1.groupby("sector").size().to_dict() == {"ural": 2, "volga": 2, "west": 2}
    assert R.eecu_hours(1_000_000, 0.0036) == pytest.approx(1.0)


def test_substitute_window_replaces_only_the_region():
    main = {"w_E2_b001_p0": np.zeros((6, 8), np.uint16), "v_E2_b001_p0": np.full((6, 8), 5, np.uint16)}
    alt = {"w_E2_b001_p0": np.full((4, 4), 3, np.uint16), "v_E2_b001_p0": np.full((4, 4), 7, np.uint16)}
    reg = np.zeros((4, 4), bool)
    reg[1:, 1:] = True
    tr_main, tr_alt = [60, 0, 0, 0, -60, 600], [60, 0, 300, 0, -60, 540]   # alt starts at col 5, row 1
    out, n = R.substitute_window(main, tr_main, alt, tr_alt, reg)
    assert n == 3 * 2                                   # alt cols 6-8 clipped to the main's 8 columns
    assert out["v_E2_b001_p0"][2:5, 6:8].tolist() == [[7, 7]] * 3
    assert out["v_E2_b001_p0"].sum() == 5 * 48 + 2 * 6 and (out["w_E2_b001_p0"] > 0).sum() == 6
    with pytest.raises(ValueError, match="aligned"):
        R.substitute_window(main, tr_main, alt, [60, 0, 310, 0, -60, 540], reg)
    with pytest.raises(ValueError, match="not in main"):
        R.substitute_window(main, tr_main, {**alt, "w_E2_b002_p0": alt["w_E2_b001_p0"]}, tr_alt, reg)
    part, _ = R.substitute_window(main, tr_main, {"w_E2_b001_p0": alt["w_E2_b001_p0"]}, tr_alt, reg)
    assert np.array_equal(part["v_E2_b001_p0"], main["v_E2_b001_p0"])


def test_contour_offsets():
    y = np.arange(0, 1000, 10.0)
    a = pd.DataFrame({"epoch": "E3", "bin": 40, "x": 0.0, "y": y})
    b = pd.DataFrame({"epoch": "E3", "bin": 40, "x": 30.0, "y": y})
    c = R.contour_offsets(a, pd.concat([b, b.assign(bin=41)]))
    assert len(c) == 1 and c["median_m"].iloc[0] == pytest.approx(30.0) and c["n_b"].iloc[0] == len(y)


def test_load_cover_is_valid(tmp_path):
    import json
    import shapely
    from shapely.geometry import mapping, Polygon
    bow = Polygon([(50.0, 45.0), (50.2, 45.2), (50.2, 45.0), (50.0, 45.2)])   # self-intersecting
    (tmp_path / "c.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {}, "geometry": mapping(bow)}]}), encoding="utf-8")
    from caspian.config import CRS_EQUAL_AREA
    g = R.load_cover(tmp_path / "c.geojson")
    h = R.load_cover(tmp_path / "c.geojson", CRS_EQUAL_AREA)
    assert g.is_valid and h.is_valid and h.area > 1e8


# --- robustness scripts on synthetic files (temporary directories only) ------------------------

import importlib.util as _ilu
from pathlib import Path as _Path

_SCRIPTS = _Path(__file__).resolve().parents[1] / "scripts"


def _script(name):
    spec = _ilu.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    m = _ilu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _write_main_tag(root, acq, masks, tag="main", key="0_0", transform=(60, 0, 0, 0, -60, 1200)):
    arrays = {k: np.asarray(v, np.uint16) for k, v in _counts(acq, masks, _main_key).items()}
    R.write_tile(root / tag / f"tile_{key}.npz", list(transform), arrays)
    return arrays


def test_r00_counts_build_rebin_and_vegetation(tmp_path):
    C = _script("r00_counts")
    acq, masks = _acquisitions(n=40)
    arrays = _write_main_tag(tmp_path, acq, masks)
    rec = C.build("bw020", ["main"], bin_factor=2, root=tmp_path)
    assert rec["bin_width_m"] == pytest.approx(0.20)
    assert C.bin_width_of("bw020", tmp_path) == pytest.approx(0.20) and C.bin_width_of("main", tmp_path) == 0.10
    _, got = R.load_tile(tmp_path / "bw020" / "tile_0_0.npz")
    want = R.merge_groups(arrays, R.rebin_mapper(2))
    assert set(got) == set(want) and all(np.array_equal(got[k], want[k]) for k in want)
    C.build("veg", ["main"], vegetation="as_dry", root=tmp_path)
    _, veg = R.load_tile(tmp_path / "veg" / "tile_0_0.npz")
    g = next(k for k in arrays if k.startswith("o_"))[2:]
    assert np.array_equal(veg[f"v_{g}"], arrays[f"o_{g}"]) and not any(k.startswith("o_") for k in veg)
    with pytest.raises(SystemExit):
        C.build("bw020", ["main"], bin_factor=2, root=tmp_path)      # never overwrites a variant


def test_r00_counts_sum_and_delta(tmp_path):
    C = _script("r00_counts")
    acq, masks = _acquisitions(n=40, seed=4)
    a_l, a_s = acq[acq["sensor"] != "MSI_S2"], acq[acq["sensor"] == "MSI_S2"]
    _write_main_tag(tmp_path, a_l, masks, tag="mainL")
    _write_main_tag(tmp_path, a_s, masks, tag="mainS")
    full = {k: np.asarray(v, np.uint16) for k, v in _counts(acq, masks, _main_key).items()}
    C.build("LS", ["mainL", "mainS"], root=tmp_path)
    _, got = R.load_tile(tmp_path / "LS" / "tile_0_0.npz")
    assert set(got) == set(full) and all(np.array_equal(got[k], full[k]) for k in full)
    # removing the S2 scenes again as a SUB delta gives the Landsat counts
    C.build("L_again", ["LS"], sub="mainS", root=tmp_path)
    _, back = R.load_tile(tmp_path / "L_again" / "tile_0_0.npz")
    _, ml = R.load_tile(tmp_path / "mainL" / "tile_0_0.npz")
    for k in ml:
        assert np.array_equal(back[k], ml[k])


def test_r00_pipeline_points_step(tmp_path, monkeypatch):
    import json
    P = _script("r00_pipeline")
    C = P.script("r00_counts")
    interim, results = tmp_path / "interim", tmp_path / "results"
    monkeypatch.setattr(P, "INTERIM", interim)
    monkeypatch.setattr(P, "RESULTS", results)
    acq, masks = _acquisitions(n=60, shape=(30, 40))
    _write_main_tag(interim / "stage2", acq, masks)
    C.build("main__bw020", ["main"], bin_factor=2, root=interim / "stage2")
    (results / "levels").mkdir(parents=True)
    (results / "levels" / "static_contour_rule.json").write_text(json.dumps({"tag": "main", "rho_star": 0.5}),
                                                                   encoding="utf-8")
    path = P.points_step("main__bw020", "main")
    pts = pd.read_csv(path)
    assert len(pts) > 0 and {"x", "y", "epoch", "bin", "rho_level", "z_m", "static"} <= set(pts.columns)
    assert np.allclose(pts["z_m"], levels.bin_center(pts["bin"], 0.20, design.BIN_ORIGIN_M))
    assert (pts["static"] == (pts["rho_level"] < 0.5)).all()
    with pytest.raises(SystemExit):
        P.main_static_rule("other")


def test_r13_stricter_lag(tmp_path):
    import rasterio
    import pyproj
    from rasterio.transform import Affine
    from caspian.config import CRS_EQUAL_AREA
    r13 = _script("r13_emergence_lag")
    n, res = 20, 30.0
    em = np.full((n, n), 18_000.0, np.float32)            # emerged on day 18000 (2019-04-12)
    with rasterio.open(tmp_path / "emergence_0_0.tif", "w", driver="GTiff", width=n, height=n, count=1,
                       dtype="float32", crs=pyproj.CRS(CRS_EQUAL_AREA).to_wkt(),
                       transform=Affine(res, 0, 0, 0, -res, n * res), nodata=np.nan) as dst:
        dst.write(em, 1)
    ts = validation.TileSet(tmp_path, "emergence", tile_m=n * res, res=res)
    t0 = np.datetime64("1970-01-01")
    pts = pd.DataFrame({"x": [300.0, 300.0, 300.0], "y": [300.0, 300.0, 300.0],
                        "time": [t0 + np.timedelta64(d, "D") for d in (18_050, 18_400, 18_200)]})
    assert len(r13.stricter(pts, ts, "ATL06", 30.0)) == 3
    kept = r13.stricter(pts, ts, "ATL06", 365.0)
    assert len(kept) == 1 and kept["time"].iloc[0] == t0 + np.timedelta64(18_400, "D")


def test_r07_gp_tiles_follow_the_surface(tmp_path, monkeypatch):
    import rasterio
    import pyproj
    from rasterio.transform import Affine
    from caspian import dem
    from caspian.config import CRS_EQUAL_AREA
    r07 = _script("r07_interpolation")
    monkeypatch.setattr(r07, "PROCESSED", tmp_path)
    xy, z = _contours()
    pts = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "z_m": z, "epoch": "E3",
                        "bin": np.repeat(np.arange(8), len(z) // 8)})
    t = r07.thinned(pts, 120.0)
    assert 0 < len(t) < len(pts)
    res, n = 150.0, 60
    gx, gy = dem.grid_centers(2000, 11_000, n, n, res)
    h = dem.tin(xy, z, gx, gy).astype(np.float32)
    (tmp_path / "dem" / "m").mkdir(parents=True)
    with rasterio.open(tmp_path / "dem" / "m" / "dem_0_0.tif", "w", driver="GTiff", width=n, height=n, count=2,
                       dtype="float32", crs=pyproj.CRS(CRS_EQUAL_AREA).to_wkt(),
                       transform=Affine(res, 0, 2000, 0, -res, 11_000), nodata=np.nan) as dst:
        dst.write(h, 1)
        dst.write(np.ones_like(h), 2)
    cov = R.fit_covariance(t[["x", "y"]].to_numpy(), t["z_m"].to_numpy(), bin_width=500.0, max_lag=8000.0)
    assert r07.gp_tiles("m", "m__gp", t, cov, nugget=1e-4, chunk=1000) == 1
    with rasterio.open(tmp_path / "dem" / "m__gp" / "dem_0_0.tif") as r:
        g, sd = r.read(1), r.read(3)
    truth = -28 + 1e-4 * gx + 2e-9 * (gy - 6000) ** 2
    ok = np.isfinite(h)
    assert np.array_equal(np.isfinite(g), ok)
    assert np.sqrt(np.nanmean((g[ok] - truth[ok]) ** 2)) < 0.02 and np.nanmin(sd) >= 0


def test_r02_pair_list():
    r02 = _script("r02_sensor")
    acq, _ = _acquisitions(n=10, seed=9)
    plan = _plan_from_acq(acq, scenes_per_acq=1)
    for s in SECTOR_ORDER:
        plan[f"setup_{s}_m"] = 0.0
    # two same-day acquisitions of different sensors at the same level
    extra = plan.iloc[[0]].copy()
    extra["acq"], extra["scene"] = "B000", "B000_0"
    extra["sensor"] = "MSI_S2" if plan.iloc[0]["sensor"] != "MSI_S2" else "OLI_L8"
    extra["t"] = plan.iloc[0]["t"] + pd.Timedelta(hours=2)
    plan = pd.concat([plan, extra], ignore_index=True)
    plan["t"] = plan["t"].astype(str)
    p = r02.pair_list(plan, tau=0.15, n=5, max_hours=24, max_dlevel=0.01)
    assert len(p) >= 1 and set(p.iloc[0][["acq_a", "acq_b"]]) == {"A000", "B000"}
    plan.loc[plan["acq"] == "B000", "setup_west_m"] = 0.5     # windy: the pair is dropped
    p2 = r02.pair_list(plan, tau=0.15, n=5, max_hours=24, max_dlevel=0.01)
    if len(p2):
        assert not ((p2["acq_a"] == "A000") & (p2["acq_b"] == "B000")).any()


def test_r04_point_sectors_and_alternatives():
    import pyproj
    from caspian.config import CRS_EQUAL_AREA
    r04 = _script("r04_screening")
    fwd = pyproj.Transformer.from_crs("EPSG:4326", CRS_EQUAL_AREA, always_xy=True)
    x, y = fwd.transform([47.6, 52.0], [45.5, 46.9])         # Kizlyar-Dagestan west; Ural delta area
    sec = r04.point_sectors(np.array(x), np.array(y))
    assert all(s in SECTOR_ORDER for s in sec)
    _, _, plan = _setup_plan()
    a = r04.alt_exclusion(plan, 0.15, "tau:0.2")
    assert np.array_equal(a, R.exclusion(plan, 0.2))
    k = r04.alt_exclusion(plan, 0.15, "komsomolets_unscreened")
    assert not k[:, SECTOR_ORDER.index("komsomolets")].any()
    with pytest.raises(SystemExit):
        r04.alt_exclusion(plan, 0.15, "nonsense")


def test_r00_pipeline_dem_difference(tmp_path, monkeypatch):
    import rasterio
    import pyproj
    from rasterio.transform import Affine
    from caspian.config import CRS_EQUAL_AREA
    P = _script("r00_pipeline")
    monkeypatch.setattr(P, "PROCESSED", tmp_path / "processed")
    monkeypatch.setattr(P, "ROBUST", tmp_path / "robust")
    fwd = pyproj.Transformer.from_crs("EPSG:4326", CRS_EQUAL_AREA, always_xy=True)
    x0, y0 = fwd.transform(50.0, 46.0)
    x0, y0 = np.floor(x0 / 30) * 30, np.floor(y0 / 30) * 30
    n = 50
    for tag, off in (("m", 0.0), ("v", 0.05)):
        d = tmp_path / "processed" / "dem" / tag
        d.mkdir(parents=True)
        h = np.full((n, n), -27.0 + off, np.float32)
        if tag == "v":
            h[0, :5] = np.nan
        with rasterio.open(d / "dem_0_0.tif", "w", driver="GTiff", width=n, height=n, count=1, dtype="float32",
                           crs=pyproj.CRS(CRS_EQUAL_AREA).to_wkt(), transform=Affine(30, 0, x0, 0, -30, y0),
                           nodata=np.nan) as dst:
            dst.write(h, 1)
    t = P.dem_difference("c", "v", "m").set_index("sector")
    assert t.loc["overall", "n_cells"] == n * n - 5 and t.loc["overall", "cells_only_main"] == 5
    assert t.loc["overall", "mean_m"] == pytest.approx(0.05, abs=1e-6)
    assert t.loc["overall", "median_m"] == pytest.approx(0.05, abs=1e-3)
