"""caspian.watersurface: SWOT heights on the working datum, sector departures, shifted
flood-level maps and the planar offset helpers (A18 item 9)."""

import numpy as np
import pandas as pd
import pytest

from caspian import hindcast_tiled as ht
from caspian import watersurface as ws


def test_free2mean_matches_the_icesat2_formulas():
    g, e = ws.free2mean([0.0, 90.0, 45.0])
    assert np.allclose(g, [0.1287, 0.1287 - 0.3848, 0.1287 - 0.3848 / 2])
    assert np.allclose(e, [0.06029, 0.06029 - 0.180873, 0.06029 - 0.180873 / 2])


def test_swot_to_working_tide_systems():
    # tide-free working datum: wse + Ns - Ne + g - e; mean tide: wse + Ns - Ne
    lat = 45.5
    g, e = ws.free2mean(lat)
    h_tf = ws.swot_to_working(-28.0, -10.4, -10.0, lat, "tide_free")
    h_mt = ws.swot_to_working(-28.0, -10.4, -10.0, lat, "mean_tide")
    assert h_mt == pytest.approx(-28.4)
    assert h_tf == pytest.approx(-28.4 + g - e)
    assert -0.040 < g - e < -0.030          # about -0.035 m at 45.5 N
    with pytest.raises(ValueError):
        ws.swot_to_working(0, 0, 0, lat, "zero_tide")


def test_swot_to_working_is_consistent_with_a_round_trip():
    # a tide-free height H on the working datum -> SWOT wse -> back
    rng = np.random.default_rng(1)
    lat = rng.uniform(44, 47, 50)
    H, ns, ne = rng.normal(-28, 1, 50), rng.normal(-10, 3, 50), rng.normal(-10, 3, 50)
    g, e = ws.free2mean(lat)
    h_free = H + ne                      # ellipsoidal, tide-free
    h_mean = h_free + e                  # surface in the mean tide system
    wse = h_mean - (ns + g)              # above SWOT's mean-tide geoid
    assert np.allclose(ws.swot_to_working(wse, ns, ne, lat), H)


def _bp():
    # sector A: blocks 1 and 2 in passes 1-3; sector B: one block in 2 passes
    return pd.DataFrame({"pass_id": [1, 1, 2, 2, 3, 3, 1, 2],
                         "block_i": [1, 2, 1, 2, 1, 2, 5, 5], "block_j": [0, 0, 0, 0, 0, 0, 0, 0],
                         "sector": ["A"] * 6 + ["B"] * 2,
                         "dep_m": [0.0, 0.2, 0.4, 0.6, 1.0, 1.2, -0.5, -0.3]})


def test_sector_departures_median_over_pass_means():
    t = ws.sector_departures(_bp(), min_blocks=2, min_passes=3).set_index("sector")
    # pass means of A: 0.1, 0.5, 1.1 -> median 0.5; IQR of [0.1, 0.5, 1.1] = 0.8 - 0.3
    assert t.loc["A", "departure_m"] == pytest.approx(0.5)
    assert t.loc["A", "iqr_m"] == pytest.approx(0.5)
    assert (t.loc["A", "n_passes"], t.loc["A", "n_blocks"], t.loc["A", "n_block_passes"]) == (3, 2, 6)
    # B has one block and two passes: too thin, no value
    assert np.isnan(t.loc["B", "departure_m"]) and "too thin" in t.loc["B", "note"]
    assert (t.loc["B", "n_passes"], t.loc["B", "n_blocks"]) == (2, 1)


def test_sector_departures_ignores_nan_and_checks_columns():
    bp = _bp()
    bp.loc[0, "dep_m"] = np.nan
    t = ws.sector_departures(bp, min_blocks=1, min_passes=1).set_index("sector")
    assert t.loc["A", "departure_m"] == pytest.approx(0.5)   # pass 1 mean is now 0.2
    with pytest.raises(ValueError):
        ws.sector_departures(bp.drop(columns="sector"))


def test_departure_lut_and_shifted_flood_map_equal_a_per_sector_threshold():
    rng = np.random.default_rng(0)
    F = rng.normal(-27, 1, (40, 60)).astype(np.float32)
    F[0, 0] = np.inf
    sector = rng.integers(0, 4, F.shape).astype(np.uint8)
    names = ["a", "b", "c"]
    d = {"a": 0.3, "b": -0.2, "c": np.nan}
    lut = ws.departure_lut(d, names, default=0.05)
    assert np.allclose(lut, [0.05, 0.3, -0.2, 0.05])
    S = ws.ShiftedFloodMap(F, sector, lut)
    assert S.shape == F.shape
    L = -27.05
    got = np.zeros(F.shape, bool)
    for r0, r1, c0, c1 in ht.tiles(F.shape, 16):
        got[r0:r1, c0:c1] = ht.flooded(S, L)(r0, r1, c0, c1)
    want = F.astype(float) < L + lut[sector]
    assert np.array_equal(got, want)
    assert np.isinf(S[0:1, 0:1]).all()
    with pytest.raises(ValueError):
        ws.ShiftedFloodMap(F, sector[:-1], lut)


def test_plane_values_and_profile_c0():
    assert ws.plane_values(1.0, 0.01, -0.02, 100.0, 50.0) == pytest.approx(1.0 + 1.0 - 1.0)
    c = np.round(np.arange(-5, 6) / 10, 2)
    bias = np.array([(c - 0.2) * 10, -(c - 0.2) * 3])   # sum of |.| is smallest (0) at c = 0.2
    r = ws.profile_c0(bias, c)
    assert r["c0_m"] == pytest.approx(0.2)
    # ties: equal objective at -0.1 and +0.1 -> the smaller |c|, then the smaller c
    flat = np.zeros((1, c.size))
    flat[0, :] = np.where(np.isin(np.round(c, 2), [-0.1, 0.1]), 0, 5)
    assert ws.profile_c0(flat, c)["c0_m"] == pytest.approx(-0.1)


def test_quantised_counts_cumulative():
    q = np.array([0, 1, 1, 3, 5, 5])        # 5 = beyond the last bin
    out = ws.quantised_counts(q, None, 5)
    # out[m] = number of q < m for m = 0..5
    assert out.tolist() == [0, 1, 3, 3, 4, 4]


def test_ellipsoid_height_reproduces_the_comparison_guide_example():
    # ICESat-2 Data Comparison User's Guide R007, section 3.1.3: WGS84 (47.0, 15.0, 1200.0 m)
    # -> T/P (47.000000123 deg, 1200.7073059 m)
    lat, h = ws.ellipsoid_height(47.0, 15.0, 1200.0)
    assert lat == pytest.approx(47.000000123, abs=2e-9)
    assert h == pytest.approx(1200.7073059, abs=1e-6)
    # and back
    lat0, h0 = ws.ellipsoid_height(lat, 15.0, h, src=ws.TOPEX, dst=ws.WGS84)
    assert lat0 == pytest.approx(47.0, abs=1e-10) and h0 == pytest.approx(1200.0, abs=1e-6)


def test_zero_degree_term_gm_part_and_w0_part():
    # GM part only: model GM smaller than the reference -> negative term of about 5 mm
    n0 = ws.zero_degree_term(3.986004415e14, 3.986004418e14)
    assert -0.0055 < n0 < -0.0045
    # a W0 above U0 lowers the geoid by (W0 - U0) / gamma
    n1 = ws.zero_degree_term(3.986004418e14, 3.986004418e14, w0=62636856.0, u0=62636851.7146)
    assert n1 == pytest.approx(-(62636856.0 - 62636851.7146) / 9.80665)
