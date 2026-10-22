"""Hindcast metrics: waterlines next to unknown cells, censoring at the zone edge,
symmetric displacement and the half-sample noise floor.

Every case is synthetic with a known true displacement, so each assertion states
what the pre-registered metric (PREREG section 9) must return.
"""

import numpy as np
import pytest

from caspian import hindcast


def _band(shape, c0, c1):
    """Boolean mask that is True in columns c0..c1-1."""
    m = np.zeros(shape, bool)
    m[:, c0:c1] = True
    return m


# --- waterline definition -------------------------------------------------------------

def test_waterline_ignores_edges_against_unknown_cells():
    shape = (20, 60)
    valid = _band(shape, 10, 60)               # cols 0-9 outside the zone (unknown)
    obs = _band(shape, 10, 40)                 # observed water cols 10-39
    kw, kl = hindcast.known_states(obs, valid)
    wl = hindcast.waterline(kw, kl)
    assert np.array_equal(np.unique(np.nonzero(wl)[1]), [39])  # col 10 is not a waterline
    # the legacy helper still treats every non-water neighbour as land
    assert hindcast.shoreline(obs)[:, 10].all()


def test_shoreline_helper_unchanged():
    m = np.zeros((5, 5), bool)
    m[1:4, 1:4] = True
    expected = m.copy()
    expected[2, 2] = False                     # interior cell
    assert np.array_equal(hindcast.shoreline(m), expected)
    full = np.ones((4, 4), bool)               # array border is not a shoreline
    assert not hindcast.shoreline(full).any()


# --- finding 2: false edges at invalid cells ------------------------------------------

def test_too_little_water_measured_to_real_waterline_not_zone_edge():
    shape = (20, 60)
    valid = _band(shape, 10, 60)
    obs = _band(shape, 10, 40)
    pred = _band(shape, 0, 20)                 # product waterline 20 px seaward of col 39
    d = hindcast.shoreline_displacement(pred, obs, valid)
    assert d["median"] == pytest.approx(20.0)
    assert d["p95"] == pytest.approx(20.0)
    assert d["signed_median"] == pytest.approx(-20.0)


@pytest.mark.parametrize("mask_obs", [True, False])
def test_cloud_hole_inside_water_does_not_shrink_displacement(mask_obs):
    shape = (30, 60)
    obs = _band(shape, 0, 40)
    valid = np.ones(shape, bool)
    valid[10:14, 20:24] = False                # 4x4 hole inside observed water
    if mask_obs:
        obs = obs & valid                      # PREREG section 9: extent within the valid mask
    pred = _band(shape, 0, 30)                 # true error: 10 px too little water
    d = hindcast.shoreline_displacement(pred, obs, valid)
    # the HEAD code gave a median of 7.81 for the masked case (distance to the hole rim)
    assert d["median"] == pytest.approx(10.0)
    assert d["p95"] == pytest.approx(10.0)
    assert d["signed_median"] == pytest.approx(-10.0)


def test_noise_floor_has_no_zeros_from_shared_false_edge():
    shape = (20, 60)
    valid = _band(shape, 10, 60)               # same zone for both halves
    a = _band(shape, 10, 40)
    b = _band(shape, 10, 42)                   # half-samples 2 px apart
    nf = hindcast.noise_floor_displacement(a, valid, b, valid, return_distances=True)
    assert nf["median"] == pytest.approx(2.0)
    assert nf["p95"] == pytest.approx(2.0)
    assert np.all(np.abs(nf["distances"]) == pytest.approx(2.0))  # no zero from col 10


# --- finding 9: predicted waterline outside the valid mask ----------------------------

def _zone_case(pred_cols, width=80):
    shape = (20, width)
    valid = _band(shape, 10, 51)               # zone cols 10-50
    obs_full = _band(shape, 0, 40)             # truth: water up to col 39
    obs = obs_full & valid
    outside = np.full(shape, np.nan)
    outside[:, :10] = 1.0                      # seaward of the zone: water
    outside[:, 51:] = 0.0                      # landward of the zone: land
    pred = _band(shape, 0, pred_cols)
    return pred, obs, valid, outside


@pytest.mark.parametrize("pred_cols,true_px", [(56, 16), (71, 31), (5, -35)])
def test_prediction_beyond_zone_edge_is_measured(pred_cols, true_px):
    pred, obs, valid, outside = _zone_case(pred_cols)
    # without known outside states: no NaN, the observed->predicted half still measures it
    d = hindcast.shoreline_displacement(pred, obs, valid)
    assert d["n"] > 0
    assert d["median"] == pytest.approx(abs(true_px))
    assert d["signed_median"] == pytest.approx(true_px)
    assert d["n_pred_unknown"] == 20           # censored half is counted, not hidden
    # with known outside states both directions are measured
    d2 = hindcast.shoreline_displacement(pred, obs, valid, outside_state=outside)
    assert d2["median"] == pytest.approx(abs(true_px))
    assert d2["signed_median"] == pytest.approx(true_px)
    assert d2["n_pred_unknown"] == 0
    assert d2["n"] == 40


def test_partial_excursion_out_of_zone_not_biased_low():
    shape = (40, 100)
    valid = _band(shape, 10, 51)
    obs = _band(shape, 0, 40) & valid
    outside = np.full(shape, np.nan)
    outside[:, :10] = 1.0
    outside[:, 51:] = 0.0
    pred = np.zeros(shape, bool)
    pred[:20, :44] = True                      # +4 px in the top half
    pred[20:, :70] = True                      # +30 px in the bottom half (outside the zone)
    d = hindcast.shoreline_displacement(pred, obs, valid, outside_state=outside,
                                        return_distances=True)
    dist = np.abs(d["distances"])
    assert np.sum(dist >= 29.5) >= 20          # the 30 px errors outside the zone are kept
    assert d["p95"] >= 29.5


def test_no_predicted_waterline_is_infinite_not_nan():
    shape = (20, 60)
    obs = _band(shape, 0, 30)
    valid = np.ones(shape, bool)
    dry = hindcast.shoreline_displacement(np.zeros(shape, bool), obs, valid)
    assert dry["median"] == np.inf and dry["signed_median"] == -np.inf and dry["n"] > 0
    wet = hindcast.shoreline_displacement(np.ones(shape, bool), obs, valid)
    assert wet["median"] == np.inf and wet["signed_median"] == np.inf


def test_no_observed_waterline_is_nan():
    shape = (10, 10)
    d = hindcast.shoreline_displacement(_band(shape, 0, 5), np.ones(shape, bool),
                                        np.ones(shape, bool))
    assert np.isnan(d["median"]) and d["n"] == 0


def test_extent_metrics_counts_errors_outside_valid():
    for pred_cols, excess_cols in [(56, 16), (71, 31), (5, -35)]:
        pred, obs, valid, outside = _zone_case(pred_cols)
        m = hindcast.extent_metrics(pred, obs, valid, cell_area=2.0, outside_state=outside)
        assert m["area_bias_total"] == pytest.approx(excess_cols * 20 * 2.0)
        assert m["area_bias_rel_total"] == pytest.approx(m["area_bias_total"] / m["area_obs"])
    # within-valid keys keep their meaning (capped at the zone edge) for comparison
    m55 = hindcast.extent_metrics(*_zone_case(56)[:3])
    m70 = hindcast.extent_metrics(*_zone_case(71)[:3])
    assert m55["area_bias"] == m70["area_bias"]
    assert "area_bias_total" not in m55
    p71, o71, v71, out71 = _zone_case(71)
    m_out = hindcast.extent_metrics(p71, o71, v71, outside_state=out71)
    assert m_out["iou_total"] < m_out["iou"]
    assert m_out["area_fp_outside"] == pytest.approx(20 * 20)


def test_region_limits_counts_to_one_sector():
    shape = (40, 100)
    valid = np.ones(shape, bool)
    obs = _band(shape, 0, 40)
    pred = np.zeros(shape, bool)
    pred[:20, :45] = True                      # sector A (top): +5 px
    pred[20:, :60] = True                      # sector B (bottom): +20 px
    top = np.zeros(shape, bool)
    top[:20] = True
    d_top = hindcast.shoreline_displacement(pred, obs, valid, region=top)
    assert d_top["median"] == pytest.approx(5.0)
    assert d_top["p95"] == pytest.approx(5.0)
    assert d_top["n"] == 40                    # 20 predicted + 20 observed waterline cells
    d_bot = hindcast.shoreline_displacement(pred, obs, valid, region=~top)
    assert d_bot["median"] > 10.0
    assert hindcast.extent_metrics(pred, obs, valid, region=top)["area_bias"] == pytest.approx(5 * 20)
    assert hindcast.extent_metrics(pred, obs, valid, region=~top)["area_bias"] == pytest.approx(20 * 20)


# --- finding 10: symmetric displacement ------------------------------------------------

def _bay_case():
    shape = (30, 60)
    obs = _band(shape, 0, 10)
    obs[8:13, 10:40] = True                    # 5 x 30 bay the product misses
    pred = _band(shape, 0, 10)
    return pred, obs, np.ones(shape, bool)


def test_missed_bay_is_penalised():
    pred, obs, valid = _bay_case()
    d = hindcast.shoreline_displacement(pred, obs, valid)
    assert d["p95"] > 20.0                     # the bay waterline is up to 30 px from the product's
    assert d["median"] > 1.0
    assert d["signed_median"] < 0.0            # too little water
    m = hindcast.extent_metrics(pred, obs, valid)
    assert m["area_bias"] < 0


def test_displacement_is_symmetric_under_swap():
    pred, obs, valid = _bay_case()
    d1 = hindcast.shoreline_displacement(pred, obs, valid, return_distances=True)
    d2 = hindcast.shoreline_displacement(obs, pred, valid, return_distances=True)
    assert d1["median"] == pytest.approx(d2["median"])
    assert d1["p95"] == pytest.approx(d2["p95"])
    assert d1["signed_median"] == pytest.approx(-d2["signed_median"])
    assert np.allclose(np.sort(d1["distances"]), np.sort(-d2["distances"]))


def test_noise_floor_is_order_invariant_with_holes():
    rng = np.random.default_rng(20261020)
    yy, xx = np.mgrid[0:80, 0:80]
    r = np.hypot(yy - 40, xx - 40)
    a = r < 25 + rng.normal(0, 1.0, r.shape)
    b = r < 26 + rng.normal(0, 1.0, r.shape)
    va = np.ones(r.shape, bool)
    vb = np.ones(r.shape, bool)
    va[5:15, 30:40] = False
    vb[60:70, 10:25] = False
    ab = hindcast.noise_floor_displacement(a, va, b, vb, return_distances=True)
    ba = hindcast.noise_floor_displacement(b, vb, a, va, return_distances=True)
    assert ab["n"] == ba["n"]
    assert ab["median"] == pytest.approx(ba["median"])
    assert ab["p95"] == pytest.approx(ba["p95"])
    assert ab["signed_median"] == pytest.approx(-ba["signed_median"])


def test_noise_floor_matches_product_definition():
    """The noise floor is the same function as the product metric, with the first
    extent's own validity mask taking the place of the product's."""
    shape = (20, 60)
    valid = _band(shape, 10, 60)
    a = _band(shape, 10, 40)
    b = _band(shape, 10, 43)
    nf = hindcast.noise_floor_displacement(a, valid, b, valid)
    sd = hindcast.shoreline_displacement(a, b, valid, pred_valid=valid)
    for k in ("median", "p95", "signed_median", "n"):
        assert nf[k] == pytest.approx(sd[k])
    assert nf["median"] == pytest.approx(3.0)


# --- known states outside the active zone ----------------------------------------------

def _point(shape, r, c):
    m = np.zeros(shape, bool)
    m[r, c] = True
    return m


def test_outside_state_from_zone():
    shape = (30, 60)
    zone = _band(shape, 10, 50)
    island = np.zeros(shape, bool)
    island[12:18, 27:33] = True                # enclosed by the zone, outside it
    zone[island] = False
    sea, land = _point(shape, 0, 0), _point(shape, 0, 59)
    st = hindcast.outside_state_from_zone(zone, sea, land, domain=None)
    assert np.all(st[:, :10] == 1.0)
    assert np.all(st[:, 50:] == 0.0)
    assert np.all(np.isnan(st[island]))        # unseeded enclosed patch: unknown, not land
    assert np.all(np.isnan(st[zone]))
    st2 = hindcast.outside_state_from_zone(zone, sea, land | _point(shape, 14, 29), domain=None)
    assert np.all(st2[island] == 0.0)          # seeded island interior: land
    with pytest.raises(ValueError, match="open-sea"):
        hindcast.outside_state_from_zone(zone, _point(shape, 0, 20), land, domain=None)
    with pytest.raises(ValueError, match="land"):
        hindcast.outside_state_from_zone(zone, sea, _point(shape, 0, 20), domain=None)


def test_enclosed_water_patch_is_unknown_not_land():
    """A pool that is water even at the low stand, enclosed by the zone (e.g. between
    emerged shoals), must not be labelled land: observed water next to it would
    otherwise get a false waterline at the zone edge."""
    shape = (40, 80)
    zone = _band(shape, 20, 60)
    pool = np.zeros(shape, bool)
    pool[15:25, 25:35] = True
    zone[pool] = False
    st = hindcast.outside_state_from_zone(zone, _point(shape, 0, 0), _point(shape, 0, 79),
                                          domain=None)
    assert np.all(np.isnan(st[pool]))
    obs_full = _band(shape, 0, 45)             # this level: water up to col 44, pool included
    valid = zone
    d = hindcast.shoreline_displacement(obs_full, obs_full & valid, valid, outside_state=st)
    assert d["p95"] == pytest.approx(0.0)      # a perfect product scores zero


def _clipped_domain_case():
    """Toy of the real geometry: the grid is larger than the domain, and the zone is
    clipped to the domain (pipeline_ee active_zone_v4/v5 end in .clip(region)). Rows
    50-59 lie outside the domain: Middle Caspian sea on the left, Mangyshlak land on
    the right, so outside the domain the sea meets the hinterland."""
    shape = (60, 100)
    domain = np.zeros(shape, bool)
    domain[:50] = True
    true_water = _band(shape, 0, 40)           # waterline at col 39 at this level
    zone = _band(shape, 30, 61) & domain
    return shape, domain, true_water, zone


def test_outside_state_with_clipped_domain_perfect_product_scores_zero():
    shape, domain, true_water, zone = _clipped_domain_case()
    sea, land = _point(shape, 10, 0), _point(shape, 10, 90)
    st = hindcast.outside_state_from_zone(zone, sea, land, domain=domain)
    assert np.all(st[:49, 61:] == 0.0)         # hinterland inside the domain: land
    assert np.all(st[:49, :30] == 1.0)
    assert np.all(np.isnan(st[50:]))           # outside the domain: unknown
    valid = zone
    obs = true_water & valid
    pred = true_water.copy()                   # a perfect product
    d = hindcast.shoreline_displacement(pred, obs, valid, outside_state=st)
    assert d["median"] == 0.0 and d["p95"] == 0.0
    m = hindcast.extent_metrics(pred, obs, valid, outside_state=st)
    # the previous version labelled the hinterland water: -2550, -5.1 and 0.16 here
    assert m["area_bias_total"] == 0.0 and m["area_fn_outside"] == 0.0
    assert m["iou_total"] == 1.0
    assert isinstance(m["area_bias_rel_total"], float)


def test_outside_state_leak_is_an_error_not_a_silent_label():
    shape, domain, true_water, zone = _clipped_domain_case()
    sea, land = _point(shape, 10, 0), _point(shape, 10, 90)
    with pytest.raises(ValueError, match="does not separate"):
        hindcast.outside_state_from_zone(zone, sea, land, domain=None)
    # an island enclosed by the zone would hide the leak from an "is any land left" check
    zone_isl = zone.copy()
    zone_isl[20:26, 44:49] = False
    with pytest.raises(ValueError, match="does not separate"):
        hindcast.outside_state_from_zone(zone_isl, sea, land, domain=None)
    # a zone that does not reach the array edge leaks around its ends
    box = np.zeros((40, 80), bool)
    box[5:35, 10:50] = True
    with pytest.raises(ValueError, match="does not separate"):
        hindcast.outside_state_from_zone(box, _point((40, 80), 20, 0), _point((40, 80), 20, 70),
                                         domain=None)


def test_domain_margin_absorbs_one_pixel_rasterisation_mismatch():
    shape, domain, true_water, zone = _clipped_domain_case()
    sea, land = _point(shape, 10, 0), _point(shape, 10, 90)
    dom_big = np.zeros(shape, bool)
    dom_big[:51] = True                        # domain mask one row larger than the zone clip
    st = hindcast.outside_state_from_zone(zone, sea, land, domain=dom_big)
    assert np.all(st[:49, 61:] == 0.0)
    with pytest.raises(ValueError, match="does not separate"):
        hindcast.outside_state_from_zone(zone, sea, land, domain=dom_big, domain_margin=0)
    with pytest.raises(TypeError):
        hindcast.outside_state_from_zone(zone, sea, land)   # domain must be stated


def _north_edge_case():
    """Toy of the northern domain edge: rows 0-9 lie north of the
    domain, on land. The sea is on the left, the hinterland on the right, and the
    zone band runs north-south across the edge. A product too deep in the north
    floods land beyond the domain edge."""
    shape = (60, 100)
    domain = np.zeros(shape, bool)
    domain[10:] = True
    zone = _band(shape, 30, 61) & domain
    true_water = _band(shape, 0, 40) & domain   # beyond the edge everything is land
    beyond = np.full(shape, np.nan)
    beyond[:10] = 0.0                           # independent observed land north of the edge
    return shape, domain, zone, true_water, beyond


def test_beyond_domain_land_scores_flooding_past_the_edge():
    shape, domain, zone, true_water, beyond = _north_edge_case()
    sea, land = _point(shape, 30, 0), _point(shape, 30, 90)
    st0 = hindcast.outside_state_from_zone(zone, sea, land, domain=domain)
    st = hindcast.outside_state_from_zone(zone, sea, land, domain=domain, beyond_state=beyond)
    assert np.all(np.isnan(st0[:10]))           # default: unknown beyond the edge
    assert np.all(st[:10][~zone[:10]] == 0.0)   # supplied: known land
    assert np.all(st[11:, :30] == 1.0) and np.all(st[11:, 61:] == 0.0)
    pred = true_water.copy()
    pred[:10, :70] = True                       # floods 700 cells of land beyond the edge
    valid = zone
    obs = true_water & valid
    m0 = hindcast.extent_metrics(pred, obs, valid, outside_state=st0)
    m = hindcast.extent_metrics(pred, obs, valid, outside_state=st)
    assert m0["area_fp_outside"] == 0.0         # censored without the outside state
    assert m["area_fp_outside"] == 700.0 and m["area_bias_total"] == 700.0
    d0 = hindcast.shoreline_displacement(pred, obs, valid, outside_state=st0)
    d = hindcast.shoreline_displacement(pred, obs, valid, outside_state=st)
    assert d["n_pred_unknown"] < d0["n_pred_unknown"]
    assert d["n_pred"] > d0["n_pred"] and d["signed_median"] >= 0
    # a perfect product is unaffected
    mp = hindcast.extent_metrics(true_water, obs, valid, outside_state=st)
    assert mp["area_bias_total"] == 0.0 and mp["iou_total"] == 1.0


def test_beyond_state_requires_domain_and_matching_shape():
    shape, domain, zone, true_water, beyond = _north_edge_case()
    sea, land = _point(shape, 30, 0), _point(shape, 30, 90)
    with pytest.raises(ValueError, match="domain"):
        hindcast.outside_state_from_zone(zone, sea, land, domain=None, beyond_state=beyond)
    with pytest.raises(ValueError, match="shape"):
        hindcast.outside_state_from_zone(zone, sea, land, domain=domain, beyond_state=beyond[1:])


def test_nearest_region_attributes_beyond_cells_to_adjacent_sector():
    shape, domain, zone, true_water, beyond = _north_edge_case()
    sectors = np.zeros(shape, int)
    sectors[:, 50:] = 1
    sectors[~domain] = -1                       # no sector outside the domain
    ext = hindcast.nearest_region(sectors, domain)
    assert np.array_equal(ext[domain], sectors[domain])
    assert np.all(ext[:10, :45] == 0) and np.all(ext[:10, 55:] == 1)
    with pytest.raises(ValueError, match="empty"):
        hindcast.nearest_region(sectors, np.zeros(shape, bool))


# --- observed levels of a region ---------------------------------------------------------

def test_observed_level_flag_for_region_without_observed_waterline():
    shape = (42, 100)
    valid = np.ones(shape, bool)
    valid[20:22] = False                       # buffer rows between the sectors: unknown
    obs = np.zeros(shape, bool)
    obs[:20, :40] = True
    obs[22:, :] = True                         # sector 1 is all water: no observed waterline
    pred = np.zeros(shape, bool)
    pred[:20, :42] = True
    pred[22:, :70] = True                      # but the product puts a waterline in it
    labels = np.zeros(shape, int)
    labels[21:] = 1
    by = hindcast.displacement_by_region(pred, obs, valid, labels)
    assert by[0]["observed_level"] is True
    assert by[1]["observed_level"] is False
    assert by[1]["n_obs"] == 0 and by[1]["n_obs_unknown"] == 0
    assert np.isfinite(by[1]["median"]) and by[1]["n_pred"] > 0   # the error is still reported
    one = hindcast.shoreline_displacement(pred, obs, valid, region=labels == 1)
    assert one["observed_level"] is False
    nan = hindcast.shoreline_displacement(pred, np.ones(shape, bool), np.ones(shape, bool))
    assert np.isnan(nan["median"]) and nan["observed_level"] is False


# --- mask inputs ---------------------------------------------------------------------------

def test_frequency_passed_as_mask_raises():
    shape = (10, 20)
    freq = np.where(_band(shape, 0, 8), 0.9, 0.3)   # water frequency, not an extent
    with pytest.raises(ValueError, match="frequency"):
        hindcast.shoreline_displacement(_band(shape, 0, 10), freq, np.ones(shape, bool))
    with pytest.raises(ValueError, match="frequency"):
        hindcast.extent_metrics(_band(shape, 0, 10), freq, np.ones(shape, bool))


def test_nan_in_float_mask_is_unknown():
    rng = np.random.default_rng(5)
    yy, xx = np.mgrid[0:40, 0:60]
    obs = xx < 30 + 2 * np.sin(yy / 4.0)
    pred = xx < 33 + rng.normal(0, 1.0, xx.shape)
    hole = (yy > 10) & (yy < 18) & (xx > 25) & (xx < 35)
    obs_f = obs.astype(float)
    obs_f[hole] = np.nan
    a = hindcast.shoreline_displacement(pred, obs_f, np.ones(obs.shape, bool))
    b = hindcast.shoreline_displacement(pred, obs & ~hole, ~hole)
    for k in ("median", "p95", "signed_median", "n", "n_pred_unknown"):
        assert a[k] == pytest.approx(b[k])
    ma = hindcast.extent_metrics(pred, obs_f, np.ones(obs.shape, bool))
    mb = hindcast.extent_metrics(pred, obs, ~hole)
    assert ma == pytest.approx(mb)


def test_by_region_matches_single_region_calls():
    rng = np.random.default_rng(3)
    yy, xx = np.mgrid[0:60, 0:90]
    obs = xx < 40 + 3 * np.sin(yy / 5.0)
    pred = xx < 44 + rng.normal(0, 1.5, xx.shape)
    valid = np.ones(obs.shape, bool)
    valid[20:26, 35:45] = False
    labels = (yy // 20).astype(int)            # three sectors
    by = hindcast.displacement_by_region(pred, obs, valid, labels, pixel_size=30.0)
    assert sorted(by) == [0, 1, 2]
    for k in (0, 1, 2):
        one = hindcast.shoreline_displacement(pred, obs, valid, 30.0, region=labels == k)
        for key in ("median", "p95", "signed_median", "n", "n_pred_unknown", "n_obs_unknown"):
            assert by[k][key] == pytest.approx(one[key])
    nf = hindcast.noise_floor_by_region(obs, valid, pred, np.ones_like(valid), labels, ids=[1])
    one = hindcast.noise_floor_displacement(obs, valid, pred, np.ones_like(valid), region=labels == 1)
    assert list(nf) == [1] and nf[1]["median"] == pytest.approx(one["median"])


def test_beyond_land_state_land_only_beyond_dilation():
    high = np.zeros((40, 40), bool)
    high[:, :10] = True                       # observed near-maximum water in the west
    obs = np.ones((40, 40), bool)
    obs[30:, 30:] = False                     # an observation gap in the south-east
    s = hindcast.beyond_land_state(high, obs, pixel_size=250.0, dilate_m=2000.0)
    assert np.all(np.isnan(s[:, :10]))        # water is never assigned
    assert np.all(np.isnan(s[:, 10:18]))      # within 2 km (8 cells) of water: unknown
    assert np.all(s[:20, 19:30] == 0)         # beyond 2 km: land
    assert np.all(np.isnan(s[30:, 30:]))      # the gap is unknown
    assert np.isnan(s[29, 25]) and np.isnan(s[25, 29])   # and so is its 2 km neighbourhood
    assert set(np.unique(s[np.isfinite(s)])) == {0.0}


def test_beyond_land_state_works_with_outside_state_from_zone():
    # toy northern edge: domain = rows 10+, zone band rows 12-16, sea rows 17+,
    # 1995-96 water rows 13+; land north of the domain
    shape = (30, 20)
    domain = np.zeros(shape, bool)
    domain[10:, :] = True
    zone = np.zeros(shape, bool)
    zone[12:17, :] = True
    sea = np.zeros(shape, bool)
    sea[25:, :] = True
    land = np.zeros(shape, bool)
    land[11, :] = True
    high = np.zeros(shape, bool)
    high[13:, :] = True
    b = hindcast.beyond_land_state(high, np.ones(shape, bool), pixel_size=1.0, dilate_m=2.0)
    st = hindcast.outside_state_from_zone(zone, sea, land, domain=domain, beyond_state=b)
    assert np.all(st[:11] == 0)            # beyond the domain and its rim: land from beyond_state
    assert np.all(st[11] == 0)             # the seeded hinterland
    assert np.all(np.isnan(st[12:17]))     # the zone
    assert np.all(st[17:] == 1)            # the open sea
    st0 = hindcast.outside_state_from_zone(zone, sea, land, domain=domain)
    assert np.all(np.isnan(st0[:11]))      # without it, unknown (the earlier behaviour)
