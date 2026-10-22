"""caspian.hindcast_tiled: every tiled function equals its whole-grid counterpart in
caspian.hindcast, with tile edges cutting through components, waterlines, holes and
the domain rim (synthetic grids; tile sizes that do not divide the grid)."""

import numpy as np
import pytest
from scipy import ndimage

from caspian import hindcast
from caspian import hindcast_tiled as ht

KEYS = ("median", "p95", "signed_median", "n", "n_pred", "n_obs", "n_pred_unknown", "n_obs_unknown",
        "observed_level")


def _field(shape, seed, sigma=3.0):
    rng = np.random.default_rng(seed)
    f = ndimage.gaussian_filter(rng.normal(size=shape), sigma)
    return f / f.std()


def _same(a: dict, b: dict):
    for k in KEYS:
        assert a[k] == pytest.approx(b[k], nan_ok=True), k


# --- connectivity --------------------------------------------------------------------------

@pytest.mark.parametrize("tile", [7, 13, 50, 1000])
@pytest.mark.parametrize("connectivity", [4, 8])
def test_flood_fill_equals_whole_grid(tile, connectivity):
    z = _field((61, 83), 1)
    z[20:25, 30:40] = np.nan
    seed = np.zeros(z.shape, bool)
    seed[np.unravel_index(np.nanargmin(z), z.shape)] = True
    seed[np.unravel_index(np.nanargmin(np.where(np.arange(83) > 60, z, np.inf)), z.shape)] = True
    partial = []
    for level in (-0.5, 0.0, 0.4):
        expect = hindcast.flood_fill(z, level, seed, connectivity=connectivity)
        assert 0 < expect.sum() <= np.sum(z < level)
        partial.append(expect.sum() < np.sum(z < level))   # some basins below the level not seeded
        assert np.array_equal(ht.flood_fill(z, level, seed, tile=tile, connectivity=connectivity), expect)
        # a float32 memory map is compared in float64, as hindcast.flood_fill does
        z32 = z.astype(np.float32)
        assert np.array_equal(ht.flood_fill(z32, level, seed, tile=tile, connectivity=connectivity),
                              hindcast.flood_fill(z32, level, seed, connectivity=connectivity))
    assert any(partial)


def test_snake_across_many_seams_and_a_diagonal_at_a_tile_corner():
    shape = (40, 40)
    z = np.ones(shape)
    z[1, 1:39] = -1                            # a channel winding through every tile
    z[1:39, 38] = -1
    z[38, 2:39] = -1
    z[10:39, 2] = -1
    seed = np.zeros(shape, bool)
    seed[38, 30] = True
    for tile in (5, 6, 9):
        assert np.array_equal(ht.flood_fill(z, 0.0, seed, tile=tile), hindcast.flood_fill(z, 0.0, seed))
    # two basins touching only diagonally, exactly at the corner of four tiles
    z = np.ones((10, 10))
    z[:5, :5] = -1
    z[5:, 5:] = -1
    seed = np.zeros(z.shape, bool)
    seed[0, 0] = True
    for c in (8, 4):
        got = ht.flood_fill(z, 0.0, seed, tile=5, connectivity=c)
        assert np.array_equal(got, hindcast.flood_fill(z, 0.0, seed, connectivity=c))
        assert got[9, 9] == (c == 8)


def test_component_any_with_callables_and_no_marks():
    m = _field((30, 30), 2) < 0
    marks = np.zeros(m.shape, bool)
    assert not ht.component_any(m, marks, tile=7).any()
    got = ht.component_any(lambda r0, r1, c0, c1: m[r0:r1, c0:c1], lambda r0, r1, c0, c1: m[r0:r1, c0:c1],
                           shape=m.shape, tile=7)
    assert np.array_equal(got, m)
    with pytest.raises(ValueError, match="shape"):
        ht.component_any(lambda *w: None, lambda *w: None)
    # several mark sets share one labelling and give the separate results
    rng = np.random.default_rng(5)
    sets = [rng.random(m.shape) < p for p in (0.002, 0.01, 0.0)]
    both = ht.components_any(m, sets, tile=7)
    assert all(np.array_equal(b, ht.component_any(m, s, tile=11)) for b, s in zip(both, sets))
    assert both[0].any() and not both[2].any()


def _clipped(shape=(60, 100)):
    """hindcast's clipped-domain toy plus an enclosed pool and a seeded island."""
    domain = np.zeros(shape, bool)
    domain[:50] = True
    zone = np.zeros(shape, bool)
    zone[:, 30:61] = True
    zone &= domain
    zone[20:25, 40:45] = False                 # pool enclosed by the zone (unknown)
    zone[30:34, 50:54] = False                 # island (land, seeded)
    sea = np.zeros(shape, bool)
    sea[10, 0] = True
    land = np.zeros(shape, bool)
    land[10, 90] = True
    land[31, 51] = True
    beyond = np.full(shape, np.nan)
    beyond[52:, 70:] = 0.0
    beyond[55:, :20] = 1.0
    return zone, sea, land, domain, beyond


@pytest.mark.parametrize("tile", [7, 16, 1000])
def test_outside_state_equals_whole_grid(tile):
    zone, sea, land, domain, beyond = _clipped()
    for kw in ({"domain": domain}, {"domain": domain, "beyond_state": beyond},
               {"domain": domain, "domain_margin": 3}, {"domain": domain, "connectivity": 4}):
        expect = hindcast.outside_state_from_zone(zone, sea, land, **kw)
        got = ht.outside_state(zone, sea, land, tile=tile, **kw)
        assert got.dtype == np.int8
        assert np.array_equal(np.where(np.isnan(expect), -1, expect), got)


def test_outside_state_raises_where_the_whole_grid_function_does():
    zone, sea, land, domain, beyond = _clipped()
    with pytest.raises(ValueError, match="does not separate"):
        ht.outside_state(zone, sea, land, domain=None, tile=9)
    with pytest.raises(ValueError, match="open-sea"):
        ht.outside_state(zone, np.zeros_like(sea), land, domain=domain, tile=9)
    with pytest.raises(ValueError, match="land"):
        ht.outside_state(zone, sea, zone & False, domain=domain, tile=9)
    with pytest.raises(ValueError, match="domain"):
        ht.outside_state(zone, sea, land, domain=None, beyond_state=beyond, tile=9)
    with pytest.raises(ValueError, match="shape"):
        ht.outside_state(zone, sea, land, domain=domain, beyond_state=beyond[1:], tile=9)


@pytest.mark.parametrize("halo", [1, 4, 64])
def test_nearest_region_matches_whole_grid_distance(halo):
    shape = (50, 70)
    domain = np.zeros(shape, bool)
    domain[8:40, 10:55] = True
    domain[8:20, 10:25] = False                # a notch
    labels = np.zeros(shape, np.int8)
    labels[domain] = 1 + (np.add.outer(np.arange(50), 2 * np.arange(70))[domain] // 23) % 4
    expect = hindcast.nearest_region(labels, domain)
    got = ht.nearest_region(labels, domain, tile=11, halo=halo)
    assert np.array_equal(got[domain], labels[domain])
    # outside the domain: the chosen label belongs to a domain cell at the minimum distance
    dist = ndimage.distance_transform_edt(~domain)
    for lab in np.unique(labels[domain]):
        d_lab = ndimage.distance_transform_edt(~(domain & (labels == lab)))
        assert np.all(np.isclose(d_lab[got == lab], dist[got == lab]))
    assert np.mean(got == expect) > 0.98       # differences only at equidistant ties
    with pytest.raises(ValueError, match="empty"):
        ht.nearest_region(labels, np.zeros(shape, bool), tile=11)


# --- displacement and area metrics ----------------------------------------------------------

def _case(seed, shape=(57, 73)):
    """Observed and predicted extents with holes, known outside states, product gaps."""
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    obs = (xx + 6 * _field(shape, seed)) < 40
    pred = (xx + 6 * _field(shape, seed + 1)) < 44
    valid = np.ones(shape, bool)
    valid[:, :8] = False                       # seaward of the zone
    valid[:, 62:] = False                      # landward of the zone
    valid[20:27, 30:38] = False                # a cloud hole across the waterline
    st = np.full(shape, np.nan)
    st[:, :8] = 1.0
    st[:, 66:] = 0.0                           # cols 62-65 unknown
    pv = np.ones(shape, bool)
    pv[40:45, 50:70] = False                   # product gap
    labels = (yy // 19).astype(int) + 1        # three sectors
    return pred, obs, valid, st, pv, labels


@pytest.mark.parametrize("seed", [3, 4])
@pytest.mark.parametrize("tile", [6, 17, 1000])
def test_displacement_equals_whole_grid(seed, tile):
    pred, obs, valid, st, pv, labels = _case(seed)
    obs_f = obs.astype(float)
    obs_f[5:9, 45:50] = np.nan                 # unknown inside valid
    for kw in ({}, {"outside_state": st}, {"outside_state": st, "pred_valid": pv}):
        expect = hindcast.displacement_by_region(pred, obs_f, valid, labels, 60.0, **kw)
        got = ht.displacement_by_region(pred, obs_f, valid, labels, 60.0, tile=tile, **kw)
        assert sorted(got) == sorted(expect)
        for k in expect:
            _same(got[k], expect[k])
        whole = hindcast.shoreline_displacement(pred, obs_f, valid, 60.0, **kw)
        cells = ht.displacement_cells(pred, obs_f, valid, labels, 60.0, tile=tile, **kw)
        _same(ht.summarise(cells, [None])[None], whole)
    # signed distances, as a multiset
    d1 = hindcast.displacement_by_region(pred, obs, valid, labels, outside_state=st, return_distances=True)
    d2 = ht.displacement_by_region(pred, obs, valid, labels, outside_state=st, return_distances=True, tile=tile)
    for k in d1:
        assert np.allclose(np.sort(d1[k]["distances"]), np.sort(d2[k]["distances"]))


def test_displacement_without_waterlines_is_inf_or_nan_as_whole_grid():
    pred, obs, valid, st, pv, labels = _case(5)
    for p, o in ((np.zeros_like(pred), obs), (np.ones_like(pred), obs), (pred, np.ones_like(obs))):
        expect = hindcast.displacement_by_region(p, o, np.ones_like(valid), labels)
        got = ht.displacement_by_region(p, o, np.ones_like(valid), labels, tile=9)
        for k in expect:
            _same(got[k], expect[k])
    dry = ht.displacement_by_region(np.zeros_like(pred), obs, np.ones_like(valid), labels, tile=9)
    assert all(r["median"] == np.inf for r in dry.values())


@pytest.mark.parametrize("tile", [6, 23])
def test_noise_floor_equals_whole_grid(tile):
    _, a, va, st, _, labels = _case(6)
    _, b, vb, _, _, _ = _case(7)
    va[30:35, 10:20] = False                   # each half has its own valid mask
    for kw in ({}, {"outside_state": st}):
        expect = hindcast.noise_floor_by_region(a, va, b, vb, labels, 60.0, **kw)
        got = ht.noise_floor_by_region(a, va, b, vb, labels, 60.0, tile=tile, **kw)
        for k in expect:
            _same(got[k], expect[k])
    # integer codes (1 water, 0 land, negative unknown) mean the same as a float mask with NaN
    code = np.where(a, 1, 0).astype(np.int8)
    code[2:4, 2:4] = -1
    af = a.astype(float)
    af[2:4, 2:4] = np.nan
    r1 = ht.noise_floor_by_region(code, va, b, vb, labels, tile=tile)
    r2 = hindcast.noise_floor_by_region(af, va, b, vb, labels)
    for k in r2:
        _same(r1[k], r2[k])


@pytest.mark.parametrize("tile", [6, 1000])
def test_extent_counts_reproduce_extent_metrics(tile):
    pred, obs, valid, st, pv, labels = _case(8)
    obs_f = obs.astype(float)
    obs_f[5:9, 45:50] = np.nan
    pred_nan = np.where(pv, pred, np.nan)
    ids = [1, 2, 3, None]
    for kw, outside in (({}, False), ({"outside_state": st}, True)):
        cnt = ht.extent_counts(pred, obs_f, valid, labels, ids, pred_valid=pv, tile=tile, **kw)
        for k in ids:
            region = None if k is None else labels == k
            expect = hindcast.extent_metrics(pred_nan, obs_f, valid, 0.0036, region=region, **kw)
            got = ht.metrics_from_counts(cnt[k], 0.0036, outside=outside)
            assert got == pytest.approx(expect, nan_ok=True)
    assert ht.metrics_from_counts(dict.fromkeys(ht.COUNT_KEYS, 0))["iou"] != ht.metrics_from_counts(
        dict.fromkeys(ht.COUNT_KEYS, 0))["iou"]      # NaN with nothing to compare


def test_pooled_summary_matches_the_concatenated_distances():
    pred, obs, valid, st, pv, labels = _case(9)
    r1 = hindcast.shoreline_displacement(pred, obs, valid, outside_state=st, return_distances=True)
    r2 = hindcast.shoreline_displacement(pred[::-1], obs[::-1], valid[::-1], outside_state=st[::-1],
                                         return_distances=True)
    p = ht.pooled([r1["distances"], r2["distances"]], 3, 4, 5, 6, True)
    d = np.abs(np.r_[r1["distances"], r2["distances"]])
    assert p["median"] == pytest.approx(np.median(d)) and p["p95"] == pytest.approx(np.percentile(d, 95))
    assert p["n"] == d.size and (p["n_pred"], p["n_obs"], p["n_pred_unknown"], p["n_obs_unknown"]) == (3, 4, 5, 6)
    inf = ht.pooled([np.array([-np.inf, -np.inf]), np.array([-np.inf])])
    assert inf["median"] == np.inf and inf["signed_median"] == -np.inf
    empty = ht.pooled([])
    assert np.isnan(empty["median"]) and empty["n"] == 0


# --- sea connection of the observed extent -----------------------------------------------------

def _sea_case():
    shape = (30, 40)
    water = np.zeros(shape, bool)
    water[:, :15] = True                       # the sea, entering the zone
    water[5:8, 14:30] = True                   # a channel joined to it
    water[15:18, 25:29] = True                 # a lake bordered by observed land
    water[22:25, 30:34] = True                 # a pond touching a cloud hole
    water[10, 36] = True                       # water diagonal to the channel? no: separate speck
    valid = np.ones(shape, bool)
    valid[:, :10] = False                      # seaward of the zone
    valid[23:27, 34:37] = False                # cloud hole next to the pond
    st = np.full(shape, np.nan)
    st[:, :10] = 1.0
    seed = np.zeros(shape, bool)
    seed[0, 0] = True
    return water, valid, st, seed


@pytest.mark.parametrize("tile", [4, 7, 100])
def test_sea_classes(tile):
    water, valid, st, seed = _sea_case()
    code = ht.sea_classes(water, valid, st, seed, tile=tile)
    assert np.all(code[~valid] == -2)
    assert np.all(code[5:8, 14:30] == 1)                       # channel joined to the sea
    assert np.all(code[:, 10:15] == 1)
    assert np.all(code[15:18, 25:29] == 0)                     # enclosed lake: land
    assert np.all(code[22:25, 30:34] == -1)                    # touches the cloud hole: unknown
    assert code[10, 36] == 0
    assert np.all(code[valid & ~water] == 0)
    # connected through the known open sea outside the zone only
    w2 = water.copy()
    w2[:, 10:15] = False
    w2[0:2, 10:30] = True
    v2 = valid.copy()
    v2[0:2, 10:12] = False                     # gap inside the zone, but outside state is water there
    st2 = st.copy()
    st2[0:2, 10:12] = 1.0
    c2 = ht.sea_classes(w2, v2, st2, seed, tile=tile)
    assert np.all(c2[0:2, 12:30] == 1)
    # the array border counts as unknown
    w3 = np.zeros((10, 10), bool)
    w3[4:6, 7:] = True
    c3 = ht.sea_classes(w3, np.ones((10, 10), bool), np.full((10, 10), np.nan), np.eye(10, dtype=bool) & False,
                        tile=tile)
    assert np.all(c3[4:6, 7:] == -1)


# --- flood-level map: F < L is the flood fill at L --------------------------------------------

@pytest.mark.parametrize("tile", [7, 16, 1000])
@pytest.mark.parametrize("connectivity", [4, 8])
def test_flood_level_map_thresholds_equal_the_flood_fill(tile, connectivity):
    z = _field((61, 83), 11).astype(np.float32)
    z[20:25, 30:40] = np.nan
    seed = np.zeros(z.shape, bool)
    seed[np.unravel_index(np.nanargmin(z), z.shape)] = True
    seed[5, 80] = True                          # a seed that is high ground
    F = ht.flood_level_map(z, seed, tile=tile, connectivity=connectivity)
    assert F.dtype == np.float32 and np.isinf(F[20:25, 30:40]).all()
    levels = np.r_[np.nanmin(z) - 1, np.quantile(z[np.isfinite(z)], [0.1, 0.3, 0.5, 0.7, 0.9]), np.nanmax(z) + 1]
    for L in levels:
        expect = hindcast.flood_fill(z, float(L), seed, connectivity=connectivity)
        assert np.array_equal(np.asarray(F, float) < L, expect)
        assert np.array_equal(ht.flooded(F, float(L))(0, 61, 0, 83), expect)
        assert np.array_equal(ht.flood_fill(z, float(L), seed, tile=tile, connectivity=connectivity), expect)
    # every finite F is one of the elevations (copied, never computed)
    assert np.isin(F[np.isfinite(F)], z[np.isfinite(z)]).all()


def test_flood_level_map_across_many_seams(tmp_path):
    shape = (40, 40)
    z = np.full(shape, 5.0)
    z[1, 1:39] = -1                            # the winding channel of the flood-fill test,
    z[1:39, 38] = -2                           # with a different floor in each leg
    z[38, 2:39] = -3
    z[10:39, 2] = -4
    seed = np.zeros(shape, bool)
    seed[38, 30] = True
    out = np.lib.format.open_memmap(tmp_path / "F.npy", mode="w+", dtype=np.float64, shape=shape)
    F = ht.flood_level_map(z, seed, tile=5, out=out)
    for L in (-3.5, -2.5, -1.5, -0.5, 0.0, 5.5):
        assert np.array_equal(np.asarray(F) < L, hindcast.flood_fill(z, L, seed))
    assert F[1, 1] == -1 and F[10, 2] == -3 and F[38, 30] == -3   # the highest floor on the way


# --- the private caspian.hindcast helpers that hindcast_tiled relies on -------------------------

def test_pinned_binary():
    """hindcast._binary: (mask, missing) for boolean, 0/1 and float-with-NaN masks; a
    frequency raises. hindcast_tiled.extent_counts depends on exactly this."""
    b = np.array([True, False])
    m, miss = hindcast._binary(b, "x")
    assert m is b and not miss.any()
    m, miss = hindcast._binary(np.array([1.0, 0.0, np.nan]), "x")
    assert m.tolist() == [True, False, False] and miss.tolist() == [False, False, True]
    m, miss = hindcast._binary(np.array([1, 0], np.int8), "x")
    assert m.tolist() == [True, False] and not miss.any()
    with pytest.raises(ValueError, match="frequency"):
        hindcast._binary(np.array([0.5]), "x")


def test_pinned_summarise():
    """hindcast._summarise on a hand-built prep: the keys, the counts (region and known
    flags), median/p95/signed median of the pooled |distances|, predicted cells first in
    ``distances``, the infinite rule and the NaN rule. hindcast_tiled.summarise and
    pooled build such preps."""
    prep = {"shape": (6,), "any_obs": True, "ip": np.arange(3), "io": 3 + np.arange(3),
            "known_p": np.array([True, True, False]), "known_o": np.array([True, False, True]),
            "signed_p": np.array([-60.0, 120.0, 999.0]), "signed_o": np.array([60.0, 999.0, -180.0])}
    r = hindcast._summarise(prep, None, True)
    assert set(r) == {"median", "p95", "signed_median", "n", "n_pred", "n_obs", "n_pred_unknown",
                      "n_obs_unknown", "observed_level", "distances"}
    assert (r["n"], r["n_pred"], r["n_obs"], r["n_pred_unknown"], r["n_obs_unknown"]) == (4, 2, 2, 1, 1)
    assert r["distances"].tolist() == [-60.0, 120.0, 60.0, -180.0]
    assert r["median"] == 90.0 and r["p95"] == pytest.approx(np.percentile([60, 120, 60, 180], 95))
    assert r["signed_median"] == 0.0 and r["observed_level"] is True
    reg = np.array([True, False, False, False, False, True])
    r = hindcast._summarise(prep, reg, False)
    assert (r["n"], r["n_pred"], r["n_obs"], r["observed_level"]) == (2, 1, 1, True)
    inf = dict(prep, signed_o=np.array([np.inf, np.inf, np.inf]), known_p=np.zeros(3, bool))
    r = hindcast._summarise(inf, None, False)
    assert r["median"] == np.inf and r["p95"] == np.inf and r["signed_median"] == np.inf
    r = hindcast._summarise(dict(prep, any_obs=False), None, True)
    assert np.isnan(r["median"]) and r["n"] == 0 and r["distances"].size == 0
