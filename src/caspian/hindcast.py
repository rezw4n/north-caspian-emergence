"""A7: hindcast audit of bathymetry products.

For an observed water level, a product's predicted water extent is the set of
cells below that level that are hydrologically connected to the open sea (seed
mask). Predicted and observed extents are compared by area bias, IoU, F1 and
shoreline displacement (PREREG section 9).

Known and unknown states. A cell's observed state (water or land) is known
inside ``valid`` (observed, enough valid scenes). Outside it the state is
unknown, unless the caller supplies a logically known state (``outside_state``:
1 = water, 0 = land, NaN = unknown), for example open sea seaward of the active
zone and land beyond the high-stand shoreline (``outside_state_from_zone``, which
needs the domain mask because the zone is clipped to the domain). Cells beyond
the domain edge are unknown unless an independent ``beyond_state`` is supplied
(D-050), so that a product flooding land beyond the domain is not censored;
``nearest_region`` attributes such cells to the adjacent sector.
A product's predicted state is known wherever ``pred_valid`` (default: every
cell, since a flood fill is defined on the whole grid). Water masks are boolean
or 0/1; a float mask may carry NaN for unknown, and any other value (e.g. a water
frequency passed instead of the thresholded extent) raises ValueError.

Waterlines. A waterline cell is a known-water cell with at least one
known-land 4-neighbour. An edge between water and a cell of unknown state (the
zone edge, a cloud or low-count hole, the array border) is not a waterline.

Shoreline displacement is symmetric. Distances from each measurable predicted
waterline cell to the nearest observed waterline cell are pooled with distances
from each measurable observed waterline cell to the nearest predicted waterline
cell. A predicted waterline cell is measurable where its observed state is
known; an observed waterline cell where its predicted state is known. Cells
that are not measurable are counted (``n_pred_unknown``, ``n_obs_unknown``),
never silently dropped. The observational noise floor uses the same function
(``noise_floor_displacement``).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

_FOUR = ndimage.generate_binary_structure(2, 1)
_EIGHT = ndimage.generate_binary_structure(2, 2)


def flood_fill(elevation, level: float, seed, connectivity: int = 8):
    """Cells with elevation < level connected (4 or 8) to any seed cell that is also
    below the level. NaN elevation counts as not flooded."""
    z = np.asarray(elevation, float)
    below = np.isfinite(z) & (z < level)
    labels, _ = ndimage.label(below, structure=_EIGHT if connectivity == 8 else _FOUR)
    seeded = np.unique(labels[np.asarray(seed, bool) & below])
    seeded = seeded[seeded > 0]
    return np.isin(labels, seeded)


def _binary(x, name: str):
    """(mask, missing) from a boolean, 0/1 or float-with-NaN water mask. Raises
    ValueError for any other value, so that a water frequency cannot be passed
    where the thresholded extent is meant (every nonzero frequency would count as
    water)."""
    a = np.asarray(x)
    if a.dtype == bool:
        return a, np.zeros(a.shape, bool)
    a = a.astype(float)
    missing = np.isnan(a)
    if not np.all((a[~missing] == 0) | (a[~missing] == 1)):
        raise ValueError(f"{name} must be a boolean or 0/1 mask (NaN = unknown), "
                         "not a frequency; threshold it first")
    return a == 1, missing


def known_states(water, valid, outside_state=None):
    """(known_water, known_land) boolean masks.

    Inside ``valid`` the state is ``water`` (a NaN in a float ``water`` mask is
    unknown). Outside it the state is ``outside_state`` where that is 1 (water) or
    0 (land), and unknown elsewhere (NaN or any other value, or everywhere outside
    ``valid`` if ``outside_state`` is None)."""
    w, missing = _binary(water, "water")
    v = np.asarray(valid, bool) & ~missing
    kw, kl = w & v, ~w & v
    if outside_state is not None:
        o = np.asarray(outside_state, float)
        kw = kw | (~v & (o == 1))
        kl = kl | (~v & (o == 0))
    return kw, kl


def outside_state_from_zone(zone, sea_seed, land_seed, *, domain, connectivity: int = 8,
                            domain_margin: int = 1, beyond_state=None):
    """Logically known state outside the active zone: float array, 1 = water,
    0 = land, NaN = unknown.

    Inside ``domain`` the cells outside the zone are split into connected
    components (outside the zone). A component holding an open-sea seed is water;
    a component holding a ``land_seed`` cell is land (hinterland beyond the
    high-stand shoreline; island interiors if seeded). Every other component is
    unknown: an enclosed patch outside the zone may be an island interior, which is
    land at every level, or a lagoon or pool that is water even at the low stand
    but sea-connected only at some levels, and topology cannot tell them apart.
    Zone cells are unknown.

    ``domain`` is the analysis domain on this grid (the rasterised DOMAIN polygon
    of ``caspian.geography``), or None to declare that every cell of the array is
    inside it. It is required because the production zones (v4 to v6) are clipped to
    the domain. On a grid larger than the domain, the cells outside it are outside
    the zone as well, and they join the open sea to the hinterland. South of the
    domain the Middle Caspian meets Mangyshlak, and east of Cape Tyub-Karagan the
    domain's southern edge runs over land. Cells outside the domain are unknown, and
    so is a rim of ``domain_margin`` cells inside its edge, which guards against
    the domain mask and the zone being rasterised slightly differently.

    ``beyond_state`` (optional; same convention as the result: 1 water, 0 land,
    NaN unknown) gives the state of the cells outside the domain and in its rim
    (outside the zone), which are otherwise unknown. Without it, a product that
    floods land beyond the domain edge is not scored there: its predicted water is
    counted only as unmeasured (D-039 found GEBCO_2022, GEBCO_2026 and SRTM15+
    flooding land north of 47.35 N at the record-maximum level). It must come from
    observations independent of the products and hold at every evaluated level,
    e.g. land where a cell lies farther than the zone dilation from the observed
    near-maximum 1995-96 water extent (D-050; PREREG Amendment 10). It needs
    ``domain``.

    Raises ValueError if no sea seed or no land seed lies outside the zone inside
    the domain, or if one component holds both. The last case means that the zone
    does not separate the open sea from the land on this grid, which happens, for
    example, when the grid is larger than the domain and ``domain`` is not passed.

    Validity: zone v6 (D-029; its predecessor v5, D-027, had neither the sea
    connection nor the dark-water rule) is the sea-connected band between the near-maximum 1995-96 extent
    and the near-minimum 2025-26 extent, dilated by 2 km. So the sea component is
    water, and the seeded hinterland land, for sea-connected water at levels between
    the 2025-26 low stand and the 1995-96 high stand. The low-stand extent is built
    from May-September 2025-26 scenes (levels down to about -28.71 m), while
    levelled scenes reach -28.905 m; a check at -28.80 to -28.86 m found about 1% of
    the 2 km band seaward of the zone classified as land (D-049). Pass as ``zone``
    the footprint over which water frequencies were actually computed (the stage-2
    tile cover, buffered), not zone v6 itself, so that such waterlines are observed
    rather than labelled. The states do not hold in scenes affected by surges.
    """
    z = np.asarray(zone, bool)
    if domain is None:
        dom = np.ones(z.shape, bool)
    else:
        dom = np.asarray(domain, bool)
        if dom.shape != z.shape:
            raise ValueError("domain and zone must have the same shape")
        if domain_margin > 0:
            dom = ndimage.binary_erosion(dom, structure=_EIGHT, iterations=domain_margin,
                                         border_value=1)
    outside = ~z & dom
    labels, _ = ndimage.label(outside, structure=_EIGHT if connectivity == 8 else _FOUR)

    def _components(seed, what):
        ids = np.unique(labels[np.asarray(seed, bool) & outside])
        ids = ids[ids > 0]
        if ids.size == 0:
            raise ValueError(f"no {what} seed cell lies outside the zone inside the domain")
        return ids

    sea = _components(sea_seed, "open-sea")
    land = _components(land_seed, "land")
    both = np.intersect1d(sea, land)
    if both.size:
        raise ValueError(
            "the zone does not separate the open sea from the land on this grid: a land "
            "seed is connected to the open-sea seed outside the zone (pass the domain "
            "mask if the grid is larger than the domain)")
    state = np.full(z.shape, np.nan)
    state[np.isin(labels, sea)] = 1.0
    state[np.isin(labels, land)] = 0.0
    if beyond_state is not None:
        if domain is None:
            raise ValueError("beyond_state needs the domain mask")
        b = np.asarray(beyond_state, float)
        if b.shape != z.shape:
            raise ValueError("beyond_state and zone must have the same shape")
        beyond = ~dom & ~z                     # outside the domain, or in its rim
        state[beyond & (b == 1)] = 1.0
        state[beyond & (b == 0)] = 0.0
    return state


def beyond_land_state(high, observed, pixel_size: float, dilate_m: float):
    """Beyond-domain state supplying land only (D-067): 0 where an observed cell lies
    farther than ``dilate_m`` from every water cell of ``high`` (the observed
    near-maximum extent), NaN elsewhere. Unobserved cells count as possible water, so
    that a gap in the observations never creates land; water is never assigned."""
    obs = np.asarray(observed, bool)
    maybe_water = np.asarray(high, bool) | ~obs
    if not maybe_water.any():
        dist = np.full(obs.shape, np.inf)
    else:
        dist = ndimage.distance_transform_edt(~maybe_water) * pixel_size
    out = np.full(obs.shape, np.nan)
    out[obs & (dist > dilate_m)] = 0.0
    return out


def nearest_region(labels, domain):
    """Region labels (e.g. sectors) extended to every cell outside ``domain``: such a
    cell takes the label of the nearest cell inside the domain (Euclidean distance on
    the grid). Used with ``beyond_state`` so that product errors beyond the domain
    edge are attributed to the adjacent sector in the per-sector metrics (D-050);
    cells inside the domain keep their own label."""
    lab = np.asarray(labels)
    dom = np.asarray(domain, bool)
    if dom.shape != lab.shape:
        raise ValueError("labels and domain must have the same shape")
    if not dom.any():
        raise ValueError("the domain mask is empty")
    _, idx = ndimage.distance_transform_edt(~dom, return_indices=True)
    return lab[tuple(idx)]


def extent_metrics(pred, obs, valid, cell_area: float = 1.0, *, outside_state=None,
                   region=None) -> dict:
    """Area bias (pred - obs), relative bias, IoU and F1 of water extents within
    ``valid`` (and ``region``, if given).

    With ``outside_state``, errors in cells outside ``valid`` whose state is
    logically known are added: predicted water on known land (``area_fp_outside``)
    and predicted land on known water (``area_fn_outside``). They enter
    ``area_bias_total``, ``area_bias_rel_total``, ``iou_total`` and ``f1_total``,
    so the bias is not capped at the zone edge. Correctly predicted cells outside
    ``valid`` (open sea, hinterland) are not counted, since they would make IoU
    trivially high. The within-``valid`` keys are unchanged. Cells where ``obs``
    or ``pred`` is NaN (float masks) are left out."""
    reg = True if region is None else np.asarray(region, bool)
    pr, pr_missing = _binary(pred, "pred")
    ob, ob_missing = _binary(obs, "obs")
    v = np.asarray(valid, bool) & reg & ~ob_missing & ~pr_missing
    p = pr & v
    o = ob & v
    tp = np.sum(p & o)
    fp = np.sum(p & ~o)
    fn = np.sum(~p & o)
    a_pred, a_obs = p.sum() * cell_area, o.sum() * cell_area
    union = tp + fp + fn
    out = {
        "area_pred": float(a_pred),
        "area_obs": float(a_obs),
        "area_bias": float(a_pred - a_obs),
        "area_bias_rel": float((a_pred - a_obs) / a_obs) if a_obs > 0 else float("nan"),
        "iou": float(tp / union) if union else float("nan"),
        "f1": float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else float("nan"),
    }
    if outside_state is not None:
        st = np.asarray(outside_state, float)
        out_reg = ~np.asarray(valid, bool) & reg & ~pr_missing
        fp_out = int(np.sum(pr & out_reg & (st == 0)))
        fn_out = int(np.sum(~pr & out_reg & (st == 1)))
        bias_t = float(a_pred - a_obs + (fp_out - fn_out) * cell_area)
        fp_t, fn_t = fp + fp_out, fn + fn_out
        out.update({
            "area_fp_outside": float(fp_out * cell_area),
            "area_fn_outside": float(fn_out * cell_area),
            "area_bias_total": bias_t,
            "area_bias_rel_total": float(bias_t / a_obs) if a_obs > 0 else float("nan"),
            "iou_total": float(tp / (tp + fp_t + fn_t)) if (tp + fp_t + fn_t) else float("nan"),
            "f1_total": float(2 * tp / (2 * tp + fp_t + fn_t)) if (2 * tp + fp_t + fn_t) else float("nan"),
        })
    return out


def waterline(known_water, known_land):
    """Known-water cells with at least one known-land 4-neighbour. Cells beyond the
    array edge are unknown."""
    return np.asarray(known_water, bool) & ndimage.binary_dilation(
        np.asarray(known_land, bool), structure=_FOUR)


def shoreline(mask):
    """Water cells that touch a non-water cell (4-neighbourhood); the array border is
    not a shoreline. Treats every non-water cell as land: for masks with unknown
    cells use ``known_states`` and ``waterline`` instead."""
    m = np.asarray(mask, bool)
    return waterline(m, ~m)


def _nan_result(**counts) -> dict:
    nan = float("nan")
    return {"median": nan, "p95": nan, "signed_median": nan, "n": 0, **counts}


def _waterline_distances(pred, obs, valid, pixel_size, pred_valid, outside_state) -> dict:
    """Both waterlines and the signed distance at each of their cells (flat indices).

    The two distance transforms are computed once here, so that several regions
    (sectors) can be summarised without recomputing them."""
    p, p_missing = _binary(pred, "pred")
    pv = (np.ones(p.shape, bool) if pred_valid is None else np.asarray(pred_valid, bool)) & ~p_missing
    o_w, o_l = known_states(obs, valid, outside_state)
    p_w = p & pv
    o_sh = waterline(o_w, o_l)
    p_sh = waterline(p_w, ~p & pv)
    ip, io = np.flatnonzero(p_sh), np.flatnonzero(o_sh)
    prep = {"shape": p.shape, "any_obs": bool(io.size), "ip": ip, "io": io,
            "known_p": (o_w | o_l).ravel()[ip],   # observed state known: sign defined
            "known_o": pv.ravel()[io]}            # predicted state known: sign defined
    if io.size:
        d_p = ndimage.distance_transform_edt(~o_sh).ravel()[ip] * pixel_size
    else:
        d_p = np.full(ip.size, np.nan)
    if ip.size:
        d_o = ndimage.distance_transform_edt(~p_sh).ravel()[io] * pixel_size
    else:
        d_o = np.full(io.size, np.inf)            # the product has no waterline
    # predicted waterline inside observed water: too little water (negative);
    # observed waterline inside predicted water: too much water (positive)
    prep["signed_p"] = np.where(o_w.ravel()[ip], -1.0, 1.0) * d_p
    prep["signed_o"] = np.where(p_w.ravel()[io], 1.0, -1.0) * d_o
    return prep


def _summarise(prep: dict, region, return_distances: bool) -> dict:
    if region is None:
        in_p = np.ones(prep["ip"].size, bool)
        in_o = np.ones(prep["io"].size, bool)
    else:
        reg = np.asarray(region, bool).ravel()
        in_p, in_o = reg[prep["ip"]], reg[prep["io"]]
    m_p, m_o = in_p & prep["known_p"], in_o & prep["known_o"]
    counts = {"n_pred": int(m_p.sum()), "n_obs": int(m_o.sum()),
              "n_pred_unknown": int((in_p & ~prep["known_p"]).sum()),
              "n_obs_unknown": int((in_o & ~prep["known_o"]).sum()),
              # the region has an observed waterline at this level (measurable or not);
              # product-independent, so it defines the region's observed levels
              "observed_level": bool(in_o.any())}
    if not prep["any_obs"] or not (m_p.any() or m_o.any()):
        res = _nan_result(**counts)
        if return_distances:
            res["distances"] = np.empty(0)
        return res
    signed = np.r_[prep["signed_p"][m_p], prep["signed_o"][m_o]]
    d = np.abs(signed)
    if np.isinf(d).any():
        # only when the product has no waterline: every distance is infinite
        sgn = np.sign(np.median(np.sign(signed)))
        stats = {"median": float("inf"), "p95": float("inf"),
                 "signed_median": float(sgn * np.inf) if sgn != 0 else float("nan")}
    else:
        stats = {"median": float(np.median(d)), "p95": float(np.percentile(d, 95)),
                 "signed_median": float(np.median(signed))}
    res = {**stats, "n": int(d.size), **counts}
    if return_distances:
        res["distances"] = signed
    return res


def shoreline_displacement(pred, obs, valid, pixel_size: float = 1.0, *, pred_valid=None,
                           outside_state=None, region=None,
                           return_distances: bool = False) -> dict:
    """Symmetric shoreline displacement between a predicted and an observed extent.

    Pools (i) the distance from each predicted waterline cell to the nearest
    observed waterline cell and (ii) the distance from each observed waterline
    cell to the nearest predicted waterline cell (see the module docstring for
    known states and waterlines). Returns the median and 95th percentile of the
    pooled distances, and the signed median. Positive means the product predicts
    too much water: a predicted waterline cell on observed land, or an observed
    waterline cell inside predicted water.

    ``region`` (e.g. a sector) limits which waterline cells are measured; the
    nearest waterline of the other extent is searched over the whole array (for
    several sectors use ``displacement_by_region``, which computes the distance
    transforms once). ``pred_valid`` marks where the predicted state is known
    (default everywhere; pass the finite-DEM mask for a product with gaps, or the
    observation mask when ``pred`` is itself an observed extent; see
    ``noise_floor_displacement``).

    Returns NaN with n = 0 if there is no observed waterline in the array or no
    measurable cell in ``region``. If the product has no waterline anywhere in
    the array (all predicted water or all predicted land), the distances are
    infinite and the median is +inf, with the sign of the error, never NaN.

    ``observed_level`` is True when ``region`` contains an observed waterline cell
    (measurable or not). It does not depend on the product, so it defines a
    region's observed levels for the H1 majority rule (PREREG section 9). A region
    with no observed waterline can still return a finite displacement: its
    predicted waterline cells, where the observed state is known, are measured to
    the nearest observed waterline anywhere in the array. That is a real product
    error and it is reported, but such a level is not one of the region's observed
    levels.

    Keys: median, p95, signed_median, n (= n_pred + n_obs), n_pred, n_obs,
    n_pred_unknown, n_obs_unknown, observed_level, and with ``return_distances``
    the signed distances (predicted cells first)."""
    prep = _waterline_distances(pred, obs, valid, pixel_size, pred_valid, outside_state)
    return _summarise(prep, region, return_distances)


def displacement_by_region(pred, obs, valid, labels, pixel_size: float = 1.0, *, ids=None,
                           pred_valid=None, outside_state=None,
                           return_distances: bool = False) -> dict:
    """``shoreline_displacement`` for each region id in the integer raster
    ``labels`` (e.g. sectors), sharing one pair of distance transforms. ``ids``
    defaults to every value present in ``labels``. Returns {id: result}."""
    lab = np.asarray(labels)
    prep = _waterline_distances(pred, obs, valid, pixel_size, pred_valid, outside_state)
    ids = np.unique(lab) if ids is None else ids
    return {k: _summarise(prep, lab == k, return_distances) for k in ids}


def noise_floor_displacement(water_a, valid_a, water_b, valid_b, pixel_size: float = 1.0, *,
                             outside_state=None, region=None,
                             return_distances: bool = False) -> dict:
    """Observational noise floor (PREREG section 9): displacement between two
    independent observed extents of the same level bin, e.g. the odd and even
    scene half-samples, or E1 against E2/E3.

    Same definition as ``shoreline_displacement``, with extent A in the place of
    the product and its own validity mask as ``pred_valid``; ``outside_state`` is
    applied to both extents. Because the measure is symmetric, swapping A and B
    leaves median, p95 and n unchanged and only flips the sign of signed_median."""
    a_w, a_l = known_states(water_a, valid_a, outside_state)
    return shoreline_displacement(a_w, water_b, valid_b, pixel_size, pred_valid=a_w | a_l,
                                  outside_state=outside_state, region=region,
                                  return_distances=return_distances)


def noise_floor_by_region(water_a, valid_a, water_b, valid_b, labels, pixel_size: float = 1.0, *,
                          ids=None, outside_state=None, return_distances: bool = False) -> dict:
    """``noise_floor_displacement`` for each region id in ``labels``, sharing one pair
    of distance transforms. Returns {id: result}."""
    a_w, a_l = known_states(water_a, valid_a, outside_state)
    return displacement_by_region(a_w, water_b, valid_b, labels, pixel_size, ids=ids,
                                  pred_valid=a_w | a_l, outside_state=outside_state,
                                  return_distances=return_distances)
