"""A7: the hindcast metrics of caspian.hindcast, computed tile by tile on large grids.

The 60 m hindcast grid (the product bounding box, about 11,250 x 6,700 cells) is too
large for whole-array processing on the project laptop (15 GB): one call of
hindcast.displacement_by_region runs two whole-grid distance transforms, each of
which holds an int32 feature transform, an index grid and float64 distances (an
estimated 2 to 3 GB at peak, not measured), and hindcast.flood_fill copies the
heights to float64 (0.6 GB) before labelling. A margin around each tile does not solve it, because the cell that joins
a basin to the open sea, or a waterline's nearest counterpart, can lie any distance
away. The functions here keep the whole-grid definitions of caspian.hindcast and
change only how they are computed:

- Connectivity (flood fill, outside states, sea connection of the observed extent).
  Each tile is labelled on its own with the structure caspian.hindcast uses. Labels
  that touch across a tile edge (including diagonally, for 8-connectivity) are
  joined in a graph (scipy.sparse.csgraph.connected_components), and a component is
  marked if any of its pieces holds a mark. This is exactly the whole-grid
  labelling; no margin is involved. Each tile is labelled twice (once to find the
  joins, once to write the result), so only seam lines are held in between.
- Flood levels (D-070). ``flood_level_map`` gives every cell the level at which the
  flood fill first reaches it (the minimax path height from the seed), so that the
  flood fill at any level is ``F < level``; tiles are reconstructed with a one-cell
  halo and redone until no tile edge changes, which is exact.
- Waterline cells. The 4-neighbour test of hindcast.waterline needs one cell of
  halo, so each tile is read with a one-cell halo, hindcast.known_states and
  hindcast.waterline are applied to it, and only the tile's own cells are kept.
- Distances. The nearest waterline cell of the other extent anywhere in the grid,
  by an exact nearest-neighbour query between cell centres (scipy.spatial.cKDTree):
  the Euclidean distance that the exact distance transform of caspian.hindcast
  returns.
- Summaries. hindcast._summarise, so the median, p95, signed median, the infinite
  and NaN rules and the counts are those of caspian.hindcast.
- Area metrics. Cell counts per tile and region are summed, and the formulas of
  hindcast.extent_metrics are applied to the sums (``metrics_from_counts``).
- Sector labels beyond the domain (hindcast.nearest_region): a distance transform
  per tile with a halo that grows until no cell outside the window can be closer.

Inputs may be arrays, memory maps, or callables (r0, r1, c0, c1) -> the array of
that window, so that a mask can be derived on the fly (e.g. heights below a level).
tests/test_hindcast_tiled.py checks every function against its whole-grid
counterpart in caspian.hindcast, with tile edges cutting through components,
waterlines, holes and the domain rim.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from . import hindcast

TILE = 2048
_FOUR = ndimage.generate_binary_structure(2, 1)
_EIGHT = ndimage.generate_binary_structure(2, 2)


# --- tiling helpers ------------------------------------------------------------------

def _get(a, r0, r1, c0, c1):
    if a is None:
        return None
    if callable(a):
        return a(r0, r1, c0, c1)
    return a[r0:r1, c0:c1]


def _shape(*arrays):
    for a in arrays:
        if a is not None and not callable(a):
            return tuple(np.shape(a))
    raise ValueError("pass shape= when every input is a callable")


def tiles(shape, tile: int = TILE):
    """(r0, r1, c0, c1) of the tiles covering ``shape``, row by row."""
    rows, cols = shape
    return [(r0, min(r0 + tile, rows), c0, min(c0 + tile, cols))
            for r0 in range(0, rows, tile) for c0 in range(0, cols, tile)]


def _window(shape, r0, r1, c0, c1, halo: int):
    """The tile's window with ``halo`` cells on each side, clipped at the grid edge, and
    the tile's slice inside it."""
    rows, cols = shape
    w0, w1, v0, v1 = max(r0 - halo, 0), min(r1 + halo, rows), max(c0 - halo, 0), min(c1 + halo, cols)
    return (w0, w1, v0, v1), (slice(r0 - w0, r1 - w0), slice(c0 - v0, c1 - v0))


def _mask(a):
    """A water mask in caspian.hindcast's convention (boolean, or float with NaN for
    unknown). Integer codes: 1 water, 0 land, negative unknown."""
    a = np.asarray(a)
    if a.dtype == bool or a.dtype.kind == "f":
        return a
    return np.where(a < 0, np.nan, a).astype(float)


# --- connectivity ----------------------------------------------------------------------

def _seam_pairs(a, b, connectivity: int):
    """Label pairs that touch across a seam; ``a`` and ``b`` are the lines of cells on
    its two sides, in the same order."""
    n = a.size
    out = []
    for s in ((0,) if connectivity == 4 else (-1, 0, 1)):
        x, y = (a[:n - s], b[s:]) if s >= 0 else (a[-s:], b[:n + s])
        k = (x > 0) & (y > 0)
        out.append(np.c_[x[k], y[k]])
    return out


def component_any(mask, marks, *, shape=None, tile: int = TILE, connectivity: int = 8) -> np.ndarray:
    """Cells of ``mask`` whose connected component over the whole grid holds a cell of
    ``marks`` (marks outside the mask are ignored). Boolean array."""
    shape = shape or _shape(mask, marks)
    return components_any(mask, [marks], shape=shape, tile=tile, connectivity=connectivity)[0]


def components_any(mask, marks_list, *, shape=None, tile: int = TILE, connectivity: int = 8) -> list:
    """``component_any`` for several sets of marks, sharing one labelling."""
    shape = shape or _shape(mask, *marks_list)
    st = _EIGHT if connectivity == 8 else _FOUR
    rows, cols = range(0, shape[0], tile), range(0, shape[1], tile)
    offsets, edges, n_total = {}, {}, 0
    marked = [[] for _ in marks_list]

    def label(ti, tj):
        r0, c0 = rows[ti], cols[tj]
        r1, c1 = min(r0 + tile, shape[0]), min(c0 + tile, shape[1])
        m = np.asarray(_get(mask, r0, r1, c0, c1), bool)
        lab, n = ndimage.label(m, structure=st)
        return (r0, r1, c0, c1), m, lab, n

    for ti in range(len(rows)):
        for tj in range(len(cols)):
            (r0, r1, c0, c1), m, lab, n = label(ti, tj)
            g = np.where(lab > 0, lab.astype(np.int64) + n_total, 0)
            for k, marks in enumerate(marks_list):
                mk = np.asarray(_get(marks, r0, r1, c0, c1), bool) & m
                marked[k].append(np.unique(g[mk]))
            edges[ti, tj] = (g[0].copy(), g[-1].copy(), g[:, 0].copy(), g[:, -1].copy())
            offsets[ti, tj] = n_total
            n_total += n
    outs = [np.zeros(shape, bool) for _ in marks_list]
    ids = [np.concatenate(x) for x in marked]
    if n_total == 0 or all(i.size == 0 for i in ids):
        return outs
    pairs = [np.empty((0, 2), np.int64)]
    for ti in range(len(rows) - 1):           # seams between tile rows, full width
        up = np.concatenate([edges[ti, tj][1] for tj in range(len(cols))])
        dn = np.concatenate([edges[ti + 1, tj][0] for tj in range(len(cols))])
        pairs += _seam_pairs(up, dn, connectivity)
    for tj in range(len(cols) - 1):           # seams between tile columns, full height
        lf = np.concatenate([edges[ti, tj][3] for ti in range(len(rows))])
        rt = np.concatenate([edges[ti, tj + 1][2] for ti in range(len(rows))])
        pairs += _seam_pairs(lf, rt, connectivity)
    p = np.concatenate(pairs)
    graph = sparse.coo_matrix((np.ones(len(p), np.int8), (p[:, 0] - 1, p[:, 1] - 1)),
                              shape=(n_total, n_total))
    n_comp, comp = connected_components(graph, directed=False)
    lab_hit = []
    for i in ids:
        hit = np.zeros(n_comp, bool)
        hit[comp[i - 1]] = True
        lab_hit.append(np.r_[False, hit[comp]])
    for ti in range(len(rows)):
        for tj in range(len(cols)):
            (r0, r1, c0, c1), m, lab, n = label(ti, tj)   # ndimage.label is deterministic
            g = np.where(lab > 0, lab.astype(np.int64) + offsets[ti, tj], 0)
            for out, lh in zip(outs, lab_hit):
                out[r0:r1, c0:c1] = lh[g]
    return outs


def flood_fill(elevation, level: float, seed, *, tile: int = TILE, connectivity: int = 8) -> np.ndarray:
    """hindcast.flood_fill over tiles: cells with elevation < level (compared in
    float64, as there) connected to a seed cell that is also below the level. NaN
    counts as not flooded. ``elevation`` may be a memory map."""
    def below(r0, r1, c0, c1):
        z = np.asarray(elevation[r0:r1, c0:c1], float)
        return np.isfinite(z) & (z < level)
    return component_any(below, seed, shape=tuple(np.shape(elevation)), tile=tile, connectivity=connectivity)


def flood_level_map(elevation, seed, *, tile: int = TILE, connectivity: int = 8, out=None) -> np.ndarray:
    """The flood level F of every cell (D-070): the lowest level at which
    hindcast.flood_fill(elevation, level, seed) floods it, i.e. the minimax path
    height from the seed (over paths of finite cells from a seed cell, the least
    possible highest elevation on the path, both ends included); +inf where no such
    path exists. For every level L, flood_fill(elevation, L, seed) == (F < L) with F
    compared in float64 (``flooded``), so a flood fill at any level, or a shifted
    level, is a threshold of F.

    Method: grayscale reconstruction by erosion (skimage.morphology.reconstruction;
    marker = elevation at the seeds and +inf elsewhere, mask = elevation with NaN as
    +inf, 3 x 3 footprint for 8-connectivity), tile by tile with a one-cell halo whose
    values are the neighbours' current F. A tile is redone whenever a neighbour's
    edge changes, until nothing changes. The iteration only lowers F, never below
    the minimax level, and its fixed point satisfies the minimax recursion
    everywhere, so it ends at that level exactly. Values are copied, never computed,
    so F holds exact elevation values. ``out`` may be a preallocated array or memory
    map of the elevation's float dtype."""
    from collections import deque

    from skimage.morphology import reconstruction
    shape = tuple(np.shape(elevation))
    dtype = np.result_type(np.asarray(elevation[:1, :1]).dtype, np.float32)
    F = np.empty(shape, dtype) if out is None else out
    for r0, r1, c0, c1 in tiles(shape, tile):
        z = np.asarray(elevation[r0:r1, c0:c1], dtype)
        s = np.asarray(_get(seed, r0, r1, c0, c1), bool) & np.isfinite(z)
        F[r0:r1, c0:c1] = np.where(s, z, np.inf)
    footprint = _EIGHT if connectivity == 8 else _FOUR
    nr, nc = -(-shape[0] // tile), -(-shape[1] // tile)
    queue = deque((ti, tj) for ti in range(nr) for tj in range(nc))
    queued = set(queue)
    while queue:
        ti, tj = queue.popleft()
        queued.discard((ti, tj))
        r0, r1, c0, c1 = ti * tile, min((ti + 1) * tile, shape[0]), tj * tile, min((tj + 1) * tile, shape[1])
        (w0, w1, v0, v1), core = _window(shape, r0, r1, c0, c1, 1)
        marker = np.array(F[w0:w1, v0:v1])
        if not np.isfinite(marker).any():
            continue                           # nothing reaches this window yet
        z = np.asarray(elevation[w0:w1, v0:v1], dtype)
        mask = np.where(np.isnan(z), np.inf, z).astype(dtype)
        new = reconstruction(marker, mask, method="erosion", footprint=footprint)[core]
        old = np.asarray(F[r0:r1, c0:c1])
        lower = new < old
        if not lower.any():
            continue
        F[r0:r1, c0:c1] = np.where(lower, new, old)
        if lower[0].any() or lower[-1].any() or lower[:, 0].any() or lower[:, -1].any():
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    nb = (ti + di, tj + dj)
                    if (di or dj) and 0 <= nb[0] < nr and 0 <= nb[1] < nc and nb not in queued:
                        queue.append(nb)
                        queued.add(nb)
    return F


def flooded(F, level: float):
    """``F < level`` compared in float64 (as hindcast.flood_fill compares heights), as a
    window callable (r0, r1, c0, c1) for the tiled functions."""
    return lambda r0, r1, c0, c1: np.asarray(F[r0:r1, c0:c1], float) < level


def eroded(domain, margin: int, *, shape=None, tile: int = TILE) -> np.ndarray:
    """``domain`` eroded ``margin`` times by the 8-neighbourhood, cells beyond the grid
    counting as inside (the rule of hindcast.outside_state_from_zone)."""
    shape = shape or _shape(domain)
    out = np.zeros(shape, bool)
    for r0, r1, c0, c1 in tiles(shape, tile):
        (w0, w1, v0, v1), core = _window(shape, r0, r1, c0, c1, margin)
        d = np.asarray(_get(domain, w0, w1, v0, v1), bool)
        if margin > 0:
            d = ndimage.binary_erosion(d, structure=_EIGHT, iterations=margin, border_value=1)
        out[r0:r1, c0:c1] = d[core]
    return out


def outside_state(zone, sea_seed, land_seed, *, domain, connectivity: int = 8, domain_margin: int = 1,
                  beyond_state=None, shape=None, tile: int = TILE) -> np.ndarray:
    """hindcast.outside_state_from_zone over tiles, as int8: 1 water, 0 land, -1
    unknown (NaN there). The same errors are raised: no sea or land seed outside the
    zone inside the domain, a component holding both, beyond_state without a domain or
    with another shape."""
    shape = shape or _shape(zone, domain)
    for name, a in (("domain", domain), ("beyond_state", beyond_state)):
        if a is not None and not callable(a) and tuple(np.shape(a)) != tuple(shape):
            raise ValueError(f"{name} and zone must have the same shape")
    if beyond_state is not None and domain is None:
        raise ValueError("beyond_state needs the domain mask")
    dom = np.ones(shape, bool) if domain is None else eroded(domain, domain_margin, shape=shape, tile=tile)

    def outside(r0, r1, c0, c1):
        return ~np.asarray(_get(zone, r0, r1, c0, c1), bool) & dom[r0:r1, c0:c1]

    for what, seed in (("open-sea", sea_seed), ("land", land_seed)):
        if not any(np.any(np.asarray(_get(seed, *t), bool) & outside(*t)) for t in tiles(shape, tile)):
            raise ValueError(f"no {what} seed cell lies outside the zone inside the domain")
    sea, land = components_any(outside, [sea_seed, land_seed], shape=shape, tile=tile, connectivity=connectivity)
    if np.any(sea & land):
        raise ValueError(
            "the zone does not separate the open sea from the land on this grid: a land "
            "seed is connected to the open-sea seed outside the zone (pass the domain "
            "mask if the grid is larger than the domain)")
    state = np.full(shape, -1, np.int8)
    state[sea] = 1
    state[land] = 0
    if beyond_state is not None:
        for r0, r1, c0, c1 in tiles(shape, tile):
            b = np.asarray(_get(beyond_state, r0, r1, c0, c1), float)
            beyond = ~dom[r0:r1, c0:c1] & ~np.asarray(_get(zone, r0, r1, c0, c1), bool)
            s = state[r0:r1, c0:c1]
            s[beyond & (b == 1)] = 1
            s[beyond & (b == 0)] = 0
    return state


def sea_classes(water, valid, outside_state, sea_seed, *, shape=None, tile: int = TILE,
                connectivity: int = 8) -> np.ndarray:
    """The observed extent restricted to sea-connected water (PREREG Amendment 7, item 2;
    D-033), as int8 codes: inside ``valid``, 1 = water 8-connected through known water
    to the open-sea seed, 0 = land (observed dry, or observed water whose component is
    bordered only by known land), -1 = unknown (observed water, not so connected, whose
    component touches a cell of unknown state, 8-neighbourhood); -2 outside ``valid``
    (the state there is ``outside_state``'s).

    Known water is observed water inside ``valid`` plus ``outside_state`` == 1 outside
    it; known land likewise with 0. Cells beyond the grid edge are unknown, as in
    caspian.hindcast. ``water`` is the thresholded observed extent (boolean)."""
    shape = shape or _shape(water, valid, outside_state)

    def states(r0, r1, c0, c1):
        w = np.asarray(_get(water, r0, r1, c0, c1), bool)
        v = np.asarray(_get(valid, r0, r1, c0, c1), bool)
        st = np.asarray(_get(outside_state, r0, r1, c0, c1), float)
        return w, v, (w & v) | (~v & (st == 1)), (~w & v) | (~v & (st == 0))

    def known_water(r0, r1, c0, c1):
        return states(r0, r1, c0, c1)[2]

    def touches_unknown(r0, r1, c0, c1):
        (w0, w1, v0, v1), core = _window(shape, r0, r1, c0, c1, 1)
        _, _, kw, kl = states(w0, w1, v0, v1)
        near = ndimage.binary_dilation(~kw & ~kl, structure=_EIGHT, border_value=1)
        return (near & kw)[core]

    sea, touch = components_any(known_water, [sea_seed, touches_unknown], shape=shape, tile=tile,
                                connectivity=connectivity)
    code = np.full(shape, -2, np.int8)
    for r0, r1, c0, c1 in tiles(shape, tile):
        w, v, _, _ = states(r0, r1, c0, c1)
        s, t = sea[r0:r1, c0:c1], touch[r0:r1, c0:c1]
        c = code[r0:r1, c0:c1]
        c[v] = 0
        c[v & w & s] = 1
        c[v & w & ~s & t] = -1
    return code


def nearest_region(labels, domain, *, shape=None, tile: int = TILE, halo: int = 256) -> np.ndarray:
    """hindcast.nearest_region over tiles. Each tile's distance transform runs on a
    window with ``halo`` cells around it; the halo doubles until, for every cell of
    the tile, the nearest domain cell found is no farther than the window's edge (or
    the window reaches the grid edge), so that no cell outside the window can be
    closer. Ties between equidistant domain cells may resolve differently from the
    whole-grid transform; the distance is the same."""
    shape = shape or _shape(labels, domain)
    if not any(np.asarray(_get(domain, *t), bool).any() for t in tiles(shape, tile)):
        raise ValueError("the domain mask is empty")
    out = None
    for r0, r1, c0, c1 in tiles(shape, tile):
        h = halo
        while True:
            (w0, w1, v0, v1), core = _window(shape, r0, r1, c0, c1, h)
            dom = np.asarray(_get(domain, w0, w1, v0, v1), bool)
            lab = np.asarray(_get(labels, w0, w1, v0, v1))
            if out is None:
                out = np.empty(shape, lab.dtype)
            if dom[core].all():
                out[r0:r1, c0:c1] = lab[core]
                break
            if dom.any():
                dist, idx = ndimage.distance_transform_edt(~dom, return_indices=True)
                rr, cc = np.arange(r0, r1)[:, None], np.arange(c0, c1)[None, :]
                room = np.full((r1 - r0, c1 - c0), np.inf)
                if w0 > 0:
                    room = np.minimum(room, rr - w0 + 1)
                if w1 < shape[0]:
                    room = np.minimum(room, w1 - rr)
                if v0 > 0:
                    room = np.minimum(room, cc - v0 + 1)
                if v1 < shape[1]:
                    room = np.minimum(room, v1 - cc)
                if np.all(dist[core] <= room):
                    out[r0:r1, c0:c1] = lab[idx[0][core], idx[1][core]]
                    break
            h *= 2
    return out


# --- shoreline displacement ------------------------------------------------------------

def _cells(pred_states, obs, valid, labels, shape, tile, outside_state) -> dict:
    """Waterline cells of both extents, tile by tile with a one-cell halo. For each
    cell: global row and column, whether the other extent's state is known there
    (``known``), whether the other extent is water there (``wet``), and its label."""
    keys = ("rows", "cols", "known", "wet", "label")
    P, O = {k: [] for k in keys}, {k: [] for k in keys}
    for r0, r1, c0, c1 in tiles(shape, tile):
        (w0, w1, v0, v1), core = _window(shape, r0, r1, c0, c1, 1)
        st = _get(outside_state, w0, w1, v0, v1)
        o_w, o_l = hindcast.known_states(_mask(_get(obs, w0, w1, v0, v1)),
                                         np.asarray(_get(valid, w0, w1, v0, v1), bool), st)
        p_w, p_l = pred_states(w0, w1, v0, v1, st)
        o_sh, p_sh = hindcast.waterline(o_w, o_l)[core], hindcast.waterline(p_w, p_l)[core]
        o_w, o_l, p_w, p_l = o_w[core], o_l[core], p_w[core], p_l[core]
        lab = np.asarray(_get(labels, r0, r1, c0, c1))
        for D, sh, kn, wet in ((P, p_sh, o_w | o_l, o_w), (O, o_sh, p_w | p_l, p_w)):
            r, c = np.nonzero(sh)
            for k, v in zip(keys, (r + r0, c + c0, kn[r, c], wet[r, c], lab[r, c])):
                D[k].append(v)
    return {"pred": {k: np.concatenate(v) for k, v in P.items()},
            "obs": {k: np.concatenate(v) for k, v in O.items()}}


def _distances(cells: dict, pixel_size: float) -> dict:
    p, o = cells["pred"], cells["obs"]
    xp, xo = np.c_[p["rows"], p["cols"]].astype(float), np.c_[o["rows"], o["cols"]].astype(float)
    if len(xo) and len(xp):
        d_p = cKDTree(xo).query(xp, workers=-1)[0] * pixel_size
    else:
        d_p = np.full(len(xp), np.nan)
    if len(xp) and len(xo):
        d_o = cKDTree(xp).query(xo, workers=-1)[0] * pixel_size
    else:
        d_o = np.full(len(xo), np.inf)             # the first extent has no waterline
    # as caspian.hindcast: a predicted waterline cell inside observed water is too
    # little water (negative); an observed one inside predicted water, too much
    cells["signed_p"] = np.where(p["wet"], -1.0, 1.0) * d_p
    cells["signed_o"] = np.where(o["wet"], 1.0, -1.0) * d_o
    return cells


def summarise(cells: dict, ids, *, return_distances: bool = False) -> dict:
    """Per-region results with caspian.hindcast's summary. ``ids`` are label values;
    None in ``ids`` stands for the whole grid."""
    p, o = cells["pred"], cells["obs"]
    n_p = p["rows"].size
    prep = {"shape": (n_p + o["rows"].size,), "any_obs": bool(o["rows"].size),
            "ip": np.arange(n_p), "io": n_p + np.arange(o["rows"].size),
            "known_p": p["known"], "known_o": o["known"],
            "signed_p": cells["signed_p"], "signed_o": cells["signed_o"]}
    lab = np.r_[p["label"], o["label"]]
    return {k: hindcast._summarise(prep, None if k is None else lab == k, return_distances) for k in ids}


def _ids(labels, shape, tile):
    return sorted(set().union(*(np.unique(np.asarray(_get(labels, *t))).tolist() for t in tiles(shape, tile))))


def displacement_cells(pred, obs, valid, labels, pixel_size: float = 1.0, *, pred_valid=None,
                       outside_state=None, shape=None, tile: int = TILE) -> dict:
    """Waterline cells and signed distances of hindcast.displacement_by_region."""
    shape = shape or _shape(pred, obs, valid, labels)

    def pred_states(w0, w1, v0, v1, st):
        pr = _mask(_get(pred, w0, w1, v0, v1))
        pv = _get(pred_valid, w0, w1, v0, v1)
        return hindcast.known_states(pr, np.ones(pr.shape, bool) if pv is None else np.asarray(pv, bool))

    return _distances(_cells(pred_states, obs, valid, labels, shape, tile, outside_state), pixel_size)


def noise_floor_cells(water_a, valid_a, water_b, valid_b, labels, pixel_size: float = 1.0, *,
                      outside_state=None, shape=None, tile: int = TILE) -> dict:
    """Waterline cells and signed distances of hindcast.noise_floor_by_region: extent A
    (with its own valid mask and ``outside_state``) in the place of the product."""
    shape = shape or _shape(water_a, valid_a, water_b, valid_b, labels)

    def pred_states(w0, w1, v0, v1, st):
        return hindcast.known_states(_mask(_get(water_a, w0, w1, v0, v1)),
                                     np.asarray(_get(valid_a, w0, w1, v0, v1), bool), st)

    return _distances(_cells(pred_states, water_b, valid_b, labels, shape, tile, outside_state), pixel_size)


def displacement_by_region(pred, obs, valid, labels, pixel_size: float = 1.0, *, ids=None, pred_valid=None,
                           outside_state=None, return_distances: bool = False, shape=None,
                           tile: int = TILE) -> dict:
    """hindcast.displacement_by_region over tiles."""
    shape = shape or _shape(pred, obs, valid, labels)
    cells = displacement_cells(pred, obs, valid, labels, pixel_size, pred_valid=pred_valid,
                               outside_state=outside_state, shape=shape, tile=tile)
    return summarise(cells, _ids(labels, shape, tile) if ids is None else ids, return_distances=return_distances)


def noise_floor_by_region(water_a, valid_a, water_b, valid_b, labels, pixel_size: float = 1.0, *, ids=None,
                          outside_state=None, return_distances: bool = False, shape=None,
                          tile: int = TILE) -> dict:
    """hindcast.noise_floor_by_region over tiles."""
    shape = shape or _shape(water_a, valid_a, water_b, valid_b, labels)
    cells = noise_floor_cells(water_a, valid_a, water_b, valid_b, labels, pixel_size,
                              outside_state=outside_state, shape=shape, tile=tile)
    return summarise(cells, _ids(labels, shape, tile) if ids is None else ids, return_distances=return_distances)


def pooled(distances, n_pred: int = 0, n_obs: int = 0, n_pred_unknown: int = 0, n_obs_unknown: int = 0,
           observed_level: bool = False) -> dict:
    """caspian.hindcast's summary of signed distances pooled from several results (e.g.
    the epochs of one level, PREREG Amendment 7 item 1); the counts are the caller's
    sums."""
    d = np.concatenate([np.asarray(x, float) for x in distances]) if len(distances) else np.empty(0)
    prep = {"shape": (d.size,), "any_obs": bool(d.size), "ip": np.arange(d.size), "io": np.empty(0, int),
            "known_p": np.ones(d.size, bool), "known_o": np.empty(0, bool),
            "signed_p": d, "signed_o": np.empty(0)}
    res = hindcast._summarise(prep, None, False)
    res.update(n_pred=int(n_pred), n_obs=int(n_obs), n_pred_unknown=int(n_pred_unknown),
               n_obs_unknown=int(n_obs_unknown), observed_level=bool(observed_level))
    return res


# --- area metrics ------------------------------------------------------------------------

COUNT_KEYS = ("tp", "fp", "fn", "fp_out", "fn_out")


def extent_counts(pred, obs, valid, labels, ids, *, regions=None, pred_valid=None, outside_state=None,
                  shape=None, tile: int = TILE) -> dict:
    """The cell counts behind hindcast.extent_metrics, per region id (None = the whole
    grid) and per named mask in ``regions`` ({name: boolean array or callable}, e.g.
    the domain): true and false positives and false negatives within ``valid``, and
    with ``outside_state`` the errors on known states outside it (``fp_out``,
    ``fn_out``). ``pred_valid`` False counts as a NaN prediction (left out)."""
    shape = shape or _shape(pred, obs, valid, labels)
    regions = regions or {}
    out = {k: dict.fromkeys(COUNT_KEYS, 0) for k in [*ids, *regions]}
    for r0, r1, c0, c1 in tiles(shape, tile):
        pr, pr_missing = hindcast._binary(_mask(_get(pred, r0, r1, c0, c1)), "pred")
        pv = _get(pred_valid, r0, r1, c0, c1)
        if pv is not None:
            pr_missing = pr_missing | ~np.asarray(pv, bool)
        ob, ob_missing = hindcast._binary(_mask(_get(obs, r0, r1, c0, c1)), "obs")
        va = np.asarray(_get(valid, r0, r1, c0, c1), bool)
        v = va & ~ob_missing & ~pr_missing
        p, o = pr & v, ob & v
        masks = {"tp": p & o, "fp": p & ~o, "fn": ~p & o}
        st = _get(outside_state, r0, r1, c0, c1)
        if st is not None:
            st = np.asarray(st, float)
            out_reg = ~va & ~pr_missing
            masks["fp_out"], masks["fn_out"] = pr & out_reg & (st == 0), ~pr & out_reg & (st == 1)
        lab = np.asarray(_get(labels, r0, r1, c0, c1))
        regs = [(k, None if k is None else lab == k) for k in ids]
        regs += [(k, np.asarray(_get(r, r0, r1, c0, c1), bool)) for k, r in regions.items()]
        for k, reg in regs:
            for name, m in masks.items():
                out[k][name] += int(m.sum() if reg is None else (m & reg).sum())
    return out


def metrics_from_counts(c: dict, cell_area: float = 1.0, outside: bool = True) -> dict:
    """hindcast.extent_metrics from summed counts (same keys and formulas)."""
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    a_pred, a_obs = (tp + fp) * cell_area, (tp + fn) * cell_area
    union = tp + fp + fn
    out = {
        "area_pred": float(a_pred),
        "area_obs": float(a_obs),
        "area_bias": float(a_pred - a_obs),
        "area_bias_rel": float((a_pred - a_obs) / a_obs) if a_obs > 0 else float("nan"),
        "iou": float(tp / union) if union else float("nan"),
        "f1": float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else float("nan"),
    }
    if outside:
        fp_out, fn_out = c["fp_out"], c["fn_out"]
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
