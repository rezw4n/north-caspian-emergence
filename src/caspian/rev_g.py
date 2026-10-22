"""Helpers for the revision analyses of H2 and of ETOPO 2022 (PREREG Amendment 18
item 11). Additive; nothing in the registered
pipeline calls them.

H2
- ``paired_diffs``: DEM-minus-product metric differences on one set of points.
- ``leave_one_out``: those differences with each group (test block or sector) left
  out, and ``jackknife_se`` over the left-out values.
- ``mark_not_estimable``: block-bootstrap intervals that rest on fewer than two blocks
  are replaced by NaN and labelled "not estimable".
- ``cell_states``: the observed state (water, dry, not valid) of the 60 m stage-2 cell
  that holds each point, from an observed-extent code grid; ``independent_dry`` turns
  it into a dryness test.

ETOPO 2022
- ``attribute_sources``: the source of each waterline distance of a displacement run:
  a predicted waterline cell keeps its own source, an observed one takes the source of
  its nearest predicted waterline cell (the cell its distance is measured to).
- ``matched_levels``: per (sector, level), the median absolute distance of each source,
  and the levels where every source has at least ``min_n`` distances.
- ``group_nmad_bootstrap`` and ``within_block_contrast``: NMAD by group with block-
  bootstrap intervals, and the per-block contrast between two groups.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from . import validation

METRICS = {"nmad": validation.nmad, "rmse": validation.rmse, "bias": validation.bias}


# --- H2 -------------------------------------------------------------------------------

def paired_diffs(e_dem: np.ndarray, e_prod: dict, metrics=("nmad", "rmse")) -> dict:
    """{(product, metric): metric(e_dem) - metric(e_prod[product])} on the same points
    (rows where every error is finite)."""
    e_dem = np.asarray(e_dem, float)
    ok = np.isfinite(e_dem)
    for v in e_prod.values():
        ok &= np.isfinite(np.asarray(v, float))
    out = {}
    for mt in metrics:
        f = METRICS[mt]
        d = f(e_dem[ok])
        for p, v in e_prod.items():
            out[(p, mt)] = d - f(np.asarray(v, float)[ok])
    return out


def leave_one_out(e_dem, e_prod: dict, groups, metrics=("nmad", "rmse")) -> pd.DataFrame:
    """The paired differences with each group left out in turn. One row per (left-out
    group, product, metric) with ``diff``; the full-sample value is in ``diff_full`` and
    the change in ``influence`` = diff - diff_full. ``groups``: one label per point."""
    g = np.asarray(groups).astype(str)
    full = paired_diffs(e_dem, e_prod, metrics)
    rows = []
    for lab in sorted(np.unique(g)):
        keep = g != lab
        if not keep.any():
            continue
        dd = paired_diffs(np.asarray(e_dem, float)[keep], {p: np.asarray(v, float)[keep] for p, v in e_prod.items()},
                          metrics)
        for (p, mt), v in dd.items():
            rows.append({"left_out": lab, "product": p, "metric": mt, "diff": v, "diff_full": full[(p, mt)],
                         "influence": v - full[(p, mt)], "n_left_out": int((~keep).sum())})
    return pd.DataFrame(rows)


def jackknife_se(values) -> float:
    """Jackknife standard error from the n leave-one-out estimates:
    sqrt((n - 1) / n * sum((v_i - mean(v))^2)). NaN for fewer than 2 values."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    n = v.size
    if n < 2:
        return float("nan")
    return float(np.sqrt((n - 1) / n * np.sum((v - v.mean()) ** 2)))


def summarise_influence(loo: pd.DataFrame) -> pd.DataFrame:
    """Per (product, metric): the full-sample difference, the range over the left-out
    groups, the group whose removal moves it most, whether any removal changes its sign,
    and the jackknife SE."""
    rows = []
    for (p, mt), g in loo.groupby(["product", "metric"], sort=False):
        full = float(g["diff_full"].iloc[0])
        k = int(np.argmax(np.abs(g["influence"].to_numpy())))
        rows.append({"product": p, "metric": mt, "diff_full": full, "diff_min": float(g["diff"].min()),
                     "diff_max": float(g["diff"].max()), "most_influential": g["left_out"].iloc[k],
                     "max_abs_influence": float(abs(g["influence"].iloc[k])),
                     "sign_change": bool(np.any(np.sign(g["diff"].to_numpy()) != np.sign(full))),
                     "n_groups": int(len(g)), "jackknife_se": jackknife_se(g["diff"])})
    return pd.DataFrame(rows)


def mark_not_estimable(table: pd.DataFrame, interval_cols, n_blocks_col: str = "n_blocks",
                       min_blocks: int = 2) -> pd.DataFrame:
    """A copy of ``table`` in which the interval columns of rows resting on fewer than
    ``min_blocks`` blocks are NaN, with ``interval_status`` "not estimable (n blocks)"
    there and "estimated" elsewhere. Rows with no block keep NaN and say so."""
    t = table.copy()
    nb = t[n_blocks_col].fillna(0).astype(int)
    bad = nb < min_blocks
    for c in interval_cols:
        t[c] = t[c].where(~bad, np.nan)
    t["interval_status"] = np.where(nb == 0, "no block", np.where(bad, "not estimable (1 block)", "estimated"))
    return t


def cell_states(x, y, code, transform) -> np.ndarray:
    """The observed code (1 water, 0 dry, -2 not valid) of the grid cell that contains
    each point; -3 where the point lies outside the grid. ``transform`` is
    [a, b, c, d, e, f] of a north-up grid (a > 0, e < 0)."""
    a, _, c, _, e, f = (float(v) for v in transform[:6])
    col = np.floor((np.asarray(x, float) - c) / a).astype(np.int64)
    row = np.floor((np.asarray(y, float) - f) / e).astype(np.int64)
    inside = (row >= 0) & (row < code.shape[0]) & (col >= 0) & (col < code.shape[1])
    out = np.full(row.shape, -3, np.int8)
    out[inside] = np.asarray(code)[row[inside], col[inside]]
    return out


def independent_dry(states) -> np.ndarray:
    """Dry only where the cell is valid and observed dry (code 0)."""
    return np.asarray(states) == 0


# --- ETOPO 2022 -----------------------------------------------------------------------

def attribute_sources(cells: dict, pred_source) -> pd.DataFrame:
    """Waterline distances of one displacement run (``hindcast_tiled.displacement_cells``)
    with a source per distance. ``pred_source``: one code per predicted waterline cell.
    A predicted cell keeps its own source; an observed cell takes the source of its
    nearest predicted waterline cell. Known cells only; undefined (NaN) distances are
    dropped. Returns columns kind ("pred"/"obs"), row, col, label, source,
    abs_m."""
    p, o = cells["pred"], cells["obs"]
    src_p = np.asarray(pred_source)
    if src_p.shape[0] != p["rows"].size:
        raise ValueError("one source per predicted waterline cell")
    frames = []
    kp = p["known"] & ~np.isnan(cells["signed_p"])
    frames.append(pd.DataFrame({"kind": "pred", "row": p["rows"][kp], "col": p["cols"][kp], "label": p["label"][kp],
                                "source": src_p[kp], "abs_m": np.abs(cells["signed_p"][kp])}))
    ko = o["known"] & ~np.isnan(cells["signed_o"])
    if ko.any() and p["rows"].size:
        xp = np.c_[p["rows"], p["cols"]].astype(float)
        _, idx = cKDTree(xp).query(np.c_[o["rows"][ko], o["cols"][ko]].astype(float), workers=-1)
        frames.append(pd.DataFrame({"kind": "obs", "row": o["rows"][ko], "col": o["cols"][ko], "label": o["label"][ko],
                                    "source": src_p[idx], "abs_m": np.abs(cells["signed_o"][ko])}))
    return pd.concat(frames, ignore_index=True)


def matched_levels(per_level: pd.DataFrame, sources, min_n: int = 20) -> pd.DataFrame:
    """``per_level``: one row per (sector, level_m, source) with ``median_m`` and ``n``.
    Returns one row per (sector, level_m) with each source's median and n as columns
    (``median_<s>``, ``n_<s>``) and ``matched`` True where every source has at least
    ``min_n`` distances."""
    w = per_level.pivot_table(index=["sector", "level_m"], columns="source", values=["median_m", "n"],
                              aggfunc="first")
    out = pd.DataFrame(index=w.index)
    matched = np.ones(len(w), bool)
    for s in sources:
        med = w[("median_m", s)] if ("median_m", s) in w.columns else pd.Series(np.nan, index=w.index)
        n = w[("n", s)] if ("n", s) in w.columns else pd.Series(0, index=w.index)
        out[f"median_{s}"] = med.to_numpy(float)
        out[f"n_{s}"] = n.fillna(0).to_numpy(int)
        matched &= out[f"n_{s}"].to_numpy() >= min_n
    out["matched"] = matched
    return out.reset_index()


def group_nmad_bootstrap(errors: dict, group, blocks, groups=("a", "b"), n_boot: int = 2000, seed: int = 20261022,
                         conf: float = 0.95) -> pd.DataFrame:
    """NMAD of each error series in each of two point groups, the difference between the
    groups (second minus first), and for two or more series the difference of those
    differences (each series minus the first), with percentile intervals from one block
    bootstrap (validation.block_bootstrap over ``blocks``)."""
    g = np.asarray(group).astype(str)
    names = list(errors)
    E = {k: np.asarray(v, float) for k, v in errors.items()}
    ok = np.ones(g.size, bool)
    for v in E.values():
        ok &= np.isfinite(v)
    g0, g1 = groups

    def stats(ix):
        ix = ix[ok[ix]]
        a, b = ix[g[ix] == g0], ix[g[ix] == g1]
        out = []
        for k in names:
            na = validation.nmad(E[k][a]) if a.size else np.nan
            nb = validation.nmad(E[k][b]) if b.size else np.nan
            out += [na, nb, nb - na]
        return out

    point = stats(np.arange(g.size))
    boot = validation.block_bootstrap(np.asarray(blocks), stats, n_boot, seed)
    rows = []
    for i, k in enumerate(names):
        for j, what in enumerate((f"nmad_{g0}", f"nmad_{g1}", f"diff_{g1}_minus_{g0}")):
            lo, hi = validation.percentile_interval(boot[:, 3 * i + j], conf)
            rows.append({"series": k, "statistic": what, "value": point[3 * i + j], "ci_low": lo, "ci_high": hi})
    if len(names) > 1:
        for i, k in enumerate(names[1:], start=1):
            dd = boot[:, 3 * i + 2] - boot[:, 2]
            lo, hi = validation.percentile_interval(dd, conf)
            rows.append({"series": f"{k} minus {names[0]}", "statistic": f"diff_{g1}_minus_{g0}",
                         "value": point[3 * i + 2] - point[2], "ci_low": lo, "ci_high": hi})
    t = pd.DataFrame(rows)
    t["n_boot"], t["seed"], t["n_blocks"] = n_boot, seed, int(np.unique(np.asarray(blocks)[ok]).size)
    return t


def within_block_contrast(errors: dict, group, blocks, groups=("a", "b"), min_n: int = 50) -> pd.DataFrame:
    """Per block with at least ``min_n`` points in each of the two groups: the NMAD of
    each error series in each group and the difference (second group minus first)."""
    g = np.asarray(group).astype(str)
    blk = np.asarray(blocks).astype(str)
    E = {k: np.asarray(v, float) for k, v in errors.items()}
    ok = np.ones(g.size, bool)
    for v in E.values():
        ok &= np.isfinite(v)
    g0, g1 = groups
    rows = []
    for b in sorted(np.unique(blk[ok])):
        m = ok & (blk == b)
        a, c = m & (g == g0), m & (g == g1)
        if a.sum() < min_n or c.sum() < min_n:
            continue
        r = {"block": b, f"n_{g0}": int(a.sum()), f"n_{g1}": int(c.sum())}
        for k, v in E.items():
            na, nc = validation.nmad(v[a]), validation.nmad(v[c])
            r.update({f"{k}_nmad_{g0}": na, f"{k}_nmad_{g1}": nc, f"{k}_diff": nc - na})
        rows.append(r)
    return pd.DataFrame(rows)
