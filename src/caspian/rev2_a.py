"""Second-round revision analysis A: PREREG Amendment 19 items 7, 8 and 11 (D-147).

Pure functions (arrays and tables in, arrays and tables out), tested in
tests/test_rev2_a.py. The drivers are scripts/rev2_a_*.py; the waterline engine is that
of caspian.rev_r (``cells_from``, ``obs_states``) with caspian.hindcast_tiled's
``_distances`` and ``summarise``.

Item 7(a), the total variant with the DAHITI reference constant c_ref: a DAHITI level L
is the EIGEN-6C4 height L - c_ref, so a corrected product cell is water when
F < L - c_ref (``total_threshold``; products.flood_level with a product offset of
+c_ref).

Item 8, unknown predicted cells. The registered metric (caspian.hindcast._summarise)
leaves out a predicted waterline cell whose observed state is unknown, while every
observed waterline cell is measured. Two recomputations of the same ``cells``:
  (i)  ``measure_all``: those cells are measured too (their distance to the nearest
       observed waterline cell is already in ``cells``; its sign is +, since whether the
       cell is wet is not known);
  (ii) ``drop_measured``: observed waterline cells with an unknown cell within 2 cells
       (centre distance <= 120 m; ``near_unknown``) are not measured. They stay
       nearest-neighbour targets, as unknown predicted cells do in the registered metric.
``unknown_share`` gives, per product and sector, the share of predicted waterline cells
left out because the observed state is unknown, from a7_hindcast's displacement table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import products, rev_r


# --- item 7(a) ---------------------------------------------------------------------------------

def total_threshold(level_m, c_ref_m: float):
    """Flood level of the total variant with the DAHITI reference constant: L - c_ref, in the
    order of operations of products.flood_level (product offset +c_ref)."""
    return products.flood_level(level_m, 0.0, float(c_ref_m), 0.0)


# --- item 8 ------------------------------------------------------------------------------------

def disk_offsets(radius_cells: int = 2) -> np.ndarray:
    """(k, 2) row and column offsets of the cells whose centre lies within ``radius_cells``
    cells of a cell's centre (Euclidean), the cell itself excluded. Radius 2 gives 12."""
    r = int(radius_cells)
    dr, dc = np.mgrid[-r:r + 1, -r:r + 1]
    keep = (dr ** 2 + dc ** 2 <= radius_cells ** 2) & ((dr != 0) | (dc != 0))
    return np.c_[dr[keep], dc[keep]].astype(np.int64)


def unknown_at(code, outside, rows, cols) -> np.ndarray:
    """True where the observed state is unknown at the cells (rows, cols): neither known
    water nor known land by rev_r.obs_states (a7_hindcast codes with the outside states).
    Cells beyond the grid edge are unknown, as in caspian.hindcast."""
    code = np.asarray(code)
    H, W = code.shape
    rows, cols = np.asarray(rows, np.int64), np.asarray(cols, np.int64)
    out = np.ones(rows.shape, bool)
    inside = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
    if inside.any():
        r, c = rows[inside], cols[inside]
        kw, kl = rev_r.obs_states(code[r, c], np.asarray(outside)[r, c])
        out[inside] = ~(kw | kl)
    return out


def near_unknown(rows, cols, code, outside, radius_cells: int = 2) -> np.ndarray:
    """True for each cell (rows, cols) that has an unknown cell (``unknown_at``) within
    ``radius_cells`` cells, centre to centre, the cell itself not counted."""
    rows, cols = np.asarray(rows, np.int64), np.asarray(cols, np.int64)
    flag = np.zeros(rows.shape, bool)
    for dr, dc in disk_offsets(radius_cells):
        flag |= unknown_at(code, outside, rows + dr, cols + dc)
    return flag


def _with_known(cells: dict, side: str, known) -> dict:
    out = {**cells, side: {**cells[side], "known": np.asarray(known, bool)}}
    return out


def measure_all(cells: dict, sides=("pred",)) -> dict:
    """``cells`` (after hindcast_tiled._distances) with every waterline cell of ``sides``
    measured: its ``known`` flag set True. Distances are not recomputed (they already hold
    every cell's distance to the other extent's nearest waterline cell)."""
    out = cells
    for s in sides:
        out = _with_known(out, s, np.ones(cells[s]["rows"].shape, bool))
    return out


def drop_measured(cells: dict, side: str, flag) -> dict:
    """``cells`` with the waterline cells of ``side`` where ``flag`` is True no longer
    measured (``known`` False); they remain targets of the other side's distances."""
    flag = np.asarray(flag, bool)
    if flag.shape != cells[side]["rows"].shape:
        raise ValueError("flag must have one value per waterline cell")
    return _with_known(cells, side, np.asarray(cells[side]["known"], bool) & ~flag)


def unknown_share(disp: pd.DataFrame, epochs=("E1", "E2", "E3")) -> pd.DataFrame:
    """Per product, variant and sector, the share of predicted waterline cells left out
    because the observed state is unknown, from a7_hindcast's displacement table (unit rows:
    ``epoch`` in ``epochs``; ``n_pred`` measured, ``n_pred_unknown`` left out).

    pooled_share = sum n_pred_unknown / sum (n_pred + n_pred_unknown) over the units;
    unit shares over the units with at least one predicted waterline cell in the sector:
    their median, quartiles and number."""
    d = disp[disp["epoch"].isin(list(epochs)) & (disp["sector"] != "all")].copy()
    d["tot"] = d["n_pred"].astype(float) + d["n_pred_unknown"].astype(float)
    rows = []
    for (p, v, s), g in d.groupby(["product", "variant", "sector"], sort=True):
        tot = float(g["tot"].sum())
        unit = (g["n_pred_unknown"].astype(float) / g["tot"])[g["tot"] > 0]
        rows.append({"product": p, "variant": v, "sector": s, "n_units": int(len(g)),
                     "n_units_with_pred": int(unit.size),
                     "n_pred_measured": int(g["n_pred"].sum()), "n_pred_unknown": int(g["n_pred_unknown"].sum()),
                     "pooled_share": float(g["n_pred_unknown"].sum() / tot) if tot > 0 else float("nan"),
                     "unit_share_median": float(unit.median()) if unit.size else float("nan"),
                     "unit_share_q25": float(unit.quantile(0.25)) if unit.size else float("nan"),
                     "unit_share_q75": float(unit.quantile(0.75)) if unit.size else float("nan")})
    return pd.DataFrame(rows)


# --- verdict tables ------------------------------------------------------------------------------

VALIDATED = ["west", "interfluve", "northeast", "open_shelf"]       # PREREG Amendment 18 item 1
SECTOR_REPORT_ORDER = VALIDATED + ["volga", "ural", "komsomolets"]   # validated sectors first


def sector_counts(summ: pd.DataFrame) -> dict:
    """Per product: mispredicted sectors in all seven, in the four validated ones and in the
    six other than Komsomolets Bay (NA counts as not mispredicted), and the lists."""
    out = {}
    for p, g in summ.groupby("product"):
        mis = g["mispredicts"].astype("boolean").fillna(False).astype(bool)
        na = g["mispredicts"].isna()
        out[p] = {"all7": int(mis.sum()), "validated4": int(mis[g["sector"].isin(VALIDATED)].sum()),
                  "six_without_komsomolets": int(mis[g["sector"] != "komsomolets"].sum()),
                  "sectors": [s for s in SECTOR_REPORT_ORDER if s in set(g.loc[mis, "sector"])],
                  "not_evaluable": [s for s in SECTOR_REPORT_ORDER if s in set(g.loc[na, "sector"])]}
    return out


def compare_verdicts(base: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Per product and sector: evaluable and exceeding levels and the verdict in two runs,
    whether it changed (NA against a value counts as a change), and whether the sector is
    validated; validated sectors first."""
    k = ["product", "sector"]
    cols = ["n_evaluable", "n_exceeding", "mispredicts"]
    m = base[k + cols].merge(new[k + cols], on=k, how="outer", suffixes=("_reference", "_new"))
    a = m["mispredicts_reference"].astype("boolean")
    b = m["mispredicts_new"].astype("boolean")
    m["changed"] = (a.isna() != b.isna()) | (a.fillna(False) != b.fillna(False))
    m["validated"] = m["sector"].isin(VALIDATED)
    m["_o"] = m["sector"].map({s: i for i, s in enumerate(SECTOR_REPORT_ORDER)})
    return m.sort_values(["product", "_o"]).drop(columns="_o").reset_index(drop=True)
