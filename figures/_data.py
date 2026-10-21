"""Reduced-size reads of the large data products behind the results figures, cached
under data/interim/figures/ (git-ignored). Nothing here computes a result: the
functions decimate rasters, select rows and subsample points for display, and the
ICESat-2 selection is checked against the committed validation tables before use.

Each cache file stores the size and modification time of every input it was built
from, and is rebuilt when any of them changes. Peak memory stays well under 1 GB
(rasters are read decimated tile by tile; point tables are read in chunks of
selected columns).

- mosaic(kind, res_m): the DEM height ("dem"), its interpolation distance ("dist")
  or the emergence date ("emergence", days since 1970) on one grid at res_m metres,
  from data/processed/dem/main/ (60 km tiles at 30 m; height and distance averaged,
  emergence date by nearest pixel).
- icesat2_test(source): the ICESat-2 points of the H2 population (D-084: in the
  emerged band and E3 vegetation share below 0.5, from the cached annotations of
  scripts/a6_validate_dem.py) in the 50 km test blocks (results/study_area/
  blocks_50km.csv), with the reference height on the DEM reference (H + c, the
  training-block constant of results/validation/main/<source>_datum_constant.json)
  and the product heights with their training offsets (the "matched" variant).
- waterlines_by_block(): counts of E2 + E3 waterline points per 25 km block, and
  waterlines_in_box(): every waterline point inside a box.
"""

from __future__ import annotations

import json
from pathlib import Path

import _common as C
import numpy as np
import pandas as pd

from caspian.config import DATA, INTERIM, RESULTS

CACHE = INTERIM / "figures"
DEM_DIR = DATA / "processed" / "dem" / "main"
PTS_DIR = INTERIM / "validation" / "main"
WL = INTERIM / "waterlines" / "main" / "points.csv.gz"
TILE_M = 60_000.0
TILE_PX = 2000


def _stamp(paths) -> str:
    return json.dumps([[Path(p).name, Path(p).stat().st_size, Path(p).stat().st_mtime_ns] for p in sorted(map(str, paths))])


def _cached(name: str, inputs, build) -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{name}.npz"
    stamp = _stamp(inputs)
    if path.exists():
        z = np.load(path, allow_pickle=False)
        if str(z["_stamp"]) == stamp:
            return {k: z[k] for k in z.files if k != "_stamp"}
    out = build()
    np.savez_compressed(path, _stamp=np.array(stamp), **out)
    return out


def _tiles(prefix: str, folder: Path) -> list[tuple[int, int, Path]]:
    out = []
    for f in sorted(folder.glob(f"{prefix}_*.tif")):
        i, j = (int(v) for v in f.stem[len(prefix) + 1:].split("_"))
        out.append((i, j, f))
    return out


def mosaic(kind: str, res_m: float = 150.0) -> dict:
    """{"z": array (north up), "extent": [x0, x1, y0, y1]} on a res_m grid."""
    import rasterio
    from rasterio.enums import Resampling

    folder, prefix, band, rs = {"dem": (DEM_DIR, "dem", 1, Resampling.average),
                                "dist": (DEM_DIR, "dem", 2, Resampling.average),
                                "emergence": (DEM_DIR / "emergence", "emergence", 1, Resampling.nearest)}[kind]
    tiles = _tiles(prefix, folder)
    n = int(round(TILE_PX * 30.0 / res_m))
    if abs(n * res_m - TILE_M) > 1e-6:
        raise ValueError(f"{res_m} m does not divide the 60 km tile")

    def build():
        i0, i1 = min(t[0] for t in tiles), max(t[0] for t in tiles)
        j0, j1 = min(t[1] for t in tiles), max(t[1] for t in tiles)
        z = np.full(((j1 - j0 + 1) * n, (i1 - i0 + 1) * n), np.nan, np.float32)
        for i, j, f in tiles:
            with rasterio.open(f) as r:
                if abs(r.transform.a - 30.0) > 1e-9 or r.width != TILE_PX or abs(r.transform.c - i * TILE_M) > 1e-6:
                    raise ValueError(f"unexpected tile geometry in {f.name}")
                a = r.read(band, out_shape=(n, n), resampling=rs)
            row = (j1 - j) * n
            col = (i - i0) * n
            z[row:row + n, col:col + n] = a
        ext = np.array([i0 * TILE_M, (i1 + 1) * TILE_M, j0 * TILE_M, (j1 + 1) * TILE_M], float)
        return {"z": z, "extent": ext}

    return _cached(f"mosaic_{kind}_{int(res_m)}m", [t[2] for t in tiles], build)


def icesat2_test(source: str) -> pd.DataFrame:
    """Test-block points of the H2 population (see the module docstring)."""
    from caspian import blocking

    pts = PTS_DIR / f"{source}_points.csv.gz"
    ann = PTS_DIR / f"{source}_annotations.npz"
    blocks_p = RESULTS / "study_area" / "blocks_50km.csv"
    const_p = RESULTS / "validation" / "main" / f"{source}_datum_constant.json"
    offs_p = RESULTS / "validation" / "main" / f"{source}_product_offsets.json"
    prods = [f"prod:{p}" for p in C.PRODUCTS]
    cols = ["x", "y", "h_orth", "dem_h", "interp_dist", *prods]

    def build():
        a = np.load(ann)
        in_band, veg = a["in_band"], a["veg_share"]
        blocks = pd.read_csv(blocks_p)
        lut = {(i, j): (s, sec) for i, j, s, sec in zip(blocks["i"], blocks["j"], blocks["split"], blocks["sector"])}
        keep, row0 = [], 0
        for ch in pd.read_csv(pts, usecols=cols, dtype={c: "float64" for c in cols}, chunksize=1_000_000):
            n = len(ch)
            sel = in_band[row0:row0 + n] & (veg[row0:row0 + n] < 0.5)
            row0 += n
            ch = ch[sel]
            bi, bj = blocking.block_index(ch["x"], ch["y"], 50_000.0)
            meta = [lut.get((u, v), (None, None)) for u, v in zip(bi, bj)]
            split = np.array([m[0] for m in meta], dtype=object)
            t = ch[split == "test"].copy()
            t["block"] = [f"{u}_{v}" for u, v, s in zip(bi, bj, split) if s == "test"]
            t["sector"] = [m[1] for m, s in zip(meta, split) if s == "test"]
            keep.append(t)
        if row0 != len(in_band):
            raise ValueError(f"{source}: {row0} point rows but {len(in_band)} annotations")
        d = pd.concat(keep, ignore_index=True)
        out = {c.replace("prod:", "p_"): d[c].to_numpy(np.float32) for c in cols}
        out["block"] = d["block"].to_numpy(str)
        out["sector"] = d["sector"].to_numpy(str)
        return out

    z = _cached(f"icesat2_test_bare_{source}", [pts, ann, blocks_p], build)
    d = pd.DataFrame({k: z[k] for k in z})
    c = json.loads(const_p.read_text(encoding="utf-8"))["constant"]["constant_m"]
    offs = json.loads(offs_p.read_text(encoding="utf-8"))["product_offsets"]
    d["y_ref"] = d["h_orth"].astype(float) + c
    for p in C.PRODUCTS:
        d[f"{p}|matched"] = d[f"p_{p}"].astype(float) + offs[p]["offset_m"]
    _check_against_committed(d, source)
    return d


def _check_against_committed(d: pd.DataFrame, source: str) -> None:
    """The selection must reproduce the committed test-block tables exactly (count) and
    their DEM statistics (to rounding), or the figure would show different points."""
    from caspian import validation

    acc = pd.read_csv(RESULTS / "validation" / "main" / f"{source}_accuracy.csv")
    o = acc[(acc.scope == "overall") & (acc.model == "dem")].iloc[0]
    e = d["dem_h"].to_numpy(float) - d["y_ref"].to_numpy(float)
    if len(d) != int(o["n"]) or d["block"].nunique() != int(o["n_blocks"]):
        raise SystemExit(f"FAIL {source}: {len(d)} points in {d['block'].nunique()} blocks, "
                         f"committed {int(o['n'])} in {int(o['n_blocks'])}")
    if abs(validation.nmad(e) - float(o["nmad"])) > 1e-4 or abs(validation.rmse(e) - float(o["rmse"])) > 1e-4:
        raise SystemExit(f"FAIL {source}: DEM NMAD/RMSE of the selection differ from {source}_accuracy.csv")


def _wl_blocks_build():
    from caspian import blocking

    parts = []
    for ch in pd.read_csv(WL, usecols=["x", "y", "epoch", "bin", "static"], chunksize=1_000_000):
        ch = ch[ch["epoch"].isin(["E2", "E3"])]
        bi, bj = blocking.block_index(ch["x"], ch["y"], 25_000.0)
        parts.append(pd.DataFrame({"i": bi, "j": bj, "bin": ch["bin"].to_numpy(int),
                                   "static": ch["static"].astype(bool).to_numpy()}))
    p = pd.concat(parts, ignore_index=True)
    kept = p[~p["static"]]
    g = kept.groupby(["i", "j"]).agg(n=("bin", "size"), n_bins=("bin", "nunique")).reset_index()
    s = p[p["static"]].groupby(["i", "j"]).size().rename("n_static").reset_index()
    g = g.merge(s, on=["i", "j"], how="left").fillna({"n_static": 0})
    return {k: g[k].to_numpy() for k in g.columns}


def waterlines_by_block() -> pd.DataFrame:
    """E2 + E3 waterline points per 25 km block: n (used by the DEM), n_bins (distinct
    level bins among them) and n_static (removed by the static-contour rule, D-034)."""
    z = _cached("waterlines_by_block25", [WL], _wl_blocks_build)
    return pd.DataFrame(z)


def waterlines_in_box(x0: float, y0: float, x1: float, y1: float) -> pd.DataFrame:
    def build():
        parts = []
        for ch in pd.read_csv(WL, usecols=["x", "y", "epoch", "bin", "z_m", "static"], chunksize=1_000_000):
            m = (ch["x"] >= x0) & (ch["x"] < x1) & (ch["y"] >= y0) & (ch["y"] < y1)
            parts.append(ch[m])
        d = pd.concat(parts, ignore_index=True)
        return {"x": d["x"].to_numpy(), "y": d["y"].to_numpy(), "epoch": d["epoch"].to_numpy(str),
                "bin": d["bin"].to_numpy(int), "z_m": d["z_m"].to_numpy(), "static": d["static"].astype(bool).to_numpy()}

    z = _cached(f"waterlines_box_{int(x0)}_{int(y0)}_{int(x1)}_{int(y1)}", [WL], build)
    return pd.DataFrame(z)


def emerged_band(extent, shape) -> np.ndarray:
    """The D-071 emerged band (water in the 1995-96 and not water in the 2025-26 zone-v6
    extent; results/study_area/extents_v6_250m.npz), by nearest 250 m cell, on a grid of
    ``shape`` covering ``extent`` = [x0, x1, y0, y1] (north up)."""
    ex = np.load(RESULTS / "study_area" / "extents_v6_250m.npz")
    a, _, c0, _, e, f0 = ex["transform"]
    ny, nx = shape
    rx = (extent[1] - extent[0]) / nx
    ry = (extent[3] - extent[2]) / ny
    xs = extent[0] + (np.arange(nx) + 0.5) * rx
    ys = extent[3] - (np.arange(ny) + 0.5) * ry
    ci = np.floor((xs - c0) / a).astype(int)
    ri = np.floor((ys - f0) / e).astype(int)
    band = (ex["high"] == 1) & (ex["low"] == 0)
    okc = (ci >= 0) & (ci < band.shape[1])
    okr = (ri >= 0) & (ri < band.shape[0])
    out = np.zeros(shape, bool)
    out[np.ix_(okr, okc)] = band[np.ix_(ri[okr], ci[okc])]
    return out
