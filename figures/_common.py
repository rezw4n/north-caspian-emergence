"""Shared settings for the display-item scripts in figures/.

Widths follow the Nature Communications author guidelines read on 2026-10-21
(docs/journal_specs.md): single column 88 mm, double column 180 mm. They differ from
the placeholders in caspian.plotting (89 and 183 mm), which are not used here. The
venue is chosen at Gate G3; Communications Earth & Environment gives no width.

Text follows CLAUDE.md and the Nature Portfolio figure guide: sans-serif 5-7 pt,
lowercase bold 8 pt panel letters (caspian.plotting.use_style, panel_label). Files are
saved at the exact figure size (no tight cropping, which would change the width):
vector PDF with TrueType fonts, a 600 dpi LZW TIFF and a 200 dpi PNG preview, in
figures/out/.
"""

from __future__ import annotations

import json
from pathlib import Path

import caspian  # noqa: F401  (before any GDAL-linked import; D-004)
import matplotlib as mpl

from caspian import plotting

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "out"
MM = 1 / 25.4
SINGLE_MM = 88.0
DOUBLE_MM = 180.0

# Okabe-Ito hues, checked with the dataviz palette validator (light surface): CVD and
# normal-vision separation pass; orange is below 3:1 contrast, so every use carries a
# legend entry or a direct label.
SPLIT_COLOURS = {"train": "#0072B2", "calib": "#E69F00", "test": "#009E73"}
SPLIT_NAMES = {"train": "Training", "calib": "Calibration", "test": "Test"}
SECTOR_SHORT = {
    "west": "Kizlyar Bay–Dagestan–Kalmyk coast",
    "volga": "Volga delta",
    "interfluve": "Volga–Ural interfluve coast",
    "ural": "Ural delta",
    "northeast": "NE Kazakh coast (Tengiz–Kashagan)",
    "komsomolets": "Komsomolets Bay and Kaidak",
    "open_shelf": "Open shelf",
}


# Audited bathymetry products, in the order used on every results figure. Okabe-Ito
# hues, checked with the dataviz palette validator (all pairs, light surface): the
# normal-vision floor passes; ETOPO1 against SRTM15+ is in the 6-8 CVD band, so every
# product also has its own marker shape (secondary encoding). The waterline DEM is
# drawn in ink (plotting.INK), not as a categorical hue.
PRODUCTS = ("gebco_2026", "gebco_2022", "etopo_2022", "srtm15p_v2.7", "etopo1")
PRODUCT_NAMES = {"gebco_2026": "GEBCO_2026", "gebco_2022": "GEBCO_2022", "etopo_2022": "ETOPO 2022",
                 "srtm15p_v2.7": "SRTM15+ V2.7", "etopo1": "ETOPO1"}
PRODUCT_COLOURS = {"gebco_2026": "#0072B2", "gebco_2022": "#56B4E9", "etopo_2022": "#D55E00",
                   "srtm15p_v2.7": "#009E73", "etopo1": "#CC79A7"}
PRODUCT_MARKERS = {"gebco_2026": "o", "gebco_2022": "s", "etopo_2022": "D", "srtm15p_v2.7": "^", "etopo1": "v"}
SECTOR_TINY = {"west": "Kizlyar–Dagestan", "volga": "Volga delta", "interfluve": "Interfluve",
               "ural": "Ural delta", "northeast": "NE Kazakh coast", "komsomolets": "Komsomolets",
               "open_shelf": "Open shelf"}
SEA = "#d7e6f2"      # open sea, as on the study-area figure
LAND = "#ececec"     # land and seabed outside the mapped band
SCALE_KM = 100       # map scale bars (params.fig_scale_bar_km)
GRATICULE_DEG = 1    # map graticule (params.fig_graticule_deg)
MAP_RES_M = 150      # display grid of the DEM and emergence maps, 30 m decimated (params.fig_map_res_m)


def style() -> None:
    plotting.use_style()
    mpl.rcParams.update({"lines.linewidth": 0.75, "patch.linewidth": 0.5, "hatch.linewidth": 0.5})


def metrics() -> dict:
    """results/metrics.json, so that dates and counts drawn on a figure are the ones
    the text quotes."""
    return json.loads((ROOT / "results" / "metrics.json").read_text(encoding="utf-8"))


def save(fig, name: str) -> list[Path]:
    """Write figures/out/<name>.pdf, .tif (600 dpi) and .png (preview) at the figure's
    own size."""
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / name
    paths = [stem.with_suffix(".pdf"), stem.with_suffix(".tif"), stem.with_suffix(".png")]
    fig.savefig(paths[0])
    fig.savefig(paths[1], dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(paths[2], dpi=200)
    w, h = fig.get_size_inches() / MM
    print(f"{name}: {w:.1f} x {h:.1f} mm ->", ", ".join(p.relative_to(ROOT).as_posix() for p in paths))
    return paths


def scale_bar(ax, x0: float, y0: float, length_m: float, label: str, transform=None, lw: float = 1.2) -> None:
    """A plain bar of ``length_m`` (map units in metres) from (x0, y0), labelled above."""
    from matplotlib import patheffects

    kw = {} if transform is None else {"transform": transform}
    halo = [patheffects.withStroke(linewidth=1.6, foreground="white")]
    ax.plot([x0, x0 + length_m], [y0, y0], color="white", lw=lw + 1.2, solid_capstyle="butt", zorder=8, **kw)
    ax.plot([x0, x0 + length_m], [y0, y0], color=plotting.INK, lw=lw, solid_capstyle="butt", zorder=9, **kw)
    ax.annotate(label, xy=(x0 + length_m / 2, y0), xycoords=transform or "data", xytext=(0, 1.5),
                textcoords="offset points", ha="center", va="bottom", fontsize=5.5, color=plotting.INK,
                path_effects=halo, zorder=9)


# --- maps in the analysis projection (Lambert azimuthal equal area, D-002) ----------

def projection():
    """(cartopy projection, lon/lat -> equal-area transform) for caspian.config.CRS_EQUAL_AREA."""
    import cartopy.crs as ccrs
    import pyproj

    from caspian.config import CRS_EQUAL_AREA

    laea = pyproj.CRS(CRS_EQUAL_AREA)
    p = laea.to_dict()
    proj = ccrs.LambertAzimuthalEqualArea(central_longitude=p["lon_0"], central_latitude=p["lat_0"],
                                          globe=ccrs.Globe(ellipse="WGS84"))
    fwd = pyproj.Transformer.from_crs("EPSG:4326", laea, always_xy=True).transform
    return proj, fwd


def study_geometries():
    """(domain, {sector id: polygon}) in equal-area metres, from results/study_area/sectors.geojson."""
    from shapely.geometry import shape
    from shapely.ops import transform

    _, fwd = projection()
    feats = json.loads((ROOT / "results" / "study_area" / "sectors.geojson").read_text(encoding="utf-8"))["features"]
    geoms = {f["properties"]["id"]: transform(fwd, shape(f["geometry"]).segmentize(0.01)) for f in feats}
    return geoms.pop("domain"), geoms


def map_extent(domain, pad: float = 8_000.0) -> list[float]:
    x0, y0, x1, y1 = domain.bounds
    return [x0 - pad, x1 + pad, y0 - pad, y1 + pad]


def graticule(ax, labels: bool = True, size: float = 5.5):
    import cartopy.crs as ccrs
    import numpy as np

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=labels, x_inline=False, y_inline=False,
                      xlocs=np.arange(46, 56, GRATICULE_DEG), ylocs=np.arange(44, 48, GRATICULE_DEG),
                      color=plotting.GRID, lw=0.4, zorder=1)
    if labels:
        gl.top_labels = gl.right_labels = False
        gl.xlabel_style = gl.ylabel_style = {"size": size, "color": plotting.INK_MUTED}
        gl.xpadding = gl.ypadding = 2
    return gl


def sea_mask():
    """The D-029 open sea of the 2025-26 low stand (results/study_area/sea_mask_1km.npz) as
    (mask, imshow extent); for orientation only, not a product under audit."""
    import numpy as np

    sea = np.load(ROOT / "results" / "study_area" / "sea_mask_1km.npz")
    a, _, c0, _, e, f0 = sea["transform"]
    ny, nx = sea["mask"].shape
    return sea["mask"].astype(bool), [c0, c0 + a * nx, f0 + e * ny, f0]


def draw_outlines(ax, proj, domain, sectors, lw_sector: float = 0.45, lw_domain: float = 0.9, zorder: int = 6,
                  colour: str | None = None) -> None:
    c = colour or plotting.INK
    ax.add_geometries(list(sectors.values()), crs=proj, facecolor="none", edgecolor=c, lw=lw_sector, zorder=zorder)
    ax.add_geometries([domain], crs=proj, facecolor="none", edgecolor=c, lw=lw_domain, zorder=zorder + 1)


def write_crlf(path: Path, text: str) -> None:
    """Write a text file with CRLF line endings (project convention for new files)."""
    path.write_bytes(text.replace("\r\n", "\n").replace("\n", "\r\n").encode("utf-8"))
