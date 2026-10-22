"""QC figure: GEBCO_2026 elevation and Type Identifier (TID) grid over the domain.

Writes outputs/qc/gebco_2026_tid.{pdf,png}, the figure docs/DATUMS.md
cites for the TID coding of the North Caspian. Input: the
GEBCO_2026 subset written by scripts/a7_get_gebco.py (`make bathymetry-download`),
default data/raw/bathymetry/gebco_2026_ncaspian.nc, or the path given as argument.

Legend names come from the TID variable's own ``flag_meanings`` attribute (the
provider's metadata, also recorded in results/provenance/gebco_2026.json), never typed
by hand. In the GEBCO_2026 file that attribute has 19 names for 21 ``flag_values``:
the names for codes 46 and 47 run together
("iceberg_groundinggrounded_Argo_float_derived") and code 48 has none. The names
before that token are paired with the first values (codes 0-45) and those after it
with the last values (codes 70-72); a code in between would be labelled by number
only. Any other mismatch between the two attributes stops the script.

Map: longitude/latitude (WGS84), aspect 1/cos(45.6 deg); the scale bar is true at
its own latitude (geodesic on WGS84).

Usage: python figures/qc_gebco_tid.py [path/to/gebco_2026_ncaspian.nc]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import caspian  # noqa: F401  (before any GDAL-linked import)

GLUED = "iceberg_groundinggrounded_Argo_float_derived"
# Okabe-Ito colours, colour-blind safe, assigned to codes in ascending order.
CATEGORICAL = ["#BBBBBB", "#D55E00", "#0072B2", "#009E73", "#E69F00", "#56B4E9", "#CC79A7", "#F0E442", "#000000"]
ELEV_RANGE_M = (-40.0, -20.0)
SCALE_KM = 100.0


def tid_labels(flag_values: str, flag_meanings: str) -> dict[int, str]:
    """Code -> provider meaning, from the TID variable's attributes (see docstring)."""
    values = [int(v) for v in json.loads(flag_values)]
    names = flag_meanings.split()
    if len(names) == len(values):
        return dict(zip(values, names))
    glued = [i for i, t in enumerate(names) if t == GLUED]
    if len(glued) != 1 or len(values) - len(names) != 2:
        raise ValueError(f"flag_meanings ({len(names)} names) does not match flag_values ({len(values)}) "
                         "in the known GEBCO_2026 way; check the attributes before labelling")
    g = glued[0]
    tail = len(names) - g - 1
    out = dict(zip(values[:g], names[:g]))
    out.update(zip(values[len(values) - tail:], names[g + 1:]))
    return out


def scale_bar_end(lon0: float, lat: float, km: float) -> float:
    """Longitude reached by going ``km`` due east from (lon0, lat) on the WGS84 ellipsoid."""
    from pyproj import Geod

    lon1, _, _ = Geod(ellps="WGS84").fwd(lon0, lat, 90.0, km * 1000.0)
    return float(lon1)


def outline(ax, geom, **kw) -> None:
    for poly in getattr(geom, "geoms", [geom]):
        x, y = poly.exterior.xy
        ax.plot(x, y, **kw)


def main(argv: list[str]) -> int:
    import matplotlib.pyplot as plt
    import numpy as np
    import xarray as xr
    from cmocean import cm
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    from caspian import plotting
    from caspian.config import OUTPUTS, RAW
    from caspian.geography import DOMAIN, sectors

    path = Path(argv[0]) if argv else RAW / "bathymetry" / "gebco_2026_ncaspian.nc"
    if not path.exists():
        print(f"FAIL {path} is missing: run `make bathymetry-download`")
        return 1
    with xr.open_dataset(path) as ds:
        lat, lon = ds["lat"].values, ds["lon"].values
        elev = ds["elevation"].values.astype(float)
        tid = ds["tid"].values.astype(int)
        attrs = dict(ds["tid"].attrs)
    if not (np.all(np.diff(lat) > 0) and np.all(np.diff(lon) > 0)):
        print("FAIL expected ascending lat and lon")
        return 1
    names = tid_labels(attrs["flag_values"], attrs["flag_meanings"])
    codes = sorted(int(c) for c in np.unique(tid))
    if len(codes) > len(CATEGORICAL):
        print(f"FAIL {len(codes)} TID codes present, palette has {len(CATEGORICAL)}")
        return 1

    dlat, dlon = float(np.median(np.diff(lat))), float(np.median(np.diff(lon)))
    extent = [lon[0] - dlon / 2, lon[-1] + dlon / 2, lat[0] - dlat / 2, lat[-1] + dlat / 2]
    aspect = 1.0 / np.cos(np.deg2rad(0.5 * (lat[0] + lat[-1])))

    plotting.use_style()
    fig = plt.figure(figsize=(plotting.WIDTH_DOUBLE_MM * plotting.MM, 165 * plotting.MM))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.27], wspace=0.04, hspace=0.18,
                          left=0.07, right=0.99, bottom=0.05, top=0.97)
    axs = [fig.add_subplot(gs[i, 0]) for i in range(2)]
    side = [fig.add_subplot(gs[i, 1]) for i in range(2)]
    for s in side:
        s.axis("off")

    im = axs[0].imshow(elev, origin="lower", extent=extent, cmap=cm.deep_r, vmin=ELEV_RANGE_M[0],
                       vmax=ELEV_RANGE_M[1], interpolation="nearest", rasterized=True)
    cax = side[0].inset_axes([0.04, 0.1, 0.09, 0.8])
    cb = fig.colorbar(im, cax=cax, extend="both")
    cb.set_label("GEBCO_2026 elevation (m, product vertical reference)")
    cb.outline.set_linewidth(0.4)
    axs[0].set_title("GEBCO_2026 elevation", loc="left")

    cmap = ListedColormap(CATEGORICAL[:len(codes)])
    index = np.searchsorted(codes, tid)
    axs[1].imshow(index, origin="lower", extent=extent, cmap=cmap,
                  norm=BoundaryNorm(np.arange(len(codes) + 1) - 0.5, len(codes)), interpolation="nearest",
                  rasterized=True)
    handles = [Patch(facecolor=CATEGORICAL[i], edgecolor="none",
                     label=f"{c}  {names[c].replace('_', ' ')}" if c in names else f"{c}")
               for i, c in enumerate(codes)]
    side[1].legend(handles=handles, title="TID code (provider meaning)", loc="upper left",
                   bbox_to_anchor=(0.0, 0.92), borderaxespad=0.0, handlelength=1.2, alignment="left")
    axs[1].set_title("GEBCO_2026 Type Identifier (source of each cell)", loc="left")

    # Scale bar in the south-east corner, below the domain's southern edge (44.2 N there).
    lat_bar = lat[0] + 0.035 * (lat[-1] - lat[0])
    lon_end = lon[-1] - 0.03 * (lon[-1] - lon[0])
    lon_bar0 = lon_end - (scale_bar_end(lon_end, lat_bar, SCALE_KM) - lon_end)
    lon_bar1 = scale_bar_end(lon_bar0, lat_bar, SCALE_KM)
    for ax, letter in zip(axs, "ab"):
        for _, geom in sectors().values():
            outline(ax, geom, color=plotting.INK, lw=0.4)
        outline(ax, DOMAIN, color=plotting.INK, lw=0.7)
        ax.plot([lon_bar0, lon_bar1], [lat_bar, lat_bar], color=plotting.INK, lw=1.4, solid_capstyle="butt")
        ax.text(lon_bar0 - 0.08, lat_bar, f"{SCALE_KM:.0f} km", ha="right", va="center",
                fontsize=6, color=plotting.INK)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect(aspect)
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
        plotting.panel_label(ax, letter, x=-0.07)
    out = plotting.save(fig, OUTPUTS / "qc" / "gebco_2026_tid", tiff=False)
    plt.close(fig)
    counts = {c: int((tid == c).sum()) for c in codes}
    print("TID codes present (cells):", counts)
    print("wrote", ", ".join(p.relative_to(OUTPUTS.parent).as_posix() for p in out))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
