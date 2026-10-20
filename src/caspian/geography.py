"""Study domain and reporting sectors (brief section 6; D-014 in docs/DECISIONS.md).

Defined once here, in WGS84 longitude/latitude, and frozen before any
evaluation. Sectors partition the domain: coastal sectors are explicit polygons
and "open shelf" is the remainder.

Anchors on the conventional North/Middle Caspian boundary are taken from
OpenStreetMap (Nominatim, queried 2026-10-20):
  Chechen Island           43.9677 N, 47.6805 E  (place=island)
  Cape Tyub-Karagan        44.6443 N, 50.3131 E  (natural=cape)
"""

from __future__ import annotations

import json
from pathlib import Path

from shapely.geometry import Polygon, mapping
from shapely.ops import unary_union

CHECHEN_ISLAND = (47.6805, 43.9677)
CAPE_TYUB_KARAGAN = (50.3131, 44.6443)

# Domain: everything north of the Chechen Island - Cape Tyub-Karagan line, within
# a generous box that contains the 1985-2026 maximum water extent. East of the cape
# the southern edge runs over the land of Mangyshlak, so Mangyshlak Bay, the Buzachi
# coast, Komsomolets Bay and the Kaidak flats are inside.
DOMAIN = Polygon([
    (46.40, 43.9750), CHECHEN_ISLAND, CAPE_TYUB_KARAGAN, (51.00, 44.30), (52.30, 44.20),
    (54.60, 44.20), (54.60, 47.35), (46.40, 47.35),
])

# Coastal sectors (lon, lat), drawn by eye on the JRC Global Surface Water
# transitions layer and frozen before any evaluation (D-014). Their offshore edges
# (the parts shared with the open shelf) are not at a set distance from the coast.
# scripts/a0_sector_edges.py measures them against the present shoreline traced in
# each relief grid (results/study_area/sector_edges.json; study_area.layout.* in
# results/metrics.json). The reach differs by sector and grid: the west edge starts
# at Chechen Island, on the coast; the Volga delta and Komsomolets sectors reach
# least far; and most of the Komsomolets edge runs over land. Part of the emerged
# band therefore falls in the open shelf, which also holds the Buzachi coast and
# Mangyshlak Bay. (Corrected 2026-10-20, review finding 18: this comment said the
# edges sit roughly 25-60 km seaward, beyond the emerged band; that was never
# measured, and it is not true of every sector.)
_COASTAL = {
    "west": ("Kizlyar Bay - Dagestan - Kalmyk coast", [
        (46.40, 43.9750), CHECHEN_ISLAND, (47.75, 44.50), (47.85, 45.30), (47.70, 45.62),
        (46.40, 45.62)]),
    "volga": ("Volga delta", [
        (46.40, 45.62), (47.70, 45.62), (47.85, 45.30), (48.40, 45.35), (49.00, 45.60),
        (49.60, 45.90), (50.05, 46.20), (50.05, 47.35), (46.40, 47.35)]),
    "interfluve": ("Volga-Ural interfluve coast", [
        (50.05, 46.20), (51.35, 46.45), (51.35, 47.35), (50.05, 47.35)]),
    "ural": ("Ural delta", [
        (51.35, 46.45), (52.20, 46.45), (52.20, 47.35), (51.35, 47.35)]),
    "northeast": ("NE Kazakh coast (Tengiz-Kashagan)", [
        (52.20, 46.45), (52.25, 46.00), (52.45, 45.65), (54.60, 45.65), (54.60, 47.35),
        (52.20, 47.35)]),
    "komsomolets": ("Komsomolets Bay (Mertvyi Kultuk) and Kaidak", [
        (52.45, 45.65), (52.45, 45.30), (52.55, 45.25), (52.75, 44.20), (54.60, 44.20),
        (54.60, 45.65)]),
}

SECTOR_ORDER = ["west", "volga", "interfluve", "ural", "northeast", "komsomolets", "open_shelf"]


def sectors() -> dict[str, tuple[str, Polygon]]:
    """Sector id -> (display name, polygon clipped to the domain). Disjoint, and
    their union is the domain."""
    out: dict[str, tuple[str, Polygon]] = {}
    for key, (name, coords) in _COASTAL.items():
        out[key] = (name, Polygon(coords).intersection(DOMAIN))
    coastal = unary_union([g for _, g in out.values()])
    out["open_shelf"] = ("Open shelf", DOMAIN.difference(coastal))
    return {k: out[k] for k in SECTOR_ORDER}


def to_geojson(path: str | Path) -> Path:
    path = Path(path)
    feats = [{"type": "Feature", "properties": {"id": "domain", "name": "North Caspian domain"},
              "geometry": mapping(DOMAIN)}]
    for key, (name, geom) in sectors().items():
        feats.append({"type": "Feature", "properties": {"id": key, "name": name},
                      "geometry": mapping(geom)})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"type": "FeatureCollection", "features": feats}, indent=1),
                    encoding="utf-8")
    return path
