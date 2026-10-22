"""North Caspian seabed-emergence pipeline.

Modules
-------
config       paths, projection definitions, study-area constants
credentials  secret lookup (never prints or logs secret values)
io           readers for the water-level series and other inputs
masks, water, levels, surge, waterlines, dem, conformal, hindcast, exposure
             analysis modules

Import this package before rasterio (scripts and tests/conftest.py do).
"""

import glob as _glob
import os as _os
import sys as _sys

__version__ = "0.1.0"


def _drop_foreign_gdal_from_path() -> None:
    """On Windows, rasterio registers every PATH directory that contains a
    gdal*.dll as a DLL search directory. A system-wide GDAL install (here
    OSGeo4W) then shadows the environment's own DLLs and rasterio fails with
    "The specified procedure could not be found". Remove such directories from
    PATH for this process only."""
    if not _sys.platform.startswith("win") or "PATH" not in _os.environ:
        return
    prefix = _os.path.normcase(_os.path.abspath(_sys.prefix))
    kept = []
    for entry in _os.environ["PATH"].split(_os.pathsep):
        absolute = _os.path.normcase(_os.path.abspath(entry)) if entry else ""
        foreign = absolute and not absolute.startswith(prefix)
        if foreign and _glob.glob(_os.path.join(entry, "gdal*.dll")):
            continue
        kept.append(entry)
    _os.environ["PATH"] = _os.pathsep.join(kept)


_drop_foreign_gdal_from_path()
