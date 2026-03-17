"""HRL Tree Cover Density raster data access via EEA ArcGIS REST."""

from __future__ import annotations

import hashlib
import logging
import os

import httpx
import numpy as np
import rasterio
from rasterio.crs import CRS

from app.config import settings

logger = logging.getLogger(__name__)

# EEA ArcGIS REST endpoint for HRL Tree Cover Density
_REST_URL = (
    "https://image.discomap.eea.europa.eu/arcgis/rest/services"
    "/GioLandPublic/HRL_TreeCoverDensity_2018/ImageServer/exportImage"
)

CRS_3035 = CRS.from_epsg(3035)


class RasterAccessError(Exception):
    """Failed to fetch raster data."""


def _bbox_hash(bbox: tuple[float, float, float, float]) -> str:
    return hashlib.md5(str(bbox).encode()).hexdigest()[:12]


def fetch_hrl_tcd(
    bbox: tuple[float, float, float, float],
    *,
    crs: CRS = CRS_3035,
    pixel_size: int = 10,
) -> tuple[np.ndarray, dict]:
    """Fetch HRL Tree Cover Density raster for the given bounding box.

    Data is open access — no authentication required.

    Returns ``(raster_array, metadata)`` where metadata has keys
    ``transform``, ``crs``, ``nodata``.
    """
    cache_dir = os.path.join(settings.cache_dir, "hrl_tcd")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"tcd_{_bbox_hash(bbox)}.tif")

    if os.path.isfile(cache_file):
        logger.info("Loading cached HRL TCD raster from %s", cache_file)
        return _read_raster(cache_file)

    xmin, ymin, xmax, ymax = bbox
    width = max(1, int((xmax - xmin) / pixel_size))
    height = max(1, int((ymax - ymin) / pixel_size))

    params = {
        "bbox": f"{xmin},{ymin},{xmax},{ymax}",
        "bboxSR": str(crs.to_epsg()),
        "imageSR": str(crs.to_epsg()),
        "size": f"{min(width, 4096)},{min(height, 4096)}",
        "format": "tiff",
        "f": "image",
    }

    token = settings.eea_api_token
    if token:
        params["token"] = token

    logger.info("Fetching HRL TCD REST: %dx%d pixels", width, height)
    try:
        resp = httpx.get(
            _REST_URL, params=params,
            timeout=120, follow_redirects=True,
        )
        resp.raise_for_status()

        with open(cache_file, "wb") as f:
            f.write(resp.content)

        return _read_raster(cache_file)
    except (httpx.HTTPStatusError, httpx.RequestError) as exc:
        raise RasterAccessError(
            f"Failed to fetch HRL Tree Cover Density raster: {exc}"
        ) from exc


def _read_raster(filepath: str) -> tuple[np.ndarray, dict]:
    with rasterio.open(filepath) as src:
        data = src.read(1)
        meta = {
            "transform": src.transform,
            "crs": src.crs,
            "nodata": src.nodata if src.nodata is not None else 255,
        }
    return data, meta
