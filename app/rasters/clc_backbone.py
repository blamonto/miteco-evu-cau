"""CLC+ Backbone 2023 raster data access via Copernicus Data Space WMS."""

from __future__ import annotations

import hashlib
import io
import logging
import os
import time

import httpx
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds

from app.config import settings

logger = logging.getLogger(__name__)

# CLC+ Backbone green urban class codes
CLC_GREEN_CLASSES = frozenset({141, 142})

# Copernicus Data Space OAuth2 token endpoint
_TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu"
    "/auth/realms/CDSE/protocol/openid-connect/token"
)

# WMS endpoint for CLC+ Backbone on Copernicus
_WMS_URL = (
    "https://land.discomap.eea.europa.eu/arcgis/services"
    "/GioLandPublic/clc_2018/MapServer/WMSServer"
)

# CRS used by Copernicus pan-European products
CRS_3035 = CRS.from_epsg(3035)

# Token cache
_token_cache: dict[str, str | float] = {"token": "", "expires": 0.0}


class RasterAccessError(Exception):
    """Failed to fetch raster data."""


def get_copernicus_token() -> str:
    """Authenticate with Copernicus Data Space and return an access token."""
    now = time.time()
    if _token_cache["token"] and float(_token_cache["expires"]) > now + 60:
        return str(_token_cache["token"])

    client_id = settings.copernicus_client_id
    client_secret = settings.copernicus_client_secret
    if not client_id or not client_secret:
        raise RasterAccessError(
            "Copernicus credentials not configured. "
            "Set COPERNICUS_CLIENT_ID and COPERNICUS_CLIENT_SECRET."
        )

    resp = httpx.post(
        _TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    _token_cache["token"] = data["access_token"]
    _token_cache["expires"] = now + data.get("expires_in", 300)
    return str(_token_cache["token"])


def _bbox_hash(bbox: tuple[float, float, float, float]) -> str:
    return hashlib.md5(str(bbox).encode()).hexdigest()[:12]


def fetch_clc_backbone(
    bbox: tuple[float, float, float, float],
    token: str | None = None,
    *,
    crs: CRS = CRS_3035,
    pixel_size: int = 10,
) -> tuple[np.ndarray, dict]:
    """Fetch CLC+ Backbone 2023 raster for the given bounding box.

    Tries WMS GetMap first; falls back to cached file if available.

    Returns ``(raster_array, metadata)`` where metadata has keys
    ``transform``, ``crs``, ``nodata``.
    """
    cache_dir = os.path.join(settings.cache_dir, "clc_backbone")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"clc_{_bbox_hash(bbox)}.tif")

    if os.path.isfile(cache_file):
        logger.info("Loading cached CLC+ raster from %s", cache_file)
        return _read_raster(cache_file)

    # Compute pixel dimensions
    xmin, ymin, xmax, ymax = bbox
    width = max(1, int((xmax - xmin) / pixel_size))
    height = max(1, int((ymax - ymin) / pixel_size))

    # WMS GetMap request
    params = {
        "SERVICE": "WMS",
        "VERSION": "1.3.0",
        "REQUEST": "GetMap",
        "LAYERS": "clcplus_backbone_2021",
        "STYLES": "",
        "CRS": f"EPSG:{crs.to_epsg()}",
        "BBOX": f"{ymin},{xmin},{ymax},{xmax}",
        "WIDTH": str(min(width, 4096)),
        "HEIGHT": str(min(height, 4096)),
        "FORMAT": "image/tiff",
    }

    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    logger.info("Fetching CLC+ Backbone WMS: %dx%d pixels", width, height)
    try:
        resp = httpx.get(
            _WMS_URL, params=params, headers=headers,
            timeout=120, follow_redirects=True,
        )
        resp.raise_for_status()

        with open(cache_file, "wb") as f:
            f.write(resp.content)

        return _read_raster(cache_file)
    except (httpx.HTTPStatusError, httpx.RequestError) as exc:
        raise RasterAccessError(
            f"Failed to fetch CLC+ Backbone raster: {exc}"
        ) from exc


def _read_raster(filepath: str) -> tuple[np.ndarray, dict]:
    with rasterio.open(filepath) as src:
        data = src.read(1)
        meta = {
            "transform": src.transform,
            "crs": src.crs,
            "nodata": src.nodata or 0,
        }
    return data, meta
