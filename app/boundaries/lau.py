"""Eurostat LAU boundary download and filtering."""

from __future__ import annotations

import logging
import os

import geopandas as gpd
import httpx

from app.config import settings

logger = logging.getLogger(__name__)

# Known URL patterns for Eurostat GISCO LAU distribution
_URL_TEMPLATES = [
    "{base}ref-lau-{year}-01m.geojson/LAU_RG_01M_{year}_{cc}.geojson",
    "{base}ref-lau-{year}-01m.shp/LAU_RG_01M_{year}_{cc}.shp.zip",
]


class BoundaryNotFoundError(ValueError):
    """LAU code not found in boundary dataset."""


class DataDownloadError(Exception):
    """Remote data download failed."""


def _cache_path(country_code: str, year: int) -> str:
    d = os.path.join(settings.cache_dir, "lau")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"LAU_{country_code}_{year}.geojson")


def download_lau_boundaries(
    country_code: str = "ES",
    year: int | None = None,
) -> gpd.GeoDataFrame:
    """Download LAU boundaries for a country from Eurostat GISCO.

    Results are cached locally to avoid re-downloading.
    """
    year = year or settings.lau_year
    cached = _cache_path(country_code, year)

    if os.path.isfile(cached):
        logger.info("Loading cached LAU boundaries from %s", cached)
        return gpd.read_file(cached)

    last_error: Exception | None = None
    for tpl in _URL_TEMPLATES:
        url = tpl.format(base=settings.lau_base_url, year=year, cc=country_code)
        logger.info("Downloading LAU boundaries: %s", url)
        try:
            resp = httpx.get(url, timeout=60, follow_redirects=True)
            resp.raise_for_status()
            # Write raw bytes and read with geopandas
            tmp = cached + ".tmp"
            with open(tmp, "wb") as f:
                f.write(resp.content)
            gdf = gpd.read_file(tmp)
            # Save as GeoJSON for fast future reads
            gdf.to_file(cached, driver="GeoJSON")
            os.remove(tmp)
            return gdf
        except (httpx.HTTPStatusError, httpx.RequestError) as exc:
            logger.warning("Failed to download from %s: %s", url, exc)
            last_error = exc
            continue

    raise DataDownloadError(
        f"Could not download LAU boundaries for {country_code}/{year}"
    ) from last_error


def get_lau_boundary(
    lau_code: str,
    country_code: str = "ES",
    year: int | None = None,
) -> tuple[gpd.GeoDataFrame, str]:
    """Get the boundary polygon for a specific LAU code.

    Returns ``(gdf_single_row, municipality_name)``.
    Raises :class:`BoundaryNotFoundError` if the code is not found.
    """
    gdf = download_lau_boundaries(country_code, year)

    # Detect the LAU code column (varies across years)
    lau_col = None
    for candidate in ("LAU_ID", "LAU_CODE", "lau_id", "lau_code", "GISCO_ID"):
        if candidate in gdf.columns:
            lau_col = candidate
            break
    if lau_col is None:
        raise BoundaryNotFoundError(
            f"Cannot find LAU code column in dataset. Columns: {list(gdf.columns)}"
        )

    # Normalise: strip country prefix if present ("ES_28079" → "28079")
    code = lau_code.split("_")[-1] if "_" in lau_code else lau_code

    match = gdf[gdf[lau_col].astype(str).str.endswith(code)]
    if match.empty:
        raise BoundaryNotFoundError(f"LAU code '{lau_code}' not found in dataset")

    row = match.iloc[[0]]

    # Extract municipality name
    name_col = None
    for candidate in ("LAU_NAME", "GISCO_ID", "NAME_LATN", "name", "lau_name"):
        if candidate in gdf.columns:
            name_col = candidate
            break
    name = str(row.iloc[0][name_col]) if name_col else lau_code

    return row.reset_index(drop=True), name


def compute_zeu_area(gdf: gpd.GeoDataFrame) -> float:
    """Compute the ZEU area in m² (reprojects to UTM for accuracy)."""
    utm_crs = gdf.estimate_utm_crs()
    projected = gdf.to_crs(utm_crs)
    return float(projected.geometry.area.sum())
