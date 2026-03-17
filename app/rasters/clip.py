"""Raster clipping utilities."""

from __future__ import annotations

import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import Affine, from_bounds
from rasterio.crs import CRS
from pyproj import Transformer
from shapely.geometry import mapping, shape
from shapely.ops import transform as shapely_transform


def reproject_polygon(
    polygon,
    src_crs: CRS,
    dst_crs: CRS,
):
    """Reproject a Shapely geometry between coordinate reference systems."""
    if src_crs == dst_crs:
        return polygon
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return shapely_transform(transformer.transform, polygon)


def clip_raster_to_polygon(
    raster_data: np.ndarray,
    transform: Affine,
    raster_crs: CRS,
    polygon,
    polygon_crs: CRS,
    nodata: int = 0,
) -> tuple[np.ndarray, Affine]:
    """Clip a 2-D raster to a polygon, setting pixels outside to *nodata*.

    If CRS differ the polygon is reprojected to match the raster.
    Returns ``(clipped_raster, original_transform)`` — same shape as input,
    but pixels outside the polygon are set to *nodata*.
    """
    poly = reproject_polygon(polygon, polygon_crs, raster_crs)

    mask = geometry_mask(
        [mapping(poly)],
        out_shape=raster_data.shape,
        transform=transform,
        invert=False,  # True inside → we invert below
    )
    # geometry_mask returns True where data should be MASKED (outside)
    clipped = raster_data.copy()
    clipped[mask] = nodata
    return clipped, transform
