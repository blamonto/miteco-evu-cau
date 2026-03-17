"""GeoTIFF cartography writer."""

from __future__ import annotations

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine


def write_geotiff(
    data: np.ndarray,
    transform: Affine,
    crs: CRS,
    filepath: str,
    *,
    nodata: int = 0,
    dtype: str = "uint8",
) -> str:
    """Write a 2-D numpy array as a single-band GeoTIFF. Returns *filepath*."""
    with rasterio.open(
        filepath,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
        compress="deflate",
    ) as dst:
        dst.write(data.astype(dtype), 1)
    return filepath
