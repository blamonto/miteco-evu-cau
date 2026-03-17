"""CAU (Cobertura Arbórea Urbana) indicator calculation.

Uses HRL Tree Cover Density raster (0-100 % per pixel, 10 m resolution).
"""

from __future__ import annotations

import numpy as np

PIXEL_AREA_M2 = 100  # 10 m × 10 m
TCD_NODATA = 255


def calculate_cau(
    tcd_raster: np.ndarray,
    *,
    nodata: int = TCD_NODATA,
) -> float:
    """Return total m² of effective tree canopy cover within the clipped ZEU raster.

    CAU = Σ (pixel_density / 100 × pixel_area) for all valid pixels.
    """
    valid = tcd_raster != nodata
    densities = tcd_raster[valid].astype(np.float64)
    return float(np.sum(densities / 100.0 * PIXEL_AREA_M2))
