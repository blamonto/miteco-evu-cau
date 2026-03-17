"""EVU (Espacio Verde Urbano) indicator calculation.

Uses CLC+ Backbone 2023 raster data. Green urban classes:
  - 141: Green urban areas
  - 142: Sport and leisure facilities
"""

from __future__ import annotations

import numpy as np

PIXEL_AREA_M2 = 100  # 10 m × 10 m
CLC_GREEN_CLASSES = frozenset({141, 142})


def calculate_evu(
    clc_raster: np.ndarray,
    *,
    nodata: int = 0,
    green_classes: set[int] | frozenset[int] = CLC_GREEN_CLASSES,
) -> float:
    """Return total m² of urban green space within the clipped ZEU raster.

    EVU = count_of_green_pixels × pixel_area (100 m²).
    """
    valid = clc_raster != nodata
    green = np.isin(clc_raster, list(green_classes))
    return float(np.sum(valid & green) * PIXEL_AREA_M2)


def create_evu_mask(
    clc_raster: np.ndarray,
    *,
    nodata: int = 0,
    green_classes: set[int] | frozenset[int] = CLC_GREEN_CLASSES,
) -> np.ndarray:
    """Return a uint8 mask: 1 = green, 0 = not green, 255 = nodata."""
    mask = np.zeros_like(clc_raster, dtype=np.uint8)
    is_nodata = clc_raster == nodata
    is_green = np.isin(clc_raster, list(green_classes))
    mask[is_green] = 1
    mask[is_nodata] = 255
    return mask
