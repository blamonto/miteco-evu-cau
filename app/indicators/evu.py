"""EVU (Espacio Verde Urbano) indicator calculation.

Supports two CLC products:

**CORINE Land Cover (44 clases)** — recommended for MITECO:
  - 141: Green urban areas
  - 142: Sport and leisure facilities

**CLC+ Backbone (11 clases)** — alternative:
  - 4: Woody needle leaved trees
  - 5: Woody broad leaved trees
  - 6: Permanent herbaceous (closest to green urban)
  - Note: CLC+ Backbone does NOT have explicit urban green classes.
    Classes 4, 5, 6 within the ZEU are the best approximation.

The script auto-detects which product is loaded by checking pixel values.
"""

from __future__ import annotations

import numpy as np

PIXEL_AREA_M2 = 100  # 10 m × 10 m

# CORINE Land Cover classic (44 classes) — the MITECO standard
CLC_GREEN_CLASSES = frozenset({141, 142})

# CLC+ Backbone (11 classes) — alternative mapping
# Within urban ZEU, these are the vegetation classes:
#   4 = Woody needle leaved trees
#   5 = Woody broad leaved trees
#   6 = Permanent herbaceous
CLCPLUS_GREEN_CLASSES = frozenset({4, 5, 6})


def detect_clc_product(clc_raster: np.ndarray, nodata: int = 0) -> str:
    """Detect whether a CLC raster is CORINE classic (44 classes) or CLC+ Backbone (11 classes).

    Returns 'corine' or 'clcplus'.
    """
    valid = clc_raster[clc_raster != nodata]
    if valid.size == 0:
        return "corine"
    max_val = int(valid.max())
    # CLC+ Backbone has values 1-11, CORINE has values 111-523
    if max_val <= 15:
        return "clcplus"
    return "corine"


def get_green_classes(clc_raster: np.ndarray, nodata: int = 0) -> frozenset[int]:
    """Auto-detect CLC product and return appropriate green classes."""
    product = detect_clc_product(clc_raster, nodata)
    if product == "clcplus":
        return CLCPLUS_GREEN_CLASSES
    return CLC_GREEN_CLASSES


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
