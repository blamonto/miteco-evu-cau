"""Shared test fixtures for miteco-service."""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import box

RNG = np.random.default_rng(42)


# ── Synthetic CLC+ Backbone raster ──────────────────────────────────

@pytest.fixture
def synthetic_clc_raster() -> np.ndarray:
    """100×100 CLC raster with deterministic class distribution.

    Layout (10 000 pixels total):
      rows  0-29  (3000 px) → class 141  (green urban)
      rows 30-39  (1000 px) → class 142  (sport/leisure)
      rows 40-89  (5000 px) → class 111  (continuous urban fabric)
      rows 90-99  (1000 px) → nodata (0)

    Expected green pixels = 4000  →  EVU = 400 000 m²
    """
    raster = np.empty((100, 100), dtype=np.uint8)
    raster[0:30, :] = 141
    raster[30:40, :] = 142
    raster[40:90, :] = 111
    raster[90:100, :] = 0  # nodata
    return raster


# ── Synthetic HRL Tree Cover Density raster ─────────────────────────

@pytest.fixture
def synthetic_tcd_raster() -> np.ndarray:
    """100×100 TCD raster with known density values.

    Layout (10 000 pixels total):
      rows  0-19  (2000 px) → density 80%
      rows 20-49  (3000 px) → density 40%
      rows 50-89  (4000 px) → density  0%
      rows 90-99  (1000 px) → nodata (255)

    Expected CAU = (2000×0.80 + 3000×0.40 + 4000×0.00) × 100 = 280 000 m²
    """
    raster = np.empty((100, 100), dtype=np.uint8)
    raster[0:20, :] = 80
    raster[20:50, :] = 40
    raster[50:90, :] = 0
    raster[90:100, :] = 255  # nodata
    return raster


# ── Sample ZEU polygon ──────────────────────────────────────────────

@pytest.fixture
def sample_zeu_polygon():
    """1 km × 1 km square around Madrid center (EPSG:4326)."""
    cx, cy = -3.7038, 40.4168
    half = 0.005  # ~500 m at this latitude
    return box(cx - half, cy - half, cx + half, cy + half)
