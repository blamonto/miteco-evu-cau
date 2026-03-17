"""Tests for EVU indicator calculation."""

import numpy as np
import pytest

from app.indicators.evu import PIXEL_AREA_M2, calculate_evu, create_evu_mask

pytestmark = pytest.mark.unit


class TestCalculateEVU:
    def test_known_distribution(self, synthetic_clc_raster):
        """4000 green pixels × 100 m² = 400 000 m²."""
        assert calculate_evu(synthetic_clc_raster) == 400_000.0

    def test_all_green(self):
        raster = np.full((10, 10), 141, dtype=np.uint8)
        assert calculate_evu(raster) == 100 * PIXEL_AREA_M2

    def test_no_green(self):
        raster = np.full((10, 10), 111, dtype=np.uint8)
        assert calculate_evu(raster) == 0.0

    def test_nodata_excluded(self):
        raster = np.full((10, 10), 0, dtype=np.uint8)  # all nodata
        assert calculate_evu(raster) == 0.0

    def test_only_class_141(self):
        raster = np.array([[141, 111], [111, 111]], dtype=np.uint8)
        assert calculate_evu(raster) == 1 * PIXEL_AREA_M2

    def test_only_class_142(self):
        raster = np.array([[142, 111], [111, 111]], dtype=np.uint8)
        assert calculate_evu(raster) == 1 * PIXEL_AREA_M2

    def test_empty_raster(self):
        raster = np.empty((0, 0), dtype=np.uint8)
        assert calculate_evu(raster) == 0.0

    def test_single_green_pixel(self):
        raster = np.array([[141]], dtype=np.uint8)
        assert calculate_evu(raster) == PIXEL_AREA_M2

    def test_result_is_float(self, synthetic_clc_raster):
        assert isinstance(calculate_evu(synthetic_clc_raster), float)

    def test_custom_green_classes(self):
        raster = np.array([[200, 201, 111]], dtype=np.uint8)
        result = calculate_evu(raster, green_classes={200, 201})
        assert result == 2 * PIXEL_AREA_M2


class TestCreateEVUMask:
    def test_binary_output(self, synthetic_clc_raster):
        mask = create_evu_mask(synthetic_clc_raster)
        unique = set(np.unique(mask))
        assert unique <= {0, 1, 255}

    def test_green_pixels_are_ones(self):
        raster = np.array([[141, 142], [111, 0]], dtype=np.uint8)
        mask = create_evu_mask(raster)
        assert mask[0, 0] == 1
        assert mask[0, 1] == 1

    def test_non_green_pixels_are_zeros(self):
        raster = np.array([[141, 142], [111, 0]], dtype=np.uint8)
        mask = create_evu_mask(raster)
        assert mask[1, 0] == 0  # urban → 0

    def test_nodata_is_255(self):
        raster = np.array([[141, 142], [111, 0]], dtype=np.uint8)
        mask = create_evu_mask(raster)
        assert mask[1, 1] == 255  # nodata → 255
