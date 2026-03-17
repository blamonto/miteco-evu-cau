"""Tests for CAU indicator calculation."""

import numpy as np
import pytest

from app.indicators.cau import PIXEL_AREA_M2, TCD_NODATA, calculate_cau

pytestmark = pytest.mark.unit


class TestCalculateCAU:
    def test_known_distribution(self, synthetic_tcd_raster):
        """(2000×0.80 + 3000×0.40) × 100 = 280 000 m²."""
        assert calculate_cau(synthetic_tcd_raster) == pytest.approx(280_000.0)

    def test_all_100_density(self):
        raster = np.full((10, 10), 100, dtype=np.uint8)
        assert calculate_cau(raster) == pytest.approx(100 * PIXEL_AREA_M2)

    def test_all_zero_density(self):
        raster = np.full((10, 10), 0, dtype=np.uint8)
        assert calculate_cau(raster) == 0.0

    def test_nodata_excluded(self):
        raster = np.full((10, 10), TCD_NODATA, dtype=np.uint8)
        assert calculate_cau(raster) == 0.0

    def test_partial_density(self):
        raster = np.array([[50, 50], [50, 50]], dtype=np.uint8)
        expected = 4 * 0.50 * PIXEL_AREA_M2  # 200 m²
        assert calculate_cau(raster) == pytest.approx(expected)

    def test_single_pixel_50_percent(self):
        raster = np.array([[50]], dtype=np.uint8)
        assert calculate_cau(raster) == pytest.approx(0.50 * PIXEL_AREA_M2)

    def test_result_is_float(self, synthetic_tcd_raster):
        assert isinstance(calculate_cau(synthetic_tcd_raster), float)

    def test_mixed_values_and_nodata(self):
        raster = np.array([[100, TCD_NODATA], [0, 60]], dtype=np.uint8)
        # 100→100m², skip nodata, 0→0m², 60→60m²  total = 160 m²
        expected = (1.0 + 0.0 + 0.6) * PIXEL_AREA_M2
        assert calculate_cau(raster) == pytest.approx(expected)
