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

    def test_min_density_threshold(self):
        """With threshold=10, pixels below 10% are excluded."""
        raster = np.array([[5, 15], [30, 2]], dtype=np.uint8)
        # Without threshold: (5+15+30+2)/100 * 100 = 52.0
        assert calculate_cau(raster, min_density=0) == pytest.approx(52.0)
        # With threshold=10: only 15 and 30 count → (15+30)/100 * 100 = 45.0
        assert calculate_cau(raster, min_density=10) == pytest.approx(45.0)

    def test_min_density_threshold_100(self):
        """Threshold=100 only counts pixels with exactly 100% density."""
        raster = np.array([[99, 100], [50, 100]], dtype=np.uint8)
        result = calculate_cau(raster, min_density=100)
        assert result == pytest.approx(2 * PIXEL_AREA_M2)
