"""Tests for the CLI run.py pipeline."""

import numpy as np
import pytest
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from shapely.geometry import box

from run import (
    _generate_demo_clc,
    _generate_demo_tcd,
    extract_urban_nucleus,
    compute_urban_ratio,
)

pytestmark = pytest.mark.unit


class TestDemoDataGeneration:
    """Verify synthetic data generators produce valid rasters."""

    def test_demo_clc_shape(self):
        bbox = (3000000, 2000000, 3010000, 2010000)
        data, meta = _generate_demo_clc(100, 100, bbox)
        assert data.shape == (100, 100)
        assert data.dtype == np.uint8

    def test_demo_clc_has_green_classes(self):
        bbox = (3000000, 2000000, 3010000, 2010000)
        data, _ = _generate_demo_clc(200, 200, bbox)
        unique = set(np.unique(data))
        assert 141 in unique  # green urban
        assert 111 in unique  # urban fabric

    def test_demo_clc_metadata(self):
        bbox = (3000000, 2000000, 3010000, 2010000)
        _, meta = _generate_demo_clc(100, 100, bbox)
        assert meta["crs"] == CRS.from_epsg(3035)
        assert meta["nodata"] == 0

    def test_demo_tcd_shape(self):
        bbox = (3000000, 2000000, 3010000, 2010000)
        data, meta = _generate_demo_tcd(100, 100, bbox)
        assert data.shape == (100, 100)
        assert data.dtype == np.uint8

    def test_demo_tcd_values_in_range(self):
        bbox = (3000000, 2000000, 3010000, 2010000)
        data, _ = _generate_demo_tcd(200, 200, bbox)
        # All values should be 0-100 (density %)
        assert data.max() <= 100
        assert data.min() >= 0

    def test_demo_tcd_metadata(self):
        bbox = (3000000, 2000000, 3010000, 2010000)
        _, meta = _generate_demo_tcd(100, 100, bbox)
        assert meta["crs"] == CRS.from_epsg(3035)
        assert meta["nodata"] == 255


class TestUrbanNucleus:
    """Tests for urban nucleus extraction and rural auto-detection."""

    def _make_raster_with_urban_center(self, size=100):
        """Create a raster with urban classes in the center and forest around."""
        bbox = (3000000, 2000000, 3000000 + size * 10, 2000000 + size * 10)
        data = np.full((size, size), 23, dtype=np.uint8)  # 23 = broad-leaved forest
        # Put urban classes in center 20×20 block
        c = size // 2
        r = 10
        data[c - r:c + r, c - r:c + r] = 111  # continuous urban fabric
        # Add some green urban in the urban area
        data[c - r:c - r + 5, c - r:c - r + 5] = 141  # green urban
        transform = from_bounds(*bbox, size, size)
        meta = {"transform": transform, "crs": CRS.from_epsg(3035), "nodata": 0}
        lau_polygon = box(*bbox)
        return data, meta, lau_polygon

    def test_extract_urban_nucleus_finds_center(self):
        data, meta, lau_polygon = self._make_raster_with_urban_center()
        nucleus = extract_urban_nucleus(data, meta, lau_polygon, buffer_m=50)
        assert nucleus is not None
        assert not nucleus.is_empty
        # Nucleus should be much smaller than full LAU
        assert nucleus.area < lau_polygon.area

    def test_extract_urban_nucleus_clips_to_lau(self):
        data, meta, lau_polygon = self._make_raster_with_urban_center()
        nucleus = extract_urban_nucleus(data, meta, lau_polygon, buffer_m=500)
        assert nucleus is not None
        # Even with large buffer, should not exceed LAU boundary
        assert nucleus.within(lau_polygon.buffer(1))  # 1m tolerance

    def test_extract_urban_nucleus_no_urban_returns_none(self):
        """If no urban pixels exist, should return None."""
        bbox = (3000000, 2000000, 3001000, 2001000)
        data = np.full((100, 100), 23, dtype=np.uint8)  # all forest
        transform = from_bounds(*bbox, 100, 100)
        meta = {"transform": transform}
        lau_polygon = box(*bbox)
        result = extract_urban_nucleus(data, meta, lau_polygon)
        assert result is None

    def test_compute_urban_ratio_all_urban(self):
        data = np.full((50, 50), 111, dtype=np.uint8)
        assert compute_urban_ratio(data, nodata=0) == pytest.approx(1.0)

    def test_compute_urban_ratio_no_urban(self):
        data = np.full((50, 50), 23, dtype=np.uint8)  # forest
        assert compute_urban_ratio(data, nodata=0) == pytest.approx(0.0)

    def test_compute_urban_ratio_mixed(self):
        data = np.full((100, 100), 23, dtype=np.uint8)  # forest
        # Make 25% urban (top-left 50×50)
        data[:50, :50] = 111
        ratio = compute_urban_ratio(data, nodata=0)
        assert ratio == pytest.approx(0.25)

    def test_compute_urban_ratio_ignores_nodata(self):
        data = np.full((100, 100), 0, dtype=np.uint8)  # all nodata
        data[:10, :10] = 111  # 100 urban pixels
        data[10:20, :10] = 23  # 100 forest pixels
        ratio = compute_urban_ratio(data, nodata=0)
        # 100 urban out of 200 valid → 0.5
        assert ratio == pytest.approx(0.5)
