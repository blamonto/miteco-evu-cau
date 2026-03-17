"""Tests for the CLI run.py pipeline."""

import numpy as np
import pytest
from rasterio.crs import CRS
from rasterio.transform import from_bounds

from run import _generate_demo_clc, _generate_demo_tcd

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
