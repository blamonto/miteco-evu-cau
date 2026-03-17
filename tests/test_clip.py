"""Tests for raster clipping utilities."""

import numpy as np
import pytest
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from shapely.geometry import box

from app.rasters.clip import clip_raster_to_polygon, reproject_polygon

pytestmark = pytest.mark.unit

CRS_4326 = CRS.from_epsg(4326)


def _make_raster_and_transform(rows=10, cols=10, fill=111):
    """Create a simple raster covering a 1°×1° bbox at (0,0)."""
    raster = np.full((rows, cols), fill, dtype=np.uint8)
    transform = from_bounds(0, 0, 1, 1, cols, rows)
    return raster, transform


class TestClipRasterToPolygon:
    def test_preserves_values_inside(self):
        raster, transform = _make_raster_and_transform(fill=141)
        # polygon covers the full extent
        poly = box(0, 0, 1, 1)
        clipped, _ = clip_raster_to_polygon(
            raster, transform, CRS_4326, poly, CRS_4326,
        )
        assert np.all(clipped == 141)

    def test_nodata_outside_polygon(self):
        raster, transform = _make_raster_and_transform(fill=141)
        # polygon covers only the center quarter
        poly = box(0.25, 0.25, 0.75, 0.75)
        clipped, _ = clip_raster_to_polygon(
            raster, transform, CRS_4326, poly, CRS_4326, nodata=0,
        )
        # some pixels should be nodata (0)
        assert 0 in clipped
        # some pixels should remain 141
        assert 141 in clipped

    def test_returns_same_shape(self):
        raster, transform = _make_raster_and_transform(20, 30)
        poly = box(0.1, 0.1, 0.9, 0.9)
        clipped, _ = clip_raster_to_polygon(
            raster, transform, CRS_4326, poly, CRS_4326,
        )
        assert clipped.shape == (20, 30)

    def test_does_not_modify_original(self):
        raster, transform = _make_raster_and_transform(fill=141)
        original = raster.copy()
        poly = box(0.25, 0.25, 0.75, 0.75)
        clip_raster_to_polygon(
            raster, transform, CRS_4326, poly, CRS_4326, nodata=0,
        )
        np.testing.assert_array_equal(raster, original)


class TestReprojectPolygon:
    def test_same_crs_returns_identical(self):
        poly = box(0, 0, 1, 1)
        result = reproject_polygon(poly, CRS_4326, CRS_4326)
        assert poly.equals(result)

    def test_different_crs_changes_coordinates(self):
        poly = box(-3.7, 40.4, -3.6, 40.5)
        crs_3035 = CRS.from_epsg(3035)
        result = reproject_polygon(poly, CRS_4326, crs_3035)
        # Coordinates should be very different (meters vs degrees)
        assert not poly.equals(result)
        assert result.is_valid
