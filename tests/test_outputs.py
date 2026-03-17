"""Tests for GeoTIFF and shapefile output generation."""

import os

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from shapely.geometry import box

from app.outputs.geotiff import write_geotiff
from app.outputs.shapefile import write_zeu_shapefile

pytestmark = pytest.mark.integration

CRS_4326 = CRS.from_epsg(4326)


class TestWriteGeotiff:
    def test_creates_valid_file(self, tmp_path):
        data = np.full((10, 10), 141, dtype=np.uint8)
        transform = from_bounds(0, 0, 1, 1, 10, 10)
        filepath = str(tmp_path / "test.tif")
        result = write_geotiff(data, transform, CRS_4326, filepath)
        assert os.path.isfile(result)

    def test_correct_crs(self, tmp_path):
        data = np.full((10, 10), 1, dtype=np.uint8)
        transform = from_bounds(0, 0, 1, 1, 10, 10)
        filepath = str(tmp_path / "test.tif")
        write_geotiff(data, transform, CRS_4326, filepath)
        with rasterio.open(filepath) as src:
            assert src.crs == CRS_4326

    def test_values_match_input(self, tmp_path):
        data = np.arange(25, dtype=np.uint8).reshape(5, 5)
        transform = from_bounds(0, 0, 1, 1, 5, 5)
        filepath = str(tmp_path / "test.tif")
        write_geotiff(data, transform, CRS_4326, filepath)
        with rasterio.open(filepath) as src:
            read_data = src.read(1)
            np.testing.assert_array_equal(read_data, data)


class TestWriteZEUShapefile:
    def test_creates_shp_file(self, tmp_path):
        gdf = gpd.GeoDataFrame(
            {"LAU_CODE": ["28079"]},
            geometry=[box(0, 0, 1, 1)],
            crs="EPSG:4326",
        )
        result = write_zeu_shapefile(gdf, str(tmp_path), "28079")
        assert os.path.isfile(result)
        assert result.endswith(".shp")

    def test_shapefile_has_correct_geometry(self, tmp_path):
        poly = box(-3.7, 40.4, -3.6, 40.5)
        gdf = gpd.GeoDataFrame(
            {"LAU_CODE": ["28079"]},
            geometry=[poly],
            crs="EPSG:4326",
        )
        filepath = write_zeu_shapefile(gdf, str(tmp_path), "28079")
        read_back = gpd.read_file(filepath)
        assert len(read_back) == 1
        assert read_back.geometry.iloc[0].is_valid
