"""Tests for LAU boundary download and filtering."""

import json
import os
from unittest.mock import patch, MagicMock

import geopandas as gpd
import pytest
from shapely.geometry import box, mapping

from app.boundaries.lau import (
    BoundaryNotFoundError,
    compute_zeu_area,
    download_lau_boundaries,
    get_lau_boundary,
)

pytestmark = pytest.mark.integration


def _make_lau_geojson(codes_and_names: list[tuple[str, str]]):
    """Build a minimal GeoJSON FeatureCollection for LAU boundaries."""
    features = []
    for i, (code, name) in enumerate(codes_and_names):
        geom = box(i, 40, i + 0.1, 40.1)
        features.append({
            "type": "Feature",
            "properties": {"LAU_ID": code, "LAU_NAME": name},
            "geometry": mapping(geom),
        })
    return {
        "type": "FeatureCollection",
        "features": features,
    }


@pytest.fixture
def mock_lau_response():
    """Mock HTTP response with LAU GeoJSON data."""
    data = _make_lau_geojson([
        ("28079", "Madrid"),
        ("08019", "Barcelona"),
        ("41091", "Sevilla"),
    ])
    resp = MagicMock()
    resp.content = json.dumps(data).encode()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    return resp


class TestDownloadLAUBoundaries:
    def test_downloads_and_parses(self, tmp_path, mock_lau_response):
        with patch("app.boundaries.lau.settings") as mock_settings, \
             patch("app.boundaries.lau.httpx.get", return_value=mock_lau_response):
            mock_settings.cache_dir = str(tmp_path)
            mock_settings.lau_base_url = "https://example.com/"
            mock_settings.lau_year = 2021
            gdf = download_lau_boundaries("ES", 2021)
            assert len(gdf) == 3
            assert "LAU_ID" in gdf.columns

    def test_caches_downloaded_file(self, tmp_path, mock_lau_response):
        with patch("app.boundaries.lau.settings") as mock_settings, \
             patch("app.boundaries.lau.httpx.get", return_value=mock_lau_response):
            mock_settings.cache_dir = str(tmp_path)
            mock_settings.lau_base_url = "https://example.com/"
            mock_settings.lau_year = 2021
            download_lau_boundaries("ES", 2021)
            cache_file = os.path.join(tmp_path, "lau", "LAU_ES_2021.geojson")
            assert os.path.isfile(cache_file)


class TestGetLAUBoundary:
    def test_filters_by_lau_code(self, tmp_path, mock_lau_response):
        with patch("app.boundaries.lau.settings") as mock_settings, \
             patch("app.boundaries.lau.httpx.get", return_value=mock_lau_response):
            mock_settings.cache_dir = str(tmp_path)
            mock_settings.lau_base_url = "https://example.com/"
            mock_settings.lau_year = 2021
            gdf, name = get_lau_boundary("28079", "ES", 2021)
            assert len(gdf) == 1
            assert name == "Madrid"

    def test_raises_for_unknown_code(self, tmp_path, mock_lau_response):
        with patch("app.boundaries.lau.settings") as mock_settings, \
             patch("app.boundaries.lau.httpx.get", return_value=mock_lau_response):
            mock_settings.cache_dir = str(tmp_path)
            mock_settings.lau_base_url = "https://example.com/"
            mock_settings.lau_year = 2021
            with pytest.raises(BoundaryNotFoundError):
                get_lau_boundary("99999", "ES", 2021)

    def test_strips_country_prefix(self, tmp_path, mock_lau_response):
        with patch("app.boundaries.lau.settings") as mock_settings, \
             patch("app.boundaries.lau.httpx.get", return_value=mock_lau_response):
            mock_settings.cache_dir = str(tmp_path)
            mock_settings.lau_base_url = "https://example.com/"
            mock_settings.lau_year = 2021
            gdf, name = get_lau_boundary("ES_28079", "ES", 2021)
            assert name == "Madrid"


class TestComputeZEUArea:
    def test_known_square_polygon_area(self):
        # ~1 km × 1 km square near Madrid
        poly = box(-3.71, 40.41, -3.70, 40.42)
        gdf = gpd.GeoDataFrame(geometry=[poly], crs="EPSG:4326")
        area = compute_zeu_area(gdf)
        # Should be roughly 1 km² = 1e6 m² (±20% due to projection)
        assert 0.5e6 < area < 2e6

    def test_area_is_positive(self):
        poly = box(0, 0, 0.001, 0.001)
        gdf = gpd.GeoDataFrame(geometry=[poly], crs="EPSG:4326")
        assert compute_zeu_area(gdf) > 0
