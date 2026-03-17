"""Tests for baseline table assembly."""

import pytest

from app.indicators.baseline import build_baseline

pytestmark = pytest.mark.unit


class TestBuildBaseline:
    def test_correct_ratios(self):
        result = build_baseline("28079", "Madrid", 1_000_000, 200_000, 130_000)
        assert result["evu_ratio_pct"] == 20.0
        assert result["cau_ratio_pct"] == 13.0

    def test_zero_zeu_area_no_division_error(self):
        result = build_baseline("00000", "Empty", 0, 0, 0)
        assert result["evu_ratio_pct"] == 0.0
        assert result["cau_ratio_pct"] == 0.0

    def test_all_fields_present(self):
        result = build_baseline("28079", "Madrid", 1_000_000, 200_000, 130_000)
        expected_keys = {
            "lau_code", "municipality_name", "year",
            "zeu_area_m2", "evu_m2", "evu_ratio_pct",
            "cau_m2", "cau_ratio_pct",
        }
        assert set(result.keys()) == expected_keys

    def test_rounding_precision(self):
        result = build_baseline("28079", "Madrid", 3, 1, 1)
        assert result["evu_ratio_pct"] == 33.33
        assert result["cau_ratio_pct"] == 33.33

    def test_default_year(self):
        result = build_baseline("28079", "Madrid", 100, 50, 30)
        assert result["year"] == 2024

    def test_custom_year(self):
        result = build_baseline("28079", "Madrid", 100, 50, 30, year=2025)
        assert result["year"] == 2025

    def test_values_are_rounded(self):
        result = build_baseline("28079", "Madrid", 1000.556, 200.334, 130.776)
        assert result["zeu_area_m2"] == 1000.56
        assert result["evu_m2"] == 200.33
        assert result["cau_m2"] == 130.78
