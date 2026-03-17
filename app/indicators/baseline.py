"""Baseline 2024 numeric table assembly."""

from __future__ import annotations


def build_baseline(
    lau_code: str,
    municipality_name: str,
    zeu_area_m2: float,
    evu_m2: float,
    cau_m2: float,
    year: int = 2024,
) -> dict:
    """Build the baseline table as a JSON-serializable dict."""
    return {
        "lau_code": lau_code,
        "municipality_name": municipality_name,
        "year": year,
        "zeu_area_m2": round(zeu_area_m2, 2),
        "evu_m2": round(evu_m2, 2),
        "evu_ratio_pct": round((evu_m2 / zeu_area_m2) * 100, 2) if zeu_area_m2 > 0 else 0.0,
        "cau_m2": round(cau_m2, 2),
        "cau_ratio_pct": round((cau_m2 / zeu_area_m2) * 100, 2) if zeu_area_m2 > 0 else 0.0,
    }
