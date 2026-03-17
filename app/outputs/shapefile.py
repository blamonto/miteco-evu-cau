"""ZEU shapefile exporter."""

from __future__ import annotations

import os

import geopandas as gpd


def write_zeu_shapefile(
    gdf: gpd.GeoDataFrame,
    output_dir: str,
    lau_code: str,
) -> str:
    """Write the ZEU boundary as a shapefile. Returns path to the .shp file."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"ZEU_{lau_code}.shp")
    gdf.to_file(filepath, driver="ESRI Shapefile")
    return filepath
