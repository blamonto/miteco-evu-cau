#!/usr/bin/env python3
"""
MITECO EVU & CAU Indicator Calculator
======================================

Uso:
    python run.py "Madrid"
    python run.py "Somiedo" --urban-nucleus
    python run.py "Barcelona" --output-dir ./resultados

Calcula automáticamente:
  - EVU (Espacio Verde Urbano) con CLC+ Backbone 2023
  - CAU (Cobertura Arbórea Urbana) con HRL Tree Cover Density
  - Genera: shapefile ZEU, GeoTIFFs, tabla baseline JSON

Para municipios rurales, usa --urban-nucleus para recortar la ZEU
al núcleo urbano (en vez de usar todo el término municipal).
Si no se indica, el script auto-detecta municipios rurales (>50 km²
y <10% suelo urbano) y aplica el recorte automáticamente.

No necesita credenciales — usa WMS abierto de Copernicus/EEA.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
import zipfile

import geopandas as gpd
import httpx
import numpy as np
from rasterio.crs import CRS
from rasterio.transform import from_bounds

from shapely.geometry import shape, MultiPolygon
from shapely.ops import unary_union

from app.indicators.evu import calculate_evu, create_evu_mask, detect_clc_product, get_green_classes
from app.indicators.cau import calculate_cau
from app.indicators.baseline import build_baseline
from app.rasters.clip import clip_raster_to_polygon
from app.outputs.geotiff import write_geotiff
from app.outputs.shapefile import write_zeu_shapefile

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("miteco")

# ── Constants ───────────────────────────────────────────────────────

LAU_URL = (
    "https://gisco-services.ec.europa.eu/distribution/v2/lau/download/"
    "ref-lau-2021-01m.geojson.zip"
)

CLC_WMS = (
    "https://copernicus.discomap.eea.europa.eu/arcgis/services"
    "/CLC/CLC2018_WM/MapServer/WMSServer"
)

TCD_REST = (
    "https://image.discomap.eea.europa.eu/arcgis/rest/services"
    "/GioLandPublic/HRL_TreeCoverDensity_2018/ImageServer/exportImage"
)

CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")
CRS_3035 = CRS.from_epsg(3035)
CRS_4326 = CRS.from_epsg(4326)

# Catastro INSPIRE Buildings WFS (open, no auth)
CATASTRO_BU_WFS = "http://ovc.catastro.meh.es/INSPIRE/wfsBU.aspx"

# CLC classes considered "urban/artificial" for nucleus extraction
# CORINE classic (44 classes):
CLC_URBAN_CLASSES_CORINE = frozenset({
    111,  # Continuous urban fabric
    112,  # Discontinuous urban fabric
    121,  # Industrial or commercial units
    122,  # Road and rail networks
    123,  # Port areas
    124,  # Airports
    131,  # Mineral extraction sites
    132,  # Dump sites
    133,  # Construction sites
    141,  # Green urban areas
    142,  # Sport and leisure facilities
})
# CLC+ Backbone (11 classes):
CLC_URBAN_CLASSES_PLUS = frozenset({
    1,   # Sealed artificial
    2,   # Non-sealed artificial
})
# Default (will be resolved at runtime)
CLC_URBAN_CLASSES = CLC_URBAN_CLASSES_CORINE

# Auto-detection thresholds for rural municipalities
RURAL_AREA_THRESHOLD_KM2 = 50  # >50 km² considered potentially rural
RURAL_URBAN_RATIO_THRESHOLD = 0.10  # <10% urban pixels → rural


# ── Urban nucleus extraction ──────────────────────────────────────


def extract_urban_nucleus(
    clc_data: np.ndarray,
    clc_meta: dict,
    lau_polygon,
    buffer_m: float = 200,
    nodata: int = 0,
) -> MultiPolygon | None:
    """Extract urban nucleus polygon from CLC raster within LAU boundary.

    1. Auto-detects CLC product (CORINE 44-class or CLC+ Backbone 11-class)
    2. Selects pixels with urban classes
    3. Buffers each polygon by *buffer_m* metres (to include adjacent green areas)
    4. Dissolves into a single MultiPolygon
    5. Clips to the LAU boundary

    Returns None if no urban pixels are found.
    """
    from rasterio.features import shapes
    from app.indicators.evu import detect_clc_product

    product = detect_clc_product(clc_data, nodata)
    urban_classes = CLC_URBAN_CLASSES_PLUS if product == "clcplus" else CLC_URBAN_CLASSES_CORINE
    urban_mask = np.isin(clc_data, list(urban_classes)).astype(np.uint8)

    # If no urban pixels, return None
    if urban_mask.sum() == 0:
        return None

    # Vectorise urban pixels into polygons
    urban_polys = []
    for geom, value in shapes(urban_mask, transform=clc_meta["transform"]):
        if value == 1:
            urban_polys.append(shape(geom))

    if not urban_polys:
        return None

    # Dissolve all urban polygons and buffer
    dissolved = unary_union(urban_polys).buffer(buffer_m)

    # Clip to LAU boundary so we don't extend beyond the municipality
    nucleus = dissolved.intersection(lau_polygon)

    if nucleus.is_empty:
        return None

    # Normalise to MultiPolygon
    if nucleus.geom_type == "Polygon":
        nucleus = MultiPolygon([nucleus])
    elif nucleus.geom_type == "GeometryCollection":
        polys = [g for g in nucleus.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
        if not polys:
            return None
        nucleus = unary_union(polys)
        if nucleus.geom_type == "Polygon":
            nucleus = MultiPolygon([nucleus])

    return nucleus


def compute_urban_ratio(clc_data: np.ndarray, nodata: int = 0) -> float:
    """Compute fraction of valid pixels that are urban classes."""
    from app.indicators.evu import detect_clc_product
    valid = clc_data != nodata
    total = valid.sum()
    if total == 0:
        return 0.0
    product = detect_clc_product(clc_data, nodata)
    urban_classes = CLC_URBAN_CLASSES_PLUS if product == "clcplus" else CLC_URBAN_CLASSES_CORINE
    urban = np.isin(clc_data, list(urban_classes)) & valid
    return float(urban.sum() / total)


def _normalise_to_multipolygon(geom):
    """Convert any polygon-ish geometry to a MultiPolygon."""
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "Polygon":
        return MultiPolygon([geom])
    if geom.geom_type == "MultiPolygon":
        return geom
    if geom.geom_type == "GeometryCollection":
        polys = [g for g in geom.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
        if not polys:
            return None
        result = unary_union(polys)
        if result.geom_type == "Polygon":
            return MultiPolygon([result])
        return result
    return None


# ── Catastro INSPIRE WFS — official building footprints ───────────


def fetch_catastro_buildings(
    bbox_4326: tuple[float, float, float, float],
    max_queries: int = 20,
) -> gpd.GeoDataFrame | None:
    """Fetch building footprints from Spanish Cadastre INSPIRE WFS.

    The WFS has a ~1 km² limit per query, so for large areas we tile
    the bbox into smaller chunks.

    Args:
        bbox_4326: (minlon, minlat, maxlon, maxlat) in EPSG:4326
        max_queries: Cap number of WFS requests to avoid overloading

    Returns:
        GeoDataFrame with building polygons, or None if no buildings found.
    """
    import xml.etree.ElementTree as ET

    minlon, minlat, maxlon, maxlat = bbox_4326

    # Tile into ~0.01° × 0.01° chunks (~1 km² each)
    step = 0.01
    tiles = []
    lat = minlat
    while lat < maxlat:
        lon = minlon
        while lon < maxlon:
            tiles.append((
                round(lat, 5),
                round(lon, 5),
                round(min(lat + step, maxlat), 5),
                round(min(lon + step, maxlon), 5),
            ))
            lon += step
        lat += step

    if len(tiles) > max_queries:
        # Too many tiles — subsample (center + random)
        log.info("  Municipio muy grande (%d tiles). Muestreando %d tiles...",
                 len(tiles), max_queries)
        import random
        random.seed(42)
        tiles = random.sample(tiles, max_queries)

    log.info("  Consultando Catastro WFS (%d tiles)...", len(tiles))

    all_polys = []
    gml_ns = "{http://www.opengis.net/gml/3.2}"

    for i, (lat0, lon0, lat1, lon1) in enumerate(tiles):
        params = {
            "service": "WFS",
            "version": "2.0.0",
            "request": "GetFeature",
            "TYPENAMES": "bu:Building",
            "count": "500",
            "BBOX": f"{lat0},{lon0},{lat1},{lon1}",
            "SRSNAME": "EPSG:4326",
        }
        try:
            resp = httpx.get(CATASTRO_BU_WFS, params=params, timeout=30)
            if resp.status_code != 200 or b"ExceptionReport" in resp.content[:500]:
                continue
        except Exception:
            continue

        # Parse GML to extract building polygons
        try:
            root = ET.fromstring(resp.content)
        except ET.ParseError:
            continue

        # Find all posList elements (building footprint coordinates)
        for pos_list_el in root.iter(f"{gml_ns}posList"):
            try:
                coords_text = pos_list_el.text.strip()
                values = [float(v) for v in coords_text.split()]
                # Coordinates are lat, lon pairs
                if len(values) >= 6:  # Need at least 3 points
                    coords = [(values[j + 1], values[j])
                              for j in range(0, len(values) - 1, 2)]
                    if len(coords) >= 3:
                        from shapely.geometry import Polygon as ShapelyPolygon
                        poly = ShapelyPolygon(coords)
                        if poly.is_valid and not poly.is_empty:
                            all_polys.append(poly)
            except (ValueError, IndexError):
                continue

    if not all_polys:
        return None

    log.info("  -> %d footprints de edificios obtenidos del Catastro", len(all_polys))
    return gpd.GeoDataFrame(geometry=all_polys, crs="EPSG:4326")


def extract_urban_nucleus_catastro(
    bbox_4326: tuple[float, float, float, float],
    lau_polygon_3035,
    buffer_m: float = 200,
    min_cluster_buildings: int = 5,
) -> MultiPolygon | None:
    """Extract urban nucleus using official Catastro building footprints.

    1. Downloads building footprints from the Catastro INSPIRE WFS
    2. Reprojects to EPSG:3035
    3. Buffers each building by *buffer_m* metres
    4. Dissolves into clusters
    5. Filters clusters with < *min_cluster_buildings* (removes farms/isolated)
    6. Clips to LAU boundary

    Returns None if no significant urban nucleus is found.
    """
    from app.rasters.clip import reproject_polygon

    buildings_gdf = fetch_catastro_buildings(bbox_4326)
    if buildings_gdf is None or buildings_gdf.empty:
        log.warning("  No se encontraron edificios en el Catastro. "
                    "Usando fallback CLC.")
        return None

    # Reproject to EPSG:3035 for metric buffer
    buildings_3035 = buildings_gdf.to_crs(epsg=3035)

    # Buffer each building and dissolve
    buffered = buildings_3035.geometry.buffer(buffer_m)
    dissolved = unary_union(buffered)

    # Split into individual clusters
    if dissolved.geom_type == "Polygon":
        clusters = [dissolved]
    elif dissolved.geom_type == "MultiPolygon":
        clusters = list(dissolved.geoms)
    else:
        return None

    # Filter small clusters (isolated farms/barns)
    significant_clusters = []
    for cluster in clusters:
        # Count buildings in this cluster
        n_buildings = buildings_3035.geometry.intersects(cluster).sum()
        if n_buildings >= min_cluster_buildings:
            significant_clusters.append(cluster)

    if not significant_clusters:
        log.warning("  No se encontraron núcleos urbanos significativos "
                    "(mín. %d edificios).", min_cluster_buildings)
        return None

    log.info("  -> %d núcleos urbanos encontrados (%d edificios total)",
             len(significant_clusters), len(buildings_3035))

    # Merge significant clusters
    nucleus = unary_union(significant_clusters)

    # Clip to LAU boundary
    nucleus = nucleus.intersection(lau_polygon_3035)

    return _normalise_to_multipolygon(nucleus)


# ── Step 1: Download LAU boundaries and find municipality ──────────


def find_municipality(name: str) -> tuple[gpd.GeoDataFrame, str, str]:
    """Search Spanish LAU boundaries by municipality name.

    Returns (gdf_row, official_name, lau_code).
    """
    cache_dir = os.path.join(CACHE_DIR, "lau")
    cache_file = os.path.join(cache_dir, "LAU_ES_2021.gpkg")
    os.makedirs(cache_dir, exist_ok=True)

    if os.path.isfile(cache_file):
        log.info("Cargando boundaries LAU desde caché...")
        gdf = gpd.read_file(cache_file)
    else:
        log.info("Descargando boundaries LAU de Eurostat (solo la primera vez, ~50MB)...")
        resp = httpx.get(LAU_URL, timeout=300, follow_redirects=True)
        resp.raise_for_status()

        # Save and extract zip
        zip_path = os.path.join(cache_dir, "lau.zip")
        with open(zip_path, "wb") as f:
            f.write(resp.content)

        extract_dir = os.path.join(cache_dir, "extracted")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)

        # Find the GeoJSON file inside the zip
        geojson_files = []
        for root, dirs, files in os.walk(extract_dir):
            for fn in files:
                if fn.endswith(".geojson") or fn.endswith(".json"):
                    geojson_files.append(os.path.join(root, fn))

        if not geojson_files:
            # Try shapefiles
            shp_files = []
            for root, dirs, files in os.walk(extract_dir):
                for fn in files:
                    if fn.endswith(".shp"):
                        shp_files.append(os.path.join(root, fn))
            if shp_files:
                geojson_files = shp_files

        if not geojson_files:
            raise RuntimeError(
                f"No geospatial files found in ZIP. Contents: "
                f"{os.listdir(extract_dir)}"
            )

        log.info("  Leyendo %s...", os.path.basename(geojson_files[0]))
        gdf_all = gpd.read_file(geojson_files[0])

        # Filter Spain only (CNTR_CODE = ES or LAU_ID starts with ES)
        cntr_col = None
        for c in ("CNTR_CODE", "cntr_code"):
            if c in gdf_all.columns:
                cntr_col = c
                break

        if cntr_col:
            gdf = gdf_all[gdf_all[cntr_col] == "ES"].copy()
        else:
            # Try filtering by LAU_ID prefix
            lau_col_tmp = None
            for c in ("LAU_ID", "LAU_CODE"):
                if c in gdf_all.columns:
                    lau_col_tmp = c
                    break
            if lau_col_tmp:
                gdf = gdf_all[gdf_all[lau_col_tmp].str.startswith("ES")].copy()
            else:
                gdf = gdf_all

        # Cache as GeoPackage (smaller, faster)
        gdf.to_file(cache_file, driver="GPKG")
        log.info("  -> %d municipios españoles cacheados", len(gdf))

        # Cleanup
        import shutil
        shutil.rmtree(extract_dir, ignore_errors=True)
        os.remove(zip_path)

    # Find name column
    name_col = None
    for c in ("LAU_NAME", "NAME_LATN", "name"):
        if c in gdf.columns:
            name_col = c
            break
    if name_col is None:
        raise RuntimeError(f"No name column found. Columns: {list(gdf.columns)}")

    # Find LAU code column
    lau_col = None
    for c in ("LAU_ID", "LAU_CODE", "lau_id"):
        if c in gdf.columns:
            lau_col = c
            break
    if lau_col is None:
        raise RuntimeError(f"No LAU code column found. Columns: {list(gdf.columns)}")

    # Search (case-insensitive)
    import unicodedata

    def strip_accents(s):
        return "".join(
            c for c in unicodedata.normalize("NFD", s)
            if unicodedata.category(c) != "Mn"
        )

    query = name.strip().lower()
    query_norm = strip_accents(query)
    names_lower = gdf[name_col].str.lower()
    names_norm = gdf[name_col].apply(lambda x: strip_accents(str(x).lower()))

    # 1. Try exact match first
    exact = gdf[names_lower == query]
    if exact.empty:
        exact = gdf[names_norm == query_norm]

    if len(exact) == 1:
        matches = exact
    elif len(exact) > 1:
        matches = exact
    else:
        # 2. Partial match
        matches = gdf[names_lower.str.contains(query, na=False)]
        if matches.empty:
            matches = gdf[names_norm.str.contains(query_norm, na=False)]

    if matches.empty:
        print(f"\n  No se encontró '{name}'. Municipios similares:")
        all_names = gdf[name_col].dropna().tolist()
        similar = [n for n in all_names if query[:4] in n.lower()][:10]
        for n in similar:
            print(f"    - {n}")
        sys.exit(1)

    if len(matches) > 1:
        # Check if one is an exact match
        exact_in_matches = matches[names_lower[matches.index] == query]
        if len(exact_in_matches) == 1:
            selected = exact_in_matches.iloc[[0]]
        else:
            print(f"\n  Se encontraron {len(matches)} coincidencias para '{name}':")
            for i, (_, row) in enumerate(matches.head(15).iterrows()):
                marker = " <--" if row[name_col].lower() == query else ""
                print(f"    {i+1}. {row[name_col]} (LAU: {row[lau_col]}){marker}")
            choice = input("\n  Elige número [1]: ").strip() or "1"
            idx = int(choice) - 1
            selected = matches.iloc[[idx]]
    else:
        selected = matches.iloc[[0]]

    official_name = str(selected.iloc[0][name_col])
    lau_code = str(selected.iloc[0][lau_col])
    return selected.reset_index(drop=True), official_name, lau_code


# ── Step 2: Fetch CLC raster via WMS ───────────────────────────────


def fetch_clc_wms(
    bbox: tuple[float, float, float, float],
    pixel_size: int = 10,
) -> tuple[np.ndarray, dict]:
    """Fetch CLC 2018 raster via EEA WMS (open, no auth)."""
    cache_dir = os.path.join(CACHE_DIR, "clc")
    os.makedirs(cache_dir, exist_ok=True)

    xmin, ymin, xmax, ymax = bbox
    width = max(1, min(int((xmax - xmin) / pixel_size), 4096))
    height = max(1, min(int((ymax - ymin) / pixel_size), 4096))

    log.info("  Solicitando CLC WMS: %d×%d pixels...", width, height)

    # Try multiple WMS endpoints (CLC+ Backbone 2023 is not yet on WMS,
    # so we try CLC 2018 as fallback, and synthetic data as last resort)
    wms_urls = [
        # CLC+ Backbone (may not be available yet)
        (
            "https://image.discomap.eea.europa.eu/arcgis/services"
            "/Corine/CLC2018_WM/MapServer/WMSServer"
        ),
    ]

    for wms_url in wms_urls:
        params = {
            "SERVICE": "WMS",
            "VERSION": "1.3.0",
            "REQUEST": "GetMap",
            "LAYERS": "12",  # CLC 2018 Europe layer
            "STYLES": "",
            "CRS": "EPSG:3035",
            "BBOX": f"{ymin},{xmin},{ymax},{xmax}",
            "WIDTH": str(width),
            "HEIGHT": str(height),
            "FORMAT": "image/tiff",
        }

        try:
            resp = httpx.get(wms_url, params=params, timeout=120, follow_redirects=True)
            resp.raise_for_status()
        except httpx.HTTPStatusError:
            continue

        content_type = resp.headers.get("content-type", "")
        if "xml" in content_type or b"<ServiceException" in resp.content[:500]:
            continue

        import rasterio

        cache_file = os.path.join(cache_dir, "clc_latest.tif")
        with open(cache_file, "wb") as f:
            f.write(resp.content)

        try:
            with rasterio.open(cache_file) as src:
                data = src.read(1)
                if src.crs is None or src.transform.is_identity:
                    # WMS returned ungeoreferenced image
                    continue
                meta = {
                    "transform": src.transform,
                    "crs": src.crs,
                    "nodata": src.nodata or 0,
                }
            return data, meta
        except Exception:
            continue

    log.warning(
        "  CLC+ Backbone 2023 (10m) aún no disponible via WMS abierto. "
        "Usando datos simulados. Para datos reales, descarga el raster "
        "de https://land.copernicus.eu y usa --clc-file."
    )
    return _generate_demo_clc(width, height, bbox)


def _generate_demo_clc(width, height, bbox):
    """Generate synthetic CLC data for demo/testing when WMS is unavailable."""
    rng = np.random.default_rng(42)
    data = np.full((height, width), 111, dtype=np.uint8)  # Urban fabric
    # ~25% green urban areas
    green_mask = rng.random((height, width)) < 0.25
    data[green_mask] = 141
    # ~8% sport/leisure
    sport_mask = rng.random((height, width)) < 0.08
    data[sport_mask & ~green_mask] = 142
    xmin, ymin, xmax, ymax = bbox
    transform = from_bounds(xmin, ymin, xmax, ymax, width, height)
    return data, {"transform": transform, "crs": CRS_3035, "nodata": 0}


# ── Step 3: Fetch HRL TCD raster via REST ──────────────────────────


def fetch_tcd_rest(
    bbox: tuple[float, float, float, float],
    pixel_size: int = 10,
) -> tuple[np.ndarray, dict]:
    """Fetch HRL Tree Cover Density via EEA REST (open, no auth)."""
    cache_dir = os.path.join(CACHE_DIR, "tcd")
    os.makedirs(cache_dir, exist_ok=True)

    xmin, ymin, xmax, ymax = bbox
    width = max(1, min(int((xmax - xmin) / pixel_size), 4096))
    height = max(1, min(int((ymax - ymin) / pixel_size), 4096))

    log.info("  Solicitando HRL TCD REST: %d×%d pixels...", width, height)

    params = {
        "bbox": f"{xmin},{ymin},{xmax},{ymax}",
        "bboxSR": "3035",
        "imageSR": "3035",
        "size": f"{width},{height}",
        "format": "tiff",
        "f": "image",
    }

    try:
        resp = httpx.get(TCD_REST, params=params, timeout=120, follow_redirects=True)
        resp.raise_for_status()
    except httpx.HTTPStatusError:
        log.warning(
            "  HRL TCD 2024 aún no disponible via REST. "
            "Usando datos simulados."
        )
        return _generate_demo_tcd(width, height, bbox)

    content_type = resp.headers.get("content-type", "")
    if "json" in content_type or "html" in content_type or len(resp.content) < 100:
        log.warning("  REST devolvió error. Usando datos simulados para demo.")
        return _generate_demo_tcd(width, height, bbox)

    import rasterio

    cache_file = os.path.join(cache_dir, "tcd_latest.tif")
    with open(cache_file, "wb") as f:
        f.write(resp.content)

    try:
        with rasterio.open(cache_file) as src:
            data = src.read(1)
            meta = {
                "transform": src.transform,
                "crs": src.crs,
                "nodata": src.nodata if src.nodata is not None else 255,
            }
        return data, meta
    except Exception:
        log.warning("  No se pudo leer raster REST. Usando datos simulados para demo.")
        return _generate_demo_tcd(width, height, bbox)


def _generate_demo_tcd(width, height, bbox):
    """Generate synthetic TCD data for demo/testing when REST is unavailable."""
    rng = np.random.default_rng(123)
    # Mix of tree densities typical of a Spanish city
    data = np.zeros((height, width), dtype=np.uint8)
    # ~15% high density (parks)
    data[rng.random((height, width)) < 0.15] = rng.integers(60, 95, size=1)[0]
    # ~20% medium density
    medium = rng.random((height, width)) < 0.20
    data[medium & (data == 0)] = rng.integers(20, 50, size=1)[0]
    xmin, ymin, xmax, ymax = bbox
    transform = from_bounds(xmin, ymin, xmax, ymax, width, height)
    return data, {"transform": transform, "crs": CRS_3035, "nodata": 255}


# ── Main pipeline ──────────────────────────────────────────────────


def load_local_raster(filepath: str) -> tuple[np.ndarray, dict]:
    """Load a local GeoTIFF raster file."""
    import rasterio

    with rasterio.open(filepath) as src:
        data = src.read(1)
        meta = {
            "transform": src.transform,
            "crs": src.crs,
            "nodata": src.nodata or 0,
        }
    return data, meta


def run(
    municipality_name: str,
    output_dir: str | None = None,
    clc_file: str | None = None,
    tcd_file: str | None = None,
    urban_nucleus: bool | None = None,
    buffer_m: float = 200,
    tcd_threshold: int = 0,
):
    """Run the full EVU + CAU pipeline for a municipality.

    Args:
        urban_nucleus: If True, restrict ZEU to urban nucleus only.
            If None (default), auto-detect: use nucleus for rural municipalities.
        buffer_m: Buffer in metres around urban pixels (default 200m).
        tcd_threshold: Min tree cover density % to count (0-100, default 0).
    """
    start = time.time()
    print()
    print("=" * 60)
    print("  MITECO — Indicadores EVU y CAU")
    print("=" * 60)

    # 1. Find municipality
    print(f"\n[1/7] Buscando municipio: {municipality_name}")
    gdf, name, lau_code = find_municipality(municipality_name)
    print(f"  -> Encontrado: {name} (LAU: {lau_code})")

    # 2. Compute full LAU area
    print(f"\n[2/7] Calculando área término municipal...")
    utm = gdf.estimate_utm_crs()
    gdf_utm = gdf.to_crs(utm)
    lau_area_m2 = float(gdf_utm.geometry.area.sum())
    lau_area_km2 = lau_area_m2 / 1e6
    print(f"  -> Área total municipio: {lau_area_m2:,.0f} m² ({lau_area_km2:,.2f} km²)")

    # Get bbox in EPSG:3035
    gdf_3035 = gdf.to_crs(CRS_3035)
    bbox = tuple(gdf_3035.total_bounds)
    lau_polygon = gdf_3035.geometry.iloc[0]

    # 3. Fetch CLC raster (needed for both EVU and urban nucleus detection)
    if clc_file:
        print(f"\n[3/7] Cargando CLC desde fichero local: {clc_file}")
        clc_data, clc_meta = load_local_raster(clc_file)
    else:
        print(f"\n[3/7] Descargando CLC (espacio verde urbano)...")
        clc_data, clc_meta = fetch_clc_wms(bbox)

    # Clip CLC to full LAU first (for urban ratio analysis)
    clc_clipped_full, clc_transform = clip_raster_to_polygon(
        clc_data, clc_meta["transform"], clc_meta["crs"],
        lau_polygon, CRS_3035, nodata=int(clc_meta["nodata"]),
    )

    # 4. Determine ZEU: full LAU or urban nucleus
    use_nucleus = urban_nucleus
    urban_ratio = compute_urban_ratio(clc_clipped_full, nodata=int(clc_meta["nodata"]))

    if use_nucleus is None:
        # Auto-detect: rural if large area AND low urban ratio
        is_rural = (lau_area_km2 > RURAL_AREA_THRESHOLD_KM2
                     and urban_ratio < RURAL_URBAN_RATIO_THRESHOLD)
        if is_rural:
            use_nucleus = True
            print(f"\n  ⚠ Municipio rural detectado ({lau_area_km2:.1f} km², "
                  f"{urban_ratio*100:.1f}% urbano)")
            print(f"    → Recortando ZEU al núcleo urbano (buffer {buffer_m}m)")
        else:
            use_nucleus = False

    nucleus_source = None  # Track which source defined the nucleus

    if use_nucleus:
        print(f"\n[4/7] Extrayendo núcleo urbano de la ZEU...")
        nucleus = None

        # Strategy 1: Catastro INSPIRE WFS (official building footprints)
        print("  [a] Consultando Catastro INSPIRE (edificios oficiales)...")
        try:
            gdf_4326 = gdf.to_crs(CRS_4326)
            bbox_4326 = tuple(gdf_4326.total_bounds)  # minlon,minlat,maxlon,maxlat
            nucleus = extract_urban_nucleus_catastro(
                bbox_4326, lau_polygon, buffer_m=buffer_m,
            )
            if nucleus is not None:
                nucleus_source = "Catastro INSPIRE (edificios oficiales)"
                print(f"  ✓ Núcleo urbano del Catastro obtenido")
        except Exception as e:
            log.warning("  Catastro WFS no disponible: %s", e)

        # Strategy 2: CLC pixel classification (fallback)
        if nucleus is None:
            print("  [b] Fallback: extrayendo núcleo de CLC...")
            nucleus = extract_urban_nucleus(
                clc_clipped_full, {"transform": clc_transform}, lau_polygon,
                buffer_m=buffer_m,
            )
            if nucleus is not None:
                nucleus_source = "CLC pixel classification (aproximación)"

        if nucleus is None or nucleus.is_empty:
            print("  ⚠ No se encontraron núcleos urbanos. Usando LAU completo.")
            polygon = lau_polygon
            zeu_area_m2 = lau_area_m2
            zeu_type = "LAU completo (sin núcleo urbano detectado)"
            nucleus_source = "LAU (Eurostat)"
        else:
            polygon = nucleus
            # Compute nucleus area in UTM
            from shapely.ops import transform as shapely_transform
            from pyproj import Transformer
            transformer = Transformer.from_crs(CRS_3035, utm, always_xy=True)
            nucleus_utm = shapely_transform(transformer.transform, nucleus)
            zeu_area_m2 = float(nucleus_utm.area)
            zeu_type = f"Núcleo urbano (buffer {buffer_m}m)"
            print(f"  -> Núcleo urbano extraído: {zeu_area_m2:,.0f} m² "
                  f"({zeu_area_m2/1e6:,.2f} km²)")
            print(f"     ({zeu_area_m2/lau_area_m2*100:.1f}% del término municipal)")
            print(f"     Fuente: {nucleus_source}")

            # Build a GeoDataFrame for the nucleus ZEU (for shapefile export)
            gdf_3035 = gpd.GeoDataFrame(
                {"LAU_CODE": [lau_code], "name": [name], "zeu_type": [zeu_type]},
                geometry=[polygon],
                crs=CRS_3035,
            )
            gdf = gdf_3035.to_crs(gdf.crs)
    else:
        print(f"\n[4/7] ZEU = término municipal completo")
        polygon = lau_polygon
        zeu_area_m2 = lau_area_m2
        zeu_type = "LAU completo"
        nucleus_source = "LAU (Eurostat)"

    zeu_area_km2 = zeu_area_m2 / 1e6
    print(f"  -> Área ZEU: {zeu_area_m2:,.0f} m² ({zeu_area_km2:,.2f} km²)")
    print(f"     Tipo: {zeu_type}")

    # 5. Clip CLC to ZEU polygon and calculate EVU
    clc_clipped, clc_transform = clip_raster_to_polygon(
        clc_data, clc_meta["transform"], clc_meta["crs"],
        polygon, CRS_3035, nodata=int(clc_meta["nodata"]),
    )
    # Auto-detect CLC product (CORINE 44-class vs CLC+ Backbone 11-class)
    clc_product = detect_clc_product(clc_clipped, nodata=int(clc_meta["nodata"]))
    green_classes = get_green_classes(clc_clipped, nodata=int(clc_meta["nodata"]))
    if clc_product == "clcplus":
        print(f"  Detectado: CLC+ Backbone (11 clases) → verde = clases {sorted(green_classes)}")
    else:
        print(f"  Detectado: CORINE Land Cover (44 clases) → verde = clases {sorted(green_classes)}")
    evu_m2 = calculate_evu(clc_clipped, nodata=int(clc_meta["nodata"]),
                           green_classes=green_classes)
    evu_pct = (evu_m2 / zeu_area_m2 * 100) if zeu_area_m2 > 0 else 0
    print(f"  -> EVU: {evu_m2:,.0f} m² ({evu_pct:.2f}% de la ZEU)")

    # 5. Fetch and calculate CAU
    if tcd_file:
        print(f"\n[5/7] Cargando TCD desde fichero local: {tcd_file}")
        tcd_data, tcd_meta = load_local_raster(tcd_file)
    else:
        print(f"\n[5/7] Descargando HRL Tree Cover Density (cobertura arbórea)...")
        tcd_data, tcd_meta = fetch_tcd_rest(bbox)
    tcd_clipped, tcd_transform = clip_raster_to_polygon(
        tcd_data, tcd_meta["transform"], tcd_meta["crs"],
        polygon, CRS_3035, nodata=int(tcd_meta["nodata"]),
    )
    cau_m2 = calculate_cau(tcd_clipped, nodata=int(tcd_meta["nodata"]),
                           min_density=tcd_threshold)
    cau_pct = (cau_m2 / zeu_area_m2 * 100) if zeu_area_m2 > 0 else 0
    tcd_note = f" (umbral ≥{tcd_threshold}%)" if tcd_threshold > 0 else ""
    print(f"  -> CAU: {cau_m2:,.0f} m² ({cau_pct:.2f}% de la ZEU){tcd_note}")

    # 6. Generate outputs
    out = output_dir or os.path.join("output", lau_code)
    os.makedirs(out, exist_ok=True)
    print(f"\n[6/7] Generando ficheros de salida en: {out}/")

    # ZEU shapefile
    shp = write_zeu_shapefile(gdf, out, lau_code)
    print(f"  -> ZEU shapefile: {os.path.basename(shp)}")

    # EVU GeoTIFF
    evu_mask = create_evu_mask(clc_clipped, nodata=int(clc_meta["nodata"]))
    evu_tif = write_geotiff(
        evu_mask, clc_transform, CRS_3035,
        os.path.join(out, f"EVU_{lau_code}.tif"), nodata=255,
    )
    print(f"  -> EVU cartografía: {os.path.basename(evu_tif)}")

    # CAU GeoTIFF
    cau_tif = write_geotiff(
        tcd_clipped, tcd_transform, CRS_3035,
        os.path.join(out, f"CAU_{lau_code}.tif"),
        nodata=int(tcd_meta["nodata"]),
    )
    print(f"  -> CAU cartografía: {os.path.basename(cau_tif)}")

    # 7. Baseline JSON with data source metadata
    baseline = build_baseline(lau_code, name, zeu_area_m2, evu_m2, cau_m2)
    baseline["zeu_type"] = zeu_type
    baseline["lau_area_m2"] = round(lau_area_m2, 2)
    if tcd_threshold > 0:
        baseline["tcd_threshold_pct"] = tcd_threshold

    # Data sources & confidence report
    clc_source = "local file" if clc_file else "WMS (CLC 2018)"
    if not clc_file and "simulados" in str(clc_meta.get("_source", "")):
        clc_source = "SIMULATED (demo data)"
    tcd_source = "local file" if tcd_file else "EEA REST (HRL TCD 2018)"

    baseline["data_sources"] = {
        "zeu_boundary": nucleus_source or "LAU (Eurostat)",
        "evu_raster": clc_source,
        "cau_raster": tcd_source,
    }
    baseline["confidence"] = {
        "zeu": "alta" if nucleus_source and "Catastro" in nucleus_source else
               "media" if nucleus_source and "CLC" in (nucleus_source or "") else "alta",
        "evu": "alta" if clc_file else "baja (datos simulados)",
        "cau": "alta" if tcd_file else "alta (datos reales EEA)",
    }

    baseline_path = os.path.join(out, f"baseline_2024_{lau_code}.json")
    with open(baseline_path, "w") as f:
        json.dump(baseline, f, indent=2, ensure_ascii=False)
    print(f"  -> Baseline 2024: {os.path.basename(baseline_path)}")

    # Summary
    elapsed = time.time() - start
    print(f"\n[7/7] RESUMEN — Línea Base 2024")
    print("─" * 55)
    print(f"  Municipio:     {name}")
    print(f"  Código LAU:    {lau_code}")
    print(f"  Tipo ZEU:      {zeu_type}")
    if use_nucleus:
        print(f"  Área municipio:{lau_area_m2:>14,.0f} m²")
    print(f"  Área ZEU:      {zeu_area_m2:>14,.0f} m²")
    print(f"  EVU:           {evu_m2:>14,.0f} m²  ({evu_pct:.2f}%)")
    print(f"  CAU:           {cau_m2:>14,.0f} m²  ({cau_pct:.2f}%)")
    if tcd_threshold > 0:
        print(f"  TCD umbral:    ≥{tcd_threshold}%")
    print("─" * 55)
    print("  FIABILIDAD:")
    for k, v in baseline["confidence"].items():
        print(f"    {k.upper():12s} {v}")
    print("─" * 55)
    print(f"  Tiempo: {elapsed:.1f}s")
    print(f"  Outputs: {os.path.abspath(out)}/")
    print()

    return baseline


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calcula indicadores MITECO (EVU y CAU) para un municipio español",
    )
    parser.add_argument(
        "municipio",
        help='Nombre del municipio (ej: "Madrid", "Barcelona", "Sevilla")',
    )
    parser.add_argument(
        "--output-dir", "-o",
        help="Directorio de salida (default: output/<lau_code>/)",
    )
    parser.add_argument(
        "--clc-file",
        help="Raster GeoTIFF local de CLC+ Backbone (en vez de descarga WMS)",
    )
    parser.add_argument(
        "--tcd-file",
        help="Raster GeoTIFF local de HRL Tree Cover Density (en vez de descarga REST)",
    )
    parser.add_argument(
        "--urban-nucleus", "-u",
        action="store_true",
        default=None,
        help="Restringir ZEU al núcleo urbano (para municipios rurales)",
    )
    parser.add_argument(
        "--full-lau",
        action="store_true",
        help="Usar LAU completo como ZEU (desactivar auto-detección rural)",
    )
    parser.add_argument(
        "--buffer",
        type=float,
        default=200,
        help="Buffer en metros alrededor del núcleo urbano (default: 200)",
    )
    parser.add_argument(
        "--tcd-threshold",
        type=int,
        default=0,
        help="Densidad mínima de cobertura arbórea %% para contar (0-100, default: 0)",
    )
    args = parser.parse_args()

    # Resolve urban nucleus flag
    nucleus = args.urban_nucleus
    if args.full_lau:
        nucleus = False

    run(
        args.municipio,
        args.output_dir,
        args.clc_file,
        args.tcd_file,
        urban_nucleus=nucleus,
        buffer_m=args.buffer,
        tcd_threshold=args.tcd_threshold,
    )
