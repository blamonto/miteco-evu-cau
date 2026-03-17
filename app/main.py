"""WildSquare MITECO Service — EVU & CAU indicators for MITECO compliance.

Endpoints:
  POST /calculate                → trigger EVU + CAU calculation (async)
  GET  /indicators/{request_id}  → retrieve results
  GET  /health                   → health check
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

from app.boundaries.lau import (
    BoundaryNotFoundError,
    DataDownloadError,
    compute_zeu_area,
    get_lau_boundary,
)
from app.config import settings
from app.indicators.baseline import build_baseline
from app.indicators.cau import calculate_cau
from app.indicators.evu import calculate_evu, create_evu_mask
from app.outputs.geotiff import write_geotiff
from app.outputs.shapefile import write_zeu_shapefile
from app.rasters.clc_backbone import (
    RasterAccessError as CLCAccessError,
    fetch_clc_backbone,
    get_copernicus_token,
)
from app.rasters.clip import clip_raster_to_polygon
from app.rasters.hrl_tcd import (
    RasterAccessError as TCDAccessError,
    fetch_hrl_tcd,
)

logger = logging.getLogger(__name__)

app = FastAPI(title="MITECO Indicators Service", version="1.0.0")


# ── Request / Response schemas ──────────────────────────────────────


class CalculateRequest(BaseModel):
    lau_code: str
    country_code: str = "ES"
    callback_url: str | None = None


class CalculateResponse(BaseModel):
    status: str = "accepted"
    request_id: str
    lau_code: str


class IndicatorResult(BaseModel):
    request_id: str
    lau_code: str
    municipality_name: str = ""
    status: str = "processing"
    zeu_area_m2: float | None = None
    evu_m2: float | None = None
    evu_ratio_pct: float | None = None
    cau_m2: float | None = None
    cau_ratio_pct: float | None = None
    outputs: dict | None = None
    error: str | None = None


# ── In-memory results store ─────────────────────────────────────────

_results: dict[str, IndicatorResult] = {}
_results_lock = asyncio.Lock()


# ── Endpoints ───────────────────────────────────────────────────────


@app.get("/health")
async def health():
    return {"status": "ok", "service": "miteco", "version": "1.0.0"}


@app.post("/calculate", status_code=202, response_model=CalculateResponse)
async def calculate(req: CalculateRequest, bg: BackgroundTasks):
    request_id = str(uuid.uuid4())
    result = IndicatorResult(request_id=request_id, lau_code=req.lau_code)
    async with _results_lock:
        _results[request_id] = result
    bg.add_task(_run_calculation_safe, req, request_id)
    return CalculateResponse(request_id=request_id, lau_code=req.lau_code)


@app.get("/indicators/{request_id}", response_model=IndicatorResult)
async def get_indicators(request_id: str):
    async with _results_lock:
        result = _results.get(request_id)
    if result is None:
        raise HTTPException(404, f"Request {request_id} not found")
    return result


# ── Background pipeline ────────────────────────────────────────────


async def _run_calculation_safe(req: CalculateRequest, request_id: str):
    """Full EVU + CAU pipeline with top-level error handling."""
    try:
        await asyncio.to_thread(_run_pipeline, req, request_id)
    except Exception as exc:
        logger.exception("Pipeline failed for %s", request_id)
        async with _results_lock:
            r = _results[request_id]
            r.status = "failed"
            r.error = str(exc)


def _run_pipeline(req: CalculateRequest, request_id: str):
    """Synchronous pipeline (runs in thread via asyncio.to_thread)."""
    from pyproj import Transformer
    from rasterio.crs import CRS
    from rasterio.transform import from_bounds

    # 1. Download LAU boundary
    logger.info("[%s] Step 1: Fetching LAU boundary for %s", request_id, req.lau_code)
    gdf, municipality_name = get_lau_boundary(req.lau_code, req.country_code)

    _update_result(request_id, municipality_name=municipality_name)

    # 2. Compute ZEU area
    zeu_area_m2 = compute_zeu_area(gdf)
    logger.info("[%s] ZEU area: %.0f m²", request_id, zeu_area_m2)

    # 3. Get bounding box in EPSG:3035
    crs_3035 = CRS.from_epsg(3035)
    gdf_3035 = gdf.to_crs(crs_3035)
    bbox_3035 = tuple(gdf_3035.total_bounds)  # (xmin, ymin, xmax, ymax)

    # 4. Authenticate with Copernicus
    logger.info("[%s] Step 2: Authenticating with Copernicus", request_id)
    try:
        token = get_copernicus_token()
    except CLCAccessError:
        token = None
        logger.warning("[%s] Copernicus auth failed, trying without token", request_id)

    # 5. Fetch CLC+ Backbone
    logger.info("[%s] Step 3: Fetching CLC+ Backbone raster", request_id)
    clc_data, clc_meta = fetch_clc_backbone(bbox_3035, token)

    # 6. Clip CLC+ to ZEU
    polygon_3035 = gdf_3035.geometry.iloc[0]
    clc_clipped, clc_transform = clip_raster_to_polygon(
        clc_data, clc_meta["transform"], clc_meta["crs"],
        polygon_3035, crs_3035, nodata=int(clc_meta["nodata"]),
    )

    # 7. Calculate EVU
    evu_m2 = calculate_evu(clc_clipped, nodata=int(clc_meta["nodata"]))
    logger.info("[%s] EVU: %.0f m²", request_id, evu_m2)

    # 8. Fetch HRL TCD
    logger.info("[%s] Step 4: Fetching HRL Tree Cover Density raster", request_id)
    tcd_data, tcd_meta = fetch_hrl_tcd(bbox_3035)

    # 9. Clip TCD to ZEU
    tcd_clipped, tcd_transform = clip_raster_to_polygon(
        tcd_data, tcd_meta["transform"], tcd_meta["crs"],
        polygon_3035, crs_3035, nodata=int(tcd_meta["nodata"]),
    )

    # 10. Calculate CAU
    cau_m2 = calculate_cau(tcd_clipped, nodata=int(tcd_meta["nodata"]))
    logger.info("[%s] CAU: %.0f m²", request_id, cau_m2)

    # 11. Generate outputs
    output_dir = os.path.join(settings.cache_dir, "outputs", req.lau_code)
    os.makedirs(output_dir, exist_ok=True)

    # ZEU shapefile
    zeu_shp = write_zeu_shapefile(gdf, output_dir, req.lau_code)

    # EVU GeoTIFF
    evu_mask = create_evu_mask(clc_clipped, nodata=int(clc_meta["nodata"]))
    evu_tif = write_geotiff(
        evu_mask, clc_transform, crs_3035,
        os.path.join(output_dir, f"EVU_{req.lau_code}.tif"),
        nodata=255,
    )

    # CAU GeoTIFF
    cau_tif = write_geotiff(
        tcd_clipped, tcd_transform, crs_3035,
        os.path.join(output_dir, f"CAU_{req.lau_code}.tif"),
        nodata=int(tcd_meta["nodata"]),
    )

    # 12. Build baseline
    baseline = build_baseline(
        req.lau_code, municipality_name, zeu_area_m2, evu_m2, cau_m2,
    )
    baseline_path = os.path.join(output_dir, f"baseline_2024_{req.lau_code}.json")
    with open(baseline_path, "w") as f:
        json.dump(baseline, f, indent=2, ensure_ascii=False)

    # 13. Store completed result
    _update_result(
        request_id,
        status="completed",
        zeu_area_m2=zeu_area_m2,
        evu_m2=evu_m2,
        evu_ratio_pct=baseline["evu_ratio_pct"],
        cau_m2=cau_m2,
        cau_ratio_pct=baseline["cau_ratio_pct"],
        outputs={
            "zeu_shapefile": zeu_shp,
            "evu_geotiff": evu_tif,
            "cau_geotiff": cau_tif,
            "baseline_json": baseline_path,
        },
    )
    logger.info("[%s] Pipeline completed successfully", request_id)


def _update_result(request_id: str, **kwargs):
    """Synchronous helper — safe because only the background thread writes."""
    r = _results.get(request_id)
    if r:
        for k, v in kwargs.items():
            setattr(r, k, v)
