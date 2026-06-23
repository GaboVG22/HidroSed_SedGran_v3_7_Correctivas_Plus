
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import io
import math
import tempfile
from typing import Dict, List

import numpy as np

from .opentopo_engine import download_dem, bbox_area_km2


@dataclass
class TileDownloadResult:
    dem_bytes: bytes
    metadata: dict


def split_bbox(bbox: Dict[str, float], rows: int, cols: int) -> List[Dict[str, float]]:
    rows = max(1, int(rows))
    cols = max(1, int(cols))
    south, north = float(bbox["south"]), float(bbox["north"])
    west, east = float(bbox["west"]), float(bbox["east"])
    lat_edges = np.linspace(south, north, rows + 1)
    lon_edges = np.linspace(west, east, cols + 1)
    tiles = []
    for i in range(rows):
        for j in range(cols):
            tiles.append({
                "south": round(float(lat_edges[i]), 8),
                "north": round(float(lat_edges[i + 1]), 8),
                "west": round(float(lon_edges[j]), 8),
                "east": round(float(lon_edges[j + 1]), 8),
                "tile_id": f"T{i+1:02d}_{j+1:02d}",
            })
    return tiles


def recommended_tiling(area_km2: float) -> dict:
    area = float(area_km2 or 0)
    if area <= 3000:
        return {"rows": 1, "cols": 1, "mode": "normal"}
    if area <= 10000:
        return {"rows": 2, "cols": 2, "mode": "tiled"}
    if area <= 30000:
        return {"rows": 3, "cols": 3, "mode": "tiled"}
    return {"rows": 4, "cols": 4, "mode": "tiled"}


def mosaic_geotiffs(tile_files: List[Path]) -> bytes:
    import rasterio
    from rasterio.merge import merge

    datasets = []
    try:
        for p in tile_files:
            datasets.append(rasterio.open(p))
        mosaic, out_transform = merge(datasets)
        meta = datasets[0].meta.copy()
        meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_transform,
            "compress": "deflate",
            "predictor": 2,
        })
        bio = io.BytesIO()
        with rasterio.io.MemoryFile() as mem:
            with mem.open(**meta) as dst:
                dst.write(mosaic)
            bio.write(mem.read())
        return bio.getvalue()
    finally:
        for ds in datasets:
            try:
                ds.close()
            except Exception:
                pass


def download_dem_normal_or_tiled(
    dem_type: str,
    bbox: Dict[str, float],
    api_key: str,
    mode: str = "Auto",
    rows: int | None = None,
    cols: int | None = None,
    progress_callback=None,
) -> TileDownloadResult:
    """Descarga DEM desde OpenTopography en modo normal o por teselas y retorna GeoTIFF único.

    mode:
    - "Normal": una sola solicitud.
    - "Por partes": divide bbox, descarga teselas y mosaica internamente.
    - "Auto": decide por área del bbox.
    """
    area = bbox_area_km2(bbox)
    rec = recommended_tiling(area)
    mode_norm = (mode or "Auto").lower()

    if mode_norm == "normal" or (mode_norm == "auto" and rec["mode"] == "normal"):
        if progress_callback:
            progress_callback("Descargando DEM en una sola solicitud...", 0.2)
        data = download_dem(dem_type, bbox, api_key)
        if progress_callback:
            progress_callback("DEM descargado.", 1.0)
        return TileDownloadResult(dem_bytes=data, metadata={
            "download_mode": "normal",
            "bbox_area_km2": area,
            "tiles": 1,
            "rows": 1,
            "cols": 1,
        })

    r = int(rows or rec["rows"])
    c = int(cols or rec["cols"])
    tiles = split_bbox(bbox, r, c)
    tmp = Path(tempfile.mkdtemp(prefix="hidrosed_tiles_"))
    tile_paths = []
    meta_tiles = []

    for idx, tb in enumerate(tiles, start=1):
        msg = f"Descargando tesela {idx}/{len(tiles)} {tb['tile_id']}..."
        if progress_callback:
            progress_callback(msg, (idx - 1) / max(len(tiles), 1))
        tclean = {k: v for k, v in tb.items() if k != "tile_id"}
        data = download_dem(dem_type, tclean, api_key, timeout=(10, 360))
        fp = tmp / f"{tb['tile_id']}.tif"
        fp.write_bytes(data)
        tile_paths.append(fp)
        meta_tiles.append({
            "tile_id": tb["tile_id"],
            "bbox": tclean,
            "bytes": len(data),
        })

    if progress_callback:
        progress_callback("Uniendo DEM parciales en un GeoTIFF único...", 0.92)
    mosaic = mosaic_geotiffs(tile_paths)
    if progress_callback:
        progress_callback("DEM descargado y mosaico unificado listo.", 1.0)

    return TileDownloadResult(dem_bytes=mosaic, metadata={
        "download_mode": "tiled",
        "bbox_area_km2": area,
        "tiles": len(tiles),
        "rows": r,
        "cols": c,
        "tile_metadata": meta_tiles,
        "mosaic_bytes": len(mosaic),
    })
