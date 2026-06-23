
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict
from urllib.parse import urlencode

import requests

BASE_URL = "https://portal.opentopography.org/API/globaldem"


def bbox_from_margin(lat: float, lon: float, margin_value: float, margin_unit: str) -> Dict[str, float]:
    if margin_value <= 0:
        raise ValueError("El margen debe ser mayor que cero.")
    if margin_unit == "km":
        delta_lat = margin_value / 111.32
        cos_lat = max(math.cos(math.radians(lat)), 0.01)
        delta_lon = margin_value / (111.32 * cos_lat)
    else:
        delta_lat = margin_value
        delta_lon = margin_value
    return {
        "south": round(max(-90.0, lat - delta_lat), 8),
        "north": round(min(90.0, lat + delta_lat), 8),
        "west": round(max(-180.0, lon - delta_lon), 8),
        "east": round(min(180.0, lon + delta_lon), 8),
    }


def bbox_area_km2(bbox: Dict[str, float]) -> float:
    radius_km = 6371.0088
    south = math.radians(bbox["south"])
    north = math.radians(bbox["north"])
    west = math.radians(bbox["west"])
    east = math.radians(bbox["east"])
    return (radius_km**2) * abs(math.sin(north) - math.sin(south)) * abs(east - west)


def build_url(dem_type: str, bbox: Dict[str, float], api_key: str) -> str:
    params = {
        "demtype": dem_type,
        "south": str(bbox["south"]),
        "north": str(bbox["north"]),
        "west": str(bbox["west"]),
        "east": str(bbox["east"]),
        "outputFormat": "GTiff",
        "API_Key": api_key,
    }
    return f"{BASE_URL}?{urlencode(params)}"


def download_dem(dem_type: str, bbox: Dict[str, float], api_key: str, timeout=(10, 240)) -> bytes:
    if not api_key.strip():
        raise ValueError("Debe ingresar API Key de OpenTopography.")
    params = {
        "demtype": dem_type,
        "south": str(bbox["south"]),
        "north": str(bbox["north"]),
        "west": str(bbox["west"]),
        "east": str(bbox["east"]),
        "outputFormat": "GTiff",
        "API_Key": api_key.strip(),
    }
    try:
        r = requests.get(BASE_URL, params=params, timeout=timeout)
    except requests.Timeout as exc:
        raise RuntimeError("Tiempo de espera agotado. Reduzca el área o intente nuevamente.") from exc
    except requests.RequestException as exc:
        raise RuntimeError(f"No fue posible conectar con OpenTopography: {exc}") from exc

    if r.status_code == 204:
        raise RuntimeError("OpenTopography respondió 204 No Data. Revise cobertura DEM o bbox.")
    if r.status_code == 400:
        raise RuntimeError("OpenTopography respondió 400 Bad Request. Revise bbox, demtype y parámetros.")
    if r.status_code == 401:
        raise RuntimeError("OpenTopography respondió 401 Unauthorized. Revise API Key.")
    if r.status_code >= 400:
        raise RuntimeError(f"OpenTopography respondió HTTP {r.status_code}.")

    data = r.content
    if not data:
        raise RuntimeError("La respuesta de OpenTopography llegó vacía.")

    looks_tiff = data.startswith(b"II*\x00") or data.startswith(b"MM\x00*")
    lower = data[:500].lower()
    if not looks_tiff and (lower.startswith(b"<html") or b"error" in lower):
        msg = data.decode("utf-8", errors="ignore")[:500]
        raise RuntimeError(f"La respuesta no parece GeoTIFF válido: {msg}")
    return data


def save_dem_bytes(dem_bytes: bytes, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(dem_bytes)
    return out_path
