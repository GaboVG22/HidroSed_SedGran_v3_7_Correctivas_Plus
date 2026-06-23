
from __future__ import annotations

import io
import math
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import LineString
from shapely.ops import transform as shp_transform
from pyproj import CRS, Transformer

from .dem_processing import sample_dem_at_lonlat
from .kmz_utils import simple_line_kml, make_kmz_from_kml


def utm_crs_from_lonlat(lon: float, lat: float) -> CRS:
    zone = int((lon + 180) // 6) + 1
    epsg = 32700 + zone if lat < 0 else 32600 + zone
    return CRS.from_epsg(epsg)


def project_line_to_utm(line_wgs: LineString):
    c = line_wgs.centroid
    crs_utm = utm_crs_from_lonlat(c.x, c.y)
    fwd = Transformer.from_crs("EPSG:4326", crs_utm, always_xy=True)
    inv = Transformer.from_crs(crs_utm, "EPSG:4326", always_xy=True)
    line_utm = shp_transform(lambda x, y, z=None: fwd.transform(x, y), line_wgs)
    return line_utm, crs_utm, inv


def generate_preliminary_axis(control_lon: float, control_lat: float, length_km: float = 5.0, azimuth_deg: float = 0.0) -> LineString:
    crs = utm_crs_from_lonlat(control_lon, control_lat)
    fwd = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    inv = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    x0, y0 = fwd.transform(control_lon, control_lat)
    az = math.radians(azimuth_deg)
    dx = math.sin(az) * length_km * 1000
    dy = math.cos(az) * length_km * 1000
    x1, y1 = x0 + dx, y0 + dy
    lon1, lat1 = inv.transform(x1, y1)
    return LineString([(lon1, lat1), (control_lon, control_lat)])


def export_axis_kmz(line: LineString, out_path: Path) -> Path:
    kml = simple_line_kml("Eje cauce HidroSed", list(line.coords), color="ff0000ff")
    return make_kmz_from_kml(kml, out_path)


def generate_cross_sections(line_wgs: LineString, dem_source, spacing_m: float = 100.0, width_m: float = 80.0, points_each_side: int = 10):
    line_utm, crs_utm, inv = project_line_to_utm(line_wgs)
    length = line_utm.length
    if length <= 0:
        raise ValueError("El eje de cauce no tiene longitud válida.")
    if spacing_m <= 0:
        raise ValueError("El espaciamiento debe ser positivo.")

    distances = np.arange(0, length + 0.01, spacing_m)
    if distances[-1] < length:
        distances = np.append(distances, length)

    secciones = []
    puntos = []
    sec_id = 0
    for d in distances:
        p = line_utm.interpolate(float(d))
        d0 = max(0, d - min(5, spacing_m / 3))
        d1 = min(length, d + min(5, spacing_m / 3))
        p0 = line_utm.interpolate(float(d0))
        p1 = line_utm.interpolate(float(d1))
        tx, ty = p1.x - p0.x, p1.y - p0.y
        norm = math.hypot(tx, ty)
        if norm == 0:
            continue
        tx, ty = tx / norm, ty / norm
        nx, ny = -ty, tx

        sec_id += 1
        pk_m = float(d)
        offsets = np.linspace(-width_m / 2, width_m / 2, 2 * points_each_side + 1)
        min_z = float("inf")
        left_z = right_z = np.nan
        lon_eje, lat_eje = inv.transform(p.x, p.y)

        for j, off in enumerate(offsets):
            x = p.x + nx * off
            y = p.y + ny * off
            lon, lat = inv.transform(x, y)
            z = sample_dem_at_lonlat(dem_source, lon, lat)
            if np.isfinite(z):
                min_z = min(min_z, z)
            if j == 0:
                left_z = z
            if j == len(offsets) - 1:
                right_z = z
            puntos.append({
                "section_id": sec_id,
                "pk_m": pk_m,
                "offset_m": float(off),
                "lon": lon,
                "lat": lat,
                "z_m": z,
                "x_utm": x,
                "y_utm": y,
                "epsg": crs_utm.to_epsg(),
            })

        if not np.isfinite(min_z):
            min_z = np.nan
        secciones.append({
            "section_id": sec_id,
            "pk_m": pk_m,
            "lon_eje": lon_eje,
            "lat_eje": lat_eje,
            "ancho_m": float(width_m),
            "cota_fondo_m": min_z,
            "cota_borde_izq_m": left_z,
            "cota_borde_der_m": right_z,
            "n_puntos": len(offsets),
            "epsg": crs_utm.to_epsg(),
        })

    return pd.DataFrame(secciones), pd.DataFrame(puntos)


def sections_excel_bytes(secciones: pd.DataFrame, puntos: pd.DataFrame, caudales: pd.DataFrame | None = None, hidro: pd.DataFrame | None = None, sed: pd.DataFrame | None = None) -> bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        secciones.to_excel(writer, index=False, sheet_name="03_Secciones")
        puntos.to_excel(writer, index=False, sheet_name="04_Puntos_Seccion")
        if caudales is not None:
            caudales.to_excel(writer, index=False, sheet_name="05_Caudales")
        if hidro is not None:
            hidro.to_excel(writer, index=False, sheet_name="06_Hidraulica")
        if sed is not None:
            sed.to_excel(writer, index=False, sheet_name="07_Sedimentos_Socavacion")
    return bio.getvalue()
