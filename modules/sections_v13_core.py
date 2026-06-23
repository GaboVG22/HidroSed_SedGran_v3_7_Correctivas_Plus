"""
Aplicación Streamlit: Generador de secciones transversales desde KMZ/KML
Autor: ChatGPT para flujo de trabajo hidrológico/hidráulico DOH

Objetivo
--------
A partir de un KMZ/KML que contenga:
  1) Curvas de nivel como LineString / MultiGeometry
  2) Eje de cauce como LineString

La aplicación permite definir:
  - ancho de sección transversal;
  - número de secciones/perfiles;
  - distancia estándar entre secciones;
  - tramo de densificación entre km A y km B con N perfiles a menor separación;
  - exportación de perfiles, puntos de intersección y líneas de sección.

Uso local
---------
1. Instalar dependencias:
   pip install -r requirements.txt
2. Ejecutar:
   streamlit run app_secciones_kmz.py

Notas técnicas
--------------
- El cálculo geométrico se ejecuta en coordenadas UTM, con datum y huso seleccionables. Para Coquimbo, Chile, el valor por defecto es WGS84 / UTM zona 19S, EPSG:32719.
- El KMZ/KML estándar viene en lon/lat WGS84. La aplicación transforma a UTM para medir distancias y vuelve a WGS84 para exportar KML/KMZ/GeoJSON.
- La cota de curvas de nivel se intenta leer desde ExtendedData, altitud Z o nombre/descripcion. Si alguna cota no se detecta, se puede editar en pantalla.
"""

from __future__ import annotations

import io
import json
import math
import re
import zipfile
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pyproj import CRS, Transformer
from shapely.geometry import LineString, MultiLineString, Point, mapping
from shapely.ops import linemerge, transform


# --------------------------------------------------------------------------------------
# Configuración general
# --------------------------------------------------------------------------------------

KML_NS = {"kml": "http://www.opengis.net/kml/2.2"}
DEFAULT_WGS84 = "EPSG:4326"
APP_VERSION = "v13_fix_km_final_utm19s_3d"

# Datums/proyecciones UTM usados en Chile.
# Por defecto se usa WGS84 / UTM 19S para la Región de Coquimbo.
# El KMZ/KML se lee normalmente en coordenadas geográficas WGS84 y se reproyecta
# al CRS UTM elegido para todos los cálculos métricos.
DATUM_UTM_CONFIGS: Dict[str, Dict[str, Any]] = {
    "WGS84": {
        "label": "WGS84",
        "south_base": 32700,
        "north_base": 32600,
        "description": "WGS84 / UTM. Recomendado por defecto para KMZ/KML y Google Earth.",
    },
    "SIRGAS2000": {
        "label": "SIRGAS 2000",
        "south_codes": {17: 31977, 18: 31978, 19: 31979, 20: 31980, 21: 31981},
        "description": "SIRGAS 2000 / UTM. Útil si la cartografía base viene en SIRGAS.",
    },
    "PSAD56": {
        "label": "PSAD56",
        "south_codes": {17: 24877, 18: 24878, 19: 24879, 20: 24880},
        "description": "PSAD56 / UTM. Datum histórico; usar solo si la cartografía original está en PSAD56.",
    },
    "SAD69": {
        "label": "SAD69",
        "south_codes": {17: 29177, 18: 29178, 19: 29179, 20: 29180},
        "description": "SAD69 / UTM. Datum histórico; usar solo si la cartografía original está en SAD69.",
    },
}


@dataclass
class FeatureLine:
    fid: str
    name: str
    description: str
    extended: Dict[str, str]
    geometry_wgs84: LineString
    z_candidate: Optional[float]


@dataclass
class SectionDef:
    section_id: str
    km: float
    chainage_m: float
    x_axis: float
    y_axis: float
    line_metric: LineString
    origen: str = "base"
    motivo_relleno: str = ""


# --------------------------------------------------------------------------------------
# Utilidades de parsing KML/KMZ
# --------------------------------------------------------------------------------------


def _strip_html(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip().replace(",", ".")
    # evita capturar cadenas vacías o muy descriptivas
    m = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def _extract_elevation_candidate(name: str, description: str, extended: Dict[str, str], coords: List[Tuple[float, float, Optional[float]]]) -> Optional[float]:
    """Intenta detectar la cota de una curva de nivel.

    Prioridad:
    1. ExtendedData con claves tipo cota/elev/alt/z/height/contour.
    2. Altitud Z de coordenadas KML, si es constante o casi constante.
    3. Número en el nombre de la entidad.
    4. Número en la descripción.
    """
    preferred_keys = ("cota", "elev", "elevation", "alt", "altura", "z", "height", "contour", "nivel")
    for k, v in extended.items():
        lk = str(k).lower()
        if any(token in lk for token in preferred_keys):
            num = _as_float(v)
            if num is not None:
                return num

    z_values = [z for _, _, z in coords if z is not None and abs(z) > 1e-9]
    if z_values:
        if max(z_values) - min(z_values) < 0.25:
            return round(float(np.mean(z_values)), 3)

    # evita capturar km del eje como elevación por nombre; se usará principalmente para curvas
    num_name = _as_float(name)
    if num_name is not None:
        return num_name

    num_desc = _as_float(_strip_html(description))
    if num_desc is not None:
        return num_desc

    return None


def _parse_coordinates(coord_text: str) -> List[Tuple[float, float, Optional[float]]]:
    coords: List[Tuple[float, float, Optional[float]]] = []
    if not coord_text:
        return coords
    for part in coord_text.replace("\n", " ").replace("\t", " ").split():
        bits = part.split(",")
        if len(bits) < 2:
            continue
        try:
            lon = float(bits[0])
            lat = float(bits[1])
            z = float(bits[2]) if len(bits) >= 3 and bits[2] != "" else None
            coords.append((lon, lat, z))
        except ValueError:
            continue
    return coords


def _find_text(parent: ET.Element, tag: str) -> str:
    node = parent.find(f"kml:{tag}", KML_NS)
    return node.text.strip() if node is not None and node.text else ""


def _extract_extended_data(placemark: ET.Element) -> Dict[str, str]:
    data: Dict[str, str] = {}
    for node in placemark.findall(".//kml:ExtendedData/kml:Data", KML_NS):
        key = node.attrib.get("name", "")
        val_node = node.find("kml:value", KML_NS)
        val = val_node.text.strip() if val_node is not None and val_node.text else ""
        if key:
            data[key] = val
    for node in placemark.findall(".//kml:ExtendedData/kml:SchemaData/kml:SimpleData", KML_NS):
        key = node.attrib.get("name", "")
        val = node.text.strip() if node.text else ""
        if key:
            data[key] = val
    return data


def read_kml_or_kmz(uploaded_file: io.BytesIO, filename: str) -> str:
    raw = uploaded_file.read()
    if filename.lower().endswith(".kmz"):
        with zipfile.ZipFile(io.BytesIO(raw), "r") as zf:
            kml_candidates = [n for n in zf.namelist() if n.lower().endswith(".kml")]
            if not kml_candidates:
                raise ValueError("El KMZ no contiene archivo .kml interno.")
            # normalmente doc.kml, si no existe toma el primero
            kml_name = "doc.kml" if "doc.kml" in kml_candidates else kml_candidates[0]
            return zf.read(kml_name).decode("utf-8", errors="ignore")
    return raw.decode("utf-8", errors="ignore")


def extract_lines_from_kml(kml_text: str) -> List[FeatureLine]:
    root = ET.fromstring(kml_text.encode("utf-8"))
    features: List[FeatureLine] = []
    idx = 1
    for placemark in root.findall(".//kml:Placemark", KML_NS):
        name = _find_text(placemark, "name") or f"Elemento {idx}"
        description = _find_text(placemark, "description")
        extended = _extract_extended_data(placemark)

        line_nodes = placemark.findall(".//kml:LineString", KML_NS)
        for line_node in line_nodes:
            coord_node = line_node.find("kml:coordinates", KML_NS)
            coords_raw = _parse_coordinates(coord_node.text if coord_node is not None and coord_node.text else "")
            if len(coords_raw) < 2:
                continue
            coords_2d = [(lon, lat) for lon, lat, _ in coords_raw]
            geom = LineString(coords_2d)
            zcand = _extract_elevation_candidate(name, description, extended, coords_raw)
            features.append(
                FeatureLine(
                    fid=f"F{idx:04d}",
                    name=name,
                    description=description,
                    extended=extended,
                    geometry_wgs84=geom,
                    z_candidate=zcand,
                )
            )
            idx += 1
    return features


# --------------------------------------------------------------------------------------
# Geometría y perfiles
# --------------------------------------------------------------------------------------


def make_transformers(metric_epsg: str) -> Tuple[Transformer, Transformer]:
    crs_src = CRS.from_user_input(DEFAULT_WGS84)
    crs_dst = CRS.from_user_input(metric_epsg)
    fwd = Transformer.from_crs(crs_src, crs_dst, always_xy=True)
    inv = Transformer.from_crs(crs_dst, crs_src, always_xy=True)
    return fwd, inv


def utm_epsg_from_datum(datum_key: str, zone: int, hemisphere: str) -> str:
    """Devuelve EPSG UTM según datum, huso y hemisferio."""
    zone = int(zone)
    hemi = str(hemisphere).upper().strip()
    cfg = DATUM_UTM_CONFIGS.get(datum_key, DATUM_UTM_CONFIGS["WGS84"])

    if datum_key == "WGS84":
        base = int(cfg["south_base"] if hemi == "S" else cfg["north_base"])
        return f"EPSG:{base + zone}"

    if hemi != "S":
        raise ValueError(f"{cfg['label']} solo está configurado en esta aplicación para hemisferio sur.")
    code = cfg.get("south_codes", {}).get(zone)
    if code is None:
        raise ValueError(f"No hay código EPSG configurado para {cfg['label']} / UTM zona {zone}{hemi}.")
    return f"EPSG:{code}"


def crs_label(metric_epsg: str) -> str:
    """Etiqueta legible de CRS para mostrar en interfaz y Excel."""
    try:
        crs = CRS.from_user_input(metric_epsg)
        return f"{metric_epsg} — {crs.name}"
    except Exception:
        return metric_epsg


def project_geom(geom: Any, transformer: Transformer) -> Any:
    return transform(lambda x, y, z=None: transformer.transform(x, y), geom)


def get_lines_dataframe(features: List[FeatureLine], fwd: Transformer) -> pd.DataFrame:
    rows = []
    for f in features:
        g_m = project_geom(f.geometry_wgs84, fwd)
        rows.append(
            {
                "fid": f.fid,
                "nombre": f.name,
                "largo_m": round(g_m.length, 2),
                "cota_detectada_m": f.z_candidate,
                "descripcion": _strip_html(f.description)[:120],
            }
        )
    return pd.DataFrame(rows)


def _line_tangent(line: LineString, d: float) -> Tuple[float, float]:
    length = line.length
    eps = max(min(length * 0.001, 5.0), 0.25)
    d1 = max(0.0, d - eps)
    d2 = min(length, d + eps)
    if abs(d2 - d1) < 1e-9:
        d1 = max(0.0, d - 1.0)
        d2 = min(length, d + 1.0)
    p1 = line.interpolate(d1)
    p2 = line.interpolate(d2)
    vx = p2.x - p1.x
    vy = p2.y - p1.y
    norm = math.hypot(vx, vy)
    if norm < 1e-9:
        return (1.0, 0.0)
    return (vx / norm, vy / norm)


def build_section_line(axis_line: LineString, chainage_m: float, width_m: float) -> Tuple[Point, LineString]:
    p = axis_line.interpolate(chainage_m)
    tx, ty = _line_tangent(axis_line, chainage_m)
    # normal izquierda-derecha del eje
    nx, ny = -ty, tx
    half = width_m / 2.0
    p_start = (p.x - nx * half, p.y - ny * half)
    p_end = (p.x + nx * half, p.y + ny * half)
    return p, LineString([p_start, p_end])


def _unique_chainages(chainages: Iterable[float], tolerance_m: float = 0.05) -> List[float]:
    values = sorted(float(c) for c in chainages if np.isfinite(c))
    unique: List[float] = []
    for c in values:
        if not unique or abs(c - unique[-1]) > tolerance_m:
            unique.append(c)
    return unique


def generate_chainages(
    axis_length_m: float,
    km_start: float,
    km_end: float,
    standard_spacing_m: float,
    dense_km_start: Optional[float],
    dense_km_end: Optional[float],
    dense_count: int,
    include_ends: bool = True,
) -> List[float]:
    start_m = max(0.0, km_start * 1000.0)
    end_m = min(axis_length_m, km_end * 1000.0)
    if end_m < start_m:
        start_m, end_m = end_m, start_m

    chainages: List[float] = []
    if include_ends:
        chainages.extend([start_m, end_m])

    if standard_spacing_m > 0:
        current = start_m
        while current <= end_m + 1e-6:
            chainages.append(current)
            current += standard_spacing_m

    if dense_km_start is not None and dense_km_end is not None and dense_count > 0:
        ds = max(start_m, dense_km_start * 1000.0)
        de = min(end_m, dense_km_end * 1000.0)
        if de < ds:
            ds, de = de, ds
        if dense_count == 1:
            chainages.append((ds + de) / 2.0)
        else:
            chainages.extend(np.linspace(ds, de, dense_count).tolist())

    return _unique_chainages(c for c in chainages if start_m - 1e-6 <= c <= end_m + 1e-6)


def build_sections(
    axis_line: LineString,
    chainages: List[float],
    width_m: float,
    origen_por_chainage: Optional[Dict[float, Tuple[str, str]]] = None,
    chainage_tolerance_m: float = 0.10,
) -> List[SectionDef]:
    sections: List[SectionDef] = []
    origen_por_chainage = origen_por_chainage or {}

    def _origin_for(ch: float) -> Tuple[str, str]:
        for key, value in origen_por_chainage.items():
            if abs(float(key) - float(ch)) <= chainage_tolerance_m:
                return value
        return ("base", "")

    for i, ch in enumerate(chainages, start=1):
        p, sec_line = build_section_line(axis_line, ch, width_m)
        origen, motivo = _origin_for(ch)
        sections.append(
            SectionDef(
                section_id=f"S-{i:03d}",
                km=round(ch / 1000.0, 4),
                chainage_m=ch,
                x_axis=p.x,
                y_axis=p.y,
                line_metric=sec_line,
                origen=origen,
                motivo_relleno=motivo,
            )
        )
    return sections


def _intersection_points(geom: Any) -> List[Point]:
    if geom.is_empty:
        return []
    gt = geom.geom_type
    if gt == "Point":
        return [geom]
    if gt == "MultiPoint":
        return list(geom.geoms)
    if gt == "LineString":
        # Si una curva coincide con parte de la sección, se toma punto medio del tramo de superposición
        return [geom.interpolate(geom.length / 2.0)] if geom.length > 0 else []
    if gt == "MultiLineString":
        return [part.interpolate(part.length / 2.0) for part in geom.geoms if part.length > 0]
    if gt == "GeometryCollection":
        pts: List[Point] = []
        for part in geom.geoms:
            pts.extend(_intersection_points(part))
        return pts
    return []



def sections_to_dataframe(sections: List[SectionDef], inv_transformer: Transformer) -> pd.DataFrame:
    """Tabla explícita de secciones generadas, con extremos y eje en UTM y WGS84."""
    rows: List[Dict[str, Any]] = []
    for sec in sections:
        coords = list(sec.line_metric.coords)
        (x_ini, y_ini), (x_fin, y_fin) = coords[0], coords[-1]
        lon_axis, lat_axis = inv_transformer.transform(sec.x_axis, sec.y_axis)
        lon_ini, lat_ini = inv_transformer.transform(x_ini, y_ini)
        lon_fin, lat_fin = inv_transformer.transform(x_fin, y_fin)
        rows.append(
            {
                "section_id": sec.section_id,
                "km_eje": round(sec.km, 5),
                "chainage_m": round(sec.chainage_m, 3),
                "origen": sec.origen,
                "motivo_relleno": sec.motivo_relleno,
                "ancho_m": round(sec.line_metric.length, 3),
                "eje_x_utm": round(sec.x_axis, 3),
                "eje_y_utm": round(sec.y_axis, 3),
                "inicio_x_utm": round(float(x_ini), 3),
                "inicio_y_utm": round(float(y_ini), 3),
                "fin_x_utm": round(float(x_fin), 3),
                "fin_y_utm": round(float(y_fin), 3),
                "eje_lon": round(lon_axis, 8),
                "eje_lat": round(lat_axis, 8),
                "inicio_lon": round(lon_ini, 8),
                "inicio_lat": round(lat_ini, 8),
                "fin_lon": round(lon_fin, 8),
                "fin_lat": round(lat_fin, 8),
            }
        )
    return pd.DataFrame(rows)

def sample_profiles(
    sections: List[SectionDef],
    contours_metric: List[Tuple[str, float, LineString]],
    inv_transformer: Transformer,
    min_duplicate_distance_m: float = 0.05,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    point_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for sec in sections:
        # estación 0 en extremo izquierdo, estación ancho en extremo derecho
        sec_length = sec.line_metric.length
        local_points: List[Tuple[float, float, Point, str]] = []
        for contour_id, elev, contour_geom in contours_metric:
            inter = sec.line_metric.intersection(contour_geom)
            for pt in _intersection_points(inter):
                sta = sec.line_metric.project(pt)
                if -1e-6 <= sta <= sec_length + 1e-6:
                    local_points.append((sta, elev, pt, contour_id))

        # elimina duplicados por estación/cota aproximada
        local_points.sort(key=lambda x: (x[0], x[1]))
        clean: List[Tuple[float, float, Point, str]] = []
        for sta, elev, pt, cid in local_points:
            if clean and abs(sta - clean[-1][0]) <= min_duplicate_distance_m and abs(elev - clean[-1][1]) <= 0.01:
                continue
            clean.append((sta, elev, pt, cid))

        elev_axis = np.nan
        if len(clean) >= 2:
            xs = np.array([p[0] for p in clean], dtype=float)
            zs = np.array([p[1] for p in clean], dtype=float)
            # cota estimada en eje por interpolación lineal entre puntos de curva si queda dentro del rango muestreado
            axis_sta = sec_length / 2.0
            if xs.min() <= axis_sta <= xs.max():
                elev_axis = float(np.interp(axis_sta, xs, zs))

        for sta, elev, pt, cid in clean:
            lon, lat = inv_transformer.transform(pt.x, pt.y)
            point_rows.append(
                {
                    "section_id": sec.section_id,
                    "km_eje": sec.km,
                    "chainage_m": round(sec.chainage_m, 3),
                    "offset_m": round(sta - sec_length / 2.0, 3),
                    "station_from_left_m": round(sta, 3),
                    "elevacion_m": round(elev, 3),
                    "x_utm": round(pt.x, 3),
                    "y_utm": round(pt.y, 3),
                    "lon": round(lon, 8),
                    "lat": round(lat, 8),
                    "contour_id": cid,
                }
            )

        summary_rows.append(
            {
                "section_id": sec.section_id,
                "km_eje": sec.km,
                "chainage_m": round(sec.chainage_m, 3),
                "ancho_m": round(sec_length, 3),
                "n_puntos_perfil": len(clean),
                "cota_eje_estimada_m": None if np.isnan(elev_axis) else round(elev_axis, 3),
                "cota_min_m": None if not clean else round(min(p[1] for p in clean), 3),
                "cota_max_m": None if not clean else round(max(p[1] for p in clean), 3),
            }
        )

    return pd.DataFrame(point_rows), pd.DataFrame(summary_rows)


def sample_longitudinal_axis_profile(
    axis_line: LineString,
    contours_metric: List[Tuple[str, float, LineString]],
    inv_transformer: Transformer,
    min_duplicate_distance_m: float = 0.05,
) -> pd.DataFrame:
    """Obtiene un perfil longitudinal preliminar por intersección eje-curvas de nivel.

    Cada punto representa el cruce del eje del cauce con una curva de nivel. El resultado
    no sustituye un MDE/levantamiento topográfico; es una lectura geométrica de las curvas.
    """
    rows: List[Dict[str, Any]] = []
    for contour_id, elev, contour_geom in contours_metric:
        inter = axis_line.intersection(contour_geom)
        for pt in _intersection_points(inter):
            ch = axis_line.project(pt)
            if -1e-6 <= ch <= axis_line.length + 1e-6:
                lon, lat = inv_transformer.transform(pt.x, pt.y)
                rows.append(
                    {
                        "fuente": "interseccion_eje_curva",
                        "km_eje": round(ch / 1000.0, 5),
                        "chainage_m": round(ch, 3),
                        "elevacion_m": round(float(elev), 3),
                        "x_utm": round(pt.x, 3),
                        "y_utm": round(pt.y, 3),
                        "lon": round(lon, 8),
                        "lat": round(lat, 8),
                        "contour_id": contour_id,
                    }
                )

    if not rows:
        return pd.DataFrame(columns=["fuente", "km_eje", "chainage_m", "elevacion_m", "x_utm", "y_utm", "lon", "lat", "contour_id"])

    df = pd.DataFrame(rows).sort_values(["chainage_m", "elevacion_m"]).reset_index(drop=True)
    clean_rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        if clean_rows and abs(float(r["chainage_m"]) - float(clean_rows[-1]["chainage_m"])) <= min_duplicate_distance_m and abs(float(r["elevacion_m"]) - float(clean_rows[-1]["elevacion_m"])) <= 0.01:
            continue
        clean_rows.append(r.to_dict())
    return pd.DataFrame(clean_rows)


def estimated_longitudinal_from_sections(profile_summary: pd.DataFrame) -> pd.DataFrame:
    """Perfil longitudinal estimado a partir de la cota interpolada en el eje de cada sección."""
    if profile_summary.empty or "cota_eje_estimada_m" not in profile_summary.columns:
        return pd.DataFrame(columns=["fuente", "section_id", "km_eje", "chainage_m", "elevacion_m"])
    df = profile_summary.dropna(subset=["cota_eje_estimada_m"]).copy()
    if df.empty:
        return pd.DataFrame(columns=["fuente", "section_id", "km_eje", "chainage_m", "elevacion_m"])
    out = df[["section_id", "km_eje", "chainage_m", "cota_eje_estimada_m"]].rename(columns={"cota_eje_estimada_m": "elevacion_m"})
    out.insert(0, "fuente", "interpolacion_en_seccion")
    return out.sort_values("chainage_m").reset_index(drop=True)



def evaluate_section_quality(
    sections: List[SectionDef],
    profile_points: pd.DataFrame,
    profile_summary: pd.DataFrame,
    min_points: int = 3,
    max_gap_m: Optional[float] = None,
    require_axis_elevation: bool = True,
) -> pd.DataFrame:
    """Evalúa si una sección transversal está suficientemente definida por las curvas.

    Criterios usados:
    - pocos puntos de intersección sección-curva;
    - cota del eje no interpolable;
    - tramo máximo sin información mayor al umbral definido.

    Una sección débil no se elimina: se marca para revisión y puede gatillar secciones
    intermedias de relleno aguas arriba/abajo.
    """
    summary_by_id: Dict[str, Dict[str, Any]] = {}
    if not profile_summary.empty:
        for _, r in profile_summary.iterrows():
            summary_by_id[str(r["section_id"])] = r.to_dict()

    rows: List[Dict[str, Any]] = []
    for sec in sections:
        sec_points = pd.DataFrame()
        if not profile_points.empty and "section_id" in profile_points.columns:
            sec_points = profile_points[profile_points["section_id"] == sec.section_id].copy()

        n_points = int(len(sec_points))
        width = float(sec.line_metric.length)
        max_gap = width
        if not sec_points.empty and "station_from_left_m" in sec_points.columns:
            stations = sorted(float(v) for v in sec_points["station_from_left_m"].dropna().tolist())
            stations = [0.0] + stations + [width]
            max_gap = max((b - a) for a, b in zip(stations[:-1], stations[1:])) if len(stations) >= 2 else width

        summary = summary_by_id.get(sec.section_id, {})
        axis_elev = summary.get("cota_eje_estimada_m", None)
        axis_is_nan = axis_elev is None or (isinstance(axis_elev, float) and np.isnan(axis_elev)) or pd.isna(axis_elev)

        reasons: List[str] = []
        if n_points < min_points:
            reasons.append(f"pocos puntos ({n_points}<{min_points})")
        if require_axis_elevation and axis_is_nan:
            reasons.append("sin cota interpolada en eje")
        if max_gap_m is not None and max_gap > max_gap_m:
            reasons.append(f"tramo sin dato {max_gap:.1f} m>{max_gap_m:.1f} m")

        rows.append(
            {
                "section_id": sec.section_id,
                "km_eje": sec.km,
                "chainage_m": round(sec.chainage_m, 3),
                "origen": sec.origen,
                "n_puntos_perfil": n_points,
                "max_tramo_sin_dato_m": round(max_gap, 3),
                "cota_eje_estimada_m": None if axis_is_nan else axis_elev,
                "calidad": "DEBIL" if reasons else "OK",
                "motivo": "; ".join(reasons) if reasons else "",
            }
        )
    return pd.DataFrame(rows)



def evaluate_modelable_sections(
    sections: List[SectionDef],
    profile_points: pd.DataFrame,
    profile_summary: pd.DataFrame,
    section_quality: Optional[pd.DataFrame] = None,
    min_points_each_bank: int = 2,
    min_total_points: int = 4,
    require_axis_elevation: bool = True,
    min_bank_span_m: float = 0.0,
) -> pd.DataFrame:
    """Clasifica secciones aptas para modelación geométrica del cauce.

    Una sección seleccionada debe estar definida en ambas riberas del eje:
    - al menos ``min_points_each_bank`` puntos con offset negativo;
    - al menos ``min_points_each_bank`` puntos con offset positivo;
    - al menos ``min_total_points`` puntos en total;
    - opcionalmente, cota interpolada en el eje;
    - opcionalmente, cobertura mínima por ribera.

    Esto evita incorporar perfiles como el ejemplo entregado, donde el terreno queda
    representado por muy pocos puntos y no describe adecuadamente ambas márgenes.
    """
    section_quality = section_quality if section_quality is not None else pd.DataFrame()
    q_by_id: Dict[str, Dict[str, Any]] = {}
    if not section_quality.empty and "section_id" in section_quality.columns:
        for _, r in section_quality.iterrows():
            q_by_id[str(r["section_id"])] = r.to_dict()

    s_by_id: Dict[str, Dict[str, Any]] = {}
    if profile_summary is not None and not profile_summary.empty and "section_id" in profile_summary.columns:
        for _, r in profile_summary.iterrows():
            s_by_id[str(r["section_id"])] = r.to_dict()

    rows: List[Dict[str, Any]] = []
    for sec in sections:
        sid = str(sec.section_id)
        if profile_points is not None and not profile_points.empty and "section_id" in profile_points.columns:
            pts = profile_points[profile_points["section_id"].astype(str) == sid].copy()
        else:
            pts = pd.DataFrame()

        offsets = pd.to_numeric(pts["offset_m"], errors="coerce") if not pts.empty and "offset_m" in pts.columns else pd.Series(dtype=float)
        left = offsets[offsets < -1e-6]
        right = offsets[offsets > 1e-6]
        n_left = int(left.count())
        n_right = int(right.count())
        n_axis = int(((offsets >= -1e-6) & (offsets <= 1e-6)).sum()) if not offsets.empty else 0
        n_total = int(offsets.count())
        left_span = float(abs(left.min())) if n_left > 0 else 0.0
        right_span = float(right.max()) if n_right > 0 else 0.0

        summary = s_by_id.get(sid, {})
        cota_eje = summary.get("cota_eje_estimada_m", None)
        axis_missing = cota_eje is None or pd.isna(cota_eje)
        q = q_by_id.get(sid, {})
        calidad_original = str(q.get("calidad", ""))
        motivo_original = str(q.get("motivo", ""))

        reasons: List[str] = []
        if n_left < int(min_points_each_bank):
            reasons.append(f"ribera izquierda insuficiente ({n_left}<{int(min_points_each_bank)})")
        if n_right < int(min_points_each_bank):
            reasons.append(f"ribera derecha insuficiente ({n_right}<{int(min_points_each_bank)})")
        if n_total < int(min_total_points):
            reasons.append(f"puntos totales insuficientes ({n_total}<{int(min_total_points)})")
        if require_axis_elevation and axis_missing:
            reasons.append("sin cota interpolada en eje")
        if min_bank_span_m > 0 and left_span < float(min_bank_span_m):
            reasons.append(f"cobertura izquierda baja ({left_span:.1f} m<{float(min_bank_span_m):.1f} m)")
        if min_bank_span_m > 0 and right_span < float(min_bank_span_m):
            reasons.append(f"cobertura derecha baja ({right_span:.1f} m<{float(min_bank_span_m):.1f} m)")
        # La calidad previa se informa, pero no se usa como rechazo automático duro.
        # Para modelación, el criterio principal es que exista geometría a ambas riberas.
        # Una sección puede aparecer como DEBIL por brechas entre curvas, pero aun así tener
        # ambas riberas definidas y ser útil como sección preliminar.
        advertencia_calidad = f"control calidad: {motivo_original}" if calidad_original == "DEBIL" and motivo_original else ("control calidad: sección débil" if calidad_original == "DEBIL" else "")

        selected = len(reasons) == 0
        rows.append(
            {
                "section_id": sid,
                "km_eje": round(float(sec.km), 5),
                "chainage_m": round(float(sec.chainage_m), 3),
                "origen": sec.origen,
                "seleccion_modelacion": bool(selected),
                "estado_modelacion": "SELECCIONADA" if selected else "CARGA_MANUAL",
                "observacion_modelacion": "" if selected else "; ".join(reasons),
                "n_puntos_total": n_total,
                "n_puntos_izquierda": n_left,
                "n_puntos_derecha": n_right,
                "n_puntos_eje": n_axis,
                "cobertura_izquierda_m": round(left_span, 3),
                "cobertura_derecha_m": round(right_span, 3),
                "cota_eje_estimada_m": None if axis_missing else cota_eje,
                "calidad_previa": calidad_original,
                "motivo_calidad_previa": motivo_original,
                "advertencia_calidad_previa": advertencia_calidad,
            }
        )
    return pd.DataFrame(rows)


def build_longitudinal_modelacion(
    profile_summary: pd.DataFrame,
    modelable_sections: pd.DataFrame,
    longitudinal_axis: pd.DataFrame,
) -> pd.DataFrame:
    """Construye perfil longitudinal de modelación y marca en rojo los km a cargar manualmente.

    La columna ``estado_modelacion`` separa las secciones seleccionadas de aquellas que
    no cumplen criterios de ambas riberas. Para las no seleccionadas se entrega una
    cota de apoyo para graficar cuando sea posible, pero el estado queda como
    ``CARGA_MANUAL``.
    """
    if modelable_sections is None or modelable_sections.empty:
        return pd.DataFrame(
            columns=[
                "section_id", "km_eje", "chainage_m", "estado_modelacion", "elevacion_m",
                "cota_para_grafico_m", "km_seleccionada", "z_seleccionada_m",
                "km_carga_manual", "z_carga_manual_m", "observacion_modelacion"
            ]
        )

    df = modelable_sections.copy()
    # Prioriza cota eje del resumen porque puede venir más limpia que la tabla de calidad.
    if profile_summary is not None and not profile_summary.empty and {"section_id", "cota_eje_estimada_m"}.issubset(profile_summary.columns):
        merge_cols = ["section_id", "cota_eje_estimada_m"]
        if "km_eje" in profile_summary.columns:
            merge_cols.append("km_eje")
        tmp = profile_summary[merge_cols].copy()
        tmp["section_id"] = tmp["section_id"].astype(str)
        df["section_id"] = df["section_id"].astype(str)
        df = df.merge(tmp[["section_id", "cota_eje_estimada_m"]].rename(columns={"cota_eje_estimada_m": "cota_eje_resumen_m"}), on="section_id", how="left")
    else:
        df["cota_eje_resumen_m"] = np.nan

    df["elevacion_m"] = pd.to_numeric(df.get("cota_eje_resumen_m", np.nan), errors="coerce")
    if "cota_eje_estimada_m" in df.columns:
        df["elevacion_m"] = df["elevacion_m"].fillna(pd.to_numeric(df["cota_eje_estimada_m"], errors="coerce"))

    # Cota auxiliar para graficar puntos rojos sin cota de eje: interpolación desde eje-curvas o desde seleccionadas.
    aux_z = df["elevacion_m"].copy()
    missing = aux_z.isna()
    if missing.any() and longitudinal_axis is not None and not longitudinal_axis.empty and {"chainage_m", "elevacion_m"}.issubset(longitudinal_axis.columns):
        axis_df = longitudinal_axis.dropna(subset=["chainage_m", "elevacion_m"]).sort_values("chainage_m")
        if len(axis_df) >= 2:
            aux_z.loc[missing] = np.interp(
                pd.to_numeric(df.loc[missing, "chainage_m"], errors="coerce"),
                pd.to_numeric(axis_df["chainage_m"], errors="coerce"),
                pd.to_numeric(axis_df["elevacion_m"], errors="coerce"),
            )
    missing = aux_z.isna()
    if missing.any():
        valid_selected = df[(df.get("seleccion_modelacion", False) == True) & df["elevacion_m"].notna()].sort_values("chainage_m")
        if len(valid_selected) >= 2:
            aux_z.loc[missing] = np.interp(
                pd.to_numeric(df.loc[missing, "chainage_m"], errors="coerce"),
                pd.to_numeric(valid_selected["chainage_m"], errors="coerce"),
                pd.to_numeric(valid_selected["elevacion_m"], errors="coerce"),
            )

    df["cota_para_grafico_m"] = aux_z
    sel = df["seleccion_modelacion"].astype(bool)
    df["km_seleccionada"] = np.where(sel, df["km_eje"], np.nan)
    df["z_seleccionada_m"] = np.where(sel, df["elevacion_m"], np.nan)
    df["km_carga_manual"] = np.where(~sel, df["km_eje"], np.nan)
    df["z_carga_manual_m"] = np.where(~sel, df["cota_para_grafico_m"], np.nan)

    keep_cols = [
        "section_id", "km_eje", "chainage_m", "estado_modelacion", "seleccion_modelacion",
        "elevacion_m", "cota_para_grafico_m", "km_seleccionada", "z_seleccionada_m",
        "km_carga_manual", "z_carga_manual_m", "n_puntos_izquierda", "n_puntos_derecha",
        "n_puntos_total", "observacion_modelacion"
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    return df[keep_cols].sort_values("chainage_m").reset_index(drop=True)


def filter_selected_profile_points(profile_points: pd.DataFrame, modelable_sections: pd.DataFrame) -> pd.DataFrame:
    """Devuelve solo los puntos de perfiles transversales aptos para modelación."""
    if profile_points is None or profile_points.empty or modelable_sections is None or modelable_sections.empty:
        return pd.DataFrame(columns=profile_points.columns if profile_points is not None else [])
    selected_ids = set(modelable_sections.loc[modelable_sections["seleccion_modelacion"] == True, "section_id"].astype(str))
    if not selected_ids:
        return profile_points.iloc[0:0].copy()
    return profile_points[profile_points["section_id"].astype(str).isin(selected_ids)].copy()

def build_infill_chainages_from_weak_sections(
    base_chainages: List[float],
    quality_df: pd.DataFrame,
    axis_length_m: float,
    n_between: int = 1,
    min_spacing_m: float = 5.0,
) -> Tuple[List[float], Dict[float, Tuple[str, str]]]:
    """Agrega secciones intermedias alrededor de secciones débiles.

    Para cada sección débil, se densifican los tramos entre la sección débil y sus
    vecinas inmediata aguas arriba y aguas abajo. Si n_between=1, agrega el punto medio.
    Si n_between=2, agrega tercios, etc.
    """
    base = _unique_chainages(base_chainages, tolerance_m=0.05)
    origin_map: Dict[float, Tuple[str, str]] = {float(ch): ("base", "") for ch in base}
    if quality_df.empty or n_between <= 0:
        return base, origin_map

    weak = quality_df[quality_df["calidad"] == "DEBIL"].copy()
    if weak.empty:
        return base, origin_map

    additions: List[float] = []
    base_sorted = sorted(base)
    for _, r in weak.iterrows():
        ch = float(r["chainage_m"])
        motivo = str(r.get("motivo", "sección débil"))
        idx = min(range(len(base_sorted)), key=lambda i: abs(base_sorted[i] - ch)) if base_sorted else 0
        neighbor_pairs: List[Tuple[float, float]] = []
        if idx > 0:
            neighbor_pairs.append((base_sorted[idx - 1], base_sorted[idx]))
        if idx < len(base_sorted) - 1:
            neighbor_pairs.append((base_sorted[idx], base_sorted[idx + 1]))
        # Si es el único perfil, agrega puntos a ambos lados dentro del eje.
        if not neighbor_pairs:
            local = max(min_spacing_m, 10.0)
            neighbor_pairs = [(max(0.0, ch - local), ch), (ch, min(axis_length_m, ch + local))]

        for a, b in neighbor_pairs:
            if b < a:
                a, b = b, a
            span = b - a
            if span <= 2.0 * min_spacing_m:
                continue
            for j in range(1, n_between + 1):
                new_ch = a + span * j / (n_between + 1)
                if new_ch <= 0 or new_ch >= axis_length_m:
                    new_ch = min(max(new_ch, 0.0), axis_length_m)
                if min(abs(new_ch - existing) for existing in base + additions) >= min_spacing_m:
                    additions.append(new_ch)
                    origin_map[float(new_ch)] = ("relleno_debil", f"Relleno por {r['section_id']}: {motivo}")

    all_chainages = _unique_chainages(base + additions, tolerance_m=0.05)
    # Reconstruye mapa sobre las progresivas únicas resultantes.
    final_map: Dict[float, Tuple[str, str]] = {}
    for ch in all_chainages:
        found = None
        for key, value in origin_map.items():
            if abs(float(key) - float(ch)) <= 0.10:
                found = value
                break
        final_map[float(ch)] = found or ("base", "")
    return all_chainages, final_map

def _sample_line_for_plot(line: LineString, max_points: int = 250) -> List[Tuple[float, float]]:
    if line.is_empty or line.length <= 0:
        return []
    coords = list(line.coords)
    if len(coords) <= max_points:
        return [(float(x), float(y)) for x, y in coords]
    distances = np.linspace(0.0, line.length, max_points)
    return [(float(line.interpolate(d).x), float(line.interpolate(d).y)) for d in distances]


def make_plan_view_df(
    axis_line: LineString,
    sections: List[SectionDef],
    contours_metric: List[Tuple[str, float, LineString]],
    include_contours: bool = True,
    max_contours: int = 200,
) -> pd.DataFrame:
    """Construye una tabla liviana para graficar planta en coordenadas UTM."""
    rows: List[Dict[str, Any]] = []

    for order, (x, y) in enumerate(_sample_line_for_plot(axis_line, max_points=600)):
        rows.append({"grupo": "Eje del cauce", "tipo": "Eje", "orden": order, "x_utm": x, "y_utm": y})

    for sec in sections:
        for order, (x, y) in enumerate(list(sec.line_metric.coords)):
            rows.append({"grupo": sec.section_id, "tipo": "Sección", "orden": order, "x_utm": float(x), "y_utm": float(y)})

    if include_contours:
        for contour_id, elev, geom in contours_metric[:max_contours]:
            for order, (x, y) in enumerate(_sample_line_for_plot(geom, max_points=80)):
                rows.append({"grupo": f"Curva {contour_id} z={elev:g}", "tipo": "Curva de nivel", "orden": order, "x_utm": x, "y_utm": y})

    return pd.DataFrame(rows)


def longitudinal_to_geojson(long_df: pd.DataFrame) -> str:
    features = []
    if long_df.empty or not {"lon", "lat"}.issubset(long_df.columns):
        return json.dumps({"type": "FeatureCollection", "features": []}, ensure_ascii=False, indent=2)
    for _, r in long_df.iterrows():
        props = {k: (None if pd.isna(v) else v) for k, v in r.to_dict().items() if k not in ("lon", "lat")}
        features.append(
            {
                "type": "Feature",
                "properties": props,
                "geometry": {"type": "Point", "coordinates": [float(r["lon"]), float(r["lat"]), float(r["elevacion_m"])]},
            }
        )
    return json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False, indent=2)


def hec_ras_like_dataframe(profile_points: pd.DataFrame) -> pd.DataFrame:
    """Tabla preliminar station-elevation, útil para preparar geometría en HEC-RAS u otro modelo.

    La estación se entrega de izquierda a derecha desde el extremo izquierdo de la sección.
    """
    if profile_points.empty:
        return pd.DataFrame(columns=["river_station_m", "section_id", "km_eje", "station_m", "offset_m", "elevation_m"])
    df = profile_points.copy()
    out = pd.DataFrame(
        {
            "river_station_m": df["chainage_m"],
            "section_id": df["section_id"],
            "km_eje": df["km_eje"],
            "station_m": df["station_from_left_m"],
            "offset_m": df["offset_m"],
            "elevation_m": df["elevacion_m"],
        }
    )
    return out.sort_values(["river_station_m", "station_m"]).reset_index(drop=True)


def _dxf_header() -> List[str]:
    return ["0", "SECTION", "2", "HEADER", "9", "$ACADVER", "1", "AC1009", "0", "ENDSEC", "0", "SECTION", "2", "ENTITIES"]


def _dxf_footer() -> List[str]:
    return ["0", "ENDSEC", "0", "EOF"]


def _dxf_line(layer: str, x1: float, y1: float, x2: float, y2: float, z1: float = 0.0, z2: float = 0.0) -> List[str]:
    return [
        "0", "LINE", "8", layer,
        "10", f"{x1:.3f}", "20", f"{y1:.3f}", "30", f"{z1:.3f}",
        "11", f"{x2:.3f}", "21", f"{y2:.3f}", "31", f"{z2:.3f}",
    ]


def _dxf_text(layer: str, x: float, y: float, text: str, height: float = 2.5) -> List[str]:
    safe = str(text).replace("\n", " ")[:200]
    return ["0", "TEXT", "8", layer, "10", f"{x:.3f}", "20", f"{y:.3f}", "30", "0.000", "40", f"{height:.3f}", "1", safe]


def make_plan_dxf(axis_line: LineString, sections: List[SectionDef], contours_metric: List[Tuple[str, float, LineString]]) -> str:
    """DXF simple en planta UTM: eje, secciones y curvas de nivel."""
    out = _dxf_header()
    coords = list(axis_line.coords)
    for (x1, y1), (x2, y2) in zip(coords[:-1], coords[1:]):
        out.extend(_dxf_line("EJE_CAUCE", x1, y1, x2, y2))

    for sec in sections:
        (x1, y1), (x2, y2) = list(sec.line_metric.coords)
        out.extend(_dxf_line("SECCIONES", x1, y1, x2, y2))
        out.extend(_dxf_text("TEXTOS_SECCIONES", sec.x_axis, sec.y_axis, f"{sec.section_id} km {sec.km:.3f}", height=3.0))

    for contour_id, elev, geom in contours_metric:
        ccoords = list(geom.coords)
        for (x1, y1), (x2, y2) in zip(ccoords[:-1], ccoords[1:]):
            out.extend(_dxf_line("CURVAS_NIVEL", x1, y1, x2, y2, z1=elev, z2=elev))
    out.extend(_dxf_footer())
    return "\n".join(out)


def make_profiles_dxf(profile_points: pd.DataFrame, width_m: float) -> str:
    """DXF 2D cartesiano de perfiles transversales: X=offset, Y=cota.

    Para evitar superposición en CAD, cada sección se desplaza horizontalmente.
    """
    out = _dxf_header()
    if profile_points.empty:
        out.extend(_dxf_footer())
        return "\n".join(out)

    gap = max(width_m * 1.35, width_m + 20.0)
    for i, section_id in enumerate(sorted(profile_points["section_id"].unique())):
        dfp = profile_points[profile_points["section_id"] == section_id].sort_values("offset_m")
        if dfp.empty:
            continue
        base_x = i * gap
        pts = [(base_x + float(r["offset_m"]), float(r["elevacion_m"])) for _, r in dfp.iterrows()]
        for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:]):
            out.extend(_dxf_line("PERFILES_TRANSVERSALES_2D", x1, y1, x2, y2))
        ymin = float(dfp["elevacion_m"].min())
        km = float(dfp["km_eje"].iloc[0])
        out.extend(_dxf_text("TEXTOS_PERFILES", base_x - width_m / 2.0, ymin - 4.0, f"{section_id} km {km:.3f}", height=2.5))
        out.extend(_dxf_line("EJE_LOCAL", base_x, ymin - 2.0, base_x, ymin + 2.0))
    out.extend(_dxf_footer())
    return "\n".join(out)


def make_longitudinal_dxf(long_df: pd.DataFrame) -> str:
    """DXF 2D del perfil longitudinal: X=km o progresiva m, Y=cota."""
    out = _dxf_header()
    if long_df.empty:
        out.extend(_dxf_footer())
        return "\n".join(out)
    df = long_df.sort_values("chainage_m")
    pts = [(float(r["chainage_m"]), float(r["elevacion_m"])) for _, r in df.iterrows()]
    for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:]):
        out.extend(_dxf_line("PERFIL_LONGITUDINAL_2D", x1, y1, x2, y2))
    if pts:
        out.extend(_dxf_text("TEXTOS_PERFIL_LONG", pts[0][0], pts[0][1] - 4.0, "Perfil longitudinal: X=progresiva m, Y=cota m", height=2.5))
    out.extend(_dxf_footer())
    return "\n".join(out)


# --------------------------------------------------------------------------------------
# Exportadores
# --------------------------------------------------------------------------------------


def sections_to_geojson(sections: List[SectionDef], inv: Transformer) -> str:
    features = []
    for sec in sections:
        geom_wgs = project_geom(sec.line_metric, inv)
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "section_id": sec.section_id,
                    "km_eje": sec.km,
                    "chainage_m": sec.chainage_m,
                    "ancho_m": sec.line_metric.length,
                },
                "geometry": mapping(geom_wgs),
            }
        )
    return json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False, indent=2)


def points_to_geojson(profile_points: pd.DataFrame) -> str:
    features = []
    if profile_points.empty:
        return json.dumps({"type": "FeatureCollection", "features": []}, ensure_ascii=False, indent=2)
    for _, r in profile_points.iterrows():
        props = {k: (None if pd.isna(v) else v) for k, v in r.to_dict().items() if k not in ("lon", "lat")}
        features.append(
            {
                "type": "Feature",
                "properties": props,
                "geometry": {"type": "Point", "coordinates": [float(r["lon"]), float(r["lat"])]},
            }
        )
    return json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False, indent=2)




def _modelable_lookup(modelable_sections: Optional[pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
    """Devuelve un diccionario por section_id con estado de modelación.

    Se usa para colorear secciones: azul = seleccionada, rojo = carga manual.
    """
    lookup: Dict[str, Dict[str, Any]] = {}
    if modelable_sections is None or modelable_sections.empty or "section_id" not in modelable_sections.columns:
        return lookup
    for _, r in modelable_sections.iterrows():
        sid = str(r.get("section_id", ""))
        if sid:
            lookup[sid] = r.to_dict()
    return lookup


def filter_sections_for_modelacion(
    sections: List[SectionDef],
    modelable_sections: Optional[pd.DataFrame],
    selected: Optional[bool] = None,
) -> List[SectionDef]:
    """Filtra secciones según selección de modelación.

    selected=True  -> solo secciones aptas;
    selected=False -> solo secciones descartadas/carga manual;
    selected=None  -> todas.
    """
    if selected is None or modelable_sections is None or modelable_sections.empty or "section_id" not in modelable_sections.columns:
        return list(sections)
    ids = set(
        modelable_sections.loc[
            modelable_sections["seleccion_modelacion"].astype(bool) == bool(selected),
            "section_id",
        ].astype(str)
    )
    return [sec for sec in sections if str(sec.section_id) in ids]


def make_kmz_modelacion(
    sections: List[SectionDef],
    modelable_sections: Optional[pd.DataFrame],
    inv: Transformer,
    axis_line: Optional[LineString] = None,
) -> bytes:
    """KMZ limpio para Google Earth.

    Evita el enredo visual del KMZ completo: no incluye los puntos de intersección
    con etiquetas de cota; solo exporta líneas de sección separadas por estado:
    - azul: secciones correctas/aptas para modelación;
    - rojo: secciones descartadas o que requieren carga manual.
    """
    def esc(x: Any) -> str:
        return str(x).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    lookup = _modelable_lookup(modelable_sections)

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<kml xmlns="http://www.opengis.net/kml/2.2">',
        "<Document>",
        "<name>Secciones modelación: azul correctas, rojo descartadas</name>",
        # KML usa color AABBGGRR. Azul = ffff0000; rojo = ff0000ff.
        '<Style id="secSelected"><LineStyle><color>ffff0000</color><width>4</width></LineStyle></Style>',
        '<Style id="secDiscarded"><LineStyle><color>ff0000ff</color><width>4</width></LineStyle></Style>',
        '<Style id="axisLine"><LineStyle><color>ff00ffff</color><width>3</width></LineStyle></Style>',
    ]

    if axis_line is not None and not axis_line.is_empty:
        geom_axis = project_geom(axis_line, inv)
        coords_axis = " ".join([f"{x:.8f},{y:.8f},0" for x, y in list(geom_axis.coords)])
        lines.extend([
            "<Folder><name>00_Eje_cauce</name>",
            "<Placemark>",
            "<name>Eje del cauce</name>",
            "<styleUrl>#axisLine</styleUrl>",
            "<LineString><tessellate>1</tessellate>",
            f"<coordinates>{coords_axis}</coordinates>",
            "</LineString>",
            "</Placemark>",
            "</Folder>",
        ])

    def write_section_folder(folder_name: str, selected_value: bool, style_url: str) -> None:
        lines.append(f"<Folder><name>{folder_name}</name>")
        for sec in filter_sections_for_modelacion(sections, modelable_sections, selected=selected_value):
            info = lookup.get(str(sec.section_id), {})
            estado = info.get("estado_modelacion", "SELECCIONADA" if selected_value else "CARGA_MANUAL")
            obs = info.get("observacion_modelacion", "")
            n_left = info.get("n_puntos_izquierda", "")
            n_right = info.get("n_puntos_derecha", "")
            n_total = info.get("n_puntos_total", "")
            geom_wgs = project_geom(sec.line_metric, inv)
            coords = " ".join([f"{x:.8f},{y:.8f},0" for x, y in list(geom_wgs.coords)])
            lines.extend([
                "<Placemark>",
                f"<name>{esc(sec.section_id)} km {sec.km:.3f}</name>",
                f"<styleUrl>#{style_url}</styleUrl>",
                "<ExtendedData>",
                f"<Data name=\"section_id\"><value>{esc(sec.section_id)}</value></Data>",
                f"<Data name=\"km_eje\"><value>{sec.km}</value></Data>",
                f"<Data name=\"chainage_m\"><value>{sec.chainage_m:.3f}</value></Data>",
                f"<Data name=\"estado_modelacion\"><value>{esc(estado)}</value></Data>",
                f"<Data name=\"n_puntos_izquierda\"><value>{esc(n_left)}</value></Data>",
                f"<Data name=\"n_puntos_derecha\"><value>{esc(n_right)}</value></Data>",
                f"<Data name=\"n_puntos_total\"><value>{esc(n_total)}</value></Data>",
                f"<Data name=\"observacion\"><value>{esc(obs)}</value></Data>",
                "</ExtendedData>",
                "<LineString><tessellate>1</tessellate>",
                f"<coordinates>{coords}</coordinates>",
                "</LineString>",
                "</Placemark>",
            ])
        lines.append("</Folder>")

    write_section_folder("01_Secciones_correctas_AZUL", True, "secSelected")
    write_section_folder("02_Secciones_descartadas_ROJO_carga_manual", False, "secDiscarded")

    lines.append("</Document></kml>")
    kml_text = "\n".join(lines)
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("doc.kml", kml_text)
    return mem.getvalue()


def make_kmz(sections: List[SectionDef], profile_points: pd.DataFrame, inv: Transformer) -> bytes:
    def esc(x: Any) -> str:
        return str(x).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<kml xmlns="http://www.opengis.net/kml/2.2">',
        "<Document>",
        "<name>Secciones transversales generadas</name>",
        "<Style id=\"secLine\"><LineStyle><color>ff0000ff</color><width>3</width></LineStyle></Style>",
        "<Style id=\"profPoint\"><IconStyle><scale>0.7</scale><Icon><href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href></Icon></IconStyle></Style>",
    ]
    lines.append("<Folder><name>Secciones</name>")
    for sec in sections:
        geom_wgs = project_geom(sec.line_metric, inv)
        coords = " ".join([f"{x:.8f},{y:.8f},0" for x, y in list(geom_wgs.coords)])
        lines.extend(
            [
                "<Placemark>",
                f"<name>{esc(sec.section_id)} km {sec.km:.3f}</name>",
                "<styleUrl>#secLine</styleUrl>",
                "<ExtendedData>",
                f"<Data name=\"km_eje\"><value>{sec.km}</value></Data>",
                f"<Data name=\"chainage_m\"><value>{sec.chainage_m:.3f}</value></Data>",
                "</ExtendedData>",
                "<LineString><tessellate>1</tessellate>",
                f"<coordinates>{coords}</coordinates>",
                "</LineString>",
                "</Placemark>",
            ]
        )
    lines.append("</Folder>")

    lines.append("<Folder><name>Puntos perfiles</name>")
    if not profile_points.empty:
        for _, r in profile_points.iterrows():
            lines.extend(
                [
                    "<Placemark>",
                    f"<name>{esc(r['section_id'])} z={r['elevacion_m']}</name>",
                    "<styleUrl>#profPoint</styleUrl>",
                    "<ExtendedData>",
                    f"<Data name=\"section_id\"><value>{esc(r['section_id'])}</value></Data>",
                    f"<Data name=\"km_eje\"><value>{r['km_eje']}</value></Data>",
                    f"<Data name=\"offset_m\"><value>{r['offset_m']}</value></Data>",
                    f"<Data name=\"elevacion_m\"><value>{r['elevacion_m']}</value></Data>",
                    "</ExtendedData>",
                    f"<Point><coordinates>{float(r['lon']):.8f},{float(r['lat']):.8f},{float(r['elevacion_m']):.3f}</coordinates></Point>",
                    "</Placemark>",
                ]
            )
    lines.append("</Folder>")
    lines.append("</Document></kml>")

    kml_text = "\n".join(lines)
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("doc.kml", kml_text)
    return mem.getvalue()




def make_excel_download(
    sections: List[SectionDef],
    profile_points: pd.DataFrame,
    profile_summary: pd.DataFrame,
    longitudinal_axis: pd.DataFrame,
    longitudinal_estimated: pd.DataFrame,
    inv: Transformer,
    metric_epsg: str = "EPSG:32719",
    section_quality: Optional[pd.DataFrame] = None,
    modelable_sections: Optional[pd.DataFrame] = None,
    selected_profile_points: Optional[pd.DataFrame] = None,
    longitudinal_modelacion: Optional[pd.DataFrame] = None,
) -> bytes:
    """Crea un Excel consolidado con secciones, dibujos transversales y perfil longitudinal.

    Versión v9:
    - mantiene hojas resumen/tablas generales;
    - agrega una hoja índice de dibujos;
    - crea una hoja por cada sección transversal con tabla offset-cota y gráfico XY;
    - dibuja el eje del cauce en X=0 cuando existe rango de cotas suficiente.
    """
    mem = io.BytesIO()
    sections_df = sections_to_dataframe(sections, inv)
    hec_df = hec_ras_like_dataframe(profile_points)
    section_quality = section_quality if section_quality is not None else pd.DataFrame()
    modelable_sections = modelable_sections if modelable_sections is not None else pd.DataFrame()
    selected_profile_points = selected_profile_points if selected_profile_points is not None else pd.DataFrame()
    longitudinal_modelacion = longitudinal_modelacion if longitudinal_modelacion is not None else pd.DataFrame()
    selected_hec_df = hec_ras_like_dataframe(selected_profile_points) if not selected_profile_points.empty else pd.DataFrame(columns=hec_df.columns if not hec_df.empty else [])

    metodologia = pd.DataFrame(
        [
            ["Objeto", "Generar secciones transversales desde KMZ/KML con eje de cauce y curvas de nivel."],
            ["Plano transversal", "X = offset desde el eje del cauce en metros; Y = cota del terreno en metros."],
            ["Dibujo de secciones", "El Excel incluye una hoja por cada sección con tabla offset-cota y gráfico cartesiano editable."],
            ["Perfil longitudinal", "X = km/progresiva del eje; Y = cota estimada o cota de cruce con curva de nivel."],
            ["CRS cálculo", crs_label(metric_epsg)],
            ["Coordenadas de cálculo", "UTM en metros. Este/Norte UTM en tablas; lon/lat WGS84 para KMZ/KML/GeoJSON."],
            ["Advertencia", "La geometría desde curvas de nivel es preliminar; para diseño se recomienda topografía/MDE/LiDAR y validación de terreno."],
            ["Uso hidráulico", "Base geométrica para pendiente, secciones, modelación hidráulica, socavación y transporte de sedimentos."],
            ["Relleno de secciones débiles", "Opcional: identifica secciones con pocos puntos, sin cota de eje o con grandes tramos sin dato; agrega secciones intermedias alrededor de ellas."],
            ["Selección para modelación", "Una sección apta debe tener ambas riberas definidas. Las no aptas quedan como CARGA_MANUAL y se destacan en rojo en el perfil longitudinal de modelación."],
            ["KMZ limpio azul/rojo", "Se exporta un KMZ sin puntos ni etiquetas de cota: secciones correctas en azul y descartadas/carga manual en rojo, para evitar enredo visual en Google Earth."],
        ],
        columns=["Campo", "Descripción"],
    )

    def _format_km(km_value: float) -> str:
        try:
            meters = int(round(float(km_value) * 1000.0))
            return f"{meters // 1000}+{meters % 1000:03d}"
        except Exception:
            return "s/km"

    def _quality_for_section(section_id: str) -> Dict[str, Any]:
        if section_quality is None or section_quality.empty or "section_id" not in section_quality.columns:
            return {}
        q = section_quality[section_quality["section_id"].astype(str) == str(section_id)]
        if q.empty:
            return {}
        return q.iloc[0].to_dict()

    def _summary_for_section(section_id: str) -> Dict[str, Any]:
        if profile_summary is None or profile_summary.empty or "section_id" not in profile_summary.columns:
            return {}
        s = profile_summary[profile_summary["section_id"].astype(str) == str(section_id)]
        if s.empty:
            return {}
        return s.iloc[0].to_dict()

    with pd.ExcelWriter(mem, engine="xlsxwriter") as writer:
        metodologia.to_excel(writer, sheet_name="00_Metodologia", index=False)
        sections_df.to_excel(writer, sheet_name="01_Secciones", index=False)
        profile_summary.to_excel(writer, sheet_name="02_Resumen_XS", index=False)
        profile_points.to_excel(writer, sheet_name="03_Perfiles_XS", index=False)
        longitudinal_estimated.to_excel(writer, sheet_name="04_Long_Secciones", index=False)
        longitudinal_axis.to_excel(writer, sheet_name="05_Long_Eje_Curvas", index=False)
        hec_df.to_excel(writer, sheet_name="06_HEC_RAS", index=False)
        section_quality.to_excel(writer, sheet_name="07_Calidad_Relleno", index=False)
        modelable_sections.to_excel(writer, sheet_name="08_Secciones_Modelar", index=False)
        selected_profile_points.to_excel(writer, sheet_name="09_Puntos_XS_Modelar", index=False)
        longitudinal_modelacion.to_excel(writer, sheet_name="10_Long_Modelar", index=False)
        selected_hec_df.to_excel(writer, sheet_name="11_HEC_RAS_Modelar", index=False)

        workbook = writer.book
        header_fmt = workbook.add_format({"bold": True, "font_color": "white", "bg_color": "#1F4E78", "border": 1})
        header_light_fmt = workbook.add_format({"bold": True, "font_color": "white", "bg_color": "#5B9BD5", "border": 1})
        title_fmt = workbook.add_format({"bold": True, "font_size": 14, "font_color": "#1F4E78"})
        section_title_fmt = workbook.add_format({"bold": True, "font_size": 13, "font_color": "#1F4E78"})
        note_fmt = workbook.add_format({"text_wrap": True, "valign": "top"})
        warning_fmt = workbook.add_format({"text_wrap": True, "valign": "top", "font_color": "#9C0006", "bg_color": "#FFC7CE"})
        ok_fmt = workbook.add_format({"font_color": "#006100", "bg_color": "#C6EFCE"})
        weak_fmt = workbook.add_format({"font_color": "#9C0006", "bg_color": "#FFC7CE"})
        num_fmt = workbook.add_format({"num_format": "0.000"})
        km_fmt = workbook.add_format({"num_format": "0.0000"})
        int_fmt = workbook.add_format({"num_format": "0"})
        small_note_fmt = workbook.add_format({"font_size": 9, "font_color": "#666666", "text_wrap": True})

        sheet_dfs = {
            "00_Metodologia": metodologia,
            "01_Secciones": sections_df,
            "02_Resumen_XS": profile_summary,
            "03_Perfiles_XS": profile_points,
            "04_Long_Secciones": longitudinal_estimated,
            "05_Long_Eje_Curvas": longitudinal_axis,
            "06_HEC_RAS": hec_df,
            "07_Calidad_Relleno": section_quality,
            "08_Secciones_Modelar": modelable_sections,
            "09_Puntos_XS_Modelar": selected_profile_points,
            "10_Long_Modelar": longitudinal_modelacion,
            "11_HEC_RAS_Modelar": selected_hec_df,
        }

        for sheet_name, df in sheet_dfs.items():
            ws = writer.sheets[sheet_name]
            ws.freeze_panes(1, 0)
            if not df.empty:
                ws.autofilter(0, 0, max(len(df), 1), max(len(df.columns) - 1, 0))
            for col_idx, col_name in enumerate(df.columns):
                ws.write(0, col_idx, col_name, header_fmt)

                # Ancho de columna robusto para Streamlit Cloud:
                # usa iloc para tolerar nombres de columna duplicados y convierte
                # siempre a texto antes de aplicar len().
                if not df.empty:
                    col_series = df.iloc[:, col_idx]
                    sample_values = [
                        "" if pd.isna(v) else str(v)
                        for v in col_series.head(50).tolist()
                    ]
                else:
                    col_series = pd.Series(dtype=object)
                    sample_values = []

                safe_lengths = [len(str(col_name)), 10]
                safe_lengths.extend(len("" if v is None else str(v)) for v in sample_values)
                width = max(safe_lengths) + 2
                width = min(max(width, 12), 34)
                ws.set_column(
                    col_idx,
                    col_idx,
                    width,
                    num_fmt if pd.api.types.is_numeric_dtype(col_series) else None,
                )
            if sheet_name == "00_Metodologia":
                ws.set_column(0, 0, 22)
                ws.set_column(1, 1, 90, note_fmt)
                ws.set_default_row(34)
            for col_idx, col_name in enumerate(df.columns):
                if "km" in str(col_name).lower():
                    ws.set_column(col_idx, col_idx, 14, km_fmt)

        # Gráfico de perfil longitudinal preferente: estimado desde secciones; si no existe, cruce eje-curvas.
        long_df = longitudinal_estimated if not longitudinal_estimated.empty else longitudinal_axis
        long_sheet = "04_Long_Secciones" if not longitudinal_estimated.empty else "05_Long_Eje_Curvas"
        if not long_df.empty and {"km_eje", "elevacion_m"}.issubset(long_df.columns):
            ws = writer.sheets[long_sheet]
            col_km = long_df.columns.get_loc("km_eje")
            col_elev = long_df.columns.get_loc("elevacion_m")
            n = len(long_df)
            chart = workbook.add_chart({"type": "scatter", "subtype": "straight_with_markers"})
            chart.add_series(
                {
                    "name": "Perfil longitudinal",
                    "categories": [long_sheet, 1, col_km, n, col_km],
                    "values": [long_sheet, 1, col_elev, n, col_elev],
                    "marker": {"type": "circle", "size": 5},
                    "line": {"color": "#1F4E78", "width": 1.5},
                }
            )
            chart.set_title({"name": "Perfil longitudinal: km vs cota"})
            chart.set_x_axis({"name": "Km eje"})
            chart.set_y_axis({"name": "Cota (m)"})
            chart.set_legend({"none": True})
            ws.insert_chart("H3", chart, {"x_scale": 1.35, "y_scale": 1.2})

        # Gráfico de perfil longitudinal de modelación: azul = seleccionado; rojo = cargar manual.
        if not longitudinal_modelacion.empty and {"km_seleccionada", "z_seleccionada_m", "km_carga_manual", "z_carga_manual_m"}.issubset(longitudinal_modelacion.columns):
            long_model_sheet = "10_Long_Modelar"
            ws_model = writer.sheets[long_model_sheet]
            n_model = len(longitudinal_modelacion)
            col_km_sel = longitudinal_modelacion.columns.get_loc("km_seleccionada")
            col_z_sel = longitudinal_modelacion.columns.get_loc("z_seleccionada_m")
            col_km_man = longitudinal_modelacion.columns.get_loc("km_carga_manual")
            col_z_man = longitudinal_modelacion.columns.get_loc("z_carga_manual_m")
            chart_model = workbook.add_chart({"type": "scatter", "subtype": "straight_with_markers"})
            chart_model.add_series(
                {
                    "name": "Secciones seleccionadas",
                    "categories": [long_model_sheet, 1, col_km_sel, n_model, col_km_sel],
                    "values": [long_model_sheet, 1, col_z_sel, n_model, col_z_sel],
                    "marker": {"type": "circle", "size": 5, "border": {"color": "#1D4ED8"}, "fill": {"color": "#1D4ED8"}},
                    "line": {"color": "#1D4ED8", "width": 1.5},
                }
            )
            chart_model.add_series(
                {
                    "name": "Km sin sección válida - cargar manual",
                    "categories": [long_model_sheet, 1, col_km_man, n_model, col_km_man],
                    "values": [long_model_sheet, 1, col_z_man, n_model, col_z_man],
                    "marker": {"type": "circle", "size": 7, "border": {"color": "#C00000"}, "fill": {"color": "#C00000"}},
                    "line": {"color": "#C00000", "none": True},
                }
            )
            chart_model.set_title({"name": "Perfil longitudinal de modelación: seleccionadas y carga manual"})
            chart_model.set_x_axis({"name": "Km eje"})
            chart_model.set_y_axis({"name": "Cota eje / cota apoyo (m)"})
            chart_model.set_legend({"position": "bottom"})
            chart_model.set_size({"width": 760, "height": 380})
            ws_model.insert_chart("N3", chart_model)

        # Índice y dibujo individual de cada sección transversal.
        # Se usa una hoja por sección para que el dibujo quede visible y editable dentro de Excel.
        index_sheet = workbook.add_worksheet("12_Indice_XS")
        writer.sheets["12_Indice_XS"] = index_sheet
        index_headers = [
            "N°",
            "section_id",
            "Hoja dibujo",
            "km_eje",
            "Progresiva",
            "origen",
            "calidad",
            "n_puntos",
            "cota_eje_estimada_m",
            "observacion",
        ]
        index_sheet.write(0, 0, "Índice de dibujos de secciones transversales", title_fmt)
        index_sheet.write(1, 0, "Cada hoja XS_* contiene tabla offset-cota y gráfico cartesiano X=offset; Y=cota.", small_note_fmt)
        for c, h in enumerate(index_headers):
            index_sheet.write(3, c, h, header_fmt)
        index_sheet.freeze_panes(4, 0)
        index_sheet.set_column(0, 0, 6, int_fmt)
        index_sheet.set_column(1, 2, 18)
        index_sheet.set_column(3, 4, 13, km_fmt)
        index_sheet.set_column(5, 9, 24)

        # Orden estable por progresiva.
        sections_ordered = sorted(sections, key=lambda s: (float(s.chainage_m), str(s.section_id)))
        for idx, sec in enumerate(sections_ordered, start=1):
            sheet_name = f"XS_{idx:03d}"
            km_label = _format_km(sec.km)
            q_info = _quality_for_section(sec.section_id)
            s_info = _summary_for_section(sec.section_id)
            calidad = str(q_info.get("calidad", "")) if q_info else ""
            motivo = str(q_info.get("motivo", "")) if q_info else ""
            n_puntos = int(s_info.get("n_puntos_perfil", 0) or 0)
            cota_eje = s_info.get("cota_eje_estimada_m", "")

            # Registro en índice con vínculo interno.
            index_row = idx + 3
            index_sheet.write_number(index_row, 0, idx, int_fmt)
            index_sheet.write(index_row, 1, sec.section_id)
            index_sheet.write_url(index_row, 2, f"internal:'{sheet_name}'!A1", string=sheet_name)
            index_sheet.write_number(index_row, 3, float(sec.km), km_fmt)
            index_sheet.write(index_row, 4, km_label)
            index_sheet.write(index_row, 5, sec.origen)
            index_sheet.write(index_row, 6, calidad, weak_fmt if calidad == "DEBIL" else ok_fmt if calidad == "OK" else None)
            index_sheet.write_number(index_row, 7, n_puntos, int_fmt)
            if cota_eje is not None and not pd.isna(cota_eje):
                index_sheet.write_number(index_row, 8, float(cota_eje), num_fmt)
            else:
                index_sheet.write(index_row, 8, "")
            index_sheet.write(index_row, 9, motivo)

            ws = workbook.add_worksheet(sheet_name)
            writer.sheets[sheet_name] = ws
            ws.set_zoom(90)
            ws.freeze_panes(8, 0)
            ws.write(0, 0, f"Dibujo sección transversal {sec.section_id}", section_title_fmt)
            ws.write(1, 0, "Progresiva")
            ws.write(1, 1, km_label)
            ws.write(1, 2, "km_eje")
            ws.write_number(1, 3, float(sec.km), km_fmt)
            ws.write(2, 0, "Ancho sección (m)")
            ws.write_number(2, 1, float(sec.line_metric.length), num_fmt)
            ws.write(2, 2, "Origen")
            ws.write(2, 3, sec.origen)
            ws.write(3, 0, "Calidad")
            ws.write(3, 1, calidad, weak_fmt if calidad == "DEBIL" else ok_fmt if calidad == "OK" else None)
            ws.write(3, 2, "Motivo")
            ws.write(3, 3, motivo, warning_fmt if calidad == "DEBIL" else note_fmt)
            ws.write_url(5, 0, "internal:'12_Indice_XS'!A1", string="Volver al índice")
            ws.write(6, 0, "Datos del perfil transversal", header_light_fmt)

            # Dataset de perfil de la sección.
            if profile_points is not None and not profile_points.empty and "section_id" in profile_points.columns:
                sec_df = profile_points[profile_points["section_id"].astype(str) == str(sec.section_id)].sort_values("offset_m").copy()
            else:
                sec_df = pd.DataFrame()

            if sec_df.empty:
                ws.write(8, 0, "Sin puntos de intersección con curvas de nivel. Revisar ancho de sección, distancia entre curvas o complementar con MDE/topografía.", warning_fmt)
                ws.set_column(0, 0, 32)
                ws.set_column(1, 6, 16)
                continue

            # Columnas priorizadas para revisión técnica.
            preferred_cols = [
                "offset_m",
                "elevacion_m",
                "station_from_left_m",
                "x_utm",
                "y_utm",
                "lon",
                "lat",
                "contour_id",
            ]
            cols = [c for c in preferred_cols if c in sec_df.columns]
            if not cols:
                cols = list(sec_df.columns)
            plot_df = sec_df[cols].copy()
            # Escritura manual para evitar depender de to_excel con muchas hojas.
            start_row = 8
            for c, col_name in enumerate(plot_df.columns):
                ws.write(start_row, c, col_name, header_fmt)
            for r_idx, (_, row) in enumerate(plot_df.iterrows(), start=start_row + 1):
                for c_idx, col_name in enumerate(plot_df.columns):
                    v = row[col_name]
                    if v is None or pd.isna(v):
                        ws.write(r_idx, c_idx, "")
                    elif isinstance(v, (int, float, np.integer, np.floating)):
                        ws.write_number(r_idx, c_idx, float(v), num_fmt)
                    else:
                        ws.write(r_idx, c_idx, str(v))
            ws.autofilter(start_row, 0, start_row + len(plot_df), max(len(plot_df.columns) - 1, 0))
            for c, col_name in enumerate(plot_df.columns):
                width = 14
                if str(col_name).lower() in {"contour_id", "section_id"}:
                    width = 22
                ws.set_column(c, c, width, num_fmt if pd.api.types.is_numeric_dtype(plot_df[col_name]) else None)

            # Columnas auxiliares para línea vertical del eje X=0.
            offset_col = plot_df.columns.get_loc("offset_m") if "offset_m" in plot_df.columns else None
            elev_col = plot_df.columns.get_loc("elevacion_m") if "elevacion_m" in plot_df.columns else None
            n = len(plot_df)
            if offset_col is not None and elev_col is not None and n >= 1:
                y_values = pd.to_numeric(plot_df["elevacion_m"], errors="coerce").dropna()
                x_values = pd.to_numeric(plot_df["offset_m"], errors="coerce").dropna()
                if not y_values.empty and not x_values.empty:
                    y_min = float(y_values.min())
                    y_max = float(y_values.max())
                    y_pad = max((y_max - y_min) * 0.08, 0.5)
                    y_min_axis = y_min - y_pad
                    y_max_axis = y_max + y_pad
                    x_min_axis = min(float(x_values.min()), -float(sec.line_metric.length) / 2.0)
                    x_max_axis = max(float(x_values.max()), float(sec.line_metric.length) / 2.0)

                    aux_col = max(len(plot_df.columns) + 2, 10)
                    ws.write(start_row, aux_col, "eje_x", header_fmt)
                    ws.write(start_row, aux_col + 1, "eje_y", header_fmt)
                    ws.write_number(start_row + 1, aux_col, 0.0, num_fmt)
                    ws.write_number(start_row + 2, aux_col, 0.0, num_fmt)
                    ws.write_number(start_row + 1, aux_col + 1, y_min_axis, num_fmt)
                    ws.write_number(start_row + 2, aux_col + 1, y_max_axis, num_fmt)
                    ws.set_column(aux_col, aux_col + 1, 10, num_fmt)

                    chart = workbook.add_chart({"type": "scatter", "subtype": "straight_with_markers"})
                    chart.add_series(
                        {
                            "name": f"Terreno {sec.section_id}",
                            "categories": [sheet_name, start_row + 1, offset_col, start_row + n, offset_col],
                            "values": [sheet_name, start_row + 1, elev_col, start_row + n, elev_col],
                            "marker": {"type": "circle", "size": 5, "border": {"color": "#1F4E78"}, "fill": {"color": "#1F4E78"}},
                            "line": {"color": "#1F4E78", "width": 1.5},
                        }
                    )
                    chart.add_series(
                        {
                            "name": "Eje cauce X=0",
                            "categories": [sheet_name, start_row + 1, aux_col, start_row + 2, aux_col],
                            "values": [sheet_name, start_row + 1, aux_col + 1, start_row + 2, aux_col + 1],
                            "marker": {"type": "none"},
                            "line": {"color": "#C00000", "width": 1.25, "dash_type": "dash"},
                        }
                    )
                    chart.set_title({"name": f"{sec.section_id} - Perfil transversal {km_label}"})
                    chart.set_x_axis({"name": "Offset desde eje del cauce (m)", "min": x_min_axis, "max": x_max_axis})
                    chart.set_y_axis({"name": "Cota terreno (m)", "min": y_min_axis, "max": y_max_axis})
                    chart.set_legend({"position": "bottom"})
                    chart.set_size({"width": 720, "height": 380})
                    ws.insert_chart("J3", chart)
                else:
                    ws.write(8, 9, "No fue posible construir gráfico: faltan offset o cota numérica.", warning_fmt)

        # Si existe al menos una sección con puntos, mantiene una hoja de referencia con el primer perfil.
        # Esto conserva compatibilidad con versiones previas del Excel.
        if not profile_points.empty and {"section_id", "offset_m", "elevacion_m"}.issubset(profile_points.columns):
            sec0 = str(profile_points["section_id"].dropna().iloc[0])
            df_first = profile_points[profile_points["section_id"].astype(str) == sec0].sort_values("offset_m").copy()
            df_first.to_excel(writer, sheet_name="99_Grafico_XS_Ref", index=False, startrow=1)
            ws = writer.sheets["99_Grafico_XS_Ref"]
            ws.write(0, 0, f"Perfil transversal de referencia: {sec0}", header_fmt)
            for col_idx, col_name in enumerate(df_first.columns):
                ws.write(1, col_idx, col_name, header_fmt)
                ws.set_column(col_idx, col_idx, 16)
            col_offset = df_first.columns.get_loc("offset_m")
            col_elev = df_first.columns.get_loc("elevacion_m")
            n = len(df_first)
            chart = workbook.add_chart({"type": "scatter", "subtype": "straight_with_markers"})
            chart.add_series(
                {
                    "name": sec0,
                    "categories": ["99_Grafico_XS_Ref", 2, col_offset, n + 1, col_offset],
                    "values": ["99_Grafico_XS_Ref", 2, col_elev, n + 1, col_elev],
                    "marker": {"type": "circle", "size": 5},
                }
            )
            chart.set_title({"name": "Perfil transversal: offset vs cota"})
            chart.set_x_axis({"name": "Offset desde eje (m)"})
            chart.set_y_axis({"name": "Cota terreno (m)"})
            chart.set_legend({"none": True})
            ws.insert_chart("M3", chart, {"x_scale": 1.35, "y_scale": 1.2})

    mem.seek(0)
    return mem.getvalue()

def make_zip_download(
    sections: List[SectionDef],
    profile_points: pd.DataFrame,
    profile_summary: pd.DataFrame,
    longitudinal_axis: pd.DataFrame,
    longitudinal_estimated: pd.DataFrame,
    axis_metric: LineString,
    contours_metric: List[Tuple[str, float, LineString]],
    inv: Transformer,
    metric_epsg: str = "EPSG:32719",
    section_quality: Optional[pd.DataFrame] = None,
    modelable_sections: Optional[pd.DataFrame] = None,
    selected_profile_points: Optional[pd.DataFrame] = None,
    longitudinal_modelacion: Optional[pd.DataFrame] = None,
) -> bytes:
    mem = io.BytesIO()
    hec_df = hec_ras_like_dataframe(profile_points)
    section_quality = section_quality if section_quality is not None else pd.DataFrame()
    modelable_sections = modelable_sections if modelable_sections is not None else pd.DataFrame()
    selected_profile_points = selected_profile_points if selected_profile_points is not None else pd.DataFrame()
    longitudinal_modelacion = longitudinal_modelacion if longitudinal_modelacion is not None else pd.DataFrame()
    selected_hec_df = hec_ras_like_dataframe(selected_profile_points) if not selected_profile_points.empty else pd.DataFrame(columns=hec_df.columns if not hec_df.empty else [])
    if not longitudinal_modelacion.empty and {"chainage_m", "z_seleccionada_m"}.issubset(longitudinal_modelacion.columns):
        long_for_dxf = longitudinal_modelacion[["chainage_m", "z_seleccionada_m"]].dropna().rename(columns={"z_seleccionada_m": "elevacion_m"})
        if long_for_dxf.empty:
            long_for_dxf = longitudinal_estimated if not longitudinal_estimated.empty else longitudinal_axis
    else:
        long_for_dxf = longitudinal_estimated if not longitudinal_estimated.empty else longitudinal_axis
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("00_salida_excel_secciones_y_perfil_longitudinal.xlsx", make_excel_download(sections, profile_points, profile_summary, longitudinal_axis, longitudinal_estimated, inv, metric_epsg=metric_epsg, section_quality=section_quality, modelable_sections=modelable_sections, selected_profile_points=selected_profile_points, longitudinal_modelacion=longitudinal_modelacion))
        zf.writestr("01_puntos_perfiles_transversales.csv", profile_points.to_csv(index=False).encode("utf-8-sig"))
        zf.writestr("02_resumen_secciones.csv", profile_summary.to_csv(index=False).encode("utf-8-sig"))
        zf.writestr("03_perfil_longitudinal_eje_curvas.csv", longitudinal_axis.to_csv(index=False).encode("utf-8-sig"))
        zf.writestr("04_perfil_longitudinal_estimado_secciones.csv", longitudinal_estimated.to_csv(index=False).encode("utf-8-sig"))
        zf.writestr("05_tabla_tipo_hec_ras_station_elevation.csv", hec_df.to_csv(index=False).encode("utf-8-sig"))
        zf.writestr("06_calidad_y_relleno_secciones.csv", section_quality.to_csv(index=False).encode("utf-8-sig"))
        zf.writestr("07_secciones_modelacion_solo_ambas_riberas.csv", modelable_sections.to_csv(index=False).encode("utf-8-sig"))
        zf.writestr("08_puntos_perfiles_modelacion.csv", selected_profile_points.to_csv(index=False).encode("utf-8-sig"))
        zf.writestr("09_perfil_longitudinal_modelacion_carga_manual.csv", longitudinal_modelacion.to_csv(index=False).encode("utf-8-sig"))
        zf.writestr("10_tabla_tipo_hec_ras_solo_modelacion.csv", selected_hec_df.to_csv(index=False).encode("utf-8-sig"))
        zf.writestr("11_lineas_secciones.geojson", sections_to_geojson(sections, inv))
        zf.writestr("12_puntos_perfiles.geojson", points_to_geojson(profile_points))
        zf.writestr("13_puntos_perfil_longitudinal.geojson", longitudinal_to_geojson(longitudinal_axis))
        zf.writestr("14_kmz_limpio_modelacion_azul_rojo.kmz", make_kmz_modelacion(sections, modelable_sections, inv, axis_line=axis_metric))
        zf.writestr("15_secciones_y_puntos_completo.kmz", make_kmz(sections, profile_points, inv))
        zf.writestr("16_planta_utm_secciones_eje_curvas.dxf", make_plan_dxf(axis_metric, sections, contours_metric).encode("utf-8"))
        zf.writestr("17_perfiles_transversales_2d_offset_cota.dxf", make_profiles_dxf(profile_points, sections[0].line_metric.length if sections else 80.0).encode("utf-8"))
        zf.writestr("18_perfil_longitudinal_2d_progresiva_cota.dxf", make_longitudinal_dxf(long_for_dxf).encode("utf-8"))
        zf.writestr(
            "LEEME.txt",
            (
                "Salida de la aplicación Generador de secciones transversales KMZ/KML\n\n"
                "01_puntos_perfiles_transversales.csv: puntos de intersección sección-curva de nivel, con offset y cota.\n"
                "02_resumen_secciones.csv: resumen de cada sección, km, ancho, número de puntos y cota estimada del eje.\n"
                "03_perfil_longitudinal_eje_curvas.csv: cruces directos del eje con curvas de nivel.\n"
                "04_perfil_longitudinal_estimado_secciones.csv: cota estimada del eje por interpolación en cada sección transversal.\n"
                "05_tabla_tipo_hec_ras_station_elevation.csv: tabla preliminar station-elevation por sección.\n"
                "06_calidad_y_relleno_secciones.csv: control de calidad, marca secciones débiles y rellenos generados.\n"
                "07_secciones_modelacion_solo_ambas_riberas.csv: clasificación de secciones aptas y carga manual.\n"
                "08_puntos_perfiles_modelacion.csv: puntos de perfiles solo para secciones seleccionadas.\n"
                "09_perfil_longitudinal_modelacion_carga_manual.csv: perfil longitudinal con secciones seleccionadas y km de carga manual.\n"
                "10_tabla_tipo_hec_ras_solo_modelacion.csv: station-elevation solo de secciones aptas.\n"
                "11_lineas_secciones.geojson: líneas de secciones en WGS84.\n"
                "12_puntos_perfiles.geojson: puntos perfil en WGS84.\n"
                "13_puntos_perfil_longitudinal.geojson: puntos del perfil longitudinal en WGS84.\n"
                "14_secciones_y_puntos.kmz: visualización rápida en Google Earth.\n"
                "15_planta_utm_secciones_eje_curvas.dxf: geometría en planta en coordenadas UTM.\n"
                "16_perfiles_transversales_2d_offset_cota.dxf: perfiles cartesianos X=offset, Y=cota.\n"
                "17_perfil_longitudinal_2d_progresiva_cota.dxf: perfil longitudinal X=progresiva, Y=cota.\n\n"
                "Limitación: con solo curvas de nivel se obtiene una geometría preliminar. Para diseño hidráulico, socavación o transporte de sedimentos se recomienda complementar con topografía de detalle, MDE/LiDAR, rugosidad, granulometría y caudales de diseño.\n"
            ).encode("utf-8"),
        )
    return mem.getvalue()


# --------------------------------------------------------------------------------------
# Interfaz
# --------------------------------------------------------------------------------------

