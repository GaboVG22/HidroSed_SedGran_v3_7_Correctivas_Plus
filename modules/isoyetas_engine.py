
from __future__ import annotations

import io
import math
import re
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from shapely.geometry import LineString, Point, Polygon, MultiPolygon
from shapely.ops import unary_union


def _strip_ns(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def read_kmz_kml_to_text(path_or_bytes: Any) -> str:
    if isinstance(path_or_bytes, (str, Path)):
        data = Path(path_or_bytes).read_bytes()
        name = str(path_or_bytes).lower()
    elif isinstance(path_or_bytes, bytes):
        data = path_or_bytes
        name = "bytes.kmz"
    else:
        data = path_or_bytes.read()
        name = getattr(path_or_bytes, "name", "upload.kmz").lower()

    if name.endswith(".kmz") or data[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(data), "r") as z:
            kmls = [n for n in z.namelist() if n.lower().endswith(".kml")]
            if not kmls:
                raise ValueError("El KMZ de isoyetas no contiene KML.")
            return z.read(kmls[0]).decode("utf-8", errors="ignore")
    return data.decode("utf-8", errors="ignore")


def parse_coord_text(txt: str):
    coords = []
    for tok in re.split(r"\s+", (txt or "").strip()):
        if not tok:
            continue
        vals = tok.split(",")
        if len(vals) >= 2:
            try:
                coords.append((float(vals[0]), float(vals[1])))
            except Exception:
                pass
    return coords


def extract_p24_value_mm(text: str) -> float | None:
    """Extrae un valor probable de precipitación [mm] desde nombre/descripción.

    Se privilegian números razonables para precipitación máxima diaria.
    """
    if not text:
        return None
    clean = re.sub(r"<[^>]+>", " ", str(text))
    nums = []
    for m in re.finditer(r"(?<![\d\-])(\d{1,4}(?:[.,]\d+)?)\s*(?:mm|milimetros|milímetros)?", clean, flags=re.I):
        try:
            v = float(m.group(1).replace(",", "."))
            # Rango razonable para P24/isoyeta en Chile semiárido/amplio. Evita años/códigos.
            if 1.0 <= v <= 1000.0:
                nums.append(v)
        except Exception:
            pass
    if not nums:
        return None
    # Si hay varios, elegir el más probable: valores enteros de precipitación; evitar años >1000 ya filtrados.
    return float(nums[0])


def _placemark_text(pm) -> tuple[str, str]:
    name, desc = "", ""
    for e in pm.iter():
        tag = _strip_ns(e.tag)
        if tag == "name" and e.text:
            name = e.text.strip()
        if tag == "description" and e.text:
            desc = e.text.strip()
    return name, desc


def parse_isoyetas_kml(kml_text: str) -> pd.DataFrame:
    # Algunos KMZ/KML de organismos públicos contienen prefijos no declarados
    # (típicamente xsi:schemaLocation). Primero se agrega xmlns:xsi si falta.
    if "xsi:" in kml_text and "xmlns:xsi" not in kml_text:
        kml_text = kml_text.replace(
            "<kml ",
            '<kml xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" ',
            1,
        )
    try:
        root = ET.fromstring(kml_text.encode("utf-8"))
    except ET.ParseError:
        # Segundo intento conservador: quitar atributos con prefijo y prefijos de etiquetas extendidas.
        clean = re.sub(r"\s+(?!xmlns:)[A-Za-z0-9_]+:[A-Za-z0-9_\-]+=\"[^\"]*\"", "", kml_text)
        clean = re.sub(r"<(/?)([A-Za-z0-9_]+):", r"<\1", clean)
        clean = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", clean)
        root = ET.fromstring(clean.encode("utf-8"))

    rows = []
    for pm in root.iter():
        if _strip_ns(pm.tag) != "Placemark":
            continue
        name, desc = _placemark_text(pm)
        p24 = extract_p24_value_mm(f"{name} {desc}")
        geoms = []
        for e in pm.iter():
            tag = _strip_ns(e.tag)
            if tag == "LineString":
                coords = []
                for ce in e.iter():
                    if _strip_ns(ce.tag) == "coordinates":
                        coords = parse_coord_text(ce.text or "")
                        break
                if len(coords) >= 2:
                    geoms.append(("LineString", LineString(coords)))
            elif tag == "Polygon":
                coords = []
                for ce in e.iter():
                    if _strip_ns(ce.tag) == "coordinates":
                        coords = parse_coord_text(ce.text or "")
                        if len(coords) >= 4:
                            break
                if len(coords) >= 4:
                    try:
                        poly = Polygon(coords)
                        if not poly.is_valid:
                            poly = poly.buffer(0)
                        if not poly.is_empty:
                            geoms.append(("Polygon", poly))
                    except Exception:
                        pass
            elif tag == "Point":
                coords = []
                for ce in e.iter():
                    if _strip_ns(ce.tag) == "coordinates":
                        coords = parse_coord_text(ce.text or "")
                        break
                if coords:
                    geoms.append(("Point", Point(coords[0])))

        for gtype, geom in geoms:
            rows.append({
                "nombre": name or "isoyeta",
                "descripcion": desc,
                "P24_mm": p24,
                "tipo": gtype,
                "geometry_wkt": geom.wkt,
                "centroid_lon": float(geom.centroid.x),
                "centroid_lat": float(geom.centroid.y),
                "longitud_grados": float(geom.length) if hasattr(geom, "length") else None,
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df[pd.notna(df["P24_mm"])].reset_index(drop=True)
    return df


def kml_polygon_from_basin_kml(basin_kml_bytes_or_text: Any):
    if basin_kml_bytes_or_text is None:
        return None
    if isinstance(basin_kml_bytes_or_text, bytes):
        text = basin_kml_bytes_or_text.decode("utf-8", errors="ignore")
    else:
        text = str(basin_kml_bytes_or_text)
    try:
        root = ET.fromstring(text.encode("utf-8"))
    except Exception:
        return None
    polys = []
    for e in root.iter():
        if _strip_ns(e.tag) == "Polygon":
            coords = []
            for ce in e.iter():
                if _strip_ns(ce.tag) == "coordinates":
                    coords = parse_coord_text(ce.text or "")
                    if len(coords) >= 4:
                        break
            if len(coords) >= 4:
                try:
                    poly = Polygon(coords)
                    if not poly.is_valid:
                        poly = poly.buffer(0)
                    if not poly.is_empty:
                        polys.append(poly)
                except Exception:
                    pass
    if not polys:
        return None
    return unary_union(polys)


def _geom_from_wkt(wkt: str):
    from shapely import wkt as shapely_wkt
    return shapely_wkt.loads(wkt)


def estimate_p24_from_isoyetas(
    isoyetas_df: pd.DataFrame,
    lon: float,
    lat: float,
    basin_kml: Any = None,
    n_nearest: int = 3,
) -> dict[str, Any]:
    if isoyetas_df is None or isoyetas_df.empty:
        return {
            "ok": False,
            "P24_mm": None,
            "metodo": "sin_isoyetas",
            "mensaje": "No hay isoyetas válidas con valores P24.",
            "detalle_df": pd.DataFrame(),
        }

    pt = Point(float(lon), float(lat))
    basin_poly = kml_polygon_from_basin_kml(basin_kml) if basin_kml is not None else None
    rows = []
    for _, r in isoyetas_df.iterrows():
        try:
            geom = _geom_from_wkt(r["geometry_wkt"])
            p24 = float(r["P24_mm"])
            dist = float(geom.distance(pt))
            contains_pt = bool(geom.contains(pt) or geom.touches(pt))
            inter_area = 0.0
            weight_area = 0.0
            if basin_poly is not None and not basin_poly.is_empty:
                try:
                    inter = geom.intersection(basin_poly)
                    inter_area = float(inter.area) if not inter.is_empty else 0.0
                    weight_area = inter_area
                except Exception:
                    inter_area = 0.0
            rows.append({
                "nombre": r.get("nombre", ""),
                "tipo": r.get("tipo", ""),
                "P24_mm": p24,
                "distancia_grados": dist,
                "contiene_punto": contains_pt,
                "intersecta_cuenca_area_grados2": inter_area,
                "peso_area": weight_area,
            })
        except Exception:
            continue

    detail = pd.DataFrame(rows)
    if detail.empty:
        return {"ok": False, "P24_mm": None, "metodo": "sin_geometrias", "mensaje": "No se pudieron interpretar geometrías de isoyetas.", "detalle_df": detail}

    # 1) Si hay polígonos que contienen punto, promedio/mediana de ellos.
    containing = detail[detail["contiene_punto"]]
    if not containing.empty:
        value = float(containing["P24_mm"].median())
        return {
            "ok": True,
            "P24_mm": value,
            "metodo": "isoyeta_contiene_punto",
            "mensaje": "P24 estimada desde isoyeta que contiene/toca el punto de control.",
            "detalle_df": detail.sort_values(["contiene_punto", "distancia_grados"], ascending=[False, True]).head(10),
        }

    # 2) Si hay intersección de polígonos con cuenca, ponderación por área.
    area_w = detail[detail["peso_area"] > 0].copy()
    if not area_w.empty and area_w["peso_area"].sum() > 0:
        value = float((area_w["P24_mm"] * area_w["peso_area"]).sum() / area_w["peso_area"].sum())
        return {
            "ok": True,
            "P24_mm": value,
            "metodo": "promedio_ponderado_cuenca_isoyetas",
            "mensaje": "P24 estimada por ponderación espacial de isoyetas sobre la cuenca.",
            "detalle_df": detail.sort_values("peso_area", ascending=False).head(10),
        }

    # 3) Líneas/puntos: distancia inversa al punto/centroide.
    nearest = detail.sort_values("distancia_grados").head(max(1, int(n_nearest))).copy()
    eps = 1e-9
    nearest["peso_idw"] = 1.0 / (nearest["distancia_grados"] + eps)
    value = float((nearest["P24_mm"] * nearest["peso_idw"]).sum() / nearest["peso_idw"].sum())
    return {
        "ok": True,
        "P24_mm": value,
        "metodo": f"interpolacion_idw_{len(nearest)}_isoyetas",
        "mensaje": "P24 estimada por interpolación IDW con isoyetas más cercanas.",
        "detalle_df": nearest,
    }


def isoyeta_inventory(isoyetas_df: pd.DataFrame) -> pd.DataFrame:
    if isoyetas_df is None or isoyetas_df.empty:
        return pd.DataFrame(columns=["n_isoyetas", "P24_min_mm", "P24_mediana_mm", "P24_max_mm", "tipos"])
    return pd.DataFrame([{
        "n_isoyetas": int(len(isoyetas_df)),
        "P24_min_mm": float(isoyetas_df["P24_mm"].min()),
        "P24_mediana_mm": float(isoyetas_df["P24_mm"].median()),
        "P24_max_mm": float(isoyetas_df["P24_mm"].max()),
        "tipos": ", ".join(sorted(set(isoyetas_df["tipo"].dropna().astype(str)))),
    }])
