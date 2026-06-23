
from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer


ELEV_PATTERNS = [
    r"(?:cota|elev|elevation|altura|z)\s*[:=]?\s*(-?\d+(?:[.,]\d+)?)",
    r"\b(-?\d+(?:[.,]\d+)?)\s*(?:m|msnm|m\.s\.n\.m\.)\b",
    r"\b(-?\d{2,5}(?:[.,]\d+)?)\b",
]


def read_kmz_kml_bytes(uploaded_file) -> str:
    data = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
    name = getattr(uploaded_file, "name", "").lower()
    if name.endswith(".kmz"):
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            kml_names = [n for n in zf.namelist() if n.lower().endswith(".kml")]
            if not kml_names:
                raise ValueError("El KMZ de curvas de apoyo no contiene KML.")
            return zf.read(kml_names[0]).decode("utf-8", errors="ignore")
    return data.decode("utf-8", errors="ignore")


def _extract_elev(txt: str | None):
    if not txt:
        return np.nan
    clean = re.sub(r"<[^>]+>", " ", str(txt))
    for pat in ELEV_PATTERNS:
        m = re.search(pat, clean, flags=re.I)
        if m:
            try:
                return float(m.group(1).replace(",", "."))
            except Exception:
                pass
    return np.nan


def _parse_coords(coord_text: str):
    coords = []
    if not coord_text:
        return coords
    for tok in coord_text.strip().split():
        parts = tok.split(",")
        if len(parts) >= 2:
            try:
                lon = float(parts[0])
                lat = float(parts[1])
                z = float(parts[2]) if len(parts) >= 3 and parts[2] != "" else np.nan
                coords.append((lon, lat, z))
            except Exception:
                continue
    return coords


def parse_topographic_contours(kml_text: str) -> pd.DataFrame:
    """Lee curvas de apoyo topográfico desde KML/KMZ.

    Retorna puntos de línea con lon/lat/z. La cota se obtiene desde:
    - coordenada Z del KML;
    - ExtendedData/Data;
    - nombre o descripción de la curva.
    """
    import xml.etree.ElementTree as ET

    root = ET.fromstring(kml_text.encode("utf-8"))
    rows = []
    contour_id = 0
    for pm in root.findall(".//{*}Placemark"):
        name_el = pm.find("{*}name")
        desc_el = pm.find("{*}description")
        txt = ""
        if name_el is not None and name_el.text:
            txt += " " + name_el.text
        if desc_el is not None and desc_el.text:
            txt += " " + desc_el.text

        for data in pm.findall(".//{*}Data"):
            if data.attrib.get("name"):
                txt += " " + str(data.attrib.get("name"))
            val = data.find("{*}value")
            if val is not None and val.text:
                txt += " " + val.text

        elev_attr = _extract_elev(txt)
        for coord_el in pm.findall(".//{*}LineString/{*}coordinates"):
            coords = _parse_coords(coord_el.text or "")
            if len(coords) < 2:
                continue
            contour_id += 1
            for i, (lon, lat, zcoord) in enumerate(coords):
                z = zcoord if np.isfinite(zcoord) else elev_attr
                rows.append({
                    "contour_id": contour_id,
                    "vertex_id": i,
                    "lon": float(lon),
                    "lat": float(lat),
                    "z_m": float(z) if np.isfinite(z) else np.nan,
                    "source_name": name_el.text if name_el is not None and name_el.text else f"curva_{contour_id}",
                })
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No se encontraron LineString de curvas de nivel en el archivo de apoyo.")
    valid_z = int(np.isfinite(df["z_m"]).sum())
    if valid_z == 0:
        raise ValueError("Las curvas de apoyo fueron leídas, pero no se pudo reconocer la cota. Use nombres como 'Cota 450 m' o ExtendedData cota_m.")
    return df


def _utm_for_points(lon, lat):
    zone = int((float(lon) + 180) // 6) + 1
    epsg = 32700 + zone if float(lat) < 0 else 32600 + zone
    return CRS.from_epsg(epsg)


def add_utm_to_topo(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    lon0 = float(out["lon"].median())
    lat0 = float(out["lat"].median())
    crs = _utm_for_points(lon0, lat0)
    tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    xs, ys = tr.transform(out["lon"].to_numpy(float), out["lat"].to_numpy(float))
    out["x_utm"] = xs
    out["y_utm"] = ys
    out["epsg"] = crs.to_epsg()
    return out


def improve_section_points_with_topo(points_df: pd.DataFrame, topo_df: pd.DataFrame, radius_m: float = 40.0, weight_topo: float = 0.70) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ajusta cotas de puntos de sección usando curvas topográficas de apoyo.

    Método: busca puntos de curvas cercanos en UTM y calcula cota interpolada por
    distancia inversa. Si hay apoyo dentro del radio, combina:
    z_final = (1-weight_topo)*z_dem + weight_topo*z_topo
    """
    if points_df is None or points_df.empty or topo_df is None or topo_df.empty:
        return points_df, pd.DataFrame()

    pts = points_df.copy()
    topo = add_utm_to_topo(topo_df)
    topo = topo[np.isfinite(topo["z_m"])].copy()
    if topo.empty:
        return pts, pd.DataFrame()

    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(topo[["x_utm", "y_utm"]].to_numpy(float))
        qxy = pts[["x_utm", "y_utm"]].to_numpy(float)
        neigh = tree.query_ball_point(qxy, r=float(radius_m))
    except Exception:
        # Fallback lento.
        tx = topo["x_utm"].to_numpy(float)
        ty = topo["y_utm"].to_numpy(float)
        neigh = []
        for _, p in pts.iterrows():
            d = np.hypot(tx - float(p["x_utm"]), ty - float(p["y_utm"]))
            neigh.append(np.where(d <= radius_m)[0].tolist())

    z_new = []
    dist_min = []
    n_support = []
    used = []
    topo_zs = topo["z_m"].to_numpy(float)
    topo_xy = topo[["x_utm", "y_utm"]].to_numpy(float)

    for idx, p in pts.iterrows():
        ids = neigh[len(z_new)]
        z_dem = float(p["z_m"]) if np.isfinite(p["z_m"]) else np.nan
        if not ids:
            z_new.append(z_dem)
            dist_min.append(np.nan)
            n_support.append(0)
            used.append(False)
            continue
        px, py = float(p["x_utm"]), float(p["y_utm"])
        d = np.hypot(topo_xy[ids, 0] - px, topo_xy[ids, 1] - py)
        w = 1.0 / np.maximum(d, 0.5) ** 2
        z_topo = float(np.sum(w * topo_zs[ids]) / np.sum(w))
        if np.isfinite(z_dem):
            zf = (1.0 - weight_topo) * z_dem + weight_topo * z_topo
        else:
            zf = z_topo
        z_new.append(zf)
        dist_min.append(float(np.min(d)))
        n_support.append(int(len(ids)))
        used.append(True)

    pts["z_dem_original_m"] = pts["z_m"]
    pts["z_m"] = z_new
    pts["topo_support_used"] = used
    pts["topo_support_n"] = n_support
    pts["topo_support_dist_min_m"] = dist_min

    report = pts.groupby("section_id").agg(
        n_puntos=("z_m", "size"),
        n_apoyo_topo=("topo_support_used", "sum"),
        z_min=("z_m", "min"),
        z_max=("z_m", "max"),
    ).reset_index()
    report["porcentaje_apoyo_topo"] = 100.0 * report["n_apoyo_topo"] / report["n_puntos"].clip(lower=1)
    return pts, report
