
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import heapq
import math
import zipfile
from collections import deque

import numpy as np
import pandas as pd


@dataclass
class BasinResult:
    kmz_bytes: bytes
    kml_bytes: bytes
    preview_png: bytes | None
    metrics: dict


def _read_dem(path_or_bytes, max_cells: int = 1_500_000):
    import rasterio
    from rasterio import Affine
    from rasterio.io import MemoryFile

    if isinstance(path_or_bytes, (str, Path)):
        src_ctx = rasterio.open(path_or_bytes)
    else:
        mem = MemoryFile(path_or_bytes)
        src_ctx = mem.open()

    with src_ctx as src:
        data = src.read(1, masked=True).astype("float64").filled(np.nan)
        if src.nodata is not None:
            data = np.where(np.isclose(data, src.nodata), np.nan, data)
        transform = src.transform
        crs = src.crs
        factor = 1
        cells = int(data.shape[0] * data.shape[1])
        if cells > max_cells:
            factor = int(math.ceil(math.sqrt(cells / max_cells)))
            data = data[::factor, ::factor]
            transform = transform * Affine.scale(factor, factor)

    if int(np.isfinite(data).sum()) < 100:
        raise ValueError("El DEM no contiene suficientes datos válidos para delimitar la cuenca.")
    return data, transform, crs, factor


def _cell_sizes_m(transform, crs, shape):
    dx_raw = abs(float(transform.a))
    dy_raw = abs(float(transform.e))
    try:
        is_geo = bool(crs and getattr(crs, "is_geographic", False))
    except Exception:
        is_geo = False
    if is_geo:
        y_mid = transform.f + transform.e * (shape[0] / 2)
        dx = dx_raw * 111_320.0 * max(0.15, math.cos(math.radians(float(y_mid))))
        dy = dy_raw * 110_574.0
    else:
        dx, dy = dx_raw, dy_raw
    return float(dx), float(dy), float(math.sqrt(dx * dy))


def _lonlat_to_rowcol(lon, lat, transform, crs):
    if crs is not None:
        try:
            epsg = crs.to_epsg()
        except Exception:
            epsg = None
        if epsg != 4326:
            try:
                from pyproj import Transformer
                tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
                lon, lat = tr.transform(lon, lat)
            except Exception:
                pass
    inv = ~transform
    col, row = inv * (float(lon), float(lat))
    return int(round(row)), int(round(col))


def _rowcol_to_lonlat(row, col, transform, crs):
    x, y = transform * (float(col), float(row))
    if crs is not None:
        try:
            epsg = crs.to_epsg()
        except Exception:
            epsg = None
        if epsg != 4326:
            try:
                from pyproj import Transformer
                tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
                x, y = tr.transform(x, y)
            except Exception:
                pass
    return float(x), float(y)


def _priority_flood(dem, valid):
    """Priority-Flood con epsilon para evitar flats cerrados.

    La versión sin epsilon puede dejar terrazas perfectamente planas después del
    relleno de depresiones; en ese caso D8 no encuentra pendiente positiva y la
    cuenca queda incompleta. Este relleno fuerza una pendiente mínima hacia el
    borde, manteniendo cambios altimétricos despreciables para uso hidrológico.
    """
    nrows, ncols = dem.shape
    filled = dem.copy()
    visited = np.zeros_like(valid, dtype=bool)
    heap = []

    finite = dem[np.isfinite(dem)]
    relief = float(np.nanmax(finite) - np.nanmin(finite)) if finite.size else 1.0
    eps = max(1e-6, relief * 1e-9)

    for r in range(nrows):
        for c in (0, ncols - 1):
            if valid[r, c] and not visited[r, c]:
                visited[r, c] = True
                heapq.heappush(heap, (filled[r, c], r, c))
    for c in range(ncols):
        for r in (0, nrows - 1):
            if valid[r, c] and not visited[r, c]:
                visited[r, c] = True
                heapq.heappush(heap, (filled[r, c], r, c))

    neigh = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    while heap:
        z, r, c = heapq.heappop(heap)
        for dr, dc in neigh:
            rr, cc = r + dr, c + dc
            if rr < 0 or rr >= nrows or cc < 0 or cc >= ncols:
                continue
            if not valid[rr, cc] or visited[rr, cc]:
                continue
            visited[rr, cc] = True
            if filled[rr, cc] <= z:
                filled[rr, cc] = z + eps
            heapq.heappush(heap, (filled[rr, cc], rr, cc))
    return filled


def _flow_dir_d8(filled, valid, dx, dy):
    nrows, ncols = filled.shape
    dst = np.full(nrows*ncols, -1, dtype=np.int64)
    neigh = [
        (-1,-1,math.hypot(dx,dy)),(-1,0,dy),(-1,1,math.hypot(dx,dy)),
        (0,-1,dx),(0,1,dx),
        (1,-1,math.hypot(dx,dy)),(1,0,dy),(1,1,math.hypot(dx,dy)),
    ]
    for r in range(nrows):
        for c in range(ncols):
            if not valid[r,c]:
                continue
            best = -1
            best_s = 0.0
            z = filled[r,c]
            for dr, dc, dist in neigh:
                rr, cc = r+dr, c+dc
                if rr < 0 or rr >= nrows or cc < 0 or cc >= ncols or not valid[rr,cc]:
                    continue
                s = (z - filled[rr,cc]) / max(dist, 1e-9)
                if s > best_s:
                    best_s = s
                    best = rr*ncols + cc
            dst[r*ncols+c] = best
    return dst


def _flow_acc(dst, valid):
    n = dst.size
    valid_f = valid.ravel()
    indeg = np.zeros(n, dtype=np.int32)
    edges = np.where((dst >= 0) & valid_f)[0]
    np.add.at(indeg, dst[edges], 1)
    acc = np.zeros(n, dtype=np.float64)
    acc[valid_f] = 1.0
    q = deque(np.where(valid_f & (indeg == 0))[0].tolist())
    while q:
        i = q.popleft()
        j = int(dst[i])
        if j >= 0:
            acc[j] += acc[i]
            indeg[j] -= 1
            if indeg[j] == 0:
                q.append(j)
    return acc.reshape(valid.shape)


def _snap(row, col, acc, valid, radius_cells):
    """Ajusta el punto de salida al píxel de mayor acumulación dentro de un radio circular."""
    nrows, ncols = acc.shape
    row = int(np.clip(row, 0, nrows-1))
    col = int(np.clip(col, 0, ncols-1))
    r0, r1 = max(0, row-radius_cells), min(nrows, row+radius_cells+1)
    c0, c1 = max(0, col-radius_cells), min(ncols, col+radius_cells+1)
    sub = acc[r0:r1, c0:c1].copy()
    sub_valid = valid[r0:r1, c0:c1]
    yy, xx = np.indices(sub.shape)
    d2 = (yy + r0 - row)**2 + (xx + c0 - col)**2
    circular = d2 <= radius_cells**2
    sub[~sub_valid] = -1
    sub[~circular] = -1
    if np.nanmax(sub) < 0:
        raise ValueError("No se encontró celda válida para ajustar el punto de control dentro del radio definido.")
    score = sub - 1e-6*d2
    rr, cc = np.unravel_index(int(np.nanargmax(score)), score.shape)
    return int(rr+r0), int(cc+c0)



def _snap_candidates(row, col, acc, valid, radius_cells, max_candidates=80):
    """Devuelve celdas candidatas dentro del radio para evitar saltar a un río principal lejano.

    El error típico en cuencas pequeñas ocurre cuando el punto de control está cerca de
    un cauce mayor: el ajuste por máxima acumulación salta a ese cauce y la cuenca pasa
    de decenas a miles de km². Esta función conserva candidatos alternativos para que
    el área esperada controle la selección.
    """
    nrows, ncols = acc.shape
    row = int(np.clip(row, 0, nrows-1))
    col = int(np.clip(col, 0, ncols-1))
    r0, r1 = max(0, row-radius_cells), min(nrows, row+radius_cells+1)
    c0, c1 = max(0, col-radius_cells), min(ncols, col+radius_cells+1)
    sub = acc[r0:r1, c0:c1].copy()
    sub_valid = valid[r0:r1, c0:c1]
    yy, xx = np.indices(sub.shape)
    d2 = (yy + r0 - row)**2 + (xx + c0 - col)**2
    circular = d2 <= radius_cells**2
    sub[~sub_valid] = -1
    sub[~circular] = -1
    if np.nanmax(sub) < 0:
        raise ValueError("No se encontró celda válida para ajustar el punto de control dentro del radio definido.")
    flat = sub.ravel()
    order = np.argsort(flat)[::-1]
    cands = []
    seen = set()
    for idx in order:
        val = float(flat[idx])
        if val < 0:
            break
        rr, cc = np.unravel_index(int(idx), sub.shape)
        gr, gc = int(rr+r0), int(cc+c0)
        key = (gr, gc)
        if key in seen:
            continue
        seen.add(key)
        dist_cells = math.sqrt(float(d2[rr, cc]))
        cands.append({
            "row": gr,
            "col": gc,
            "acc": val,
            "dist_cells": dist_cells,
            "dist_norm": dist_cells / max(radius_cells, 1),
        })
        if len(cands) >= max_candidates:
            break
    # Garantiza que la celda original también se pruebe.
    if valid[row, col] and (row, col) not in seen:
        cands.append({
            "row": int(row),
            "col": int(col),
            "acc": float(acc[row, col]),
            "dist_cells": 0.0,
            "dist_norm": 0.0,
        })
    return cands


def _select_outlet_candidate(dst, valid, acc, row, col, radius_cells, dx, dy, expected_area_km2=None, max_area_km2=None, selection_mode="area_controlled"):
    """Selecciona outlet ajustado con control de área.

    selection_mode:
    - max_acc: comportamiento antiguo, toma máxima acumulación.
    - closest: usa celda válida más cercana al punto.
    - area_controlled: evita candidatos que exceden max_area_km2 y prioriza cercanía al área esperada.
    """
    if selection_mode == "max_acc" and expected_area_km2 is None and max_area_km2 is None:
        r1, c1 = _snap(row, col, acc, valid, radius_cells)
        basin = _upstream_mask(dst, valid, r1*valid.shape[1] + c1)
        return r1, c1, basin, [{"row": r1, "col": c1, "area_km2": float(basin.sum()*dx*dy/1_000_000), "motivo": "max_acc_antiguo"}]

    cands = _snap_candidates(row, col, acc, valid, radius_cells, max_candidates=90)
    evaluated = []
    ncols = valid.shape[1]
    for cand in cands:
        r1, c1 = int(cand["row"]), int(cand["col"])
        basin = _upstream_mask(dst, valid, r1*ncols + c1)
        area = float(int(basin.sum()) * dx * dy / 1_000_000)
        dist_norm = float(cand.get("dist_norm", 0.0))
        acc_val = max(float(cand.get("acc", 0.0)), 1.0)
        over = bool(max_area_km2 is not None and max_area_km2 > 0 and area > max_area_km2)
        if expected_area_km2 is not None and expected_area_km2 > 0 and area > 0:
            area_term = abs(math.log(area / expected_area_km2))
        else:
            area_term = 0.0
        # Puntaje menor es mejor. Se castiga exceder el área máxima y alejarse demasiado.
        score = area_term + 0.65*dist_norm - 0.04*math.log(acc_val)
        if over:
            score += 1000.0 + area/max(max_area_km2 or 1.0, 1e-9)
        if selection_mode == "closest":
            score = dist_norm - 0.01*math.log(acc_val)
        evaluated.append({
            "row": r1,
            "col": c1,
            "acc": acc_val,
            "area_km2": area,
            "dist_norm": dist_norm,
            "score": score,
            "over_max_area": over,
            "basin": basin,
        })

    if not evaluated:
        raise ValueError("No se pudo evaluar candidatos de salida para la cuenca.")

    evaluated_sorted = sorted(evaluated, key=lambda x: x["score"])
    chosen = evaluated_sorted[0]
    # Si todos exceden max_area, se selecciona el menor excedente, pero se advertirá.
    if max_area_km2 is not None and max_area_km2 > 0 and all(e["over_max_area"] for e in evaluated):
        chosen = min(evaluated, key=lambda x: x["area_km2"])
    report = [{k: v for k, v in e.items() if k != "basin"} for e in evaluated_sorted[:12]]
    return int(chosen["row"]), int(chosen["col"]), chosen["basin"], report


def _upstream_mask(dst, valid, outlet_idx):
    n = dst.size
    valid_f = valid.ravel()
    src = np.where((dst >= 0) & valid_f)[0]
    dest = dst[src]
    order = np.argsort(dest, kind="mergesort")
    dest_s = dest[order]
    src_s = src[order]
    basin = np.zeros(n, dtype=bool)
    if outlet_idx < 0 or outlet_idx >= n or not valid_f[outlet_idx]:
        return basin.reshape(valid.shape)
    stack = [int(outlet_idx)]
    basin[outlet_idx] = True
    while stack:
        target = stack.pop()
        lo = np.searchsorted(dest_s, target, side="left")
        hi = np.searchsorted(dest_s, target, side="right")
        for child in src_s[lo:hi]:
            child = int(child)
            if not basin[child]:
                basin[child] = True
                stack.append(child)
    return basin.reshape(valid.shape)


def _mask_to_polygon(mask, transform, crs, simplify_m=80.0):
    from skimage import measure
    from shapely.geometry import Polygon, MultiPolygon
    from shapely.ops import transform as shp_transform

    padded = np.pad(mask.astype(float), 1, mode="constant", constant_values=0.0)
    contours = measure.find_contours(padded, 0.5)
    polys = []
    for arr in contours:
        if len(arr) < 4:
            continue
        coords = []
        for row, col in arr:
            row = float(row) - 1.0
            col = float(col) - 1.0
            x, y = transform * (col, row)
            coords.append((x, y))
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        try:
            poly = Polygon(coords)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty:
                continue
            try:
                is_geo = bool(crs and getattr(crs, "is_geographic", False))
            except Exception:
                is_geo = False
            tol = simplify_m/111000.0 if is_geo else simplify_m
            if tol > 0:
                poly = poly.simplify(tol, preserve_topology=True)
            polys.append(poly)
        except Exception:
            pass
    if not polys:
        raise RuntimeError("No se pudo vectorizar el polígono de cuenca.")
    poly = max(polys, key=lambda p: p.area)
    if isinstance(poly, MultiPolygon):
        poly = max(poly.geoms, key=lambda p: p.area)
    if crs is not None:
        try:
            epsg = crs.to_epsg()
        except Exception:
            epsg = None
        if epsg != 4326:
            from pyproj import Transformer
            tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            poly = shp_transform(lambda x,y,z=None: tr.transform(x,y), poly)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


def _utm_crs(lon, lat):
    from pyproj import CRS
    zone = int((lon + 180)//6) + 1
    epsg = 32700 + zone if lat < 0 else 32600 + zone
    return CRS.from_epsg(epsg)


def _project_poly(poly_wgs):
    from shapely.ops import transform as shp_transform
    from pyproj import Transformer
    c = poly_wgs.centroid
    crs = _utm_crs(float(c.x), float(c.y))
    tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    return shp_transform(lambda x,y,z=None: tr.transform(x,y), poly_wgs), crs


def _morphometry(poly_wgs, basin_mask, acc, dx, dy, cell_m, outlet_rc, snapped_lon, snapped_lat, original_lon, original_lat, snapped_dist_m, flags):
    poly_m, crs = _project_poly(poly_wgs)
    area_km2 = float(poly_m.area / 1_000_000)
    area_ha = area_km2 * 100
    perimeter_km = float(poly_m.length / 1000)
    minx, miny, maxx, maxy = poly_m.bounds
    bbox_length_km = max((maxx-minx), (maxy-miny)) / 1000
    bbox_width_km = min((maxx-minx), (maxy-miny)) / 1000
    mean_width_km = area_km2 / bbox_length_km if bbox_length_km > 0 else float("nan")
    compactness_kc = 0.2821 * perimeter_km / math.sqrt(area_km2) if area_km2 > 0 else float("nan")
    form_factor = area_km2/(bbox_length_km**2) if bbox_length_km > 0 else float("nan")
    elongation_ratio = 1.128 * math.sqrt(area_km2)/bbox_length_km if bbox_length_km > 0 else float("nan")
    max_acc = float(np.nanmax(acc[basin_mask])) if int(basin_mask.sum()) else float("nan")
    return {
        "area_km2": area_km2,
        "area_ha": area_ha,
        "perimetro_km": perimeter_km,
        "epsg_morfometria": int(crs.to_epsg()),
        "centroide_lon": float(poly_wgs.centroid.x),
        "centroide_lat": float(poly_wgs.centroid.y),
        "bbox_largo_km": float(bbox_length_km),
        "bbox_ancho_km": float(bbox_width_km),
        "ancho_medio_km": float(mean_width_km),
        "coef_compacidad_kc": float(compactness_kc),
        "factor_forma": float(form_factor),
        "relacion_elongacion": float(elongation_ratio),
        "n_celdas_cuenca": int(basin_mask.sum()),
        "tamano_celda_m": float(cell_m),
        "acumulacion_salida_celdas": max_acc,
        "punto_original_lon": float(original_lon),
        "punto_original_lat": float(original_lat),
        "punto_ajustado_lon": float(snapped_lon),
        "punto_ajustado_lat": float(snapped_lat),
        "distancia_ajuste_m": float(snapped_dist_m),
        "cuenca_toca_borde_dem": bool(any("borde del DEM" in str(f) for f in flags)),
        "advertencias": flags,
    }


def _kml_poly(poly_wgs, metrics):
    coords = " ".join([f"{x:.8f},{y:.8f},0" for x,y in list(poly_wgs.exterior.coords)])
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
<name>Cuenca delimitada HidroSed</name>
<Style id="basin"><LineStyle><color>ff0000ff</color><width>2</width></LineStyle><PolyStyle><color>330000ff</color></PolyStyle></Style>
<Placemark><name>Cuenca delimitada automática</name><description>Área {metrics["area_km2"]:.3f} km2</description><styleUrl>#basin</styleUrl>
<Polygon><outerBoundaryIs><LinearRing><coordinates>{coords}</coordinates></LinearRing></outerBoundaryIs></Polygon>
</Placemark>
<Placemark><name>Punto original</name><Point><coordinates>{metrics["punto_original_lon"]:.8f},{metrics["punto_original_lat"]:.8f},0</coordinates></Point></Placemark>
<Placemark><name>Punto ajustado al cauce</name><Point><coordinates>{metrics["punto_ajustado_lon"]:.8f},{metrics["punto_ajustado_lat"]:.8f},0</coordinates></Point></Placemark>
</Document></kml>'''


def _preview(mask, acc, outlet_rc):
    import io
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig, ax = plt.subplots(figsize=(8,6))
    acc_log = np.log10(np.where(acc > 0, acc, np.nan))
    ax.imshow(acc_log)
    ax.contour(mask.astype(float), levels=[0.5], linewidths=1.5)
    ax.scatter([outlet_rc[1]], [outlet_rc[0]], s=30)
    ax.set_title("Cuenca delimitada y acumulación de flujo")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def delineate_basin(path_or_bytes, outlet_lon: float, outlet_lat: float, snap_radius_m: float = 500.0, max_cells: int = 20_000_000, simplify_m: float = 80.0, expected_area_km2: float | None = None, max_area_km2: float | None = None, selection_mode: str = 'area_controlled') -> BasinResult:
    data, transform, crs, decim = _read_dem(path_or_bytes, max_cells=max_cells)
    valid = np.isfinite(data)
    dx, dy, cell_m = _cell_sizes_m(transform, crs, data.shape)
    filled = _priority_flood(data, valid)
    dst = _flow_dir_d8(filled, valid, dx, dy)
    acc = _flow_acc(dst, valid)
    r0, c0 = _lonlat_to_rowcol(outlet_lon, outlet_lat, transform, crs)
    if not (0 <= r0 < data.shape[0] and 0 <= c0 < data.shape[1]):
        raise ValueError("El punto de control queda fuera del DEM descargado. Aumenta el margen.")
    radius_cells = max(1, int(math.ceil(snap_radius_m/max(cell_m,1e-9))))
    r1, c1, basin, candidate_report = _select_outlet_candidate(
        dst, valid, acc, r0, c0, radius_cells, dx, dy,
        expected_area_km2=expected_area_km2,
        max_area_km2=max_area_km2,
        selection_mode=selection_mode,
    )
    outlet_idx = r1*data.shape[1] + c1
    flags = []
    snapped_lon, snapped_lat = _rowcol_to_lonlat(r1, c1, transform, crs)
    snapped_dist = math.hypot((r1-r0)*dy, (c1-c0)*dx)
    basin_cells = int(basin.sum())
    area_est_km2 = float(basin_cells * dx * dy / 1_000_000)
    if max_area_km2 is not None and max_area_km2 > 0 and area_est_km2 > max_area_km2:
        flags.append(f"Control de área: la mejor cuenca estimada ({area_est_km2:.2f} km²) excede el máximo esperado ({max_area_km2:.2f} km²). Revise punto, radio o DEM.")
    if expected_area_km2 is not None and expected_area_km2 > 0:
        ratio = area_est_km2 / expected_area_km2 if expected_area_km2 else float('inf')
        if ratio > 3 or ratio < 1/3:
            flags.append(f"Control de área: cuenca estimada {area_est_km2:.2f} km² difiere mucho del área esperada {expected_area_km2:.2f} km².")
    if basin_cells < 50:
        flags.append("Cuenca muy pequeña: revisar ubicación del punto de control o aumentar el radio de ajuste.")
    if snapped_dist > max(500.0, 0.5*snap_radius_m):
        flags.append("El ajuste del punto al cauce fue alto; revisar visualmente.")
    if decim > 1:
        flags.append(f"DEM decimado por factor {decim}; para más precisión reduzca margen o aumente max_cells.")
    # Advertencia si la cuenca toca demasiados bordes del DEM: puede estar cortada.
    border_touch = int(basin[0, :].sum() + basin[-1, :].sum() + basin[:, 0].sum() + basin[:, -1].sum())
    if border_touch > max(5, 0.02 * basin_cells):
        flags.append("La cuenca toca el borde del DEM; probablemente falta área aguas arriba. Aumenta el margen de descarga del DEM.")
    poly = _mask_to_polygon(basin, transform, crs, simplify_m=simplify_m)
    metrics = _morphometry(poly, basin, acc, dx, dy, cell_m, (r1,c1), snapped_lon, snapped_lat, outlet_lon, outlet_lat, snapped_dist, flags)
    metrics["area_esperada_km2"] = float(expected_area_km2) if expected_area_km2 is not None else None
    metrics["area_maxima_control_km2"] = float(max_area_km2) if max_area_km2 is not None else None
    metrics["modo_seleccion_salida"] = str(selection_mode)
    metrics["candidatos_salida_top"] = candidate_report
    kml = _kml_poly(poly, metrics).encode("utf-8")
    import io
    kmz_buf = io.BytesIO()
    with zipfile.ZipFile(kmz_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("doc.kml", kml)
    png = _preview(basin, acc, (r1,c1))
    return BasinResult(kmz_bytes=kmz_buf.getvalue(), kml_bytes=kml, preview_png=png, metrics=metrics)


def metrics_dataframe(metrics: dict) -> pd.DataFrame:
    labels = {
        "area_km2": "Área cuenca [km²]",
        "area_ha": "Área cuenca [ha]",
        "perimetro_km": "Perímetro [km]",
        "bbox_largo_km": "Largo característico bbox [km]",
        "bbox_ancho_km": "Ancho característico bbox [km]",
        "ancho_medio_km": "Ancho medio [km]",
        "coef_compacidad_kc": "Coeficiente compacidad Kc",
        "factor_forma": "Factor de forma",
        "relacion_elongacion": "Relación de elongación",
        "tamano_celda_m": "Tamaño celda procesada [m]",
        "distancia_ajuste_m": "Distancia ajuste punto [m]",
        "acumulacion_salida_celdas": "Acumulación salida [celdas]",
    }
    return pd.DataFrame([{"parametro": lab, "clave": k, "valor": metrics.get(k)} for k, lab in labels.items()])
