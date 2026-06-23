
from __future__ import annotations

from dataclasses import dataclass
import io
import math
import zipfile

import numpy as np


@dataclass
class TiledContourOutput:
    kmz_bytes: bytes
    kml_bytes: bytes
    preview_png: bytes | None
    metadata: dict


def _kml_header(name="Curvas unificadas por teselas"):
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
<name>{name}</name>
<Style id="contour"><LineStyle><color>ff222222</color><width>1</width></LineStyle></Style>
<Style id="contour_index"><LineStyle><color>ff000000</color><width>2</width></LineStyle></Style>
"""


def _kml_footer():
    return "</Document></kml>"


def _lines_to_kml(lines, level, tile_id, index_interval=None):
    style = "contour"
    if index_interval and index_interval > 0:
        try:
            if abs((float(level) / float(index_interval)) - round(float(level) / float(index_interval))) < 1e-6:
                style = "contour_index"
        except Exception:
            pass
    parts = []
    for j, coords in enumerate(lines):
        if len(coords) < 2:
            continue
        coord_txt = " ".join([f"{x:.8f},{y:.8f},0" for x, y in coords])
        parts.append(f"""
<Placemark>
<name>Cota {level:.2f} m · {tile_id}-{j}</name>
<styleUrl>#{style}</styleUrl>
<ExtendedData>
<Data name="cota_m"><value>{level:.3f}</value></Data>
<Data name="tile"><value>{tile_id}</value></Data>
</ExtendedData>
<LineString><tessellate>1</tessellate><coordinates>{coord_txt}</coordinates></LineString>
</Placemark>
""")
    return "".join(parts)


def _coords_to_wgs84(xs, ys, crs):
    if crs is None:
        return xs, ys
    try:
        epsg = crs.to_epsg()
    except Exception:
        epsg = None
    if epsg == 4326:
        return xs, ys
    try:
        from pyproj import Transformer
        tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        return tr.transform(xs, ys)
    except Exception:
        return xs, ys


def _safe_levels(levels, zmin, zmax, max_levels_tile=600):
    vals = [float(v) for v in levels if zmin <= float(v) <= zmax]
    if len(vals) > max_levels_tile:
        step = int(math.ceil(len(vals) / max_levels_tile))
        vals = vals[::step]
    return vals


def _contours_for_array(
    arr,
    transform,
    crs,
    levels,
    tile_id,
    index_interval=None,
    max_tile_cells: int = 650_000,
    max_levels_tile: int = 600,
    max_vertices_per_line: int = 1800,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from rasterio import Affine

    finite = arr[np.isfinite(arr)]
    if finite.size < 9:
        return "", {"tile_id": tile_id, "status": "sin_datos", "factor": 1, "levels": 0, "placemarks": 0}

    factor = 1
    cells = int(arr.shape[0] * arr.shape[1])
    if cells > max_tile_cells:
        factor = int(math.ceil(math.sqrt(cells / max_tile_cells)))
        arr = arr[::factor, ::factor]
        transform = transform * Affine.scale(factor, factor)

    finite = arr[np.isfinite(arr)]
    if finite.size < 9:
        return "", {"tile_id": tile_id, "status": "sin_datos_post_downsample", "factor": factor, "levels": 0, "placemarks": 0}

    zmin = float(np.nanmin(finite))
    zmax = float(np.nanmax(finite))
    levels_tile = _safe_levels(levels, zmin, zmax, max_levels_tile=max_levels_tile)
    if not levels_tile:
        return "", {"tile_id": tile_id, "status": "sin_niveles", "factor": factor, "levels": 0, "placemarks": 0}

    masked = np.ma.masked_invalid(arr)
    fig, ax = plt.subplots(figsize=(4, 3))
    try:
        cs = ax.contour(masked, levels=levels_tile)
        parts = []
        placemarks = 0
        for level, segs in zip(cs.levels, cs.allsegs):
            lines = []
            for seg in segs:
                if seg is None or len(seg) < 2:
                    continue
                step = max(1, int(len(seg) / max_vertices_per_line))
                xs_geo = []
                ys_geo = []
                for x_col, y_row in seg[::step]:
                    x, y = transform * (float(x_col), float(y_row))
                    xs_geo.append(x)
                    ys_geo.append(y)
                lon, lat = _coords_to_wgs84(xs_geo, ys_geo, crs)
                coords = [(float(x), float(y)) for x, y in zip(lon, lat)]
                if len(coords) >= 2:
                    lines.append(coords)
            if lines:
                placemarks += len(lines)
                parts.append(_lines_to_kml(lines, float(level), tile_id, index_interval=index_interval))
        return "".join(parts), {
            "tile_id": tile_id,
            "status": "ok",
            "factor": factor,
            "levels": len(levels_tile),
            "placemarks": placemarks,
            "cells_original": cells,
            "cells_used": int(arr.shape[0] * arr.shape[1]),
        }
    finally:
        plt.close(fig)


def generate_tiled_contours_from_dem(
    dem_path,
    interval_m: float = 1.0,
    tile_rows: int = 4,
    tile_cols: int = 4,
    max_levels: int = 20000,
    index_interval_m: float | None = 10.0,
    max_tile_cells: int = 650_000,
    max_levels_tile: int = 600,
) -> TiledContourOutput:
    if interval_m < 1:
        raise ValueError("La equidistancia mínima permitida es 1 metro.")

    import rasterio
    from rasterio.windows import Window
    from rasterio.windows import transform as window_transform

    kml_parts = [_kml_header()]
    total_lines_blocks = 0
    tile_reports = []

    with rasterio.open(dem_path) as src:
        width = src.width
        height = src.height
        crs = src.crs
        nodata = src.nodata

        sample = src.read(
            1,
            out_shape=(1, max(1, min(height, 1200)), max(1, min(width, 1200))),
            masked=True,
        ).astype("float64").filled(np.nan)

        if nodata is not None:
            sample = np.where(np.isclose(sample, nodata), np.nan, sample)

        finite = sample[np.isfinite(sample)]
        if finite.size < 9:
            raise ValueError("DEM sin datos válidos suficientes.")

        zmin = float(np.nanmin(finite))
        zmax = float(np.nanmax(finite))
        start = math.ceil(zmin / interval_m) * interval_m
        end = math.floor(zmax / interval_m) * interval_m
        levels = list(np.arange(start, end + interval_m, interval_m))

        if len(levels) > max_levels:
            step = int(math.ceil(len(levels) / max_levels))
            levels = levels[::step]

        tr = int(max(1, tile_rows))
        tc = int(max(1, tile_cols))
        row_edges = np.linspace(0, height, tr + 1, dtype=int)
        col_edges = np.linspace(0, width, tc + 1, dtype=int)

        for i in range(tr):
            for j in range(tc):
                r0, r1 = int(row_edges[i]), int(row_edges[i + 1])
                c0, c1 = int(col_edges[j]), int(col_edges[j + 1])
                if r1 <= r0 or c1 <= c0:
                    continue

                win = Window(c0, r0, c1 - c0, r1 - r0)
                arr = src.read(1, window=win, masked=True).astype("float64").filled(np.nan)
                if nodata is not None:
                    arr = np.where(np.isclose(arr, nodata), np.nan, arr)

                wt = window_transform(win, src.transform)
                tile_id = f"T{i+1:02d}_{j+1:02d}"
                part, report = _contours_for_array(
                    arr,
                    wt,
                    crs,
                    levels,
                    tile_id,
                    index_interval=index_interval_m,
                    max_tile_cells=max_tile_cells,
                    max_levels_tile=max_levels_tile,
                )
                tile_reports.append(report)

                if part:
                    total_lines_blocks += part.count("<Placemark>")
                    kml_parts.append(part)

    if total_lines_blocks == 0:
        raise RuntimeError("No se generaron curvas de nivel por teselas. Aumente equidistancia o revise DEM.")

    kml_parts.append(_kml_footer())
    kml_bytes = "".join(kml_parts).encode("utf-8")

    kmz_buf = io.BytesIO()
    with zipfile.ZipFile(kmz_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("doc.kml", kml_bytes)

    preview_png = _preview_dem(dem_path)

    return TiledContourOutput(
        kmz_bytes=kmz_buf.getvalue(),
        kml_bytes=kml_bytes,
        preview_png=preview_png,
        metadata={
            "modo": "curvas_por_teselas_unificadas_cloud_safe",
            "intervalo_m": float(interval_m),
            "tile_rows": int(tile_rows),
            "tile_cols": int(tile_cols),
            "total_tiles": int(tile_rows * tile_cols),
            "placemarks": int(total_lines_blocks),
            "max_levels": int(max_levels),
            "max_levels_tile": int(max_levels_tile),
            "max_tile_cells": int(max_tile_cells),
            "distancia_minima_permitida_m": 1.0,
            "tile_reports": tile_reports,
        },
    )


def _preview_dem(dem_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import rasterio

        with rasterio.open(dem_path) as src:
            arr = src.read(
                1,
                out_shape=(1, min(src.height, 1000), min(src.width, 1000)),
                masked=True,
            ).astype("float64").filled(np.nan)
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(arr)
        ax.set_title("DEM procesado por teselas")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.03)
        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png", dpi=140)
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()
    except Exception:
        return None


def split_bbox_km2_strategy(area_km2: float) -> dict:
    area = float(area_km2 or 0)
    if area <= 2500:
        return {"tile_rows": 3, "tile_cols": 3, "nota": "Teselado suave"}
    if area <= 10000:
        return {"tile_rows": 5, "tile_cols": 5, "nota": "Teselado recomendado para cuencas grandes"}
    return {"tile_rows": 8, "tile_cols": 8, "nota": "Bbox grande; considere aumentar equidistancia o dividir descarga DEM"}
