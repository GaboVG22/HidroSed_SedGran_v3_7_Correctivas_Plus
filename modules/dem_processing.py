
from __future__ import annotations

import io
import math
import zipfile
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class ContourOutput:
    kmz_bytes: bytes
    kml_bytes: bytes
    preview_png: bytes
    metadata: dict


def read_dem(path_or_bytes):
    import rasterio
    from rasterio.io import MemoryFile

    if isinstance(path_or_bytes, (str, bytes)) and not isinstance(path_or_bytes, bytes):
        src_ctx = rasterio.open(path_or_bytes)
    elif hasattr(path_or_bytes, "__fspath__"):
        src_ctx = rasterio.open(path_or_bytes)
    else:
        mem = MemoryFile(path_or_bytes)
        src_ctx = mem.open()

    with src_ctx as src:
        data = src.read(1, masked=True).astype("float64").filled(np.nan)
        if src.nodata is not None:
            data = np.where(np.isclose(data, src.nodata), np.nan, data)
        return data, src.transform, src.crs, src.width, src.height


def coords_to_wgs84(xs, ys, crs):
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


def generate_contours(path_or_bytes, interval_m: float = 1.0, max_cells: int = 20_000_000, max_levels: int = 30000) -> ContourOutput:
    if interval_m < 1:
        raise ValueError("La distancia mínima entre curvas de nivel es 1 metro. No se permite una equidistancia menor.")

    from rasterio import Affine

    data, transform, crs, width, height = read_dem(path_or_bytes)
    finite = data[np.isfinite(data)]
    if finite.size < 25:
        raise ValueError("El DEM no contiene suficientes datos válidos.")

    cells = int(data.shape[0] * data.shape[1])
    factor = 1
    if cells > max_cells:
        factor = int(math.ceil(math.sqrt(cells / max_cells)))
        data = data[::factor, ::factor]
        transform = transform * Affine.scale(factor, factor)

    finite = data[np.isfinite(data)]
    zmin = float(np.nanmin(finite))
    zmax = float(np.nanmax(finite))
    start = math.ceil(zmin / interval_m) * interval_m
    end = math.floor(zmax / interval_m) * interval_m
    if end < start:
        raise ValueError("El rango de cotas no permite generar curvas con esa equidistancia.")

    levels = np.arange(start, end + interval_m, interval_m, dtype=float)
    if len(levels) > max_levels:
        raise ValueError(f"La equidistancia genera {len(levels):,} niveles; aumente intervalo o max_levels.")

    masked = np.ma.masked_invalid(data)
    fig_tmp, ax_tmp = plt.subplots()
    placemarks = []
    n_lines = 0
    try:
        cs = ax_tmp.contour(masked, levels=levels)
        for level, segs in zip(cs.levels, cs.allsegs):
            for seg in segs:
                if seg is None or len(seg) < 2:
                    continue
                xs_geo, ys_geo = [], []
                for x_col, y_row in seg:
                    x, y = transform * (float(x_col), float(y_row))
                    xs_geo.append(x)
                    ys_geo.append(y)
                lon, lat = coords_to_wgs84(xs_geo, ys_geo, crs)
                coords_text = " ".join(f"{float(x):.8f},{float(y):.8f},{float(level):.3f}" for x, y in zip(lon, lat))
                placemarks.append(
                    "<Placemark>"
                    f"<name>Curva {float(level):.2f} m</name>"
                    f"<description>Cota {float(level):.2f} m</description>"
                    "<Style><LineStyle><color>ff00aaff</color><width>1.2</width></LineStyle></Style>"
                    "<LineString><tessellate>1</tessellate><altitudeMode>clampToGround</altitudeMode>"
                    f"<coordinates>{coords_text}</coordinates>"
                    "</LineString></Placemark>"
                )
                n_lines += 1
    finally:
        plt.close(fig_tmp)

    if n_lines == 0:
        raise RuntimeError("No se generaron curvas de nivel.")

    kml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<kml xmlns="http://www.opengis.net/kml/2.2">\n<Document>\n'
        '<name>Curvas de nivel HidroSed</name>\n'
        f'<description>Equidistancia: {interval_m} m | Líneas: {n_lines}</description>\n'
        + "\n".join(placemarks)
        + "\n</Document>\n</kml>\n"
    ).encode("utf-8")

    kmz_buffer = io.BytesIO()
    with zipfile.ZipFile(kmz_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("doc.kml", kml)

    png = io.BytesIO()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(masked)
    ax.contour(masked, levels=levels[: min(len(levels), 80)], linewidths=0.35)
    ax.set_title("DEM + curvas de nivel")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(png, format="png", dpi=150)
    plt.close(fig)
    png.seek(0)

    meta = {
        "distancia_minima_permitida_m": 1.0,
        "control_calidad": "Curvas generadas desde DEM; equidistancia menor a 1 m bloqueada por la aplicación.",
        "equidistancia_m": float(interval_m),
        "niveles": int(len(levels)),
        "lineas": int(n_lines),
        "z_min_m": zmin,
        "z_max_m": zmax,
        "factor_decimacion": int(factor),
        "celdas_originales": cells,
        "celdas_procesadas": int(data.shape[0] * data.shape[1]),
        "crs": str(crs),
    }
    return ContourOutput(kmz_buffer.getvalue(), kml, png.getvalue(), meta)


def sample_dem_at_lonlat(path_or_bytes, lon: float, lat: float) -> float:
    import rasterio
    from rasterio.io import MemoryFile
    from pyproj import Transformer

    if isinstance(path_or_bytes, (str, bytes)) and not isinstance(path_or_bytes, bytes):
        src_ctx = rasterio.open(path_or_bytes)
    elif hasattr(path_or_bytes, "__fspath__"):
        src_ctx = rasterio.open(path_or_bytes)
    else:
        mem = MemoryFile(path_or_bytes)
        src_ctx = mem.open()

    with src_ctx as src:
        x, y = lon, lat
        try:
            epsg = src.crs.to_epsg() if src.crs else 4326
        except Exception:
            epsg = 4326
        if epsg != 4326 and src.crs:
            tr = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            x, y = tr.transform(lon, lat)
        row, col = src.index(x, y)
        arr = src.read(1, masked=True)
        if row < 0 or row >= arr.shape[0] or col < 0 or col >= arr.shape[1]:
            return float("nan")
        v = arr[row, col]
        return float(v) if not np.ma.is_masked(v) else float("nan")
