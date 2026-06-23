
from __future__ import annotations

import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _read_dem(path_or_bytes):
    import rasterio
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
        bounds = src.bounds
    return data, transform, crs, bounds


def hillshade(elevation, azimuth=315, altitude=45):
    elev = np.array(elevation, dtype=float)
    finite = np.isfinite(elev)
    if not finite.any():
        return np.zeros_like(elev)
    fill = np.nanmedian(elev[finite])
    elev = np.where(finite, elev, fill)
    dy, dx = np.gradient(elev)
    slope = np.pi / 2.0 - np.arctan(np.sqrt(dx * dx + dy * dy))
    aspect = np.arctan2(-dx, dy)
    az = np.deg2rad(azimuth)
    alt = np.deg2rad(altitude)
    shaded = np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
    shaded = (shaded - shaded.min()) / max(shaded.max() - shaded.min(), 1e-9)
    return shaded


def _plot_polygon(ax, polygon_lonlat, color="#0057d8", linewidth=2.2, face_alpha=0.08):
    if polygon_lonlat is None:
        return
    xs, ys = zip(*polygon_lonlat)
    ax.fill(xs, ys, facecolor=color, alpha=face_alpha, edgecolor=color, linewidth=linewidth, zorder=5)


def _kml_polygon_coords(kml_bytes):
    if not kml_bytes:
        return None
    import xml.etree.ElementTree as ET
    try:
        root = ET.fromstring(kml_bytes)
        coords_el = root.find(".//{*}Polygon/{*}outerBoundaryIs/{*}LinearRing/{*}coordinates")
        if coords_el is None or not coords_el.text:
            return None
        coords = []
        for tok in coords_el.text.strip().split():
            parts = tok.split(",")
            if len(parts) >= 2:
                coords.append((float(parts[0]), float(parts[1])))
        return coords if len(coords) >= 3 else None
    except Exception:
        return None


def _line_coords_from_session(axis_line):
    if axis_line is None:
        return None
    try:
        return [(float(x), float(y)) for x, y in axis_line]
    except Exception:
        return None


def make_cartographic_sheet(
    dem_path,
    basin_kml_bytes=None,
    axis_line=None,
    control_point=None,
    metrics=None,
    title="HidroSed · Delimitación de cuenca y curvas de nivel",
    contour_interval=10.0,
):
    """Create professional preview PNG/PDF-like map as PNG bytes.

    This is a cartographic output renderer. It does not replace technical GIS review,
    but creates a high-quality visual sheet for reports.
    """
    data, transform, crs, bounds = _read_dem(dem_path)
    finite = data[np.isfinite(data)]
    if finite.size < 25:
        raise ValueError("DEM insuficiente para generar lámina cartográfica.")

    # extent in DEM CRS. For most OpenTopo GTiff this is EPSG:4326.
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    shade = hillshade(data)

    zmin = float(np.nanmin(finite))
    zmax = float(np.nanmax(finite))
    ci = max(float(contour_interval), 1.0)
    start = np.ceil(zmin / ci) * ci
    end = np.floor(zmax / ci) * ci
    levels = np.arange(start, end + ci, ci)
    if len(levels) > 160:
        # prevent unreadable sheet; technical export can still be 1 m in KMZ.
        step = int(np.ceil(len(levels) / 160))
        levels = levels[::step]

    fig = plt.figure(figsize=(16, 9), dpi=150)
    ax = fig.add_axes([0.05, 0.08, 0.68, 0.82])
    side = fig.add_axes([0.76, 0.08, 0.21, 0.82])
    side.axis("off")

    ax.imshow(shade, cmap="gray", extent=extent, origin="upper", alpha=0.85)
    ax.imshow(data, cmap="terrain", extent=extent, origin="upper", alpha=0.25)

    try:
        # For geographic DEM, imshow extent maps rows/cols to lon/lat.
        x = np.linspace(bounds.left, bounds.right, data.shape[1])
        y = np.linspace(bounds.top, bounds.bottom, data.shape[0])
        X, Y = np.meshgrid(x, y)
        cs = ax.contour(X, Y, data, levels=levels, linewidths=0.35, colors="black", alpha=0.55)
        if len(levels) <= 80:
            ax.clabel(cs, inline=True, fontsize=6, fmt=lambda v: f"{v:.0f}")
    except Exception:
        pass

    basin_coords = _kml_polygon_coords(basin_kml_bytes)
    _plot_polygon(ax, basin_coords)

    axis_coords = _line_coords_from_session(axis_line)
    if axis_coords:
        xs, ys = zip(*axis_coords)
        ax.plot(xs, ys, color="#0057d8", linewidth=2.8, zorder=8, label="Eje de cauce")

    if control_point:
        lon = float(control_point.get("lon"))
        lat = float(control_point.get("lat"))
        ax.scatter([lon], [lat], s=70, c="red", edgecolors="white", linewidths=1.2, zorder=9)
        ax.text(lon, lat, "  Punto de control", fontsize=8, weight="bold", color="red", zorder=10)

    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_xlabel("Longitud / X")
    ax.set_ylabel("Latitud / Y")
    ax.grid(True, alpha=0.25, linewidth=0.4)

    # North arrow
    ax.annotate("N", xy=(0.06, 0.88), xytext=(0.06, 0.74), xycoords="axes fraction",
                arrowprops=dict(facecolor="black", width=4, headwidth=12),
                ha="center", va="center", fontsize=12, weight="bold")

    side.text(0.0, 0.98, "RESUMEN MORFOMÉTRICO", fontsize=12, weight="bold", color="#003b73")
    side.plot([0, 1], [0.955, 0.955], color="#003b73", linewidth=2)

    if metrics:
        rows = [
            ("Área", "area_km2", "km²"),
            ("Perímetro", "perimetro_km", "km"),
            ("Kc compacidad", "coef_compacidad_kc", ""),
            ("Factor forma", "factor_forma", ""),
            ("Rel. elongación", "relacion_elongacion", ""),
            ("Ancho medio", "ancho_medio_km", "km"),
            ("Largo caract.", "bbox_largo_km", "km"),
            ("Ajuste salida", "distancia_ajuste_m", "m"),
        ]
        y = 0.90
        for label, key, unit in rows:
            val = metrics.get(key)
            if val is None:
                continue
            try:
                txt = f"{float(val):,.3f}".replace(",", "X").replace(".", ",").replace("X", ".")
            except Exception:
                txt = str(val)
            side.text(0.0, y, label, fontsize=9)
            side.text(0.66, y, f"{txt} {unit}".strip(), fontsize=9, weight="bold", color="#0057d8")
            y -= 0.055

    side.text(0.0, 0.37, "LEYENDA", fontsize=12, weight="bold", color="#003b73")
    side.plot([0, 1], [0.345, 0.345], color="#003b73", linewidth=2)
    legend_items = [
        ("Curvas de nivel", "black"),
        ("Límite de cuenca", "#0057d8"),
        ("Eje de cauce", "#0057d8"),
        ("Punto de control", "red"),
    ]
    y = 0.30
    for label, color in legend_items:
        side.plot([0.02, 0.16], [y, y], color=color, linewidth=2)
        side.text(0.20, y - 0.01, label, fontsize=9)
        y -= 0.055

    side.text(0.0, 0.04, "Salida cartográfica preliminar.\nRequiere revisión técnica para diseño final.", fontsize=8, color="dimgray")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()
