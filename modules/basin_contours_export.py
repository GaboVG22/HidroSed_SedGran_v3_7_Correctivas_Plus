
from __future__ import annotations

from dataclasses import dataclass
import io
import re
import zipfile
import xml.etree.ElementTree as ET

from shapely.geometry import Polygon, LineString, MultiLineString
from shapely.ops import unary_union


@dataclass
class BasinContoursOutput:
    kmz_bytes: bytes
    kml_bytes: bytes
    preview_png: bytes | None
    metadata: dict


def _strip_ns(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _parse_coord_text(text: str):
    coords = []
    for token in re.split(r"\s+", (text or "").strip()):
        if not token:
            continue
        vals = token.split(",")
        if len(vals) >= 2:
            try:
                lon = float(vals[0])
                lat = float(vals[1])
                z = float(vals[2]) if len(vals) >= 3 and vals[2] != "" else 0.0
                coords.append((lon, lat, z))
            except Exception:
                pass
    return coords


def _first_polygon_from_kml(kml_bytes_or_text):
    if isinstance(kml_bytes_or_text, bytes):
        txt = kml_bytes_or_text.decode("utf-8", errors="ignore")
    else:
        txt = str(kml_bytes_or_text)

    root = ET.fromstring(txt.encode("utf-8"))
    for elem in root.iter():
        if _strip_ns(elem.tag) == "Polygon":
            for child in elem.iter():
                if _strip_ns(child.tag) == "coordinates":
                    coords3 = _parse_coord_text(child.text or "")
                    coords2 = [(x, y) for x, y, *_ in coords3]
                    if len(coords2) >= 4:
                        poly = Polygon(coords2)
                        if not poly.is_valid:
                            poly = poly.buffer(0)
                        return poly
    raise ValueError("No se encontró polígono de cuenca válido en KML.")


def _line_records_from_kml(kml_bytes_or_text):
    if isinstance(kml_bytes_or_text, bytes):
        txt = kml_bytes_or_text.decode("utf-8", errors="ignore")
    else:
        txt = str(kml_bytes_or_text)

    records = []
    root = ET.fromstring(txt.encode("utf-8"))

    for pm in root.iter():
        if _strip_ns(pm.tag) != "Placemark":
            continue
        name = "Curva"
        level = None
        for ch in pm.iter():
            if _strip_ns(ch.tag) == "name" and ch.text:
                name = ch.text.strip()
                m = re.search(r"(-?\d+(?:\.\d+)?)", name)
                if m:
                    try:
                        level = float(m.group(1))
                    except Exception:
                        pass
                break

        for ls in pm.iter():
            if _strip_ns(ls.tag) != "LineString":
                continue
            for coord_elem in ls.iter():
                if _strip_ns(coord_elem.tag) == "coordinates":
                    coords3 = _parse_coord_text(coord_elem.text or "")
                    coords2 = [(x, y) for x, y, *_ in coords3]
                    if len(coords2) >= 2:
                        records.append({"name": name, "level": level, "coords": coords2})
                    break
    return records


def _clip_records_to_polygon(records, polygon: Polygon):
    clipped = []
    prepared = polygon.buffer(0)
    for rec in records:
        line = LineString(rec["coords"])
        if line.is_empty:
            continue
        inter = line.intersection(prepared)
        if inter.is_empty:
            continue
        geoms = []
        if isinstance(inter, LineString):
            geoms = [inter]
        elif isinstance(inter, MultiLineString):
            geoms = list(inter.geoms)
        else:
            try:
                geoms = [g for g in inter.geoms if isinstance(g, LineString)]
            except Exception:
                geoms = []
        for g in geoms:
            coords = list(g.coords)
            if len(coords) >= 2:
                clipped.append({
                    "name": rec["name"],
                    "level": rec.get("level"),
                    "coords": [(float(x), float(y)) for x, y in coords],
                })
    return clipped


def _kml_for_basin_and_contours(polygon: Polygon, contour_records):
    exterior = list(polygon.exterior.coords)
    coord_poly = " ".join(f"{x:.8f},{y:.8f},0" for x, y in exterior)

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>\n',
        '<kml xmlns="http://www.opengis.net/kml/2.2">\n<Document>\n',
        '<name>Cuenca + curvas de nivel HidroSed</name>\n',
        '<Style id="basin"><LineStyle><color>ff0066ff</color><width>2.2</width></LineStyle><PolyStyle><color>330066ff</color></PolyStyle></Style>\n',
        '<Style id="contour"><LineStyle><color>ff222222</color><width>1</width></LineStyle></Style>\n',
        '<Style id="contour_index"><LineStyle><color>ff000000</color><width>2</width></LineStyle></Style>\n',
        '<Placemark><name>Cuenca delimitada</name><styleUrl>#basin</styleUrl><Polygon><outerBoundaryIs><LinearRing><coordinates>',
        coord_poly,
        '</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark>\n',
        '<Folder><name>Curvas de nivel recortadas a cuenca</name>\n',
    ]

    for i, rec in enumerate(contour_records):
        level = rec.get("level")
        style = "contour"
        if level is not None:
            # Curva índice simple cada 50 m o 100 m.
            try:
                if abs(level / 50.0 - round(level / 50.0)) < 1e-6:
                    style = "contour_index"
            except Exception:
                pass
        name = rec.get("name") or f"Curva {i+1}"
        coord_txt = " ".join(f"{x:.8f},{y:.8f},0" for x, y in rec["coords"])
        parts.append(
            f'<Placemark><name>{name}</name><styleUrl>#{style}</styleUrl>'
            f'<LineString><tessellate>1</tessellate><coordinates>{coord_txt}</coordinates></LineString>'
            '</Placemark>\n'
        )

    parts.append("</Folder>\n</Document>\n</kml>\n")
    return "".join(parts).encode("utf-8")


def _preview_png(polygon: Polygon, contour_records):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        for rec in contour_records:
            xs = [p[0] for p in rec["coords"]]
            ys = [p[1] for p in rec["coords"]]
            ax.plot(xs, ys, linewidth=0.35)

        x, y = polygon.exterior.xy
        ax.plot(x, y, linewidth=2.0)

        ax.set_title("Cuenca delimitada + curvas de nivel recortadas")
        ax.set_xlabel("X (EPSG:4326)")
        ax.set_ylabel("Y (EPSG:4326)")
        ax.set_aspect("equal", adjustable="box")
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()
    except Exception:
        return None


def build_basin_contours_kmz(basin_kml, contours_kml, clip_to_basin=True) -> BasinContoursOutput:
    polygon = _first_polygon_from_kml(basin_kml)
    records = _line_records_from_kml(contours_kml)

    if clip_to_basin:
        records_use = _clip_records_to_polygon(records, polygon)
        mode = "curvas_recortadas_a_cuenca"
    else:
        records_use = records
        mode = "curvas_completas_sobre_cuenca"

    if not records_use:
        raise RuntimeError("No se encontraron curvas dentro de la cuenca. Revise que las curvas y la cuenca correspondan al mismo DEM/CRS.")

    kml = _kml_for_basin_and_contours(polygon, records_use)
    kmz_buf = io.BytesIO()
    with zipfile.ZipFile(kmz_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("doc.kml", kml)

    png = _preview_png(polygon, records_use)

    return BasinContoursOutput(
        kmz_bytes=kmz_buf.getvalue(),
        kml_bytes=kml,
        preview_png=png,
        metadata={
            "modo": mode,
            "curvas_entrada": len(records),
            "curvas_exportadas": len(records_use),
            "area_bbox_cuenca_aprox": polygon.bounds,
        },
    )
