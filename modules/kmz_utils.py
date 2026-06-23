
from __future__ import annotations

import io
import re
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shapely.geometry import LineString


@dataclass
class ControlPoint:
    lat: float
    lon: float
    name: str = "Punto de control"


@dataclass
class KMLLine:
    name: str
    coords: list[tuple[float, float, float | None]]


def _decode(raw: bytes) -> str:
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", errors="ignore")


def read_kml(file_or_bytes: Any) -> str:
    if isinstance(file_or_bytes, (str, Path)):
        data = Path(file_or_bytes).read_bytes()
        name = str(file_or_bytes).lower()
    elif isinstance(file_or_bytes, bytes):
        data = file_or_bytes
        name = ""
    else:
        try:
            file_or_bytes.seek(0)
        except Exception:
            pass
        data = file_or_bytes.read()
        try:
            file_or_bytes.seek(0)
        except Exception:
            pass
        name = getattr(file_or_bytes, "name", "").lower()

    if name.endswith(".kmz") or data[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            kmls = [n for n in zf.namelist() if n.lower().endswith(".kml")]
            if not kmls:
                raise ValueError("El KMZ no contiene archivo KML interno.")
            best = max(kmls, key=lambda n: zf.read(n).lower().count(b"<coordinates"))
            return _decode(zf.read(best))
    return _decode(data)


def _parse_coord_tuple(token: str) -> tuple[float, float, float | None]:
    parts = token.strip().split(",")
    if len(parts) < 2:
        raise ValueError("Coordenada KML inválida.")
    lon = float(parts[0])
    lat = float(parts[1])
    z = float(parts[2]) if len(parts) >= 3 and parts[2] != "" else None
    return lon, lat, z


def parse_first_point(kml_text: str) -> ControlPoint:
    try:
        root = ET.fromstring(kml_text.encode("utf-8"))
        placemarks = root.findall(".//{*}Placemark")
        for pm in placemarks:
            point = pm.find(".//{*}Point")
            if point is None:
                continue
            coords = point.find(".//{*}coordinates")
            if coords is None or not coords.text:
                continue
            lon, lat, _z = _parse_coord_tuple(coords.text.strip().split()[0])
            name_el = pm.find(".//{*}name")
            name = name_el.text.strip() if name_el is not None and name_el.text else "Punto de control"
            return ControlPoint(lat=lat, lon=lon, name=name)
    except ET.ParseError:
        pass

    m = re.search(r"<Point[^>]*>.*?<coordinates[^>]*>(.*?)</coordinates>.*?</Point>", kml_text, re.I | re.S)
    if not m:
        raise ValueError("No se encontró un punto de control válido en el KMZ/KML.")
    lon, lat, _z = _parse_coord_tuple(m.group(1).strip().split()[0])
    return ControlPoint(lat=lat, lon=lon, name="Punto de control")


def parse_lines(kml_text: str) -> list[KMLLine]:
    lines: list[KMLLine] = []
    try:
        root = ET.fromstring(kml_text.encode("utf-8"))
        for pm in root.findall(".//{*}Placemark"):
            name_el = pm.find(".//{*}name")
            name = name_el.text.strip() if name_el is not None and name_el.text else "Eje"
            for ls in pm.findall(".//{*}LineString"):
                coords_el = ls.find(".//{*}coordinates")
                if coords_el is None or not coords_el.text:
                    continue
                coords = [_parse_coord_tuple(tok) for tok in coords_el.text.strip().split()]
                if len(coords) >= 2:
                    lines.append(KMLLine(name=name, coords=coords))
    except ET.ParseError:
        for m in re.finditer(r"<LineString[^>]*>.*?<coordinates[^>]*>(.*?)</coordinates>.*?</LineString>", kml_text, re.I | re.S):
            coords = [_parse_coord_tuple(tok) for tok in m.group(1).strip().split()]
            if len(coords) >= 2:
                lines.append(KMLLine(name="Eje", coords=coords))
    return lines


def line_to_shapely_wgs84(line: KMLLine) -> LineString:
    return LineString([(lon, lat) for lon, lat, _z in line.coords])


def make_kmz_from_kml(kml_text: str, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("doc.kml", kml_text.encode("utf-8"))
    return out_path


def simple_line_kml(name: str, coords_lonlat: list[tuple[float, float]], color: str = "ff0000ff") -> str:
    coords = " ".join([f"{lon:.8f},{lat:.8f},0" for lon, lat in coords_lonlat])
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
<name>{name}</name>
<Style id="line"><LineStyle><color>{color}</color><width>3</width></LineStyle></Style>
<Placemark><name>{name}</name><styleUrl>#line</styleUrl>
<LineString><tessellate>1</tessellate><coordinates>{coords}</coordinates></LineString>
</Placemark>
</Document></kml>'''
