
from __future__ import annotations

import io
import math
import zipfile
from pathlib import Path
import pandas as pd
import numpy as np


APP_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = APP_ROOT / "data" / "preloaded" / "catalogo_dga_sedimentos.csv"
CATALOG_JSON_PATH = APP_ROOT / "data" / "preloaded" / "catalogo_dga_sedimentos.json"


def load_catalog() -> pd.DataFrame:
    if CATALOG_PATH.exists():
        return pd.read_csv(CATALOG_PATH)
    return pd.DataFrame()


def read_preloaded_dataset(dataset: str, nrows: int | None = None, usecols=None) -> pd.DataFrame:
    cat = load_catalog()
    if cat.empty or dataset not in set(cat["dataset"]):
        return pd.DataFrame()
    row = cat[cat["dataset"] == dataset].iloc[0]
    path = APP_ROOT / str(row["archivo_app"])
    if not path.exists():
        return pd.DataFrame()
    with zipfile.ZipFile(path) as z:
        name = z.namelist()[0]
        with z.open(name) as f:
            return pd.read_csv(f, nrows=nrows, usecols=usecols)


def station_inventory(dataset: str, max_rows: int | None = None) -> pd.DataFrame:
    cols = ["CODIGO ESTACION","NOMBRE ESTACION","LAT","LONG","UTM_ESTE","UTM_NORTE","ALTITUD"]
    df = read_preloaded_dataset(dataset, nrows=max_rows, usecols=lambda c: c in cols)
    if df.empty or "CODIGO ESTACION" not in df.columns:
        return pd.DataFrame()
    agg = df.groupby("CODIGO ESTACION").agg(
        nombre=("NOMBRE ESTACION", "first"),
        registros=("CODIGO ESTACION", "count"),
        lat=("LAT", "first"),
        lon=("LONG", "first"),
        utm_este=("UTM_ESTE", "first"),
        utm_norte=("UTM_NORTE", "first"),
        altitud=("ALTITUD", "first"),
    ).reset_index()
    return agg


def _parse_dms_chile(v):
    """Parses DGA compact strings like 0291445 or 0712739 as decimal degrees."""
    try:
        s = str(v).strip().replace(" ", "")
        if not s or s.lower()=="nan":
            return np.nan
        # 0291445 => 29°14'45"
        s = s.zfill(7)
        deg = float(s[:3])
        minutes = float(s[3:5])
        seconds = float(s[5:7])
        return -(deg + minutes/60 + seconds/3600)
    except Exception:
        return np.nan


def rank_stations_by_point(dataset: str, lat: float, lon: float, max_rows: int | None = 250000) -> pd.DataFrame:
    inv = station_inventory(dataset, max_rows=max_rows)
    if inv.empty:
        return inv
    lat_dec = inv["lat"].map(_parse_dms_chile)
    lon_dec = inv["lon"].map(_parse_dms_chile)
    # If LONG already negative/decimal, keep it
    lat_num = pd.to_numeric(inv["lat"], errors="coerce")
    lon_num = pd.to_numeric(inv["lon"], errors="coerce")
    lat_dec = lat_dec.fillna(lat_num)
    lon_dec = lon_dec.fillna(lon_num)
    inv["lat_decimal"] = lat_dec
    inv["lon_decimal"] = lon_dec
    R = 6371.0
    la1 = np.radians(float(lat)); lo1 = np.radians(float(lon))
    la2 = np.radians(inv["lat_decimal"].astype(float)); lo2 = np.radians(inv["lon_decimal"].astype(float))
    dlat = la2-la1; dlon = lo2-lo1
    a = np.sin(dlat/2)**2 + np.cos(la1)*np.cos(la2)*np.sin(dlon/2)**2
    inv["distancia_km"] = 2*R*np.arcsin(np.sqrt(a))
    inv["score_ranking"] = inv["distancia_km"].rank(pct=True) + (1 - inv["registros"].rank(pct=True))
    return inv.sort_values(["score_ranking","distancia_km"], ascending=[True,True]).reset_index(drop=True)


def validation_station_isoyeta(p24_station: float, p24_isoyeta: float) -> dict:
    if p24_station is None or p24_isoyeta is None or p24_station <= 0 or p24_isoyeta <= 0:
        return {"estado": "sin_dato", "diferencia_pct": np.nan, "criterio": "No hay datos suficientes estación-isoyeta."}
    diff = abs(p24_station - p24_isoyeta)/p24_isoyeta*100
    if diff <= 20:
        estado = "verde"
        crit = "Consistente: diferencia ≤ 20%."
    elif diff <= 35:
        estado = "amarillo"
        crit = "Revisar: diferencia entre 20% y 35%."
    else:
        estado = "rojo"
        crit = "Inconsistente: diferencia > 35%; adoptar criterio conservador y advertencia."
    return {"estado": estado, "diferencia_pct": float(diff), "criterio": crit}
