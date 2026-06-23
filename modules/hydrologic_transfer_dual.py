from __future__ import annotations

import math
import numpy as np
import pandas as pd


def transfer_flow_area_altitude_distance(
    q_station: float,
    area_target_km2: float,
    area_station_km2: float,
    alt_target_m: float | None = None,
    alt_station_m: float | None = None,
    distance_km: float | None = None,
    exponent_b: float = 0.75,
) -> dict:
    if area_station_km2 is None or area_station_km2 <= 0 or area_target_km2 <= 0:
        return {"Q_transferido_m3s": np.nan, "valido": False, "motivo": "área estación o punto no válida"}
    q = float(q_station) * (float(area_target_km2) / float(area_station_km2)) ** float(exponent_b)
    f_alt = 1.0
    if alt_target_m is not None and alt_station_m is not None and np.isfinite(alt_target_m) and np.isfinite(alt_station_m):
        dz = float(alt_target_m) - float(alt_station_m)
        f_alt = min(max(1.0 + dz / 10000.0, 0.75), 1.25)
        q *= f_alt
    f_dist = 1.0
    if distance_km is not None and np.isfinite(distance_km):
        f_dist = max(0.70, math.exp(-float(distance_km)/250.0))
        # no reduce q adopted; this is a confidence penalty, not a hydrologic coefficient
    confidence = 9.0
    ratio = float(area_target_km2) / float(area_station_km2)
    if ratio < 0.25 or ratio > 4.0: confidence -= 1.0
    if distance_km and distance_km > 80: confidence -= 0.8
    if abs(f_alt-1.0)>0.15: confidence -= 0.5
    return {
        "Q_transferido_m3s": float(q),
        "factor_area": ratio ** float(exponent_b),
        "factor_altitud": f_alt,
        "factor_distancia_confianza": f_dist,
        "exponente_b": exponent_b,
        "confianza_transferencia": round(max(confidence, 5.0),2),
        "valido": confidence >= 7.5,
        "motivo": "transferencia dual área-altitud-distancia",
    }


def rank_hydrometric_stations(stations: pd.DataFrame, target_lat: float, target_lon: float, target_alt_m: float | None = None) -> pd.DataFrame:
    if stations is None or len(stations)==0:
        return pd.DataFrame()
    df = stations.copy()
    def hav(row):
        lat1, lon1, lat2, lon2 = map(math.radians, [target_lat, target_lon, row.get('lat', np.nan), row.get('lon', np.nan)])
        if not np.isfinite(lat2) or not np.isfinite(lon2): return np.nan
        dlat = lat2-lat1; dlon = lon2-lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
        return 6371.0*2*math.atan2(math.sqrt(a), math.sqrt(1-a))
    df['distancia_km'] = df.apply(hav, axis=1)
    if 'n_anios' not in df.columns: df['n_anios'] = df.get('anios', 10)
    if 'continuidad_pct' not in df.columns: df['continuidad_pct'] = 70.0
    alt_pen = 0
    if target_alt_m is not None and 'altitud_m' in df.columns:
        alt_pen = np.minimum(abs(pd.to_numeric(df['altitud_m'], errors='coerce')-float(target_alt_m))/20.0, 20)
    df['score_estacion'] = 100 - df['distancia_km'].fillna(999)/2 + np.minimum(df['n_anios'].fillna(0), 40) + df['continuidad_pct'].fillna(0)/5 - alt_pen
    df = df.sort_values('score_estacion', ascending=False).reset_index(drop=True)
    df['rol'] = ['principal' if i==0 else 'secundaria' if i<4 else 'control' for i in range(len(df))]
    return df
