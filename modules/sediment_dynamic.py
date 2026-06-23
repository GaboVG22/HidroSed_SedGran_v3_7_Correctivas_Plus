from __future__ import annotations

import numpy as np
import pandas as pd


def classify_sediment_zones(sed_df: pd.DataFrame) -> pd.DataFrame:
    if sed_df is None or sed_df.empty:
        return pd.DataFrame()
    df = sed_df.copy()
    theta = pd.to_numeric(df.get("Shields", np.nan), errors="coerce")
    qs = pd.to_numeric(df.get("Qs_total_m3_s", np.nan), errors="coerce")
    scour = pd.to_numeric(df.get("socavacion_general_m", np.nan), errors="coerce")
    df["tendencia_sedimentaria"] = np.select(
        [theta > 0.10, theta > 0.047, theta <= 0.047],
        ["erosion_alta", "transporte_activo", "estable_deposicion_probable"],
        default="sin_dato",
    )
    df["zona_hidrosed"] = np.select(
        [scour > 1.0, scour > 0.25, qs <= 0],
        ["socavacion_relevante", "socavacion_moderada", "depositacion_equilibrio"],
        default="transporte_sedimentos",
    )
    df["indice_riesgo_sedimento"] = np.nan_to_num(theta, nan=0) * 50 + np.nan_to_num(scour, nan=0) * 10
    return df


def summarize_zones(zone_df: pd.DataFrame) -> pd.DataFrame:
    if zone_df is None or zone_df.empty:
        return pd.DataFrame()
    return zone_df.groupby(["T_anios", "zona_hidrosed"]).agg(
        n_secciones=("section_id", "count"),
        pk_min=("pk_m", "min"),
        pk_max=("pk_m", "max"),
        socavacion_max_m=("socavacion_general_m", "max"),
        transporte_max_m3s=("Qs_total_m3_s", "max"),
        shields_max=("Shields", "max"),
    ).reset_index()
