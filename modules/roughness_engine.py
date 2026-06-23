from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, List

import numpy as np
import pandas as pd


ROUGHNESS_TABLE = pd.DataFrame([
    {"categoria": "tierra_recto_limpio", "descripcion": "Canal/cauce de tierra recto y limpio", "n_min": 0.018, "n_tipico": 0.025, "n_max": 0.033},
    {"categoria": "arena_fina", "descripcion": "Lecho arenoso fino a medio", "n_min": 0.020, "n_tipico": 0.030, "n_max": 0.040},
    {"categoria": "arena_grava", "descripcion": "Arena gruesa con grava fina", "n_min": 0.028, "n_tipico": 0.038, "n_max": 0.050},
    {"categoria": "grava_media", "descripcion": "Cauce aluvial con grava media", "n_min": 0.032, "n_tipico": 0.045, "n_max": 0.060},
    {"categoria": "grava_bolones", "descripcion": "Grava gruesa con bolones menores", "n_min": 0.040, "n_tipico": 0.060, "n_max": 0.085},
    {"categoria": "cauce_montana", "descripcion": "Cauce de montaña, irregular, rocoso", "n_min": 0.050, "n_tipico": 0.075, "n_max": 0.120},
    {"categoria": "vegetacion_baja", "descripcion": "Márgenes con vegetación baja", "n_min": 0.035, "n_tipico": 0.055, "n_max": 0.080},
    {"categoria": "vegetacion_densa", "descripcion": "Márgenes con vegetación densa", "n_min": 0.070, "n_tipico": 0.110, "n_max": 0.160},
])

COWAN_FACTORS = {
    "n0_material": {
        "tierra": 0.020,
        "roca_cortada": 0.025,
        "grava_fina": 0.024,
        "grava_media": 0.028,
        "grava_gruesa_bolones": 0.032,
    },
    "n1_irregularidad": {
        "suave": 0.000,
        "menor": 0.005,
        "moderada": 0.010,
        "severa": 0.020,
    },
    "n2_variacion_seccion": {
        "gradual": 0.000,
        "ocasional": 0.005,
        "frecuente": 0.010,
        "muy_frecuente": 0.015,
    },
    "n3_obstrucciones": {
        "despreciable": 0.000,
        "menor": 0.010,
        "apreciable": 0.025,
        "severa": 0.050,
    },
    "n4_vegetacion": {
        "nula": 0.000,
        "baja": 0.005,
        "media": 0.010,
        "alta": 0.025,
        "muy_alta": 0.050,
    },
    "m_sinuosidad": {
        "baja": 1.00,
        "media": 1.15,
        "alta": 1.30,
    },
}


def clamp_n(n: float) -> float:
    if not np.isfinite(n):
        return np.nan
    return float(min(max(n, 0.010), 0.200))


def cowan_n(
    material: str = "grava_media",
    irregularidad: str = "moderada",
    variacion_seccion: str = "ocasional",
    obstrucciones: str = "menor",
    vegetacion: str = "baja",
    sinuosidad: str = "media",
) -> dict:
    n0 = COWAN_FACTORS["n0_material"].get(material, 0.028)
    n1 = COWAN_FACTORS["n1_irregularidad"].get(irregularidad, 0.010)
    n2 = COWAN_FACTORS["n2_variacion_seccion"].get(variacion_seccion, 0.005)
    n3 = COWAN_FACTORS["n3_obstrucciones"].get(obstrucciones, 0.010)
    n4 = COWAN_FACTORS["n4_vegetacion"].get(vegetacion, 0.005)
    m = COWAN_FACTORS["m_sinuosidad"].get(sinuosidad, 1.15)
    n = (n0 + n1 + n2 + n3 + n4) * m
    return {
        "metodo": "Cowan",
        "n_manning": clamp_n(n),
        "n0_material": n0,
        "n1_irregularidad": n1,
        "n2_variacion_seccion": n2,
        "n3_obstrucciones": n3,
        "n4_vegetacion": n4,
        "m_sinuosidad": m,
        "observacion": "Rugosidad estimada por suma de componentes; debe verificarse en terreno si el cálculo es de diseño.",
    }


def strickler_n(d84_m: float | None = None, d50_m: float | None = None) -> dict:
    """Estimación granular n ≈ d84^(1/6)/26; alternativa con d50 si falta D84.

    Sirve como estimación preliminar para lechos granulares, no para márgenes vegetados.
    """
    d = None
    fuente = None
    if d84_m and np.isfinite(d84_m) and d84_m > 0:
        d = d84_m
        fuente = "D84"
    elif d50_m and np.isfinite(d50_m) and d50_m > 0:
        d = 2.2 * d50_m
        fuente = "D50 convertido a D84 aprox."
    if d is None:
        return {"metodo": "Strickler granular", "n_manning": np.nan, "fuente": "sin granulometría", "observacion": "No se pudo estimar n granular."}
    n = (float(d) ** (1.0 / 6.0)) / 26.0
    return {
        "metodo": "Strickler granular",
        "n_manning": clamp_n(n),
        "d_referencia_m": float(d),
        "fuente": fuente,
        "observacion": "Estimación para cauce principal granular; no representa vegetación ni obstrucciones.",
    }


def table_n(category: str) -> dict:
    row = ROUGHNESS_TABLE[ROUGHNESS_TABLE["categoria"] == category]
    if row.empty:
        row = ROUGHNESS_TABLE[ROUGHNESS_TABLE["categoria"] == "grava_media"]
    r = row.iloc[0].to_dict()
    return {"metodo": "Tabla Manning", "n_manning": float(r["n_tipico"]), **r}


def compose_roughness_manual(n_left: float, n_channel: float, n_right: float, source: str = "manual") -> pd.DataFrame:
    rows = [
        {"zona": "margen_izquierda", "n_manning": clamp_n(n_left), "fuente": source},
        {"zona": "cauce_principal", "n_manning": clamp_n(n_channel), "fuente": source},
        {"zona": "margen_derecha", "n_manning": clamp_n(n_right), "fuente": source},
    ]
    return pd.DataFrame(rows)


def suggested_roughness(
    category: str = "grava_media",
    d50_m: float | None = None,
    d84_m: float | None = None,
    material: str = "grava_media",
    irregularidad: str = "moderada",
    vegetacion: str = "baja",
    sinuosidad: str = "media",
) -> pd.DataFrame:
    candidates = []
    candidates.append(table_n(category))
    candidates.append(cowan_n(material=material, irregularidad=irregularidad, vegetacion=vegetacion, sinuosidad=sinuosidad))
    candidates.append(strickler_n(d84_m=d84_m, d50_m=d50_m))
    df = pd.DataFrame(candidates)
    vals = pd.to_numeric(df["n_manning"], errors="coerce").dropna()
    adopted = float(vals.median()) if len(vals) else float(table_n(category)["n_manning"])
    df["n_adoptado_recomendado"] = adopted
    df["confianza"] = np.where(df["n_manning"].notna(), 8.8, 6.0)
    return df


def section_roughness_by_pk(sections_df: pd.DataFrame, roughness_value: float) -> pd.DataFrame:
    if sections_df is None or len(sections_df) == 0:
        return pd.DataFrame()
    out = sections_df[["section_id", "pk_m"]].copy()
    out["n_manning"] = clamp_n(roughness_value)
    out["fuente_n"] = "valor uniforme adoptado"
    return out


def roughness_confidence(mode: str, has_granulometry: bool, has_field_calibration: bool, zones: int = 1) -> dict:
    score = 6.5
    notes = []
    if mode == "manual":
        score += 0.8; notes.append("n ingresado manualmente")
    if mode == "tabla":
        score += 0.9; notes.append("n estimado por tabla")
    if mode == "cowan":
        score += 1.5; notes.append("n estimado por Cowan")
    if has_granulometry:
        score += 1.0; notes.append("granulometría disponible")
    if has_field_calibration:
        score += 1.2; notes.append("calibración con nivel/caudal observado")
    if zones >= 3:
        score += 0.5; notes.append("rugosidad diferenciada por zonas")
    score = min(score, 9.5)
    return {"confianza_rugosidad": round(score, 2), "observaciones": "; ".join(notes) or "rugosidad preliminar"}
