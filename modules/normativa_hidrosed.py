
from __future__ import annotations

import pandas as pd


def normative_hydraulic_hydrology_check(context: dict) -> pd.DataFrame:
    def ok(key):
        return bool(context.get(key))

    rows = [
        {
            "bloque": "Isoyetas / precipitación",
            "criterio": "P24 representativa debe provenir de isoyetas, estación o fuente trazable; valor manual solo como respaldo.",
            "dato_requerido": "P24, fuente, método de asignación",
            "estado": "OK" if ok("p24_trazable") else "Preliminar",
            "observacion": context.get("p24_observacion", "Sin P24 trazable desde isoyetas/estación."),
        },
        {
            "bloque": "IDF / duración",
            "criterio": "La intensidad debe asociarse a duración compatible con tiempo de concentración y período de retorno.",
            "dato_requerido": "Tc, curva IDF o transformación P24→I",
            "estado": "OK" if ok("idf_tc") else "Revisar",
            "observacion": "Usar IDF oficial si existe; si se transforma desde P24, dejar como estimación.",
        },
        {
            "bloque": "Manual de Carreteras",
            "criterio": "Comparar métodos hidrológicos y justificar adopción de caudal; no depender de un único método sin contraste.",
            "dato_requerido": "Racional, racional modificado, DGA/regional, transferencia si aplica",
            "estado": "OK" if ok("metodos_comparados") else "Revisar",
            "observacion": "El caudal adoptado debe quedar con criterio conservador o técnico documentado.",
        },
        {
            "bloque": "Guías DGA modificación de cauces",
            "criterio": "Topografía, planta/perfil, secciones, hidrología, hidráulica, rugosidad y condiciones de borde deben ser trazables.",
            "dato_requerido": "Cuenca, eje, secciones, DEM/curvas, Q(T), n, borde aguas abajo",
            "estado": "OK" if ok("geometria_hidraulica") else "Revisar",
            "observacion": "Si hay secciones sintéticas o DEM de baja resolución, marcar nivel preliminar.",
        },
        {
            "bloque": "HEC-RAS / hidráulica 1D",
            "criterio": "Las secciones deben actuar como sistema conectado y ordenado por PK; revisar régimen y pérdidas.",
            "dato_requerido": "Secciones conectadas, pendiente, n, Q, condición de borde",
            "estado": "OK" if ok("hecras_like") else "Preliminar",
            "observacion": "La app no reemplaza HEC-RAS oficial calibrado; sirve como cálculo tipo HEC-RAS preliminar.",
        },
        {
            "bloque": "Sedimentos y socavación",
            "criterio": "La socavación debe distinguir movilidad, transporte, depositación y tipo de material; preferir granulometría real.",
            "dato_requerido": "D50, D84/D90, Shields, esfuerzo cortante, perfil hidráulico",
            "estado": "OK" if ok("granulometria_real") else "Preliminar",
            "observacion": "Con granulometría por defecto los resultados son preliminares.",
        },
    ]
    df = pd.DataFrame(rows)
    score = 0
    for st in df["estado"]:
        score += 1.0 if st == "OK" else (0.55 if st == "Preliminar" else 0.35)
    df["puntaje_bloque"] = df["estado"].map({"OK": 1.0, "Preliminar": 0.55, "Revisar": 0.35}).fillna(0.35)
    return df


def normative_confidence_score(check_df: pd.DataFrame) -> float:
    if check_df is None or check_df.empty or "puntaje_bloque" not in check_df.columns:
        return 0.0
    return float(10.0 * check_df["puntaje_bloque"].mean())
