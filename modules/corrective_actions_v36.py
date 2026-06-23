
from __future__ import annotations

import io
import math
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd

from .data_catalog_engine import read_preloaded_dataset, rank_stations_by_point, validation_station_isoyeta
from .hydrology_normative_v35 import frequency_analysis_gumbel


APP_ROOT = Path(__file__).resolve().parents[1]
REGIONAL_COEFFS_PATH = APP_ROOT / "data" / "regional_coeffs_hidrologia_v36.csv"


def load_regional_coefficients() -> pd.DataFrame:
    if REGIONAL_COEFFS_PATH.exists():
        return pd.read_csv(REGIONAL_COEFFS_PATH)
    return pd.DataFrame()


def _to_datetime(s):
    return pd.to_datetime(s, dayfirst=True, errors="coerce")


def flow_annual_maxima_frequency(station_code: str | int, periods=None) -> dict:
    """Conecta caudales diarios reales con análisis de frecuencia Gumbel.

    Retorna máximos anuales, Q(T) y un reporte de suficiencia.
    """
    periods = periods or [2, 5, 10, 25, 50, 100, 200]
    code = str(station_code)
    try:
        df = read_preloaded_dataset(
            "caudal_diario",
            usecols=["CODIGO ESTACION", "NOMBRE ESTACION", "FECHA", "Caudal_diario"],
        )
    except Exception:
        df = pd.DataFrame()
    if df.empty:
        return {"ok": False, "annual_max": pd.DataFrame(), "frequency": pd.DataFrame(), "report": pd.DataFrame([{"estado":"sin_datos","detalle":"No se pudo leer caudal_diario."}])}

    df = df[df["CODIGO ESTACION"].astype(str) == code].copy()
    if df.empty:
        return {"ok": False, "annual_max": pd.DataFrame(), "frequency": pd.DataFrame(), "report": pd.DataFrame([{"estado":"sin_estacion","detalle":f"No hay datos para estación {code}."}])}
    df["fecha_dt"] = _to_datetime(df["FECHA"])
    df["caudal_m3s"] = pd.to_numeric(df["Caudal_diario"], errors="coerce")
    df = df.dropna(subset=["fecha_dt", "caudal_m3s"])
    df["anio"] = df["fecha_dt"].dt.year
    ann = df.groupby("anio").agg(
        Qmax_diario_m3s=("caudal_m3s","max"),
        registros=("caudal_m3s","count"),
        nombre=("NOMBRE ESTACION","first"),
    ).reset_index()
    ann = ann[ann["registros"] >= 30].copy()
    freq = frequency_analysis_gumbel(ann["Qmax_diario_m3s"], periods)
    if not freq.empty:
        freq["CODIGO ESTACION"] = code
        freq["NOMBRE ESTACION"] = ann["nombre"].iloc[0] if not ann.empty else ""
        freq["adoptable"] = len(ann) >= 10
        freq["observacion"] = "Frecuencia conectada a máximos anuales diarios reales; requiere área aportante para transferencia final."
    report = pd.DataFrame([{
        "CODIGO ESTACION": code,
        "nombre": ann["nombre"].iloc[0] if not ann.empty else "",
        "anios_validos": int(len(ann)),
        "primer_anio": int(ann["anio"].min()) if not ann.empty else None,
        "ultimo_anio": int(ann["anio"].max()) if not ann.empty else None,
        "estado": "OK" if len(ann) >= 10 else "REVISAR",
        "detalle": "Serie suficiente para Gumbel preliminar." if len(ann) >= 10 else "Serie corta; no adoptar sin revisión.",
    }])
    return {"ok": not freq.empty, "annual_max": ann, "frequency": freq, "report": report}


def _station_annual_pmax(code: str | int) -> pd.DataFrame:
    code = str(code)
    try:
        df = read_preloaded_dataset(
            "precipitacion_max_24h",
            usecols=["CODIGO ESTACION","NOMBRE ESTACION","ANIO","Precipitación_max_anual_24horas"]
        )
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df
    df = df[df["CODIGO ESTACION"].astype(str) == code].copy()
    df["ANIO"] = pd.to_numeric(df["ANIO"], errors="coerce").astype("Int64")
    df["Pmax24_mm"] = pd.to_numeric(df["Precipitación_max_anual_24horas"], errors="coerce")
    return df.dropna(subset=["ANIO","Pmax24_mm"])


def p24_station_frequency(station_code: str | int, periods=None) -> dict:
    periods = periods or [2,5,10,25,50,100,200]
    ann = _station_annual_pmax(station_code)
    if ann.empty:
        return {"ok": False, "frequency": pd.DataFrame(), "annual": ann, "report": pd.DataFrame([{"estado":"sin_datos"}])}
    # Reuse Gumbel function field name Q_m3s, then rename.
    freq = frequency_analysis_gumbel(ann["Pmax24_mm"], periods)
    if freq.empty:
        return {"ok": False, "frequency": pd.DataFrame(), "annual": ann, "report": pd.DataFrame([{"estado":"serie_corta", "n":len(ann)}])}
    freq = freq.rename(columns={"Q_m3s":"P24_mm"})
    freq["metodo"] = "Frecuencia_Gumbel_Pmax24_estacion"
    freq["CODIGO ESTACION"] = str(station_code)
    return {"ok": True, "frequency": freq, "annual": ann, "report": pd.DataFrame([{"estado":"OK", "n":len(ann), "P24_10_estacion_mm": float(freq[freq["T_anios"]==10]["P24_mm"].iloc[0]) if (freq["T_anios"]==10).any() else np.nan}])}


def fill_pluviometric_gaps(primary_code: str | int, secondary_code: str | int) -> dict:
    """Rellena lagunas anuales de Pmax24 por regresión si hay traslape suficiente, si no razón normal."""
    p = _station_annual_pmax(primary_code)[["ANIO","Pmax24_mm"]].rename(columns={"Pmax24_mm":"P_principal"})
    s = _station_annual_pmax(secondary_code)[["ANIO","Pmax24_mm"]].rename(columns={"Pmax24_mm":"P_secundaria"})
    if p.empty or s.empty:
        return {"ok": False, "filled": pd.DataFrame(), "report": pd.DataFrame([{"estado":"sin_datos"}])}
    m = p.merge(s, on="ANIO", how="outer").sort_values("ANIO")
    overlap = m.dropna(subset=["P_principal","P_secundaria"])
    method = "sin_relleno"
    r2 = np.nan
    if len(overlap) >= 8:
        x = overlap["P_secundaria"].to_numpy(float)
        y = overlap["P_principal"].to_numpy(float)
        A = np.vstack([x, np.ones_like(x)]).T
        a,b = np.linalg.lstsq(A, y, rcond=None)[0]
        yhat = a*x+b
        ss_res = float(((y-yhat)**2).sum())
        ss_tot = float(((y-y.mean())**2).sum())
        r2 = 1-ss_res/ss_tot if ss_tot>0 else np.nan
        if np.isfinite(r2) and r2 >= 0.60:
            mask = m["P_principal"].isna() & m["P_secundaria"].notna()
            m.loc[mask, "P_rellena"] = a*m.loc[mask, "P_secundaria"] + b
            method = "regresion_lineal"
        else:
            ratio = overlap["P_principal"].mean()/max(overlap["P_secundaria"].mean(),1e-9)
            mask = m["P_principal"].isna() & m["P_secundaria"].notna()
            m.loc[mask, "P_rellena"] = ratio*m.loc[mask, "P_secundaria"]
            method = "razon_normal_por_bajo_R2"
    elif len(overlap) >= 3:
        ratio = overlap["P_principal"].mean()/max(overlap["P_secundaria"].mean(),1e-9)
        mask = m["P_principal"].isna() & m["P_secundaria"].notna()
        m.loc[mask, "P_rellena"] = ratio*m.loc[mask, "P_secundaria"]
        method = "razon_normal"
    else:
        m["P_rellena"] = np.nan
    m["P_final"] = m["P_principal"].fillna(m.get("P_rellena"))
    filled_count = int(m["P_principal"].isna().sum() - m["P_final"].isna().sum())
    report = pd.DataFrame([{
        "principal": str(primary_code), "secundaria": str(secondary_code),
        "traslape_anios": int(len(overlap)), "metodo": method, "R2": r2,
        "lagunas_rellenadas": filled_count,
        "estado": "OK" if method in ["regresion_lineal","razon_normal","razon_normal_por_bajo_R2"] else "REVISAR",
        "advertencia": "Relleno preliminar; revisar homogeneidad y régimen antes de adoptar."
    }])
    return {"ok": method!="sin_relleno", "filled": m, "report": report}


def station_isoyeta_semiphore(station_code: str | int, p24_isoyeta_mm: float) -> dict:
    fr = p24_station_frequency(station_code, periods=[10])
    if not fr.get("ok") or fr["frequency"].empty:
        return {"ok": False, "validation": pd.DataFrame([{"estado":"sin_dato_estacion"}]), "frequency": pd.DataFrame()}
    p24_station = float(fr["frequency"]["P24_mm"].iloc[0])
    val = validation_station_isoyeta(p24_station, float(p24_isoyeta_mm))
    out = pd.DataFrame([{
        "CODIGO ESTACION": str(station_code),
        "P24_10_estacion_mm": p24_station,
        "P24_10_isoyeta_mm": float(p24_isoyeta_mm),
        "diferencia_pct": val["diferencia_pct"],
        "semaforo": val["estado"],
        "criterio": val["criterio"],
        "P24_adoptada_conservadora_mm": max(p24_station, float(p24_isoyeta_mm)),
    }])
    return {"ok": True, "validation": out, "frequency": fr["frequency"]}


def calibrate_manning_from_observed(profile_df: pd.DataFrame, observed_df: pd.DataFrame) -> dict:
    """Calibración simple: busca factor global de Manning que minimiza RMSE de cotas.

    Usa sensibilidad aproximadamente lineal entre n-20, base y n+20 cuando esté disponible.
    """
    if profile_df is None or profile_df.empty or observed_df is None or observed_df.empty:
        return {"ok": False, "calibration": pd.DataFrame(), "report": pd.DataFrame([{"estado":"sin_datos"}])}
    obs = observed_df.copy()
    # Normalize columns
    cols = {c.lower().strip():c for c in obs.columns}
    sid_col = cols.get("section_id") or cols.get("seccion") or cols.get("id")
    wse_col = cols.get("cota_observada_m") or cols.get("wse_observada_m") or cols.get("cota_agua_observada_m")
    T_col = cols.get("t_anios") or cols.get("tr") or cols.get("periodo")
    if not sid_col or not wse_col:
        return {"ok": False, "calibration": pd.DataFrame(), "report": pd.DataFrame([{"estado":"columnas_invalidas", "detalle":"Requiere section_id y cota_observada_m."}])}
    obs["section_id_str"] = obs[sid_col].astype(str)
    obs["cota_observada_m"] = pd.to_numeric(obs[wse_col], errors="coerce")
    prof = profile_df.copy()
    prof["section_id_str"] = prof["section_id"].astype(str)
    if T_col and "T_anios" in prof.columns:
        obs["T_anios_tmp"] = pd.to_numeric(obs[T_col], errors="coerce")
        m = prof.merge(obs[["section_id_str","T_anios_tmp","cota_observada_m"]], left_on=["section_id_str","T_anios"], right_on=["section_id_str","T_anios_tmp"], how="inner")
    else:
        m = prof.merge(obs[["section_id_str","cota_observada_m"]], on="section_id_str", how="inner")
    if m.empty:
        return {"ok": False, "calibration": pd.DataFrame(), "report": pd.DataFrame([{"estado":"sin_traslape"}])}
    factors = np.linspace(0.8,1.2,17)
    rows=[]
    for f in factors:
        # Approximate cota modelada as base + proportional to delta if sensitivity columns exist.
        base = pd.to_numeric(m.get("cota_agua_m"), errors="coerce")
        if f < 1 and "wse_n_menos20_m" in m.columns:
            pred = base + (pd.to_numeric(m["wse_n_menos20_m"], errors="coerce")-base)*((1-f)/0.2)
        elif f > 1 and "wse_n_mas20_m" in m.columns:
            pred = base + (pd.to_numeric(m["wse_n_mas20_m"], errors="coerce")-base)*((f-1)/0.2)
        else:
            pred = base
        err = pred - m["cota_observada_m"]
        rows.append({"factor_n": float(f), "RMSE_m": float(np.sqrt(np.nanmean(err**2))), "sesgo_m": float(np.nanmean(err)), "n_puntos": int(err.notna().sum())})
    cal = pd.DataFrame(rows).sort_values("RMSE_m")
    best = cal.iloc[0]
    report = pd.DataFrame([{"estado":"OK", "factor_n_recomendado": best["factor_n"], "RMSE_m": best["RMSE_m"], "n_observaciones": int(best["n_puntos"])}])
    return {"ok": True, "calibration": cal, "report": report}


def sediment_applicability_ranges(sediment_df: pd.DataFrame) -> pd.DataFrame:
    if sediment_df is None or sediment_df.empty:
        return pd.DataFrame()
    rows=[]
    for _, r in sediment_df.iterrows():
        D50 = float(r.get("D50_m", np.nan))
        theta = float(r.get("Shields", np.nan))
        Fr = float(r.get("Froude", np.nan)) if "Froude" in r else np.nan
        notes=[]
        if np.isfinite(D50):
            if D50 < 0.0002 or D50 > 0.1:
                notes.append("MPM fuera/rango débil por D50; revisar material.")
            if D50 <= 0.002:
                notes.append("Engelund-Hansen más aplicable a arena/transporte total.")
            else:
                notes.append("Engelund-Hansen referencial en grava; usar con cautela.")
        if np.isfinite(theta):
            if theta < 0.047:
                notes.append("Shields bajo crítico: lecho estable.")
            elif theta > 0.20:
                notes.append("Shields alto: transporte activo; revisar acorazamiento.")
        if np.isfinite(Fr) and Fr > 1:
            notes.append("Régimen supercrítico: revisar hipótesis y socavación.")
        rows.append({
            "section_id": r.get("section_id"),
            "pk_m": r.get("pk_m"),
            "T_anios": r.get("T_anios"),
            "D50_m": D50,
            "Shields": theta,
            "semaforo_sedimentos": "verde" if len(notes)<=1 else ("amarillo" if len(notes)<=2 else "rojo"),
            "observaciones_rango": "; ".join(notes) if notes else "OK",
        })
    return pd.DataFrame(rows)


def generate_calculation_memory_text(context: dict) -> str:
    lines = []
    lines.append("# Memoria de cálculo automática HidroSed")
    lines.append("")
    lines.append("## Datos y fuentes")
    for k,v in context.get("fuentes", {}).items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Parámetros adoptados")
    for k,v in context.get("parametros", {}).items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## QA y trazabilidad")
    for k,v in context.get("qa", {}).items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Dictamen")
    lines.append(context.get("dictamen", "Memoria generada automáticamente; revisar y complementar con antecedentes del proyecto."))
    return "\n".join(lines)


def unit_tests_v36() -> pd.DataFrame:
    tests=[]
    try:
        # hydrology module sanity
        from .hydrology_normative_v35 import run_normative_hydrology
        for area, regime in [(5,"pluvial"), (87,"pluvial"), (1200,"pluvial"), (800,"nivo-pluvial"), (15000,"pluvial")]:
            out=run_normative_hydrology(area, max(2, area**0.5), 0.02, 0.45, 80, 2.14, [2,10,100], regime, 250, area if "nivo" in regime else None)
            tests.append({"prueba":f"hidrologia_area_{area}_{regime}", "estado":"OK" if not out["caudales_adoptados"].empty else "FALLA", "detalle":out["cumplimiento"].iloc[0].to_dict()})
    except Exception as exc:
        tests.append({"prueba":"hidrologia_normativa", "estado":"FALLA", "detalle":str(exc)})
    try:
        coeff=load_regional_coefficients()
        tests.append({"prueba":"coeficientes_regionales_csv", "estado":"OK" if not coeff.empty else "FALLA", "detalle":f"filas={len(coeff)}"})
    except Exception as exc:
        tests.append({"prueba":"coeficientes_regionales_csv", "estado":"FALLA", "detalle":str(exc)})
    return pd.DataFrame(tests)
