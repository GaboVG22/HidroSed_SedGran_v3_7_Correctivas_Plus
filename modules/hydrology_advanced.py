from __future__ import annotations

import math
import numpy as np
import pandas as pd

PERIODS = [2, 5, 10, 25, 50, 100, 200]
GROWTH_P24 = {2: 0.65, 5: 0.85, 10: 1.00, 25: 1.25, 50: 1.45, 100: 1.65, 200: 1.90}
DGA_JP_MAX = {2: 0.30, 5: 0.66, 10: 1.00, 20: 1.61, 25: 1.85, 50: 2.76, 75: 3.42, 100: 3.94, 200: 5.20}


def _safe(v, default=np.nan):
    try:
        x = float(v)
        return x if np.isfinite(x) else default
    except Exception:
        return default


def tc_kirpich(length_km: float, slope: float) -> float:
    L = _safe(length_km, 0) * 1000
    S = max(_safe(slope, 0), 1e-6)
    if L <= 0:
        return np.nan
    return 0.01947 * (L ** 0.77) * (S ** -0.385) / 60.0


def tc_temez(length_km: float, slope: float) -> float:
    L = _safe(length_km, 0)
    S = max(_safe(slope, 0), 1e-6)
    if L <= 0:
        return np.nan
    return 0.30 * ((L / (S ** 0.25)) ** 0.76)


def tc_giandotti(area_km2: float, length_km: float, dz_m: float | None = None, slope: float | None = None) -> float:
    A = _safe(area_km2, 0)
    L = _safe(length_km, 0)
    H = _safe(dz_m, np.nan)
    if not np.isfinite(H) or H <= 0:
        H = max(_safe(slope, 0.01) * L * 1000, 1.0)
    if A <= 0 or L <= 0:
        return np.nan
    return (4 * math.sqrt(A) + 1.5 * L) / (0.8 * math.sqrt(H))


def tc_methods(area_km2: float, length_km: float, slope: float, dz_m: float | None = None) -> pd.DataFrame:
    rows = [
        {"metodo": "Kirpich", "tc_h": tc_kirpich(length_km, slope), "uso": "cuencas pequeñas/pendientes"},
        {"metodo": "Témez", "tc_h": tc_temez(length_km, slope), "uso": "cuencas rurales medianas"},
        {"metodo": "Giandotti", "tc_h": tc_giandotti(area_km2, length_km, dz_m, slope), "uso": "contraste morfométrico"},
    ]
    df = pd.DataFrame(rows)
    vals = pd.to_numeric(df["tc_h"], errors="coerce").dropna()
    if len(vals):
        adopted = float(vals.median())
        df["tc_adoptado_h"] = adopted
    else:
        df["tc_adoptado_h"] = np.nan
    return df


def idf_intensity_from_p24(p24_mm: float, tc_h: float, exponent: float = 0.35) -> float:
    p24 = max(_safe(p24_mm, 0), 0)
    tc = max(_safe(tc_h, 1.0), 0.10)
    cd = min(1.0, max(0.10, (tc / 24.0) ** exponent))
    return p24 * cd / tc


def p24_by_period(p24_10_mm: float, periods=None, growth=None) -> dict:
    periods = periods or PERIODS
    growth = growth or GROWTH_P24
    return {int(T): _safe(p24_10_mm, 0) * growth.get(int(T), 1.0) for T in periods}


def rational_flows(area_km2: float, C: float, p24_10_mm: float, tc_h: float, periods=None) -> pd.DataFrame:
    periods = periods or PERIODS
    p24 = p24_by_period(p24_10_mm, periods)
    rows = []
    for T in periods:
        i = idf_intensity_from_p24(p24[int(T)], tc_h)
        Q = 0.278 * _safe(C, 0.35) * i * _safe(area_km2, 0)
        rows.append({"T_anios": float(T), "metodo": "Racional_IDF", "P24_mm": p24[int(T)], "Tc_h": tc_h, "intensidad_mm_h": i, "Q_m3s": Q})
    return pd.DataFrame(rows)


def rational_modified_flows(area_km2: float, C: float, p24_10_mm: float, tc_h: float, periods=None) -> pd.DataFrame:
    df = rational_flows(area_km2, C, p24_10_mm, tc_h, periods).copy()
    A = max(_safe(area_km2, 0), 0)
    # Atenuación simple por almacenamiento/área: conserva racional en cuencas pequeñas y modera cuencas mayores.
    attenuation = 1.0 / (1.0 + 0.004 * max(A - 25.0, 0.0) ** 0.85)
    attenuation = max(0.35, min(1.0, attenuation))
    df["metodo"] = "Racional_modificado"
    df["factor_atenuacion_area"] = attenuation
    df["Q_m3s"] = df["Q_m3s"] * attenuation
    return df


def dga_ac_q10(area_km2: float, p24_10_mm: float, alpha: float = 2.14) -> float:
    A = max(_safe(area_km2, 0), 0)
    P = max(_safe(p24_10_mm, 0), 0)
    if A <= 0 or P <= 0:
        return np.nan
    q10_md = 1.94e-7 * (A ** 0.776) * (P ** 3.108)
    return q10_md * _safe(alpha, 2.14)


def _growth_dga(T):
    T = float(T)
    if int(T) in DGA_JP_MAX:
        return DGA_JP_MAX[int(T)]
    xs = sorted(DGA_JP_MAX)
    if T <= xs[0]:
        return DGA_JP_MAX[xs[0]]
    if T >= xs[-1]:
        return DGA_JP_MAX[xs[-1]]
    for a, b in zip(xs[:-1], xs[1:]):
        if a <= T <= b:
            la, lb, lt = math.log(a), math.log(b), math.log(T)
            return DGA_JP_MAX[a] + (DGA_JP_MAX[b]-DGA_JP_MAX[a])*(lt-la)/(lb-la)
    return DGA_JP_MAX[10]


def dga_ac_flows(area_km2: float, p24_10_mm: float, alpha: float = 2.14, periods=None) -> pd.DataFrame:
    periods = periods or PERIODS
    q10 = dga_ac_q10(area_km2, p24_10_mm, alpha)
    rows = []
    for T in periods:
        factor = _growth_dga(T)
        rows.append({"T_anios": float(T), "metodo": "DGA_AC_regional", "factor_QT_Q10": factor, "Q_m3s": q10 * factor})
    return pd.DataFrame(rows)


def transfer_flows(area_km2: float, station_area_km2: float, station_q100: float, b_exp: float = 0.75, f_alt: float = 1.0, f_dist: float = 1.0, periods=None) -> pd.DataFrame:
    periods = periods or PERIODS
    A = _safe(area_km2, 0); As = _safe(station_area_km2, 0); Q100 = _safe(station_q100, 0)
    if A <= 0 or As <= 0 or Q100 <= 0:
        return pd.DataFrame()
    q100 = Q100 * (A / As) ** _safe(b_exp, 0.75) * _safe(f_alt, 1.0) * _safe(f_dist, 1.0)
    rows = []
    for T in periods:
        q = q100 * GROWTH_P24.get(int(T), 1.0) / GROWTH_P24[100]
        rows.append({"T_anios": float(T), "metodo": "Transferencia_area_altitud_distancia", "Q_m3s": q, "Q100_transferido_m3s": q100})
    return pd.DataFrame(rows)


def method_recommendation(area_km2: float, has_station: bool = False, basin_regime: str = "pluvial") -> pd.DataFrame:
    A = _safe(area_km2, 0)
    rows = []
    if A <= 25:
        rows.append(["Racional_IDF", "Alta", "Cuenca pequeña: método directo, sensible al C e IDF."])
        rows.append(["Racional_modificado", "Media", "Útil como contraste si hay almacenamiento o respuesta atenuada."])
    elif A <= 250:
        rows.append(["Racional_modificado", "Alta", "Cuenca mediana: atenúa el racional puro."])
        rows.append(["DGA_AC_regional", "Alta", "Contraste regional recomendado."])
    else:
        rows.append(["DGA_AC_regional", "Alta", "Cuenca grande: preferir regionalización, transferencia o HUS."])
        rows.append(["Racional_modificado", "Media-baja", "Solo como contraste preliminar."])
    if has_station:
        rows.append(["Transferencia_area_altitud_distancia", "Alta", "Existe estación de referencia; documentar similitud y área."])
    if "nivo" in basin_regime.lower():
        rows.append(["Ajuste nivo-pluvial", "Condicional", "Revisar aporte nival, altitud, estacionalidad y evento crítico."])
    return pd.DataFrame(rows, columns=["metodo", "prioridad", "criterio"])


def build_hydrology(area_km2, length_km, slope, C, p24_10, alpha, periods=None, include_transfer=False, station_area=0, station_q100=0, b_exp=0.75, f_alt=1.0, f_dist=1.0, dz_m=None, basin_regime="pluvial"):
    periods = periods or PERIODS
    tc_df = tc_methods(area_km2, length_km, slope, dz_m)
    tc_vals = pd.to_numeric(tc_df["tc_h"], errors="coerce").dropna()
    tc_adopted = float(tc_vals.median()) if len(tc_vals) else np.nan
    dfs = [
        rational_flows(area_km2, C, p24_10, tc_adopted, periods),
        rational_modified_flows(area_km2, C, p24_10, tc_adopted, periods),
        dga_ac_flows(area_km2, p24_10, alpha, periods),
    ]
    if include_transfer:
        tf = transfer_flows(area_km2, station_area, station_q100, b_exp, f_alt, f_dist, periods)
        if not tf.empty:
            dfs.append(tf)
    all_df = pd.concat([d for d in dfs if d is not None and not d.empty], ignore_index=True)
    rec_df = method_recommendation(area_km2, include_transfer, basin_regime)
    unc = all_df.groupby("T_anios")["Q_m3s"].agg(Q_min="min", Q_mediana="median", Q_max="max", Q_promedio="mean").reset_index()
    unc["rango_relativo_pct"] = np.where(unc["Q_mediana"] > 0, (unc["Q_max"] - unc["Q_min"]) / unc["Q_mediana"] * 100, np.nan)
    return tc_df, all_df, rec_df, unc


def adopt_flows(all_df: pd.DataFrame, mode: str = "envolvente_maxima") -> pd.DataFrame:
    if all_df is None or all_df.empty:
        return pd.DataFrame()
    if mode == "mediana_metodos":
        out = all_df.groupby("T_anios")["Q_m3s"].median().reset_index()
        out["metodo_adoptado"] = "mediana_metodos"
    elif mode == "promedio_metodos":
        out = all_df.groupby("T_anios")["Q_m3s"].mean().reset_index()
        out["metodo_adoptado"] = "promedio_metodos"
    else:
        out = all_df.groupby("T_anios")["Q_m3s"].max().reset_index()
        out["metodo_adoptado"] = "envolvente_maxima"
    return out.sort_values("T_anios").reset_index(drop=True)
