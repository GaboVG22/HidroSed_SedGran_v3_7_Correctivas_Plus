
from __future__ import annotations

import math
import numpy as np
import pandas as pd

from .hydraulic_hecras_like import (
    G, _section_points_safe, _section_props_at_wse,
    _normal_depth_irregular, _critical_depth_simple
)


def water_density_kgm3(temp_c: float = 15.0) -> float:
    """Densidad aproximada del agua dulce según temperatura [°C]."""
    try:
        T = float(temp_c)
    except Exception:
        T = 15.0
    # Fórmula empírica UNESCO simplificada para 0-40 °C.
    rho = 1000.0 * (1 - ((T + 288.9414) / (508929.2 * (T + 68.12963))) * (T - 3.9863) ** 2)
    return float(rho)


def _safe_num(x, default=np.nan):
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def manning_sensitivity(
    sections_df: pd.DataFrame,
    points_df: pd.DataFrame,
    profile_df: pd.DataFrame,
    n_manning: float,
    slope_energy: float,
) -> pd.DataFrame:
    rows = []
    if profile_df is None or profile_df.empty:
        return pd.DataFrame()
    for _, row in profile_df.iterrows():
        sid = int(row.get("section_id", -1))
        sec_row = sections_df[sections_df.get("section_id").astype(str) == str(sid)].iloc[0] if "section_id" in sections_df and (sections_df.get("section_id").astype(str) == str(sid)).any() else pd.Series({"section_id": sid})
        df_sec, geom_status = _section_points_safe(points_df, sec_row)
        Q = _safe_num(row.get("Q_m3s"), np.nan)
        zmin = float(df_sec["z_m"].min())
        if not np.isfinite(Q) or Q <= 0:
            continue
        ys = {}
        for factor in [0.8, 1.0, 1.2]:
            n2 = max(n_manning * factor, 1e-5)
            y = _normal_depth_irregular(df_sec, Q, max(slope_energy, 1e-8), n2)
            p = _section_props_at_wse(df_sec, zmin + y)
            V = Q / p["area_m2"] if p["area_m2"] > 0 else np.nan
            ys[f"n_{int(factor*100)}"] = {
                "n": n2, "tirante_m": y, "cota_agua_m": zmin + y,
                "velocidad_m_s": V, "area_m2": p["area_m2"],
                "radio_hidraulico_m": p["radio_hidraulico_m"],
            }
        base = ys["n_100"]
        delta_wse = max(abs(ys["n_80"]["cota_agua_m"] - base["cota_agua_m"]), abs(ys["n_120"]["cota_agua_m"] - base["cota_agua_m"]))
        delta_v_pct = max(
            abs(ys["n_80"]["velocidad_m_s"] - base["velocidad_m_s"]),
            abs(ys["n_120"]["velocidad_m_s"] - base["velocidad_m_s"])
        ) / max(abs(base["velocidad_m_s"]), 1e-9) * 100
        rows.append({
            "section_id": sid,
            "pk_m": _safe_num(row.get("pk_m"), np.nan),
            "T_anios": _safe_num(row.get("T_anios"), np.nan),
            "Q_m3s": Q,
            "n_base": n_manning,
            "wse_n_menos20_m": ys["n_80"]["cota_agua_m"],
            "wse_n_base_m": base["cota_agua_m"],
            "wse_n_mas20_m": ys["n_120"]["cota_agua_m"],
            "delta_wse_max_m": delta_wse,
            "vel_n_menos20_m_s": ys["n_80"]["velocidad_m_s"],
            "vel_n_base_m_s": base["velocidad_m_s"],
            "vel_n_mas20_m_s": ys["n_120"]["velocidad_m_s"],
            "delta_velocidad_max_pct": delta_v_pct,
            "sensibilidad_alta_manning": bool(delta_wse > 0.30 or delta_v_pct > 25),
        })
    return pd.DataFrame(rows)


def enhance_hydraulic_profile(
    profile_df: pd.DataFrame,
    sections_df: pd.DataFrame,
    points_df: pd.DataFrame,
    n_manning: float,
    slope_energy: float,
    manning_sensitivity_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if profile_df is None or profile_df.empty:
        return pd.DataFrame()
    out = profile_df.copy()
    rows = []
    for _, row in out.iterrows():
        sid = int(row.get("section_id", -1))
        sec_match = sections_df[sections_df["section_id"].astype(str) == str(sid)] if "section_id" in sections_df else pd.DataFrame()
        sec_row = sec_match.iloc[0] if not sec_match.empty else pd.Series({"section_id": sid})
        df_sec, geom_status = _section_points_safe(points_df, sec_row)
        Q = _safe_num(row.get("Q_m3s"), np.nan)
        zmin = float(df_sec["z_m"].min())
        y_n = _normal_depth_irregular(df_sec, Q, max(slope_energy, 1e-8), n_manning) if np.isfinite(Q) and Q > 0 else np.nan
        y_c = _critical_depth_simple(df_sec, Q) if np.isfinite(Q) and Q > 0 else np.nan
        p_n = _section_props_at_wse(df_sec, zmin + y_n) if np.isfinite(y_n) else {}
        Vn = Q / p_n.get("area_m2", np.nan) if np.isfinite(Q) and p_n.get("area_m2", 0) > 0 else np.nan
        E_specific = row.get("tirante_max_m", np.nan) + (row.get("velocidad_m_s", np.nan)**2)/(2*G) if np.isfinite(row.get("tirante_max_m", np.nan)) and np.isfinite(row.get("velocidad_m_s", np.nan)) else np.nan
        # pendiente local aproximada entre secciones si existe
        rows.append({
            "section_id": sid,
            "T_anios": _safe_num(row.get("T_anios"), np.nan),
            "tirante_normal_manning_m": y_n,
            "cota_normal_manning_m": zmin + y_n if np.isfinite(y_n) else np.nan,
            "calado_critico_m": y_c,
            "area_mojada_normal_m2": p_n.get("area_m2", np.nan),
            "perimetro_mojado_normal_m": p_n.get("perimetro_mojado_m", np.nan),
            "radio_hidraulico_normal_m": p_n.get("radio_hidraulico_m", np.nan),
            "ancho_superficial_normal_m": p_n.get("ancho_superior_m", np.nan),
            "velocidad_normal_m_s": Vn,
            "energia_especifica_m": E_specific,
            "comparacion_perfil_vs_normal_m": row.get("cota_agua_m", np.nan) - (zmin + y_n) if np.isfinite(y_n) else np.nan,
            "geometria_extrapolada_riberas": bool(row.get("cota_agua_m", 0) > df_sec["z_m"].max()),
        })
    add = pd.DataFrame(rows)
    merged = out.merge(add, on=["section_id", "T_anios"], how="left")
    if manning_sensitivity_df is not None and not manning_sensitivity_df.empty:
        keep = ["section_id", "T_anios", "delta_wse_max_m", "delta_velocidad_max_pct", "sensibilidad_alta_manning"]
        merged = merged.merge(manning_sensitivity_df[keep], on=["section_id", "T_anios"], how="left")
    return merged


def sediment_transport_advanced(
    profile_df: pd.DataFrame,
    d50_m: float,
    d75_m: float | None = None,
    d90_m: float = 0.20,
    slope_energy: float = 0.01,
    temp_c: float = 15.0,
) -> pd.DataFrame:
    if profile_df is None or profile_df.empty:
        return pd.DataFrame()
    rho = water_density_kgm3(temp_c)
    rhos = 2650.0
    Rsub = (rhos - rho) / rho
    theta_cr = 0.047
    d50 = max(float(d50_m), 1e-6)
    d75 = max(float(d75_m) if d75_m and np.isfinite(float(d75_m)) else d50*1.8, 1e-6)
    d90 = max(float(d90_m), 1e-6)
    rows = []
    for _, row in profile_df.iterrows():
        Rh = _safe_num(row.get("radio_hidraulico_m"), np.nan)
        B = _safe_num(row.get("ancho_superior_m"), np.nan)
        y = _safe_num(row.get("tirante_max_m"), np.nan)
        zbed = _safe_num(row.get("cota_fondo_m"), np.nan)
        V = _safe_num(row.get("velocidad_m_s"), np.nan)
        Q = _safe_num(row.get("Q_m3s"), np.nan)
        Fr = _safe_num(row.get("Froude"), np.nan)
        S = max(float(slope_energy), 1e-8)
        tau = rho * G * Rh * S if np.isfinite(Rh) else np.nan
        theta = tau / ((rhos - rho) * G * d50) if np.isfinite(tau) else np.nan
        excess = max(theta - theta_cr, 0) if np.isfinite(theta) else np.nan
        qb_mpm = 8.0 * (excess ** 1.5) * math.sqrt(max(Rsub,1e-9) * G * d50**3) if np.isfinite(excess) else np.nan
        # Engelund-Hansen total load indicative formulation: strong nonlinearity with V.
        # Uses dimensionless transport parameter scaled with Shields. Kept as preliminary if sandy/gravelly.
        cf = tau / max(rho * V * V, 1e-12) if np.isfinite(tau) and np.isfinite(V) and V > 0 else np.nan
        phi_eh = 0.05 * max(theta, 0)**2.5 / max(cf, 1e-6) if np.isfinite(theta) and np.isfinite(cf) else np.nan
        qs_eh_unit = phi_eh * math.sqrt(max(Rsub,1e-9) * G * d50**3) if np.isfinite(phi_eh) else np.nan
        q_total_eh = qs_eh_unit * B if np.isfinite(qs_eh_unit) and np.isfinite(B) else np.nan
        q_bed_total = qb_mpm * B if np.isfinite(qb_mpm) and np.isfinite(B) else np.nan
        # Simple local/general preliminary scour
        scour_general = max(0, 0.12*y + 1.8*max(theta-theta_cr,0)*d90) if np.isfinite(y) and np.isfinite(theta) else np.nan
        scour_local = max(0, 0.25*y*max(Fr-0.6,0) + 0.75*max(theta-theta_cr,0)*d90) if np.isfinite(y) and np.isfinite(Fr) and np.isfinite(theta) else np.nan
        mobility = "sin_dato"
        if np.isfinite(theta):
            if theta < theta_cr:
                mobility = "estable / sin movilidad incipiente"
            elif theta < 2*theta_cr:
                mobility = "movilidad incipiente"
            else:
                mobility = "transporte activo"
        rows.append({
            "section_id": int(row.get("section_id", -1)),
            "pk_m": _safe_num(row.get("pk_m"), np.nan),
            "T_anios": _safe_num(row.get("T_anios"), np.nan),
            "Q_m3s": Q,
            "rho_agua_kg_m3": rho,
            "temperatura_C": temp_c,
            "D50_m": d50,
            "D75_m": d75,
            "D90_m": d90,
            "tau_Pa": tau,
            "Shields": theta,
            "Shields_critico": theta_cr,
            "estado_movilidad_lecho": mobility,
            "qb_MPM_m2_s": qb_mpm,
            "Qb_MPM_m3_s": q_bed_total,
            "qs_EH_unit_m2_s": qs_eh_unit,
            "Qs_EH_total_m3_s": q_total_eh,
            "Qs_total_m3_s": np.nanmax([q_total_eh, q_bed_total]) if np.isfinite(q_total_eh) or np.isfinite(q_bed_total) else np.nan,
            "socavacion_general_m": scour_general,
            "socavacion_local_prelim_m": scour_local,
            "socavacion_total_prelim_m": scour_general + scour_local if np.isfinite(scour_general) and np.isfinite(scour_local) else np.nan,
            "cota_fondo_socavado_m": zbed - scour_general - scour_local if np.isfinite(zbed) and np.isfinite(scour_general) and np.isfinite(scour_local) else np.nan,
            "estado": "movil" if np.isfinite(theta) and theta > theta_cr else "estable/preliminar",
        })
    return pd.DataFrame(rows)


def hydraulic_qa(
    sections_df: pd.DataFrame,
    points_df: pd.DataFrame,
    profile_df: pd.DataFrame,
    n_manning: float,
    slope_energy: float,
) -> pd.DataFrame:
    rows = []
    if sections_df is None or sections_df.empty:
        return pd.DataFrame()
    sens_lookup = {}
    if "sensibilidad_alta_manning" in profile_df.columns:
        tmp = profile_df.groupby("section_id")["sensibilidad_alta_manning"].max()
        sens_lookup = tmp.to_dict()
    for _, sec_row in sections_df.iterrows():
        sid = int(sec_row.get("section_id", -1))
        df_sec, geom_status = _section_points_safe(points_df, sec_row)
        warnings = []
        npts = len(df_sec)
        x = pd.to_numeric(df_sec["offset_m"], errors="coerce")
        z = pd.to_numeric(df_sec["z_m"], errors="coerce")
        width = float(x.max()-x.min()) if len(x) else np.nan
        dx = np.diff(np.sort(x.dropna().unique()))
        max_spacing = float(np.max(dx)) if len(dx) else np.nan
        if npts < 6: warnings.append("seccion_con_pocos_puntos")
        if z.isna().any() or (z.fillna(0)==0).mean() > 0.5: warnings.append("cotas_nulas_o_invalidas")
        if x.duplicated().any(): warnings.append("distancias_transversales_duplicadas")
        if not np.isfinite(width) or width <= 0: warnings.append("ancho_hidraulico_no_positivo")
        if np.isfinite(max_spacing) and np.isfinite(width) and max_spacing > max(width/8, 5): warnings.append("separacion_transversal_demasiado_gruesa")
        if slope_energy <= 0: warnings.append("pendiente_no_positiva")
        if slope_energy > 0.05: warnings.append("pendiente_excesiva_para_hipotesis_1D")
        if not (0.015 <= n_manning <= 0.080): warnings.append("manning_fuera_de_rango_usual")
        prof_sid = profile_df[profile_df["section_id"].astype(str)==str(sid)] if profile_df is not None and not profile_df.empty and "section_id" in profile_df.columns else pd.DataFrame()
        if not prof_sid.empty and "Froude" in prof_sid.columns:
            frmax = pd.to_numeric(prof_sid["Froude"], errors="coerce").max()
            if np.isfinite(frmax) and 0.8 <= frmax <= 1.2: warnings.append("regimen_cercano_a_critico")
            if np.isfinite(frmax) and frmax > 1.0: warnings.append("regimen_supercritico")
        if geom_status != "real" or (not prof_sid.empty and prof_sid.get("geometria_fallback", pd.Series(False)).fillna(False).any()):
            warnings.append("geometria_extrapolada_o_fallback")
        if sens_lookup.get(sid, False):
            warnings.append("sensibilidad_alta_a_manning")
        rows.append({
            "section_id": sid,
            "pk_m": float(sec_row.get("pk_m", np.nan)),
            "n_puntos": int(npts),
            "ancho_m": width,
            "separacion_max_m": max_spacing,
            "geometria_status": geom_status,
            "n_warnings": len(warnings),
            "warnings": "; ".join(warnings) if warnings else "OK",
            "estado_QA": "OK" if not warnings else ("REVISAR" if len(warnings)<=2 else "CRITICO"),
        })
    return pd.DataFrame(rows)


def confidence_report(profile_df, sediment_df, qa_df, sensitivity_df, mc_df=None) -> pd.DataFrame:
    score = 9.0
    penalties = []
    if profile_df is None or profile_df.empty:
        score -= 3; penalties.append("sin_perfil_hidraulico")
    if sediment_df is None or sediment_df.empty:
        score -= 1.5; penalties.append("sin_sedimentos")
    if qa_df is not None and not qa_df.empty:
        frac_bad = (qa_df["estado_QA"].isin(["REVISAR","CRITICO"])).mean()
        score -= min(2.0, frac_bad*2.0)
        if frac_bad>0: penalties.append(f"qa_con_observaciones_{frac_bad:.0%}")
        frac_crit = (qa_df["estado_QA"]=="CRITICO").mean()
        score -= min(1.5, frac_crit*2.0)
    if sensitivity_df is not None and not sensitivity_df.empty:
        frac_sens = sensitivity_df.get("sensibilidad_alta_manning", pd.Series(False)).fillna(False).mean()
        score -= min(1.0, frac_sens*1.5)
        if frac_sens>0: penalties.append(f"sensibilidad_manning_alta_{frac_sens:.0%}")
    if mc_df is None or (hasattr(mc_df, "empty") and mc_df.empty):
        score -= 0.4; penalties.append("sin_monte_carlo")
    score = max(1.0, min(10.0, score))
    return pd.DataFrame([{
        "puntaje_confianza_1_10": round(score, 2),
        "nivel": "alto" if score>=8.7 else ("medio" if score>=7 else "preliminar"),
        "penalizaciones": "; ".join(penalties) if penalties else "sin_penalizaciones_relevantes",
        "recomendacion": "aceptable para revisión técnica preliminar avanzada" if score>=8.7 else "requiere completar QA/calibración/datos observados",
    }])


def monte_carlo_uncertainty(
    profile_df: pd.DataFrame,
    d50_m: float,
    n_manning: float,
    slope_energy: float,
    q_sigma: float = 0.15,
    n_sigma: float = 0.20,
    d50_sigma: float = 0.25,
    slope_sigma: float = 0.25,
    n_iter: int = 100,
    random_seed: int = 42,
) -> pd.DataFrame:
    if profile_df is None or profile_df.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(random_seed)
    rows = []
    base = profile_df.copy()
    # Monte Carlo reduced: perturb formula outputs from main rows without resolving profile completely.
    for _, row in base.iterrows():
        Q0 = _safe_num(row.get("Q_m3s"), np.nan)
        Rh0 = _safe_num(row.get("radio_hidraulico_m"), np.nan)
        B0 = _safe_num(row.get("ancho_superior_m"), np.nan)
        V0 = _safe_num(row.get("velocidad_m_s"), np.nan)
        if not all(np.isfinite(v) for v in [Q0,Rh0,B0,V0]):
            continue
        scours=[]; qs=[]; shields=[]
        for _i in range(int(n_iter)):
            Q = Q0 * max(0.1, rng.normal(1.0, q_sigma))
            n = n_manning * max(0.1, rng.normal(1.0, n_sigma))
            d50 = d50_m * max(0.1, rng.normal(1.0, d50_sigma))
            S = slope_energy * max(0.05, rng.normal(1.0, slope_sigma))
            V = V0 * (Q/Q0) * (n_manning/n)**0.4
            tau = 1000*G*Rh0*S
            theta = tau/((2650-1000)*G*d50)
            excess=max(theta-0.047,0)
            qb = 8*(excess**1.5)*math.sqrt(1.65*G*d50**3)*B0
            sc = max(0, 0.12*row.get("tirante_max_m",0)+1.8*excess*d50*3.0)
            qs.append(qb); scours.append(sc); shields.append(theta)
        rows.append({
            "section_id": int(row.get("section_id",-1)),
            "pk_m": _safe_num(row.get("pk_m"), np.nan),
            "T_anios": _safe_num(row.get("T_anios"), np.nan),
            "scour_p05_m": float(np.percentile(scours,5)),
            "scour_p50_m": float(np.percentile(scours,50)),
            "scour_p95_m": float(np.percentile(scours,95)),
            "Qs_p05_m3_s": float(np.percentile(qs,5)),
            "Qs_p50_m3_s": float(np.percentile(qs,50)),
            "Qs_p95_m3_s": float(np.percentile(qs,95)),
            "Shields_p50": float(np.percentile(shields,50)),
            "n_iter": int(n_iter),
        })
    return pd.DataFrame(rows)
