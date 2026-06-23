
from __future__ import annotations

import math
import numpy as np
import pandas as pd


def normal_depth_rectangular(Q: float, B: float, S: float, n: float = 0.035) -> float:
    if Q <= 0 or B <= 0 or S <= 0:
        return float("nan")
    def q_of_y(y):
        A = B * y
        P = B + 2 * y
        R = A / P
        return (1 / n) * A * (R ** (2/3)) * (S ** 0.5)
    lo, hi = 1e-4, 50.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if q_of_y(mid) < Q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def hydraulic_and_sediment(secciones: pd.DataFrame, q_design: pd.DataFrame, slope: float = 0.01, n_manning: float = 0.035, d50_m: float = 0.045, d90_m: float = 0.20) -> tuple[pd.DataFrame, pd.DataFrame]:
    if secciones is None or len(secciones) == 0 or q_design is None or len(q_design) == 0:
        return pd.DataFrame(), pd.DataFrame()

    rows_h = []
    rows_s = []
    rho = 1000
    rhos = 2650
    g = 9.81
    Rsub = (rhos - rho) / rho

    for _, srow in secciones.iterrows():
        B = float(srow.get("ancho_m", 50) or 50)
        zbed = float(srow.get("cota_fondo_m", np.nan))
        pk = float(srow.get("pk_m", 0))
        sid = int(srow.get("section_id", 0))

        for _, qrow in q_design.iterrows():
            T = float(qrow["T_anios"])
            Q = float(qrow["Q_m3s"])
            y = normal_depth_rectangular(Q, B, slope, n_manning)
            A = B * y if np.isfinite(y) else np.nan
            V = Q / A if A and A > 0 else np.nan
            Fr = V / math.sqrt(g * y) if np.isfinite(V) and y > 0 else np.nan
            tau = rho * g * y * slope if np.isfinite(y) else np.nan
            theta = tau / ((rhos - rho) * g * d50_m) if d50_m > 0 and np.isfinite(tau) else np.nan
            theta_cr = 0.047
            excess = max(theta - theta_cr, 0) if np.isfinite(theta) else np.nan
            qb_mpm = 8 * (excess ** 1.5) * math.sqrt(Rsub * g * d50_m**3) if np.isfinite(excess) else np.nan
            qs_total = qb_mpm * B if np.isfinite(qb_mpm) else np.nan
            scour_general = max(0, 0.15 * y + 2.0 * max(theta - theta_cr, 0) * d90_m) if np.isfinite(y) and np.isfinite(theta) else np.nan
            z_scour = zbed - scour_general if np.isfinite(zbed) and np.isfinite(scour_general) else np.nan

            rows_h.append({
                "section_id": sid, "pk_m": pk, "T_anios": T, "Q_m3s": Q,
                "ancho_m": B, "pendiente": slope, "n_manning": n_manning,
                "tirante_m": y, "area_m2": A, "velocidad_m_s": V, "Froude": Fr,
                "cota_fondo_m": zbed, "cota_agua_m": zbed + y if np.isfinite(zbed) and np.isfinite(y) else np.nan,
            })
            rows_s.append({
                "section_id": sid, "pk_m": pk, "T_anios": T,
                "tau_Pa": tau, "Shields": theta, "D50_m": d50_m, "D90_m": d90_m,
                "qb_MPM_m2_s": qb_mpm, "Qs_total_m3_s": qs_total,
                "socavacion_general_m": scour_general, "cota_fondo_socavado_m": z_scour,
                "estado": "movil" if np.isfinite(theta) and theta > theta_cr else "estable/preliminar",
            })
    return pd.DataFrame(rows_h), pd.DataFrame(rows_s)
