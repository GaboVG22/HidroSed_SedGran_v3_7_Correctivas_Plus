
from __future__ import annotations

import math
import pandas as pd

DEFAULT_T = [2, 5, 10, 25, 50, 100, 200]


def time_concentration_kirpich(length_km: float, slope_m_m: float) -> float:
    if length_km <= 0 or slope_m_m <= 0:
        return float("nan")
    L = length_km * 1000
    tc_min = 0.01947 * (L ** 0.77) * (slope_m_m ** -0.385)
    return tc_min / 60.0


def rational_method(area_km2: float, C: float, intensities_mm_h: dict[float, float]) -> pd.DataFrame:
    rows = []
    for T, i in intensities_mm_h.items():
        q = 0.278 * C * float(i) * area_km2
        rows.append({"T_anios": float(T), "metodo": "Racional", "intensidad_mm_h": float(i), "Q_m3s": q})
    return pd.DataFrame(rows)


def dga_ac_q10(area_km2: float, p24_10_mm: float, alpha: float = 2.14) -> float:
    if area_km2 <= 0 or p24_10_mm <= 0:
        return float("nan")
    q10_md = 1.94e-7 * (area_km2 ** 0.776) * (p24_10_mm ** 3.108)
    return q10_md * alpha


DGA_JP_MAX = {2: 0.30, 5: 0.66, 10: 1.00, 20: 1.61, 25: 1.85, 50: 2.76, 75: 3.42, 100: 3.94, 200: 5.20}


def dga_ac_series(area_km2: float, p24_10_mm: float, periods=None, alpha: float = 2.14) -> pd.DataFrame:
    periods = periods or DEFAULT_T
    q10 = dga_ac_q10(area_km2, p24_10_mm, alpha)
    rows = []
    for T in periods:
        factor = DGA_JP_MAX.get(int(T))
        if factor is None:
            xs = sorted(DGA_JP_MAX)
            if T <= xs[0]:
                factor = DGA_JP_MAX[xs[0]]
            elif T >= xs[-1]:
                factor = DGA_JP_MAX[xs[-1]]
            else:
                factor = DGA_JP_MAX[10]
                for a, b in zip(xs[:-1], xs[1:]):
                    if a <= T <= b:
                        la, lb, lt = math.log(a), math.log(b), math.log(T)
                        fa, fb = DGA_JP_MAX[a], DGA_JP_MAX[b]
                        factor = fa + (fb - fa) * (lt - la) / (lb - la)
                        break
        rows.append({"T_anios": float(T), "metodo": "DGA-AC Jp Limari Max", "factor_QT_Q10": factor, "Q_m3s": q10 * factor})
    return pd.DataFrame(rows)


def combine_design_flows(*dfs) -> pd.DataFrame:
    frames = [df for df in dfs if df is not None and len(df) > 0]
    if not frames:
        return pd.DataFrame()
    all_df = pd.concat(frames, ignore_index=True)
    rec = all_df.groupby("T_anios")["Q_m3s"].max().reset_index()
    rec["metodo_adoptado"] = "envolvente_maxima"
    return rec
