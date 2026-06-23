from __future__ import annotations

import math
from typing import List, Tuple
import numpy as np
import pandas as pd

G = 9.81


def trapezoid_points(bottom_width_m: float, depth_m: float, z_bed_m: float, side_slope_left_hv: float, side_slope_right_hv: float, points_per_side: int = 4) -> pd.DataFrame:
    b = float(bottom_width_m)
    y = float(depth_m)
    zl = float(side_slope_left_hv)
    zr = float(side_slope_right_hv)
    left_top = -b / 2 - zl * y
    left_bottom = -b / 2
    right_bottom = b / 2
    right_top = b / 2 + zr * y
    xs = []
    zs = []
    # left slope from top to bottom
    for i in range(points_per_side):
        f = i / max(points_per_side - 1, 1)
        xs.append(left_top + f * (left_bottom - left_top))
        zs.append(z_bed_m + y * (1 - f))
    xs.extend([left_bottom, right_bottom])
    zs.extend([z_bed_m, z_bed_m])
    for i in range(points_per_side):
        f = i / max(points_per_side - 1, 1)
        xs.append(right_bottom + f * (right_top - right_bottom))
        zs.append(z_bed_m + y * f)
    df = pd.DataFrame({"offset_m": xs, "z_m": zs})
    df = df.drop_duplicates(subset=["offset_m"]).sort_values("offset_m").reset_index(drop=True)
    return df


def generate_trapezoid_reach_sections(
    length_m: float,
    spacing_m: float,
    bottom_width_m: float,
    depth_m: float,
    side_slope_left_hv: float,
    side_slope_right_hv: float,
    slope_longitudinal: float,
    z0_m: float = 100.0,
    start_section_id: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    length_m = max(float(length_m), float(spacing_m))
    nsec = int(math.floor(length_m / spacing_m)) + 1
    sections = []
    points = []
    for i in range(nsec):
        pk = i * float(spacing_m)
        sid = start_section_id + i
        zbed = float(z0_m) - pk * float(slope_longitudinal)
        sections.append({
            "section_id": sid,
            "pk_m": pk,
            "origen": "trapezoidal_estimado",
            "ancho_fondo_m": bottom_width_m,
            "profundidad_geom_m": depth_m,
            "talud_izq_hv": side_slope_left_hv,
            "talud_der_hv": side_slope_right_hv,
            "confianza_seccion": 7.2,
        })
        pdf = trapezoid_points(bottom_width_m, depth_m, zbed, side_slope_left_hv, side_slope_right_hv)
        pdf["section_id"] = sid
        pdf["pk_m"] = pk
        points.append(pdf)
    return pd.DataFrame(sections), pd.concat(points, ignore_index=True)


def trapezoid_properties(bottom_width_m: float, side_slope_left_hv: float, side_slope_right_hv: float, depth_y_m: float) -> dict:
    b = float(bottom_width_m); m1 = float(side_slope_left_hv); m2 = float(side_slope_right_hv); y = max(float(depth_y_m), 0.0)
    A = b * y + 0.5 * (m1 + m2) * y * y
    T = b + (m1 + m2) * y
    P = b + y * math.sqrt(1 + m1 * m1) + y * math.sqrt(1 + m2 * m2)
    R = A / P if P > 0 else np.nan
    D = A / T if T > 0 else np.nan
    return {"area_m2": A, "ancho_superior_m": T, "perimetro_mojado_m": P, "radio_hidraulico_m": R, "profundidad_media_m": D}


def normal_depth_trapezoid(Q_m3s: float, bottom_width_m: float, side_slope_left_hv: float, side_slope_right_hv: float, slope: float, n_manning: float) -> float:
    Q = float(Q_m3s); S = max(float(slope), 1e-7); n = max(float(n_manning), 1e-5)
    lo, hi = 0.001, 1.0
    def q_at(y):
        p = trapezoid_properties(bottom_width_m, side_slope_left_hv, side_slope_right_hv, y)
        return (1/n) * p["area_m2"] * (p["radio_hidraulico_m"] ** (2/3)) * math.sqrt(S) if p["area_m2"] > 0 and p["radio_hidraulico_m"] > 0 else 0
    while q_at(hi) < Q and hi < 200:
        hi *= 1.6
    for _ in range(80):
        mid = 0.5*(lo+hi)
        if q_at(mid) < Q: lo = mid
        else: hi = mid
    return float(0.5*(lo+hi))


def critical_depth_trapezoid(Q_m3s: float, bottom_width_m: float, side_slope_left_hv: float, side_slope_right_hv: float) -> float:
    Q = float(Q_m3s)
    ys = np.geomspace(0.005, 100.0, 400)
    best_y, best_E = np.nan, np.inf
    for y in ys:
        p = trapezoid_properties(bottom_width_m, side_slope_left_hv, side_slope_right_hv, y)
        A = p["area_m2"]
        if A <= 0: continue
        E = y + (Q/A)**2/(2*G)
        if E < best_E:
            best_E = E; best_y = y
    return float(best_y)


def trapezoid_capacity_table(q_values: List[float], bottom_width_m: float, depth_m: float, side_slope_left_hv: float, side_slope_right_hv: float, slope: float, n_manning: float) -> pd.DataFrame:
    rows = []
    for Q in q_values:
        yn = normal_depth_trapezoid(Q, bottom_width_m, side_slope_left_hv, side_slope_right_hv, slope, n_manning)
        yc = critical_depth_trapezoid(Q, bottom_width_m, side_slope_left_hv, side_slope_right_hv)
        p = trapezoid_properties(bottom_width_m, side_slope_left_hv, side_slope_right_hv, yn)
        V = Q / p["area_m2"] if p["area_m2"] > 0 else np.nan
        Fr = V / math.sqrt(G * max(p["profundidad_media_m"], 1e-9)) if np.isfinite(V) else np.nan
        rows.append({"Q_m3s": Q, "y_normal_m": yn, "y_critico_m": yc, "velocidad_m_s": V, "Froude": Fr, **p, "sobrepasa_profundidad_geom": yn > depth_m})
    return pd.DataFrame(rows)
