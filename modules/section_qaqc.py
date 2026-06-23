
from __future__ import annotations

import numpy as np
import pandas as pd


def _interp_fill_section(df: pd.DataFrame, min_points_for_interp: int = 3) -> pd.DataFrame:
    out = df.sort_values("offset_m").copy()
    z = out["z_m"].to_numpy(float)
    x = out["offset_m"].to_numpy(float)
    valid = np.isfinite(z)
    out["z_original_m"] = z
    out["z_filled"] = False
    if valid.sum() >= min_points_for_interp and (~valid).any():
        z_new = z.copy()
        z_new[~valid] = np.interp(x[~valid], x[valid], z[valid])
        out["z_m"] = z_new
        out.loc[~valid, "z_filled"] = True
    return out


def select_and_fill_sections(
    sections_df: pd.DataFrame,
    points_df: pd.DataFrame,
    min_valid_points: int = 9,
    min_total_points: int = 11,
    max_nan_pct: float = 25.0,
    min_wettable_width_m: float = 5.0,
    fill_missing: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Rellena puntos faltantes, evalúa calidad y selecciona secciones válidas.

    Criterios:
    - mínimo de puntos totales;
    - mínimo de puntos con cota válida;
    - porcentaje máximo de cotas faltantes;
    - ancho útil mínimo;
    - presencia de fondo reconocible.
    """
    if sections_df is None or points_df is None or sections_df.empty or points_df.empty:
        return sections_df, points_df, pd.DataFrame()

    sections = sections_df.copy()
    points = points_df.copy()
    processed = []
    report_rows = []

    for sid, g in points.groupby("section_id"):
        g0 = g.sort_values("offset_m").copy()
        n_total = len(g0)
        n_valid_ini = int(np.isfinite(g0["z_m"]).sum())
        nan_pct_ini = 100.0 * (1.0 - n_valid_ini / max(n_total, 1))

        if fill_missing:
            g1 = _interp_fill_section(g0)
        else:
            g1 = g0.copy()
            g1["z_original_m"] = g1["z_m"]
            g1["z_filled"] = False

        n_valid = int(np.isfinite(g1["z_m"]).sum())
        nan_pct = 100.0 * (1.0 - n_valid / max(n_total, 1))
        valid_offsets = g1.loc[np.isfinite(g1["z_m"]), "offset_m"]
        useful_width = float(valid_offsets.max() - valid_offsets.min()) if len(valid_offsets) >= 2 else 0.0
        zmin = float(np.nanmin(g1["z_m"])) if n_valid else np.nan
        zmax = float(np.nanmax(g1["z_m"])) if n_valid else np.nan
        relief = zmax - zmin if np.isfinite(zmin) and np.isfinite(zmax) else np.nan

        reasons = []
        if n_total < min_total_points:
            reasons.append(f"puntos_totales<{min_total_points}")
        if n_valid < min_valid_points:
            reasons.append(f"puntos_validos<{min_valid_points}")
        if nan_pct > max_nan_pct:
            reasons.append(f"nan_pct>{max_nan_pct}")
        if useful_width < min_wettable_width_m:
            reasons.append(f"ancho_util<{min_wettable_width_m}m")
        if not np.isfinite(zmin):
            reasons.append("sin_cota_fondo")
        if np.isfinite(relief) and relief < 0.05:
            reasons.append("relieve_transversal_muy_bajo")

        estado = "válida" if not reasons else "descartada"
        g1["section_qc_status"] = estado
        g1["section_qc_reasons"] = "; ".join(reasons)
        processed.append(g1)

        report_rows.append({
            "section_id": int(sid),
            "estado": estado,
            "razones": "; ".join(reasons),
            "n_puntos_total": n_total,
            "n_puntos_validos_ini": n_valid_ini,
            "n_puntos_validos_final": n_valid,
            "nan_pct_ini": nan_pct_ini,
            "nan_pct_final": nan_pct,
            "n_puntos_rellenos": int(g1["z_filled"].sum()) if "z_filled" in g1 else 0,
            "ancho_util_m": useful_width,
            "cota_fondo_m": zmin,
            "cota_max_m": zmax,
            "relieve_transversal_m": relief,
        })

    pts_out = pd.concat(processed, ignore_index=True) if processed else points
    report = pd.DataFrame(report_rows)
    valid_ids = set(report.loc[report["estado"] == "válida", "section_id"].astype(int).tolist())

    sec_out = sections[sections["section_id"].astype(int).isin(valid_ids)].copy()
    pts_valid = pts_out[pts_out["section_id"].astype(int).isin(valid_ids)].copy()

    if not sec_out.empty:
        # Actualiza cotas resumidas desde puntos corregidos.
        agg = pts_valid.groupby("section_id").agg(
            cota_fondo_m=("z_m", "min"),
            n_puntos=("z_m", "size"),
            n_puntos_validos=("z_m", lambda s: int(np.isfinite(s).sum())),
        ).reset_index()
        sec_out = sec_out.drop(columns=[c for c in ["cota_fondo_m", "n_puntos", "n_puntos_validos"] if c in sec_out.columns], errors="ignore")
        sec_out = sec_out.merge(agg, on="section_id", how="left")
        sec_out["qc_estado"] = "válida"

    return sec_out, pts_valid, report


def section_report_summary(report: pd.DataFrame) -> dict:
    if report is None or report.empty:
        return {"n_total": 0, "n_validas": 0, "n_descartadas": 0, "pct_validas": 0.0}
    n = len(report)
    nv = int((report["estado"] == "válida").sum())
    return {
        "n_total": int(n),
        "n_validas": nv,
        "n_descartadas": int(n - nv),
        "pct_validas": 100.0 * nv / max(n, 1),
    }
