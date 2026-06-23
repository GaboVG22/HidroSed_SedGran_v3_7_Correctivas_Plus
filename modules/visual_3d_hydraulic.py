
from __future__ import annotations

import io
import numpy as np
import pandas as pd


def _merge_profile(points_df: pd.DataFrame, hydraulic_df: pd.DataFrame | None, sediment_df: pd.DataFrame | None):
    pts = points_df.copy()
    if hydraulic_df is not None and not hydraulic_df.empty:
        # Mostrar por defecto el mayor periodo de retorno disponible.
        h = hydraulic_df.copy()
        tmax = h["T_anios"].max()
        h = h[h["T_anios"] == tmax][["section_id", "T_anios", "cota_agua_m", "velocidad_m_s", "Froude"]]
        pts = pts.merge(h, on="section_id", how="left")
    else:
        pts["cota_agua_m"] = np.nan
        pts["velocidad_m_s"] = np.nan
        pts["Froude"] = np.nan
        pts["T_anios"] = np.nan

    if sediment_df is not None and not sediment_df.empty:
        s = sediment_df.copy()
        tmax = s["T_anios"].max()
        keep = [c for c in ["section_id", "T_anios", "socavacion_general_m", "cota_fondo_socavado_m", "Shields", "estado"] if c in s.columns]
        s = s[s["T_anios"] == tmax][keep]
        s = s.drop(columns=["T_anios"], errors="ignore")
        pts = pts.merge(s, on="section_id", how="left")
    else:
        pts["socavacion_general_m"] = np.nan
        pts["cota_fondo_socavado_m"] = np.nan
        pts["Shields"] = np.nan
        pts["estado"] = ""

    return pts


def create_3d_profile_figure(
    sections_df: pd.DataFrame,
    points_df: pd.DataFrame,
    hydraulic_df: pd.DataFrame | None = None,
    sediment_df: pd.DataFrame | None = None,
    vertical_exaggeration: float = 1.0,
    show_water: bool = True,
    show_scour: bool = True,
    show_deposition: bool = True,
    initial_view: str = "Isométrica",
):
    import plotly.graph_objects as go

    if sections_df is None or points_df is None or sections_df.empty or points_df.empty:
        raise ValueError("No existen secciones/puntos suficientes para visualización 3D.")

    pts = _merge_profile(points_df, hydraulic_df, sediment_df)
    fig = go.Figure()

    # Terreno y secciones.
    for sid, g in pts.groupby("section_id"):
        g = g.sort_values("offset_m")
        pk = float(g["pk_m"].iloc[0])
        fig.add_trace(go.Scatter3d(
            x=[pk] * len(g),
            y=g["offset_m"],
            z=g["z_m"] * vertical_exaggeration,
            mode="lines",
            line=dict(width=3, color="saddlebrown"),
            name="Terreno/sección",
            showlegend=bool(sid == pts["section_id"].min()),
            hovertemplate="PK %{x:.1f} m<br>Offset %{y:.1f} m<br>Cota terreno %{z:.2f}<extra></extra>",
        ))

        if show_water and np.isfinite(g["cota_agua_m"]).any():
            wse = float(g["cota_agua_m"].dropna().iloc[0])
            wet = g[g["z_m"] <= wse]
            if len(wet) >= 2:
                fig.add_trace(go.Scatter3d(
                    x=[pk] * len(wet),
                    y=wet["offset_m"],
                    z=[wse * vertical_exaggeration] * len(wet),
                    mode="lines",
                    line=dict(width=5, color="deepskyblue"),
                    name="Lámina de agua",
                    showlegend=bool(sid == pts["section_id"].min()),
                    hovertemplate="PK %{x:.1f} m<br>Cota agua " + f"{wse:.2f} m" + "<extra></extra>",
                ))

        if show_scour and "cota_fondo_socavado_m" in g and np.isfinite(g["cota_fondo_socavado_m"]).any():
            zsc = float(g["cota_fondo_socavado_m"].dropna().iloc[0])
            # Marca fondo socavado como línea corta al centro de sección.
            center = g.iloc[(g["offset_m"].abs()).argmin()]
            fig.add_trace(go.Scatter3d(
                x=[pk, pk],
                y=[float(center["offset_m"]) - 2, float(center["offset_m"]) + 2],
                z=[zsc * vertical_exaggeration, zsc * vertical_exaggeration],
                mode="lines",
                line=dict(width=7, color="red"),
                name="Fondo socavado",
                showlegend=bool(sid == pts["section_id"].min()),
                hovertemplate="PK %{x:.1f} m<br>Cota fondo socavado %{z:.2f}<extra></extra>",
            ))

    # Perfil longitudinal de fondo.
    if not sections_df.empty:
        sec = sections_df.sort_values("pk_m")
        zcol = "cota_fondo_m" if "cota_fondo_m" in sec.columns else None
        if zcol:
            fig.add_trace(go.Scatter3d(
                x=sec["pk_m"],
                y=[0] * len(sec),
                z=sec[zcol] * vertical_exaggeration,
                mode="lines+markers",
                line=dict(width=5, color="black"),
                marker=dict(size=3),
                name="Perfil longitudinal fondo",
                hovertemplate="PK %{x:.1f} m<br>Fondo %{z:.2f}<extra></extra>",
            ))

    # Puntos críticos por Froude/Shields.
    crit = pts.copy()
    crit["critico"] = False
    if "Froude" in crit:
        crit["critico"] = crit["critico"] | (crit["Froude"] >= 0.8)
    if "Shields" in crit:
        crit["critico"] = crit["critico"] | (crit["Shields"] >= 0.047)
    crit = crit[crit["critico"]]
    if len(crit):
        fig.add_trace(go.Scatter3d(
            x=crit["pk_m"],
            y=crit["offset_m"],
            z=crit["z_m"] * vertical_exaggeration,
            mode="markers",
            marker=dict(size=4, color="orange", symbol="diamond"),
            name="Zona crítica hidráulica/sedimento",
            hovertemplate="PK %{x:.1f}<br>Offset %{y:.1f}<br>Froude %{customdata[0]:.2f}<br>Shields %{customdata[1]:.3f}<extra></extra>",
            customdata=np.stack([
                crit.get("Froude", pd.Series(np.nan, index=crit.index)).fillna(np.nan),
                crit.get("Shields", pd.Series(np.nan, index=crit.index)).fillna(np.nan),
            ], axis=1)
        ))

    fig.update_layout(
        title="Perfil longitudinal 3D con secciones, lámina de agua y fenómenos hidráulicos",
        scene=dict(
            xaxis_title="PK [m]",
            yaxis_title="Offset transversal [m]",
            zaxis_title=f"Cota x {vertical_exaggeration:g}",
            aspectmode="data",
        ),
        height=750,
        legend=dict(orientation="h"),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig




VIEW_CAMERAS_3D = {
    "Rotación libre": None,
    "Planta / superior": dict(eye=dict(x=0.0, y=0.0, z=2.8), up=dict(x=0, y=1, z=0), center=dict(x=0, y=0, z=0)),
    "Lateral": dict(eye=dict(x=0.0, y=2.6, z=0.15), up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0)),
    "Aguas arriba": dict(eye=dict(x=-2.6, y=0.0, z=0.35), up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0)),
    "Aguas abajo": dict(eye=dict(x=2.6, y=0.0, z=0.35), up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0)),
    "Isométrica": dict(eye=dict(x=1.65, y=1.65, z=1.25), up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0)),
}


def apply_3d_view(fig, view_name: str = "Isométrica"):
    """Aplica cámara inicial fija sin desactivar rotación libre interactiva."""
    cam = VIEW_CAMERAS_3D.get(view_name)
    if cam is not None:
        fig.update_layout(scene_camera=cam, uirevision="hidrosed_3d_free_rotation")
    else:
        fig.update_layout(uirevision="hidrosed_3d_free_rotation")
    return fig


def figure_to_html_bytes(fig) -> bytes:
    html = fig.to_html(include_plotlyjs="cdn", full_html=True)
    return html.encode("utf-8")


def create_section_selection_3d_figure(
    sections_df: pd.DataFrame,
    points_df: pd.DataFrame | None = None,
    modelable_df: pd.DataFrame | None = None,
    vertical_exaggeration: float = 1.0,
    show_accepted: bool = True,
    show_filled: bool = True,
    show_removed: bool = True,
    initial_view: str = "Isométrica",
):
    """Perfil longitudinal 3D previo de QA de secciones.

    Colores:
    - Verde: aceptadas/modelables.
    - Azul: rellenadas/sintéticas/relleno/interpoladas.
    - Rojo: eliminadas/no modelables/carga manual.
    """
    import plotly.graph_objects as go

    if sections_df is None or sections_df.empty:
        raise ValueError("No existen secciones para visualizar.")

    sec = sections_df.copy()
    if "section_id" not in sec.columns:
        sec["section_id"] = range(1, len(sec)+1)

    # Merge modelable information if available.
    if modelable_df is not None and not modelable_df.empty and "section_id" in modelable_df.columns:
        md = modelable_df.copy()
        md["section_id_str"] = md["section_id"].astype(str)
        sec["section_id_str"] = sec["section_id"].astype(str)
        keep = [c for c in [
            "section_id_str", "seleccion_modelacion", "estado_modelacion",
            "observacion_modelacion", "n_puntos_izquierda", "n_puntos_derecha",
            "n_puntos_total", "cota_para_grafico_m", "elevacion_m"
        ] if c in md.columns]
        sec = sec.merge(md[keep], on="section_id_str", how="left")
    else:
        sec["seleccion_modelacion"] = True
        sec["estado_modelacion"] = "ACEPTADA"
        sec["observacion_modelacion"] = ""

    # Determine PK and z.
    if "pk_m" not in sec.columns:
        if "chainage_m" in sec.columns:
            sec["pk_m"] = pd.to_numeric(sec["chainage_m"], errors="coerce")
        elif "km_eje" in sec.columns:
            sec["pk_m"] = pd.to_numeric(sec["km_eje"], errors="coerce") * 1000
        else:
            sec["pk_m"] = range(len(sec))

    z_candidates = ["cota_fondo_m", "cota_min_m", "cota_eje_estimada_m", "cota_para_grafico_m", "elevacion_m"]
    z_col = next((c for c in z_candidates if c in sec.columns), None)
    if z_col is None:
        sec["z_plot_m"] = 0.0
    else:
        sec["z_plot_m"] = pd.to_numeric(sec[z_col], errors="coerce")
        if sec["z_plot_m"].isna().all():
            sec["z_plot_m"] = 0.0
        else:
            sec["z_plot_m"] = sec["z_plot_m"].interpolate(limit_direction="both").fillna(sec["z_plot_m"].median())

    # Determine status.
    def _status(row):
        sel = bool(row.get("seleccion_modelacion", True)) if not pd.isna(row.get("seleccion_modelacion", True)) else True
        origen = str(row.get("origen", "")).lower()
        estado = str(row.get("estado_modelacion", "")).lower()
        if not sel or "carga" in estado or "descart" in estado or "elimin" in estado:
            return "Eliminada"
        if "rell" in origen or "interpol" in origen or "fallback" in origen or "sint" in origen or "rell" in estado:
            return "Rellenada"
        return "Aceptada"

    sec["estado_vista"] = sec.apply(_status, axis=1)
    fig = go.Figure()

    cfg = {
        "Aceptada": {"color": "green", "show": bool(show_accepted), "name": "Aceptadas correctamente"},
        "Rellenada": {"color": "royalblue", "show": bool(show_filled), "name": "Rellenadas/interpoladas"},
        "Eliminada": {"color": "red", "show": bool(show_removed), "name": "Eliminadas/no modelables"},
    }

    # Perfil longitudinal base
    sec_sorted = sec.sort_values("pk_m")
    fig.add_trace(go.Scatter3d(
        x=sec_sorted["pk_m"],
        y=[0]*len(sec_sorted),
        z=sec_sorted["z_plot_m"] * vertical_exaggeration,
        mode="lines",
        line=dict(color="gray", width=4),
        name="Perfil longitudinal preliminar",
        showlegend=True,
        hovertemplate="PK %{x:.1f} m<br>Cota %{z:.2f}<extra></extra>",
    ))

    # Section lines with colors.
    for status, meta in cfg.items():
        if not meta["show"]:
            continue
        sub = sec[sec["estado_vista"] == status].copy()
        first = True
        for _, row in sub.iterrows():
            pk = float(row.get("pk_m", 0.0))
            z = float(row.get("z_plot_m", 0.0)) * vertical_exaggeration
            width = row.get("ancho_m", row.get("width_m", row.get("ancho_referencia_m", 40.0)))
            try:
                width = float(width)
                if not np.isfinite(width) or width <= 0:
                    width = 40.0
            except Exception:
                width = 40.0
            obs = str(row.get("observacion_modelacion", ""))
            fig.add_trace(go.Scatter3d(
                x=[pk, pk],
                y=[-width/2, width/2],
                z=[z, z],
                mode="lines+markers",
                line=dict(color=meta["color"], width=7),
                marker=dict(size=3, color=meta["color"]),
                name=meta["name"],
                showlegend=bool(first),
                customdata=[[str(row.get("section_id", "")), obs]]*2,
                hovertemplate="Sección %{customdata[0]}<br>PK %{x:.1f} m<br>Estado " + status + "<br>%{customdata[1]}<extra></extra>",
            ))
            first = False

    # Optional terrain/profile points
    if points_df is not None and not points_df.empty and {"section_id", "offset_m", "z_m"}.issubset(points_df.columns):
        pts = points_df.copy()
        if "pk_m" not in pts.columns:
            if "chainage_m" in pts.columns:
                pts["pk_m"] = pd.to_numeric(pts["chainage_m"], errors="coerce")
            else:
                pk_lookup = sec.set_index(sec["section_id"].astype(str))["pk_m"].to_dict()
                pts["pk_m"] = pts["section_id"].astype(str).map(pk_lookup)
        pts = pts.dropna(subset=["pk_m", "offset_m", "z_m"])
        if not pts.empty:
            sample = pts if len(pts) <= 5000 else pts.sample(5000, random_state=42)
            fig.add_trace(go.Scatter3d(
                x=sample["pk_m"],
                y=sample["offset_m"],
                z=sample["z_m"] * vertical_exaggeration,
                mode="markers",
                marker=dict(size=2, color="saddlebrown", opacity=0.45),
                name="Puntos topográficos de secciones",
                showlegend=True,
                hovertemplate="PK %{x:.1f}<br>Offset %{y:.1f}<br>Cota %{z:.2f}<extra></extra>",
            ))

    fig.update_layout(
        title="Perfil longitudinal 3D previo · QA de secciones seleccionadas",
        scene=dict(
            xaxis_title="PK [m]",
            yaxis_title="Offset transversal [m]",
            zaxis_title=f"Cota x {vertical_exaggeration:g}",
            aspectmode="data",
        ),
        height=720,
        legend=dict(orientation="h"),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    fig = apply_3d_view(fig, initial_view)
    return fig

