
from __future__ import annotations

import io
import math
import re
from typing import Any
import numpy as np
import pandas as pd


TARGET_DIAMETERS = [5, 10, 16, 25, 30, 35, 50, 60, 65, 75, 84, 90, 95]


DEFAULT_PROFILES_MM = {
    "Arena fina / cauce arenoso": {
        "descripcion": "Lecho dominado por arena fina a media. Alta movilidad; usar con precaución en cauces de montaña.",
        "material": "arena fina-media",
        "D5": 0.08, "D10": 0.12, "D16": 0.16, "D25": 0.22, "D30": 0.28,
        "D35": 0.35, "D50": 0.55, "D60": 0.75, "D65": 0.90, "D75": 1.30,
        "D84": 1.80, "D90": 2.40, "D95": 3.20,
    },
    "Arena media-gruesa": {
        "descripcion": "Lecho arenoso grueso con fracción gravilla. Útil como perfil preliminar en esteros de baja pendiente.",
        "material": "arena gruesa-gravilla",
        "D5": 0.20, "D10": 0.30, "D16": 0.45, "D25": 0.70, "D30": 0.90,
        "D35": 1.10, "D50": 1.80, "D60": 2.60, "D65": 3.20, "D75": 4.80,
        "D84": 6.50, "D90": 9.00, "D95": 13.0,
    },
    "Grava fina": {
        "descripcion": "Lecho de grava fina con arena. Perfil preliminar para quebradas con transporte activo moderado.",
        "material": "grava fina",
        "D5": 0.70, "D10": 1.20, "D16": 2.00, "D25": 3.20, "D30": 4.00,
        "D35": 5.00, "D50": 8.00, "D60": 11.0, "D65": 13.0, "D75": 18.0,
        "D84": 25.0, "D90": 35.0, "D95": 50.0,
    },
    "Grava media": {
        "descripcion": "Lecho de grava media. Buen perfil por defecto para cauces aluviales de montaña media.",
        "material": "grava media",
        "D5": 2.00, "D10": 4.00, "D16": 6.00, "D25": 9.00, "D30": 11.0,
        "D35": 14.0, "D50": 22.0, "D60": 30.0, "D65": 36.0, "D75": 50.0,
        "D84": 70.0, "D90": 95.0, "D95": 130.0,
    },
    "Grava gruesa / bolones pequeños": {
        "descripcion": "Lecho de grava gruesa con bolones pequeños. Mayor resistencia a movilidad incipiente.",
        "material": "grava gruesa-bolones",
        "D5": 5.00, "D10": 9.00, "D16": 14.0, "D25": 22.0, "D30": 28.0,
        "D35": 35.0, "D50": 55.0, "D60": 75.0, "D65": 90.0, "D75": 125.0,
        "D84": 180.0, "D90": 250.0, "D95": 350.0,
    },
    "Mixto aluvial semiárido": {
        "descripcion": "Perfil mixto preliminar para cauces aluviales semiáridos con arenas, gravas y bolones.",
        "material": "mixto aluvial",
        "D5": 0.50, "D10": 1.00, "D16": 2.50, "D25": 6.00, "D30": 8.00,
        "D35": 11.0, "D50": 28.0, "D60": 45.0, "D65": 58.0, "D75": 95.0,
        "D84": 150.0, "D90": 230.0, "D95": 380.0,
    },
}


def _as_float(x):
    try:
        if pd.isna(x):
            return np.nan
        return float(str(x).replace(",", "."))
    except Exception:
        return np.nan


def _clean_col(c: Any) -> str:
    s = str(c).strip().lower()
    for a, b in {"á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u", "%": "porcentaje"}.items():
        s = s.replace(a, b)
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def _find_col(cols, options):
    cols_l = list(cols)
    for opt in options:
        if opt in cols_l:
            return opt
    for c in cols_l:
        if any(opt in c for opt in options):
            return c
    return None


def default_profiles_dataframe() -> pd.DataFrame:
    rows = []
    for name, data in DEFAULT_PROFILES_MM.items():
        row = {"perfil": name, "material": data.get("material"), "descripcion": data.get("descripcion")}
        for p in TARGET_DIAMETERS:
            row[f"D{p}_mm"] = data.get(f"D{p}", np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def classify_material(d50_mm: float, d84_mm: float | None = None) -> str:
    if not np.isfinite(d50_mm):
        return "sin clasificación"
    if d50_mm < 0.063:
        return "limo/arcilla"
    if d50_mm < 2.0:
        return "arena"
    if d50_mm < 16.0:
        return "grava fina"
    if d50_mm < 64.0:
        return "grava media-gruesa"
    if d50_mm < 256.0:
        return "bolones"
    return "bloques"


def _complete_metrics(row_mm: dict[str, float]) -> dict[str, float]:
    out = {}
    for p in TARGET_DIAMETERS:
        val = row_mm.get(f"D{p}", row_mm.get(f"D{p}_mm", np.nan))
        out[f"D{p}_mm"] = _as_float(val)
        out[f"D{p}_m"] = out[f"D{p}_mm"] / 1000.0 if np.isfinite(out[f"D{p}_mm"]) else np.nan

    d16 = out.get("D16_mm", np.nan)
    d84 = out.get("D84_mm", np.nan)
    if np.isfinite(d16) and np.isfinite(d84) and d16 > 0 and d84 > 0:
        dm = math.sqrt(d16 * d84)
    else:
        vals = [out.get(f"D{p}_mm", np.nan) for p in [16, 30, 50, 60, 84]]
        vals = [v for v in vals if np.isfinite(v) and v > 0]
        dm = float(np.exp(np.mean(np.log(vals)))) if vals else np.nan
    out["Dm_mm"] = dm
    out["Dm_m"] = dm / 1000.0 if np.isfinite(dm) else np.nan

    d10, d30, d60 = out.get("D10_mm", np.nan), out.get("D30_mm", np.nan), out.get("D60_mm", np.nan)
    out["Cu"] = d60 / d10 if np.isfinite(d60) and np.isfinite(d10) and d10 > 0 else np.nan
    out["Cc"] = (d30**2) / (d10 * d60) if all(np.isfinite(v) and v > 0 for v in [d10, d30, d60]) else np.nan
    out["clasificacion"] = classify_material(out.get("D50_mm", np.nan), out.get("D84_mm", np.nan))
    return out


def profile_to_characteristics(profile_name: str) -> dict[str, Any]:
    if profile_name not in DEFAULT_PROFILES_MM:
        profile_name = list(DEFAULT_PROFILES_MM.keys())[0]
    data = DEFAULT_PROFILES_MM[profile_name]
    metrics = _complete_metrics(data)
    metrics["perfil"] = profile_name
    metrics["material"] = data.get("material", "")
    metrics["descripcion"] = data.get("descripcion", "")
    metrics["fuente"] = "perfil_tipo"
    metrics["confianza_granulometria"] = 0.55
    return metrics


def read_excel_any(uploaded) -> dict[str, pd.DataFrame]:
    name = getattr(uploaded, "name", "").lower()
    data = uploaded.read()
    bio = io.BytesIO(data)
    if name.endswith(".csv"):
        return {"CSV": pd.read_csv(bio)}
    return {str(k): v for k, v in pd.read_excel(bio, sheet_name=None).items() if v is not None and not v.empty}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [_clean_col(c) for c in out.columns]
    return out


def _characteristics_from_d_columns(df: pd.DataFrame) -> pd.DataFrame:
    norm = _normalize_columns(df)
    rows = []
    for i, r in norm.iterrows():
        row = {}
        for p in TARGET_DIAMETERS:
            col = _find_col(norm.columns, [f"d{p}", f"d{p}_mm", f"d_{p}", f"d_{p}_mm"])
            row[f"D{p}_mm"] = _as_float(r[col]) if col else np.nan
        if any(np.isfinite(row.get(f"D{p}_mm", np.nan)) for p in TARGET_DIAMETERS):
            id_col = _find_col(norm.columns, ["id_muestra", "muestra", "sample", "codigo"])
            pk_col = _find_col(norm.columns, ["pk_m", "pk", "km", "progresiva"])
            row["id_muestra"] = r[id_col] if id_col else f"M{i+1}"
            row["PK_m"] = _as_float(r[pk_col]) if pk_col else np.nan
            row.update(_complete_metrics(row))
            rows.append(row)
    return pd.DataFrame(rows)


def _characteristics_from_sieve_curve(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    norm = _normalize_columns(df)
    size_col = _find_col(norm.columns, ["abertura_mm", "tamiz_mm", "diametro_mm", "diametro", "d_mm", "size_mm", "tamiz"])
    pass_col = _find_col(norm.columns, ["porcentaje_pasa", "pasa", "porcentaje_que_pasa", "pct_pasa", "passing", "percent_passing"])
    retained_col = _find_col(norm.columns, ["porcentaje_retenido", "retenido", "pct_retenido", "percent_retained"])
    weight_col = _find_col(norm.columns, ["peso_retenido_g", "peso_retenido", "peso", "weight_retained"])
    if size_col is None:
        return pd.DataFrame(), pd.DataFrame()

    curve = pd.DataFrame()
    curve["diametro_mm"] = pd.to_numeric(norm[size_col].map(_as_float), errors="coerce")
    if pass_col is not None:
        curve["porcentaje_pasa"] = pd.to_numeric(norm[pass_col].map(_as_float), errors="coerce")
    elif retained_col is not None:
        retained = pd.to_numeric(norm[retained_col].map(_as_float), errors="coerce").fillna(0)
        curve["porcentaje_pasa"] = 100.0 - retained.cumsum()
    elif weight_col is not None:
        w = pd.to_numeric(norm[weight_col].map(_as_float), errors="coerce").fillna(0)
        total = float(w.sum())
        if total <= 0:
            return pd.DataFrame(), pd.DataFrame()
        curve["porcentaje_pasa"] = 100.0 - (100.0 * w / total).cumsum()
    else:
        return pd.DataFrame(), pd.DataFrame()

    curve = curve[np.isfinite(curve["diametro_mm"]) & np.isfinite(curve["porcentaje_pasa"])]
    curve = curve[curve["diametro_mm"] > 0]
    if curve.empty:
        return pd.DataFrame(), pd.DataFrame()

    curve["porcentaje_pasa"] = curve["porcentaje_pasa"].clip(0, 100)
    curve = curve.groupby("diametro_mm", as_index=False)["porcentaje_pasa"].mean()
    curve = curve.sort_values("porcentaje_pasa")
    p = curve["porcentaje_pasa"].to_numpy(dtype=float)
    d = curve["diametro_mm"].to_numpy(dtype=float)
    p_unique, idx = np.unique(p, return_index=True)
    p = p_unique
    d = d[idx]
    row = {}
    if len(p) >= 2:
        logd = np.log10(d)
        for target in TARGET_DIAMETERS:
            row[f"D{target}_mm"] = float(10 ** np.interp(target, p, logd)) if np.nanmin(p) <= target <= np.nanmax(p) else np.nan
    row.update(_complete_metrics(row))
    row["id_muestra"] = "curva_excel"
    return pd.DataFrame([row]), curve.sort_values("diametro_mm")


def extract_granulometry_from_excel(uploaded) -> dict[str, Any]:
    sheets = read_excel_any(uploaded)
    all_chars, curves, diagnostics = [], {}, []
    for sheet, df in sheets.items():
        chars_direct = _characteristics_from_d_columns(df)
        if not chars_direct.empty:
            chars_direct["hoja"] = sheet
            all_chars.append(chars_direct)
            diagnostics.append(f"{sheet}: diámetros característicos detectados.")
            continue
        chars_curve, curve = _characteristics_from_sieve_curve(df)
        if not chars_curve.empty:
            chars_curve["hoja"] = sheet
            all_chars.append(chars_curve)
            curves[sheet] = curve
            diagnostics.append(f"{sheet}: curva granulométrica interpretada.")
    if not all_chars:
        return {"ok": False, "characteristics": None, "samples": pd.DataFrame(), "curves": {}, "diagnostics": ["No se detectaron columnas válidas de granulometría."]}

    chars = pd.concat(all_chars, ignore_index=True)
    adopt = {}
    for p in TARGET_DIAMETERS:
        col = f"D{p}_mm"
        adopt[col] = float(chars[col].dropna().median()) if col in chars and not chars[col].dropna().empty else np.nan
    adopt.update(_complete_metrics(adopt))
    adopt["perfil"] = "Excel granulometría real"
    adopt["material"] = adopt["clasificacion"]
    adopt["descripcion"] = "Diámetros característicos calculados desde Excel/CSV cargado."
    adopt["fuente"] = "excel_usuario"
    adopt["confianza_granulometria"] = 0.90
    return {"ok": True, "characteristics": adopt, "samples": chars, "curves": curves, "diagnostics": diagnostics}


def characteristic_table(metrics: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for key in ["D5", "D10", "D16", "D25", "D30", "D35", "D50", "D60", "D65", "D75", "D84", "D90", "D95", "Dm"]:
        rows.append({"diametro": key, "mm": metrics.get(f"{key}_mm", np.nan), "m": metrics.get(f"{key}_m", np.nan)})
    rows.append({"diametro": "Cu", "mm": metrics.get("Cu", np.nan), "m": np.nan})
    rows.append({"diametro": "Cc", "mm": metrics.get("Cc", np.nan), "m": np.nan})
    return pd.DataFrame(rows)


def method_diameter_table(metrics: dict[str, Any]) -> pd.DataFrame:
    rows = [
        ("Shields movilidad incipiente", "D50", "Inicio de movimiento del material dominante"),
        ("Manning-Strickler rugosidad", "D50 / D84", "Rugosidad por tamaño representativo o material grueso"),
        ("Meyer-Peter & Müller", "D50 / Dm", "Transporte de fondo"),
        ("HEC-18 socavación general", "D50", "Material de lecho representativo"),
        ("Socavación local / protección", "D50 / D84", "Mayor robustez con fracción gruesa"),
        ("Acorazamiento del lecho", "D84 / D90", "Fracción gruesa y estabilidad superficial"),
        ("Depositación / finos móviles", "D16 / D30", "Fracción fina disponible para transporte"),
        ("Yang / Engelund-Hansen", "D50", "Transporte total o arenoso"),
    ]
    out = []
    for method, dia, note in rows:
        values = []
        for token in dia.replace(" ", "").split("/"):
            val = metrics.get(f"{token}_mm", metrics.get(token, np.nan))
            if isinstance(val, (int, float)) and np.isfinite(val):
                values.append(f"{token}={val:.2f} mm")
        out.append({"metodologia": method, "diametro_usado": dia, "valor_disponible": "; ".join(values), "criterio": note})
    return pd.DataFrame(out)


def profile_curve_dataframe(metrics: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for p in TARGET_DIAMETERS:
        d = metrics.get(f"D{p}_mm", np.nan)
        if np.isfinite(d):
            rows.append({"diametro_mm": d, "porcentaje_pasa": float(p)})
    return pd.DataFrame(rows).sort_values("diametro_mm")


def confidence_label(metrics: dict[str, Any]) -> str:
    c = float(metrics.get("confianza_granulometria", 0.0) or 0.0)
    if c >= 0.85:
        return "alta"
    if c >= 0.65:
        return "media"
    return "preliminar"
