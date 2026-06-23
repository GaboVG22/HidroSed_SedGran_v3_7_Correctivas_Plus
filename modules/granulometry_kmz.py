from __future__ import annotations

import io, zipfile, re, math
import pandas as pd
import numpy as np
from xml.etree import ElementTree as ET


def read_kmz_or_kml_to_text(uploaded_file) -> str:
    data = uploaded_file.read() if hasattr(uploaded_file, "read") else bytes(uploaded_file)
    name = getattr(uploaded_file, "name", "").lower()
    if name.endswith(".kmz") or data[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            kmls = [n for n in z.namelist() if n.lower().endswith(".kml")]
            if not kmls:
                raise ValueError("KMZ sin KML interno.")
            return z.read(kmls[0]).decode("utf-8", errors="ignore")
    return data.decode("utf-8", errors="ignore")


def _ns(tag):
    return tag.split("}",1)[0]+"}" if tag.startswith("{") else ""


def parse_granulometry_points(kml_text: str) -> pd.DataFrame:
    root = ET.fromstring(kml_text.encode("utf-8"))
    ns = _ns(root.tag)
    rows=[]
    for pm in root.iterfind(f".//{ns}Placemark"):
        name_el = pm.find(f"{ns}name")
        name = name_el.text.strip() if name_el is not None and name_el.text else f"muestra_{len(rows)+1}"
        coord_el = pm.find(f".//{ns}Point/{ns}coordinates")
        if coord_el is None or not coord_el.text:
            continue
        first = coord_el.text.strip().split()[0]
        parts = first.split(",")
        if len(parts) < 2: continue
        lon, lat = float(parts[0]), float(parts[1])
        desc_el = pm.find(f"{ns}description")
        desc = desc_el.text if desc_el is not None and desc_el.text else ""
        # id_muestra desde nombre o descripción
        id_m = name
        m = re.search(r"id[_\s-]*muestra\s*[:=]\s*([A-Za-z0-9_\-]+)", desc, re.I)
        if m: id_m = m.group(1)
        rows.append({"id_muestra": id_m, "nombre_kmz": name, "lat": lat, "lon": lon, "descripcion": desc})
    return pd.DataFrame(rows)


def normalize_granulometry_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    if "id_muestra" not in out.columns:
        out["id_muestra"] = [f"G{i+1}" for i in range(len(out))]
    for c in ["D5","D10","D16","D25","D30","D50","D60","D75","D84","D90","D95"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    unidad = str(out.get("unidad", pd.Series(["mm"])).iloc[0]).lower() if len(out) else "mm"
    factor = 0.001 if "mm" in unidad else 1.0
    for c in ["D5","D10","D16","D25","D30","D50","D60","D75","D84","D90","D95"]:
        if c in out.columns:
            out[c+"_m"] = out[c] * factor
    return out


def validate_granulometry(df: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for _, r in df.iterrows():
        vals=[]
        for c in ["D10_m","D16_m","D30_m","D50_m","D84_m","D90_m"]:
            if c in df.columns and pd.notna(r.get(c)):
                vals.append((c, float(r[c])))
        ok_order = all(vals[i][1] <= vals[i+1][1] for i in range(len(vals)-1)) if len(vals)>1 else True
        ok_positive = all(v>0 for _,v in vals)
        rows.append({"id_muestra": r.get("id_muestra"), "ok_orden_granulometrico": ok_order, "ok_positivo": ok_positive, "n_diametros": len(vals), "confianza_granulometria": 9.0 if ok_order and ok_positive and len(vals)>=3 else 7.0})
    return pd.DataFrame(rows)


def assign_granulometry_to_sections(sections_df: pd.DataFrame, granulometry_df: pd.DataFrame, sample_pk_col: str = "pk_m") -> pd.DataFrame:
    if sections_df is None or len(sections_df)==0 or granulometry_df is None or len(granulometry_df)==0:
        return pd.DataFrame()
    sec = sections_df[["section_id","pk_m"]].copy()
    g = granulometry_df.copy()
    if sample_pk_col not in g.columns:
        # fallback: evenly spread samples along reach
        maxpk = float(sec["pk_m"].max()) if len(sec) else 0
        g[sample_pk_col] = np.linspace(0, maxpk, len(g))
    out = sec.copy()
    for d in ["D50_m","D84_m","D90_m","D95_m"]:
        if d in g.columns:
            valid = g[[sample_pk_col,d]].dropna().sort_values(sample_pk_col)
            if len(valid)==1:
                out[d] = float(valid[d].iloc[0])
            elif len(valid)>1:
                out[d] = np.interp(out["pk_m"], valid[sample_pk_col].to_numpy(float), valid[d].to_numpy(float))
    out["fuente_granulometria"] = "interpolacion longitudinal por PK"
    return out
