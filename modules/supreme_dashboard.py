from __future__ import annotations
import pandas as pd

CSS = """
<style>
:root { --hs-blue:#0b3d5c; --hs-cyan:#00a6d6; --hs-green:#12b886; --hs-red:#e03131; --hs-gold:#f59f00; }
.block-container {padding-top: 1.2rem; padding-bottom: 3rem;}
.hs-hero {background: linear-gradient(135deg,#061826 0%,#0b3d5c 45%,#057a9b 100%); border-radius:24px; padding:26px 30px; color:white; box-shadow:0 10px 30px rgba(0,0,0,.25); margin-bottom: 1rem;}
.hs-hero h1 {font-size: 2.2rem; margin:0; letter-spacing:.3px;}
.hs-hero p {opacity:.93; font-size:1.02rem; margin-top:.5rem; max-width: 1100px;}
.hs-card {background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.16); border-radius:18px; padding:16px 18px; margin:4px 0;}
.hs-kpi {border-radius:18px; padding:15px 16px; border:1px solid rgba(120,120,120,.18); background:linear-gradient(180deg,rgba(255,255,255,.06),rgba(255,255,255,.02)); box-shadow:0 6px 18px rgba(0,0,0,.08);}
.hs-kpi .label {font-size:.78rem; opacity:.75; text-transform:uppercase; letter-spacing:.08em;}
.hs-kpi .value {font-size:1.55rem; font-weight:800; margin-top:.2rem;}
.hs-kpi .sub {font-size:.78rem; opacity:.70; margin-top:.1rem;}
.hs-pill {display:inline-block; padding:5px 10px; border-radius:999px; background:#e7f5ff; color:#0b3d5c; margin:2px 4px 2px 0; font-size:.80rem; font-weight:600;}
.hs-alert {border-left:5px solid var(--hs-gold); padding:10px 14px; background:#fff9db; border-radius:10px; margin:.5rem 0;}
</style>
"""

def confidence_label(score: float) -> str:
    if score >= 9.0: return "Excelente"
    if score >= 8.8: return "Aprobado alto"
    if score >= 8.0: return "Alto en revisión"
    if score >= 7.0: return "Medio"
    return "Insuficiente"

def confidence_color(score: float) -> str:
    if score >= 8.8: return "#12b886"
    if score >= 8.0: return "#f59f00"
    return "#e03131"

def kpi_html(label: str, value: str, sub: str = "", score: float | None = None) -> str:
    border = "rgba(120,120,120,.18)" if score is None else confidence_color(score)
    return f"""<div class='hs-kpi' style='border-color:{border}'><div class='label'>{label}</div><div class='value'>{value}</div><div class='sub'>{sub}</div></div>"""

def global_confidence_report(scores: dict) -> pd.DataFrame:
    rows=[]
    for k,v in scores.items():
        try: s=float(v)
        except Exception: s=0.0
        rows.append({"bloque": k, "confianza": round(s,2), "estado": confidence_label(s)})
    return pd.DataFrame(rows)
