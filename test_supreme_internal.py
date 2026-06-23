from pathlib import Path
import sys, math, zipfile, io
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from modules.roughness_engine import cowan_n, suggested_roughness, compose_roughness_manual
from modules.synthetic_trapezoid_sections import generate_trapezoid_reach_sections, trapezoid_capacity_table
from modules.granulometry_kmz import normalize_granulometry_table, validate_granulometry, assign_granulometry_to_sections
from modules.hydrologic_transfer_dual import transfer_flow_area_altitude_distance, rank_hydrometric_stations
from modules.hydraulic_hecras_like import hecras_like_steady_profile, sediment_from_hecras_profile
from modules.tiled_contours import split_bbox_km2_strategy
from modules.opentopo_tiled_download import split_bbox, recommended_tiling


def q_design():
    return pd.DataFrame({'T_anios':[2,5,10,25,50,100], 'Q_m3s':[5,8,12,20,30,45]})


def test_01_compile_imports():
    import py_compile
    py_compile.compile(str(Path(__file__).parent/'app.py'), doraise=True)
    return True, 'app.py compila OK; streamlit se instala desde requirements en Streamlit Cloud'


def test_02_roughness_cowan():
    r=cowan_n('grava_media','moderada','ocasional','menor','baja','media')
    assert 0.02 <= r['n_manning'] <= 0.09
    return True, f"n={r['n_manning']:.3f}"


def test_03_roughness_suggested():
    df=suggested_roughness('grava_media', d50_m=0.04, d84_m=0.10)
    assert len(df)>=3 and df['n_adoptado_recomendado'].notna().any()
    return True, f"n_adopt={df['n_adoptado_recomendado'].dropna().iloc[0]:.3f}"


def test_04_trapezoid_sections():
    sec, pts=generate_trapezoid_reach_sections(1000,100,6,2,1.5,1.5,0.008)
    assert len(sec)==11 and len(pts)>30 and pts['z_m'].notna().all()
    return True, f"secciones={len(sec)} puntos={len(pts)}"


def test_05_trapezoid_capacity():
    cap=trapezoid_capacity_table([5,15,30],6,2,1.5,1.5,0.008,0.04)
    assert len(cap)==3 and cap['y_normal_m'].gt(0).all()
    return True, f"ymax={cap['y_normal_m'].max():.2f}"


def test_06_hydraulic_with_trapezoid():
    sec, pts=generate_trapezoid_reach_sections(500,100,8,2.5,1.5,1.5,0.006)
    prof=hecras_like_steady_profile(sec, pts, q_design(), n_manning=0.04, slope_energy=0.006)
    assert len(prof)>0 and prof['cota_agua_m'].notna().all()
    # physical guard: no absurd water surface jump > 80 m in synthetic reach
    assert prof.groupby('T_anios')['cota_agua_m'].agg(lambda s: s.max()-s.min()).max() < 80
    return True, f"filas={len(prof)}"


def test_07_sediment_from_profile():
    sec, pts=generate_trapezoid_reach_sections(500,100,8,2.5,1.5,1.5,0.006)
    prof=hecras_like_steady_profile(sec, pts, q_design(), n_manning=0.04, slope_energy=0.006)
    sed=sediment_from_hecras_profile(prof, d50_m=0.04, d90_m=0.12, slope_energy=0.006)
    assert len(sed)==len(prof) and sed['Shields'].notna().any()
    return True, f"sed filas={len(sed)}"


def test_08_granulometry_assign():
    sec, pts=generate_trapezoid_reach_sections(1000,100,6,2,1.5,1.5,0.008)
    raw=pd.DataFrame({'id_muestra':['G1','G2','G3'], 'D50':[30,45,60], 'D84':[70,90,120], 'D90':[100,130,160], 'unidad':['mm','mm','mm'], 'pk_m':[0,500,1000]})
    g=normalize_granulometry_table(raw)
    val=validate_granulometry(g)
    ass=assign_granulometry_to_sections(sec,g)
    assert val['ok_orden_granulometrico'].all() and len(ass)==len(sec) and ass['D50_m'].notna().all()
    return True, f"asignadas={len(ass)}"


def test_09_transfer_dual():
    tr=transfer_flow_area_altitude_distance(100, 200, 150, 800, 600, 25, 0.75)
    assert tr['valido'] and tr['Q_transferido_m3s']>0 and tr['confianza_transferencia']>=8
    return True, f"Qtr={tr['Q_transferido_m3s']:.2f}"


def test_10_tiling_logic():
    bbox={'south':-31.0,'north':-29.0,'west':-72.0,'east':-70.0}
    tiles=split_bbox(bbox,3,3)
    rec=recommended_tiling(15000)
    strat=split_bbox_km2_strategy(15000)
    assert len(tiles)==9 and rec['mode']=='tiled' and strat['tile_rows']>=3
    return True, f"tiles={len(tiles)} rec={rec}"


def run_all():
    funcs=[test_01_compile_imports,test_02_roughness_cowan,test_03_roughness_suggested,test_04_trapezoid_sections,test_05_trapezoid_capacity,test_06_hydraulic_with_trapezoid,test_07_sediment_from_profile,test_08_granulometry_assign,test_09_transfer_dual,test_10_tiling_logic]
    rows=[]
    for i,fn in enumerate(funcs,1):
        try:
            ok,msg=fn()
        except Exception as e:
            ok=False; msg=f"{type(e).__name__}: {e}"
        rows.append({'corrida':i, 'prueba':fn.__name__, 'resultado':'OK' if ok else 'FALLA', 'detalle':msg})
    return pd.DataFrame(rows)

if __name__=='__main__':
    df=run_all()
    print(df.to_string(index=False))
    if not (df['resultado']=='OK').all():
        raise SystemExit(1)
