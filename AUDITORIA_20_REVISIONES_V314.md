# Auditoría 20 revisiones — HidroSed v3.1.4 BBox demcop30 integrado

- [OK] 01_compilacion_python
- [OK] 02_misma_base_v311
- [OK] 03_version_v314
- [OK] 04_bbox_democop30_integrado
- [OK] 05_preajustes_tamano_cuenca
- [OK] 06_unidad_km_default
- [OK] 07_margen_por_perfil
- [OK] 08_area_bbox_no_cuenca
- [OK] 09_advertencia_bbox_ratio
- [OK] 10_dem_manual_opcional
- [OK] 11_opentopo_mantiene
- [OK] 12_descarga_por_teselas
- [OK] 13_antisnap_area_controlled
- [OK] 14_defaults_cuenca_desde_bbox
- [OK] 15_radio_automatico_perfil
- [OK] 16_no_mensaje_solo_20km2
- [OK] 17_curvas_siguen
- [OK] 18_allsegs
- [OK] 19_qa_candidatos
- [OK] 20_readme_actualizado

## Nota

Se integró la lógica de la aplicación demcop30_streamlit dentro de la aplicación v3.1.1:
- El BBox se controla por preajuste de tamaño esperado.
- El BBox se explica como ventana DEM, no área de cuenca.
- La delimitación Anti-Snap se conserva en la pestaña 3.
- La generación de curvas/KMZ se conserva.
