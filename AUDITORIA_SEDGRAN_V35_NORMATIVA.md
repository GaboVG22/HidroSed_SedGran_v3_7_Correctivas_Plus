# Auditoría HidroSed SedGran v3.5 Normativa DGA/MC/IDF/Pmax123

## Resultado

- Compilación Python: OK
- Smoke test hidrología normativa: OK
- Puntaje de prueba hidrología normativa: 9.15/10

## Chequeos

- [OK] 01_compila_python
- [OK] 02_version_v35
- [OK] 03_manual_dga_metodos
- [OK] 04_manual_carreteras_sensibilidad
- [OK] 05_idf_k_id_i24
- [OK] 06_pmax_123
- [OK] 07_hidrogramas_linsley_gray
- [OK] 08_cauda_minimos
- [OK] 09_validacion_rangos
- [OK] 10_app_ui_compacta
- [OK] 11_q_design_normativo
- [OK] 12_excel_v35
- [OK] 13_score_9
- [OK] 14_base_dga_sigue
- [OK] 15_isoyetas_sigue

## Metodología corregida

- DGA-AC pluvial con validación automática de rango 20–10.000 km².
- Verni-King modificado como método regional paramétrico auditable.
- Racional con mayoración automática de C para T=25, 50 y 100+.
- Racional modificado con abatimiento espacial.
- IDF desde P24 con coeficientes de duración k=Id/I24 y coeficientes de frecuencia T/T10.
- P24, P48 y P72.
- Hidrogramas unitarios sintéticos Linsley y Gray.
- DGA-AC deshielo/nival con validación 50–6.000 km² y T≤100.
- Caudales mínimos DGA 30, 7 y 1 día como módulo preliminar trazable.
- QA hidrológico con estados OK, ADVERTENCIA, NO_APLICA, OK_PRELIMINAR y REVISAR.
- La adopción normativa puede ser envolvente máxima, mediana adoptable o promedio adoptable.

## Nota técnica

La aplicación queda ordenada manteniendo la hidrología normativa dentro de una sección compacta con pestañas. Los coeficientes regionales quedan auditables/editables porque los manuales usan zonas homogéneas, mapas e isoyetas que deben seleccionarse según ubicación y disponibilidad de datos.
