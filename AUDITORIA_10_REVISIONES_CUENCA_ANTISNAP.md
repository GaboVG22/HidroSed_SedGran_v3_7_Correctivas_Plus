# Auditoría 10 revisiones — Corrección delimitación de cuenca v3.1.1

- [OK] 01_compila_python
- [OK] 02_default_radio_250m
- [OK] 03_modo_controlado_area
- [OK] 04_area_esperada_ui
- [OK] 05_area_maxima_permitida_ui
- [OK] 06_delineate_params_area
- [OK] 07_candidatos_salida
- [OK] 08_rechazo_sobre_area
- [OK] 09_qa_candidatos_en_app
- [OK] 10_titulo_hotfix_antisnap

## Corrección aplicada

Problema detectado:
- El ajuste del punto de control tomaba la celda de mayor acumulación dentro de un radio grande.
- En cuencas pequeñas cercanas a un río principal, el punto podía saltar al cauce mayor.
- Resultado típico: cuenca calculada de miles de km² cuando la real era decenas de km².

Solución:
- Radio por defecto reducido a 250 m.
- Nuevo modo recomendado: Controlado por área.
- Nueva área esperada aproximada.
- Nueva área máxima permitida.
- Evaluación de múltiples candidatos de salida.
- Selección evita candidatos que exceden el área máxima.
- Tabla QA de candidatos evaluados.
