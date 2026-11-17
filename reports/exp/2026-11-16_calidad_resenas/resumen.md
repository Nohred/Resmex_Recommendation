# Calidad de las reseñas que forman los arquetipos

Experimento `2026-11-16_calidad_resenas` (`configs/2026-11-16_calidad_resenas.yaml`,
`scripts/sweep_quality.py`). Encoder `ollama:nomic-embed-text-v2-moe` (caché de la base), k = 30 con
logsumexp τ = 0.01 (líder del barrido de arquetipos) fijos en todas las variantes; λ recalibrado por
variante con `choose_lambda` en val (polaridad 4–5, sin la mitad de prueba).

**Consultas idénticas en todas las variantes.** Holdout enmascarado con los `row_id` de
`eval/holdout_ids.json` (ya existía; no se regeneró), partido por `eval/holdout_calib_test.json`;
sus vectores (documento y HyDE / Anti-HyDE) se reutilizan del barrido de arquetipos. Gold del equipo
desde la caché del LLM: cero llamadas nuevas. Para elegir solo se usaron calibración y las 25
consultas `dev` + `calibration` del gold; la mitad de prueba y las 15 consultas `test` se evaluaron una
vez, al final. Los filtros solo tocan las reseñas de train que entran al K-Means; la penalización de
popularidad y el Spearman usan siempre el conteo completo de train.

## Fase 1 · Diagnóstico

`reports/diagnostico_resenas.md` (desglose por tipo y polaridad, 10 ejemplos reales por categoría) y
`diagnostico_por_pueblo.csv`. En corto, sobre 207,867 reseñas:

- Palabras de contenido: mediana 20 (P5 = 9); solo 2.3 % tiene menos de 8. Restaurant es el más corto
  (mediana 17) y Hotel el más largo (32).
- Genéricas (≤ P10 de palabras, IDF < P25 y coseno al centroide > P75): 3.6 % las tres, 15.8 % dos de
  tres; Restaurant 7.5 % / 25.8 %, Attractive 1.4 % / 6.9 %. Crecen con la polaridad.
- Otro idioma: 0.14 % (casi todo inglés; Isla Mujeres, Loreto, Tulum).
- Casi duplicados (> 0.95 en el pueblo): 106 sobrantes (0.05 %), sobre todo traducciones repetidas.
- Mencionan su pueblo o estado: 27.6 % (14.6 % en polaridad 1, 29.2 % en polaridad 5).
- Hotel / Restaurant solo de servicio (léxico grueso, cota alta): 36 % de esos tipos; 48 % en polaridad 1.

## Fase 2 · Variantes

**Ruido del K-Means.** Con la semilla del YAML (42) ninguna variante superaba a la referencia y varias
"perdían" de forma significativa, pero quitar 9 reseñas (`min_contenido_3`) ya movía el MRR macro de
calibración −0.008 y quitar 195, −0.035. La referencia con otras semillas da 0.556 (42), 0.527 (7),
0.533 (123) y 0.515 (2025): la semilla 42 de `k30|lse` fue la ganadora del barrido de arquetipos en esta
misma calibración y trae el sesgo del ganador. Por eso cada variante se repitió con 4 semillas
(42, 7, 123, 2025) y se compara contra la referencia **con su misma semilla**; los Δ e IC95 usan el
recíproco del rango de cada consulta promediado sobre las semillas (bootstrap pareado por consulta,
1,000 remuestreos). El bootstrap de una sola semilla (`barrido_calidad.md`) subestima la incertidumbre.

Tabla completa: `barrido_semillas.md`. Ningún pueblo bajó de 200 reseñas (el mínimo fue 457, Tapalpa con
`genericas_30`); el piso nunca tuvo que devolver reseñas.

| Variante | Reseñas de train | Mín. por pueblo | MRR macro calib. (media ± DE) | Δ calib. [IC95] | nDCG@5 gold (25) | Δ gold [IC95] | Spearman |
|---|---|---|---|---|---|---|---|
| sin filtro (ref.) | 187,080 | 653 | 0.5327 ± 0.017 | — | 0.709 | — | −0.07 ± 0.06 |
| ≥ 3 palabras de contenido | 187,071 | 653 | 0.5331 ± 0.011 | +0.000 [−0.006, +0.007] | 0.717 | +0.008 [−0.003, +0.021] | −0.06 ± 0.07 |
| ≥ 5 | 186,885 | 652 | 0.5249 ± 0.008 | −0.008 [−0.022, +0.006] | 0.735 | **+0.026 [+0.003, +0.052]** | +0.05 ± 0.13 |
| ≥ 8 | 182,796 | 641 | 0.5392 ± 0.013 | +0.007 [−0.011, +0.023] | 0.702 | −0.007 [−0.050, +0.026] | −0.12 ± 0.08 |
| sin 10 % más genérico | 168,373 | 588 | 0.5217 ± 0.010 | −0.011 [−0.028, +0.004] | 0.733 | +0.024 [−0.002, +0.051] | +0.07 ± 0.08 |
| sin 20 % | 149,664 | 522 | 0.5164 ± 0.006 | −0.016 [−0.037, +0.004] | 0.738 | **+0.029 [+0.006, +0.053]** | +0.04 ± 0.05 |
| sin 30 % | 130,956 | 457 | 0.5167 ± 0.004 | −0.016 [−0.038, +0.004] | 0.704 | −0.006 [−0.043, +0.029] | −0.03 ± 0.04 |
| sin otro idioma | 186,823 | 653 | 0.5351 ± 0.014 | +0.002 [−0.008, +0.012] | 0.713 | +0.004 [−0.014, +0.026] | −0.04 ± 0.10 |
| sin casi duplicados | 186,989 | 653 | 0.5333 ± 0.023 | +0.001 [−0.011, +0.012] | 0.710 | +0.001 [−0.015, +0.019] | −0.02 ± 0.13 |
| sin idioma ni duplicados | 186,732 | 653 | 0.5332 ± 0.008 | +0.001 [−0.013, +0.013] | 0.702 | −0.007 [−0.024, +0.007] | −0.07 ± 0.09 |
| peso = informatividad | 187,080 | 653 | 0.5334 ± 0.014 | +0.001 [−0.017, +0.017] | 0.719 | +0.010 [−0.022, +0.041] | +0.06 ± 0.03 |
| peso = informatividad × polaridad/5 | 187,080 | 653 | 0.5244 ± 0.012 | −0.008 [−0.027, +0.011] | 0.745 | **+0.035 [+0.003, +0.073]** | +0.08 ± 0.06 |
| Hotel, Restaurant × 0.5 | 187,080 | 653 | 0.5193 ± 0.009 | −0.013 [−0.032, +0.004] | 0.728 | **+0.018 [+0.001, +0.035]** | +0.05 ± 0.11 |
| Hotel, Restaurant × 0.25 | 187,080 | 653 | 0.5047 ± 0.009 | **−0.028 [−0.050, −0.007]** | 0.730 | +0.021 [−0.011, +0.054] | +0.04 ± 0.08 |
| muestra 1,500 / pueblo (ref.) | 52,268 | 653 | 0.5248 ± 0.011 | — | 0.706 | — | +0.03 ± 0.05 |
| muestra sin nombres | 52,268 | 653 | 0.5333 ± 0.011 | +0.009 [−0.011, +0.028] | 0.736 | +0.031 [−0.007, +0.064] | −0.00 ± 0.06 |

Lectura:

- **Holdout (reseña → pueblo):** ninguna idea mejora fuera del ruido (todas las |Δ| ≤ 0.017 salvo
  `tipo_025`). La única señal clara es negativa: bajar mucho el peso de Hotel y Restaurant cuesta
  −0.028 y es negativa en las 4 semillas.
- **Gold (necesidad de viaje → pueblos):** las ideas que quitan o restan peso a lo genérico y al
  servicio suben el nDCG@5 entre +0.018 y +0.035, con IC sin el 0. Es un trade-off con el holdout: sus
  consultas son reseñas reales (también las genéricas, que encuentran arquetipos genéricos), mientras
  el gold describe intereses. Con 25 consultas y un gold sin adjudicar, es indicio, no prueba.
- **Spearman:** su desviación entre semillas (0.03–0.13) es del orden de las diferencias entre
  variantes; ninguna cambia el sesgo de popularidad de forma interpretable.
- **Enmascarar nombres** (solo en la muestra, ~10 min de codificación; cambió el 28.6 % de los
  textos) es la mejor idea en calibración y la segunda en el gold, sin salir del ruido.

## Combinación de las 2 mejores ideas

Regla fijada antes de verla (`best_ideas`): la mejor variante de cada idea por MRR macro medio de
calibración (desempate: gold) y las 2 ideas con mayor Δ medio. Salieron **6 · nombres enmascarados**
y **1 · ≥ 8 palabras de contenido**. Como la 6 solo existe en la muestra, la combinación es
*muestra de 1,500 por pueblo + nombres enmascarados + ≥ 8 palabras* (50,908 reseñas; mínimo 641,
Tapalpa) y su referencia es `muestra_1500`.

| Calibración (4 semillas) | Combinación | Referencia | Δ [IC95] |
|---|---|---|---|
| MRR macro holdout (n = 300) | 0.5415 ± 0.011 | 0.5248 ± 0.011 | +0.017 [−0.002, +0.037] (por semilla −0.012 / +0.036 / +0.031 / +0.011) |
| nDCG@5 gold (25) | 0.731 | 0.706 | +0.025 [−0.017, +0.063] |
| Spearman | +0.04 ± 0.07 | +0.03 ± 0.05 | — |

**Prueba (una sola vez; `prueba.json`).** Recíproco del rango promediado sobre las 4 semillas por
consulta y bootstrap pareado; λ = 0.015 (combinación) y 0.015–0.02 (referencia), de calibración.

| Prueba | Combinación [IC95] | Referencia [IC95] | Δ [IC95] |
|---|---|---|---|
| Holdout, reseña como documento (n = 300) · MRR macro | 0.582 [0.536, 0.623] | 0.574 [0.528, 0.617] | +0.007 [−0.011, +0.026] |
| · Hit@5 | 0.714 [0.668, 0.758] | 0.695 [0.644, 0.738] | +0.019 [−0.007, +0.044] |
| Holdout, HyDE + Anti-HyDE (n = 120; qwen2.5:3b 96, Groq 24) · MRR macro | 0.484 [0.416, 0.554] | 0.461 [0.398, 0.531] | +0.022 [−0.011, +0.055] |
| Gold `test` (n = 15) · nDCG@5 | 0.678 [0.557, 0.798] | 0.698 [0.594, 0.809] | −0.020 [−0.059, +0.018] |

**Decisión: no se adopta ningún filtro.** La combinación apunta en la dirección correcta en el holdout
(también con HyDE), pero ningún intervalo excluye el 0 y en el gold `test` va en contra. Además está
medida sobre la muestra de 1,500 por pueblo: adoptarla en el corpus completo exige recodificar train
enmascarado (~30 min) y volver a medir. `review_quality` queda en el código, apagado por defecto.

## Revisión a mano

`descartadas_muestra.md`: 20 reseñas descartadas al azar por la combinación (todas por el mínimo de 8
palabras, 1,360 de 52,268). Una lectura rápida: alrededor de la mitad son elogio vacío (15, 17, 19), pero
varias sí dicen algo del lugar (12 tortugas en Mazunte, 16 el cenote de Valladolid, 18 la plaza junto al
ferry de Isla Mujeres, 10 la vegetación de Xilitla). El umbral de 8 palabras es tosco.

## Archivos

`barrido_calidad.{json,md}` (semilla 42), `barrido_semillas.{json,md}` (4 semillas, con métricas por
semilla y por consulta), `combinacion.{json,md}`, `prueba.json`, `descartadas_muestra.md`,
`diagnostico_por_pueblo.csv`. Tiempos: barrido 33 min, semillas 65 min, codificación de la muestra
enmascarada 10.4 min (todo en `artifacts/exp/2026-11-16_calidad_resenas/`, no versionado).
