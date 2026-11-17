# Registro de experimentos

Una fila por corrida, en orden cronológico; las filas y conclusiones anteriores no se editan
(ver `.claude/skills/experimento/SKILL.md`). Métricas en validación (reseñas con polaridad
4–5; acierto = el pueblo real en el Top-k): Hit@1 y Hit@5 micro, MRR macro y Spearman con
consultas balanceadas, todas de la variante con arquetipos y el λ elegido. HyDE Hit@5 sale de
la muestra de 100 reseñas.

| Fecha | Nombre | Qué cambió | Encoder | k | λ | Hit@1 | Hit@5 | MRR macro | Spearman bal. | HyDE Hit@5 | Indexado (min) | Commit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-11-11 | base | Referencia: `config.yaml` (resultados en `reports/metrics.json`) | ollama:nomic-embed-text-v2-moe | 10 | 0.015 | 0.5343 | 0.7401 | 0.6314 | −0.0797 | 0.58 | 86 | f110f53 |
| 2026-11-16 | 2026-11-16_baseline_nomic_k10 | Nada: misma configuración que `config.yaml`; solo `make evaluate` (el reporte con el protocolo del equipo, `make report`, queda pendiente) | ollama:nomic-embed-text-v2-moe | 10 | 0.015 | 0.5343 | 0.7401 | 0.6314 | −0.0797 | 0.58 | — | 185cd6b |
| 2026-11-16 | 2026-11-16_arquetipos_k_agg | Barrido de k (fijo 1–30, k = a·ln n, silhouette) × agregación (máximo, media de 2, logsumexp, máximo sin clústeres pequeños), λ recalibrado por combinación; 61 combinaciones con `scripts/sweep_archetypes.py` (resultados en `barrido.json`, no en `metrics.json`) | ollama:nomic-embed-text-v2-moe | 1–30 | por combinación | — | — | — | — | — | 10 (caché; solo K-Means) | aeb9d2c |
| 2026-11-16 | 2026-11-16_arq_k30_lse | Ganadora del barrido: k = 30, agregación logsumexp (τ = 0.01) y λ = 0.03; `make evaluate` sin HyDE | ollama:nomic-embed-text-v2-moe | 30 | 0.03 | 0.5608 | 0.7622 | 0.6646 | −0.0420 | — | 10 (caché; K-Means 10.3 min, etiquetado interrumpido) | 390e092 |
| 2026-11-16 | 2026-11-16_calidad_resenas | Filtros y pesos de calidad de las reseñas de train que forman los arquetipos (14 variantes + muestra de 1,500 / pueblo con nombres enmascarados), k = 30 lse fijo, λ recalibrado, 4 semillas del K-Means; `scripts/sweep_quality.py` (resultados en `resumen.md`, no en `metrics.json`) | ollama:nomic-embed-text-v2-moe | 30 | 0.015–0.05 por variante | — | — | — | — | — | 99 (caché; K-Means de 16 variantes × 4 semillas, + 10.4 min de codificar la muestra enmascarada) | b2c63d9 (barrido) · 3ff12d2 (semillas, combinación, prueba) |
| 2026-11-16 | 2026-11-16_enc_nomic | Selección de encoder en muestra (tope de 1,500 reseñas/pueblo, 48,566 de train); referencia con el encoder de la base | ollama:nomic-embed-text-v2-moe | 10 | 0.02 | 0.5240 | 0.7158 | 0.6177 | −0.0119 | — (HyDE va en `make report`) | 19.8 | f9152a8 |
| 2026-11-16 | 2026-11-16_enc_bge_m3 | Encoder bge-m3 sin prefijos, misma muestra | ollama:bge-m3 | 10 | 0.015 | 0.5203 | 0.7238 | 0.6178 | +0.0234 | — (HyDE va en `make report`) | 45.4 | aeb9d2c |
| 2026-11-16 | 2026-11-16_enc_bge_m3_largo | bge-m3 sin truncar (`max_chars` 9000, `num_ctx` 4096) | ollama:bge-m3@ctx4096 | 10 | 0.01 | 0.5207 | 0.7258 | 0.6192 | +0.0045 | — (HyDE va en `make report`) | 16.8 | aeb9d2c |
| 2026-11-16 | 2026-11-16_enc_qwen3 | Encoder qwen3-embedding:0.6b (instrucción solo en consultas), misma muestra | ollama:qwen3-embedding:0.6b | 10 | 0.04 | 0.4956 | 0.7028 | 0.5972 | −0.0020 | — (HyDE va en `make report`) | 34.2 | aeb9d2c |
| 2026-11-16 | 2026-11-16_enc_qwen3_largo | qwen3-embedding sin truncar (`max_chars` 9000, `num_ctx` 4096) | ollama:qwen3-embedding:0.6b@ctx4096 | 10 | 0.04 | 0.4932 | 0.7043 | 0.5972 | −0.0453 | — (HyDE va en `make report`) | 47.9 | b83a790 |
| 2026-11-16 | 2026-11-16_enc_gemma | Encoder embeddinggemma con sus prefijos, misma muestra | ollama:embeddinggemma | 10 | 0.02 | 0.5398 | 0.7386 | 0.6346 | +0.0282 | — (HyDE va en `make report`) | 58.8 | aeb9d2c |
| 2026-11-16 | 2026-11-16_enc_gemma_largo | embeddinggemma con su contexto completo (`max_chars` 6000) | ollama:embeddinggemma | 10 | 0.01 | 0.5367 | 0.7377 | 0.6343 | +0.0156 | — (HyDE va en `make report`) | 21.9 | 390e092 |
| 2026-11-16 | 2026-11-16_enc_e5_large | Encoder multilingual-e5-large (sentence-transformers, fp16, `passage: `/`query: `), misma muestra | st:intfloat/multilingual-e5-large@len512@fp16 | 10 | 0.01 | 0.5237 | 0.7230 | 0.6204 | −0.0561 | — (HyDE va en `make report`) | 16.7 | aeb9d2c |
| 2026-11-16 | 2026-11-16_comparacion_encoders | Sin corrida: `scripts/compare_encoders.py` lee los ocho `make report` de la ronda `enc_*` y calcula el bootstrap pareado contra `enc_nomic` (`comparacion.md`) | — | 10 | — | — | — | — | — | — | — | ba7ff07 |

## Conclusiones

- **base**: punto de comparación para las rondas de experimentos. Los arquetipos superan al
  centroide único y λ = 0.015 casi elimina el sesgo de popularidad sin perder MRR macro; HyDE
  empeora cuando la consulta ya es una reseña real (n = 100, ±10 puntos).
- **2026-11-16_baseline_nomic_k10**: reproduce la base cifra por cifra (mismo índice; HyDE desde
  la caché). **Pendiente**: el reporte completo con el protocolo del equipo (gold, holdout
  enmascarado, robustez) no se ha corrido; `make report` está listo pero necesita ~380 llamadas
  al LLM (≈ 3 días de la cuota gratuita de Groq, o un modo recortado). Sin ese reporte no hay
  todavía comparación con las métricas del equipo.
- **2026-11-16_arquetipos_k_agg**: el holdout enmascarado se partió en calibración y prueba
  (`eval/holdout_calib_test.json`, 300 / 300). En calibración (reseña como documento) el MRR macro
  sube con k hasta el borde de la rejilla: k = 30 con logsumexp τ = 0.01 da 0.556 frente a 0.481 de
  k = 10 con máximo (Δ +0.075, IC95 [+0.037, +0.114]); el k adaptativo por log(n) no supera al k
  fijo equivalente, silhouette elige k = 3 en casi todos los pueblos (0.413) e ignorar clústeres
  pequeños empeora con k ≥ 20. λ sigue haciendo falta y más que antes: Spearman con λ = 0 pasa de
  +0.53 (k = 10) a +0.80 (k = 30) y el λ recalibrado sube de 0.015 a 0.03. Con HyDE en calibración
  (n = 120; 79 con qwen2.5:3b porque Groq agotó la cuota diaria) las tres finalistas quedan dentro
  del ruido entre sí y se conserva la líder. **Prueba (una sola vez)**: documento, n = 300, MRR
  macro 0.585 vs 0.537 (Δ +0.048, IC95 [+0.012, +0.084]); HyDE, n = 120 (96 con qwen), 0.495 vs
  0.427 (Δ +0.068, IC95 [+0.026, +0.107]). Reservas: el óptimo está en el borde de la rejilla
  (k > 30 no se probó) y el holdout premia reconocer negocios concretos (con k = 30 aparecen
  arquetipos de un solo hotel o restaurante y, en pueblos chicos, clústeres de 2–8 reseñas); falta
  contrastar con el gold antes de adoptar. `config.yaml` no cambia hasta que se apruebe.
- **2026-11-16_arq_k30_lse**: índice de la ganadora. En validación sube el MRR macro de 0.6314 a
  0.6646 y Hit@5 micro de 0.7401 a 0.7622 frente a la base, con Spearman −0.04 a λ = 0.03 (sin
  penalización sería +0.755). Etiquetas del LLM incompletas: 240 de 1,200 arquetipos (8 pueblos, Gemini)
  antes de que Gemini y Groq agotaran la cuota diaria; el resto tiene la etiqueta de palabras clave
  y se completa con `build_index.py --relabel`. No se corrió `make report` (sin cuota de LLM).
- **2026-11-16_calidad_resenas**: diagnóstico en `reports/diagnostico_resenas.md` (3.6 % de reseñas
  genéricas, 0.14 % en otro idioma, 0.05 % casi duplicadas, 27.6 % nombran su pueblo; 36 % de Hotel y
  Restaurant hablan solo del servicio). El ruido del K-Means domina: la referencia k = 30 lse da
  0.556 / 0.527 / 0.533 / 0.515 de MRR macro en calibración con 4 semillas (la 42 traía el sesgo del
  ganador del barrido anterior), así que todo se compara por semilla y con la media. Ningún filtro
  mejora el holdout fuera del ruido; solo bajar Hotel y Restaurant a 0.25 empeora (−0.028, IC sin el
  0). En el gold (25 consultas sin las `test`), quitar o restar peso a lo genérico sube el nDCG@5
  entre +0.018 y +0.035 con IC sin el 0. Combinación de las 2 mejores ideas en calibración (nombres
  enmascarados + ≥ 8 palabras, en la muestra de 1,500): **prueba** documento +0.007 [−0.011, +0.026],
  HyDE +0.022 [−0.011, +0.055], gold `test` −0.020 [−0.059, +0.018]. No se adopta; `review_quality`
  queda apagado por defecto. Las conclusiones del barrido de arquetipos con una sola semilla
  (Δ +0.075 de k = 30) conviene revisarlas con este mismo ruido de semilla en mente.
- **2026-11-16_enc_\* (ocho corridas) y 2026-11-16_comparacion_encoders**: muestra de 1,500 reseñas por
  pueblo, k = 10, misma partición, semilla y holdout (`tope_1500`), λ recalibrado por encoder y las
  mismas expansiones HyDE desde la caché. **embeddinggemma queda primero** en todo lo que no pasa por
  HyDE: MRR macro en validación 0.635 frente a 0.618 de nomic (bge-m3 0.618, e5-large 0.620, qwen3
  0.597); holdout enmascarado con la reseña como documento 0.485 frente a 0.468 (Δ +0.017, IC95
  [−0.010, +0.043]: dentro del ruido); gold Hit@1 0.775 frente a 0.675 y error en el bottom 0.075
  frente a 0.200 (40 consultas, heurístico). Con HyDE (n = 200) todos empatan con nomic (gemma
  −0.005 [−0.049, +0.039]) y e5-large queda al borde de ser peor (−0.042 [−0.085, +0.001]). bge-m3,
  qwen3 y e5-large no superan a nomic en ninguna métrica del holdout. **Contexto largo**: quitar el
  truncado no cambia nada medible (gemma 0.6343 vs 0.6346 en validación, bge-m3 0.6192 vs 0.6178,
  qwen3 0.5972 en ambos; holdout dentro del IC), porque solo ~1 % de las reseñas pasa de 1,500
  caracteres: `max_chars: 1500` se conserva. Los minutos de indexado no son comparables entre filas
  (el perfil de energía de la GPU cambió durante la ronda; el benchmark de abajo es la referencia).
  Decisión: embeddinggemma pasa a verificarse con k = 30 y logsumexp sobre el corpus completo
  (`2026-11-16_gemma_k30_lse`); se adopta solo si supera en validación a nomic con k = 30.
- **2026-11-16_baseline_nomic_k10 (reporte del equipo)**: el `make report` que quedaba pendiente ya
  se corrió (commit f9152a8, sin robustez): gold Hit@1 0.650, P@5 en consultas amplias 0.691, MRR
  0.767, nDCG@5 0.631; contrato, determinismo y latencia en caliente cumplen la rúbrica, Hit@1 y
  P@5 no. Es el punto de comparación del score final.

## Benchmarks de encoders (2026-11-16)

Preparación de la selección de encoder: todavía **sin índices ni métricas** (la evaluación se
implementa en un paso aparte). `make benchmark CONFIG=configs/2026-11-16_enc_<nombre>.yaml`
sobre 500 reseñas (345 caracteres de media) en la RTX 2070 Max-Q, perfil de energía
"balanced", con un solo modelo cargado en la GPU. Commit del código: 43a8b2c +cambios (los
de este commit). Minutos estimados = reseñas / textos por segundo; solo embeddings, sin
K-Means ni etiquetas.

| Config (`2026-11-16_enc_…`) | Encoder | Dim | Contexto (tokens) | max_chars | Textos/s | Muestra, 53,963 (min) | Corpus, 207,867 (min) |
|---|---|---|---|---|---|---|---|
| nomic | ollama:nomic-embed-text-v2-moe | 768 | 512 | 1500 | 119.6 | 8 | 29 |
| bge_m3 | ollama:bge-m3 | 1024 | 8192 | 1500 | 83.2 | 11 | 42 |
| bge_m3_largo | ollama:bge-m3@ctx4096 | 1024 | 8192 | 9000 | 81.8 | 11 | 42 |
| qwen3 | ollama:qwen3-embedding:0.6b | 1024 | 32768 | 1500 | 61.3 | 15 | 57 |
| qwen3_largo | ollama:qwen3-embedding:0.6b@ctx4096 | 1024 | 32768 | 9000 | 60.1 | 15 | 58 |
| gemma | ollama:embeddinggemma | 768 | 2048 | 1500 | 70.2 | 13 | 49 |
| gemma_largo | ollama:embeddinggemma | 768 | 2048 | 6000 | 69.2 | 13 | 50 |
| e5_large | st:intfloat/multilingual-e5-large@len512@fp16 | 1024 | 512 | 1500 | 80.5 | 11 | 43 |

- **Costo del contexto largo**: quitar el truncado a 1,500 caracteres baja el rendimiento
  entre 1 y 2 % (bge-m3, qwen3, embeddinggemma), porque solo el ~1 % de las reseñas pasa de
  1,500 caracteres. Con 500 textos entran muy pocas reseñas largas, así que la cifra es
  orientativa; el split y las filas no cambian con `max_chars` (verificado).
- **e5-large**: en FP32 con lote de 32 dio 25.4 textos/s (corpus ≈ 136 min); con `fp16` y lote
  de 128, 80.5 (dos corridas: 80.6 y 80.4), con coseno medio 0.9999995 contra FP32.
- **Las ocho corridas de muestra suman ≈ 1.6 h de embeddings** con estas cifras. Son un
  piso: cada benchmark dura 4–8 s y el índice base del 2026-11-11 corrió a 42 textos/s
  (86 min, GPU en ahorro de energía) frente a los 113–120 que marca el benchmark de nomic.
