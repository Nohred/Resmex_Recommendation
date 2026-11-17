# Comparación de encoders (muestra)

Referencia: `2026-11-16_enc_nomic`. Todas las filas comparten k, partición, semilla, holdout (`tope_1500` de `eval/holdout_ids.json`) y las mismas expansiones HyDE del LLM (salen de la caché); solo cambian el encoder, sus prefijos, `max_chars` y el λ recalibrado.

## Métricas

Holdout enmascarado: reseña como documento (n = 600) y HyDE + Anti-HyDE (n = 200). Gold: 40 consultas, relevancia binaria. Latencia: `Recommender.recommend` con el LLM en caché (embeddings + búsqueda), p50 / p95 en segundos.

| Experimento | Encoder | max_chars | λ | MRR macro doc | MRR macro HyDE | Hit@5 doc | Hit@5 HyDE | nDCG@5 gold | Error bottom (HyDE / gold) | Spearman sesgo (gold / val) | Latencia p50 / p95 | Embeddings (min) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-11-16_enc_nomic | ollama:nomic-embed-text-v2-moe | 1500 | 0.02 | 0.468 | 0.341 | 0.598 | 0.445 | 0.664 | 0.130 / 0.200 | -0.053 / -0.012 | 0.024 / 0.026 | 18.8 |
| 2026-11-16_enc_bge_m3 | ollama:bge-m3 | 1500 | 0.015 | 0.460 | 0.326 | 0.580 | 0.435 | 0.715 | 0.130 / 0.175 | 0.065 / 0.023 | 0.366 / 0.376 | 44.0 |
| 2026-11-16_enc_bge_m3_largo | ollama:bge-m3@ctx4096 | 9000 | 0.01 | 0.465 | 0.324 | 0.603 | 0.410 | 0.699 | 0.145 / 0.175 | 0.268 / 0.004 | 0.035 / 0.036 | 15.7 |
| 2026-11-16_enc_qwen3 | ollama:qwen3-embedding:0.6b | 1500 | 0.04 | 0.453 | 0.330 | 0.590 | 0.435 | 0.686 | 0.105 / 0.200 | -0.038 / -0.002 | 0.052 / 0.057 | 33.0 |
| 2026-11-16_enc_qwen3_largo | ollama:qwen3-embedding:0.6b@ctx4096 | 9000 | 0.04 | 0.459 | 0.324 | 0.600 | 0.425 | 0.656 | 0.110 / 0.225 | 0.037 / -0.045 | 0.053 / 0.056 | 46.6 |
| 2026-11-16_enc_gemma | ollama:embeddinggemma | 1500 | 0.02 | 0.485 | 0.336 | 0.617 | 0.440 | 0.671 | 0.090 / 0.075 | -0.003 / 0.028 | 0.040 / 0.042 | 57.8 |
| 2026-11-16_enc_gemma_largo | ollama:embeddinggemma | 6000 | 0.01 | 0.490 | 0.341 | 0.627 | 0.455 | 0.690 | 0.105 / 0.150 | 0.041 / 0.016 | 0.059 / 0.084 | 18.8 |
| 2026-11-16_enc_e5_large | st:intfloat/multilingual-e5-large@len512@fp16 | 1500 | 0.01 | 0.452 | 0.299 | 0.593 | 0.435 | 0.708 | 0.130 / 0.175 | -0.095 / -0.056 | 0.021 / 0.021 | 15.4 |

- **Spearman sesgo**: gold = apariciones en el top 5 del gold contra nº de reseñas del pueblo; val = frecuencia en el top 5 con consultas balanceadas de validación (`make evaluate`). Cerca de 0 es mejor.
- **Error bottom**: fracción de consultas con un pueblo correcto en el Bottom 5 (azar: 5/40 = 0.125 en el holdout).

## Diferencia contra `2026-11-16_enc_nomic` (bootstrap pareado, IC95)

| Experimento | ΔMRR macro doc | ΔHit@5 doc | ΔMRR macro HyDE | ΔHit@5 HyDE |
|---|---|---|---|---|
| 2026-11-16_enc_bge_m3 | -0.008 [-0.035, +0.018] | -0.018 [-0.058, +0.022] | -0.015 [-0.051, +0.022] | -0.010 [-0.065, +0.045] |
| 2026-11-16_enc_bge_m3_largo | -0.003 [-0.031, +0.024] | +0.005 [-0.038, +0.045] | -0.017 [-0.055, +0.022] | -0.035 [-0.095, +0.025] |
| 2026-11-16_enc_qwen3 | -0.015 [-0.042, +0.011] | -0.008 [-0.048, +0.033] | -0.011 [-0.052, +0.032] | -0.010 [-0.070, +0.050] |
| 2026-11-16_enc_qwen3_largo | -0.009 [-0.038, +0.018] | +0.002 [-0.035, +0.038] | -0.017 [-0.057, +0.025] | -0.020 [-0.080, +0.040] |
| 2026-11-16_enc_gemma | +0.017 [-0.010, +0.043] | +0.018 [-0.025, +0.055] | -0.005 [-0.049, +0.039] | -0.005 [-0.070, +0.055] |
| 2026-11-16_enc_gemma_largo | +0.022 [-0.006, +0.049] | +0.028 [-0.012, +0.065] | +0.000 [-0.043, +0.043] | +0.010 [-0.055, +0.070] |
| 2026-11-16_enc_e5_large | -0.016 [-0.043, +0.011] | -0.005 [-0.048, +0.035] | -0.042 [-0.085, +0.001] | -0.010 [-0.080, +0.065] |

`*` = el IC95 no contiene el 0.

### Definición del bootstrap

- Pareado por consulta: los dos encoders se evalúan con las mismas reseñas del holdout (mismo `row_id`) y, en HyDE, con el mismo texto generado por el LLM.
- 1,000 remuestreos (semilla 2025): en cada uno se sacan n consultas con reemplazo y el mismo remuestreo se aplica al candidato y a la referencia.
- Estadístico: diferencia candidato − referencia de (a) MRR macro (media de 1/posición por pueblo y luego media entre pueblos; un pueblo ausente en un remuestreo no cuenta en él) y (b) Hit@5 (media por consulta).
- IC95 por percentiles: percentiles 2.5 y 97.5 de las diferencias. Sin corrección por comparaciones múltiples.
- Solo mide la incertidumbre por qué consultas cayeron en el holdout; no la del K-Means, la del LLM ni la de la muestra de reseñas indexadas.

## Procedencia

| Experimento | Commit | Holdout | Reseñas train | LLM en HyDE del holdout | Determinismo |
|---|---|---|---|---|---|
| 2026-11-16_enc_nomic | f9152a8 | tope_1500 | 48566 | groq:openai/gpt-oss-120b × 200 | 1.00 |
| 2026-11-16_enc_bge_m3 | aeb9d2c | tope_1500 | 48566 | groq:openai/gpt-oss-120b × 200 | 1.00 |
| 2026-11-16_enc_bge_m3_largo | aeb9d2c | tope_1500 | 48566 | groq:openai/gpt-oss-120b × 200 | 1.00 |
| 2026-11-16_enc_qwen3 | aeb9d2c | tope_1500 | 48566 | groq:openai/gpt-oss-120b × 200 | 1.00 |
| 2026-11-16_enc_qwen3_largo | b83a790 | tope_1500 | 48566 | groq:openai/gpt-oss-120b × 200 | 1.00 |
| 2026-11-16_enc_gemma | aeb9d2c | tope_1500 | 48566 | groq:openai/gpt-oss-120b × 200 | 1.00 |
| 2026-11-16_enc_gemma_largo | 390e092 | tope_1500 | 48566 | groq:openai/gpt-oss-120b × 200 | 1.00 |
| 2026-11-16_enc_e5_large | aeb9d2c | tope_1500 | 48566 | groq:openai/gpt-oss-120b × 200 | 1.00 |
