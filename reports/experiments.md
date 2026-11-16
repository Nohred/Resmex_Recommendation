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

## Conclusiones

- **base**: punto de comparación para las rondas de experimentos. Los arquetipos superan al
  centroide único y λ = 0.015 casi elimina el sesgo de popularidad sin perder MRR macro; HyDE
  empeora cuando la consulta ya es una reseña real (n = 100, ±10 puntos).
- **2026-11-16_baseline_nomic_k10**: reproduce la base cifra por cifra (mismo índice; HyDE desde
  la caché). **Pendiente**: el reporte completo con el protocolo del equipo (gold, holdout
  enmascarado, robustez) no se ha corrido; `make report` está listo pero necesita ~380 llamadas
  al LLM (≈ 3 días de la cuota gratuita de Groq, o un modo recortado). Sin ese reporte no hay
  todavía comparación con las métricas del equipo.

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
