# Registro de experimentos

Una fila por corrida, en orden cronológico; las filas y conclusiones anteriores no se editan
(ver `.claude/skills/experimento/SKILL.md`). Métricas en validación (reseñas con polaridad
4–5; acierto = el pueblo real en el Top-k): Hit@1 y Hit@5 micro, MRR macro y Spearman con
consultas balanceadas, todas de la variante con arquetipos y el λ elegido. HyDE Hit@5 sale de
la muestra de 100 reseñas.

| Fecha | Nombre | Qué cambió | Encoder | k | λ | Hit@1 | Hit@5 | MRR macro | Spearman bal. | HyDE Hit@5 | Indexado (min) | Commit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-11-11 | base | Referencia: `config.yaml` (resultados en `reports/metrics.json`) | ollama:nomic-embed-text-v2-moe | 10 | 0.015 | 0.5343 | 0.7401 | 0.6314 | −0.0797 | 0.58 | 86 | f110f53 |

## Conclusiones

- **base**: punto de comparación para las rondas de experimentos. Los arquetipos superan al
  centroide único y λ = 0.015 casi elimina el sesgo de popularidad sin perder MRR macro; HyDE
  empeora cuando la consulta ya es una reseña real (n = 100, ±10 puntos).
