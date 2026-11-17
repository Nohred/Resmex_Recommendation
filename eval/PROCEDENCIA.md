# Procedencia de `eval/`

`gold_queries.json` y `query_bank.json` son copias **sin editar** del repo del equipo,
<https://github.com/DyAndrade-AI/restmex-orbita> (commit `3ecbcfd`, 2026-11-12), tomadas el
2026-11-16. No se modifican aquí: si el equipo los actualiza, se vuelven a copiar completos.

| Archivo | SHA-256 |
|---|---|
| `gold_queries.json` | `e565ef79cbc66d3593f41662b1dd91911b3e9d437341dc570bc6f1505f7d62a9` |
| `query_bank.json` | `dc978be2b193078e7f2420edc32ea3c16b51339f43af944dff8f07d047854cae` |

- **Gold** (40 consultas, relevancia binaria): seed heurístico escrito por un asistente de IA a
  petición del equipo, con conocimiento general y palabras clave del corpus. Su autor vio
  salidas previas del sistema del equipo y está pendiente de adjudicación humana: sirve como
  prueba de regresión, no como evidencia independiente.
- **Banco** (60 consultas: las 40 del gold y 20 sin etiquetas) con particiones `dev`,
  `calibration` y `test` asignadas por hash del `id`. `test` es la evaluación humana final del
  equipo: no se usa para elegir encoder, k, λ ni prompts.

`scripts/evaluate_report.py` (`make report`) los lee de aquí y registra sus SHA-256 en el reporte.
