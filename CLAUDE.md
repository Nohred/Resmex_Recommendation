# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# RestMex_Recommendation

Recomendador semántico de Pueblos Mágicos sobre Rest-Mex 2025 (~208 mil reseñas, 40 pueblos).
Detalle, decisiones y resultados de referencia: `README.md`, `reports/metrics.json`,
`reports/experiments.md` (registro) y `docs/informe/informe.pdf` (qué se probó y qué se concluyó).

## Arquitectura

1. `data.py`: CSV → limpieza (ftfy, NFKC), título + reseña, truncado a 1,500 caracteres,
   deduplicado y partición 90/10 estratificada por pueblo (`train` / `val`).
2. `encoders.py`: embeddings L2 con `embeddinggemma` vía Ollama (prefijo de documento
   `title: none | text: `); caché reanudable en disco, compartida entre experimentos.
3. `archetypes.py`: K-Means por pueblo (k = 30, solo `train`) → 1,200 arquetipos con palabras
   clave TF-IDF y etiqueta del LLM. `review_quality.py` (filtros y pesos) está apagado por defecto.
4. `query_expansion.py` + `llm.py` + `prompts.py`: HyDE / Anti-HyDE con la cadena
   Gemini → Groq → qwen2.5:3b → plantilla, con cuotas y caché (`artifacts/llm_cache`).
5. `retrieval.py` + `aggregation.py` + `ranking.py`: coseno contra arquetipos, agregación por
   pueblo según `ranking.aggregation` (`lse`, máximo suave con τ = 0.01; `max` es el anterior) y
   `score = sim − λ·log(1+n)/log(1+n_max)` con λ = 0.03; Top 5 y Bottom 5 (Anti-HyDE).
   `ArchetypeIndex.town_scores` es el único punto donde se agrega: ranking, mapa de la web
   (`ranking.town_overview`) y evaluación pasan por ahí.
6. `recommender.py` y `app/`: orquestación en dos etapas (`expand` → `rank`) y web FastAPI
   (`POST /recommend`, `/recommend/stream` en NDJSON, `GET /towns`); frontend sin build ni CDN
   en `app/static/` (`tests/test_web_assets.py` verifica que funcione sin red).
7. `scripts/build_index.py` (offline), `scripts/evaluate.py` (métricas y figuras),
   `scripts/warmup.py` (caché del LLM). `config.py` carga el YAML en dataclasses.
8. Evaluación del equipo: `report.py` + `scripts/evaluate_report.py` (gold, holdout enmascarado
   con los `row_id` de `eval/holdout_ids.json`, ablación, robustez) y `bootstrap.py` (IC pareado).
   Barridos: `scripts/sweep_archetypes.py`, `scripts/sweep_quality.py`, `scripts/compare_encoders.py`.

## Comandos

Requieren `conda activate bigdata` (con el Python base faltan dependencias) y Ollama activo
para todo menos `make test`.

```bash
make index      # datos → embeddings → arquetipos → etiquetas (reanudable; ~70 min sin caché)
make evaluate   # métricas → <reports_dir>/metrics.json y figures/
make report     # protocolo del equipo: gold, holdout enmascarado, robustez → <reports_dir>/evaluation_report.{json,md}
make test       # pytest (Ollama, Gemini y Groq simulados; no usa red)
PYTHONPATH=src python -m pytest tests/test_ranking.py::<nombre> -q   # una sola prueba
make serve      # web en http://127.0.0.1:8000
make warmup     # precalienta la caché del LLM con las consultas de ejemplo
make benchmark  # textos/s del encoder con 500 reseñas (antes de lanzar un índice largo)
```

Todos aceptan `CONFIG=configs/<archivo>.yaml` (equivale a `RESTMEX_CONFIG`); sin él usan
`config.yaml`. Para pasar banderas: `python scripts/build_index.py --config <yaml> --no-labels`,
`python scripts/evaluate.py --config <yaml> --hyde-sample 0`.

`make report` (protocolo del equipo) usa `llm.eval_chain` y la caché del LLM: reanuda sin
repetir llamadas. Completo y sin caché son ~380 llamadas ≈ 454 mil tokens; la capa gratuita
de Groq da 200 mil tokens/día (≈ 160 llamadas) y 8 mil/min, así que no cabe en un día.
`make report ARGS="--estimate"` cuenta las llamadas nuevas sin hacer ninguna. Modo recortado:
`make report CONFIG=<yaml> ARGS="--hyde-holdout 80 --skip-robustness"` (~145 llamadas);
`--max-llm-calls N` deja el resto pendiente y `--publish` copia a `docs/` y `reports/`.

## Experimentos

- Un YAML de `configs/` hereda con `extends: config.yaml` (merge profundo: los diccionarios
  se mezclan, escalares y listas se reemplazan) y solo declara lo que cambia.
- `paths.artifacts_dir` / `paths.reports_dir` por experimento: `artifacts/exp/<nombre>/` y
  `reports/exp/<nombre>/`. Un experimento que solo cambia el ranking o HyDE puede heredar
  `artifacts_dir` y fijar únicamente `reports_dir`.
- La caché de embeddings depende solo de encoder + preprocesamiento (huella del nombre del
  encoder, el prefijo y los textos limpios), no de k ni de λ. Vive en `paths.embeddings_dir`
  (`artifacts/`): si la huella coincide con la base se reutiliza `artifacts/embeddings.npy`;
  si no, se crea `artifacts/embeddings_cache/<huella>/`. El directorio del experimento guarda
  `embeddings_ref.json` como puntero.
- Registro acumulativo: `reports/experiments.md`.
- La base cambió el 2026-11-16 (antes: nomic, k = 10, máximo, λ = 0.015; reproducible con
  `configs/2026-11-11_base_nomic_k10.yaml`). Los YAML anteriores que heredan de `config.yaml`
  describen sus corridas contra el `config.yaml` del commit de su fila en el registro, no el actual.
- `artifacts/embeddings.npy` es la caché del encoder base; las de otros encoders (incluida la de
  nomic) viven en `artifacts/embeddings_cache/<huella>/`.
- El ruido de semilla del K-Means es ±0.017 de MRR macro en el holdout: una diferencia menor
  entre dos índices no decide nada sin repetir semillas.
- Informe: `python docs/informe/generar_tablas.py && (cd docs/informe && latexmk -pdf informe.tex)`;
  las tablas salen de los JSON de `reports/`, no se editan a mano.

## Reglas

a) **No elegir hiperparámetros mirando la partición de prueba.** Encoder, k, preprocesamiento,
   λ y prompts se eligen solo con `val`. Hoy el repo únicamente tiene `train` / `val` (ambas
   salen del CSV de entrenamiento); si se agrega una partición de prueba, se evalúa una sola
   vez con la configuración ya elegida y nunca se usa para comparar variantes.
b) **Nunca subir `data/`, `artifacts/` ni `.env`.** Nada de `git add -f`; las claves no se
   imprimen ni se copian a logs, configs o reportes.
c) **Toda corrida se registra con la skill `experimento`** (`.claude/skills/experimento/`),
   incluidas las evaluaciones de la configuración base.
d) **Cada variante vive en su propio YAML en `configs/`**; `config.yaml` no se edita para
   probar algo. Solo cambia cuando una variante se adopta como nueva base, en su propio commit.
e) **Código, comentarios, docs y mensajes de commit en español**, como el resto del repo.

Además: no modificar `data/` ni borrar o sobrescribir resultados previos en `artifacts/` y
`reports/` (cada experimento escribe en su propio directorio).
