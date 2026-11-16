# RestMex_Recommendation

Recomendador semántico de Pueblos Mágicos sobre Rest-Mex 2025 (~208 mil reseñas, 40 pueblos).
Detalle, decisiones y resultados de referencia: `README.md` y `reports/metrics.json`.

## Arquitectura

1. `data.py`: CSV → limpieza (ftfy, NFKC), título + reseña, truncado a 1,500 caracteres,
   deduplicado y partición 90/10 estratificada por pueblo (`train` / `val`).
2. `encoders.py`: embeddings L2 con `nomic-embed-text-v2-moe` vía Ollama (prefijo
   `search_document: `); caché reanudable en disco, compartida entre experimentos.
3. `archetypes.py`: K-Means por pueblo (k = 10, solo `train`) → 400 arquetipos con palabras
   clave TF-IDF y etiqueta del LLM.
4. `query_expansion.py` + `llm.py` + `prompts.py`: HyDE / Anti-HyDE con la cadena
   Gemini → Groq → qwen2.5:3b → plantilla, con cuotas y caché (`artifacts/llm_cache`).
5. `retrieval.py` + `ranking.py`: coseno contra arquetipos, máximo por pueblo y
   `score = sim − λ·log(1+n)/log(1+n_max)`; Top 5 y Bottom 5 (Anti-HyDE).
6. `recommender.py` y `app/`: orquestación en línea y web FastAPI (`POST /recommend`).
7. `scripts/build_index.py` (offline), `scripts/evaluate.py` (métricas y figuras),
   `scripts/warmup.py` (caché del LLM). `config.py` carga el YAML en dataclasses.

## Comandos

Requieren `conda activate bigdata` (con el Python base faltan dependencias) y Ollama activo
para todo menos `make test`.

```bash
make index      # datos → embeddings → arquetipos → etiquetas (reanudable; 30–85 min sin caché)
make evaluate   # métricas → <reports_dir>/metrics.json y figures/
make test       # pytest (Ollama, Gemini y Groq simulados; no usa red)
make serve      # web en http://127.0.0.1:8000
make warmup     # precalienta la caché del LLM con las consultas de ejemplo
```

Todos aceptan `CONFIG=configs/<archivo>.yaml` (equivale a `RESTMEX_CONFIG`); sin él usan
`config.yaml`. Para pasar banderas: `python scripts/build_index.py --config <yaml> --no-labels`,
`python scripts/evaluate.py --config <yaml> --hyde-sample 0`.

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
