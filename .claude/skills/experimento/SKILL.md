---
name: experimento
description: "Ejecuta y registra un experimento del recomendador (encoder, k, preprocesamiento, λ, HyDE). Úsala siempre que se cambie la configuración del modelo o se corra una evaluación."
---

# Experimento del recomendador

Cada experimento es un YAML en `configs/`, una corrida y una fila nueva en
`reports/experiments.md`. Los resultados anteriores no se tocan: sirven para comparar, y una
comparación solo vale si lo que se compara sigue intacto.

Entorno: `conda activate bigdata` y Ollama activo.

## 1. Crear la configuración

Archivo `configs/<AAAA-MM-DD>_<nombre>.yaml` (fecha de hoy; nombre corto en minúsculas con
guiones bajos, p. ej. `2026-11-20_k20`). Solo se declara lo que cambia:

```yaml
extends: config.yaml
paths:
  artifacts_dir: artifacts/exp/<AAAA-MM-DD>_<nombre>
  reports_dir: reports/exp/<AAAA-MM-DD>_<nombre>
archetypes:
  k: 20
```

- No edites `config.yaml`. Si ya existe ese archivo, `artifacts/exp/<...>` o
  `reports/exp/<...>` con ese nombre, es otra corrida: usa un nombre nuevo (`_v2`, `_b`).
- Si el experimento no cambia el índice (solo λ, `top_k`, cadena de LLM o HyDE), omite
  `artifacts_dir` para heredar el índice base y fija únicamente `reports_dir`.
- Si el cambio está en el código (prompts, limpieza de texto), haz commit antes de correr
  para que el hash del registro identifique ese código.
- Los hiperparámetros se eligen con validación, nunca con una partición de prueba.

## 2. Construir el índice (solo si hace falta)

| Qué cambió | ¿`index`? | Costo |
|---|---|---|
| Encoder (`encoder.*`) o preprocesamiento (`data.max_chars`, limpieza) | Sí | Recalcula embeddings: 30–85 min |
| k, `n_init`, semilla o partición (`archetypes.*`, `data.val_fraction`) | Sí | Reutiliza los embeddings; solo reagrupa (minutos) |
| λ, `top_k`, HyDE, cadena de LLM | No | — |

```bash
make index CONFIG=configs/<archivo>.yaml
# sin etiquetas del LLM (no afectan las métricas y gastan cuota):
python scripts/build_index.py --config configs/<archivo>.yaml --no-labels
```

La última línea del log dice `Indexado en X min (embeddings: Y min, caché en …)`: anota X.
Si los embeddings salieron de la caché compartida, Y ≈ 0; es lo esperado cuando no cambió ni
el encoder ni el preprocesamiento. Si esperabas reutilizarlos y se recalculan, detente y
revisa la configuración antes de gastar una hora.

## 3. Evaluar

```bash
make evaluate CONFIG=configs/<archivo>.yaml
# sin la comparación con HyDE (no llama al LLM):
python scripts/evaluate.py --config configs/<archivo>.yaml --hyde-sample 0
```

Escribe `<reports_dir>/metrics.json` y `<reports_dir>/figures/`. Corre la parte de HyDE cuando
el experimento toque HyDE, los prompts o el encoder; en los demás casos se puede omitir e
indicarlo en el registro.

Al leer `metrics.json`:

- La fila que se reporta es la variante con arquetipos y `λ = lambda_chosen`. Su clave dice
  "10 arquetipos" aunque k sea otro (texto fijo en `scripts/evaluate.py`); el k real se
  confirma con `n_archetypes / n_towns`.
- `evaluate.py` barre λ y elige `lambda_chosen` con su criterio en validación; el
  `ranking.lambda` del YAML solo afecta a la web. Registra los dos si difieren.

## 4. Registrar

Agrega **una fila al final** de `reports/experiments.md` (si no existe, créalo con el
encabezado de abajo). Columnas:

```markdown
| Fecha | Nombre | Qué cambió | Encoder | k | λ | Hit@1 | Hit@5 | MRR macro | Spearman bal. | HyDE Hit@5 | Indexado (min) | Commit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
```

- **Fecha**: la de la corrida. **Nombre**: el del YAML sin extensión.
- **Qué cambió**: frente a `config.yaml`, en una frase.
- **Encoder**: campo `encoder` de `metrics.json`. **λ**: `lambda_chosen`.
- **Hit@1 / Hit@5** (micro), **MRR macro** y **Spearman bal.** (`spearman_balanced`): de la
  variante con arquetipos y λ elegido. **HyDE Hit@5**: de `hyde_comparison`, o `—`.
- **Indexado (min)**: el X del paso 2; `—` si no se corrió `index`, y `(caché)` junto al
  número si los embeddings se reutilizaron.
- **Commit**: `git rev-parse --short HEAD` al correr; añade `+cambios` si había
  modificaciones sin commit en `src/` o `scripts/`.

Debajo de la tabla, en la sección "Conclusiones", agrega una entrada
`- **<nombre>**: …` de 1–2 frases: qué pasó frente a la base (o al experimento con el que se
compara) y qué se decide. Si la diferencia cabe en el ruido (p. ej. HyDE con n = 100 tiene
±10 puntos), dilo en lugar de declarar un ganador.

## Lo que nunca se hace

- Sobrescribir resultados: no reutilizar un `reports_dir` o `artifacts_dir` que ya tenga
  salidas de otra corrida, ni correr un experimento contra `reports/` (el de la base).
- Editar o borrar filas o conclusiones anteriores del registro. Si una corrida salió mal, se
  agrega una fila nueva que lo aclare; la vieja se queda.
- Subir `data/`, `artifacts/` o `.env`. Sí se versionan el YAML, `reports/exp/<nombre>/` y
  el registro.
