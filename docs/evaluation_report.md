# Reporte de evaluación automática

- **Encoder:** ollama:embeddinggemma · **k:** 30 arquetipos por pueblo (1200 en 40 pueblos) · **λ:** 0.03
- **Cadena de LLM (evaluación):** groq/openai/gpt-oss-120b → ollama/qwen2.5:3b → plantilla · temperature 0.4
- **LLM usado en esta corrida:** groq:openai/gpt-oss-120b × 332, ollama:qwen2.5:3b × 213 (127 llamadas nuevas, 418 desde la caché)
- **Commit:** a70eac5 · **Fecha:** 2026-11-16T22:18:03-06:00 · **Duración:** 1519.6 s · **Config:** `config.yaml`

> Protocolo, métricas y umbrales del equipo (`restmex-orbita`, commit 3ecbcfd). “5 mejores” = mayor afinidad con la consulta; “5 peores” = los más parecidos al Anti-HyDE (lo opuesto a lo pedido), no los de peor calidad.

## 1. Rúbrica

| Criterio | Valor | Umbral | Estado |
|---|---|---|---|
| Precision@5 (gold, 11 consultas con ≥5 relevantes) | 0.764 | 0.7 | ✅ |
| Hit@1 (gold) | 0.725 | 0.8 | ❌ |
| Contrato 5 + 5 | 1.000 | 1.0 | ✅ |
| Determinismo (con caché de LLM) | 1.000 | 1.0 | ✅ |
| Latencia p95 en caliente, LLM en caché (s, n = 40) | 0.053 | 3.0 | ✅ |
| Latencia p95 en frío, sin caché de LLM (s, n = 0) | — | 3.0 | — |

Latencia en proceso de `Recommender.recommend` (LLM + embeddings + búsqueda). En frío: p50 — s, p95 — s, máx. — s (n = 0, consultas que no estaban en la caché del LLM; la pausa entre llamadas para respetar la cuota no se cuenta). En caliente: p50 0.046 s, p95 0.053 s, máx. 0.064 s (n = 40). La web usa primero Gemini; aquí la latencia en frío es la de la cadena de evaluación.

## 2. Gold (seed heurístico del equipo, pendiente de adjudicación humana)

`eval/gold_queries.json` copiado sin cambios (SHA-256 `e565ef79cbc66d35…`; autor: asistente de IA (Claude) a petición del equipo; vio salidas del sistema del equipo: sí). Pipeline completo de la web (HyDE + Anti-HyDE + λ = 0.03) sobre 40 consultas.

| Métrica | Sistema | Popularidad | Azar |
|---|---|---|---|
| precision_at_5 | 0.450 | 0.135 | 0.095 |
| capped_recall_at_5 | 0.739 | 0.182 | 0.143 |
| hit_at_1 | 0.725 | 0.175 | 0.096 |
| hit_at_5 | 0.925 | 0.375 | 0.371 |
| mrr | 0.819 | 0.278 | 0.238 |
| ndcg_at_5 | 0.720 | 0.162 | 0.119 |
| bottom_error_rate | 0.100 | 0.300 | 0.364 |
| exclusion_violation_rate | 0.667 | 1.000 | 0.422 |

Top/Bottom de `recommend` idénticos al ranking evaluado: 1.000. Consultas resueltas con la plantilla sin LLM: 0. Las 3 consultas con exclusión miden `exclusion_violation_rate`; este sistema no interpreta exclusiones de forma explícita.

Por partición del banco de consultas del equipo (`test` no se usa para ajustar nada):

| Partición | Consultas | P@5 | Hit@1 | Hit@5 | MRR | nDCG@5 |
|---|---|---|---|---|---|---|
| calibration | 6 | 0.467 | 0.833 | 0.833 | 0.861 | 0.784 |
| dev | 19 | 0.453 | 0.684 | 0.895 | 0.799 | 0.706 |
| test | 15 | 0.440 | 0.733 | 1.000 | 0.828 | 0.713 |

## 3. Holdout enmascarado (sin consultas escritas a mano)

Cada reseña reservada de validación (no vista al construir los arquetipos) actúa como consulta con nombres de pueblo y estado enmascarados; el destino correcto es su pueblo.

600 reseñas de `val` (polaridad 4–5, 80–600 caracteres, 15 por pueblo, seed 2025). El enmascarado modificó el 60.3 % de las reseñas. El texto enmascarado queda en forma canónica (minúsculas, sin acentos ni puntuación), como en el equipo.

| Variante | n | Hit@1 | Hit@5 | MRR | Pueblo correcto en bottom 5 | Spearman(Hit@5 por pueblo, nº reseñas) |
|---|---|---|---|---|---|---|
| reseña como documento (sin LLM) | 600 | 0.470 | 0.707 | 0.581 | 0.003 | 0.052 |
| consulta cruda (search_query) | 600 | 0.470 | 0.702 | 0.577 | 0.007 | -0.017 |
| HyDE + Anti-HyDE (pipeline web) | 200 | 0.320 | 0.585 | 0.437 | 0.095 | 0.197 |
| Popularidad | 600 | 0.025 | 0.125 | 0.107 | 0.125 | — |
| Azar (esperado) | — | 0.025 | 0.125 | — | 0.125 | — |

HyDE se evalúa en una submuestra balanceada (`--hyde-holdout 200`; pendientes por cuota: 0; LLM: groq:openai/gpt-oss-120b × 64, ollama:qwen2.5:3b × 136). En las variantes sin Anti-HyDE el bottom es el reverso del ranking. Mismas 200 reseñas en las tres variantes:

| Variante (misma submuestra) | Hit@1 | Hit@5 | MRR |
|---|---|---|---|
| reseña como documento (sin LLM) | 0.460 | 0.675 | 0.566 |
| consulta cruda (search_query) | 0.485 | 0.680 | 0.579 |
| HyDE + Anti-HyDE (pipeline web) | 0.320 | 0.585 | 0.437 |

### Fuga de nombres: el mismo holdout sin enmascarar

| Texto de la consulta | Variante | Hit@1 | Hit@5 | MRR |
|---|---|---|---|---|
| original (sin enmascarar) | reseña como documento (sin LLM) | 0.593 | 0.787 | 0.682 |
| original (sin enmascarar) | consulta cruda (search_query) | 0.593 | 0.760 | 0.675 |
| canónico sin enmascarar | reseña como documento (sin LLM) | 0.588 | 0.778 | 0.676 |
| canónico sin enmascarar | consulta cruda (search_query) | 0.600 | 0.770 | 0.680 |
| canónico enmascarado | reseña como documento (sin LLM) | 0.470 | 0.707 | 0.581 |
| canónico enmascarado | consulta cruda (search_query) | 0.470 | 0.702 | 0.577 |

Con la reseña como documento, la fuga de nombres inflaba Hit@5 en +7.2 puntos (canónico sin enmascarar − enmascarado); pasar el texto a forma canónica cuesta además +0.8 puntos (original − canónico), efecto del formato y no de la fuga.

HyDE sin enmascarar no se corrió (`--hyde-unmasked 0`): duplicaría las llamadas al LLM y el prompt ya prohíbe nombres propios en el HyDE.

## 4. Ablación

Cada fila cambia una sola pieza del sistema completo. Es descriptiva: los hiperparámetros se eligen en validación (`scripts/evaluate.py`), no con estas tablas.

### Gold (40 consultas)

| Configuración | P@5 | Hit@1 | Hit@5 | MRR | nDCG@5 | Error en bottom |
|---|---|---|---|---|---|---|
| completo (arquetipos, λ, HyDE, bottom Anti-HyDE) | 0.450 | 0.725 | 0.925 | 0.819 | 0.720 | 0.100 |
| 1 centroide por pueblo | 0.410 | 0.650 | 0.925 | 0.786 | 0.654 | 0.075 |
| λ = 0 | 0.445 | 0.700 | 0.925 | 0.811 | 0.719 | 0.100 |
| consulta cruda (search_query, sin HyDE) | 0.425 | 0.850 | 0.975 | 0.912 | 0.784 | 0.100 |
| bottom = reverso del ranking | 0.450 | 0.725 | 0.925 | 0.819 | 0.720 | 0.050 |

### Holdout, submuestra con HyDE (n = 200)

| Configuración | P@5 | Hit@1 | Hit@5 | MRR | nDCG@5 | Error en bottom |
|---|---|---|---|---|---|---|
| completo (arquetipos, λ, HyDE, bottom Anti-HyDE) | 0.117 | 0.320 | 0.585 | 0.437 | 0.449 | 0.095 |
| 1 centroide por pueblo | 0.095 | 0.200 | 0.475 | 0.335 | 0.343 | 0.125 |
| λ = 0 | 0.110 | 0.315 | 0.550 | 0.433 | 0.435 | 0.095 |
| consulta cruda (search_query, sin HyDE) | 0.136 | 0.485 | 0.680 | 0.575 | 0.584 | 0.075 |
| bottom = reverso del ranking | 0.117 | 0.320 | 0.585 | 0.439 | 0.449 | 0.000 |

### Holdout completo, sin LLM (n = 600)

| Consulta | Configuración | Hit@1 | Hit@5 | MRR | Error en bottom |
|---|---|---|---|---|---|
| reseña como documento (sin LLM) | arquetipos, λ elegido | 0.470 | 0.707 | 0.581 | 0.003 |
| reseña como documento (sin LLM) | arquetipos, λ = 0 | 0.465 | 0.695 | 0.578 | 0.003 |
| reseña como documento (sin LLM) | 1 centroide por pueblo, λ elegido | 0.250 | 0.525 | 0.387 | 0.012 |
| consulta cruda (search_query) | arquetipos, λ elegido | 0.470 | 0.702 | 0.577 | 0.007 |
| consulta cruda (search_query) | arquetipos, λ = 0 | 0.478 | 0.702 | 0.580 | 0.007 |
| consulta cruda (search_query) | 1 centroide por pueblo, λ elegido | 0.252 | 0.548 | 0.394 | 0.008 |

## 5. Sesgo por volumen de reseñas

Spearman(apariciones en top 5 del gold, nº de reseñas) = 0.108 (p = 0.506). Pueblo más expuesto: Cuatro_Cienegas (11 apariciones). Pueblos que nunca aparecen en el top 5 del gold (4): Bernal, Metepec, Patzcuaro, San_Cristobal_de_las_Casas.

## 6. Robustez

Jaccard medio del top 5 ante variantes (sin acentos en mayúsculas, relleno, cláusulas invertidas): 0.546 (mínimo 0.000; 120 pares; pendientes por cuota: 0). Cada variante es un prompt nuevo para el LLM: aquí se mide la sensibilidad del HyDE a cambios superficiales.

## 7. Casos adversariales

Tasa de aprobación: 0.800. Este sistema no tiene nivel de confianza, así que un caso `reject_or_low` solo pasa si la consulta se rechaza. La validación (≥ 2 palabras, no solo exclusiones, al menos una palabra con contenido) vive en `restmex/validation.py`; la API además limita la consulta a 600 caracteres, por lo que el caso “muy larga” solo se sirve en proceso.

| Caso | Esperado | Resultado | OK |
|---|---|---|---|
| vacía | reject | rechazada: Describe al menos dos palabras sobre el viaje que buscas. | ✅ |
| una palabra | reject | rechazada: Describe al menos dos palabras sobre el viaje que buscas. | ✅ |
| solo stopwords | reject_or_low | rechazada: La consulta no describe ningún rasgo del destino; agrega más detalles. | ✅ |
| fuera de vocabulario | reject_or_low | servida | ❌ |
| solo números | reject_or_low | rechazada: La consulta no describe ningún rasgo del destino; agrega más detalles. | ✅ |
| emoji | serve | servida | ✅ |
| html | serve | servida | ✅ |
| muy larga | serve | servida | ✅ |
| todo negado | reject | rechazada: La consulta solo contiene exclusiones; di también qué buscas. | ✅ |
| pueblo excluido | serve | servida; Valladolid en el top 5 | ❌ |

## 8. Determinismo

Dos corridas de las 40 consultas del gold con la caché del LLM: rankings idénticos en 1.000 de las consultas. Huella SHA-256 de los rankings: `5a4f3fadbfa03a909f2e5b38a3f2a026397a50f539eb99647f86cd2d92b3b56f`.

Sin caché el sistema **no** es determinista: el LLM genera HyDE y Anti-HyDE con temperature 0.4, así que dos llamadas con la misma consulta producen textos distintos y el ranking puede cambiar. El determinismo que se reporta es el de la caché en disco (`artifacts/llm_cache`): la primera respuesta válida de cada prompt queda fija, y con ella el encoder y el ranking son reproducibles. Borrar la caché cambia la huella.

## 9. Consultas con fallos (Hit@5 = 0 o relevante en el bottom 5)

- **playa-snorkel** (relevante en bottom: Tulum) “Quiero playas tranquilas, agua turquesa y snorkel” → top Loreto, Cuatro_Cienegas, Isla_Mujeres, Sayulita, TodosSantos; bottom Tulum, Valladolid, Palenque, Taxco, Tapalpa; esperados Isla_Mujeres, Tulum, Sayulita, Mazunte, TodosSantos, Loreto, Bacalar
- **cenotes** (relevante en bottom: Izamal) “Busco cenotes para nadar y naturaleza” → top Tulum, Valladolid, Bacalar, Loreto, Cuatro_Cienegas; bottom Isla_Mujeres, Palenque, Izamal, Ixtapan_de_la_Sal, Tlaquepaque; esperados Valladolid, Tulum, Bacalar, Izamal
- **gastronomia** (relevante en bottom: Patzcuaro) “Quiero gastronomía mexicana, mercados y restaurantes tradicionales” → top Tlaquepaque, Palenque, Cuetzalan, Teotihuacan, Tepotzotlan; bottom Isla_Mujeres, TodosSantos, Loreto, Tulum, Patzcuaro; esperados Patzcuaro, Tlaquepaque, Izamal, Valladolid, San_Cristobal_de_las_Casas, Tepoztlan, Cholula, Atlixco, Ajijic
- **colonial** (relevante en bottom: Izamal, Patzcuaro) “Deseo arquitectura colonial, iglesias y calles históricas” → top Dolores_Hidalgo, Atlixco, Valladolid, Tepotzotlan, Taxco; bottom Tlaquepaque, Isla_Mujeres, Izamal, Sayulita, Patzcuaro; esperados Taxco, San_Cristobal_de_las_Casas, Patzcuaro, Izamal, Cholula, Dolores_Hidalgo, Valladolid, Tepotzotlan, Orizaba, Atlixco
- **plata** (Hit@5 = 0) “Comprar joyería de plata y recorrer calles empedradas en cerro” → top Atlixco, Huasca_de_Ocampo, Coatepec, Tepoztlan, Real_de_Catorce; bottom Tulum, Isla_Mujeres, Loreto, Mazunte, Sayulita; esperados Taxco
- **monolito** (Hit@5 = 0) “Escalar un monolito enorme y comer gorditas” → top Tepoztlan, Malinalco, Tapalpa, Huasca_de_Ocampo, Teotihuacan; bottom Tlaquepaque, Tequisquiapan, Isla_Mujeres, Ixtapan_de_la_Sal, Patzcuaro; esperados Bernal
- **independencia** (Hit@5 = 0) “Historia de la independencia de México y nieves de sabores raros” → top Tlaquepaque, Atlixco, Taxco, Valladolid, Tequila; bottom Isla_Mujeres, Sayulita, Tulum, TodosSantos, Cuatro_Cienegas; esperados Dolores_Hidalgo

## Exportación para la evaluación humana ciega

`reports/rankings_query_bank.json`: ranking completo de 60 de 60 consultas de `eval/query_bank.json` (rechazadas por la validación: 0; pendientes por cuota: 0).
