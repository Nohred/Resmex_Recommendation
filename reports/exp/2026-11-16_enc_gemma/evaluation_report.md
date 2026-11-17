# Reporte de evaluación automática

- **Encoder:** ollama:embeddinggemma · **k:** 10 arquetipos por pueblo (400 en 40 pueblos) · **λ:** 0.02
- **Cadena de LLM (evaluación):** groq/openai/gpt-oss-120b → ollama/qwen2.5:3b → plantilla · temperature 0.4
- **LLM usado en esta corrida:** groq:openai/gpt-oss-120b × 401, ollama:qwen2.5:3b × 24 (0 llamadas nuevas, 425 desde la caché)
- **Commit:** aeb9d2c · **Fecha:** 2026-11-16T12:34:27-06:00 · **Duración:** 77.3 s · **Config:** `configs/2026-11-16_enc_gemma.yaml`

> Protocolo, métricas y umbrales del equipo (`restmex-orbita`, commit 3ecbcfd). “5 mejores” = mayor afinidad con la consulta; “5 peores” = los más parecidos al Anti-HyDE (lo opuesto a lo pedido), no los de peor calidad.

## 1. Rúbrica

| Criterio | Valor | Umbral | Estado |
|---|---|---|---|
| Precision@5 (gold, 11 consultas con ≥5 relevantes) | 0.800 | 0.7 | ✅ |
| Hit@1 (gold) | 0.775 | 0.8 | ❌ |
| Contrato 5 + 5 | 1.000 | 1.0 | ✅ |
| Determinismo (con caché de LLM) | 1.000 | 1.0 | ✅ |
| Latencia p95 en caliente, LLM en caché (s, n = 40) | 0.042 | 3.0 | ✅ |
| Latencia p95 en frío, sin caché de LLM (s, n = 0) | — | 3.0 | — |

Latencia en proceso de `Recommender.recommend` (LLM + embeddings + búsqueda). En frío: p50 — s, p95 — s, máx. — s (n = 0, consultas que no estaban en la caché del LLM; la pausa entre llamadas para respetar la cuota no se cuenta). En caliente: p50 0.040 s, p95 0.042 s, máx. 0.042 s (n = 40). La web usa primero Gemini; aquí la latencia en frío es la de la cadena de evaluación.

## 2. Gold (seed heurístico del equipo, pendiente de adjudicación humana)

`eval/gold_queries.json` copiado sin cambios (SHA-256 `e565ef79cbc66d35…`; autor: asistente de IA (Claude) a petición del equipo; vio salidas del sistema del equipo: sí). Pipeline completo de la web (HyDE + Anti-HyDE + λ = 0.02) sobre 40 consultas.

| Métrica | Sistema | Popularidad | Azar |
|---|---|---|---|
| precision_at_5 | 0.410 | 0.120 | 0.095 |
| capped_recall_at_5 | 0.657 | 0.193 | 0.143 |
| hit_at_1 | 0.775 | 0.050 | 0.096 |
| hit_at_5 | 0.850 | 0.450 | 0.371 |
| mrr | 0.831 | 0.234 | 0.238 |
| ndcg_at_5 | 0.671 | 0.141 | 0.119 |
| bottom_error_rate | 0.075 | 0.300 | 0.364 |
| exclusion_violation_rate | 0.667 | 0.000 | 0.422 |

Top/Bottom de `recommend` idénticos al ranking evaluado: 1.000. Consultas resueltas con la plantilla sin LLM: 0. Las 3 consultas con exclusión miden `exclusion_violation_rate`; este sistema no interpreta exclusiones de forma explícita.

Por partición del banco de consultas del equipo (`test` no se usa para ajustar nada):

| Partición | Consultas | P@5 | Hit@1 | Hit@5 | MRR | nDCG@5 |
|---|---|---|---|---|---|---|
| calibration | 6 | 0.400 | 0.833 | 0.833 | 0.861 | 0.717 |
| dev | 19 | 0.442 | 0.737 | 0.842 | 0.803 | 0.663 |
| test | 15 | 0.373 | 0.800 | 0.867 | 0.854 | 0.661 |

## 3. Holdout enmascarado (sin consultas escritas a mano)

Cada reseña reservada de validación (no vista al construir los arquetipos) actúa como consulta con nombres de pueblo y estado enmascarados; el destino correcto es su pueblo.

600 reseñas de `val` (polaridad 4–5, 80–600 caracteres, 15 por pueblo, seed 2025). El enmascarado modificó el 53.3 % de las reseñas. El texto enmascarado queda en forma canónica (minúsculas, sin acentos ni puntuación), como en el equipo.

| Variante | n | Hit@1 | Hit@5 | MRR | Pueblo correcto en bottom 5 | Spearman(Hit@5 por pueblo, nº reseñas) |
|---|---|---|---|---|---|---|
| reseña como documento (sin LLM) | 600 | 0.358 | 0.617 | 0.485 | 0.012 | 0.096 |
| consulta cruda (search_query) | 600 | 0.375 | 0.635 | 0.496 | 0.010 | 0.191 |
| HyDE + Anti-HyDE (pipeline web) | 200 | 0.215 | 0.440 | 0.336 | 0.090 | 0.051 |
| Popularidad | 600 | 0.025 | 0.125 | 0.107 | 0.125 | — |
| Azar (esperado) | — | 0.025 | 0.125 | — | 0.125 | — |

HyDE se evalúa en una submuestra balanceada (`--hyde-holdout 200`; pendientes por cuota: 0; LLM: groq:openai/gpt-oss-120b × 200). En las variantes sin Anti-HyDE el bottom es el reverso del ranking. Mismas 200 reseñas en las tres variantes:

| Variante (misma submuestra) | Hit@1 | Hit@5 | MRR |
|---|---|---|---|
| reseña como documento (sin LLM) | 0.320 | 0.605 | 0.454 |
| consulta cruda (search_query) | 0.320 | 0.580 | 0.445 |
| HyDE + Anti-HyDE (pipeline web) | 0.215 | 0.440 | 0.336 |

### Fuga de nombres: el mismo holdout sin enmascarar

| Texto de la consulta | Variante | Hit@1 | Hit@5 | MRR |
|---|---|---|---|---|
| original (sin enmascarar) | reseña como documento (sin LLM) | 0.490 | 0.722 | 0.595 |
| original (sin enmascarar) | consulta cruda (search_query) | 0.490 | 0.713 | 0.595 |
| canónico sin enmascarar | reseña como documento (sin LLM) | 0.483 | 0.693 | 0.586 |
| canónico sin enmascarar | consulta cruda (search_query) | 0.493 | 0.707 | 0.593 |
| canónico enmascarado | reseña como documento (sin LLM) | 0.358 | 0.617 | 0.485 |
| canónico enmascarado | consulta cruda (search_query) | 0.375 | 0.635 | 0.496 |

Con la reseña como documento, la fuga de nombres inflaba Hit@5 en +7.7 puntos (canónico sin enmascarar − enmascarado); pasar el texto a forma canónica cuesta además +2.8 puntos (original − canónico), efecto del formato y no de la fuga.

HyDE sin enmascarar no se corrió (`--hyde-unmasked 0`): duplicaría las llamadas al LLM y el prompt ya prohíbe nombres propios en el HyDE.

## 4. Ablación

Cada fila cambia una sola pieza del sistema completo. Es descriptiva: los hiperparámetros se eligen en validación (`scripts/evaluate.py`), no con estas tablas.

### Gold (40 consultas)

| Configuración | P@5 | Hit@1 | Hit@5 | MRR | nDCG@5 | Error en bottom |
|---|---|---|---|---|---|---|
| completo (arquetipos, λ, HyDE, bottom Anti-HyDE) | 0.410 | 0.775 | 0.850 | 0.831 | 0.671 | 0.075 |
| 1 centroide por pueblo | 0.415 | 0.625 | 0.925 | 0.778 | 0.650 | 0.075 |
| λ = 0 | 0.410 | 0.775 | 0.850 | 0.830 | 0.671 | 0.075 |
| consulta cruda (search_query, sin HyDE) | 0.435 | 0.875 | 1.000 | 0.931 | 0.782 | 0.075 |
| bottom = reverso del ranking | 0.410 | 0.775 | 0.850 | 0.831 | 0.671 | 0.050 |

### Holdout, submuestra con HyDE (n = 200)

| Configuración | P@5 | Hit@1 | Hit@5 | MRR | nDCG@5 | Error en bottom |
|---|---|---|---|---|---|---|
| completo (arquetipos, λ, HyDE, bottom Anti-HyDE) | 0.088 | 0.215 | 0.440 | 0.336 | 0.330 | 0.090 |
| 1 centroide por pueblo | 0.095 | 0.225 | 0.475 | 0.350 | 0.352 | 0.090 |
| λ = 0 | 0.089 | 0.215 | 0.445 | 0.336 | 0.332 | 0.090 |
| consulta cruda (search_query, sin HyDE) | 0.116 | 0.320 | 0.580 | 0.442 | 0.453 | 0.075 |
| bottom = reverso del ranking | 0.088 | 0.215 | 0.440 | 0.338 | 0.330 | 0.015 |

### Holdout completo, sin LLM (n = 600)

| Consulta | Configuración | Hit@1 | Hit@5 | MRR | Error en bottom |
|---|---|---|---|---|---|
| reseña como documento (sin LLM) | arquetipos, λ elegido | 0.358 | 0.617 | 0.485 | 0.012 |
| reseña como documento (sin LLM) | arquetipos, λ = 0 | 0.362 | 0.615 | 0.485 | 0.012 |
| reseña como documento (sin LLM) | 1 centroide por pueblo, λ elegido | 0.247 | 0.520 | 0.380 | 0.025 |
| consulta cruda (search_query) | arquetipos, λ elegido | 0.375 | 0.635 | 0.496 | 0.010 |
| consulta cruda (search_query) | arquetipos, λ = 0 | 0.378 | 0.632 | 0.498 | 0.007 |
| consulta cruda (search_query) | 1 centroide por pueblo, λ elegido | 0.257 | 0.530 | 0.391 | 0.025 |

## 5. Sesgo por volumen de reseñas

Spearman(apariciones en top 5 del gold, nº de reseñas) = -0.003 (p = 0.984). Pueblo más expuesto: Zacatlan (14 apariciones). Pueblos que nunca aparecen en el top 5 del gold (3): Bernal, Metepec, Taxco.

## 6. Robustez

No se corrió o quedó pendiente por cuota (0 pares pendientes).

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

Dos corridas de las 40 consultas del gold con la caché del LLM: rankings idénticos en 1.000 de las consultas. Huella SHA-256 de los rankings: `7eb8c5ac49b3da8b0e9fced36b2828a9105e415d864653b1bf4d5a8328edda14`.

Sin caché el sistema **no** es determinista: el LLM genera HyDE y Anti-HyDE con temperature 0.4, así que dos llamadas con la misma consulta producen textos distintos y el ranking puede cambiar. El determinismo que se reporta es el de la caché en disco (`artifacts/llm_cache`): la primera respuesta válida de cada prompt queda fija, y con ella el encoder y el ranking son reproducibles. Borrar la caché cambia la huella.

## 9. Consultas con fallos (Hit@5 = 0 o relevante en el bottom 5)

- **cenotes** (relevante en bottom: Izamal) “Busco cenotes para nadar y naturaleza” → top Tulum, Valladolid, Loreto, Bacalar, Cuatro_Cienegas; bottom Izamal, Palenque, Ixtapan_de_la_Sal, Valle_de_Bravo, Sayulita; esperados Valladolid, Tulum, Bacalar, Izamal
- **colonial** (relevante en bottom: Patzcuaro, Atlixco) “Deseo arquitectura colonial, iglesias y calles históricas” → top Dolores_Hidalgo, Teotihuacan, Tequisquiapan, Cholula, Tepotzotlan; bottom Tlaquepaque, Patzcuaro, Atlixco, TodosSantos, Ixtapan_de_la_Sal; esperados Taxco, San_Cristobal_de_las_Casas, Patzcuaro, Izamal, Cholula, Dolores_Hidalgo, Valladolid, Tepotzotlan, Orizaba, Atlixco
- **termales** (Hit@5 = 0) “Busco un viaje relajante con spa y aguas termales” → top Tepoztlan, Malinalco, Valle_de_Bravo, Loreto, Atlixco; bottom Tlaquepaque, TodosSantos, Valladolid, Tulum, Patzcuaro; esperados Ixtapan_de_la_Sal, Tequisquiapan
- **plata** (Hit@5 = 0) “Comprar joyería de plata y recorrer calles empedradas en cerro” → top Huasca_de_Ocampo, Coatepec, Atlixco, Tequisquiapan, Real_de_Catorce; bottom Tulum, Loreto, Mazunte, Isla_Mujeres, Izamal; esperados Taxco
- **teleferico** (Hit@5 = 0; relevante en bottom: Orizaba, Zacatlan) “Subir en teleférico y visitar museos en la ciudad” → top Tlaquepaque, Atlixco, Tequisquiapan, Teotihuacan, Coatepec; bottom Ixtapan_de_la_Sal, Tapalpa, Orizaba, Tepoztlan, Zacatlan; esperados Orizaba, Taxco, Tepotzotlan, Zacatlan
- **monolito** (Hit@5 = 0) “Escalar un monolito enorme y comer gorditas” → top Tepoztlan, Tapalpa, Orizaba, Malinalco, Zacatlan; bottom Ixtapan_de_la_Sal, Atlixco, Tlaquepaque, Tequisquiapan, Patzcuaro; esperados Bernal
- **prismas** (Hit@5 = 0) “Formaciones de prismas basálticos, haciendas y tirolesa” → top Creel, Chiapa_de_Corzo, Palenque, Zacatlan, Tapalpa; bottom Orizaba, Atlixco, Tlaquepaque, Tequisquiapan, Ixtapan_de_la_Sal; esperados Huasca_de_Ocampo
- **independencia** (Hit@5 = 0) “Historia de la independencia de México y nieves de sabores raros” → top Tlaquepaque, Tequisquiapan, Tepoztlan, Valladolid, Coatepec; bottom Sayulita, Tulum, Isla_Mujeres, TodosSantos, Loreto; esperados Dolores_Hidalgo

## Exportación para la evaluación humana ciega

`reports/exp/2026-11-16_enc_gemma/rankings_query_bank.json`: ranking completo de 60 de 60 consultas de `eval/query_bank.json` (rechazadas por la validación: 0; pendientes por cuota: 0).
