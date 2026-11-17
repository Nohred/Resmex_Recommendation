# Reporte de evaluación automática

- **Encoder:** ollama:embeddinggemma · **k:** 10 arquetipos por pueblo (400 en 40 pueblos) · **λ:** 0.01
- **Cadena de LLM (evaluación):** groq/openai/gpt-oss-120b → ollama/qwen2.5:3b → plantilla · temperature 0.4
- **LLM usado en esta corrida:** groq:openai/gpt-oss-120b × 401, ollama:qwen2.5:3b × 24 (0 llamadas nuevas, 425 desde la caché)
- **Commit:** 390e092 · **Fecha:** 2026-11-16T14:12:13-06:00 · **Duración:** 267.5 s · **Config:** `configs/2026-11-16_enc_gemma_largo.yaml`

> Protocolo, métricas y umbrales del equipo (`restmex-orbita`, commit 3ecbcfd). “5 mejores” = mayor afinidad con la consulta; “5 peores” = los más parecidos al Anti-HyDE (lo opuesto a lo pedido), no los de peor calidad.

## 1. Rúbrica

| Criterio | Valor | Umbral | Estado |
|---|---|---|---|
| Precision@5 (gold, 11 consultas con ≥5 relevantes) | 0.764 | 0.7 | ✅ |
| Hit@1 (gold) | 0.775 | 0.8 | ❌ |
| Contrato 5 + 5 | 1.000 | 1.0 | ✅ |
| Determinismo (con caché de LLM) | 1.000 | 1.0 | ✅ |
| Latencia p95 en caliente, LLM en caché (s, n = 40) | 0.084 | 3.0 | ✅ |
| Latencia p95 en frío, sin caché de LLM (s, n = 0) | — | 3.0 | — |

Latencia en proceso de `Recommender.recommend` (LLM + embeddings + búsqueda). En frío: p50 — s, p95 — s, máx. — s (n = 0, consultas que no estaban en la caché del LLM; la pausa entre llamadas para respetar la cuota no se cuenta). En caliente: p50 0.059 s, p95 0.084 s, máx. 0.086 s (n = 40). La web usa primero Gemini; aquí la latencia en frío es la de la cadena de evaluación.

## 2. Gold (seed heurístico del equipo, pendiente de adjudicación humana)

`eval/gold_queries.json` copiado sin cambios (SHA-256 `e565ef79cbc66d35…`; autor: asistente de IA (Claude) a petición del equipo; vio salidas del sistema del equipo: sí). Pipeline completo de la web (HyDE + Anti-HyDE + λ = 0.01) sobre 40 consultas.

| Métrica | Sistema | Popularidad | Azar |
|---|---|---|---|
| precision_at_5 | 0.415 | 0.120 | 0.095 |
| capped_recall_at_5 | 0.704 | 0.193 | 0.143 |
| hit_at_1 | 0.775 | 0.050 | 0.096 |
| hit_at_5 | 0.900 | 0.450 | 0.371 |
| mrr | 0.831 | 0.234 | 0.238 |
| ndcg_at_5 | 0.690 | 0.141 | 0.119 |
| bottom_error_rate | 0.150 | 0.300 | 0.364 |
| exclusion_violation_rate | 0.667 | 0.000 | 0.422 |

Top/Bottom de `recommend` idénticos al ranking evaluado: 1.000. Consultas resueltas con la plantilla sin LLM: 0. Las 3 consultas con exclusión miden `exclusion_violation_rate`; este sistema no interpreta exclusiones de forma explícita.

Por partición del banco de consultas del equipo (`test` no se usa para ajustar nada):

| Partición | Consultas | P@5 | Hit@1 | Hit@5 | MRR | nDCG@5 |
|---|---|---|---|---|---|---|
| calibration | 6 | 0.467 | 0.833 | 1.000 | 0.875 | 0.817 |
| dev | 19 | 0.432 | 0.737 | 0.895 | 0.803 | 0.669 |
| test | 15 | 0.373 | 0.800 | 0.867 | 0.848 | 0.665 |

## 3. Holdout enmascarado (sin consultas escritas a mano)

Cada reseña reservada de validación (no vista al construir los arquetipos) actúa como consulta con nombres de pueblo y estado enmascarados; el destino correcto es su pueblo.

600 reseñas de `val` (polaridad 4–5, 80–600 caracteres, 15 por pueblo, seed 2025). El enmascarado modificó el 53.3 % de las reseñas. El texto enmascarado queda en forma canónica (minúsculas, sin acentos ni puntuación), como en el equipo.

| Variante | n | Hit@1 | Hit@5 | MRR | Pueblo correcto en bottom 5 | Spearman(Hit@5 por pueblo, nº reseñas) |
|---|---|---|---|---|---|---|
| reseña como documento (sin LLM) | 600 | 0.363 | 0.627 | 0.490 | 0.013 | 0.065 |
| consulta cruda (search_query) | 600 | 0.365 | 0.622 | 0.492 | 0.005 | 0.167 |
| HyDE + Anti-HyDE (pipeline web) | 200 | 0.220 | 0.455 | 0.341 | 0.105 | 0.114 |
| Popularidad | 600 | 0.025 | 0.125 | 0.107 | 0.125 | — |
| Azar (esperado) | — | 0.025 | 0.125 | — | 0.125 | — |

HyDE se evalúa en una submuestra balanceada (`--hyde-holdout 200`; pendientes por cuota: 0; LLM: groq:openai/gpt-oss-120b × 200). En las variantes sin Anti-HyDE el bottom es el reverso del ranking. Mismas 200 reseñas en las tres variantes:

| Variante (misma submuestra) | Hit@1 | Hit@5 | MRR |
|---|---|---|---|
| reseña como documento (sin LLM) | 0.335 | 0.615 | 0.466 |
| consulta cruda (search_query) | 0.325 | 0.560 | 0.454 |
| HyDE + Anti-HyDE (pipeline web) | 0.220 | 0.455 | 0.341 |

### Fuga de nombres: el mismo holdout sin enmascarar

| Texto de la consulta | Variante | Hit@1 | Hit@5 | MRR |
|---|---|---|---|---|
| original (sin enmascarar) | reseña como documento (sin LLM) | 0.488 | 0.720 | 0.594 |
| original (sin enmascarar) | consulta cruda (search_query) | 0.498 | 0.713 | 0.601 |
| canónico sin enmascarar | reseña como documento (sin LLM) | 0.477 | 0.698 | 0.583 |
| canónico sin enmascarar | consulta cruda (search_query) | 0.488 | 0.697 | 0.591 |
| canónico enmascarado | reseña como documento (sin LLM) | 0.363 | 0.627 | 0.490 |
| canónico enmascarado | consulta cruda (search_query) | 0.365 | 0.622 | 0.492 |

Con la reseña como documento, la fuga de nombres inflaba Hit@5 en +7.2 puntos (canónico sin enmascarar − enmascarado); pasar el texto a forma canónica cuesta además +2.2 puntos (original − canónico), efecto del formato y no de la fuga.

HyDE sin enmascarar no se corrió (`--hyde-unmasked 0`): duplicaría las llamadas al LLM y el prompt ya prohíbe nombres propios en el HyDE.

## 4. Ablación

Cada fila cambia una sola pieza del sistema completo. Es descriptiva: los hiperparámetros se eligen en validación (`scripts/evaluate.py`), no con estas tablas.

### Gold (40 consultas)

| Configuración | P@5 | Hit@1 | Hit@5 | MRR | nDCG@5 | Error en bottom |
|---|---|---|---|---|---|---|
| completo (arquetipos, λ, HyDE, bottom Anti-HyDE) | 0.415 | 0.775 | 0.900 | 0.831 | 0.690 | 0.150 |
| 1 centroide por pueblo | 0.415 | 0.625 | 0.925 | 0.774 | 0.649 | 0.075 |
| λ = 0 | 0.415 | 0.775 | 0.900 | 0.829 | 0.689 | 0.150 |
| consulta cruda (search_query, sin HyDE) | 0.430 | 0.875 | 1.000 | 0.927 | 0.774 | 0.150 |
| bottom = reverso del ranking | 0.415 | 0.775 | 0.900 | 0.834 | 0.690 | 0.050 |

### Holdout, submuestra con HyDE (n = 200)

| Configuración | P@5 | Hit@1 | Hit@5 | MRR | nDCG@5 | Error en bottom |
|---|---|---|---|---|---|---|
| completo (arquetipos, λ, HyDE, bottom Anti-HyDE) | 0.091 | 0.220 | 0.455 | 0.341 | 0.339 | 0.105 |
| 1 centroide por pueblo | 0.095 | 0.225 | 0.475 | 0.350 | 0.352 | 0.090 |
| λ = 0 | 0.090 | 0.220 | 0.450 | 0.341 | 0.337 | 0.105 |
| consulta cruda (search_query, sin HyDE) | 0.112 | 0.325 | 0.560 | 0.451 | 0.451 | 0.075 |
| bottom = reverso del ranking | 0.091 | 0.220 | 0.455 | 0.344 | 0.339 | 0.030 |

### Holdout completo, sin LLM (n = 600)

| Consulta | Configuración | Hit@1 | Hit@5 | MRR | Error en bottom |
|---|---|---|---|---|---|
| reseña como documento (sin LLM) | arquetipos, λ elegido | 0.363 | 0.627 | 0.490 | 0.013 |
| reseña como documento (sin LLM) | arquetipos, λ = 0 | 0.362 | 0.630 | 0.489 | 0.015 |
| reseña como documento (sin LLM) | 1 centroide por pueblo, λ elegido | 0.253 | 0.522 | 0.383 | 0.025 |
| consulta cruda (search_query) | arquetipos, λ elegido | 0.365 | 0.622 | 0.492 | 0.005 |
| consulta cruda (search_query) | arquetipos, λ = 0 | 0.365 | 0.623 | 0.492 | 0.005 |
| consulta cruda (search_query) | 1 centroide por pueblo, λ elegido | 0.258 | 0.527 | 0.392 | 0.025 |

## 5. Sesgo por volumen de reseñas

Spearman(apariciones en top 5 del gold, nº de reseñas) = 0.041 (p = 0.800). Pueblo más expuesto: Valladolid (15 apariciones). Pueblos que nunca aparecen en el top 5 del gold (2): Bernal, Metepec.

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

Dos corridas de las 40 consultas del gold con la caché del LLM: rankings idénticos en 1.000 de las consultas. Huella SHA-256 de los rankings: `191ca93707e64227e9098a843d2de2fb653b8d707f0e0640ccb7e2a23a8c1069`.

Sin caché el sistema **no** es determinista: el LLM genera HyDE y Anti-HyDE con temperature 0.4, así que dos llamadas con la misma consulta producen textos distintos y el ranking puede cambiar. El determinismo que se reporta es el de la caché en disco (`artifacts/llm_cache`): la primera respuesta válida de cada prompt queda fija, y con ella el encoder y el ranking son reproducibles. Borrar la caché cambia la huella.

## 9. Consultas con fallos (Hit@5 = 0 o relevante en el bottom 5)

- **playa-snorkel** (relevante en bottom: TodosSantos) “Quiero playas tranquilas, agua turquesa y snorkel” → top Loreto, Tulum, Isla_Mujeres, Sayulita, Mazunte; bottom TodosSantos, Tapalpa, Valladolid, Zacatlan, Teotihuacan; esperados Isla_Mujeres, Tulum, Sayulita, Mazunte, TodosSantos, Loreto, Bacalar
- **cenotes** (relevante en bottom: Izamal) “Busco cenotes para nadar y naturaleza” → top Tulum, Valladolid, Loreto, Bacalar, TodosSantos; bottom Izamal, Palenque, Tepoztlan, Ixtapan_de_la_Sal, Valle_de_Bravo; esperados Valladolid, Tulum, Bacalar, Izamal
- **gastronomia** (relevante en bottom: Tepoztlan) “Quiero gastronomía mexicana, mercados y restaurantes tradicionales” → top San_Cristobal_de_las_Casas, Valladolid, Tequila, Tlaquepaque, Zacatlan; bottom Isla_Mujeres, Tulum, Loreto, Tepoztlan, Teotihuacan; esperados Patzcuaro, Tlaquepaque, Izamal, Valladolid, San_Cristobal_de_las_Casas, Tepoztlan, Cholula, Atlixco, Ajijic
- **colonial** (relevante en bottom: Patzcuaro, Atlixco) “Deseo arquitectura colonial, iglesias y calles históricas” → top Dolores_Hidalgo, Tequisquiapan, Tlaquepaque, Valladolid, Teotihuacan; bottom Isla_Mujeres, Patzcuaro, Atlixco, TodosSantos, Ixtapan_de_la_Sal; esperados Taxco, San_Cristobal_de_las_Casas, Patzcuaro, Izamal, Cholula, Dolores_Hidalgo, Valladolid, Tepotzotlan, Orizaba, Atlixco
- **termales** (Hit@5 = 0) “Busco un viaje relajante con spa y aguas termales” → top Tepoztlan, Malinalco, Loreto, Atlixco, Cuetzalan; bottom Isla_Mujeres, Tlaquepaque, Tulum, Valladolid, Patzcuaro; esperados Ixtapan_de_la_Sal, Tequisquiapan
- **lago** (relevante en bottom: Patzcuaro) “Caminar junto al lago, malecón tranquilo y comida con vista al agua” → top Ajijic, Creel, Bacalar, Loreto, Mazunte; bottom Sayulita, Isla_Mujeres, Tlaquepaque, Patzcuaro, TodosSantos; esperados Ajijic, Patzcuaro, Valle_de_Bravo, Bacalar, Creel
- **teleferico** (Hit@5 = 0; relevante en bottom: Orizaba, Zacatlan, Taxco) “Subir en teleférico y visitar museos en la ciudad” → top Tlaquepaque, Atlixco, Tequisquiapan, Teotihuacan, Coatepec; bottom Ixtapan_de_la_Sal, Tapalpa, Orizaba, Zacatlan, Taxco; esperados Orizaba, Taxco, Tepotzotlan, Zacatlan
- **monolito** (Hit@5 = 0) “Escalar un monolito enorme y comer gorditas” → top Tepoztlan, Tapalpa, Orizaba, Malinalco, Zacatlan; bottom Ixtapan_de_la_Sal, Atlixco, Patzcuaro, Tlaquepaque, Tequisquiapan; esperados Bernal
- **prismas** (Hit@5 = 0) “Formaciones de prismas basálticos, haciendas y tirolesa” → top Creel, Chiapa_de_Corzo, Palenque, Valladolid, Zacatlan; bottom Orizaba, Atlixco, Tlaquepaque, Tequisquiapan, Isla_Mujeres; esperados Huasca_de_Ocampo

## Exportación para la evaluación humana ciega

`reports/exp/2026-11-16_enc_gemma_largo/rankings_query_bank.json`: ranking completo de 60 de 60 consultas de `eval/query_bank.json` (rechazadas por la validación: 0; pendientes por cuota: 0).
