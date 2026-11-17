# Reporte de evaluación automática

- **Encoder:** ollama:nomic-embed-text-v2-moe · **k:** 10 arquetipos por pueblo (400 en 40 pueblos) · **λ:** 0.015
- **Cadena de LLM (evaluación):** groq/openai/gpt-oss-120b → ollama/qwen2.5:3b → plantilla · temperature 0.4
- **LLM usado en esta corrida:** groq:openai/gpt-oss-120b × 201, ollama:qwen2.5:3b × 24 (64 llamadas nuevas, 161 desde la caché)
- **Commit:** f9152a8 · **Fecha:** 2026-11-16T09:43:37-06:00 · **Duración:** 782.7 s · **Config:** `configs/2026-11-16_baseline_nomic_k10.yaml`

> Protocolo, métricas y umbrales del equipo (`restmex-orbita`, commit 3ecbcfd). “5 mejores” = mayor afinidad con la consulta; “5 peores” = los más parecidos al Anti-HyDE (lo opuesto a lo pedido), no los de peor calidad.

## 1. Rúbrica

| Criterio | Valor | Umbral | Estado |
|---|---|---|---|
| Precision@5 (gold, 11 consultas con ≥5 relevantes) | 0.691 | 0.7 | ❌ |
| Hit@1 (gold) | 0.650 | 0.8 | ❌ |
| Contrato 5 + 5 | 1.000 | 1.0 | ✅ |
| Determinismo (con caché de LLM) | 1.000 | 1.0 | ✅ |
| Latencia p95 en caliente, LLM en caché (s, n = 40) | 0.030 | 3.0 | ✅ |
| Latencia p95 en frío, sin caché de LLM (s, n = 39) | 2.674 | 3.0 | ✅ |

Latencia en proceso de `Recommender.recommend` (LLM + embeddings + búsqueda). En frío: p50 1.760 s, p95 2.674 s, máx. 3.704 s (n = 39, consultas que no estaban en la caché del LLM; la pausa entre llamadas para respetar la cuota no se cuenta). En caliente: p50 0.024 s, p95 0.030 s, máx. 0.031 s (n = 40). La web usa primero Gemini; aquí la latencia en frío es la de la cadena de evaluación.

## 2. Gold (seed heurístico del equipo, pendiente de adjudicación humana)

`eval/gold_queries.json` copiado sin cambios (SHA-256 `e565ef79cbc66d35…`; autor: asistente de IA (Claude) a petición del equipo; vio salidas del sistema del equipo: sí). Pipeline completo de la web (HyDE + Anti-HyDE + λ = 0.015) sobre 40 consultas.

| Métrica | Sistema | Popularidad | Azar |
|---|---|---|---|
| precision_at_5 | 0.380 | 0.135 | 0.095 |
| capped_recall_at_5 | 0.650 | 0.182 | 0.143 |
| hit_at_1 | 0.650 | 0.175 | 0.096 |
| hit_at_5 | 0.875 | 0.375 | 0.371 |
| mrr | 0.767 | 0.278 | 0.238 |
| ndcg_at_5 | 0.631 | 0.162 | 0.119 |
| bottom_error_rate | 0.125 | 0.300 | 0.364 |
| exclusion_violation_rate | 0.667 | 1.000 | 0.422 |

Top/Bottom de `recommend` idénticos al ranking evaluado: 1.000. Consultas resueltas con la plantilla sin LLM: 0. Las 3 consultas con exclusión miden `exclusion_violation_rate`; este sistema no interpreta exclusiones de forma explícita.

Por partición del banco de consultas del equipo (`test` no se usa para ajustar nada):

| Partición | Consultas | P@5 | Hit@1 | Hit@5 | MRR | nDCG@5 |
|---|---|---|---|---|---|---|
| calibration | 6 | 0.433 | 0.667 | 0.833 | 0.719 | 0.677 |
| dev | 19 | 0.410 | 0.737 | 0.947 | 0.850 | 0.694 |
| test | 15 | 0.320 | 0.533 | 0.800 | 0.681 | 0.534 |

## 3. Holdout enmascarado (sin consultas escritas a mano)

Cada reseña reservada de validación (no vista al construir los arquetipos) actúa como consulta con nombres de pueblo y estado enmascarados; el destino correcto es su pueblo.

600 reseñas de `val` (polaridad 4–5, 80–600 caracteres, 15 por pueblo, seed 2025). El enmascarado modificó el 60.3 % de las reseñas. El texto enmascarado queda en forma canónica (minúsculas, sin acentos ni puntuación), como en el equipo.

| Variante | n | Hit@1 | Hit@5 | MRR | Pueblo correcto en bottom 5 | Spearman(Hit@5 por pueblo, nº reseñas) |
|---|---|---|---|---|---|---|
| reseña como documento (sin LLM) | 600 | 0.382 | 0.643 | 0.508 | 0.008 | -0.030 |
| consulta cruda (search_query) | 600 | 0.373 | 0.645 | 0.502 | 0.008 | -0.108 |
| HyDE + Anti-HyDE (pipeline web) | 0 | — | — | — | — | — |
| Popularidad | 600 | 0.025 | 0.125 | 0.107 | 0.125 | — |
| Azar (esperado) | — | 0.025 | 0.125 | — | 0.125 | — |

HyDE se evalúa en una submuestra balanceada (`--hyde-holdout 0`; pendientes por cuota: 0; LLM: —). En las variantes sin Anti-HyDE el bottom es el reverso del ranking. Mismas 0 reseñas en las tres variantes:

| Variante (misma submuestra) | Hit@1 | Hit@5 | MRR |
|---|---|---|---|
| reseña como documento (sin LLM) | 0.000 | 0.000 | 0.000 |
| consulta cruda (search_query) | 0.000 | 0.000 | 0.000 |
| HyDE + Anti-HyDE (pipeline web) | 0.000 | 0.000 | 0.000 |

### Fuga de nombres: el mismo holdout sin enmascarar

| Texto de la consulta | Variante | Hit@1 | Hit@5 | MRR |
|---|---|---|---|---|
| original (sin enmascarar) | reseña como documento (sin LLM) | 0.547 | 0.747 | 0.642 |
| original (sin enmascarar) | consulta cruda (search_query) | 0.543 | 0.737 | 0.635 |
| canónico sin enmascarar | reseña como documento (sin LLM) | 0.515 | 0.732 | 0.616 |
| canónico sin enmascarar | consulta cruda (search_query) | 0.520 | 0.737 | 0.621 |
| canónico enmascarado | reseña como documento (sin LLM) | 0.382 | 0.643 | 0.508 |
| canónico enmascarado | consulta cruda (search_query) | 0.373 | 0.645 | 0.502 |

Con la reseña como documento, la fuga de nombres inflaba Hit@5 en +8.8 puntos (canónico sin enmascarar − enmascarado); pasar el texto a forma canónica cuesta además +1.5 puntos (original − canónico), efecto del formato y no de la fuga.

HyDE sin enmascarar no se corrió (`--hyde-unmasked 0`): duplicaría las llamadas al LLM y el prompt ya prohíbe nombres propios en el HyDE.

## 4. Ablación

Cada fila cambia una sola pieza del sistema completo. Es descriptiva: los hiperparámetros se eligen en validación (`scripts/evaluate.py`), no con estas tablas.

### Gold (40 consultas)

| Configuración | P@5 | Hit@1 | Hit@5 | MRR | nDCG@5 | Error en bottom |
|---|---|---|---|---|---|---|
| completo (arquetipos, λ, HyDE, bottom Anti-HyDE) | 0.380 | 0.650 | 0.875 | 0.767 | 0.631 | 0.125 |
| 1 centroide por pueblo | 0.300 | 0.475 | 0.775 | 0.619 | 0.497 | 0.175 |
| λ = 0 | 0.375 | 0.675 | 0.875 | 0.772 | 0.632 | 0.125 |
| consulta cruda (search_query, sin HyDE) | 0.400 | 0.825 | 0.975 | 0.889 | 0.746 | 0.200 |
| bottom = reverso del ranking | 0.380 | 0.650 | 0.875 | 0.766 | 0.631 | 0.025 |

### Holdout, submuestra con HyDE (n = 0)

| Configuración | P@5 | Hit@1 | Hit@5 | MRR | nDCG@5 | Error en bottom |
|---|---|---|---|---|---|---|

### Holdout completo, sin LLM (n = 600)

| Consulta | Configuración | Hit@1 | Hit@5 | MRR | Error en bottom |
|---|---|---|---|---|---|
| reseña como documento (sin LLM) | arquetipos, λ elegido | 0.382 | 0.643 | 0.508 | 0.008 |
| reseña como documento (sin LLM) | arquetipos, λ = 0 | 0.383 | 0.643 | 0.510 | 0.008 |
| reseña como documento (sin LLM) | 1 centroide por pueblo, λ elegido | 0.228 | 0.483 | 0.354 | 0.010 |
| consulta cruda (search_query) | arquetipos, λ elegido | 0.373 | 0.645 | 0.502 | 0.008 |
| consulta cruda (search_query) | arquetipos, λ = 0 | 0.368 | 0.648 | 0.499 | 0.010 |
| consulta cruda (search_query) | 1 centroide por pueblo, λ elegido | 0.225 | 0.493 | 0.358 | 0.010 |

## 5. Sesgo por volumen de reseñas

Spearman(apariciones en top 5 del gold, nº de reseñas) = 0.103 (p = 0.525). Pueblo más expuesto: Valladolid (23 apariciones). Pueblos que nunca aparecen en el top 5 del gold (4): Bernal, Metepec, San_Cristobal_de_las_Casas, Tepotzotlan.

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

Dos corridas de las 40 consultas del gold con la caché del LLM: rankings idénticos en 1.000 de las consultas. Huella SHA-256 de los rankings: `0ef15a08edc0653db4b66085733bb9a8458a6cc8f47080cd9630dbb1ee88c16a`.

Sin caché el sistema **no** es determinista: el LLM genera HyDE y Anti-HyDE con temperature 0.4, así que dos llamadas con la misma consulta producen textos distintos y el ranking puede cambiar. El determinismo que se reporta es el de la caché en disco (`artifacts/llm_cache`): la primera respuesta válida de cada prompt queda fija, y con ella el encoder y el ranking son reproducibles. Borrar la caché cambia la huella.

## 9. Consultas con fallos (Hit@5 = 0 o relevante en el bottom 5)

- **vino** (relevante en bottom: Tequisquiapan) “Me interesan viñedos, vino y recorridos por bodegas” → top Dolores_Hidalgo, Parras, Valladolid, Atlixco, TodosSantos; bottom Tequisquiapan, Metepec, Patzcuaro, Tepoztlan, Tlaquepaque; esperados Parras, Dolores_Hidalgo, Bernal, Tequisquiapan
- **cenotes** (relevante en bottom: Bacalar) “Busco cenotes para nadar y naturaleza” → top Tulum, Valladolid, Loreto, Xilitla, TodosSantos; bottom Ixtapan_de_la_Sal, Bacalar, Atlixco, Metepec, Tepoztlan; esperados Valladolid, Tulum, Bacalar, Izamal
- **colonial** (relevante en bottom: Orizaba, Atlixco) “Deseo arquitectura colonial, iglesias y calles históricas” → top Dolores_Hidalgo, Malinalco, Tequisquiapan, Teotihuacan, Valladolid; bottom Tlaquepaque, TodosSantos, Metepec, Orizaba, Atlixco; esperados Taxco, San_Cristobal_de_las_Casas, Patzcuaro, Izamal, Cholula, Dolores_Hidalgo, Valladolid, Tepotzotlan, Orizaba, Atlixco
- **lago** (relevante en bottom: Patzcuaro) “Caminar junto al lago, malecón tranquilo y comida con vista al agua” → top Loreto, Ajijic, Creel, Bacalar, Valladolid; bottom TodosSantos, Atlixco, Tequisquiapan, Patzcuaro, Tepotzotlan; esperados Ajijic, Patzcuaro, Valle_de_Bravo, Bacalar, Creel
- **teleferico** (Hit@5 = 0; relevante en bottom: Zacatlan) “Subir en teleférico y visitar museos en la ciudad” → top Teotihuacan, Tlaquepaque, Xilitla, Valladolid, Tepoztlan; bottom TodosSantos, Zacatlan, Atlixco, Tapalpa, Bacalar; esperados Orizaba, Taxco, Tepotzotlan, Zacatlan
- **monolito** (Hit@5 = 0) “Escalar un monolito enorme y comer gorditas” → top Tepoztlan, Tapalpa, TodosSantos, Cuetzalan, Huasca_de_Ocampo; bottom Tequisquiapan, Atlixco, Valladolid, Ixtapan_de_la_Sal, Ajijic; esperados Bernal
- **barrancas** (Hit@5 = 0) “Barrancas profundas, tren y cultura rarámuri” → top Zacatlan, Valladolid, Huasca_de_Ocampo, Chiapa_de_Corzo, Tapalpa; bottom TodosSantos, Loreto, Sayulita, Ixtapan_de_la_Sal, Tequisquiapan; esperados Creel
- **independencia** (Hit@5 = 0) “Historia de la independencia de México y nieves de sabores raros” → top Valladolid, Tequila, Tlaquepaque, Atlixco, Patzcuaro; bottom TodosSantos, Sayulita, Metepec, Bacalar, Tulum; esperados Dolores_Hidalgo
- **virreinato** (Hit@5 = 0) “Museo del virreinato y arte religioso barroco” → top Valladolid, Malinalco, Orizaba, Tequila, Tepoztlan; bottom Sayulita, TodosSantos, Isla_Mujeres, Loreto, Bacalar; esperados Tepotzotlan, Taxco, Cholula, Patzcuaro

## Exportación para la evaluación humana ciega

`reports/exp/2026-11-16_baseline_nomic_k10/rankings_query_bank.json`: ranking completo de 60 de 60 consultas de `eval/query_bank.json` (rechazadas por la validación: 0; pendientes por cuota: 0).
