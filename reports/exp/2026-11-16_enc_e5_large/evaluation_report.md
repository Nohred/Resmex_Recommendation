# Reporte de evaluación automática

- **Encoder:** st:intfloat/multilingual-e5-large@len512@fp16 · **k:** 10 arquetipos por pueblo (400 en 40 pueblos) · **λ:** 0.01
- **Cadena de LLM (evaluación):** groq/openai/gpt-oss-120b → ollama/qwen2.5:3b → plantilla · temperature 0.4
- **LLM usado en esta corrida:** groq:openai/gpt-oss-120b × 401, ollama:qwen2.5:3b × 24 (0 llamadas nuevas, 425 desde la caché)
- **Commit:** aeb9d2c · **Fecha:** 2026-11-16T13:27:40-06:00 · **Duración:** 39.9 s · **Config:** `configs/2026-11-16_enc_e5_large.yaml`

> Protocolo, métricas y umbrales del equipo (`restmex-orbita`, commit 3ecbcfd). “5 mejores” = mayor afinidad con la consulta; “5 peores” = los más parecidos al Anti-HyDE (lo opuesto a lo pedido), no los de peor calidad.

## 1. Rúbrica

| Criterio | Valor | Umbral | Estado |
|---|---|---|---|
| Precision@5 (gold, 11 consultas con ≥5 relevantes) | 0.727 | 0.7 | ✅ |
| Hit@1 (gold) | 0.725 | 0.8 | ❌ |
| Contrato 5 + 5 | 1.000 | 1.0 | ✅ |
| Determinismo (con caché de LLM) | 1.000 | 1.0 | ✅ |
| Latencia p95 en caliente, LLM en caché (s, n = 40) | 0.021 | 3.0 | ✅ |
| Latencia p95 en frío, sin caché de LLM (s, n = 0) | — | 3.0 | — |

Latencia en proceso de `Recommender.recommend` (LLM + embeddings + búsqueda). En frío: p50 — s, p95 — s, máx. — s (n = 0, consultas que no estaban en la caché del LLM; la pausa entre llamadas para respetar la cuota no se cuenta). En caliente: p50 0.021 s, p95 0.021 s, máx. 0.023 s (n = 40). La web usa primero Gemini; aquí la latencia en frío es la de la cadena de evaluación.

## 2. Gold (seed heurístico del equipo, pendiente de adjudicación humana)

`eval/gold_queries.json` copiado sin cambios (SHA-256 `e565ef79cbc66d35…`; autor: asistente de IA (Claude) a petición del equipo; vio salidas del sistema del equipo: sí). Pipeline completo de la web (HyDE + Anti-HyDE + λ = 0.01) sobre 40 consultas.

| Métrica | Sistema | Popularidad | Azar |
|---|---|---|---|
| precision_at_5 | 0.425 | 0.120 | 0.095 |
| capped_recall_at_5 | 0.729 | 0.193 | 0.143 |
| hit_at_1 | 0.725 | 0.050 | 0.096 |
| hit_at_5 | 0.950 | 0.450 | 0.371 |
| mrr | 0.835 | 0.234 | 0.238 |
| ndcg_at_5 | 0.708 | 0.141 | 0.119 |
| bottom_error_rate | 0.175 | 0.300 | 0.364 |
| exclusion_violation_rate | 0.667 | 0.000 | 0.422 |

Top/Bottom de `recommend` idénticos al ranking evaluado: 1.000. Consultas resueltas con la plantilla sin LLM: 0. Las 3 consultas con exclusión miden `exclusion_violation_rate`; este sistema no interpreta exclusiones de forma explícita.

Por partición del banco de consultas del equipo (`test` no se usa para ajustar nada):

| Partición | Consultas | P@5 | Hit@1 | Hit@5 | MRR | nDCG@5 |
|---|---|---|---|---|---|---|
| calibration | 6 | 0.433 | 0.667 | 0.833 | 0.760 | 0.716 |
| dev | 19 | 0.463 | 0.737 | 1.000 | 0.868 | 0.765 |
| test | 15 | 0.373 | 0.733 | 0.933 | 0.822 | 0.633 |

## 3. Holdout enmascarado (sin consultas escritas a mano)

Cada reseña reservada de validación (no vista al construir los arquetipos) actúa como consulta con nombres de pueblo y estado enmascarados; el destino correcto es su pueblo.

600 reseñas de `val` (polaridad 4–5, 80–600 caracteres, 15 por pueblo, seed 2025). El enmascarado modificó el 53.3 % de las reseñas. El texto enmascarado queda en forma canónica (minúsculas, sin acentos ni puntuación), como en el equipo.

| Variante | n | Hit@1 | Hit@5 | MRR | Pueblo correcto en bottom 5 | Spearman(Hit@5 por pueblo, nº reseñas) |
|---|---|---|---|---|---|---|
| reseña como documento (sin LLM) | 600 | 0.323 | 0.593 | 0.452 | 0.013 | -0.180 |
| consulta cruda (search_query) | 600 | 0.318 | 0.560 | 0.439 | 0.012 | -0.203 |
| HyDE + Anti-HyDE (pipeline web) | 200 | 0.170 | 0.435 | 0.299 | 0.130 | -0.378 |
| Popularidad | 600 | 0.025 | 0.125 | 0.107 | 0.125 | — |
| Azar (esperado) | — | 0.025 | 0.125 | — | 0.125 | — |

HyDE se evalúa en una submuestra balanceada (`--hyde-holdout 200`; pendientes por cuota: 0; LLM: groq:openai/gpt-oss-120b × 200). En las variantes sin Anti-HyDE el bottom es el reverso del ranking. Mismas 200 reseñas en las tres variantes:

| Variante (misma submuestra) | Hit@1 | Hit@5 | MRR |
|---|---|---|---|
| reseña como documento (sin LLM) | 0.315 | 0.575 | 0.445 |
| consulta cruda (search_query) | 0.295 | 0.535 | 0.416 |
| HyDE + Anti-HyDE (pipeline web) | 0.170 | 0.435 | 0.299 |

### Fuga de nombres: el mismo holdout sin enmascarar

| Texto de la consulta | Variante | Hit@1 | Hit@5 | MRR |
|---|---|---|---|---|
| original (sin enmascarar) | reseña como documento (sin LLM) | 0.468 | 0.707 | 0.579 |
| original (sin enmascarar) | consulta cruda (search_query) | 0.468 | 0.660 | 0.563 |
| canónico sin enmascarar | reseña como documento (sin LLM) | 0.460 | 0.683 | 0.563 |
| canónico sin enmascarar | consulta cruda (search_query) | 0.437 | 0.637 | 0.539 |
| canónico enmascarado | reseña como documento (sin LLM) | 0.323 | 0.593 | 0.452 |
| canónico enmascarado | consulta cruda (search_query) | 0.318 | 0.560 | 0.439 |

Con la reseña como documento, la fuga de nombres inflaba Hit@5 en +9.0 puntos (canónico sin enmascarar − enmascarado); pasar el texto a forma canónica cuesta además +2.3 puntos (original − canónico), efecto del formato y no de la fuga.

HyDE sin enmascarar no se corrió (`--hyde-unmasked 0`): duplicaría las llamadas al LLM y el prompt ya prohíbe nombres propios en el HyDE.

## 4. Ablación

Cada fila cambia una sola pieza del sistema completo. Es descriptiva: los hiperparámetros se eligen en validación (`scripts/evaluate.py`), no con estas tablas.

### Gold (40 consultas)

| Configuración | P@5 | Hit@1 | Hit@5 | MRR | nDCG@5 | Error en bottom |
|---|---|---|---|---|---|---|
| completo (arquetipos, λ, HyDE, bottom Anti-HyDE) | 0.425 | 0.725 | 0.950 | 0.835 | 0.708 | 0.175 |
| 1 centroide por pueblo | 0.375 | 0.600 | 0.925 | 0.723 | 0.611 | 0.125 |
| λ = 0 | 0.425 | 0.725 | 0.950 | 0.835 | 0.711 | 0.175 |
| consulta cruda (search_query, sin HyDE) | 0.415 | 0.900 | 0.975 | 0.934 | 0.771 | 0.175 |
| bottom = reverso del ranking | 0.425 | 0.725 | 0.950 | 0.835 | 0.708 | 0.025 |

### Holdout, submuestra con HyDE (n = 200)

| Configuración | P@5 | Hit@1 | Hit@5 | MRR | nDCG@5 | Error en bottom |
|---|---|---|---|---|---|---|
| completo (arquetipos, λ, HyDE, bottom Anti-HyDE) | 0.087 | 0.170 | 0.435 | 0.299 | 0.303 | 0.130 |
| 1 centroide por pueblo | 0.083 | 0.190 | 0.415 | 0.299 | 0.297 | 0.125 |
| λ = 0 | 0.082 | 0.155 | 0.410 | 0.289 | 0.286 | 0.130 |
| consulta cruda (search_query, sin HyDE) | 0.107 | 0.295 | 0.535 | 0.415 | 0.417 | 0.080 |
| bottom = reverso del ranking | 0.087 | 0.170 | 0.435 | 0.303 | 0.303 | 0.015 |

### Holdout completo, sin LLM (n = 600)

| Consulta | Configuración | Hit@1 | Hit@5 | MRR | Error en bottom |
|---|---|---|---|---|---|
| reseña como documento (sin LLM) | arquetipos, λ elegido | 0.323 | 0.593 | 0.452 | 0.013 |
| reseña como documento (sin LLM) | arquetipos, λ = 0 | 0.327 | 0.592 | 0.451 | 0.010 |
| reseña como documento (sin LLM) | 1 centroide por pueblo, λ elegido | 0.223 | 0.492 | 0.358 | 0.017 |
| consulta cruda (search_query) | arquetipos, λ elegido | 0.318 | 0.560 | 0.439 | 0.012 |
| consulta cruda (search_query) | arquetipos, λ = 0 | 0.313 | 0.560 | 0.437 | 0.012 |
| consulta cruda (search_query) | 1 centroide por pueblo, λ elegido | 0.230 | 0.492 | 0.361 | 0.018 |

## 5. Sesgo por volumen de reseñas

Spearman(apariciones en top 5 del gold, nº de reseñas) = -0.095 (p = 0.561). Pueblo más expuesto: Malinalco (14 apariciones). Pueblos que nunca aparecen en el top 5 del gold (2): Metepec, Tlaquepaque.

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

Dos corridas de las 40 consultas del gold con la caché del LLM: rankings idénticos en 1.000 de las consultas. Huella SHA-256 de los rankings: `b8e46c4c2a552307e4a88ff5c2f0805809a9bf89e6295efac50aa310436f7e5a`.

Sin caché el sistema **no** es determinista: el LLM genera HyDE y Anti-HyDE con temperature 0.4, así que dos llamadas con la misma consulta producen textos distintos y el ranking puede cambiar. El determinismo que se reporta es el de la caché en disco (`artifacts/llm_cache`): la primera respuesta válida de cada prompt queda fija, y con ella el encoder y el ranking son reproducibles. Borrar la caché cambia la huella.

## 9. Consultas con fallos (Hit@5 = 0 o relevante en el bottom 5)

- **playa-snorkel** (relevante en bottom: TodosSantos) “Quiero playas tranquilas, agua turquesa y snorkel” → top Tulum, Sayulita, Loreto, Mazunte, Isla_Mujeres; bottom Cuatro_Cienegas, TodosSantos, Taxco, Zacatlan, Tapalpa; esperados Isla_Mujeres, Tulum, Sayulita, Mazunte, TodosSantos, Loreto, Bacalar
- **vino** (relevante en bottom: Tequisquiapan) “Me interesan viñedos, vino y recorridos por bodegas” → top Dolores_Hidalgo, Parras, Tequila, Atlixco, Cuetzalan; bottom Ixtapan_de_la_Sal, Tepoztlan, TodosSantos, Tequisquiapan, Valladolid; esperados Parras, Dolores_Hidalgo, Bernal, Tequisquiapan
- **cenotes** (relevante en bottom: Izamal) “Busco cenotes para nadar y naturaleza” → top Valladolid, Tulum, Cuatro_Cienegas, Palenque, Zacatlan; bottom Izamal, Cuetzalan, Ixtapan_de_la_Sal, Malinalco, Loreto; esperados Valladolid, Tulum, Bacalar, Izamal
- **gastronomia** (relevante en bottom: Valladolid) “Quiero gastronomía mexicana, mercados y restaurantes tradicionales” → top Atlixco, Tequila, Izamal, Coatepec, Taxco; bottom TodosSantos, Teotihuacan, Valladolid, Loreto, Mazunte; esperados Patzcuaro, Tlaquepaque, Izamal, Valladolid, San_Cristobal_de_las_Casas, Tepoztlan, Cholula, Atlixco, Ajijic
- **colonial** (relevante en bottom: Orizaba, Atlixco, Patzcuaro) “Deseo arquitectura colonial, iglesias y calles históricas” → top Dolores_Hidalgo, Malinalco, Tequisquiapan, Teotihuacan, Izamal; bottom Ixtapan_de_la_Sal, Tlaquepaque, Orizaba, Atlixco, Patzcuaro; esperados Taxco, San_Cristobal_de_las_Casas, Patzcuaro, Izamal, Cholula, Dolores_Hidalgo, Valladolid, Tepotzotlan, Orizaba, Atlixco
- **artesania** (relevante en bottom: Tlaquepaque) “Artesanías de barro, mariachi y tiendas de arte popular” → top Tequisquiapan, Patzcuaro, Tequila, Coatepec, Atlixco; bottom Ixtapan_de_la_Sal, Tlaquepaque, Teotihuacan, Real_de_Catorce, Tepoztlan; esperados Tlaquepaque, Metepec, Patzcuaro, Atlixco
- **relojes** (Hit@5 = 0) “Mirador sobre la barranca, fábrica de relojes y sidra” → top Tequila, Taxco, Dolores_Hidalgo, Parras, Patzcuaro; bottom Sayulita, Tulum, Ixtapan_de_la_Sal, Isla_Mujeres, Atlixco; esperados Zacatlan
- **independencia** (Hit@5 = 0) “Historia de la independencia de México y nieves de sabores raros” → top Patzcuaro, Tequila, Coatepec, Atlixco, Tepoztlan; bottom Sayulita, TodosSantos, Ixtapan_de_la_Sal, Tulum, Isla_Mujeres; esperados Dolores_Hidalgo
- **familia** (relevante en bottom: Valle_de_Bravo) “Vacaciones familiares con niños, albercas y toboganes” → top Tulum, Ixtapan_de_la_Sal, Loreto, Cuetzalan, Izamal; bottom Huasca_de_Ocampo, Valle_de_Bravo, Tepoztlan, Teotihuacan, Atlixco; esperados Ixtapan_de_la_Sal, Tequisquiapan, Valle_de_Bravo

## Exportación para la evaluación humana ciega

`reports/exp/2026-11-16_enc_e5_large/rankings_query_bank.json`: ranking completo de 60 de 60 consultas de `eval/query_bank.json` (rechazadas por la validación: 0; pendientes por cuota: 0).
