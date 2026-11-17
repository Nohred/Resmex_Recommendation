# Reporte de evaluación automática

- **Encoder:** ollama:qwen3-embedding:0.6b · **k:** 10 arquetipos por pueblo (400 en 40 pueblos) · **λ:** 0.04
- **Cadena de LLM (evaluación):** groq/openai/gpt-oss-120b → ollama/qwen2.5:3b → plantilla · temperature 0.4
- **LLM usado en esta corrida:** groq:openai/gpt-oss-120b × 401, ollama:qwen2.5:3b × 24 (0 llamadas nuevas, 425 desde la caché)
- **Commit:** aeb9d2c · **Fecha:** 2026-11-16T13:10:12-06:00 · **Duración:** 89.5 s · **Config:** `configs/2026-11-16_enc_qwen3.yaml`

> Protocolo, métricas y umbrales del equipo (`restmex-orbita`, commit 3ecbcfd). “5 mejores” = mayor afinidad con la consulta; “5 peores” = los más parecidos al Anti-HyDE (lo opuesto a lo pedido), no los de peor calidad.

## 1. Rúbrica

| Criterio | Valor | Umbral | Estado |
|---|---|---|---|
| Precision@5 (gold, 11 consultas con ≥5 relevantes) | 0.746 | 0.7 | ✅ |
| Hit@1 (gold) | 0.675 | 0.8 | ❌ |
| Contrato 5 + 5 | 1.000 | 1.0 | ✅ |
| Determinismo (con caché de LLM) | 1.000 | 1.0 | ✅ |
| Latencia p95 en caliente, LLM en caché (s, n = 40) | 0.057 | 3.0 | ✅ |
| Latencia p95 en frío, sin caché de LLM (s, n = 0) | — | 3.0 | — |

Latencia en proceso de `Recommender.recommend` (LLM + embeddings + búsqueda). En frío: p50 — s, p95 — s, máx. — s (n = 0, consultas que no estaban en la caché del LLM; la pausa entre llamadas para respetar la cuota no se cuenta). En caliente: p50 0.052 s, p95 0.057 s, máx. 0.059 s (n = 40). La web usa primero Gemini; aquí la latencia en frío es la de la cadena de evaluación.

## 2. Gold (seed heurístico del equipo, pendiente de adjudicación humana)

`eval/gold_queries.json` copiado sin cambios (SHA-256 `e565ef79cbc66d35…`; autor: asistente de IA (Claude) a petición del equipo; vio salidas del sistema del equipo: sí). Pipeline completo de la web (HyDE + Anti-HyDE + λ = 0.04) sobre 40 consultas.

| Métrica | Sistema | Popularidad | Azar |
|---|---|---|---|
| precision_at_5 | 0.415 | 0.120 | 0.095 |
| capped_recall_at_5 | 0.728 | 0.193 | 0.143 |
| hit_at_1 | 0.675 | 0.050 | 0.096 |
| hit_at_5 | 0.950 | 0.450 | 0.371 |
| mrr | 0.783 | 0.234 | 0.238 |
| ndcg_at_5 | 0.686 | 0.141 | 0.119 |
| bottom_error_rate | 0.200 | 0.300 | 0.364 |
| exclusion_violation_rate | 0.667 | 0.000 | 0.422 |

Top/Bottom de `recommend` idénticos al ranking evaluado: 1.000. Consultas resueltas con la plantilla sin LLM: 0. Las 3 consultas con exclusión miden `exclusion_violation_rate`; este sistema no interpreta exclusiones de forma explícita.

Por partición del banco de consultas del equipo (`test` no se usa para ajustar nada):

| Partición | Consultas | P@5 | Hit@1 | Hit@5 | MRR | nDCG@5 |
|---|---|---|---|---|---|---|
| calibration | 6 | 0.433 | 0.500 | 0.833 | 0.597 | 0.660 |
| dev | 19 | 0.453 | 0.789 | 0.947 | 0.862 | 0.754 |
| test | 15 | 0.360 | 0.600 | 1.000 | 0.758 | 0.611 |

## 3. Holdout enmascarado (sin consultas escritas a mano)

Cada reseña reservada de validación (no vista al construir los arquetipos) actúa como consulta con nombres de pueblo y estado enmascarados; el destino correcto es su pueblo.

600 reseñas de `val` (polaridad 4–5, 80–600 caracteres, 15 por pueblo, seed 2025). El enmascarado modificó el 53.3 % de las reseñas. El texto enmascarado queda en forma canónica (minúsculas, sin acentos ni puntuación), como en el equipo.

| Variante | n | Hit@1 | Hit@5 | MRR | Pueblo correcto en bottom 5 | Spearman(Hit@5 por pueblo, nº reseñas) |
|---|---|---|---|---|---|---|
| reseña como documento (sin LLM) | 600 | 0.318 | 0.590 | 0.453 | 0.010 | -0.190 |
| consulta cruda (search_query) | 600 | 0.328 | 0.623 | 0.465 | 0.012 | -0.168 |
| HyDE + Anti-HyDE (pipeline web) | 200 | 0.215 | 0.435 | 0.330 | 0.105 | 0.180 |
| Popularidad | 600 | 0.025 | 0.125 | 0.107 | 0.125 | — |
| Azar (esperado) | — | 0.025 | 0.125 | — | 0.125 | — |

HyDE se evalúa en una submuestra balanceada (`--hyde-holdout 200`; pendientes por cuota: 0; LLM: groq:openai/gpt-oss-120b × 200). En las variantes sin Anti-HyDE el bottom es el reverso del ranking. Mismas 200 reseñas en las tres variantes:

| Variante (misma submuestra) | Hit@1 | Hit@5 | MRR |
|---|---|---|---|
| reseña como documento (sin LLM) | 0.325 | 0.585 | 0.449 |
| consulta cruda (search_query) | 0.315 | 0.590 | 0.447 |
| HyDE + Anti-HyDE (pipeline web) | 0.215 | 0.435 | 0.330 |

### Fuga de nombres: el mismo holdout sin enmascarar

| Texto de la consulta | Variante | Hit@1 | Hit@5 | MRR |
|---|---|---|---|---|
| original (sin enmascarar) | reseña como documento (sin LLM) | 0.452 | 0.703 | 0.567 |
| original (sin enmascarar) | consulta cruda (search_query) | 0.450 | 0.712 | 0.571 |
| canónico sin enmascarar | reseña como documento (sin LLM) | 0.438 | 0.680 | 0.553 |
| canónico sin enmascarar | consulta cruda (search_query) | 0.447 | 0.705 | 0.567 |
| canónico enmascarado | reseña como documento (sin LLM) | 0.318 | 0.590 | 0.453 |
| canónico enmascarado | consulta cruda (search_query) | 0.328 | 0.623 | 0.465 |

Con la reseña como documento, la fuga de nombres inflaba Hit@5 en +9.0 puntos (canónico sin enmascarar − enmascarado); pasar el texto a forma canónica cuesta además +2.3 puntos (original − canónico), efecto del formato y no de la fuga.

HyDE sin enmascarar no se corrió (`--hyde-unmasked 0`): duplicaría las llamadas al LLM y el prompt ya prohíbe nombres propios en el HyDE.

## 4. Ablación

Cada fila cambia una sola pieza del sistema completo. Es descriptiva: los hiperparámetros se eligen en validación (`scripts/evaluate.py`), no con estas tablas.

### Gold (40 consultas)

| Configuración | P@5 | Hit@1 | Hit@5 | MRR | nDCG@5 | Error en bottom |
|---|---|---|---|---|---|---|
| completo (arquetipos, λ, HyDE, bottom Anti-HyDE) | 0.415 | 0.675 | 0.950 | 0.783 | 0.686 | 0.200 |
| 1 centroide por pueblo | 0.365 | 0.550 | 0.775 | 0.662 | 0.575 | 0.200 |
| λ = 0 | 0.420 | 0.650 | 0.975 | 0.767 | 0.681 | 0.175 |
| consulta cruda (search_query, sin HyDE) | 0.430 | 0.925 | 1.000 | 0.952 | 0.797 | 0.225 |
| bottom = reverso del ranking | 0.415 | 0.675 | 0.950 | 0.786 | 0.686 | 0.025 |

### Holdout, submuestra con HyDE (n = 200)

| Configuración | P@5 | Hit@1 | Hit@5 | MRR | nDCG@5 | Error en bottom |
|---|---|---|---|---|---|---|
| completo (arquetipos, λ, HyDE, bottom Anti-HyDE) | 0.087 | 0.215 | 0.435 | 0.330 | 0.325 | 0.105 |
| 1 centroide por pueblo | 0.086 | 0.175 | 0.430 | 0.300 | 0.302 | 0.135 |
| λ = 0 | 0.087 | 0.210 | 0.435 | 0.327 | 0.323 | 0.105 |
| consulta cruda (search_query, sin HyDE) | 0.118 | 0.315 | 0.590 | 0.447 | 0.459 | 0.055 |
| bottom = reverso del ranking | 0.087 | 0.215 | 0.435 | 0.334 | 0.325 | 0.035 |

### Holdout completo, sin LLM (n = 600)

| Consulta | Configuración | Hit@1 | Hit@5 | MRR | Error en bottom |
|---|---|---|---|---|---|
| reseña como documento (sin LLM) | arquetipos, λ elegido | 0.318 | 0.590 | 0.453 | 0.010 |
| reseña como documento (sin LLM) | arquetipos, λ = 0 | 0.318 | 0.587 | 0.452 | 0.010 |
| reseña como documento (sin LLM) | 1 centroide por pueblo, λ elegido | 0.212 | 0.463 | 0.344 | 0.027 |
| consulta cruda (search_query) | arquetipos, λ elegido | 0.328 | 0.623 | 0.465 | 0.012 |
| consulta cruda (search_query) | arquetipos, λ = 0 | 0.327 | 0.618 | 0.465 | 0.012 |
| consulta cruda (search_query) | 1 centroide por pueblo, λ elegido | 0.218 | 0.468 | 0.346 | 0.023 |

## 5. Sesgo por volumen de reseñas

Spearman(apariciones en top 5 del gold, nº de reseñas) = -0.038 (p = 0.815). Pueblo más expuesto: Tulum (17 apariciones). Pueblos que nunca aparecen en el top 5 del gold (2): Bernal, Metepec.

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

Dos corridas de las 40 consultas del gold con la caché del LLM: rankings idénticos en 1.000 de las consultas. Huella SHA-256 de los rankings: `316d3538c836453eefecd40fe22be6bec30c2b3bba8b926954bf42f5e533fa7e`.

Sin caché el sistema **no** es determinista: el LLM genera HyDE y Anti-HyDE con temperature 0.4, así que dos llamadas con la misma consulta producen textos distintos y el ranking puede cambiar. El determinismo que se reporta es el de la caché en disco (`artifacts/llm_cache`): la primera respuesta válida de cada prompt queda fija, y con ella el encoder y el ranking son reproducibles. Borrar la caché cambia la huella.

## 9. Consultas con fallos (Hit@5 = 0 o relevante en el bottom 5)

- **playa-snorkel** (relevante en bottom: Mazunte, Bacalar) “Quiero playas tranquilas, agua turquesa y snorkel” → top Tulum, Isla_Mujeres, Loreto, TodosSantos, Sayulita; bottom Mazunte, Bacalar, Cuatro_Cienegas, Huasca_de_Ocampo, Malinalco; esperados Isla_Mujeres, Tulum, Sayulita, Mazunte, TodosSantos, Loreto, Bacalar
- **piramides** (relevante en bottom: Malinalco) “Busco pirámides, arqueología e historia prehispánica” → top Teotihuacan, Cholula, Palenque, Tulum, Cuetzalan; bottom Malinalco, Coatepec, Tlaquepaque, Tequisquiapan, Atlixco; esperados Teotihuacan, Cholula, Palenque, Tulum, Malinalco, Izamal, Tepoztlan, Cuetzalan
- **gastronomia** (relevante en bottom: Tepoztlan, Patzcuaro) “Quiero gastronomía mexicana, mercados y restaurantes tradicionales” → top Atlixco, Valladolid, San_Cristobal_de_las_Casas, Malinalco, TodosSantos; bottom Tepoztlan, Bacalar, Patzcuaro, Tulum, Valle_de_Bravo; esperados Patzcuaro, Tlaquepaque, Izamal, Valladolid, San_Cristobal_de_las_Casas, Tepoztlan, Cholula, Atlixco, Ajijic
- **colonial** (relevante en bottom: Patzcuaro, Atlixco) “Deseo arquitectura colonial, iglesias y calles históricas” → top Taxco, Izamal, Cholula, Dolores_Hidalgo, Tlaquepaque; bottom Tequisquiapan, Patzcuaro, Real_de_Catorce, Atlixco, Coatepec; esperados Taxco, San_Cristobal_de_las_Casas, Patzcuaro, Izamal, Cholula, Dolores_Hidalgo, Valladolid, Tepotzotlan, Orizaba, Atlixco
- **desierto** (relevante en bottom: Real_de_Catorce) “Quiero desierto y paisajes naturales, sin playa” → top Cuatro_Cienegas, TodosSantos, Malinalco, Mazunte, Palenque; bottom Coatepec, Atlixco, Ajijic, Real_de_Catorce, Sayulita; esperados Cuatro_Cienegas, Real_de_Catorce, Parras
- **lago** (relevante en bottom: Patzcuaro) “Caminar junto al lago, malecón tranquilo y comida con vista al agua” → top Ajijic, Creel, Bacalar, Tulum, Sayulita; bottom Loreto, Atlixco, TodosSantos, Patzcuaro, Tlaquepaque; esperados Ajijic, Patzcuaro, Valle_de_Bravo, Bacalar, Creel
- **teleferico** (relevante en bottom: Zacatlan) “Subir en teleférico y visitar museos en la ciudad” → top Tlaquepaque, Orizaba, Tequisquiapan, Coatepec, Tepoztlan; bottom Malinalco, Creel, Atlixco, Zacatlan, TodosSantos; esperados Orizaba, Taxco, Tepotzotlan, Zacatlan
- **monolito** (Hit@5 = 0) “Escalar un monolito enorme y comer gorditas” → top Tepoztlan, Tapalpa, Creel, Chiapa_de_Corzo, Zacatlan; bottom TodosSantos, Atlixco, Bacalar, Malinalco, Coatepec; esperados Bernal
- **volcan** (Hit@5 = 0; relevante en bottom: Atlixco) “Clima templado, viveros de flores y vista al volcán” → top Zacatlan, Malinalco, Chiapa_de_Corzo, Xilitla, Cuatro_Cienegas; bottom Atlixco, TodosSantos, Real_de_Catorce, Ajijic, Coatepec; esperados Atlixco, Cholula, Tepoztlan

## Exportación para la evaluación humana ciega

`reports/exp/2026-11-16_enc_qwen3/rankings_query_bank.json`: ranking completo de 60 de 60 consultas de `eval/query_bank.json` (rechazadas por la validación: 0; pendientes por cuota: 0).
