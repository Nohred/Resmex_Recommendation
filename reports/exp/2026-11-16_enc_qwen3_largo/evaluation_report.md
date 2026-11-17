# Reporte de evaluación automática

- **Encoder:** ollama:qwen3-embedding:0.6b@ctx4096 · **k:** 10 arquetipos por pueblo (400 en 40 pueblos) · **λ:** 0.04
- **Cadena de LLM (evaluación):** groq/openai/gpt-oss-120b → ollama/qwen2.5:3b → plantilla · temperature 0.4
- **LLM usado en esta corrida:** groq:openai/gpt-oss-120b × 401, ollama:qwen2.5:3b × 24 (0 llamadas nuevas, 425 desde la caché)
- **Commit:** b83a790 · **Fecha:** 2026-11-16T15:01:42-06:00 · **Duración:** 89.8 s · **Config:** `configs/2026-11-16_enc_qwen3_largo.yaml`

> Protocolo, métricas y umbrales del equipo (`restmex-orbita`, commit 3ecbcfd). “5 mejores” = mayor afinidad con la consulta; “5 peores” = los más parecidos al Anti-HyDE (lo opuesto a lo pedido), no los de peor calidad.

## 1. Rúbrica

| Criterio | Valor | Umbral | Estado |
|---|---|---|---|
| Precision@5 (gold, 11 consultas con ≥5 relevantes) | 0.709 | 0.7 | ✅ |
| Hit@1 (gold) | 0.650 | 0.8 | ❌ |
| Contrato 5 + 5 | 1.000 | 1.0 | ✅ |
| Determinismo (con caché de LLM) | 1.000 | 1.0 | ✅ |
| Latencia p95 en caliente, LLM en caché (s, n = 40) | 0.056 | 3.0 | ✅ |
| Latencia p95 en frío, sin caché de LLM (s, n = 0) | — | 3.0 | — |

Latencia en proceso de `Recommender.recommend` (LLM + embeddings + búsqueda). En frío: p50 — s, p95 — s, máx. — s (n = 0, consultas que no estaban en la caché del LLM; la pausa entre llamadas para respetar la cuota no se cuenta). En caliente: p50 0.053 s, p95 0.056 s, máx. 0.058 s (n = 40). La web usa primero Gemini; aquí la latencia en frío es la de la cadena de evaluación.

## 2. Gold (seed heurístico del equipo, pendiente de adjudicación humana)

`eval/gold_queries.json` copiado sin cambios (SHA-256 `e565ef79cbc66d35…`; autor: asistente de IA (Claude) a petición del equipo; vio salidas del sistema del equipo: sí). Pipeline completo de la web (HyDE + Anti-HyDE + λ = 0.04) sobre 40 consultas.

| Métrica | Sistema | Popularidad | Azar |
|---|---|---|---|
| precision_at_5 | 0.390 | 0.120 | 0.095 |
| capped_recall_at_5 | 0.680 | 0.193 | 0.143 |
| hit_at_1 | 0.650 | 0.050 | 0.096 |
| hit_at_5 | 0.925 | 0.450 | 0.371 |
| mrr | 0.771 | 0.234 | 0.238 |
| ndcg_at_5 | 0.656 | 0.141 | 0.119 |
| bottom_error_rate | 0.225 | 0.300 | 0.364 |
| exclusion_violation_rate | 1.000 | 0.000 | 0.422 |

Top/Bottom de `recommend` idénticos al ranking evaluado: 1.000. Consultas resueltas con la plantilla sin LLM: 0. Las 3 consultas con exclusión miden `exclusion_violation_rate`; este sistema no interpreta exclusiones de forma explícita.

Por partición del banco de consultas del equipo (`test` no se usa para ajustar nada):

| Partición | Consultas | P@5 | Hit@1 | Hit@5 | MRR | nDCG@5 |
|---|---|---|---|---|---|---|
| calibration | 6 | 0.367 | 0.500 | 0.667 | 0.604 | 0.588 |
| dev | 19 | 0.421 | 0.737 | 0.947 | 0.827 | 0.710 |
| test | 15 | 0.360 | 0.600 | 1.000 | 0.767 | 0.614 |

## 3. Holdout enmascarado (sin consultas escritas a mano)

Cada reseña reservada de validación (no vista al construir los arquetipos) actúa como consulta con nombres de pueblo y estado enmascarados; el destino correcto es su pueblo.

600 reseñas de `val` (polaridad 4–5, 80–600 caracteres, 15 por pueblo, seed 2025). El enmascarado modificó el 53.3 % de las reseñas. El texto enmascarado queda en forma canónica (minúsculas, sin acentos ni puntuación), como en el equipo.

| Variante | n | Hit@1 | Hit@5 | MRR | Pueblo correcto en bottom 5 | Spearman(Hit@5 por pueblo, nº reseñas) |
|---|---|---|---|---|---|---|
| reseña como documento (sin LLM) | 600 | 0.328 | 0.600 | 0.459 | 0.012 | -0.202 |
| consulta cruda (search_query) | 600 | 0.327 | 0.625 | 0.466 | 0.013 | -0.147 |
| HyDE + Anti-HyDE (pipeline web) | 200 | 0.205 | 0.425 | 0.324 | 0.110 | 0.115 |
| Popularidad | 600 | 0.025 | 0.125 | 0.107 | 0.125 | — |
| Azar (esperado) | — | 0.025 | 0.125 | — | 0.125 | — |

HyDE se evalúa en una submuestra balanceada (`--hyde-holdout 200`; pendientes por cuota: 0; LLM: groq:openai/gpt-oss-120b × 200). En las variantes sin Anti-HyDE el bottom es el reverso del ranking. Mismas 200 reseñas en las tres variantes:

| Variante (misma submuestra) | Hit@1 | Hit@5 | MRR |
|---|---|---|---|
| reseña como documento (sin LLM) | 0.330 | 0.570 | 0.447 |
| consulta cruda (search_query) | 0.300 | 0.570 | 0.434 |
| HyDE + Anti-HyDE (pipeline web) | 0.205 | 0.425 | 0.324 |

### Fuga de nombres: el mismo holdout sin enmascarar

| Texto de la consulta | Variante | Hit@1 | Hit@5 | MRR |
|---|---|---|---|---|
| original (sin enmascarar) | reseña como documento (sin LLM) | 0.458 | 0.692 | 0.570 |
| original (sin enmascarar) | consulta cruda (search_query) | 0.457 | 0.722 | 0.578 |
| canónico sin enmascarar | reseña como documento (sin LLM) | 0.442 | 0.690 | 0.557 |
| canónico sin enmascarar | consulta cruda (search_query) | 0.453 | 0.708 | 0.574 |
| canónico enmascarado | reseña como documento (sin LLM) | 0.328 | 0.600 | 0.459 |
| canónico enmascarado | consulta cruda (search_query) | 0.327 | 0.625 | 0.466 |

Con la reseña como documento, la fuga de nombres inflaba Hit@5 en +9.0 puntos (canónico sin enmascarar − enmascarado); pasar el texto a forma canónica cuesta además +0.2 puntos (original − canónico), efecto del formato y no de la fuga.

HyDE sin enmascarar no se corrió (`--hyde-unmasked 0`): duplicaría las llamadas al LLM y el prompt ya prohíbe nombres propios en el HyDE.

## 4. Ablación

Cada fila cambia una sola pieza del sistema completo. Es descriptiva: los hiperparámetros se eligen en validación (`scripts/evaluate.py`), no con estas tablas.

### Gold (40 consultas)

| Configuración | P@5 | Hit@1 | Hit@5 | MRR | nDCG@5 | Error en bottom |
|---|---|---|---|---|---|---|
| completo (arquetipos, λ, HyDE, bottom Anti-HyDE) | 0.390 | 0.650 | 0.925 | 0.771 | 0.656 | 0.225 |
| 1 centroide por pueblo | 0.365 | 0.550 | 0.775 | 0.662 | 0.575 | 0.200 |
| λ = 0 | 0.390 | 0.650 | 0.925 | 0.771 | 0.658 | 0.225 |
| consulta cruda (search_query, sin HyDE) | 0.440 | 0.925 | 1.000 | 0.950 | 0.801 | 0.200 |
| bottom = reverso del ranking | 0.390 | 0.650 | 0.925 | 0.773 | 0.656 | 0.050 |

### Holdout, submuestra con HyDE (n = 200)

| Configuración | P@5 | Hit@1 | Hit@5 | MRR | nDCG@5 | Error en bottom |
|---|---|---|---|---|---|---|
| completo (arquetipos, λ, HyDE, bottom Anti-HyDE) | 0.085 | 0.205 | 0.425 | 0.324 | 0.317 | 0.110 |
| 1 centroide por pueblo | 0.086 | 0.175 | 0.430 | 0.300 | 0.302 | 0.135 |
| λ = 0 | 0.083 | 0.200 | 0.415 | 0.320 | 0.309 | 0.110 |
| consulta cruda (search_query, sin HyDE) | 0.114 | 0.300 | 0.570 | 0.433 | 0.443 | 0.065 |
| bottom = reverso del ranking | 0.085 | 0.205 | 0.425 | 0.327 | 0.317 | 0.035 |

### Holdout completo, sin LLM (n = 600)

| Consulta | Configuración | Hit@1 | Hit@5 | MRR | Error en bottom |
|---|---|---|---|---|---|
| reseña como documento (sin LLM) | arquetipos, λ elegido | 0.328 | 0.600 | 0.459 | 0.012 |
| reseña como documento (sin LLM) | arquetipos, λ = 0 | 0.323 | 0.600 | 0.456 | 0.012 |
| reseña como documento (sin LLM) | 1 centroide por pueblo, λ elegido | 0.213 | 0.463 | 0.345 | 0.027 |
| consulta cruda (search_query) | arquetipos, λ elegido | 0.327 | 0.625 | 0.466 | 0.013 |
| consulta cruda (search_query) | arquetipos, λ = 0 | 0.322 | 0.625 | 0.464 | 0.013 |
| consulta cruda (search_query) | 1 centroide por pueblo, λ elegido | 0.218 | 0.468 | 0.345 | 0.023 |

## 5. Sesgo por volumen de reseñas

Spearman(apariciones en top 5 del gold, nº de reseñas) = 0.037 (p = 0.820). Pueblo más expuesto: Chiapa_de_Corzo (16 apariciones). Pueblos que nunca aparecen en el top 5 del gold (2): Bernal, Metepec.

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

Dos corridas de las 40 consultas del gold con la caché del LLM: rankings idénticos en 1.000 de las consultas. Huella SHA-256 de los rankings: `cbbebb8ac81b87a9047146f32e77a82f1280530c37bf68d14352694000f79de1`.

Sin caché el sistema **no** es determinista: el LLM genera HyDE y Anti-HyDE con temperature 0.4, así que dos llamadas con la misma consulta producen textos distintos y el ranking puede cambiar. El determinismo que se reporta es el de la caché en disco (`artifacts/llm_cache`): la primera respuesta válida de cada prompt queda fija, y con ella el encoder y el ranking son reproducibles. Borrar la caché cambia la huella.

## 9. Consultas con fallos (Hit@5 = 0 o relevante en el bottom 5)

- **playa-snorkel** (relevante en bottom: Bacalar) “Quiero playas tranquilas, agua turquesa y snorkel” → top Tulum, Isla_Mujeres, Loreto, Sayulita, TodosSantos; bottom Bacalar, Cuatro_Cienegas, Malinalco, Huasca_de_Ocampo, Zacatlan; esperados Isla_Mujeres, Tulum, Sayulita, Mazunte, TodosSantos, Loreto, Bacalar
- **piramides** (relevante en bottom: Malinalco) “Busco pirámides, arqueología e historia prehispánica” → top Teotihuacan, Palenque, Cholula, Tulum, Cuetzalan; bottom Malinalco, Coatepec, Tlaquepaque, Tequisquiapan, Atlixco; esperados Teotihuacan, Cholula, Palenque, Tulum, Malinalco, Izamal, Tepoztlan, Cuetzalan
- **gastronomia** (relevante en bottom: Patzcuaro, Tepoztlan) “Quiero gastronomía mexicana, mercados y restaurantes tradicionales” → top Atlixco, Valladolid, San_Cristobal_de_las_Casas, TodosSantos, Malinalco; bottom Tulum, Bacalar, Patzcuaro, Tepoztlan, Mazunte; esperados Patzcuaro, Tlaquepaque, Izamal, Valladolid, San_Cristobal_de_las_Casas, Tepoztlan, Cholula, Atlixco, Ajijic
- **colonial** (relevante en bottom: Patzcuaro, Atlixco) “Deseo arquitectura colonial, iglesias y calles históricas” → top Dolores_Hidalgo, Teotihuacan, Taxco, Izamal, Cholula; bottom Tlaquepaque, Tequisquiapan, Patzcuaro, Real_de_Catorce, Atlixco; esperados Taxco, San_Cristobal_de_las_Casas, Patzcuaro, Izamal, Cholula, Dolores_Hidalgo, Valladolid, Tepotzotlan, Orizaba, Atlixco
- **desierto** (relevante en bottom: Real_de_Catorce) “Quiero desierto y paisajes naturales, sin playa” → top Cuatro_Cienegas, TodosSantos, Malinalco, Palenque, Chiapa_de_Corzo; bottom Coatepec, Atlixco, Ajijic, Real_de_Catorce, Tequisquiapan; esperados Cuatro_Cienegas, Real_de_Catorce, Parras
- **cafe** (relevante en bottom: Cuetzalan) “Pueblo cafetalero con niebla, café de altura y clima fresco” → top Coatepec, Malinalco, Palenque, Xilitla, Atlixco; bottom Tequisquiapan, Patzcuaro, Metepec, Ajijic, Cuetzalan; esperados Coatepec, Cuetzalan, Xilitla, Orizaba, San_Cristobal_de_las_Casas, Zacatlan
- **lago** (relevante en bottom: Patzcuaro) “Caminar junto al lago, malecón tranquilo y comida con vista al agua” → top Ajijic, Creel, Bacalar, Tulum, Loreto; bottom Atlixco, Patzcuaro, TodosSantos, Tlaquepaque, Sayulita; esperados Ajijic, Patzcuaro, Valle_de_Bravo, Bacalar, Creel
- **teleferico** (relevante en bottom: Zacatlan) “Subir en teleférico y visitar museos en la ciudad” → top Tlaquepaque, Orizaba, Tequisquiapan, Coatepec, Izamal; bottom Malinalco, Zacatlan, Creel, Atlixco, Tapalpa; esperados Orizaba, Taxco, Tepotzotlan, Zacatlan
- **monolito** (Hit@5 = 0) “Escalar un monolito enorme y comer gorditas” → top Tepoztlan, Tapalpa, Creel, Chiapa_de_Corzo, Cuetzalan; bottom Atlixco, TodosSantos, Bacalar, Malinalco, Coatepec; esperados Bernal
- **independencia** (Hit@5 = 0) “Historia de la independencia de México y nieves de sabores raros” → top Tlaquepaque, Valladolid, Tequisquiapan, Malinalco, Zacatlan; bottom Tulum, Sayulita, Mazunte, Isla_Mujeres, Atlixco; esperados Dolores_Hidalgo
- **volcan** (Hit@5 = 0; relevante en bottom: Atlixco) “Clima templado, viveros de flores y vista al volcán” → top Zacatlan, Malinalco, Chiapa_de_Corzo, Xilitla, Cuatro_Cienegas; bottom Atlixco, Real_de_Catorce, TodosSantos, Ajijic, Coatepec; esperados Atlixco, Cholula, Tepoztlan

## Exportación para la evaluación humana ciega

`reports/exp/2026-11-16_enc_qwen3_largo/rankings_query_bank.json`: ranking completo de 60 de 60 consultas de `eval/query_bank.json` (rechazadas por la validación: 0; pendientes por cuota: 0).
