Media ± desviación entre 4 semillas del K-Means (42, 7, 123, 2025). Δ: recíproco del rango de cada consulta promediado sobre las semillas y bootstrap pareado (IC95; * = sin el 0).

| Variante | Idea | Reseñas que quedan | Mín. por pueblo | λ por semilla | MRR macro calib. | Δ vs ref. [IC95] | Δ por semilla | nDCG@5 gold (25) | Δ gold [IC95] | Spearman |
|---|---|---|---|---|---|---|---|---|---|---|
| muestra_1500 | referencia de la muestra | 52,268 de 52,268 (100.0 %) | 653 (Tapalpa) | 0.02, 0.015, 0.015, 0.015 | 0.5248 ± 0.0110 | — | — | 0.7059 ± 0.0386 | — | +0.034 ± 0.048 |
| muestra_1500_sin_nombres | 6 · nombres enmascarados (muestra) | 52,268 de 52,268 (100.0 %) | 653 (Tapalpa) | 0.015, 0.015, 0.02, 0.02 | 0.5333 ± 0.0106 | +0.0085 [-0.011, +0.028] | -0.003 +0.023 +0.001 +0.013 | 0.7364 ± 0.0062 | +0.0305 [-0.007, +0.064] | -0.002 ± 0.060 |
| min_contenido_8 | 1 · mínimo de palabras de contenido | 182,796 de 187,080 (97.7 %) | 641 (Tapalpa) | 0.03, 0.03, 0.03, 0.03 | 0.5392 ± 0.0125 | +0.0065 [-0.011, +0.023] | -0.013 +0.015 -0.011 +0.035 | 0.7024 ± 0.0140 | -0.0068 [-0.050, +0.026] | -0.115 ± 0.084 |
| combinacion | 6 · nombres enmascarados (muestra) + 1 · mínimo de palabras de contenido | 50,908 de 52,268 (97.4 %) | 641 (Tapalpa) | 0.015, 0.015, 0.015, 0.015 | 0.5415 ± 0.0113 | +0.0168 [-0.002, +0.037] | -0.012 +0.036 +0.031 +0.011 | 0.7308 ± 0.0072 | +0.0249 [-0.017, +0.063] | +0.036 ± 0.069 |
