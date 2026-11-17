"""Bootstrap pareado por consulta para comparar dos sistemas sobre las mismas consultas.

Definición (la misma que se documenta en el reporte de comparación):

1. Las dos variantes se evalúan sobre exactamente las mismas consultas (mismo `row_id`).
2. En cada uno de `n_boot` remuestreos se sacan `n` consultas con reemplazo; el mismo
   remuestreo se aplica a los dos sistemas (por eso es pareado).
3. En cada remuestreo se calcula la métrica de cada sistema y su diferencia (candidato − base).
4. El IC95 son los percentiles 2.5 y 97.5 de las diferencias. Si el intervalo no contiene
   el 0, la diferencia no se explica por qué consultas tocaron en la muestra.

El promedio macro pesa igual a cada pueblo: media por pueblo y luego media entre pueblos
(los pueblos que no salen en un remuestreo no cuentan en ese remuestreo).
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def macro_mean(values: np.ndarray, groups: np.ndarray, n_groups: int) -> float:
    """Media por grupo y luego media entre los grupos presentes."""
    sums = np.bincount(groups, weights=values, minlength=n_groups)
    counts = np.bincount(groups, minlength=n_groups)
    present = counts > 0
    return float(np.mean(sums[present] / counts[present]))


def paired_bootstrap(
    candidate: Sequence[float],
    baseline: Sequence[float],
    groups: Optional[Sequence[int]] = None,
    n_boot: int = 1000,
    seed: int = 2025,
) -> dict:
    """Diferencia (candidato − base) de la media por consulta, con IC95 por percentiles.

    `groups` (un entero por consulta, p. ej. el pueblo) cambia la métrica a promedio macro.
    """
    a, b = np.asarray(candidate, dtype=np.float64), np.asarray(baseline, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1 or not len(a):
        raise ValueError("Las dos variantes deben tener las mismas consultas (vectores 1-D no vacíos)")
    n = len(a)
    if groups is None:
        stat = lambda v, idx: float(v[idx].mean())  # noqa: E731
    else:
        g = np.asarray(groups, dtype=np.int64)
        n_groups = int(g.max()) + 1
        stat = lambda v, idx: macro_mean(v[idx], g[idx], n_groups)  # noqa: E731
    everything = np.arange(n)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[i] = stat(a, idx) - stat(b, idx)
    low, high = np.percentile(diffs, [2.5, 97.5])
    return {
        "candidate": round(stat(a, everything), 4),
        "baseline": round(stat(b, everything), 4),
        "diff": round(stat(a, everything) - stat(b, everything), 4),
        "ci95": [round(float(low), 4), round(float(high), 4)],
        "significant": bool(low > 0 or high < 0),
        "n": n,
        "n_boot": n_boot,
        "seed": seed,
    }
