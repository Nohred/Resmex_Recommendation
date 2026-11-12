"""Métricas de evaluación: Hit@k, MRR, calibración de λ y sesgo de popularidad."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy.stats import spearmanr

from .ranking import adjusted_scores


def true_ranks(scores: np.ndarray, true_town: np.ndarray) -> np.ndarray:
    """Posición (1 = mejor) del pueblo real de cada consulta. `scores`: (Q, T)."""
    own = scores[np.arange(len(scores)), true_town][:, None]
    return 1 + (scores > own).sum(axis=1)


def retrieval_metrics(ranks: np.ndarray, true_town: np.ndarray, n_towns: int) -> dict[str, float]:
    """Hit@1, Hit@5 y MRR: micro (por consulta) y macro (promedio por pueblo).

    El promedio macro pesa igual a todos los pueblos; el micro está dominado por
    los pueblos con más reseñas (Tulum, Isla Mujeres).
    """
    per_query = {"hit@1": ranks <= 1, "hit@5": ranks <= 5, "mrr": 1.0 / ranks}
    out: dict[str, float] = {"n": int(len(ranks))}
    for name, values in per_query.items():
        out[name] = float(np.mean(values))
        by_town = [values[true_town == t].mean() for t in range(n_towns) if np.any(true_town == t)]
        out[f"macro_{name}"] = float(np.mean(by_town))
    return out


def top_k_frequency(scores: np.ndarray, n_towns: int, k: int = 5) -> np.ndarray:
    """Cuántas veces aparece cada pueblo en el Top-k."""
    top = np.argpartition(-scores, kth=min(k, scores.shape[1] - 1) - 1, axis=1)[:, :k]
    return np.bincount(top.ravel(), minlength=n_towns)


def popularity_bias(scores: np.ndarray, n_reviews: np.ndarray, k: int = 5) -> dict[str, Any]:
    """Spearman entre la frecuencia en el Top-k y el número de reseñas del pueblo."""
    freq = top_k_frequency(scores, len(n_reviews), k)
    rho, p = spearmanr(freq, n_reviews)
    return {"spearman": float(rho), "p_value": float(p), "frequency": freq.tolist()}


def balanced_sample(true_town: np.ndarray, n_towns: int, seed: int = 42) -> np.ndarray:
    """Índices con el mismo número de consultas por pueblo (el mínimo disponible).

    Con consultas balanceadas, un sistema sin sesgo mostraría a todos los pueblos
    con la misma frecuencia; así la correlación mide el sesgo del sistema y no
    la distribución de las consultas.
    """
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(true_town == t) for t in range(n_towns)]
    m = min(len(g) for g in groups if len(g))
    return np.sort(np.concatenate([rng.choice(g, size=m, replace=False) for g in groups if len(g)]))


def lambda_sweep(
    sims: np.ndarray,
    true_town: np.ndarray,
    n_reviews: np.ndarray,
    n_max: float,
    lambdas: Sequence[float],
    balanced_idx: np.ndarray,
) -> list[dict[str, Any]]:
    """Métricas y sesgo de popularidad para cada λ."""
    rows = []
    n_towns = sims.shape[1]
    for lam in lambdas:
        adj = adjusted_scores(sims, n_reviews, n_max, lam)
        row: dict[str, Any] = {"lambda": float(lam)}
        row.update(retrieval_metrics(true_ranks(adj, true_town), true_town, n_towns))
        bias = popularity_bias(adj[balanced_idx], n_reviews)
        row["spearman_balanced"] = bias["spearman"]
        row["spearman_all"] = popularity_bias(adj, n_reviews)["spearman"]
        row["top5_frequency_balanced"] = bias["frequency"]
        rows.append(row)
    return rows


def choose_lambda(rows: list[dict[str, Any]], tolerance: float = 0.01) -> float:
    """Criterio de calibración.

    Entre los λ cuyo MRR macro queda a menos de `tolerance` (relativo) del mejor,
    elige el de menor |Spearman| (menos sesgo de popularidad con consultas
    balanceadas). Empates → el λ más pequeño.
    """
    best = max(r["macro_mrr"] for r in rows)
    ok = [r for r in rows if r["macro_mrr"] >= best * (1 - tolerance)]
    return float(min(ok, key=lambda r: (abs(r["spearman_balanced"]), r["lambda"]))["lambda"])
