"""Agregación de similitudes arquetipo → pueblo y políticas para elegir k por pueblo.

`retrieval.max_per_town` (máximo) es lo que usa la web; aquí viven las alternativas que se
comparan en el barrido de arquetipos (`scripts/sweep_archetypes.py`):

- `max`      máximo sobre los arquetipos del pueblo (actual).
- `top2`     media de las 2 similitudes más altas (un pueblo con 1 arquetipo usa esa).
- `lse`      máximo suave: τ · log Σ exp(sim / τ). τ → 0 tiende al máximo; con τ grande pesa
             a todos los arquetipos y suma τ · log(k) a los pueblos con más arquetipos.
- `max_big`  máximo ignorando los arquetipos pequeños (`keep` = arquetipos que cuentan).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

MODES = ("max", "top2", "lse", "max_big")


def small_cluster_mask(sizes: np.ndarray, shares: np.ndarray, town_of: np.ndarray,
                       min_share: float = 0.02, min_size: int = 30) -> np.ndarray:
    """Arquetipos que cuentan para `max_big`: share ≥ `min_share` y tamaño ≥ `min_size`.

    Si un pueblo se quedara sin arquetipos, conserva todos los suyos.
    """
    keep = (np.asarray(shares) >= min_share) & (np.asarray(sizes) >= min_size)
    for t in np.unique(town_of):
        cols = town_of == t
        if not keep[cols].any():
            keep[cols] = True
    return keep


def aggregate_per_town(
    sims: np.ndarray,
    town_of: np.ndarray,
    n_towns: int,
    mode: str = "max",
    tau: float = 0.05,
    keep: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Reduce similitudes (Q, A) a puntajes por pueblo (Q, T) con la agregación `mode`."""
    if mode not in MODES:
        raise ValueError(f"Agregación desconocida: {mode} (opciones: {MODES})")
    if mode == "max_big" and keep is None:
        raise ValueError("`max_big` necesita la máscara `keep`")
    sims = np.asarray(sims, dtype=np.float32)
    out = np.full((sims.shape[0], n_towns), -np.inf, dtype=np.float64)
    for t in range(n_towns):
        cols = np.flatnonzero((town_of == t) & keep) if mode == "max_big" else np.flatnonzero(town_of == t)
        if len(cols) == 0:
            continue
        sub = sims[:, cols].astype(np.float64)
        if mode in ("max", "max_big") or sub.shape[1] == 1:
            out[:, t] = sub.max(axis=1)
        elif mode == "top2":
            out[:, t] = np.partition(sub, -2, axis=1)[:, -2:].mean(axis=1)
        else:  # lse, estable: se resta el máximo antes de exponenciar
            top = sub.max(axis=1, keepdims=True)
            out[:, t] = top[:, 0] + tau * np.log(np.exp((sub - top) / tau).sum(axis=1))
    return out


def k_log(n_reviews: int, a: float, k_min: int = 3, k_max: int = 30) -> int:
    """k adaptativo: clip(round(a · ln(n_reseñas)), k_min, k_max)."""
    return int(np.clip(round(a * np.log(n_reviews)), k_min, k_max))
