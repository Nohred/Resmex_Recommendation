"""Índice de arquetipos y emparejamiento por similitud coseno (máximo por pueblo)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ArchetypeIndex:
    """Arquetipos en memoria. Los vectores están normalizados (L2).

    `town_of[a]` es el índice del pueblo (en `towns`) del arquetipo `a`.
    """

    vectors: np.ndarray            # (A, d)
    meta: list[dict[str, Any]]     # A elementos
    towns: list[str]               # T pueblos (orden alfabético)
    town_of: np.ndarray            # (A,) int
    n_reviews: np.ndarray          # (T,) reseñas de entrenamiento por pueblo
    n_max: int
    stats: dict[str, Any]

    @classmethod
    def from_parts(cls, vectors: np.ndarray, meta: list[dict[str, Any]], stats: dict[str, Any]) -> "ArchetypeIndex":
        towns = sorted(stats["towns"].keys())
        pos = {t: i for i, t in enumerate(towns)}
        town_of = np.asarray([pos[m["town"]] for m in meta], dtype=np.int64)
        n_reviews = np.asarray([stats["towns"][t]["n_reviews"] for t in towns], dtype=np.float64)
        if len(meta) != len(vectors):
            raise ValueError("Los metadatos no coinciden con los vectores de arquetipos")
        return cls(
            vectors=np.asarray(vectors, dtype=np.float32),
            meta=meta,
            towns=towns,
            town_of=town_of,
            n_reviews=n_reviews,
            n_max=int(stats.get("n_max", n_reviews.max())),
            stats=stats,
        )

    @classmethod
    def load(cls, artifacts_dir: Path) -> "ArchetypeIndex":
        """Carga `archetypes.npy`, `archetypes_meta.json` y `town_stats.json`."""
        vectors = np.load(artifacts_dir / "archetypes.npy")
        meta = json.loads((artifacts_dir / "archetypes_meta.json").read_text(encoding="utf-8"))
        stats = json.loads((artifacts_dir / "town_stats.json").read_text(encoding="utf-8"))
        return cls.from_parts(vectors, meta, stats)

    @property
    def n_towns(self) -> int:
        return len(self.towns)

    def town_scores(self, query_vecs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Similitud máxima por pueblo.

        `query_vecs`: (d,) o (Q, d), normalizados. Devuelve
        (scores (Q, T), arquetipo ganador (Q, T) como índice global).
        """
        q = np.atleast_2d(np.asarray(query_vecs, dtype=np.float32))
        return max_per_town(q @ self.vectors.T, self.town_of, self.n_towns)


def max_per_town(sims: np.ndarray, town_of: np.ndarray, n_towns: int) -> tuple[np.ndarray, np.ndarray]:
    """Reduce (Q, A) similitudes a (Q, T) con el máximo por pueblo y su argmax."""
    n_q = sims.shape[0]
    scores = np.full((n_q, n_towns), -np.inf, dtype=np.float32)
    best = np.full((n_q, n_towns), -1, dtype=np.int64)
    for t in range(n_towns):
        cols = np.flatnonzero(town_of == t)
        if len(cols) == 0:
            continue
        sub = sims[:, cols]
        arg = sub.argmax(axis=1)
        scores[:, t] = sub[np.arange(n_q), arg]
        best[:, t] = cols[arg]
    return scores, best
