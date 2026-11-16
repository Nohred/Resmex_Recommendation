"""Índice de arquetipos y emparejamiento por similitud coseno (máximo por pueblo)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .aggregation import aggregate_per_town


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
    aggregation: str = "max"       # "max" | "lse" (`ranking.aggregation`)
    lse_tau: float = 0.01

    def configure(self, ranking: Optional[Any]) -> "ArchetypeIndex":
        """Aplica la agregación de `cfg.ranking` (sin argumento se queda el máximo)."""
        if ranking is not None:
            if ranking.aggregation not in ("max", "lse"):
                raise ValueError(f"ranking.aggregation desconocida: {ranking.aggregation} (opciones: max, lse)")
            self.aggregation, self.lse_tau = ranking.aggregation, float(ranking.lse_tau)
        return self

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
    def load(cls, artifacts_dir: Path, ranking: Optional[Any] = None) -> "ArchetypeIndex":
        """Carga `archetypes.npy`, `archetypes_meta.json` y `town_stats.json`."""
        vectors = np.load(artifacts_dir / "archetypes.npy")
        meta = json.loads((artifacts_dir / "archetypes_meta.json").read_text(encoding="utf-8"))
        stats = json.loads((artifacts_dir / "town_stats.json").read_text(encoding="utf-8"))
        return cls.from_parts(vectors, meta, stats).configure(ranking)

    @property
    def n_towns(self) -> int:
        return len(self.towns)

    def town_scores(self, query_vecs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Similitud por pueblo: máximo sobre sus arquetipos o máximo suave (`lse`).

        `query_vecs`: (d,) o (Q, d), normalizados. Devuelve
        (scores (Q, T), arquetipo más parecido (Q, T) como índice global).
        """
        q = np.atleast_2d(np.asarray(query_vecs, dtype=np.float32))
        sims = q @ self.vectors.T
        scores, best = max_per_town(sims, self.town_of, self.n_towns)
        if self.aggregation == "lse":
            scores = aggregate_per_town(sims, self.town_of, self.n_towns, "lse", tau=self.lse_tau).astype(np.float32)
        return scores, best


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
