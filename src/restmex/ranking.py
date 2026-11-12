"""Penalización por popularidad (hubness) y ranking final Top / Bottom."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .retrieval import ArchetypeIndex


def popularity_penalty(n_reviews: np.ndarray, n_max: float, lam: float) -> np.ndarray:
    """`λ · log(1 + n_pueblo) / log(1 + n_max)` (vale λ para el pueblo más reseñado)."""
    n = np.asarray(n_reviews, dtype=np.float64)
    return lam * np.log1p(n) / np.log1p(float(n_max))


def adjusted_scores(sim: np.ndarray, n_reviews: np.ndarray, n_max: float, lam: float) -> np.ndarray:
    """`score_ajustado = sim_hyde − penalización`. Acepta (T,) o (Q, T)."""
    return np.asarray(sim, dtype=np.float64) - popularity_penalty(n_reviews, n_max, lam)


@dataclass
class TownResult:
    """Un pueblo dentro del Top o del Bottom, con su explicación."""

    rank: int
    town: str
    town_display: str
    region: str
    n_reviews: int
    sim_hyde: float
    sim_anti: float
    penalty: float
    score: float                    # score_ajustado
    archetype_id: int               # arquetipo que produjo el máximo mostrado
    archetype_label: str
    keywords: list[str]
    review: str                     # reseña representativa
    review_polarity: float
    mean_polarity: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _result(index: ArchetypeIndex, t: int, rank: int, sim_h: float, sim_a: float,
            penalty: float, arch: int) -> TownResult:
    town = index.towns[t]
    meta = index.meta[arch]
    rep = meta["representatives"][0] if meta.get("representatives") else {"text": "", "polarity": 0.0}
    return TownResult(
        rank=rank,
        town=town,
        town_display=meta.get("town_display", town),
        region=meta.get("region", ""),
        n_reviews=int(index.n_reviews[t]),
        sim_hyde=float(sim_h),
        sim_anti=float(sim_a),
        penalty=float(penalty),
        score=float(sim_h - penalty),
        archetype_id=int(arch),
        archetype_label=meta.get("label", ""),
        keywords=list(meta.get("keywords", [])),
        review=rep["text"],
        review_polarity=float(rep["polarity"]),
        mean_polarity=float(meta.get("mean_polarity", 0.0)),
    )


def rank_towns(
    index: ArchetypeIndex,
    hyde_vec: np.ndarray,
    anti_vec: np.ndarray,
    lam: float = 0.05,
    top_k: int = 5,
    bottom_k: int = 5,
) -> tuple[list[TownResult], list[TownResult]]:
    """Ranking final.

    - Top: los `top_k` pueblos con mayor `score_ajustado` (HyDE − penalización).
    - Bottom: los `bottom_k` con mayor similitud al Anti-HyDE, excluyendo el Top.
    El arquetipo explicativo es el ganador frente a HyDE (Top) o Anti-HyDE (Bottom).
    """
    sims, best = index.town_scores(np.vstack([hyde_vec, anti_vec]))
    sim_h, sim_a = sims[0], sims[1]
    penalty = popularity_penalty(index.n_reviews, index.n_max, lam)
    adjusted = sim_h - penalty

    top_idx = [int(t) for t in np.argsort(-adjusted, kind="stable")[:top_k]]
    excluded = set(top_idx)
    bottom_idx = [int(t) for t in np.argsort(-sim_a, kind="stable") if int(t) not in excluded][:bottom_k]

    top = [
        _result(index, t, r + 1, sim_h[t], sim_a[t], penalty[t], int(best[0, t]))
        for r, t in enumerate(top_idx)
    ]
    bottom = [
        _result(index, t, r + 1, sim_h[t], sim_a[t], penalty[t], int(best[1, t]))
        for r, t in enumerate(bottom_idx)
    ]
    return top, bottom
