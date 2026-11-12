"""Ranking, penalización por popularidad y máximo por pueblo."""
from __future__ import annotations

import numpy as np
import pytest

from restmex.ranking import adjusted_scores, popularity_penalty, rank_towns
from restmex.retrieval import max_per_town

E = np.eye(4, dtype=np.float32)


def test_penalty_formula() -> None:
    pen = popularity_penalty(np.array([0, 99, 9999]), 9999, lam=0.1)
    assert pen[0] == 0.0
    assert pen[1] == pytest.approx(0.1 * np.log(100) / np.log(10000))
    assert pen[2] == pytest.approx(0.1)          # el más reseñado paga λ completo


def test_penalty_is_monotonic_and_zero_without_lambda() -> None:
    n = np.array([10, 100, 1000, 10000])
    assert np.all(np.diff(popularity_penalty(n, 10000, 0.05)) > 0)
    assert np.all(popularity_penalty(n, 10000, 0.0) == 0)


def test_adjusted_scores_broadcast() -> None:
    sim = np.array([[0.5, 0.5], [0.2, 0.9]])
    out = adjusted_scores(sim, np.array([10, 1000]), 1000, 0.1)
    assert out.shape == (2, 2)
    assert out[0, 0] > out[0, 1]                 # misma similitud: gana el menos popular


def test_max_per_town_and_argmax() -> None:
    sims = np.array([[0.1, 0.9, 0.3, 0.2], [0.8, 0.1, 0.0, 0.5]], dtype=np.float32)
    scores, best = max_per_town(sims, np.array([0, 0, 1, 1]), 2)
    np.testing.assert_allclose(scores, [[0.9, 0.3], [0.8, 0.5]])
    np.testing.assert_array_equal(best, [[1, 2], [0, 3]])


def test_top_uses_best_archetype(index) -> None:
    top, _ = rank_towns(index, E[0], E[2], lam=0.0, top_k=2, bottom_k=2)
    assert top[0].town == "Arqueo"
    assert top[0].sim_hyde == pytest.approx(1.0)
    assert top[0].archetype_label == "Arqueo-0"
    assert top[0].review == "reseña de Arqueo 0"
    assert [t.rank for t in top] == [1, 2]


def test_bottom_excludes_top(index) -> None:
    # HyDE y Anti-HyDE iguales: sin exclusión, Top y Bottom coincidirían.
    top, bottom = rank_towns(index, E[1], E[1], lam=0.0, top_k=2, bottom_k=2)
    assert not {t.town for t in top} & {b.town for b in bottom}
    assert len(bottom) == 2


def test_bottom_ranked_by_anti_similarity(index) -> None:
    _, bottom = rank_towns(index, E[0], E[2], lam=0.0, top_k=1, bottom_k=3)
    sims = [b.sim_anti for b in bottom]
    assert sims == sorted(sims, reverse=True)
    assert bottom[0].town == "Grande"            # su arquetipo fiestero es el más cercano
    assert bottom[0].archetype_label == "Grande-1"


def test_penalty_demotes_popular_town(index) -> None:
    q = (E[0] + E[1]) / np.sqrt(2)               # Grande tiene el arquetipo más cercano
    top_plain, _ = rank_towns(index, q, E[2], lam=0.0, top_k=1)
    top_pen, _ = rank_towns(index, q, E[2], lam=0.6, top_k=1)
    assert top_plain[0].town == "Grande"
    assert top_pen[0].town != "Grande"
    assert top_pen[0].score == pytest.approx(top_pen[0].sim_hyde - top_pen[0].penalty)


def test_bottom_k_capped_by_available_towns(index) -> None:
    top, bottom = rank_towns(index, E[0], E[1], lam=0.0, top_k=3, bottom_k=5)
    assert len(top) == 3 and len(bottom) == 1
