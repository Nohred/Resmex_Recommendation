"""Métricas de evaluación y calibración de λ."""
from __future__ import annotations

import numpy as np
import pytest

from restmex.evaluate import (
    balanced_sample, choose_lambda, lambda_sweep, popularity_bias, retrieval_metrics, top_k_frequency, true_ranks,
)


def test_true_ranks_and_metrics() -> None:
    scores = np.array([[0.9, 0.1, 0.2], [0.3, 0.2, 0.8], [0.1, 0.5, 0.4]])
    true = np.array([0, 1, 2])
    ranks = true_ranks(scores, true)
    np.testing.assert_array_equal(ranks, [1, 3, 2])
    m = retrieval_metrics(ranks, true, 3)
    assert m["hit@1"] == pytest.approx(1 / 3)
    assert m["hit@5"] == 1.0
    assert m["mrr"] == pytest.approx((1 + 1 / 3 + 1 / 2) / 3)


def test_macro_weights_towns_equally() -> None:
    ranks = np.array([1, 1, 1, 1, 9])            # 4 consultas del pueblo 0 (aciertos), 1 del pueblo 1 (fallo)
    true = np.array([0, 0, 0, 0, 1])
    m = retrieval_metrics(ranks, true, 2)
    assert m["hit@1"] == pytest.approx(0.8) and m["macro_hit@1"] == pytest.approx(0.5)


def test_top_k_frequency_and_bias() -> None:
    scores = np.array([[3, 2, 1, 0], [3, 2, 0, 1], [3, 0, 2, 1]], dtype=float)
    np.testing.assert_array_equal(top_k_frequency(scores, 4, k=2), [3, 2, 1, 0])
    bias = popularity_bias(np.tile([4.0, 3, 2, 1, 0, -1], (5, 1)), np.array([600, 500, 400, 300, 200, 100]), k=2)
    assert bias["spearman"] > 0.5


def test_balanced_sample() -> None:
    true = np.array([0] * 10 + [1] * 3 + [2] * 5)
    idx = balanced_sample(true, 3)
    assert np.bincount(true[idx]).tolist() == [3, 3, 3]


def test_lambda_sweep_reduces_bias_and_choice() -> None:
    rng = np.random.default_rng(0)
    n_reviews = np.array([10000, 3000, 500, 100, 50])
    true = rng.integers(0, 5, 400)
    sims = rng.normal(0, 0.02, (400, 5)) + 0.04 * np.log1p(n_reviews) / np.log1p(10000)   # sesgo hacia populares
    sims[np.arange(400), true] += 0.03
    rows = lambda_sweep(sims, true, n_reviews, 10000, [0.0, 0.04, 0.3], balanced_sample(true, 5))
    assert rows[0]["spearman_balanced"] > rows[1]["spearman_balanced"]
    assert choose_lambda(rows) == 0.04           # quita el sesgo sin perder MRR macro
    assert rows[2]["macro_mrr"] < rows[1]["macro_mrr"]   # λ excesivo castiga de más
