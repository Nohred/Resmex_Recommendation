"""Agregación arquetipo → pueblo y k adaptativo del barrido de arquetipos."""
import numpy as np
import pytest

from restmex.aggregation import aggregate_per_town, k_log, small_cluster_mask
from restmex.retrieval import max_per_town

TOWN_OF = np.array([0, 0, 0, 1, 1, 2])
SIMS = np.array([[0.9, 0.5, 0.1, 0.6, 0.6, 0.7]], dtype=np.float32)


def test_max_matches_the_retrieval_used_by_the_web() -> None:
    rng = np.random.default_rng(0)
    sims = rng.random((7, 6)).astype(np.float32)
    np.testing.assert_allclose(aggregate_per_town(sims, TOWN_OF, 3), max_per_town(sims, TOWN_OF, 3)[0])


def test_top2_averages_the_two_best_and_handles_single_archetype_towns() -> None:
    np.testing.assert_allclose(aggregate_per_town(SIMS, TOWN_OF, 3, "top2")[0], [0.7, 0.6, 0.7], atol=1e-6)


def test_lse_tends_to_max_with_small_tau_and_counts_archetypes_with_large_tau() -> None:
    cold = aggregate_per_town(SIMS, TOWN_OF, 3, "lse", tau=1e-3)[0]
    np.testing.assert_allclose(cold, [0.9, 0.6 + 1e-3 * np.log(2), 0.7], atol=1e-5)
    warm = aggregate_per_town(SIMS, TOWN_OF, 3, "lse", tau=10.0)[0]
    assert warm[0] > warm[1] > warm[2]               # con τ grande manda el número de arquetipos


def test_max_big_ignores_small_clusters_but_never_empties_a_town() -> None:
    sizes = np.array([500, 400, 10, 20, 5, 1000])
    shares = np.array([0.55, 0.44, 0.01, 0.8, 0.2, 1.0])
    keep = small_cluster_mask(sizes, shares, TOWN_OF, min_share=0.02, min_size=30)
    assert keep.tolist() == [True, True, False, True, True, True]     # el pueblo 1 conserva los suyos
    sims = np.array([[0.1, 0.2, 0.9, 0.3, 0.4, 0.5]], dtype=np.float32)
    np.testing.assert_allclose(aggregate_per_town(sims, TOWN_OF, 3, "max_big", keep=keep)[0], [0.2, 0.4, 0.5], atol=1e-6)
    with pytest.raises(ValueError):
        aggregate_per_town(sims, TOWN_OF, 3, "max_big")
    with pytest.raises(ValueError):
        aggregate_per_town(sims, TOWN_OF, 3, "mediana")


def test_k_log_is_clipped() -> None:
    assert k_log(653, 1) == 6 and k_log(40804, 1) == 11 and k_log(40804, 3) == 30 and k_log(5, 1) == 3


def test_index_uses_the_configured_aggregation() -> None:
    from restmex.config import RankingConfig
    from restmex.retrieval import ArchetypeIndex

    vectors = np.eye(4, dtype=np.float32)[:3]
    meta = [{"town": "a"}, {"town": "a"}, {"town": "b"}]
    stats = {"towns": {"a": {"n_reviews": 10}, "b": {"n_reviews": 5}}}
    index = ArchetypeIndex.from_parts(vectors, meta, stats)
    q = np.array([0.6, 0.6, 0.5, 0.0], dtype=np.float32)
    hard, best = index.town_scores(q)
    soft, best_soft = index.configure(RankingConfig(aggregation="lse", lse_tau=0.5)).town_scores(q)
    np.testing.assert_allclose(hard[0], [0.6, 0.5], atol=1e-6)
    np.testing.assert_allclose(soft[0], [0.6 + 0.5 * np.log(2), 0.5], atol=1e-6)
    assert best.tolist() == best_soft.tolist()
    with pytest.raises(ValueError):
        index.configure(RankingConfig(aggregation="mediana"))
