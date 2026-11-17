"""Bootstrap pareado por consulta."""
import numpy as np
import pytest

from restmex.bootstrap import macro_mean, paired_bootstrap


def test_macro_mean_weighs_groups_equally() -> None:
    values, groups = np.array([1.0, 1.0, 1.0, 0.0]), np.array([0, 0, 0, 1])
    assert macro_mean(values, groups, 3) == 0.5          # el grupo 2 no aparece: no cuenta


def test_paired_bootstrap_is_deterministic_and_detects_a_clear_gain() -> None:
    rng = np.random.default_rng(0)
    base = rng.random(300)
    groups = np.arange(300) % 10
    better = paired_bootstrap(base + 0.1, base, groups)
    assert better == paired_bootstrap(base + 0.1, base, groups)
    assert better["diff"] == pytest.approx(0.1, abs=1e-4) and better["significant"]
    assert better["ci95"][0] <= better["diff"] <= better["ci95"][1]
    same = paired_bootstrap(base, base)
    assert same["diff"] == 0 and same["ci95"] == [0.0, 0.0] and not same["significant"]


def test_paired_bootstrap_noise_is_not_significant() -> None:
    rng = np.random.default_rng(1)
    a, b = rng.random(200), rng.random(200)
    out = paired_bootstrap(a, b, n_boot=500)
    assert out["ci95"][0] < 0 < out["ci95"][1] and out["n_boot"] == 500


def test_paired_bootstrap_rejects_unpaired_inputs() -> None:
    with pytest.raises(ValueError):
        paired_bootstrap([1.0, 0.0], [1.0])
