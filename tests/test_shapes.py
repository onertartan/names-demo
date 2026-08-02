import numpy as np
import pytest

from modules.experimental.shapes import (
    ALL_SHAPES,
    DISPLAY_NAMES,
    SHAPES,
    GenConfig,
    difficulty,
    make_dataset,
    prototypes,
)


def test_all_shapes_are_znormalized():
    protos = prototypes(ALL_SHAPES)
    assert np.allclose(protos.mean(axis=-1), 0.0, atol=1e-10)
    assert np.allclose(protos.std(axis=-1), 1.0, atol=1e-10)


def test_shape_registry_is_consistent():
    assert set(SHAPES.keys()) == set(ALL_SHAPES)
    assert set(DISPLAY_NAMES.keys()) == set(ALL_SHAPES)
    assert len(ALL_SHAPES) == 16


def test_correlation_matrix_bounds_and_diagonal():
    protos = prototypes(ALL_SHAPES)
    T = protos.shape[1]
    corr = (protos @ protos.T) / T
    assert np.all(np.abs(corr) <= 1.0 + 1e-10)
    assert np.allclose(np.diag(corr), 1.0, atol=1e-10)


def test_peak_trough_anticorrelated():
    protos = prototypes(["peak", "trough"])
    T = protos.shape[1]
    corr = (protos @ protos.T) / T
    assert corr[0, 1] == pytest.approx(-1.0, abs=1e-10)


def test_difficulty_all_shapes_closest_pair():
    result = difficulty(ALL_SHAPES)
    assert set(result.closest_pair) == {"linear_up", "sigmoid"}
    assert result.rho_max == pytest.approx(0.973, abs=0.01)


def test_difficulty_peak_trough_is_easy():
    # Regression test: abs(rho) would wrongly flag this anti-correlated
    # (easiest) pair as the hardest.
    result = difficulty(["peak", "trough"])
    assert result.verdict == "kolay"


def test_make_dataset_shapes_and_normalization():
    names = ["linear_up", "peak", "sine_1"]
    cfg = GenConfig(T=64, n_per_cluster=5)
    rng = np.random.default_rng(0)
    X, y = make_dataset(names, cfg, rng)

    assert X.shape == (len(names) * cfg.n_per_cluster, cfg.T)
    assert y.shape == (len(names) * cfg.n_per_cluster,)
    assert set(np.unique(y).tolist()) == {0, 1, 2}
    assert np.allclose(X.mean(axis=-1), 0.0, atol=1e-10)
    assert np.allclose(X.std(axis=-1), 1.0, atol=1e-10)


def test_make_dataset_reproducible_with_same_seed():
    names = ["linear_up", "peak"]
    cfg = GenConfig(T=32, n_per_cluster=4)
    X1, y1 = make_dataset(names, cfg, np.random.default_rng(42))
    X2, y2 = make_dataset(names, cfg, np.random.default_rng(42))
    assert np.array_equal(X1, X2)
    assert np.array_equal(y1, y2)
