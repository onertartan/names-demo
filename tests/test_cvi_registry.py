import numpy as np
import pandas as pd
import pytest

from clustering.evaluation.cvi_registry import CVI, CVI_REGISTRY, dunn_index
from clustering.models.kmeans import KMeansEngine
from modules.experimental.shape_library import ShapeInstance
from modules.experimental.synthetic_data_generator import TimeSeriesSyntheticDataGenerator

EXPECTED_KEYS = [
    "Silhouette Score (cosine)",
    "Silhouette Score (euclidean)",
    "Davies-Bouldin Index",
    "Calinski-Harabasz Index",
    "Dunn Index",
]


def _three_blobs(rng):
    # Centres placed away from the origin so the cosine metric sees three
    # distinct directions, not a degenerate cluster around zero.
    centers = np.array([[10.0, 10.0], [30.0, 10.0], [10.0, 30.0]])
    X = np.vstack([center + 0.2 * rng.standard_normal((30, 2)) for center in centers])
    labels_true = np.repeat([1, 2, 3], 30)
    return X, labels_true


def test_registry_keys_unchanged_from_legacy_storage():
    assert list(CVI_REGISTRY) == EXPECTED_KEYS
    for key, cvi in CVI_REGISTRY.items():
        assert cvi.key == key
        assert cvi.label and cvi.mean_column and cvi.std_column


def test_registry_polarity_flags():
    assert CVI_REGISTRY["Davies-Bouldin Index"].maximize is False
    for key in EXPECTED_KEYS:
        if key != "Davies-Bouldin Index":
            assert CVI_REGISTRY[key].maximize is True


def test_polarity_against_known_answer():
    # The true partition must beat a shuffled one in the direction each
    # entry's maximize flag claims.
    rng = np.random.default_rng(0)
    X, labels_good = _three_blobs(rng)
    labels_bad = rng.permutation(labels_good)
    for cvi in CVI_REGISTRY.values():
        score_good = cvi.fn(X, labels_good)
        score_bad = cvi.fn(X, labels_bad)
        if cvi.maximize:
            assert score_good > score_bad, cvi.key
        else:
            assert score_good < score_bad, cvi.key


def test_dunn_known_value():
    X = np.array([[0.0], [0.1], [10.0], [10.1]])
    labels = np.array([1, 1, 2, 2])
    assert dunn_index(X, labels) == pytest.approx(9.9 / 0.1, rel=1e-9)


def test_dunn_degenerate_cases_yield_nan():
    X = np.array([[0.0], [1.0], [2.0]])
    assert np.isnan(dunn_index(X, np.array([1, 1, 1])))       # single cluster
    X_flat = np.array([[0.0], [0.0], [5.0], [5.0]])
    assert np.isnan(dunn_index(X_flat, np.array([1, 1, 2, 2])))  # zero diameters


def _tiny_sweep(**kwargs):
    rng = np.random.default_rng(0)
    X, _ = _three_blobs(rng)
    df = pd.DataFrame(X)
    return KMeansEngine.optimal_k_analysis(
        df, range(2), range(2, 4), {"n_clusters": 2, "n_init": 2},
        save_folder="", **kwargs)


def test_raising_metric_yields_nan_without_aborting(monkeypatch):
    def broken(X, labels):
        raise RuntimeError("boom")

    monkeypatch.setitem(
        CVI_REGISTRY, "Broken Index",
        CVI("Broken Index", "Broken", broken, True, "Broken_mean", "Broken_std"))

    df_summary, metrics_all, metrics_mean, *_ = _tiny_sweep()

    broken_cells = np.array(metrics_all["Broken Index"], dtype=float)
    assert broken_cells.shape == (2, 2)
    assert np.isnan(broken_cells).all()
    assert np.isnan(np.asarray(metrics_mean["Broken Index"], dtype=float)).all()
    assert df_summary["Broken_mean"].isna().all()

    healthy = np.array(metrics_all["Silhouette Score (euclidean)"], dtype=float)
    assert np.isfinite(healthy).all()
    assert np.isfinite(df_summary["Silhouette_mean (euclidean)"]).all()


def test_summary_columns_preserve_legacy_names_and_order():
    df_summary, *_ = _tiny_sweep()
    assert list(df_summary.columns) == [
        "Silhouette_mean (cosine)", "Silhouette_std (cosine)",
        "Silhouette_mean (euclidean)", "Silhouette_std (euclidean)",
        "DaviesBouldin_mean", "DaviesBouldin_std",
        "CalinskiHarabasz_mean", "CalinskiHarabasz_std",
        "Dunn_mean", "Dunn_std",
        "ARI_mean", "ARI_std",
        "Inertia_mean", "Inertia_std",   # KMeans extras, unchanged
    ]
    assert np.isfinite(df_summary["Dunn_mean"]).all()


def test_silhouettes_select_same_k_on_time_series_path():
    # On z-normalized series the two silhouette distances are monotone
    # transforms of the same correlations (d_euc = sqrt(2T * d_cos)), so
    # both variants order candidate partitions identically and select the
    # same k -- the property the experiment depends on, and it holds at
    # every sigma. Their *values* diverge systematically with noise, because
    # silhouette is a ratio statistic and the square root compresses large
    # distances: measured cosine minus euclidean on this class list is
    # 0.105 at sigma=0.1, 0.179 at 0.2, 0.224 at 0.3 (cosine higher).
    instances = [ShapeInstance("peak", 1925), ShapeInstance("trough", 1955),
                 ShapeInstance("level_shift", 1970)]
    k_values = range(2, 11)
    for sigma in (0.1, 0.2, 0.3):
        df, _ = TimeSeriesSyntheticDataGenerator(
            {"instances": instances, "n_per_cluster": 20,
             "sigma": sigma, "seed": 0}).generate()
        X = df.to_numpy()
        scores_cos, scores_euc = [], []
        for k in k_values:
            labels = KMeansEngine(n_clusters=k, random_state=0,
                                  n_init=5).fit_predict(df)
            scores_cos.append(CVI_REGISTRY["Silhouette Score (cosine)"].fn(X, labels))
            scores_euc.append(CVI_REGISTRY["Silhouette Score (euclidean)"].fn(X, labels))
        assert int(np.argmax(scores_cos)) == int(np.argmax(scores_euc)), f"sigma={sigma}"
