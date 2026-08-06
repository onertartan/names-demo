import numpy as np
import pandas as pd
import pytest

from clustering.evaluation.cvi_registry import (
    CVI,
    CVI_REGISTRY,
    _distance_cache,
    dunn_index,
    generalized_dunn,
)
from clustering.models.kmeans import KMeansEngine
from modules.experimental.shape_library import ShapeInstance
from modules.experimental.synthetic_data_generator import TimeSeriesSyntheticDataGenerator

EXPECTED_KEYS = [
    "Silhouette Score (cosine)",
    "Silhouette Score (euclidean)",
    "Davies-Bouldin Index",
    "Calinski-Harabasz Index",
    "Dunn Index",
    "Dunn Index (d2/D2)",
    "Dunn Index (d4/D1)",
    "Dunn Index (d3/D2)",
]

DUNN_KEYS = EXPECTED_KEYS[4:]


def _outlier_partition():
    # Cluster 1 carries an outlier pulled toward the others; this is where
    # the family members diverge most and where a refactor slip would be
    # least visible.
    X = np.array([
        [0.0, 0.0], [0.3, 0.1], [0.1, 0.4], [2.5, 2.5],
        [10.0, 0.0], [10.2, 0.3], [10.4, 0.1],
        [0.0, 10.0], [0.3, 10.2], [0.1, 10.4],
    ])
    labels = np.array([1, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    return X, labels


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


def test_dunn_d1_D1_pinned_to_pre_refactor_values():
    # Values recorded from the pre-refactor dunn_index (commit e70e2e6)
    # before generalized_dunn existed; the parameterized core must
    # reproduce the original variant exactly.
    X1 = np.array([[0.0], [0.1], [10.0], [10.1]])
    labels1 = np.array([1, 1, 2, 2])
    assert dunn_index(X1, labels1) == pytest.approx(99.0, rel=1e-12)
    assert generalized_dunn(X1, labels1, d="d1", D="D1") == pytest.approx(99.0, rel=1e-12)

    X2, labels2 = _outlier_partition()
    pinned = 2.2360679774997894
    assert dunn_index(X2, labels2) == pytest.approx(pinned, rel=1e-12)
    assert generalized_dunn(X2, labels2, d="d1", D="D1") == pytest.approx(pinned, rel=1e-12)


def test_dunn_variants_diverge_on_outlier_partition():
    X, labels = _outlier_partition()
    values = {key: CVI_REGISTRY[key].fn(X, labels) for key in DUNN_KEYS}
    assert all(np.isfinite(value) for value in values.values())
    distinct = sorted(values.values())
    assert all(later - earlier > 1e-6
               for earlier, later in zip(distinct, distinct[1:])), values


def test_all_dunn_variants_nan_on_degenerate_partitions():
    X = np.array([[0.0, 0.0], [1.0, 0.5], [2.0, 1.0], [3.0, 1.5]])
    single_cluster = np.array([1, 1, 1, 1])
    all_singletons = np.array([1, 2, 3, 4])
    for key in DUNN_KEYS:
        assert np.isnan(CVI_REGISTRY[key].fn(X, single_cluster)), key
        assert np.isnan(CVI_REGISTRY[key].fn(X, all_singletons)), key


def test_distance_cache_matches_uncached_and_never_goes_stale():
    from scipy.spatial.distance import pdist, squareform

    rng = np.random.default_rng(0)
    X_a = rng.standard_normal((40, 5))
    X_b = rng.standard_normal((40, 5))  # same shape and dtype, new content
    labels = np.array([1] * 20 + [2] * 20)

    _distance_cache["fingerprint"] = None  # cold start
    cold = dunn_index(X_a, labels)
    warm = dunn_index(X_a, labels)         # served from the cache
    assert warm == cold

    # The seed-swap scenario: a weaker fingerprint would hand X_a's
    # distances to X_b here. Compare against an uncached reference.
    value_b = dunn_index(X_b, labels)
    distances_b = squareform(pdist(X_b))
    reference = distances_b[:20, 20:].min() / max(
        distances_b[:20, :20].max(), distances_b[20:, 20:].max())
    assert value_b == pytest.approx(reference, rel=1e-12)
    assert value_b != cold


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
        "Dunn_d2D2_mean", "Dunn_d2D2_std",
        "Dunn_d4D1_mean", "Dunn_d4D1_std",
        "Dunn_d3D2_mean", "Dunn_d3D2_std",
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
    # 0.103 at sigma=0.1, 0.177 at 0.2, 0.222 at 0.3 (cosine higher).
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
