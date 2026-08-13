import numpy as np
import pandas as pd
import pytest

from clustering.models.hierarchical import HierarchicalBaseClusteringEngine


def _blobs_df():
    rng = np.random.default_rng(0)
    centers = np.array([[0.0, 0.0], [8.0, 0.0], [0.0, 8.0]])
    X = np.vstack([center + 0.3 * rng.standard_normal((10, 2))
                   for center in centers])
    return pd.DataFrame(X)


def test_linkage_method_defaults_to_average():
    engine = HierarchicalBaseClusteringEngine(n_clusters=3, metric="cosine")
    assert engine.linkage_method == "average"


def test_linkage_method_is_honoured_not_hardcoded():
    engine = HierarchicalBaseClusteringEngine(
        n_clusters=3, metric="euclidean", linkage_method="ward")
    assert engine.linkage_method == "ward"
    engine = HierarchicalBaseClusteringEngine(
        n_clusters=3, metric="cosine", linkage_method="complete")
    assert engine.linkage_method == "complete"


def test_euclidean_only_linkages_reject_other_metrics():
    for linkage_method in ("ward", "centroid", "median"):
        with pytest.raises(ValueError, match="euclidean"):
            HierarchicalBaseClusteringEngine(
                n_clusters=3, metric="cosine", linkage_method=linkage_method)


def test_ward_fit_predict_recovers_blobs():
    df = _blobs_df()
    engine = HierarchicalBaseClusteringEngine(
        n_clusters=3, metric="euclidean", linkage_method="ward")
    labels = engine.fit_predict(df)["clusters"].to_numpy()
    assert len(np.unique(labels)) == 3
    # each true blob lands wholly inside one cluster
    for block in (labels[:10], labels[10:20], labels[20:]):
        assert len(np.unique(block)) == 1


def test_average_cosine_path_still_works():
    df = _blobs_df() + 5.0  # away from the origin so cosine is well-behaved
    engine = HierarchicalBaseClusteringEngine(
        n_clusters=3, metric="cosine", linkage_method="average")
    labels = engine.fit_predict(df)["clusters"].to_numpy()
    assert len(np.unique(labels)) == 3
