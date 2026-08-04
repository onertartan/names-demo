"""Cluster validity index (CVI) registry.

Adding an index is one ``CVI_REGISTRY`` entry and nothing else: the sweep
(``BaseClustering.optimal_k_analysis``), the summary dataframe, and
``OptimalKPlotter`` all iterate this registry instead of naming metrics
inline. Engine-specific extras (inertia, AIC/BIC/NLL) are not CVIs and do
not belong here — engines declare them separately.

Cost arithmetic for new entries: every entry runs once per ``(seed, k)``.
``number_of_seeds`` is a UI setting (default 3, max 100) and ``k_values``
is ``range(2, 11)``, so a sweep evaluates each metric 27 to 900 times.
``dunn_index`` builds the full pairwise distance matrix — O(n^2 d) time,
O(n^2) memory. Measured on the dev machine: ~6 ms/call at n=200 (the Time
Series path, where the high-seed runs happen), ~0.2 s/call at n=1681 (name
clustering, use-all-data) giving ~5 s per sweep at 3 seeds and ~3 min at
100, and ~2 s/call at n=5000 (~54 s / ~30 min). Any new O(n^2) metric
inherits the same multiplier.

On the two silhouettes: every series on the Time Series path is
z-normalized, so all vectors have equal norm. For equal-norm vectors
``||a - b||^2 = 2T(1 - rho)`` while ``cos(a, b) = rho`` — cosine and
euclidean distance are monotone transforms of each other, so the two
silhouette variants track each other closely there and are not independent
evidence. They can still diverge on the name-count paths, where scaling
choices leave norms unequal. Both entries stay: a large divergence on the
Time Series path signals that normalization is not doing what we expect.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


@dataclass(frozen=True)
class CVI:
    key: str          # stable id used in metrics_all and saved sweep output
    label: str        # display label for figure panels
    fn: Callable[[np.ndarray, np.ndarray], float]   # (X, labels) -> score
    maximize: bool    # True: argmax picks k; False (Davies-Bouldin): argmin
    mean_column: str  # summary-CSV column names; the legacy entries pin the
    std_column: str   # exact strings previously saved CSVs already use


def _silhouette_cosine(X: np.ndarray, labels: np.ndarray) -> float:
    """Silhouette with cosine distance. See the module docstring: on the
    z-normalized Time Series path this is not independent of the euclidean
    variant."""
    return silhouette_score(X, labels, metric="cosine")


def _silhouette_euclidean(X: np.ndarray, labels: np.ndarray) -> float:
    """Silhouette with euclidean distance. See the module docstring for why
    the two silhouettes agree on the z-normalized Time Series path."""
    return silhouette_score(X, labels, metric="euclidean")


def dunn_index(X: np.ndarray, labels: np.ndarray) -> float:
    """Minimum inter-cluster distance over maximum intra-cluster diameter.

    O(n^2 d) time and O(n^2) memory — see the module docstring for measured
    sweep costs before adding large-n paths. Degenerate partitions (a single
    cluster, or all-zero diameters) return NaN rather than raising.
    """
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    unique = np.unique(labels)
    if unique.size < 2:
        return float("nan")
    distances = squareform(pdist(X))
    masks = [labels == u for u in unique]
    max_intra = max(
        distances[np.ix_(mask, mask)].max() if np.count_nonzero(mask) > 1 else 0.0
        for mask in masks
    )
    if max_intra == 0.0:
        return float("nan")
    min_inter = min(
        distances[np.ix_(mask_a, mask_b)].min()
        for i, mask_a in enumerate(masks)
        for mask_b in masks[i + 1:]
    )
    return float(min_inter / max_intra)


CVI_REGISTRY: dict[str, CVI] = {
    entry.key: entry
    for entry in (
        CVI("Silhouette Score (cosine)", "Silhouette Score (cosine)",
            _silhouette_cosine, True,
            "Silhouette_mean (cosine)", "Silhouette_std (cosine)"),
        CVI("Silhouette Score (euclidean)", "Silhouette Score (euclidean)",
            _silhouette_euclidean, True,
            "Silhouette_mean (euclidean)", "Silhouette_std (euclidean)"),
        CVI("Davies-Bouldin Index", "Davies-Bouldin Index",
            davies_bouldin_score, False,
            "DaviesBouldin_mean", "DaviesBouldin_std"),
        CVI("Calinski-Harabasz Index", "Calinski-Harabasz Index",
            calinski_harabasz_score, True,
            "CalinskiHarabasz_mean", "CalinskiHarabasz_std"),
        CVI("Dunn Index", "Dunn Index",
            dunn_index, True,
            "Dunn_mean", "Dunn_std"),
    )
}
