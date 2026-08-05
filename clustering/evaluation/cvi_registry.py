"""Cluster validity index (CVI) registry.

Adding an index is one ``CVI_REGISTRY`` entry and nothing else: the sweep
(``BaseClustering.optimal_k_analysis``), the summary dataframe, and
``OptimalKPlotter`` all iterate this registry instead of naming metrics
inline. Engine-specific extras (inertia, AIC/BIC/NLL) are not CVIs and do
not belong here — engines declare them separately.

Cost arithmetic for new entries: every entry runs once per ``(seed, k)``.
``number_of_seeds`` is a UI setting (default 3, max 100) and ``k_values``
is ``range(2, 11)``, so a sweep evaluates each metric 27 to 900 times.
The Dunn family shares one condensed pairwise-distance vector through a
content-addressed single-entry cache (``_condensed_distances``), so
``pdist`` — O(n^2 d) time, O(n^2) memory; measured ~6 ms at n=200 (the
Time Series path), ~0.2 s at n=1681 (name clustering, use-all-data), ~2 s
at n=5000 — runs once per distinct X (in the sweep: once per seed), and
each variant call pays only its block reductions. sklearn's
silhouette / Davies-Bouldin / Calinski-Harabasz compute their own
distances internally and neither use nor populate this cache. Any new
O(n^2) metric added here inherits the seeds-times-k multiplier unless it
reuses the cache.

The Dunn entries are members of Bezdek & Pal's generalized Dunn family:
an inter-cluster separation measure ``d`` over an intra-cluster diameter
measure ``D`` — min over cluster pairs of d, divided by max over clusters
of D, maximized. The full family is 18 variants (six separation measures
crossed with three diameters); only the measures below are implemented,
and a further one is a single entry in the lookup dicts. The variant
choice is not cosmetic: on the SSA baby-name data, variants scored on
identical partitions select k anywhere from 2 to 9 — a wider spread than
between Dunn, Silhouette, Davies-Bouldin and Calinski-Harabasz combined.
Implemented measures, in words:

- ``d1`` — minimum distance between points in different clusters
  (single linkage; Dunn's original 1974 numerator)
- ``d2`` — maximum distance between points in different clusters
  (complete linkage)
- ``d3`` — mean distance over all cross-cluster point pairs
  (average linkage)
- ``d4`` — euclidean distance between cluster centroids
- ``D1`` — maximum pairwise distance within a cluster (its diameter;
  Dunn's original denominator)
- ``D2`` — mean pairwise distance within a cluster

References: Bezdek & Pal (1995), "Cluster validation with generalized
Dunn's indices", Proc. 2nd NZ Int. Two-Stream Conf. on ANN and Expert
Systems, 190-193; Bezdek, Li, Attikiouzel & Windham (1998), IEEE Trans.
Systems, Man, and Cybernetics-B 28(3):301-315.

On the two silhouettes: every series on the Time Series path is
z-normalized, so all vectors have equal norm. For equal-norm vectors
``||a - b||^2 = 2T(1 - rho)`` while ``cos(a, b) = rho`` — cosine and
euclidean distance are monotone transforms of each other, so the two
variants order partitions identically and select the same k; they are not
independent evidence there. Their *values* still diverge systematically
with noise, because silhouette is a ratio statistic and the square root
compresses large distances (measured cosine minus euclidean on the easy
three-class list: 0.105/0.179/0.224 at sigma 0.1/0.2/0.3, cosine higher).
They can genuinely differ on the name-count paths, where scaling choices
leave norms unequal.
"""
from __future__ import annotations

import hashlib
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
    the two silhouettes select the same k on the z-normalized Time Series
    path while their values diverge with noise."""
    return silhouette_score(X, labels, metric="euclidean")


# ---------------------------------------------------------------------------
# Shared pairwise-distance cache for the Dunn family
# ---------------------------------------------------------------------------

_distance_cache: dict = {"fingerprint": None, "array": None,
                         "condensed": None, "square": None}


def _square_distances(X: np.ndarray) -> np.ndarray:
    """Full pairwise-distance matrix for X through a single-entry cache.

    The fingerprint hashes the full buffer (shape, dtype, blake2b-128 of
    the bytes): across sweep seeds X keeps its shape and dtype while the
    noise draw changes, so anything weaker — a sampled or shape-only key —
    would silently return one seed's distances for another seed's data. On
    a fingerprint hit the cached array is additionally compared with
    ``np.array_equal`` before use, so even a hash collision or an in-place
    mutation falls through to recomputation, never to stale distances.
    Both checks together cost O(n d), about 2% of the pdist they avoid.

    The cache holds both the condensed pdist vector and its squareform
    expansion: measurement showed rebuilding the n^2 matrix per variant
    call cost more than the reductions themselves. The held matrix is the
    same size the uncached implementation allocated transiently on every
    call.

    X must already be float and C-contiguous. Callers must not mutate the
    returned matrix.
    """
    fingerprint = (X.shape, X.dtype.str,
                   hashlib.blake2b(X, digest_size=16).digest())
    if (fingerprint == _distance_cache["fingerprint"]
            and _distance_cache["array"] is not None
            and np.array_equal(_distance_cache["array"], X)):
        return _distance_cache["square"]
    condensed = pdist(X)
    _distance_cache["fingerprint"] = fingerprint
    _distance_cache["array"] = X
    _distance_cache["condensed"] = condensed
    _distance_cache["square"] = squareform(condensed)
    return _distance_cache["square"]


# ---------------------------------------------------------------------------
# Generalized Dunn family (Bezdek & Pal) — measures in words and references
# are in the module docstring.
# ---------------------------------------------------------------------------

def _separation_single_linkage(X, distances, mask_a, mask_b):
    """d1: minimum distance between points in different clusters."""
    return distances[np.ix_(mask_a, mask_b)].min()


def _separation_complete_linkage(X, distances, mask_a, mask_b):
    """d2: maximum distance between points in different clusters."""
    return distances[np.ix_(mask_a, mask_b)].max()


def _separation_average_linkage(X, distances, mask_a, mask_b):
    """d3: mean distance over all cross-cluster point pairs."""
    return distances[np.ix_(mask_a, mask_b)].mean()


def _separation_centroid(X, distances, mask_a, mask_b):
    """d4: euclidean distance between the two cluster centroids."""
    return float(np.linalg.norm(X[mask_a].mean(axis=0) - X[mask_b].mean(axis=0)))


def _diameter_max(distances, mask):
    """D1: maximum pairwise distance within a cluster (its diameter).
    Single-member clusters contribute 0.0."""
    if np.count_nonzero(mask) < 2:
        return 0.0
    return distances[np.ix_(mask, mask)].max()


def _diameter_mean(distances, mask):
    """D2: mean pairwise distance within a cluster. Single-member clusters
    contribute 0.0."""
    count = int(np.count_nonzero(mask))
    if count < 2:
        return 0.0
    return distances[np.ix_(mask, mask)].sum() / (count * (count - 1))


# Adding a further measure is one entry in the matching dict.
_SEPARATION_MEASURES: dict[str, Callable] = {
    "d1": _separation_single_linkage,
    "d2": _separation_complete_linkage,
    "d3": _separation_average_linkage,
    "d4": _separation_centroid,
}

_DIAMETER_MEASURES: dict[str, Callable] = {
    "D1": _diameter_max,
    "D2": _diameter_mean,
}


def generalized_dunn(X: np.ndarray, labels: np.ndarray, *, d: str, D: str) -> float:
    """Generalized Dunn index: min over cluster pairs of the separation
    measure ``d``, divided by max over clusters of the diameter measure
    ``D``; maximized. Any result reported as "Dunn" must name the variant.

    Degenerate handling, identical for every variant: fewer than two
    clusters returns NaN, a zero maximum diameter returns NaN (which covers
    an all-singletons partition), single-member clusters contribute
    diameter 0.0.
    """
    if d not in _SEPARATION_MEASURES:
        raise ValueError(f"unknown separation measure {d!r}")
    if D not in _DIAMETER_MEASURES:
        raise ValueError(f"unknown diameter measure {D!r}")
    X = np.ascontiguousarray(np.asarray(X, dtype=float))
    labels = np.asarray(labels)
    unique = np.unique(labels)
    if unique.size < 2:
        return float("nan")
    distances = _square_distances(X)
    masks = [labels == u for u in unique]
    diameter = _DIAMETER_MEASURES[D]
    max_diameter = max(diameter(distances, mask) for mask in masks)
    if max_diameter == 0.0:
        return float("nan")
    separation = _SEPARATION_MEASURES[d]
    min_separation = min(
        separation(X, distances, mask_a, mask_b)
        for i, mask_a in enumerate(masks)
        for mask_b in masks[i + 1:]
    )
    return float(min_separation / max_diameter)


def dunn_index(X: np.ndarray, labels: np.ndarray) -> float:
    """Dunn's original 1974 index, Bezdek & Pal's delta_1 / Delta_1 (d1/D1).

    Numerator: single-linkage inter-cluster distance — the closest pair of
    points in different clusters. Denominator: the largest cluster diameter,
    i.e. the farthest pair of points within any one cluster.

    Both numerator and denominator rest on a single point pair, which makes
    this the most outlier-sensitive member of the family (Bezdek et al.,
    1998). GAEngine imports this exact variant as its fitness; its
    behaviour must not change.
    """
    return generalized_dunn(X, labels, d="d1", D="D1")


def _dunn_d2_D2(X: np.ndarray, labels: np.ndarray) -> float:
    """Generalized Dunn d2/D2: complete-linkage separation (maximum
    distance between points in different clusters) over mean within-cluster
    pairwise distance.

    In the family for an empirical reason, not a taxonomic one: on the two
    SSA baby-name files this was the only variant selecting the same k on
    both, while d1/D1 moved from k=7 to k=2."""
    return generalized_dunn(X, labels, d="d2", D="D2")


def _dunn_d4_D1(X: np.ndarray, labels: np.ndarray) -> float:
    """Generalized Dunn d4/D1: centroid separation (euclidean distance
    between cluster centroids) over the maximum cluster diameter."""
    return generalized_dunn(X, labels, d="d4", D="D1")


def _dunn_d3_D2(X: np.ndarray, labels: np.ndarray) -> float:
    """Generalized Dunn d3/D2: average-linkage separation (mean distance
    over all cross-cluster point pairs) over mean within-cluster pairwise
    distance."""
    return generalized_dunn(X, labels, d="d3", D="D2")


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
        CVI("Dunn Index", "Dunn (d1/D1)",
            dunn_index, True,
            "Dunn_mean", "Dunn_std"),
        CVI("Dunn Index (d2/D2)", "Dunn (d2/D2)",
            _dunn_d2_D2, True,
            "Dunn_d2D2_mean", "Dunn_d2D2_std"),
        CVI("Dunn Index (d4/D1)", "Dunn (d4/D1)",
            _dunn_d4_D1, True,
            "Dunn_d4D1_mean", "Dunn_d4D1_std"),
        CVI("Dunn Index (d3/D2)", "Dunn (d3/D2)",
            _dunn_d3_D2, True,
            "Dunn_d3D2_mean", "Dunn_d3D2_std"),
    )
}
