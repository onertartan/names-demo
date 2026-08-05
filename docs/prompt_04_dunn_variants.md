# Task: generalized Dunn variants in the CVI registry

Small stage. The registry currently has one Dunn entry. It is Bezdek & Pal's
`d1/D1` — minimum single-linkage inter-cluster distance over maximum cluster
diameter, i.e. Dunn's original 1974 definition. Add two more variants and make the
family structure explicit.

**Add, do not replace.** `d1/D1` is what "the Dunn index" means in the literature and
in every result already stored under `results/`. Removing it would break comparability
with published benchmarks and with prior sweeps, and the instability of that specific
variant is a finding this experiment needs to be able to report.

## Why

Bezdek & Pal generalized Dunn's index into a family: an inter-cluster distance `d`
crossed with an intra-cluster diameter `D`. The variant choice is not cosmetic. On the
SSA baby-name data, twelve variants scored on identical k-means partitions select k
anywhere from 2 to 9 — a wider spread than between Dunn, Silhouette, Davies–Bouldin
and Calinski–Harabasz combined.

Bezdek, Li, Attikiouzel & Windham (1998), *IEEE Trans. SMC-B* 28(3):301–315, report
that `d1` — the minimum distance between points in a pair of sets — is the least
reliable separation measure when clusters form volumetric clouds, and that variants
using average intra-cluster distance outperform the original on outlier-prone data.
Our own measurement is consistent: `d1/D1` was the only index to shift from k=7 to
k=2 between the male and female files.

## 1. Generalized implementation

Replace the body of the single Dunn function with a parameterized core in the registry
module:

```python
def generalized_dunn(X, labels, *, d: str, D: str) -> float: ...
```

Separation measures (`d`):

| key | definition |
|---|---|
| `d1` | min over pairs of points in different clusters (single linkage) |
| `d3` | mean over all such pairs (average linkage) |
| `d4` | distance between cluster centroids |

Diameter measures (`D`):

| key | definition |
|---|---|
| `D1` | max pairwise distance within a cluster |
| `D2` | mean pairwise distance within a cluster |

Index value is `min over cluster pairs of d(Ci, Cj)` divided by
`max over clusters of D(Ci)`, maximized.

Keep the current degenerate handling exactly: fewer than two clusters returns NaN, a
zero denominator returns NaN, single-member clusters contribute diameter 0.0.

Implement only the four `d` and two `D` measures above — not the full 18-variant grid.
Adding a further one should be a single entry in the lookup dict.

## 2. Registry entries

Three entries:

| `key` | `label` | d/D |
|---|---|---|
| `Dunn Index` | `Dunn (d1/D1)` | d1 / D1 |
| `Dunn Index (d4/D1)` | `Dunn (d4/D1)` | d4 / D1 |
| `Dunn Index (d3/D2)` | `Dunn (d3/D2)` | d3 / D2 |

**The `key` of the existing entry must stay `Dunn Index` verbatim** — it is what the
current `metrics_all` and the saved sweep output use, and the storage-compatibility
rule from the previous stage still applies. Only its `label` changes. The two new
entries need new keys and their own `mean_column` / `std_column` names following the
existing convention (`Dunn_d4D1_mean` / `_std`, `Dunn_d3D2_mean` / `_std`).

All three are `maximize=True`.

## 3. Share the distance matrix — this matters

The registry calls each entry's `fn(X, labels)` independently, so three Dunn entries
would each call `pdist(X)` and pay the O(n²) cost three times over. Within one
`(seed, k)` evaluation `X` is identical across every entry, and across the whole k
sweep at a fixed seed it is identical too.

Add a single-entry cache for the condensed pairwise distance vector, keyed on a cheap
fingerprint of `X` (shape, dtype, and a hash of the underlying buffer). Invalidate on
mismatch. Do not use `functools.lru_cache` — ndarrays are unhashable and wrapping them
in a hashable proxy invites subtle staleness bugs.

Then **measure and report the actual cost** of the three variants together versus the
single variant today, on the largest path currently exercised, using the real
`seeds × len(k_values)` multiplier already documented in the module docstring. If the
three-variant sweep is more than roughly 1.5× the one-variant sweep, the cache is not
working — say so rather than shipping it.

## 4. Docstrings

The module or function docstring must state, for each entry, exactly which `d` and `D`
it uses in words, not only by key — the whole reason this stage exists is that
"the Dunn index" was ambiguous enough to produce a false mismatch when two independent
implementations were compared. Cite Bezdek & Pal (1995), *Proc. 2nd NZ Int.
Two-Stream Conf. on ANN and Expert Systems*, 190–193, and the 1998 IEEE paper above.

Note in the `d1/D1` docstring that both its numerator and denominator rest on a single
point pair, which is why it is the most outlier-sensitive member of the family.

## 5. Tests

- The three entries exist with the keys above; `Dunn Index` is unchanged.
- `generalized_dunn(..., d="d1", D="D1")` reproduces the current `dunn_index` output
  exactly on a fixed synthetic partition — pin this with the current values so a
  refactor cannot silently change the original variant.
- The three variants give different values on a partition where they should, and all
  three return NaN on a single-cluster partition and on an all-singletons partition.
- The distance cache returns identical results to an uncached path.
- `tests/test_app_smoke.py` stays green; the figure gains two panels, which the
  existing registry-driven panel-count test should already cover.

## Constraints

- No change to `k_values`, no new sweep runner, no change to the other four indices.
- Storage layout and existing column names unchanged; new columns appended.
- `ga_clusterer.dunn_index` keeps importing the `d1/D1` variant so `toy_datasets.py`
  behaviour is unchanged.
