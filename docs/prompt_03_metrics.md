# Task: make the CVI set pluggable, and add Dunn

Small, contained stage in `tartan-analytics`. `BaseClustering.optimal_k_analysis`
currently computes a fixed set of validity indices inline. This stage turns that set
into a registry and adds Dunn as the first entry exercised through it.

The point is not Dunn specifically — it is that adding the *next* index should be one
entry in a dict, not an edit threaded through the sweep, the storage, and the plots.

## What exists today

Per `(seed, k)`, `optimal_k_analysis` computes:

- Silhouette (cosine)
- Silhouette (euclidean)
- Davies–Bouldin
- Calinski–Harabasz

Engine-specific extras that are **not** CVIs and stay where they are: inertia
(KMeans / TimeSeriesKMeans, drives the elbow curve), AIC / BIC / negative
log-likelihood (GMM). Leave these alone; they are not part of the registry.

A working `dunn_index` already lives in `clustering/models/ga_clusterer.py`, used only
by `toy_datasets.py`. Reuse it; do not write a second implementation.

`clustering/evaluation/metrics.py` is dead code — its Gap, Dunn, and BIC sit inside a
module-level triple-quoted string, and the live remainder is a pasted sklearn example
nothing imports. Do not revive it and do not import from it. Deleting it is welcome as
a separate commit.

## 1. The registry

```python
@dataclass(frozen=True)
class CVI:
    key: str                                        # stable id used in storage
    label: str                                      # display label for plots
    fn: Callable[[np.ndarray, np.ndarray], float]   # (X, labels) -> score
    maximize: bool                                  # True: argmax picks k; False: argmin

CVI_REGISTRY: dict[str, CVI]
```

Put it wherever `optimal_k_analysis` can reach it without a circular import —
`clustering/` is the natural home. `optimal_k_analysis` iterates the registry instead
of naming indices inline.

**The `key` strings must exactly match the names used today**, so previously saved
sweep output stays readable and comparable. Rename nothing.

`maximize` matters: Davies–Bouldin is minimized, the rest are maximized. Today that
polarity is implicit in the plotting code; make it explicit on the entry and have the
"selected k" logic read it from there.

A metric that raises or returns a degenerate value for a given partition must yield
`NaN` for that `(seed, k)` cell rather than aborting the sweep. Downstream averaging
already has to tolerate `NaN`; confirm it does.

## 2. Add Dunn

One registry entry, `maximize=True`, wrapping `dunn_index`.

Dunn computes all pairwise distances, so it is O(n²) in samples where the others are
not. On the geo paths (81 provinces, 50 states) that is irrelevant. On name clustering
the sample count can reach several thousand and the sweep runs it for every
`(seed, k)`. Measure it on the largest existing path before declaring done, and if it
dominates the sweep, say so and propose a guard rather than silently shipping a much
slower sweep.

## 3. Make the plots follow the registry

`OptimalKPlotter.plot_optimal_k_analysis` currently draws a fixed set of columns in one
figure — silhouette (euclidean), silhouette (cosine), and inertia when the engine is
KMeans. Adding a metric to the sweep without touching the plotter leaves it computed
and invisible, so this belongs in the same stage.

Drive the columns off two sources:

1. **Every entry in `CVI_REGISTRY`**, in registry order. No hardcoded metric names in
   the plotter.
2. **Engine-specific extras**, appended after the registry columns: inertia for
   KMeans / TimeSeriesKMeans, AIC / BIC / negative log-likelihood for GMM. These are
   not CVIs and must not enter the registry — the plotter asks the engine what extras
   it produced.

Requirements:

- **Wrap into a grid.** Five CVIs plus inertia in a single row leaves each panel
  unreadable. Lay out at most three or four columns per row and add rows as needed;
  compute the grid from the column count rather than fixing it.
- **Orient each panel by `maximize`.** Mark the selected k on every panel — argmax for
  maximized indices, argmin for Davies–Bouldin. Read the polarity from the registry
  entry, never from a name check in the plotter.
- **Handle `NaN` columns.** A metric that failed for some `(seed, k)` still gets a
  panel; the missing points are simply absent from the curve. A metric that is `NaN`
  everywhere gets a panel labelled as unavailable rather than an empty axis or a crash.
- **Keep the existing per-seed / mean-with-spread rendering.** The current plot shows
  individual seed curves plus the mean; preserve that, do not simplify it while
  generalizing the column set.
- Panel titles come from `CVI.label`, not from the storage key.

`print_optimal_k_analysis` and the summary dataframe should pick up new metrics the
same way — from the registry, not from a list written out by hand.

## 4. No k = 1

`base_page.py` hardcodes `k_values = range(2, 11)`. **Leave it.** Every ground-truth
class list in this experiment has k ≥ 2, so `k = 1` is never the correct answer and
the indices that cannot express it lose nothing. Do not add Gap, do not make
`k_values` configurable, do not add a `k = 1` branch anywhere.

If a future stage needs it, that is a separate job touching all five pages.

## 5. Note on the two silhouettes

Add this to the registry entries' docstrings, because it affects how results are read:

every series on the Time Series path is z-normalized, so all vectors have equal norm.
For equal-norm vectors, `||a - b||^2 = 2T(1 - rho)` while `cos(a, b) = rho` — cosine
and euclidean distance are monotone transforms of each other. The two silhouette
variants therefore track each other almost exactly on that path and are not
independent evidence. They can still diverge on the name-count paths, where scaling
choices leave norms unequal.

Keep both entries. A large divergence between them on the Time Series path is a useful
signal that normalization is not doing what we expect.

## 6. Tests

- The registry contains the four existing entries plus Dunn, with the `key` strings
  unchanged from what the current code writes.
- `maximize` polarity is correct for each entry, asserted against a synthetic case
  where the right answer is known.
- A metric raising inside the sweep produces `NaN` in that cell and does not abort.
- `tests/test_app_smoke.py` stays green — all seven paths, unchanged behaviour apart
  from one extra metric appearing in the output and in the figure.
- The figure has one panel per registry entry plus the engine's extras, and the panel
  count changes when a registry entry is added — assert this by adding a dummy entry
  in the test rather than by counting a fixed number.
- A metric that is `NaN` for every `(seed, k)` still produces a labelled panel and does
  not raise.
- On the Time Series path with a known-easy three-class list, the two silhouette
  variants agree to within a small tolerance.

## Constraints

- No new CVI implementations beyond wiring `dunn_index`. Later metrics arrive as
  registry entries.
- Do not touch `k_values`, the definition of the engine-specific extras, or anything on
  the geo paths beyond the registry refactor and the plotter generalization.
- Existing storage layout and file naming unchanged.
