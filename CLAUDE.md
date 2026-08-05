# tartan-analytics
## Before starting work

Run `git status` and report anything uncommitted before making the first
edit. Do not stage or commit files you did not modify yourself; if a file
you need to touch already has uncommitted changes, say so and wait.

Streamlit app for name/surname and population analytics across Türkiye and the USA,
plus a cluster validity experiment in `modules/experimental/`.

Entry point: `streamlit run main.py`. Pages are declared in `main.py` via
`st.navigation`; each page module ends with `PageClass().run()`.

## Architecture

- `modules/base_page.py` — `BasePage` ABC. Owns `tab_clustering()`, which runs the
  clustering engine, the optimal-k sweep, and the PCA plot. Most pages inherit
  through `base_page_names.py` (name data) or `base_page_common.py` (population data).
- `clustering/` — engines behind `models/factory.get_engine_class`, plus
  `BaseClustering.optimal_k_analysis`, which sweeps `k_values` across random seeds and
  returns per-metric curves, consensus labels, and cross-seed ARI. **This is the CVI
  sweep machinery. Do not write a second one.**
- `viz/` — plotters and `gui_helpers/`. Streamlit widget code lives in `gui_helpers`,
  not in page modules.
- `modules/experimental/` — the cluster validity experiment (below).
- `docs/` — staged specs the experimental modules were built from.

## Cluster validity experiment

Measures which cluster validity index correctly recovers a known number of clusters in
synthetic time series. Reached through the **Experiment** page → *Synthetic Data* tab.

`shapes.py` provides 16 z-normalized shape prototypes and a difficulty diagnostic.
`shape_library.py` positions those shapes on a year axis and turns them into named
classes. `synthetic_data_generator.py` holds the `SyntheticDataGenerator` ABC; each
dataset family is a subclass returning `(df, ground_truth_labels)` and setting
`self.ground_truth_labels`, which `tab_clustering` forwards to `optimal_k_analysis`.

Read `docs/prompt_01_shapes.md`, `docs/prompt_02_library_and_ui.md`, and
`docs/prompt_03_metrics.md` before changing the public surface of anything in
`modules/experimental/` or the sweep.

### Indices actually computed

Per `(seed, k)` in `optimal_k_analysis`: Silhouette (cosine), Silhouette (euclidean),
Davies–Bouldin, Calinski–Harabasz, and four generalized Dunn variants — d1/D1, d2/D2,
d4/D1, d3/D2 (Bezdek & Pal separation × diameter measures, sharing one
content-addressed pairwise-distance cache in `cvi_registry.py`). Davies–Bouldin is
minimized; the rest are maximized.

**Any result reported as "Dunn" must name the variant.** `d1/D1` is the reference
variant — Dunn's original 1974 index, kept for comparability with published
benchmarks and with everything already stored under `results/`. `d2/D2` earned its
place empirically, not taxonomically: on the two SSA files it was the only variant
selecting the same k on both, while `d1/D1` moved from k=7 to k=2.

**Adding an index is one `CVI_REGISTRY` entry and nothing else.** The sweep, the
summary table, and `OptimalKPlotter` all iterate the registry — no metric name is
written out by hand in any of them, and panel orientation comes from the entry's
`maximize` flag rather than a name check. If you find yourself editing a plotter to
show a new metric, the registry wiring has been broken.

Engine-specific extras, not CVIs and not part of the registry: inertia (KMeans /
TimeSeriesKMeans, drives the elbow curve), AIC / BIC / negative log-likelihood (GMM).
The plotter appends these after the registry columns by asking the engine what it
produced.

**The two silhouettes are not independent on the Time Series path.** Every series is
z-normalized, so all vectors have equal norm; for equal-norm vectors
`||a - b||^2 = 2T(1 - rho)` and `cos(a, b) = rho`, making cosine and euclidean
distance monotone transforms of each other. Orderings therefore agree exactly — both
variants select the same k — but their *values* diverge systematically with noise,
because silhouette is a ratio statistic and the square root compresses large
distances: measured cosine minus euclidean on the easy three-class list is 0.105 at
sigma 0.1, 0.179 at 0.2, 0.224 at 0.3, cosine always higher. A divergence far outside
that pattern means normalization is not behaving as expected. The two can
legitimately differ on the name-count paths, where norms are unequal.

### Not implemented — do not assume otherwise

- **Gap statistic.** Absent. The only Gap code in the repo sits inside a module-level
  triple-quoted string in `clustering/evaluation/metrics.py`, which is dead — nothing
  imports it, and the live remainder is a pasted sklearn example. Do not revive it.
- **`k = 1`.** `k_values` is hardcoded to `range(2, 11)` in `base_page.py` and is
  shared by all five pages. Every ground-truth class list in this experiment has
  k ≥ 2, so `k = 1` is never a correct answer and nothing is lost by its absence.
  Do not add it as a side effect of other work.

### Invariants — these look like bugs but are correct. Do not "fix" them.

- **Signed rho, never `abs(rho)`.** `shapes.difficulty` finds the closest pair by the
  maximum of the *signed* correlation. In z-normalized space
  `||a - b||^2 = 2T(1 - rho)`, so `rho = +1` is closest (hard to separate) and
  `rho = -1` is farthest (easy). Taking `abs()` would flag `peak`/`trough` — the
  easiest pair — as the hardest.
- **Operation order in `make_dataset`: amplitude → noise → z-normalization.**
  Z-normalization removes the amplitude factor mathematically, but because noise is
  added *after* scaling, low-amplitude series keep proportionally more noise.
  Reordering silently makes `amplitude_jitter` a no-op. This sequence must live in
  exactly one function.
- **No elastic distances anywhere — no DTW, no soft-DTW, no warping.** Class identity
  is the pair (base shape, time position), so `peak@1925` and `peak@1960` are
  deliberately different clusters. Warping aligns them and collapses exactly the
  distinction this experiment exists to measure. Euclidean family only.
- **The class list is fixed for a run.** The user composes it in the UI; the ground
  truth k comes from that list. Across sweep seeds, only the noise draw changes — the
  instance list and therefore k_true must not be resampled, and `optimal_k_analysis`'s
  `kwargs["centers"] = k` is ignored by the time-series generator.
- **The synthetic time axis is years 1901–2000, `T = 100`,** grid step `1/99`. Fixed by
  `shape_library.py`; `GenConfig.T` stays generic at 128 for other callers.
- **Positioned shapes are clipped at the window edge, never wrapped.** A peak centred
  at 1905 is a truncated peak, which is a legitimate class. Periodic boundaries would
  make `@1905` and `@1995` near-identical.

### Width convention

Normative. `w_t = W / 99` for a width `W` given in years. Prototypes are evaluated on
`np.linspace(0, 1, 100)`; route every year conversion through `year_to_t`.

| Shape | `width` means | Formula |
|---|---|---|
| `peak`, `trough` | FWHM | `exp(-((t-c)/h)^2)`, `h = w_t / (2*sqrt(ln 2))` |
| `impulse`, `cylinder` | full support | `abs(t - c) <= w_t/2` (inclusive) |
| `sigmoid` | 10%–90% transition | `1/(1+exp(-s(t-c)))`, `s = 2*ln(9)/w_t` |
| `skewed_peak`, `funnel` | full duration of the rise | — |
| `level_shift` | n/a | step at `c`, `t >= c` (inclusive) |
| `sine_1`, `sine_2`, `damped_sine` | n/a | `position` is a phase shift in years |

Bounds are inclusive, unlike the base lambdas in `shapes.py`, which use strict
comparisons and are pinned by `tests/test_shapes.py`. That difference is intentional.

`suggested_min_gap` places the pair centred in the window
(`y1 = 1901 + (99 - g)//2`) and scans `g` upward; `rho(g)` is non-monotone for the
periodic shapes, so bisection is unsafe. Reference values live in
`docs/prompt_02_library_and_ui.md` §2. If one drifts, fix the code — never re-tune the
geometry to match a number.

## Conventions

- Identifiers, docstrings, and comments in English. User-facing Streamlit strings in
  Turkish, confined to display-name dicts and `viz/gui_helpers/`.
- **No non-ASCII in `print()` anywhere.** The development console is cp1254; a
  non-ASCII print raises `UnicodeEncodeError` and kills the page render. This is a
  hard crash, not mojibake.
- Session state keys follow `"<key>_" + page_name`. `utils.SessionAdapter` and
  `PageKeys` wrap this; prefer them over raw `st.session_state` in new code.
- Type hints on new code; `from __future__ import annotations`.
- No global RNG. Stochastic functions take an `np.random.Generator`.
- New data goes to `data/preprocessed/`; sweep output to `results/<country>/<tab>/`.

## Testing

- `pytest -m "not smoke"` — fast unit loop, roughly 20s.
- `pytest tests/test_app_smoke.py` — seven AppTest paths covering every page.
  `stx.tab_bar` and `st.file_uploader` are custom components that cannot run headless
  and are monkeypatched; everything downstream of them is real. Tab clicks and file
  uploads still need occasional manual verification.
- Verify by running the real path, not by static reasoning.
- Fixes for pre-existing bugs found en route go in their own commit, separate from
  feature work, and get their own line in the summary.

## Known rough edges

- `run_optimal_k_analysis_helper` attaches consensus labels from the sweep (seeds
  `0..n_seeds-1`) to the frame generated in `render()` with the user's seed, so plotted
  labels come from a different draw than the plotted points. Pre-existing; affects
  blobs and time series alike.
- `clustering/evaluation/metrics.py` is dead — a triple-quoted string plus a pasted
  sklearn example, imported by nothing. Safe to delete.
- `base_page_names.preprocess_clustering` mutates `df_year_male` / `df_year_female`
  slices in place and will raise `SettingWithCopyWarning`. It also appends `_female` to
  the male frame and `_male` to the female frame, which looks inverted.

## Commands

```
streamlit run main.py
pytest -m "not smoke"
pytest tests/test_app_smoke.py
```
