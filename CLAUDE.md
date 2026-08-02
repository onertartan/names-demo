# tartan-analytics

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

Measures which cluster validity index (silhouette, Calinski–Harabasz,
Davies–Bouldin, Dunn, Gap) correctly recovers a known number of clusters in synthetic
time series. Reached through the **Experiment** page → *Synthetic Data* tab.

`shapes.py` provides 16 z-normalized shape prototypes and a difficulty diagnostic.
`shape_library.py` positions those shapes on a year axis and turns them into named
classes. `synthetic_data_generator.py` holds the `SyntheticDataGenerator` ABC; each
dataset family is a subclass returning `(df, ground_truth_labels)` and setting
`self.ground_truth_labels`, which `tab_clustering` forwards to `optimal_k_analysis`.

Read `docs/prompt_01_shapes.md` and `docs/prompt_02_library_and_ui.md` before changing
the public surface of anything in `modules/experimental/`.

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
- **Gap is the only index defined at `k = 1`.** The others return NaN there by design.

### Width convention

Normative. `w_t = W / 99` for a width `W` given in years.

| Shape | `width` means | Formula |
|---|---|---|
| `peak`, `trough` | FWHM | `exp(-((t-c)/h)^2)`, `h = w_t / (2*sqrt(ln 2))` |
| `impulse`, `cylinder` | full support | `abs(t - c) <= w_t/2` |
| `sigmoid` | 10%–90% transition | `1/(1+exp(-s(t-c)))`, `s = 2*ln(9)/w_t` |
| `skewed_peak`, `funnel` | full duration of the rise | — |
| `level_shift` | n/a | step at `c` |
| `sine_1`, `sine_2`, `damped_sine` | n/a | `position` is a phase shift in years |

Reference correlations live in `docs/prompt_02_library_and_ui.md` §2 and are covered
by `tests/test_shape_library.py`. If a value drifts, fix the code — never re-tune the
geometry to match a number.

## Conventions

- Identifiers, docstrings, and comments in English. User-facing Streamlit strings in
  Turkish, confined to display-name dicts and `viz/gui_helpers/`.
- Avoid Turkish characters in code outside those places — dotted/dotless I breaks
  `str.lower()` in unexpected ways. Note `base_page_names.py` sets
  `locale.setlocale(locale.LC_ALL, 'tr_TR.utf8')` for name sorting; that is
  intentional and separate.
- Session state keys follow `"<key>_" + page_name`. `utils.SessionAdapter` and
  `PageKeys` wrap this; prefer them over raw `st.session_state` in new code.
- Type hints on new code; `from __future__ import annotations`.
- No global RNG. Stochastic functions take an `np.random.Generator`.
- New data goes to `data/preprocessed/`; sweep output to `results/<country>/<tab>/`.

## Known rough edges

- `run_optimal_k_analysis_helper` attaches consensus labels from the sweep (seeds
  `0..n_seeds-1`) to the frame generated in `render()` with the user's seed, so plotted
  labels come from a different draw than the plotted points. Pre-existing; affects
  blobs today and time series tomorrow.
- `base_clustering` reads `data_generator.n_features` whenever `"experiment"` is in
  `save_folder`, but `data_generator` is `None` on the Experiment page's Names Data
  path. Guarded as of the Time Series work; do not remove the guard.
- `base_page_names.preprocess_clustering` mutates `df_year_male` / `df_year_female`
  slices in place and will raise `SettingWithCopyWarning`. It also appends `_female` to
  the male frame and `_male` to the female frame, which looks inverted. Not yet
  addressed.
- Debug `print()` calls are scattered through `base_page_common.py`. Do not add more;
  removing them is welcome.

## Commands

```
streamlit run main.py
pytest -q
```
