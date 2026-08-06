# Task: time-positioned shape classes and the Time Series synthetic data tab

Extends the existing `tartan-analytics` Streamlit app. `modules/experimental/shapes.py`
is already in place (see `docs/prompt_01_shapes.md`). This stage adds time-positioned
shape instances and wires them into the **Experiment → Synthetic Data → Time Series**
tab, which currently exists as a stub.

Read the existing code before writing anything. This task is mostly integration,
not greenfield.

## The design change driving this stage

Ground-truth class identity is now the pair **(base shape, time position)**, not the
base shape alone. Two Gaussian peaks centred at 1925 and 1960 are *different*
clusters, deliberately. The experiment treats misalignment as a real difference, so
the same base shape may appear several times in one dataset at different positions.

Two consequences that hold throughout the repo:

1. **Only Euclidean-family distances are valid.** DTW and other elastic distances
   warp along the time axis, which would align `peak@1925` with `peak@1960` and
   collapse the two clusters this design exists to separate. Do not add elastic
   distance options to `clustering/` or anywhere else.
2. **Not every shape can be meaningfully repositioned** — see §2.

## 0. What already exists

- `modules/experimental/shapes.py` — `TIERS`, `ALL_SHAPES`, `DISPLAY_NAMES`,
  `SHAPES`, `zscore`, `prototypes`, `GenConfig`, `make_dataset`, `Difficulty`,
  `difficulty`. Do not rewrite these.
- `modules/experimental/synthetic_data_generator.py` — the `SyntheticDataGenerator`
  ABC and `BlobsSyntheticDataGenerator`. The new generator is a sibling subclass.
- `modules/experimental/experiment.py` — the `Experiment` page. `render_tabs()`
  already defines a `time_series` sub-tab; `render()` does not handle it yet.
- `viz/gui_helpers/base_page_names/render_tabs_helpers.py` — contains
  `render_synthetic_data()`, the blobs parameter UI. Mirror its style.
- `viz/plotters/synthetic_data_plotter.py` — `SyntheticDataPlotter.plot_synthetic_data`.
- `BasePage.tab_clustering(...)` and `BaseClustering.optimal_k_analysis(...)` already
  run the CVI sweep across `k_values` and random seeds. **Do not write a new sweep
  runner or new CVI implementations** — this stage only has to hand them a dataset.

## 1. Two minimal splits in `shapes.py`

Both follow the same pattern: extract a prototype-taking core, leave the existing
name-taking function as a thin wrapper. Public signatures and behaviour of the
existing functions must not change, and `tests/test_shapes.py` must pass unmodified.

```python
def make_dataset_from_prototypes(
    protos: np.ndarray, cfg: GenConfig, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]: ...

def difficulty_from_prototypes(
    protos: np.ndarray, labels: list[str], sigma: float = 0.3, amp_min: float = 1.0
) -> Difficulty: ...
```

Positioned instances cannot go through the name-taking versions, because those build
prototypes from base-shape keys and cannot see positions.

These splits matter because two invariants must each live in exactly one place: the
**amplitude → noise → z-normalization** order, and the **signed-rho** closest-pair
logic. Do not reimplement either in the new module.

Leave `GenConfig.T`'s default at 128. The year axis fixes `T = 146`, but that is the
caller's job (§2), not a change to the generic library.

## 2. New module: `modules/experimental/shape_library.py`

Depends on `numpy` and `shapes` only. No streamlit, no pandas, no sklearn.
JSON serialization of a class list plus config lives here, not in the UI, so the
round-trip test can run without Streamlit.

### Time axis

```python
YEARS: np.ndarray = np.arange(1880, 2026)   # dtype=int, length 146
YEAR_MIN, YEAR_MAX = 1880, 2025
T_YEARS = 146
```

The grid is `np.linspace(0, 1, 146)`, so the step is `1/145`, not `1/146` — the
divisor is `YEAR_MAX - YEAR_MIN`, not `T_YEARS`, and must be derived from the
constants rather than written as a literal:

```
year_to_t(y) = (y - YEAR_MIN) / 145
w_t          = width_in_years / 145
```

Keep the conversion in exactly one pair of helpers (`year_to_t`, `t_to_year`); do not
scatter the arithmetic. Getting the 145 wrong is the single most likely cause of the
reference values below failing to reproduce.

### Position kinds

```python
class PositionKind(str, Enum):
    CENTER = "center"   # event symmetric about a year
    ONSET  = "onset"    # something begins at a year
    PHASE  = "phase"    # periodic; position is a phase shift in years
    NONE   = "none"     # global shape, repositioning is meaningless
```

| Shape | Kind | Position means | Width default (years) |
|---|---|---|---|
| `peak` | CENTER | year of maximum | 15 |
| `trough` | CENTER | year of minimum | 15 |
| `impulse` | CENTER | year of the spike | 6 |
| `cylinder` | CENTER | midpoint of the plateau | 50 |
| `skewed_peak` | ONSET | year the rise begins | 30 |
| `sine_1` | PHASE | year of first upward zero crossing | — |
| `sine_2` | PHASE | same | — |
| `damped_sine` | PHASE | year the oscillation starts | — |
| `level_shift` | ONSET | year of the step | — |
| `sigmoid` | ONSET | midpoint year of the transition | 20 |
| `funnel` | ONSET | year the ramp begins | 50 |
| `linear_up`, `linear_down`, `exponential`, `saturating`, `trend_seasonal` | NONE | — | — |

### Width convention

**Normative. Implement to this definition; do not tune the geometry to make the
reference numbers below come out right.** If a value misses by more than `atol=0.01`,
report it rather than adjusting.

| Shape | `width` means | Formula |
|---|---|---|
| `peak`, `trough` | FWHM | `exp(-((t-c)/h)^2)` with `h = w_t / (2*sqrt(ln 2))` |
| `impulse`, `cylinder` | full support | `abs(t - c) <= w_t / 2` |
| `sigmoid` | 10%–90% transition duration | `1/(1+exp(-s(t-c)))` with `s = 2*ln(9)/w_t` |
| `skewed_peak`, `funnel` | full duration of the rise | — |
| `level_shift` | n/a | step at `c` |
| `sine_1`, `sine_2`, `damped_sine` | n/a | `position` is a phase shift in years |

Shapes outside the window are **clipped, never wrapped**. A peak centred at 1885 with
width 15 is a truncated peak, which is a legitimate class. Periodic boundaries would
make `@1885` and `@2020` near-identical.

### The instance model

```python
@dataclass(frozen=True)
class ShapeInstance:
    base: str                 # key from shapes.ALL_SHAPES
    position: int | None      # year; None for PositionKind.NONE
    width: int | None = None  # years; None uses the table default

    @property
    def key(self) -> str:     # stable id, e.g. "peak@1925w15"
    @property
    def label(self) -> str:   # Turkish UI label, e.g. "Tepe @1925"
```

`instance_prototypes(instances: list[ShapeInstance]) -> np.ndarray` returns
`(k, 146)`, z-normalized, reusing `shapes.zscore`.

Validation raising `ValueError` naming the offending instance:

- `position` in `[1880, 2025]`
- `width` between 2 and 146
- `position` is `None` exactly when the base shape is `PositionKind.NONE`
- no duplicate `key` in the list

### Separation warnings

```python
def separation_matrix(instances) -> np.ndarray            # (k, k) signed rho
def flag_pairs(instances, threshold: float = 0.7) -> list[tuple[str, str, float]]
def suggested_min_gap(base: str, width: int | None) -> int | None
```

`flag_pairs` returns pairs whose signed rho exceeds the threshold, sorted descending.
Difficulty statistics come from `shapes.difficulty_from_prototypes` (§1) — do not
reimplement the signed-rho logic.

`suggested_min_gap` finds the smallest whole-year separation bringing rho below 0.4,
computed numerically, **not hard-coded**. Returns `None` when no gap within the window
achieves it.

Two conventions the gap scan depends on, both normative:

- **Placement is centred.** For a candidate gap `g`, place the pair at
  `y1 = YEAR_MIN + (T_YEARS - 1 - g)//2` and `y2 = y1 + g`. Anchoring at a fixed year instead
  changes the answer for edge-sensitive shapes, because clipping at the window
  boundary alters the correlation. Centred placement also matches how quick-add
  spreads instances across the window.
- **Ascending scan, not bisection.** `rho(g)` is not monotone for the periodic
  shapes, so bisection can land on the wrong side. Scan `g` upward and return the
  first value with `rho < 0.4` (strict).

State both in the module docstring; neither is visible from the signature.

### Reference values

Signed rho after z-normalization. **Evaluate on `np.linspace(0, 1, 146)`**, not on
`(YEARS - 1880)/145` — the two differ by one ulp at a boundary grid point, which is
enough to move `cylinder` from −0.200 to −0.180. Route every conversion through
`year_to_t` so no caller rebuilds the grid a second way.

| Pair | rho |
|---|---|
| `peak@1935w15` vs `peak@1945w15` | +0.456 |
| `peak@1935w15` vs `peak@1950w15` | +0.113 |
| `peak@1935w15` vs `peak@1955w15` | −0.082 |
| `peak@1935w5` vs `peak@1945w5` | −0.050 |
| `peak@1935w30` vs `peak@1965w30` | −0.086 |
| `impulse@1930w6` vs `impulse@1940w6` | −0.046 |
| `cylinder@1930w50` vs `cylinder@1960w50` | +0.096 |
| `sigmoid@1930w20` vs `sigmoid@1960w20` | **+0.754** |
| `level_shift@1930` vs `level_shift@1960` | **+0.656** |
| `level_shift@1930` vs `level_shift@1980` | +0.489 |
| `peak@1935w15` vs `trough@1935w15` | −1.000 |

`suggested_min_gap` expected results:

| base | width | gap |
|---|---|---|
| `peak` | 5 | 4 |
| `peak` | 15 | 11 |
| `peak` | 30 | 19 |
| `impulse` | 6 | 3 |
| `cylinder` | 50 | 20 |
| `sigmoid` | 20 | 73 |
| `level_shift` | — | 63 |

The last two rows are the reason `flag_pairs` exists. Translation barely separates a
step or a broad sigmoid, because two of them agree everywhere except the window
between them. A 63-year gap allows at most two usable `level_shift` classes in a
146-year window. A UI that lets a user build three without warning produces an
unsolvable problem and makes the CVIs look broken when they are not.

## 3. New generator class

Add to `modules/experimental/synthetic_data_generator.py`, alongside
`BlobsSyntheticDataGenerator`:

```python
class TimeSeriesSyntheticDataGenerator(SyntheticDataGenerator):
    def generate(self) -> tuple[pd.DataFrame, np.ndarray]: ...
```

It must honour the existing ABC contract, because `BasePage.tab_clustering` depends
on it:

- returns `(df, ground_truth_labels)`
- sets `self.ground_truth_labels` and `self.n_features` before returning
- takes its parameters through the `kwargs` dict passed to `__init__`, matching how
  `BlobsSyntheticDataGenerator` is constructed in `experiment.py`

Expected `kwargs`: `instances` (list of `ShapeInstance`), plus the `GenConfig` fields
`n_per_cluster`, `sigma`, `znorm`, `amplitude_jitter`, `amp_range`, and `seed`.

The returned DataFrame must have columns equal to `YEARS` (integer year labels), so
downstream plotting and `tab_clustering_pca` — which does
`df_pivot.drop(columns=["clusters"])` — keep working. Build it from
`shapes.make_dataset_from_prototypes` with `GenConfig(T=146, ...)`.

While you are in this file, fix the ABC's return annotation: it declares
`-> pd.DataFrame` but every implementation returns a tuple.

### Interaction with the sweep

`BaseClustering.optimal_k_analysis` mutates the generator's kwargs per
`(seed, k)`: it sets `kwargs["random_state"] = seed` and `kwargs["centers"] = k`, then
regenerates. Leave `optimal_k_analysis` untouched and absorb this in the generator:

- **`random_state`, when present, overrides `seed`.** Each injected seed must produce
  a *new noise draw* over the same fixed prototypes. That variation is the
  experiment's repetition.
- **`centers` is ignored.** The number of ground-truth classes is fixed by the
  instance list, not by the candidate k. Regenerating per candidate k therefore yields
  identical data at every k, which is what this experiment requires.

## 4. UI helper

Add `render_time_series_synthetic_data()` to
`viz/gui_helpers/base_page_names/render_tabs_helpers.py`, next to
`render_synthetic_data()`. Same return style: a `kwargs` dict ready for the generator.
Turkish interface text. Matplotlib figures belong in a small new plotter module beside
`SyntheticDataPlotter`, not inline in the helper.

**Class builder**

- Add a class by picking a base shape from a `selectbox` grouped by `shapes.TIERS`
  (use `DISPLAY_NAMES` for labels), then a position year and width where applicable.
  Hide position and width controls entirely for `PositionKind.NONE` shapes.
- Show the composed classes as a list with a delete button per row.
- A "quick add" control: pick a base shape and a count `n`, insert `n` instances
  spread evenly across the window at or above `suggested_min_gap`. If the gap cannot
  be satisfied, insert nothing and show an error naming the constraint. Raise the same
  error for `NONE` shapes, which cannot be repeated.
- Persist the class list in `st.session_state`, keyed with the page name, following
  the `"<key>_" + page_name` convention. Prefer `SessionAdapter` / `PageKeys`.

**Generation settings**

`sigma` (0.05–1.0, default 0.3), `n_per_cluster` (5–100, default 20),
`amplitude_jitter` toggle default off with a caption noting it produces elongated
clusters, `seed` number input.

**Live diagnostics**

Recompute on every change: `k` and total series count; `rho_max`, `theta_min_deg`,
`ratio`, `verdict` from `shapes.difficulty_from_prototypes`; the `flag_pairs` list as
`st.warning`, phrased like `"Seviye kayması @1930 ile Seviye kayması @1960 çok benzer
(ρ=0.53) — bu iki sınıf ayrılamayabilir."`; and the separation matrix as a heatmap.

**Preview**

Overlay of the `k` prototypes on the year axis, one colour per class, legend from
`ShapeInstance.label`, x ticks on decades. Plus a sample of generated series (5 per
class) at the current `sigma`, alpha-blended, sharing the same axis.

**Export**

Download the class definition and config as JSON, with a matching upload control that
restores it. This is what makes a run reproducible — treat it as required.

## 5. Wire it into `experiment.py`

Currently `render()` sends anything that is not `tab_geo_clustering` to
`BlobsSyntheticDataGenerator`, so the `time_series` sub-tab silently produces blobs.
Branch on all three cases explicitly: `tab_geo_clustering`, `blobs`, `time_series`.

**Fix the tab key mismatch.** `render_tabs()` writes the *sub*-tab id (`"blobs"` /
`"time_series"`) into `st.session_state["selected_tab_" + page_name]`, but
`BasePage.tab_clustering` checks for `"tab_synthetic_clustering"` before calling
`SyntheticDataPlotter`, so that branch never fires. Keep the main-tab id in
`selected_tab` and put the sub-tab id in its own key — add a `selected_sub_tab`
property to `PageKeys`. `base_page.py` should then need no functional change, and the
geo path must keep working unaltered.

**Plotting on the `time_series` sub-tab.** `SyntheticDataPlotter` draws a 2-D scatter
of the first two feature columns, which for time series means year 1880 against year
1881 — meaningless axes. Use the new sub-tab key to skip that scatter for
`time_series` and draw instead an overlay of the generated series on the year axis,
coloured by predicted cluster. The PCA plot continues to run as normal. The `blobs`
sub-tab keeps the scatter.

**Guard the pre-existing crash.** `base_clustering` reads `data_generator.n_features`
whenever `"experiment"` is in `save_folder`, but `data_generator` is `None` on the
Experiment page's Names Data path, so the optimal-k sweep raises `AttributeError`
there. Add the guard, and the `os.makedirs` guard for the `np.save` target directory
on the same path.

## 6. Tests

Add `tests/test_shape_library.py`:

- `year_to_t` / `t_to_year` round-trip for 1880, 1950, 2025, and `year_to_t(2025) == 1.0`
- every reference rho in §2 reproduces within `atol=0.01`, and every
  `suggested_min_gap` value reproduces exactly
- validation raises on out-of-range years, duplicate keys, a position given for a
  `NONE` shape, and a position omitted for a positionable one
- clipping: `peak@1885w15` is truncated, not wrapped — assert the final years sit
  near the series minimum
- JSON round-trip of a class list reproduces identical prototypes
- `TimeSeriesSyntheticDataGenerator` returns a DataFrame whose columns equal `YEARS`,
  sets `ground_truth_labels`, gives an identical frame twice for one seed, and gives a
  *different* frame for two different injected `random_state` values while
  `ground_truth_labels` stays unchanged
- passing `centers` to the generator changes nothing

The existing `tests/test_shapes.py` must pass unmodified.

## Constraints

- `shape_library.py` imports only `numpy` and `shapes`. Streamlit stays in
  `viz/gui_helpers/`; pandas enters only in the generator.
- Identifiers, docstrings, comments in English. Turkish only in `DISPLAY_NAMES`,
  `ShapeInstance.label`, and UI strings.
- No global RNG — stochastic functions take an `np.random.Generator`.
- Do not add new CVI implementations or a new sweep runner; `clustering/` owns that.
