# Task: `shapes.py` — time series prototype library and data generator

This will be integrated into a Streamlit project. In this stage, write **only**
the shape library, the data generator, and the difficulty diagnostic. Clustering,
CVI computation, and the UI come in later stages — leave room for them, but do
not write them.

## Context

The experiment measures which cluster validity index (silhouette,
Calinski–Harabasz, Davies–Bouldin, Dunn, Gap) correctly recovers the a priori
known number of clusters. Classes are time-aligned and shape-based. The shape
pool is tiered by geometric complexity.

## 1. Shape library

16 prototypes on `t = np.linspace(0, 1, T)`. Each is defined in raw form, then
z-normalized.

| Key | Formula | Tier |
|---|---|---|
| `linear_up` | `t` | monotone |
| `linear_down` | `-t` | monotone |
| `exponential` | `exp(3t)` | monotone |
| `saturating` | `1 - exp(-4t)` | monotone |
| `sigmoid` | `1 / (1 + exp(-10(t - 0.5)))` | single_turn |
| `peak` | `exp(-((t - 0.5)/0.15)^2)` | single_turn |
| `trough` | `-exp(-((t - 0.5)/0.15)^2)` | single_turn |
| `skewed_peak` | `t^2 * exp(-6t)` | single_turn |
| `level_shift` | `1[t > 0.5]` | piecewise |
| `impulse` | `1[0.47 < t < 0.53]` | piecewise |
| `cylinder` | `1[0.25 < t < 0.75]` | piecewise |
| `funnel` | `1[t < 0.75] * clip(t - 0.25, 0, None) / 0.5` | piecewise |
| `sine_1` | `sin(2πt)` | oscillatory |
| `sine_2` | `sin(4πt)` | oscillatory |
| `damped_sine` | `exp(-3t) * sin(6πt)` | oscillatory |
| `trend_seasonal` | `2t + 0.5 sin(6πt)` | composite |

Expose these as:

- `TIERS: dict[str, list[str]]` — tier name → shape keys, in the order above.
  The UI will group checkboxes by tier.
- `ALL_SHAPES: list[str]` — flattened `TIERS`.
- `DISPLAY_NAMES: dict[str, str]`  

Implement the formulas as a `dict[str, Callable[[np.ndarray], np.ndarray]]`, not
an `if/elif` chain — adding a shape should be one line.

`cylinder` and `funnel` come from Cylinder–Bell–Funnel; `linear_up`,
`linear_down`, `level_shift`, and `sine_1` overlap with Synthetic Control. That
overlap is deliberate so results stay comparable to published benchmarks — do
not alter these formulas.

## 2. Normalization

`zscore(x, axis=-1)` — zero mean, unit standard deviation. Clamp the divisor to
1.0 when the standard deviation is below `1e-12` (constant-series guard).

`prototypes(names: list[str], T: int = 128) -> np.ndarray` returns `(k, T)`,
z-normalized.

## 3. Data generator

```python
@dataclass
class GenConfig:
    T: int = 128
    n_per_cluster: int = 20
    sigma: float = 0.3
    znorm: bool = True
    amplitude_jitter: bool = False
    amp_range: tuple[float, float] = (0.5, 2.0)
```

`make_dataset(names, cfg, rng) -> tuple[np.ndarray, np.ndarray]` where `X` has
shape `(k * n_per_cluster, T)` and `y` holds the ground-truth labels.

The order of operations is load-bearing: **amplitude, then noise, then
z-normalization.**

```
X = a * prototype + sigma * gauss
if znorm: X = zscore(X)
```

With `amplitude_jitter=False`, `a = 1` — this is the default path and yields
spherical clusters. With `True`, `a ~ Uniform(amp_range)`.

Rationale, which belongs in the docstring: z-normalization mathematically
removes the amplitude factor, since `z(a·φ) = z(φ)`. But because noise is added
*after* the amplitude scaling, low-amplitude series retain proportionally more
noise. So enabling jitter produces elongated (ray-shaped) clusters rather than
no effect at all. Do not reorder these steps.

## 4. Difficulty diagnostic

```python
@dataclass
class Difficulty:
    rho_max: float
    theta_min_deg: float
    theta_spread_deg: float
    ratio: float
    closest_pair: tuple[str, str]
    verdict: str  # "kolay" / "orta" / "zor"
```

`difficulty(names, T=128, sigma=0.3, amp_min=1.0) -> Difficulty`

**There is an easy mistake here — read this before implementing.** When finding
the closest pair from the correlation matrix, take the maximum of the **signed**
rho, not `abs(rho)`. The reason: in z-normalized space,

```
||phi_i - phi_j||^2 = 2T(1 - rho)
```

so `rho = +1` means closest (hard to separate) and `rho = -1` means farthest
(easy). Using `abs()` would flag the `peak`/`trough` pair — which is in fact the
easiest pair to separate — as the hardest. Fill the diagonal with `-inf` and take
`argmax`.

Remaining quantities:

- `theta_min_deg = degrees(arccos(rho_max))`
- `theta_spread_deg = degrees(sigma / amp_min)` — the angular deviation noise
  induces away from the prototype direction
- `ratio = theta_min_deg / theta_spread_deg`
- `verdict`: `ratio >= 3` → `"kolay"`, `>= 1.5` → `"orta"`, else `"zor"`

Expected output, for verification:

| Pool | closest_pair | rho_max | theta_min | ratio (σ=0.3) |
|---|---|---|---|---|
| all 16 shapes | `linear_up` / `sigmoid` | +0.973 | 13.2° | 0.77 |
| `[peak, trough]` | `peak` / `trough` | −1.000 | 180.0° | 10.47 |
| oscillatory tier | `sine_2` / `damped_sine` | +0.141 | 81.9° | 4.76 |

If your numbers disagree, the formulas or the sign handling are wrong.

## 5. Tests

Write `tests/test_shapes.py` for pytest:

1. Every prototype has `mean ≈ 0` and `std ≈ 1` (atol=1e-10).
2. `abs(rho) <= 1` everywhere; the correlation diagonal is exactly 1.
3. `peak` and `trough` correlate at ≈ −1.
4. `difficulty(ALL_SHAPES).closest_pair` equals `{linear_up, sigmoid}` as a set,
   and `rho_max` is within 0.01 of 0.973.
5. `difficulty(["peak", "trough"]).verdict == "kolay"` — regression test for the
   sign error described above.
6. `make_dataset` returns the expected shapes; with `znorm=True` every row has
   `mean ≈ 0` and `std ≈ 1`.
7. The same seed produces an identical `X` twice.

## Constraints

- `numpy` is the only dependency. Keep `sklearn` out of this file.
- Type hints throughout; use `from __future__ import annotations`.
- No global RNG — every stochastic function takes an `np.random.Generator`.
- Document the design rationale in docstrings, especially the signed rho and the
  amplitude/noise ordering. Both look like bugs to a future reader and are
  likely to be "fixed" otherwise.

## Later stages (do not write yet)

2. `metrics.py` — the CVIs (Dunn and Gap implemented by hand; Gap must support
   `k=1`) plus the single-trial and sweep runners.
3. `app.py` — tiered checkboxes, live difficulty readout, result plots.
