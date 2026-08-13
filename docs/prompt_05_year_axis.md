# Task: move the synthetic time axis to 1880–2025

The real SSA baby-name data now runs 1880–2025. The synthetic axis is still
1901–2000, so a generated dataset cannot be compared against a real one without
truncating one of them. Move the axis.

This is mechanically small — one constant block and one divisor — but it invalidates
every reference value in `docs/prompt_02_library_and_ui.md` §2 and several tests.
Corrected values are given below; they are normative, same as before. Implement to the
stated definitions and report any value that misses by more than `atol=0.01` rather
than adjusting the geometry.

Do this on its own, after the `arctan` fix is committed. Do not combine the two — if a
reference number fails, I need to know which change caused it.

## 1. The constants

In `modules/experimental/shape_library.py`:

```python
YEARS: np.ndarray = np.arange(1880, 2026)   # dtype=int, length 146
YEAR_MIN, YEAR_MAX = 1880, 2025
T_YEARS = 146
```

The grid is `np.linspace(0, 1, 146)`, so the step is `1/145`:

```
year_to_t(y) = (y - YEAR_MIN) / 145
w_t          = width_in_years / 145
```

The divisor is `YEAR_MAX - YEAR_MIN`, not `T_YEARS`. It was 99 for a 100-point grid
and is 145 for a 146-point grid. **Derive it from the constants rather than writing
either number as a literal** — a hardcoded 99 surviving somewhere is the most likely
failure mode of this change, and it will produce values that are close enough to look
plausible.

Validation range becomes `[1880, 2025]`. Width bounds become 2 to 146.

## 2. Widths stay in years

Do not rescale the per-shape width defaults. `peak` stays 15 years, `cylinder` stays
50, `impulse` stays 6. A class represents a real historical event — "names peaking in
the 1950s" — so a width fixed in years keeps its meaning, while a width fixed as a
fraction of the window would silently change what the class denotes when the window
changes again.

The width convention itself is unchanged: FWHM for `peak`/`trough`, full support for
`impulse`/`cylinder`, 10%–90% transition for `sigmoid`, inclusive bounds throughout.

## 3. Corrected reference values

Signed rho on the 1880–2025 grid after z-normalization. Replaces the table in
`prompt_02` §2:

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

`suggested_min_gap`, same centred placement and ascending scan as before:

| base | width | gap (old, 1901–2000) | gap (new) |
|---|---|---|---|
| `peak` | 5 | 4 | 4 |
| `peak` | 15 | 11 | 11 |
| `peak` | 30 | 17 | 19 |
| `impulse` | 6 | 4 | 3 |
| `cylinder` | 50 | 16 | 20 |
| `sigmoid` | 20 | 55 | **73** |
| `level_shift` | — | 43 | **63** |

Note the direction of travel for the transition shapes: a wider window makes them
*harder* to separate by position, not easier, because two steps now share even more of
the window in common. This strengthens the existing finding rather than changing it —
keep the `flag_pairs` warning behaviour exactly as it is.

## 4. Clipping test needs a new fixture

The old test used `peak@1905w15` to assert truncation rather than wrapping. 1905 is
now 25 years inside the window and no longer clipped. Use **`peak@1885w15`** — five
years from the window start. Assert the same property: the final years sit at the
series minimum, which they cannot do if the missing left half had been wrapped around.

## 5. Everything that follows from the constants

- `year_to_t(2025) == 1.0` exactly; add it alongside the existing round-trip cases,
  and update those cases to 1880, 1950, 2025.
- `instance_prototypes` returns `(k, 146)`.
- `TimeSeriesSyntheticDataGenerator` builds `GenConfig(T=146, ...)`, sets
  `self.n_features = 146`, and its DataFrame carries 146 integer year columns. The test
  asserting `columns == YEARS` should pass unchanged if it reads `YEARS` rather than a
  literal.
- The `separation_matrix` and `difficulty_from_prototypes` paths need no change —
  they take prototypes, not years.
- `shapes.GenConfig.T` keeps its generic default of 128. Only the caller changes.
- Any JSON fixture in the tests with 1901–2000 positions needs its positions checked
  against the new validation range — they remain valid, but a fixture asserting a
  specific prototype length will not.

## 6. Documentation

- `docs/prompt_02_library_and_ui.md` §2: the constants block, the `/99` → `/145`
  arithmetic, both reference tables, and the clipping test fixture.
- `CLAUDE.md`: the invariant currently reads "years 1901–2000, `T = 100`, grid step
  `1/99`". Update to 1880–2025, `T = 146`, step `1/145`, and the width convention line
  that quotes `w_t = W / 99`.

Run `git status` first and report anything uncommitted before editing.
