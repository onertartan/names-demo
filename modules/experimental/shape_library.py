"""Time-positioned shape classes on the fixed 1901-2000 year axis.

Positions base shapes from ``shapes.SHAPES`` at explicit years, making the
pair (base shape, time position) the ground-truth class identity. Two peaks
centred at different years are deliberately *different* classes; the
experiment treats misalignment as a real difference, which is also why only
Euclidean-family distances are valid downstream (no DTW anywhere).

Geometry conventions (normative; see docs/prompt_02_library_and_ui.md §2):

- Years 1901-2000, ``T_YEARS = 100`` grid points, normalized step ``1/99``.
  All year<->t conversion goes through ``year_to_t`` / ``t_to_year``. The
  evaluation grid itself is ``np.linspace(0, 1, 100)``: mathematically equal
  to ``year_to_t(YEARS)`` but not bit-identical, and the reference
  correlations pin the linspace values (final-ulp differences change which
  grid points fall inside the rectangular shapes' closed supports).
- ``width`` is in years; ``w_t = width / 99``. Meaning per shape: FWHM for
  ``peak``/``trough``; full support for ``impulse``/``cylinder``; the
  10%-90% transition duration for ``sigmoid``; the full duration of the
  rise for ``skewed_peak``/``funnel``.
- Boundary comparisons of the positioned piecewise shapes are *inclusive*:
  ``abs(t - c) <= w_t / 2`` for ``impulse``/``cylinder``, ``t >= c`` for
  ``level_shift``. This deliberately differs from the base lambdas in
  ``shapes.py``, which use strict ``>`` / ``<`` and are pinned by
  ``tests/test_shapes.py``; the positioned variants are a separate
  parameterization, not a drift to "fix".
- Shapes are clipped at the window edges, never wrapped. A peak centred at
  1905 is a truncated peak, which is a legitimate class.
- ``suggested_min_gap`` places its trial pair symmetrically about the
  window centre, ``y1 = 1901 + (99 - g)//2`` and ``y2 = y1 + g``. Placement
  is part of the definition: anchoring at a fixed year instead changes the
  answer for edge-sensitive shapes (sigmoid, cylinder, level_shift),
  because clipping at the window boundary alters the correlation.
- The gap scan walks g upward and returns the first value with strict
  ``rho < 0.4``. ``rho(g)`` is not monotone for the periodic shapes, so
  bisection could land on the wrong side.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum

import numpy as np

from modules.experimental import shapes

YEARS: np.ndarray = np.arange(1901, 2001)   # dtype=int, length 100
YEAR_MIN, YEAR_MAX = 1901, 2000
T_YEARS = 100

_SPAN = YEAR_MAX - YEAR_MIN                 # 99 -- the grid step denominator
_T_GRID = np.linspace(0.0, 1.0, T_YEARS)
_GAP_RHO_TARGET = 0.4


def year_to_t(year: int | float | np.ndarray) -> float | np.ndarray:
    """Convert years (scalar or array) to normalized t; the step is 1/99."""
    t = (np.asarray(year, dtype=float) - YEAR_MIN) / _SPAN
    return float(t) if np.ndim(t) == 0 else t


def t_to_year(t: float | np.ndarray) -> float | np.ndarray:
    """Inverse of ``year_to_t``; returns float years."""
    year = np.asarray(t, dtype=float) * _SPAN + YEAR_MIN
    return float(year) if np.ndim(year) == 0 else year


class PositionKind(str, Enum):
    CENTER = "center"   # event symmetric about a year
    ONSET = "onset"     # something begins at a year
    PHASE = "phase"     # periodic; position is a phase shift in years
    NONE = "none"       # global shape, repositioning is meaningless


POSITION_KINDS: dict[str, PositionKind] = {
    "peak": PositionKind.CENTER,
    "trough": PositionKind.CENTER,
    "impulse": PositionKind.CENTER,
    "cylinder": PositionKind.CENTER,
    "skewed_peak": PositionKind.ONSET,
    "level_shift": PositionKind.ONSET,
    "sigmoid": PositionKind.ONSET,
    "funnel": PositionKind.ONSET,
    "sine_1": PositionKind.PHASE,
    "sine_2": PositionKind.PHASE,
    "damped_sine": PositionKind.PHASE,
    "linear_up": PositionKind.NONE,
    "linear_down": PositionKind.NONE,
    "exponential": PositionKind.NONE,
    "saturating": PositionKind.NONE,
    "trend_seasonal": PositionKind.NONE,
}

# Shapes with a width parameter and its default in years. Shapes absent here
# (level_shift, the sines, and the NONE kinds) take no width at all.
WIDTH_DEFAULTS: dict[str, int] = {
    "peak": 15,
    "trough": 15,
    "impulse": 6,
    "cylinder": 50,
    "skewed_peak": 30,
    "sigmoid": 20,
    "funnel": 50,
}

WIDTH_MIN, WIDTH_MAX = 2, 100


@dataclass(frozen=True)
class ShapeInstance:
    """One ground-truth class: a base shape at a time position.

    ``position`` is a year, ``None`` exactly for ``PositionKind.NONE``
    shapes. ``width`` is in years; ``None`` means the default from
    ``WIDTH_DEFAULTS`` (and stays ``None`` for shapes without a width).
    """
    base: str
    position: int | None
    width: int | None = None

    @property
    def kind(self) -> PositionKind:
        return POSITION_KINDS[self.base]

    @property
    def effective_width(self) -> int | None:
        """Width in years with the default applied; None for widthless shapes."""
        if self.width is not None:
            return self.width
        return WIDTH_DEFAULTS.get(self.base)

    @property
    def key(self) -> str:
        """Stable id, e.g. ``"peak@1925w15"`` (default width made explicit)."""
        key = self.base
        if self.position is not None:
            key += f"@{self.position}"
        if self.effective_width is not None:
            key += f"w{self.effective_width}"
        return key

    @property
    def label(self) -> str:
        """Turkish UI label, e.g. ``"Tepe @1925"``."""
        name = shapes.DISPLAY_NAMES.get(self.base, self.base)
        if self.position is None:
            return name
        return f"{name} @{self.position}"


def _describe(inst: ShapeInstance) -> str:
    return (f"ShapeInstance(base={inst.base!r}, position={inst.position}, "
            f"width={inst.width})")


def validate_instances(instances: list[ShapeInstance]) -> None:
    """Validate a class list, raising ValueError naming the offending instance."""
    seen: set[str] = set()
    for inst in instances:
        kind = POSITION_KINDS.get(inst.base)
        if kind is None:
            raise ValueError(f"{_describe(inst)}: unknown base shape {inst.base!r}")
        if kind is PositionKind.NONE:
            if inst.position is not None:
                raise ValueError(f"{_describe(inst)}: {inst.base!r} is a global "
                                 f"shape; position must be None")
        else:
            if inst.position is None:
                raise ValueError(f"{_describe(inst)}: {inst.base!r} needs a "
                                 f"position year")
            if not YEAR_MIN <= inst.position <= YEAR_MAX:
                raise ValueError(f"{_describe(inst)}: position must be in "
                                 f"[{YEAR_MIN}, {YEAR_MAX}]")
        if inst.base in WIDTH_DEFAULTS:
            if not WIDTH_MIN <= inst.effective_width <= WIDTH_MAX:
                raise ValueError(f"{_describe(inst)}: width must be in "
                                 f"[{WIDTH_MIN}, {WIDTH_MAX}]")
        elif inst.width is not None:
            raise ValueError(f"{_describe(inst)}: {inst.base!r} takes no width "
                             f"parameter")
        if inst.key in seen:
            raise ValueError(f"{_describe(inst)}: duplicate key {inst.key!r}")
        seen.add(inst.key)


def _positioned_series(base: str, t: np.ndarray, c: float,
                       w_t: float | None) -> np.ndarray:
    """Evaluate one positioned shape on the normalized grid (raw values).

    Boundary comparisons of the piecewise shapes are inclusive; see the
    module docstring. Values outside [0, 1] never wrap: the grid simply
    stops at the window edges, truncating whatever falls outside.
    """
    if base in ("peak", "trough"):
        half_width = w_t / (2.0 * np.sqrt(np.log(2.0)))   # FWHM convention
        series = np.exp(-(((t - c) / half_width) ** 2))
        return series if base == "peak" else -series
    if base in ("impulse", "cylinder"):
        return (np.abs(t - c) <= w_t / 2.0).astype(float)
    if base == "sigmoid":
        steepness = 2.0 * np.log(9.0) / w_t               # 10%-90% in w_t
        return 1.0 / (1.0 + np.exp(-steepness * (t - c)))
    if base == "level_shift":
        return (t >= c).astype(float)
    if base == "skewed_peak":
        # The base form u^2 exp(-6u) peaks at u = 1/3, so u = (t - c)/(3 w_t)
        # makes the rise span exactly [c, c + w_t].
        u = (t - c) / (3.0 * w_t)
        return np.where(u >= 0.0, u ** 2 * np.exp(-6.0 * u), 0.0)
    if base == "funnel":
        # Linear ramp 0 -> 1 over [c, c + w_t], dropping to 0 at the end;
        # the drop keeps the base lambda's strict `<`.
        return (t < c + w_t) * np.clip(t - c, 0.0, None) / w_t
    if base == "sine_1":
        return np.sin(2.0 * np.pi * (t - c))
    if base == "sine_2":
        return np.sin(4.0 * np.pi * (t - c))
    if base == "damped_sine":
        u = t - c
        return np.where(u >= 0.0, np.exp(-3.0 * u) * np.sin(6.0 * np.pi * u), 0.0)
    raise ValueError(f"shape {base!r} has no positioned form")


def instance_prototypes(instances: list[ShapeInstance]) -> np.ndarray:
    """Evaluate and z-normalize instances on the year grid; returns (k, 100).

    Validates the list first. ``PositionKind.NONE`` shapes evaluate their
    base lambda from ``shapes.SHAPES`` directly.
    """
    validate_instances(instances)
    raw = np.empty((len(instances), T_YEARS))
    for row, inst in enumerate(instances):
        if inst.kind is PositionKind.NONE:
            raw[row] = shapes.SHAPES[inst.base](_T_GRID)
        else:
            width = inst.effective_width
            w_t = None if width is None else width / _SPAN
            raw[row] = _positioned_series(inst.base, _T_GRID,
                                          year_to_t(inst.position), w_t)
    return shapes.zscore(raw, axis=-1)


def separation_matrix(instances: list[ShapeInstance]) -> np.ndarray:
    """Signed correlation rho between all instance prototypes; shape (k, k)."""
    protos = instance_prototypes(instances)
    corr = (protos @ protos.T) / T_YEARS
    return np.clip(corr, -1.0, 1.0)


def flag_pairs(instances: list[ShapeInstance],
               threshold: float = 0.7) -> list[tuple[str, str, float]]:
    """Pairs whose signed rho exceeds ``threshold``, sorted descending.

    Returns ``(key_i, key_j, rho)`` triples; the UI maps keys to labels.
    """
    corr = separation_matrix(instances)
    pairs = [
        (instances[i].key, instances[j].key, float(corr[i, j]))
        for i in range(len(instances))
        for j in range(i + 1, len(instances))
        if corr[i, j] > threshold
    ]
    return sorted(pairs, key=lambda pair: pair[2], reverse=True)


def suggested_min_gap(base: str, width: int | None = None) -> int | None:
    """Smallest whole-year gap g with rho(base@y, base@y+g) < 0.4, or None.

    The pair is placed symmetrically about the window centre so the result
    reflects translation alone rather than edge clipping. rho(g) is not
    monotone for the periodic shapes, so this scans g ascending instead of
    bisecting. ``PositionKind.NONE`` shapes cannot be repositioned and
    return None.
    """
    kind = POSITION_KINDS.get(base)
    if kind is None:
        raise ValueError(f"unknown base shape {base!r}")
    if kind is PositionKind.NONE:
        return None
    if base not in WIDTH_DEFAULTS:
        width = None
    elif width is None:
        width = WIDTH_DEFAULTS[base]
    for gap in range(1, T_YEARS):
        year_1 = YEAR_MIN + (T_YEARS - 1 - gap) // 2
        pair = [ShapeInstance(base, year_1, width),
                ShapeInstance(base, year_1 + gap, width)]
        if separation_matrix(pair)[0, 1] < _GAP_RHO_TARGET:
            return gap
    return None


def instances_to_json(instances: list[ShapeInstance],
                      config: dict | None = None) -> str:
    """Serialize a class list plus generation config for export/reload."""
    validate_instances(instances)
    payload = {
        "version": 1,
        "instances": [
            {"base": inst.base, "position": inst.position, "width": inst.width}
            for inst in instances
        ],
        "config": dict(config) if config else {},
    }
    return json.dumps(payload, indent=2)


def instances_from_json(text: str) -> tuple[list[ShapeInstance], dict]:
    """Inverse of ``instances_to_json``; validates the restored class list."""
    payload = json.loads(text)
    instances = [
        ShapeInstance(item["base"], item.get("position"), item.get("width"))
        for item in payload.get("instances", [])
    ]
    validate_instances(instances)
    return instances, payload.get("config", {})
