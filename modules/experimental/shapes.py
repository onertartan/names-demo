"""Time-series shape prototype library and synthetic dataset generator.

Sixteen z-normalized shape prototypes, grouped into tiers by geometric
complexity, used to build synthetic time-series clusters with a known
ground-truth label. This module covers the shape library, the dataset
generator, and a difficulty diagnostic; clustering, CVI computation, and
the Streamlit UI are added in later stages.
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np

TIERS: dict[str, list[str]] = {
    "monotone": ["linear_up", "linear_down", "exponential", "saturating"],
    "single_turn": ["sigmoid", "peak", "trough", "skewed_peak"],
    "piecewise": ["level_shift", "impulse", "cylinder", "funnel"],
    "oscillatory": ["sine_1", "sine_2", "damped_sine"],
    "composite": ["trend_seasonal"],
}

ALL_SHAPES: list[str] = [name for names in TIERS.values() for name in names]

DISPLAY_NAMES: dict[str, str] = {
    "linear_up": "Lineer artan",
    "linear_down": "Lineer azalan",
    "exponential": "Üstel artış",
    "saturating": "Doygun artış",
    "sigmoid": "Sigmoid eğrisi",
    "peak": "Tepe",
    "trough": "Çukur",
    "skewed_peak": "Çarpık tepe",
    "level_shift": "Seviye kayması",
    "impulse": "Darbe",
    "cylinder": "Silindir",
    "funnel": "Huni",
    "sine_1": "Sinüs (1 periyot)",
    "sine_2": "Sinüs (2 periyot)",
    "damped_sine": "Sönümlü sinüs",
    "trend_seasonal": "Trend + mevsimsellik",
}

# cylinder/funnel come from Cylinder-Bell-Funnel; linear_up/linear_down/level_shift/
# sine_1 overlap with Synthetic Control. Formulas are kept as-is so results stay
# comparable to those published benchmarks.
SHAPES: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "linear_up": lambda t: t,
    "linear_down": lambda t: -t,
    "exponential": lambda t: np.exp(3 * t),
    "saturating": lambda t: 1 - np.exp(-4 * t),
    "sigmoid": lambda t: 1 / (1 + np.exp(-10 * (t - 0.5))),
    "peak": lambda t: np.exp(-(((t - 0.5) / 0.15) ** 2)),
    "trough": lambda t: -np.exp(-(((t - 0.5) / 0.15) ** 2)),
    "skewed_peak": lambda t: t ** 2 * np.exp(-6 * t),
    "level_shift": lambda t: (t > 0.5).astype(float),
    "impulse": lambda t: ((t > 0.47) & (t < 0.53)).astype(float),
    "cylinder": lambda t: ((t > 0.25) & (t < 0.75)).astype(float),
    "funnel": lambda t: (t < 0.75) * np.clip(t - 0.25, 0, None) / 0.5,
    "sine_1": lambda t: np.sin(2 * np.pi * t),
    "sine_2": lambda t: np.sin(4 * np.pi * t),
    "damped_sine": lambda t: np.exp(-3 * t) * np.sin(6 * np.pi * t),
    "trend_seasonal": lambda t: 2 * t + 0.5 * np.sin(6 * np.pi * t),
}


def zscore(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Zero mean, unit standard deviation along axis.

    Divisor is clamped to 1.0 when std < 1e-12, so a constant series maps
    to an all-zero vector instead of NaN/inf.
    """
    mean = x.mean(axis=axis, keepdims=True)
    std = x.std(axis=axis, keepdims=True)
    std = np.where(std < 1e-12, 1.0, std)
    return (x - mean) / std


def prototypes(names: list[str], T: int = 128) -> np.ndarray:
    """Evaluate and z-normalize shape prototypes on t = linspace(0, 1, T).

    Returns an array of shape (len(names), T).
    """
    t = np.linspace(0, 1, T)
    raw = np.stack([SHAPES[name](t) for name in names])
    return zscore(raw, axis=-1)


@dataclass
class GenConfig:
    T: int = 128
    n_per_cluster: int = 20
    sigma: float = 0.3
    znorm: bool = True
    amplitude_jitter: bool = False
    amp_range: tuple[float, float] = (0.5, 2.0)


def make_dataset(
    names: list[str], cfg: GenConfig, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Draw a labeled synthetic time-series dataset from shape prototypes.

    Per sample: X = a * prototype + sigma * gauss, then, if cfg.znorm, X is
    z-normalized. The order -- amplitude, then noise, then normalization --
    is load-bearing and must not change.

    z-normalization mathematically cancels the amplitude factor, since
    z(a * phi) == z(phi) for a > 0. But because noise is added *before* that
    final normalization, a low-amplitude draw retains proportionally more
    noise than a high-amplitude one once everything is renormalized. So
    amplitude_jitter=True is not a no-op: it stretches each cluster into an
    elongated, ray-like shape instead of leaving it spherical.

    With amplitude_jitter=False (default), a=1 for every sample and clusters
    come out spherical. With True, a ~ Uniform(*cfg.amp_range) per sample.
    """
    protos = prototypes(names, cfg.T)
    k, n = len(names), cfg.n_per_cluster

    X = np.empty((k * n, cfg.T))
    y = np.empty(k * n, dtype=int)
    for i in range(k):
        sl = slice(i * n, (i + 1) * n)
        if cfg.amplitude_jitter:
            a = rng.uniform(cfg.amp_range[0], cfg.amp_range[1], size=(n, 1))
        else:
            a = 1.0
        gauss = rng.standard_normal((n, cfg.T))
        X[sl] = a * protos[i] + cfg.sigma * gauss
        y[sl] = i

    if cfg.znorm:
        X = zscore(X, axis=-1)
    return X, y


@dataclass
class Difficulty:
    rho_max: float
    theta_min_deg: float
    theta_spread_deg: float
    ratio: float
    closest_pair: tuple[str, str]
    verdict: str  # "kolay" / "orta" / "zor"


def difficulty(
    names: list[str], T: int = 128, sigma: float = 0.3, amp_min: float = 1.0
) -> Difficulty:
    """Diagnose how hard a shape pool is to cluster correctly.

    Finds the closest pair of prototypes in z-normalized space by the
    *signed* max correlation, not abs(rho). For z-normalized phi_i, phi_j:

        ||phi_i - phi_j||^2 = 2T(1 - rho)

    so rho = +1 is the closest pair (hardest to separate) and rho = -1 is
    the farthest apart (easiest). Using abs(rho) would instead flag an
    anti-correlated pair like peak/trough -- actually the easiest pair to
    separate -- as the hardest.

    theta_min_deg is the angle between the two closest prototype directions;
    theta_spread_deg is the angular deviation noise induces away from a
    prototype's own direction at amplitude amp_min. ratio = theta_min /
    theta_spread: >=3 the noise cones are well clear of each other ("kolay"),
    >=1.5 some overlap ("orta"), otherwise they likely overlap ("zor").
    """
    protos = prototypes(names, T)
    corr = (protos @ protos.T) / T
    np.fill_diagonal(corr, -np.inf)
    i, j = np.unravel_index(np.argmax(corr), corr.shape)
    rho_max = float(np.clip(corr[i, j], -1.0, 1.0))

    theta_min_deg = float(np.degrees(np.arccos(rho_max)))
    theta_spread_deg = float(np.degrees(sigma / amp_min))
    ratio = theta_min_deg / theta_spread_deg

    if ratio >= 3:
        verdict = "kolay"
    elif ratio >= 1.5:
        verdict = "orta"
    else:
        verdict = "zor"

    return Difficulty(
        rho_max=rho_max,
        theta_min_deg=theta_min_deg,
        theta_spread_deg=theta_spread_deg,
        ratio=ratio,
        closest_pair=(names[i], names[j]),
        verdict=verdict,
    )
