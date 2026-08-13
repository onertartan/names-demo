"""Headless batch driver for the CVI benchmark (kosum protokolu v5.3 + S-01..S-07).

Reads ``protocol/run_matrix_v4.csv`` row by row and runs the frozen
simulation: data generation per (cell, seed) via the pinned seed
architecture, four sklearn estimators with frozen parameters, the eight
CVI_REGISTRY indices, and the frozen k-hat / recording / QC rules.

Normative sources (verified by SHA256 at every start; the run refuses to
begin on any mismatch): the two protocol documents, the frozen manifest,
the two Block-D prototype files, and the four pinned code modules.

Interpretation note (written to provenance; user veto open): the
"driver calls TimeSeriesSyntheticDataGenerator + optimal_k_analysis"
sentence in v5.3's Manifest section is a descriptive v3-legacy line;
optimal_k_analysis does not implement the pinned seed architecture and is
Streamlit-bound. The normative code blocks are "Seed mimarisi" and
"Algoritma parametreleri"; this driver is a headless batch loop conforming
to those blocks verbatim and does not touch optimal_k_analysis or the repo
engines (they remain for the UI).

All console output is ASCII (the development console is cp1254).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

from clustering.evaluation.cvi_registry import CVI_REGISTRY
from modules.experimental import shapes
from modules.experimental.seed_scheme import CellSeeds, make_cell_seeds
from modules.experimental.shape_library import (
    POSITION_KINDS,
    T_YEARS,
    ShapeInstance,
    instance_prototypes,
)

# ---------------------------------------------------------------------------
# Pinned constants
# ---------------------------------------------------------------------------

PINNED_HASHES = {
    "protocol/run_matrix_v4.csv":
        "34e1217e1d36b7282311ca5e51ec25c2106ac43e75712ade15cfec446458fd64",
    "protocol/blokD_prototip_erkek_k5.npy":
        "3cdf4f26f0884c6f8d36c2c2d811d63522a3f35bc55dd91138796a4c3ccbd53e",
    "protocol/blokD_prototip_kadin_k5.npy":
        "16ebb091ba12d87cca93209f383226193e498e20e8d10c5db4843a3e9f706d81",
    "modules/experimental/shapes.py":
        "8375f6d4cd85bbe01758f7441208052e6120919b9035a606cfe7910f8bd04fc9",
    "modules/experimental/seed_scheme.py":
        "f9d2d66e4f0535224ed97ef08590c983b113b8cbefaadae2529eb69fe2219ef6",
    "modules/experimental/shape_library.py":
        "38f407dbd6b8b4710edf2ce42c568861f11ca664345a73431175c20d10c68c92",
    "modules/experimental/synthetic_data_generator.py":
        "22f79767045c25424dff57728ef3c9ccb86579457b728f59f2753cd9538ef9d9",
    "protocol/01_kosum_protokolu_v5_3.md":
        "cf8b453f0a05e2e6fed0c8b73692c2a5361fdfe60a22b3c9fe8c612e0e13db67",
    "protocol/kosum_protokolu_v5_3_sapma_eki_S01_S07_FINAL.md":
        "99c17c42711fd27dd2e55baf55f5ed41388b14a39a29e196f8eb1def34a5d0a7",
}

K_CANDIDATES = tuple(range(2, 11))
N_SEEDS = 100
ALGORITHMS = ("kmeans", "ward", "complete", "gmm")
TIE_RTOL, TIE_ATOL = 1e-10, 1e-12
AMP_RANGE = (0.5, 2.0)

# gurultu label <-> phi consistency (manifest is binding; a contradiction stops the run)
GURULTU_PHI = {"beyaz": None, "ar1_080": 0.8, "ar1_090": 0.9,
               "ar1_097": 0.97, "ar1_099": 0.99}

# Class-(i) mandatory generation fields (S-03a): empty on a row to be run -> stop.
CLASS_I_FIELDS = ("sigma", "gurultu", "siniflar", "n_per_cluster", "jitter", "aykiri_n")

_TOKEN_RE = re.compile(r"^([a-z_0-9]+?)(?:@(\d{4}))?(?:w(\d+))?$")


class ProtocolViolation(RuntimeError):
    """Fail-fast error: a binding protocol/manifest condition does not hold."""


# ---------------------------------------------------------------------------
# Hash verification (run start; refuse on mismatch)
# ---------------------------------------------------------------------------

def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_pinned_hashes(repo_root: Path) -> dict[str, str]:
    """Verify every pinned input; return the verified map for provenance."""
    verified, failures = {}, []
    for rel, expected in PINNED_HASHES.items():
        path = repo_root / rel
        if not path.exists():
            failures.append(f"MISSING {rel}")
            continue
        actual = sha256_of(path)
        verified[rel] = actual
        if actual != expected:
            failures.append(f"MISMATCH {rel}: {actual}")
    if failures:
        raise ProtocolViolation("pinned-input verification failed: "
                                + "; ".join(failures))
    return verified


# ---------------------------------------------------------------------------
# Manifest parsing + S-03 validation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CellSpec:
    blok: str
    hucre_id: int
    seed_key: int
    k_true: int
    sigma: float                     # nominal; binding only when sigma_vektoru is empty
    gurultu: str
    phi: float | None
    n_per_cluster: int               # nominal; binding only when kume_boyutlari is empty
    sizes: tuple[int, ...]           # resolved per-class sizes (binding)
    sigmas: tuple[float, ...]        # resolved per-class sigmas (binding)
    jitter: bool
    aykiri_n: int
    aykiri_sigma: float | None
    siniflar: str
    cinsiyet: str
    prototip_set_id: str
    prototip_hash: str


def _parse_cell(row: pd.Series) -> CellSpec:
    hucre = row["hucre_id"]
    for fieldname in CLASS_I_FIELDS:
        if row[fieldname] == "":
            raise ProtocolViolation(
                f"hucre {hucre}: class-(i) field '{fieldname}' is empty (S-03a)")
    gurultu = row["gurultu"]
    if gurultu not in GURULTU_PHI:
        raise ProtocolViolation(f"hucre {hucre}: unknown gurultu '{gurultu}'")
    expected_phi = GURULTU_PHI[gurultu]
    if expected_phi is None:
        if row["phi"] != "":
            raise ProtocolViolation(
                f"hucre {hucre}: phi must be NA on beyaz rows (S-03a class ii)")
        phi = None
    else:
        if row["phi"] == "":
            raise ProtocolViolation(f"hucre {hucre}: AR row with empty phi (S-03a)")
        phi = float(row["phi"])
        if phi != expected_phi:
            raise ProtocolViolation(
                f"hucre {hucre}: phi {phi} contradicts gurultu '{gurultu}'")

    k_true = int(row["k_true"])
    n_per = int(row["n_per_cluster"])
    sigma = float(row["sigma"])

    if row["kume_boyutlari"] != "":
        sizes = tuple(int(part) for part in row["kume_boyutlari"].split(";"))
        if len(sizes) != k_true:
            raise ProtocolViolation(
                f"hucre {hucre}: kume_boyutlari length {len(sizes)} != k_true {k_true}")
    else:
        sizes = (n_per,) * k_true
    if row["sigma_vektoru"] != "":
        sigmas = tuple(float(part) for part in row["sigma_vektoru"].split(";"))
        if len(sigmas) != k_true:
            raise ProtocolViolation(
                f"hucre {hucre}: sigma_vektoru length {len(sigmas)} != k_true {k_true}")
    else:
        sigmas = (sigma,) * k_true

    aykiri_n = int(row["aykiri_n"])
    if aykiri_n > 0:
        if row["aykiri_sigma"] == "":
            raise ProtocolViolation(
                f"hucre {hucre}: aykiri_n>0 but aykiri_sigma empty (S-03a class ii)")
        aykiri_sigma = float(row["aykiri_sigma"])
    else:
        aykiri_sigma = None

    return CellSpec(
        blok=row["blok"],
        hucre_id=int(hucre),
        seed_key=int(row["seed_key_hucre"]),
        k_true=k_true,
        sigma=sigma,
        gurultu=gurultu,
        phi=phi,
        n_per_cluster=n_per,
        sizes=sizes,
        sigmas=sigmas,
        jitter=bool(int(row["jitter"])),
        aykiri_n=aykiri_n,
        aykiri_sigma=aykiri_sigma,
        siniflar=row["siniflar"],
        cinsiyet=row["cinsiyet"],
        prototip_set_id=row["prototip_set_id"],
        prototip_hash=row["prototip_hash"],
    )


def parse_manifest(path: Path) -> list[CellSpec]:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if len(frame) != 631:
        raise ProtocolViolation(f"manifest has {len(frame)} rows, expected 631")
    return [_parse_cell(row) for _, row in frame.iterrows()]


# ---------------------------------------------------------------------------
# Prototypes
# ---------------------------------------------------------------------------

def token_to_instance(token: str) -> ShapeInstance:
    match = _TOKEN_RE.match(token)
    if not match or match.group(1) not in POSITION_KINDS:
        raise ProtocolViolation(f"unparseable class token '{token}'")
    base, pos, width = match.groups()
    instance = ShapeInstance(base, int(pos) if pos else None,
                             int(width) if width else None)
    if instance.key != token:
        # Round-trip assert: the manifest token must be the instance's own
        # stable key (any implicit-default drift fails fast).
        raise ProtocolViolation(
            f"token round-trip failed: '{token}' -> key '{instance.key}'")
    return instance


def build_prototypes(spec: CellSpec, protocol_dir: Path) -> np.ndarray:
    if spec.blok == "D_koprusu":
        path = protocol_dir / spec.siniflar
        actual = sha256_of(path)
        if actual != spec.prototip_hash:
            raise ProtocolViolation(
                f"hucre {spec.hucre_id}: prototype hash mismatch for {spec.siniflar}")
        protos = np.load(path)
        if protos.shape != (5, T_YEARS) or protos.dtype != np.float64:
            raise ProtocolViolation(
                f"hucre {spec.hucre_id}: prototype array must be (5, {T_YEARS}) "
                f"float64, got {protos.shape} {protos.dtype}")
        return protos  # rows used directly; NO re-z-normalization
    tokens = [token.strip() for token in spec.siniflar.split("|")]
    if len(tokens) != spec.k_true:
        raise ProtocolViolation(
            f"hucre {spec.hucre_id}: {len(tokens)} class tokens != k_true {spec.k_true}")
    return instance_prototypes([token_to_instance(token) for token in tokens])


# ---------------------------------------------------------------------------
# Data generation (frozen chain)
# ---------------------------------------------------------------------------

def generate_raw(protos: np.ndarray, sizes: tuple[int, ...],
                 sigmas: tuple[float, ...], jitter: bool, phi: float | None,
                 rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Class-major raw generation (amplitude -> noise; z-norm is the caller's
    LAST step, after any outlier replacement).

    Mirrors shapes.make_dataset_from_prototypes' class-major loop, extended
    to per-class n_i / sigma_i (the pinned heterogeneous rows). On
    homogeneous inputs it is bit-identical to the pinned function -- the
    same rng calls in the same order (regression-tested).
    """
    total = int(np.sum(sizes))
    n_timesteps = protos.shape[1]
    X = np.empty((total, n_timesteps), dtype=np.float64)
    y = np.empty(total, dtype=int)
    start = 0
    for class_index, (n_i, sigma_i) in enumerate(zip(sizes, sigmas)):
        sl = slice(start, start + n_i)
        if jitter:
            a = rng.uniform(AMP_RANGE[0], AMP_RANGE[1], size=(n_i, 1))
        else:
            a = 1.0
        if phi is None:
            noise = rng.standard_normal((n_i, n_timesteps))
        else:
            noise = shapes.ar1_noise(n_i, n_timesteps, phi, rng)
        X[sl] = a * protos[class_index] + sigma_i * noise
        y[sl] = class_index
        start += n_i
    return X, y


def generate_cell_data(spec: CellSpec, protos: np.ndarray,
                       rng: np.random.Generator
                       ) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    """Frozen order (provenance): base generation -> outlier index selection
    -> regeneration in ascending index order -> row z-norm (ddof=0)."""
    X_raw, y = generate_raw(protos, spec.sizes, spec.sigmas, spec.jitter,
                            spec.phi, rng)
    outlier_ids: list[int] = []
    outlier_classes: list[int] = []
    if spec.aykiri_n > 0:
        total = X_raw.shape[0]
        # Unstratified, without replacement, from all N indices; N stays
        # fixed (replacement of series, not addition).
        selected = np.sort(rng.choice(total, size=spec.aykiri_n, replace=False))
        for index in selected:  # ascending order, frozen
            class_index = int(y[index])
            if spec.phi is None:
                noise = rng.standard_normal((1, X_raw.shape[1]))
            else:
                noise = shapes.ar1_noise(1, X_raw.shape[1], spec.phi, rng)
            # Same prototype, same label, the cell's noise structure,
            # ABSOLUTE sigma_out (not a multiple of the class sigma); a=1.
            X_raw[index] = protos[class_index] + spec.aykiri_sigma * noise[0]
            outlier_ids.append(int(index))
            outlier_classes.append(class_index)
    X = shapes.zscore(X_raw, axis=-1)  # final transform, ddof=0
    return X, y, outlier_ids, outlier_classes


# ---------------------------------------------------------------------------
# Algorithms (frozen sklearn parameters; repo engines are NOT used)
# ---------------------------------------------------------------------------

def fit_algorithm(X: np.ndarray, algorithm: str, k: int,
                  seeds: CellSeeds) -> tuple[np.ndarray, bool | None]:
    """Returns (labels, gmm_converged_or_None). Exceptions propagate to the
    caller, which invalidates the candidate and logs (S-04, no retry)."""
    if algorithm == "kmeans":
        model = KMeans(n_clusters=k, n_init=50, init="k-means++", max_iter=300,
                       tol=1e-4, algorithm="lloyd", random_state=seeds.kmeans_seed)
        return model.fit_predict(X), None
    if algorithm == "ward":
        return AgglomerativeClustering(n_clusters=k, linkage="ward",
                                       metric="euclidean").fit_predict(X), None
    if algorithm == "complete":
        return AgglomerativeClustering(n_clusters=k, linkage="complete",
                                       metric="euclidean").fit_predict(X), None
    if algorithm == "gmm":
        model = GaussianMixture(n_components=k, covariance_type="diag",
                                reg_covar=1e-6, tol=1e-3, max_iter=500,
                                n_init=5, init_params="kmeans",
                                random_state=seeds.gmm_seed)
        labels = model.fit_predict(X)
        # Non-convergence: best lower-bound solution among n_init is
        # accepted, converged recorded, NO rerun (protocol + S-04).
        return labels, bool(model.converged_)
    raise ValueError(f"unknown algorithm {algorithm!r}")


# ---------------------------------------------------------------------------
# k-hat selection (frozen direction / tie / non-finite policy)
# ---------------------------------------------------------------------------

def select_k_hat(scores_by_k: dict[int, float], maximize: bool
                 ) -> tuple[int | None, int, int]:
    """Returns (k_hat_or_None, k_hat_tie, cvi_failure) over VALID candidates'
    scores. Non-finite scores drop only this CVI's candidate; no finite
    candidate left -> (None, 0, 1)."""
    finite = {k: s for k, s in scores_by_k.items() if np.isfinite(s)}
    if not finite:
        return None, 0, 1
    best = max(finite.values()) if maximize else min(finite.values())
    ties = [k for k, s in finite.items()
            if np.isclose(s, best, rtol=TIE_RTOL, atol=TIE_ATOL)]
    return min(ties), int(len(ties) > 1), 0


def corr_ari_spearman(scores_by_k: dict[int, float], ari_by_k: dict[int, float],
                      maximize: bool) -> float:
    """Direction-standardized Spearman(S(k), ARI(k)) over valid+finite
    candidates; < 3 candidates -> NaN (recorded as NA)."""
    ks = [k for k, s in scores_by_k.items() if np.isfinite(s)]
    if len(ks) < 3:
        return float("nan")
    ks.sort()
    standardized = [scores_by_k[k] if maximize else -scores_by_k[k] for k in ks]
    ari = [ari_by_k[k] for k in ks]
    result = spearmanr(standardized, ari)
    return float(result.correlation)


# ---------------------------------------------------------------------------
# Realized-geometry QC (S-05 formula pins)
# ---------------------------------------------------------------------------

def qc_for_seed(Xz: np.ndarray, y: np.ndarray, protos_nominal: np.ndarray
                ) -> dict:
    n_timesteps = Xz.shape[1]
    # 1. sigma_generator_deviation: pooled ddof=0 residual to the NOMINAL
    # generator prototype (generator fidelity only).
    residual_nominal = Xz - protos_nominal[y]
    sigma_generator_deviation = float(np.sqrt(np.mean(residual_nominal ** 2)))
    # 2. sigma_achieved: sample class prototypes from TRUE labels,
    # P_hat_c = z(mean z_i); pooled, observation-weighted, ddof=0 (S-05).
    classes = np.unique(y)
    prototypes_hat = np.vstack([
        shapes.zscore(Xz[y == c].mean(axis=0), axis=-1) for c in classes])
    residual_hat = Xz - prototypes_hat[np.searchsorted(classes, y)]
    sigma_achieved = float(np.sqrt(np.mean(residual_hat ** 2)))
    # 3. rho_max_achieved: SIGNED max Pearson between sample prototypes;
    # abs() is forbidden (S-05). Pair recorded 1-based in class order.
    best_rho, best_pair = -np.inf, ""
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            rho = float(prototypes_hat[i] @ prototypes_hat[j] / n_timesteps)
            if rho > best_rho:
                best_rho, best_pair = rho, f"{classes[i] + 1}-{classes[j] + 1}"
    return {"sigma_generator_deviation": sigma_generator_deviation,
            "sigma_achieved": sigma_achieved,
            "rho_max_achieved": best_rho,
            "rho_max_pair": best_pair}


# ---------------------------------------------------------------------------
# One (cell, seed) evaluation
# ---------------------------------------------------------------------------

def evaluate_cell_seed(spec: CellSpec, protos: np.ndarray, tohum: int,
                       records: dict[str, list], exceptions: list[dict]) -> None:
    seeds = make_cell_seeds(spec.seed_key, tohum)
    X, y_true, outlier_ids, outlier_classes = generate_cell_data(
        spec, protos, seeds.rng_data)

    qc = qc_for_seed(X, y_true, protos)
    records["dataqc"].append({
        "hucre_id": spec.hucre_id, "tohum": tohum, **qc,
        "outlier_series_id": ";".join(map(str, outlier_ids)),
        "outlier_class_id": ";".join(map(str, outlier_classes)),
    })

    for algorithm in ALGORITHMS:
        candidates: dict[int, np.ndarray] = {}
        converged_by_k: dict[int, bool | None] = {}
        for k in K_CANDIDATES:
            try:
                labels, converged = fit_algorithm(X, algorithm, k, seeds)
                candidates[k] = labels
                converged_by_k[k] = converged
            except Exception as err:  # fit exception -> invalid, log, no retry
                exceptions.append({
                    "hucre_id": spec.hucre_id, "algoritma": algorithm,
                    "tohum": tohum, "k": k, "stage": "fit",
                    "type": type(err).__name__, "message": str(err)[:500]})

        valid = {k: labels for k, labels in candidates.items()
                 if len(np.unique(labels)) == k}
        degenerate = sorted(set(candidates) - set(valid))
        algorithm_failure = int(len(valid) == 0)

        ari_by_k = {k: float(adjusted_rand_score(y_true, labels))
                    for k, labels in candidates.items()}
        for k in K_CANDIDATES:
            row = {"hucre_id": spec.hucre_id, "algoritma": algorithm,
                   "tohum": tohum, "k": k,
                   "fitted": int(k in candidates),
                   "valid": int(k in valid),
                   "ari": ari_by_k.get(k, float("nan"))}
            if algorithm == "gmm":
                row["converged"] = converged_by_k.get(k)
            records["candidates"].append(row)

        # CVI scores on VALID candidates only; a CVI exception drops only
        # that (CVI, k) (S-04) -- implemented as a NaN cell.
        scores: dict[str, dict[int, float]] = {key: {} for key in CVI_REGISTRY}
        for k, labels in valid.items():
            for key, cvi in CVI_REGISTRY.items():
                try:
                    scores[key][k] = float(cvi.fn(X, labels))
                except Exception as err:
                    scores[key][k] = float("nan")
                    exceptions.append({
                        "hucre_id": spec.hucre_id, "algoritma": algorithm,
                        "tohum": tohum, "k": k, "stage": f"cvi:{key}",
                        "type": type(err).__name__, "message": str(err)[:500]})

        # ari_at_ktrue: CVI-independent, mechanical ARI(y_true, y_hat at
        # k = k_true); NA if that fit failed or the algorithm failed wholly.
        if algorithm_failure or spec.k_true not in candidates:
            ari_at_ktrue = float("nan")
        else:
            ari_at_ktrue = ari_by_k[spec.k_true]

        records["alg"].append({
            "hucre_id": spec.hucre_id, "algoritma": algorithm, "tohum": tohum,
            "ari_at_ktrue": ari_at_ktrue,
            "degenerate_candidates": ";".join(map(str, degenerate)),
            "n_degenerate": len(degenerate),
            "algorithm_failure": algorithm_failure,
            "gmm_converged_at_ktrue": (converged_by_k.get(spec.k_true)
                                       if algorithm == "gmm" else None),
            "outlier_series_id": ";".join(map(str, outlier_ids)),
            "outlier_class_id": ";".join(map(str, outlier_classes)),
        })

        for key, cvi in CVI_REGISTRY.items():
            if algorithm_failure:
                k_hat, tie, cvi_failure = None, 0, 0
            else:
                k_hat, tie, cvi_failure = select_k_hat(scores[key], cvi.maximize)
            correct = int(k_hat == spec.k_true) if k_hat is not None else 0
            bias = (k_hat - spec.k_true) if k_hat is not None else float("nan")
            records["cvi"].append({
                "hucre_id": spec.hucre_id, "algoritma": algorithm,
                "indeks": key, "tohum": tohum,
                "k_hat": float(k_hat) if k_hat is not None else float("nan"),
                "correct": correct,
                "bias": float(bias) if k_hat is not None else float("nan"),
                "corr_ari": (corr_ari_spearman(scores[key], ari_by_k, cvi.maximize)
                             if not algorithm_failure else float("nan")),
                "k_hat_tie": tie,
                "cvi_failure": cvi_failure,
                "algorithm_failure": algorithm_failure,
            })


# ---------------------------------------------------------------------------
# Cell runner with per-cell checkpoint/resume
# ---------------------------------------------------------------------------

def run_cell(spec: CellSpec, protocol_dir: Path, out_dir: Path,
             n_seeds: int = N_SEEDS) -> float:
    cell_dir = out_dir / "cells"
    cell_dir.mkdir(parents=True, exist_ok=True)
    done_marker = cell_dir / f"{spec.hucre_id:04d}.done"
    if done_marker.exists():
        return 0.0
    start_time = time.perf_counter()
    protos = build_prototypes(spec, protocol_dir)
    records: dict[str, list] = {"cvi": [], "alg": [], "candidates": [],
                                "dataqc": []}
    exceptions: list[dict] = []
    for tohum in range(n_seeds):
        evaluate_cell_seed(spec, protos, tohum, records, exceptions)

    prefix = f"{spec.hucre_id:04d}"
    pd.DataFrame(records["cvi"]).to_parquet(cell_dir / f"{prefix}_cvi.parquet")
    pd.DataFrame(records["alg"]).to_parquet(cell_dir / f"{prefix}_alg.parquet")
    pd.DataFrame(records["candidates"]).to_parquet(
        cell_dir / f"{prefix}_candidates.parquet")
    pd.DataFrame(records["dataqc"]).to_parquet(
        cell_dir / f"{prefix}_dataqc.parquet")
    if exceptions:  # separate exception log (S-04)
        with open(cell_dir / f"{prefix}_exceptions.jsonl", "w",
                  encoding="utf-8") as handle:
            for entry in exceptions:
                handle.write(json.dumps(entry) + "\n")
    done_marker.touch()
    return time.perf_counter() - start_time


def summarize_cell_qc(out_dir: Path, cell_ids: list[int]) -> pd.DataFrame:
    """Cell-level QC: median + IQR over seeds (protocol Kayit)."""
    rows = []
    for hucre_id in cell_ids:
        path = out_dir / "cells" / f"{hucre_id:04d}_dataqc.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        row = {"hucre_id": hucre_id}
        for column in ("sigma_generator_deviation", "sigma_achieved",
                       "rho_max_achieved"):
            q1, med, q3 = frame[column].quantile([0.25, 0.5, 0.75])
            row[f"{column}_median"] = med
            row[f"{column}_iqr"] = q3 - q1
        row["rho_max_pair_mode"] = frame["rho_max_pair"].mode().iloc[0]
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Provenance + entry points
# ---------------------------------------------------------------------------

def write_provenance(out_dir: Path, verified: dict[str, str],
                     extra: dict | None = None) -> None:
    import scipy
    import sklearn
    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,  # metric= requires sklearn >= 1.2
        "verified_hashes": verified,
        "k_candidates": list(K_CANDIDATES),
        "n_seeds": N_SEEDS,
        "tie_policy": {"rtol": TIE_RTOL, "atol": TIE_ATOL, "rule": "min(T)"},
        "interpretation_note": (
            "v5_3 Manifest section's 'driver calls "
            "TimeSeriesSyntheticDataGenerator + optimal_k_analysis' sentence "
            "is v3-legacy descriptive text; the normative Seed mimarisi and "
            "Algoritma parametreleri code blocks are implemented verbatim in "
            "this headless driver. optimal_k_analysis and the repo engines "
            "are not used (they do not implement the pinned seed "
            "architecture and are Streamlit-bound)."),
        "outlier_execution_order": (
            "base generation -> index selection (rng_data, unstratified, "
            "without replacement) -> regeneration in ascending index order "
            "(same prototype, same label, cell noise structure, absolute "
            "sigma_out, a=1) -> row z-norm ddof=0"),
        "ari_at_ktrue_reading": (
            "computed whenever the k=k_true fit succeeded (degenerate "
            "included); NA on fit exception at k_true or algorithm_failure"),
        "label_convention": "class ids 0-based in siniflar token order",
    }
    if extra:
        payload.update(extra)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "provenance.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def run(manifest_path: Path, out_dir: Path, protocol_dir: Path,
        cell_filter: set[int] | None = None,
        block_filter: set[str] | None = None,
        n_seeds: int = N_SEEDS) -> None:
    repo_root = protocol_dir.parent
    verified = verify_pinned_hashes(repo_root)
    print("pinned-input verification: OK (9 files)")
    specs = parse_manifest(manifest_path)
    if cell_filter is not None:
        specs = [spec for spec in specs if spec.hucre_id in cell_filter]
    if block_filter is not None:
        specs = [spec for spec in specs if spec.blok in block_filter]
    if n_seeds != N_SEEDS:
        print(f"WARNING: n_seeds={n_seeds} deviates from the pinned {N_SEEDS}; "
              f"recorded in provenance")
    write_provenance(out_dir, verified,
                     extra={"n_seeds_effective": n_seeds,
                            "n_cells_selected": len(specs)})
    total = len(specs)
    for position, spec in enumerate(specs, 1):
        elapsed = run_cell(spec, protocol_dir, out_dir, n_seeds=n_seeds)
        status = "skipped (done)" if elapsed == 0.0 else f"{elapsed:8.1f}s"
        print(f"[{position:4d}/{total}] hucre {spec.hucre_id:4d} "
              f"({spec.blok:14s}) {status}")
    qc = summarize_cell_qc(out_dir, [spec.hucre_id for spec in specs])
    qc.to_csv(out_dir / "cell_qc_summary.csv", index=False)
    print(f"done: {total} cells; cell QC summary written")


def benchmark(manifest_path: Path, out_dir: Path, protocol_dir: Path) -> None:
    """Checklist item 11: one representative Block-A cell + the
    representative Block-B cell (B11, hucre 626), full 100 seeds each;
    full-run estimate follows. Full run only after user approval."""
    repo_root = protocol_dir.parent
    verified = verify_pinned_hashes(repo_root)
    print("pinned-input verification: OK (9 files)")
    specs = {spec.hucre_id: spec for spec in parse_manifest(manifest_path)}
    representative_a = specs[449]   # R cell: M_karisik k=5 r062 sigma=0.5 ar1_097
    representative_b = specs[626]   # B11: adverse combined stress
    write_provenance(out_dir, verified, extra={"mode": "benchmark"})
    timings = {}
    for spec in (representative_a, representative_b):
        elapsed = run_cell(spec, protocol_dir, out_dir)
        timings[spec.hucre_id] = elapsed
        print(f"benchmark hucre {spec.hucre_id}: {elapsed:.1f}s "
              f"({elapsed / N_SEEDS:.2f}s/seed)")
    per_cell = float(np.mean(list(timings.values())))
    estimate_hours = per_cell * 631 / 3600.0
    print(f"estimated full run, single worker: {per_cell:.1f}s/cell x 631 "
          f"= {estimate_hours:.1f}h")
    with open(out_dir / "benchmark.json", "w", encoding="utf-8") as handle:
        json.dump({"timings_s": timings, "per_cell_s": per_cell,
                   "single_worker_estimate_h": estimate_hours}, handle, indent=2)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="CVI benchmark driver (v5.3)")
    parser.add_argument("--manifest", default="protocol/run_matrix_v4.csv")
    parser.add_argument("--protocol-dir", default="protocol")
    parser.add_argument("--out", required=True,
                        help="output directory (no default by design)")
    parser.add_argument("--cells", default=None,
                        help="comma-separated hucre_id filter")
    parser.add_argument("--blocks", default=None,
                        help="comma-separated blok filter")
    parser.add_argument("--seeds", type=int, default=N_SEEDS)
    parser.add_argument("--benchmark", action="store_true",
                        help="run checklist item 11 instead of a full run")
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest)
    protocol_dir = Path(args.protocol_dir)
    out_dir = Path(args.out)
    if args.benchmark:
        benchmark(manifest_path, out_dir, protocol_dir)
        return
    cell_filter = ({int(part) for part in args.cells.split(",")}
                   if args.cells else None)
    block_filter = set(args.blocks.split(",")) if args.blocks else None
    run(manifest_path, out_dir, protocol_dir,
        cell_filter=cell_filter, block_filter=block_filter,
        n_seeds=args.seeds)


if __name__ == "__main__":
    main()
