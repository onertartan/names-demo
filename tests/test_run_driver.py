"""Mandatory pre-run tests for the CVI benchmark driver (checklist 4-10 +
manifest/prototype/data-sharing/heterogeneity items). Checklist item 11
(runtime benchmark) is a driver mode, not a unit test.
"""
import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import davies_bouldin_score, silhouette_samples

import modules.experimental.run_driver as run_driver
from clustering.evaluation.cvi_registry import CVI_REGISTRY, dunn_index, generalized_dunn
from modules.experimental import shapes
from modules.experimental.run_driver import (
    K_CANDIDATES,
    ProtocolViolation,
    build_prototypes,
    corr_ari_spearman,
    evaluate_cell_seed,
    generate_cell_data,
    generate_raw,
    parse_manifest,
    qc_for_seed,
    run_cell,
    select_k_hat,
    token_to_instance,
    verify_pinned_hashes,
)
from modules.experimental.seed_scheme import make_cell_seeds
from modules.experimental.shape_library import T_YEARS

REPO_ROOT = Path(__file__).resolve().parent.parent
PROTOCOL_DIR = REPO_ROOT / "protocol"
MANIFEST = PROTOCOL_DIR / "run_matrix_v4.csv"


@pytest.fixture(scope="module")
def specs():
    return {spec.hucre_id: spec for spec in parse_manifest(MANIFEST)}


# ---------------------------------------------------------------------------
# Pinned-input verification
# ---------------------------------------------------------------------------

def test_pinned_hashes_verify_against_repo():
    verified = verify_pinned_hashes(REPO_ROOT)
    assert set(verified) == set(run_driver.PINNED_HASHES)


def test_hash_mismatch_refuses_to_start(monkeypatch):
    tampered = dict(run_driver.PINNED_HASHES)
    tampered["modules/experimental/shapes.py"] = "0" * 64
    monkeypatch.setattr(run_driver, "PINNED_HASHES", tampered)
    with pytest.raises(ProtocolViolation, match="MISMATCH"):
        verify_pinned_hashes(REPO_ROOT)


# ---------------------------------------------------------------------------
# Manifest parse round-trip + S-03 fail-fast
# ---------------------------------------------------------------------------

def test_manifest_parse_roundtrip(specs):
    assert len(specs) == 631
    blocks = pd.Series([spec.blok for spec in specs.values()]).value_counts()
    assert blocks["A_harita"] == 600 and blocks["B_duyarlilik"] == 15
    assert blocks["C_sabit_ratio"] == 12 and blocks["D_koprusu"] == 4
    # B0 exception: hucre 612 carries R's seed key
    assert specs[612].seed_key == 449 and specs[612].hucre_id == 612
    assert all(spec.seed_key == spec.hucre_id
               for spec in specs.values() if spec.hucre_id != 612)
    # heterogeneous rows resolve binding vectors
    assert specs[620].sigmas == (0.5, 0.35, 0.5, 0.7, 0.5)
    assert specs[626].sizes == (10, 15, 10, 5, 10)
    assert specs[626].aykiri_n == 3 and specs[626].aykiri_sigma == 1.5
    # homogeneous rows expand nominal scalars
    assert specs[449].sizes == (10,) * 5 and specs[449].sigmas == (0.5,) * 5
    assert specs[627].blok == "D_koprusu" and specs[627].prototip_hash


def _row_as_series(hucre_id: int, **overrides) -> pd.Series:
    frame = pd.read_csv(MANIFEST, dtype=str, keep_default_na=False)
    row = frame[frame["hucre_id"] == str(hucre_id)].iloc[0].copy()
    for key, value in overrides.items():
        row[key] = value
    return row


def test_class_i_empty_field_stops(specs):
    with pytest.raises(ProtocolViolation, match="sigma"):
        run_driver._parse_cell(_row_as_series(449, sigma=""))
    with pytest.raises(ProtocolViolation, match="phi"):
        run_driver._parse_cell(_row_as_series(449, phi=""))


def test_gurultu_phi_contradictions_stop(specs):
    with pytest.raises(ProtocolViolation, match="contradicts"):
        run_driver._parse_cell(_row_as_series(449, phi="0.8"))
    with pytest.raises(ProtocolViolation, match="beyaz"):
        run_driver._parse_cell(_row_as_series(0, phi="0.97"))


# ---------------------------------------------------------------------------
# Class tokens and Block-D prototypes
# ---------------------------------------------------------------------------

def test_token_round_trip():
    for token in ("peak@1944w15", "level_shift@1955", "trough@1975w15"):
        assert token_to_instance(token).key == token
    with pytest.raises(ProtocolViolation, match="round-trip"):
        token_to_instance("peak@1944")  # implicit default width must not pass
    with pytest.raises(ProtocolViolation):
        token_to_instance("not a token")


def test_block_d_prototypes_hash_shape_dtype(specs):
    protos = build_prototypes(specs[627], PROTOCOL_DIR)
    assert protos.shape == (5, T_YEARS) and protos.dtype == np.float64
    tampered = dataclasses.replace(specs[627], prototip_hash="0" * 64)
    with pytest.raises(ProtocolViolation, match="hash mismatch"):
        build_prototypes(tampered, PROTOCOL_DIR)


# ---------------------------------------------------------------------------
# Checklist 5 -- AR(1): stationary init, unit variance
# ---------------------------------------------------------------------------

def test_ar1_stationary_unit_variance():
    rng = np.random.default_rng(0)
    x = shapes.ar1_noise(20000, 20, 0.97, rng)
    # Var(x_0) ~ 1 rules out the forbidden scaled-innovation init
    # (which would give 1 - phi^2 = 0.06) and x_0 = 0.
    assert np.var(x[:, 0]) == pytest.approx(1.0, abs=0.05)
    for t in (1, 5, 19):
        assert np.var(x[:, t]) == pytest.approx(1.0, abs=0.05)
    lag1 = np.corrcoef(x[:, :-1].ravel(), x[:, 1:].ravel())[0, 1]
    assert lag1 == pytest.approx(0.97, abs=0.01)
    with pytest.raises(ValueError):
        shapes.ar1_noise(2, 5, 1.0, rng)


# ---------------------------------------------------------------------------
# Checklist 6 -- seed architecture
# ---------------------------------------------------------------------------

def _generate(spec):
    protos = build_prototypes(spec, PROTOCOL_DIR)
    seeds = make_cell_seeds(spec.seed_key, 0)
    return generate_cell_data(spec, protos, seeds.rng_data)


def test_seed_architecture_identity_and_separation(specs):
    x_first, y_first, _, _ = _generate(specs[449])
    x_second, y_second, _, _ = _generate(specs[449])
    assert np.array_equal(x_first, x_second)          # same (seed_key, tohum)
    assert np.array_equal(y_first, y_second)
    x_other, _, _, _ = _generate(specs[450])
    assert not np.array_equal(x_first, x_other)        # different cell -> new stream


def test_b0_reproduces_a449_bitwise(specs):
    # Free integrity check: hucre 612 (seed_key 449) must regenerate the
    # A-449 realizations exactly; anything else is a pipeline error.
    x_a, y_a, _, _ = _generate(specs[449])
    x_b, y_b, _, _ = _generate(specs[612])
    assert np.array_equal(x_a, x_b)
    assert np.array_equal(y_a, y_b)


# ---------------------------------------------------------------------------
# Checklist 7 -- z-norm ddof=0: ||x||^2 = T exactly
# ---------------------------------------------------------------------------

def test_znorm_rows_have_norm_squared_T(specs):
    x, _, _, _ = _generate(specs[449])
    norms_sq = np.sum(x ** 2, axis=1)
    assert np.allclose(norms_sq, T_YEARS, rtol=1e-10, atol=1e-7)


# ---------------------------------------------------------------------------
# Checklist 4 -- registry singleton / degenerate conventions
# ---------------------------------------------------------------------------

def test_singleton_conventions():
    X = np.array([[0.0], [0.1], [10.0]])
    labels = np.array([0, 0, 1])
    # silhouette: singleton observation s(i) = 0 (Rousseeuw convention)
    samples = silhouette_samples(X, labels)
    assert samples[2] == 0.0
    assert np.isfinite(CVI_REGISTRY["Silhouette Score (euclidean)"].fn(X, labels))
    # Dunn: singleton diameter 0 in both D1 and D2
    assert dunn_index(X, labels) == pytest.approx(9.9 / 0.1, rel=1e-9)
    d3_sep = (9.9 + 10.0) / 2.0
    assert generalized_dunn(X, labels, d="d3", D="D2") == pytest.approx(
        d3_sep / 0.1, rel=1e-9)
    # Davies-Bouldin: singleton S_i = 0
    s0, s1 = 0.05, 0.0
    centroid_distance = 9.95
    expected_db = ((s0 + s1) / centroid_distance)  # same max ratio both clusters
    assert davies_bouldin_score(X, labels) == pytest.approx(expected_db, rel=1e-9)


def test_all_invalid_gives_algorithm_failure(monkeypatch, specs):
    def degenerate_fit(X, algorithm, k, seeds):
        return np.zeros(X.shape[0], dtype=int), (False if algorithm == "gmm" else None)

    monkeypatch.setattr(run_driver, "fit_algorithm", degenerate_fit)
    records = {"cvi": [], "alg": [], "candidates": [], "dataqc": []}
    protos = build_prototypes(specs[449], PROTOCOL_DIR)
    evaluate_cell_seed(specs[449], protos, 0, records, exceptions=[])
    alg_rows = pd.DataFrame(records["alg"])
    assert (alg_rows["algorithm_failure"] == 1).all()
    assert alg_rows["ari_at_ktrue"].isna().all()
    cvi_rows = pd.DataFrame(records["cvi"])
    assert (cvi_rows["correct"] == 0).all()          # denominator stays 100
    assert cvi_rows["k_hat"].isna().all()
    assert cvi_rows["bias"].isna().all()
    assert (cvi_rows["cvi_failure"] == 0).all()      # separate field, not set here


# ---------------------------------------------------------------------------
# Checklist 9 -- k-hat direction / tie / non-finite policy
# ---------------------------------------------------------------------------

def test_k_hat_direction():
    assert select_k_hat({2: 0.2, 3: 0.9, 4: 0.5}, maximize=True) == (3, 0, 0)
    assert select_k_hat({2: 0.9, 3: 0.2, 4: 0.5}, maximize=False) == (3, 0, 0)


def test_k_hat_tie_takes_min_k_and_flags():
    k_hat, tie, failure = select_k_hat({2: 0.5, 3: 0.5, 4: 0.1}, maximize=True)
    assert (k_hat, tie, failure) == (2, 1, 0)


def test_k_hat_tie_is_scale_aware():
    # CH-scale values: relative closeness decides, not an absolute epsilon
    k_hat, tie, _ = select_k_hat({2: 1e6, 3: 1e6 * (1 + 1e-11)}, maximize=True)
    assert (k_hat, tie) == (2, 1)
    k_hat, tie, _ = select_k_hat({2: 1e6, 3: 1e6 * (1 + 1e-9)}, maximize=True)
    assert (k_hat, tie) == (3, 0)


def test_k_hat_non_finite_policy():
    assert select_k_hat({2: float("nan"), 3: 0.4}, maximize=True) == (3, 0, 0)
    k_hat, tie, failure = select_k_hat({2: float("nan"), 3: float("inf") * 0},
                                       maximize=True)
    assert k_hat is None and failure == 1            # cvi_failure, not algorithm


def test_corr_ari_direction_standardized_and_min_candidates():
    db_scores = {2: 3.0, 3: 2.0, 4: 1.0}             # DB: lower = better
    ari = {2: 0.1, 3: 0.5, 4: 0.9}
    rho = corr_ari_spearman(db_scores, ari, maximize=False)
    assert rho == pytest.approx(1.0)                 # -DB rises with ARI
    assert np.isnan(corr_ari_spearman({2: 1.0, 3: 2.0}, {2: 0.1, 3: 0.2},
                                      maximize=True))


# ---------------------------------------------------------------------------
# Checklist 10 -- QC estimators hand-verified on the R cell (449)
# ---------------------------------------------------------------------------

def test_qc_estimators_hand_computation_cell_449(specs):
    spec = specs[449]
    protos = build_prototypes(spec, PROTOCOL_DIR)
    x, y, _, _ = _generate(spec)
    qc = qc_for_seed(x, y, protos)

    # Independent hand computation, S-05 formulas written out explicitly.
    residual_nominal = x - protos[y]
    sg_dev = np.sqrt((residual_nominal ** 2).sum() / residual_nominal.size)
    assert qc["sigma_generator_deviation"] == pytest.approx(sg_dev, rel=1e-12)

    prototypes_hat = []
    for c in range(spec.k_true):
        mean_series = x[y == c].mean(axis=0)
        mean_series = (mean_series - mean_series.mean()) / mean_series.std()  # ddof=0
        prototypes_hat.append(mean_series)
    prototypes_hat = np.vstack(prototypes_hat)
    residual_hat = x - prototypes_hat[y]
    s_ach = np.sqrt((residual_hat ** 2).sum() / residual_hat.size)
    assert qc["sigma_achieved"] == pytest.approx(s_ach, rel=1e-12)

    best, best_pair = -np.inf, ""
    for i in range(spec.k_true):
        for j in range(i + 1, spec.k_true):
            rho = float(prototypes_hat[i] @ prototypes_hat[j]) / T_YEARS  # signed
            if rho > best:
                best, best_pair = rho, f"{i + 1}-{j + 1}"
    assert qc["rho_max_achieved"] == pytest.approx(best, rel=1e-12)
    assert qc["rho_max_pair"] == best_pair


# ---------------------------------------------------------------------------
# Heterogeneous extension: bit-identity regression on homogeneous rows
# ---------------------------------------------------------------------------

def test_homogeneous_generation_bit_identical_to_pinned_function(specs):
    spec = specs[449]
    protos = build_prototypes(spec, PROTOCOL_DIR)

    rng_driver = make_cell_seeds(spec.seed_key, 0).rng_data
    x_driver, y_driver = generate_raw(protos, spec.sizes, spec.sigmas,
                                      spec.jitter, spec.phi, rng_driver)
    x_driver = shapes.zscore(x_driver, axis=-1)

    rng_pinned = make_cell_seeds(spec.seed_key, 0).rng_data
    cfg = shapes.GenConfig(T=T_YEARS, n_per_cluster=spec.n_per_cluster,
                           sigma=spec.sigma, znorm=True,
                           amplitude_jitter=spec.jitter, phi=spec.phi)
    x_pinned, y_pinned = shapes.make_dataset_from_prototypes(protos, cfg, rng_pinned)

    assert np.array_equal(x_driver, x_pinned)
    assert np.array_equal(y_driver, y_pinned)


# ---------------------------------------------------------------------------
# Data-sharing invariant: all four algorithms receive the identical X
# ---------------------------------------------------------------------------

def test_all_algorithms_see_same_array(monkeypatch, specs):
    seen = []

    def recording_fit(X, algorithm, k, seeds):
        seen.append(id(X))
        labels = np.arange(X.shape[0]) % k          # valid partition, cheap
        return labels, (True if algorithm == "gmm" else None)

    monkeypatch.setattr(run_driver, "fit_algorithm", recording_fit)
    records = {"cvi": [], "alg": [], "candidates": [], "dataqc": []}
    protos = build_prototypes(specs[449], PROTOCOL_DIR)
    evaluate_cell_seed(specs[449], protos, 0, records, exceptions=[])
    assert len(seen) == 4 * len(K_CANDIDATES)
    assert len(set(seen)) == 1


# ---------------------------------------------------------------------------
# Outlier mechanism (B6/B11)
# ---------------------------------------------------------------------------

def test_outlier_replacement_keeps_n_and_labels(specs):
    spec = specs[626]  # B11: sizes 10;15;10;5;10, outliers 3 at sigma 1.5
    protos = build_prototypes(spec, PROTOCOL_DIR)
    seeds = make_cell_seeds(spec.seed_key, 0)
    x, y, outlier_ids, outlier_classes = generate_cell_data(
        spec, protos, seeds.rng_data)
    assert x.shape == (50, T_YEARS)                  # replacement: N fixed
    assert len(outlier_ids) == 3
    assert outlier_ids == sorted(outlier_ids)        # ascending, frozen order
    assert [int(y[i]) for i in outlier_ids] == outlier_classes  # labels kept
    assert np.array_equal(np.bincount(y), np.array(spec.sizes))
    # deterministic: same (seed_key, tohum) -> same outlier draw
    x2, _, outlier_ids2, _ = generate_cell_data(
        spec, protos, make_cell_seeds(spec.seed_key, 0).rng_data)
    assert outlier_ids2 == outlier_ids and np.array_equal(x, x2)


# ---------------------------------------------------------------------------
# Integration micro-run + per-cell checkpoint/resume
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_run_cell_micro_and_resume(tmp_path, specs):
    spec = specs[449]
    elapsed = run_cell(spec, PROTOCOL_DIR, tmp_path, n_seeds=2)
    assert elapsed > 0
    cvi = pd.read_parquet(tmp_path / "cells" / "0449_cvi.parquet")
    assert len(cvi) == 2 * 4 * len(CVI_REGISTRY)
    assert set(cvi["indeks"]) == set(CVI_REGISTRY)
    alg = pd.read_parquet(tmp_path / "cells" / "0449_alg.parquet")
    assert len(alg) == 2 * 4
    candidates = pd.read_parquet(tmp_path / "cells" / "0449_candidates.parquet")
    assert len(candidates) == 2 * 4 * len(K_CANDIDATES)
    assert candidates["ari"].notna().all()           # ARI recorded at every k
    dataqc = pd.read_parquet(tmp_path / "cells" / "0449_dataqc.parquet")
    assert len(dataqc) == 2
    # resume: completed cell is skipped
    assert run_cell(spec, PROTOCOL_DIR, tmp_path, n_seeds=2) == 0.0
