"""Frozen seed architecture for the CVI benchmark run.

Source: kosum_protokolu_v5_3.md, "Seed mimarisi" (normative code block,
lines ~162-172), verbatim:

    PROTO_TAG = 20260810
    root      = np.random.SeedSequence([PROTO_TAG, seed_key_hucre, tohum])
    data_ss, alg_ss = root.spawn(2)
    rng_data  = np.random.default_rng(data_ss)   # ALL data-side randomness:
                                                 # noise, jitter, outlier-index
                                                 # selection (and outlier
                                                 # regeneration)
    kmeans_seed, gmm_seed = [int(s.generate_state(1)[0]) for s in alg_ss.spawn(2)]

REVISION r2 (2026-08-13): the first release of this module built the data
Generator directly from the root SeedSequence, following the abbreviated
scheme in devir_notu_v3. That produced a DIFFERENT stream than the
protocol's rng_data (which comes from root.spawn(2)[0]). Corrected here;
the abbreviated form must not be used.

Rules (protocol):
- ``seed_key_hucre`` is a manifest column: = hucre_id everywhere except B0
  (hucre 612 -> 449). ALWAYS read it from run_matrix_v4.csv; never derive.
- ``tohum`` = 0..99.
- The X array generated for one (seed_key_hucre, tohum) is given UNCHANGED
  to all four algorithms. Ward/Complete are deterministic and take no seed;
  KMeans/GMM take kmeans_seed/gmm_seed. seed_data != seed_alg by
  construction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PROTO_TAG: int = 20260810


@dataclass(frozen=True)
class CellSeeds:
    rng_data: np.random.Generator
    kmeans_seed: int
    gmm_seed: int


def make_cell_seeds(seed_key_hucre: int, tohum: int) -> CellSeeds:
    """Frozen-architecture seeds for one (cell, replication) pair."""
    root = np.random.SeedSequence([PROTO_TAG, int(seed_key_hucre), int(tohum)])
    data_ss, alg_ss = root.spawn(2)
    rng_data = np.random.default_rng(data_ss)
    kmeans_seed, gmm_seed = [int(s.generate_state(1)[0]) for s in alg_ss.spawn(2)]
    return CellSeeds(rng_data=rng_data, kmeans_seed=kmeans_seed, gmm_seed=gmm_seed)


def make_rng(seed_key_hucre: int, tohum: int) -> np.random.Generator:
    """Data-side Generator only (= make_cell_seeds(...).rng_data)."""
    return make_cell_seeds(seed_key_hucre, tohum).rng_data
