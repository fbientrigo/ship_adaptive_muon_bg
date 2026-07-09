"""Guard-rail tests for the committed Muon NTuples v1.0 samples and subsampling.

These need no FairShip, no ROOT, no GPU, and no network — only the small sample
files committed under ``data/samples/`` and NumPy. They protect two things:

1. the committed samples stay loadable, valid, and format-consistent;
2. the range-preserving subsampling logic stays deterministic and keeps the
   observed envelope.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from ship_muon_bg.data_contracts import (
    load_muon_npz,
    load_muon_pkl,
    representative_subset,
    validate_muon_array,
)
from ship_muon_bg.data_contracts import schema

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES_DIR = os.path.join(REPO_ROOT, "data", "samples")

SAMPLE_STEMS = ("muons_FullMC_sample", "muonsFullMC_afterMS_sample")

# The whole point of the exercise: everything committed under data/samples/ must
# fit comfortably in git.
SIZE_BUDGET_BYTES = 10 * 1024 * 1024


def _stem_path(stem, ext):
    return os.path.join(SAMPLES_DIR, f"{stem}{ext}")


@pytest.mark.parametrize("stem", SAMPLE_STEMS)
def test_sample_pkl_loads_and_validates(stem):
    array = load_muon_pkl(_stem_path(stem, ".pkl.gz"))
    assert array.ndim == 2 and array.shape[1] == schema.N_COLUMNS
    # One w == 0 row (a per-column extreme) is expected; allow it per the README.
    validate_muon_array(array, allow_zero_weight=True)


@pytest.mark.parametrize("stem", SAMPLE_STEMS)
def test_npz_and_pkl_are_identical(stem):
    from_pkl = load_muon_pkl(_stem_path(stem, ".pkl.gz"))
    from_npz = load_muon_npz(_stem_path(stem, ".npz"))
    assert from_pkl.shape == from_npz.shape
    assert np.array_equal(from_pkl, from_npz)


@pytest.mark.parametrize("stem", SAMPLE_STEMS)
def test_manifest_matches_committed_sample(stem):
    with open(_stem_path(stem, "_manifest.json"), encoding="utf-8") as handle:
        manifest = json.load(handle)
    array = load_muon_npz(_stem_path(stem, ".npz"))
    assert manifest["subset_n_rows"] == array.shape[0]
    assert manifest["columns"] == list(schema.COLUMNS)


def test_size_budget_under_10mb():
    total = 0
    for root, _dirs, files in os.walk(SAMPLES_DIR):
        for name in files:
            total += os.path.getsize(os.path.join(root, name))
    assert total < SIZE_BUDGET_BYTES, f"data/samples is {total} bytes (>= 10 MiB)"


def _synthetic_full(n=5000, seed=7):
    rng = np.random.default_rng(seed)
    px = rng.normal(0, 2, n)
    py = rng.normal(0, 2, n)
    pz = np.abs(rng.normal(40, 20, n))
    x = rng.normal(0, 3, n)
    y = rng.normal(0, 3, n)
    z = np.full(n, 28.9)
    ids = rng.choice([13.0, -13.0], size=n)
    w = rng.uniform(0.1, 5.0, n)
    return np.column_stack([px, py, pz, x, y, z, ids, w]).astype(np.float64)


def test_subset_is_deterministic():
    full = _synthetic_full()
    a, idx_a = representative_subset(full, 500, seed=42)
    b, idx_b = representative_subset(full, 500, seed=42)
    assert np.array_equal(idx_a, idx_b)
    assert np.array_equal(a, b)


def test_subset_preserves_per_column_extremes():
    full = _synthetic_full()
    subset, _idx = representative_subset(full, 400, seed=42)
    # Every column's full min and max must survive into the subset.
    for col in range(full.shape[1]):
        assert subset[:, col].min() == full[:, col].min()
        assert subset[:, col].max() == full[:, col].max()


def test_subset_returns_whole_array_when_target_exceeds_n():
    full = _synthetic_full(n=100)
    subset, idx = representative_subset(full, 1000, seed=1)
    assert subset.shape == full.shape
    assert np.array_equal(idx, np.arange(full.shape[0]))
