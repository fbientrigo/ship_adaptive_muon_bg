"""Guard-rail tests for the v0 ``data_contracts`` layer.

These tests require no FairShip, no ROOT, no GPU, and only a tiny committed PKL
fixture. They are the gate that must stay green before any modelling code lands.
"""

from __future__ import annotations

import gzip
import os
import pickle

import numpy as np
import pytest

from ship_muon_bg.data_contracts import (
    FiniteError,
    IdError,
    ShapeError,
    WeightError,
    dataset_hash,
    fit_normalization,
    load_muon_pkl,
    make_split,
    validate_muon_array,
)


def _write_pkl(tmp_path, array, name="data.pkl.gz"):
    path = os.path.join(tmp_path, name)
    with gzip.open(path, "wb") as handle:
        pickle.dump(np.asarray(array, dtype=np.float64), handle, protocol=4)
    return path


def _valid_array(n=16, seed=0):
    rng = np.random.default_rng(seed)
    px = rng.normal(0, 1, n)
    py = rng.normal(0, 1, n)
    pz = np.abs(rng.normal(40, 5, n))
    x = rng.normal(0, 1, n)
    y = rng.normal(0, 1, n)
    z = np.full(n, 30.0)
    ids = rng.choice([13.0, -13.0], size=n)
    w = rng.uniform(0.1, 1.0, n)
    return np.column_stack([px, py, pz, x, y, z, ids, w]).astype(np.float64)


# 1. Valid fixture loads correctly.
def test_valid_fixture_loads(tiny_pkl_path):
    array = load_muon_pkl(tiny_pkl_path)
    assert array.shape == (64, 8)
    assert array.dtype == np.float64
    validate_muon_array(array)  # must not raise


# 2. Invalid shape fails.
def test_invalid_shape_fails(tmp_path):
    bad = _valid_array()[:, :7]  # 7 columns
    with pytest.raises(ShapeError):
        validate_muon_array(bad)


def test_empty_array_fails():
    with pytest.raises(ShapeError):
        validate_muon_array(np.empty((0, 8), dtype=np.float64))


# 3. NaN / inf fails.
@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_nan_inf_fails(bad_value):
    array = _valid_array()
    array[3, 2] = bad_value
    with pytest.raises(FiniteError):
        validate_muon_array(array)


# 4. Non-positive weights fail.
@pytest.mark.parametrize("bad_weight", [0.0, -0.5])
def test_non_positive_weight_fails(bad_weight):
    array = _valid_array()
    array[5, 7] = bad_weight
    with pytest.raises(WeightError):
        validate_muon_array(array)


def test_zero_weight_allowed_when_opted_in():
    array = _valid_array()
    array[5, 7] = 0.0
    validate_muon_array(array, allow_zero_weight=True)  # must not raise


# 5. Non-integer id fails.
def test_non_integer_id_fails():
    array = _valid_array()
    array[2, 6] = 13.5
    with pytest.raises(IdError):
        validate_muon_array(array)


# 6 & 7. Deterministic split with same seed; different split with different seed.
def test_split_deterministic_same_seed():
    split_a = make_split(64, seed=1234, val_fraction=0.25)
    split_b = make_split(64, seed=1234, val_fraction=0.25)
    assert split_a["train_indices"] == split_b["train_indices"]
    assert split_a["val_indices"] == split_b["val_indices"]


def test_split_differs_with_different_seed():
    split_a = make_split(64, seed=1234, val_fraction=0.25)
    split_b = make_split(64, seed=4321, val_fraction=0.25)
    assert split_a["train_indices"] != split_b["train_indices"]


def test_split_partitions_all_indices_without_overlap():
    split = make_split(64, seed=7, val_fraction=0.25)
    train = set(split["train_indices"])
    val = set(split["val_indices"])
    assert train.isdisjoint(val)
    assert train | val == set(range(64))


# 8. Normalization is fit on train only.
def test_normalization_fit_on_train_only():
    array = _valid_array(n=40, seed=3)
    split = make_split(40, seed=99, val_fraction=0.25)
    train_idx = np.asarray(split["train_indices"])

    norm = fit_normalization(array, train_idx)
    assert norm["fit_on"] == "train"
    assert norm["n_train_rows"] == train_idx.size

    # The recorded mean must equal the TRAIN-rows mean, not the full-dataset mean.
    px_train_mean = float(array[train_idx, 0].mean())
    px_full_mean = float(array[:, 0].mean())
    assert norm["params"]["px"]["mean"] == pytest.approx(px_train_mean)
    # With a real split the two means differ (guards against accidental full fit).
    assert norm["params"]["px"]["mean"] != pytest.approx(px_full_mean)


def test_dataset_hash_is_deterministic_and_order_sensitive():
    array = _valid_array(seed=5)
    assert dataset_hash(array) == dataset_hash(array.copy())
    reordered = array[::-1].copy()
    assert dataset_hash(array) != dataset_hash(reordered)


def test_loader_roundtrip(tmp_path):
    array = _valid_array()
    path = _write_pkl(tmp_path, array)
    loaded = load_muon_pkl(path)
    np.testing.assert_allclose(loaded, array)


# 9. No ROOT / FairShip import in core.
def test_no_root_or_fairship_import_in_core():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    core_dir = os.path.join(repo_root, "src", "ship_muon_bg")
    offenders = []
    for dirpath, _dirs, files in os.walk(core_dir):
        for name in files:
            if not name.endswith(".py"):
                continue
            full = os.path.join(dirpath, name)
            with open(full, "r", encoding="utf-8") as handle:
                for lineno, line in enumerate(handle, start=1):
                    stripped = line.strip()
                    if not (stripped.startswith("import ") or stripped.startswith("from ")):
                        continue
                    lowered = stripped.lower()
                    if (
                        "import root" in lowered
                        or "fairship" in lowered
                        or stripped == "import ROOT"
                    ):
                        offenders.append(f"{full}:{lineno}: {stripped}")
    assert not offenders, "core must not import ROOT/FairShip:\n" + "\n".join(offenders)
