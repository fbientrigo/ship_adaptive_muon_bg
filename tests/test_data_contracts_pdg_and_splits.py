"""Unit tests for the PDG-filtering and three-way-split additions to
``ship_muon_bg.data_contracts`` (D9/D7 fixture-to-full-data execution
contract). Synthetic unit fixtures only; the repository afterMS sample is
exercised separately in ``tests/test_empirical_campaign.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from ship_muon_bg.data_contracts import (
    IdError,
    filter_by_pdg,
    make_split,
    make_three_way_split,
    pdg_counts,
    schema,
)


def _mixed_pdg_array(n=200, seed=0):
    rng = np.random.default_rng(seed)
    px = rng.normal(0, 1, n)
    py = rng.normal(0, 1, n)
    pz = np.abs(rng.normal(40, 5, n)) + 1.0
    x = rng.normal(0, 1, n)
    y = rng.normal(0, 1, n)
    z = np.full(n, 28.9)
    ids = rng.choice([13.0, -13.0], size=n)
    w = rng.uniform(0.1, 5.0, n)
    return np.column_stack([px, py, pz, x, y, z, ids, w]).astype(np.float64)


# --- filter_by_pdg -----------------------------------------------------------


def test_filter_by_pdg_selects_exactly_matching_rows():
    array = _mixed_pdg_array()
    filtered, indices = filter_by_pdg(array, 13)
    assert np.array_equal(array[indices], filtered)
    assert np.all(np.rint(filtered[:, schema.COLUMN_INDEX["id"]]) == 13)
    other, other_indices = filter_by_pdg(array, -13)
    # Every row belongs to exactly one PDG track (mutually exclusive, jointly exhaustive).
    assert set(indices.tolist()) & set(other_indices.tolist()) == set()
    assert filtered.shape[0] + other.shape[0] == array.shape[0]


def test_filter_by_pdg_indices_are_sorted_and_index_into_source():
    array = _mixed_pdg_array()
    _filtered, indices = filter_by_pdg(array, 13)
    assert np.array_equal(indices, np.sort(indices))
    assert indices.min() >= 0 and indices.max() < array.shape[0]


def test_filter_by_pdg_rejects_non_muon_pdg_id():
    array = _mixed_pdg_array()
    with pytest.raises(IdError):
        filter_by_pdg(array, 211)  # a pion PDG id, not a supported muon track


def test_filter_by_pdg_empty_result_for_absent_track():
    array = _mixed_pdg_array(n=50)
    array[:, schema.COLUMN_INDEX["id"]] = 13.0  # force every row to one track
    filtered, indices = filter_by_pdg(array, -13)
    assert filtered.shape == (0, schema.N_COLUMNS)
    assert indices.shape == (0,)


def test_pdg_counts_matches_manual_histogram():
    array = _mixed_pdg_array()
    counts = pdg_counts(array)
    ids = np.rint(array[:, schema.COLUMN_INDEX["id"]]).astype(int)
    for pdg_id in (13, -13):
        assert counts[pdg_id] == int(np.count_nonzero(ids == pdg_id))
    assert sum(counts.values()) == array.shape[0]


# --- make_three_way_split -----------------------------------------------------


def test_three_way_split_partitions_are_disjoint_and_complete():
    split = make_three_way_split(1000, seed=42, val_fraction=0.2, test_fraction=0.2)
    train = set(split["train_indices"])
    val = set(split["val_indices"])
    test = set(split["test_indices"])
    assert not (train & val)
    assert not (train & test)
    assert not (val & test)
    assert train | val | test == set(range(1000))
    assert split["n_train"] + split["n_val"] + split["n_test"] == 1000


def test_three_way_split_is_deterministic_for_same_seed():
    first = make_three_way_split(500, seed=7, val_fraction=0.25, test_fraction=0.15)
    second = make_three_way_split(500, seed=7, val_fraction=0.25, test_fraction=0.15)
    assert first == second


def test_three_way_split_differs_from_two_way_split_at_same_seed():
    """The composition uses seeds spawned from the input seed, not the seed
    itself, so it must not silently collide with a direct make_split call."""

    two_way = make_split(500, seed=7, val_fraction=0.25)
    three_way = make_three_way_split(500, seed=7, val_fraction=0.25, test_fraction=0.15)
    assert three_way["seed_test_split"] != 7
    assert three_way["seed_val_split"] != 7
    assert set(two_way["val_indices"]) != set(three_way["val_indices"])


def test_three_way_split_approximate_fractions():
    split = make_three_way_split(10000, seed=1, val_fraction=0.2, test_fraction=0.3)
    assert split["n_test"] == pytest.approx(3000, abs=50)
    remaining = 10000 - split["n_test"]
    assert split["n_val"] == pytest.approx(int(round(remaining * 0.2)), abs=50)


def test_three_way_split_records_weight_agnostic_selection():
    split = make_three_way_split(200, seed=1, val_fraction=0.2, test_fraction=0.2)
    assert split["weights_affect_selection"] is False


def test_three_way_split_rejects_invalid_fractions():
    with pytest.raises(ValueError):
        make_three_way_split(100, seed=1, val_fraction=1.5, test_fraction=0.2)
    with pytest.raises(ValueError):
        make_three_way_split(100, seed=1, val_fraction=0.2, test_fraction=0.0)


def test_three_way_split_dataset_hash_is_recorded():
    split = make_three_way_split(100, seed=1, val_fraction=0.2, test_fraction=0.2, dataset_hash="abc123")
    assert split["dataset_hash"] == "abc123"
