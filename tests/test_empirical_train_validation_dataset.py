"""Test-payload closure contract for ``build_empirical_train_validation_dataset``.

Gate E (test-payload evaluation) must remain closed for any D9 conditional
experiment: the loading path used there must never materialize, index, or
otherwise inspect a single test-partition feature value. This module proves
that mechanistically (the returned object structurally has no ``test``
partition) and causally (mutating only the rows that fall in the test split
does not change anything the training path reads).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ship_muon_bg.data_contracts.splitting import make_three_way_split
from ship_muon_bg.data_contracts.subsampling import save_subset_npz
from ship_muon_bg.density_lab.empirical import (
    EmpiricalDatasetSpec,
    build_empirical_dataset,
    build_empirical_train_validation_dataset,
)

N_ROWS = 30
SEED = 7
VAL_FRACTION = 0.2
TEST_FRACTION = 0.2


def _synthetic_array(mutate_test_rows: bool, test_indices: np.ndarray) -> np.ndarray:
    array = np.zeros((N_ROWS, 8), dtype=np.float64)
    array[:, 0] = np.arange(N_ROWS, dtype=np.float64)  # px, distinct per row
    array[:, 6] = 13.0  # id: single PDG track, so row order == filtered order
    array[:, 7] = 1.0  # w
    if mutate_test_rows:
        array[test_indices, 0] += 5000.0  # still well within the momentum bound
        array[test_indices, 3] += 7.0  # x, also mutated
    return array


def _known_test_indices() -> np.ndarray:
    # Depends only on (n_rows, seed, fractions) -- never on row content -- so
    # it can be computed without loading any file.
    split = make_three_way_split(
        N_ROWS, seed=SEED, val_fraction=VAL_FRACTION, test_fraction=TEST_FRACTION
    )
    return np.asarray(split["test_indices"], dtype=int)


def test_train_validation_dataset_has_no_test_partition_attribute(tmp_path: Path):
    test_indices = _known_test_indices()
    path = tmp_path / "clean_sample.npz"
    save_subset_npz(path, _synthetic_array(False, test_indices))
    spec = EmpiricalDatasetSpec(
        dataset_path=str(path), pdg_id=13, seed=SEED,
        val_fraction=VAL_FRACTION, test_fraction=TEST_FRACTION, max_rows=None,
    )
    dataset = build_empirical_train_validation_dataset(spec)
    assert not hasattr(dataset, "test")
    manifest = dataset.manifest()
    assert manifest["test_payload_loaded"] is False
    assert manifest["test_used_for_training"] is False
    assert manifest["test_used_for_preprocessing"] is False
    assert manifest["test_used_for_model_selection"] is False
    assert manifest["test_used_for_evaluation"] is False
    assert manifest["test_row_count"] == test_indices.size
    assert "partitions" in manifest and set(manifest["partitions"]) == {"train", "validation"}


def test_train_validation_dataset_matches_full_split_counts_and_hash(tmp_path: Path):
    test_indices = _known_test_indices()
    path = tmp_path / "clean_sample.npz"
    array = _synthetic_array(False, test_indices)
    save_subset_npz(path, array)
    spec = EmpiricalDatasetSpec(
        dataset_path=str(path), pdg_id=13, seed=SEED,
        val_fraction=VAL_FRACTION, test_fraction=TEST_FRACTION, max_rows=None,
    )
    closed = build_empirical_train_validation_dataset(spec)
    full = build_empirical_dataset(spec)
    assert closed.test_row_count == full.test.n_rows
    np.testing.assert_array_equal(closed.train.raw, full.train.raw)
    np.testing.assert_array_equal(closed.validation.raw, full.validation.raw)


def test_mutating_only_test_rows_never_changes_train_or_validation_or_test_provenance(tmp_path: Path):
    """Mutate feature values ONLY at the deterministic test-row positions.

    The closed loading path must return bit-identical train/validation rows
    and an identical test row count / test split hash either way -- proof it
    never reads the mutated (test-only) content. ``source_file_dataset_hash``
    is intentionally excluded from this comparison: it is a whole-file
    identity hash computed over the raw loaded file before any split is
    applied (pre-existing, unrelated to this contract), not declared
    test-partition provenance.
    """

    test_indices = _known_test_indices()
    clean_path = tmp_path / "clean_sample.npz"
    mutated_path = tmp_path / "mutated_test_rows_sample.npz"
    save_subset_npz(clean_path, _synthetic_array(False, test_indices))
    save_subset_npz(mutated_path, _synthetic_array(True, test_indices))

    def _spec(path: Path) -> EmpiricalDatasetSpec:
        return EmpiricalDatasetSpec(
            dataset_path=str(path), pdg_id=13, seed=SEED,
            val_fraction=VAL_FRACTION, test_fraction=TEST_FRACTION, max_rows=None,
        )

    clean = build_empirical_train_validation_dataset(_spec(clean_path))
    mutated = build_empirical_train_validation_dataset(_spec(mutated_path))

    np.testing.assert_array_equal(clean.train.raw, mutated.train.raw)
    np.testing.assert_array_equal(clean.validation.raw, mutated.validation.raw)
    assert clean.test_row_count == mutated.test_row_count == test_indices.size
    assert clean.test_split_hash == mutated.test_split_hash

    # Sanity: the mutation actually touched something (otherwise this test
    # would pass vacuously) -- confirmed by inspecting the *full* (test-
    # materializing) loader on each file directly, never by the closed path.
    full_clean = build_empirical_dataset(_spec(clean_path))
    full_mutated = build_empirical_dataset(_spec(mutated_path))
    assert not np.array_equal(full_clean.test.raw, full_mutated.test.raw)
