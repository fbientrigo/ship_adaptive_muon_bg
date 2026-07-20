import numpy as np

from ship_muon_bg.afterms import split


def test_deterministic_repeatable():
    labels_a = split.assign_split(
        np.arange(50_000), dataset_hash="deadbeef" * 8, seed=7
    )
    labels_b = split.assign_split(
        np.arange(50_000), dataset_hash="deadbeef" * 8, seed=7
    )
    np.testing.assert_array_equal(labels_a, labels_b)


def test_no_row_overlap_across_splits():
    manifest = split.split_manifest(100_000, dataset_hash="cafebabe" * 8, seed=3)
    labels = split.assign_split(
        np.arange(100_000), dataset_hash="cafebabe" * 8, seed=3
    )
    train = set(np.nonzero(labels == "train")[0].tolist())
    val = set(np.nonzero(labels == "validation")[0].tolist())
    test = set(np.nonzero(labels == "test")[0].tolist())
    assert not (train & val)
    assert not (train & test)
    assert not (val & test)
    assert len(train) + len(val) + len(test) == 100_000
    assert manifest["realized_counts"]["train"] == len(train)


def test_realized_fractions_close_to_target():
    manifest = split.split_manifest(500_000, dataset_hash="0123abcd" * 8, seed=11)
    assert abs(manifest["realized_fractions"]["train"] - 0.80) < 0.01
    assert abs(manifest["realized_fractions"]["validation"] - 0.10) < 0.01
    assert abs(manifest["realized_fractions"]["test"] - 0.10) < 0.01


def test_split_independent_of_total_row_count():
    # A row's assignment must not depend on n_rows or on other rows present --
    # the same dataset_hash/seed/row_index must give the same label whether
    # queried alone or as part of a larger index array (no shard-count coupling).
    dataset_hash = "feedface" * 8
    seed = 42
    idx = np.array([5, 500, 50_000])
    labels_alone = split.assign_split(idx, dataset_hash=dataset_hash, seed=seed)
    labels_in_context = split.assign_split(
        np.arange(200_000), dataset_hash=dataset_hash, seed=seed
    )[idx]
    np.testing.assert_array_equal(labels_alone, labels_in_context)


def test_different_seed_changes_assignment():
    idx = np.arange(10_000)
    labels_a = split.assign_split(idx, dataset_hash="aa" * 32, seed=1)
    labels_b = split.assign_split(idx, dataset_hash="aa" * 32, seed=2)
    assert not np.array_equal(labels_a, labels_b)
