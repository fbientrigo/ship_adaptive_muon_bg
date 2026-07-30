"""Deterministic train/validation split manifest.

All randomness is driven by an explicit integer ``seed`` via
``numpy.random.default_rng``. There is no ``time.time()`` seeding. The split is
recorded as explicit row-index lists so it can be reproduced and audited exactly.
"""

from __future__ import annotations

import numpy as np

SPLIT_SCHEMA_VERSION = "0"


def make_split(n_rows, *, seed, val_fraction=0.2, dataset_hash=None):
    """Build a deterministic train/validation split manifest.

    Parameters
    ----------
    n_rows : int
        Number of rows in the dataset.
    seed : int
        Explicit deterministic seed. Required; never derived from wall-clock time.
    val_fraction : float
        Fraction of rows assigned to validation, in ``(0, 1)``.
    dataset_hash : str or None
        Hash of the dataset the split applies to, recorded for provenance.

    Returns
    -------
    dict
        A JSON-serializable manifest with explicit ``train_indices`` and
        ``val_indices``.
    """
    if not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an explicit integer")
    if n_rows < 1:
        raise ValueError("n_rows must be >= 1")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in the open interval (0, 1)")

    rng = np.random.default_rng(int(seed))
    permutation = rng.permutation(n_rows)
    n_val = max(1, int(round(n_rows * val_fraction)))
    n_val = min(n_val, n_rows - 1)  # keep at least one training row

    val_indices = np.sort(permutation[:n_val])
    train_indices = np.sort(permutation[n_val:])

    return {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "strategy": "random_permutation",
        "seed": int(seed),
        "val_fraction": float(val_fraction),
        "n_rows": int(n_rows),
        "n_train": int(train_indices.size),
        "n_val": int(val_indices.size),
        "train_indices": train_indices.astype(int).tolist(),
        "val_indices": val_indices.astype(int).tolist(),
        "dataset_hash": dataset_hash,
    }


def make_three_way_split(
    n_rows, *, seed, val_fraction=0.2, test_fraction=0.2, dataset_hash=None
):
    """Build a deterministic train/validation/test split manifest.

    Composed from two calls to :func:`make_split` (never reimplementing its
    permutation logic): the first carves ``test_fraction`` of all rows off as
    the test split; the second splits the *remaining* pool into train/val
    using ``val_fraction`` of that pool. The two calls use independent seeds
    derived from ``seed`` via ``numpy.random.SeedSequence.spawn``, so this
    composition never collides with a direct 2-way :func:`make_split` call
    using the same integer ``seed``.

    Weights never influence row selection: this is a uniform random
    permutation over row indices, exactly like :func:`make_split`.

    Returns
    -------
    dict
        A JSON-serializable manifest with explicit ``train_indices``,
        ``val_indices``, ``test_indices`` (each sorted, mutually disjoint,
        covering all ``n_rows`` rows).
    """

    if not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an explicit integer")
    if n_rows < 1:
        raise ValueError("n_rows must be >= 1")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in the open interval (0, 1)")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be in the open interval (0, 1)")

    child_test, child_val = np.random.SeedSequence(int(seed)).spawn(2)
    seed_test = int(child_test.generate_state(1)[0])
    seed_val = int(child_val.generate_state(1)[0])

    carve = make_split(n_rows, seed=seed_test, val_fraction=test_fraction, dataset_hash=dataset_hash)
    test_indices = np.asarray(carve["val_indices"], dtype=int)
    pool_indices = np.asarray(carve["train_indices"], dtype=int)

    pool_split = make_split(pool_indices.size, seed=seed_val, val_fraction=val_fraction, dataset_hash=dataset_hash)
    train_indices = np.sort(pool_indices[np.asarray(pool_split["train_indices"], dtype=int)])
    val_indices = np.sort(pool_indices[np.asarray(pool_split["val_indices"], dtype=int)])
    test_indices = np.sort(test_indices)

    return {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "strategy": "composed_two_way_random_permutation",
        "seed": int(seed),
        "seed_test_split": seed_test,
        "seed_val_split": seed_val,
        "val_fraction": float(val_fraction),
        "test_fraction": float(test_fraction),
        "n_rows": int(n_rows),
        "n_train": int(train_indices.size),
        "n_val": int(val_indices.size),
        "n_test": int(test_indices.size),
        "train_indices": train_indices.astype(int).tolist(),
        "val_indices": val_indices.astype(int).tolist(),
        "test_indices": test_indices.astype(int).tolist(),
        "weights_affect_selection": False,
        "dataset_hash": dataset_hash,
    }
