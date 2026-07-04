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
