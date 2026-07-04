"""Feature-normalization metadata, fit on the training split only.

Normalization parameters are computed from training rows exclusively to avoid
leakage, then persisted as explicit metadata (not a pickled transformer). The
default method is per-feature standardization (mean/std), which is minimal,
invertible, and dependency-free.
"""

from __future__ import annotations

import numpy as np

from . import schema

NORMALIZATION_SCHEMA_VERSION = "0"


def fit_normalization(array, train_indices, *, method="standard", dataset_hash=None):
    """Fit per-feature normalization metadata on the training rows only.

    Only the continuous feature columns (:data:`schema.FEATURE_COLUMNS`) are
    standardized; ``id`` and ``w`` are excluded.

    Parameters
    ----------
    array : numpy.ndarray
        Full ``(N, 8)`` dataset.
    train_indices : array-like of int
        Row indices of the training split. Normalization uses these rows ONLY.
    method : str
        Currently only ``"standard"`` (mean/std) is supported.
    dataset_hash : str or None
        Provenance hash recorded in the metadata.

    Returns
    -------
    dict
        JSON-serializable normalization metadata.
    """
    if method != "standard":
        raise ValueError(f"unsupported normalization method: {method!r}")

    train_idx = np.asarray(train_indices, dtype=int)
    if train_idx.size < 1:
        raise ValueError("train_indices must be non-empty")

    feature_cols = schema.column_indices(schema.FEATURE_COLUMNS)
    train_features = array[np.ix_(train_idx, feature_cols)]

    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0)
    # Guard against zero-variance features to keep the transform invertible.
    safe_std = np.where(std > 0.0, std, 1.0)

    return {
        "schema_version": NORMALIZATION_SCHEMA_VERSION,
        "method": method,
        "feature_order": list(schema.FEATURE_COLUMNS),
        "fit_on": "train",
        "n_train_rows": int(train_idx.size),
        "params": {
            name: {"mean": float(mean[i]), "std": float(safe_std[i])}
            for i, name in enumerate(schema.FEATURE_COLUMNS)
        },
        "dataset_hash": dataset_hash,
    }


def apply_normalization(array, normalization):
    """Apply previously-fitted normalization metadata to feature columns.

    Returns a standardized copy of the feature columns in ``feature_order``.
    Provided as a helper (and for tests); the contract layer itself only fits
    and records metadata.
    """
    feature_order = normalization["feature_order"]
    feature_cols = schema.column_indices(feature_order)
    features = array[:, feature_cols].astype(np.float64, copy=True)
    for i, name in enumerate(feature_order):
        p = normalization["params"][name]
        features[:, i] = (features[:, i] - p["mean"]) / p["std"]
    return features
