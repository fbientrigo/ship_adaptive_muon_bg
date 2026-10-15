"""Deterministic global train/validation/test split (Phase B, §5.1).

Each row's split assignment is a pure function of
``(dataset_hash, row_index, split_seed)`` via a stable per-row hash -- not a
full in-memory permutation. This means: (a) the split never depends on the
total row count or on how many shards a split is later divided into, and
(b) a row's split can be recomputed independently, without materializing the
whole dataset, which matters for a genuinely out-of-core pipeline.

No source-lineage/group identifier exists in this contract, so row-disjoint
splitting does not prove source-muon independence -- see
``LIMITATION_NO_SOURCE_LINEAGE``.
"""

from __future__ import annotations

import numpy as np

SPLIT_SCHEMA_VERSION = "0"
DEFAULT_FRACTIONS = {"train": 0.80, "validation": 0.10, "test": 0.10}

LIMITATION_NO_SOURCE_LINEAGE = (
    "No source-lineage/group identifier is currently available, therefore "
    "row-disjoint splitting does not prove source-muon independence."
)

_MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def _mix64(x):
    """splitmix64 finalizer: not cryptographic, just a well-distributed,
    deterministic avalanche mix suitable for reproducible row assignment."""
    x = x & _MASK64
    # Modulo-2^64 wraparound is intentional here (splitmix64 is defined over
    # uint64 arithmetic); NumPy's scalar overflow warning is expected noise
    # for these two multiplies, not a correctness signal, so it is suppressed
    # at the narrowest possible scope.
    with np.errstate(over="ignore"):
        x = (x ^ (x >> np.uint64(33))) * np.uint64(0xFF51AFD7ED558CCD) & _MASK64
        x = (x ^ (x >> np.uint64(33))) * np.uint64(0xC4CEB9FE1A85EC53) & _MASK64
    x = x ^ (x >> np.uint64(33))
    return x


def _row_salt(dataset_hash, seed):
    """Fold ``dataset_hash`` (hex str) and integer ``seed`` into one uint64 salt."""
    h_component = int(dataset_hash[:16], 16) if dataset_hash else 0
    salt = (np.uint64(h_component) ^ np.uint64(int(seed) & 0xFFFFFFFFFFFFFFFF)) & _MASK64
    return _mix64(salt)


def row_uniform(row_indices, *, dataset_hash, seed):
    """Deterministic uniform value in [0, 1) for each row index.

    Pure function of (dataset_hash, row_index, seed): independent of n_rows
    and of any shard boundary.
    """
    row_indices = np.asarray(row_indices, dtype=np.uint64)
    salt = _row_salt(dataset_hash, seed)
    mixed = _mix64((row_indices * np.uint64(0x9E3779B97F4A7C15)) ^ salt)
    # top 53 bits -> float in [0, 1), matching double mantissa precision.
    return (mixed >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def assign_split(row_indices, *, dataset_hash, seed, fractions=None):
    """Return an array of {"train","validation","test"} labels, one per row index."""
    fractions = dict(fractions or DEFAULT_FRACTIONS)
    total = sum(fractions.values())
    if not np.isclose(total, 1.0):
        raise ValueError("split fractions must sum to 1.0, got {}".format(total))

    u = row_uniform(row_indices, dataset_hash=dataset_hash, seed=seed)
    train_edge = fractions["train"]
    val_edge = train_edge + fractions["validation"]

    labels = np.full(u.shape, "test", dtype=object)
    labels[u < train_edge] = "train"
    labels[(u >= train_edge) & (u < val_edge)] = "validation"
    return labels


def split_manifest(n_rows, *, dataset_hash, seed, fractions=None):
    """Build a JSON-serializable manifest recording the split assignment and
    its realized fractions (which only approximate the requested fractions,
    since assignment is per-row-independent rather than a fixed-size draw)."""
    fractions = dict(fractions or DEFAULT_FRACTIONS)
    row_indices = np.arange(n_rows, dtype=np.uint64)
    labels = assign_split(row_indices, dataset_hash=dataset_hash, seed=seed, fractions=fractions)

    counts = {name: int(np.count_nonzero(labels == name)) for name in fractions}
    realized_fractions = {
        name: (counts[name] / n_rows if n_rows else None) for name in fractions
    }

    return {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "strategy": "stable_row_hash",
        "dataset_hash": dataset_hash,
        "seed": int(seed),
        "n_rows": int(n_rows),
        "requested_fractions": fractions,
        "realized_counts": counts,
        "realized_fractions": realized_fractions,
        "limitation_source_lineage": LIMITATION_NO_SOURCE_LINEAGE,
    }
