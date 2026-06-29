"""Deterministic content hashing for muon datasets.

The hash binds a dataset's raw content (in a canonical byte layout) to its
column schema and contract version, so any downstream artifact can be traced to
its exact input. Row order is part of provenance: reordered rows hash
differently by design.
"""

from __future__ import annotations

import hashlib

import numpy as np

from . import schema


def dataset_hash(array, *, contract_version=schema.CONTRACT_VERSION, columns=schema.COLUMNS):
    """Return a SHA-256 hex digest over the canonicalized array plus schema.

    Canonicalization: C-contiguous ``float64`` bytes, prefixed by the contract
    version, column schema, and shape, so two files with identical content hash
    identically and a schema/version change changes the hash.
    """
    canonical = np.ascontiguousarray(array, dtype=np.float64)
    hasher = hashlib.sha256()
    header = "|".join(
        [
            f"contract={contract_version}",
            f"columns={','.join(columns)}",
            f"shape={canonical.shape[0]}x{canonical.shape[1]}",
            "dtype=float64",
        ]
    )
    hasher.update(header.encode("utf-8"))
    hasher.update(b"\x00")
    hasher.update(canonical.tobytes(order="C"))
    return hasher.hexdigest()
