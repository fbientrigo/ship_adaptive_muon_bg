"""Fixed schema for the local post-shield muon PKL contract (v0).

These are **post-shield muon states**. The schema records the column layout and
units only; it does not impose DIS/event-level energy or momentum conservation.
Full event-level conservation belongs to the downstream ``simulation_backend``,
never to this contract.
"""

from __future__ import annotations

# Contract version. Bump on any breaking change to the column layout or hash
# canonicalization so that ``dataset_hash`` values stay comparable within a version.
CONTRACT_VERSION = "0"

# Fixed column order of the ``(N, 8)`` array.
COLUMNS = ("px", "py", "pz", "x", "y", "z", "id", "w")
N_COLUMNS = len(COLUMNS)

# Per-column units, for provenance/reporting only (not enforced as physics).
UNITS = {
    "px": "GeV/c",
    "py": "GeV/c",
    "pz": "GeV/c",
    "x": "m",
    "y": "m",
    "z": "m",
    "id": "PDG code (int-valued)",
    "w": "dimensionless",
}

# Column-index groups used by validation and normalization.
COLUMN_INDEX = {name: i for i, name in enumerate(COLUMNS)}

MOMENTUM_COLUMNS = ("px", "py", "pz")
POSITION_COLUMNS = ("x", "y", "z")

# Continuous feature columns that normalization standardizes. ``id`` (a PDG code)
# and ``w`` (an event weight) are deliberately excluded.
FEATURE_COLUMNS = MOMENTUM_COLUMNS + POSITION_COLUMNS

ID_COLUMN = "id"
WEIGHT_COLUMN = "w"

# Expected muon PDG ids. Values outside this set are warned about (recorded in the
# dataset report), not rejected.
EXPECTED_MUON_IDS = (13, -13)


def column_indices(names):
    """Return the integer indices for the given column ``names``."""
    return [COLUMN_INDEX[name] for name in names]
