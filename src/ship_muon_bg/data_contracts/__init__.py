"""Local post-shield muon PKL data contract (v0).

Enforces the ``(N, 8)`` ``[px, py, pz, x, y, z, id, w]`` contract for trusted
local legacy muon PKL files: load, validate, hash, deterministically split, and
record train-only normalization metadata. Pure Python + NumPy; imports neither
FairShip nor ROOT.
"""

from __future__ import annotations

from . import schema
from .errors import (
    BoundsError,
    DataContractError,
    FiniteError,
    IdError,
    LoaderError,
    ShapeError,
    WeightError,
)
from .hashing import dataset_hash
from .loader import load_muon_pkl
from .normalization import apply_normalization, fit_normalization
from .report import build_dataset_report, process_pkl, write_artifacts
from .splitting import make_split
from .distribution_compare import (
    compare_distributions,
    ks_2samp,
    moment_summary,
)
from .subsampling import (
    load_muon_npz,
    representative_subset,
    save_subset_npz,
    save_subset_pkl_gz,
)
from .validation import (
    DEFAULT_BOUNDS,
    run_checks,
    validate_bounds,
    validate_finite,
    validate_id_integer,
    validate_muon_array,
    validate_shape,
    validate_weights,
)

__all__ = [
    "schema",
    "DataContractError",
    "ShapeError",
    "FiniteError",
    "WeightError",
    "IdError",
    "BoundsError",
    "LoaderError",
    "load_muon_pkl",
    "load_muon_npz",
    "representative_subset",
    "save_subset_npz",
    "save_subset_pkl_gz",
    "compare_distributions",
    "ks_2samp",
    "moment_summary",
    "dataset_hash",
    "make_split",
    "fit_normalization",
    "apply_normalization",
    "DEFAULT_BOUNDS",
    "validate_muon_array",
    "validate_shape",
    "validate_finite",
    "validate_weights",
    "validate_id_integer",
    "validate_bounds",
    "run_checks",
    "build_dataset_report",
    "process_pkl",
    "write_artifacts",
]
