"""Local post-shield muon data contracts (v0).

Enforces the ``(N, 8)`` ``[px, py, pz, x, y, z, id, w]`` contract for trusted
local legacy muon files: load, validate, hash, deterministically split, record
train-only normalization metadata, and construct explicit invertible density
feature views. Pure Python + NumPy; imports neither FairShip nor ROOT.
"""

from __future__ import annotations

from . import schema
from .errors import (
    BoundsError,
    DataContractError,
    FeatureViewConfigError,
    FeatureViewDomainError,
    FeatureViewError,
    FeatureViewShapeError,
    FiniteError,
    IdError,
    LoaderError,
    ShapeError,
    WeightError,
)
from .feature_views import (
    CARTESIAN_LOGPZ_FEATURES,
    CARTESIAN_LOGPZ_VIEW_ID,
    FEATURE_VIEW_EXPERIMENT_ID,
    FEATURE_VIEW_SCHEMA_VERSION,
    IDENTITY_CARTESIAN_FEATURES,
    IDENTITY_CARTESIAN_VIEW_ID,
    N_DENSITY_FEATURES,
    PHYSICAL_STATE_COLUMNS,
    SLOPE_LOGPZ_FEATURES,
    SLOPE_LOGPZ_VIEW_ID,
    SUPPORTED_FEATURE_VIEW_IDS,
    FeatureView,
    feature_view_experiment_manifest,
)
from .hashing import dataset_hash
from .loader import load_muon_pkl
from .normalization import apply_normalization, fit_normalization
from .report import build_dataset_report, process_pkl, write_artifacts
from .splitting import make_split
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
    "FeatureViewError",
    "FeatureViewConfigError",
    "FeatureViewShapeError",
    "FeatureViewDomainError",
    "FeatureView",
    "feature_view_experiment_manifest",
    "FEATURE_VIEW_SCHEMA_VERSION",
    "FEATURE_VIEW_EXPERIMENT_ID",
    "IDENTITY_CARTESIAN_VIEW_ID",
    "CARTESIAN_LOGPZ_VIEW_ID",
    "SLOPE_LOGPZ_VIEW_ID",
    "SUPPORTED_FEATURE_VIEW_IDS",
    "PHYSICAL_STATE_COLUMNS",
    "IDENTITY_CARTESIAN_FEATURES",
    "CARTESIAN_LOGPZ_FEATURES",
    "SLOPE_LOGPZ_FEATURES",
    "N_DENSITY_FEATURES",
    "load_muon_pkl",
    "load_muon_npz",
    "representative_subset",
    "save_subset_npz",
    "save_subset_pkl_gz",
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
