"""Validation for the local post-shield muon PKL contract.

Checks fail fast and loud with typed errors (see :mod:`errors`). Bounds checks
are *units-sanity* guards (catching e.g. mm-vs-m or MeV-vs-GeV mistakes); they
are **not** physics-validity claims.
"""

from __future__ import annotations

import numpy as np

from . import schema
from .errors import BoundsError, FiniteError, IdError, ShapeError, WeightError

# Default units-sanity bounds (absolute value), as ``(low, high)`` on each column
# group. These are deliberately loose: they catch unit mistakes, not physics.
DEFAULT_BOUNDS = {
    "momentum_abs_max": 1.0e4,  # |p| component, GeV/c
    "position_abs_max": 1.0e3,  # |position| component, m
}


def validate_shape(array):
    """Validate the array is 2-D ``(N, 8)`` with ``N >= 1``."""
    if array.ndim != 2:
        raise ShapeError(f"expected a 2-D array, got ndim={array.ndim}")
    n_rows, n_cols = array.shape
    if n_cols != schema.N_COLUMNS:
        raise ShapeError(
            f"expected {schema.N_COLUMNS} columns {schema.COLUMNS}, got {n_cols}"
        )
    if n_rows < 1:
        raise ShapeError("expected at least one row, got an empty array")


def validate_finite(array):
    """Validate there are no ``NaN`` or ``inf`` values anywhere."""
    if not np.all(np.isfinite(array)):
        bad = int(np.count_nonzero(~np.isfinite(array)))
        raise FiniteError(f"array contains {bad} non-finite (NaN/inf) value(s)")


def validate_weights(array, *, allow_zero=False):
    """Validate the weight column ``w`` is finite and positive.

    Parameters
    ----------
    allow_zero : bool
        If ``True``, accept ``w == 0`` (policy choice). Default rejects ``w <= 0``.
    """
    w = array[:, schema.COLUMN_INDEX[schema.WEIGHT_COLUMN]]
    if not np.all(np.isfinite(w)):
        raise WeightError("weight column 'w' contains non-finite values")
    threshold = 0.0
    bad_mask = (w < threshold) if allow_zero else (w <= threshold)
    if np.any(bad_mask):
        bound = ">= 0" if allow_zero else "> 0"
        raise WeightError(
            f"weight column 'w' must be {bound}; found {int(np.count_nonzero(bad_mask))} violation(s)"
        )


def validate_id_integer(array):
    """Validate the PDG ``id`` column is integer-valued (within float tolerance)."""
    ids = array[:, schema.COLUMN_INDEX[schema.ID_COLUMN]]
    if not np.all(np.isfinite(ids)):
        raise IdError("id column contains non-finite values")
    if not np.all(ids == np.rint(ids)):
        raise IdError("id column 'id' must be integer-valued (PDG codes)")


def validate_bounds(array, bounds=None):
    """Validate momentum/position columns lie within units-sanity bounds."""
    cfg = {**DEFAULT_BOUNDS, **(bounds or {})}

    mom = array[:, schema.column_indices(schema.MOMENTUM_COLUMNS)]
    if np.any(np.abs(mom) > cfg["momentum_abs_max"]):
        raise BoundsError(
            f"momentum component exceeds units-sanity bound {cfg['momentum_abs_max']} GeV/c"
        )

    pos = array[:, schema.column_indices(schema.POSITION_COLUMNS)]
    if np.any(np.abs(pos) > cfg["position_abs_max"]):
        raise BoundsError(
            f"position component exceeds units-sanity bound {cfg['position_abs_max']} m"
        )


def validate_muon_array(array, *, bounds=None, allow_zero_weight=False):
    """Run the full contract validation, raising the first typed failure.

    Order: shape -> finite -> weights -> id -> bounds. Shape and finiteness are
    checked first because later checks assume a well-formed, finite array.
    """
    validate_shape(array)
    validate_finite(array)
    validate_weights(array, allow_zero=allow_zero_weight)
    validate_id_integer(array)
    validate_bounds(array, bounds=bounds)
    return array


def run_checks(array, *, bounds=None, allow_zero_weight=False):
    """Run all checks without raising; return an ordered list of outcomes.

    Used by the dataset report so validation results are recorded as data rather
    than only surfaced as exceptions.
    """
    checks = [
        ("shape", lambda: validate_shape(array)),
        ("finite", lambda: validate_finite(array)),
        ("weights_positive", lambda: validate_weights(array, allow_zero=allow_zero_weight)),
        ("id_integer", lambda: validate_id_integer(array)),
        ("units_bounds", lambda: validate_bounds(array, bounds=bounds)),
    ]
    outcomes = []
    for name, fn in checks:
        try:
            fn()
            outcomes.append({"check": name, "passed": True, "detail": None})
        except Exception as exc:  # noqa: BLE001 - recorded as report data
            outcomes.append(
                {"check": name, "passed": False, "detail": f"{type(exc).__name__}: {exc}"}
            )
    return outcomes
