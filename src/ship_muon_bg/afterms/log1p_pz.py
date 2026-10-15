"""cartesian_log1p_pz_v0: logarithmic pz preprocessing variant (Phase C, §6.3).

Maps ``(px, py, pz, x, y) -> (px, py, log1p(pz/s_pz), x, y)``. Domain
``pz >= 0``. Unlike the existing ``cartesian_logpz_v0`` feature view (plain
``log(pz)``, which is undefined at ``pz == 0``), this variant is finite at
``pz == 0`` by construction.

Forward:   u = log1p(pz / s_pz)
Inverse:   pz = s_pz * expm1(u)
Jacobian:  log|du/dpz| = -log(s_pz + pz)   (only where pz >= 0)
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

VIEW_ID = "cartesian_log1p_pz_v0"
DEFAULT_S_PZ_GEV = 1.0


class NegativePzError(ValueError):
    """Raised when pz < 0 rows are present and the caller did not opt to count them."""


def validate_pz_domain(pz, *, raise_on_negative=True):
    """Return ``(negative_count, negative_mask)`` for ``pz < 0``.

    Never clips or drops rows. If ``raise_on_negative`` is True (default) and
    any pz < 0 exists, raises ``NegativePzError`` instead of silently
    proceeding -- callers that want to record-and-block instead of raising
    should pass ``raise_on_negative=False``.
    """
    pz = np.asarray(pz, dtype=np.float64)
    mask = pz < 0.0
    count = int(np.count_nonzero(mask))
    if count and raise_on_negative:
        raise NegativePzError(
            "{} row(s) have pz < 0; cartesian_log1p_pz_v0 requires pz >= 0 "
            "and never clips or drops rows silently".format(count)
        )
    return count, mask


def forward_log1p_pz(pz, *, s_pz=DEFAULT_S_PZ_GEV):
    """u_pz = log1p(pz / s_pz). Caller must ensure pz >= 0 (see validate_pz_domain)."""
    pz = np.asarray(pz, dtype=np.float64)
    validate_pz_domain(pz, raise_on_negative=True)
    return np.log1p(pz / float(s_pz))


def inverse_log1p_pz(u_pz, *, s_pz=DEFAULT_S_PZ_GEV):
    """pz = s_pz * expm1(u_pz)."""
    u_pz = np.asarray(u_pz, dtype=np.float64)
    return float(s_pz) * np.expm1(u_pz)


def forward_log_abs_det_jacobian(pz, *, s_pz=DEFAULT_S_PZ_GEV):
    """log|du_pz/dpz| = -log(s_pz + pz), for pz >= 0."""
    pz = np.asarray(pz, dtype=np.float64)
    validate_pz_domain(pz, raise_on_negative=True)
    return -np.log(float(s_pz) + pz)


def transform_rows(rows, *, s_pz=DEFAULT_S_PZ_GEV):
    """rows: (N, 5) [px, py, pz, x, y] -> (N, 5) [px, py, u_pz, x, y]."""
    rows = np.asarray(rows, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != 5:
        raise ValueError("expected rows of shape (N, 5) [px, py, pz, x, y]")
    px, py, pz, x, y = rows.T
    u_pz = forward_log1p_pz(pz, s_pz=s_pz)
    return np.column_stack((px, py, u_pz, x, y))


def inverse_transform_rows(features, *, s_pz=DEFAULT_S_PZ_GEV):
    """features: (N, 5) [px, py, u_pz, x, y] -> (N, 5) [px, py, pz, x, y]."""
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or features.shape[1] != 5:
        raise ValueError("expected features of shape (N, 5) [px, py, u_pz, x, y]")
    px, py, u_pz, x, y = features.T
    pz = inverse_log1p_pz(u_pz, s_pz=s_pz)
    return np.column_stack((px, py, pz, x, y))


def manifest(*, s_pz=DEFAULT_S_PZ_GEV):
    return {
        "view_id": VIEW_ID,
        "s_pz_gev": float(s_pz),
        "forward_definition": "log1p(pz / s_pz)",
        "inverse_definition": "s_pz * expm1(u_pz)",
        "log_abs_det_jacobian_definition": "-log(s_pz + pz)",
        "domain": "pz >= 0",
        "physical_input_columns": ["px", "py", "pz", "x", "y"],
        "feature_output_columns": ["px", "py", "log1p_pz", "x", "y"],
    }


def config_hash(*, s_pz=DEFAULT_S_PZ_GEV):
    encoded = json.dumps(
        manifest(s_pz=s_pz), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
