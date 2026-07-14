"""Exact, invertible NumPy-only transforms for controlled targets (v0).

These transforms turn the exact Gaussian-mixture base targets (the D0-D2
family) into curved / heteroscedastic / skewed targets (D3-D5) while keeping
everything analytically exact:

- an exact analytic inverse (no numerical root finding);
- an exact analytic ``log|det J|`` (checked against finite differences in the
  tests);
- ``float64`` throughout, deterministic, no input mutation;
- ``pz`` (physical column index 2) is **never** modified by any transform, so
  the positive-``pz`` policy stays auditable and no feature view is
  privileged.

All transforms operate in canonical physical coordinates
``[px, py, pz, x, y]`` and are dimensionally documented via explicit
coordinate scales in their manifests.

These are numerical benchmark transforms, not SHiP physics. See
``docs/contracts/controlled_targets_v0.md`` (D3-D5 section).
"""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import numpy as np

from .controlled_targets import (
    ControlledTargetDomainError,
    ControlledTargetShapeError,
    N_PHYSICAL_DIMS,
)

# Physical column layout: [px, py, pz, x, y]. pz is index 2 and is frozen.
_PZ_INDEX = 2


def _validate_physical(array: Any, *, name: str = "physical") -> np.ndarray:
    """Coerce/validate an ``(N, 5)`` finite float64 physical array (copy)."""

    try:
        out = np.array(array, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ControlledTargetShapeError(
            "{} is not coercible to float64: {}".format(name, exc)
        ) from exc
    if out.ndim != 2 or out.shape[0] < 1 or out.shape[1] != N_PHYSICAL_DIMS:
        raise ControlledTargetShapeError(
            "expected {} shape (N, {}) with N >= 1, got {}".format(
                name, N_PHYSICAL_DIMS, out.shape
            )
        )
    if not np.isfinite(out).all():
        raise ControlledTargetDomainError(
            "{} contains NaN or inf".format(name)
        )
    return out


class ExactTransform:
    """Base class for exact, invertible, ``pz``-preserving target transforms.

    Subclasses implement :meth:`forward`, :meth:`inverse`,
    :meth:`forward_log_abs_det_jacobian`, :meth:`inverse_log_abs_det_jacobian`
    and :meth:`manifest`. The base class provides input validation helpers and
    a shared ``pz``-invariance assertion used by the test suite.
    """

    transform_id: str = "identity"

    def forward(self, base_physical: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def inverse(self, transformed_physical: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def forward_log_abs_det_jacobian(self, base_physical: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def inverse_log_abs_det_jacobian(
        self, transformed_physical: np.ndarray
    ) -> np.ndarray:
        raise NotImplementedError

    def manifest(self) -> Dict[str, Any]:
        raise NotImplementedError

    @staticmethod
    def _assert_pz_unchanged(base: np.ndarray, transformed: np.ndarray) -> None:
        if not np.allclose(base[:, _PZ_INDEX], transformed[:, _PZ_INDEX], rtol=0, atol=0):
            raise ControlledTargetDomainError(
                "transform illegally modified pz (physical column index 2)"
            )


class TriangularBananaTransform(ExactTransform):
    """Smooth triangular ("banana") coupling with unit Jacobian determinant.

    Using normalized ``px`` (``u_px / scale_px``) as the root coordinate:

    - ``py`` receives a quadratic function of normalized ``px``;
    - ``x`` receives a linear function of normalized ``px``;
    - ``pz``, ``px`` and ``y`` are unchanged.

    Because ``px`` is unchanged and each dependent output depends on itself
    with coefficient 1 (plus already-known ``px``), the Jacobian is unit
    lower-triangular and ``log|det J| = 0`` exactly, in both directions.
    """

    transform_id = "triangular_banana_v0"

    def __init__(
        self,
        *,
        scale_px: float,
        scale_py: float,
        scale_x: float,
        curvature: float,
        shear: float,
    ) -> None:
        for label, value in (
            ("scale_px", scale_px),
            ("scale_py", scale_py),
            ("scale_x", scale_x),
        ):
            v = float(value)
            if not np.isfinite(v) or v <= 0.0:
                raise ValueError("{} must be finite and > 0".format(label))
        self.scale_px = float(scale_px)
        self.scale_py = float(scale_py)
        self.scale_x = float(scale_x)
        self.curvature = float(curvature)
        self.shear = float(shear)

    def forward(self, base_physical: np.ndarray) -> np.ndarray:
        u = _validate_physical(base_physical, name="base_physical")
        t = u.copy()
        norm_px = u[:, 0] / self.scale_px
        t[:, 1] = u[:, 1] + self.curvature * (norm_px ** 2) * self.scale_py
        t[:, 3] = u[:, 3] + self.shear * norm_px * self.scale_x
        self._assert_pz_unchanged(u, t)
        return t

    def inverse(self, transformed_physical: np.ndarray) -> np.ndarray:
        t = _validate_physical(transformed_physical, name="transformed_physical")
        u = t.copy()
        norm_px = t[:, 0] / self.scale_px  # px is unchanged, so t_px == u_px
        u[:, 1] = t[:, 1] - self.curvature * (norm_px ** 2) * self.scale_py
        u[:, 3] = t[:, 3] - self.shear * norm_px * self.scale_x
        return u

    def forward_log_abs_det_jacobian(self, base_physical: np.ndarray) -> np.ndarray:
        u = _validate_physical(base_physical, name="base_physical")
        return np.zeros(u.shape[0], dtype=np.float64)

    def inverse_log_abs_det_jacobian(
        self, transformed_physical: np.ndarray
    ) -> np.ndarray:
        t = _validate_physical(transformed_physical, name="transformed_physical")
        return np.zeros(t.shape[0], dtype=np.float64)

    def manifest(self) -> Dict[str, Any]:
        return {
            "transform_id": self.transform_id,
            "kind": "triangular_banana",
            "root_coordinate": "px",
            "pz_preserved": True,
            "unit_determinant": True,
            "coordinate_scales": {
                "scale_px": self.scale_px,
                "scale_py": self.scale_py,
                "scale_x": self.scale_x,
            },
            "curvature": self.curvature,
            "shear": self.shear,
            "forward_definition": [
                "px' = px",
                "py' = py + curvature * (px/scale_px)^2 * scale_py",
                "pz' = pz",
                "x'  = x + shear * (px/scale_px) * scale_x",
                "y'  = y",
            ],
        }


class SinhArcsinhSkewTransform(ExactTransform):
    """Elementwise sinh-arcsinh skew on ``px, py, x, y`` (``pz`` untouched).

    For a coordinate with scale ``s`` and skew ``b``:

    ``t = s * sinh(b + asinh(u / s))``

    which is strictly monotonic with a closed-form inverse
    ``u = s * sinh(asinh(t / s) - b)`` and analytic log-derivative
    ``log|dt/du| = log cosh(b + asinh(z)) - 0.5 log(1 + z^2)`` where
    ``z = u / s``. The derivative is never singular (``cosh >= 1``). ``pz``'s
    skew coefficient is fixed to ``0``, which makes the map the exact identity
    on ``pz`` with zero log-Jacobian contribution.
    """

    transform_id = "sinh_arcsinh_skew_v0"

    def __init__(self, *, scales: Sequence[float], skews: Sequence[float]) -> None:
        scales = np.asarray(scales, dtype=np.float64)
        skews = np.asarray(skews, dtype=np.float64)
        if scales.shape != (N_PHYSICAL_DIMS,) or skews.shape != (N_PHYSICAL_DIMS,):
            raise ValueError("scales and skews must have shape (5,)")
        if not (np.isfinite(scales).all() and np.all(scales > 0.0)):
            raise ValueError("scales must be finite and strictly positive")
        if not np.isfinite(skews).all():
            raise ValueError("skews must be finite")
        if skews[_PZ_INDEX] != 0.0:
            raise ValueError("pz skew must be exactly 0.0 (pz is preserved)")
        self.scales = scales
        self.skews = skews
        # Coordinates with zero skew are the exact identity; treat them as such
        # (bitwise value preservation, exactly-zero log-Jacobian contribution)
        # rather than relying on ``sinh(asinh(z)) == z`` in floating point.
        self._active = skews != 0.0

    def forward(self, base_physical: np.ndarray) -> np.ndarray:
        u = _validate_physical(base_physical, name="base_physical")
        z = u / self.scales
        t = u.copy()
        active = self._active
        t[:, active] = (
            self.scales[active]
            * np.sinh(self.skews[active] + np.arcsinh(z[:, active]))
        )
        self._assert_pz_unchanged(u, t)
        return t

    def inverse(self, transformed_physical: np.ndarray) -> np.ndarray:
        t = _validate_physical(transformed_physical, name="transformed_physical")
        z_out = t / self.scales
        u = t.copy()
        active = self._active
        u[:, active] = (
            self.scales[active]
            * np.sinh(np.arcsinh(z_out[:, active]) - self.skews[active])
        )
        return u

    def forward_log_abs_det_jacobian(self, base_physical: np.ndarray) -> np.ndarray:
        u = _validate_physical(base_physical, name="base_physical")
        z = u / self.scales
        arg = self.skews + np.arcsinh(z)
        # log cosh(arg) - 0.5 log(1 + z^2) per coordinate; identity (zero-skew)
        # coordinates contribute exactly 0.
        log_term = _log_cosh(arg) - 0.5 * np.log1p(z * z)
        log_term[:, ~self._active] = 0.0
        return np.sum(log_term, axis=1)

    def inverse_log_abs_det_jacobian(
        self, transformed_physical: np.ndarray
    ) -> np.ndarray:
        return -self.forward_log_abs_det_jacobian(
            self.inverse(transformed_physical)
        )

    def manifest(self) -> Dict[str, Any]:
        return {
            "transform_id": self.transform_id,
            "kind": "sinh_arcsinh_skew",
            "pz_preserved": True,
            "coordinate_scales": self.scales.tolist(),
            "skews": self.skews.tolist(),
            "forward_definition": "t_i = s_i * sinh(b_i + asinh(u_i / s_i))",
        }


def _log_cosh(x: np.ndarray) -> np.ndarray:
    """Numerically stable ``log(cosh(x))``."""

    abs_x = np.abs(x)
    # log cosh(x) = |x| + log((1 + exp(-2|x|)) / 2)
    return abs_x + np.log1p(np.exp(-2.0 * abs_x)) - np.log(2.0)


class ComposedTransform(ExactTransform):
    """Compose exact transforms as ``forward = transforms[-1] o ... o [0]``.

    ``forward`` applies the transforms left to right; ``inverse`` applies their
    inverses right to left. Log-Jacobians accumulate additively, evaluated at
    the intermediate coordinates each stage sees.
    """

    transform_id = "composed_v0"

    def __init__(self, transforms: Sequence[ExactTransform]) -> None:
        transforms = tuple(transforms)
        if not transforms:
            raise ValueError("ComposedTransform requires at least one transform")
        self.transforms: Tuple[ExactTransform, ...] = transforms

    def forward(self, base_physical: np.ndarray) -> np.ndarray:
        current = _validate_physical(base_physical, name="base_physical")
        for transform in self.transforms:
            current = transform.forward(current)
        return current

    def inverse(self, transformed_physical: np.ndarray) -> np.ndarray:
        current = _validate_physical(
            transformed_physical, name="transformed_physical"
        )
        for transform in reversed(self.transforms):
            current = transform.inverse(current)
        return current

    def forward_log_abs_det_jacobian(self, base_physical: np.ndarray) -> np.ndarray:
        current = _validate_physical(base_physical, name="base_physical")
        total = np.zeros(current.shape[0], dtype=np.float64)
        for transform in self.transforms:
            total = total + transform.forward_log_abs_det_jacobian(current)
            current = transform.forward(current)
        return total

    def inverse_log_abs_det_jacobian(
        self, transformed_physical: np.ndarray
    ) -> np.ndarray:
        # log|det d(base)/d(transformed)| = -forward_ladj at the mapped base.
        base = self.inverse(transformed_physical)
        return -self.forward_log_abs_det_jacobian(base)

    def manifest(self) -> Dict[str, Any]:
        return {
            "transform_id": self.transform_id,
            "kind": "composed",
            "pz_preserved": all(
                t.manifest().get("pz_preserved", False) for t in self.transforms
            ),
            "stages": [t.manifest() for t in self.transforms],
        }


def numerical_forward_log_abs_det_jacobian(
    transform: ExactTransform, base_physical: np.ndarray, *, eps: float = 1e-6
) -> np.ndarray:
    """Central finite-difference reference for ``log|det J_forward|``.

    Used only by the test suite to validate the analytic determinants.
    """

    base = _validate_physical(base_physical, name="base_physical")
    n_rows = base.shape[0]
    jac = np.zeros((n_rows, N_PHYSICAL_DIMS, N_PHYSICAL_DIMS), dtype=np.float64)
    for j in range(N_PHYSICAL_DIMS):
        plus = base.copy()
        minus = base.copy()
        plus[:, j] += eps
        minus[:, j] -= eps
        deriv = (transform.forward(plus) - transform.forward(minus)) / (2.0 * eps)
        jac[:, :, j] = deriv
    sign, logabsdet = np.linalg.slogdet(jac)
    return logabsdet
