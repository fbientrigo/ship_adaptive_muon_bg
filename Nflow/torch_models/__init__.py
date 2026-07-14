"""PyTorch density models for the controlled density lab.

Importing this subpackage imports torch. It is therefore imported lazily by
the model registry (only when the ``affine_coupling`` family is requested) and
never from ``Nflow/__init__.py``, so the core import path stays NumPy-only.
"""

from __future__ import annotations

from .affine_coupling import AffineCouplingFlow

__all__ = ["AffineCouplingFlow"]
