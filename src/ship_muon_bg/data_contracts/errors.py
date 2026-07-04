"""Typed errors for the data contract.

Validation failures raise these; they are never silently coerced. Keeping
distinct types lets callers (and tests) discriminate failure modes precisely.
"""

from __future__ import annotations


class DataContractError(ValueError):
    """Base class for all data-contract validation failures."""


class ShapeError(DataContractError):
    """Array is not a 2-D ``(N, 8)`` float array with ``N >= 1``."""


class FiniteError(DataContractError):
    """Array contains ``NaN`` or ``inf``."""


class WeightError(DataContractError):
    """Weight column ``w`` has non-finite or non-positive values."""


class IdError(DataContractError):
    """PDG ``id`` column is not integer-valued."""


class BoundsError(DataContractError):
    """A column violates its configured units-sanity bounds."""


class LoaderError(DataContractError):
    """The PKL payload could not be loaded into a usable NumPy array."""
