"""Deterministic weighted statistics for the local post-shield muon contract.

Physical MC weights (`w`) are not optional metadata (see
``docs/d9_direct_utility_tilt_sampling_v0.md``): any quantile-based threshold
used to define a diagnostic region must be computed with these weights, never
by treating rows as uniformly likely. This module is the single place that
does it.
"""

from __future__ import annotations

import numpy as np


class WeightedQuantileError(ValueError):
    """A weighted-quantile precondition was violated."""


def weighted_quantile(values, weights, q: float) -> float:
    """Deterministic weighted quantile, "inverted-CDF" (step) convention.

    Definition: sort rows by value ascending, breaking ties by original row
    index (a fully deterministic total order regardless of input order or
    duplicate values); the cumulative weight fraction is formed over that
    order; the returned quantile is the **smallest observed value** whose
    cumulative weight fraction is ``>= q``. This is a step-function
    convention -- the result is always one of the input values, never an
    interpolated point between two of them -- so it stays well-defined when
    most rows carry zero weight (as in this repository's MC weight column,
    which is legitimately zero for some rows).

    A row's own weight does not need to be positive for it to be selected:
    what matters is the cumulative weight reaching ``q`` at or before it, in
    sorted order. In particular the minimum value is always the ``q=0``
    quantile regardless of its own weight, and a maximum-valued row with zero
    weight contributes no mass and is skipped in favor of the largest value
    that does carry weight.

    Parameters
    ----------
    values : array-like, shape (N,)
        Finite, real-valued observations.
    weights : array-like, shape (N,)
        Non-negative, finite weights, same shape as ``values``. Zero weights
        are valid. Negative, NaN, or infinite weights raise.
    q : float
        Quantile level in ``[0, 1]``.

    Returns
    -------
    float
        The weighted quantile value (always one of the elements of ``values``).

    Raises
    ------
    WeightedQuantileError
        Shape mismatch, empty input, non-finite ``values``, negative or
        non-finite ``weights``, non-positive total weight, or ``q`` outside
        ``[0, 1]``.
    """

    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or weights.ndim != 1:
        raise WeightedQuantileError("values and weights must be 1-D")
    if values.shape != weights.shape:
        raise WeightedQuantileError(
            "values and weights must have the same shape, got {} vs {}".format(
                values.shape, weights.shape
            )
        )
    if values.shape[0] == 0:
        raise WeightedQuantileError("values/weights must be non-empty")
    if not np.isfinite(values).all():
        raise WeightedQuantileError("values must be finite")
    if not np.isfinite(weights).all():
        raise WeightedQuantileError("weights must be finite (no NaN/inf)")
    if np.any(weights < 0.0):
        raise WeightedQuantileError("weights must be non-negative")
    total = float(weights.sum())
    if total <= 0.0:
        raise WeightedQuantileError("total weight must be positive (got {})".format(total))
    if not isinstance(q, (int, float)) or isinstance(q, bool) or not (0.0 <= float(q) <= 1.0):
        raise WeightedQuantileError("q must be a number in [0, 1], got {!r}".format(q))

    order = np.lexsort((np.arange(values.shape[0]), values))
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative_fraction = np.cumsum(sorted_weights) / total
    idx = int(np.searchsorted(cumulative_fraction, float(q), side="left"))
    idx = min(idx, sorted_values.shape[0] - 1)
    return float(sorted_values[idx])


def weighted_prevalence(indicator, weights) -> float:
    """``sum(w_i * indicator_i) / sum(w_i)`` -- the weighted mean of a 0/1 array.

    Used for the physical weighted nominal prevalence ``p_0(B_toy)``. Raises
    the same :class:`WeightedQuantileError` preconditions as
    :func:`weighted_quantile` on the weights (non-negative, finite, positive
    total); ``indicator`` must be a finite 0/1 (or boolean) array.
    """

    indicator = np.asarray(indicator, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if indicator.shape != weights.shape:
        raise WeightedQuantileError(
            "indicator and weights must have the same shape, got {} vs {}".format(
                indicator.shape, weights.shape
            )
        )
    if not np.isfinite(weights).all():
        raise WeightedQuantileError("weights must be finite (no NaN/inf)")
    if np.any(weights < 0.0):
        raise WeightedQuantileError("weights must be non-negative")
    total = float(weights.sum())
    if total <= 0.0:
        raise WeightedQuantileError("total weight must be positive (got {})".format(total))
    if not np.isfinite(indicator).all() or not np.all((indicator == 0.0) | (indicator == 1.0)):
        raise WeightedQuantileError("indicator must be a finite 0/1 array")
    return float(np.sum(weights * indicator) / total)
