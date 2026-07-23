"""D9 weighted-objective estimator: fixed_global_weight_normalization_v1 (§7).

D8 found that the historical D7 weighted training loop normalized each
minibatch by that minibatch's own weight sum::

    batch_loss = sum(w_i * nll_i) / sum(w_i)   # i in this minibatch only

That is a *self-normalized minibatch ratio*: its target shifts with whichever
rows happen to land in a batch, and is NOT an unbiased minibatch-gradient
estimator of the fixed, whole-split objective

    L(theta) = sum_i w_i * loss_i(theta) / sum_i w_i     (i over the full split)

This module implements the fixed-global-normalization estimator instead: the
denominator ``W_train = sum_i w_i`` is computed ONCE over the complete
declared training split and held fixed for every minibatch of that split.
Validation NLL is accumulated exactly as ``sum_i w_i * loss_i / sum_i w_i``
over the complete validation split (a single global ratio, not a per-batch
one -- there is only one "batch" for validation here by construction).

No utility multiplier is implemented or referenced anywhere in this module.
"""

from __future__ import annotations

import numpy as np

ESTIMATOR_NAME = "fixed_global_weight_normalization_v1"


def split_weight_total(weights: np.ndarray) -> float:
    """``W_train`` (or ``W_validation``): sum of weights over a COMPLETE split.

    Must be computed once per split, before any minibatch iteration, and
    reused as a fixed constant for every batch of that split.
    """

    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1:
        raise ValueError(f"expected a 1D weight array, got shape {weights.shape}")
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError(f"split weight total must be finite and positive, got {total}")
    return total


def weighted_batch_loss_numpy(nll_batch: np.ndarray, weight_batch: np.ndarray, *, n_train: int, w_train_total: float) -> float:
    """Reference (numpy) form of the fixed-global-normalization batch loss.

    ``batch_loss = (N_train / W_train) * mean_batch(w_i * nll_i)``

    This targets the same fixed full-split objective for every batch: its
    expectation under uniform minibatch sampling is ``L(theta)`` (up to the
    minibatch mean's own sampling variance), unlike the historical
    self-normalized-minibatch ratio, whose target changes with each batch's
    own weight sum.
    """

    nll_batch = np.asarray(nll_batch, dtype=np.float64)
    weight_batch = np.asarray(weight_batch, dtype=np.float64)
    if nll_batch.shape != weight_batch.shape:
        raise ValueError("nll_batch and weight_batch must have the same shape")
    return (float(n_train) / float(w_train_total)) * float(np.mean(weight_batch * nll_batch))


def weighted_batch_loss_torch(nll_batch, weight_batch, *, n_train: int, w_train_total: float):
    """Torch form of the same estimator, for use inside a training loop.

    ``nll_batch``/``weight_batch`` are 1D tensors for the current minibatch.
    Returns a scalar tensor with gradient connected to ``nll_batch``.
    """

    return (float(n_train) / float(w_train_total)) * (weight_batch * nll_batch).mean()


def weighted_validation_nll(nll_full: np.ndarray, weight_full: np.ndarray) -> float:
    """``sum_i w_i * nll_i / sum_i w_i`` over the COMPLETE validation split.

    This is a single global ratio computed once over the whole split -- never
    accumulated as a running per-minibatch average of per-minibatch ratios,
    which would silently reintroduce the historical self-normalized-minibatch
    estimator this module exists to replace.
    """

    nll_full = np.asarray(nll_full, dtype=np.float64)
    weight_full = np.asarray(weight_full, dtype=np.float64)
    if nll_full.shape != weight_full.shape:
        raise ValueError("nll_full and weight_full must have the same shape")
    w_total = split_weight_total(weight_full)
    return float(np.sum(weight_full * nll_full) / w_total)


def unweighted_batch_loss_numpy(nll_batch: np.ndarray) -> float:
    """Row-empirical unweighted objective: plain mean NLL over the batch."""

    return float(np.mean(np.asarray(nll_batch, dtype=np.float64)))


def unweighted_validation_nll(nll_full: np.ndarray) -> float:
    return float(np.mean(np.asarray(nll_full, dtype=np.float64)))
