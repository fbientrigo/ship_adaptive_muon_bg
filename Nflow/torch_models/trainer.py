"""Deterministic weighted-NLL trainer for the affine coupling flow."""

from __future__ import annotations

import hashlib
import io
import time
from typing import List, Optional

import numpy as np
import torch

from Nflow.interfaces import FIT_STATUS_FAILED, FIT_STATUS_OK, FitResult


class NonFiniteLossError(RuntimeError):
    """Raised when training or validation produces a non-finite value."""


def _validated_weight(value: Optional[np.ndarray], n: int, name: str) -> np.ndarray:
    weight = np.ones(n, dtype=np.float64) if value is None else np.asarray(value, dtype=np.float64)
    if weight.shape != (n,):
        raise ValueError("{} must have shape ({},)".format(name, n))
    if not np.isfinite(weight).all() or np.any(weight < 0.0):
        raise ValueError("{} must be finite and nonnegative".format(name))
    if float(weight.sum()) <= 0.0:
        raise ValueError("{} total must be positive".format(name))
    return weight


def _weighted_nll(module, tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Compute a weighted negative log-likelihood.

    Parameters
    ----------
    module:
        Flow module exposing ``log_prob``.
    tensor:
        Normalized training rows.
    weight:
        Finite, non-negative row weights. They are not rescaled.

    Returns
    -------
    torch.Tensor
        ``sum(weight * nll) / sum(weight)``.
    """

    return torch.sum(weight * -module.log_prob(tensor)) / torch.sum(weight)


def _state_hash(module) -> str:
    buffer = io.BytesIO()
    torch.save(module.state_dict(), buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _component_metrics(module, x, weight, labels, rare_id, prefix):
    with torch.no_grad():
        row_nll = -module.log_prob(x)
    out = {prefix + "_nll": float(torch.sum(weight * row_nll) / torch.sum(weight))}
    if labels is not None and rare_id is not None:
        rare = labels == int(rare_id)
        for name, mask in (("rare", rare), ("main", ~rare)):
            if bool(mask.any()):
                out[prefix + "_{}_nll".format(name)] = float(
                    torch.sum(weight[mask] * row_nll[mask]) / torch.sum(weight[mask])
                )
            else:
                out[prefix + "_{}_nll".format(name)] = None
    return out


def train_flow(
    estimator,
    x_train: np.ndarray,
    x_validation: Optional[np.ndarray],
    *,
    seed: int,
    sample_weight: Optional[np.ndarray] = None,
    validation_sample_weight: Optional[np.ndarray] = None,
    component_id: Optional[np.ndarray] = None,
    validation_component_id: Optional[np.ndarray] = None,
    rare_component_id: Optional[int] = None,
) -> FitResult:
    """Fit an affine flow with deterministic batches and weighted NLL.

    Weights retain their supplied values. Every loss is normalized only by the
    sum of weights in the rows being reported.
    """

    started = time.perf_counter()
    module, device, dtype = estimator._module, estimator.device, estimator.torch_dtype
    train = np.asarray(x_train, dtype=np.float64)
    if train.ndim != 2 or train.shape[1] != estimator.dimension or not np.isfinite(train).all():
        raise ValueError("x_train must be finite (n, {})".format(estimator.dimension))
    train_weight = _validated_weight(sample_weight, train.shape[0], "sample_weight")
    train_labels = None if component_id is None else np.asarray(component_id, dtype=np.int64)
    if train_labels is not None and train_labels.shape != (train.shape[0],):
        raise ValueError("component_id must have shape (n_train,)")
    train_tensor = torch.as_tensor(train, dtype=dtype, device=device)
    train_weight_tensor = torch.as_tensor(train_weight, dtype=dtype, device=device)
    train_labels_tensor = None if train_labels is None else torch.as_tensor(train_labels, device=device)

    val_tensor = val_weight_tensor = val_labels_tensor = None
    if x_validation is not None:
        val = np.asarray(x_validation, dtype=np.float64)
        if val.ndim != 2 or val.shape[1] != estimator.dimension or not np.isfinite(val).all():
            raise ValueError("x_validation must be finite (n, {})".format(estimator.dimension))
        val_weight = _validated_weight(validation_sample_weight, val.shape[0], "validation_sample_weight")
        val_labels = None if validation_component_id is None else np.asarray(validation_component_id, dtype=np.int64)
        if val_labels is not None and val_labels.shape != (val.shape[0],):
            raise ValueError("validation_component_id must have shape (n_validation,)")
        val_tensor = torch.as_tensor(val, dtype=dtype, device=device)
        val_weight_tensor = torch.as_tensor(val_weight, dtype=dtype, device=device)
        val_labels_tensor = None if val_labels is None else torch.as_tensor(val_labels, device=device)

    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    optimizer = torch.optim.Adam(
        module.parameters(), lr=estimator.learning_rate,
        weight_decay=estimator.weight_decay,
    )
    n = train_tensor.shape[0]
    batch_size = min(estimator.batch_size, n)
    history: List[dict] = []
    warnings: List[str] = []
    best_val = float("inf")
    best_state = {k: v.detach().clone() for k, v in module.state_dict().items()}
    best_step = 0
    stale = 0
    try:
        for epoch in range(estimator.max_epochs):
            module.train()
            permutation = torch.randperm(n, generator=generator, device=device)
            max_gradient_norm = 0.0
            for start in range(0, n, batch_size):
                idx = permutation[start:start + batch_size]
                optimizer.zero_grad()
                loss = _weighted_nll(module, train_tensor[idx], train_weight_tensor[idx])
                if not torch.isfinite(loss):
                    raise NonFiniteLossError("non-finite training loss at epoch {}".format(epoch))
                loss.backward()
                if estimator.grad_clip_norm is not None:
                    norm = torch.nn.utils.clip_grad_norm_(module.parameters(), estimator.grad_clip_norm)
                else:
                    norm = torch.sqrt(sum(
                        torch.sum(p.grad * p.grad) for p in module.parameters() if p.grad is not None
                    ))
                max_gradient_norm = max(max_gradient_norm, float(norm))
                optimizer.step()

            module.eval()
            record = {"step": epoch, "epoch": epoch + 1}
            record.update(_component_metrics(
                module, train_tensor, train_weight_tensor, train_labels_tensor,
                rare_component_id, "train",
            ))
            record["gradient_norm"] = max_gradient_norm
            record["max_abs_log_scale"] = float(module.max_abs_log_scale(train_tensor))
            record["weight_normalization"] = "sum_weights"
            record["train_weight_total"] = float(train_weight.sum())
            if val_tensor is not None:
                record.update(_component_metrics(
                    module, val_tensor, val_weight_tensor, val_labels_tensor,
                    rare_component_id, "validation",
                ))
                value = record["validation_nll"]
                if not np.isfinite(value):
                    raise NonFiniteLossError("non-finite validation loss at epoch {}".format(epoch))
                if value < best_val - 1e-9:
                    best_val, best_step, stale = value, epoch, 0
                    best_state = {k: v.detach().clone() for k, v in module.state_dict().items()}
                else:
                    stale += 1
            else:
                value = record["train_nll"]
                if value < best_val - 1e-9:
                    best_val, best_step = value, epoch
                    best_state = {k: v.detach().clone() for k, v in module.state_dict().items()}
            # Hash every epoch so the history is auditable; checkpoint_interval
            # remains the fixed persistence cadence for memorization runs.
            record["checkpoint_hash"] = _state_hash(module)
            history.append(record)
            if estimator.early_stopping and val_tensor is not None and stale >= estimator.patience:
                warnings.append("early stopped at epoch {} (patience {})".format(epoch, estimator.patience))
                break
    except NonFiniteLossError as exc:
        return FitResult(
            status=FIT_STATUS_FAILED, seed=int(seed), train_history=history,
            wall_time_seconds=time.perf_counter() - started, warnings=[str(exc)],
        )

    if not estimator.memorization_mode:
        module.load_state_dict(best_state)
    return FitResult(
        status=FIT_STATUS_OK, seed=int(seed), train_history=history,
        best_step=(estimator.max_epochs - 1 if estimator.memorization_mode else best_step),
        best_validation_nll=(None if val_tensor is None else (
            history[-1]["validation_nll"] if estimator.memorization_mode else best_val
        )),
        wall_time_seconds=time.perf_counter() - started, warnings=warnings,
    )
