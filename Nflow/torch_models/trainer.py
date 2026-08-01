"""Deterministic weighted-NLL trainer for the affine coupling flow."""

from __future__ import annotations

import hashlib
import io
import time
from typing import Any, Callable, Dict, List, Mapping, Optional

import numpy as np
import torch

from Nflow.interfaces import FIT_STATUS_FAILED, FIT_STATUS_OK, FitResult

# Absolute tolerance for the arm-C invariant sum_h m_h * w_h == batch_size
# (Horvitz-Thompson weights built from realized integer counts sum exactly to
# the batch size; this only ever fires on a real construction bug).
_HT_WEIGHT_SUM_TOLERANCE = 1e-6


def _log_prob(module, tensor: torch.Tensor, condition: Optional[torch.Tensor] = None):
    """Preserve one-argument compatibility for legacy test doubles/models."""

    return module.log_prob(tensor) if condition is None else module.log_prob(tensor, condition)


class NonFiniteLossError(RuntimeError):
    """Raised when training or validation produces a non-finite value."""


class HTInvariantError(RuntimeError):
    """Raised when a fixed-composition batch's weight sum deviates from B.

    This is the A4 fixed-denominator invariant (sum_h m_h * w_h == batch_size)
    failing at runtime -- a technical construction failure, never silently
    absorbed into a biased loss.
    """


def _validated_weight(value: Optional[np.ndarray], n: int, name: str) -> np.ndarray:
    weight = np.ones(n, dtype=np.float64) if value is None else np.asarray(value, dtype=np.float64)
    if weight.shape != (n,):
        raise ValueError("{} must have shape ({},)".format(name, n))
    if not np.isfinite(weight).all() or np.any(weight < 0.0):
        raise ValueError("{} must be finite and nonnegative".format(name))
    if float(weight.sum()) <= 0.0:
        raise ValueError("{} total must be positive".format(name))
    return weight


def _weighted_nll(
    module,
    tensor: torch.Tensor,
    weight: torch.Tensor,
    condition: Optional[torch.Tensor] = None,
) -> torch.Tensor:
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

    return torch.sum(weight * -_log_prob(module, tensor, condition)) / torch.sum(weight)


def _fixed_denominator_nll(
    module,
    tensor: torch.Tensor,
    weight: torch.Tensor,
    denominator: float,
    condition: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fixed-composition Horvitz-Thompson loss: never normalizes by sum(weight).

    Parameters
    ----------
    module:
        Flow module exposing ``log_prob``.
    tensor:
        Normalized rows for one fixed-composition minibatch.
    weight:
        Per-row Horvitz-Thompson correction weights ``w_h = p_h * B / m_h``.
    denominator:
        The fixed, known batch size ``B`` (never the random ``sum(weight)``;
        assumption A4).

    Returns
    -------
    torch.Tensor
        ``sum(weight * nll) / denominator``.
    """

    return torch.sum(weight * -_log_prob(module, tensor, condition)) / denominator


def _derived_seed(seed: int, *salts: int) -> int:
    seq = np.random.SeedSequence([int(seed), *[int(s) for s in salts]])
    return int(seq.generate_state(1)[0])


def _stratum_gradient_norms(
    module,
    x: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    rare_id: int,
    condition: Optional[torch.Tensor] = None,
) -> Dict[str, Optional[float]]:
    """Diagnostic rare/main gradient norms on one batch.

    Computed via separate backward passes that never call ``optimizer.step``
    and never alter optimizer state; gradients are cleared before returning
    so the next real training step starts clean. Never affects clipping.
    """

    norms: Dict[str, Optional[float]] = {}
    rare_mask = labels == int(rare_id)
    for name, mask in (("rare", rare_mask), ("main", ~rare_mask)):
        if not bool(mask.any()):
            norms[name] = None
            continue
        module.zero_grad(set_to_none=True)
        sub_condition = None if condition is None else condition[mask]
        sub_loss = torch.mean(-_log_prob(module, x[mask], sub_condition))
        sub_loss.backward()
        norm = torch.sqrt(sum(
            torch.sum(p.grad * p.grad) for p in module.parameters() if p.grad is not None
        ))
        norms[name] = float(norm)
    module.zero_grad(set_to_none=True)
    return norms


def _state_hash(module) -> str:
    buffer = io.BytesIO()
    torch.save(module.state_dict(), buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _component_metrics(
    module, x, weight, labels, rare_id, prefix, condition=None
):
    with torch.no_grad():
        row_nll = -_log_prob(module, x, condition)
    value = float(torch.sum(weight * row_nll) / torch.sum(weight))
    feature_prefix = "feature_space_" + prefix
    out = {feature_prefix + "_nll": value, prefix + "_nll": value}
    if labels is not None and rare_id is not None:
        rare = labels == int(rare_id)
        for name, mask in (("rare", rare), ("main", ~rare)):
            if bool(mask.any()):
                value = float(
                    torch.sum(weight[mask] * row_nll[mask]) / torch.sum(weight[mask])
                )
                out[feature_prefix + "_{}_nll".format(name)] = value
                out[prefix + "_{}_nll".format(name)] = value
            else:
                out[feature_prefix + "_{}_nll".format(name)] = None
                out[prefix + "_{}_nll".format(name)] = None
    return out


def train_flow(
    estimator,
    x_train: Optional[np.ndarray],
    x_validation: Optional[np.ndarray],
    *,
    seed: int,
    sample_weight: Optional[np.ndarray] = None,
    validation_sample_weight: Optional[np.ndarray] = None,
    component_id: Optional[np.ndarray] = None,
    validation_component_id: Optional[np.ndarray] = None,
    rare_component_id: Optional[int] = None,
    batch_plan: Optional[Any] = None,
    loss_normalization: Optional[str] = None,
    condition: Optional[np.ndarray] = None,
    validation_condition: Optional[np.ndarray] = None,
    epoch_sampler: Optional[Callable[[int], Mapping[str, Any]]] = None,
) -> FitResult:
    """Fit an affine flow with deterministic batches and weighted NLL.

    ``batch_plan=None`` and ``epoch_sampler=None`` (the default) is the exact
    legacy path: minibatches are permutation slices of the fixed training
    partition ``x_train`` and the loss is normalized by the sum of weights in
    the rows being reported (``_weighted_nll``, ``loss_normalization ==
    "sum_weights"``).

    ``batch_plan`` (a ``density_lab.sampling.MinibatchPlan``) selects the
    fixed-composition Horvitz-Thompson path (arm C, Issue #17): minibatches
    are the plan's explicit precomputed index arrays and the loss
    (``_fixed_denominator_nll``) is normalized by the fixed, known batch size
    -- never by the (here deterministic, but never trusted as such)
    ``sum(weight)``. The same precomputed plan is replayed identically every
    epoch (only cosmetic within-batch row order varies by epoch); see the
    rare-aware estimator contract for why this does not affect per-step
    unbiasedness.

    ``epoch_sampler`` (arm D, direct per-epoch sampling) is a callable
    ``epoch_sampler(epoch) -> {"x": ndarray, "condition": Optional[ndarray],
    "metadata": dict}`` invoked once per epoch to draw that epoch's entire
    training set fresh (e.g. macro-balanced physical-nominal draws with
    replacement). ``x_train`` is ignored (may be ``None``) in this mode; the
    loss is always the ordinary unweighted NLL (``sample_weight`` must be
    ``None``), and ``batch_plan`` must not also be supplied. Each epoch's
    ``metadata`` is recorded verbatim onto that epoch's history record under
    ``"epoch_sampler_metadata"``. This makes each step's gradient an unbiased
    estimator of the macro-balanced population gradient
    ``E_epoch_draws[gradient] = (1/2) grad L_13 + (1/2) grad L_-13``; it does
    *not* imply the final trained parameters are themselves an unbiased
    estimator of any macro target -- SGD over a nonconvex loss has no such
    guarantee even with per-step-unbiased gradients.
    """

    started = time.perf_counter()
    module, device, dtype = estimator._module, estimator.device, estimator.torch_dtype

    if epoch_sampler is not None:
        if batch_plan is not None:
            raise ValueError("epoch_sampler is mutually exclusive with batch_plan")
        if sample_weight is not None:
            raise ValueError(
                "epoch_sampler implies the ordinary unweighted NLL; sample_weight must be None"
            )
        train_tensor = train_weight_tensor = train_labels_tensor = train_condition_tensor = None
    else:
        train = np.asarray(x_train, dtype=np.float64)
        if train.ndim != 2 or train.shape[1] != estimator.dimension or not np.isfinite(train).all():
            raise ValueError("x_train must be finite (n, {})".format(estimator.dimension))
        train_weight = _validated_weight(sample_weight, train.shape[0], "sample_weight")
        train_labels = None if component_id is None else np.asarray(component_id, dtype=np.int64)
        if train_labels is not None and train_labels.shape != (train.shape[0],):
            raise ValueError("component_id must have shape (n_train,)")
        train_tensor = torch.as_tensor(train, dtype=dtype, device=device)
        train_condition_tensor = estimator._condition_to_tensor(
            condition, train.shape[0]
        )
        train_weight_tensor = torch.as_tensor(train_weight, dtype=dtype, device=device)
        train_labels_tensor = None if train_labels is None else torch.as_tensor(train_labels, device=device)

    if batch_plan is not None:
        if loss_normalization is not None and loss_normalization != batch_plan.denominator_mode:
            raise ValueError(
                "loss_normalization {!r} is inconsistent with batch_plan.denominator_mode "
                "{!r}".format(loss_normalization, batch_plan.denominator_mode)
            )
        if train_labels_tensor is None or rare_component_id is None:
            raise ValueError(
                "batch_plan requires component_id and rare_component_id (per-stratum "
                "diagnostics and the fixed-composition loss both need labelled rows)"
            )

    val_tensor = val_weight_tensor = val_labels_tensor = val_condition_tensor = None
    if x_validation is not None:
        val = np.asarray(x_validation, dtype=np.float64)
        if val.ndim != 2 or val.shape[1] != estimator.dimension or not np.isfinite(val).all():
            raise ValueError("x_validation must be finite (n, {})".format(estimator.dimension))
        val_weight = _validated_weight(validation_sample_weight, val.shape[0], "validation_sample_weight")
        val_labels = None if validation_component_id is None else np.asarray(validation_component_id, dtype=np.int64)
        if val_labels is not None and val_labels.shape != (val.shape[0],):
            raise ValueError("validation_component_id must have shape (n_validation,)")
        val_tensor = torch.as_tensor(val, dtype=dtype, device=device)
        val_condition_tensor = estimator._condition_to_tensor(
            validation_condition, val.shape[0]
        )
        val_weight_tensor = torch.as_tensor(val_weight, dtype=dtype, device=device)
        val_labels_tensor = None if val_labels is None else torch.as_tensor(val_labels, device=device)

    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    optimizer = torch.optim.Adam(
        module.parameters(), lr=estimator.learning_rate,
        weight_decay=estimator.weight_decay,
    )
    if train_tensor is not None:
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
            if epoch_sampler is not None:
                # -- arm D: fresh per-epoch direct sampling, ordinary unweighted NLL --
                draw = epoch_sampler(int(epoch))
                x_epoch = np.asarray(draw["x"], dtype=np.float64)
                if (
                    x_epoch.ndim != 2
                    or x_epoch.shape[1] != estimator.dimension
                    or not np.isfinite(x_epoch).all()
                ):
                    raise ValueError(
                        "epoch_sampler(epoch) 'x' must be finite (n, {})".format(estimator.dimension)
                    )
                epoch_condition = draw.get("condition")
                epoch_metadata = dict(draw.get("metadata", {}))
                train_tensor = torch.as_tensor(x_epoch, dtype=dtype, device=device)
                train_condition_tensor = estimator._condition_to_tensor(
                    epoch_condition, x_epoch.shape[0]
                )
                train_weight_tensor = torch.ones(x_epoch.shape[0], dtype=dtype, device=device)
                train_labels_tensor = None
                n = train_tensor.shape[0]
                batch_size = min(estimator.batch_size, n)
                epoch_seed = _derived_seed(seed, epoch, 0x64395F)
                epoch_generator = torch.Generator(device=device)
                epoch_generator.manual_seed(epoch_seed)
                permutation = torch.randperm(n, generator=epoch_generator, device=device)
                max_gradient_norm = 0.0
                for start in range(0, n, batch_size):
                    idx = permutation[start:start + batch_size]
                    optimizer.zero_grad()
                    batch_condition = (
                        None
                        if train_condition_tensor is None
                        else train_condition_tensor[idx]
                    )
                    loss = _weighted_nll(
                        module,
                        train_tensor[idx],
                        train_weight_tensor[idx],
                        batch_condition,
                    )
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
                plan_fields = {"epoch_sampler_metadata": epoch_metadata}
            elif batch_plan is None:
                # -- legacy path: exact permutation-sliced, sum-weights loss --
                permutation = torch.randperm(n, generator=generator, device=device)
                max_gradient_norm = 0.0
                for start in range(0, n, batch_size):
                    idx = permutation[start:start + batch_size]
                    optimizer.zero_grad()
                    batch_condition = (
                        None
                        if train_condition_tensor is None
                        else train_condition_tensor[idx]
                    )
                    loss = _weighted_nll(
                        module,
                        train_tensor[idx],
                        train_weight_tensor[idx],
                        batch_condition,
                    )
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
                plan_fields: Dict[str, Any] = {}
            else:
                # -- arm C: explicit fixed-composition plan, fixed-B loss --
                max_gradient_norm = 0.0
                rare_counts: List[int] = []
                batch_weight_sums: List[float] = []
                ht_losses: List[float] = []
                unweighted_losses: List[float] = []
                last_x = last_weight = last_labels = None
                for step in range(batch_plan.steps_per_epoch):
                    shuffle_seed = _derived_seed(seed, epoch, step)
                    idx_np = batch_plan.step_indices(step, shuffle_seed=shuffle_seed)
                    weight_np = batch_plan.step_weights(step, shuffle_seed=shuffle_seed)
                    weight_sum = float(weight_np.sum())
                    if abs(weight_sum - batch_plan.batch_size) > _HT_WEIGHT_SUM_TOLERANCE:
                        raise HTInvariantError(
                            "fixed-composition weight sum {} deviates from batch_size {} "
                            "by more than tolerance {} at epoch {} step {}".format(
                                weight_sum, batch_plan.batch_size,
                                _HT_WEIGHT_SUM_TOLERANCE, epoch, step,
                            )
                        )
                    batch_weight_sums.append(weight_sum)
                    idx = torch.as_tensor(idx_np, device=device, dtype=torch.long)
                    weight_tensor = torch.as_tensor(weight_np, dtype=dtype, device=device)
                    x_batch = train_tensor[idx]
                    condition_batch = (
                        None
                        if train_condition_tensor is None
                        else train_condition_tensor[idx]
                    )
                    labels_batch = train_labels_tensor[idx]
                    optimizer.zero_grad()
                    loss = _fixed_denominator_nll(
                        module,
                        x_batch,
                        weight_tensor,
                        float(batch_plan.batch_size),
                        condition_batch,
                    )
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
                    with torch.no_grad():
                        ht_losses.append(float(loss))
                        unweighted_losses.append(
                            float(
                                torch.mean(
                                    -_log_prob(module, x_batch, condition_batch)
                                )
                            )
                        )
                    rare_counts.append(int(torch.sum(labels_batch == int(rare_component_id))))
                    last_x, last_weight, last_labels, last_condition = (
                        x_batch,
                        weight_tensor,
                        labels_batch,
                        condition_batch,
                    )

                # Diagnostic per-stratum gradient norms: bounded cadence (once
                # per epoch, on the epoch's last batch), never updates
                # parameters or optimizer state, never affects clipping.
                stratum_norms = _stratum_gradient_norms(
                    module,
                    last_x,
                    last_weight,
                    last_labels,
                    rare_component_id,
                    last_condition,
                )
                plan_fields = {
                    "minibatch_rare_count_min": min(rare_counts),
                    "minibatch_rare_count_max": max(rare_counts),
                    "minibatch_rare_count_mean": float(np.mean(rare_counts)),
                    "batch_weight_sum_min": min(batch_weight_sums),
                    "batch_weight_sum_max": max(batch_weight_sums),
                    "steps": batch_plan.steps_per_epoch,
                    "dropped_steps": 0,
                    "horvitz_thompson_loss": float(np.mean(ht_losses)),
                    "unweighted_minibatch_mean_loss": float(np.mean(unweighted_losses)),
                    "rare_gradient_norm": stratum_norms["rare"],
                    "main_gradient_norm": stratum_norms["main"],
                    "gradient_norm_cadence": "per_epoch_last_step",
                }

            module.eval()
            record = {"step": epoch, "epoch": epoch + 1}
            record.update(_component_metrics(
                module, train_tensor, train_weight_tensor, train_labels_tensor,
                rare_component_id, "train", train_condition_tensor,
            ))
            record["gradient_norm"] = max_gradient_norm
            record["max_abs_log_scale"] = float(
                module.max_abs_log_scale(train_tensor, train_condition_tensor)
            )
            if epoch_sampler is not None:
                record["weight_normalization"] = "epoch_direct_sampling_unweighted"
            elif batch_plan is None:
                record["weight_normalization"] = "sum_weights"
            else:
                record["weight_normalization"] = batch_plan.denominator_mode
            record["train_weight_total"] = float(train_weight_tensor.sum())
            record.update(plan_fields)
            if val_tensor is not None:
                record.update(_component_metrics(
                    module, val_tensor, val_weight_tensor, val_labels_tensor,
                    rare_component_id, "validation", val_condition_tensor,
                ))
                value = record["feature_space_validation_nll"]
                if not np.isfinite(value):
                    raise NonFiniteLossError("non-finite validation loss at epoch {}".format(epoch))
                if value < best_val - 1e-9:
                    best_val, best_step, stale = value, epoch, 0
                    best_state = {k: v.detach().clone() for k, v in module.state_dict().items()}
                else:
                    stale += 1
            else:
                value = record["feature_space_train_nll"]
                if value < best_val - 1e-9:
                    best_val, best_step = value, epoch
                    best_state = {k: v.detach().clone() for k, v in module.state_dict().items()}
            # Hash every epoch so the history is auditable. This is a
            # tensor-state-only hash (learned parameters), distinct from
            # AffineCouplingFlow.checkpoint_hash(), which also fingerprints
            # the functional config, init_seed and reconstructed
            # permutations at save time. checkpoint_interval must be 1:
            # only the final state is ever persisted as a checkpoint
            # artifact; this per-epoch value is an audit-only history, not
            # periodic checkpoint persistence.
            record["state_dict_hash"] = _state_hash(module)
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
            history[-1]["feature_space_validation_nll"] if estimator.memorization_mode else best_val
        )),
        wall_time_seconds=time.perf_counter() - started, warnings=warnings,
    )
