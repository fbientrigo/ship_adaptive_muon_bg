"""Mini-batch NLL trainer for the affine-coupling flow.

Deterministic (explicit seed, no hidden global seeding), with validation NLL,
early stopping, best-checkpoint restoration, configurable gradient clipping,
and a hard failure on non-finite loss. Records per-epoch history plus device
and dtype. Returns a :class:`~Nflow.interfaces.FitResult`.
"""

from __future__ import annotations

import time
from typing import Any, List, Optional

import numpy as np
import torch

from Nflow.interfaces import FIT_STATUS_FAILED, FIT_STATUS_OK, FitResult


class NonFiniteLossError(RuntimeError):
    """Raised when the training or validation loss becomes non-finite."""


def _nll(module, tensor: torch.Tensor) -> torch.Tensor:
    return -module.log_prob(tensor).mean()


def train_flow(
    estimator,
    x_train: np.ndarray,
    x_validation: Optional[np.ndarray],
    *,
    seed: int,
) -> FitResult:
    start = time.perf_counter()
    module = estimator._module
    device = estimator.device
    dtype = estimator.torch_dtype

    x_train = np.asarray(x_train, dtype=np.float64)
    if x_train.ndim != 2 or x_train.shape[1] != estimator.dimension:
        raise ValueError("x_train must be (n, {})".format(estimator.dimension))
    train_tensor = torch.as_tensor(x_train, dtype=dtype, device=device)
    val_tensor = (
        torch.as_tensor(np.asarray(x_validation, dtype=np.float64), dtype=dtype, device=device)
        if x_validation is not None
        else None
    )

    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))

    optimizer = torch.optim.Adam(
        module.parameters(),
        lr=estimator.learning_rate,
        weight_decay=estimator.weight_decay,
    )

    n = train_tensor.shape[0]
    batch_size = min(estimator.batch_size, n)
    history: List[dict] = []
    warnings_list: List[str] = []

    best_val = float("inf")
    best_state = {k: v.detach().clone() for k, v in module.state_dict().items()}
    best_step = 0
    epochs_without_improvement = 0

    def eval_nll(tensor) -> float:
        module.eval()
        with torch.no_grad():
            value = float(_nll(module, tensor).item())
        return value

    try:
        for epoch in range(estimator.max_epochs):
            module.train()
            perm = torch.randperm(n, generator=generator, device=device)
            epoch_losses = []
            for start_idx in range(0, n, batch_size):
                idx = perm[start_idx : start_idx + batch_size]
                batch = train_tensor[idx]
                optimizer.zero_grad()
                loss = _nll(module, batch)
                if not torch.isfinite(loss):
                    raise NonFiniteLossError(
                        "non-finite training loss at epoch {}".format(epoch)
                    )
                loss.backward()
                if estimator.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        module.parameters(), estimator.grad_clip_norm
                    )
                optimizer.step()
                epoch_losses.append(float(loss.item()))

            train_nll = float(np.mean(epoch_losses))
            record: dict = {"step": epoch, "train_nll": train_nll}
            if val_tensor is not None:
                val_nll = eval_nll(val_tensor)
                if not np.isfinite(val_nll):
                    raise NonFiniteLossError(
                        "non-finite validation loss at epoch {}".format(epoch)
                    )
                record["validation_nll"] = val_nll
                if val_nll < best_val - 1e-9:
                    best_val = val_nll
                    best_state = {
                        k: v.detach().clone() for k, v in module.state_dict().items()
                    }
                    best_step = epoch
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
            else:
                # no validation: track best by train loss
                if train_nll < best_val - 1e-9:
                    best_val = train_nll
                    best_state = {
                        k: v.detach().clone() for k, v in module.state_dict().items()
                    }
                    best_step = epoch
            history.append(record)

            if (
                val_tensor is not None
                and epochs_without_improvement >= estimator.patience
            ):
                warnings_list.append(
                    "early stopped at epoch {} (patience {})".format(
                        epoch, estimator.patience
                    )
                )
                break
    except NonFiniteLossError as exc:
        return FitResult(
            status=FIT_STATUS_FAILED,
            seed=int(seed),
            train_history=history,
            wall_time_seconds=time.perf_counter() - start,
            warnings=[str(exc)],
        )

    # restore best checkpoint
    module.load_state_dict(best_state)
    best_validation_nll = None if val_tensor is None else best_val

    return FitResult(
        status=FIT_STATUS_OK,
        seed=int(seed),
        train_history=history,
        best_step=best_step,
        best_validation_nll=best_validation_nll,
        wall_time_seconds=time.perf_counter() - start,
        warnings=warnings_list,
    )
