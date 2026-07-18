"""Tested affine-coupling normalizing flow (RealNVP-style) in PyTorch.

Not adapted from ``Nflow/legacy`` -- a clean, small implementation:

- standard-normal base;
- alternating binary masks over the five dimensions;
- exact forward (latent->data) and inverse (data->latent) with exact total
  log-determinant;
- MLP conditioner with configurable blocks / width / depth / activation;
- bounded log-scale (``tanh * max_log_scale``);
- configurable float32/float64;
- explicit device selection (cpu / cuda / auto);
- deterministic seeded sampling;
- checkpoint = ``state_dict`` + JSON-safe config, with a recorded hash.

Trained separately per PDG id (no charge conditioning). The public data
boundary is NumPy float64 arrays.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import nn

from Nflow.interfaces import FitResult

_ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
}


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device in ("cpu", "cuda"):
        return torch.device(device)
    raise ValueError("device must be 'cpu', 'cuda' or 'auto', got {!r}".format(device))


def _build_mlp(dim: int, width: int, depth: int, activation: str) -> nn.Sequential:
    if activation not in _ACTIVATIONS:
        raise ValueError(
            "unknown activation {!r}; expected one of {}".format(
                activation, sorted(_ACTIVATIONS)
            )
        )
    act = _ACTIVATIONS[activation]
    layers: list = [nn.Linear(dim, width), act()]
    for _ in range(depth):
        layers += [nn.Linear(width, width), act()]
    layers += [nn.Linear(width, 2 * dim)]
    network = nn.Sequential(*layers)
    for layer in network[:-1]:
        if isinstance(layer, nn.Linear):
            if activation == "relu":
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
                with torch.no_grad():
                    layer.weight.mul_(0.25)
            elif activation == "silu":
                nn.init.xavier_uniform_(layer.weight)
                with torch.no_grad():
                    layer.weight.mul_(0.25)
            nn.init.zeros_(layer.bias)
    return network


class _CouplingLayer(nn.Module):
    """One affine coupling layer with a fixed binary mask."""

    def __init__(
        self,
        dim: int,
        mask: torch.Tensor,
        width: int,
        depth: int,
        activation: str,
        max_log_scale: float,
    ) -> None:
        super().__init__()
        self.register_buffer("mask", mask)
        self.net = _build_mlp(dim, width, depth, activation)
        self.max_log_scale = float(max_log_scale)
        # zero-init the final layer so the flow starts near identity.
        final = self.net[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def _s_t(self, conditioned_on: torch.Tensor):
        h = self.net(conditioned_on * self.mask)
        s_raw, t = h.chunk(2, dim=-1)
        keep = 1.0 - self.mask
        s = torch.tanh(s_raw) * self.max_log_scale * keep
        t = t * keep
        return s, t

    def forward_map(self, z: torch.Tensor):
        """latent -> data; returns (x, log|det dx/dz|)."""

        s, t = self._s_t(z)
        x = z * torch.exp(s) + t
        return x, s.sum(dim=-1)

    def inverse_map(self, x: torch.Tensor):
        """data -> latent; returns (z, log|det dz/dx|)."""

        s, t = self._s_t(x)
        z = (x - t) * torch.exp(-s)
        return z, -s.sum(dim=-1)


class _FlowModule(nn.Module):
    """A stack of alternating-mask coupling layers over a standard normal base."""

    def __init__(
        self,
        *,
        dim: int,
        number_of_blocks: int,
        hidden_width: int,
        hidden_depth: int,
        activation: str,
        max_log_scale: float,
        mixing_mode: str,
        seed: int,
    ) -> None:
        super().__init__()
        self.dim = dim
        if mixing_mode not in ("alternating_only", "fixed_random_permutation"):
            raise ValueError("unknown mixing_mode {!r}".format(mixing_mode))
        layers = []
        permutations = []
        rng = np.random.default_rng(int(seed))
        for i in range(number_of_blocks):
            pattern = (torch.arange(dim) + i) % 2
            mask = pattern.to(torch.get_default_dtype())
            layers.append(
                _CouplingLayer(
                    dim, mask, hidden_width, hidden_depth, activation, max_log_scale
                )
            )
            permutation = (
                rng.permutation(dim) if mixing_mode == "fixed_random_permutation"
                else np.arange(dim)
            )
            permutations.append(torch.as_tensor(permutation, dtype=torch.long))
        self.layers = nn.ModuleList(layers)
        self.mixing_mode = mixing_mode
        for index, permutation in enumerate(permutations):
            self.register_buffer("permutation_{}".format(index), permutation)
        self._log_base_const = 0.5 * dim * float(np.log(2.0 * np.pi))

    def inverse(self, x: torch.Tensor):
        z = x
        total = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        for index, layer in enumerate(self.layers):
            permutation = getattr(self, "permutation_{}".format(index))
            z = z[:, permutation]
            z, ld = layer.inverse_map(z)
            total = total + ld
        return z, total

    def forward(self, z: torch.Tensor):
        x = z
        for index in reversed(range(len(self.layers))):
            layer = self.layers[index]
            x, _ = layer.forward_map(x)
            permutation = getattr(self, "permutation_{}".format(index))
            inverse_permutation = torch.argsort(permutation)
            x = x[:, inverse_permutation]
        return x

    def permutations(self):
        return [
            getattr(self, "permutation_{}".format(i)).detach().cpu().tolist()
            for i in range(len(self.layers))
        ]

    def base_log_prob(self, z: torch.Tensor) -> torch.Tensor:
        return -self._log_base_const - 0.5 * torch.sum(z * z, dim=-1)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        z, log_det = self.inverse(x)
        return self.base_log_prob(z) + log_det

    def max_abs_log_scale(self, x: torch.Tensor) -> torch.Tensor:
        """Maximum absolute realized coupling log scale on ``x``."""

        value = torch.zeros((), dtype=x.dtype, device=x.device)
        current = x
        with torch.no_grad():
            for index, layer in enumerate(self.layers):
                permutation = getattr(self, "permutation_{}".format(index))
                current = current[:, permutation]
                scale, _ = layer._s_t(current)
                value = torch.maximum(value, torch.max(torch.abs(scale)))
                current, _ = layer.inverse_map(current)
        return value


class AffineCouplingFlow:
    """DensityEstimator wrapper around the affine-coupling flow module."""

    family = "affine_coupling"

    def __init__(
        self,
        *,
        dimension: int,
        device: str = "cpu",
        number_of_blocks: int = 4,
        hidden_width: int = 64,
        hidden_depth: int = 2,
        activation: str = "relu",
        max_log_scale: float = 3.0,
        dtype: str = "float32",
        learning_rate: float = 1e-3,
        batch_size: int = 256,
        max_epochs: int = 50,
        patience: int = 10,
        grad_clip_norm: Optional[float] = 5.0,
        weight_decay: float = 0.0,
        mixing_mode: str = "alternating_only",
        memorization_mode: bool = False,
        dropout: float = 0.0,
        data_augmentation: bool = False,
        input_noise_std: float = 0.0,
        early_stopping: bool = True,
        checkpoint_interval: int = 1,
    ) -> None:
        self.dimension = int(dimension)
        self.requested_device = device
        self.device = _resolve_device(device)
        self.number_of_blocks = int(number_of_blocks)
        self.hidden_width = int(hidden_width)
        self.hidden_depth = int(hidden_depth)
        self.activation = activation
        self.max_log_scale = float(max_log_scale)
        if dtype not in ("float32", "float64"):
            raise ValueError("dtype must be 'float32' or 'float64'")
        self.dtype_name = dtype
        self.torch_dtype = torch.float32 if dtype == "float32" else torch.float64
        self.learning_rate = float(learning_rate)
        self.batch_size = int(batch_size)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.grad_clip_norm = grad_clip_norm
        self.weight_decay = float(weight_decay)
        self.mixing_mode = mixing_mode
        self.memorization_mode = bool(memorization_mode)
        self.dropout = float(dropout)
        self.data_augmentation = bool(data_augmentation)
        self.input_noise_std = float(input_noise_std)
        self.early_stopping = bool(early_stopping)
        self.checkpoint_interval = int(checkpoint_interval)
        if self.checkpoint_interval < 1:
            raise ValueError("checkpoint_interval must be >= 1")
        if self.memorization_mode and any((
            self.weight_decay != 0.0, self.dropout != 0.0,
            self.data_augmentation, self.input_noise_std != 0.0,
            self.early_stopping,
        )):
            raise ValueError(
                "memorization_mode requires weight_decay=0, dropout=0, no "
                "augmentation/noise, and early_stopping=False"
            )
        self._module: Optional[_FlowModule] = None
        self._init_seed = 0
        self._build_module(seed=self._init_seed)

    def _build_module(self, *, seed: int = 0) -> None:
        # Seed Torch's global RNG before constructing so weight initialization
        # is deterministic in ``seed``: two identical specs with the same seed
        # start from identical hidden-layer weights (the final coupling layer is
        # zero-initialized). ``fit`` re-seeds from the run seed before training.
        self._init_seed = int(seed)
        prev = torch.get_default_dtype()
        torch.set_default_dtype(self.torch_dtype)
        torch.manual_seed(int(seed))
        try:
            module = _FlowModule(
                dim=self.dimension,
                number_of_blocks=self.number_of_blocks,
                hidden_width=self.hidden_width,
                hidden_depth=self.hidden_depth,
                activation=self.activation,
                max_log_scale=self.max_log_scale,
                mixing_mode=self.mixing_mode,
                seed=int(seed),
            )
        finally:
            torch.set_default_dtype(prev)
        self._module = module.to(self.device).to(self.torch_dtype)

    # -- estimator interface -------------------------------------------------

    def fit(
        self,
        x_train: np.ndarray,
        *,
        x_validation: Optional[np.ndarray] = None,
        seed: int = 0,
        sample_weight: Optional[np.ndarray] = None,
        validation_sample_weight: Optional[np.ndarray] = None,
        component_id: Optional[np.ndarray] = None,
        validation_component_id: Optional[np.ndarray] = None,
        rare_component_id: Optional[int] = None,
    ) -> FitResult:
        # Reset weights deterministically from the run seed so identical
        # RunSpecs with the same seed produce identical checkpoints/metrics.
        self._build_module(seed=int(seed))
        from .trainer import train_flow

        return train_flow(
            self, x_train, x_validation, seed=int(seed),
            sample_weight=sample_weight,
            validation_sample_weight=validation_sample_weight,
            component_id=component_id,
            validation_component_id=validation_component_id,
            rare_component_id=rare_component_id,
        )

    def _to_tensor(self, x: np.ndarray) -> torch.Tensor:
        array = np.asarray(x, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != self.dimension:
            raise ValueError(
                "expected (n, {}) array, got {}".format(self.dimension, array.shape)
            )
        return torch.as_tensor(array, dtype=self.torch_dtype, device=self.device)

    def log_prob(self, x: np.ndarray) -> np.ndarray:
        self._module.eval()
        with torch.no_grad():
            tensor = self._to_tensor(x)
            lp = self._module.log_prob(tensor)
        return lp.detach().cpu().numpy().astype(np.float64)

    def sample(self, n: int, *, seed: int) -> np.ndarray:
        self._module.eval()
        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(seed))
        with torch.no_grad():
            z = torch.randn(
                int(n),
                self.dimension,
                dtype=self.torch_dtype,
                device=self.device,
                generator=generator,
            )
            x = self._module(z)
        return x.detach().cpu().numpy().astype(np.float64)

    def parameter_count(self) -> int:
        return int(sum(p.numel() for p in self._module.parameters()))

    def config(self) -> Dict[str, Any]:
        return {
            "family": self.family,
            "dimension": self.dimension,
            "number_of_blocks": self.number_of_blocks,
            "hidden_width": self.hidden_width,
            "hidden_depth": self.hidden_depth,
            "activation": self.activation,
            "max_log_scale": self.max_log_scale,
            "dtype": self.dtype_name,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "grad_clip_norm": self.grad_clip_norm,
            "weight_decay": self.weight_decay,
            "mixing_mode": self.mixing_mode,
            "memorization_mode": self.memorization_mode,
            "dropout": self.dropout,
            "data_augmentation": self.data_augmentation,
            "input_noise_std": self.input_noise_std,
            "early_stopping": self.early_stopping,
            "checkpoint_interval": self.checkpoint_interval,
        }

    def manifest(self) -> Dict[str, Any]:
        manifest = dict(self.config())
        manifest.update(
            {
                "parameter_count": self.parameter_count(),
                "requested_device": self.requested_device,
                "actual_device": str(self.device),
                "torch_version": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
                "init_seed": self._init_seed,
                "permutations": self._module.permutations(),
                "permutation_log_abs_det": 0.0,
                "loss_normalization": "sum_weights",
            }
        )
        return manifest

    # -- checkpointing -------------------------------------------------------

    def state_dict_bytes(self) -> bytes:
        import io

        buffer = io.BytesIO()
        torch.save(self._module.state_dict(), buffer)
        return buffer.getvalue()

    def checkpoint_hash(self) -> str:
        return hashlib.sha256(self.state_dict_bytes()).hexdigest()

    def save(self, output_dir: Path) -> Dict[str, Any]:
        output_dir = Path(output_dir)
        checkpoint_dir = output_dir / "checkpoint"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self._module.state_dict(), checkpoint_dir / "state_dict.pt")
        (checkpoint_dir / "model_config.json").write_text(json.dumps({
            "config": self.config(), "requested_device": self.requested_device,
            "init_seed": self._init_seed,
            "permutations": self._module.permutations(),
        }, indent=2))
        checkpoint_hash = self.checkpoint_hash()
        (checkpoint_dir / "checkpoint_hash.txt").write_text(checkpoint_hash)
        return {
            "family": self.family,
            "checkpoint_dir": "checkpoint",
            "checkpoint_hash": checkpoint_hash,
        }

    @classmethod
    def load(cls, input_dir: Path, *, device: str = "cpu") -> "AffineCouplingFlow":
        checkpoint_dir = Path(input_dir) / "checkpoint"
        payload = json.loads((checkpoint_dir / "model_config.json").read_text())
        config = payload["config"]
        config.pop("family", None)
        model = cls(device=device, **config)
        model._build_module(seed=int(payload.get("init_seed", 0)))
        state = torch.load(
            checkpoint_dir / "state_dict.pt", map_location=model.device
        )
        model._module.load_state_dict(state)
        return model
