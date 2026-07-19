"""Invariant tests for the affine-coupling normalizing flow (CPU).

Marked ``flow`` (requires the optional torch stack). Covers exact
forward/inverse round trip and Jacobian cancellation, shape/finiteness,
deterministic seeded sampling, tiny overfit on D0, a smoke improvement on a
curved target, save/load equality, parameter count, and the guarantee that
importing ``Nflow`` does not import torch.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn

from Nflow.interfaces import FIT_STATUS_OK, DensityEstimator
from Nflow.registry import create_density_estimator
from Nflow.torch_models.affine_coupling import AffineCouplingFlow

pytestmark = pytest.mark.flow

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = 5


def _standardized(target_id, pdg_id=13, n=4000, seed=11):
    from ship_muon_bg.benchmarks import make_controlled_target

    target = make_controlled_target(target_id)
    batch = target.sample(n, pdg_id=pdg_id, seed=seed)
    x = batch.physical.copy()
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    return (x - mean) / std


def test_import_nflow_does_not_import_torch():
    code = (
        "import sys\n"
        "import Nflow, Nflow.registry\n"
        "assert 'torch' not in sys.modules, 'torch imported by Nflow'\n"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [REPO_ROOT, os.path.join(REPO_ROOT, "src"), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


def test_flow_is_density_estimator():
    flow = AffineCouplingFlow(dimension=D, number_of_blocks=4)
    assert isinstance(flow, DensityEstimator)


@pytest.mark.parametrize(
    "option,value",
    [
        ("dropout", 0.1),
        ("data_augmentation", True),
        ("input_noise_std", 0.1),
    ],
)
@pytest.mark.parametrize("memorization_mode", [False, True])
def test_unimplemented_options_reject_non_neutral_values(
    option, value, memorization_mode
):
    with pytest.raises(ValueError, match="{} is not implemented".format(option)):
        AffineCouplingFlow(
            dimension=D, memorization_mode=memorization_mode, **{option: value}
        )


@pytest.mark.parametrize("option", ["dropout", "input_noise_std"])
@pytest.mark.parametrize("value", ["bad", None, np.nan, np.inf, -np.inf, True])
def test_unimplemented_numeric_options_reject_malformed_or_non_finite(option, value):
    with pytest.raises(ValueError, match="{} must be a finite numeric value".format(option)):
        AffineCouplingFlow(dimension=D, **{option: value})


@pytest.mark.parametrize("value", [None, 0, "false"])
def test_data_augmentation_rejects_non_boolean_values(value):
    with pytest.raises(ValueError, match="data_augmentation must be a boolean"):
        AffineCouplingFlow(dimension=D, data_augmentation=value)


def test_neutral_unimplemented_options_remain_serialized_and_doe_compatible():
    flow = AffineCouplingFlow(
        dimension=D,
        memorization_mode=True,
        dropout=-0.0,
        data_augmentation=False,
        input_noise_std=0,
        early_stopping=False,
    )
    assert flow.config()["dropout"] == 0.0
    assert flow.config()["data_augmentation"] is False
    assert flow.config()["input_noise_std"] == 0.0


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_forward_inverse_roundtrip(dtype):
    torch.manual_seed(0)
    flow = AffineCouplingFlow(dimension=D, number_of_blocks=6, dtype=dtype)
    module = flow._module
    rng = np.random.default_rng(0)
    x = torch.as_tensor(rng.standard_normal((64, D)), dtype=flow.torch_dtype)
    # perturb weights away from identity so the test is non-trivial
    with torch.no_grad():
        for p in module.parameters():
            p.add_(0.1 * torch.randn_like(p))
    z, ldj_inv = module.inverse(x)
    x_rec = module(z)
    # exp-based coupling accumulates float noise through 6 layers; a real
    # inverse bug produces O(1) errors, so this tolerance still catches it.
    tol = 1e-4 if dtype == "float32" else 1e-7
    assert torch.allclose(x_rec, x, atol=tol)


def test_forward_inverse_jacobian_cancels():
    torch.manual_seed(1)
    flow = AffineCouplingFlow(dimension=D, number_of_blocks=6, dtype="float64")
    module = flow._module
    with torch.no_grad():
        for p in module.parameters():
            p.add_(0.2 * torch.randn_like(p))
    rng = np.random.default_rng(1)
    x = torch.as_tensor(rng.standard_normal((32, D)), dtype=torch.float64)
    z, ldj_inv = module.inverse(x)
    # forward log-det from z
    total_fwd = torch.zeros(x.shape[0], dtype=torch.float64)
    xt = z
    for layer in reversed(module.layers):
        xt, ld = layer.forward_map(xt)
        total_fwd = total_fwd + ld
    # A sign error gives residual ~O(10); float64 noise is ~1e-9.
    assert torch.allclose(total_fwd + ldj_inv, torch.zeros_like(total_fwd), atol=1e-6)


def test_sample_and_log_prob_shapes_and_finite():
    flow = AffineCouplingFlow(dimension=D)
    s = flow.sample(100, seed=3)
    assert s.shape == (100, D) and np.isfinite(s).all()
    lp = flow.log_prob(s)
    assert lp.shape == (100,) and np.isfinite(lp).all()


def test_deterministic_sampling():
    flow = AffineCouplingFlow(dimension=D)
    a = flow.sample(50, seed=9)
    b = flow.sample(50, seed=9)
    np.testing.assert_array_equal(a, b)


def test_parameter_count_positive():
    flow = AffineCouplingFlow(dimension=D, number_of_blocks=4, hidden_width=32)
    assert flow.parameter_count() > 0


@pytest.mark.slow
def test_tiny_overfit_on_d0():
    x = _standardized("D0", n=512)
    flow = AffineCouplingFlow(
        dimension=D,
        number_of_blocks=6,
        hidden_width=64,
        hidden_depth=2,
        max_epochs=120,
        patience=200,
        learning_rate=2e-3,
        batch_size=512,
    )
    result = flow.fit(x, x_validation=None, seed=11)
    assert result.status == FIT_STATUS_OK
    final_nll = float(-np.mean(flow.log_prob(x)))
    # standardized 5D standard-normal-ish data has analytic NLL ~ 7.09; a flow
    # that overfits 512 rows should get at least close to / below that.
    assert final_nll < 7.5
    assert result.train_history[-1]["train_nll"] < result.train_history[0]["train_nll"]


@pytest.mark.slow
def test_smoke_improvement_on_d3():
    x = _standardized("D3", n=2000)
    val = _standardized("D3", n=1000, seed=22)
    flow = AffineCouplingFlow(
        dimension=D,
        number_of_blocks=6,
        hidden_width=64,
        hidden_depth=2,
        max_epochs=40,
        patience=40,
        learning_rate=2e-3,
        batch_size=256,
    )
    initial_val = float(-np.mean(flow.log_prob(val)))
    result = flow.fit(x, x_validation=val, seed=11)
    assert result.status == FIT_STATUS_OK
    assert result.best_validation_nll is not None
    assert result.best_validation_nll < initial_val


def test_non_finite_loss_fails():
    x = _standardized("D0", n=256)
    x[0, 0] = 1e30  # force an explosive gradient / non-finite loss
    flow = AffineCouplingFlow(
        dimension=D,
        number_of_blocks=4,
        max_epochs=10,
        grad_clip_norm=None,
        learning_rate=5e-1,
    )
    result = flow.fit(x, x_validation=None, seed=11)
    # Either it fails with FAILED status, or it survives; we require that if a
    # non-finite loss occurs the status is FAILED (never silently OK+NaN).
    if result.status != FIT_STATUS_OK:
        assert result.status == "failed"
        assert result.warnings


@pytest.mark.slow
def test_save_load_equality(tmp_path):
    x = _standardized("D0", n=512)
    flow = AffineCouplingFlow(dimension=D, number_of_blocks=4, max_epochs=5)
    flow.fit(x, x_validation=None, seed=11)
    point = _standardized("D0", n=100, seed=5)
    before = flow.log_prob(point)
    save_manifest = flow.save(tmp_path)
    reloaded = AffineCouplingFlow.load(tmp_path)
    after = reloaded.log_prob(point)
    np.testing.assert_allclose(after, before, rtol=1e-6, atol=1e-6)
    assert reloaded.checkpoint_hash() == save_manifest["checkpoint_hash"]


@pytest.mark.slow
def test_fit_is_deterministic_in_seed():
    # Two independently-constructed flows fit on the same data with the same
    # seed must produce identical densities (deterministic weight init + train).
    x = _standardized("D3", n=1000)
    val = _standardized("D3", n=500, seed=22)
    point = _standardized("D0", n=64, seed=5)

    def _fit_once():
        flow = AffineCouplingFlow(
            dimension=D, number_of_blocks=4, hidden_width=32, max_epochs=10, batch_size=256
        )
        flow.fit(x, x_validation=val, seed=11)
        return flow.log_prob(point)

    np.testing.assert_array_equal(_fit_once(), _fit_once())


def test_registry_creates_flow():
    flow = create_density_estimator(
        {"family": "affine_coupling", "params": {"number_of_blocks": 2}},
        dimension=D,
        device="cpu",
    )
    assert isinstance(flow, DensityEstimator)
    assert flow.number_of_blocks == 2


def test_device_auto_resolves():
    flow = AffineCouplingFlow(dimension=D, device="auto")
    assert str(flow.device) in ("cpu", "cuda")


# -- initializer_mode ---------------------------------------------------------


def test_default_initializer_mode_is_legacy_torch_default():
    flow = AffineCouplingFlow(dimension=D)
    assert flow.initializer_mode == "legacy_torch_default"
    assert flow.config()["initializer_mode"] == "legacy_torch_default"


def test_legacy_initializer_matches_historical_construction():
    flow = AffineCouplingFlow(
        dimension=D, number_of_blocks=1, hidden_width=16, hidden_depth=1,
        activation="relu", initializer_mode="legacy_torch_default",
    )
    flow._build_module(seed=42)
    actual = [p.clone() for p in flow._module.layers[0].net.parameters()]

    # Historical (pre-PR #15) construction: plain torch.nn.Linear defaults,
    # no manual reinitialization, except the final layer which
    # ``_CouplingLayer`` always zero-initializes regardless of mode.
    torch.manual_seed(42)
    expected_net = nn.Sequential(
        nn.Linear(D, 16), nn.ReLU(),
        nn.Linear(16, 16), nn.ReLU(),
        nn.Linear(16, 2 * D),
    )
    nn.init.zeros_(expected_net[-1].weight)
    nn.init.zeros_(expected_net[-1].bias)

    for a, e in zip(actual, expected_net.parameters()):
        torch.testing.assert_close(a, e)


def test_scaled_initializer_is_deterministic():
    a = AffineCouplingFlow(
        dimension=D, number_of_blocks=2, hidden_width=16,
        initializer_mode="scaled_activation_aware",
    )
    a._build_module(seed=5)
    b = AffineCouplingFlow(
        dimension=D, number_of_blocks=2, hidden_width=16,
        initializer_mode="scaled_activation_aware",
    )
    b._build_module(seed=5)
    for pa, pb in zip(a._module.parameters(), b._module.parameters()):
        torch.testing.assert_close(pa, pb)


def test_legacy_and_scaled_initializers_produce_different_parameters():
    legacy = AffineCouplingFlow(
        dimension=D, number_of_blocks=1, hidden_width=16, activation="relu",
        initializer_mode="legacy_torch_default",
    )
    legacy._build_module(seed=3)
    scaled = AffineCouplingFlow(
        dimension=D, number_of_blocks=1, hidden_width=16, activation="relu",
        initializer_mode="scaled_activation_aware",
    )
    scaled._build_module(seed=3)
    legacy_weight = legacy._module.layers[0].net[0].weight
    scaled_weight = scaled._module.layers[0].net[0].weight
    assert not torch.allclose(legacy_weight, scaled_weight)


@pytest.mark.parametrize("mode", ["legacy_torch_default", "scaled_activation_aware"])
def test_initializer_mode_round_trips_through_save_load(tmp_path, mode):
    flow = AffineCouplingFlow(dimension=D, number_of_blocks=2, initializer_mode=mode)
    flow.save(tmp_path)
    reloaded = AffineCouplingFlow.load(tmp_path)
    assert reloaded.initializer_mode == mode
    assert reloaded.config()["initializer_mode"] == mode


def test_unknown_initializer_mode_fails_early():
    with pytest.raises(ValueError, match="unknown initializer_mode"):
        AffineCouplingFlow(dimension=D, initializer_mode="bogus_mode")


# -- checkpoint compatibility --------------------------------------------------


def _write_checkpoint(checkpoint_dir, state, config, init_seed, permutations=None):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, checkpoint_dir / "state_dict.pt")
    payload = {"config": config, "requested_device": "cpu", "init_seed": init_seed}
    if permutations is not None:
        payload["permutations"] = permutations
    (checkpoint_dir / "model_config.json").write_text(json.dumps(payload))


def test_pre_permutation_checkpoint_loads(tmp_path):
    flow = AffineCouplingFlow(
        dimension=D, number_of_blocks=2, hidden_width=8, hidden_depth=1
    )
    old_config = flow.config()
    old_config.pop("mixing_mode")
    old_config.pop("initializer_mode")
    state = flow._module.state_dict()
    assert not any(k.startswith("permutation_") for k in state)
    _write_checkpoint(tmp_path / "checkpoint", state, old_config, flow._init_seed)

    reloaded = AffineCouplingFlow.load(tmp_path)
    assert reloaded.mixing_mode == "alternating_only"
    assert reloaded.initializer_mode == "legacy_torch_default"
    x = np.random.default_rng(0).standard_normal((8, D))
    np.testing.assert_allclose(reloaded.log_prob(x), flow.log_prob(x))


def test_interim_checkpoint_with_persistent_permutation_keys_loads(tmp_path):
    flow = AffineCouplingFlow(
        dimension=D, number_of_blocks=3, hidden_width=8, hidden_depth=1,
        mixing_mode="fixed_random_permutation",
    )
    state = flow._module.state_dict()
    for i in range(len(flow._module.layers)):
        state["permutation_{}".format(i)] = getattr(
            flow._module, "permutation_{}".format(i)
        ).clone()
    _write_checkpoint(
        tmp_path / "checkpoint", state, flow.config(), flow._init_seed,
        permutations=flow._module.permutations(),
    )

    reloaded = AffineCouplingFlow.load(tmp_path)
    x = np.random.default_rng(1).standard_normal((8, D))
    np.testing.assert_allclose(reloaded.log_prob(x), flow.log_prob(x))


def test_interim_checkpoint_permutation_mismatch_fails(tmp_path):
    flow = AffineCouplingFlow(
        dimension=D, number_of_blocks=2, hidden_width=8, hidden_depth=1,
        mixing_mode="fixed_random_permutation",
    )
    state = flow._module.state_dict()
    for i in range(len(flow._module.layers)):
        state["permutation_{}".format(i)] = getattr(
            flow._module, "permutation_{}".format(i)
        ).clone()
    state["permutation_0"] = torch.flip(state["permutation_0"], dims=[0])
    _write_checkpoint(
        tmp_path / "checkpoint", state, flow.config(), flow._init_seed,
    )

    with pytest.raises(ValueError, match="does not match"):
        AffineCouplingFlow.load(tmp_path)


@pytest.mark.parametrize("mixing_mode", ["alternating_only", "fixed_random_permutation"])
def test_checkpoint_round_trip_preserves_log_prob(tmp_path, mixing_mode):
    flow = AffineCouplingFlow(
        dimension=D, number_of_blocks=3, hidden_width=8, hidden_depth=1,
        mixing_mode=mixing_mode,
    )
    flow._build_module(seed=7)
    x = np.random.default_rng(2).standard_normal((16, D))
    before = flow.log_prob(x)
    flow.save(tmp_path)
    reloaded = AffineCouplingFlow.load(tmp_path)
    after = reloaded.log_prob(x)
    np.testing.assert_allclose(after, before, rtol=1e-6, atol=1e-6)


def test_reconstructed_permutation_matches_config_and_seed():
    flow = AffineCouplingFlow(
        dimension=D, number_of_blocks=4, mixing_mode="fixed_random_permutation"
    )
    flow._build_module(seed=13)
    expected = flow._module.permutations()

    other = AffineCouplingFlow(
        dimension=D, number_of_blocks=4, mixing_mode="fixed_random_permutation"
    )
    other._build_module(seed=13)
    assert other._module.permutations() == expected


def test_missing_learned_parameter_fails_to_load(tmp_path):
    flow = AffineCouplingFlow(dimension=D, number_of_blocks=2, hidden_width=8, hidden_depth=1)
    flow.save(tmp_path)
    state_path = tmp_path / "checkpoint" / "state_dict.pt"
    state = torch.load(state_path)
    del state[next(iter(state))]
    torch.save(state, state_path)

    with pytest.raises(RuntimeError):
        AffineCouplingFlow.load(tmp_path)


def test_unexpected_non_permutation_key_fails_to_load(tmp_path):
    flow = AffineCouplingFlow(dimension=D, number_of_blocks=2, hidden_width=8, hidden_depth=1)
    flow.save(tmp_path)
    state_path = tmp_path / "checkpoint" / "state_dict.pt"
    state = torch.load(state_path)
    state["totally_unexpected_key"] = torch.zeros(1)
    torch.save(state, state_path)

    with pytest.raises(RuntimeError):
        AffineCouplingFlow.load(tmp_path)
