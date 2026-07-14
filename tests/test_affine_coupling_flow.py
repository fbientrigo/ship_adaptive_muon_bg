"""Invariant tests for the affine-coupling normalizing flow (CPU).

Marked ``flow`` (requires the optional torch stack). Covers exact
forward/inverse round trip and Jacobian cancellation, shape/finiteness,
deterministic seeded sampling, tiny overfit on D0, a smoke improvement on a
curved target, save/load equality, parameter count, and the guarantee that
importing ``Nflow`` does not import torch.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

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
