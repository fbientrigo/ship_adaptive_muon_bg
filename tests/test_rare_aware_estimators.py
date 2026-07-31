"""Rare-aware fixed-composition Horvitz-Thompson minibatch estimator (Issue #17).

Section order mirrors the implementation plan:

1. Compatibility pins -- must stay green through every later step. These
   values were captured from the unmodified base commit
   ``4c2abe8913130a35a3cbf00fbec4b81d2ae961ab`` and must never change.
2. Planner unit tests.
3. Estimator property tests (collapse identity, divergence, algebra,
   permutation invariance, determinism).
4. Integration tests across all four arms.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

from ship_muon_bg.density_lab.artifacts import derive_run_id
from ship_muon_bg.density_lab.config import ExperimentConfig, canonical_hash
from ship_muon_bg.density_lab.doe import generate_blocked_maximin_lhs

CONFIG_ROOT = "configs/density_lab/doe_v0/"

# --- Section 1: compatibility pins ------------------------------------------
#
# Captured once, before any implementation change, from the exact base
# commit. If any of these move, config/run identity has drifted and
# implementation must stop (BLOCKED_CONFIG_IDENTITY_DRIFT).

_PINNED_EXPERIMENT_CONFIG_HASH = {
    "d3_memorization_control_v0.json": "177f36cb1494c7709a6946367e5812d3fff069e6194d35f36f5865130f02d255",
    "d5_memorization_matrix_v0.json": "c719e223bd55e01038c9312ad4531eabc08fa927f066f85df464f3c9a467ed5b",
    "d5_memorization_smoke_v0.json": "4cdf0b73ac8f74241216cd8043b3548209dfd875a520668bd7a2d3ef42d1a719",
}

_PINNED_N_RUNS = {
    "d3_memorization_control_v0.json": 24,
    "d5_memorization_matrix_v0.json": 144,
    "d5_memorization_smoke_v0.json": 9,
}

_PINNED_RUN_IDS_HEAD_TAIL = {
    "d3_memorization_control_v0.json": (
        "D3_default_p13_identity_cartesian_v0_A_b08_w192_d2_lr0.00317796_mls2.84045_bs0128_seed1_476876eb284b",
        "D3_default_p13_identity_cartesian_v0_C_b09_w128_d2_lr0.000375055_mls1.65544_bs0256_seed1_ee60e3957a52",
    ),
    "d5_memorization_matrix_v0.json": (
        "D5_rare_1e-3_p13_identity_cartesian_v0_A_b08_w192_d2_lr0.00317796_mls2.84045_bs0128_seed1_1ce69cf48b24",
        "D5_rare_1e-3_p13_identity_cartesian_v0_C_b09_w128_d2_lr0.000375055_mls1.65544_bs0256_seed1_265f5a4b9cb3",
    ),
    "d5_memorization_smoke_v0.json": (
        "D5_rare_1e-3_p13_identity_cartesian_v0_A_b08_w192_d2_lr0.00317796_mls2.84045_bs0128_seed1_c3e52038064f",
        "D5_rare_1e-3_p13_identity_cartesian_v0_C_b04_w192_d2_lr0.00249866_mls3.0361_bs0128_seed1_1d9badd15e13",
    ),
}

_PINNED_DOE_V0_CANONICAL_HASH = "bd041ccd289fe24dfa41dec109657e1f8938295bb3a4b96028d6afea28a60430"

# Golden Arm-A (IID baseline) regression fixture, captured pre-change with
# ``AffineCouplingFlow(dimension=5, number_of_blocks=2, hidden_width=16,
# hidden_depth=1, max_epochs=3, batch_size=16, early_stopping=False,
# checkpoint_interval=1)`` fit on a 64-row D5(rare_1e-3, pdg=13, seed=7) IID
# training partition with ``seed=42``.
_GOLDEN_ARM_A_STATE_DICT_HASH = "afcaa5215fc4c24113ae42a91ef473d2847eaac7e044edb2787fa95258c2d025"
_GOLDEN_ARM_A_BEST_STEP = 2
_GOLDEN_ARM_A_N_HISTORY = 3
_GOLDEN_ARM_A_FEATURE_SPACE_TRAIN_NLL = 1015.8333740234375
_GOLDEN_ARM_A_FEATURE_SPACE_VALIDATION_NLL = 978.6005859375
_GOLDEN_ARM_A_WEIGHT_NORMALIZATION = "sum_weights"
_GOLDEN_ARM_A_STATE_DICT_SHAPES = {
    "layers.0.mask": (5,),
    "layers.0.net.0.weight": (16, 5),
    "layers.0.net.0.bias": (16,),
    "layers.0.net.2.weight": (16, 16),
    "layers.0.net.2.bias": (16,),
    "layers.0.net.4.weight": (10, 16),
    "layers.0.net.4.bias": (10,),
    "layers.1.mask": (5,),
    "layers.1.net.0.weight": (16, 5),
    "layers.1.net.0.bias": (16,),
    "layers.1.net.2.weight": (16, 16),
    "layers.1.net.2.bias": (16,),
    "layers.1.net.4.weight": (10, 16),
    "layers.1.net.4.bias": (10,),
}
_GOLDEN_ARM_A_PARAMETER_COUNT = 1076
_GOLDEN_ARM_A_PROBE = np.asarray(
    [[0.0, 0.0, 0.0, 0.0, 0.0],
     [0.25, -0.5, 1.0, -1.5, 2.0],
     [-1.0, 0.5, -0.25, 0.75, -0.8]],
    dtype=np.float32,
)
_GOLDEN_ARM_A_PROBE_LOG_PROB = np.asarray(
    [-4.720993995666504, -8.852683067321777, -5.9291276931762695],
    dtype=np.float64,
)


@pytest.mark.parametrize(
    "config_name",
    ["d3_memorization_control_v0.json", "d5_memorization_matrix_v0.json", "d5_memorization_smoke_v0.json"],
)
def test_pinned_experiment_config_hash_and_run_count(config_name):
    config = ExperimentConfig.from_json_file(CONFIG_ROOT + config_name)
    assert config.config_hash() == _PINNED_EXPERIMENT_CONFIG_HASH[config_name]
    assert canonical_hash(config.to_dict()) == _PINNED_EXPERIMENT_CONFIG_HASH[config_name]
    runs = config.runs()
    assert len(runs) == _PINNED_N_RUNS[config_name]


@pytest.mark.parametrize(
    "config_name",
    ["d3_memorization_control_v0.json", "d5_memorization_matrix_v0.json", "d5_memorization_smoke_v0.json"],
)
def test_pinned_run_ids_head_and_tail(config_name):
    config = ExperimentConfig.from_json_file(CONFIG_ROOT + config_name)
    runs = config.runs()
    head_expected, tail_expected = _PINNED_RUN_IDS_HEAD_TAIL[config_name]
    assert derive_run_id(runs[0]) == head_expected
    assert derive_run_id(runs[-1]) == tail_expected


def test_pinned_doe_v0_canonical_hash_unchanged():
    payload = generate_blocked_maximin_lhs(doe_seed=20260717, candidate_count=256)
    assert payload["canonical_hash"] == _PINNED_DOE_V0_CANONICAL_HASH


def test_pinned_doe_v0_committed_artifacts_unchanged():
    import json
    from pathlib import Path

    manifest = json.loads(Path(CONFIG_ROOT + "d5_memorization_doe_v0.manifest.json").read_text())
    assert manifest["canonical_hash"] == _PINNED_DOE_V0_CANONICAL_HASH
    payload = json.loads(Path(CONFIG_ROOT + "d5_memorization_doe_v0.json").read_text())
    assert payload["canonical_hash"] == _PINNED_DOE_V0_CANONICAL_HASH


def test_legacy_weighted_nll_hand_computation_unchanged():
    torch = pytest.importorskip("torch")
    from Nflow.torch_models.trainer import _weighted_nll

    class KnownLogProb:
        def log_prob(self, value):
            return value[:, 0]

    value = torch.tensor([[1.0], [2.0]])
    weight = torch.tensor([1.0, 3.0])
    assert float(_weighted_nll(KnownLogProb(), value, weight)) == pytest.approx(-1.75)


def test_density_lab_import_stays_numpy_only():
    script = (
        "import sys\n"
        "import ship_muon_bg.density_lab\n"
        "forbidden = ('torch', 'sklearn', 'matplotlib', 'mlflow')\n"
        "leaked = [m for m in forbidden if m in sys.modules]\n"
        "assert not leaked, leaked\n"
        "print('ok')\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ok" in result.stdout


def _run_golden_arm_a():
    torch = pytest.importorskip("torch")
    from ship_muon_bg.density_lab.datasets import build_controlled_dataset
    from ship_muon_bg.density_lab.sampling import IID_TARGET
    from Nflow.torch_models.affine_coupling import AffineCouplingFlow

    dataset = build_controlled_dataset(
        target_id="D5", variant="rare_1e-3", pdg_id=13,
        n_train=64, n_validation=64, n_test=32, seed=7,
        regime=IID_TARGET, sampling_rare_fraction=None,
    )
    flow = AffineCouplingFlow(
        dimension=5, number_of_blocks=2, hidden_width=16, hidden_depth=1,
        max_epochs=3, batch_size=16, early_stopping=False, checkpoint_interval=1,
    )
    result = flow.fit(
        dataset.train.physical, x_validation=dataset.validation.physical, seed=42,
        sample_weight=dataset.train.sample_weight,
        validation_sample_weight=dataset.validation.sample_weight,
        component_id=dataset.train.component_id,
        validation_component_id=dataset.validation.component_id,
        rare_component_id=None,
    )
    return flow, result


def test_golden_arm_a_portable_functional_regression():
    torch = pytest.importorskip("torch")
    flow_a, result_a = _run_golden_arm_a()
    flow_b, result_b = _run_golden_arm_a()

    assert result_a.status == "ok"
    assert result_a.best_step == _GOLDEN_ARM_A_BEST_STEP
    assert len(result_a.train_history) == _GOLDEN_ARM_A_N_HISTORY
    final = result_a.train_history[-1]
    assert final["weight_normalization"] == _GOLDEN_ARM_A_WEIGHT_NORMALIZATION
    assert final["feature_space_train_nll"] == pytest.approx(
        _GOLDEN_ARM_A_FEATURE_SPACE_TRAIN_NLL, rel=1e-7, abs=1e-5
    )
    assert final["feature_space_validation_nll"] == pytest.approx(
        _GOLDEN_ARM_A_FEATURE_SPACE_VALIDATION_NLL, rel=1e-7, abs=1e-5
    )

    state = flow_a._module.state_dict()
    assert list(state) == list(_GOLDEN_ARM_A_STATE_DICT_SHAPES)
    assert {name: tuple(value.shape) for name, value in state.items()} == _GOLDEN_ARM_A_STATE_DICT_SHAPES
    assert flow_a.parameter_count() == _GOLDEN_ARM_A_PARAMETER_COUNT
    assert all(bool(torch.isfinite(value).all()) for value in state.values())

    assert result_b.status == result_a.status
    assert result_b.best_step == result_a.best_step
    assert result_b.train_history[-1]["state_dict_hash"] == final["state_dict_hash"]
    np.testing.assert_allclose(
        flow_a.log_prob(_GOLDEN_ARM_A_PROBE),
        _GOLDEN_ARM_A_PROBE_LOG_PROB,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        flow_a.log_prob(_GOLDEN_ARM_A_PROBE),
        flow_b.log_prob(_GOLDEN_ARM_A_PROBE),
        rtol=1e-7,
        atol=1e-7,
    )


@pytest.mark.reference_golden
def test_golden_arm_a_reference_state_dict_hash():
    if os.environ.get("SHIP_RUN_REFERENCE_GOLDEN") != "1":
        pytest.skip(
            "exact model-byte identity requires the frozen reference environment; "
            "set SHIP_RUN_REFERENCE_GOLDEN=1 to run this check"
        )
    _torch = pytest.importorskip("torch")
    _flow, result = _run_golden_arm_a()
    assert result.status == "ok"
    assert result.best_step == _GOLDEN_ARM_A_BEST_STEP
    assert len(result.train_history) == _GOLDEN_ARM_A_N_HISTORY
    final = result.train_history[-1]
    assert final["state_dict_hash"] == _GOLDEN_ARM_A_STATE_DICT_HASH


# --- Section 2: planner unit tests ------------------------------------------

from ship_muon_bg.density_lab.sampling import (  # noqa: E402
    IID_TARGET,
    STRATIFIED_DIAGNOSTIC,
    STRATIFIED_SELF_NORMALIZED_PROVISIONAL,
    STRATIFIED_HT_FIXED_COMPOSITION,
    ESTIMATOR_HORVITZ_THOMPSON_FIXED,
    ESTIMATOR_MINIBATCH_MEAN,
    ESTIMATOR_SELF_NORMALIZED,
    POOL_FIXED_COMPOSITION,
    POOL_IID,
    MinibatchPlan,
    plan_fixed_composition_batches,
    resolve_regime,
)


def _toy_component_id(n_main=900, n_rare=100):
    return np.concatenate(
        (np.zeros(n_main, dtype=np.int64), np.ones(n_rare, dtype=np.int64))
    )


def test_resolve_regime_maps_all_four_named_arms_without_changing_legacy():
    assert resolve_regime(IID_TARGET) == (POOL_IID, ESTIMATOR_MINIBATCH_MEAN)
    assert resolve_regime(STRATIFIED_DIAGNOSTIC) == (
        POOL_FIXED_COMPOSITION, ESTIMATOR_MINIBATCH_MEAN,
    )
    assert resolve_regime(STRATIFIED_SELF_NORMALIZED_PROVISIONAL) == (
        POOL_FIXED_COMPOSITION, ESTIMATOR_SELF_NORMALIZED,
    )
    assert resolve_regime(STRATIFIED_HT_FIXED_COMPOSITION) == (
        POOL_FIXED_COMPOSITION, ESTIMATOR_HORVITZ_THOMPSON_FIXED,
    )
    with pytest.raises(ValueError):
        resolve_regime("not_a_regime")


def test_planner_realized_counts_weights_and_weight_sum_identity():
    component_id = _toy_component_id()
    plan = plan_fixed_composition_batches(
        component_id=component_id, rare_id=1,
        target_stratum_masses={"main": 0.999, "rare": 0.001},
        batch_size=128, counts={"main": 118, "rare": 10},
        replacement="without_replacement_within_epoch",
        steps_per_epoch_rule="min_stratum_pass", seed=42,
    )
    assert plan.counts_by_stratum == {"main": 118, "rare": 10}
    # w_h built from realized integer counts, not the configured rare_fraction.
    assert plan.weights_by_stratum["rare"] == pytest.approx(0.001 * 128 / 10)
    assert plan.weights_by_stratum["main"] == pytest.approx(0.999 * 128 / 118)
    for step in range(plan.steps_per_epoch):
        indices = plan.step_indices(step)
        weights = plan.step_weights(step)
        assert indices.shape == (128,)
        # sum_h m_h w_h == B exactly (float64, tight tolerance).
        assert abs(float(weights.sum()) - 128.0) < 1e-9
        assert int((component_id[indices] == 1).sum()) == 10
        assert int((component_id[indices] == 0).sum()) == 118


def test_planner_deterministic_for_same_seed():
    component_id = _toy_component_id()
    kwargs = dict(
        component_id=component_id, rare_id=1,
        target_stratum_masses={"main": 0.999, "rare": 0.001},
        batch_size=128, counts={"main": 118, "rare": 10}, seed=42,
    )
    first = plan_fixed_composition_batches(**kwargs)
    second = plan_fixed_composition_batches(**kwargs)
    assert first.plan_hash == second.plan_hash
    for step in range(first.steps_per_epoch):
        np.testing.assert_array_equal(first.step_indices(step), second.step_indices(step))


def test_planner_indices_in_range_and_no_intra_step_duplicates_without_replacement():
    component_id = _toy_component_id(n_main=50, n_rare=20)
    plan = plan_fixed_composition_batches(
        component_id=component_id, rare_id=1,
        target_stratum_masses={"main": 0.7, "rare": 0.3},
        batch_size=10, counts={"main": 7, "rare": 3},
        replacement="without_replacement_within_epoch",
        steps_per_epoch_rule="min_stratum_pass", seed=1,
    )
    n = component_id.shape[0]
    for step in range(plan.steps_per_epoch):
        indices = plan.step_indices(step)
        assert indices.min() >= 0 and indices.max() < n
        assert len(set(indices.tolist())) == len(indices)


def test_planner_step_weights_stay_aligned_under_shuffle():
    component_id = _toy_component_id()
    plan = plan_fixed_composition_batches(
        component_id=component_id, rare_id=1,
        target_stratum_masses={"main": 0.999, "rare": 0.001},
        batch_size=128, counts={"main": 118, "rare": 10}, seed=3,
    )
    indices = plan.step_indices(0, shuffle_seed=99)
    weights = plan.step_weights(0, shuffle_seed=99)
    expected = np.where(
        component_id[indices] == 1,
        plan.weights_by_stratum["rare"],
        plan.weights_by_stratum["main"],
    )
    np.testing.assert_allclose(weights, expected)


def test_planner_rejects_zero_count_for_positive_mass_stratum():
    component_id = _toy_component_id()
    with pytest.raises(ValueError, match="positive target mass"):
        plan_fixed_composition_batches(
            component_id=component_id, rare_id=1,
            target_stratum_masses={"main": 0.999, "rare": 0.001},
            batch_size=128, counts={"main": 128, "rare": 0}, seed=1,
        )


def test_planner_rejects_count_sum_mismatch():
    component_id = _toy_component_id()
    with pytest.raises(ValueError, match="!= batch_size"):
        plan_fixed_composition_batches(
            component_id=component_id, rare_id=1,
            target_stratum_masses={"main": 0.999, "rare": 0.001},
            batch_size=128, counts={"main": 100, "rare": 10}, seed=1,
        )


def test_planner_rejects_scarce_stratum_below_requested_count():
    component_id = _toy_component_id(n_main=900, n_rare=5)
    with pytest.raises(ValueError, match="fewer than the requested fixed count"):
        plan_fixed_composition_batches(
            component_id=component_id, rare_id=1,
            target_stratum_masses={"main": 0.999, "rare": 0.001},
            batch_size=128, counts={"main": 118, "rare": 10}, seed=1,
        )


def test_planner_rejects_invalid_mass_sum():
    component_id = _toy_component_id()
    with pytest.raises(ValueError, match="must sum to 1.0"):
        plan_fixed_composition_batches(
            component_id=component_id, rare_id=1,
            target_stratum_masses={"main": 0.9, "rare": 0.05},
            batch_size=128, counts={"main": 118, "rare": 10}, seed=1,
        )


def test_planner_rejects_zero_step_plan():
    # A stratum with zero target mass may request zero rows (A3 only binds
    # positive-mass strata), but a zero count also means zero step capacity
    # for that stratum -- reaching the zero-step guard directly rather than
    # the scarce-stratum guard.
    component_id = _toy_component_id(n_main=100, n_rare=0)
    with pytest.raises(ValueError, match="steps_per_epoch would be 0"):
        plan_fixed_composition_batches(
            component_id=component_id, rare_id=1,
            target_stratum_masses={"main": 1.0, "rare": 0.0},
            batch_size=20, counts={"main": 20, "rare": 0}, seed=1,
        )


def test_planner_rejects_non_unit_physical_weights():
    component_id = _toy_component_id()
    physical_weight = np.ones(component_id.shape[0])
    physical_weight[0] = 2.0
    with pytest.raises(ValueError, match="unit physical row weights"):
        plan_fixed_composition_batches(
            component_id=component_id, rare_id=1,
            target_stratum_masses={"main": 0.999, "rare": 0.001},
            batch_size=128, counts={"main": 118, "rare": 10}, seed=1,
            physical_weight=physical_weight,
        )


def test_planner_rejects_unsupported_replacement_mode():
    component_id = _toy_component_id()
    with pytest.raises(ValueError, match="replacement must be one of"):
        plan_fixed_composition_batches(
            component_id=component_id, rare_id=1,
            target_stratum_masses={"main": 0.999, "rare": 0.001},
            batch_size=128, counts={"main": 118, "rare": 10}, seed=1,
            replacement="bogus_mode",
        )


def test_planner_recycle_scarce_stratum_records_recycling():
    component_id = _toy_component_id(n_main=900, n_rare=12)
    plan = plan_fixed_composition_batches(
        component_id=component_id, rare_id=1,
        target_stratum_masses={"main": 0.999, "rare": 0.001},
        batch_size=128, counts={"main": 118, "rare": 10},
        steps_per_epoch_rule="recycle_scarce_stratum", seed=1,
    )
    assert plan.stratum_recycled["rare"] is True
    assert plan.recycle_count_by_stratum["rare"] >= 1
    assert plan.stratum_recycled["main"] is False


# --- Section 3: estimator property tests ------------------------------------


def test_horvitz_thompson_algebraic_identity_matches_stratum_means():
    rng = np.random.default_rng(0)
    component_id = _toy_component_id(n_main=900, n_rare=100)
    values = np.where(
        component_id == 1,
        rng.normal(loc=5.0, scale=1.0, size=component_id.shape[0]),
        rng.normal(loc=0.0, scale=1.0, size=component_id.shape[0]),
    )
    plan = plan_fixed_composition_batches(
        component_id=component_id, rare_id=1,
        target_stratum_masses={"main": 0.999, "rare": 0.001},
        batch_size=128, counts={"main": 118, "rare": 10}, seed=5,
    )
    indices = plan.step_indices(0)
    weights = plan.step_weights(0)
    ht_mean = float(np.sum(weights * values[indices]) / plan.batch_size)
    stratum_mean = {
        h: float(np.mean(values[plan.step_indices_by_stratum[0][h]]))
        for h in ("main", "rare")
    }
    expected = 0.999 * stratum_mean["main"] + 0.001 * stratum_mean["rare"]
    assert ht_mean == pytest.approx(expected)


def test_collapse_identity_fixed_composition_ht_equals_self_normalized():
    """Under exact fixed composition, sum(weight) == B deterministically, so
    the HT mean and the self-normalized ratio (sum(w*l)/sum(w)) coincide."""

    rng = np.random.default_rng(1)
    component_id = _toy_component_id(n_main=900, n_rare=100)
    loss = rng.normal(size=component_id.shape[0]) ** 2
    plan = plan_fixed_composition_batches(
        component_id=component_id, rare_id=1,
        target_stratum_masses={"main": 0.999, "rare": 0.001},
        batch_size=128, counts={"main": 118, "rare": 10}, seed=7,
    )
    indices = plan.step_indices(0)
    weights = plan.step_weights(0)
    ht_loss = float(np.sum(weights * loss[indices]) / plan.batch_size)
    self_normalized_loss = float(np.sum(weights * loss[indices]) / np.sum(weights))
    assert ht_loss == pytest.approx(self_normalized_loss, rel=1e-12)
    assert np.sum(weights) == pytest.approx(plan.batch_size)


def test_random_composition_diverges_from_fixed_denominator():
    """Under random (hypergeometric) minibatch composition, the self-
    normalized ratio and a fixed-B-denominator estimator built from the same
    per-row weights differ on average -- this is exactly arm D's bias."""

    p_rare, p_main = 0.001, 0.999
    a_rare, a_main = 0.05, 0.95
    rare_weight, main_weight = p_rare / a_rare, p_main / a_main
    rng = np.random.default_rng(2)
    n_pool, batch_size, n_draws = 2000, 128, 500
    component_id = rng.choice([0, 1], size=n_pool, p=[1 - a_rare, a_rare])
    loss_main, loss_rare = 1.0, 50.0  # deliberately asymmetric per-stratum loss

    self_normalized_values = []
    fixed_denominator_values = []
    for draw in range(n_draws):
        idx = np.random.default_rng(1000 + draw).choice(n_pool, size=batch_size, replace=False)
        labels = component_id[idx]
        weight = np.where(labels == 1, rare_weight, main_weight)
        loss = np.where(labels == 1, loss_rare, loss_main)
        self_normalized_values.append(float(np.sum(weight * loss) / np.sum(weight)))
        fixed_denominator_values.append(float(np.sum(weight * loss) / batch_size))

    mean_diff = float(np.mean(self_normalized_values) - np.mean(fixed_denominator_values))
    assert abs(mean_diff) > 1e-6


def test_ht_estimate_invariant_under_row_permutation_and_relabeling():
    rng = np.random.default_rng(3)
    component_id = _toy_component_id(n_main=900, n_rare=100)
    loss = rng.normal(size=component_id.shape[0]) ** 2
    plan = plan_fixed_composition_batches(
        component_id=component_id, rare_id=1,
        target_stratum_masses={"main": 0.999, "rare": 0.001},
        batch_size=128, counts={"main": 118, "rare": 10}, seed=9,
    )
    indices = plan.step_indices(0)
    weights = plan.step_weights(0)
    baseline = float(np.sum(weights * loss[indices]) / plan.batch_size)

    order = rng.permutation(len(indices))
    permuted = float(np.sum(weights[order] * loss[indices[order]]) / plan.batch_size)
    assert permuted == pytest.approx(baseline)

    # Relabel components 0<->1 consistently (swap the meaning of "rare id");
    # the resulting HT estimate is unchanged when the stratum masses are
    # relabelled the same way.
    relabeled_component_id = 1 - component_id
    relabeled_plan = plan_fixed_composition_batches(
        component_id=relabeled_component_id, rare_id=0,
        target_stratum_masses={"main": 0.999, "rare": 0.001},
        batch_size=128, counts={"main": 118, "rare": 10}, seed=9,
    )
    relabeled_indices = relabeled_plan.step_indices(0)
    relabeled_weights = relabeled_plan.step_weights(0)
    relabeled_value = float(
        np.sum(relabeled_weights * loss[relabeled_indices]) / relabeled_plan.batch_size
    )
    assert relabeled_value == pytest.approx(baseline)


# --- SamplingSpec extension: default-omitting serialization, arm C validation


def test_sampling_spec_default_omitting_serialization_matches_legacy_shape():
    from ship_muon_bg.density_lab.config import SamplingSpec

    legacy = SamplingSpec(regime=STRATIFIED_DIAGNOSTIC, sampling_rare_fraction=0.5)
    assert legacy.to_dict() == {
        "regime": STRATIFIED_DIAGNOSTIC,
        "sampling_rare_fraction": 0.5,
    }
    assert SamplingSpec().to_dict() == {"regime": IID_TARGET, "sampling_rare_fraction": None}


def test_sampling_spec_arm_c_serializes_new_fields_only_when_set():
    from ship_muon_bg.density_lab.config import SamplingSpec

    spec = SamplingSpec(
        regime=STRATIFIED_HT_FIXED_COMPOSITION, sampling_rare_fraction=0.05,
        minibatch_rare_count=10, minibatch_batch_size=128,
    )
    payload = spec.to_dict()
    assert payload["minibatch_rare_count"] == 10
    assert payload["minibatch_batch_size"] == 128
    assert "replacement" not in payload  # default not serialized
    assert spec.resolved_validation_partition_law() == "iid_target"


def test_sampling_spec_arm_c_requires_integer_counts():
    from ship_muon_bg.density_lab.config import ConfigError, SamplingSpec

    with pytest.raises(ConfigError, match="requires an integer"):
        SamplingSpec(
            regime=STRATIFIED_HT_FIXED_COMPOSITION, sampling_rare_fraction=0.05,
        ).validate()
    with pytest.raises(ConfigError, match="1 <= minibatch_rare_count"):
        SamplingSpec(
            regime=STRATIFIED_HT_FIXED_COMPOSITION, sampling_rare_fraction=0.05,
            minibatch_rare_count=128, minibatch_batch_size=128,
        ).validate()


def test_sampling_spec_rejects_minibatch_fields_outside_arm_c():
    from ship_muon_bg.density_lab.config import ConfigError, SamplingSpec

    with pytest.raises(ConfigError, match="only valid for regime"):
        SamplingSpec(
            regime=IID_TARGET, minibatch_rare_count=10, minibatch_batch_size=128,
        ).validate()


def test_sampling_spec_rejects_unimplemented_incomplete_batch_rule():
    from ship_muon_bg.density_lab.config import ConfigError, SamplingSpec

    with pytest.raises(ConfigError, match="incomplete_batch_rule"):
        SamplingSpec(incomplete_batch_rule="allow_short_last").validate()


def test_sampling_spec_legacy_regime_resolves_validation_partition_law_unchanged():
    from ship_muon_bg.density_lab.config import SamplingSpec

    for regime in (IID_TARGET, STRATIFIED_DIAGNOSTIC, STRATIFIED_SELF_NORMALIZED_PROVISIONAL):
        kwargs = {"regime": regime}
        if regime != IID_TARGET:
            kwargs["sampling_rare_fraction"] = 0.5
        assert SamplingSpec(**kwargs).resolved_validation_partition_law() == "matches_training_regime"


# --- Validation partition correctness (Step 5) -------------------------------


def test_validation_partition_law_default_matches_legacy_stratified_behavior():
    from ship_muon_bg.density_lab.datasets import build_controlled_dataset

    dataset = build_controlled_dataset(
        target_id="D5", variant="rare_1e-3", pdg_id=13,
        n_train=200, n_validation=100, n_test=100, seed=3,
        regime=STRATIFIED_SELF_NORMALIZED_PROVISIONAL, sampling_rare_fraction=0.5,
    )
    assert dataset.validation_partition_law == "matches_training_regime"
    assert dataset.validation.sampling_manifest["regime"] == STRATIFIED_SELF_NORMALIZED_PROVISIONAL
    manifest = dataset.manifest()
    assert manifest["validation_partition_is_iid_target"] is False
    assert manifest["validation_no_leakage"] is True


def test_validation_partition_law_iid_target_forces_iid_validation():
    from ship_muon_bg.density_lab.datasets import build_controlled_dataset

    dataset = build_controlled_dataset(
        target_id="D5", variant="rare_1e-3", pdg_id=13,
        n_train=200, n_validation=100, n_test=100, seed=3,
        regime=STRATIFIED_SELF_NORMALIZED_PROVISIONAL, sampling_rare_fraction=0.5,
        validation_partition_law="iid_target",
    )
    assert dataset.validation.sampling_manifest["regime"] == IID_TARGET
    assert dataset.train.sampling_manifest["regime"] == STRATIFIED_SELF_NORMALIZED_PROVISIONAL
    manifest = dataset.manifest()
    assert manifest["validation_partition_is_iid_target"] is True
    assert manifest["validation_no_leakage"] is True
    partition_manifest = dataset.train.manifest()
    assert partition_manifest["partition_sampling_law"] == STRATIFIED_SELF_NORMALIZED_PROVISIONAL
    assert partition_manifest["stratum_pool_counts"] == {"main": 100, "rare": 100}


# --- Trainer plan-driven path (Step 6) --------------------------------------


def _toy_flow_dataset(n_main=500, n_rare=50, seed=0):
    rng = np.random.default_rng(seed)
    main_x = rng.normal(loc=0.0, scale=1.0, size=(n_main, 5))
    rare_x = rng.normal(loc=5.0, scale=1.0, size=(n_rare, 5))
    x = np.concatenate([main_x, rare_x])
    component_id = np.concatenate(
        (np.zeros(n_main, dtype=np.int64), np.ones(n_rare, dtype=np.int64))
    )
    return x, component_id


def test_trainer_plan_driven_path_uses_fixed_denominator_and_realized_counts():
    torch = pytest.importorskip("torch")
    from Nflow.torch_models.affine_coupling import AffineCouplingFlow
    from Nflow.torch_models.trainer import train_flow

    x, component_id = _toy_flow_dataset()
    plan = plan_fixed_composition_batches(
        component_id=component_id, rare_id=1,
        target_stratum_masses={"main": 0.999, "rare": 0.001},
        batch_size=32, counts={"main": 30, "rare": 2}, seed=11,
    )
    flow = AffineCouplingFlow(
        dimension=5, number_of_blocks=2, hidden_width=16, hidden_depth=1,
        max_epochs=3, batch_size=32, early_stopping=False, checkpoint_interval=1,
    )
    flow._build_module(seed=1)
    result = train_flow(
        flow, x, x, seed=1, component_id=component_id, rare_component_id=1,
        batch_plan=plan, loss_normalization="fixed_batch_size",
    )
    assert result.status == "ok"
    final = result.train_history[-1]
    assert final["weight_normalization"] == "fixed_batch_size"
    assert final["steps"] == plan.steps_per_epoch
    assert final["dropped_steps"] == 0
    assert final["batch_weight_sum_min"] == pytest.approx(32.0)
    assert final["batch_weight_sum_max"] == pytest.approx(32.0)
    assert final["minibatch_rare_count_min"] == 2
    assert final["minibatch_rare_count_max"] == 2
    assert final["rare_gradient_norm"] is not None
    assert final["main_gradient_norm"] is not None
    assert "feature_space_train_rare_nll" in final
    assert "feature_space_train_main_nll" in final


def test_trainer_plan_driven_path_requires_labels_and_rare_id():
    torch = pytest.importorskip("torch")
    from Nflow.torch_models.affine_coupling import AffineCouplingFlow
    from Nflow.torch_models.trainer import train_flow

    x, component_id = _toy_flow_dataset()
    plan = plan_fixed_composition_batches(
        component_id=component_id, rare_id=1,
        target_stratum_masses={"main": 0.999, "rare": 0.001},
        batch_size=32, counts={"main": 30, "rare": 2}, seed=11,
    )
    flow = AffineCouplingFlow(
        dimension=5, number_of_blocks=2, hidden_width=16, hidden_depth=1,
        max_epochs=1, batch_size=32, early_stopping=False, checkpoint_interval=1,
    )
    flow._build_module(seed=1)
    with pytest.raises(ValueError, match="batch_plan requires"):
        train_flow(flow, x, x, seed=1, batch_plan=plan)


def test_trainer_plan_driven_path_rejects_inconsistent_loss_normalization():
    torch = pytest.importorskip("torch")
    from Nflow.torch_models.affine_coupling import AffineCouplingFlow
    from Nflow.torch_models.trainer import train_flow

    x, component_id = _toy_flow_dataset()
    plan = plan_fixed_composition_batches(
        component_id=component_id, rare_id=1,
        target_stratum_masses={"main": 0.999, "rare": 0.001},
        batch_size=32, counts={"main": 30, "rare": 2}, seed=11,
    )
    flow = AffineCouplingFlow(
        dimension=5, number_of_blocks=2, hidden_width=16, hidden_depth=1,
        max_epochs=1, batch_size=32, early_stopping=False, checkpoint_interval=1,
    )
    flow._build_module(seed=1)
    with pytest.raises(ValueError, match="inconsistent"):
        train_flow(
            flow, x, x, seed=1, component_id=component_id, rare_component_id=1,
            batch_plan=plan, loss_normalization="sum_weights",
        )


def test_trainer_diagnostic_gradient_pass_never_touches_optimizer_state():
    torch = pytest.importorskip("torch")
    from Nflow.torch_models.affine_coupling import AffineCouplingFlow
    from Nflow.torch_models.trainer import train_flow

    x, component_id = _toy_flow_dataset()
    plan = plan_fixed_composition_batches(
        component_id=component_id, rare_id=1,
        target_stratum_masses={"main": 0.999, "rare": 0.001},
        batch_size=32, counts={"main": 30, "rare": 2}, seed=11,
    )
    flow = AffineCouplingFlow(
        dimension=5, number_of_blocks=2, hidden_width=16, hidden_depth=1,
        max_epochs=2, batch_size=32, early_stopping=False, checkpoint_interval=1,
    )
    flow._build_module(seed=1)
    result_a = train_flow(
        flow, x, x, seed=5, component_id=component_id, rare_component_id=1,
        batch_plan=plan, loss_normalization="fixed_batch_size",
    )
    flow2 = AffineCouplingFlow(
        dimension=5, number_of_blocks=2, hidden_width=16, hidden_depth=1,
        max_epochs=2, batch_size=32, early_stopping=False, checkpoint_interval=1,
    )
    flow2._build_module(seed=1)
    result_b = train_flow(
        flow2, x, x, seed=5, component_id=component_id, rare_component_id=1,
        batch_plan=plan, loss_normalization="fixed_batch_size",
    )
    # Determinism through the diagnostic pass proves it left no side effects
    # that would perturb the optimizer trajectory differently run to run.
    assert result_a.train_history[-1]["state_dict_hash"] == result_b.train_history[-1]["state_dict_hash"]


def test_affine_coupling_manifest_reports_actual_loss_normalization():
    torch = pytest.importorskip("torch")
    from Nflow.torch_models.affine_coupling import AffineCouplingFlow

    x, component_id = _toy_flow_dataset()
    flow = AffineCouplingFlow(
        dimension=5, number_of_blocks=2, hidden_width=16, hidden_depth=1,
        max_epochs=1, batch_size=32, early_stopping=False, checkpoint_interval=1,
    )
    assert flow.manifest()["loss_normalization"] == "sum_weights"
    plan = plan_fixed_composition_batches(
        component_id=component_id, rare_id=1,
        target_stratum_masses={"main": 0.999, "rare": 0.001},
        batch_size=32, counts={"main": 30, "rare": 2}, seed=1,
    )
    flow.fit(
        x, x_validation=x, seed=1, component_id=component_id, rare_component_id=1,
        batch_plan=plan, loss_normalization="fixed_batch_size",
    )
    manifest = flow.manifest()
    assert manifest["loss_normalization"] == "fixed_batch_size"
    assert manifest["estimator_family"] == "affine_coupling"
    assert set(manifest["supported_loss_normalizations"]) == {"sum_weights", "fixed_batch_size"}


def test_gaussian_baselines_raise_on_unsupported_loss_normalization():
    from Nflow.baselines.gaussian import DiagonalGaussian, FullGaussian
    from Nflow.baselines.gmm import GaussianMixtureEstimator

    x = np.random.default_rng(0).normal(size=(20, 3))
    for cls in (DiagonalGaussian, FullGaussian):
        model = cls(dimension=3) if cls is DiagonalGaussian else cls(dimension=3)
        with pytest.raises(NotImplementedError, match="loss_normalization"):
            model.fit(x, loss_normalization="fixed_batch_size")
        with pytest.raises(NotImplementedError, match="batch_plan"):
            model.fit(x, batch_plan=object())
    gmm = GaussianMixtureEstimator(dimension=3, n_components=1)
    with pytest.raises(NotImplementedError, match="batch_plan"):
        gmm.fit(x, batch_plan=object())


# --- Section 4: campaign integration tests (Step 8) --------------------------


def _arm_c_run_spec(store_experiment_id, *, minibatch_rare_count=4, minibatch_batch_size=32,
                     seed=11, n_train=200, n_validation=100, n_test=100):
    from ship_muon_bg.density_lab.config import (
        DatasetSpec, EvaluationSpec, FeatureViewSpec, ModelSpec, RunSpec,
        SamplingSpec, TargetSpec,
    )

    model = ModelSpec(
        name="tiny_affine", family="affine_coupling",
        params={
            "number_of_blocks": 2, "hidden_width": 16, "hidden_depth": 1,
            "max_epochs": 2, "batch_size": minibatch_batch_size,
            "early_stopping": False, "checkpoint_interval": 1,
        },
    )
    sampling = SamplingSpec(
        regime=STRATIFIED_HT_FIXED_COMPOSITION, sampling_rare_fraction=0.5,
        minibatch_rare_count=minibatch_rare_count, minibatch_batch_size=minibatch_batch_size,
    )
    return RunSpec(
        experiment_id=store_experiment_id,
        target=TargetSpec("D5", "rare_1e-3"),
        pdg_id=13,
        feature_view=FeatureViewSpec("identity_cartesian_v0"),
        model=model,
        seed=seed,
        dataset=DatasetSpec(n_train=n_train, n_validation=n_validation, n_test=n_test),
        evaluation=EvaluationSpec(ess_sample_count=300, c2st_sample_count=200, rare_sample_count=300),
        device="cpu",
        sampling=sampling,
    )


@pytest.mark.lab
def test_campaign_arm_c_run_writes_expected_manifest_and_claim(tmp_path):
    torch = pytest.importorskip("torch")
    from ship_muon_bg.density_lab.artifacts import ArtifactStore
    from ship_muon_bg.density_lab.campaign import run_single

    store = ArtifactStore("itest_arm_c", root=tmp_path)
    run_spec = _arm_c_run_spec("itest_arm_c")
    record = run_single(run_spec, store, device="cpu")
    assert record["status"] == "completed"

    import json
    metrics = json.loads(store.run_paths(run_spec).metrics.read_text())
    assert metrics["estimator_family"] == "fixed_composition_horvitz_thompson_minibatch"
    assert metrics["unbiasedness_status"] == "established_under_stated_assumptions"
    assert metrics["fit_claim"] == "unbiased_target_risk_estimator_under_stated_assumptions"
    assert metrics["validation_objective_law"] == "iid_target"
    assert metrics["validation_nll_is_target_risk_estimate"] is True
    assert metrics["minibatch_plan"]["counts_by_stratum"] == {"main": 28, "rare": 4}
    assert metrics["diagnostic_only"] is False
    training_final = metrics["training_final"]
    assert training_final["weight_normalization"] == "fixed_batch_size"
    assert training_final["minibatch_rare_count_min"] == 4
    assert training_final["minibatch_rare_count_max"] == 4

    dataset_manifest = json.loads(store.run_paths(run_spec).run_dir.joinpath("dataset_manifest.json").read_text())
    assert dataset_manifest["validation_partition_is_iid_target"] is True
    assert dataset_manifest["partitions"]["validation"]["partition_sampling_law"] == IID_TARGET
    assert dataset_manifest["partitions"]["train"]["partition_sampling_law"] == STRATIFIED_HT_FIXED_COMPOSITION

    model_manifest = json.loads(store.run_paths(run_spec).run_dir.joinpath("model_manifest.json").read_text())
    assert model_manifest["loss_normalization"] == "fixed_batch_size"

    hashes = json.loads(store.run_paths(run_spec).run_status.read_text())["hashes"]
    assert hashes["minibatch_plan_hash"]


@pytest.mark.lab
def test_campaign_arm_c_resume_and_force_work(tmp_path):
    torch = pytest.importorskip("torch")
    from ship_muon_bg.density_lab.artifacts import ArtifactStore
    from ship_muon_bg.density_lab.campaign import run_single

    store = ArtifactStore("itest_arm_c_resume", root=tmp_path)
    run_spec = _arm_c_run_spec("itest_arm_c_resume")
    first = run_single(run_spec, store, device="cpu")
    assert first["status"] == "completed"
    second = run_single(run_spec, store, device="cpu")
    assert second["status"] == "skipped_completed"
    third = run_single(run_spec, store, device="cpu", force=True)
    assert third["status"] == "completed"


@pytest.mark.lab
def test_campaign_unsupported_baseline_fails_technically_under_arm_c(tmp_path):
    from ship_muon_bg.density_lab.artifacts import ArtifactStore
    from ship_muon_bg.density_lab.campaign import run_single
    from ship_muon_bg.density_lab.config import (
        DatasetSpec, EvaluationSpec, FeatureViewSpec, ModelSpec, RunSpec,
        SamplingSpec, TargetSpec,
    )

    store = ArtifactStore("itest_arm_c_gmm", root=tmp_path)
    model = ModelSpec(name="gmm", family="gaussian_mixture", params={"n_components": 1})
    sampling = SamplingSpec(
        regime=STRATIFIED_HT_FIXED_COMPOSITION, sampling_rare_fraction=0.5,
        minibatch_rare_count=4, minibatch_batch_size=32,
    )
    run_spec = RunSpec(
        experiment_id="itest_arm_c_gmm", target=TargetSpec("D5", "rare_1e-3"), pdg_id=13,
        feature_view=FeatureViewSpec("identity_cartesian_v0"), model=model, seed=11,
        dataset=DatasetSpec(n_train=200, n_validation=100, n_test=100),
        evaluation=EvaluationSpec(ess_sample_count=300, c2st_sample_count=200, rare_sample_count=300),
        device="cpu", sampling=sampling,
    )
    record = run_single(run_spec, store, device="cpu")
    assert record["status"] == "failed"
    import json
    status_payload = json.loads(store.run_paths(run_spec).run_status.read_text())
    assert status_payload["technical_status"] == "failed"
    assert status_payload["scientific_status"] is None


@pytest.mark.lab
def test_campaign_arm_b_never_claims_fit_to_target():
    from ship_muon_bg.density_lab.artifacts import ArtifactStore
    from ship_muon_bg.density_lab.campaign import run_single
    from ship_muon_bg.density_lab.config import (
        DatasetSpec, EvaluationSpec, FeatureViewSpec, ModelSpec, RunSpec,
        SamplingSpec, TargetSpec,
    )
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        store = ArtifactStore("itest_arm_b", root=tmp)
        model = ModelSpec(name="gauss", family="full_gaussian", params={})
        sampling = SamplingSpec(regime=STRATIFIED_DIAGNOSTIC, sampling_rare_fraction=0.5)
        run_spec = RunSpec(
            experiment_id="itest_arm_b", target=TargetSpec("D5", "rare_1e-3"), pdg_id=13,
            feature_view=FeatureViewSpec("identity_cartesian_v0"), model=model, seed=11,
            dataset=DatasetSpec(n_train=200, n_validation=100, n_test=100),
            evaluation=EvaluationSpec(ess_sample_count=300, c2st_sample_count=200, rare_sample_count=300),
            device="cpu", sampling=sampling,
        )
        record = run_single(run_spec, store, device="cpu")
        assert record["status"] == "completed"
        import json
        metrics = json.loads(store.run_paths(run_spec).metrics.read_text())
        assert metrics["fit_claim"] == "diagnostic_only_not_a_fit_to_original_target_density"
        assert metrics["fit_claim"] != "fit_to_original_target_density"


# --- Reporting + inactive gate (Step 10) -------------------------------------


def test_estimator_evidence_scope_gate_is_inactive_and_report_only():
    from ship_muon_bg.density_lab.gates import ScientificGateSpec, evaluate_scientific_gates

    healthy_metrics = {
        "held_out": {"held_out_nll": 1.0, "non_finite_count": 0},
        "forward_kl": {"forward_kl": 1.0, "non_finite_count": 0},
        "non_finite_density": {"non_finite_count": 0, "non_finite_density_rate": 0.0},
        "importance_ess": {"ess_over_n": 0.5},
        "estimator_family": "fixed_composition_horvitz_thompson_minibatch",
        "fit_claim": "unbiased_target_risk_estimator_under_stated_assumptions",
        "unbiasedness_status": "established_under_stated_assumptions",
    }
    result = evaluate_scientific_gates(
        healthy_metrics, target_id="D5", gate_spec=ScientificGateSpec().resolve(
            type("E", (), {"catastrophic_ess_threshold": 0.01})()
        ),
    )
    gate = next(g for g in result.gate_results if g["gate_id"] == "estimator_evidence_scope")
    assert gate["active"] is False
    assert gate["outcome"] == "report"
    assert gate["value"]["permitted_claim"] == "unbiased_target_risk_estimator_under_stated_assumptions"
    # An inactive report-only gate never enters scientific_failure_reasons.
    assert all(r["gate_id"] != "estimator_evidence_scope" for r in result.scientific_failure_reasons)


def test_reporting_series_never_merges_estimator_arms():
    from ship_muon_bg.density_lab.reporting import PlotSeriesKey, _plot_series_key

    base = {"target_label": "D5-1e-3", "target_stage": "transformed", "diagnostic_only": False}
    key_a = _plot_series_key(dict(base, sampling_regime=IID_TARGET))
    key_b = _plot_series_key(dict(base, sampling_regime=STRATIFIED_DIAGNOSTIC, diagnostic_only=True))
    key_c = _plot_series_key(dict(base, sampling_regime=STRATIFIED_HT_FIXED_COMPOSITION))
    key_d = _plot_series_key(dict(base, sampling_regime=STRATIFIED_SELF_NORMALIZED_PROVISIONAL))
    assert len({key_a, key_b, key_c, key_d}) == 4
    assert key_c.estimator == "horvitz_thompson_fixed"
    assert key_a.estimator == "minibatch_mean"
    assert key_d.estimator == "self_normalized"
