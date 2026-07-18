"""Focused invariants for the D5 memorization DOE core."""

from __future__ import annotations

import json

import numpy as np
import pytest

from ship_muon_bg.benchmarks import make_controlled_target
from ship_muon_bg.density_lab.config import ConfigError, ExperimentConfig
from ship_muon_bg.density_lab.artifacts import derive_run_id
from ship_muon_bg.density_lab.datasets import build_controlled_dataset
from ship_muon_bg.density_lab.doe import generate_blocked_maximin_lhs
from ship_muon_bg.density_lab.metrics import exact_binomial_interval
from ship_muon_bg.density_lab.sampling import (
    IID_TARGET,
    STRATIFIED_SELF_NORMALIZED_PROVISIONAL,
    STRATIFIED_DIAGNOSTIC,
    sample_controlled,
    validate_sample_weight,
)
from ship_muon_bg.density_lab.targets import resolve_target


@pytest.mark.parametrize("regime", [STRATIFIED_DIAGNOSTIC, STRATIFIED_SELF_NORMALIZED_PROVISIONAL])
def test_exact_stratified_counts_weights_and_determinism(regime):
    target = make_controlled_target("D5", variant="rare_1e-3")
    kwargs = dict(
        target=target, pdg_id=13, n=1000, seed=17, regime=regime,
        sampling_rare_fraction=0.2,
    )
    first = sample_controlled(**kwargs)
    second = sample_controlled(**kwargs)
    rare_id = target.rare_component_id(pdg_id=13)
    assert np.count_nonzero(first.component_id == rare_id) == 200
    np.testing.assert_array_equal(first.physical, second.physical)
    np.testing.assert_array_equal(first.sample_weight, second.sample_weight)
    if regime == STRATIFIED_SELF_NORMALIZED_PROVISIONAL:
        np.testing.assert_allclose(first.sample_weight[first.component_id == rare_id], 0.001 / 0.2)
        np.testing.assert_allclose(first.sample_weight[first.component_id != rare_id], 0.999 / 0.8)
        assert first.manifest["estimator_family"] == "self_normalized_importance_weighted_minibatch"
        assert first.manifest["unbiasedness_status"] == "not_established"
        assert first.manifest["scientific_scope"] == "provisional_target_estimator"
    else:
        np.testing.assert_array_equal(first.sample_weight, np.ones(1000))
        assert first.manifest["diagnostic_only"] is True
    assert first.manifest["seed"] == 17
    assert len(first.manifest["dataset_hash"]) == 64


def test_iid_and_partition_validation_never_use_stratified_validation():
    dataset = build_controlled_dataset(
        target_id="D5", variant="rare_1e-3", pdg_id=13,
        n_train=200, n_validation=100, n_test=100, seed=3,
        regime=STRATIFIED_SELF_NORMALIZED_PROVISIONAL, sampling_rare_fraction=0.5,
    )
    assert dataset.train.sampling_manifest["regime"] == STRATIFIED_SELF_NORMALIZED_PROVISIONAL
    assert dataset.validation.sampling_manifest["regime"] == STRATIFIED_SELF_NORMALIZED_PROVISIONAL
    assert dataset.test_nominal.sampling_manifest["regime"] == IID_TARGET
    assert dataset.manifest()["validation_no_leakage"] is True


@pytest.mark.parametrize("weight", [[1.0, -1.0], [0.0, 0.0], [1.0, np.nan]])
def test_weight_validation_rejects_invalid_values(weight):
    with pytest.raises(ValueError):
        validate_sample_weight(np.asarray(weight), 2)


def test_d5_base_stage_precedes_transform_and_retains_labels():
    base = resolve_target("D5", "rare_1e-3", "base_before_d4")
    transformed = resolve_target("D5", "rare_1e-3", "transformed")
    a = base.sample(200, pdg_id=13, seed=9)
    b = transformed.sample(200, pdg_id=13, seed=9)
    np.testing.assert_array_equal(a.component_id, b.component_id)
    assert not np.array_equal(a.physical, b.physical)
    assert base.manifest()["transform_stage"] == "base_before_d4"


def test_blocked_doe_is_deterministic_maximin_and_duplicate_free():
    first = generate_blocked_maximin_lhs(doe_seed=123, candidate_count=32)
    second = generate_blocked_maximin_lhs(doe_seed=123, candidate_count=32)
    assert first == second
    assert len(first["configs"]) == 24
    assert {row["block"] for row in first["configs"]} == {"A", "B", "C"}
    assert all(sum(row["block"] == block for row in first["configs"]) == 8 for block in "ABC")
    keys = {
        tuple(sorted((key, value) for key, value in row.items() if key != "doe_id"))
        for row in first["configs"]
    }
    assert len(keys) == 24
    assert all(value > 0.0 for value in first["minimum_normalized_pairwise_distance_by_block"].values())
    assert len(first["canonical_hash"]) == 64
    for row in first["configs"]:
        assert 2 <= row["number_of_blocks"] <= 10
        assert row["hidden_width"] in {32, 48, 64, 96, 128, 192}
        assert 1 <= row["hidden_depth"] <= 3
        assert -4.0 <= row["log10_learning_rate"] <= -2.3
        assert 1.0 <= row["max_log_scale"] <= 5.0
        assert row["batch_size"] in {128, 256, 512}


def test_versioned_matrix_and_smoke_configs_are_runnable_definitions():
    root = "configs/density_lab/doe_v0/"
    matrix = ExperimentConfig.from_json_file(root + "d5_memorization_matrix_v0.json")
    control = ExperimentConfig.from_json_file(root + "d3_memorization_control_v0.json")
    smoke = ExperimentConfig.from_json_file(root + "d5_memorization_smoke_v0.json")
    assert len(matrix.models) == 24
    assert [target.stage for target in matrix.targets] == ["base_before_d4", "transformed"]
    assert {sampling.regime for sampling in matrix.sampling_regimes} == {
        IID_TARGET, STRATIFIED_DIAGNOSTIC, STRATIFIED_SELF_NORMALIZED_PROVISIONAL,
    }
    assert [target.target_id for target in control.targets] == ["D3"]
    assert [sampling.regime for sampling in control.sampling_regimes] == [IID_TARGET]
    assert len(matrix.runs()) == 144
    assert len(control.runs()) == 24
    assert len(matrix.runs()) + len(control.runs()) == 168
    assert len(smoke.runs()) == 9
    assert smoke.evaluation.rare_sample_count == 2000


def test_d3_stratified_pair_is_rejected_before_run_expansion():
    target = make_controlled_target("D3")
    assert getattr(target, "rare_mass", None) is None
    with pytest.raises(ValueError, match="labelled rare component"):
        sample_controlled(
            target, pdg_id=13, n=8, seed=4,
            regime=STRATIFIED_DIAGNOSTIC, sampling_rare_fraction=0.5,
        )
    payload = {
        "experiment_id": "invalid_d3_stratified",
        "targets": [{"target_id": "D3"}],
        "pdg_ids": [13],
        "feature_views": [{"view_id": "identity_cartesian_v0"}],
        "models": [{"name": "a", "family": "affine_coupling"}],
        "seeds": [1],
        "sampling_regimes": [{"regime": STRATIFIED_DIAGNOSTIC, "sampling_rare_fraction": 0.5}],
    }
    with pytest.raises(ConfigError, match="explicitly labelled rare component"):
        ExperimentConfig.from_dict(payload)


@pytest.mark.parametrize(
    "config_name", ["d5_memorization_matrix_v0.json", "d3_memorization_control_v0.json"]
)
def test_generated_target_sampling_pairs_construct_small_datasets(config_name):
    config = ExperimentConfig.from_json_file("configs/density_lab/doe_v0/" + config_name)
    pairs = {
        (
            run.target.target_id, run.target.variant, run.target.stage,
            run.sampling.regime, run.sampling.sampling_rare_fraction,
        )
        for run in config.runs()
    }
    assert pairs
    for target_id, variant, stage, regime, fraction in pairs:
        dataset = build_controlled_dataset(
            target_id=target_id, variant=variant, target_stage=stage, pdg_id=13,
            n_train=8, n_validation=8, n_test=8, seed=4,
            regime=regime, sampling_rare_fraction=fraction,
        )
        assert dataset.train.sampling_manifest["regime"] == regime


def test_exact_binomial_interval_contains_observed_mass():
    low, high = exact_binomial_interval(10, 1000)
    assert low < 0.01 < high
    assert exact_binomial_interval(0, 100)[0] == 0.0


def test_run_identity_includes_new_model_sampling_and_doe_factors():
    payload = {
        "experiment_id": "identity",
        "targets": [{"target_id": "D5", "variant": "rare_1e-3"}],
        "pdg_ids": [13],
        "feature_views": [{"view_id": "identity_cartesian_v0"}],
        "models": [{"name": "a", "family": "affine_coupling", "params": {"mixing_mode": "alternating_only"}}],
        "seeds": [1],
        "doe_seed": 10,
        "sampling_regimes": [{"regime": "iid_target"}],
    }
    base = ExperimentConfig.from_dict(payload).runs()[0]
    changed = dict(payload)
    changed["doe_seed"] = 11
    changed["models"] = [{"name": "a", "family": "affine_coupling", "params": {"mixing_mode": "fixed_random_permutation"}}]
    changed["sampling_regimes"] = [{"regime": STRATIFIED_SELF_NORMALIZED_PROVISIONAL, "sampling_rare_fraction": 0.5}]
    other = ExperimentConfig.from_dict(changed).runs()[0]
    assert base.config_hash() != other.config_hash()
    assert derive_run_id(base) != derive_run_id(other)


torch = pytest.importorskip("torch")

from Nflow.torch_models.affine_coupling import AffineCouplingFlow
from Nflow.torch_models.trainer import _weighted_nll


def test_weighted_loss_matches_hand_computation():
    class KnownLogProb:
        def log_prob(self, value):
            return value[:, 0]

    value = torch.tensor([[1.0], [2.0]])
    weight = torch.tensor([1.0, 3.0])
    assert float(_weighted_nll(KnownLogProb(), value, weight)) == pytest.approx(-1.75)


def test_seeded_permutation_is_bijective_deterministic_and_round_trips():
    first = AffineCouplingFlow(dimension=5, mixing_mode="fixed_random_permutation")
    first._build_module(seed=7)
    second = AffineCouplingFlow(dimension=5, mixing_mode="fixed_random_permutation")
    second._build_module(seed=7)
    assert first._module.permutations() == second._module.permutations()
    assert all(sorted(value) == list(range(5)) for value in first._module.permutations())
    x = torch.randn(16, 5)
    z, inverse_logdet = first._module.inverse(x)
    np.testing.assert_allclose(first._module(z).detach(), x, atol=1e-6)
    assert np.isfinite(inverse_logdet.detach()).all()


def test_memorization_mode_enforcement_and_epoch_metrics():
    with pytest.raises(ValueError, match="memorization_mode requires"):
        AffineCouplingFlow(dimension=5, memorization_mode=True)
    rng = np.random.default_rng(2)
    x = rng.standard_normal((64, 5))
    labels = np.concatenate((np.zeros(56, dtype=int), np.ones(8, dtype=int)))
    weights = np.where(labels == 1, 0.1, 1.1)
    flow = AffineCouplingFlow(
        dimension=5, number_of_blocks=2, hidden_width=16, hidden_depth=1,
        max_epochs=2, batch_size=32, memorization_mode=True,
        early_stopping=False, checkpoint_interval=1,
    )
    result = flow.fit(
        x, x_validation=x, seed=4, sample_weight=weights,
        validation_sample_weight=np.ones(64), component_id=labels,
        validation_component_id=labels, rare_component_id=1,
    )
    assert len(result.train_history) == 2
    required = {
        "feature_space_train_nll", "feature_space_train_main_nll",
        "feature_space_train_rare_nll", "feature_space_validation_nll",
        "feature_space_validation_main_nll", "feature_space_validation_rare_nll",
        "gradient_norm",
        "max_abs_log_scale", "checkpoint_hash", "weight_normalization",
    }
    assert required <= result.train_history[-1].keys()
    assert result.best_step == 1
