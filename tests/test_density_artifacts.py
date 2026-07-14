"""Tests for config hashing, run-id derivation, artifacts and datasets."""

from __future__ import annotations

import json

import numpy as np
import pytest

from ship_muon_bg.density_lab import (
    ArtifactStore,
    ExperimentConfig,
    build_controlled_dataset,
    derive_run_id,
)
from ship_muon_bg.density_lab.config import (
    ConfigError,
    DatasetSpec,
    EvaluationSpec,
    FeatureViewSpec,
    ModelSpec,
    RunSpec,
    TargetSpec,
)

SMOKE_CONFIG = "configs/density_lab/smoke_v0.json"


def _run_spec(**overrides):
    base = dict(
        experiment_id="t",
        target=TargetSpec("D2", None),
        pdg_id=13,
        feature_view=FeatureViewSpec("identity_cartesian_v0"),
        model=ModelSpec("full_gaussian", "full_gaussian", {}),
        seed=11,
        dataset=DatasetSpec(),
        evaluation=EvaluationSpec(),
        device="cpu",
    )
    base.update(overrides)
    return RunSpec(**base)


# --- config -----------------------------------------------------------------


def test_config_loads_and_validates():
    config = ExperimentConfig.from_json_file(SMOKE_CONFIG)
    config.validate()
    assert config.experiment_id == "smoke_v0"
    runs = config.runs()
    assert len(runs) == len(config.targets) * len(config.pdg_ids) * len(
        config.feature_views
    ) * len(config.models) * len(config.seeds)


def test_config_hash_deterministic_and_sensitive():
    config = ExperimentConfig.from_json_file(SMOKE_CONFIG)
    same = ExperimentConfig.from_json_file(SMOKE_CONFIG)
    assert config.config_hash() == same.config_hash()


def test_config_rejects_bad_pdg_id():
    with pytest.raises(ConfigError):
        ExperimentConfig(
            experiment_id="x",
            targets=[TargetSpec("D0")],
            pdg_ids=[99],
            feature_views=[FeatureViewSpec("identity_cartesian_v0")],
            models=[ModelSpec("m", "diagonal_gaussian")],
            seeds=[1],
        ).validate()


def test_changed_config_changes_run_hash():
    a = _run_spec()
    b = _run_spec(seed=22)
    assert a.config_hash() != b.config_hash()
    assert derive_run_id(a) != derive_run_id(b)


def test_run_id_deterministic():
    assert derive_run_id(_run_spec()) == derive_run_id(_run_spec())


def test_device_is_part_of_run_identity():
    # A CPU-forced run of an "auto" config must not collide with an auto run.
    cpu = _run_spec(device="cpu")
    auto = _run_spec(device="auto")
    assert cpu.config_hash() != auto.config_hash()
    assert derive_run_id(cpu) != derive_run_id(auto)
    assert cpu.to_dict()["device"] == "cpu"


# --- datasets ---------------------------------------------------------------


def test_matched_rows_across_feature_views():
    # The dataset is view-independent, so A/B1/B2 share identical physical rows.
    ds1 = build_controlled_dataset(
        target_id="D2", variant=None, pdg_id=13,
        n_train=500, n_validation=200, n_test=500, seed=11,
    )
    ds2 = build_controlled_dataset(
        target_id="D2", variant=None, pdg_id=13,
        n_train=500, n_validation=200, n_test=500, seed=11,
    )
    np.testing.assert_array_equal(ds1.train.physical, ds2.train.physical)
    np.testing.assert_array_equal(ds1.test_nominal.physical, ds2.test_nominal.physical)
    assert ds1.train.raw_dataset_hash == ds2.train.raw_dataset_hash


def test_partitions_are_independent_draws():
    ds = build_controlled_dataset(
        target_id="D0", variant=None, pdg_id=13,
        n_train=500, n_validation=500, n_test=500, seed=11,
    )
    # independent draws -> train and test should not be identical rows
    assert not np.array_equal(ds.train.physical, ds.test_nominal.physical)
    assert ds.train.seed != ds.test_nominal.seed


def test_d5_dataset_carries_rare_mask_without_forcing():
    ds = build_controlled_dataset(
        target_id="D5", variant="rare_1e-2", pdg_id=13,
        n_train=20000, n_validation=2000, n_test=20000, seed=11,
    )
    assert ds.train.rare_region_mask is not None
    frac = ds.train.rare_region_mask.mean()
    # not forced/oversampled: close to the nominal rare mass
    assert 0.005 < frac < 0.02


# --- artifacts / resume -----------------------------------------------------


def test_artifact_write_and_resume(tmp_path):
    store = ArtifactStore("t", root=tmp_path)
    run_spec = _run_spec()
    assert store.is_complete(run_spec) is False
    store.write_run(
        run_spec,
        environment={"git_commit": "abc"},
        dataset_manifest={"target_id": "D2"},
        feature_pipeline_manifest={"feature_view_id": "identity_cartesian_v0"},
        model_manifest={"family": "full_gaussian"},
        fit_result={"status": "ok"},
        metrics={"forward_kl": {"forward_kl": 0.1}},
        training_history=[{"step": 0, "train_nll": 1.0}],
        samples={"model_samples_physical": np.zeros((3, 5))},
        status="completed",
        run_id=derive_run_id(run_spec),
    )
    assert store.is_complete(run_spec) is True
    paths = store.run_paths(run_spec)
    for name in (
        "experiment_config.json", "environment.json", "dataset_manifest.json",
        "feature_pipeline_manifest.json", "model_manifest.json", "fit_result.json",
        "metrics.json", "training_history.jsonl", "samples.npz", "run_status.json",
    ):
        assert (paths.run_dir / name).exists(), name
    # NPZ reloads
    data = np.load(paths.samples)
    assert data["model_samples_physical"].shape == (3, 5)
    # training history is valid JSONL
    lines = (paths.training_history).read_text().strip().splitlines()
    assert json.loads(lines[0])["step"] == 0


def test_failed_rewrite_clears_stale_optional_artifacts(tmp_path):
    store = ArtifactStore("t", root=tmp_path)
    run_spec = _run_spec()
    paths = store.run_paths(run_spec)
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    # simulate a prior successful run leaving samples + model parameters
    np.savez(paths.samples, model_samples_physical=np.zeros((3, 5)))
    (paths.run_dir / "model_parameters.npz").write_bytes(b"stale")
    (paths.run_dir / "checkpoint").mkdir()
    (paths.run_dir / "checkpoint" / "state_dict.pt").write_bytes(b"stale")

    # a failed re-attempt writes no samples and no save_manifest
    store.write_run(
        run_spec,
        environment={},
        dataset_manifest={},
        feature_pipeline_manifest={},
        model_manifest={},
        fit_result={"status": "failed"},
        metrics={"status": "error"},
        training_history=[],
        samples={},
        status="failed",
        run_id=derive_run_id(run_spec),
    )
    assert not paths.samples.exists()
    assert not (paths.run_dir / "model_parameters.npz").exists()
    assert not (paths.run_dir / "checkpoint").exists()
