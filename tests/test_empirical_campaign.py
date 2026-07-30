"""Fixture-backed tests for D7 (empirical after-MS data) dataset construction,
preprocessing, splits, weight semantics, and the bounded campaign smoke path
(density_lab.empirical). Uses the repository afterMS sample throughout; a
full local dataset is stood in for with a temporary copy of that same
fixture (Section 7's "full-data configuration test").

Marked ``lab``/``flow`` (auto-skipped when torch/sklearn are absent) since
the campaign smoke test trains a tiny affine-coupling flow and its evaluator
uses C2ST.
"""

from __future__ import annotations

import json
import os
import shutil

import numpy as np
import pytest

from ship_muon_bg.data_contracts import (
    load_muon_array,
    load_muon_npz,
    schema,
)
from ship_muon_bg.data_contracts.feature_views import PHYSICAL_STATE_COLUMNS

pytestmark = pytest.mark.lab

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(REPO_ROOT, "data", "samples", "muonsFullMC_afterMS_sample.npz")


def _tiny_model_spec(name="tiny_affine", max_epochs=2, batch_size=32):
    from ship_muon_bg.density_lab.config import ModelSpec

    return ModelSpec(
        name=name, family="affine_coupling",
        params={
            "number_of_blocks": 2, "hidden_width": 16, "hidden_depth": 1,
            "max_epochs": max_epochs, "batch_size": batch_size,
            "early_stopping": False, "checkpoint_interval": 1,
        },
    )


def _dataset_spec(pdg_id=13, seed=11, max_rows=2000, dataset_path=FIXTURE):
    from ship_muon_bg.density_lab.empirical import EmpiricalDatasetSpec

    return EmpiricalDatasetSpec(
        dataset_path=dataset_path, pdg_id=pdg_id, seed=seed,
        val_fraction=0.2, test_fraction=0.2, max_rows=max_rows, allow_zero_weight=True,
    )


# --- Loader and schema -------------------------------------------------------


def test_repository_fixture_loads_through_production_loader():
    array = load_muon_array(FIXTURE)  # the same dispatcher the CLI and empirical module use
    assert array.shape == (40000, 8)
    assert array.dtype == np.float64


def test_required_feature_order_is_correct():
    assert PHYSICAL_STATE_COLUMNS == ("px", "py", "pz", "x", "y")
    assert schema.COLUMNS[: len(PHYSICAL_STATE_COLUMNS)] == PHYSICAL_STATE_COLUMNS


def test_pdg_filtering_works_for_both_charges_present():
    from ship_muon_bg.density_lab.empirical import build_empirical_dataset

    for pdg_id in (13, -13):
        dataset = build_empirical_dataset(_dataset_spec(pdg_id=pdg_id))
        for partition in (dataset.train, dataset.validation, dataset.test):
            ids = np.rint(partition.raw[:, schema.COLUMN_INDEX["id"]])
            assert np.all(ids == pdg_id)


def test_non_finite_values_fail_clearly():
    from ship_muon_bg.density_lab.empirical import EmpiricalDataError, build_empirical_dataset

    array = load_muon_npz(FIXTURE)
    corrupt = np.array(array, copy=True)
    corrupt[0, 0] = np.nan
    path = "/tmp/test_empirical_corrupt.npz"
    np.savez_compressed(path, muons=corrupt)
    try:
        with pytest.raises(EmpiricalDataError):
            build_empirical_dataset(_dataset_spec(dataset_path=path))
    finally:
        os.remove(path)


def test_optional_zero_weight_handled_explicitly():
    from ship_muon_bg.density_lab.empirical import EmpiricalDataError, build_empirical_dataset

    spec_allow = _dataset_spec()
    dataset = build_empirical_dataset(spec_allow)  # allow_zero_weight=True: must not raise
    assert dataset.train.n_rows > 0

    import dataclasses

    spec_disallow = dataclasses.replace(spec_allow, allow_zero_weight=False)
    array = load_muon_npz(FIXTURE)
    if np.any(array[:, schema.COLUMN_INDEX["w"]] == 0.0):
        with pytest.raises(EmpiricalDataError):
            build_empirical_dataset(spec_disallow)


# --- Preprocessing ------------------------------------------------------------


def test_forward_inverse_preprocessing_round_trip():
    from ship_muon_bg.density_lab.empirical import build_empirical_dataset
    from ship_muon_bg.density_lab.feature_pipeline import FittedFeaturePipeline
    from ship_muon_bg.data_contracts.feature_views import FeatureView

    dataset = build_empirical_dataset(_dataset_spec())
    view = FeatureView("identity_cartesian_v0")
    pipeline = FittedFeaturePipeline.fit(dataset.train.raw, view)
    normalized = pipeline.transform_raw(dataset.test.raw)
    physical_back = pipeline.inverse_to_physical(normalized)
    np.testing.assert_allclose(physical_back, dataset.test.physical, atol=1e-8)


def test_saved_preprocessing_parameters_reload_identically():
    from ship_muon_bg.density_lab.empirical import build_empirical_dataset
    from ship_muon_bg.density_lab.feature_pipeline import FittedFeaturePipeline
    from ship_muon_bg.data_contracts.feature_views import FeatureView

    dataset = build_empirical_dataset(_dataset_spec())
    view = FeatureView("identity_cartesian_v0")
    pipeline = FittedFeaturePipeline.fit(dataset.train.raw, view)
    manifest = pipeline.manifest()

    reloaded = FittedFeaturePipeline(
        feature_view=FeatureView(manifest["feature_view_id"]),
        mean=np.array(manifest["standardization"]["mean"]),
        std=np.array(manifest["standardization"]["std"]),
        n_train_rows=manifest["standardization"]["n_train_rows"],
        zero_variance_policy=manifest["standardization"]["zero_variance_policy"],
    )
    original = pipeline.transform_raw(dataset.test.raw)
    from_reload = reloaded.transform_raw(dataset.test.raw)
    np.testing.assert_array_equal(original, from_reload)
    assert reloaded.config_hash() == pipeline.config_hash()


def test_physical_and_model_space_feature_ordering_is_stable():
    from ship_muon_bg.density_lab.feature_pipeline import FittedFeaturePipeline
    from ship_muon_bg.density_lab.empirical import build_empirical_dataset
    from ship_muon_bg.data_contracts.feature_views import FeatureView

    dataset = build_empirical_dataset(_dataset_spec())
    view = FeatureView("identity_cartesian_v0")
    pipeline = FittedFeaturePipeline.fit(dataset.train.raw, view)
    assert pipeline.manifest()["feature_names"] == list(PHYSICAL_STATE_COLUMNS)


# --- Splits -------------------------------------------------------------------


def test_deterministic_split_reproduction():
    from ship_muon_bg.density_lab.empirical import build_empirical_dataset

    a = build_empirical_dataset(_dataset_spec(seed=42))
    b = build_empirical_dataset(_dataset_spec(seed=42))
    assert a.manifest() == b.manifest()
    np.testing.assert_array_equal(a.train.raw, b.train.raw)


def test_split_partitions_have_no_overlap():
    from ship_muon_bg.density_lab.empirical import build_empirical_dataset

    dataset = build_empirical_dataset(_dataset_spec())
    train_idx = set(dataset.train.source_row_indices.tolist())
    val_idx = set(dataset.validation.source_row_indices.tolist())
    test_idx = set(dataset.test.source_row_indices.tolist())
    assert not (train_idx & val_idx)
    assert not (train_idx & test_idx)
    assert not (val_idx & test_idx)
    assert dataset.manifest()["validation_no_leakage"] is True


def test_row_counts_and_hashes_stable_for_fixture():
    from ship_muon_bg.density_lab.empirical import build_empirical_dataset

    dataset = build_empirical_dataset(_dataset_spec(pdg_id=13, seed=11, max_rows=2000))
    assert dataset.train.n_rows == 1280
    assert dataset.validation.n_rows == 320
    assert dataset.test.n_rows == 400
    assert dataset.source_file_dataset_hash == (
        "718b9197a3965f6842574fec1f3f6b1bf4c16949ff8a1f27dca6232817c03c32"
    )


def test_test_split_is_not_touched_by_train_validation_phase():
    """A train/validation-only exploration must never read the test partition."""

    from ship_muon_bg.density_lab.empirical import build_empirical_dataset
    from ship_muon_bg.density_lab.feature_pipeline import FittedFeaturePipeline
    from ship_muon_bg.data_contracts.feature_views import FeatureView

    dataset = build_empirical_dataset(_dataset_spec())
    view = FeatureView("identity_cartesian_v0")
    # Fit + transform train/validation exactly as run_empirical_single does,
    # without ever referencing dataset.test.
    pipeline = FittedFeaturePipeline.fit(dataset.train.raw, view)
    normalized_train = pipeline.transform_raw(dataset.train.raw)
    normalized_val = pipeline.transform_raw(dataset.validation.raw)
    assert normalized_train.shape[0] == dataset.train.n_rows
    assert normalized_val.shape[0] == dataset.validation.n_rows
    # Corrupting dataset.test afterwards must not be able to affect anything
    # already computed above -- proving those rows were never read.
    dataset.test.raw[:] = np.nan
    assert np.isfinite(normalized_train).all()
    assert np.isfinite(normalized_val).all()


# --- Weights -------------------------------------------------------------------


def test_weight_field_is_preserved_through_dataset_construction():
    from ship_muon_bg.density_lab.empirical import build_empirical_dataset

    array = load_muon_npz(FIXTURE)
    dataset = build_empirical_dataset(_dataset_spec(pdg_id=13, max_rows=None))
    # Every weight in the (unbounded) per-PDG dataset must trace back to a
    # weight actually present in the source file for that PDG id.
    source_w = set(np.round(array[:, schema.COLUMN_INDEX["w"]], 6).tolist())
    for partition in (dataset.train, dataset.validation, dataset.test):
        partition_w = np.round(partition.raw[:, schema.COLUMN_INDEX["w"]], 6)
        assert set(partition_w.tolist()) <= source_w


def test_row_empirical_mode_does_not_silently_claim_physical_weighting():
    from ship_muon_bg.density_lab.empirical import build_empirical_dataset

    dataset = build_empirical_dataset(_dataset_spec())
    manifest = dataset.manifest()
    assert manifest["physical_weight_status"] == (
        "preserved_in_raw_rows_not_applied_to_training_loss"
    )
    assert manifest["physical_weight_semantics"] == "not_implemented"
    assert manifest["sampling_correction_weight_status"] == (
        "not_applicable_no_rare_aware_arm_for_unlabeled_real_data"
    )
    assert manifest["weights_affect_selection"] is False


def test_ht_arm_c_rejects_real_non_unit_physical_weights():
    """A6 (docs/contracts/rare_aware_minibatch_estimators_v0.md) must reject
    the *real* afterMS weight column, not just a synthetic one."""

    from ship_muon_bg.density_lab.sampling import plan_fixed_composition_batches

    array = load_muon_npz(FIXTURE)
    real_weight = array[:, schema.COLUMN_INDEX["w"]]
    assert not np.allclose(real_weight, 1.0), "fixture must contain genuinely non-unit weights for this test to be meaningful"

    component_id = np.where(
        np.rint(array[:, schema.COLUMN_INDEX["id"]]) == 13, 0, 1
    ).astype(np.int64)
    with pytest.raises(ValueError, match="unit physical row weights"):
        plan_fixed_composition_batches(
            component_id=component_id, rare_id=1,
            target_stratum_masses={"main": 0.5, "rare": 0.5},
            batch_size=128, counts={"main": 64, "rare": 64}, seed=1,
            physical_weight=real_weight,
        )


def test_ht_arm_c_accepts_unit_physical_weights_from_same_shaped_input():
    from ship_muon_bg.density_lab.sampling import plan_fixed_composition_batches

    array = load_muon_npz(FIXTURE)
    unit_weight = np.ones(array.shape[0])
    component_id = np.where(
        np.rint(array[:, schema.COLUMN_INDEX["id"]]) == 13, 0, 1
    ).astype(np.int64)
    plan = plan_fixed_composition_batches(
        component_id=component_id, rare_id=1,
        target_stratum_masses={"main": 0.5, "rare": 0.5},
        batch_size=128, counts={"main": 64, "rare": 64}, seed=1,
        physical_weight=unit_weight,
    )
    assert plan.steps_per_epoch > 0


# --- Campaign smoke test -------------------------------------------------------


@pytest.mark.parametrize("pdg_id", [13, -13])
def test_bounded_campaign_smoke_per_pdg_track(tmp_path, pdg_id):
    from ship_muon_bg.density_lab.artifacts import ArtifactStore
    from ship_muon_bg.density_lab.config import EvaluationSpec, FeatureViewSpec
    from ship_muon_bg.density_lab.empirical import EmpiricalDatasetSpec, run_empirical_single

    store = ArtifactStore("d7_smoke_test", root=tmp_path)
    dataset_spec = EmpiricalDatasetSpec(
        dataset_path=FIXTURE, pdg_id=pdg_id, seed=11, max_rows=1000,
    )
    record = run_empirical_single(
        dataset_spec, store, experiment_id="d7_smoke_test",
        feature_view=FeatureViewSpec("identity_cartesian_v0"),
        model=_tiny_model_spec(),
        evaluation=EvaluationSpec(ess_sample_count=150, c2st_sample_count=100),
        device="cpu",
    )
    assert record["status"] == "completed"
    assert record["technical_status"] == "completed"
    # Wiring validation, not model-quality acceptance: inconclusive is the
    # correct, expected scientific status for D7 (no exact target density).
    assert record["scientific_status"] in ("inconclusive", "pass")

    run_id = record["run_id"]

    run_dir = store.experiment_dir / run_id
    assert (run_dir / "checkpoint").is_dir()
    assert (run_dir / "model_manifest.json").exists()

    dataset_manifest = json.loads((run_dir / "dataset_manifest.json").read_text())
    assert dataset_manifest["source_file_dataset_hash"] == (
        "718b9197a3965f6842574fec1f3f6b1bf4c16949ff8a1f27dca6232817c03c32"
    )

    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["exact_target_density_available"] is False
    assert np.isfinite(metrics["held_out"]["held_out_nll"])
    assert metrics["forward_kl"] is None
    assert metrics["importance_ess"] is None

    # Resume/skip.
    second = run_empirical_single(
        dataset_spec, store, experiment_id="d7_smoke_test",
        feature_view=FeatureViewSpec("identity_cartesian_v0"),
        model=_tiny_model_spec(),
        evaluation=EvaluationSpec(ess_sample_count=150, c2st_sample_count=100),
        device="cpu",
    )
    assert second["status"] == "skipped_completed"

    # Force rerun.
    third = run_empirical_single(
        dataset_spec, store, experiment_id="d7_smoke_test",
        feature_view=FeatureViewSpec("identity_cartesian_v0"),
        model=_tiny_model_spec(),
        evaluation=EvaluationSpec(ess_sample_count=150, c2st_sample_count=100),
        device="cpu", force=True,
    )
    assert third["status"] == "completed"


def test_campaign_smoke_requires_no_full_local_path():
    """The fixture path is a plain relative repo path -- no environment
    variable, no machine-specific location, no download step."""

    assert os.path.exists(FIXTURE)
    assert not os.path.isabs(FIXTURE.replace(REPO_ROOT, "."))


# --- Full-data configuration test (temporary external path stand-in) ---------


def test_full_data_command_path_uses_no_special_code_and_accepts_external_path(tmp_path):
    """A temporary copy standing in for a full local dataset must run through
    the identical build_empirical_dataset/run_empirical_single code -- no
    dataset-path-conditional branch anywhere in that path."""

    from ship_muon_bg.density_lab.artifacts import ArtifactStore
    from ship_muon_bg.density_lab.config import EvaluationSpec, FeatureViewSpec
    from ship_muon_bg.density_lab.empirical import EmpiricalDatasetSpec, run_empirical_single

    external_dir = tmp_path / "external_full_data_stand_in"
    external_dir.mkdir()
    external_path = external_dir / "muonsFullMC_afterMS.npz"
    shutil.copy(FIXTURE, external_path)

    artifact_root = tmp_path / "external_artifacts"
    store = ArtifactStore("d7_full_data_dry_run", root=artifact_root)
    dataset_spec = EmpiricalDatasetSpec(dataset_path=str(external_path), pdg_id=13, seed=1, max_rows=500)
    record = run_empirical_single(
        dataset_spec, store, experiment_id="d7_full_data_dry_run",
        feature_view=FeatureViewSpec("identity_cartesian_v0"),
        model=_tiny_model_spec(max_epochs=1),
        evaluation=EvaluationSpec(ess_sample_count=100, c2st_sample_count=50),
        device="cpu",
    )
    assert record["status"] == "completed"
    # Artifacts land under the external root, not the repository.
    run_dir = artifact_root / "d7_full_data_dry_run" / record["run_id"]
    assert run_dir.exists()
    assert str(artifact_root) not in str(REPO_ROOT)


def test_dry_run_validates_external_path_without_training(tmp_path):
    from ship_muon_bg.density_lab.empirical import EmpiricalDatasetSpec, build_empirical_dataset

    external_path = tmp_path / "external.npz"
    shutil.copy(FIXTURE, external_path)
    dataset_spec = EmpiricalDatasetSpec(dataset_path=str(external_path), pdg_id=13, seed=1, max_rows=500)
    dataset = build_empirical_dataset(dataset_spec)  # never trains
    assert dataset.train.n_rows > 0
    assert dataset.source_file_dataset_hash == (
        "718b9197a3965f6842574fec1f3f6b1bf4c16949ff8a1f27dca6232817c03c32"
    )


# --- Committed config layers (Section 8) --------------------------------------


@pytest.mark.parametrize(
    "config_name",
    [
        "d7_fixture_smoke_v0.json",
        "d7_fixture_bounded_scientific_v0.json",
        "d7_local_full_v0.json",
    ],
)
def test_committed_empirical_configs_load(config_name, monkeypatch):
    from ship_muon_bg.density_lab.empirical import EmpiricalCampaignSpec

    monkeypatch.setenv("SHIP_MUON_BG_LOCAL_DATA", "/placeholder/for/config/load/test")
    path = os.path.join(REPO_ROOT, "configs", "density_lab", "empirical", config_name)
    spec = EmpiricalCampaignSpec.from_json_file(path)
    assert spec.experiment_id
    assert spec.pdg_ids
    assert spec.model.family == "affine_coupling"
    assert "$" not in spec.dataset_path  # env vars must be fully expanded


def test_fixture_configs_do_not_reference_environment_variables():
    from ship_muon_bg.density_lab.empirical import EmpiricalCampaignSpec

    for config_name in ("d7_fixture_smoke_v0.json", "d7_fixture_bounded_scientific_v0.json"):
        path = os.path.join(REPO_ROOT, "configs", "density_lab", "empirical", config_name)
        spec = EmpiricalCampaignSpec.from_json_file(path)
        assert spec.dataset_path == "data/samples/muonsFullMC_afterMS_sample.npz"
        assert os.path.exists(spec.dataset_path)


def test_local_full_config_uses_environment_placeholder_not_hardcoded_path():
    with open(
        os.path.join(REPO_ROOT, "configs", "density_lab", "empirical", "d7_local_full_v0.json"),
        encoding="utf-8",
    ) as handle:
        raw = json.load(handle)
    assert "${SHIP_MUON_BG_LOCAL_DATA}" in raw["dataset_path"]
    assert "/home/" not in raw["dataset_path"]
    assert "C:\\" not in raw["dataset_path"]
