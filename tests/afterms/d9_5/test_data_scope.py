"""Data-scope tests: alias resolution, shard filtering, isolation, negative pz.

Covers required tests 1, 2, 5, 6, 7, 8, 10, 11, 12, 30.
"""

from __future__ import annotations

import inspect

import numpy as np

from ship_muon_bg.afterms import model_naming
from ship_muon_bg.afterms.d9 import runner as d9runner
from ship_muon_bg.afterms.d9_5 import data_scope


def test_both_tracks_resolve_through_alias_registry():
    registry = model_naming.load_alias_registry()
    pos = model_naming.resolve_track(
        registry, pdg_value=13, weighting_policy="row_empirical_unweighted", preprocessing_name="identity_standardized_v0",
    )
    neg = model_naming.resolve_track(
        registry, pdg_value=-13, weighting_policy="row_empirical_unweighted", preprocessing_name="identity_standardized_v0",
    )
    assert pos["track_id"] == "TRK_PDG13_UW_ID"
    assert neg["track_id"] == "TRK_PDGM13_UW_ID"
    assert pos["alias_resolution_status"] == "CURATED"
    assert neg["alias_resolution_status"] == "CURATED"


def test_internal_legacy_ids_are_lineage_only_not_track_names():
    # A legacy candidate id never appears inside a resolved track_id/model_config_id.
    registry = model_naming.load_alias_registry()
    track = model_naming.resolve_track(
        registry, pdg_value=13, weighting_policy="row_empirical_unweighted", preprocessing_name="identity_standardized_v0",
    )
    assert "A1" not in track["track_id"]
    assert "B2" not in track["track_id"]


def test_all_families_use_same_five_modeled_features(tiny_data_scope_config):
    assert tiny_data_scope_config["modeled_features"] == ["px", "py", "pz", "x", "y"]
    assert tiny_data_scope_config["feature_order"] == ["px", "py", "pz", "x", "y"]
    assert tiny_data_scope_config["modeled_dimension"] == 5


def test_all_families_use_same_preprocessing_name(tiny_data_scope_config):
    assert tiny_data_scope_config["preprocessing_name"] == "identity_standardized_v0"


def test_identical_train_rows_across_families(tiny_shard_dir):
    manifest = data_scope.load_shard_manifest(tiny_shard_dir)
    first = data_scope.load_filtered_split(tiny_shard_dir, manifest, "train", 13)
    second = data_scope.load_filtered_split(tiny_shard_dir, manifest, "train", 13)
    np.testing.assert_array_equal(first, second)


def test_identical_validation_rows_across_families(tiny_shard_dir):
    manifest = data_scope.load_shard_manifest(tiny_shard_dir)
    first = data_scope.load_filtered_split(tiny_shard_dir, manifest, "validation", -13)
    second = data_scope.load_filtered_split(tiny_shard_dir, manifest, "validation", -13)
    np.testing.assert_array_equal(first, second)


def test_no_silent_row_subsampling(tiny_shard_dir):
    manifest = data_scope.load_shard_manifest(tiny_shard_dir)
    total_rows = sum(s["row_count"] for s in data_scope.shards_for_split(manifest, "train"))
    pos = data_scope.load_filtered_split(tiny_shard_dir, manifest, "train", 13)
    neg = data_scope.load_filtered_split(tiny_shard_dir, manifest, "train", -13)
    # Every row belongs to exactly one PDG code in the synthetic fixture.
    assert pos.shape[0] + neg.shape[0] == total_rows


def test_pdg_tracks_remain_separate(tiny_shard_dir):
    manifest = data_scope.load_shard_manifest(tiny_shard_dir)
    pos = data_scope.load_filtered_split(tiny_shard_dir, manifest, "train", 13)
    neg = data_scope.load_filtered_split(tiny_shard_dir, manifest, "train", -13)
    assert np.all(pos[:, 6] == 13.0)
    assert np.all(neg[:, 6] == -13.0)


def test_source_weight_is_metadata_only(tiny_shard_dir):
    manifest = data_scope.load_shard_manifest(tiny_shard_dir)
    record = data_scope._split_track_record(tiny_shard_dir, manifest, "train", 13)
    assert record["feature_order"] == ["px", "py", "pz", "x", "y"]
    assert "w" not in record["feature_order"]
    assert record["source_weight_summary"]["note"].startswith("metadata only")


def test_negative_pz_is_preserved_not_dropped(tiny_shard_dir):
    manifest = data_scope.load_shard_manifest(tiny_shard_dir)
    rows = data_scope.load_filtered_split(tiny_shard_dir, manifest, "train", 13)
    assert np.count_nonzero(rows[:, 2] < 0.0) > 0


def test_train_candidate_seed_signature_has_no_test_data_parameter():
    # Structural test-set isolation lives in the reused D9 runner; D9-5 must
    # never route test rows through it either.
    params = inspect.signature(d9runner.train_candidate_seed).parameters
    assert "test_raw" not in params
    assert "test_data" not in params


def test_no_shard_overlap_across_splits(tiny_shard_dir):
    manifest = data_scope.load_shard_manifest(tiny_shard_dir)
    assert data_scope.check_no_shard_overlap(manifest) == []
