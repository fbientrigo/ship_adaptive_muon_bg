"""NF_AC adapter tests: scout-derived config, no scout-checkpoint reuse,
multi-seed identity, exact resume, alias/hash separation.

Covers required tests 21, 22, 23, 24, 25.
"""

from __future__ import annotations

import pytest

from ship_muon_bg.afterms.d9 import runner as d9runner
from ship_muon_bg.afterms.d9_5 import data_scope, nf_ac_adapter


@pytest.fixture
def make_adapter(tiny_nf_architecture, tiny_nf_execution_policy, tiny_nf_optimizer_settings, tiny_evaluation_policy):
    def _make(track_id, pdg_value, artifact_root, execution_policy=None):
        return nf_ac_adapter.NfAcAdapter(
            track_id=track_id, pdg_value=pdg_value, architecture=tiny_nf_architecture,
            execution_policy=execution_policy or tiny_nf_execution_policy,
            optimizer_settings=tiny_nf_optimizer_settings, evaluation_policy=tiny_evaluation_policy,
            artifact_root=artifact_root, device="cpu",
        )
    return _make


def test_scout_config_derived_from_named_report_argmin(tiny_scout_report):
    resolved = nf_ac_adapter.select_scout_promoted_nf_config("TRK_PDG13_UW_ID", scout_report_path=tiny_scout_report)
    assert resolved["model_config_id"] == "NF_AC_b04_w016_d01"  # lower val NLL among completed variants
    assert resolved["source_run_id"] == "toy_a__cap_2.0x"
    assert resolved["config_label"] == "scout_promoted_nf_config"
    assert resolved["scout_validation_metric"] == 1.5


def test_scout_selection_excludes_non_completed_rows(tiny_scout_report):
    resolved = nf_ac_adapter.select_scout_promoted_nf_config("TRK_PDG13_UW_ID", scout_report_path=tiny_scout_report)
    # best_validation_metric=0.1 belongs to a status=="failed_technical" row and must never win.
    assert resolved["scout_validation_metric"] != 0.1


def test_scout_selection_raises_for_unknown_track(tiny_scout_report):
    with pytest.raises(nf_ac_adapter.ScoutSelectionError):
        nf_ac_adapter.select_scout_promoted_nf_config("TRK_UNKNOWN", scout_report_path=tiny_scout_report)


def test_alias_metadata_does_not_alter_semantic_hash(make_adapter, tmp_path):
    adapter = make_adapter("TRK_PDG13_UW_ID", 13, tmp_path)
    candidate_config = adapter._build_candidate_config(seed=20260720)
    # No capacity_label / alias string leaks into the dict that feeds
    # semantic_training_hash wholesale (d9/contract.py).
    assert set(candidate_config["architecture"].keys()) == {"number_of_blocks", "hidden_width", "hidden_depth"}


def test_nf_run_dir_is_isolated_from_scout_artifacts(make_adapter, tiny_shard_dir, tmp_path):
    manifest = data_scope.load_shard_manifest(tiny_shard_dir)
    train_raw = data_scope.load_filtered_split(tiny_shard_dir, manifest, "train", 13)
    validation_raw = data_scope.load_filtered_split(tiny_shard_dir, manifest, "validation", 13)

    adapter = make_adapter("TRK_PDG13_UW_ID", 13, tmp_path)
    result = adapter.fit(train_raw, validation_raw, seed=20260720)

    assert result["status"] == d9runner.STATUS_COMPLETED
    assert str(adapter._run_dir).startswith(str(tmp_path))
    assert "afterms_d9_model_arena_v0" not in str(adapter._run_dir)


def test_nf_multi_seed_run_identities_are_distinct(make_adapter, tiny_shard_dir, tmp_path):
    manifest = data_scope.load_shard_manifest(tiny_shard_dir)
    train_raw = data_scope.load_filtered_split(tiny_shard_dir, manifest, "train", 13)
    validation_raw = data_scope.load_filtered_split(tiny_shard_dir, manifest, "validation", 13)

    adapter_a = make_adapter("TRK_PDG13_UW_ID", 13, tmp_path)
    result_a = adapter_a.fit(train_raw, validation_raw, seed=20260720)
    adapter_b = make_adapter("TRK_PDG13_UW_ID", 13, tmp_path)
    result_b = adapter_b.fit(train_raw, validation_raw, seed=20260721)

    assert result_a["status"] == d9runner.STATUS_COMPLETED
    assert result_b["status"] == d9runner.STATUS_COMPLETED
    assert adapter_a._run_dir != adapter_b._run_dir
    assert adapter_a._run_dir.name == "seed_20260720"
    assert adapter_b._run_dir.name == "seed_20260721"


def test_nf_exact_resume_remains_valid(make_adapter, tiny_shard_dir, tmp_path):
    manifest = data_scope.load_shard_manifest(tiny_shard_dir)
    train_raw = data_scope.load_filtered_split(tiny_shard_dir, manifest, "train", 13)
    validation_raw = data_scope.load_filtered_split(tiny_shard_dir, manifest, "validation", 13)
    execution_policy = {"minimum_epochs": 1, "maximum_epochs": 3, "early_stopping_patience": 5}

    epoch_polls = {"count": 0}

    def interrupt_after_epoch_one():
        epoch_polls["count"] += 1
        return epoch_polls["count"] > 1

    adapter1 = make_adapter("TRK_PDG13_UW_ID", 13, tmp_path, execution_policy=execution_policy)
    result1 = adapter1.fit(train_raw, validation_raw, seed=20260720, interrupt_flag=interrupt_after_epoch_one)
    assert result1["status"] == d9runner.STATUS_INTERRUPTED

    adapter2 = make_adapter("TRK_PDG13_UW_ID", 13, tmp_path, execution_policy=execution_policy)
    result2 = adapter2.fit(train_raw, validation_raw, seed=20260720, resume=True)
    assert result2["status"] == d9runner.STATUS_COMPLETED
    assert result2["final_epoch"] >= 2


def test_nf_deterministic_sampling_after_reload(make_adapter, tiny_shard_dir, tmp_path):
    manifest = data_scope.load_shard_manifest(tiny_shard_dir)
    train_raw = data_scope.load_filtered_split(tiny_shard_dir, manifest, "train", 13)
    validation_raw = data_scope.load_filtered_split(tiny_shard_dir, manifest, "validation", 13)

    adapter = make_adapter("TRK_PDG13_UW_ID", 13, tmp_path)
    adapter.fit(train_raw, validation_raw, seed=20260720)
    adapter.save_bundle(adapter._run_dir)

    reloaded = nf_ac_adapter.NfAcAdapter.load_bundle(adapter._run_dir)
    import numpy as np
    np.testing.assert_allclose(adapter.sample(20, seed=5), reloaded.sample(20, seed=5))
