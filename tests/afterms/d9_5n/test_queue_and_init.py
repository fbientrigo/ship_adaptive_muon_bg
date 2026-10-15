"""Frozen queue resolution, init idempotency, and disk-derived run-state
mapping. Never touches the real production scout report or shard scope --
``tiny_scout_report``/``tmp_path`` only.
"""

from __future__ import annotations

import json

import pytest

from ship_muon_bg.afterms.d9 import runner as d9runner
from ship_muon_bg.afterms.d9_5 import model_adapter as ma
from ship_muon_bg.afterms.d9_5 import nightly_runner as nr


def test_queue_order_is_deterministic_six_items_nf_ac_only(tmp_path, tiny_scout_report):
    items = nr.resolve_run_queue_items(tmp_path, scout_report_path=tiny_scout_report)
    assert [(it["track_id"], it["seed"]) for it in items] == list(nr.FROZEN_QUEUE)
    assert len(items) == 6
    assert all(it["model_family"] == "NF_AC" for it in items)


def test_queue_has_no_test_evaluation_phase(tmp_path, tiny_scout_report):
    items = nr.resolve_run_queue_items(tmp_path, scout_report_path=tiny_scout_report)
    # The nightly queue only ever names fit/train work items -- never an
    # evaluate-test phase, and it holds no reference to the test split.
    blob = json.dumps(items)
    assert "evaluate" not in blob.lower()
    assert "test_shard" not in blob.lower()


def test_resolve_run_queue_items_is_deterministic_across_calls(tmp_path, tiny_scout_report):
    a = nr.resolve_run_queue_items(tmp_path, scout_report_path=tiny_scout_report)
    b = nr.resolve_run_queue_items(tmp_path, scout_report_path=tiny_scout_report)
    assert a == b


def test_init_creates_expected_tree_and_files(tmp_path, tiny_scout_report):
    result = nr.init_nightly_runner(tmp_path, scout_report_path=tiny_scout_report)
    assert result["action"] == "initialized"
    root = nr.nightly_root(tmp_path)
    for sub in ("incidents", "logs", "nightly_blocks", "locks"):
        assert (root / sub).is_dir()
    queue = ma.read_json(root / "run_queue.json")
    assert queue["schema_id"] == "d9_5_nightly_runner_state_v0"
    assert len(queue["items"]) == 6
    campaign_state = ma.read_json(root / "campaign_state.json")
    assert campaign_state["blocks_completed"] == 0


def test_init_is_idempotent_when_queue_matches(tmp_path, tiny_scout_report):
    first = nr.init_nightly_runner(tmp_path, scout_report_path=tiny_scout_report)
    assert first["action"] == "initialized"
    second = nr.init_nightly_runner(tmp_path, scout_report_path=tiny_scout_report)
    assert second["action"] == "noop_already_initialized"


def test_init_refuses_incompatible_queue_overwrite(tmp_path, tiny_scout_report):
    nr.init_nightly_runner(tmp_path, scout_report_path=tiny_scout_report)
    queue_path = nr.nightly_root(tmp_path) / "run_queue.json"
    payload = ma.read_json(queue_path)
    payload["items"][0]["model_config_id"] = "NF_AC_DIFFERENT_ID"
    ma.atomic_write_json(queue_path, payload)

    with pytest.raises(nr.IncompatibleQueueError):
        nr.init_nightly_runner(tmp_path, scout_report_path=tiny_scout_report)


def test_derive_run_state_pending_when_no_status(tmp_path):
    state, status = nr.derive_run_state(tmp_path / "runs" / "does_not_exist")
    assert state == nr.RUN_PENDING
    assert status is None


def test_derive_run_state_completed(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    ma.atomic_write_json(run_dir / "status.json", {"status": d9runner.STATUS_COMPLETED, "final_epoch": 3})
    state, status = nr.derive_run_state(run_dir)
    assert state == nr.RUN_COMPLETED
    assert status["final_epoch"] == 3


def test_derive_run_state_interrupted_without_marker_is_retryable(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    ma.atomic_write_json(run_dir / "status.json", {"status": d9runner.STATUS_INTERRUPTED, "last_completed_epoch": 1})
    state, status = nr.derive_run_state(run_dir)
    assert state == nr.RUN_FAILED_RETRYABLE


def test_derive_run_state_interrupted_with_deadline_marker(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    ma.atomic_write_json(run_dir / "status.json", {"status": d9runner.STATUS_INTERRUPTED, "last_completed_epoch": 1})
    nr._write_deadline_marker(run_dir, "block_test")
    state, status = nr.derive_run_state(run_dir)
    assert state == nr.RUN_INTERRUPTED_AT_BLOCK_DEADLINE


def test_derive_run_state_contract_mismatch_is_blocked(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    ma.atomic_write_json(run_dir / "status.json", {"status": d9runner.STATUS_EXCLUDED_CONTRACT_MISMATCH})
    state, _ = nr.derive_run_state(run_dir)
    assert state == nr.RUN_FAILED_BLOCKED


def test_determine_current_run_walks_in_order_and_none_when_all_complete(tmp_path, tiny_scout_report):
    nr.init_nightly_runner(tmp_path, scout_report_path=tiny_scout_report)
    items = nr.reconcile_queue_and_persist(tmp_path)
    current = nr.determine_current_run(items, tmp_path)
    assert current["track_id"] == "TRK_PDG13_UW_ID"
    assert current["seed"] == 20260720

    for item in items:
        run_dir = tmp_path / item["run_dir"]
        run_dir.mkdir(parents=True, exist_ok=True)
        ma.atomic_write_json(run_dir / "status.json", {"status": d9runner.STATUS_COMPLETED, "final_epoch": 1})
    items = nr.reconcile_queue_and_persist(tmp_path)
    assert nr.determine_current_run(items, tmp_path) is None


def test_reconcile_fills_hash_fields_from_status_json(tmp_path, tiny_scout_report):
    nr.init_nightly_runner(tmp_path, scout_report_path=tiny_scout_report)
    items = nr.resolve_run_queue_items(tmp_path, scout_report_path=tiny_scout_report)
    run_dir = tmp_path / items[0]["run_dir"]
    run_dir.mkdir(parents=True, exist_ok=True)
    ma.atomic_write_json(run_dir / "status.json", {
        "status": d9runner.STATUS_COMPLETED, "final_epoch": 2,
        "semantic_training_hash": "abc123", "execution_policy_hash": "def456",
    })
    reconciled = nr.reconcile_queue_and_persist(tmp_path)
    assert reconciled[0]["semantic_training_hash"] == "abc123"
    assert reconciled[0]["execution_policy_hash"] == "def456"
    # sampling_contract_version is never written into status.json by the
    # frozen runner -- stays null, per spec ("read when present, else null").
    assert reconciled[0]["sampling_contract_version"] is None


def test_canary_artifact_root_is_never_conflated_with_production_root(tmp_path):
    canary_root = tmp_path / "afterms_d9_5_nightly_canary_v0"
    production_root = tmp_path / "afterms_d9_5_model_family_arena_v0"
    assert nr.nightly_root(canary_root) != nr.nightly_root(production_root)
    assert "canary" in str(nr.nightly_root(canary_root))
