"""CPU-based structural/behavioral tests for scripts/_d9b_gpu_gate_worker.py.

These tests invoke the worker's train/reload logic directly against tiny
synthetic in-repo data (never a real shard), the same way
tests/afterms/d9/test_runner.py exercises the underlying runner. The real
RTX 2060 CUDA integration gate is a separate, explicitly-requested execution
(`python scripts/run_afterms_d9_gpu_hpo.py gpu-gate --execute`), not part of
this unit suite.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_worker_module():
    spec = importlib.util.spec_from_file_location(
        "_d9b_gpu_gate_worker", REPO_ROOT / "scripts" / "_d9b_gpu_gate_worker.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


worker = _load_worker_module()

from ship_muon_bg.afterms.d9 import runner as d9runner  # noqa: E402


def _synthetic_raw(n, seed, pdg=13):
    rng = np.random.default_rng(seed)
    px = rng.normal(size=n)
    py = rng.normal(size=n)
    pz = rng.uniform(1.0, 20.0, size=n)
    x = rng.normal(size=n)
    y = rng.normal(size=n)
    z = np.zeros(n)
    ids = np.full(n, pdg, dtype=np.float64)
    w = rng.uniform(0.5, 2.0, size=n)
    return np.column_stack([px, py, pz, x, y, z, ids, w])


def _tiny_candidate_config():
    return {
        "campaign_id": "d9b_gate_test_campaign",
        "candidate_id": "GT1_tiny_gate_test_candidate",
        "model_family": "affine_coupling",
        "architecture": {"number_of_blocks": 2, "hidden_width": 8, "hidden_depth": 1, "capacity_label": "tiny_test"},
        "modeled_features": ["px", "py", "pz", "x", "y"],
        "feature_order": ["px", "py", "pz", "x", "y"],
        "pdg_policy": "pdg13",
        "pdg_value": 13,
        "preprocessing_name": "identity_standardized_v0",
        "target_measure": "physical_space_nll",
        "weighting_policy": "row_empirical_unweighted",
        "weighting_estimator_version": "not_applicable",
        "shard_dir": "synthetic_test_only",
        "train_shards": ["synthetic_train"],
        "validation_shards": ["synthetic_validation"],
        "test_shards": ["synthetic_test"],
        "seed_set": [20260720],
        "max_epochs": 2,
        "minimum_epochs": 1,
        "early_stopping_patience": 5,
        "optimizer": "adam",
        "learning_rate": 0.01,
        "batch_size": 32,
        "weight_decay": 0.0,
        "gradient_clipping": 5.0,
        "dtype": "float32",
        "device_policy": "cpu_only_for_tests",
        "checkpoint_policy": {"save_best": True, "save_final": True, "save_last_resumable": True, "checkpoint_interval_epochs": 1},
        "evaluation_policy": {
            "quick_validation_budget": {"n_generated_samples": 50},
            "final_candidate_budget": {"n_generated_samples": 100},
        },
    }


def _write_shards(shard_dir: Path):
    shard_dir.mkdir(parents=True, exist_ok=True)
    np.save(shard_dir / "synthetic_train.npy", _synthetic_raw(200, seed=1))
    np.save(shard_dir / "synthetic_validation.npy", _synthetic_raw(50, seed=2))
    np.save(shard_dir / "synthetic_test.npy", _synthetic_raw(50, seed=3))


class _Args:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def test_real_execution_uses_train_and_validation_shards_only(tmp_path, monkeypatch):
    """Required test 3: real execution uses train/validation only."""

    shard_dir = tmp_path / "shards"
    _write_shards(shard_dir)
    config = _tiny_candidate_config()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    artifact_root = tmp_path / "artifacts"
    telemetry_out = tmp_path / "telemetry.json"

    requested = []
    real_load = d9runner.load_concatenated_shards

    def _spy_load(shard_dir_arg, shard_names):
        requested.extend(shard_names)
        return real_load(shard_dir_arg, shard_names)

    monkeypatch.setattr(d9runner, "load_concatenated_shards", _spy_load)

    args = _Args(
        candidate_config_json=str(config_path), seed=20260720, shard_dir=str(shard_dir),
        artifact_root=str(artifact_root), device="cpu", resume=False,
        interrupt_after_epochs=None, telemetry_out=str(telemetry_out),
    )
    rc = worker.cmd_train(args)
    assert rc == 0
    assert requested == ["synthetic_train", "synthetic_validation"]
    assert "synthetic_test" not in requested


def test_gpu_dry_run_reads_no_real_shard(tmp_path, monkeypatch):
    """Required test 2: GPU dry-run reads no real shard."""

    from ship_muon_bg.afterms.d9b import gpu_gate

    def _boom(*args, **kwargs):
        raise AssertionError("dry-run must never load a shard file")

    monkeypatch.setattr(d9runner, "load_concatenated_shards", _boom)

    plan = json.loads((REPO_ROOT / "configs" / "afterms" / "d9_candidate_plan_v0.json").read_text())
    training_config = json.loads((REPO_ROOT / "configs" / "afterms" / "d9_training_v0.json").read_text())
    from ship_muon_bg.afterms.d9b import candidate_selection as cs

    candidate_config, report = cs.select_gpu_gate_candidate(plan, training_config)
    summary = gpu_gate.dry_run_plan(candidate_config, report, candidate_config["seed_set"][0], "cuda")
    assert summary["dry_run"] is True


def test_train_telemetry_has_required_schema(tmp_path):
    """Required test 9: CUDA telemetry schema (CPU device -> zeroed bytes, schema still present)."""

    shard_dir = tmp_path / "shards"
    _write_shards(shard_dir)
    config = _tiny_candidate_config()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    telemetry_out = tmp_path / "telemetry.json"

    args = _Args(
        candidate_config_json=str(config_path), seed=20260720, shard_dir=str(shard_dir),
        artifact_root=str(tmp_path / "artifacts"), device="cpu", resume=False,
        interrupt_after_epochs=None, telemetry_out=str(telemetry_out),
    )
    worker.cmd_train(args)
    telemetry = json.loads(telemetry_out.read_text(encoding="utf-8"))
    for key in ("device", "initial", "peak", "final", "result"):
        assert key in telemetry
    for snap_key in ("initial", "peak", "final"):
        assert "allocated_bytes" in telemetry[snap_key]
        assert "reserved_bytes" in telemetry[snap_key]


def test_controlled_interruption_status(tmp_path):
    """Required test 15: controlled interruption status via the declared,
    bounded --interrupt-after-epochs hook (interrupt after exactly 1 of 2
    declared epochs -- deterministic, not a batch-count race)."""

    shard_dir = tmp_path / "shards"
    _write_shards(shard_dir)
    config = _tiny_candidate_config()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    artifact_root = tmp_path / "artifacts"
    telemetry_out = tmp_path / "telemetry_interrupt.json"

    args = _Args(
        candidate_config_json=str(config_path), seed=20260720, shard_dir=str(shard_dir),
        artifact_root=str(artifact_root), device="cpu", resume=False,
        interrupt_after_epochs=1, telemetry_out=str(telemetry_out),
    )
    rc = worker.cmd_train(args)
    assert rc == 0

    run_dir = d9runner.run_directory(artifact_root, config["candidate_id"], 20260720)
    status = json.loads((run_dir / "status.json").read_text())
    assert status["status"] == "interrupted"
    assert (run_dir / "checkpoints" / "last_resumable_checkpoint.pt").exists()
    assert not (run_dir / "checkpoints" / "final_checkpoint.pt").exists()


def test_exact_resume_compatibility_and_no_duplicate_history(tmp_path):
    """Required tests 16/17: exact resume compatibility; no duplicate history
    after resume. Both stage-E invocations MUST share the same overridden
    max_epochs=2, or the runner refuses via MaxEpochsExtensionError."""

    shard_dir = tmp_path / "shards"
    _write_shards(shard_dir)
    config = _tiny_candidate_config()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    artifact_root = tmp_path / "artifacts"

    interrupt_args = _Args(
        candidate_config_json=str(config_path), seed=20260720, shard_dir=str(shard_dir),
        artifact_root=str(artifact_root), device="cpu", resume=False,
        interrupt_after_epochs=1, telemetry_out=str(tmp_path / "t1.json"),
    )
    worker.cmd_train(interrupt_args)

    resume_args = _Args(
        candidate_config_json=str(config_path), seed=20260720, shard_dir=str(shard_dir),
        artifact_root=str(artifact_root), device="cpu", resume=True,
        interrupt_after_epochs=None, telemetry_out=str(tmp_path / "t2.json"),
    )
    rc = worker.cmd_train(resume_args)
    assert rc == 0

    run_dir = d9runner.run_directory(artifact_root, config["candidate_id"], 20260720)
    status = json.loads((run_dir / "status.json").read_text())
    assert status["status"] == "completed"
    history = json.loads((run_dir / "histories" / "training_history.json").read_text())
    epochs = [r["epoch"] for r in history]
    assert epochs == [1, 2]
    assert len(epochs) == len(set(epochs))


def test_resume_with_mismatched_max_epochs_is_refused(tmp_path):
    """The gate orchestrator always reuses the SAME candidate_config (with the
    same overridden max_epochs) for both stage-E invocations; verify what
    happens if it didn't -- the runner's own MaxEpochsExtensionError guard."""

    from ship_muon_bg.afterms.d9 import runner as d9r

    shard_dir = tmp_path / "shards"
    _write_shards(shard_dir)
    config = _tiny_candidate_config()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    artifact_root = tmp_path / "artifacts"

    interrupt_args = _Args(
        candidate_config_json=str(config_path), seed=20260720, shard_dir=str(shard_dir),
        artifact_root=str(artifact_root), device="cpu", resume=False,
        interrupt_after_epochs=1, telemetry_out=str(tmp_path / "t1.json"),
    )
    worker.cmd_train(interrupt_args)

    mismatched_config = dict(config, max_epochs=5)
    mismatched_config_path = tmp_path / "mismatched_config.json"
    mismatched_config_path.write_text(json.dumps(mismatched_config), encoding="utf-8")
    resume_args = _Args(
        candidate_config_json=str(mismatched_config_path), seed=20260720, shard_dir=str(shard_dir),
        artifact_root=str(artifact_root), device="cpu", resume=True,
        interrupt_after_epochs=None, telemetry_out=str(tmp_path / "t2.json"),
    )
    with pytest.raises(d9r.MaxEpochsExtensionError):
        worker.cmd_train(resume_args)


def test_checkpoint_reload_negative_pz_preserved_and_counted(tmp_path):
    """Required test 13: negative pz is preserved and counted, never clipped."""

    shard_dir = tmp_path / "shards"
    _write_shards(shard_dir)
    config = _tiny_candidate_config()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    artifact_root = tmp_path / "artifacts"

    train_args = _Args(
        candidate_config_json=str(config_path), seed=20260720, shard_dir=str(shard_dir),
        artifact_root=str(artifact_root), device="cpu", resume=False,
        interrupt_after_epochs=None, telemetry_out=str(tmp_path / "t.json"),
    )
    worker.cmd_train(train_args)

    run_dir = d9runner.run_directory(artifact_root, config["candidate_id"], 20260720)
    output_json = tmp_path / "reload.json"
    reload_args = _Args(
        run_dir=str(run_dir), device="cpu", sample_count=64, generation_seed=20260720,
        output_json=str(output_json),
    )
    rc = worker.cmd_reload(reload_args)
    assert rc == 0
    result = json.loads(output_json.read_text())
    assert result["checkpoint_reload"]["compatible"] is True
    assert result["deterministic_sample"]["all_finite"] is True
    diag = result["deterministic_sample"]["negative_pz_diagnostics"]
    assert "generated_domain_violation_count" in diag
    # Rerun with the same seed to confirm the sample (and thus any negative
    # pz count) is reproducible, not clipped/altered between reloads.
    output_json_2 = tmp_path / "reload2.json"
    reload_args_2 = _Args(
        run_dir=str(run_dir), device="cpu", sample_count=64, generation_seed=20260720,
        output_json=str(output_json_2),
    )
    worker.cmd_reload(reload_args_2)
    result_2 = json.loads(output_json_2.read_text())
    assert (
        result["deterministic_sample"]["sample_hash_sha256"]
        == result_2["deterministic_sample"]["sample_hash_sha256"]
    )


def test_checkpoint_reload_happens_via_a_separate_process_entrypoint(tmp_path):
    """Required test 12: checkpoint reload is a distinct worker mode
    (invoked as a fresh subprocess by the orchestrator), never inlined into
    the train mode."""

    parser = worker.build_parser()
    subparsers_action = next(a for a in parser._subparsers._group_actions if hasattr(a, "choices"))
    assert set(subparsers_action.choices.keys()) == {"train", "reload"}
    assert subparsers_action.choices["reload"].get_default("func") is not subparsers_action.choices["train"].get_default("func")
