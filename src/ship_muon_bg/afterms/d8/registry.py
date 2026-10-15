"""Phase B (D8 spec §5-§6): canonical run registry over frozen D7 jobs 04-12.

Builds one `RunRecord` per historical run, reading only `metrics.json` /
`metrics_run{1,2}.json` under `<input_artifact_dir>/jobs/<job_name>/` plus the
frozen contract in `legacy_d7_adapter`. Never re-derives config from tensor
shapes; every architecture fact comes from that adapter's producer-verified
tables.

Loss semantics (D8 spec §6), fixed by reading the producer's evaluation code
path directly (`run_neural_training_subprocess` @ f9d8246):

- Test-set NLL (`test_feature_space_nll` / `physical_space_nll` in
  `metrics.json`) is ALWAYS a plain row-empirical mean over `test_shard_000`
  -- the producer's test evaluation never applies sample weights, even for
  jobs whose *training* loop was weighted. `feature_space_nll` and `test_nll`
  below both surface this same row-empirical test quantity; `physical_space_nll`
  is its physical-space counterpart where the coordinate Jacobian is defined
  (never for `quantile_normal_v0`, per §4.4).
- Per-epoch `train_loss`/`validation_loss` in `history` ARE the training
  target actually optimized: row-empirical for unweighted runs, or
  `self_normalized_minibatch_ratio` (`sum(w*-logp)/sum(w)` per minibatch,
  confirmed from source -- not a fixed global normalization or resampling)
  for weighted runs. `validation_nll` here is read from that history, so a
  weighted run's `validation_nll` is a weighted quantity even though its
  `test_nll` is not; this is a source-verified fact, not an inconsistency.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ship_muon_bg.afterms.d8 import legacy_d7_adapter as adapter

SCHEMA_VERSION = "d8_run_registry_v0"


@dataclass
class RunRecord:
    schema_version: str
    run_id: str
    job_id: str
    producer_git_commit: str
    evaluation_git_commit: Optional[str]
    model_family: str
    model_capacity: Optional[str]
    modeled_features: Tuple[str, ...]
    modeled_dimension: int
    pdg_policy: str
    pdg_value: Optional[int]
    preprocessing_name: str
    target_measure: str
    weighting_policy: bool
    weighting_reduction: Optional[str]
    split_identity: Dict[str, Any]
    content_dataset_hash: Optional[str]
    legacy_raw_file_sha256_prefix16: Optional[str]
    training_seed: int
    generation_seed: Optional[int]
    training_budget_epochs: Optional[int]
    parameter_count: Optional[int]
    wall_time: Optional[float]
    history: Optional[List[Dict[str, Any]]]
    best_validation_epoch: Optional[int]
    final_epoch: Optional[int]
    available_checkpoint_epoch: Optional[int]
    feature_space_nll: Optional[float]
    physical_space_nll: Optional[float]
    validation_nll: Optional[float]
    test_nll: Optional[float]
    checkpoint_path: Optional[str]
    checkpoint_hash: Optional[str]
    reconstruction_status: str
    reconstruction_scope: Optional[str]
    exclusion_reasons: List[str] = field(default_factory=list)
    # Bonus fields beyond the spec's required list -- needed by reconstruction.py
    # downstream and cheap to carry, not a second registry.
    architecture: Dict[str, Any] = field(default_factory=dict)
    checkpoint_kind: str = "none"

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "job_id": self.job_id,
            "producer_git_commit": self.producer_git_commit,
            "evaluation_git_commit": self.evaluation_git_commit,
            "model_family": self.model_family,
            "model_capacity": self.model_capacity,
            "modeled_features": list(self.modeled_features),
            "modeled_dimension": self.modeled_dimension,
            "pdg_policy": self.pdg_policy,
            "pdg_value": self.pdg_value,
            "preprocessing_name": self.preprocessing_name,
            "target_measure": self.target_measure,
            "weighting_policy": self.weighting_policy,
            "weighting_reduction": self.weighting_reduction,
            "split_identity": self.split_identity,
            "content_dataset_hash": self.content_dataset_hash,
            "legacy_raw_file_sha256_prefix16": self.legacy_raw_file_sha256_prefix16,
            "training_seed": self.training_seed,
            "generation_seed": self.generation_seed,
            "training_budget_epochs": self.training_budget_epochs,
            "parameter_count": self.parameter_count,
            "wall_time": self.wall_time,
            "history": self.history,
            "best_validation_epoch": self.best_validation_epoch,
            "final_epoch": self.final_epoch,
            "available_checkpoint_epoch": self.available_checkpoint_epoch,
            "feature_space_nll": self.feature_space_nll,
            "physical_space_nll": self.physical_space_nll,
            "validation_nll": self.validation_nll,
            "test_nll": self.test_nll,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_hash": self.checkpoint_hash,
            "reconstruction_status": self.reconstruction_status,
            "reconstruction_scope": self.reconstruction_scope,
            "exclusion_reasons": list(self.exclusion_reasons),
            "architecture": self.architecture,
            "checkpoint_kind": self.checkpoint_kind,
        }


def _best_validation_epoch(history: List[Dict[str, Any]]) -> Tuple[Optional[int], Optional[float]]:
    finite = [(h["epoch"], h["validation_loss"]) for h in history if h.get("validation_loss") is not None]
    if not finite:
        return None, None
    best_epoch, best_val = min(finite, key=lambda pair: pair[1])
    return int(best_epoch), float(best_val)


def _split_identity(shard_dir: Path) -> Dict[str, Any]:
    identity: Dict[str, Any] = {}
    for split, shard_name in adapter.SPLIT_SHARD_NAMES.items():
        sidecar = json.loads(Path(shard_dir, f"{shard_name}.json").read_text())
        identity[split] = {
            "shard_name": shard_name,
            "dataset_hash": sidecar.get("dataset_hash"),
            "shard_hash": sidecar.get("shard_hash"),
            "indices_hash": sidecar.get("indices_hash"),
            "construction_seed": sidecar.get("construction_seed"),
        }
    return identity


def _weighted_run_semantics(weighted: bool) -> Tuple[str, Optional[str]]:
    if weighted:
        return "production_weighted_nominal", "self_normalized_minibatch_ratio"
    return "row_empirical_unweighted", None


def _neural_run_record(
    *,
    job_name: str,
    run_key: str,
    metrics_blob: Dict[str, Any],
    variant_id: str,
    model_name: str,
    weighted: bool,
    shard_dir: Path,
    split_identity: Dict[str, Any],
    content_dataset_hash: Optional[str],
    legacy_prefix16: Optional[str],
    evaluation_git_commit: Optional[str],
    checkpoint_info: adapter.CheckpointInfo,
    recorded_checkpoint_hash: Optional[str],
) -> RunRecord:
    job_id = job_name.split("_", 1)[0]
    pdg_value, pdg_policy = adapter.JOB_PDG[job_name]
    family = adapter.MODEL_FAMILY_OF_NAME[model_name]
    capacity = {"affine_tiny": "tiny", "affine_small": "small", "affine_medium": "medium"}.get(model_name)

    history = metrics_blob.get("history")
    m = metrics_blob.get("metrics", {})
    target_measure, weighting_reduction = _weighted_run_semantics(weighted)

    best_epoch, best_val = (None, None)
    final_epoch = None
    if history:
        best_epoch, best_val = _best_validation_epoch(history)
        final_epoch = int(history[-1]["epoch"]) if history else None

    training_budget_epochs = 5 if family == "affine_coupling" else 1

    return RunRecord(
        schema_version=SCHEMA_VERSION,
        run_id=f"{job_name}__{run_key}",
        job_id=job_id,
        producer_git_commit=adapter.PRODUCER_GIT_COMMIT,
        evaluation_git_commit=evaluation_git_commit,
        model_family=family,
        model_capacity=capacity,
        modeled_features=adapter.MODELED_FEATURES_5D,
        modeled_dimension=5,
        pdg_policy=pdg_policy,
        pdg_value=pdg_value,
        preprocessing_name=variant_id,
        target_measure=target_measure,
        weighting_policy=weighted,
        weighting_reduction=weighting_reduction,
        split_identity=split_identity,
        content_dataset_hash=content_dataset_hash,
        legacy_raw_file_sha256_prefix16=legacy_prefix16,
        training_seed=adapter.TRAINING_SEED,
        generation_seed=m.get("generation_seed"),
        training_budget_epochs=training_budget_epochs,
        parameter_count=m.get("parameter_count"),
        wall_time=m.get("wall_time_seconds"),
        history=history,
        best_validation_epoch=best_epoch,
        final_epoch=final_epoch,
        available_checkpoint_epoch=final_epoch if checkpoint_info.checkpoint_path else None,
        feature_space_nll=m.get("test_feature_space_nll"),
        physical_space_nll=m.get("physical_space_nll"),
        validation_nll=best_val,
        test_nll=m.get("test_feature_space_nll"),
        checkpoint_path=checkpoint_info.checkpoint_path,
        checkpoint_hash=recorded_checkpoint_hash,
        reconstruction_status=checkpoint_info.reconstruction_status,
        reconstruction_scope=(
            adapter.RECONSTRUCTION_SCOPE_MODERN if checkpoint_info.reconstruction_status == "RECONSTRUCTIBLE" else None
        ),
        exclusion_reasons=list(checkpoint_info.exclusion_reasons),
        architecture=(
            {"family": family, **adapter.AFFINE_TRAIN_PARAMS.get(model_name, {})}
            if family == "affine_coupling"
            else {"family": family, **adapter.BASELINE_TRAIN_PARAMS.get(model_name, {})}
        ),
        checkpoint_kind=checkpoint_info.checkpoint_kind,
    )


def _job04_run_record(
    *,
    metrics_json: Dict[str, Any],
    split_identity: Dict[str, Any],
    content_dataset_hash: Optional[str],
    legacy_prefix16: Optional[str],
    evaluation_git_commit: Optional[str],
    checkpoint_info: adapter.CheckpointInfo,
    recorded_checkpoint_hash: Optional[str],
) -> RunRecord:
    history = metrics_json.get("history")
    m = metrics_json.get("metrics", {})
    best_epoch, best_val = _best_validation_epoch(history) if history else (None, None)
    final_epoch = int(history[-1]["epoch"]) if history else None

    return RunRecord(
        schema_version=SCHEMA_VERSION,
        run_id="04_legacy_available_code_realnvp_quantile__default",
        job_id="04",
        producer_git_commit=adapter.PRODUCER_GIT_COMMIT,
        evaluation_git_commit=evaluation_git_commit,
        model_family=adapter.JOB04_LEGACY_CONFIG["family"],
        model_capacity=None,
        modeled_features=adapter.MODELED_FEATURES_4D,
        modeled_dimension=4,
        pdg_policy="combined_unfiltered",
        pdg_value=None,
        preprocessing_name="quantile_transformer_legacy",
        target_measure="legacy_row_empirical",
        weighting_policy=False,
        weighting_reduction=None,
        split_identity=split_identity,
        content_dataset_hash=content_dataset_hash,
        legacy_raw_file_sha256_prefix16=legacy_prefix16,
        training_seed=adapter.TRAINING_SEED,
        generation_seed=m.get("generation_seed"),
        training_budget_epochs=5,
        parameter_count=m.get("parameter_count"),
        wall_time=m.get("wall_time_seconds"),
        history=history,
        best_validation_epoch=best_epoch,
        final_epoch=final_epoch,
        available_checkpoint_epoch=final_epoch if checkpoint_info.checkpoint_path else None,
        feature_space_nll=m.get("test_feature_space_nll"),
        physical_space_nll=m.get("physical_space_nll"),
        validation_nll=best_val,
        test_nll=m.get("test_feature_space_nll"),
        checkpoint_path=checkpoint_info.checkpoint_path,
        checkpoint_hash=recorded_checkpoint_hash,
        reconstruction_status=checkpoint_info.reconstruction_status,
        reconstruction_scope=(
            adapter.RECONSTRUCTION_SCOPE_JOB04 if checkpoint_info.reconstruction_status == "RECONSTRUCTIBLE" else None
        ),
        exclusion_reasons=list(checkpoint_info.exclusion_reasons)
        + ["job 04's fitted QuantileTransformer was never persisted; reconstruction refits it, which is "
           "sklearn-version-sensitive because the historical `subsample` parameter was left at whatever "
           "the installed scikit-learn's default was, not pinned to `None`"],
        architecture=dict(adapter.JOB04_LEGACY_CONFIG),
        checkpoint_kind=checkpoint_info.checkpoint_kind,
    )


def build_registry(input_artifact_dir: Path, shard_dir: Path) -> List[RunRecord]:
    """Build one `RunRecord` per historical run found under jobs 04-12."""

    input_artifact_dir = Path(input_artifact_dir)
    shard_dir = Path(shard_dir)
    jobs_dir = input_artifact_dir / "jobs"

    afterms_audit = json.loads((input_artifact_dir / "afterms_audit.json").read_text())
    nightly_summary_path = input_artifact_dir / "report" / "nightly_summary.json"
    nightly_summary = json.loads(nightly_summary_path.read_text()) if nightly_summary_path.exists() else {}
    shard_manifest = json.loads((shard_dir / "shard_manifest.json").read_text())

    content_dataset_hash = adapter.audit_content_dataset_hash(afterms_audit)
    legacy_prefix16 = (
        adapter.legacy_raw_file_sha256_prefix16(nightly_summary) if nightly_summary else None
    )
    split_identity = _split_identity(shard_dir)
    evaluation_git_commit = nightly_summary.get("git_commit")

    records: List[RunRecord] = []

    # Job 04 (single default run, legacy family, no run_label dict).
    job04_dir = jobs_dir / "04_legacy_available_code_realnvp_quantile"
    job04_metrics_path = job04_dir / "metrics.json"
    if job04_metrics_path.exists():
        job04_metrics = json.loads(job04_metrics_path.read_text())
        ckpt_info = adapter.resolve_checkpoint_info(
            "04_legacy_available_code_realnvp_quantile", "default"
        )
        recorded_hash = job04_metrics.get("metrics", {}).get("checkpoint_hash")
        records.append(
            _job04_run_record(
                metrics_json=job04_metrics,
                split_identity=split_identity,
                content_dataset_hash=content_dataset_hash,
                legacy_prefix16=legacy_prefix16,
                evaluation_git_commit=evaluation_git_commit,
                checkpoint_info=ckpt_info,
                recorded_checkpoint_hash=recorded_hash,
            )
        )

    # Jobs 05-11: run_label-keyed metrics.json dicts.
    for job_name in (
        "05_affine_preprocessing_ab_pdg13",
        "06_affine_preprocessing_ab_pdg_minus13",
        "07_affine_weight_ab_pdg13",
        "08_affine_weight_ab_pdg_minus13",
        "09_affine_capacity_smoke_pdg13",
        "10_gaussian_controls_pdg13",
        "11_gaussian_controls_pdg_minus13",
    ):
        job_dir = jobs_dir / job_name
        metrics_path = job_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        job_metrics = json.loads(metrics_path.read_text())
        variants = adapter.JOB_VARIANTS[job_name]
        model_names = adapter.JOB_MODEL_NAMES[job_name]
        weighted_job = adapter.JOB_WEIGHTED.get(job_name, False)
        weight_policies = [False, True] if weighted_job else [False]

        for variant_id in variants:
            for model_name in model_names:
                for weighted in weight_policies:
                    label = adapter.run_label(variant_id, model_name, weighted)
                    entry = job_metrics.get(label)
                    if entry is None:
                        continue
                    if entry.get("status") == "blocked":
                        continue
                    ckpt_info = adapter.resolve_checkpoint_info(job_name, label)
                    recorded_hash = entry.get("metrics", {}).get("checkpoint_hash")
                    records.append(
                        _neural_run_record(
                            job_name=job_name,
                            run_key=label,
                            metrics_blob=entry,
                            variant_id=variant_id,
                            model_name=model_name,
                            weighted=weighted,
                            shard_dir=shard_dir,
                            split_identity=split_identity,
                            content_dataset_hash=content_dataset_hash,
                            legacy_prefix16=legacy_prefix16,
                            evaluation_git_commit=evaluation_git_commit,
                            checkpoint_info=ckpt_info,
                            recorded_checkpoint_hash=recorded_hash,
                        )
                    )

    # Job 12: two repeat runs, read from metrics_run{1,2}.json (metrics.json
    # there is a merged comparison blob, not a run_label-keyed dict).
    job12_dir = jobs_dir / "12_memory_release_repeat_smoke"
    label = adapter.run_label("identity_standardized_v0", "affine_tiny", weighted=False)
    for repeat_index, fname in ((1, "metrics_run1.json"), (2, "metrics_run2.json")):
        path = job12_dir / fname
        if not path.exists():
            continue
        run_metrics = json.loads(path.read_text())
        entry = run_metrics.get(label)
        if entry is None:
            continue
        ckpt_info = adapter.job12_run_checkpoint_info(repeat_index)
        recorded_hash = entry.get("metrics", {}).get("checkpoint_hash")
        record = _neural_run_record(
            job_name="12_memory_release_repeat_smoke",
            run_key=f"{label}__repeat{repeat_index}",
            metrics_blob=entry,
            variant_id="identity_standardized_v0",
            model_name="affine_tiny",
            weighted=False,
            shard_dir=shard_dir,
            split_identity=split_identity,
            content_dataset_hash=content_dataset_hash,
            legacy_prefix16=legacy_prefix16,
            evaluation_git_commit=evaluation_git_commit,
            checkpoint_info=ckpt_info,
            recorded_checkpoint_hash=recorded_hash,
        )
        record.run_id = f"12_memory_release_repeat_smoke__{label}__repeat{repeat_index}"
        records.append(record)

    return records


def write_registry(records: List[RunRecord], output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = [r.to_json_dict() for r in records]
    (output_dir / "run_registry.json").write_text(json.dumps(payload, indent=2), encoding='utf-8')

    fieldnames = list(payload[0].keys()) if payload else []
    csv_buffer = io.StringIO()
    writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
    writer.writeheader()
    for row in payload:
        flat = dict(row)
        for key in ("modeled_features", "history", "split_identity", "exclusion_reasons", "architecture"):
            if key in flat:
                flat[key] = json.dumps(flat[key])
        writer.writerow(flat)
    (output_dir / "run_registry.csv").write_text(csv_buffer.getvalue(), encoding='utf-8')

    md_buffer = io.StringIO()
    md_buffer.write("# D8 Run Registry\n\n")
    md_buffer.write("| run_id | family | pdg | preprocessing | weighted | validation_nll | test_nll | physical_nll | reconstruction_status |\n")
    md_buffer.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
    for row in payload:
        md_buffer.write(
            "| {run_id} | {model_family} | {pdg_policy} | {preprocessing_name} | {weighting_policy} | "
            "{validation_nll} | {test_nll} | {physical_space_nll} | {reconstruction_status} |\n".format(**row)
        )
    (output_dir / "run_registry.md").write_text(md_buffer.getvalue(), encoding='utf-8')
