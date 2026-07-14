"""Campaign runner: execute the run matrix with resume and failure isolation.

Each run executes independently. A failed run records its status and traceback
and the campaign continues. Completed identical run hashes are skipped unless
``force`` is set. Runs never silently overwrite an incompatible artifact:
resume is keyed on the canonical config hash embedded in ``run_id``.
"""

from __future__ import annotations

import dataclasses
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..benchmarks import embed_physical_to_raw
from ..data_contracts.feature_views import FeatureView
from .artifacts import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    ArtifactStore,
    derive_run_id,
)
from .datasets import build_controlled_dataset
from .environment import capture_environment, utc_timestamp
from .evaluator import evaluate_run
from .feature_pipeline import FittedFeaturePipeline

DIMENSION = 5


def _make_feature_view(spec) -> FeatureView:
    if spec.pz_unit_gev is None:
        return FeatureView(spec.view_id)
    return FeatureView(spec.view_id, pz_unit_gev=spec.pz_unit_gev)


def _dataset_key(run_spec) -> tuple:
    return (
        run_spec.target.target_id,
        run_spec.target.variant,
        run_spec.pdg_id,
        run_spec.seed,
        run_spec.dataset.n_train,
        run_spec.dataset.n_validation,
        run_spec.dataset.n_test,
    )


def run_single(
    run_spec,
    store: ArtifactStore,
    *,
    force: bool = False,
    dataset_cache: Optional[Dict[tuple, Any]] = None,
    device: str = "cpu",
    tracker=None,
) -> Dict[str, Any]:
    """Execute one run; return a status record. Never raises for run failures."""

    from Nflow.registry import create_density_estimator

    run_id = derive_run_id(run_spec)
    if not force and store.is_complete(run_spec):
        return {"run_id": run_id, "status": "skipped_completed"}

    started_at = utc_timestamp()
    try:
        # -- dataset (cached so matched A/B1/B2 runs share identical rows) --
        key = _dataset_key(run_spec)
        if dataset_cache is not None and key in dataset_cache:
            dataset = dataset_cache[key]
        else:
            dataset = build_controlled_dataset(
                target_id=run_spec.target.target_id,
                variant=run_spec.target.variant,
                pdg_id=run_spec.pdg_id,
                n_train=run_spec.dataset.n_train,
                n_validation=run_spec.dataset.n_validation,
                n_test=run_spec.dataset.n_test,
                seed=run_spec.seed,
            )
            if dataset_cache is not None:
                dataset_cache[key] = dataset

        from ..benchmarks import make_controlled_target

        target = make_controlled_target(
            run_spec.target.target_id, variant=run_spec.target.variant
        )

        view = _make_feature_view(run_spec.feature_view)
        raw_train = embed_physical_to_raw(
            dataset.train.physical, pdg_id=run_spec.pdg_id, plane_z=0.0
        )
        raw_val = embed_physical_to_raw(
            dataset.validation.physical, pdg_id=run_spec.pdg_id, plane_z=0.0
        )
        pipeline = FittedFeaturePipeline.fit(raw_train, view)
        normalized_train = pipeline.transform_raw(raw_train)
        normalized_val = pipeline.transform_raw(raw_val)

        model = create_density_estimator(
            run_spec.model, dimension=DIMENSION, device=device
        )
        fit_result = model.fit(
            normalized_train, x_validation=normalized_val, seed=run_spec.seed
        )

        environment = capture_environment(requested_device=device)
        if fit_result.status != "ok":
            store.write_run(
                run_spec,
                environment=environment,
                dataset_manifest=dataset.manifest(),
                feature_pipeline_manifest=pipeline.manifest(),
                model_manifest=model.manifest(),
                fit_result=fit_result.to_dict(),
                metrics={"status": "fit_failed"},
                training_history=fit_result.train_history,
                samples={},
                status=STATUS_FAILED,
                run_id=run_id,
                error="fit returned status={}".format(fit_result.status),
            )
            return {"run_id": run_id, "status": STATUS_FAILED, "reason": "fit_failed"}

        metrics, physical_q = evaluate_run(
            target=target,
            pdg_id=run_spec.pdg_id,
            test_physical=dataset.test_nominal.physical,
            test_rare_mask=dataset.test_nominal.rare_region_mask,
            pipeline=pipeline,
            model=model,
            evaluation=run_spec.evaluation,
            seed=run_spec.seed,
        )
        metrics["fit_wall_time_seconds"] = fit_result.wall_time_seconds
        metrics["ended_at"] = utc_timestamp()

        paths = store.run_paths(run_spec)
        paths.run_dir.mkdir(parents=True, exist_ok=True)
        save_manifest = model.save(paths.run_dir)

        hashes = {
            "run_config_hash": run_spec.config_hash(),
            "target_config_hash": target.config_hash(),
            "feature_pipeline_hash": pipeline.config_hash(),
            "model_config_hash": run_spec.model.config_hash(),
            "train_dataset_hash": dataset.train.raw_dataset_hash,
            "test_dataset_hash": dataset.test_nominal.raw_dataset_hash,
            "checkpoint_hash": save_manifest.get("checkpoint_hash"),
        }
        store.write_run(
            run_spec,
            environment=environment,
            dataset_manifest=dataset.manifest(),
            feature_pipeline_manifest=pipeline.manifest(),
            model_manifest=model.manifest(),
            fit_result=fit_result.to_dict(),
            metrics=metrics,
            training_history=fit_result.train_history,
            samples={"model_samples_physical": physical_q},
            status=STATUS_COMPLETED,
            run_id=run_id,
            save_manifest=save_manifest,
            hashes=hashes,
        )
        if tracker is not None:
            try:
                tracker.log_run(
                    run_id=run_id,
                    params=run_spec.to_dict(),
                    metrics=metrics,
                    run_dir=paths.run_dir,
                )
            except Exception:  # tracking is best-effort, never fatal
                pass
        return {
            "run_id": run_id,
            "status": STATUS_COMPLETED,
            "started_at": started_at,
        }
    except Exception as exc:  # isolate: record and continue the campaign
        tb = traceback.format_exc()
        try:
            paths = store.run_paths(run_spec)
            paths.run_dir.mkdir(parents=True, exist_ok=True)
            store.write_run(
                run_spec,
                environment=capture_environment(requested_device=device),
                dataset_manifest={"status": "unavailable"},
                feature_pipeline_manifest={"status": "unavailable"},
                model_manifest={"status": "unavailable"},
                fit_result={"status": "failed"},
                metrics={"status": "error"},
                training_history=[],
                samples={},
                status=STATUS_FAILED,
                run_id=run_id,
                error=tb,
            )
        except Exception:  # pragma: no cover - best-effort failure record
            pass
        return {"run_id": run_id, "status": STATUS_FAILED, "error": str(exc)}


def _selected(value, selection) -> bool:
    return selection is None or value in selection


def run_campaign(
    config,
    *,
    root: Optional[Path] = None,
    force: bool = False,
    target_ids=None,
    model_names=None,
    seeds=None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the full campaign matrix; return a summary of run statuses."""

    config.validate()
    store = ArtifactStore(config.experiment_id, root=root)
    device = device or config.resources.device
    tracker = None
    if config.tracking.mode != "local":
        try:
            from .tracking import make_tracker

            tracker = make_tracker(config.tracking)
        except Exception:  # optional adapter missing/misconfigured -> local only
            tracker = None
    dataset_cache: Dict[tuple, Any] = {}
    records: List[Dict[str, Any]] = []
    for run_spec in config.runs():
        if not _selected(run_spec.target.target_id, target_ids):
            continue
        if not _selected(run_spec.model.name, model_names):
            continue
        if not _selected(run_spec.seed, seeds):
            continue
        # Materialize the effective device into the run identity so run_id /
        # config_hash / experiment_config.json reflect the device actually used;
        # a CPU-forced run of an "auto" config cannot then be skipped or
        # overwritten by a later differing-device run.
        run_spec = dataclasses.replace(run_spec, device=device)
        record = run_single(
            run_spec,
            store,
            force=force,
            dataset_cache=dataset_cache,
            device=run_spec.device,
            tracker=tracker,
        )
        records.append(record)
    summary = {
        "experiment_id": config.experiment_id,
        "experiment_config_hash": config.config_hash(),
        "n_runs": len(records),
        "n_completed": sum(1 for r in records if r["status"] == STATUS_COMPLETED),
        "n_failed": sum(1 for r in records if r["status"] == STATUS_FAILED),
        "n_skipped": sum(1 for r in records if r["status"] == "skipped_completed"),
        "runs": records,
    }
    store.experiment_dir.mkdir(parents=True, exist_ok=True)
    (store.experiment_dir / "campaign_summary.json").write_text(
        __import__("json").dumps(summary, indent=2, default=str)
    )
    return summary
