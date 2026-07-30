"""D7 -- empirical after-MS real-data dataset, evaluation, and campaign entry.

D7 ("empirical after-MS data", ``docs/contracts/density_problem_contract_v0.md``
section 7) has **no closed-form target density**. Every metric here that would
require an exact ``p(x)`` -- forward KL, importance-weight ESS, component-
posterior mass, rare-mode diagnostics -- is never computed; this module
computes exactly the subset of ``density_lab.metrics`` that only needs
held-out real rows and model samples, reusing those functions unchanged.

Single code path, repository sample vs. full local dataset: this module never
branches on scale. The repository afterMS sample
(``data/samples/muonsFullMC_afterMS_sample.npz``) and a full local
``muonsFullMC_afterMS.pkl`` go through the exact same
:func:`build_empirical_dataset` / :func:`run_empirical_single` code; only the
``dataset_path``, ``max_rows``, and artifact ``root`` differ, and those are
ordinary configuration, never hardcoded.

Deliberately reused, unmodified:

- ``ship_muon_bg.data_contracts`` for loading, schema validation, hashing,
  PDG filtering, and the deterministic three-way split;
- ``density_lab.feature_pipeline.FittedFeaturePipeline`` (already generic
  over any raw ``(N, 8)`` array; no synthetic-target assumption anywhere in
  it);
- ``Nflow.registry.create_density_estimator`` and the family's unmodified
  ``.fit()`` legacy path (``batch_plan=None``, ``sample_weight=None`` --
  D7 has no component labels, so no rare-aware estimator arm applies; this is
  exactly arm A, the unbiased IID baseline, applied to real data);
- ``density_lab.artifacts.ArtifactStore`` / ``config.RunSpec`` /
  ``config.canonical_hash`` for identity, resume/skip, and artifact writing
  (D7 runs live in the same ``artifacts/density_lab/<experiment_id>/<run_id>/``
  layout as D0-D5 runs, just with ``target.target_id == "D7_empirical"``);
  ``gates.evaluate_scientific_gates`` (D5-only gates never fire for D7).

Physical weight policy: the real ``w`` column is preserved end-to-end in
every partition and provenance record, but is **not** applied to the training
loss by default (``sample_weight=None``). Row-empirical D7 training never
silently claims physical weighting; see ``physical_weight_status`` in the
dataset manifest and assumption A6 of
``docs/contracts/rare_aware_minibatch_estimators_v0.md``.
"""

from __future__ import annotations

import dataclasses
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from ..data_contracts import (
    dataset_hash as compute_dataset_hash,
    filter_by_pdg,
    load_muon_array,
    pdg_counts,
    schema,
    validate_muon_array,
)
from ..data_contracts.feature_views import PHYSICAL_STATE_COLUMNS
from ..data_contracts.splitting import make_three_way_split
from ..data_contracts.subsampling import representative_subset
from .artifacts import STATUS_COMPLETED, STATUS_FAILED, ArtifactStore, derive_run_id
from .config import (
    DatasetSpec,
    EvaluationSpec,
    FeatureViewSpec,
    ModelSpec,
    ResourceSpec,
    RunSpec,
    SamplingSpec,
    TargetSpec,
    canonical_hash,
)
from .environment import capture_environment, utc_timestamp
from .feature_pipeline import FittedFeaturePipeline
from .gates import SCIENTIFIC_STATUSES, STATUS_UNAVAILABLE, ScientificGateSpec, evaluate_scientific_gates
from . import metrics as M

EMPIRICAL_TARGET_ID = "D7_empirical"
EMPIRICAL_DATASET_SCHEMA_VERSION = "0"
DIMENSION = 5

# Explicit, documented ordering: load -> validate -> filter by PDG -> cap to
# max_rows (if requested) -> three-way split. Row limits apply *after* PDG
# filtering (a row budget is a per-track budget, not a whole-file budget that
# could be exhausted by the other track before this one is ever selected),
# and the split is built on the already-filtered, already-capped pool.
ROW_LIMIT_ORDER = "after_pdg_filter_before_split"


class EmpiricalDataError(ValueError):
    """A D7 dataset configuration or on-disk file is unusable."""


@dataclasses.dataclass(frozen=True)
class EmpiricalDatasetSpec:
    """Configuration for one PDG track's D7 dataset build.

    ``dataset_path`` is the only thing that differs between the repository
    fixture and a full local dataset; every other field is bounded
    operational configuration (row budget, split fractions, seed).
    """

    dataset_path: str
    pdg_id: int
    seed: int
    val_fraction: float = 0.2
    test_fraction: float = 0.2
    max_rows: Optional[int] = None
    allow_zero_weight: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_path": self.dataset_path,
            "pdg_id": int(self.pdg_id),
            "seed": int(self.seed),
            "val_fraction": float(self.val_fraction),
            "test_fraction": float(self.test_fraction),
            "max_rows": self.max_rows,
            "allow_zero_weight": bool(self.allow_zero_weight),
        }


@dataclasses.dataclass(frozen=True)
class EmpiricalDatasetPartition:
    """One split's real, raw (N, 8) rows, with full index provenance."""

    partition: str
    raw: np.ndarray
    source_row_indices: np.ndarray  # row indices into the ORIGINAL loaded file
    n_rows: int

    @property
    def physical(self) -> np.ndarray:
        """The five modelled physical columns [px, py, pz, x, y]."""

        return np.ascontiguousarray(self.raw[:, : len(PHYSICAL_STATE_COLUMNS)])

    def manifest(self) -> Dict[str, Any]:
        return {
            "partition": self.partition,
            "n_rows": self.n_rows,
            "raw_dataset_hash": compute_dataset_hash(self.raw),
            "source_row_index_min": int(self.source_row_indices.min()) if self.n_rows else None,
            "source_row_index_max": int(self.source_row_indices.max()) if self.n_rows else None,
        }


@dataclasses.dataclass(frozen=True)
class EmpiricalDataset:
    """The three D7 partitions for one PDG track, plus full provenance."""

    dataset_spec: EmpiricalDatasetSpec
    source_file_dataset_hash: str
    source_file_n_rows: int
    pdg_counts_in_source: Dict[int, int]
    filtered_n_rows: int
    rows_used_after_cap: int
    train: EmpiricalDatasetPartition
    validation: EmpiricalDatasetPartition
    test: EmpiricalDatasetPartition

    def manifest(self, *, redact_path: bool = False) -> Dict[str, Any]:
        hashes = [
            self.train.manifest()["raw_dataset_hash"],
            self.validation.manifest()["raw_dataset_hash"],
            self.test.manifest()["raw_dataset_hash"],
        ]
        return {
            "schema_version": EMPIRICAL_DATASET_SCHEMA_VERSION,
            "target_id": EMPIRICAL_TARGET_ID,
            "dataset_path": "REDACTED" if redact_path else self.dataset_spec.dataset_path,
            "pdg_id": int(self.dataset_spec.pdg_id),
            "seed": int(self.dataset_spec.seed),
            "source_file_dataset_hash": self.source_file_dataset_hash,
            "source_file_n_rows": self.source_file_n_rows,
            "pdg_counts_in_source": {str(k): v for k, v in self.pdg_counts_in_source.items()},
            "filtered_n_rows": self.filtered_n_rows,
            "row_limit_order": ROW_LIMIT_ORDER,
            "requested_max_rows": self.dataset_spec.max_rows,
            "rows_used_after_cap": self.rows_used_after_cap,
            "val_fraction": self.dataset_spec.val_fraction,
            "test_fraction": self.dataset_spec.test_fraction,
            "weights_affect_selection": False,
            "validation_no_leakage": len(set(hashes)) == 3,
            "physical_weight_status": "preserved_in_raw_rows_not_applied_to_training_loss",
            "physical_weight_semantics": "not_implemented",
            "sampling_correction_weight_status": "not_applicable_no_rare_aware_arm_for_unlabeled_real_data",
            "partitions": {
                "train": self.train.manifest(),
                "validation": self.validation.manifest(),
                "test": self.test.manifest(),
            },
        }

    def config_hash(self) -> str:
        return canonical_hash(self.manifest())


def build_empirical_dataset(spec: EmpiricalDatasetSpec) -> EmpiricalDataset:
    """Load, validate, PDG-filter, (optionally) cap, and three-way split.

    Identical code path for the repository afterMS sample and a full local
    dataset -- only ``spec.dataset_path`` differs. Raises
    :class:`EmpiricalDataError`-wrapped typed contract errors on any schema
    violation (a technical failure, never a scientific negative).
    """

    if spec.pdg_id not in schema.EXPECTED_MUON_IDS:
        raise EmpiricalDataError(
            "pdg_id must be one of {}, got {!r}".format(schema.EXPECTED_MUON_IDS, spec.pdg_id)
        )
    try:
        array = load_muon_array(spec.dataset_path)
        validate_muon_array(array, allow_zero_weight=spec.allow_zero_weight)
    except Exception as exc:  # re-raise typed, with dataset path context only
        raise EmpiricalDataError(
            "dataset at {!r} failed load/validate: {}".format(spec.dataset_path, exc)
        ) from exc

    source_hash = compute_dataset_hash(array)
    source_counts = pdg_counts(array)
    filtered, source_indices = filter_by_pdg(array, spec.pdg_id)
    if filtered.shape[0] == 0:
        raise EmpiricalDataError(
            "no rows for pdg_id={} in dataset at {!r}".format(spec.pdg_id, spec.dataset_path)
        )

    if spec.max_rows is not None:
        if spec.max_rows < 1:
            raise EmpiricalDataError("max_rows must be >= 1, got {}".format(spec.max_rows))
        capped, cap_selected = representative_subset(filtered, int(spec.max_rows), seed=spec.seed)
        capped_source_indices = source_indices[cap_selected]
    else:
        capped, capped_source_indices = filtered, source_indices

    n_rows = int(capped.shape[0])
    min_rows_needed = 3  # >=1 row per split at minimum; practical floor checked below
    if n_rows < min_rows_needed:
        raise EmpiricalDataError(
            "only {} row(s) available for pdg_id={} after filtering/capping; "
            "need at least {}".format(n_rows, spec.pdg_id, min_rows_needed)
        )

    split = make_three_way_split(
        n_rows, seed=spec.seed, val_fraction=spec.val_fraction,
        test_fraction=spec.test_fraction, dataset_hash=compute_dataset_hash(capped),
    )
    for name in ("n_train", "n_val", "n_test"):
        if split[name] < 2:
            raise EmpiricalDataError(
                "split produced {}={} rows (< 2) for pdg_id={}; increase max_rows or "
                "adjust val_fraction/test_fraction".format(name, split[name], spec.pdg_id)
            )

    partitions = {}
    for key, split_key in (("train", "train_indices"), ("validation", "val_indices"), ("test", "test_indices")):
        idx = np.asarray(split[split_key], dtype=int)
        partitions[key] = EmpiricalDatasetPartition(
            partition=key,
            raw=np.ascontiguousarray(capped[idx], dtype=np.float64),
            source_row_indices=capped_source_indices[idx],
            n_rows=int(idx.size),
        )

    return EmpiricalDataset(
        dataset_spec=spec,
        source_file_dataset_hash=source_hash,
        source_file_n_rows=int(array.shape[0]),
        pdg_counts_in_source=source_counts,
        filtered_n_rows=int(filtered.shape[0]),
        rows_used_after_cap=n_rows,
        train=partitions["train"],
        validation=partitions["validation"],
        test=partitions["test"],
    )


# --- evaluation (no exact target density; see module docstring) -------------


def _derived_seed(base_seed: int, salt: int) -> int:
    seq = np.random.SeedSequence([int(base_seed), int(salt)])
    return int(seq.generate_state(1)[0])


def evaluate_empirical_run(
    *,
    dataset: EmpiricalDataset,
    pipeline: FittedFeaturePipeline,
    model,
    evaluation: EvaluationSpec,
    seed: int,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """D7 metric bundle: only what never requires an exact target density.

    Never computes ``forward_kl``, ``importance_ess``,
    ``component_posterior_mass``, or rare-mode diagnostics -- there is no
    ``p(x)`` for D7 to compute them against (see the module docstring).
    Those keys are recorded explicitly as ``None`` (not merely absent) so a
    reader never mistakes "not applicable" for "forgot to compute".
    """

    results: Dict[str, Any] = {"physical_space": True, "exact_target_density_available": False}

    test_physical = dataset.test.physical
    n_samples = int(max(evaluation.ess_sample_count, evaluation.c2st_sample_count))
    sample_seed = _derived_seed(seed, 1)
    import time

    t0 = time.perf_counter()
    normalized_samples = model.sample(n_samples, seed=sample_seed)
    sample_seconds = time.perf_counter() - t0
    physical_q = pipeline.inverse_to_physical(normalized_samples)
    finite_rows = np.isfinite(physical_q).all(axis=1)
    physical_q_finite = physical_q[finite_rows]

    # log_prob throughput capacity metric (density_problem_contract_v0.md
    # section 8); no target density exists to convert this into an ESS.
    t1 = time.perf_counter()
    _ = model.log_prob(normalized_samples)
    logprob_seconds = time.perf_counter() - t1

    # The pipeline needs raw (N, 8) rows for the feature view + pz Jacobian;
    # dataset.test.raw is that raw form (dataset.test.physical is only the 5
    # modelled columns, used for the sample-vs-held-out-rows metrics below).
    normalized_test = pipeline.transform_raw(dataset.test.raw)
    normalized_lp_test = np.asarray(model.log_prob(normalized_test), dtype=np.float64)
    q_log_on_test = pipeline.normalized_to_physical_log_prob(normalized_lp_test, dataset.test.raw)

    results["held_out"] = M.held_out_nll(q_log_on_test)
    results["physical_space_held_out_nll"] = results["held_out"]["held_out_nll"]
    results["forward_kl"] = None
    results["importance_ess"] = None
    results["component_posterior_mass_on_test"] = None
    results["rare_mode"] = None

    n_c2st = min(evaluation.c2st_sample_count, test_physical.shape[0], physical_q_finite.shape[0])
    if n_c2st >= 10:
        results["c2st"] = M.c2st(
            test_physical[:n_c2st], physical_q_finite[:n_c2st], seed=_derived_seed(seed, 2)
        )

    results["tail_quantile_errors"] = M.tail_quantile_errors(
        test_physical, physical_q_finite,
        quantiles=evaluation.tail_quantiles, column_names=PHYSICAL_STATE_COLUMNS,
    )
    results["exceedance_probability_errors"] = M.exceedance_probability_errors(
        test_physical, physical_q_finite, thresholds=evaluation.exceedance_pz_thresholds,
    )
    results["non_finite_density"] = M.non_finite_density_rate(q_log_on_test)
    results["support_violation"] = M.support_violation_rate(physical_q)
    results["duplicates"] = M.duplicate_diagnostics(physical_q_finite, atol=evaluation.near_duplicate_atol)

    results["throughput"] = {
        "sample_rows": int(n_samples),
        "sample_seconds": float(sample_seconds),
        "sample_rows_per_second": float(n_samples / sample_seconds) if sample_seconds > 0 else None,
        "log_prob_rows": int(n_samples),
        "log_prob_seconds": float(logprob_seconds),
        "log_prob_rows_per_second": float(n_samples / logprob_seconds) if logprob_seconds > 0 else None,
    }
    results["parameter_count"] = int(model.parameter_count())

    feature_lp = normalized_lp_test
    results["feature_space_held_out_nll"] = (
        float(-np.mean(feature_lp[np.isfinite(feature_lp)])) if np.isfinite(feature_lp).any() else float("nan")
    )
    results["debug_feature_space_nll"] = results["feature_space_held_out_nll"]

    return results, physical_q


# --- campaign orchestration (single PDG track / both tracks) ----------------


def _make_empirical_run_spec(
    *,
    experiment_id: str,
    dataset: EmpiricalDataset,
    feature_view: FeatureViewSpec,
    model: ModelSpec,
    evaluation: EvaluationSpec,
    scientific_gates: ScientificGateSpec,
    device: str,
) -> RunSpec:
    target = TargetSpec(
        target_id=EMPIRICAL_TARGET_ID,
        variant=dataset.source_file_dataset_hash[:16],
        stage="empirical",
    )
    ds = DatasetSpec(
        n_train=dataset.train.n_rows,
        n_validation=dataset.validation.n_rows,
        n_test=dataset.test.n_rows,
    )
    return RunSpec(
        experiment_id=experiment_id,
        target=target,
        pdg_id=int(dataset.dataset_spec.pdg_id),
        feature_view=feature_view,
        model=model,
        seed=int(dataset.dataset_spec.seed),
        dataset=ds,
        evaluation=evaluation,
        device=device,
        scientific_gates=scientific_gates,
        sampling=SamplingSpec(),  # default iid_target; no rare-aware arm applies to D7
    )


def run_empirical_single(
    dataset_spec: EmpiricalDatasetSpec,
    store: ArtifactStore,
    *,
    experiment_id: str,
    feature_view: FeatureViewSpec,
    model: ModelSpec,
    evaluation: Optional[EvaluationSpec] = None,
    scientific_gates: Optional[ScientificGateSpec] = None,
    device: str = "cpu",
    force: bool = False,
) -> Dict[str, Any]:
    """Build one PDG track's D7 dataset and run one bounded training+eval.

    Mirrors ``density_lab.campaign.run_single``'s contract exactly (same
    ``ArtifactStore``, same resume/skip-on-identical-config-hash behavior,
    same technical/scientific status separation): never raises for a run
    failure, always returns a status record, and a technical failure is
    never recorded as a scientific negative.
    """

    from Nflow.registry import create_density_estimator

    evaluation = evaluation or EvaluationSpec()
    scientific_gates = scientific_gates or ScientificGateSpec()
    model.validate()

    try:
        dataset = build_empirical_dataset(dataset_spec)
    except EmpiricalDataError as exc:
        return {
            "run_id": None, "status": STATUS_FAILED, "reason": "dataset_build_failed",
            "error": str(exc), "technical_status": STATUS_FAILED, "scientific_status": None,
        }

    run_spec = _make_empirical_run_spec(
        experiment_id=experiment_id, dataset=dataset, feature_view=feature_view,
        model=model, evaluation=evaluation, scientific_gates=scientific_gates, device=device,
    )
    run_spec.dataset.validate()
    run_spec.evaluation.validate()
    run_spec.scientific_gates.validate(run_spec.evaluation)

    run_id = derive_run_id(run_spec)
    if not force and store.is_complete(run_spec):
        status = store.read_run_status(run_spec) or {}
        scientific_status = status.get("scientific_status")
        if not isinstance(scientific_status, str) or scientific_status not in SCIENTIFIC_STATUSES:
            scientific_status = STATUS_UNAVAILABLE
        return {
            "run_id": run_id, "status": "skipped_completed",
            "technical_status": status.get("technical_status", STATUS_COMPLETED),
            "scientific_status": scientific_status,
        }

    started_at = utc_timestamp()
    try:
        view = _build_feature_view(feature_view)
        pipeline = FittedFeaturePipeline.fit(dataset.train.raw, view)
        normalized_train = pipeline.transform_raw(dataset.train.raw)
        normalized_val = pipeline.transform_raw(dataset.validation.raw)

        estimator = create_density_estimator(model, dimension=DIMENSION, device=device)
        fit_result = estimator.fit(
            normalized_train, x_validation=normalized_val, seed=run_spec.seed,
        )
        environment = capture_environment(requested_device=device)

        if fit_result.status != "ok":
            store.write_run(
                run_spec, environment=environment,
                dataset_manifest=dataset.manifest(),
                feature_pipeline_manifest=pipeline.manifest(),
                model_manifest=estimator.manifest(),
                fit_result=fit_result.to_dict(),
                metrics={"status": "fit_failed"},
                training_history=fit_result.train_history,
                samples={}, status=STATUS_FAILED, run_id=run_id,
                error="fit returned status={}".format(fit_result.status),
            )
            return {"run_id": run_id, "status": STATUS_FAILED, "reason": "fit_failed"}

        metrics, physical_q = evaluate_empirical_run(
            dataset=dataset, pipeline=pipeline, model=estimator,
            evaluation=run_spec.evaluation, seed=run_spec.seed,
        )
        metrics["fit_wall_time_seconds"] = fit_result.wall_time_seconds
        metrics["training_final"] = fit_result.train_history[-1] if fit_result.train_history else {}
        metrics["estimator_family"] = "unweighted_iid_target"
        metrics["unbiasedness_status"] = "not_applicable"
        metrics["scientific_scope"] = "empirical_no_closed_form_target_density"
        metrics["fit_claim"] = "empirical_fit_no_exact_target_density_to_compare_against"
        metrics["sampling_regime"] = "iid_target"
        metrics["diagnostic_only"] = False
        metrics["ended_at"] = utc_timestamp()

        gate_result = evaluate_scientific_gates(
            metrics, target_id=EMPIRICAL_TARGET_ID, gate_spec=run_spec.resolved_gate_spec(),
        )
        metrics["scientific_gates"] = gate_result.to_dict()

        paths = store.run_paths(run_spec)
        paths.run_dir.mkdir(parents=True, exist_ok=True)
        save_manifest = estimator.save(paths.run_dir)

        hashes = {
            "run_config_hash": run_spec.config_hash(),
            "dataset_manifest_hash": dataset.config_hash(),
            "feature_pipeline_hash": pipeline.config_hash(),
            "model_config_hash": run_spec.model.config_hash(),
            "train_dataset_hash": dataset.train.manifest()["raw_dataset_hash"],
            "test_dataset_hash": dataset.test.manifest()["raw_dataset_hash"],
            "checkpoint_hash": save_manifest.get("checkpoint_hash"),
            "source_file_dataset_hash": dataset.source_file_dataset_hash,
        }
        store.write_run(
            run_spec, environment=environment,
            dataset_manifest=dataset.manifest(), feature_pipeline_manifest=pipeline.manifest(),
            model_manifest=estimator.manifest(), fit_result=fit_result.to_dict(),
            metrics=metrics, training_history=fit_result.train_history,
            samples={"model_samples_physical": physical_q},
            status=STATUS_COMPLETED, run_id=run_id, save_manifest=save_manifest,
            hashes=hashes, scientific_status=gate_result.scientific_status,
            decision_scope=gate_result.decision_scope,
            scientific_failure_reasons=gate_result.scientific_failure_reasons,
        )
        return {
            "run_id": run_id, "status": STATUS_COMPLETED, "technical_status": STATUS_COMPLETED,
            "scientific_status": gate_result.scientific_status,
            "scientific_failure_reasons": gate_result.scientific_failure_reasons,
            "started_at": started_at,
        }
    except Exception as exc:  # isolate: record and continue, matching campaign.run_single
        tb = traceback.format_exc()
        try:
            paths = store.run_paths(run_spec)
            paths.run_dir.mkdir(parents=True, exist_ok=True)
            store.write_run(
                run_spec, environment=capture_environment(requested_device=device),
                dataset_manifest={"status": "unavailable"}, feature_pipeline_manifest={"status": "unavailable"},
                model_manifest={"status": "unavailable"}, fit_result={"status": "failed"},
                metrics={"status": "error"}, training_history=[], samples={},
                status=STATUS_FAILED, run_id=run_id, error=tb,
            )
        except Exception:  # pragma: no cover - best-effort failure record
            pass
        return {"run_id": run_id, "status": STATUS_FAILED, "error": str(exc)}


def _build_feature_view(spec: FeatureViewSpec):
    """Construct a ``FeatureView`` from a ``config.FeatureViewSpec``.

    Mirrors ``density_lab.campaign._make_feature_view`` exactly; kept as a
    small local copy rather than importing a module-private helper.
    """

    from ..data_contracts.feature_views import FeatureView

    if spec.pz_unit_gev is None:
        return FeatureView(spec.view_id)
    return FeatureView(spec.view_id, pz_unit_gev=spec.pz_unit_gev)


@dataclasses.dataclass(frozen=True)
class EmpiricalCampaignSpec:
    """JSON-serializable D7 campaign configuration (the "config layer" seam).

    ``dataset_path`` is the *only* external-data configuration seam: no other
    field, no source code, and no other tracked file needs to change between
    running this against the repository fixture and a full local dataset.
    Environment-variable references in ``dataset_path`` (e.g.
    ``"${SHIP_MUON_BG_LOCAL_DATA}/muonsFullMC_afterMS.pkl"``) are expanded at
    load time by :meth:`from_dict` / :meth:`from_json_file` (via
    ``os.path.expandvars`` + ``os.path.expanduser``), so a committed config
    file never needs to contain a machine-specific absolute path.
    """

    experiment_id: str
    dataset_path: str
    pdg_ids: Tuple[int, ...]
    feature_view: FeatureViewSpec
    model: ModelSpec
    seed: int
    val_fraction: float = 0.2
    test_fraction: float = 0.2
    max_rows: Optional[int] = None
    allow_zero_weight: bool = True
    evaluation: EvaluationSpec = dataclasses.field(default_factory=EvaluationSpec)
    scientific_gates: ScientificGateSpec = dataclasses.field(default_factory=ScientificGateSpec)
    device: str = "cpu"
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": EMPIRICAL_DATASET_SCHEMA_VERSION,
            "experiment_id": self.experiment_id,
            "description": self.description,
            "dataset_path": self.dataset_path,
            "pdg_ids": list(self.pdg_ids),
            "feature_view": {"view_id": self.feature_view.view_id, "pz_unit_gev": self.feature_view.pz_unit_gev},
            "model": self.model.to_dict(),
            "seed": int(self.seed),
            "val_fraction": float(self.val_fraction),
            "test_fraction": float(self.test_fraction),
            "max_rows": self.max_rows,
            "allow_zero_weight": bool(self.allow_zero_weight),
            "evaluation": self.evaluation.to_dict(),
            "scientific_gates": self.scientific_gates.to_dict(),
            "device": self.device,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EmpiricalCampaignSpec":
        import os

        raw_path = payload["dataset_path"]
        expanded_path = os.path.expanduser(os.path.expandvars(raw_path))
        return cls(
            experiment_id=payload["experiment_id"],
            description=payload.get("description", ""),
            dataset_path=expanded_path,
            pdg_ids=tuple(int(p) for p in payload["pdg_ids"]),
            feature_view=FeatureViewSpec(
                payload["feature_view"]["view_id"], payload["feature_view"].get("pz_unit_gev")
            ),
            model=ModelSpec(
                name=payload["model"]["name"], family=payload["model"]["family"],
                params=dict(payload["model"].get("params", {})),
                training_budget_id=payload["model"].get("training_budget_id", "default"),
            ),
            seed=int(payload["seed"]),
            val_fraction=float(payload.get("val_fraction", 0.2)),
            test_fraction=float(payload.get("test_fraction", 0.2)),
            max_rows=(None if payload.get("max_rows") is None else int(payload["max_rows"])),
            allow_zero_weight=bool(payload.get("allow_zero_weight", True)),
            evaluation=EvaluationSpec(**{
                k: (tuple(v) if isinstance(v, list) else v)
                for k, v in payload.get("evaluation", {}).items()
            }),
            scientific_gates=ScientificGateSpec(**payload.get("scientific_gates", {})),
            device=payload.get("device", "cpu"),
        )

    @classmethod
    def from_json_file(cls, path) -> "EmpiricalCampaignSpec":
        import json as _json

        with open(path, encoding="utf-8") as handle:
            return cls.from_dict(_json.load(handle))

    def config_hash(self) -> str:
        return canonical_hash(self.to_dict())


def run_empirical_campaign_from_spec(
    spec: EmpiricalCampaignSpec, *, root: Optional[Path] = None, force: bool = False
) -> Dict[str, Any]:
    """Run :func:`run_empirical_campaign` from an :class:`EmpiricalCampaignSpec`."""

    return run_empirical_campaign(
        experiment_id=spec.experiment_id, dataset_path=spec.dataset_path,
        pdg_ids=list(spec.pdg_ids), feature_view=spec.feature_view, model=spec.model,
        seed=spec.seed, val_fraction=spec.val_fraction, test_fraction=spec.test_fraction,
        max_rows=spec.max_rows, allow_zero_weight=spec.allow_zero_weight,
        evaluation=spec.evaluation, scientific_gates=spec.scientific_gates,
        device=spec.device, root=root, force=force,
    )


def run_empirical_campaign(
    *,
    experiment_id: str,
    dataset_path: str,
    pdg_ids: List[int],
    feature_view: FeatureViewSpec,
    model: ModelSpec,
    seed: int,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
    max_rows: Optional[int] = None,
    allow_zero_weight: bool = True,
    evaluation: Optional[EvaluationSpec] = None,
    scientific_gates: Optional[ScientificGateSpec] = None,
    device: str = "cpu",
    root: Optional[Path] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """Run one PDG track's D7 build+train+eval for every requested PDG id.

    Each PDG track is fully independent (models are trained separately per
    PDG id, matching every other estimator in this repository); a failure on
    one track never aborts the other.
    """

    store = ArtifactStore(experiment_id, root=root)
    records: List[Dict[str, Any]] = []
    for pdg_id in pdg_ids:
        dataset_spec = EmpiricalDatasetSpec(
            dataset_path=dataset_path, pdg_id=int(pdg_id), seed=int(seed),
            val_fraction=val_fraction, test_fraction=test_fraction,
            max_rows=max_rows, allow_zero_weight=allow_zero_weight,
        )
        record = run_empirical_single(
            dataset_spec, store, experiment_id=experiment_id, feature_view=feature_view,
            model=model, evaluation=evaluation, scientific_gates=scientific_gates,
            device=device, force=force,
        )
        records.append(record)
    summary = {
        "experiment_id": experiment_id,
        "target_id": EMPIRICAL_TARGET_ID,
        "n_runs": len(records),
        "n_completed": sum(1 for r in records if r["status"] == STATUS_COMPLETED),
        "n_failed": sum(1 for r in records if r["status"] == STATUS_FAILED),
        "n_skipped": sum(1 for r in records if r["status"] == "skipped_completed"),
        "runs": records,
    }
    store.experiment_dir.mkdir(parents=True, exist_ok=True)
    import json

    (store.experiment_dir / "campaign_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    return summary
