"""Fixture-only Phase 2 conditional-charge normalizing-flow pilot (v1).

v1 corrects four v0 contract defects (``docs/reviews/d9_conditional_charge_sampling_v1.md``):

1. the declared condition is the physical muon electric charge sign
   (``muon_electric_charge_sign``), not the PDG numeric sign -- PDG 13 names
   the negatively-charged mu- (condition -1), PDG -13 names the
   positively-charged mu+ (condition +1);
2. preprocessing is a single pooled train-only macro-balanced
   physical-weight standardization fit directly on the train partitions,
   never on a resampled draw pool;
3. training data is drawn fresh every epoch (:class:`_EpochDirectSampler`),
   never one fixed resample reused across all epochs;
4. the test payload is never materialized -- only its deterministic row
   count and split-index hash are computed (Gate E stays closed).
"""

from __future__ import annotations

import csv
import io
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from ..data_contracts import schema
from ..data_contracts.hashing import dataset_hash
from ..data_contracts.feature_views import FeatureView, IDENTITY_CARTESIAN_VIEW_ID
from .config import canonical_hash
from .empirical import (
    EmpiricalDatasetSpec,
    EmpiricalTrainValidationDataset,
    build_empirical_train_validation_dataset,
)
from .feature_pipeline import FittedFeaturePipeline
from .utility_tilt import (
    AliasSampler,
    ToyThresholds,
    compact_distribution_summary,
    compute_b_toy,
    empirical_draw_diagnostics,
    fit_toy_thresholds,
    nominal_prevalence,
    theoretical_concentration_diagnostics,
)

CONDITIONAL_CHARGE_SCHEMA_VERSION = "1"
CONDITION_FIELD_NAME = "muon_electric_charge_sign"

# Physical electric charge sign, keyed by PDG id. PDG codes are
# particle/antiparticle codes, not charge codes: PDG 13 names the mu-
# (physical charge -1) and PDG -13 names the mu+ (physical charge +1) -- the
# PDG numeric sign is the *opposite* of the physical charge sign for this
# pair. v0 mistakenly used the PDG numeric sign here; v1 uses the physical
# charge.
MUON_ELECTRIC_CHARGE_SIGN_BY_PDG: Dict[int, float] = {13: -1.0, -13: 1.0}

_ALLOWED_CONFIG_KEYS = {
    "schema_version",
    "experiment_id",
    "description",
    "fixture_only",
    "dataset_path",
    "pdg_ids",
    "seed",
    "max_rows_per_charge",
    "draws_per_epoch_per_charge",
    "validation_fraction",
    "test_fraction",
    "sample_count_per_charge",
    "feature_view",
    "model",
    "symmetry_audit",
}


class ConditionalChargeError(ValueError):
    """Invalid or unsafe conditional-charge pilot configuration."""


def _validate_config(config: Mapping[str, Any], repo_root: Path) -> Dict[str, Any]:
    unknown = sorted(set(config) - _ALLOWED_CONFIG_KEYS)
    if unknown:
        raise ConditionalChargeError(
            "undocumented config fields are refused: {}".format(unknown)
        )
    if config.get("schema_version") != CONDITIONAL_CHARGE_SCHEMA_VERSION:
        raise ConditionalChargeError(
            "schema_version must be {!r}".format(CONDITIONAL_CHARGE_SCHEMA_VERSION)
        )
    if config.get("fixture_only") is not True:
        raise ConditionalChargeError("fixture_only must be true")
    dataset_path = (repo_root / str(config["dataset_path"])).resolve()
    fixture_root = (repo_root / "data" / "samples").resolve()
    if fixture_root not in dataset_path.parents or "_sample." not in dataset_path.name:
        raise ConditionalChargeError(
            "Phase 2 pilot accepts only a repository data/samples/*_sample.* fixture"
        )
    if tuple(int(value) for value in config.get("pdg_ids", ())) != (13, -13):
        raise ConditionalChargeError("pdg_ids must be exactly [13, -13]")
    if config.get("feature_view", {}).get("view_id") != IDENTITY_CARTESIAN_VIEW_ID:
        raise ConditionalChargeError(
            "only the no-transform identity_cartesian_v0 baseline is documented"
        )
    symmetry = config.get("symmetry_audit", {})
    if symmetry.get("transform", "none") != "none":
        raise ConditionalChargeError(
            "symmetry_audit.transform must be 'none'; undocumented transforms are refused"
        )
    model = config.get("model", {})
    if model.get("family") != "affine_coupling":
        raise ConditionalChargeError("model.family must be 'affine_coupling'")
    if model.get("conditional") is not True:
        raise ConditionalChargeError("model.conditional must be true")
    if model.get("condition_name") != CONDITION_FIELD_NAME:
        raise ConditionalChargeError(
            "model.condition_name must be {!r}".format(CONDITION_FIELD_NAME)
        )
    if list(model.get("condition_values", ())) != [-1, 1]:
        raise ConditionalChargeError("model.condition_values must be [-1, 1]")
    params = dict(model.get("params", {}))
    if params.get("condition_dim", 1) != 1:
        raise ConditionalChargeError("conditional charge requires model condition_dim=1")
    for name in ("max_rows_per_charge", "draws_per_epoch_per_charge", "sample_count_per_charge"):
        if not isinstance(config.get(name), int) or config[name] < 2:
            raise ConditionalChargeError("{} must be an integer >= 2".format(name))
    resolved = dict(config)
    resolved["dataset_path"] = str(dataset_path)
    resolved["model"] = dict(model)
    resolved["model"]["params"] = params
    resolved["model"]["params"]["condition_dim"] = 1
    resolved["symmetry_audit"] = {
        "transform": "none",
        "sample_count": int(
            symmetry.get("sample_count", config["sample_count_per_charge"])
        ),
    }
    return resolved


def _condition(pdg_id: int, n: int) -> np.ndarray:
    try:
        value = MUON_ELECTRIC_CHARGE_SIGN_BY_PDG[int(pdg_id)]
    except (KeyError, TypeError, ValueError) as exc:
        raise ConditionalChargeError("pdg_id must be 13 or -13") from exc
    return np.full((int(n), 1), value, dtype=np.float64)


def muon_electric_charge_sign(pdg_id: int) -> float:
    """Return the physical muon electric charge sign: PDG 13 (mu-) -> -1, PDG -13 (mu+) -> +1."""

    return float(_condition(pdg_id, 1)[0, 0])


def _probabilities_from_weights(raw: np.ndarray) -> np.ndarray:
    weights = np.asarray(raw[:, schema.COLUMN_INDEX["w"]], dtype=np.float64)
    if not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise ConditionalChargeError("weights must be finite and nonnegative")
    total = float(weights.sum())
    if total <= 0.0:
        raise ConditionalChargeError("weight total must be positive")
    return weights / total


def _derived_epoch_seed(global_seed: int, epoch: int, pdg_id: int, salt: int) -> int:
    """Deterministic seed from (global_seed, epoch, pdg_id[, salt])."""

    return int(
        np.random.SeedSequence(
            [int(global_seed), int(epoch), abs(int(pdg_id)), int(pdg_id < 0), int(salt)]
        ).generate_state(1)[0]
    )


class _EpochDirectSampler:
    """Per-epoch macro-balanced physical-nominal direct sampling (arm D).

    Built once per training run from the two charges' TRAIN partitions only.
    Each call to :meth:`draw` (one per training epoch) draws
    ``I_{e,t}|c ~ pi_i|c`` independently per charge from a deterministic seed
    derived from ``(global_seed, epoch, pdg_id)`` -- never one fixed pool
    reused across epochs. No sample weights are ever attached to the drawn
    rows (``sample_weight_applied_to_loss`` is always ``False``): the
    per-charge physical-weight nominal measure enters only through the
    sampling probabilities, and the loss on the drawn rows is the ordinary
    unweighted NLL.

    This makes each step's gradient an unbiased estimator of the
    macro-balanced two-charge population gradient::

        E_epoch_draws[gradient] = (1/2) grad L_13 + (1/2) grad L_-13

    It does *not* imply the final trained parameters are themselves an
    unbiased estimator of any macro target -- that is a claim about a single
    stochastic-gradient step, never about the fixed point of nonconvex SGD.
    """

    def __init__(
        self,
        *,
        datasets: Mapping[int, EmpiricalTrainValidationDataset],
        pipeline: FittedFeaturePipeline,
        draws_per_epoch_per_charge: int,
        seed: int,
    ) -> None:
        if int(draws_per_epoch_per_charge) < 1:
            raise ConditionalChargeError("draws_per_epoch_per_charge must be positive")
        self._pipeline = pipeline
        self.draws_per_epoch_per_charge = int(draws_per_epoch_per_charge)
        self._seed = int(seed)
        self._raw: Dict[int, np.ndarray] = {}
        self._pi: Dict[int, np.ndarray] = {}
        self._samplers: Dict[int, AliasSampler] = {}
        self.source_probability_table_hash: Dict[int, str] = {}
        self.thresholds: Dict[int, ToyThresholds] = {}
        self._b_toy_train: Dict[int, np.ndarray] = {}
        self.nominal_b_toy_prevalence: Dict[int, float] = {}
        self._cumulative_unique: Dict[int, set] = {13: set(), -13: set()}
        for pdg_id in (13, -13):
            raw = datasets[pdg_id].train.raw
            weights = raw[:, schema.COLUMN_INDEX["w"]]
            pi = _probabilities_from_weights(raw)
            source_hash = dataset_hash(raw)
            self._raw[pdg_id] = raw
            self._pi[pdg_id] = pi
            self.source_probability_table_hash[pdg_id] = source_hash
            self._samplers[pdg_id] = AliasSampler.from_probabilities(pi, source_hash=source_hash)
            thresholds = fit_toy_thresholds(
                datasets[pdg_id].train.physical, weights, pdg_id=int(pdg_id)
            )
            self.thresholds[pdg_id] = thresholds
            b_toy_train = compute_b_toy(datasets[pdg_id].train.physical, thresholds)
            self._b_toy_train[pdg_id] = b_toy_train
            self.nominal_b_toy_prevalence[pdg_id] = nominal_prevalence(b_toy_train, weights)
        self.epoch_records: List[Dict[str, Any]] = []

    def draw(self, epoch: int) -> Dict[str, Any]:
        epoch = int(epoch)
        per_charge_raw: Dict[int, np.ndarray] = {}
        per_charge_meta: Dict[str, Any] = {}
        for pdg_id in (13, -13):
            derived_seed = _derived_epoch_seed(self._seed, epoch, pdg_id, salt=0)
            indices = self._samplers[pdg_id].draw(
                self.draws_per_epoch_per_charge, seed=derived_seed
            )
            per_charge_raw[pdg_id] = self._raw[pdg_id][indices]
            self._cumulative_unique[pdg_id].update(int(i) for i in np.unique(indices))
            diagnostics = empirical_draw_diagnostics(indices, b_toy=self._b_toy_train[pdg_id])
            per_charge_meta[str(pdg_id)] = {
                "pdg_id": int(pdg_id),
                "condition": muon_electric_charge_sign(pdg_id),
                "derived_seed": derived_seed,
                "draw_count": self.draws_per_epoch_per_charge,
                "draw_hash": canonical_hash(indices.tolist()),
                "source_probability_table_hash": self.source_probability_table_hash[pdg_id],
                "alias_table_hash": self._samplers[pdg_id].table_hash(),
                "unique_rows_drawn": diagnostics["unique_rows_drawn"],
                "unique_rows_fraction": diagnostics["unique_rows_fraction"],
                "max_reuse_count": diagnostics["max_reuse_count"],
                "empirical_b_toy_fraction": diagnostics["empirical_b_toy_fraction"],
                "cumulative_unique_source_rows": len(self._cumulative_unique[pdg_id]),
            }
        raw_combined = np.concatenate((per_charge_raw[13], per_charge_raw[-13]))
        condition = np.concatenate(
            (
                _condition(13, self.draws_per_epoch_per_charge),
                _condition(-13, self.draws_per_epoch_per_charge),
            )
        )
        shuffle_seed = _derived_epoch_seed(self._seed, epoch, 0, salt=1)
        permutation = np.random.default_rng(shuffle_seed).permutation(raw_combined.shape[0])
        raw_combined = raw_combined[permutation]
        condition = condition[permutation]
        x = self._pipeline.transform_raw(raw_combined)

        record = {
            "epoch": epoch,
            "charge_counts": {
                "13": self.draws_per_epoch_per_charge,
                "-13": self.draws_per_epoch_per_charge,
            },
            "sample_weight_applied_to_loss": False,
            "per_charge": per_charge_meta,
        }
        self.epoch_records.append(record)
        return {"x": x, "condition": condition, "metadata": record}

    def manifest(self) -> Dict[str, Any]:
        return {
            "schema_version": CONDITIONAL_CHARGE_SCHEMA_VERSION,
            "sampling_regime": "conditional_macro_balanced_physical_nominal_direct_per_epoch",
            "draws_per_epoch_per_charge": self.draws_per_epoch_per_charge,
            "draws_per_epoch_total": 2 * self.draws_per_epoch_per_charge,
            "global_seed": self._seed,
            "sample_weight_applied_to_loss": False,
            "gradient_estimator_target": (
                "E_epoch_draws[gradient] = (1/2) * grad L_13 + (1/2) * grad L_-13"
            ),
            "unbiasedness_scope": (
                "per_step_stochastic_gradient_only; the final trained parameters "
                "are not claimed to be an unbiased estimator of any macro target"
            ),
            "charges": {
                str(pdg_id): {
                    "pdg_id": int(pdg_id),
                    "condition": muon_electric_charge_sign(pdg_id),
                    "source_probability_table_hash": self.source_probability_table_hash[pdg_id],
                    "alias_table_hash": self._samplers[pdg_id].table_hash(),
                    "theoretical": theoretical_concentration_diagnostics(
                        self._pi[pdg_id], draw_budget=self.draws_per_epoch_per_charge
                    ),
                    "nominal_b_toy_prevalence": self.nominal_b_toy_prevalence[pdg_id],
                    "thresholds": {
                        "t_pT": float(self.thresholds[pdg_id].t_pT),
                        "t_R": float(self.thresholds[pdg_id].t_R),
                    },
                    "cumulative_unique_source_rows": len(self._cumulative_unique[pdg_id]),
                    "cumulative_unique_source_rows_fraction": (
                        float(len(self._cumulative_unique[pdg_id])) / self._raw[pdg_id].shape[0]
                    ),
                }
                for pdg_id in (13, -13)
            },
            "epochs": list(self.epoch_records),
            "n_epochs_recorded": len(self.epoch_records),
        }


def _weighted_nll(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    return float(np.sum(weights * -values) / np.sum(weights))


def _split_hash(dataset: EmpiricalTrainValidationDataset) -> str:
    return canonical_hash({
        "train_source_row_indices": dataset.train.source_row_indices.tolist(),
        "validation_source_row_indices": dataset.validation.source_row_indices.tolist(),
        "test_split_hash": dataset.test_split_hash,
    })


def _current_git_commit() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _assert_finite_diagnostics(value: Any, path: str = "diagnostics") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _assert_finite_diagnostics(child, "{}.{}".format(path, key))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_finite_diagnostics(child, "{}[{}]".format(path, index))
    elif isinstance(value, (float, np.floating)) and not np.isfinite(value):
        raise ConditionalChargeError("{} is non-finite".format(path))


def run_symmetry_audit(
    *,
    empirical: Mapping[int, Mapping[str, np.ndarray]],
    generated: Optional[Mapping[int, np.ndarray]] = None,
    transform: str = "none",
    sample_count: int = 1024,
) -> Dict[str, Any]:
    """Compare the two charge distributions without imposing a symmetry.

    v1 intentionally accepts only the no-transform baseline. A future declared
    ``g`` must enter here with repository-backed coordinate and geometry
    evidence before any transformed comparison is run.
    """

    if transform != "none":
        raise ConditionalChargeError(
            "undocumented hard symmetry transformation refused: {!r}".format(transform)
        )
    left = empirical[13]
    right = empirical[-13]
    left_summary = compact_distribution_summary(left["physical"], weights=left["weights"])
    right_summary = compact_distribution_summary(right["physical"], weights=right["weights"])
    left_mean = np.array(
        [left_summary["per_feature"][name]["mean"] for name in ("px", "py", "pz", "x", "y")]
    )
    right_mean = np.array(
        [right_summary["per_feature"][name]["mean"] for name in ("px", "py", "pz", "x", "y")]
    )
    result: Dict[str, Any] = {
        "status": "completed_no_transform_baseline",
        "evidence_status": "OPEN",
        "evidence": "No repository evidence defines an exact charge-reflection map after the Muon Shield.",
        "transform": "none",
        "hard_symmetry_enforced": False,
        "comparison": "P(X|C=13) versus P(X|C=-13)",
        "empirical_weighted_mean_difference_13_minus_m13": (left_mean - right_mean).tolist(),
        "empirical_weighted_summary_13": left_summary,
        "empirical_weighted_summary_m13": right_summary,
        "sample_count": int(sample_count),
    }
    if generated is not None:
        generated_left = compact_distribution_summary(generated[13])
        generated_right = compact_distribution_summary(generated[-13])
        result["generated_summary_13"] = generated_left
        result["generated_summary_m13"] = generated_right
        result["generated_mean_difference_13_minus_m13"] = (
            np.asarray([generated_left["per_feature"][name]["mean"] for name in ("px", "py", "pz", "x", "y")])
            - np.asarray([generated_right["per_feature"][name]["mean"] for name in ("px", "py", "pz", "x", "y")])
        ).tolist()
    return result


def run_fixture_pilot(
    config: Mapping[str, Any],
    *,
    output_dir: Path,
    repo_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run the bounded macro-balanced, physical-nominal, per-epoch-direct-sampling pilot.

    Test rows are never loaded for either charge (``build_empirical_train_validation_dataset``);
    only the deterministic test row count and split-index hash are recorded.
    """

    repo_root = (
        Path(repo_root).resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[3]
    )
    resolved = _validate_config(config, repo_root)
    seed = int(resolved["seed"])
    datasets: Dict[int, EmpiricalTrainValidationDataset] = {}
    for pdg_id in (13, -13):
        datasets[pdg_id] = build_empirical_train_validation_dataset(
            EmpiricalDatasetSpec(
                dataset_path=resolved["dataset_path"],
                pdg_id=pdg_id,
                seed=seed,
                val_fraction=float(resolved.get("validation_fraction", 0.2)),
                test_fraction=float(resolved.get("test_fraction", 0.2)),
                max_rows=int(resolved["max_rows_per_charge"]),
            )
        )

    feature_view = FeatureView(IDENTITY_CARTESIAN_VIEW_ID)
    pipeline = FittedFeaturePipeline.fit_macro_weighted(
        raw_by_charge={pdg_id: datasets[pdg_id].train.raw for pdg_id in (13, -13)},
        weights_by_charge={
            pdg_id: datasets[pdg_id].train.raw[:, schema.COLUMN_INDEX["w"]]
            for pdg_id in (13, -13)
        },
        feature_view=feature_view,
    )

    validation_raw = np.concatenate(
        (datasets[13].validation.raw, datasets[-13].validation.raw)
    )
    validation_condition = np.concatenate(
        (
            _condition(13, datasets[13].validation.n_rows),
            _condition(-13, datasets[-13].validation.n_rows),
        )
    )
    validation_weights = np.concatenate(
        tuple(
            datasets[pdg_id].validation.raw[:, schema.COLUMN_INDEX["w"]]
            / datasets[pdg_id].validation.raw[:, schema.COLUMN_INDEX["w"]].sum()
            for pdg_id in (13, -13)
        )
    )

    from Nflow.torch_models.affine_coupling import AffineCouplingFlow

    model = AffineCouplingFlow(
        dimension=pipeline.dimension,
        device=resolved["model"].get("device", "cpu"),
        **resolved["model"]["params"],
    )

    epoch_sampler = _EpochDirectSampler(
        datasets=datasets,
        pipeline=pipeline,
        draws_per_epoch_per_charge=int(resolved["draws_per_epoch_per_charge"]),
        seed=seed,
    )

    fit = model.fit(
        None,
        x_validation=pipeline.transform_raw(validation_raw),
        seed=seed,
        validation_sample_weight=validation_weights,
        validation_condition=validation_condition,
        epoch_sampler=epoch_sampler.draw,
    )
    if fit.status != "ok":
        raise ConditionalChargeError("conditional flow fit failed: {}".format(fit.warnings))

    per_charge = []
    per_charge_generated = []
    generated = {}
    for pdg_id in (13, -13):
        partition = datasets[pdg_id].validation
        normalized = pipeline.transform_raw(partition.raw)
        normalized_lp = model.log_prob(
            normalized, condition=_condition(pdg_id, partition.n_rows)
        )
        physical_lp = pipeline.normalized_to_physical_log_prob(
            normalized_lp, partition.raw
        )
        weights = partition.raw[:, schema.COLUMN_INDEX["w"]]
        samples_normalized = model.sample(
            int(resolved["sample_count_per_charge"]),
            seed=seed + 1000 + (0 if pdg_id == 13 else 1),
            condition=_condition(pdg_id, int(resolved["sample_count_per_charge"])),
        )
        generated[pdg_id] = samples_normalized
        samples_physical = pipeline.inverse_to_physical(samples_normalized)
        thresholds = epoch_sampler.thresholds[pdg_id]
        empirical_summary = compact_distribution_summary(
            partition.physical, weights=weights,
        )
        generated_summary = compact_distribution_summary(samples_physical)
        generated_b_toy = compute_b_toy(samples_physical, thresholds)
        nominal_b_toy_occupancy = epoch_sampler.nominal_b_toy_prevalence[pdg_id]
        generated_b_toy_occupancy = float(generated_b_toy.mean())
        validation_entry = {
            "pdg_id": int(pdg_id),
            "condition": muon_electric_charge_sign(pdg_id),
            "validation_rows": partition.n_rows,
            "validation_weight_total": float(weights.sum()),
            "physical_validation_nll_weighted": _weighted_nll(physical_lp, weights),
            "physical_validation_log_prob_finite_fraction": float(np.isfinite(physical_lp).mean()),
            "empirical_weighted_summary": empirical_summary,
        }
        generated_entry = {
            "pdg_id": int(pdg_id),
            "condition": muon_electric_charge_sign(pdg_id),
            "sample_count": int(samples_physical.shape[0]),
            "generated_sample_finite_fraction": float(np.isfinite(samples_physical).all(axis=1).mean()),
            "generated_log_prob_finite_fraction": float(np.isfinite(model.log_prob(
                samples_normalized, condition=_condition(pdg_id, samples_normalized.shape[0])
            )).mean()),
            "nominal_b_toy_occupancy": nominal_b_toy_occupancy,
            "generated_b_toy_occupancy": generated_b_toy_occupancy,
            "generated_over_nominal_b_toy_ratio": (
                (generated_b_toy_occupancy / nominal_b_toy_occupancy)
                if nominal_b_toy_occupancy > 0.0
                else None
            ),
            "generated_summary": generated_summary,
        }
        per_charge.append(validation_entry)
        per_charge_generated.append(generated_entry)

    audit_n = int(resolved["symmetry_audit"]["sample_count"])
    generated_physical = {
        pdg_id: pipeline.inverse_to_physical(generated[pdg_id])
        for pdg_id in (13, -13)
    }
    symmetry_audit = run_symmetry_audit(
        empirical={
            pdg_id: {
                "physical": datasets[pdg_id].train.physical,
                "weights": datasets[pdg_id].train.raw[:, schema.COLUMN_INDEX["w"]],
            }
            for pdg_id in (13, -13)
        },
        generated=generated_physical,
        sample_count=audit_n,
    )
    audit_minus = model.sample(
        audit_n, seed=seed + 2000, condition=_condition(13, audit_n)
    )
    audit_plus = model.sample(
        audit_n, seed=seed + 2000, condition=_condition(-13, audit_n)
    )
    symmetry_audit["same_latent_diagnostic"] = {
        "sample_count": audit_n,
        "normalized_paired_mean_absolute_difference": float(
            np.mean(np.abs(audit_minus - audit_plus))
        ),
    }
    sampling_manifest = epoch_sampler.manifest()
    training_metrics = {
        "fit": fit.to_dict(),
        "training_history": fit.train_history,
        "loss_contract": {
            "sample_weight_applied_to_loss": False,
            "ordinary_unweighted_nll_after_direct_sampling": True,
        },
    }
    per_charge_macro = [row["physical_validation_nll_weighted"] for row in per_charge]
    test_provenance = {
        str(pdg_id): {
            "test_row_count": datasets[pdg_id].test_row_count,
            "test_split_hash": datasets[pdg_id].test_split_hash,
            "test_payload_loaded": False,
            "test_used_for_training": False,
            "test_used_for_preprocessing": False,
            "test_used_for_model_selection": False,
            "test_used_for_evaluation": False,
        }
        for pdg_id in (13, -13)
    }
    summary = {
        "schema_version": CONDITIONAL_CHARGE_SCHEMA_VERSION,
        "experiment_id": resolved["experiment_id"],
        "config_hash": canonical_hash(config),
        "status": "completed_fixture_pilot",
        "scope": "fixture_only_not_physics_acceptance",
        "training_distribution": "macro_balanced_physical_nominal_direct_per_epoch",
        "draws_per_epoch_per_charge": int(resolved["draws_per_epoch_per_charge"]),
        "condition_field_name": CONDITION_FIELD_NAME,
        "condition_definition": {
            "pdg_13": MUON_ELECTRIC_CHARGE_SIGN_BY_PDG[13],
            "pdg_-13": MUON_ELECTRIC_CHARGE_SIGN_BY_PDG[-13],
        },
        "source_file_dataset_hash": datasets[13].source_file_dataset_hash,
        "test_provenance": test_provenance,
        "lineage": {
            "code_commit": _current_git_commit(),
            "source_file_dataset_hash": datasets[13].source_file_dataset_hash,
            "charges": {
                str(pdg_id): {
                    "source_file_dataset_hash": datasets[pdg_id].source_file_dataset_hash,
                    "split_hash": _split_hash(datasets[pdg_id]),
                    "train_rows": datasets[pdg_id].train.n_rows,
                    "validation_rows": datasets[pdg_id].validation.n_rows,
                    "test_row_count": datasets[pdg_id].test_row_count,
                    "test_split_hash": datasets[pdg_id].test_split_hash,
                    "test_payload_loaded": False,
                    "source_probability_table_hash": epoch_sampler.source_probability_table_hash[pdg_id],
                }
                for pdg_id in (13, -13)
            },
            "preprocessing_hash": pipeline.config_hash(),
            "preprocessing_weighting": pipeline.weighting,
            "model_config_hash": canonical_hash(model.config()),
            "checkpoint_hash": model.checkpoint_hash(),
        },
        "pipeline_manifest": pipeline.manifest(),
        "model_manifest": model.manifest(),
        "fit": training_metrics["fit"],
        "validation": {
            "per_charge": per_charge_macro,
            "macro_nll": float(np.mean(per_charge_macro)),
            "worst_charge_nll": float(np.max(per_charge_macro)),
        },
        "per_charge_validation": per_charge,
        "per_charge_generated_summary": per_charge_generated,
        "sampling_manifest": sampling_manifest,
        "symmetry_audit": symmetry_audit,
    }
    _assert_finite_diagnostics(summary)
    _write_artifacts(
        Path(output_dir),
        summary,
        sampling_manifest=sampling_manifest,
        training_metrics=training_metrics,
        per_charge_generated_summary=per_charge_generated,
        symmetry_audit=symmetry_audit,
    )
    return summary


def _write_artifacts(
    output_dir: Path,
    summary: Mapping[str, Any],
    *,
    sampling_manifest: Mapping[str, Any],
    training_metrics: Mapping[str, Any],
    per_charge_generated_summary: Any,
    symmetry_audit: Mapping[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    fieldnames = list(summary["per_charge_validation"][0])
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(summary["per_charge_validation"])
    (output_dir / "per_charge_validation.csv").write_text(
        stream.getvalue(), encoding="utf-8"
    )
    rows = summary["per_charge_validation"]
    summary_stream = io.StringIO(newline="")
    summary_writer = csv.DictWriter(
        summary_stream,
        fieldnames=[
            "pdg_id", "condition", "validation_nll_weighted",
            "nominal_b_toy_occupancy", "generated_b_toy_occupancy",
            "generated_over_nominal_b_toy_ratio",
        ],
        lineterminator="\n",
    )
    summary_writer.writeheader()
    for row, generated_row in zip(rows, per_charge_generated_summary):
        summary_writer.writerow({
            "pdg_id": row["pdg_id"],
            "condition": row["condition"],
            "validation_nll_weighted": row["physical_validation_nll_weighted"],
            "nominal_b_toy_occupancy": generated_row["nominal_b_toy_occupancy"],
            "generated_b_toy_occupancy": generated_row["generated_b_toy_occupancy"],
            "generated_over_nominal_b_toy_ratio": generated_row["generated_over_nominal_b_toy_ratio"],
        })
    (output_dir / "conditional_fixture_summary.csv").write_text(
        summary_stream.getvalue(), encoding="utf-8"
    )
    files = {
        "conditional_sampling_manifest.json": sampling_manifest,
        "conditional_training_metrics.json": training_metrics,
        "per_charge_validation.json": summary["per_charge_validation"],
        "per_charge_generated_summary.json": per_charge_generated_summary,
        "symmetry_audit.json": symmetry_audit,
        "conditional_fixture_summary.json": summary,
    }
    for name, payload in files.items():
        (output_dir / name).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    lines = [
        "# Conditional-charge NF fixture pilot (v1)",
        "",
        "- Status: `{}`".format(summary["status"]),
        "- Scope: `{}`".format(summary["scope"]),
        "- Training: `{}`".format(summary["training_distribution"]),
        "- Condition: `{}` (PDG 13 -> -1, PDG -13 -> +1)".format(summary["condition_field_name"]),
        "- Validation: per-charge weighted NLL; macro mean and worst charge are reported",
        "- Symmetry transform: `none` (audit only; no hard symmetry)",
        "- Test rows: never materialized (Gate E closed); only row count and split hash are recorded",
        "- Physical-rate estimate: not produced",
        "",
        "| PDG id | condition | weighted validation NLL | nominal B_toy | generated B_toy | ratio |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row, generated_row in zip(rows, per_charge_generated_summary):
        ratio = generated_row["generated_over_nominal_b_toy_ratio"]
        lines.append(
            "| {pdg_id} | {condition:.0f} | {physical_validation_nll_weighted:.12g} | "
            "{nominal_b_toy_occupancy:.12g} | {generated_b_toy_occupancy:.12g} | {ratio} |".format(
                ratio=("{:.6g}".format(ratio) if ratio is not None else "n/a"),
                **dict(generated_row, **row)
            )
        )
    markdown = "\n".join(lines) + "\n"
    (output_dir / "report.md").write_text(markdown, encoding="utf-8")
    (output_dir / "conditional_fixture_summary.md").write_text(
        markdown, encoding="utf-8"
    )
