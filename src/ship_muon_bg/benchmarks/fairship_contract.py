"""Versioned contract for the utility-guided FairShip benchmark.

This module only shapes and validates benchmark records.  It does not run
FairShip, estimate an endpoint, or turn a technical censoring event into a
physics result.  The three arms deliberately share one report shape while
keeping their generation measures distinct.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence


class BenchmarkArm(str, Enum):
    P0 = "P0"
    PU_DIRECT = "PU_DIRECT"
    Q_THETA = "Q_THETA"


SCHEMA_VERSION = "utility_guided_fairship_benchmark_v0"
ARM_ORDER = tuple(arm.value for arm in BenchmarkArm)
_SAMPLING_METHOD = {
    "P0": "nominal_p0_sampling",
    "PU_DIRECT": "direct_pu_sampling",
    "Q_THETA": "learned_q_theta_generation",
}
_LINEAGE_REF_FIELDS = (
    "source_state_ref",
    "execution_ref",
    "interaction_realization_refs",
    "observation_refs",
    "decision_ref",
)


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


@dataclass(frozen=True)
class BenchmarkConfig:
    """The immutable, shared provenance and cohort policy for all arms."""

    benchmark_id: str
    fairship_configuration_id: str
    geometry_tag: str
    source_state_definition_id: str
    downstream_endpoint_definition_id: str | None = None
    cohort_manifest_id: str = "predeclared_cohorts_v0"
    no_redraw_until_success: bool = True
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "benchmark_id",
            "fairship_configuration_id",
            "geometry_tag",
            "source_state_definition_id",
            "cohort_manifest_id",
        ):
            _nonempty(getattr(self, name), name)
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version!r}")
        if not isinstance(self.no_redraw_until_success, bool):
            raise TypeError("no_redraw_until_success must be a bool")
        if not self.no_redraw_until_success:
            raise ValueError("benchmark requires predeclared cohorts; redraw-until-success is forbidden")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "BenchmarkConfig":
        if not isinstance(value, Mapping):
            raise TypeError("benchmark config must be a mapping")
        allowed = {"schema_version", "benchmark_id", "fairship_configuration_id", "geometry_tag",
                   "source_state_definition_id", "downstream_endpoint_definition_id", "cohort_manifest_id",
                   "no_redraw_until_success"}
        if set(value) - allowed:
            raise ValueError("benchmark config has unknown fields")
        return cls(
            benchmark_id=value["benchmark_id"],
            fairship_configuration_id=value["fairship_configuration_id"],
            geometry_tag=value["geometry_tag"],
            source_state_definition_id=value["source_state_definition_id"],
            downstream_endpoint_definition_id=value.get("downstream_endpoint_definition_id"),
            cohort_manifest_id=value.get("cohort_manifest_id", "predeclared_cohorts_v0"),
            no_redraw_until_success=value.get("no_redraw_until_success", True),
            schema_version=value.get("schema_version", SCHEMA_VERSION),
        )

    @classmethod
    def from_json(cls, path: Path | str) -> "BenchmarkConfig":
        return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "benchmark_id": self.benchmark_id,
            "fairship_configuration_id": self.fairship_configuration_id,
            "geometry_tag": self.geometry_tag,
            "source_state_definition_id": self.source_state_definition_id,
            "downstream_endpoint_definition_id": self.downstream_endpoint_definition_id,
            "cohort_manifest_id": self.cohort_manifest_id,
            "no_redraw_until_success": self.no_redraw_until_success,
        }


def load_config(path: Path | str) -> BenchmarkConfig:
    """Load the one versioned benchmark configuration used by all arms."""

    return BenchmarkConfig.from_json(path)


def _empty_arm(arm: str) -> dict[str, Any]:
    return {
        "arm_id": arm,
        "generation": {
            "measure": arm,
            "sampling_method": _SAMPLING_METHOD[arm],
            "predeclared_cohort": False,
            "redraw_until_success": False,
            "cohort_manifest_ref": None,
            "cohort_manifest_sha256": None,
            "candidate_provenance": None,
            "proposal_provenance": None,
            "fairship_configuration_id": None,
        },
        "utility_tilt": {
            "status": "not_run",
            "normalization": None,
            "support": None,
            "enrichment_vs_concentration": None,
            "ess": None,
            "ess_is_diagnostic_only": True,
            "empirical_utility_association": None,
        },
        "proposal_fidelity": {
            "status": "not_run",
            "fit_metric": None,
            "physical_space_diagnostics": None,
            "high_utility_region_occupancy": None,
            "two_sample_diagnostic": None,
            "declared_target_measure": "PU_DIRECT" if arm == "Q_THETA" else arm,
            "comparison_target": "PU_DIRECT" if arm == "Q_THETA" else arm,
        },
        "fairship_outcomes": {
            "status": "not_run",
            "candidate_count": None,
            "valid_execution_count": None,
            "technical_failure_count": None,
            "physics_rejection_count": None,
            "accepted_candidate_count": None,
            "technical_failures": [],
            "physics_outcomes": [],
        },
    }


def _lineage_refs(record: Mapping[str, Any], *, technical: bool) -> dict[str, Any]:
    refs = record.get("lineage_refs")
    if not isinstance(refs, Mapping):
        raise ValueError("each outcome requires lineage_refs for source/execution/interaction/observation/decision")
    unknown = set(refs) - set(_LINEAGE_REF_FIELDS)
    missing = set(_LINEAGE_REF_FIELDS) - set(refs)
    if unknown or missing:
        raise ValueError(f"lineage_refs must contain exactly {_LINEAGE_REF_FIELDS}")
    result = {"source_state_ref": _nonempty(refs["source_state_ref"], "source_state_ref"),
              "execution_ref": _nonempty(refs["execution_ref"], "execution_ref")}
    for field in ("interaction_realization_refs", "observation_refs"):
        values = refs[field]
        if not isinstance(values, (tuple, list)) or any(not isinstance(item, str) or not item for item in values):
            raise ValueError(f"{field} must be a sequence of non-empty references")
        result[field] = list(values)
    decision = refs["decision_ref"]
    if technical:
        if decision is not None:
            raise ValueError("technical_failure must not carry a decision_ref")
    else:
        result["decision_ref"] = _nonempty(decision, "decision_ref")
    result["decision_ref"] = decision
    return result


def summarize_fairship_outcomes(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize canonical execution/physics records without conflation.

    Each record must carry one execution status and, only for a successful
    execution, one physics outcome.  ``records`` is intentionally a flat
    summary input: source state, execution, interaction realization, and
    observation remain separate upstream entities.
    """

    technical: list[dict[str, Any]] = []
    physics: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        candidate_id = _nonempty(record.get("candidate_id"), "candidate_id")
        if candidate_id in seen:
            raise ValueError(f"duplicate candidate_id: {candidate_id!r}")
        seen.add(candidate_id)
        execution_status = record.get("execution_status")
        outcome = record.get("physics_outcome")
        if execution_status == "technical_failure":
            if outcome is not None:
                raise ValueError("technical_failure must not carry a physics_outcome")
            technical.append({"candidate_id": candidate_id, "reason": record.get("failure_reason", ""),
                              "lineage_refs": _lineage_refs(record, technical=True)})
        elif execution_status == "succeeded":
            if outcome not in {"physics_rejection", "accepted_candidate"}:
                raise ValueError("succeeded execution requires physics_rejection or accepted_candidate")
            physics.append({"candidate_id": candidate_id, "outcome": outcome,
                            "lineage_refs": _lineage_refs(record, technical=False)})
        else:
            raise ValueError("execution_status must be technical_failure or succeeded")
    accepted = sum(item["outcome"] == "accepted_candidate" for item in physics)
    rejected = sum(item["outcome"] == "physics_rejection" for item in physics)
    return {
        "status": "computed",
        "candidate_count": len(seen),
        "valid_execution_count": len(physics),
        "technical_failure_count": len(technical),
        "physics_rejection_count": rejected,
        "accepted_candidate_count": accepted,
        "technical_failures": technical,
        "physics_outcomes": physics,
    }


_TOP_LEVEL_KEYS = {"schema_version", "benchmark_id", "provenance", "lineage_axes", "cohort_policy", "weight_policy", "arms"}
_ARM_KEYS = {"arm_id", "generation", "utility_tilt", "proposal_fidelity", "fairship_outcomes"}
_GENERATION_KEYS = {"measure", "sampling_method", "predeclared_cohort", "redraw_until_success", "cohort_manifest_ref",
                    "cohort_manifest_sha256", "candidate_provenance", "proposal_provenance", "fairship_configuration_id",
                    "cohort_id", "candidate_ids", "candidate_count"}
_GENERATION_REQUIRED = {"measure", "sampling_method", "predeclared_cohort", "redraw_until_success", "cohort_manifest_ref",
                        "cohort_manifest_sha256", "candidate_provenance", "proposal_provenance", "fairship_configuration_id"}
_FAIRSHIP_KEYS = {"status", "candidate_count", "valid_execution_count", "technical_failure_count", "physics_rejection_count",
                  "accepted_candidate_count", "technical_failures", "physics_outcomes"}
_COHORT_KEYS = {"cohort_id", "candidate_ids", "predeclared", "redraw_until_success", "manifest_ref", "manifest_sha256",
                "candidate_provenance", "proposal_provenance"}
_PROPOSAL_KEYS = {"proposal_id", "proposal_version", "checkpoint_id"}
_CANDIDATE_KEYS = {"source_state_definition_id", "dataset_hash", "candidate_table_ref"}
_UTILITY_KEYS = {"status", "normalization", "support", "enrichment_vs_concentration", "ess", "ess_is_diagnostic_only", "empirical_utility_association"}
_FIDELITY_KEYS = {"status", "fit_metric", "physical_space_diagnostics", "high_utility_region_occupancy", "two_sample_diagnostic",
                  "declared_target_measure", "comparison_target"}


def _validate_fairship_summary(value: Mapping[str, Any], cohort_ids: Sequence[str] | None = None) -> None:
    if set(value) - _FAIRSHIP_KEYS:
        raise ValueError("unknown fairship_outcomes field")
    if value.get("status") not in {"not_run", "computed"}:
        raise ValueError("FairShip status must be not_run or computed")
    if value.get("status") != "computed":
        return
    required = ("candidate_count", "valid_execution_count", "technical_failure_count",
                "physics_rejection_count", "accepted_candidate_count", "technical_failures", "physics_outcomes")
    if any(field not in value for field in required):
        raise ValueError("computed FairShip outcomes require all count and record fields")
    counts = tuple(value[field] for field in required[:5])
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in counts):
        raise ValueError("FairShip outcome counts must be non-negative integers")
    technical = value["technical_failures"]
    physics = value["physics_outcomes"]
    if not isinstance(technical, list) or not isinstance(physics, list):
        raise ValueError("FairShip outcome records must be lists")
    technical_ids = [item.get("candidate_id") for item in technical]
    physics_ids = [item.get("candidate_id") for item in physics]
    if len(technical_ids) != len(set(technical_ids)) or len(physics_ids) != len(set(physics_ids)):
        raise ValueError("FairShip outcome ids must be unique within each bucket")
    if set(technical_ids) & set(physics_ids):
        raise ValueError("technical and physics outcome buckets must be disjoint")
    if value["candidate_count"] != value["valid_execution_count"] + value["technical_failure_count"]:
        raise ValueError("candidate_count must equal valid plus technical executions")
    if value["valid_execution_count"] != value["physics_rejection_count"] + value["accepted_candidate_count"]:
        raise ValueError("valid_execution_count must equal physics rejection plus accepted counts")
    if len(technical) != value["technical_failure_count"] or len(physics) != value["valid_execution_count"]:
        raise ValueError("FairShip counts must match outcome records")
    if cohort_ids is not None and set(technical_ids + physics_ids) != set(cohort_ids):
        raise ValueError("FairShip outcome ids must resolve to the declared cohort manifest")
    for item in technical:
        if set(item) != {"candidate_id", "reason", "lineage_refs"}:
            raise ValueError("technical outcome has unknown or missing fields")
    for item in physics:
        if set(item) != {"candidate_id", "outcome", "lineage_refs"} or item["outcome"] not in {"physics_rejection", "accepted_candidate"}:
            raise ValueError("physics outcome has unknown or invalid fields")
        if not isinstance(item["lineage_refs"], Mapping):
            raise ValueError("physics outcome must preserve lineage_refs")


def build_report(
    config: BenchmarkConfig,
    arm_data: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Create the identical report structure for P0, direct PU, and Q_theta.

    ``arm_data`` may contain measured sections.  Missing sections remain
    ``not_run``; no empirical FairShip value is inferred.  Candidate ids are
    accepted only through a predeclared cohort declaration.
    """

    supplied = arm_data or {}
    unknown = set(supplied) - set(ARM_ORDER)
    if unknown:
        raise ValueError(f"unknown benchmark arms: {sorted(unknown)}")
    arms: dict[str, Any] = {}
    for arm in ARM_ORDER:
        report = _empty_arm(arm)
        data = supplied.get(arm, {})
        if not isinstance(data, Mapping):
            raise TypeError(f"{arm} arm data must be a mapping")
        unknown_data = set(data) - {"cohort", "utility_tilt", "proposal_fidelity", "fairship_outcomes"}
        if unknown_data:
            raise ValueError(f"unknown {arm} arm fields: {sorted(unknown_data)}")
        cohort = data.get("cohort")
        if cohort is not None:
            if not isinstance(cohort, Mapping) or set(cohort) - _COHORT_KEYS:
                raise ValueError(f"{arm} cohort has unknown fields")
            if not cohort.get("predeclared", False):
                raise ValueError(f"{arm} cohort must be predeclared")
            if cohort.get("redraw_until_success", False):
                raise ValueError("redraw_until_success is forbidden")
            for field in ("manifest_ref", "manifest_sha256"):
                _nonempty(cohort.get(field), field)
            candidate_ids = tuple(_nonempty(item, "candidate_id") for item in cohort.get("candidate_ids", ()))
            if len(candidate_ids) != len(set(candidate_ids)):
                raise ValueError(f"{arm} cohort contains duplicate candidate ids")
            candidate_provenance = cohort.get("candidate_provenance")
            proposal_provenance = cohort.get("proposal_provenance")
            if not isinstance(candidate_provenance, Mapping) or set(candidate_provenance) != _CANDIDATE_KEYS:
                raise ValueError(f"{arm} cohort requires candidate provenance fields {_CANDIDATE_KEYS}")
            if not isinstance(proposal_provenance, Mapping) or set(proposal_provenance) != _PROPOSAL_KEYS:
                raise ValueError(f"{arm} cohort requires proposal provenance fields {_PROPOSAL_KEYS}")
            for field in _CANDIDATE_KEYS:
                _nonempty(candidate_provenance.get(field), field)
            for field in ("proposal_id", "proposal_version"):
                _nonempty(proposal_provenance.get(field), field)
            checkpoint_id = proposal_provenance.get("checkpoint_id")
            if arm == "Q_THETA":
                _nonempty(checkpoint_id, "checkpoint_id")
            elif checkpoint_id is not None:
                _nonempty(checkpoint_id, "checkpoint_id")
            report["generation"].update({
                "predeclared_cohort": True,
                "cohort_id": _nonempty(cohort.get("cohort_id"), "cohort_id"),
                "candidate_ids": list(candidate_ids),
                "candidate_count": len(candidate_ids),
                "cohort_manifest_ref": cohort["manifest_ref"],
                "cohort_manifest_sha256": cohort["manifest_sha256"],
                "candidate_provenance": dict(candidate_provenance),
                "proposal_provenance": dict(proposal_provenance),
                "fairship_configuration_id": config.fairship_configuration_id,
            })
        for section in ("utility_tilt", "proposal_fidelity", "fairship_outcomes"):
            if section in data:
                if not isinstance(data[section], Mapping):
                    raise TypeError(f"{arm} {section} must be a mapping")
                allowed = {"utility_tilt": _UTILITY_KEYS, "proposal_fidelity": _FIDELITY_KEYS,
                           "fairship_outcomes": _FAIRSHIP_KEYS}[section]
                if set(data[section]) - allowed:
                    raise ValueError(f"unknown {arm} {section} field")
                report[section].update(data[section])
        report["generation"]["redraw_until_success"] = False
        if report["fairship_outcomes"].get("status") == "computed" and cohort is None:
            raise ValueError(f"{arm} computed FairShip outcomes require a predeclared cohort")
        if report["fairship_outcomes"].get("status") == "computed":
            _validate_fairship_summary(report["fairship_outcomes"], report["generation"]["candidate_ids"])
        if report["fairship_outcomes"].get("candidate_count") is not None and cohort is not None:
            if report["fairship_outcomes"]["candidate_count"] != len(report["generation"]["candidate_ids"]):
                raise ValueError(f"{arm} FairShip count does not match declared cohort")
        target = "PU_DIRECT" if arm == "Q_THETA" else arm
        if report["proposal_fidelity"].get("comparison_target") != target or report["proposal_fidelity"].get("declared_target_measure") != target:
            raise ValueError(f"{arm} proposal fidelity must compare against declared {target}")
        arms[arm] = report
    result = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_id": config.benchmark_id,
        "provenance": {
            "fairship_configuration_id": config.fairship_configuration_id,
            "geometry_tag": config.geometry_tag,
            "source_state_definition_id": config.source_state_definition_id,
            "downstream_endpoint_definition_id": config.downstream_endpoint_definition_id,
            "cohort_manifest_id": config.cohort_manifest_id,
        },
        "lineage_axes": {
            "source_state": "declared input state and physical source weight",
            "fs_sim_execution": "execution health and technical censoring",
            "interaction_realization": "zero_or_more_realizations_per_execution",
            "observation": "measured detector/downstream evidence",
            "intermediate_Y_k_is_not_endpoint_B": True,
        },
        "cohort_policy": {
            "predeclared_before_fairship": True,
            "no_redraw_until_success": config.no_redraw_until_success,
        },
        "weight_policy": {
            "physical_source_weight_field": "physical_source_weight",
            "utility_multiplier_field": "utility_multiplier",
            "distinct_objects": True,
        },
        "arms": arms,
    }
    validate_report(result)
    return result


def validate_report(report: Mapping[str, Any]) -> None:
    """Small structural gate suitable for CI and artifact checks."""

    if not isinstance(report, Mapping) or set(report) != _TOP_LEVEL_KEYS:
        raise ValueError("report has unknown or missing top-level fields")
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported benchmark report schema")
    if tuple(report.get("arms", {}).keys()) != ARM_ORDER:
        raise ValueError("report must contain P0, PU_DIRECT, Q_THETA in order")
    for arm in ARM_ORDER:
        value = report["arms"][arm]
        if set(value) != _ARM_KEYS:
            raise ValueError(f"{arm} report has unknown or missing fields")
        if set(value["generation"]) - _GENERATION_KEYS:
            raise ValueError(f"{arm} generation has unknown fields")
        if not _GENERATION_REQUIRED <= set(value["generation"]):
            raise ValueError(f"{arm} generation has missing provenance fields")
        if set(value["utility_tilt"]) != _UTILITY_KEYS or set(value["proposal_fidelity"]) != _FIDELITY_KEYS:
            raise ValueError(f"{arm} report sections do not match the standard schema")
        if set(value["fairship_outcomes"]) != _FAIRSHIP_KEYS:
            raise ValueError(f"{arm} FairShip section does not match the standard schema")
        if value["generation"]["sampling_method"] != _SAMPLING_METHOD[arm]:
            raise ValueError(f"wrong sampling method for {arm}")
        if value["generation"].get("redraw_until_success"):
            raise ValueError("redraw_until_success is forbidden")
        if value["utility_tilt"].get("ess_is_diagnostic_only") is not True:
            raise ValueError("ESS must be marked diagnostic only")
        if value["utility_tilt"].get("status") not in {"not_run", "computed"}:
            raise ValueError("utility_tilt status must be not_run or computed")
        if value["proposal_fidelity"].get("status") not in {"not_run", "computed"}:
            raise ValueError("proposal_fidelity status must be not_run or computed")
        target = "PU_DIRECT" if arm == "Q_THETA" else arm
        if value["proposal_fidelity"].get("comparison_target") != target:
            raise ValueError(f"{arm} proposal fidelity must compare against declared {target}")
        if value["fairship_outcomes"].get("status") == "computed":
            if not value["generation"].get("predeclared_cohort"):
                raise ValueError(f"{arm} computed FairShip outcomes require a predeclared cohort")
        _validate_fairship_summary(value["fairship_outcomes"], value["generation"].get("candidate_ids"))


__all__ = [
    "ARM_ORDER",
    "SCHEMA_VERSION",
    "BenchmarkArm",
    "BenchmarkConfig",
    "build_report",
    "load_config",
    "summarize_fairship_outcomes",
    "validate_report",
]
