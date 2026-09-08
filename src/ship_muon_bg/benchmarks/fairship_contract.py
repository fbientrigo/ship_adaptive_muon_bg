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


SCHEMA_VERSION = "utility_guided_fairship_benchmark_v0"
ARM_ORDER = ("P0", "PU_DIRECT", "Q_THETA")
_SAMPLING_METHOD = {
    "P0": "nominal_p0_sampling",
    "PU_DIRECT": "direct_pu_sampling",
    "Q_THETA": "learned_q_theta_generation",
}


class BenchmarkArm(str, Enum):
    P0 = "P0"
    PU_DIRECT = "PU_DIRECT"
    Q_THETA = "Q_THETA"


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
            "comparison_target": arm,
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
            technical.append({"candidate_id": candidate_id, "reason": record.get("failure_reason", "")})
        elif execution_status == "succeeded":
            if outcome not in {"physics_rejection", "accepted_candidate"}:
                raise ValueError("succeeded execution requires physics_rejection or accepted_candidate")
            physics.append({"candidate_id": candidate_id, "outcome": outcome})
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
        cohort = data.get("cohort")
        if cohort is not None:
            if not cohort.get("predeclared", False):
                raise ValueError(f"{arm} cohort must be predeclared")
            if cohort.get("redraw_until_success", False):
                raise ValueError("redraw_until_success is forbidden")
            candidate_ids = tuple(_nonempty(item, "candidate_id") for item in cohort.get("candidate_ids", ()))
            if len(candidate_ids) != len(set(candidate_ids)):
                raise ValueError(f"{arm} cohort contains duplicate candidate ids")
            report["generation"].update({
                "predeclared_cohort": True,
                "cohort_id": _nonempty(cohort.get("cohort_id"), "cohort_id"),
                "candidate_ids": list(candidate_ids),
                "candidate_count": len(candidate_ids),
            })
        for section in ("utility_tilt", "proposal_fidelity", "fairship_outcomes"):
            if section in data:
                report[section].update(data[section])
        report["generation"]["redraw_until_success"] = False
        if report["fairship_outcomes"].get("candidate_count") is not None and cohort is not None:
            if report["fairship_outcomes"]["candidate_count"] != len(report["generation"]["candidate_ids"]):
                raise ValueError(f"{arm} FairShip count does not match declared cohort")
        arms[arm] = report
    return {
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


def validate_report(report: Mapping[str, Any]) -> None:
    """Small structural gate suitable for CI and artifact checks."""

    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported benchmark report schema")
    if tuple(report.get("arms", {}).keys()) != ARM_ORDER:
        raise ValueError("report must contain P0, PU_DIRECT, Q_THETA in order")
    for arm in ARM_ORDER:
        value = report["arms"][arm]
        if value["generation"]["sampling_method"] != _SAMPLING_METHOD[arm]:
            raise ValueError(f"wrong sampling method for {arm}")
        if value["generation"].get("redraw_until_success"):
            raise ValueError("redraw_until_success is forbidden")
        if value["utility_tilt"].get("ess_is_diagnostic_only") is not True:
            raise ValueError("ESS must be marked diagnostic only")


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
