"""D9B GPU-gate candidate selection (Gate B, section 2).

Derives the smallest enabled, reconstructible, unweighted modern 5D D9
candidate from the frozen D9 candidate plan (``configs/afterms/d9_candidate_plan_v0.json``)
joined against the D9 training config (``configs/afterms/d9_training_v0.json``)
-- never a hardcoded candidate_id.

"Smallest" is measured by ``parameter_count``. Among the D9 plan's six
primary candidates, three unweighted candidates (A2/B1/B2) tie at the same
parameter_count -- the tie-break is ``candidate_id`` ascending, applied
explicitly and reported in ``selection_report`` so the resolution is
reproducible and auditable rather than an implicit ordering artifact.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

UNWEIGHTED_POLICY = "row_empirical_unweighted"
RECONSTRUCTIBLE_STATUS = "RECONSTRUCTIBLE"
MODERN_DIMENSION = 5


class NoEligibleCandidateError(RuntimeError):
    """No D9 plan candidate satisfies the GPU-gate eligibility criteria."""


def eligible_plan_candidates(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Filter the D9 plan's candidates to those eligible for the GPU gate:
    enabled by default, reconstructible, row-empirical-unweighted, and a
    modern 5D candidate (never legacy 4D, never a weighted track)."""

    eligible = []
    for c in plan.get("candidates", []):
        if not c.get("D9_enabled_by_default"):
            continue
        if c.get("reconstruction_status") != RECONSTRUCTIBLE_STATUS:
            continue
        if c.get("weighting_policy") != UNWEIGHTED_POLICY:
            continue
        if c.get("modeled_dimension") != MODERN_DIMENSION:
            continue
        eligible.append(c)
    return eligible


def select_gpu_gate_plan_candidate(plan: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return ``(selected_plan_candidate, selection_report)``.

    Ranks eligible candidates by ``(parameter_count, candidate_id)`` ascending
    and returns the first. Raises ``NoEligibleCandidateError`` if no
    candidate is eligible, or if an eligible candidate is missing
    ``parameter_count`` (plan corruption -- never silently coerced to 0).
    """

    eligible = eligible_plan_candidates(plan)
    if not eligible:
        raise NoEligibleCandidateError(
            "no D9 candidate is simultaneously D9_enabled_by_default=true, "
            f"reconstruction_status={RECONSTRUCTIBLE_STATUS!r}, "
            f"weighting_policy={UNWEIGHTED_POLICY!r}, modeled_dimension={MODERN_DIMENSION}"
        )
    missing_params = [c["candidate_id"] for c in eligible if "parameter_count" not in c]
    if missing_params:
        raise NoEligibleCandidateError(f"eligible candidates missing parameter_count: {missing_params}")

    ranked = sorted(eligible, key=lambda c: (c["parameter_count"], c["candidate_id"]))
    selected = ranked[0]
    minimum_params = ranked[0]["parameter_count"]
    tied = [c["candidate_id"] for c in ranked if c["parameter_count"] == minimum_params]

    report = {
        "selection_rule": (
            "among candidates with D9_enabled_by_default=true, "
            f"reconstruction_status={RECONSTRUCTIBLE_STATUS!r}, "
            f"weighting_policy={UNWEIGHTED_POLICY!r}, modeled_dimension=={MODERN_DIMENSION}: "
            "pick min(parameter_count), tie-break candidate_id ascending"
        ),
        "eligible_candidate_ids_ranked": [c["candidate_id"] for c in ranked],
        "eligible_parameter_counts": {c["candidate_id"]: c["parameter_count"] for c in ranked},
        "selected_candidate_id": selected["candidate_id"],
        "minimum_parameter_count": minimum_params,
        "tied_at_minimum_candidate_ids": tied,
        "tie_break_applied": len(tied) > 1,
    }
    return selected, report


def select_gpu_gate_candidate(
    plan: Dict[str, Any], training_config: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return ``(training_config_candidate, selection_report)``: the full
    trainable candidate config (architecture, shards, optimizer, ...) for the
    plan candidate ``select_gpu_gate_plan_candidate`` resolves to."""

    plan_candidate, report = select_gpu_gate_plan_candidate(plan)
    candidate_id = plan_candidate["candidate_id"]
    by_id = {c["candidate_id"]: c for c in training_config.get("candidates", [])}
    if candidate_id not in by_id:
        raise NoEligibleCandidateError(
            f"selected plan candidate {candidate_id!r} has no matching entry in the training config"
        )
    return by_id[candidate_id], report
