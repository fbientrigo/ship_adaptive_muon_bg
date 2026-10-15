"""Parsing and structural validation for the D9 candidate plan (§5).

This module never selects or re-derives candidates itself -- it only loads and
validates the frozen `configs/afterms/d9_candidate_plan_v0.json`, which was
derived by hand from the frozen D8 evaluation artifacts. Validation here
exists to catch plan-file corruption/drift, not to re-run candidate selection.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

REQUIRED_CANDIDATE_FIELDS = (
    "candidate_id",
    "source_d8_run_id",
    "source_arena_id",
    "pdg_value",
    "target_measure",
    "weighting_policy",
    "preprocessing",
    "model_family",
    "model_capacity",
    "modeled_features",
    "modeled_dimension",
    "D8_validation_evidence",
    "D8_support_evidence",
    "reconstruction_status",
    "selection_reason",
    "exclusion_reason",
    "D9_enabled_by_default",
)

VALID_STATUSES = ("D9_PLAN_VALID", "D9_PLAN_PARTIAL", "D9_PLAN_BLOCKED_BY_D8_EVIDENCE")

# Fields whose text content must never cite test-split metrics as a
# selection basis (D9 spec §5.2, required test 2/3).
_TEST_METRIC_MARKERS = ("test_nll", "test_nll=", "test_validation", "using test")


class PlanValidationError(ValueError):
    """The candidate plan JSON failed a structural or policy check."""


def load_plan(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_source_hashes(plan: Dict[str, Any], repo_root: Path) -> List[str]:
    """Return a list of mismatch descriptions (empty if all match)."""

    mismatches = []
    for label, entry in plan.get("source_d8_files", {}).items():
        path = Path(repo_root, entry["path"])
        if not path.exists():
            mismatches.append(f"{label}: missing file {path}")
            continue
        actual = file_sha256(path)
        if actual != entry["sha256"]:
            mismatches.append(
                f"{label}: sha256 mismatch, expected {entry['sha256']}, got {actual}"
            )
    return mismatches


def validate_plan(plan: Dict[str, Any]) -> List[str]:
    """Structural + policy validation. Returns a list of violations (empty if valid)."""

    violations: List[str] = []

    if plan.get("status") not in VALID_STATUSES:
        violations.append(f"unknown status {plan.get('status')!r}")

    candidates = plan.get("candidates", [])
    excluded = plan.get("excluded_candidates", [])

    for c in candidates:
        missing = [f for f in REQUIRED_CANDIDATE_FIELDS if f not in c]
        if missing:
            violations.append(f"candidate {c.get('candidate_id')!r} missing fields: {missing}")

    max_count = plan.get("maximum_primary_neural_candidate_count")
    if max_count is not None and len(candidates) > max_count:
        violations.append(
            f"{len(candidates)} primary candidates exceeds cap of {max_count}"
        )

    # Required test 2/3: selection reasons must not cite test-split metrics.
    for c in candidates:
        reason = (c.get("selection_reason") or "")
        for marker in _TEST_METRIC_MARKERS:
            if marker in reason:
                violations.append(
                    f"candidate {c['candidate_id']!r} selection_reason cites a test-split "
                    f"marker ({marker!r}); selection must use validation evidence only"
                )

    # Required test 4: weighted/unweighted tracks stay separate.
    policies_by_pdg: Dict[Any, set] = {}
    for c in candidates:
        key = c.get("pdg_value")
        policies_by_pdg.setdefault(key, set()).add(c.get("weighting_policy"))

    # Required test 5: PDG +13/-13 stay separate (no candidate mixes both).
    for c in candidates:
        pdg = c.get("pdg_value")
        if pdg not in (13, -13, None):
            violations.append(f"candidate {c['candidate_id']!r} has invalid pdg_value {pdg!r}")

    # Required test 6: legacy 4D / modern 5D stay separate.
    for c in candidates:
        if c.get("modeled_dimension") != 5 or list(c.get("modeled_features", [])) != [
            "px", "py", "pz", "x", "y"
        ]:
            violations.append(
                f"candidate {c['candidate_id']!r} is not a modern 5D candidate; "
                "legacy 4D must live only in excluded_candidates"
            )
    for e in excluded:
        if e.get("modeled_dimension") == 5 and e.get("D9_enabled_by_default"):
            violations.append(
                f"excluded candidate {e.get('candidate_id')!r} is a 5D candidate enabled by default; "
                "excluded candidates must have D9_enabled_by_default=false"
            )

    # Required test 8/9: physical weight `w` and utility multipliers never present.
    payload_text = json.dumps(plan)
    if '"w"' in payload_text and "modeled_features" in payload_text:
        for c in candidates:
            if "w" in c.get("modeled_features", []):
                violations.append(f"candidate {c['candidate_id']!r} lists 'w' as a modeled feature")
    if "utility_multiplier" in payload_text or "utility_weight" in payload_text:
        violations.append("plan references a utility multiplier/weight, which must not exist in D9")

    return violations
