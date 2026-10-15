#!/usr/bin/env python3
"""build_afterms_model_alias_inventory.py: read-only GAUSS/GMM alias inventory (D9C Phase C).

Reads the frozen D8 run registry (``artifacts/afterms_d8_evaluation_v0/registry/run_registry.json``)
and derives GAUSS/GMM ``model_config_id`` aliases from verified evidence only:
the registry's own ``architecture``/``modeled_dimension`` fields, plus the
producer's hardcoded GMM covariance type (``Nflow/baselines/gmm.py``:
``covariance_type="full"`` is not a configurable field of
``GaussianMixtureEstimator``, so citing it is producer-code evidence, not an
invented value).

Never fits, refits, or reconstructs a model; never writes to
``artifacts/afterms_d8_evaluation_v0/`` or any other frozen D7/D8 path.
Writes only under ``artifacts/afterms_model_alias_inventory_v0/`` (gitignored,
not committed).

The D8 registry mislabels ``weighting_policy``/``target_measure`` for these
six rows (``weighting_policy: false``, ``target_measure: "row_empirical_unweighted"``);
none of D7/D8's Gaussian/GMM control jobs (10, 11) appear in the producer's
``JOB_WEIGHTED`` set (``src/ship_muon_bg/afterms/d8/legacy_d7_adapter.py``), so
every one of these controls is unweighted -- that fact is asserted directly
here rather than trusted from the mislabeled registry field.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from ship_muon_bg.afterms import model_naming  # noqa: E402

D8_REGISTRY_PATH = REPO_ROOT / "artifacts" / "afterms_d8_evaluation_v0" / "registry" / "run_registry.json"
OUTPUT_DIR = REPO_ROOT / "artifacts" / "afterms_model_alias_inventory_v0"

# Nflow/baselines/gmm.py's GaussianMixtureEstimator.fit() always constructs
# sklearn's GaussianMixture with covariance_type="full" -- this is a hardcoded
# literal in the producer's fit() call, never a constructor parameter of
# GaussianMixtureEstimator, so it cannot vary run to run.
GMM_COVARIANCE_TYPE_EVIDENCE = "full"
GMM_COVARIANCE_TYPE_SOURCE = (
    "Nflow/baselines/gmm.py (GaussianMixtureEstimator.fit hardcodes covariance_type='full'; "
    "not a configurable field, so every gaussian_mixture run in this registry used it)"
)
UNWEIGHTED_EVIDENCE_SOURCE = (
    "src/ship_muon_bg/afterms/d8/legacy_d7_adapter.py JOB_WEIGHTED (jobs 10/11 absent, "
    "so both Gaussian/GMM control jobs are unweighted -- the registry's own "
    "weighting_policy/target_measure fields for these rows are mislabeled)"
)

GAUSS_GMM_FAMILIES = ("diagonal_gaussian", "full_gaussian", "gaussian_mixture")
UNWEIGHTED_POLICY = "row_empirical_unweighted"


def load_registry_records() -> List[Dict[str, Any]]:
    return json.loads(D8_REGISTRY_PATH.read_text(encoding="utf-8"))


def build_inventory() -> Dict[str, Any]:
    alias_registry = model_naming.load_alias_registry()
    records = load_registry_records()

    resolved: List[Dict[str, Any]] = []
    unresolved: List[Dict[str, Any]] = []

    for rec in records:
        family = rec.get("model_family")
        if family not in GAUSS_GMM_FAMILIES:
            continue

        pdg_value = rec.get("pdg_value")
        preprocessing_name = rec.get("preprocessing_name")
        dimension = rec.get("modeled_dimension")
        arch = rec.get("architecture", {}) or {}

        track = model_naming.resolve_track(
            alias_registry,
            pdg_value=pdg_value,
            weighting_policy=UNWEIGHTED_POLICY,
            preprocessing_name=preprocessing_name,
        )

        evidence_parts = [
            f"artifacts/afterms_d8_evaluation_v0/registry/run_registry.json#{rec['run_id']}",
            UNWEIGHTED_EVIDENCE_SOURCE,
        ]

        if family in ("diagonal_gaussian", "full_gaussian"):
            if dimension is None:
                unresolved.append({
                    "run_id": rec["run_id"], "model_family": family,
                    "missing_fields": ["modeled_dimension"],
                })
                continue
            model_config = model_naming.gauss_model_config(family=family, dimension=dimension)
        else:  # gaussian_mixture
            n_components = arch.get("n_components")
            covariance_type = GMM_COVARIANCE_TYPE_EVIDENCE if n_components is not None else None
            model_config = model_naming.gmm_model_config(
                n_components=n_components,
                covariance_type=covariance_type,
                dimension=dimension,
            )
            if model_config.get("alias_resolution_status") == "UNRESOLVED":
                unresolved.append({
                    "run_id": rec["run_id"], "model_family": family,
                    "missing_fields": model_config["unresolved_missing_fields"],
                })
                continue
            evidence_parts.append(GMM_COVARIANCE_TYPE_SOURCE)

        alias_record = model_naming.build_alias_record(
            internal_candidate_id=rec["run_id"],
            legacy_run_id=rec["run_id"],
            track=track,
            model_config=model_config,
            preprocessing_name=preprocessing_name,
            weighting_policy=UNWEIGHTED_POLICY,
            pdg_value=pdg_value,
            modeled_features=rec.get("modeled_features", []),
            modeled_dimension=dimension,
            alias_registry_version=alias_registry["schema_version"],
            alias_evidence_source="; ".join(evidence_parts),
        )
        alias_record["d8_reconstruction_status"] = rec.get("reconstruction_status")
        alias_record["d8_checkpoint_path"] = rec.get("checkpoint_path")
        alias_record["d8_job_id"] = rec.get("job_id")
        resolved.append(alias_record)

    return {"resolved": resolved, "unresolved": unresolved}


def write_inventory(inventory: Dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    resolved = inventory["resolved"]
    unresolved = inventory["unresolved"]

    (OUTPUT_DIR / "existing_model_inventory.json").write_text(
        json.dumps(resolved, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    with (OUTPUT_DIR / "existing_model_inventory.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "internal_candidate_id", "track_id", "track_label", "model_family_id",
            "model_config_id", "display_name", "d8_reconstruction_status",
            "alias_resolution_status", "alias_evidence_source",
        ])
        for r in resolved:
            writer.writerow([
                r["internal_candidate_id"], r["track_id"], r["track_label"], r["model_family_id"],
                r["model_config_id"], r["display_name"], r["d8_reconstruction_status"],
                r["alias_resolution_status"], r["alias_evidence_source"],
            ])

    (OUTPUT_DIR / "unresolved_aliases.json").write_text(
        json.dumps(unresolved, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )

    md = [
        "# Existing D7/D8 GAUSS/GMM model alias inventory",
        "",
        "Read-only inventory derived from `artifacts/afterms_d8_evaluation_v0/registry/run_registry.json`. "
        "No model is refit or reconstructed here; missing historical checkpoints stay missing "
        "(`d8_reconstruction_status`).",
        "",
        "| internal_candidate_id (D8 run_id) | track_id | model_config_id | reconstruction_status |",
        "|---|---|---|---|",
    ]
    for r in resolved:
        md.append(
            f"| {r['internal_candidate_id']} | {r['track_id']} | {r['model_config_id']} | "
            f"{r['d8_reconstruction_status']} |"
        )
    md += ["", f"Unresolved: {len(unresolved)}", ""]
    for u in unresolved:
        md.append(f"- `{u['run_id']}` ({u['model_family']}): missing {u['missing_fields']}")
    (OUTPUT_DIR / "alias_inventory_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"Wrote {len(resolved)} resolved / {len(unresolved)} unresolved GAUSS/GMM aliases to {OUTPUT_DIR}")


def main() -> int:
    write_inventory(build_inventory())
    return 0


if __name__ == "__main__":
    sys.exit(main())
