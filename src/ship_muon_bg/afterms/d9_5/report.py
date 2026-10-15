"""D9-5 validation-only selection freeze and final report (Sec 12/13/16).

``family_selection_manifest.json`` is the hard gate between validation and
test: :func:`require_frozen_selection` is what ``evaluate-test`` calls before
touching a test shard (Sec 17, required tests 27/28) -- refusal is structural
(a missing file), not a convention. The final report never carries a global
cross-track winner (Sec 15/16/31): every section is keyed by ``track_id`` and
nothing compares across tracks.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

from . import model_adapter as ma

FREEZE_MANIFEST_FILENAME = "family_selection_manifest.json"


class SelectionNotFrozenError(RuntimeError):
    pass


def _primary_value(aggregate_record: Dict[str, Any]) -> Any:
    if aggregate_record.get("fitting_policy") == "deterministic_single_fit":
        return aggregate_record.get(aggregate_record["metric_key"])
    return aggregate_record.get("mean")


def build_family_selection_manifest(per_track_aggregates: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """``per_track_aggregates``: ``{track_id: [aggregate_record, ...]}``, each
    record from :func:`aggregation.aggregate_seeds` or
    :func:`aggregation.deterministic_fit_record`. Primary metric is validation
    physical-space mean NLL (Sec 12); ranking is within-track only, never
    across tracks (Sec 15/31)."""

    tracks_out: Dict[str, Any] = {}
    for track_id, aggregates in per_track_aggregates.items():
        eligible = [a for a in aggregates if _primary_value(a) is not None]
        ranked = sorted(eligible, key=lambda a: _primary_value(a))
        tracks_out[track_id] = {
            "candidates": aggregates,
            "ranked_by_validation_physical_nll": [
                {"model_config_id": a["model_config_id"], "value": _primary_value(a)} for a in ranked
            ],
        }

    payload: Dict[str, Any] = {
        "schema_version": "d9_5_family_selection_manifest_v0",
        "tracks": tracks_out,
        "primary_metric_statement": (
            "Validation physical-space mean NLL is primary but not the only scientific diagnostic "
            "(Sec 12); no global cross-track ranking is produced or implied (Sec 15/31)."
        ),
    }
    payload["content_hash"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    return payload


def write_freeze_manifest(output_dir: Path, manifest: Dict[str, Any]) -> Path:
    path = Path(output_dir) / FREEZE_MANIFEST_FILENAME
    ma.atomic_write_json(path, manifest)
    return path


def is_selection_frozen(output_dir: Path) -> bool:
    return (Path(output_dir) / FREEZE_MANIFEST_FILENAME).exists()


def require_frozen_selection(output_dir: Path) -> Dict[str, Any]:
    """Sec 12/17, required tests 27/28: ``evaluate-test`` calls this before
    reading any test shard; a missing manifest is a hard refusal."""

    path = Path(output_dir) / FREEZE_MANIFEST_FILENAME
    if not path.exists():
        raise SelectionNotFrozenError(
            f"no frozen validation-only family selection manifest at {path}; run "
            "'freeze-selection' before 'evaluate-test'"
        )
    return ma.read_json(path)


_CSV_FIELDS = ("track_id", "model_family_id", "model_config_id", "test_physical_nll", "test_feature_nll", "finite_log_prob_fraction")


def render_final_report_json(per_track_test_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    return {
        "schema_version": "d9_5_model_family_arena_report_v0",
        "tracks": per_track_test_results,
        "no_global_cross_track_winner": True,
    }


def _csv_rows(per_track_test_results: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    rows = []
    for track_id, results in per_track_test_results.items():
        for record in results:
            rows.append({
                "track_id": track_id,
                "model_family_id": record.get("model_family_id"),
                "model_config_id": record.get("model_config_id"),
                "test_physical_nll": record.get("physical_nll"),
                "test_feature_nll": record.get("feature_nll"),
                "finite_log_prob_fraction": record.get("finite_log_prob_fraction"),
            })
    return rows


def render_final_report_md(per_track_test_results: Dict[str, List[Dict[str, Any]]]) -> str:
    lines = ["# D9-5 Model Family Arena Report", ""]
    for track_id, results in per_track_test_results.items():
        lines.append(f"## {track_id}")
        lines.append("")
        lines.append("| model_config_id | test physical NLL | test feature NLL | finite log-prob fraction |")
        lines.append("|---|---|---|---|")
        for record in results:
            lines.append(
                f"| {record.get('model_config_id')} | {record.get('physical_nll')} | "
                f"{record.get('feature_nll')} | {record.get('finite_log_prob_fraction')} |"
            )
        lines.append("")
    lines.append("No global cross-track winner is reported (Sec 15/16/31); "
                  "TRK_PDG13_UW_ID and TRK_PDGM13_UW_ID are never ranked against each other.")
    return "\n".join(lines)


def write_final_report(output_dir: Path, per_track_test_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    json_path = output_dir / "d9_5_model_family_arena.json"
    csv_path = output_dir / "d9_5_model_family_arena.csv"
    md_path = output_dir / "d9_5_model_family_arena.md"

    ma.atomic_write_json(json_path, render_final_report_json(per_track_test_results))

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_CSV_FIELDS))
        writer.writeheader()
        writer.writerows(_csv_rows(per_track_test_results))

    md_path.write_text(render_final_report_md(per_track_test_results), encoding="utf-8")
    return {"json": json_path, "csv": csv_path, "md": md_path}
