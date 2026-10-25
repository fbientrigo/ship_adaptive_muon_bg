"""Parquet export of persisted ``EvaluationBundle`` records and immutable training snapshots.

Evidence stays where FairShip wrote it (``bundle.json`` + ROOT next to it).  ``export`` copies the five record kinds of
``bundle_records`` into one Parquet table each, row for row, adding only ``request_id`` and ``bundle_file`` so every row
leads back to its raw artifacts.  Nested fields (provenance, payloads, evidence lists) are stored as JSON text.  Shared
metadata (draws, scores, configuration) lives once in ``manifest.json``, not in every row.

``snapshot`` freezes one training input: which datasets, which stage, which estimator, which U and sampling policy,
and a grouped split.  Its id is the hash of all of that; an existing snapshot is never rewritten.  Rows are per-state
aggregates recomputed by ``tagging.aggregation`` from the exported records -- derived, never written back.

Query with DuckDB, e.g. ``duckdb.sql("select * from 'dataset/decisions.parquet' where stage_definition_id = ?")``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from ship_muon_bg.entities.decision import DecisionEvaluationStatus, StageDecision
from ship_muon_bg.entities.identifiers import content_hash, definition_id
from ship_muon_bg.entities.lineage import ExecutionStatus, FSSimExecution, ReconstructedCandidate
from ship_muon_bg.tagging.aggregation import aggregate_state_stage_evaluations

KINDS = ("executions", "realizations", "candidates", "observations", "decisions")


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _flat(row: Mapping[str, Any]) -> dict[str, Any]:
    return {k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in row.items()}


def export(bundle_files: Sequence[Path], out: Path, metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Write ``<out>/<kind>.parquet`` + ``manifest.json``.  Refuses physical/non-physical mixing and duplicate ids."""
    out.mkdir(parents=True, exist_ok=True)
    tables: dict[str, list[dict]] = {k: [] for k in KINDS}
    sources, physical = [], set()
    for path in sorted(Path(p) for p in bundle_files):
        b = json.loads(path.read_text())
        physical.add(b["is_physical"])
        sources.append({"bundle_file": str(path), "sha256": sha256(path), "request_id": b["request_id"], "backend": b["backend_name"]})
        for kind in KINDS:
            tables[kind] += [{"request_id": b["request_id"], "bundle_file": str(path), **_flat(r)} for r in b[kind]]
    if len(physical) > 1:
        raise ValueError("refusing to export physical and non-physical bundles together")
    if len({s["request_id"] for s in sources}) != len(sources):
        raise ValueError("duplicate request_id across bundles")
    files = {}
    for kind, rows in tables.items():
        id_col = {"executions": "execution_id", "realizations": "realization_id", "candidates": "candidate_id",
                  "observations": "observation_id", "decisions": "decision_id"}[kind]
        if len({r[id_col] for r in rows}) != len(rows):
            raise ValueError(f"duplicate {id_col} in {kind}")
        path = out / f"{kind}.parquet"
        pq.write_table(pa.Table.from_pylist(rows) if rows else pa.table({"request_id": pa.array([], pa.string())}), path)
        files[kind] = {"rows": len(rows), "sha256": sha256(path)}
    manifest = {"schema": "minimal_loop_dataset_v0", "is_physical": physical.pop() if physical else None,
                "tables": files, "bundles": sources, "metadata": dict(metadata)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return manifest


def _read(dataset: Path, kind: str) -> list[dict]:
    return pq.read_table(dataset / f"{kind}.parquet").to_pylist()


def load_records(datasets: Iterable[Path]) -> tuple[list[FSSimExecution], list[ReconstructedCandidate], list[StageDecision]]:
    """Rebuild the entities ``aggregate_state_stage_evaluations`` needs, after checking every table hash."""
    ex, cand, dec = [], [], []
    for d in datasets:
        manifest = json.loads((d / "manifest.json").read_text())
        for kind, info in manifest["tables"].items():
            if sha256(d / f"{kind}.parquet") != info["sha256"]:
                raise ValueError(f"{d / kind}.parquet does not match its manifest")
        ex += [FSSimExecution(r["execution_id"], r["subject_id"], r["fs_sim_configuration_id"], ExecutionStatus(r["execution_status"]),
                              r["seed"], r["failure_reason"], json.loads(r["provenance"])) for r in _read(d, "executions")]
        cand += [ReconstructedCandidate(r["candidate_id"], r["execution_id"], r.get("realization_id"), r.get("candidate_index"))
                 for r in _read(d, "candidates")]
        dec += [StageDecision(r["decision_id"], r["subject_ref"], r["stage_definition_id"], DecisionEvaluationStatus(r["evaluation_status"]),
                              r["decision"], r["reason"], tuple(json.loads(r["evidence_references"])), json.loads(r["config_provenance"]))
                for r in _read(d, "decisions")]
    return ex, cand, dec


def snapshot(datasets: Sequence[Path], stage_definition_id: str, out_root: Path, *, policy: Mapping[str, Any],
             split_seed: int, validation_fraction: float = 0.3) -> Path:
    """Freeze one training snapshot under ``<out_root>/<snapshot_id>/``.  ``policy`` must name the U version and the
    sampling policy that produced the states; it is part of the id.  The split is by source state (all executions of a
    state fall on the same side), decided by a hash of (seed, subject_id): deterministic, order-free."""
    datasets = [Path(d).resolve() for d in datasets]
    ex, cand, dec = load_records(datasets)
    if stage_definition_id not in {d.stage_definition_id for d in dec}:  # never fall back to another (e.g. newer) definition
        raise ValueError(f"stage {stage_definition_id} is not in the evidence")
    agg = aggregate_state_stage_evaluations(executions=ex, candidates=cand, decisions=dec, stage_definition_ids=[stage_definition_id])
    config = agg.require_single_configuration()  # incompatible configurations are never pooled
    if len({r.options_digest for r in agg.rows}) > 1:
        raise ValueError("more than one options digest for the same configuration")
    estimator = definition_id("eta_hat_positive_over_valid_v0", {"numerator": "positive executions", "denominator": "valid executions",
                              "rollup": sorted({r.rollup_rule for r in agg.rows}), "unit": "execution of one source state",
                              "independence": "not assumed between executions of one state"})
    spec = {"datasets": [{"path": str(d), "manifest_sha256": sha256(d / "manifest.json")} for d in datasets],
            "stage_definition_id": stage_definition_id, "estimator_definition_id": estimator, "fs_sim_configuration_id": config,
            "options_digest": agg.rows[0].options_digest if agg.rows else None, "policy": dict(policy),
            "split": {"group": "subject_id", "seed": split_seed, "validation_fraction": validation_fraction}}
    sid = "snap-" + content_hash(spec)[:16]
    target = out_root / sid
    if target.exists():
        return target  # content-addressed: same spec, same snapshot; nothing is overwritten
    target.mkdir(parents=True)
    rows = []
    for r in agg.rows:
        u = int(hashlib.sha256(f"{split_seed}:{r.subject_id}".encode()).hexdigest()[:8], 16) / 2 ** 32
        rows.append({"subject_id": r.subject_id, "split": "validation" if u < validation_fraction else "train",
                     "executions": r.execution_count, "valid": r.valid_count, "positive": r.positive_count, "negative": r.negative_count,
                     "technical_failure": r.technical_failure_count, "censored": r.technically_censored_count,
                     "not_evaluated": r.not_evaluated_count, "eta_hat": r.eta_hat})
    pq.write_table(pa.Table.from_pylist(rows), target / "rows.parquet")
    (target / "snapshot.json").write_text(json.dumps({**spec, "snapshot_id": sid, "rows_sha256": sha256(target / "rows.parquet")},
                                                     indent=1, sort_keys=True) + "\n")
    return target
