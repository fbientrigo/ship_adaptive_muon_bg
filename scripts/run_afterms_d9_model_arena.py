#!/usr/bin/env python3
"""run_afterms_d9_model_arena.py: D9 model-capacity arena (exploratory scouting).

Not part of the D9 production campaign contract. This is a capacity-variation
sweep -- for each of the 6 enabled D9 candidates ("models"), train 5 capacity
variations (scaled number_of_blocks/hidden_width, depth held fixed) for a
fixed number of epochs at a single fixed seed, then produce modular
comparison evidence within each model and across models.

Writes to a distinct artifact tree (default
artifacts/afterms_d9_model_arena_v0/, campaign_id
"afterms_d9_model_arena_v0") so it never collides with the real D9 production
campaign (campaign_id "afterms_d9_training_v0", which as of this script has
not been run). Reuses ``runner.train_candidate_seed`` completely unmodified
-- only candidate_id / architecture / max_epochs / campaign_id are
overridden per variation, exactly the same mechanism the real campaign CLI
uses for a normal (candidate, seed) run.

Comparability rule: the training loop only ever computes feature-space NLL
(no physical-space Jacobian correction -- that only happens in evaluate.py's
test-time evaluation, which reads the test shard and is out of scope here).
Feature-space NLL is therefore only comparable across candidates that share
the same (preprocessing_name, weighting_policy) pair. Cross-model comparison
below is grouped by that pair; it never ranks across groups.

Default action is a dry run (prints the planned run matrix only). Pass
--execute to actually train.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from ship_muon_bg.afterms.d9 import checkpoint as ckpt  # noqa: E402
from ship_muon_bg.afterms.d9 import runner as d9runner  # noqa: E402
from ship_muon_bg.afterms.d9 import training_config as d9tc  # noqa: E402
from ship_muon_bg.afterms.d9 import plan as d9plan  # noqa: E402
from ship_muon_bg.afterms import model_naming  # noqa: E402

DEFAULT_TRAINING_CONFIG_PATH = REPO_ROOT / "configs" / "afterms" / "d9_training_v0.json"
DEFAULT_CANDIDATE_PLAN_PATH = REPO_ROOT / "configs" / "afterms" / "d9_candidate_plan_v0.json"
DEFAULT_ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "afterms_d9_model_arena_v0"
ARENA_CAMPAIGN_ID = "afterms_d9_model_arena_v0"
ARENA_EXPERIMENT_ID = "AFFINE_COUPLING_CAPACITY_SCOUT_V0"
DEFAULT_SCALES = (0.5, 0.75, 1.0, 1.25, 1.5)
DEFAULT_SEED = 20260720
DEFAULT_EPOCHS = 10
VARIANT_SUFFIX_RE = re.compile(r"^(?P<base>.+)__arena_cap_(?P<scale>[0-9.]+)x$")


def comparability_group(candidate_config: Dict[str, Any]) -> Tuple[str, str]:
    return (candidate_config["preprocessing_name"], candidate_config["weighting_policy"])


def scaled_architecture(base_arch: Dict[str, Any], scale: float) -> Dict[str, Any]:
    blocks = max(2, round(base_arch["number_of_blocks"] * scale))
    width = max(8, round(base_arch["hidden_width"] * scale))
    depth = base_arch["hidden_depth"]
    return {
        "number_of_blocks": blocks,
        "hidden_width": width,
        "hidden_depth": depth,
        "capacity_label": f"arena_scale_{scale:.2f}x",
    }


def variation_candidate_id(base_candidate_id: str, scale: float) -> str:
    return f"{base_candidate_id}__arena_cap_{scale:.2f}x"


def build_variation_config(base_config: Dict[str, Any], scale: float, epochs: int) -> Dict[str, Any]:
    cfg = json.loads(json.dumps(base_config))  # deep copy, JSON-safe
    cfg["campaign_id"] = ARENA_CAMPAIGN_ID
    cfg["candidate_id"] = variation_candidate_id(base_config["candidate_id"], scale)
    cfg["architecture"] = scaled_architecture(base_config["architecture"], scale)
    cfg["max_epochs"] = epochs
    if cfg["minimum_epochs"] > epochs:
        cfg["minimum_epochs"] = epochs
    return cfg


_shard_cache: Dict[Tuple[str, ...], Any] = {}


def load_shards_cached(shard_dir: Path, shard_names: List[str]):
    key = tuple(shard_names)
    if key not in _shard_cache:
        _shard_cache[key] = d9runner.load_concatenated_shards(shard_dir, shard_names)
    return _shard_cache[key]


def collect_variant_row(artifact_root: Path, base_candidate_id: str, base_config: Dict[str, Any], scale: float, seed: int) -> Dict[str, Any]:
    variant_id = variation_candidate_id(base_candidate_id, scale)
    run_dir = d9runner.run_directory(artifact_root, variant_id, seed)
    row: Dict[str, Any] = {
        "variant_candidate_id": variant_id,
        "capacity_scale": scale,
        "architecture": scaled_architecture(base_config["architecture"], scale),
    }
    status_path = run_dir / "status.json"
    if not status_path.exists():
        row["status"] = "not_run"
        return row
    status = json.loads(status_path.read_text(encoding="utf-8"))
    row["status"] = status.get("status")
    row["best_validation_metric"] = status.get("best_validation_metric")
    row["best_validation_epoch"] = status.get("best_validation_epoch")
    row["final_epoch"] = status.get("final_epoch")

    history_path = run_dir / "histories" / "training_history.json"
    if history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))
        row["history"] = history
        if history:
            row["final_train_feature_nll"] = history[-1]["train_feature_nll"]
            row["final_validation_feature_nll"] = history[-1]["validation_feature_nll"]
            total_wall = sum(h.get("wall_time_seconds", 0.0) for h in history)
            row["total_wall_seconds"] = total_wall
            row["mean_epoch_wall_seconds"] = total_wall / len(history)

    final_ckpt_path = run_dir / "checkpoints" / ckpt.FILENAME_BY_SCOPE[ckpt.SCOPE_FINAL]
    if final_ckpt_path.exists():
        bundle = ckpt.load_bundle(final_ckpt_path)
        row["parameter_count"] = int(sum(t.numel() for t in bundle["model_state_dict"].values()))
    return row


def _rank_key(row: Dict[str, Any]):
    metric = row.get("best_validation_metric")
    return (metric is None, metric if metric is not None else float("inf"))


def build_comparison_reports(
    artifact_root: Path,
    selected_ids: List[str],
    all_candidates: Dict[str, Dict[str, Any]],
    scales: List[float],
    seed: int,
) -> None:
    per_model_dir = artifact_root / "per_model"
    per_model_dir.mkdir(parents=True, exist_ok=True)
    model_champions: Dict[str, Any] = {}

    for cid in selected_ids:
        base_config = all_candidates[cid]
        rows = [collect_variant_row(artifact_root, cid, base_config, s, seed) for s in scales]
        ranked = sorted(rows, key=_rank_key)
        champion = ranked[0] if ranked and ranked[0].get("best_validation_metric") is not None else None
        model_champions[cid] = champion

        out_dir = per_model_dir / cid
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "base_candidate_id": cid,
            "base_architecture": base_config["architecture"],
            "preprocessing_name": base_config["preprocessing_name"],
            "weighting_policy": base_config["weighting_policy"],
            "pdg_policy": base_config["pdg_policy"],
            "seed": seed,
            "variations": rows,
            "champion_variant_candidate_id": champion["variant_candidate_id"] if champion else None,
        }
        (out_dir / "comparison.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

        with (out_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow([
                "variant_candidate_id", "capacity_scale", "blocks", "width", "depth",
                "parameter_count", "status", "best_validation_metric", "best_validation_epoch",
                "final_train_feature_nll", "final_validation_feature_nll", "mean_epoch_wall_seconds",
            ])
            for r in ranked:
                arch = r.get("architecture", {})
                writer.writerow([
                    r.get("variant_candidate_id"), r.get("capacity_scale"),
                    arch.get("number_of_blocks"), arch.get("hidden_width"), arch.get("hidden_depth"),
                    r.get("parameter_count"), r.get("status"),
                    r.get("best_validation_metric"), r.get("best_validation_epoch"),
                    r.get("final_train_feature_nll"), r.get("final_validation_feature_nll"),
                    r.get("mean_epoch_wall_seconds"),
                ])

        md = [
            f"# Model arena -- {cid}",
            "",
            f"Base architecture: `{base_config['architecture']}`",
            f"Preprocessing: `{base_config['preprocessing_name']}` | Weighting: `{base_config['weighting_policy']}` | PDG policy: `{base_config['pdg_policy']}`",
            "",
            f"Ranked by validation feature-space NLL (lower is better), seed fixed at {seed}:",
            "",
            "| rank | variant | scale | blocks/width/depth | params | status | best_val_nll | best_epoch |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for i, r in enumerate(ranked, 1):
            arch = r.get("architecture", {})
            md.append(
                f"| {i} | {r.get('variant_candidate_id')} | {r.get('capacity_scale')}x | "
                f"{arch.get('number_of_blocks')}/{arch.get('hidden_width')}/{arch.get('hidden_depth')} | "
                f"{r.get('parameter_count')} | {r.get('status')} | {r.get('best_validation_metric')} | "
                f"{r.get('best_validation_epoch')} |"
            )
        if champion:
            md += ["", f"**Champion variation:** `{champion['variant_candidate_id']}` (scale {champion['capacity_scale']}x)"]
        (out_dir / "comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    groups: Dict[Tuple[str, str], List[str]] = {}
    for cid in selected_ids:
        groups.setdefault(comparability_group(all_candidates[cid]), []).append(cid)

    cross_dir = artifact_root / "cross_model"
    cross_dir.mkdir(parents=True, exist_ok=True)
    cross_payload: Dict[str, Any] = {"seed": seed, "groups": []}
    md = [
        "# Model arena -- cross-model comparison",
        "",
        "Champion variation (lowest validation feature-space NLL among each model's capacity "
        "variations) per model, grouped by comparability axis.",
        "",
        "**Comparability rule:** feature-space validation NLL is only meaningfully comparable "
        "across models that share the same `(preprocessing_name, weighting_policy)` pair -- the "
        "training loop never applies the physical-space Jacobian correction (only `evaluate.py`'s "
        "test-time evaluation does, and that requires the test shard, out of scope here). Do not "
        "rank models across different preprocessing pipelines or weighted-vs-unweighted on this "
        "number.",
        "",
    ]
    for key, cids in groups.items():
        preprocessing_name, weighting_policy = key
        group_rows = [(cid, model_champions.get(cid)) for cid in cids]
        group_rows.sort(key=lambda t: _rank_key(t[1]) if t[1] else (True, float("inf")))
        md.append(f"## Group: preprocessing={preprocessing_name}, weighting={weighting_policy}")
        md.append("")
        md.append("| rank | model | pdg_policy | champion variant | scale | best_val_nll | params |")
        md.append("|---|---|---|---|---|---|---|")
        for i, (cid, champ) in enumerate(group_rows, 1):
            if champ is None:
                md.append(f"| {i} | {cid} | {all_candidates[cid]['pdg_policy']} | (no successful run) | - | - | - |")
                continue
            md.append(
                f"| {i} | {cid} | {all_candidates[cid]['pdg_policy']} | {champ['variant_candidate_id']} | "
                f"{champ['capacity_scale']}x | {champ.get('best_validation_metric')} | {champ.get('parameter_count')} |"
            )
        md.append("")
        cross_payload["groups"].append({
            "preprocessing_name": preprocessing_name,
            "weighting_policy": weighting_policy,
            "models": [{"candidate_id": cid, "champion": champ} for cid, champ in group_rows],
        })

    (cross_dir / "comparison.json").write_text(json.dumps(cross_payload, indent=2, default=str), encoding="utf-8")
    (cross_dir / "comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    with (cross_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "comparability_group_preprocessing", "comparability_group_weighting", "model",
            "pdg_policy", "champion_variant", "capacity_scale", "best_validation_metric", "parameter_count",
        ])
        for key, cids in groups.items():
            for cid in cids:
                champ = model_champions.get(cid)
                writer.writerow([
                    key[0], key[1], cid, all_candidates[cid]["pdg_policy"],
                    champ.get("variant_candidate_id") if champ else None,
                    champ.get("capacity_scale") if champ else None,
                    champ.get("best_validation_metric") if champ else None,
                    champ.get("parameter_count") if champ else None,
                ])

    print(f"\nComparison reports written under {artifact_root}")


def discover_scout_variant_ids(artifact_root: Path, base_candidate_id: str) -> List[str]:
    """List existing scout variant run directories for one base candidate,
    read-only (glob only, never derives ids from a scale multiplier)."""

    runs_dir = artifact_root / "runs"
    if not runs_dir.is_dir():
        return []
    prefix = f"{base_candidate_id}__arena_cap_"
    return sorted(entry.name for entry in runs_dir.iterdir() if entry.is_dir() and entry.name.startswith(prefix))


def load_variant_run_evidence(artifact_root: Path, variant_id: str, seed: int) -> Dict[str, Any] | None:
    run_dir = d9runner.run_directory(artifact_root, variant_id, seed)
    training_config_path = run_dir / "training_config.json"
    status_path = run_dir / "status.json"
    if not training_config_path.exists() or not status_path.exists():
        return None
    return {
        "training_config": json.loads(training_config_path.read_text(encoding="utf-8")),
        "status": json.loads(status_path.read_text(encoding="utf-8")),
    }


def _variant_rank_key(row: Dict[str, Any]):
    metric = row.get("best_validation_metric")
    return (metric is None, metric if metric is not None else float("inf"))


def build_named_alias_outputs(
    artifact_root: Path,
    candidate_plan: Dict[str, Any],
    training_config: Dict[str, Any],
    registry: Dict[str, Any],
    selected_ids: List[str],
    seed: int,
) -> Dict[str, Any]:
    """Read-only: derive human-readable aliases for the existing scout runs
    under ``artifact_root/runs`` and write them under ``artifact_root/named``.

    Never trains, never renames an existing run directory or arena file --
    every value comes from the frozen candidate plan / training config or
    from a specific run's own recorded ``training_config.json``/``status.json``.
    """

    plan_by_id = {c["candidate_id"]: c for c in candidate_plan.get("candidates", [])}
    training_by_id = {c["candidate_id"]: c for c in training_config["candidates"]}

    base_records: Dict[str, Dict[str, Any]] = {}
    track_sections: Dict[str, Dict[str, Any]] = {}
    inventory_rows: List[Dict[str, Any]] = []
    all_records_for_collision_check: List[Dict[str, Any]] = []

    for base_id in selected_ids:
        plan_entry = plan_by_id[base_id]
        training_entry = training_by_id[base_id]
        base_record = model_naming.resolve_d9_training_candidate_alias(
            registry=registry,
            plan_entry=plan_entry,
            training_entry=training_entry,
            evidence_source="configs/afterms/d9_candidate_plan_v0.json + configs/afterms/d9_training_v0.json",
        )
        base_records[base_id] = base_record
        all_records_for_collision_check.append(base_record)

        variant_entries: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        for variant_id in discover_scout_variant_ids(artifact_root, base_id):
            match = VARIANT_SUFFIX_RE.match(variant_id)
            scale = float(match.group("scale")) if match else None
            evidence = load_variant_run_evidence(artifact_root, variant_id, seed)
            if evidence is None:
                continue
            variant_record = model_naming.resolve_scout_variant_alias(
                registry=registry,
                variant_training_config=evidence["training_config"],
                base_candidate_id=base_id,
                capacity_scale=scale,
                evidence_source=(
                    f"artifacts/afterms_d9_model_arena_v0/runs/{variant_id}/seed_{seed}/training_config.json"
                ),
            )
            status = evidence["status"]
            row = {
                "internal_candidate_id": variant_record["internal_candidate_id"],
                "legacy_base_candidate_id": base_id,
                "legacy_capacity_scale": scale,
                "track_id": variant_record["track_id"],
                "model_config_id": variant_record["model_config_id"],
                "display_name": variant_record["display_name"],
                "status": status.get("status"),
                "best_validation_metric": status.get("best_validation_metric"),
                "best_validation_epoch": status.get("best_validation_epoch"),
                "final_epoch": status.get("final_epoch"),
            }
            variant_entries.append((variant_record, row))
            all_records_for_collision_check.append(variant_record)

        model_naming.assert_no_model_config_collisions(all_records_for_collision_check)

        ranked = sorted(variant_entries, key=lambda pair: _variant_rank_key(pair[1]))
        inventory_rows.extend(row for _, row in ranked)

        track_id = base_record["track_id"]
        track_sections[track_id] = {
            "track_id": track_id,
            "track_label": base_record["track_label"],
            "legacy_base_candidate_id": base_id,
            "base_model_config_id": base_record["model_config_id"],
            "variants": [
                {**variant_record, "best_validation_metric": row["best_validation_metric"],
                 "best_validation_epoch": row["best_validation_epoch"], "status": row["status"]}
                for variant_record, row in ranked
            ],
        }

    named_dir = artifact_root / "named"
    named_dir.mkdir(parents=True, exist_ok=True)

    alias_snapshot = {
        "alias_registry_version": registry["schema_version"],
        "scout_experiment_id": ARENA_EXPERIMENT_ID,
        "seed": seed,
        "base_candidates": base_records,
    }
    (named_dir / "alias_snapshot.json").write_text(
        json.dumps(alias_snapshot, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )

    (named_dir / "run_inventory_named.json").write_text(
        json.dumps(inventory_rows, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    with (named_dir / "run_inventory_named.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "internal_candidate_id", "legacy_base_candidate_id", "legacy_capacity_scale",
            "track_id", "model_config_id", "display_name", "status",
            "best_validation_metric", "best_validation_epoch", "final_epoch",
        ])
        for row in inventory_rows:
            writer.writerow([
                row["internal_candidate_id"], row["legacy_base_candidate_id"], row["legacy_capacity_scale"],
                row["track_id"], row["model_config_id"], row["display_name"], row["status"],
                row["best_validation_metric"], row["best_validation_epoch"], row["final_epoch"],
            ])

    md_lines = [
        "# Per-track model comparison",
        "",
        "Legacy candidate ids are internal identifiers, not model names -- see "
        "`docs/reviews/afterms_model_alias_registry_v0.md`. Rankings below are "
        "**within a single track only**; feature-space validation NLL is never "
        "compared across tracks (different empirical targets are not on the same "
        "ranking axis).",
        "",
    ]
    csv_rows: List[List[Any]] = []
    for track_id in sorted(track_sections):
        section = track_sections[track_id]
        md_lines += [
            f"## Track {track_id}",
            "",
            f"{section['track_label']}",
            "",
            f"Base model: `{section['base_model_config_id']}` (legacy candidate `{section['legacy_base_candidate_id']}`)",
            "",
            "| rank | model_config_id | legacy_scale | status | best_val_nll | best_epoch |",
            "|---|---|---|---|---|---|",
        ]
        for i, variant in enumerate(section["variants"], 1):
            md_lines.append(
                f"| {i} | {variant['model_config_id']} | {variant['scout_legacy_capacity_scale']} | "
                f"{variant['status']} | {variant['best_validation_metric']} | {variant['best_validation_epoch']} |"
            )
            csv_rows.append([
                track_id, i, variant["model_config_id"], variant["internal_candidate_id"],
                variant["scout_legacy_capacity_scale"], variant["status"],
                variant["best_validation_metric"], variant["best_validation_epoch"],
            ])
        md_lines.append("")

    (named_dir / "per_track_model_comparison.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    with (named_dir / "per_track_model_comparison.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "track_id", "rank", "model_config_id", "internal_candidate_id",
            "legacy_capacity_scale", "status", "best_validation_metric", "best_validation_epoch",
        ])
        for row in csv_rows:
            writer.writerow(row)

    summary_lines = [
        f"# {ARENA_EXPERIMENT_ID}",
        "",
        "Per-candidate capacity-variation sweep (5 capacity variants x 6 D9 "
        "candidates, single seed, 10 epochs). This is exploratory scouting, "
        "not the D9 production multi-seed campaign, and every variant here is "
        "an `NF_AC` (affine-coupling) configuration -- no other model family "
        "was run in this experiment.",
        "",
        "No global cross-track ranking is produced: each base candidate lives "
        "on its own track (distinct empirical target), so feature-space "
        "validation NLL is only compared within a track's own capacity sweep.",
        "",
    ]
    for track_id in sorted(track_sections):
        section = track_sections[track_id]
        best = section["variants"][0] if section["variants"] else None
        summary_lines.append(f"- Track `{track_id}` ({section['track_label']}), legacy candidate "
                              f"`{section['legacy_base_candidate_id']}`: "
                              + (f"best variant `{best['model_config_id']}` "
                                 f"(val NLL {best['best_validation_metric']})" if best else "no completed variants"))
    (named_dir / "affine_capacity_scout_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"Named alias reports written under {named_dir}")
    return {
        "alias_snapshot": alias_snapshot,
        "inventory_rows": inventory_rows,
        "track_sections": track_sections,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="D9 model arena: per-candidate capacity-variation sweep, exploratory scouting before any real multi-seed HPO campaign.",
    )
    parser.add_argument("--training-config-path", type=Path, default=DEFAULT_TRAINING_CONFIG_PATH)
    parser.add_argument("--candidate-plan-path", type=Path, default=DEFAULT_CANDIDATE_PLAN_PATH)
    parser.add_argument("--alias-registry-path", type=Path, default=model_naming.DEFAULT_REGISTRY_PATH)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--candidates", type=str, default=None, help="comma-separated candidate_ids; default = all candidates in the training config")
    parser.add_argument("--scales", type=str, default=",".join(str(s) for s in DEFAULT_SCALES))
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--execute", action="store_true", help="actually train (default: dry-run, prints the planned run matrix only)")
    parser.add_argument("--report-only", action="store_true", help="skip training; rebuild comparison reports from existing run artifacts only")
    parser.add_argument(
        "--relabel-existing", action="store_true",
        help="read-only: derive human-readable track/model aliases for existing runs under "
             "--artifact-root and write them under <artifact-root>/named/. Never trains, never "
             "renames or rewrites an existing arena file.",
    )
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)

    training_config = d9tc.load_training_config(args.training_config_path)
    shard_dir = REPO_ROOT / training_config["shard_dir"]
    all_candidates = {c["candidate_id"]: c for c in training_config["candidates"]}
    selected_ids = [c.strip() for c in args.candidates.split(",")] if args.candidates else list(all_candidates.keys())
    scales = [float(s) for s in args.scales.split(",")]

    artifact_root = args.artifact_root
    artifact_root.mkdir(parents=True, exist_ok=True)

    if args.relabel_existing:
        candidate_plan = d9plan.load_plan(args.candidate_plan_path)
        registry = model_naming.load_alias_registry(args.alias_registry_path)
        build_named_alias_outputs(artifact_root, candidate_plan, training_config, registry, selected_ids, args.seed)
        return 0

    if args.report_only:
        build_comparison_reports(artifact_root, selected_ids, all_candidates, scales, args.seed)
        return 0

    run_matrix = [(all_candidates[cid], scale) for cid in selected_ids for scale in scales]

    print(f"Model arena: {len(selected_ids)} models x {len(scales)} capacity variations = {len(run_matrix)} runs")
    print(f"scales={scales} epochs={args.epochs} seed={args.seed} device={args.device} artifact_root={artifact_root}")
    for base_config, scale in run_matrix:
        vcfg = build_variation_config(base_config, scale, args.epochs)
        arch = vcfg["architecture"]
        print(f"  {vcfg['candidate_id']}: blocks={arch['number_of_blocks']} width={arch['hidden_width']} depth={arch['hidden_depth']}")

    if not args.execute:
        print("\nDry run only (pass --execute to actually train). No GPU work performed.")
        return 0

    manifest_path = artifact_root / "arena_manifest.json"
    manifest: Dict[str, Any] = {
        "campaign_id": ARENA_CAMPAIGN_ID,
        "seed": args.seed,
        "device": args.device,
        "epochs": args.epochs,
        "scales": scales,
        "candidates": selected_ids,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "runs": [],
    }

    def write_manifest() -> None:
        tmp = artifact_root / ".arena_manifest.json.tmp"
        tmp.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        tmp.replace(manifest_path)

    write_manifest()

    import torch

    for base_config, scale in run_matrix:
        vcfg = build_variation_config(base_config, scale, args.epochs)
        variant_id = vcfg["candidate_id"]
        print(f"\n=== {variant_id} (seed {args.seed}) ===", flush=True)

        train_raw = load_shards_cached(shard_dir, vcfg["train_shards"])
        validation_raw = load_shards_cached(shard_dir, vcfg["validation_shards"])

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        run_record: Dict[str, Any] = {
            "base_candidate_id": base_config["candidate_id"],
            "variant_candidate_id": variant_id,
            "capacity_scale": scale,
            "architecture": vcfg["architecture"],
        }
        run_start = time.perf_counter()
        try:
            # resume=True: idempotent across orchestrator restarts. A fresh
            # variant has no checkpoint yet (starts at epoch 0); a variant
            # left mid-run by a prior interrupted orchestrator process
            # resumes exactly at its last completed epoch boundary (Gate B.1
            # exact-resume contract); an already-completed variant reloads
            # its final checkpoint and re-verifies rather than retraining.
            result = d9runner.train_candidate_seed(
                vcfg, args.seed, train_raw, validation_raw,
                artifact_root=artifact_root, repo_root=REPO_ROOT, device=args.device, resume=True,
            )
            run_record["status"] = result["status"]
            run_record["result"] = result
            print(f"  -> {result['status']} best_val={result.get('best_validation_metric')} best_epoch={result.get('best_validation_epoch')}")
        except Exception as exc:  # noqa: BLE001 -- one variant's failure must not abort the sweep
            run_record["status"] = "orchestrator_caught_exception"
            run_record["error"] = f"{type(exc).__name__}: {exc}"
            print(f"  -> FAILED: {run_record['error']}", flush=True)

        run_record["wall_seconds"] = time.perf_counter() - run_start
        manifest["runs"].append(run_record)
        write_manifest()

    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_manifest()

    build_comparison_reports(artifact_root, selected_ids, all_candidates, scales, args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
