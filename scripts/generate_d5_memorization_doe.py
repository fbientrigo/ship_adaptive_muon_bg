"""Generate the versioned D5 memorization DOE definitions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ship_muon_bg.density_lab.doe import generate_blocked_maximin_lhs, write_doe


def _model(row):
    params = {key: row[key] for key in (
        "number_of_blocks", "hidden_width", "hidden_depth", "learning_rate",
        "batch_size", "max_log_scale", "activation", "mixing_mode",
    )}
    params.update({
        "memorization_mode": True, "weight_decay": 0.0, "dropout": 0.0,
        "data_augmentation": False, "input_noise_std": 0.0,
        "early_stopping": False, "max_epochs": 200,
        "checkpoint_interval": 10,
        "grad_clip_norm": 5.0,
    })
    return {"name": row["doe_id"], "family": "affine_coupling", "params": params}


def _write_campaign_configs(payload, output_dir):
    common = {
        "pdg_ids": [13], "feature_views": [{"view_id": "identity_cartesian_v0"}],
        "seeds": [1], "resources": {"device": "cpu"},
        "doe_seed": payload["doe_seed"],
        "tracking": {"mode": "local", "experiment_name": "d5_memorization_doe_v0"},
    }
    matrix = dict(common)
    matrix.update({
        "experiment_id": "d5_memorization_doe_v0",
        "description": (
            "D5 experimental matrix: 144 valid runs; no physics winner may be "
            "selected from one seed."
        ),
        "targets": [
            {"target_id": "D5", "variant": "rare_1e-3", "stage": "base_before_d4"},
            {"target_id": "D5", "variant": "rare_1e-3", "stage": "transformed"},
        ],
        "models": [_model(row) for row in payload["configs"]],
        "dataset": {"n_train": 65536, "n_validation": 8192, "n_test": 32768},
        "evaluation": {"rare_sample_count": 100000},
        "sampling_regimes": [
            {"regime": "iid_target"},
            {"regime": "stratified_unweighted_diagnostic", "sampling_rare_fraction": 0.5},
            {"regime": "stratified_self_normalized_provisional", "sampling_rare_fraction": 0.5},
        ],
    })
    d3_control = dict(common)
    d3_control.update({
        "experiment_id": "d3_memorization_control_v0",
        "description": (
            "D3 IID control matrix: 24 valid runs; D3 has no labelled rare "
            "component and is never paired with rare-aware sampling."
        ),
        "targets": [{"target_id": "D3"}],
        "models": [_model(row) for row in payload["configs"]],
        "dataset": {"n_train": 65536, "n_validation": 8192, "n_test": 32768},
        "evaluation": {"rare_sample_count": 100000},
        "sampling_regimes": [{"regime": "iid_target"}],
    })
    first_by_block = [next(row for row in payload["configs"] if row["block"] == block) for block in "ABC"]
    smoke = dict(common)
    smoke.update({
        "experiment_id": "d5_memorization_doe_smoke_v0",
        "description": "Bounded CPU wiring smoke: one config per block and one seed; never rank winners.",
        "targets": [{"target_id": "D5", "variant": "rare_1e-3", "stage": "transformed"}],
        "models": [_model(row) for row in first_by_block],
        "dataset": {"n_train": 512, "n_validation": 256, "n_test": 256},
        "evaluation": {"ess_sample_count": 512, "c2st_sample_count": 256, "rare_sample_count": 2000},
        "sampling_regimes": [
            {"regime": "iid_target"},
            {"regime": "stratified_unweighted_diagnostic", "sampling_rare_fraction": 0.5},
            {"regime": "stratified_self_normalized_provisional", "sampling_rare_fraction": 0.5},
        ],
    })
    paths = []
    for name, config in (
        ("d5_memorization_matrix_v0.json", matrix),
        ("d3_memorization_control_v0.json", d3_control),
        ("d5_memorization_smoke_v0.json", smoke),
    ):
        path = output_dir / name
        path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--doe-seed", type=int, default=20260717)
    parser.add_argument("--output-dir", type=Path, default=Path("configs/density_lab/doe_v0"))
    args = parser.parse_args()
    payload = generate_blocked_maximin_lhs(doe_seed=args.doe_seed)
    paths = write_doe(payload, args.output_dir)
    paths += _write_campaign_configs(payload, args.output_dir)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
