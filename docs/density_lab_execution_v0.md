# Controlled Density Lab — Execution Guide (v0)

The controlled density lab fits and evaluates density models against the exact
controlled targets D0-D5. It is a numerical benchmarking harness, **not** a
SHiP physics or background-rate tool.

## Install

```bash
pip install -e .[dev]                 # NumPy-only core + pytest
pip install -e .[dev,flow,lab]        # + torch (flow) + scikit-learn/matplotlib
pip install -e .[dev,flow,lab,tracking]  # + optional MLflow adapter
```

The core import path stays NumPy-only: importing `Nflow`, `ship_muon_bg`, or
`ship_muon_bg.density_lab` never imports torch / scikit-learn / matplotlib /
mlflow. Heavy dependencies are imported lazily where used.

## Test

```bash
python -m pytest -q            # runs core; flow/lab tests auto-skip if deps absent
```

Markers: `flow` (torch), `lab` (scikit-learn/matplotlib), `slow` (small training
runs; not performance benchmarks).

## Architecture

```
exact targets (D0-D5)            benchmarks/controlled_targets.py, target_transforms.py, target_regions.py
  -> datasets (matched rows)     density_lab/datasets.py
  -> FittedFeaturePipeline       density_lab/feature_pipeline.py   (view + train-only 5D standardization)
  -> DensityEstimator            Nflow/registry.py -> baselines/{gaussian,gmm}.py, torch_models/affine_coupling.py
  -> evaluator (physical space)  density_lab/evaluator.py + metrics.py
  -> artifacts + report          density_lab/artifacts.py, reporting.py, tracking.py
```

Density accounting is exact:

```
physical_log_prob = normalized_model_log_prob
                  + normalization forward log-Jacobian   (sum(-log std))
                  + FeatureView forward log-Jacobian
```

## Run a campaign

```bash
python scripts/run_density_lab.py --config configs/density_lab/smoke_v0.json
python scripts/run_density_lab.py --config <cfg> --targets D3 D5 --models affine_small --seeds 11
python scripts/run_density_lab.py --config <cfg> --force --device cpu
```

Runs execute independently; a failed run records its status and traceback and
the campaign continues. Completed identical run hashes are skipped unless
`--force`. `run_id` is derived from the canonical config hash, so a changed
config produces a new run directory.

Suites: `smoke_v0` (fast CPU correctness), `reference_v0` (first presentable
result, five seeds), `capacity_v0` (an explicit affine-flow capacity ladder
vs Gaussian baselines), `capacity_pilot_d5_cpu_v0` (a provisional engineering
CPU-only D5 rare-mode capacity pilot; see below).

## Artifacts

Each run writes `artifacts/density_lab/<experiment_id>/<run_id>/`:
`experiment_config.json`, `environment.json`, `dataset_manifest.json`,
`feature_pipeline_manifest.json`, `model_manifest.json`, `fit_result.json`,
`metrics.json`, `training_history.jsonl`, `samples.npz`,
`model_parameters.npz` or `checkpoint/`, and `run_status.json`. The
`artifacts/` directory is gitignored.

`metrics.json` carries the scientific gate block under `scientific_gates`
(`scientific_status`, `scientific_failure_reasons`, `gate_results`,
`gate_config_hash`). `run_status.json` records both `technical_status` and
`scientific_status` (see below).

## Build the report

```bash
python scripts/build_density_report.py --campaign-dir artifacts/density_lab/smoke_v0
```

Reads artifacts only (never retrains). Writes `report/`:
`benchmark_summary.{json,csv,md}`, `scientific_gate_summary.{json,md}`,
`limitations.md`, `quality_by_target.png`, `rare_mode_recovery.png`,
`feature_view_comparison.png`, `capacity_frontier.png`. The benchmark CSV has
separate `technical_status` and `scientific_status` columns.

## Runtime micro-benchmark

```bash
python scripts/benchmark_density_runtime.py --family affine_coupling --device cpu
```

Warms up, synchronizes CUDA when relevant, reports median/dispersion, and
separates fit/log_prob/sample. It sets no wall-time thresholds; CPU may be
faster than GPU for these small 5D models and that is not an error.

## Gates and thresholds

Threshold manifests label each value as `mathematical_invariant`,
`catastrophic_guard` (e.g. ESS/N < 0.01), `provisional_engineering_gate`, or
`preregistered_scientific_gate`. Provisional gates are not final physics
criteria. Hard failures (non-finite loss/density, invalid Jacobian, round-trip
failure, missing artifact, invalid D5 calibration) stop the affected run and
are recorded.

## Technical status vs scientific status

The lab reports two **independent** statuses per run.

- **Technical status** (`technical_status`, mirrored by the historical `status`
  field): did the run execute without an execution error? Values: `completed`
  or `failed`. A `failed` run raised an exception or its fit did not converge;
  its traceback is preserved.
- **Scientific status** (`scientific_status`): is the fitted model
  scientifically acceptable, given its metric bundle? Values: `pass`, `fail`,
  `catastrophic`, `inconclusive`.

These are deliberately orthogonal. A run can be:

```
technical_status = completed     (the code ran fine)
scientific_status = catastrophic (the model collapsed the D5 rare mode)
```

A model that fails a scientific gate is **never** relabeled a technical failure.
Averages in the report exclude scientifically catastrophic runs from "clean"
means, but keep them counted and individually visible
(`scientific_gate_summary.{json,md}`), so a collapse is never smoothed away.

### The scientific gate layer (`density_lab/gates.py`)

`evaluate_scientific_gates(metrics, target_id=..., gate_spec=...)` is a pure,
model-independent classifier. It **consumes** the existing metric bundle and
never re-estimates ESS/KL/C2ST. It imports only the standard library — no
numpy, torch, scikit-learn, matplotlib, or mlflow. Each gate declares one of the
four threshold classes above so a provisional engineering reference can never be
confused with a preregistered physics criterion.

Default classifications:

| gate | threshold class | status on trigger |
| --- | --- | --- |
| non-finite density/loss | `mathematical_invariant` | `catastrophic` |
| ESS/N below catastrophic floor | `catastrophic_guard` | `catastrophic` |
| D5 zero generated rare-region samples | `catastrophic_guard` | `catastrophic` |
| mandatory metric missing/malformed | (any) | `inconclusive` |
| rare-region mass ratio | `provisional_engineering_gate` | reported only |
| forward KL, C2ST | `preregistered_scientific_gate` | reported only |

Rare-region mass ratio, forward KL, and C2ST are **reported**, never used as a
sole pass/fail selector by default. Status aggregation follows a documented
precedence: `catastrophic > fail > inconclusive > pass`. `catastrophic`
dominates so a hard failure is reported even when an unrelated metric is
missing; `inconclusive` means a mandatory metric was absent or malformed, so a
responsible `pass`/`fail` cannot be assigned.

ESS boundary semantics: the catastrophic guard fires iff
`ess_over_n < catastrophic_ess_threshold` (strict less-than). A value exactly at
the threshold passes the guard, matching `metrics.importance_ess`.

### One source of truth for the ESS threshold

The ESS catastrophic floor lives in `EvaluationSpec.catastrophic_ess_threshold`.
`ScientificGateSpec.catastrophic_ess_threshold` defaults to `null` meaning
"inherit that value". Setting it to a value that contradicts the evaluation
config is a `ConfigError`. The gate spec is part of the run identity: changing
any gate field changes `run_id`, so a re-run under different gates gets its own
artifact directory and resume/skip stays deterministic.

## D5 CPU capacity pilot (provisional engineering)

`configs/density_lab/capacity_pilot_d5_cpu_v0.json` is a **provisional
engineering CPU pilot** — not the final capacity frontier and not a physics
result. It probes, on CPU with a single seed, whether affine-coupling capacity
removes the D5 `rare_1e-3` zero-rare-mode failure and improves ESS/N relative to
a `gmm_k4` baseline:

```bash
python scripts/run_density_lab.py --config configs/density_lab/capacity_pilot_d5_cpu_v0.json
python scripts/build_density_report.py --campaign-dir artifacts/density_lab/capacity_pilot_d5_cpu_v0
```

Models run independently, so partial progress is preserved and the campaign is
resumable. The pilot answers engineering questions only (does affine capacity
recover the rare mode? does ESS/N improve? does `gmm_k4` remain competitive?);
it is not evidence about D7, FairShip, a proxy density, or any SHiP physics
rate.
