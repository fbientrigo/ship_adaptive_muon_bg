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
vs Gaussian baselines).

## Artifacts

Each run writes `artifacts/density_lab/<experiment_id>/<run_id>/`:
`experiment_config.json`, `environment.json`, `dataset_manifest.json`,
`feature_pipeline_manifest.json`, `model_manifest.json`, `fit_result.json`,
`metrics.json`, `training_history.jsonl`, `samples.npz`,
`model_parameters.npz` or `checkpoint/`, and `run_status.json`. The
`artifacts/` directory is gitignored.

## Build the report

```bash
python scripts/build_density_report.py --campaign-dir artifacts/density_lab/smoke_v0
```

Reads artifacts only (never retrains). Writes `report/`:
`benchmark_summary.{json,csv,md}`, `limitations.md`, `quality_by_target.png`,
`rare_mode_recovery.png`, `feature_view_comparison.png`, `capacity_frontier.png`.

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
