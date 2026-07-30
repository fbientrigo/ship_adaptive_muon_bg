# D7 (empirical after-MS data) — full local replay runbook (v0)

How to rerun the D7 (empirical after-MS, `docs/contracts/density_problem_contract_v0.md`
section 7) workflow on the complete local `muonsFullMC_afterMS.pkl` dataset,
after it has been verified on the small committed repository sample
(`data/samples/muonsFullMC_afterMS_sample.npz`, 40,000 rows).

**Status: `FULL_DATA_REPLAY_PATH_IMPLEMENTED_AND_FIXTURE_VERIFIED`.** Every
command, code path, test, and artifact schema below was exercised against the
repository fixture (`tests/test_empirical_campaign.py`,
`tests/test_build_dataset_report_cli.py`). The complete local dataset has
**not** been run through this workflow. Full-scale runtime, memory use, tail
fidelity, and scientific results are unverified until a local replay actually
completes -- the fixture proves code-path and artifact compatibility, not
full-data scientific performance. The same code is intended to run locally
without any implementation change: only `--dataset` / `dataset_path` and
bounded operational settings (`--max-rows`, `--artifact-root`, seed, batch
size) differ (`docs/contracts/rare_aware_minibatch_estimators_v0.md`'s
estimator mathematics and gate behavior are unmodified and out of scope for
this document; see section "What this replay does not change" below).

## Prerequisites

**Environment.** `pip install -e .[dev,flow,lab]` (NumPy + pytest + torch +
scikit-learn/matplotlib). D7 training uses the `affine_coupling` family
(`Nflow.torch_models.affine_coupling`, requires `.[flow]`) and its C2ST
metric (requires `.[lab]`). Optional MLflow tracking (`.[tracking]`) is not
required.

**Supported input format.** A gzip-compressed pickle whose payload is a
single `numpy.ndarray`, or an NPZ with the array under key `"muons"`
(`ship_muon_bg.data_contracts.load_muon_array` dispatches by file extension
-- the same dispatcher the repository fixture uses). See
`data/samples/README.md` for how to obtain
`muonsFullMC_afterMS.pkl` (GitHub release asset,
[`v1.0.0-fullmc-afterms`](https://github.com/fbientrigo/NFlow/releases/tag/v1.0.0-fullmc-afterms)).

**Required columns (fixed order).** `[px, py, pz, x, y, z, id, w]`, 8
columns, `float64`-coercible. `id` is a PDG code (`13` = mu-, `-13` = mu+);
`w` is a dimensionless event weight (see "Physical weight handling" below).

**Units and coordinate convention.** Momenta in GeV/c; positions in metres,
`z = 0` at the entrance of the Muon Shield -- the after-Muon-Shield sample's
`z` is already shifted to that convention (`z ≈ 28.9 m`, the shield exit /
scoring plane). No event-level four-momentum conservation is assumed or
checked; this is a post-shield muon *state* contract, not a physics
simulation contract.

**Disk and memory checks.** The full file is ~565 MB on disk
(564,941,616 B, 13,779,080 rows, per `data/samples/README.md`). Loaded as a
raw `float64` array it is:

```
13,779,080 rows x 8 columns x 8 bytes = 881,861,120 bytes = 841 MiB
```

per PDG track after filtering this roughly halves (both tracks are present
in the same file; a single-PDG in-memory copy is smaller, not larger). Budget
headroom for: the loaded raw array, one PDG-filtered copy, the 5-column
normalized training/validation copies, and torch's own working memory during
training. There is no streaming/chunked training path (see "Scale-safe
loading" below) -- the full per-PDG raw array, and its normalized copies, are
held in memory for the duration of a run. Compute your own exact estimate for
whatever row count you actually use:

```bash
python scripts/build_dataset_report.py --dataset <path> --validate-only \
    --allow-zero-weight --seed 1234 --max-rows <N> | python -c \
    "import json,sys; print(json.load(sys.stdin)['memory_footprint_estimate'])"
```

**Output artifact location.** `artifacts/density_lab/<experiment_id>/<run_id>/`
under `--artifact-root` (default: `artifacts/density_lab/` under the repo,
gitignored). Point `--artifact-root` at an external directory for a full
local run so nothing lands under the repository by default.

## Scale-safe loading (read this before a full run)

Stated honestly, not aspirationally: **the current loaders fully materialize
the file into memory.** `load_muon_pkl` unpickles the whole gzip payload;
`load_muon_npz` reads the whole compressed array (NPZ is a zip archive, so it
cannot be memory-mapped the way an uncompressed `.npy` can -- `mmap_mode` is
not available for either committed format). There is **no streaming or
chunked training path** in this repository; a full local run holds the
PDG-filtered raw array, its normalized copies, and the model's own working
memory simultaneously (see the memory-footprint estimate above and in Step
1's report). Do not expect this workflow to handle a dataset that does not
fit in your machine's RAM.

What *is* implemented and does reduce peak memory, in the order applied
(`density_lab.empirical.build_empirical_dataset`, matching `ROW_LIMIT_ORDER`
in that module):

1. **PDG filtering before the row-budget cap** (`data_contracts.filter_by_pdg`):
   only one PDG track's rows are kept before anything else touches them, so a
   `--max-rows` cap or the split never has to consider the other track's rows.
2. **Deterministic bounded sampling** (`--max-rows`, via
   `data_contracts.representative_subset`): caps the row count actually
   copied into train/validation/test, at your choice, after PDG filtering.
3. **Explicit `float64` dtype handling** throughout (`data_contracts.schema`,
   `hashing.dataset_hash`): no silent dtype promotion/demotion anywhere in
   the pipeline that could hide a memory blow-up.

What is **not** implemented: column projection at load time (all 8 columns
are always loaded, even though training only uses 5), true chunked/streaming
ingestion, and memory-mapped reads. If the full file does not fit in memory
as-is, the only currently-supported mitigation is `--max-rows` (a smaller,
still-representative subsample) -- there is no partial-file or
lower-memory-footprint code path to fall back to, and this document will not
claim otherwise.

## Step 1 — validate the local dataset (never trains)

```bash
python scripts/build_dataset_report.py \
    --dataset "$SHIP_MUON_BG_LOCAL_DATA/muonsFullMC_afterMS.pkl" \
    --validate-only --allow-zero-weight --seed 1234 \
    --output artifacts/dataset_validation/afterms_full_report.json
```

(`$SHIP_MUON_BG_LOCAL_DATA` is any environment variable you set to your local
data directory -- nothing here is hardcoded; substitute the literal path if
you prefer.)

**Expected output:** a JSON report with `n_rows` matching the file's known
row count, `expected_pdg_counts` for `13`/`-13` both non-zero,
`validation` with every check `passed: true`, `weight_column_status`, and
`memory_footprint_estimate`. Exit code `0` on a fully valid dataset, `1` if
any contract check fails (the report still prints, listing which check and
why).

**Blockers this step catches, before any training time is spent:** schema
violations, non-finite values, non-positive weights (unless
`--allow-zero-weight`), non-integer `id`, momentum/position out of
units-sanity bounds, and (via `--campaign-config`, optional) insufficient
rows for your intended split. It does *not* run the exact-duplicate-row
check at full scale (13.8M rows exceeds the bounded default limit --
`row_budget`/`duplicate_rows.checked: false` with an explanation; rerun with
`--max-rows` if you need that check).

## Step 2 — run a small local smoke test (bounded rows)

Confirms the local file and environment without starting the full campaign:

```bash
python scripts/run_empirical_campaign.py \
    --config configs/density_lab/empirical/d7_fixture_smoke_v0.json \
    --dataset "$SHIP_MUON_BG_LOCAL_DATA/muonsFullMC_afterMS.pkl" \
    --artifact-root /path/to/local/artifacts/d7_smoke
```

This reuses the fixture *smoke* config (small model, `max_rows: 2000`, 3
epochs) with only `dataset_path` overridden -- the same bounded-wiring
budget as the fixture smoke test, just pointed at your local file. Expect
this to finish in well under a minute per PDG track on CPU.

## Step 3 — run one PDG track

```bash
python scripts/run_empirical_campaign.py \
    --config configs/density_lab/empirical/d7_local_full_v0.json \
    --dataset "$SHIP_MUON_BG_LOCAL_DATA/muonsFullMC_afterMS.pkl" \
    --pdg-ids 13 \
    --artifact-root /path/to/local/artifacts/d7_full
```

`d7_local_full_v0.json` is the layer-3/4 template (`max_rows: null`, i.e. no
cap -- every available row for that PDG track): edit its `model.params`
(`max_epochs`, `batch_size`, architecture) to match your intended budget
before running, or override via a small copy of the file (config
inheritance: keep the dataset/evaluation fields, change only what your run
needs).

## Step 4 — run or schedule both PDG tracks

```bash
python scripts/run_empirical_campaign.py \
    --config configs/density_lab/empirical/d7_local_full_v0.json \
    --dataset "$SHIP_MUON_BG_LOCAL_DATA/muonsFullMC_afterMS.pkl" \
    --artifact-root /path/to/local/artifacts/d7_full
```

(omitting `--pdg-ids` runs every PDG id in the config, i.e. both `13` and
`-13`). The two tracks are fully independent -- separate datasets, separate
models, separate artifacts (`Nflow` never conditions on charge) -- so they
may run **serially** (the command above, simplest) or **independently** in
two separate processes/invocations with `--pdg-ids 13` and `--pdg-ids -13`
respectively, e.g. on two machines or two terminal sessions. A failure on one
track never aborts the other either way.

**Resources.** No GPU is assumed. `"device": "cpu"` in the committed configs
is explicit, not a default guess; pass `--device cuda` (or edit the config)
if you have one available and want to use it -- `device` only changes where
tensors live, not any estimator or metric behavior. There is no
worker/dataloader-count setting to tune: training reads one in-memory NumPy
array per partition (no `DataLoader`, no multiprocessing workers).

## Step 5 — resume after interruption

Every run is content-addressed: `run_id` is derived from the canonical
config hash (dataset hash + PDG id + feature view + model + seed +
evaluation + gates), not a timestamp. Rerunning the **same** command after an
interruption:

```bash
python scripts/run_empirical_campaign.py --config <same config> \
    --dataset <same path> --artifact-root <same root>
```

skips any run whose `run_status.json` already records `status: completed`
with a matching config hash (`status: "skipped_completed"` in the printed
summary), and restarts (from scratch -- there is no mid-epoch checkpoint
resume) any run that is missing or was left `failed`. Force a clean rerun of
an already-completed run with `--force`.

## Step 6 — verify completion

For each `<run_id>` under `<artifact_root>/<experiment_id>/`:

- `run_status.json` -- `technical_status: "completed"`, a `scientific_status`
  (expect `"inconclusive"`: D7 has no closed-form target density, so the
  finiteness/ESS gates that require `forward_kl`/`importance_ess` cannot
  reach `"pass"` -- this is the documented, expected outcome, not a defect;
  see "What `scientific_status` means for D7" below), and a `hashes` block.
- `dataset_manifest.json` -- `source_file_dataset_hash` (content hash of your
  full local file), `pdg_counts_in_source`, per-partition row counts and
  hashes, `validation_no_leakage: true`, `physical_weight_status`,
  `sampling_correction_weight_status`.
- `model_manifest.json` -- `loss_normalization: "sum_weights"` (D7 trains
  with the legacy unweighted-IID path; no rare-aware arm applies without
  component labels), `checkpoint_hash`.
- `checkpoint/` -- the saved model state (`checkpoint_hash.txt`,
  `model_config.json`, `state_dict.pt`), reloadable exactly like any other
  `AffineCouplingFlow` checkpoint in this repository.
- `metrics.json` -- `exact_target_density_available: false`,
  `forward_kl: null`, `importance_ess: null` (explicit, not merely absent),
  finite `held_out.held_out_nll`, `c2st`, `tail_quantile_errors`,
  `duplicates`, `support_violation`, and `scientific_gates.gate_results`
  (including a `report`-only `estimator_evidence_scope` entry).
- `training_history.jsonl` -- one JSON record per epoch (loss, gradient
  norm, `state_dict_hash`).
- `feature_pipeline_manifest.json` -- the fitted 5-feature standardization
  (mean/std, fit on train rows only) and its Jacobian accounting.
- `<experiment_id>/campaign_summary.json` -- `n_runs` / `n_completed` /
  `n_failed` / `n_skipped` across both PDG tracks.

## Step 7 — compare with the repository-fixture experiment

**Structurally identical (must match):** every key listed in Step 6 is
present with the same shape/type; `estimator_family`,
`sampling_regime: "iid_target"`, `fit_claim`, `scientific_scope`,
`physical_weight_status`, `sampling_correction_weight_status`,
`weights_affect_selection: false`; `forward_kl`/`importance_ess` are `null`
in both; the gate id list is identical.

**Expected to differ, because of scale and sample composition (not a bug):**
`n_rows`/`n_train`/`n_val`/`n_test` (the fixture is a 40,000-row uniform +
range-anchor subsample; the full file is ~13.8M rows and its own natural
distribution); `source_file_dataset_hash` and every `raw_dataset_hash` (the
files differ); `held_out.held_out_nll`, `c2st_accuracy`,
`tail_quantile_errors`, `duplicates`, and every other data-dependent metric
value; wall time; `scientific_status` *could* differ from `"inconclusive"`
only if a future gate revision adds a D7-appropriate pass criterion -- as
shipped, both should read `"inconclusive"` for the same structural reason.

## What `scientific_status` means for D7

`"inconclusive"` here means the existing finiteness/ESS gates cannot certify
`"pass"` without `forward_kl`/`importance_ess`, which D7 legitimately never
computes (no closed-form target density -- `docs/contracts/
density_problem_contract_v0.md` section 7). It does **not** mean the run
failed, and it must never be read or reported as a scientific negative: a
technical failure (bad schema, OOM, crash) is `technical_status: "failed"`,
which is a wholly separate field. Judge a completed D7 run instead by the
metrics section 9.2 of that contract calls mandatory: held-out NLL, C2ST,
support/duplicate diagnostics, per-charge (PDG) consistency, and seed-to-seed
variation -- run the same config at a second seed and compare.

## What this replay does not change

No estimator mathematics, gate threshold, gate ID, gate schema version, or
rare-aware minibatch contract (`docs/contracts/
rare_aware_minibatch_estimators_v0.md`) is touched by this workflow. D7 has
no component labels, so no rare-aware arm (B/C/D) applies to it -- D7
training is exactly arm A (the unbiased IID baseline) run on real, unlabeled
rows; the fixed-composition Horvitz-Thompson estimator (arm C) still
requires unit physical weights (assumption A6) and still rejects the real,
non-unit afterMS `w` column exactly as it rejects any non-unit weight array
(`tests/test_empirical_campaign.py::test_ht_arm_c_rejects_real_non_unit_physical_weights`).

## Failure recovery

| Symptom | Cause | Recovery |
| --- | --- | --- |
| `EmpiricalDataError: ... failed load/validate` | Invalid schema (wrong shape/columns, non-finite values, non-integer `id`, non-positive weight without `--allow-zero-weight`, units-sanity bound exceeded) | Run Step 1 (`--validate-only`) first; it reports exactly which check failed and why. Fix the source file or pass `--allow-zero-weight` if zero-weight rows are expected (one such row is expected per the known full afterMS file). |
| Reported `source_file_dataset_hash` differs from a previously recorded one for "the same" file | The file's *content* changed (re-download, different release, accidental edit) -- the hash is over file content, not the path | Confirm you have the intended release asset (`data/samples/README.md` records the known-good `source_sha256`); if the file legitimately changed, this is a **new** dataset identity and a new `run_id` is correct, not an error to suppress. |
| `EmpiricalDataError: only N row(s) available for pdg_id=... after filtering/capping; need at least 3` (or a `make_three_way_split` `ValueError`) | Insufficient rows for the requested PDG track after `--max-rows`/split fractions | Raise `--max-rows` (or omit it), or check the PDG actually has enough rows in the source file (`expected_pdg_counts` in the Step 1 report). |
| No rows / dataset build fails for one PDG track only | Missing PDG track (a file legitimately containing only one charge, or a corrupted subset) | Check `expected_pdg_counts` from Step 1; run only the present track (`--pdg-ids 13` or `--pdg-ids -13`); do not report the missing track as a technical failure of the *other* track -- they are independent. |
| `ValueError: ... requires unit physical row weights (assumption A6)` | You (or a future config) tried to pass the real, non-unit afterMS `w` column into `plan_fixed_composition_batches` / the fixed-composition Horvitz-Thompson estimator | By design: D7 has no component labels, so no rare-aware arm is offered by `run_empirical_campaign`; this error only fires if you call the rare-aware estimator machinery directly and by hand with real weights. It is correct behavior (A6), not a bug -- do not weaken it. |
| Process killed / `MemoryError` | Out of memory at full scale | Reduce `--max-rows`, reduce `batch_size`, run one PDG track at a time (never both processes concurrently on the same machine unless you've budgeted memory for both), or use the memory-footprint command in "Prerequisites" to size your run before starting it. There is no streaming path (see "Scale-safe loading" below) to fall back to. |
| Interrupted run (Ctrl-C, crash, power loss) mid-training | No mid-epoch checkpoint resume exists | Rerun the identical command (Step 5): a run that never reached `status: completed` restarts from scratch, it does not resume from a partial epoch. This is a real limitation, stated here rather than implied to work. |
| A `checkpoint/` directory exists but a later run in the same `run_id` directory reports a hash mismatch | `checkpoint_hash` verifies the functional fingerprint (architecture + init seed + permutations + dtype) matches the saved state; a config edit that changes any of those fields under an unchanged `run_id` would be a bug, not an expected outcome | Do not hand-edit files under `artifacts/`; if you changed the model config, that produces (correctly) a new `run_id` -- let the new run write its own directory rather than reusing the old one. |
| A stale-looking `run_status.json` (`status: "running"` never observed, or a run directory with partial files) | This implementation writes artifacts atomically at the *end* of a run (`ArtifactStore.write_run`); a partial directory means the process died before that write, so the run never reached `completed` | Confirm with `status.get("technical_status")`; if not `"completed"`, treat it as unwritten and just rerun (Step 5) -- there is no separate lock file to remove. |
| Changed preprocessing or dataset identity between two "resumed" runs | The `run_id`/config hash already encodes the feature view, model config, seed, and (via the dataset manifest) row counts -- but *not* the dataset file's content hash directly in the `run_id` label (only in `dataset_manifest.json`/`hashes.source_file_dataset_hash`) | Always check `hashes.source_file_dataset_hash` in `run_status.json` before treating two runs as comparable; a `run_id` collision with a different `source_file_dataset_hash` means two different files produced the same row counts by coincidence -- inspect before trusting either as canonical. |

Every failure above is a **technical** outcome (`technical_status: "failed"`
or a raised exception before any artifact is written) and must never be
reported as a scientific negative -- a run that never completed has no
scientific status to report at all.
