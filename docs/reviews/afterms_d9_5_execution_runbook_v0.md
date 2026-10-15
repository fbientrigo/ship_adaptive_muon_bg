# D9-5 Execution Runbook

Branch: `experiment/d9-5-afterms-model-family-arena-v0`. Python:
`.venv\Scripts\python.exe` only. Device: RTX 2060 Laptop GPU, 6 GB VRAM,
float32, one CUDA process at a time, no AMP, `num_workers=0`.

CLI: `scripts\run_afterms_d9_5_model_family_arena.py`. Every fitting action
requires `--execute`; every subcommand supports `--dry-run` or is read-only
by construction (`status`, `summarize`).

## Gate status

- [x] Gate A -- data-scope audit, adapters, tests (`812 + 52 = 864` D9-5
      tests pass; full suite 864 passed / 2 skipped / 0 failed).
- [x] Gate B -- GAUSS_DIAG / GAUSS_FULL fitted + validated on both tracks.
      Real results in `artifacts/afterms_d9_5_model_family_arena_v0/runs/`.
- [x] Gate C -- GMM 3 seeds (20260720/21/22) fitted + validated on both
      tracks; every seed converged (14-20 EM iterations) with all 4
      components occupied.
- [x] Gate D -- NF_AC seed training, run via the D9-5N unattended nightly
      campaign (`docs/reviews/afterms_d9_5_nightly_execution_v0.md`).
      **Closed as Phase 0 (nominal seed-stability baseline) with 5 of 6
      frozen runs at a genuine terminal state**; the 6th
      (`TRK_PDGM13_UW_ID` seed `20260722`) was never started -- the user
      explicitly directed the campaign to stop after seeing 5 runs' results,
      judging that sufficient for Phase 0. This is a deliberate early
      closure decision, not a technical failure; zero blocking incidents
      were recorded across the whole campaign. Total measured GPU time:
      ~91.8 GPU-hours summed over 384 completed epochs, ~94.4 hours of
      wall-clock campaign runtime (2026-07-25 07:12 UTC through the final
      `abort`). See `docs/reviews/afterms_d9_5_model_family_arena_v0.md`
      Sec 11 for the full closure record (per-run results table,
      hyperparameters, architecture table, invariant verification, and
      items explicitly deferred to Phase 1).
- [ ] Gate E -- freeze-selection, evaluate-test, final report. Blocked on
      Gate D (all four families per track must be fitted before
      freeze-selection). **Not started; the test split has not been
      opened.**

## Gate A -- orientation, data audit, adapters (compute-free)

```powershell
git branch --show-current
git status --short
git log --oneline --decorate -12

.venv\Scripts\python.exe scripts\run_afterms_d9_5_model_family_arena.py audit --dry-run
.venv\Scripts\python.exe scripts\run_afterms_d9_5_model_family_arena.py audit
.venv\Scripts\python.exe scripts\run_afterms_d9_5_model_family_arena.py plan --dry-run

.venv\Scripts\python.exe -m pytest -q tests/afterms/d9
.venv\Scripts\python.exe -m pytest -q tests/afterms/d9b
.venv\Scripts\python.exe -m pytest -q tests/afterms/d9_5
.venv\Scripts\python.exe -m pytest -q
```

`audit` (no `--dry-run`) writes
`artifacts\afterms_d9_5_model_family_arena_v0\data_scope\{data_scope_manifest.json,
data_scope_audit.md, ram_preflight.json}`.

## Gate B -- Gaussian controls (fast, CPU, deterministic)

```powershell
.venv\Scripts\python.exe scripts\run_afterms_d9_5_model_family_arena.py fit `
  --track-id TRK_PDG13_UW_ID --model-family GAUSS_DIAG --execute
.venv\Scripts\python.exe scripts\run_afterms_d9_5_model_family_arena.py fit `
  --track-id TRK_PDG13_UW_ID --model-family GAUSS_FULL --execute
.venv\Scripts\python.exe scripts\run_afterms_d9_5_model_family_arena.py fit `
  --track-id TRK_PDGM13_UW_ID --model-family GAUSS_DIAG --execute
.venv\Scripts\python.exe scripts\run_afterms_d9_5_model_family_arena.py fit `
  --track-id TRK_PDGM13_UW_ID --model-family GAUSS_FULL --execute

.venv\Scripts\python.exe scripts\run_afterms_d9_5_model_family_arena.py validate `
  --track-id TRK_PDG13_UW_ID --model-family GAUSS_DIAG --execute
# ... repeat validate for GAUSS_FULL and both models on TRK_PDGM13_UW_ID
```

## Gate C -- GMM (3 seeds x 2 tracks, CPU, sklearn EM)

```powershell
foreach ($track in @("TRK_PDG13_UW_ID","TRK_PDGM13_UW_ID")) {
  foreach ($seed in @(20260720,20260721,20260722)) {
    .venv\Scripts\python.exe scripts\run_afterms_d9_5_model_family_arena.py fit `
      --track-id $track --model-family GMM --seed $seed --execute
    .venv\Scripts\python.exe scripts\run_afterms_d9_5_model_family_arena.py validate `
      --track-id $track --model-family GMM --seed $seed --execute
  }
}
```

If a GMM fit ever needs a chunked/streaming EM implementation because the
declared full scope cannot be fit safely, STOP and report
`D9_5_BLOCKED_BY_GMM_DATA_SCALE` -- do not silently reduce the data. The Gate
A RAM preflight (`ram_preflight.json`) already confirmed this is not needed
for the current dataset.

### Gate B/C actual validation results (quick_validation_budget)

| track | model_config_id | seed | validation feature NLL | validation physical NLL |
|---|---|---|---|---|
| TRK_PDG13_UW_ID | GAUSS_DIAG_d05 | deterministic | 7.0991 | 9.0655 |
| TRK_PDG13_UW_ID | GAUSS_FULL_d05 | deterministic | 5.4348 | 7.4011 |
| TRK_PDG13_UW_ID | GMM_k04_covFULL_d05 | 20260720 | 2.7536 | 4.7199 |
| TRK_PDG13_UW_ID | GMM_k04_covFULL_d05 | 20260721 | 2.9739 | 4.9402 |
| TRK_PDG13_UW_ID | GMM_k04_covFULL_d05 | 20260722 | 3.1177 | 5.0840 |
| TRK_PDGM13_UW_ID | GAUSS_DIAG_d05 | deterministic | 7.0929 | 9.0623 |
| TRK_PDGM13_UW_ID | GAUSS_FULL_d05 | deterministic | 5.4240 | 7.3934 |
| TRK_PDGM13_UW_ID | GMM_k04_covFULL_d05 | 20260720 | 2.7494 | 4.7188 |
| TRK_PDGM13_UW_ID | GMM_k04_covFULL_d05 | 20260721 | 2.9654 | 4.9348 |
| TRK_PDGM13_UW_ID | GMM_k04_covFULL_d05 | 20260722 | 3.1099 | 5.0793 |

All finite-log-prob fractions are 1.0 (no non-finite log-densities in any
fit). Within each track, GMM's best seed (20260720) achieves markedly lower
validation physical NLL than either Gaussian control -- unsurprising given
its much larger parameter count, and not yet comparable to NF_AC since Gate D
has not run. **This is a Gate B/C snapshot, not a validation-only selection**
-- `freeze-selection` has not been run (it requires all four families,
including NF_AC) and no test-split evaluation has occurred.

## Gate D -- NF_AC (3 seeds x 2 tracks, GPU, one process at a time)

**Status: CLOSED as Phase 0 (nominal seed-stability baseline), 5 of 6 runs
executed to a genuine terminal state.** Run via the D9-5N unattended
campaign (`start-campaign`), not the manual per-run commands below (those
remain valid for a single manual run/resume, e.g. after a `FAILED_BLOCKED`
incident).

### Gate D actual results (Phase 0 closure)

| track | model_config_id | seed | terminal epoch | reason | best epoch | best val feature NLL | final val feature NLL |
|---|---|---|---|---|---|---|---|
| TRK_PDG13_UW_ID | NF_AC_b08_w128_d02 | 20260720 | 100 | max epochs | 78 | 1.3032 | 1.3203 |
| TRK_PDG13_UW_ID | NF_AC_b08_w128_d02 | 20260721 | 75 | early-stopped | 50 | 1.3022 | 1.3223 |
| TRK_PDG13_UW_ID | NF_AC_b08_w128_d02 | 20260722 | 54 | early-stopped | 17 | 1.3221 | 1.3351 |
| TRK_PDGM13_UW_ID | NF_AC_b06_w096_d02 | 20260720 | 100 | max epochs | 76 | 1.3123 | 1.3356 |
| TRK_PDGM13_UW_ID | NF_AC_b06_w096_d02 | 20260721 | 55 (last checkpoint) | **stopped by explicit user request** (not natural) | 38 | 1.3136 | n/a (mid-training) |
| TRK_PDGM13_UW_ID | NF_AC_b06_w096_d02 | 20260722 | -- | **not started** | -- | -- | -- |

Feature-space NLL only (as recorded directly in each run's
`training_history.json`/checkpoint); the physical-space NLL Jacobian
addition (Sec 5 of the arena doc) has not been computed for NF_AC because
`validate` was not run for these seeds -- that is deferred, not inferred
here. See `docs/reviews/afterms_d9_5_model_family_arena_v0.md` Sec 11 for
the full closure record: hyperparameters, invariant verification (identical
`execution_policy_hash`/`evaluation_policy_hash`/per-track dataset hash
across every run), incident count (zero), and items explicitly deferred to
Phase 1.

Total measured compute: 384 completed epochs, ~91.8 GPU-hours (sum of
per-epoch wall time across all 5 executed runs), ~94.4 hours of campaign
wall-clock (2026-07-25 07:12 UTC to the final `abort`), across 12
auto-chained 8-hour blocks with zero blocking incidents.

### Manual per-run commands (for a single run/resume, not the closed campaign)

```powershell
foreach ($track in @("TRK_PDG13_UW_ID","TRK_PDGM13_UW_ID")) {
  foreach ($seed in @(20260720,20260721,20260722)) {
    .venv\Scripts\python.exe scripts\run_afterms_d9_5_model_family_arena.py fit `
      --track-id $track --model-family NF_AC --seed $seed --device cuda --execute
  }
}
```

**Resume** an interrupted seed (same command, add `--resume`):

```powershell
.venv\Scripts\python.exe scripts\run_afterms_d9_5_model_family_arena.py fit `
  --track-id TRK_PDG13_UW_ID --model-family NF_AC --seed 20260720 --device cuda --execute --resume
```

Then validate each completed seed:

```powershell
.venv\Scripts\python.exe scripts\run_afterms_d9_5_model_family_arena.py validate `
  --track-id TRK_PDG13_UW_ID --model-family NF_AC --seed 20260720 --execute
```

### Compute estimate (recorded before starting Gate D, Sec 20)

**Pre-execution estimate** (from the D8/D9 producer's own per-epoch
measurement, 53.7 s/epoch on a single ~501k-row train shard, RTX 2060 Max-Q,
scaled ~11x to the full 22-shard per-track training scope): on the order of
10 minutes/epoch.

**Measured** (single real epoch, `TRK_PDG13_UW_ID`, `NF_AC_b08_w128_d02`,
full 5,484,126-row training scope, RTX 2060 Laptop GPU, throwaway probe run
then discarded): **1011.8 s/epoch (~16.9 minutes/epoch)** -- markedly higher
than the pre-execution scale-up, confirming per-epoch cost must be measured
directly rather than assumed from a single-shard extrapolation.

With the frozen execution policy (`minimum_epochs=20`, `maximum_epochs=100`,
`early_stopping_patience=25`) and the measured ~16.9 min/epoch:

- minimum bound per seed (early stop right at `minimum_epochs`):
  `20 * 16.9 min ≈ 5.6 hours`
- worst case per seed (no early stop, full `maximum_epochs`):
  `100 * 16.9 min ≈ 28.1 hours`
- six runs (3 seeds x 2 tracks), run strictly serially (one CUDA process at a
  time, per the primary-device policy): **~34 to ~169 GPU-hours total**.

This will not complete inside one interactive session -- that is the expected
path (Sec 20/21: "if the complete campaign cannot finish in one session,
finish the current atomic run, preserve status/checkpoints, return exact
resume commands"). Peak VRAM per run is the standardized training tensor
resident on device (~110 MB/track at float32) plus the affine-coupling module
and optimizer state -- comfortably under 6 GB; the bottleneck is wall time,
not memory.

## Gate E -- freeze selection, frozen test evaluation, final report

```powershell
.venv\Scripts\python.exe scripts\run_afterms_d9_5_model_family_arena.py freeze-selection --execute

foreach ($track in @("TRK_PDG13_UW_ID","TRK_PDGM13_UW_ID")) {
  foreach ($family in @("NF_AC","GAUSS_DIAG","GAUSS_FULL","GMM")) {
    # NF_AC/GMM: add --seed 20260720 / 20260721 / 20260722 per run
    .venv\Scripts\python.exe scripts\run_afterms_d9_5_model_family_arena.py evaluate-test `
      --track-id $track --model-family $family --budget final_candidate_budget --execute
  }
}

.venv\Scripts\python.exe scripts\run_afterms_d9_5_model_family_arena.py summarize `
  --budget final_candidate_budget --execute
```

`evaluate-test` refuses with a non-zero exit and
`ERROR: no frozen validation-only family selection manifest ...` if
`freeze-selection --execute` has not already run for this artifact root.

## Safe resume commands (general)

- Re-run any `fit`/`validate`/`evaluate-test` command exactly as issued; every
  write is atomic (temp file + `os.replace`), so a killed process never
  leaves a corrupt bundle to resume from.
- NF_AC only: add `--resume` to continue an interrupted seed from its
  `last_resumable_checkpoint.pt`; add `--extend-max-epochs <N>` only if
  deliberately revising `maximum_epochs` upward (never silently).
- `status` (read-only) reports what is actually on disk per (track, family[,
  seed]) without touching any shard or fitting anything:

```powershell
.venv\Scripts\python.exe scripts\run_afterms_d9_5_model_family_arena.py status
```
