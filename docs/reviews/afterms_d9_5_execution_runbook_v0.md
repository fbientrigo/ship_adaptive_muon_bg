# D9-5 Execution Runbook

Branch: `experiment/d9-5-afterms-model-family-arena-v0`. Python:
`.venv\Scripts\python.exe` only. Device: RTX 2060 Laptop GPU, 6 GB VRAM,
float32, one CUDA process at a time, no AMP, `num_workers=0`.

CLI: `scripts\run_afterms_d9_5_model_family_arena.py`. Every fitting action
requires `--execute`; every subcommand supports `--dry-run` or is read-only
by construction (`status`, `summarize`).

## Gate status

- [x] Gate A -- data-scope audit, adapters, tests (`812 + 52 = 864` D9-5
      focused tests pass; see Test Result below for the full-suite number).
- [ ] Gate B -- GAUSS_DIAG / GAUSS_FULL fits on both tracks.
- [ ] Gate C -- GMM 3-seed fits on both tracks.
- [ ] Gate D -- NF_AC 3-seed training on both tracks (estimated 18-60 GPU
      hours serial; will not complete in one interactive session -- see
      Sec 20 estimate below).
- [ ] Gate E -- freeze-selection, evaluate-test, final report.

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

## Gate D -- NF_AC (3 seeds x 2 tracks, GPU, one process at a time)

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

From the D8/D9 producer's own per-epoch measurement (53.7 s/epoch on a single
~501k-row train shard, RTX 2060 Max-Q) scaled to the full 22-shard per-track
training scope (~11x more rows/epoch): **on the order of 10 minutes/epoch**.
With the frozen execution policy (`minimum_epochs=20`, `maximum_epochs=100`,
`early_stopping_patience=25`), a single seed's worst case is
`100 * ~10 min ≈ 16.7 hours`; six runs (3 seeds x 2 tracks), run strictly
serially (one CUDA process at a time, per the primary-device policy), worst
case is on the order of **18-60+ GPU-hours** depending on how early stopping
actually triggers. This will not complete inside one interactive session --
that is the expected path (Sec 20/21: "if the complete campaign cannot finish
in one session, finish the current atomic run, preserve status/checkpoints,
return exact resume commands"). Peak VRAM per run is the standardized
training tensor resident on device (~110 MB/track at float32) plus the
affine-coupling module and optimizer state -- comfortably under 6 GB.

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
