# After-MS Nightly Smoke — Manual Execution Runbook v0

Companion to `afterms_evaluation_audit_v0.md`. All commands below were derived
from `python scripts/run_afterms_nightly_queue.py --help` and
`python scripts/watch_afterms_nightly_queue.py --help`, run against the
current tree (commit `5e37c92` on `experiment/d7-afterms-sharded-smoke-v0`) —
no flag here is invented. Every command block distinguishes commands that
exist today from commands that do not yet exist (arena/KS/energy/C2ST tooling
is **not implemented** anywhere in this repo — do not run anything implying
it is).

Working directory for every command below:
`C:\Users\Asus\Documents\FisicoFabi\tesis\ship_adaptive_muon_bg`

---

## 1. Environment

**PURPOSE:** Activate the project's virtual environment.
**COMMAND:**
```powershell
& "C:\Users\Asus\Documents\FisicoFabi\tesis\ship_adaptive_muon_bg\.venv\Scripts\Activate.ps1"
```
**EXPECTED OUTPUT:** Prompt gains a `(.venv)` prefix.
**ARTIFACTS WRITTEN:** none.
**SAFE TO RE-RUN:** yes.
**TRAINS A MODEL:** no.

**PURPOSE:** Confirm Python/PyTorch/CUDA versions match what the audit
recorded (`torch_version = "2.13.0+cu126"` in this session; prior artifacts
were produced under `2.13.0+cpu`, `cuda_available: false`).
**COMMAND:**
```powershell
python -c "import torch; print('python_exe=', __import__('sys').executable); print('torch=', torch.__version__); print('cuda_available=', torch.cuda.is_available())"
```
**EXPECTED OUTPUT:** Three lines: the venv's python path, the torch version
string, and `True`/`False` for CUDA.
**ARTIFACTS WRITTEN:** none.
**SAFE TO RE-RUN:** yes.
**TRAINS A MODEL:** no.

---

## MANUAL VALIDATION COMMANDS

**PURPOSE:** Run only the focused patch tests for this mission.
**COMMAND:**
```powershell
python -m pytest -q tests/afterms/test_report_builder.py
```
**EXPECTED OUTPUT:** `17 passed` (0 failed).
**ARTIFACTS WRITTEN:** none (tests use `tmp_path`, not the repo's `artifacts/`).
**SAFE TO RE-RUN:** yes.
**TRAINS A MODEL:** no.

**PURPOSE:** Run the full repository test suite.
**COMMAND:**
```powershell
python -m pytest -q
```
**EXPECTED OUTPUT:** `552 passed, 2 skipped` (0 failed), as verified this
session.
**ARTIFACTS WRITTEN:** none.
**SAFE TO RE-RUN:** yes.
**TRAINS A MODEL:** no.

**PURPOSE:** Validate the queue configuration without executing anything.
**COMMAND:**
```powershell
python scripts/run_afterms_nightly_queue.py --dry-run
```
**EXPECTED OUTPUT:** Prints the git commit, the 14-job queue (`00_...` through
`13_build_nightly_report`), and one `[DRY-RUN] ... would execute` line per
job.
**ARTIFACTS WRITTEN:** none.
**SAFE TO RE-RUN:** yes.
**TRAINS A MODEL:** no.

---

## MANUAL TRAINING COMMANDS

These commands run real (but 5-epoch smoke-scale) training. None were
executed this session.

**PURPOSE:** Run only jobs 00-03 (environment smoke, shard build, shard
validation, preprocessing round-trip) — no neural training.
**COMMAND:**
```powershell
python scripts/run_afterms_nightly_queue.py --jobs 00_environment_and_dataset_smoke 01_build_afterms_shards 02_validate_afterms_shards 03_preprocessing_roundtrip_and_plots
```
**EXPECTED OUTPUT:** Sequential per-job `[Job ...]` progress lines; each job's
`status.json` reports `"status": "completed"`.
**ARTIFACTS WRITTEN:** `artifacts/afterms_nightly_v0/jobs/{00,01,02,03}_*/`,
`data/shards/afterms_nightly_v0/`.
**SAFE TO RE-RUN:** yes (deterministic given the same raw dataset file).
**TRAINS A MODEL:** no.

**PURPOSE:** Run one selected neural smoke job in isolation (example: job 05,
the affine preprocessing A/B comparison for PDG 13).
**COMMAND:**
```powershell
python scripts/run_afterms_nightly_queue.py --jobs 05_affine_preprocessing_ab_pdg13
```
**EXPECTED OUTPUT:** Per-epoch train/val loss printed for each of the 3
preprocessing variants; `artifacts/afterms_nightly_v0/jobs/05_affine_preprocessing_ab_pdg13/status.json` → `"completed"`.
**ARTIFACTS WRITTEN:** `.../jobs/05_affine_preprocessing_ab_pdg13/{metrics.json,run.log,status.json,*_model.pt}`.
**SAFE TO RE-RUN:** yes, but overwrites that job's existing artifacts —
back them up first if you want to diff before/after.
**TRAINS A MODEL:** yes.

**PURPOSE:** Run the remaining nightly training jobs (04 through 12)
sequentially, then build the final report (job 13).
**COMMAND:**
```powershell
python scripts/run_afterms_nightly_queue.py --jobs 04_legacy_available_code_realnvp_quantile 05_affine_preprocessing_ab_pdg13 06_affine_preprocessing_ab_pdg_minus13 07_affine_weight_ab_pdg13 08_affine_weight_ab_pdg_minus13 09_affine_capacity_smoke_pdg13 10_gaussian_controls_pdg13 11_gaussian_controls_pdg_minus13 12_memory_release_repeat_smoke 13_build_nightly_report
```
Or, to run the entire default queue (00 through 13) in one invocation:
```powershell
python scripts/run_afterms_nightly_queue.py
```
**EXPECTED OUTPUT:** Sequential completion of every listed job; final
`nightly_summary.md` states `NIGHTLY_SMOKES_COMPLETE` only if every job in
00-12 reports `"completed"`.
**ARTIFACTS WRITTEN:** all `artifacts/afterms_nightly_v0/jobs/*/` directories,
`artifacts/afterms_nightly_v0/report/{nightly_summary.json,nightly_summary.md,nightly_results.csv}`.
**SAFE TO RE-RUN:** yes, but overwrites existing job artifacts.
**TRAINS A MODEL:** yes (jobs 04-12).

---

## RECOVERY/RESUME COMMANDS

**PURPOSE:** Resume an interrupted queue from the last completed job (skips
jobs whose `status.json` already says `"completed"`).
**COMMAND:**
```powershell
python scripts/run_afterms_nightly_queue.py --resume
```
**EXPECTED OUTPUT:** Already-completed jobs are skipped; execution continues
from the first incomplete job.
**ARTIFACTS WRITTEN:** only for the jobs actually (re-)executed.
**SAFE TO RE-RUN:** yes.
**TRAINS A MODEL:** only for jobs that resume into a training job.

**PURPOSE:** Force exactly one named job to re-execute even under `--resume`
(e.g. to redo job 09 after fixing something, without redoing everything
before it).
**COMMAND:**
```powershell
python scripts/run_afterms_nightly_queue.py --resume --force-job 09_affine_capacity_smoke_pdg13
```
**EXPECTED OUTPUT:** All other completed jobs are skipped; job 09 executes
regardless of its prior `status.json`.
**ARTIFACTS WRITTEN:** `artifacts/afterms_nightly_v0/jobs/09_affine_capacity_smoke_pdg13/*` (overwritten).
**SAFE TO RE-RUN:** yes.
**TRAINS A MODEL:** yes (for a neural job).

**PURPOSE:** Stop the queue after a named job completes (e.g. to inspect
artifacts before continuing).
**COMMAND:**
```powershell
python scripts/run_afterms_nightly_queue.py --stop-after 03_preprocessing_roundtrip_and_plots
```
**EXPECTED OUTPUT:** Queue executes through job 03 and exits; jobs 04+ are
not started.
**ARTIFACTS WRITTEN:** jobs 00-03 only.
**SAFE TO RE-RUN:** yes.
**TRAINS A MODEL:** no (jobs 00-03 are non-neural).

---

## WATCHER COMMANDS

**PURPOSE:** Watch queue progress once (single snapshot).
**COMMAND:**
```powershell
python scripts/watch_afterms_nightly_queue.py
```
**EXPECTED OUTPUT:** A status snapshot of all job directories under
`--artifact-dir` (default `artifacts/afterms_nightly_v0`).
**ARTIFACTS WRITTEN:** `artifacts/afterms_nightly_v0/watch_packet.{json,md}`
(confirmed against `write_watch_packet()` in `scripts/watch_afterms_nightly_queue.py`).
**SAFE TO RE-RUN:** yes.
**TRAINS A MODEL:** no.

**PURPOSE:** Watch queue progress continuously while a queue run is in
another terminal.
**COMMAND:**
```powershell
python scripts/watch_afterms_nightly_queue.py --loop --interval 30
```
**EXPECTED OUTPUT:** Repeated snapshots every 30 seconds until interrupted
(Ctrl+C).
**ARTIFACTS WRITTEN:** same as above, repeatedly overwritten.
**SAFE TO RE-RUN:** yes.
**TRAINS A MODEL:** no.

---

## MANUAL REPORT/EVALUATION COMMANDS

**PURPOSE:** Regenerate the final nightly report from existing job artifacts
without running or re-running any training. This is the exact command this
mission used to verify the reporting patch (twice, producing byte-identical
output) against the real `artifacts/afterms_nightly_v0/` directory.
**COMMAND:**
```powershell
python scripts/run_afterms_nightly_queue.py --run-job 13_build_nightly_report --artifact-dir artifacts/afterms_nightly_v0 --shard-dir data/shards/afterms_nightly_v0
```
**EXPECTED OUTPUT:** No training output; only report-writing. Regenerates
`artifacts/afterms_nightly_v0/report/{nightly_summary.json,nightly_summary.md,nightly_results.csv}`.
**ARTIFACTS WRITTEN:** the three report files above (overwritten in place —
confirmed byte-identical across repeated runs this session, given unchanged
job artifacts).
**SAFE TO RE-RUN:** yes.
**TRAINS A MODEL:** no.

The following are explicitly **NOT implemented** in this repository as of
this session — do not attempt them, and do not accept any command claiming
to invoke them: a model arena / champion-selection CLI, a training-curve
gallery, a pair-plot generator, a KS+Holm-correction 1D test battery, an
energy-distance 2D permutation test, or a C2ST-with-permutation-null suite.
Only `c2st` (plain accuracy + ROC-AUC, no permutation null),
`tail_quantile_errors`, `exceedance_probability_errors`,
`duplicate_diagnostics`, `marginal_summaries`, and
`generated_domain_violations` exist today, computed automatically inside
`evaluate_generated_samples` during training — there is no separate CLI to
invoke them standalone against an already-trained checkpoint (the mission
explicitly ruled out adding a general CLI framework or new evaluation suite
this session).

---

## LOCATING ARTIFACTS

**PURPOSE:** Find logs, checkpoints, metrics, histories, generated samples,
and final reports for a given job.
**COMMAND:**
```powershell
Get-ChildItem -Recurse artifacts\afterms_nightly_v0\jobs\<job_name> | Select-Object FullName, Length
Get-ChildItem artifacts\afterms_nightly_v0\report
```
Layout, confirmed against the real artifact tree this session:
- Logs: `artifacts/afterms_nightly_v0/jobs/<job_name>/run.log`
- Job status: `artifacts/afterms_nightly_v0/jobs/<job_name>/status.json`
- Metrics + per-epoch history: `artifacts/afterms_nightly_v0/jobs/<job_name>/metrics.json`
  (top-level `"history"`/`"run_history"` list for per-epoch rows; `"metrics"`
  or per-variant `{variant}.metrics` dict for final scalars)
- Checkpoints: `artifacts/afterms_nightly_v0/jobs/<job_name>/*_model.pt`
  (bare `state_dict()`, not a bundled config+hash — see audit §1 Q7)
- Generated samples: not persisted to disk as a standalone artifact today;
  they exist only in-memory during `evaluate_generated_samples` and are
  summarized (not stored raw) into `metrics.json`'s `marginal_summaries` /
  `generated_domain_violations` — there is no `generated_sample_artifact`
  file to locate yet (a documented future-facing field, not built this
  session).
- Final reports: `artifacts/afterms_nightly_v0/report/{nightly_summary.json,nightly_summary.md,nightly_results.csv}`
**ARTIFACTS WRITTEN:** none (read-only).
**SAFE TO RE-RUN:** yes.
**TRAINS A MODEL:** no.

---

## GPU / PROCESS STATE (PowerShell)

**PURPOSE:** Check GPU process and memory state.
**COMMAND:**
```powershell
nvidia-smi --query-gpu=memory.free,memory.total --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
```
**EXPECTED OUTPUT:** Free/total VRAM in MiB; a row per process currently
using the GPU (empty if none). On a CPU-only host (as this artifact set was
produced on), `nvidia-smi` may not be installed — that is expected, not an
error to fix.
**ARTIFACTS WRITTEN:** none.
**SAFE TO RE-RUN:** yes.
**TRAINS A MODEL:** no.

**PURPOSE:** Check whether a detached/background queue process is still
alive.
**COMMAND:**
```powershell
Get-Process python -ErrorAction SilentlyContinue | Select-Object Id, ProcessName, StartTime, CPU
```
**EXPECTED OUTPUT:** One row per running `python.exe` process, or nothing if
none are running. Cross-reference `Id` against the PID you started the queue
with (`Start-Process` returns a `Process` object with `.Id`) to confirm it is
specifically the queue runner and not an unrelated Python process.
**ARTIFACTS WRITTEN:** none.
**SAFE TO RE-RUN:** yes.
**TRAINS A MODEL:** no.

---

## Remaining Limitations (see audit for full detail)

- No per-job standalone report-regeneration CLI beyond `--run-job
  13_build_nightly_report`, which requires `artifacts/afterms_nightly_v0/jobs/*/status.json`
  and `metrics.json` to already exist — it does not accept an arbitrary
  artifact directory produced elsewhere without matching that layout.
- No arena, training-curve, pair-plot, KS, energy-distance, or C2ST-null CLI
  exists — do not invent flags for these.
