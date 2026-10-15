# D8 After-MS Smoke Arena — Manual Execution Runbook v0

Companion to the D8 mission spec. D8 is a read-only evaluation arena over the
frozen `artifacts/afterms_nightly_v1` campaign (producer commit `f9d8246`).
It never runs D7's queue, optimizer code, generative `fit`, Gaussian/GMM fit,
shard construction, or raw-data split construction. The one exception is
fitting the diagnostic C2ST classifier (a fixed logistic regression), which
is a statistics-only diagnostic, not a generative model.

Working directory for every command below:
`C:\Users\Asus\Documents\FisicoFabi\tesis\ship_adaptive_muon_bg`

Branch: `experiment/d8-afterms-model-arena-v0` (created from the frozen D7
HEAD `56355d0` on `experiment/d7-afterms-sharded-smoke-v0`).

---

## 1. Environment

**PURPOSE:** Activate the project's virtual environment.
**COMMAND:**
```powershell
& "C:\Users\Asus\Documents\FisicoFabi\tesis\ship_adaptive_muon_bg\.venv\Scripts\Activate.ps1"
```
**EXPECTED OUTPUT:** Prompt gains a `(.venv)` prefix.
**ARTIFACTS WRITTEN:** none.
**TRAINS GENERATIVE MODEL:** no.
**SAFE TO RE-RUN:** yes.

**PURPOSE:** Confirm the current branch and that it was created from the
frozen D7 HEAD.
**COMMAND:**
```powershell
git branch --show-current
git log --oneline -1 experiment/d7-afterms-sharded-smoke-v0
```
**EXPECTED OUTPUT:** `experiment/d8-afterms-model-arena-v0`, and the D7 branch
tip is `56355d0 docs(afterms): document affine checkpoint_hash's
functional-fingerprint semantics`.
**ARTIFACTS WRITTEN:** none.
**TRAINS GENERATIVE MODEL:** no.
**SAFE TO RE-RUN:** yes.

---

## 2. Tests

**PURPOSE:** Run only the D8 focused test suite (tiny synthetic fixtures,
never the real 13.7M-row PKL).
**COMMAND:**
```powershell
python -m pytest -q tests/afterms/d8
```
**EXPECTED OUTPUT:** all tests pass, 0 failed.
**ARTIFACTS WRITTEN:** none (tests use `tmp_path`).
**TRAINS GENERATIVE MODEL:** no.
**SAFE TO RE-RUN:** yes.

**PURPOSE:** Run the full repository test suite (D7 + D8) to confirm no
regressions.
**COMMAND:**
```powershell
python -m pytest -q
```
**EXPECTED OUTPUT:** `623 passed, 2 skipped` (0 failed) or higher as the suite
grows; never fewer passes than the D7 baseline (`583 passed, 2 skipped`).
**ARTIFACTS WRITTEN:** none.
**TRAINS GENERATIVE MODEL:** no.
**SAFE TO RE-RUN:** yes.

---

## 3. Dry run

**PURPOSE:** Report the discovered runs, legacy adapter mapping, arenas,
exclusions, reconstruction plan, visualization candidates, and evaluation
budgets without writing anything to disk.
**COMMAND:**
```powershell
python scripts/build_afterms_smoke_arena.py --dry-run
```
**EXPECTED OUTPUT:** a JSON object ending in `"status": "DRY_RUN_OK"`, with
`phase_a_audit.classification` = `D8_INPUT_VALID`.
**ARTIFACTS WRITTEN:** none (dry run never touches the filesystem).
**TRAINS GENERATIVE MODEL:** no.
**SAFE TO RE-RUN:** yes.

---

## 4. Full execution

**PURPOSE:** Run the full D8 pipeline (audit, registry, arenas, curves,
reconstruction, deterministic samples, statistics, plots, report) against the
default frozen inputs.
**COMMAND:**
```powershell
python scripts/build_afterms_smoke_arena.py
```
**EXPECTED OUTPUT:** `AFTERMS_SMOKE_ARENA_COMPLETE` on stdout.
**ARTIFACTS WRITTEN:** everything under `artifacts/afterms_d8_evaluation_v0/`
(`audit/`, `registry/`, `arenas/`, `training_curves/`, `generated_samples/`,
`reference_samples/`, `statistics/`, `sample_matrices/`, `pz_diagnostics/`,
`report/`). Refuses to overwrite a non-empty output directory unless `--force`
is passed.
**TRAINS GENERATIVE MODEL:** no (reconstructs and samples from already-frozen
checkpoints only; the one classifier fit is the fixed diagnostic C2ST logistic
regression).
**SAFE TO RE-RUN:** yes, with `--force`.

**PURPOSE:** Re-run with `--force` to verify deterministic regeneration
(Gate 3 requirement).
**COMMAND:**
```powershell
python scripts/build_afterms_smoke_arena.py --force
Compare-Object (Get-Content artifacts/afterms_d8_evaluation_v0/generated_samples/*.json) (Get-Content <prior-run-copy>/generated_samples/*.json)
```
**EXPECTED OUTPUT:** identical `sample_hash` values in every
`generated_samples/*.json` across both runs (verified empirically: `diff -rq`
between two full runs' `generated_samples/` reports no differences).
**ARTIFACTS WRITTEN:** overwrites `artifacts/afterms_d8_evaluation_v0/`.
**TRAINS GENERATIVE MODEL:** no.
**SAFE TO RE-RUN:** yes.

**PURPOSE:** Run with smaller bounded budgets (faster, e.g. for local
iteration).
**COMMAND:**
```powershell
python scripts/build_afterms_smoke_arena.py --force --sample-size 2000 --energy-sample-size 500 --permutations 49 --bootstrap-repetitions 50 --c2st-sample-size 2000
```
**EXPECTED OUTPUT:** `AFTERMS_SMOKE_ARENA_COMPLETE`, same structure, smaller
sample counts recorded in every provenance file's `sample_count` /
`evaluation_budgets`.
**ARTIFACTS WRITTEN:** same tree, smaller files.
**TRAINS GENERATIVE MODEL:** no.
**SAFE TO RE-RUN:** yes.

---

## 5. Inspecting outputs

**PURPOSE:** Inspect the canonical run registry.
**COMMAND:**
```powershell
Get-Content artifacts/afterms_d8_evaluation_v0/registry/run_registry.md
```
**EXPECTED OUTPUT:** one row per historical run (22 rows against the real
frozen campaign), with `null` (never `0`/`NaN`/`""`) for genuinely missing
fields (e.g. `physical_space_nll` for every `quantile_normal_v0` and legacy
run).
**ARTIFACTS WRITTEN:** none (read-only).
**TRAINS GENERATIVE MODEL:** no.
**SAFE TO RE-RUN:** yes.

**PURPOSE:** Inspect the arenas and provisional champions.
**COMMAND:**
```powershell
Get-Content artifacts/afterms_d8_evaluation_v0/arenas/model_arena.md
```
**EXPECTED OUTPUT:** one section per arena (A-F plus G, the job-12 memory
diagnostic), each with a `provisional_smoke_champion` (or "no eligible
candidates") and, where the champion is not reconstructible, a separate
`visualization_candidate`.
**ARTIFACTS WRITTEN:** none.
**TRAINS GENERATIVE MODEL:** no.
**SAFE TO RE-RUN:** yes.

**PURPOSE:** View training curves and diagnostic plots.
**COMMAND:**
```powershell
ii artifacts/afterms_d8_evaluation_v0/training_curves
ii artifacts/afterms_d8_evaluation_v0/sample_matrices
ii artifacts/afterms_d8_evaluation_v0/pz_diagnostics
```
**EXPECTED OUTPUT:** File Explorer opens each folder; one `loss_curve__*.png`
per multi-epoch neural run (none for one-shot Gaussian/GMM baselines), plus
grouped comparison plots, sample-matrix plots, and pz-diagnostic plots for
every visualization candidate.
**ARTIFACTS WRITTEN:** none.
**TRAINS GENERATIVE MODEL:** no.
**SAFE TO RE-RUN:** yes.

**PURPOSE:** Read the distribution-test results.
**COMMAND:**
```powershell
Get-Content artifacts/afterms_d8_evaluation_v0/statistics/one_dimensional_tests.json
Get-Content artifacts/afterms_d8_evaluation_v0/statistics/two_dimensional_tests.json
Get-Content artifacts/afterms_d8_evaluation_v0/statistics/ndimensional_c2st.json
```
**EXPECTED OUTPUT:** per-run-id KS/energy-distance/C2ST results with
Holm-adjusted p-values (1D/2D, unweighted only) or a `"deferred": true` block
(2D, weighted arenas) with a documented reason -- never a classical KS
p-value attached to a weighted test.
**ARTIFACTS WRITTEN:** none.
**TRAINS GENERATIVE MODEL:** no.
**SAFE TO RE-RUN:** yes.

**PURPOSE:** Check the frozen D7 input hashes remain unchanged after any D8
run.
**COMMAND:**
```powershell
git status --short artifacts/afterms_nightly_v1 data/shards/afterms_nightly_v1
```
**EXPECTED OUTPUT:** no output (clean; D8 never writes into D7 paths).
**ARTIFACTS WRITTEN:** none.
**TRAINS GENERATIVE MODEL:** no.
**SAFE TO RE-RUN:** yes.

**PURPOSE:** Read the final report and D9 recommendations.
**COMMAND:**
```powershell
Get-Content artifacts/afterms_d8_evaluation_v0/report/afterms_smoke_arena.md
```
**EXPECTED OUTPUT:** the 17-section report (frozen D7 identity through
non-claims), ending in one of `AFTERMS_SMOKE_ARENA_COMPLETE` /
`_PARTIAL` / `_BLOCKED_BY_INPUTS` / `_BLOCKED_BY_IMPLEMENTATION`. Section 16
("D9 recommendations") lists concrete follow-ups for a future campaign
(persisting Gaussian/GMM checkpoints, adding a source-lineage identifier, a
verified weighted energy-distance estimator).
**ARTIFACTS WRITTEN:** none.
**TRAINS GENERATIVE MODEL:** no.
**SAFE TO RE-RUN:** yes.
