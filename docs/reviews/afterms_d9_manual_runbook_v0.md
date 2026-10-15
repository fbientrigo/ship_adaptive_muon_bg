# D9 Directed Multi-Seed Training — Manual Execution Runbook v0

Companion to the D9 mission spec. D9 implements a directed multi-seed
training campaign for 6 primary neural candidates derived from the frozen D8
evaluation (`artifacts/afterms_d8_evaluation_v0`, producer commit `f9d8246`).
**This mission implemented and validated the machinery only; the real
multi-seed campaign was never executed.** Every command below that trains a
real candidate is your responsibility to run, after reviewing
`configs/afterms/d9_candidate_plan_v0.json` and
`configs/afterms/d9_training_v0.json`.

Working directory for every command below:
`C:\Users\Asus\Documents\FisicoFabi\tesis\ship_adaptive_muon_bg`

Branch: `experiment/d9-afterms-directed-training-v0` (created from the frozen
D8 HEAD `59338ab` on `experiment/d8-afterms-model-arena-v0`).

---

## 1. Environment

**PURPOSE:** Activate the project's virtual environment.
**COMMAND:**
```powershell
& "C:\Users\Asus\Documents\FisicoFabi\tesis\ship_adaptive_muon_bg\.venv\Scripts\Activate.ps1"
```
**EXPECTED OUTPUT:** Prompt gains a `(.venv)` prefix.
**ARTIFACTS WRITTEN:** none.
**TRAINS A GENERATIVE MODEL:** no.
**SAFE TO RE-RUN:** yes.

## 2. Check Git branch and CUDA

**PURPOSE:** Confirm the current branch and that CUDA (RTX 2060) is visible
to PyTorch before training anything.
**COMMAND:**
```powershell
git branch --show-current
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```
**EXPECTED OUTPUT:** `experiment/d9-afterms-directed-training-v0`; a torch
version, `True`, `NVIDIA GeForce RTX 2060 with Max-Q Design` (or `False`/`None`
if run on a CPU-only machine -- training still works, just slower).
**ARTIFACTS WRITTEN:** none.
**TRAINS A GENERATIVE MODEL:** no.
**SAFE TO RE-RUN:** yes.

## 3. Run focused D9 tests

**PURPOSE:** Run only the D9 test suite (synthetic fixtures + tiny CPU
smokes, never the real shards).
**COMMAND:**
```powershell
python -m pytest -q tests/afterms/d9
```
**EXPECTED OUTPUT:** all tests pass, 0 failed.
**ARTIFACTS WRITTEN:** none (tests use `tmp_path`).
**TRAINS A GENERATIVE MODEL:** yes -- several tests train a tiny 2-epoch
synthetic model on CPU as part of verifying the runner; none touch the real
D7 shards or `artifacts/afterms_d9_training_v0`.
**SAFE TO RE-RUN:** yes.

## 4. Run the full suite

**PURPOSE:** Confirm no regressions across D7 + D8 + D9.
**COMMAND:**
```powershell
python -m pytest -q
```
**EXPECTED OUTPUT:** all tests pass; see this mission's final response for the
exact pass count.
**ARTIFACTS WRITTEN:** none.
**TRAINS A GENERATIVE MODEL:** yes (same tiny synthetic smokes as above).
**SAFE TO RE-RUN:** yes.

## 5. Inspect the D9 candidate plan

**PURPOSE:** Review which 6 candidates were selected and why before training
anything.
**COMMAND:**
```powershell
Get-Content configs\afterms\d9_candidate_plan_v0.json | ConvertFrom-Json | Select-Object -ExpandProperty candidates | Format-Table candidate_id, track_id, source_d8_run_id
Get-Content docs\reviews\afterms_d9_candidate_plan_v0.md
```
**EXPECTED OUTPUT:** 6 candidates (A1, A2, B1, B2, C1, D1); the `.md` explains
each `selection_reason`, including A2's disclosed capacity confound.
**ARTIFACTS WRITTEN:** none.
**TRAINS A GENERATIVE MODEL:** no.
**SAFE TO RE-RUN:** yes.

## 6. Run the D9 dry-run

**PURPOSE:** See every intended (candidate, seed) run and its estimated
runtime before training anything.
**COMMAND:**
```powershell
python scripts\run_afterms_d9_campaign.py plan --dry-run
```
**EXPECTED OUTPUT:** an 18-row table (6 candidates x 3 seeds) with estimated
per-run runtime and a total estimated serial GPU time (~5.8 hours,
provisional; see `non_convergence_disclaimer` in
`configs/afterms/d9_training_v0.json`), ending with "No training was executed
by this dry-run."
**ARTIFACTS WRITTEN:** none.
**TRAINS A GENERATIVE MODEL:** no.
**SAFE TO RE-RUN:** yes.

## 7. Train one candidate and one seed

**PURPOSE:** Run the first real D9 training job.
**COMMAND:**
```powershell
python scripts\run_afterms_d9_campaign.py train --candidate-id A1_capacity_medium_identity_pdg13_unweighted --seed 20260720 --device cuda
```
**EXPECTED OUTPUT:** per-epoch progress is not printed by default (see
histories/ for the JSON log); on completion, a JSON result with
`"status": "completed"`, `best_validation_metric`, `best_validation_epoch`.
**ARTIFACTS WRITTEN:**
`artifacts/afterms_d9_training_v0/runs/A1_capacity_medium_identity_pdg13_unweighted/seed_20260720/`
(status.json, training_config.json, preprocessing/, checkpoints/, histories/).
**TRAINS A GENERATIVE MODEL:** yes -- this is a real training job (~27 min
estimated on an RTX 2060, per the dry-run table).
**SAFE TO RE-RUN:** yes without `--resume` only if you intend to restart from
scratch; the runner will overwrite `last_resumable_checkpoint.pt` for this
exact (candidate, seed) but never another seed's files.

## 8. Watch training progress

**PURPOSE:** Monitor an in-progress or completed run without re-training.
**COMMAND:**
```powershell
Get-Content artifacts\afterms_d9_training_v0\runs\A1_capacity_medium_identity_pdg13_unweighted\seed_20260720\histories\training_history.json | ConvertFrom-Json | Select-Object -Last 5
python scripts\run_afterms_d9_campaign.py status --candidate-id A1_capacity_medium_identity_pdg13_unweighted
```
**EXPECTED OUTPUT:** the last few epoch records (train/validation NLL, wall
time); `status` line shows `running`, `completed`, or `interrupted`.
**ARTIFACTS WRITTEN:** none.
**TRAINS A GENERATIVE MODEL:** no.
**SAFE TO RE-RUN:** yes.

## 9. Safely interrupting

**PURPOSE:** Stop an in-progress run cleanly, preserving the last resumable
checkpoint.
**COMMAND:** In the terminal running step 7's command, press `Ctrl+C` once
and wait for it to exit (do not send a second Ctrl+C or close the window).
**EXPECTED OUTPUT:** the process exits; `status.json` for that run shows
`"status": "interrupted"`.
**ARTIFACTS WRITTEN:** `checkpoints/last_resumable_checkpoint.pt` (already
current as of the last completed epoch); `final_checkpoint.pt` is NOT
written.
**TRAINS A GENERATIVE MODEL:** no (this step only stops one).
**SAFE TO RE-RUN:** n/a.

## 10. Resuming the exact interrupted run

**PURPOSE:** Continue exactly the interrupted (candidate, seed) run, never a
different one.
**COMMAND:**
```powershell
python scripts\run_afterms_d9_campaign.py train --candidate-id A1_capacity_medium_identity_pdg13_unweighted --seed 20260720 --device cuda --resume
```
**EXPECTED OUTPUT:** training continues from the last completed epoch; if
the training contract (architecture/preprocessing/optimizer/dataset/relevant
source) changed since the interrupted run, this instead raises
`CheckpointCompatibilityError` and refuses to resume.
**ARTIFACTS WRITTEN:** same run directory as step 7, now with more epochs and
(on completion) `final_checkpoint.pt`.
**TRAINS A GENERATIVE MODEL:** yes (continuation of a real training job).
**SAFE TO RE-RUN:** yes (idempotent past completion -- resuming a completed
run's `last_resumable_checkpoint.pt` will simply have nothing left to do once
`max_epochs`/early stopping is already satisfied).

## 11. Checking best/final checkpoints

**PURPOSE:** Confirm the three checkpoint files exist and are distinct.
**COMMAND:**
```powershell
Get-ChildItem artifacts\afterms_d9_training_v0\runs\A1_capacity_medium_identity_pdg13_unweighted\seed_20260720\checkpoints
python -c "import torch; b=torch.load('artifacts/afterms_d9_training_v0/runs/A1_capacity_medium_identity_pdg13_unweighted/seed_20260720/checkpoints/best_checkpoint.pt', weights_only=True); print(b['checkpoint_scope'], b['epoch'], b['best_validation_metric'])"
```
**EXPECTED OUTPUT:** `best_checkpoint.pt`, `final_checkpoint.pt`,
`last_resumable_checkpoint.pt`; the best bundle's `checkpoint_scope=="best"`
and its epoch is typically earlier than the final epoch.
**ARTIFACTS WRITTEN:** none.
**TRAINS A GENERATIVE MODEL:** no.
**SAFE TO RE-RUN:** yes.

## 12. Evaluating one completed run

**PURPOSE:** Run test-split evaluation for one completed (candidate, seed).
**COMMAND:**
```powershell
python scripts\run_afterms_d9_campaign.py evaluate --candidate-id A1_capacity_medium_identity_pdg13_unweighted --seed 20260720 --budget quick_validation_budget --device cuda
```
**EXPECTED OUTPUT:** JSON with `test_feature_nll`, `test_physical_nll`,
negative-pz diagnostics, 1D/2D/C2ST test results.
**ARTIFACTS WRITTEN:**
`.../seed_20260720/evaluation/quick_validation_budget.json`.
**TRAINS A GENERATIVE MODEL:** no.
**SAFE TO RE-RUN:** yes.

## 13. Executing the remaining seeds sequentially

**PURPOSE:** Complete the other two seeds for this candidate, one at a time
(no parallel GPU training).
**COMMAND:**
```powershell
python scripts\run_afterms_d9_campaign.py train --candidate-id A1_capacity_medium_identity_pdg13_unweighted --seed 20260721 --device cuda
python scripts\run_afterms_d9_campaign.py train --candidate-id A1_capacity_medium_identity_pdg13_unweighted --seed 20260722 --device cuda
```
**EXPECTED OUTPUT:** two more completed runs under
`runs/A1_capacity_medium_identity_pdg13_unweighted/seed_20260721` and
`seed_20260722`.
**ARTIFACTS WRITTEN:** as in step 7, per seed.
**TRAINS A GENERATIVE MODEL:** yes (two more real training jobs).
**SAFE TO RE-RUN:** yes, per the same rule as step 7.

## 14. Summarizing a candidate

**PURPOSE:** Aggregate the 3-seed results for one candidate.
**COMMAND:**
```powershell
python scripts\run_afterms_d9_campaign.py summarize --candidate-id A1_capacity_medium_identity_pdg13_unweighted
```
**EXPECTED OUTPUT:** JSON with `completed_seed_count`, `median_best_validation_nll`,
`std_best_validation_nll`, `best_epoch_distribution`, etc.
**ARTIFACTS WRITTEN:** none (reads from disk only).
**TRAINS A GENERATIVE MODEL:** no.
**SAFE TO RE-RUN:** yes.

## 15. Comparing candidates across seeds

**PURPOSE:** Aggregate all 6 candidates at once.
**COMMAND:**
```powershell
python scripts\run_afterms_d9_campaign.py summarize
```
**EXPECTED OUTPUT:** JSON list of all 6 candidates' aggregate stats, for
side-by-side comparison (e.g. A1 vs A2, B1 vs B2).
**ARTIFACTS WRITTEN:** none.
**TRAINS A GENERATIVE MODEL:** no.
**SAFE TO RE-RUN:** yes.

## 16. Running quick statistical evaluation

**PURPOSE:** Run the cheap evaluation budget across all completed runs of a
candidate, for an early look.
**COMMAND:**
```powershell
foreach ($seed in 20260720, 20260721, 20260722) {
  python scripts\run_afterms_d9_campaign.py evaluate --candidate-id A1_capacity_medium_identity_pdg13_unweighted --seed $seed --budget quick_validation_budget --device cuda
}
```
**EXPECTED OUTPUT:** three `evaluation/quick_validation_budget.json` files.
**ARTIFACTS WRITTEN:** as in step 12, per seed.
**TRAINS A GENERATIVE MODEL:** no.
**SAFE TO RE-RUN:** yes.

## 17. Running final statistical evaluation

**PURPOSE:** Run the full evaluation budget, once seeds are reviewed and a
candidate's results look stable. This budget is materially more expensive
(20000 generated samples, 5000 C2ST subsample vs 2000/500) -- run it only
after reviewing the quick budget.
**COMMAND:**
```powershell
python scripts\run_afterms_d9_campaign.py evaluate --candidate-id A1_capacity_medium_identity_pdg13_unweighted --seed 20260720 --budget final_candidate_budget --device cuda
```
**EXPECTED OUTPUT:** `evaluation/final_candidate_budget.json` with the larger
sample/statistics budget.
**ARTIFACTS WRITTEN:** `.../evaluation/final_candidate_budget.json`.
**TRAINS A GENERATIVE MODEL:** no.
**SAFE TO RE-RUN:** yes.

## 18. Locating all artifacts

**PURPOSE:** Find everything D9 has written so far.
**COMMAND:**
```powershell
Get-ChildItem -Recurse artifacts\afterms_d9_training_v0 | Select-Object FullName
```
**EXPECTED OUTPUT:** the full tree under `campaign_manifest.json`,
`candidate_plan/`, `runs/<candidate_id>/seed_<seed>/{status.json,
training_config.json, preprocessing/, checkpoints/, histories/,
evaluation/}`.
**ARTIFACTS WRITTEN:** none (read-only listing).
**TRAINS A GENERATIVE MODEL:** no.
**SAFE TO RE-RUN:** yes.

## 19. Verifying D7/D8 immutability

**PURPOSE:** Confirm this mission never modified the frozen D7/D8 inputs.
**COMMAND:**
```powershell
python -m pytest -q tests/afterms/d9/test_plan.py::test_d7_d8_inputs_unchanged
```
**EXPECTED OUTPUT:** 1 passed -- re-verifies the SHA-256 of
`report/afterms_smoke_arena.json`, `arenas/model_arena.json`, and
`registry/run_registry.json` against the values recorded in
`configs/afterms/d9_candidate_plan_v0.json`.
**ARTIFACTS WRITTEN:** none.
**TRAINS A GENERATIVE MODEL:** no.
**SAFE TO RE-RUN:** yes.

## 20. Verifying no training process remains

**PURPOSE:** Confirm no orphaned Python/CUDA process is left running after
manual training (especially after a Ctrl+C interrupt).
**COMMAND:**
```powershell
Get-Process python -ErrorAction SilentlyContinue
nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader
```
**EXPECTED OUTPUT:** no `python.exe` process still attached to this
`run_afterms_d9_campaign.py` invocation; `nvidia-smi` reports no compute
process holding the GPU from this run.
**ARTIFACTS WRITTEN:** none.
**TRAINS A GENERATIVE MODEL:** no.
**SAFE TO RE-RUN:** yes.

---

## Notes

- Every `train`/`evaluate` command above defaults `--device` to `cpu` if
  omitted; pass `--device cuda` explicitly to use the RTX 2060.
- `train` always requires both `--candidate-id` and `--seed`; there is no
  command in this CLI that trains more than one (candidate, seed) per
  invocation.
- The shard scope for this v0 training config is `train_shard_000` /
  `validation_shard_000` / `test_shard_000` only (same scope D7's five-epoch
  smoke used) -- see `shard_scope_note` in
  `configs/afterms/d9_training_v0.json`.
