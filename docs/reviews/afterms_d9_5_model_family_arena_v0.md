# D9-5: Modern Same-Track Generative Model Family Arena

Status: **Gates A-C complete, Gate D (Phase 0) closed**. Gates A-C: data-scope
audit + common model-adapter contract + Gaussian/GMM/NF_AC adapters +
evaluation/aggregation/report layer + CLI + tests, then real
GAUSS_DIAG/GAUSS_FULL/GMM fits and validation on the full declared scope for
both tracks. Gate D ran an unattended nightly-block campaign
(`docs/reviews/afterms_d9_5_nightly_execution_v0.md`) that executed 5 of the
6 frozen NF_AC seed runs to a genuine terminal state (100/75/54/100/55
epochs); the sixth (`TRK_PDGM13_UW_ID` seed `20260722`) was never started.
The user then explicitly directed the campaign to stop, judging the 5
executed runs sufficient evidence for **Phase 0 (nominal seed-stability
baseline)** -- this is a deliberate early closure, not a completed 6/6
campaign and not a failure. See Sec 11 below for the full Phase 0 closure
record, and `docs/reviews/afterms_d9_5_execution_runbook_v0.md` for the
Gate B/C/D results tables and exact reproduce/inspect commands.

**Phase 0 is a nominal seed-stability baseline only.** It says nothing about
physical MC weighting, utility-weighted enrichment, synthetic tags, new
architectures, optimizer sweeps, or Flow Matching -- none of that was
implemented or attempted here, and none of it is implied by these results.
Gate E (freeze-selection, frozen test-split evaluation) has **not** run; the
test split has not been opened by any Gate D/Phase-0 activity.

Branch: `experiment/d9-5-afterms-model-family-arena-v0`, forked from D9C HEAD
`22d38b5`.

## 1. Scientific question

Under the same empirical after-MS target, the same five-dimensional state,
the same data splits and the same evaluation protocol, how do affine-coupling
NF, diagonal Gaussian, full Gaussian and a four-component full-covariance GMM
compare? This is the first modern, direct, same-track comparison across these
four families in this repository. It is explicitly **not** a FairShip
downstream-signal comparison -- see Sec 10 (Limitations) below.

## 2. Data source and five-dimensional state

Source dataset: `data/shards/afterms_nightly_v1` (dataset hash
`44336e8e3629149c813026cf21334c6fc2db75ae9ffa12e78ed2b064f3a54579`), the same
frozen shard set D9/D9B/D9C already use. Raw rows are
`(px, py, pz, x, y, z, id, w)`; the modeled state is exactly

    px, py, pz, x, y

(dimension 5). `id` (PDG code) selects the track; `w` (source weight) is
recorded as metadata only and never enters a feature vector, a loss, or a
sampling weight anywhere in D9-5.

## 3. Track definitions

Two independent, never-pooled, never-cross-ranked tracks:

- `TRK_PDG13_UW_ID` -- PDG +13, row-empirical-unweighted, identity-standardized.
- `TRK_PDGM13_UW_ID` -- PDG -13, row-empirical-unweighted, identity-standardized.

Both resolve through `ship_muon_bg.afterms.model_naming.resolve_track` against
`configs/afterms/model_alias_registry_v0.json` -- the same alias machinery
D9C introduced; D9-5 adds no new alias codes. log1p-pz and production-weighted
tracks are out of scope (weighting interpretation remains OPEN, tracked
outside D9-5).

## 4. Exact rows/shards used

Declared policy: `configs/afterms/d9_5_data_scope_v0.json` (`shard_scope:
"complete_declared_scope"` -- all 22 train / 3 validation / 3 test shards, no
row subsampling, no silent negative-pz dropping). Resolved evidence: the audit
run in `artifacts/afterms_d9_5_model_family_arena_v0/data_scope/` records,
per split and track, the concrete shard names/hashes, raw and PDG-filtered
row counts, per-feature finite-value counts, negative-pz counts and a content
hash of the filtered scope.

Resolved row counts from the frozen audit (`data_scope_manifest.json`):

| track | train rows | validation rows | test rows |
|---|---|---|---|
| TRK_PDG13_UW_ID | 5,484,126 | 685,646 | 685,139 |
| TRK_PDGM13_UW_ID | 5,539,934 | 691,239 | 692,996 |

No negative-`pz` rows exist anywhere in this dataset (0 across every
split/track); the negative-pz-preservation machinery in D9-5's pipeline is
therefore inert on this data but remains active/tested, since it is a
declared invariant, not an assumption about this one dataset.

A host-RAM preflight (`ram_preflight.json`, Sec 6.1 of the mission) confirms
the complete declared scope fits comfortably in host RAM for every family
(peak estimates in the few-hundred-MB range per track); **no chunked/streaming
GMM implementation was required and `D9_5_BLOCKED_BY_GMM_DATA_SCALE` does not
apply**.

## 5. Preprocessing

`identity_standardized_v0` for every family (mean/std of the training split's
5 features, fit once per track on the full training scope). Physical-space
NLL is the primary cross-family likelihood metric:

    log p_x(x) = log p_z(T(x)) + log|det J_T|,  log|det J_T| = -sum(log(std))

Every adapter's `test_log_prob`/`validation_log_prob` returns **feature-space**
log p(z) only; the shared evaluation layer (`d9_5/evaluation.py`) adds the
Jacobian once, so no family can silently double-count or drop that constant.
This is verified directly: `tests/afterms/d9_5/test_gaussian_adapter.py::
test_gaussian_physical_nll_matches_independent_raw_space_calculation` checks
that (standardized-space log p(z) + Jacobian) exactly matches an
independently, directly computed raw-space Gaussian NLL on the same rows.

## 6. Model configurations

| model_config_id | family | notes |
|---|---|---|
| `NF_AC_...` (per track) | affine-coupling NF | derived from the named `AFFINE_COUPLING_CAPACITY_SCOUT_V0` report, never hardcoded -- see Sec 7 |
| `GAUSS_DIAG_d05` | diagonal Gaussian | deterministic single fit, two-pass streaming sufficient statistics |
| `GAUSS_FULL_d05` | full-covariance Gaussian | deterministic single fit, two-pass streaming sufficient statistics |
| `GMM_k04_covFULL_d05` | 4-component full-covariance GMM | 3 seeds (20260720/21/22), sklearn EM driven step-by-step to record a genuine lower-bound curve |

Exact settings (regularization, `n_init`, `max_iter`, `tol`, seeds) are frozen
in `configs/afterms/d9_5_model_family_arena_v0.json`.

**Gate B/C validation results (quick_validation_budget, real full-scope
data)** -- see the runbook for the full table; summary: every GMM seed
converged (14-20 EM iterations) with all 4 components occupied on both
tracks, and GMM's best seed reaches validation physical NLL ~4.72
(TRK_PDG13_UW_ID) / ~4.72 (TRK_PDGM13_UW_ID) versus GAUSS_FULL's ~7.40/~7.39
and GAUSS_DIAG's ~9.07/~9.06. This is expected given GMM's much larger
parameter count and is **not yet a same-track four-family comparison** --
NF_AC has not been fitted (Gate D) and no validation-only selection has been
frozen.

## 7. NF_AC configuration selection (scout-promoted)

`ship_muon_bg.afterms.d9_5.nf_ac_adapter.select_scout_promoted_nf_config`
resolves the architecture per track from
`artifacts/afterms_d9_model_arena_v0/named/run_inventory_named.json` (the
named `AFFINE_COUPLING_CAPACITY_SCOUT_V0` report): `argmin(best_validation_metric)`
over that track's `status == "completed"` capacity variants. Confirmed
resolution:

- `TRK_PDG13_UW_ID` -> `NF_AC_b08_w128_d02` (scout val NLL 1.3992097402270696, run `A1_capacity_medium_identity_pdg13_unweighted__arena_cap_1.00x`)
- `TRK_PDGM13_UW_ID` -> `NF_AC_b06_w096_d02` (scout val NLL 1.4511939737719584, run `B2_capacity_small_identity_pdg_minus13_unweighted__arena_cap_1.50x`)

The scout used a single seed (20260720) and 10 epochs; it selected
**architecture capacity only**, on a single training shard, not a converged
final model. D9-5 labels this `scout_promoted_nf_config` (never "final
champion") and trains three new seeds from initialization on the full
declared scope for each track -- the scout's own checkpoints are never
reused (structurally: D9-5's run directories live entirely under
`artifacts/afterms_d9_5_model_family_arena_v0/`, isolated from the scout's
`artifacts/afterms_d9_model_arena_v0/` tree).

The alias/label string `scout_promoted_nf_config` is presentation metadata
only -- it is never folded into the hashed `candidate_config` that feeds
`semantic_training_hash` (see `nf_ac_adapter.py`'s `_build_candidate_config`
and required test 25).

## 8. Fitting/training policies

- **Gaussian controls**: deterministic single fit, two-pass chunk-bounded
  streaming sufficient statistics (verified algebraically identical to direct
  `mean`/covariance computation on the same rows -- required tests 13/14).
- **GMM**: 3 deterministic seeds (20260720/21/22), `n_components=4`,
  `covariance_type="full"`, `reg_covar=1e-6`, `n_init=1`, `max_iter=200`,
  `tol=1e-3` -- every setting frozen in
  `configs/afterms/d9_5_model_family_arena_v0.json`, never silently changed.
  Non-convergence is recorded, never hidden (`converged: false` +
  a warning), and is not automatically interpreted as poor physics.
- **NF_AC**: execution policy frozen *before* the first seed of either track
  trains: `minimum_epochs=20`, `maximum_epochs=100`,
  `early_stopping_patience=25` (see the config file's `justification` field
  for the reasoning and Sec 11.1 bounds). Three seeds (20260720/21/22) trained
  from initialization, sequentially, one CUDA process at a time, float32, no
  AMP, `num_workers=0` (Windows / RTX 2060 Laptop, 6 GB VRAM).

## 9. Likelihood semantics, validation-only selection, frozen test evaluation

All four families share one preprocessing contract and one Jacobian
convention (Sec 5). Validation-only selection is a hard gate:
`ship_muon_bg.afterms.d9_5.report.require_frozen_selection` raises unless
`family_selection_manifest.json` already exists (written only by
`freeze-selection --execute`); `evaluate-test` calls this before reading any
test shard (required tests 27/28). The primary validation metric is
physical-space mean NLL, explicitly stated as primary-but-not-only
(`report.build_family_selection_manifest`'s `primary_metric_statement`).
Ranking is always within-track; no code path produces or implies a global
cross-track ranking (required test 31).

## 10. Limitations

- This is a pre-FairShip, pre-GEANT4, proxy-free comparison of generative
  fits to an empirical after-MS distribution. It says nothing about which
  family better reproduces the downstream DIS-relevant subset, and no D9-5
  output should be read as a FairShip readiness signal.
- The NF_AC architecture is scout-promoted (single seed, 10 epochs, one
  shard) -- a capacity choice with real evidence behind it, but a much
  smaller evidence base than the full D9-5 multi-seed campaign that trains
  on top of it.
- GMM `k=4` and Gaussian model classes are fixed, not searched; this is a
  same-track comparison of these four specific configurations, not a
  hyperparameter search over model families.
- Source weight `w`'s physical interpretation remains an open question
  outside D9-5's scope; production-weighted tracks are excluded entirely.
- PDG +13 and PDG -13 are two separate empirical targets and are never
  ranked against each other.

## 11. Phase 0 closure (Gate D, NF_AC nominal seed-stability baseline)

**Purpose.** Phase 0 exists to answer one narrow question: for the frozen,
scout-promoted NF_AC architecture per track, trained under one frozen
optimizer/execution policy, how stable is the resulting validation NLL across
independent seed reinitializations? It is a **nominal seed-stability
baseline only**. It is explicitly **not** an enrichment experiment: no
physical MC weighting, no utility weighting, no synthetic tags, no data
enrichment, no new architectures, no optimizer sweeps, and no Flow Matching
were implemented or evaluated in Phase 0. None of the numbers below should be
read as validating physical tail behavior, MC-weight interpretation, utility
enrichment, or downstream FairShip usefulness.

**Frozen shared hyperparameters** (identical `execution_policy_hash` and
`evaluation_policy_hash` confirmed across every executed run -- see
verification below):

| Setting | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 256 |
| Weight decay | 0.0 |
| Gradient clipping | 5.0 |
| Dtype / AMP | float32, no AMP |
| Minimum / maximum epochs | 20 / 100 |
| Early stopping patience | 25 epochs |
| Weighting policy | `row_empirical_unweighted` (all runs) |
| Preprocessing | `identity_standardized_v0` (all runs) |

**Per-track architecture** (frozen before any seed trained; consistent
across every seed of its track):

| Track | PDG | `model_config_id` | Blocks | Hidden width | Hidden depth |
|---|---|---|---|---|---|
| `TRK_PDG13_UW_ID` | 13 | `NF_AC_b08_w128_d02` | 8 | 128 | 2 |
| `TRK_PDGM13_UW_ID` | -13 | `NF_AC_b06_w096_d02` | 6 | 96 | 2 |

**Six-run results.** The frozen queue defines 6 runs (3 seeds x 2 tracks).
**5 of 6 reached a genuine terminal state** (natural completion, either
`maximum_epochs` or early stopping); the 6th was never started, and Phase 0
was explicitly closed by the user before it began -- this is a deliberate
scope decision, not a technical failure or an invariant violation being
silently repaired.

| # | Run | Terminal epoch | Reason | Best epoch | Best val NLL | Final train NLL | Final val NLL |
|---|---|---|---|---|---|---|---|
| 1 | PDG13 / seed 20260720 | 100 | max epochs | 78 | 1.3032 | 1.3163 | 1.3203 |
| 2 | PDG13 / seed 20260721 | 75 | early-stopped | 50 | 1.3022 | 1.3175 | 1.3223 |
| 3 | PDG13 / seed 20260722 | 54 | early-stopped | 17 | 1.3221 | 1.3390 | 1.3351 |
| 4 | PDG-M13 / seed 20260720 | 100 | max epochs | 76 | 1.3123 | 1.3315 | 1.3356 |
| 5 | PDG-M13 / seed 20260721 | 55 (last checkpointed epoch) | **stopped by explicit user request, not a natural terminal state** | 38 | 1.3136 | n/a (mid-training) | n/a (mid-training) |
| 6 | PDG-M13 / seed 20260722 | -- | **not started** | -- | -- | -- | -- |

Run 5 was stopped via the nightly runner's PID-scoped `abort` (never a
by-name kill); its last completed epoch (55) and its checkpoints
(`best_checkpoint.pt` at epoch 38, `last_resumable_checkpoint.pt` at epoch
55) are valid and usable, but its `training_config.json` and
`preprocessing/preprocessing.json` do not exist on disk -- these files are
written by the training code only at natural completion, not at every
epoch, so their absence for run 5 is expected and does not indicate
corruption. All fields needed for audit (architecture, hashes, policy)
were instead read directly from its checkpoint bundle, which the
training loop writes every epoch (`checkpoint_written: true`).

**Artifact locations:**

```
artifacts/afterms_d9_5_model_family_arena_v0/runs/<track>/<model_config_id>/seed_<seed>/
  status.json                       (absent-final-state for run 5: still shows "running")
  training_config.json              (missing for run 5; see note above)
  histories/training_history.json   (present, sequential, no duplicate epochs, for all 5 executed runs)
  checkpoints/{best_checkpoint.pt, last_resumable_checkpoint.pt}
  preprocessing/preprocessing.json  (missing for run 5; see note above)
```

**Source dataset.** Hash `44336e8e3629149c813026cf21334c6fc2db75ae9ffa12e78ed2b064f3a54579`
(full declared shard scope, Sec 4). Per-run checkpoint `dataset_hash`/
`split_hashes` cover only the train+validation shards actually loaded for
that run (`{"train": ..., "validation": ...}`, no `"test"` key) -- confirmed
identical across all seeds of a track, confirming no run silently used a
different data scope or a different track's data.

**Split-integrity statement.** `data_scope_audit.md` (Gate A) already
recorded zero shard-overlap violations across train/validation/test for both
tracks; nothing in Gate D touches shard assignment. **The Gate E/test split
was not opened at any point during Gate D or this closure** -- no
`freeze-selection`, `evaluate-test`, or `summarize` artifacts exist anywhere
under `artifacts/afterms_d9_5_model_family_arena_v0/`, and no test shard
appears in any run's `dataset_hash`/`split_hashes`.

**Incidents.** Zero blocking-failure incidents were recorded in
`nightly_runner/incidents/` across the entire campaign (12 auto-chained
8-hour blocks). The only non-natural stop was the final, explicit,
user-directed `abort` of run 5 -- a deliberate closure action, not a
technical failure.

**Verified invariants** (read directly from each run's checkpoint bundle):

- All 5 executed runs have `model_family: affine_coupling`.
- Only the seed varies within each track; architecture is identical across
  all seeds of a track and differs correctly between tracks (8/128/2 vs
  6/96/2).
- `weighting_policy: row_empirical_unweighted` on every run -- no physical
  MC weight or utility weight entered the loss anywhere.
- `execution_policy_hash` and `evaluation_policy_hash` are byte-identical
  across all 5 executed runs -- the frozen policy was never silently
  changed mid-campaign.
- `split_hashes["train"]`/`["validation"]` are identical across all seeds of
  a given track, and differ correctly between the two tracks -- no run
  silently reused another run's data or another run's result.
- Each run has its own independent `seed_<seed>/` output directory; no
  overwrite or cross-run reuse observed.
- After the final `abort`, both the per-block supervisor lock and the
  campaign lock were confirmed released (the campaign lock had gone stale
  on disk after the hard stop and was cleaned up via the existing
  `nightly_runner.reconcile_lock` function -- no new code was written for
  this); no CUDA process remained (`nvidia-smi` confirmed empty compute-app
  list).

**Unresolved items deferred to Phase 1 (explicitly not attempted here):**

- `TRK_PDGM13_UW_ID` seed `20260722` was never trained; Phase 0's PDG-M13
  seed-stability read rests on 2 seeds, not 3.
- Run 5's seed-stability contribution is based on a non-terminal checkpoint
  (epoch 55, best at epoch 38), not a naturally completed/early-stopped run.
- `freeze-selection`/`evaluate-test`/`summarize` (Gate E) have not run for
  any family, including NF_AC -- no test-split evaluation, no cross-family
  validation-only selection, no final report exists yet.
- Physical MC weighting, utility weighting, synthetic tags, data
  enrichment, new architectures, optimizer sweeps, and Flow Matching are
  all out of scope for Phase 0 and remain open for a deliberately separate
  Phase 1 decision.

**Reproduce/inspect commands:**

```powershell
git log --oneline -5
.venv\Scripts\python.exe scripts\run_afterms_d9_5_nightly.py status
.venv\Scripts\python.exe scripts\run_afterms_d9_5_nightly.py history
.venv\Scripts\python.exe scripts\run_afterms_d9_5_model_family_arena.py status
```

To inspect a specific run's raw evidence directly:

```powershell
Get-Content artifacts\afterms_d9_5_model_family_arena_v0\runs\TRK_PDG13_UW_ID\NF_AC_b08_w128_d02\seed_20260720\status.json
```

**Commit.** This closure was committed as `chore(d9): close Gate D seed
arena` on branch `experiment/d9-5n-afterms-nightly-gpu-runner-v0`; see
`git log --oneline -1` for the exact hash on this branch.
