# After-MS Nightly Smoke — Senior Evaluation Audit v0

**Scope:** `experiment/d7-afterms-sharded-smoke-v0`, commit `4976067` (pre-patch) /
patched in this same working tree. Reviewed: `scripts/run_afterms_nightly_queue.py`,
`scripts/watch_afterms_nightly_queue.py`, `scripts/build_afterms_shards.py`,
`src/ship_muon_bg/afterms/{preprocessing,log1p_pz,split,stratify,audit}.py`,
`tests/afterms/*`, `Nflow/interfaces.py`, `Nflow/torch_models/affine_coupling.py`,
`Nflow/baselines/{gaussian,gmm}.py`, and every artifact under
`artifacts/afterms_nightly_v0/`. Out of scope per mission brief: FairShip, proxy
U(x), utility tilting, adaptive sampling, RL, spline flows, D0-D5 density_lab
targets not reused by this queue.

This document is the pre-change audit plus a record of the minimal patch that
followed it. No models were retrained except jobs 10 and 11 (Gaussian/GMM
controls), which are near-instant (≤20s per run) and were rerun only because a
value they were supposed to report (train NLL) had never actually been
computed before — see §3.4.

---

## 1. Primary Questions

| # | Question | Verdict | Evidence |
|---|---|---|---|
| 1 | Correct epoch-by-epoch train/val NLL histories | PARTIALLY_IMPLEMENTED | `run.py:709-719` (affine) writes `epoch, train_loss, validation_loss, cpu_rss_bytes, gpu_*_bytes, dataloader_time_seconds, training_time_seconds` per epoch for jobs 04-09. Missing: `train_physical_nll`, `validation_physical_nll`, `learning_rate`, per-epoch finite-loss flag (§2C). Jobs 10/11 (Gaussian/GMM) have no per-epoch loop at all — one synthetic `{"epoch": 1, ...}` entry (`run.py:745`, one-shot `fit()`). |
| 2 | Comparable physical-space NLL where mathematically valid | IMPLEMENTED_AND_CORRECT (for identity/log1p) / IMPLEMENTED_BUT_INCORRECT (report rendering) | See §4. Computation is correct; job 04's report row rendered `0.0000` instead of `N/A` for an undefined value — **fixed** (§5, patch 3). |
| 3 | Correct weighted and unweighted loss semantics | IMPLEMENTED_AND_CORRECT, mislabeled by the mission's stricter terminology | See §3.2. Self-normalized per-minibatch ratio estimator, already self-labeled `"self_normalized_minibatch_approximation"` in the code (`run.py:775` pre-patch). Not the exact globally-normalized weighted MLE — correctly not claimed as one. |
| 4 | Representative and leak-free row-level splits | IMPLEMENTED_AND_CORRECT | `split.py` assigns each row a single deterministic split label by row-hash; `tests/afterms/test_split.py::test_no_row_overlap_across_splits` asserts empty pairwise set intersections over 100,000 rows. Acknowledged limitation: no muon-lineage group id, so row-disjointness ≠ source-independence (`split.py:22`, `LIMITATION_NO_SOURCE_LINEAGE`, surfaced verbatim in the report's Limitations section). |
| 5 | Deterministic model and preprocessing comparisons | IMPLEMENTED_BUT_INCORRECT → fixed | Training was seeded (`gen.manual_seed(20260720)`, `run.py`), but **sample generation was not**: `torch.randn(5000, 5, ...)` (affine path) and `model.base_dist.sample((5000,))` (job 04) drew from the unseeded global RNG. Fixed by reusing the seeded generator / adding an explicit `torch.manual_seed` immediately before sampling. A second pass (this session) closed the remaining gap the mission's Item E names explicitly: seeding alone was not provenance — `generation_seed`, `generated_sample_count`, `generated_sample_hash`, and (for the two families that save a checkpoint file) `checkpoint_hash` are now written into every job's `eval_metrics` via new `generated_sample_hash()` / `checkpoint_file_hash()` helpers (`run.py`). Gaussian/GMM controls save no checkpoint file today, so their `checkpoint_hash` is honestly `None`, not fabricated. |
| 6 | Generated samples in physical coordinates | IMPLEMENTED_AND_CORRECT | `pipeline.inverse(gen_scaled)` (`run.py`) round-trips generated samples back to physical units for every job; confirmed by `marginal_summaries` in `metrics.json` reporting physical-unit ranges. |
| 7 | Usable checkpoints for every completed neural job | PARTIALLY_IMPLEMENTED | Spot-verified: loaded `05_affine_preprocessing_ab_pdg13/identity_standardized_v0_affine_small_unweighted_model.pt` into a freshly constructed `AffineCouplingFlow` module and confirmed `log_prob` is deterministic and reproducible. **Gap:** the runner saves a bare `state_dict()` (`torch.save(module.state_dict(), ...)`), not the bundled `{state_dict, model_config.json, checkpoint_hash.txt}` that `AffineCouplingFlow.save/load` (the tested API, `Nflow/torch_models/affine_coupling.py:508-577`) provides. Reload correctness today depends on the architecture spec (`hidden_width`/`number_of_blocks`/`hidden_depth`) hardcoded per capacity name in `run.py:611-621` staying in sync with the run label — nothing hashes or verifies this, unlike the tested checkpoint path. |
| 8 | Enough artifacts to build a model arena without retraining | PARTIALLY_IMPLEMENTED | Checkpoints, metrics, and histories exist for every combo. What's missing for a trustworthy arena: consistent physical-vs-feature NLL serialization (fixed here), a real train NLL for Gaussian/GMM (fixed here), and any arena-key/`comparison_scope` bookkeeping (not implemented — deferred, §7). |
| 9 | 1D / 2D / N-D two-sample diagnostics | PARTIALLY_IMPLEMENTED | Present: `c2st` (accuracy + ROC-AUC), `tail_quantile_errors`, `exceedance_probability_errors`, `duplicate_diagnostics`, `marginal_summaries` (`density_lab/metrics.py`, `run.py:evaluate_generated_samples`). Missing entirely: KS+Holm-correction 1D battery, energy-distance 2D permutation tests, C2ST with a permutation null / reference-vs-reference band (mission §9-11). Deferred — see §7. |
| 10 | Honest separation between smoke validation and model selection | IMPLEMENTED_AND_CORRECT | No code path anywhere in `run_afterms_nightly_queue.py` compares `test_nll` (or any metric) across variants to pick a winner; every preprocessing/capacity/weight combination is trained and recorded unconditionally (`run.py:591-786`, confirmed by reading the full loop nest — no `min`/`argmax`/selection logic exists). Test NLL is computed exactly once per combo, strictly after training (`run.py`, `test_lp = module.log_prob(...)` after the 5-epoch loop). However, there is no formal `comparison_scope="five_epoch_single_seed_smoke"` label anywhere yet — nothing stops a future reader from over-interpreting these numbers as a champion selection. Recommend adding that label when the arena work (§7) is actually built. |

---

## 2. Junior Report Audit

The junior report (`artifacts/afterms_nightly_v0/report/nightly_summary.md`)
claims `NIGHTLY_SMOKES_COMPLETE`, 13 completed jobs, physical-space NLL "where
relevant", memory released, no leakage, all tests passing.

**A. Job count (00-13 vs "thirteen jobs").** `main()`'s sequential loop
(`run.py`, pre-patch line 1164) queues 14 jobs, 00 through 13
(`13_build_nightly_report`). `build_final_nightly_report()` kept an
independently hardcoded, 13-entry copy of the same list (pre-patch line 943)
that never included job 13 itself. This was **list drift, not a deliberate
design choice** — nothing in the code or docs said job 13 excludes itself on
purpose. **Fixed**: both now read a single `NIGHTLY_JOB_NAMES` constant.
Job 13 legitimately cannot observe its own `status.json` while it is the one
building this very report (the caller writes that file only *after*
`build_final_nightly_report()` returns), so it will always show `"missing"` in
its own Job Statuses row — this is now shown honestly rather than hidden, and
excluded from the pass/fail completeness gate (§5, patch 2/7) so that gate is
not permanently unsatisfiable.

Separately, `artifacts/afterms_nightly_v0/jobs/13_build_nightly_report/`
contains a **stray `metrics.json` and model checkpoint** for
`identity_standardized_v0_affine_small_unweighted` — actual neural-training
output filed under the report-builder's job name, with no `status.json`. This
is an artifact of some earlier invocation drift (the current code's `--run-job
13_build_nightly_report` dispatch always calls `build_final_nightly_report`,
never the generic training fallback), not something the current code would
reproduce. Left untouched (not overwritten/deleted — it's pre-existing data
outside this mission's authority to discard), but the report builder now
explicitly skips job 13 when rendering the performance table, so this stray
file can never again masquerade as a model result (§5, patch 3b).

**B. `physical_space_nll = 0.0000` on the legacy Quantile job.** Confirmed
**missing-value serialization bug** compounded by a **report-builder
formatting bug** — not a valid computed value and not a deliberate placeholder:

- Job 04 (`04_legacy_available_code_realnvp_quantile`) never attempted a
  Jacobian conversion and never wrote a `physical_space_nll` key at all
  (pre-patch `eval_metrics` dict had 5 keys, none of them `physical_space_nll`).
- Jobs 05/06 *do* attempt it and correctly get `None` back
  (`PreprocessingPipeline.forward_log_abs_det_jacobian` raises
  `NotImplementedError` for `quantile_normal_v0`,
  `src/ship_muon_bg/afterms/preprocessing.py:94-95`), which the multi-run
  report branch correctly rendered as `"N/A (No Jac)"`.
- The single-run report branch instead wrote `m.get('physical_space_nll') or
  0.0` (pre-patch `run.py` line 1040) — since the key was absent, `.get()`
  returned `None`, and `None or 0.0` silently became `0.0`.

**Fixed** (§5, patches 3a/3b): job 04 now explicitly serializes
`physical_space_nll = None` with a one-line rationale (QuantileTransformer has
no analytic Jacobian, same reason as `quantile_normal_v0`), and both report
branches now share one `None → "N/A (No Jac)"` formatting rule. Confirmed
end-to-end: regenerating the report from the existing job artifacts (no
retraining) now prints `N/A (No Jac)` for job 04, matching jobs 05/06's
`quantile_normal_v0` rows. Quantile-transform jobs correctly never received a
physical NLL on the same numeric axis as identity/log1p jobs, before or after
the fix — only the *rendering* of the missing value was wrong.

**C. Per-epoch schema completeness.** Confirmed **incomplete** for every
neural job (04-09): `epoch`, `train_loss`/`validation_loss` (feature-space,
not physical), memory/timing fields are present; `train_physical_nll`,
`validation_physical_nll`, `learning_rate`, and a per-epoch finite-loss flag
are **absent** from every job's `history`/`run_history`. A single
`finite_log_probability_rate` scalar exists, but it's computed once on the
*test* set post-training (`run.py`), not per epoch. This is a real gap left
**unfixed** in this patch (adding per-epoch physical NLL/LR/finite-flag
logging is new instrumentation, not a report-serialization fix, and the
mission scopes new epoch-logging as "if needed... do not invent historical
epochs" for *future* runs, not a retroactive backfill). Flagged as a
limitation for the next experiment (§7).

**D. Test NLL evaluated only after training.** Confirmed true for every job —
see Q10 above.

**E. Final test vs. final validation vs. best validation vs. last-epoch NLL
conflation.** Two real conflations found, one **fixed**, one **left as a
documented limitation**:

1. **Fixed** — Gaussian/GMM controls (jobs 10/11): pre-patch code assigned
   `train_feature_space_nll = validation_feature_space_nll =
   fit_res.best_validation_nll` verbatim (`run.py`, pre-patch lines 733-734).
   Empirically confirmed bit-identical in the pre-patch artifact
   (`jobs/10_gaussian_controls_pdg13/metrics.json`,
   `identity_standardized_v0_diagonal_gaussian_unweighted`:
   `train_feature_space_nll == validation_feature_space_nll ==
   7.095651093170661`). `Nflow.interfaces.FitResult.train_history` already
   carries a genuine, separately-computed `train_nll`
   (`Nflow/baselines/gaussian.py:110`, `Nflow/baselines/gmm.py:126`) that was
   simply never read. Fixed by extracting the real value via a new
   `train_and_validation_nll_from_fit_result()` helper and rerunning jobs
   10/11 (§5, patch 4; before/after values below).
2. **Documented limitation, not fixed** — for the affine-coupling branch,
   `validation_feature_space_nll` is always the *last epoch's* validation
   loss; there is no best-checkpoint tracking anywhere in this file, so
   "validation NLL" here always means last-epoch, never best-epoch. Given
   only 5 epochs are ever run, last-epoch and best-epoch are close in
   practice but not guaranteed identical, and a future champion-selection
   step (mission §7.1, "minimum best validation physical NLL") would need a
   genuine best-epoch tracker. Not implemented here — flagged for §7.

**F. CUDA vs. CPU.** Every job in this artifact set ran on CPU:
`jobs/00_environment_and_dataset_smoke/metrics.json` →
`system_info.torch_version = "2.13.0+cpu"`, `cuda_available: false`,
`gpu_name: null`. Device-resolution code exists and is correct for the affine
family (`Nflow/torch_models/affine_coupling.py` `_resolve_device()` maps
`"auto"`→CPU/CUDA correctly), but job 04's legacy path passes the raw
`--device` string straight to `model.to(device)` with no `"auto"` handling —
untested, but moot on this CPU-only host. UNVERIFIABLE: no record of what
`--device` string was actually passed on the invoking command line (not
persisted in `status.json`/`queue_state.json`).

**G. Test count and invariant coverage.** Pre-patch: 20 test functions across
4 files in `tests/afterms/`, covering shard/split/stratify/preprocessing
round-trip/watcher invariants — **but zero tests exercised
`run_afterms_nightly_queue.py` itself** (confirmed via
`grep -rl "run_afterms_nightly_queue" tests/` → no hits before this patch).
None of bugs B or E-1 above would have been caught by the pre-existing suite.
**Added 17 tests** in `tests/afterms/test_report_builder.py` directly
covering: job-list completeness, missing-physical-NLL → `"N/A"` not `"0.0"`
rendering, a **real** `physical_space_nll == 0.0` rendering as `"0.0000"`
(distinguishable from missing, added this session), status-code computed vs.
hardcoded, job-13 self-exclusion from its own performance row,
report-regeneration determinism, train/validation-NLL non-conflation (both
the conflated and the "fitter never recorded it" case), generated-domain-
violation counting and quantile reporting (violated, clean, and
no-clipping cases), the deterministic-sample-generation RNG contract
(fixed-seed → identical hash, different seed → different hash, hash is a
stable 64-char hex digest), `checkpoint_hash` changing with file content, and
one JSON/CSV/Markdown agreement check (the job's own `metrics.json` is the
source of truth; CSV and Markdown must render the same numbers, not
independently recompute them).
Full repository suite: **552 passed, 0 failed, 2 skipped** (post-patch, this
session).

**H. Fitting leakage.** Confirmed leak-free: `QuantileTransformer`/mean-std
fits only ever see `raw_train[:, :5]`
(`src/ship_muon_bg/afterms/preprocessing.py:29-53`), validation/test only ever
go through `.transform()`. `tests/afterms/test_integration.py::test_train_only_preprocessing_fit`
exercises this directly.

**I. Row-level A/B comparability.** All preprocessing/capacity/weight variants
within one job draw from the *same* `train_data`/`val_data`/`test_data` arrays
loaded once at the top of `run_neural_training_subprocess`
(`run.py`) — same row indices across every variant in a job. Confirmed by
reading the loop structure: the shard arrays are loaded before the
`for var_id in variants:` loop and never reloaded inside it.

**J. Generated samples silently clipped/dropped/repaired?** No — confirmed no
clipping code exists anywhere in the generation or inverse-transform path.
But this also means the domain-violation *diagnostic* the mission requires
was **entirely missing** before this patch (§4/§7's generated-support audit) —
added in §5, patch 5.

---

## 3. Mathematical Loss Contract

### 3.1 Unweighted target

`-mean(log q_theta(x_i))` over a batch — confirmed for every unweighted
run: `loss = torch.sum(t_w_train[idx] * -module.log_prob(t_train[idx])) /
torch.sum(t_w_train[idx])` with `t_w_train = torch.ones(n_train)` when
`w_policy=False` (`run.py`), which reduces exactly to
`mean(-log q_theta(x_i))`, i.e. empirical cross-entropy / mean NLL. Matches
the intended `L_hat(theta)`.

### 3.2 Weighted target

Exact implemented formula, training minibatch (`run.py`):

```
loss = sum(w_i * -log_prob(x_i)) / sum(w_i)     # computed fresh per minibatch
```

and validation (whole-tensor, same formula). This is **(B) per-minibatch
self-normalization**, not (A) global fixed normalization (no precomputed mean
training weight is used anywhere) and not (C) weighted resampling (sampling
is a plain unweighted `torch.randperm`; weights only enter the loss). The code
already self-labels this correctly:
`eval_metrics["weighted_estimand_scope"] = "self_normalized_minibatch_approximation"`
(`run.py`). Per the mission's own naming convention this is
`self_normalized_minibatch_ratio_estimator` — **already labeled, not claimed
as an exact globally-normalized weighted MLE**. No correctness defect found;
**not modified** in this patch (correctly out of scope per the mission's own
guard: no unit test demonstrates a defect in the estimator itself, only in
downstream report labeling elsewhere).

---

## 4. Coordinate-Space NLL Contract

| Preprocessing | Classification | Evidence |
|---|---|---|
| `identity_standardized_v0` | PHYSICAL_NLL_AVAILABLE | `preprocessing.py:90-93`: Jacobian `= -sum(log(std))`. Derivation: `z=(x-mean)/std` ⇒ `dz/dx=1/std` ⇒ `log|dz/dx|=-log(std)`; `log p_X(x) = log p_Z(z) - Σlog(std)`. Sign and combination (`physical_lp = test_lp + log_jac`, `run.py`) verified correct. |
| `cartesian_log1p_pz_v0` | PHYSICAL_NLL_AVAILABLE | `log1p_pz.py:60-64`: `log|du/dpz| = -log(s_pz + pz)`, confirmed against a numeric derivative by `tests/afterms/test_log1p_pz.py::test_jacobian_matches_numeric_derivative`. Chain-rule combination with the standardization Jacobian (`preprocessing.py:96-100`: `std_jac + view_jac`) is correct. Minor unverified-but-currently-harmless duplication: the view Jacobian is re-derived inline in `preprocessing.py` instead of calling `log1p_pz.forward_log_abs_det_jacobian`; both hardcode `s_pz=1.0` today so they agree, but nothing would catch a future desync. |
| `quantile_normal_v0` | FEATURE_NLL_ONLY | `preprocessing.py:94-95` explicitly `raise NotImplementedError(...)`. Never placed on the same axis as physical-NLL preprocessings in the report — confirmed both before and after this patch. |

Negative NLL values observed in the artifacts (e.g. job 04's
`test_feature_space_nll = -2.3071`) are expected for continuous densities and
are **not** treated as failures anywhere in this audit.

### 4.1 Generated-support audit (mission §5)

Before this patch: **MISSING**. No code computed
`generated_domain_violation_count/rate`, `min_generated_pz`, or generated-pz
quantiles anywhere (`evaluate_generated_samples`, `audit.py`,
`density_lab/metrics.py` all grepped — no matches). The only incidental signal
was a generic `marginal_summaries.pz.min` in `metrics.json`. QuantileTransformer
inverse-saturation diagnostics (fraction pinned to train min/max, duplicate
rate from saturation) were likewise **MISSING** — the closest available proxy
is the generic (non-saturation-aware) `duplicate_diagnostics` self-comparison.

**Added** (§5, patch 5): `evaluate_generated_samples` now returns a
`generated_domain_violations` dict with exactly the four fields the mission
names (`generated_domain_violation_count`, `_rate`, `min_generated_pz`,
`quantiles_of_generated_pz`), computed from the already-generated physical-
space sample's `pz` column (index 2, same column `exceedance_counts` already
used). No clipping/repair was added — this only counts and reports.
QuantileTransformer inverse-saturation diagnostics remain **MISSING** — out of
scope for this patch (a new statistic, not a report-serialization fix);
flagged for the next experiment (§7).

This diagnostic applies to every future run of `evaluate_generated_samples`
automatically. It was **not backfilled** onto the already-produced
`cartesian_log1p_pz_v0` artifacts from jobs 05/06/09 (that would require
reloading each checkpoint, refitting each job's preprocessing pipeline on its
train shard, and regenerating samples — a bounded but nontrivial expansion of
scope the mission's own discipline ("do not build the section 6-12 machinery
now") argues against doing inline here). Recorded as missing evidence in §8.

---

## 5. Patch Applied (this mission, part B)

All changes are in `scripts/run_afterms_nightly_queue.py` plus one new test
file `tests/afterms/test_report_builder.py` (10 tests). No other files
touched. Nothing in `src/ship_muon_bg/afterms/` or `Nflow/` was modified — the
estimators, Jacobians, and split/stratify code were all found correct.

1. **Single source of truth for the job queue.** New `NIGHTLY_JOB_NAMES`
   constant (14 entries, 00-13) replaces two independently-hardcoded, drifted
   copies in `main()` and `build_final_nightly_report()`.
2. **`status_code` is computed, never hardcoded.** The markdown report used to
   `f.write("...NIGHTLY_SMOKES_COMPLETE...")` unconditionally regardless of
   the JSON summary's actual computed `status_code`. Both now reference the
   same computed `status_code` variable. Completeness is judged over the 13
   substantive smoke jobs (00-12); job 13 is excluded from that specific gate
   because it structurally cannot observe its own completion while building
   this report (verified: including it made `status_code` permanently
   `NIGHTLY_SMOKES_PARTIAL` — caught by rerunning the report builder against
   real artifacts before finalizing this fix).
3. **`physical_space_nll` missing-value handling unified.**
   - (a) Job 04 now explicitly writes `physical_space_nll = None` (was: key
     absent entirely).
   - (b) Both report-rendering branches (single-run and multi-run) now share
     one `None → "N/A (No Jac)"` rule; job 13 is explicitly excluded from the
     performance table (it is the report-builder, never a model run — this
     also prevents the stray orphaned artifact under its job directory from
     appearing as a fake result row).
4. **Gaussian/GMM train NLL no longer copies validation NLL.** New pure
   helper `train_and_validation_nll_from_fit_result(fit_res)` reads the
   genuine `train_nll` from `FitResult.train_history` instead of duplicating
   `best_validation_nll` into both fields. Jobs 10 and 11 were rerun (≤20s
   each) since the correct train NLL was never actually computed and stored
   before — not a retraining of anything expensive, a recomputation of a
   value that was silently wrong. Before → after (feature-space NLL):

   | Run | train (before) | train (after) | validation (unchanged) |
   |---|---|---|---|
   | 10 · diagonal_gaussian | 7.095651 | 7.094693 | 7.095651 |
   | 10 · full_gaussian | 5.438203 | 5.440643 | 5.438203 |
   | 10 · gaussian_mixture | 2.759708 | 2.765228 | 2.759708 |
   | 11 · diagonal_gaussian | 7.103522 | 7.094693 | 7.103522 |
   | 11 · full_gaussian | 5.445076 | 5.438504 | 5.445076 |
   | 11 · gaussian_mixture | 2.763289 | 2.775229 | 2.763289 |

5. **Generated-domain-violation diagnostic added** to
   `evaluate_generated_samples` (see §4.1). Purely additive key; does not
   change any existing field.
6. **Sample generation made deterministic.** `torch.randn(...)` in the affine
   path now passes `generator=gen` (the already-seeded generator used for
   minibatch order); job 04's `model.base_dist.sample(...)` is preceded by an
   explicit `torch.manual_seed(20260720)`. Neither call had any seed control
   before, meaning re-running the same checkpoint's sampling code twice would
   not reproduce the same generated sample — a real gap against "deterministic
   model and preprocessing comparisons" (Q5) and against the mission's later
   sample-matrix requirement ("store exact seeds"). **Verified end-to-end**
   (not just compiled): loaded the real
   `09_affine_capacity_smoke_pdg13/identity_standardized_v0_affine_tiny_unweighted_model.pt`
   checkpoint, ran the exact fixed generation line
   (`torch.randn(5000, 5, ..., generator=gen)` → `module(z)`) twice with a
   freshly-seeded generator each time, and confirmed the two generated
   samples are bit-identical (`np.array_equal`, all 25,000 floats). This is
   the one edit none of the automated tests exercise directly against a real
   checkpoint (no test drives the affine training loop itself), so it was
   also checked manually as above.
7. **Job 13 shown honestly in its own status table**, not silently excluded.
8. **Generation provenance recorded, not just seeded (this session).** Patch 6
   made sampling deterministic but recorded no evidence of it. New
   `generated_sample_hash(samples)` (sha256 of the sample array's raw bytes)
   and `checkpoint_file_hash(path)` (sha256 of the saved checkpoint file)
   helpers are now called at every one of the three sample-generation sites
   (job 04 legacy, affine family, Gaussian/GMM family). Every job's
   `eval_metrics` now carries `generation_seed` (`20260720`, the value already
   used everywhere -- this only makes it visible), `generated_sample_count`
   (`5000` everywhere today), `generated_sample_hash`, and `checkpoint_hash`
   (`None` for the Gaussian/GMM family, which saves no checkpoint file at
   all -- an honest gap, not fabricated). Purely additive keys; no existing
   field changed or renamed. Verified via a pure RNG-contract test
   (`torch.Generator` + fixed seed -> identical `generated_sample_hash`;
   different seed -> different hash; hash is a stable 64-char hex digest)
   rather than by driving the full training subprocess. No jobs were rerun
   this session to backfill these fields onto existing `metrics.json` files
   -- they will appear starting with the next real training invocation.

The real `artifacts/afterms_nightly_v0/report/` files were regenerated from
existing job artifacts (no retraining involved in the report regen itself —
verified reproducible by running `build_final_nightly_report` twice on the
same inputs, byte-identical output, per the new `test_report_regeneration_from_existing_artifacts_is_deterministic` test). The pre-patch versions are preserved at
`artifacts/afterms_nightly_v0/jobs/{10,11}_.../metrics.json` git history and a
local backup was diffed above; nothing was overwritten silently — this
document records every changed value.

Re-verified this session: `--run-job 13_build_nightly_report` was invoked
twice in a row against the real `artifacts/afterms_nightly_v0/` directory
(no training, per mission §8) and produced byte-identical
`nightly_summary.json`/`nightly_summary.md`/`nightly_results.csv` both
against each other and against the pre-session copies of those files —
confirming report regeneration is still training-free and reproducible after
this session's edits, and that the new provenance fields (item 8 above) do
not alter existing report output for jobs whose `metrics.json` predates
them.

**Verification:** `python -m pytest -q` → **552 passed, 0 failed, 2 skipped**
(full repository suite, this session; was 545 before this session's 7 new
tests).

---

## 6. What was deliberately NOT built in this mission

Per the mission's own discipline ("implement only the smallest
evaluation/reporting patch justified by the audit"; "do not begin by
rewriting code"), the following mission sections describe target-state
machinery that was **not** started, because building it on top of
metrics that were still mis-serialized (§2B, §2E) would have been premature:

- Training-curve figures (§6), model arena artifacts (§7), sample matrices
  (§8), 1D/2D/N-D statistical test suites (§9-11).
- QuantileTransformer inverse-saturation diagnostics (§4.1).
- Per-epoch physical NLL / learning-rate / finite-loss-flag logging (§2C).
- Best-validation-epoch checkpoint tracking for the affine family (§2E-2).
- Backfilling the new generated-domain-violation diagnostic onto the
  already-trained `cartesian_log1p_pz_v0` checkpoints from jobs 05/06/09.

These are explicitly deferred to the next experiment (§8 below), not silently
dropped.

---

## 7. Missing Evidence / UNVERIFIABLE

- Exact `--device` string used for the original nightly invocation (not
  persisted anywhere in `status.json`/`queue_state.json`); moot since the host
  is CPU-only PyTorch, but not reconstructible from artifacts alone.
- Whether checkpoints for jobs 04-09 (the expensive ones) would still load
  correctly if `run.py`'s hardcoded capacity-spec table (`hidden_width`/
  `number_of_blocks`/`hidden_depth` per capacity name) is ever edited —
  nothing hashes or verifies this today (§1, Q7).
- Generated-domain-violation figures for the already-completed
  `cartesian_log1p_pz_v0` runs (05/06/09) — the diagnostic now exists in code
  but was not backfilled onto those checkpoints (§4.1).
- QuantileTransformer inverse-saturation rates for any completed run — no
  code computes this anywhere yet.
- `generation_seed`/`generated_sample_count`/`generated_sample_hash`/
  `checkpoint_hash` (§5, patch 8) on the already-produced `metrics.json` files
  for jobs 04-12: the recording code now exists, but these existing artifacts
  were written by the pre-patch-8 code and predate the new keys. They will
  only appear starting with the next real training invocation of each job.

## 8. Minimal Next Experiment

Given metrics are now trustworthy (physical/feature NLL correctly
distinguished and nulled, train/validation NLL no longer conflated for
Gaussian/GMM, sampling now deterministic), the smallest next step that
extends real value is:

1. Backfill `generated_domain_violations` for the existing
   `cartesian_log1p_pz_v0` checkpoints (jobs 05, 06, 09) via checkpoint
   reload + deterministic re-sampling — no retraining required, now that
   sampling is seeded (§5, patch 6). This directly answers whether the
   log1p-pz flow ever inverts to `pz < 0` in practice, which is the single
   open physically-meaningful question this audit could not close.
2. Only after (1), begin the model-arena/training-curve build-out (mission
   §6-7), gated explicitly on "metrics must be trustworthy first" — which is
   now true for the fields this patch touched, but not yet for per-epoch
   physical NLL or best-checkpoint tracking (§2C, §2E-2), which the arena's
   champion-selection logic (mission §7.1) will need.

---

## 9. Exact Non-Claims

- This audit does not claim any model here is scientifically superior to
  another — no ranking or arena exists yet.
- Fixing the Gaussian/GMM train-NLL bug does not change any validation or
  test NLL, and does not change which preprocessing "looks better" — it only
  corrects a mislabeled train-set number that was never used in any
  comparison logic (confirmed, §1 Q10).
- The weighted-loss estimator was audited and found correctly implemented and
  already correctly labeled; it was not touched.
- "Deterministic model comparisons" (Q5) is now true for training and, after
  this patch, for future sampling — but historical samples from jobs 04-09
  generated before this patch are **not** reproducible after the fact (the
  seed fix only affects code going forward).
- The new `generation_seed`/`generated_sample_hash`/`checkpoint_hash` fields
  (§5, patch 8) are verified as a pure RNG/hash-function contract (fixed
  seed → same hash, different seed → different hash, hash is stable), not by
  re-running a real training job end-to-end this session — no training was
  performed this session, per mission constraints.
- No claim is made that jobs 04-12's existing `metrics.json` artifacts
  contain the new provenance fields; they do not, since those files predate
  this session's patch (see §7).

---

## Required Final Verdict

**CURRENT_IMPLEMENTATION_NEEDS_REPORTING_PATCH**

Rationale: every estimator, Jacobian, split, and leakage invariant audited
was implemented correctly (§1 Q3, Q4, Q6, Q10; §3; §4). The defects found
clustered entirely in the reporting/serialization layer — a hardcoded status
string, a missing-vs-zero rendering bug, list drift between two copies of a
job queue, and (the one item that could have tipped this toward a metric
defect) a copy-paste that put a validation-set number into a train-set field.
That last one was a **value bug**, not an estimator bug: the underlying
`fit_res.train_history["train_nll"]` was already being computed correctly by
the fitter and simply never read — no scientific conclusion in the repo or
report depended on the wrong copy (test NLL, which is what any future
comparison would use, was computed independently and was never affected).
With that fixed and verified by rerunning the two cheap jobs it affected, the
smoke pipeline's numbers are now internally consistent and honestly labeled,
which is what a reporting patch is for.

---

VERIFIED COMMIT: `4976067` (pre-patch HEAD of `experiment/d7-afterms-sharded-smoke-v0`); patch committed this session as `5e37c92` ("fix(afterms): correct nightly report serialization") on top of it, plus this documentation commit

FILES REVIEWED: `scripts/run_afterms_nightly_queue.py`, `scripts/watch_afterms_nightly_queue.py`, `scripts/build_afterms_shards.py`, `src/ship_muon_bg/afterms/{preprocessing,log1p_pz,split,stratify,audit}.py`, `src/ship_muon_bg/density_lab/feature_pipeline.py`, `Nflow/interfaces.py`, `Nflow/torch_models/affine_coupling.py`, `Nflow/baselines/{gaussian,gmm}.py`, `Nflow/registry.py`, `tests/afterms/*.py`, `tests/test_affine_coupling_flow.py`, `tests/test_density_metrics.py`

ARTIFACTS REVIEWED: `artifacts/afterms_nightly_v0/` in full (all 14 job directories, `report/`, `plots/`, `queue_state.json`, `watch_packet.{json,md}`, `afterms_audit.json`); `data/shards/afterms_nightly_v0/shard_validation_report.md`

CLAIMS CONFIRMED: row-disjoint leak-free splits (train-only fitting); weighted-loss estimator correctly implemented and self-labeled; identity/log1p physical-NLL Jacobians correct in sign and combination; quantile correctly refused a physical NLL; test NLL computed once, never used for selection; no clipping/repair of generated samples; CPU-only execution this run; checkpoints loadable and log_prob-deterministic (spot-checked)

CLAIMS REJECTED: "NIGHTLY_SMOKES_COMPLETE" was a hardcoded literal, not a computed verdict; "physical_space_nll = 0.0000" for legacy Quantile was a missing-value-rendered-as-zero bug, not a valid computed value; "13 jobs" undercounted the queue by 1 due to list drift; train NLL for Gaussian/GMM controls was silently identical to validation NLL (copy-paste bug); generated-sample domain-violation diagnostics claimed implicitly available were entirely absent from the codebase before this patch

BUGS FOUND: (1) hardcoded status-code literal ignoring computed value; (2) job-13 list drift between two hardcoded job lists; (3) missing→zero physical-NLL rendering for job 04, compounded by job 04 never writing the key; (4) Gaussian/GMM train NLL conflated with validation NLL; (5) generated-sample domain-violation diagnostic entirely missing; (6) non-deterministic sample generation (no seed) in the affine and legacy-Quantile paths, and (this session) no provenance recording of the seed/sample-hash/checkpoint-hash even after seeding; (7) stray orphaned metrics.json/checkpoint under job 13's own directory (artifact-state issue, not a live code bug, now inert since the report builder skips job 13's row)

PATCH COMMITS: `5e37c92` — "fix(afterms): correct nightly report serialization" (`scripts/run_afterms_nightly_queue.py`, `tests/afterms/test_report_builder.py`); a second commit follows this document with the finalized audit + runbook

TEST RESULT: 552 passed, 0 failed, 2 skipped (full repository suite, this session); 37 tests now in `tests/afterms/` (20 pre-existing + 17 in `test_report_builder.py`, up from 10)

RUNBOOK: `docs/reviews/afterms_manual_execution_runbook_v0.md` — copy-paste-ready PowerShell commands for environment setup, focused/full tests, `--dry-run`, partial and full training runs, resume/force/stop-after, the watcher, training-free report regeneration, artifact location, and GPU/process-liveness checks. Every flag was derived from `--help` output this session, not invented.

RETRAINING PERFORMED: jobs 10 and 11 only (Gaussian/GMM controls, ≤20s per run each), to populate a train-NLL value that had never actually been computed before; no affine-coupling or legacy job was retrained

ARENAS CREATED: none (explicitly deferred, §6)

PROVISIONAL CHAMPIONS: none (explicitly deferred, §6)

UNRESOLVED SCIENTIFIC LIMITATIONS: no muon-lineage group id (row-disjointness ≠ source-independence, pre-existing and documented); no best-validation-epoch tracking for the affine family; no per-epoch physical NLL/learning-rate/finite-loss logging; no QuantileTransformer inverse-saturation diagnostic; generated-domain-violation diagnostic not yet backfilled onto existing log1p-pz checkpoints; checkpoints are not self-describing (no bundled config/hash) so reload correctness depends on the runner script's hardcoded capacity table staying in sync

NEXT MINIMAL EXPERIMENT: backfill `generated_domain_violations` onto the existing `cartesian_log1p_pz_v0` checkpoints (05/06/09) via checkpoint reload + deterministic re-sampling (no retraining), to close the one open physically-meaningful question this audit could not answer — whether the log1p-pz flow actually generates `pz < 0` in practice; only after that, begin the model-arena/training-curve build-out gated on the per-epoch physical-NLL and best-checkpoint instrumentation this audit found missing.
