# D9 conditional-charge sampling v1

## Scope and lineage

This branch starts at the verified `experiment/d9-conditional-charge-nf-v0` tip
`eea984e77b5d4767c6edee5545a007476cc4e297` ("docs(d9): record conditional
charge fixture pilot"). It does not merge the historical row-empirical D9-5N
branch and does not modify any closed Phase 0/Phase 1 result.

The implementation remains fixture-only: it does not train on the 13.8M-row
full dataset, run utility tilting, call FairShip/GEANT4, estimate a physical
rate, or evaluate a model on the test split. This document corrects four
scientific contract defects identified in the v0 pilot
(`docs/reviews/d9_conditional_charge_nf_v0.md`) before any full-data gate.

## 1. Why the v0 condition name was ambiguous

v0 declared the model condition as `charge_sign` with
`CONDITION_BY_PDG = {13: +1.0, -13: -1.0}` -- i.e. the *PDG numeric sign*.
PDG codes are particle/antiparticle labels, not charge labels: PDG `13`
names the mu- (physical electric charge **-1**) and PDG `-13` names the mu+
(physical electric charge **+1**). For this particle pair the PDG numeric
sign is the exact *opposite* of the physical charge sign, so a field named
`charge_sign` carrying the PDG sign silently encoded the wrong physical
quantity under a name that reads as "the charge."

## 2. Corrected physical charge convention

The condition is renamed `muon_electric_charge_sign` everywhere (source
constant `CONDITION_FIELD_NAME`, config field `model.condition_name`,
manifests, tests, this document) and now carries the physical charge:

```
PDG 13  (mu-) -> muon_electric_charge_sign = -1
PDG -13 (mu+) -> muon_electric_charge_sign = +1
```

`MUON_ELECTRIC_CHARGE_SIGN_BY_PDG = {13: -1.0, -13: 1.0}` in
`src/ship_muon_bg/density_lab/conditional_charge.py` is the single source of
this mapping. No field named `charge_sign` under the old PDG-sign convention
remains in any v1 code path, config, or artifact field; the v0 config files
and doc are left untouched as an immutable historical record of what was
run, but they are no longer accepted by the v1 validator (`schema_version`
and `condition_name` both changed).

This is a **label-convention fix, not a physical symmetry claim or a hard
coordinate transform**: it changes which scalar is attached to which PDG
track, nothing about the model, the Jacobian, or the training objective.
`tests/test_conditional_charge_nf.py::test_charge_mapping_uses_physical_charge_not_pdg_numeric_sign`
is the regression test:

```python
assert muon_electric_charge_sign(13) == -1.0   # mu-
assert muon_electric_charge_sign(-13) == 1.0   # mu+
```

## 3. Exact train-only macro-weighted preprocessing

v0 fit the standardization on the *resampled draw pool* (a seed-dependent
sample), not on the train partitions directly. v1 adds
`FittedFeaturePipeline.fit_macro_weighted` (`src/ship_muon_bg/density_lab/feature_pipeline.py`),
fit once, directly, on the full train partitions of both charges:

```
pi_i|c = w_i / sum_{j: C_j=c} w_j

mu       = (1/2) * sum_c sum_i pi_i|c * x_i
variance = (1/2) * sum_c sum_i pi_i|c * (x_i - mu)^2
```

Properties, all verified by test:

- train partitions only, never validation or test;
- no sampler-seed dependence (the formula is a closed-form sum, not a draw);
- one shared scaler for both charges;
- finite, strictly positive scales (`FittedFeaturePipeline.__init__` now
  asserts this unconditionally);
- deterministic preprocessing hash (`pipeline.config_hash()`);
- preserves the five ordinary physical generated features `[px, py, pz, x, y]`.

`tests/test_density_feature_pipeline.py` adds exact small-array tests for
the weighted mean/variance formula (hand-computed against a tiny two-charge
array) and a test proving each charge gets an equal 1/2 share regardless of
its row count (a charge with 2 rows and a charge with 5 rows contribute
equally to `mu`/`variance`, unlike a naive pooled/row-count-weighted mean).

Seed-11 fixture-study preprocessing hash:
`d858fb2f0dd0522c6bfa9098615dc6ef77264d32360d9d5f4222e15524c8aa8c`
(`weighting: macro_balanced_physical_weight_train_only`).

## 4. Per-epoch direct sampling derivation

v0 built one resampled draw pool *before* training and reused it as
`x_train` for every epoch (`Nflow.torch_models.trainer.train_flow`'s legacy
path just re-permutes the same fixed tensor). v1 adds a new trainer arm
("arm D", `epoch_sampler` in `Nflow/torch_models/trainer.py`) that draws a
*fresh* macro-balanced sample every epoch:

```
I_{e,t}|c ~ pi_i|c   for each epoch e, each charge c
```

with a deterministic derived seed from `(global_seed, epoch, pdg_id)`
(`_derived_epoch_seed` in `conditional_charge.py`); `_EpochDirectSampler`
holds one `AliasSampler` per charge (built once, from the train partition
only) and calls `.draw(...)` fresh on every `epoch_sampler(epoch)`
invocation. Each epoch draws exactly equal charge counts
(`draws_per_epoch_per_charge` per charge), with replacement, using the
train-only physical-weight measure, macro-balanced between charges. No
sample weight is ever attached to the drawn rows -- the loss on them is
ordinary unweighted NLL (`sample_weight_applied_to_loss: false` throughout).

This makes the per-step gradient an unbiased estimator of the macro
population gradient:

```
E_epoch_draws[gradient estimator] = (1/2) * grad L_13 + (1/2) * grad L_-13
```

**This is a claim about one stochastic-gradient step, not about the final
trained parameters.** SGD over a nonconvex loss has no general guarantee
that the fixed point of many such steps is itself an unbiased estimator of
anything; v0's "fixed resample reused across all epochs" was a strictly
worse approximation (all epochs see the exact same 4,096-row realization,
so its sampling noise never averages out across epochs) -- v1 does not
claim to have removed all approximation, only the epoch-reuse defect.

Recorded per epoch (`sampling_manifest.epochs[i]`, and per training-history
record via `epoch_sampler_metadata`): per-charge `draw_hash`,
`source_probability_table_hash`, `alias_table_hash`, `unique_rows_drawn`,
`unique_rows_fraction`, `max_reuse_count`, `cumulative_unique_source_rows`,
plus `charge_counts` and `sample_weight_applied_to_loss: false`. The
Adam optimizer state persists across epochs exactly as before (only the
per-epoch *data* is fresh, never the model or optimizer); the trainer had no
existing epoch-boundary resume mechanism to preserve, so this is unaffected.

Regression tests in `tests/test_conditional_charge_nf.py` and
`tests/test_affine_coupling_flow.py` verify: within-charge `pi`
reproduction and zero-weight exclusion; distinct draws/hashes across epochs
from the same sampler; identical epoch-0 draws from two freshly-built
samplers with the same seed; non-decreasing, bounded cumulative unique-row
coverage; that the trainer calls `epoch_sampler` exactly once per epoch with
the right epoch index; that `epoch_sampler` is mutually exclusive with
`batch_plan` and with `sample_weight`; and (`@pytest.mark.slow`) that
optimizer state genuinely persists across epochs of fresh draws (loss trend
improves over 60 epochs of i.i.d. resampling from a fixed population, which
could not happen if the model or optimizer were silently reset each epoch).

## 5. Test-payload closure (Gate E stays closed)

v0's `build_empirical_dataset` always materialized `dataset.test.raw` for
both charges, even though nothing read it -- the array existed in memory.
v1 adds `build_empirical_train_validation_dataset`
(`src/ship_muon_bg/density_lab/empirical.py`): identical
load -> validate -> PDG-filter -> cap -> three-way-split path (refactored
into a shared `_load_filter_cap`/`_build_three_way_split` so both loaders
use one implementation), but `capped` is only ever indexed by
`train_indices`/`val_indices`; `test_indices` is used **solely** to compute
`len(test_indices)` and `canonical_hash({"test_indices": ...})` -- no test
feature value is ever read. The returned `EmpiricalTrainValidationDataset`
has no `test` attribute at all (structurally cannot hold test rows).

Recorded: `test_row_count`, `test_split_hash`,
`test_payload_loaded: false`, `test_used_for_training: false`,
`test_used_for_preprocessing: false`, `test_used_for_model_selection: false`,
`test_used_for_evaluation: false` (per charge, in `summary["test_provenance"]`
and `summary["lineage"]["charges"][pdg]`).

`tests/test_empirical_train_validation_dataset.py` proves closure
mechanistically (`not hasattr(dataset, "test")`) and causally: two synthetic
fixture files identical except for feature values at the exact (independently
precomputed) test-row positions produce **bit-identical** train/validation
partitions and an identical test row count/split hash through the closed
loader, while the same mutation is independently confirmed to change the
*full* (test-materializing) loader's `test.raw` -- i.e. the mutation is real,
and the closed path is provably blind to it. (`source_file_dataset_hash`,
a whole-file identity hash computed before any split is applied, is
explicitly excluded from that comparison -- it is file provenance, not
declared test-partition provenance, and predates this contract.)

## 6. Validation contract (unchanged, exact)

Validation is never resampled. For each charge:

```
L_val,c = -sum_i w_i log q_theta(x_i|c) / sum_i w_i
```

reported alongside the macro mean and worst charge; per-charge weighted
empirical summaries, generated summaries, weighted/generated correlation
matrices, train-only `B_toy` prevalence, generated `B_toy` occupancy, and
finite fractions (`compact_distribution_summary`, unchanged from v0).
`B_toy` thresholds remain train-only and charge-specific
(`_EpochDirectSampler` fits them once from each charge's train partition).

## 7. Three-seed fixture study

Fixture: `data/samples/muonsFullMC_afterMS_sample.npz` only. CPU. One shared
architecture for both the smoke and the study: 6 coupling blocks, hidden
width 64, hidden depth 2, batch size 256
(`configs/density_lab/conditional_charge/d9_fixture_smoke_v1.json`,
`d9_fixture_study_v1.json`). Study: `max_rows_per_charge=16000`,
`draws_per_epoch_per_charge=4096` (even, identical across all three seeds),
`max_epochs=30`, `patience=5`, `early_stopping=true`.

### A. Two-epoch smoke

`status=completed_fixture_pilot`, `fit.status=ok`, 2 history steps,
`validation.macro_nll=9.4662`, `worst_charge_nll=9.5361`. Test provenance:
`test_row_count=102` per charge, `test_payload_loaded=false` (both charges
share the fixture's row budget and seed, so an identical `test_split_hash`
across charges is expected -- the split depends only on row count/seed, never
on content). This is a wiring smoke test only, not a capacity result.

### B. Corrected 30-epoch study, seeds 11/12/13

All three seeds converged (`fit.status=ok`) and all three early-stopped
before the 30-epoch cap (patience 5): seed 11 at epoch 27 (best step 21,
0-indexed), seed 12 at epoch 21 (best step 15), seed 13 at epoch 25 (best
step 19). Per-seed, per-charge:

| seed | PDG | condition | final weighted val NLL | nominal B_toy | generated B_toy | ratio gen/nom | cumulative unique rows | last-epoch max reuse |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 11 | 13  | -1 | 5.7472 | 0.024185 | 0.031250 | 1.2921 | 9547 / 10240 (93.2%) | 19 |
| 11 | -13 | +1 | 5.8611 | 0.026383 | 0.011230 | 0.4257 | 9131 / 10240 (89.2%) | 19 |
| 12 | 13  | -1 | 5.1397 | 0.023518 | 0.017090 | 0.7267 | 9037 / 10240 (88.3%) | 21 |
| 12 | -13 | +1 | 5.1667 | 0.021430 | 0.021973 | 1.0253 | 8625 / 10240 (84.2%) | 20 |
| 13 | 13  | -1 | 5.1026 | 0.027061 | 0.020996 | 0.7759 | 9446 / 10240 (92.2%) | 18 |
| 13 | -13 | +1 | 4.6322 | 0.027540 | 0.016602 | 0.6028 | 8887 / 10240 (86.8%) | 16 |

"Last-epoch max reuse" is the largest number of times any single source row
was drawn within that seed's final recorded epoch's
`draws_per_epoch_per_charge=4096` draws (final epoch = 27/21/25 for seeds
11/12/13 respectively; see the early-stopping detail above).

Aggregate (median / min / max across the three seeds):

| metric | PDG 13 | PDG -13 |
|---|---:|---:|
| final weighted validation NLL | 5.1397 / 5.1026 / 5.7472 | 5.1667 / 4.6322 / 5.8611 |
| generated B_toy occupancy | 0.02100 / 0.01709 / 0.03125 | 0.01660 / 0.01123 / 0.02197 |
| generated/nominal B_toy ratio | 0.7759 / 0.7267 / 1.2921 | 0.6028 / 0.4257 / 1.0253 |

Macro validation NLL across seeds: median 5.1532, min 4.8674, max 5.8042.
`generated_sample_finite_fraction` and `generated_log_prob_finite_fraction`
were `1.0` for every seed and charge -- no non-finite generation.

Full per-epoch draw hashes, per-charge alias/probability-table hashes, and
`n_eff`/expected-unique-draws diagnostics are in each seed's
`conditional_sampling_manifest.json` (not committed; see Reproduction).

### Does the correction improve nominal tail reproduction?

Both directions occur across seeds: the generated/nominal `B_toy` ratio
ranges from 0.43 to 1.29 (three seeds, two charges = six ratios), i.e. no
seed produced dramatic B_toy over- or under-coverage in one consistent
direction, but no seed reproduced the nominal tail occupancy tightly either
(only one of six ratios, PDG -13 seed 12, lands within 3% of 1.0). **This
three-seed CPU fixture run is not sufficient to conclude whether the v1
correction improves nominal tail reproduction relative to v0 or a
non-conditional baseline** -- v0's pilot used a different (10-epoch, smaller
draw budget, buggy-condition) configuration, so the two are not a clean A/B
comparison, and three seeds is too small a sample to separate genuine
capacity behavior from seed noise. See OPEN below.

## 8. Reproduction

```bash
python scripts/run_conditional_charge_nf.py \
  configs/density_lab/conditional_charge/d9_fixture_smoke_v1.json \
  --output-dir artifacts/density_lab/d9_conditional_charge_sampling_v1/smoke

python scripts/run_conditional_charge_multiseed_v1.py \
  configs/density_lab/conditional_charge/d9_fixture_study_v1.json \
  --output-root artifacts/density_lab/d9_conditional_charge_sampling_v1/study \
  --seeds 11 12 13
```

Neither script nor config touches `data/full` or any full-data path; both
are hard-refused by `_validate_config` unless `dataset_path` resolves under
`data/samples/*_sample.*`.

## 9. Lineage

- Parent commit: `eea984e77b5d4767c6edee5545a007476cc4e297`.
- Fixture source hash: `718b9197a3965f6842574fec1f3f6b1bf4c16949ff8a1f27dca6232817c03c32`
  (`data/samples/muonsFullMC_afterMS_sample.npz`, both charges).
- Seed-11 study preprocessing hash:
  `d858fb2f0dd0522c6bfa9098615dc6ef77264d32360d9d5f4222e15524c8aa8c`.
- Seed-11 study model config hash:
  `d30b2329f49ba428d323963a75a27e683dd3debb05eb04ad5f4cbd4ed43d5df0`.
- Seed-11 study checkpoint hash:
  `3a943750981ae7d39c4e6728397baa70a093fcc94f5fb143397681c37fcc7732`.
- Seed-11 study test row count: 3,200 per charge; test split hash
  `1db5e13b5012e43a6e3b7ea93f6957699270dd85a0451cc5c45fcf72d8e8a720`.

Generated artifacts (`conditional_sampling_manifest.json`,
`conditional_training_metrics.json`, `per_charge_validation.json`,
`per_charge_generated_summary.json`, `symmetry_audit.json`,
`conditional_fixture_summary.json/.csv/.md`, `multiseed_study_summary.json`)
are not committed -- no generated samples, checkpoints, or large artifacts
are added to the repository, matching v0 practice.

## Authoritative empirical contract (preserved, unchanged)

- `w_mc` defines the current empirical nominal measure used throughout this
  document and code; nothing here redefines it.
- Utility remains separate: this branch introduces no utility tilting
  anywhere in the conditional-charge path (`symmetry_audit.transform` stays
  refused to anything but `"none"`; no `h_alpha_delta`/`pi_tilt` call exists
  in `conditional_charge.py`).
- Exact downstream physical-rate weight semantics remain **OPEN** --
  unchanged from v0; this branch does not estimate a physical rate.
- A technical failure (e.g. a fit returning `status != "ok"`) is never
  recorded as a physics negative; `run_fixture_pilot` raises
  `ConditionalChargeError` rather than silently reporting a scientific
  result.
- FairShip/GEANT4 remains the final oracle; nothing here calls FairShip or
  claims to substitute for it.

## Status boundaries

- **VERIFIED:** branch provenance from the exact required parent SHA;
  physical charge-sign condition (`muon_electric_charge_sign`, PDG
  13 -> -1, PDG -13 -> +1) with a regression test; exact train-only
  macro-balanced preprocessing (closed-form, not resampled, not
  seed-dependent) with exact small-array tests; per-epoch direct sampling
  with deterministic per-`(seed, epoch, pdg_id)` draws, verified distinct
  across epochs and reproducible across fresh sampler instances; persistent
  optimizer state across epochs of fresh draws; test payload never
  materialized for either charge, verified both structurally and by a
  content-mutation-invariance test; finite two-epoch smoke and 30-epoch
  three-seed study, all seeds `fit.status=ok`, all generated samples and
  log-probs finite; full local test suite and focused conditional/sampler/
  preprocessing/trainer tests pass (one pre-existing, unrelated
  environment-precision failure -- see Test results below); deterministic
  compact artifacts.
- **PROJECT DECISION:** one shared conditional model; `muon_electric_charge_sign`
  as the condition name and physical-charge convention; macro-balanced
  physical-nominal direct sampling with equal per-epoch charge counts; no
  sample weight applied to the loss (ordinary unweighted NLL after
  sampling); no double weighting; no hard symmetry; no utility tilting.
- **PROVISIONAL:** three CPU fixture seeds are sufficient to test execution,
  per-epoch resampling correctness, conditioning, per-charge stability, and
  diagnostic production under the corrected contract. They do not establish
  physical superiority, a symmetry claim, or a conclusive answer to whether
  the correction improves nominal tail reproduction (six generated/nominal
  `B_toy` ratios spanning 0.43-1.29 across three seeds and two charges is
  directionally inconclusive; see section 7).
- **OPEN:** whether the v1 correction quantitatively improves nominal tail
  reproduction versus v0 or a non-conditional baseline (no clean A/B
  comparison exists yet); a larger multiseed or architecture-varied study;
  full-data training; conditional-vs-separate-model scientific comparison;
  an evidence-backed charge-reflection map (`symmetry_audit.evidence_status`
  stays `OPEN`); exact downstream physical-rate weight semantics; FairShip
  validation; any physical-rate estimate.

## Test results

Full local suite: `python -m pytest -q` -> 710 passed, 3 skipped (2 require
the optional `mlflow` extra, 1 `reference_golden` outside its frozen
environment, `SHIP_RUN_REFERENCE_GOLDEN` unset), 1 failed
(`tests/test_rare_aware_estimators.py::test_golden_arm_a_portable_functional_regression`,
a ~1.2e-4 float32 mismatch against a `pytest.approx(..., abs=1e-5)`
tolerance). This failure is **pre-existing and unrelated to this branch**:
verified by `git stash` and re-running the identical test against the
unmodified parent commit `eea984e77b5d4767c6edee5545a007476cc4e297`, where it
fails identically (same obtained/expected values) -- an environment/torch-build
floating-point drift against the frozen golden value, not a regression
introduced here. No `xfail` is used anywhere to hide it.

Focused: `pytest tests/test_conditional_charge_nf.py
tests/test_density_feature_pipeline.py tests/test_affine_coupling_flow.py
tests/test_empirical_train_validation_dataset.py -q` -> all pass.
`reference_golden`-marked tests correctly skip outside
`SHIP_RUN_REFERENCE_GOLDEN=1`; none are xfailed.
