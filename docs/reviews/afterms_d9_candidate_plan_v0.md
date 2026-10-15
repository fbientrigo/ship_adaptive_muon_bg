# D9 Candidate Plan v0 -- Explanation

Machine-readable source: `configs/afterms/d9_candidate_plan_v0.json`.

## 1. Provenance

Derived entirely from the frozen D8 evaluation (`artifacts/afterms_d8_evaluation_v0`),
producer commit `f9d8246f281a0e80a8583f8c70b4a0c2e9fc0870`. Before writing this plan, the
following D8 files were hashed (SHA-256) and recorded in the plan JSON's
`source_d8_files`:

| file | sha256 |
| --- | --- |
| `report/afterms_smoke_arena.json` | `6a1a4375c3fd99dfc9e49d6f2ced56c68f92afe2e6dae8024ebce71dd7a38b62` |
| `arenas/model_arena.json` | `a374616f15fbc5282b061b4ce0c75d046009ec751a399574f9ad2c53ed8a91cc` |
| `registry/run_registry.json` | `d789982edd7b86b431a33aeba20c8523ecad75966d0f66f7af9073ca07cd1272` |

These are re-verified by `tests/afterms/d9/test_plan.py::test_d7_d8_inputs_unchanged`
(required test 35) and must be re-checked before and after any future edit to this
plan.

## 2. Weighted-track status

The project authority of the physical source weight `w` remains unresolved. This plan
does **not** silently declare the weighted track (C/D) the sole true nominal target,
and does **not** silently discard it either. Both weighted and unweighted tracks are
kept as separate estimands. The decision is recorded as `OPEN` in
`weighted_track_decision_status`.

## 3. Selected candidates (6, at the stated cap)

| candidate_id | track | source D8 run | role | preprocessing | capacity | params |
| --- | --- | --- | --- | --- | --- | --- |
| A1_capacity_medium_identity_pdg13_unweighted | A | `09_affine_capacity_smoke_pdg13__identity_standardized_v0_affine_medium_unweighted` | champion | identity_standardized_v0 | medium | 280656 |
| A2_capacity_small_cartesian_pdg13_unweighted | A | `05_affine_preprocessing_ab_pdg13__cartesian_log1p_pz_v0_affine_small_unweighted` | challenger | cartesian_log1p_pz_v0 | small | 37416 |
| B1_capacity_small_cartesian_pdg_minus13_unweighted | B | `06_affine_preprocessing_ab_pdg_minus13__cartesian_log1p_pz_v0_affine_small_unweighted` | champion | cartesian_log1p_pz_v0 | small | 37416 |
| B2_capacity_small_identity_pdg_minus13_unweighted | B | `06_affine_preprocessing_ab_pdg_minus13__identity_standardized_v0_affine_small_unweighted` | challenger | identity_standardized_v0 | small | 37416 |
| C1_weighted_small_identity_pdg13 | C | `07_affine_weight_ab_pdg13__identity_standardized_v0_affine_small_weighted` | champion (only) | identity_standardized_v0 | small | 37416 |
| D1_weighted_small_identity_pdg_minus13 | D | `08_affine_weight_ab_pdg_minus13__identity_standardized_v0_affine_small_weighted` | champion (only) | identity_standardized_v0 | small | 37416 |

### Human-readable aliases (D9C)

The `candidate_id`s above are legacy internal identifiers, not model names.
`docs/reviews/afterms_model_alias_registry_v0.md` defines a presentation-only
alias layer on top of them (never renaming an id or touching a hash):

- `NF_AC_b08_w128_d02` on `TRK_PDG13_UW_ID` (legacy candidate `A1_capacity_medium_identity_pdg13_unweighted`)
- `NF_AC_b04_w064_d02` on `TRK_PDG13_UW_LOGPZ` (legacy candidate `A2_capacity_small_cartesian_pdg13_unweighted`)
- `NF_AC_b04_w064_d02` on `TRK_PDGM13_UW_LOGPZ` (legacy candidate `B1_capacity_small_cartesian_pdg_minus13_unweighted`)
- `NF_AC_b04_w064_d02` on `TRK_PDGM13_UW_ID` (legacy candidate `B2_capacity_small_identity_pdg_minus13_unweighted`)
- `NF_AC_b04_w064_d02` on `TRK_PDG13_W_ID` (legacy candidate `C1_weighted_small_identity_pdg13`)
- `NF_AC_b04_w064_d02` on `TRK_PDGM13_W_ID` (legacy candidate `D1_weighted_small_identity_pdg_minus13`)

### Why A2/B2 exist (the one scientific question a lone champion can't answer)

D8's five-epoch smoke shows the winning preprocessing **flips by PDG sign** on the
physical-validation-NLL axis:

- PDG+13: `identity_standardized_v0` wins (medium 3.4925, small 3.6094) over
  `cartesian_log1p_pz_v0` (small 3.5507).
- PDG-13: `cartesian_log1p_pz_v0` wins (small 3.5164) over `identity_standardized_v0`
  (small 3.6739).

A2 and B2 exist to test whether that flip is a real preprocessing-by-sign effect or
five-epoch/single-seed noise -- a question the champion alone cannot answer.

**B2 is a clean, same-capacity (37416 params both) preprocessing-only comparison**
against B1.

**A2 is not clean**: A1 (champion) is capacity=medium (280656 params) while A2 is
capacity=small (37416 params), so preprocessing and capacity both vary relative to A1.
This confound is disclosed in A2's `selection_reason` rather than hidden. The
same-capacity alternative (identity/small, 3.6094) was not chosen as A2 because it is
materially indistinguishable from A1's own small-capacity identity variant and does
not, by itself, probe a new question.

C and D have no challenger: each of their D8 arenas contains exactly one candidate
(`n_candidates=1`). No challenger is added merely to fill the cap.

This yields **6 primary neural candidates**, the stated maximum, justified candidate
by candidate rather than treated as a target headcount.

## 4. Excluded candidates

| candidate_id | reason (short) |
| --- | --- |
| E_quantile_pdg13 / E_quantile_pdg_minus13 | feature-space-only per frozen D8 contract; physical-space Jacobian explicitly deferred (D8 recommendation #4); 6-slot cap already filled by physical-space-comparable tracks |
| F_legacy_combined_pdg_4d | historical 4D context only; D8 report does not justify a required 5D regression run |
| G_memory_repeat_combined_unfiltered_5d | D8's own report labels this a determinism/memory diagnostic, not a genuine model comparison |
| gaussian_gmm_controls_pdg13 / pdg_minus13 | optional deterministic controls, disabled by default; ranked worst in both A and B arenas; no historical checkpoint (`MISSING_HISTORICAL_CHECKPOINT`) |

Full reasoning for each is in `configs/afterms/d9_candidate_plan_v0.json` ->
`excluded_candidates[].exclusion_reason`.

## 5. Target measure

All 6 primary candidates target `physical_space_nll`. `identity_standardized_v0` and
`cartesian_log1p_pz_v0` both have analytic forward log-abs-det Jacobians in
`PreprocessingPipeline` (`src/ship_muon_bg/afterms/preprocessing.py`); `07`/`08`
(weighted) already recorded a `physical_nll` in the D8 registry using this Jacobian, so
the weighted tracks are not restricted to feature-space NLL going forward.

D8's weighted `validation_nll` for C1/D1 (2.951137, 2.900266) was produced under the
historical `self_normalized_minibatch_approximation` estimator (see D8's loss-semantics
notes: "Per-epoch validation_loss for weighted runs is a self_normalized_minibatch_ratio
quantity"). D9 does not inherit that estimator (see
`configs/afterms/d9_training_v0.json` and `src/ship_muon_bg/afterms/d9/weighted_objective.py`);
those D8 numbers are cited here only as prior smoke evidence, not as the D9 objective.

## 6. Non-claims carried forward from D8

All of D8's non-claims (`report/afterms_smoke_arena.json` -> `non_claims`) still apply:
five-epoch smoke establishes plausible wiring, not convergence; no production-readiness
or downstream-acceptance claim; the D7 train/validation/test split has no
source-lineage/group identifier and does not establish source-muon independence.

D9_PLAN_VALID
