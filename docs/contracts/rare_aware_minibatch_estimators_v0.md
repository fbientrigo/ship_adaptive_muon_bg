# Rare-aware minibatch estimators (v0)

Normative contract for the four estimator arms implemented for
[Issue #17](https://github.com/fbientrigo/ship_adaptive_muon_bg/issues/17):
IID (A), fixed-composition diagnostic (B), fixed-composition
Horvitz-Thompson (C), and the legacy self-normalized estimator (D). This
document is scoped to numerical controlled benchmarks (D0-D5); it makes no
SHiP physics, background-rate, or FairShip claim.

## 1. Label-augmented target setup

Let `(H, X)` have the target joint law: `H = h` with probability `p_h`
("stratum mass"), `X | H = h ~ p_h(.)` ("stratum conditional"). The marginal
is `p(x) = sum_h p_h * p_h(x)`. The training risk is

```
L(theta) = E_p[ell_theta(X)] = sum_h p_h * E_{p_h}[ell_theta(X)]
```

with `ell_theta(x) = -log q_theta(x)`. Training draws from a different label
marginal `a_h` (the realized fixed-composition allocation) with the *same*
conditionals `p_h(.)`. The Radon-Nikodym derivative on the joint `(h, x)`
space is the per-stratum constant `p_h / a_h` -- exact, auditable, and
requiring no density evaluation.

## 2. Why latent mixture labels, not geometric regions, define D5 strata

D5's rare component is a labelled Gaussian mixture component
(`benchmarks/controlled_targets.py:_build_d5`), not a region of feature
space: `x` values from the rare and main components overlap. Every formula
below operates on the label `h` (`component_id`), never on a geometric
predicate over `x`. `target.stratum_masses(pdg_id=...)` returns the exact
`{"main": 1 - p_rare, "rare": p_rare}` and `target.sample_stratum(n, ...,
stratum=...)` draws exactly from `p_h(.)` -- the renormalized sub-mixture
over non-rare components ("main") or the rare component itself ("rare").

## 3. Formulas for arms A-D

Notation: batch size `B`; fixed per-stratum counts `m_h` with `sum_h m_h =
B`; realized allocation `a_h = m_h / B`; HT weight `w_h = p_h / a_h = p_h *
B / m_h`.

| Arm | Estimator | Sampling law |
|---|---|---|
| **A** IID target | `L_A = (1/B) sum_i ell_theta(x_i)`, `x_i ~ p` i.i.d. | uniform draw from the target (or a uniformly-permuted finite pool) |
| **B** fixed-composition diagnostic | `L_B = (1/B) sum_h sum_i ell_theta(x_{h,i})` | `m_h` fixed; `x_{h,i} ~ p_h` |
| **C** fixed-composition Horvitz-Thompson | `L_C = (1/B) sum_h sum_i w_h * ell_theta(x_{h,i}) = sum_h (p_h/m_h) sum_i ell_theta(x_{h,i})` | `m_h` fixed; `x_{h,i} ~ p_h`, strata independent |
| **D** legacy self-normalized | `L_D = sum_i w_{c(i)} ell_theta(x_i) / sum_i w_{c(i)}` | pool stratified at fraction `a_h`; *minibatch* composition random (permutation slice of the pool) |

Arm B targets the shifted risk `L_s = sum_h a_h * E_{p_h}[ell_theta]`, with
bias `sum_h (a_h - p_h) * E_{p_h}[ell_theta]` whenever `a_h != p_h`: it is
**intentionally biased** for the original target risk `L(theta)` and is
never reported as a fit to it.

Arm C's gradient is `nabla L_C = sum_h (p_h/m_h) sum_i nabla_theta
ell_theta(x_{h,i})`, linear in `ell_theta`, so unbiasedness of the loss
passes to the gradient under assumption A5 (below) with no extra argument.

## 4. Fixed vs. random per minibatch

| Quantity | A | B | C | D |
|---|---|---|---|---|
| Batch size `B` | fixed (short last batch allowed) | fixed | fixed | fixed (short last batch allowed) |
| Per-stratum counts `m_h` | random | **fixed** | **fixed** | random |
| Row weights | 1 | 1 | fixed `w_h` | fixed per row |
| `sum(weight)` | `= B`, fixed | `= B`, fixed | `= B`, **fixed** | **random** |
| Loss denominator | `B` | `B` | `B` | random `sum(weight)` |

## 5. The deterministic-weight-sum identity

For arm C, `sum_h m_h * w_h = sum_h m_h * (p_h * B / m_h) = B * sum_h p_h =
B`. This holds **exactly**, independent of which `p_h` or `m_h` are chosen,
as long as composition is exactly fixed. Two consequences:

1. **Collapse**: when composition is exactly fixed, arm C's value is
   numerically identical to the self-normalized ratio
   `sum(w * ell) / sum(w)`, because the denominator `sum(w)` is
   deterministically `B`. All of arm D's bias comes from its minibatch
   composition being *random* (a permutation slice of a stratified pool),
   never from the choice of denominator formula. Tested directly:
   `tests/test_rare_aware_estimators.py::test_collapse_identity_fixed_composition_ht_equals_self_normalized`.
2. **Divergence**: under random composition, the self-normalized ratio and a
   fixed-`B` estimator built from the *same* per-row weights differ, with a
   non-zero mean difference over repeated draws -- this is exactly arm D's
   mechanism of bias. Tested:
   `test_random_composition_diverges_from_fixed_denominator`.

Both identities are also verified analytically at scale in
`scripts/validate_rare_aware_estimators.py --stage analytic`
(`arm_C.weight_sum_identity_holds` on every grid point).

## 6. Assumptions A1-A6 (arm C's unbiasedness claim)

Recorded verbatim in the sampling manifest
(`unbiasedness_assumptions`) and referenced by ID everywhere the claim
appears.

- **A1 -- exact strata**: strata are exhaustive and mutually exclusive in
  the label-augmented space, `sum_h p_h = 1`. For D5, `p_h` is exact by
  construction (`_d5_base_params` rescales main-component weights by
  `1 - rare_mass`).
- **A2 -- correct conditional sampling**: within each stratum, rows are
  drawn from the *unmodified* target conditional `p_h(.)`
  (`target.sample_stratum`), never a modified or truncated version.
- **A3 -- non-empty fixed allocation**: for every stratum with `p_h > 0`,
  `m_h >= 1`, fixed before the batch is drawn.
  `plan_fixed_composition_batches` raises if this is violated.
- **A4 -- fixed denominator**: the loss denominator is the known batch size
  `B` (or a realized short-batch `B'`, not implemented in this stage --
  `drop_last` is the only supported `incomplete_batch_rule`). Never
  `sum(weight)`. Enforced at runtime by
  `Nflow.torch_models.trainer.HTInvariantError`.
- **A5 -- integrability and gradient interchange**: `E_{p_h}|ell_theta| <
  infinity` and `nabla_theta ell_theta` is locally dominated so `nabla E =
  E nabla`. Not over-proven here: the affine-coupling flow's `max_log_scale`
  bound (already enforced and logged as `max_abs_log_scale`) keeps the
  density and its gradient bounded on the training domain.
- **A6 -- unit physical weights**: physical/MC row weights must be
  identically 1 under arm C. `plan_fixed_composition_batches` raises if a
  non-unit `physical_weight` array is supplied. Sampling-correction weights
  (`w_h`) and physical weights are separate concepts; physical-weight
  semantics are **not implemented** in this stage
  (`physical_weight_semantics: "not_implemented"` in every sampling
  manifest).

## 7. Finite-pool vs. generative conditional sampling

Two pool modes exist for fixed-composition training:

- **Mode F (finite pool, the implemented default)**: the training partition
  is drawn once with fixed composition (`sample_controlled`'s
  `pool_law="fixed_composition"`), and minibatches are index slices into
  that fixed partition (`plan_fixed_composition_batches`). This preserves
  dataset hashes, resume, and checkpoints.
- **Mode G (generative)**: each batch is drawn fresh from the target
  conditionals, exact i.i.d. with no finite-population correction. Used only
  by the analytic validation script (Stage 1), never by campaign runs, to
  avoid coupling dataset identity to an infinite-pool assumption.

Both assumptions hold: under SRSWOR from a finite pool (mode F), the
stratum sample mean is unbiased for the pool's stratum mean, so arm C stays
unbiased per step (conditional on the pool) while gaining a finite-population
variance reduction `1 - m_h/N_h`.

## 8. Replacement semantics

Default `replacement="without_replacement_within_epoch"`: each stratum's
pool indices are permuted once per epoch and consumed in successive
`m_h`-sized slices, guaranteeing no duplicate row within one stratum's
sub-batch. `"with_replacement"` draws each step's `m_h` rows independently
(duplicates possible), offered for variance comparison. Both modes are unit
tested in `tests/test_rare_aware_estimators.py`.

**Caveat**: the current trainer implementation replays one precomputed
`MinibatchPlan` identically every training epoch (only cosmetic within-batch
row order varies by a per-epoch shuffle seed). This does not affect any
single step's unbiasedness (the property this document and the validation
script certify), but with a large `max_epochs` relative to
`steps_per_epoch`, the model can see only `steps_per_epoch * m_h` distinct
rows per stratum for the whole run -- a memorization risk flagged here
explicitly, mitigated in the bounded validation config by a small
`max_epochs` (<= 30).

## 9. Incomplete-batch semantics

Legacy arms A and D are unchanged: the last batch of an epoch may be short
(`range(0, n, batch_size)`), with the corresponding denominator (`B'` or
`sum(weight)`) unaffected. Arms B and C use `incomplete_batch_rule =
"drop_last"` (the only implemented value): `steps_per_epoch` is chosen so
every planned batch is a full, exactly-composed batch --  no short trailing
batch is ever planned. `"allow_short_last"` is a named-but-unimplemented
value that `SamplingSpec.validate()` rejects explicitly.

## 10. Scarce-stratum failure rules

- `N_h < m_h` (fewer available rows than requested): hard `ValueError` from
  `plan_fixed_composition_batches`, before any training. Always a
  **technical** failure (`technical_status=failed`, `scientific_status=None`
  via the campaign's outer exception handler), never a scientific negative.
- `N_h >= m_h` for every stratum but capacities differ:
  `steps_per_epoch_rule="min_stratum_pass"` (default) sets `steps_per_epoch
  = min_h floor(N_h/m_h)` -- unbiased, no recycling, some pool rows of larger
  strata go unused within an epoch.
- `steps_per_epoch_rule="recycle_scarce_stratum"` (opt-in): `steps_per_epoch
  = max_h floor(N_h/m_h)`; a stratum whose own capacity is smaller gets
  re-permuted mid-epoch. Still unbiased per step; recorded via
  `stratum_recycled` / `recycle_count_by_stratum` in the sampling/plan
  manifest.
- A configuration that would produce zero steps under either rule raises
  `ValueError` rather than silently running a zero-step epoch.

## 11. Physical weights vs. sampling corrections

Two orthogonal factors: `physical_weight` (MC/event weight; semantics **not
implemented**) and `sampling_correction_weight` `w_h`. If physical weights
were non-unit, the correct combined estimator would be a *within-stratum
ratio* `sum_h p_h * (sum_i c_i ell_i / sum_i c_i)`, which is only
asymptotically unbiased -- so A6 (unit physical weights) is a precondition
for arm C's exact unbiasedness claim, not a simplification of convenience.

## 12. Permitted claims

- **Arm A**: the unbiased IID baseline for the original target risk.
- **Arm C** (`stratified_horvitz_thompson_fixed_composition`): "unbiased for
  the target risk and its gradient **under assumptions A1-A6**, for a
  labelled-component target with exactly known stratum masses" --
  `fit_claim = "unbiased_target_risk_estimator_under_stated_assumptions"`.
  Verified against an exact analytic gradient by Monte Carlo
  (`scripts/validate_rare_aware_estimators.py --stage analytic`).
- **Arm B** (`stratified_unweighted_diagnostic`): an intentionally biased
  diagnostic of capacity/optimization under a shifted training distribution
  -- `fit_claim = "diagnostic_only_not_a_fit_to_original_target_density"`.
- **Arm D** (`stratified_self_normalized_provisional`): a self-normalized
  provisional estimator whose unbiasedness is not established; its measured
  bias may be reported -- `fit_claim =
  "provisional_target_estimator_not_validated_as_original_target_density"`.

## 13. Forbidden claims

- Arm B never fits the original target density.
- Arm D is never relabelled unbiased.
- Arm C's unbiasedness never transfers to data without exact known stratum
  masses and component labels (in particular: **not** real, unlabeled
  afterMS data).
- No minibatch stratum allocation, including 50/50, is promoted as optimal.
- No ESS acceptance threshold exists for training-weight ESS
  (`training_weight_ess_over_n`); it is report-only and no gate reads it.
- No SHiP physics, background-rate, or FairShip/GEANT4 claim.
- No architecture winner is declared by this validation.
- A technical failure (unsupported family, scarce stratum, malformed
  config) is never reported as a scientific negative.

## 14. Backward-compatibility guarantees

- `SAMPLING_REGIMES` is append-only: `iid_target`,
  `stratified_unweighted_diagnostic`,
  `stratified_self_normalized_provisional` keep their exact order and
  semantics; `stratified_horvitz_thompson_fixed_composition` is appended.
- `SamplingSpec.to_dict()` serializes new fields (`minibatch_rare_count`,
  `minibatch_batch_size`, `replacement`, `incomplete_batch_rule`,
  `steps_per_epoch_rule`, `validation_partition_law`) only when they differ
  from their defaults (the `EvaluationSpec._include_rare_sample_count`
  idiom), so every existing committed config's `config_hash` / `run_id` is
  byte-identical (pinned by
  `tests/test_rare_aware_estimators.py::test_pinned_*`).
- `_weighted_nll` (legacy loss) and `loss_normalization == "sum_weights"`
  for legacy arms are unchanged; a new `_fixed_denominator_nll` implements
  arm C.
- `AffineCouplingFlow._functional_fingerprint_payload` /
  `checkpoint_hash_schema` are untouched: the estimator arm is a
  training-only choice, never a function-changing one, so existing
  checkpoints remain loadable and arm C introduces no new checkpoint schema.
- The frozen D5 memorization DOE v0 canonical hash, JSON, and manifest are
  never edited in place; a new estimator contract for future DOE
  generations lives under `SAMPLING_ESTIMATOR_CONTRACT_V1` in
  `density_lab/doe.py`.
- Scientific gate IDs, severities, and `GATE_SCHEMA_VERSION` are unchanged;
  the one new `estimator_evidence_scope` gate is inactive and report-only
  and never changes `scientific_status`.
- `metrics.importance_ess.ess_over_n` remains the sole quantity the ESS
  catastrophic gate reads; training-weight ESS lives under
  `training_weight_ess` / `training_weight_ess_over_n` in the sampling
  manifest, a disjoint namespace.

## 15. Bounded validation design

Two bounded stages (`scripts/validate_rare_aware_estimators.py`), CPU-only,
no real afterMS data:

**Stage 1 (`--stage analytic`)**: a quadratic loss `ell_theta(x) =
0.5||x-theta||^2` with exact gradient `nabla L = theta - sum_h p_h * mu_h`.
Grid: rare mass in `{1e-3, 1e-2}`, allocation in `{p_rare, 0.05, 0.2, 0.5}`,
batch size in `{128, 512}`, replicate counts `{125, 500, 2000}`. Reports
exact gradient, empirical mean gradient, bias, MC standard error,
standardized bias `z`, gradient covariance trace, the weight-sum identity,
and wall time for every arm/configuration. Preregistered checks (not
scientific gates): arms A/C have `|z| < 4` at every grid point; arms B/D
show `|z| > 6` at the single deliberately asymmetric configured case
(smallest tested rare mass, allocation 0.5, largest replicate count); A/C
Monte Carlo error shrinks approximately as `R^-1/2`; D's bias at that same
asymmetric case does not vanish merely from more replicates. Runs in
seconds.

**Stage 2 (`--stage d5`)**: `configs/density_lab/estimator_validation_v0.json`
-- D5 `rare_1e-3`, `pdg_id=13`, one small affine-coupling model (4 blocks,
width 64, depth 1, batch size 128, <= 30 epochs), `n_train=8192`,
`n_validation=4096`, `n_test=4096`, seeds `{11, 22, 33}`, 5 sampling
configurations (arm A; arm B at allocation 0.5; arm C at allocation ~0.05
and at 0.5; arm D at allocation 0.5) = 15 runs, measured at ~2 minutes wall
time on CPU (well under the ~30-minute bound). Also evaluates the gradient
(and rare/main gradient norms) at each arm-C model's *initial* parameters
using the same batch plan the campaign trains with, at no extra training
cost. Reports: exact fixed stratum counts per arm-B/C minibatch; train/
validation NLL by stratum; generated D5 rare-region mass and its exact
binomial interval (existing `metrics.exact_binomial_interval`); training-
weight ESS and the existing model-vs-target `importance_ess`; wall time;
technical failures; low-power outcomes (a zero rare count is
`inconclusive_low_power`, never demonstrated rare-mode collapse). No
allocation or architecture is declared a winner.
