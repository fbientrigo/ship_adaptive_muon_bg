# D9 conditional utility-tilt v0 (fixture arena)

## 1. Lineage and branch boundaries

| item | value |
| --- | --- |
| parent branch | `experiment/d9-conditional-charge-sampling-v1` |
| parent commit | `71bcbaaf5480cd65bef3ff8c461e00f7a8cdedf6` ("docs(d9): record conditional sampling v1 study") |
| this branch | `experiment/d9-conditional-utility-tilt-v0` |
| branch base | created directly from the parent commit; no merge, no rebase, no cherry-pick |
| fixture | `data/samples/muonsFullMC_afterMS_sample.npz` |
| fixture dataset hash | `718b9197a3965f6842574fec1f3f6b1bf4c16949ff8a1f27dca6232817c03c32` |
| parent full-dataset hash (from the fixture manifest) | `44336e8e3629149c813026cf21334c6fc2db75ae9ffa12e78ed2b064f3a54579` (13 779 080 rows, never loaded) |
| arena executed at code commit | `1cf0af0776b8ddabf35e925fd8803ab4eec5f478` (recorded in every artifact's `lineage.code_commit`) |

Not done, by construction: no merge into the parent or `main`, no rebase of the
parent, no start from `feat/d9-conditional-charge-sampling-v1` or `main`, no
cherry-pick of the historical D9-5N row-empirical line, no full-dataset
training, no FairShip/GEANT4 call, no hyperparameter sweep, no hard charge
symmetry, no test-payload access, and no physical-rate estimate.

### v1 contracts preserved

Every mechanism frozen by the v1 gate remains active and is exercised by test:

- physical charge convention `PDG 13 (mu-) -> -1`, `PDG -13 (mu+) -> +1`
  (`MUON_ELECTRIC_CHARGE_SIGN_BY_PDG`, re-used, not redefined);
- one shared conditional affine-coupling flow, `condition_dim = 1`;
- exact train-only macro-weighted preprocessing;
- deterministic per-epoch direct sampling;
- equal draw counts between charges in every epoch;
- ordinary unweighted NLL after direct sampling;
- no double weighting;
- exact weighted validation without resampling;
- closed test payload;
- train-only, charge-specific `B_toy` thresholds;
- no hard symmetry;
- FairShip/GEANT4 remains the final physical oracle.

## 2. Mathematical target

For each charge `c` the nominal empirical measure is unchanged from v1:

```
pi_i^(0,c) = w_i / sum_{j: C_j = c} w_j
```

The synthetic utility multiplier is the repository's single canonical
definition (`ship_muon_bg.density_lab.utility_tilt.h_alpha_delta`), imported
and never restated:

```
r_i(delta, alpha) = [delta + (1 - delta) * U_i] ** alpha
```

with the binary synthetic utility (`compute_utility_a`)

```
U_A(x_i) = 1  if x_i in B_toy,c
         = 0  otherwise
```

and the train-fitted, charge-specific region

```
p_T   = sqrt(px^2 + py^2)
R_xy  = sqrt(x^2 + y^2)
t_pT,c = weighted train quantile Q_0.95(p_T)     (train partition of charge c only)
t_R,c  = weighted train quantile Q_0.05(R_xy)    (train partition of charge c only)

B_toy,c = 1[p_T > t_pT,c and R_xy < t_R,c]
```

The within-charge tilted law is

```
pi_i^(U,c) = w_i r_i / sum_{j: C_j = c} w_j r_j
```

The `NOMINAL_PHYSICAL` arm sets `r_i = 1` exactly, so `pi^(NOMINAL,c)` is
bitwise the v1 nominal table (regression-tested against
`conditional_charge._EpochDirectSampler`).

### Macro charge prior

```
P(C = 13) = P(C = -13) = 1/2
```

is enforced structurally, not statistically: every epoch draws exactly
`draws_per_epoch_per_charge` rows for PDG 13 and the same number for PDG -13.
The tilt renormalizes mass *inside* a charge only; the total tilted mass of a
charge is divided out by its own denominator and can never move the 50/50
charge allocation. `manifest()["charge_prior_unaffected_by_tilt"] = true`, and
`test_macro_charge_prior_is_unaffected_by_the_tilt` asserts equal condition
counts under the strongest tilt in the arena.

### Direct-sampling derivation (why the loss carries no weight)

Drawing `I_(e,t) | c ~ pi^(variant,c)` and then evaluating the *ordinary
unweighted* NLL on the drawn rows already realizes the intended per-charge
objective in expectation:

```
E_{i ~ pi^(variant,c)}[ -log q_theta(x_i | c) ]
    = - sum_i pi_i^(variant,c) log q_theta(x_i | c)
```

Multiplying that loss by `w_i`, `r_i`, or `w_i r_i` again would square the
measure. With equal per-charge draw counts, the per-step gradient satisfies

```
E_epoch_draws[gradient] = (1/2) grad L^(variant)_13 + (1/2) grad L^(variant)_-13
```

This is a statement about **one stochastic-gradient step only**. It does not
imply the final trained parameters are an unbiased estimator of any macro
target: SGD over a non-convex loss carries no such guarantee even with
per-step-unbiased gradients. The manifest records this scope verbatim under
`unbiasedness_scope`.

## 3. Arena variants

Exactly four variants, using the repository's canonical variant ids
(`utility_tilt.make_tilt_id`) — no new id strings, no duplication of the
utility formula, and the historical 20-variant grid is never materialized:

| task label | canonical variant id | delta | alpha | `r` on `B_toy` | `r` off `B_toy` | rho |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| NOMINAL | `NOMINAL_PHYSICAL` | – | – | 1 | 1 | 1 |
| UA_d0p9_a04 | `UA_d0p9_a04` | 0.9 | 4 | 1 | 0.6561 | 1.524 |
| UA_d0p9_a08 | `UA_d0p9_a08` | 0.9 | 8 | 1 | 0.43047 | 2.323 |
| UA_d0p1_a01 | `UA_d0p1_a01` | 0.1 | 1 | 1 | 0.1 | 10 |

`NOMINAL` is accepted as a task-facing alias and resolved to the canonical
`NOMINAL_PHYSICAL` before anything is hashed or seeded.
`validate_conditional_utility_variant_ids` rejects an empty list, duplicates,
unknown ids, and any id outside this gate — in particular every probabilistic
`U_P` configuration and the `alpha = 16` rungs of the historical grid.

## 4. Preprocessing decision (PROJECT DECISION)

**Decision.** One fixed *nominal* macro-weighted train-only preprocessing
transform is used for all four variants. It is fit exactly once per (data
split, model seed) from

```
mu       = (1/2) sum_c sum_{i: C_i = c} pi_i^(0,c) x_i
variance = (1/2) sum_c sum_{i: C_i = c} pi_i^(0,c) (x_i - mu)^2
```

via the v1 helper `FittedFeaturePipeline.fit_macro_weighted`, never on
resampled rows, never separately per utility variant, never with tilted
weights, and never touching validation or test rows.

**Rationale.**

- it keeps the physical coordinate transform common across variants;
- it isolates the effect of changing the sampling measure — the only thing
  that differs between arms is which rows get drawn;
- it makes nominal and tilted runs directly comparable in the same normalized
  coordinates;
- it makes **no claim** that nominal preprocessing is universally optimal.

Variant-specific or tilted preprocessing remains **OPEN**.

**Evidence.** Each run records `lineage.preprocessing_hash`; the arena
additionally asserts at runtime that the hash is identical across all variants
sharing a seed and split, and records the result in
`arena_run_summary.json -> preprocessing_hash_by_model_seed`. Hashes are
reproduced in section 8 below.

## 5. Per-epoch conditional utility sampling

For every epoch `e` and charge `c`, `I_(e,t) | c ~ pi^(variant,c)` is drawn
with replacement from an O(1) Walker-alias table using a deterministic seed

```
SeedSequence[global_model_seed, epoch, |pdg_id|, pdg_id < 0,
             sha256(variant_id)[:8], salt]
```

so the variant id is a first-class component of the seed: two variants can
never share a draw sequence at identical (model seed, epoch, charge). After
drawing, both charge pools are concatenated, deterministically shuffled with a
separate salt, and the physical charge conditions are passed separately to the
flow. Training is the ordinary unweighted NLL; `w_i` and `r_i` never reach the
loss.

Recorded per epoch and charge (`conditional_utility_sampling_manifest.json`):
`variant_id`, `pdg_id`, `condition`, `source_table_hash`,
`nominal_probability_table_hash`, `utility_vector_hash`,
`multiplier_vector_hash`, `tilted_probability_table_hash`, `alias_table_hash`,
`draw_hash`, `draw_count`, `empirical_b_toy_occupancy`,
`declared_target_b_toy_probability`, `unique_source_rows`,
`unique_rows_fraction`, `max_reuse_count`, `cumulative_unique_source_rows`,
`sample_weight_applied_to_loss = false`,
`utility_weight_applied_to_loss = false`,
`sampling_with_replacement = true`.

The trainer itself refuses the combination that would double-weight:
`epoch_sampler` with a non-`None` `sample_weight` raises, and every history
record carries `weight_normalization = "epoch_direct_sampling_unweighted"`.

## 6. Validation definitions

Validation is never resampled. For every trained model and charge both
measures are computed on the untouched validation partition, using the
**train-fitted** `B_toy` thresholds to evaluate `r_i`:

```
L_val,0,c = - sum_i w_i       log q_theta(x_i|c) / sum_i w_i
L_val,U,c = - sum_i w_i r_i   log q_theta(x_i|c) / sum_i w_i r_i
```

Both are reported per charge (PDG 13, PDG -13), as a macro mean and as the
worst charge, in `per_charge_nominal_validation.json` /
`per_charge_tilted_validation.json` and in `summary.json -> validation`.

No model is selected by nominal NLL alone, and none by tilted NLL alone: the
arena is diagnostic and reports the trade-off
(`validation.selection_policy`).

## 7. Generated-distribution diagnostics

For each variant, seed, and charge, 8192 samples are generated with a
deterministic generation seed derived from (model seed, charge, variant).
Reported: finite generated-sample fraction, finite generated log-probability
fraction, generated `B_toy` occupancy, nominal and tilted target `B_toy`
probabilities, generated/nominal and generated/tilted ratios, binomial
standard error and Wilson 95% interval for the generated occupancy, weighted
empirical marginal summaries and correlations, and generated marginal
summaries and correlations.

The occupancy is compared against **the target measure actually used for
training**: `nominal` for `NOMINAL_PHYSICAL`, `tilted` for the three tilt
variants (`training_target_measure` / `training_target_b_toy_probability`).
A generated/target ratio above or below one is not read as a physical pass or
failure without the generated-sample uncertainty and the seed spread.

## 8. Results

### 8.1 Completion matrix — `CONDITIONAL_UTILITY_ARENA_VERIFIED`, 12/12

| model seed \ variant | `NOMINAL_PHYSICAL` | `UA_d0p9_a04` | `UA_d0p9_a08` | `UA_d0p1_a01` |
| --- | --- | --- | --- | --- |
| 11 | completed | completed | completed | completed |
| 12 | completed | completed | completed | completed |
| 13 | completed | completed | completed | completed |

Wall time for the whole arena: 1 min 47 s on 4 CPU threads (fit alone is
~15 s per run). Every run reports `status = completed_fixture_run`,
`generated_sample_finite_fraction = 1.0` and
`generated_log_prob_finite_fraction = 1.0`; no run produced a non-finite
sample, a non-finite log-probability, or a failed fit. There are **no
technical failures to separate from the scientific outcome**.

Fixture split, identical for every variant and seed: 10 240 train / 2 560
validation / 3 200 test rows per charge (test never materialized).

### 8.2 Preprocessing hash equality (section 4 evidence)

| model seed | preprocessing hash | identical across all four variants |
| --- | --- | --- |
| 11 | `d858fb2f0dd0522c6bfa9098615dc6ef77264d32360d9d5f4222e15524c8aa8c` | yes |
| 12 | `065d1fa2aca2045c997fa0694d930dbc892fa0ad32fc778f9c83bfcc55e5be6f` | yes |
| 13 | `e471ada5330178e6fb57ed16d1eeac317e2035ec2c91a4b1ff1db4e6d9e77c03` | yes |

The arena additionally raises if the hashes ever diverge within a seed, so a
completed run is itself proof of the equality.

Per-charge lineage hashes (source table, nominal probability table, utility
vector, multiplier vector, tilted probability table, alias table), the model
config hash, checkpoint hash, split hashes and generation seeds are recorded
in every run's `summary.json -> lineage`; the `NOMINAL_PHYSICAL` arm's
`tilted_probability_table_hash` equals its `nominal_probability_table_hash`
exactly, which is the artifact-level statement that `r = 1`.

### 8.3 Full 12-run, per-seed, per-charge table

`p0(B)` = nominal target, `pU(B)` = tilted target, `gen B` = generated
occupancy over 8192 samples, `SE` = binomial standard error, `W95` = Wilson
95% interval, `N_eff(w)` = the concentration already in `w_mc`.

| variant | seed | PDG | nominal val NLL | tilted val NLL | p0(B) | pU(B) | gen B | gen/nom | gen/tilt | SE | W95 low | W95 high | N_eff | N_eff(w) | cum unique | max reuse |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NOMINAL_PHYSICAL | 11 | 13 | 5.7752 | 5.7752 | 0.024185 | 0.024185 | 0.021729 | 0.8984 | 0.8984 | 0.001611 | 0.018788 | 0.025117 | 630.3 | 630.3 | 9661 | 22 |
| NOMINAL_PHYSICAL | 11 | -13 | 6.2466 | 6.2466 | 0.026384 | 0.026384 | 0.034912 | 1.3233 | 1.3233 | 0.002028 | 0.031150 | 0.039110 | 742.4 | 742.4 | 9290 | 21 |
| NOMINAL_PHYSICAL | 12 | 13 | 5.2659 | 5.2659 | 0.023518 | 0.023518 | 0.012085 | 0.5139 | 0.5139 | 0.001207 | 0.009937 | 0.014690 | 622.0 | 622.0 | 8418 | 25 |
| NOMINAL_PHYSICAL | 12 | -13 | 5.1839 | 5.1839 | 0.021430 | 0.021430 | 0.020508 | 0.9570 | 0.9570 | 0.001566 | 0.017656 | 0.023809 | 714.3 | 714.3 | 7935 | 19 |
| NOMINAL_PHYSICAL | 13 | 13 | 4.9937 | 4.9937 | 0.027061 | 0.027061 | 0.019165 | 0.7082 | 0.7082 | 0.001515 | 0.016414 | 0.022367 | 638.0 | 638.0 | 8724 | 22 |
| NOMINAL_PHYSICAL | 13 | -13 | 4.7185 | 4.7185 | 0.027540 | 0.027540 | 0.010742 | 0.3901 | 0.3901 | 0.001139 | 0.008728 | 0.013215 | 745.8 | 745.8 | 8053 | 20 |
| UA_d0p9_a04 | 11 | 13 | 5.4957 | 5.5457 | 0.024185 | 0.036401 | 0.021851 | 0.9035 | 0.6003 | 0.001615 | 0.018902 | 0.025248 | 638.8 | 630.3 | 9686 | 23 |
| UA_d0p9_a04 | 11 | -13 | 6.1809 | 6.2028 | 0.026384 | 0.039664 | 0.025635 | 0.9716 | 0.6463 | 0.001746 | 0.022428 | 0.029286 | 744.1 | 742.4 | 9331 | 22 |
| UA_d0p9_a04 | 12 | 13 | 5.2447 | 5.2855 | 0.023518 | 0.035408 | 0.028198 | 1.1990 | 0.7964 | 0.001829 | 0.024829 | 0.032010 | 627.1 | 622.0 | 8553 | 28 |
| UA_d0p9_a04 | 12 | -13 | 4.9638 | 5.0459 | 0.021430 | 0.032300 | 0.027588 | 1.2874 | 0.8541 | 0.001810 | 0.024256 | 0.031362 | 719.7 | 714.3 | 7970 | 25 |
| UA_d0p9_a04 | 13 | 13 | 4.8697 | 4.9225 | 0.027061 | 0.040668 | 0.038086 | 1.4074 | 0.9365 | 0.002115 | 0.034153 | 0.042452 | 645.5 | 638.0 | 9696 | 24 |
| UA_d0p9_a04 | 13 | -13 | 4.6708 | 4.7386 | 0.027540 | 0.041378 | 0.020264 | 0.7358 | 0.4897 | 0.001557 | 0.017430 | 0.023547 | 748.0 | 745.8 | 9285 | 21 |
| UA_d0p9_a08 | 11 | 13 | 5.5369 | 5.6460 | 0.024185 | 0.054442 | 0.045166 | 1.8675 | 0.8296 | 0.002294 | 0.040878 | 0.049880 | 645.9 | 630.3 | 9030 | 38 |
| UA_d0p9_a08 | 11 | -13 | 5.7763 | 5.8402 | 0.026384 | 0.059223 | 0.039673 | 1.5037 | 0.6699 | 0.002157 | 0.035657 | 0.044120 | 732.9 | 742.4 | 8440 | 29 |
| UA_d0p9_a08 | 12 | 13 | 5.2131 | 5.2783 | 0.023518 | 0.052984 | 0.052734 | 2.2423 | 0.9953 | 0.002469 | 0.048101 | 0.057787 | 627.1 | 622.0 | 8823 | 35 |
| UA_d0p9_a08 | 12 | -13 | 5.1724 | 5.3451 | 0.021430 | 0.048411 | 0.039551 | 1.8456 | 0.8170 | 0.002153 | 0.035542 | 0.043992 | 719.6 | 714.3 | 8231 | 31 |
| UA_d0p9_a08 | 13 | 13 | 4.9128 | 5.0280 | 0.027061 | 0.060690 | 0.053223 | 1.9668 | 0.8770 | 0.002480 | 0.048568 | 0.058296 | 648.7 | 638.0 | 8885 | 38 |
| UA_d0p9_a08 | 13 | -13 | 4.5097 | 4.6585 | 0.027540 | 0.061728 | 0.020264 | 0.7358 | 0.3283 | 0.001557 | 0.017430 | 0.023547 | 737.4 | 745.8 | 8225 | 32 |
| UA_d0p1_a01 | 11 | 13 | 5.5340 | 5.8586 | 0.024185 | 0.198620 | 0.157959 | 6.5312 | 0.7953 | 0.004029 | 0.150222 | 0.166017 | 496.5 | 630.3 | 9308 | 115 |
| UA_d0p1_a01 | 11 | -13 | 6.2914 | 6.1887 | 0.026384 | 0.213208 | 0.112427 | 4.2613 | 0.5273 | 0.003490 | 0.105767 | 0.119450 | 390.4 | 742.4 | 8811 | 87 |
| UA_d0p1_a01 | 12 | 13 | 5.8245 | 5.9004 | 0.023518 | 0.194094 | 0.140259 | 5.9640 | 0.7226 | 0.003837 | 0.132908 | 0.147947 | 409.6 | 622.0 | 8009 | 108 |
| UA_d0p1_a01 | 12 | -13 | 5.0386 | 5.5667 | 0.021430 | 0.179651 | 0.128174 | 5.9810 | 0.7135 | 0.003693 | 0.121109 | 0.135587 | 479.5 | 714.3 | 7474 | 92 |
| UA_d0p1_a01 | 13 | 13 | 5.0972 | 5.3933 | 0.027061 | 0.217608 | 0.160889 | 5.9455 | 0.7394 | 0.004060 | 0.153091 | 0.169004 | 440.0 | 638.0 | 8379 | 96 |
| UA_d0p1_a01 | 13 | -13 | 4.7350 | 5.1947 | 0.027540 | 0.220697 | 0.135620 | 4.9245 | 0.6145 | 0.003783 | 0.128376 | 0.143205 | 394.3 | 745.8 | 7683 | 81 |

Each `tilted val NLL` column is that run's **own** measure, so the column is
not comparable across variant rows; section 8.5 is the comparable version.

### 8.4 Aggregate across seeds — median [min, max]

| variant | PDG | nominal val NLL | generated B_toy | gen/nominal | gen/tilted | N_eff | cumulative unique rows | max reuse |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- |
| NOMINAL_PHYSICAL | 13 | 5.266 [4.994, 5.775] | 0.01917 [0.01209, 0.02173] | 0.708 [0.514, 0.898] | 0.708 [0.514, 0.898] | 630 [622, 638] | 8724 [8418, 9661] | 22 [22, 25] |
| NOMINAL_PHYSICAL | -13 | 5.184 [4.719, 6.247] | 0.02051 [0.01074, 0.03491] | 0.957 [0.390, 1.323] | 0.957 [0.390, 1.323] | 742 [714, 746] | 8053 [7935, 9290] | 20 [19, 21] |
| UA_d0p9_a04 | 13 | 5.245 [4.870, 5.496] | 0.02820 [0.02185, 0.03809] | 1.199 [0.903, 1.407] | 0.796 [0.600, 0.937] | 639 [627, 646] | 9686 [8553, 9696] | 24 [23, 28] |
| UA_d0p9_a04 | -13 | 4.964 [4.671, 6.181] | 0.02563 [0.02026, 0.02759] | 0.972 [0.736, 1.287] | 0.646 [0.490, 0.854] | 744 [720, 748] | 9285 [7970, 9331] | 22 [21, 25] |
| UA_d0p9_a08 | 13 | 5.213 [4.913, 5.537] | 0.05273 [0.04517, 0.05322] | 1.967 [1.868, 2.242] | 0.877 [0.830, 0.995] | 646 [627, 649] | 8885 [8823, 9030] | 38 [35, 38] |
| UA_d0p9_a08 | -13 | 5.172 [4.510, 5.776] | 0.03955 [0.02026, 0.03967] | 1.504 [0.736, 1.846] | 0.670 [0.328, 0.817] | 733 [720, 737] | 8231 [8225, 8440] | 31 [29, 32] |
| UA_d0p1_a01 | 13 | 5.534 [5.097, 5.825] | 0.15796 [0.14026, 0.16089] | 5.964 [5.946, 6.531] | 0.739 [0.723, 0.795] | 440 [410, 497] | 8379 [8009, 9308] | 108 [96, 115] |
| UA_d0p1_a01 | -13 | 5.039 [4.735, 6.291] | 0.12817 [0.11243, 0.13562] | 4.924 [4.261, 5.981] | 0.615 [0.527, 0.713] | 394 [390, 479] | 7683 [7474, 8811] | 87 [81, 92] |

### 8.5 Cross-measure validation (the comparable trade-off)

Because each variant's tilted NLL is a *different functional*, the arena also
scores **every trained model under every arena measure**
(`per_charge_tilted_validation.json -> cross_measure_validation_nll`,
`val_nll_under_*` in the CSV). The table below is the paired, same-seed,
same-split difference `L(tilted model) - L(NOMINAL model)` at a fixed
measure; negative means the tilt-trained model is better on that measure.
Three cells per row = model seeds 11, 12, 13.

| model | measure | PDG 13 | PDG -13 |
| --- | --- | --- | --- |
| UA_d0p9_a04 | nominal | -0.279, -0.021, -0.124 | -0.066, -0.220, -0.048 |
| UA_d0p9_a04 | UA_d0p9_a04 | -0.282, -0.039, -0.150 | -0.072, -0.228, -0.055 |
| UA_d0p9_a04 | UA_d0p1_a01 | -0.311, -0.273, -0.476 | -0.150, -0.316, -0.150 |
| UA_d0p9_a08 | nominal | -0.238, -0.053, -0.081 | -0.470, -0.012, -0.209 |
| UA_d0p9_a08 | UA_d0p9_a08 | -0.258, -0.132, -0.160 | -0.476, -0.057, -0.246 |
| UA_d0p9_a08 | UA_d0p1_a01 | -0.350, -0.509, -0.514 | -0.500, -0.241, -0.415 |
| UA_d0p1_a01 | nominal | -0.241, **+0.559**, **+0.104** | **+0.045**, -0.145, **+0.017** |
| UA_d0p1_a01 | UA_d0p1_a01 | -0.639, -0.195, -0.667 | -0.435, -0.711, -0.559 |

### 8.6 Concentration: `w_mc` versus `r_utility`

Per charge, before any tilt, the MC weights alone already concentrate the
train pool: `N_eff(w_mc) ~ 622-638` of 10 240 rows for PDG 13
(`N_eff / n_train ~ 0.061`) and `~714-746` for PDG -13 (`~0.071`), with
top-100 mass `0.19-0.25` and a zero-probability fraction of 0 (PDG -13 seed
13 has a single zero-weight row, `9.77e-5`).

The **additional** concentration introduced by `r_utility`:

| variant | amplification `pU/p0` | `N_eff` change vs `w_mc` | expected unique rows / epoch | max reuse |
| --- | ---: | ---: | ---: | ---: |
| `NOMINAL_PHYSICAL` | 1.00 | — (identical law) | 1291-1357 | 19-25 |
| `UA_d0p9_a04` | 1.502-1.507 | **+0.2 % to +1.4 %** | 1302-1374 | 21-28 |
| `UA_d0p9_a08` | 2.241-2.259 | -1.3 % to +2.5 % | 1317-1397 | 29-38 |
| `UA_d0p1_a01` | 8.01-8.38 | **-21 % to -47 %** | 1377-1523 | 81-115 |

The mild tilts do **not** add net concentration; `UA_d0p9_a04` slightly
*raises* `N_eff` on 6 of 6 charge-seed cells. The mechanism is visible in the
tables: `B_toy` rows are low-weight rows, so up-weighting them moves mass
away from the heavy tail of `w_mc` and flattens the law. Only the strong tilt
(`delta = 0.1`, `alpha = 1`, `rho = 10`) inverts this and concentrates
sharply — `N_eff` drops to 390-497 and maximum single-row reuse rises from
~20 to 81-115 draws.

Cumulative unique source rows across the whole training run stay at 7 474 -
9 696 of 10 240 (73-95 %) for every variant, so no variant collapses onto a
narrow support. High reuse is reported, not treated as a failure and not
labelled OOD.

## 9. Concentration and support diagnostics

`concentration_diagnostics.json` separates the two sources of concentration
explicitly, per variant and charge:

- `source_weight_concentration_w_mc` — the concentration already present in
  the MC weights (`N_eff`, `N_eff / n_train`, top-10 and top-100 mass,
  zero-probability fraction, expected unique rows for one epoch);
- `variant_sampling_concentration` — the same quantities under the variant's
  actual sampling law;
- `additional_concentration_from_r_utility` — the deltas between them.

Also recorded: empirical unique rows per epoch, cumulative unique rows across
training, maximum row reuse, nominal and tilted `B_toy` probabilities, and the
amplification factor `p_U(B_toy) / p_0(B_toy)`.

No universal `N_eff` threshold is defined, high reuse is never automatically
labelled OOD, and no variant is called invalid merely because rejection or
concentration is large. Numerical failures (non-finite samples or
log-probabilities, failed fits) are reported separately from statistical
concentration; `generated_sample_finite_fraction` and
`generated_log_prob_finite_fraction` are the numerical channel.

## 10. Test-payload closure

`build_empirical_train_validation_dataset` (re-used unchanged from v1) never
indexes the test rows out of the capped array: only `len(test_indices)` and a
hash of the index list are computed. Every run records, per charge:

`test_row_count`, `test_split_hash`, `test_payload_loaded = false`,
`test_used_for_training = false`, `test_used_for_preprocessing = false`,
`test_used_for_utility_thresholds = false`,
`test_used_for_model_selection = false`, `test_used_for_evaluation = false`.

`test_utility_construction_never_reads_test_feature_values` is the regression
test: it records every array handed to the threshold fitter during utility
construction and asserts each one is exactly the corresponding charge's train
partition.

## 11. Golden-test repair

The parent branch carries one portable functional regression failure in
`tests/test_rare_aware_estimators.py::test_golden_arm_a_portable_functional_regression`.

**Reproduced first, unmodified.** On the parent tree the failure is:

```
assert 978.6004638671875 == 978.6005859375 +/- 9.8e-05   (feature_space_validation_nll)
```

Measured against the captured reference, with the test otherwise untouched:

| check | result |
| --- | --- |
| `best_step` | 2 — unchanged |
| history length | 3 — unchanged |
| state-dict key order and shapes | unchanged |
| parameter count | 1076 — unchanged |
| all state values finite | yes |
| local execution deterministic (two runs in-process) | yes, bit-identical |
| `state_dict_hash` vs the frozen reference | **identical** (`afcaa521…`) |
| pinned probe log-probabilities | **identical**, 0 ULP drift |
| `feature_space_train_nll` drift | 0.0 (0 float32 ULP) |
| `feature_space_validation_nll` drift | 1.2207e-4 — **exactly 2 float32 ULP** |

So the model is bit-identical and only one reported float32 *reduction* drifts
by two ULP: the assertion was pinning a summation order (BLAS kernel, thread
count, vector width), not the fit.

**Repair — documented float32 precision policy.** Portable assertions against
a captured float32 reference now use an absolute tolerance of a fixed small
number of float32 ULPs *at the expected magnitude*:

```
_GOLDEN_FLOAT32_ULP_TOLERANCE = 8
tolerance(expected) = 8 * ulp_float32(|expected|)
```

At the golden validation-NLL magnitude one ULP is `2^-14 = 6.1035e-5`, so the
budget is `4.883e-4` — a relative tolerance of about `5e-7`, four times the
largest measured drift and orders of magnitude tighter than any change that
would matter scientifically (a changed `best_step`, optimizer, or loss moves
these scalars by far more). `test_float32_ulp_tolerance_is_a_tight_documented_rule`
pins the rule itself.

Explicitly not done: the historical exact hash is unchanged; no list of
environment-specific hashes was added; nothing is `xfail`ed; the portable
functional test is not skipped; no unrelated assertion is loosened (the local
run-A-vs-run-B determinism checks keep their original exact-style
tolerances). The `reference_golden` exact-byte test stays separately gated
behind `SHIP_RUN_REFERENCE_GOLDEN=1`.

The repair is committed separately, ahead of the utility implementation.

## 12. Reproduce

```bash
git checkout experiment/d9-conditional-utility-tilt-v0
python -m pip install -e ".[dev]" && python -m pip install torch   # CPU

# implementation / unit tests
python -m pytest -q tests/test_conditional_utility_tilt.py \
                   tests/test_conditional_charge_nf.py \
                   tests/test_affine_coupling_flow.py \
                   tests/test_density_feature_pipeline.py \
                   tests/test_empirical_train_validation_dataset.py \
                   tests/test_utility_tilt.py

# two-epoch smoke (NOMINAL + UA_d0p9_a04, seed 11)
python scripts/run_conditional_utility_arena.py \
    configs/density_lab/conditional_utility/d9_fixture_smoke_v0.json \
    --output-root runs/d9_conditional_utility_smoke_v0

# full 4 variants x 3 model seeds fixture arena
python scripts/run_conditional_utility_arena.py \
    configs/density_lab/conditional_utility/d9_fixture_arena_v0.json \
    --output-root runs/d9_conditional_utility_arena_v0

# full portable suite (zero failures expected)
python -m pytest -q

# frozen-environment exact byte identity (skips by default)
SHIP_RUN_REFERENCE_GOLDEN=1 python -m pytest -q -m reference_golden
```

### Committed artifacts

The repository's `.gitignore` keeps run/campaign outputs (`artifacts/`,
`outputs/`) out of git, and this gate does not change that policy. No
checkpoint, generated event array, dataset, large log, cache, or
machine-specific absolute path is committed.

What *is* committed, under `docs/reviews/d9_conditional_utility_tilt_v0/`
(124 KB total), is the compact arena-level summary set:

| file | content |
| --- | --- |
| `arena_run_summary.json` | completion status, requested/completed matrix, per-seed preprocessing hashes and split hashes, lineage, and all 24 flattened result rows |
| `arena_aggregate.json` | per-(variant, charge) median/min/max across the three model seeds |
| `arena_summary.csv` | the 24 result rows, one row per (variant, seed, charge) |
| `arena_summary.md` | the human-readable arena table |

The per-run files the gate specifies —
`conditional_utility_sampling_manifest.json` (30 epochs x 2 charges of draw
provenance), `conditional_utility_training_metrics.json` (full training
history), `per_charge_nominal_validation.json`,
`per_charge_tilted_validation.json`, `per_charge_generated_summary.json`,
`concentration_diagnostics.json` and `summary.json` — are written into the
run directory by every run (3.9 MB for the arena) and are deliberately left
uncommitted as run output; the commands above regenerate them bit for bit.

## 13. Scientific interpretation

**1. Does generated `B_toy` occupancy move in the direction of the declared
tilt for each charge?** Yes for PDG 13 without exception, and with one
exception for PDG -13. Comparing each of the 3 tilt variants against the
nominal arm at fixed seed and charge (9 cells per charge): PDG 13 is at or
above nominal in **9 of 9** cells and strictly monotone in the declared tilt
strength in all three seeds. PDG -13 is at or above nominal in **8 of 9**
cells; the exception is seed 11 `UA_d0p9_a04`, where the *nominal* arm itself
overshot its own target (`gen/nom = 1.32`) and so sits above the mild tilt,
and seed 13 additionally ties `a04` and `a08` at 0.02026 rather than
separating them. Across seeds the medians are strictly ordered for both
charges — PDG 13 `0.0192 -> 0.0282 -> 0.0527 -> 0.1580` against declared
targets `0.0242 -> 0.0364 -> 0.0544 -> 0.1986`, and PDG -13
`0.0205 -> 0.0256 -> 0.0396 -> 0.1282` against `0.0264 -> 0.0397 -> 0.0592 ->
0.2132`.

**2. Does it track the magnitude of the tilted target?** Yes, over an
eight-fold amplification range, with a systematic shortfall that is *not*
introduced by the tilt. Median `gen/tilted` is 0.80 / 0.65 (`a04`), 0.88 /
0.67 (`a08`), 0.74 / 0.61 (`d0p1`) for PDG 13 / PDG -13 — but the nominal arm
shows the same kind of offset against its own target (median `gen/nominal`
0.71 / 0.96, range 0.39-1.32). The flow under-populates this tail region by a
comparable factor whether or not it is tilted, so the residual is a baseline
fidelity limit of a 6-block / width-64 flow on a 10 240-row fixture, not a
failure of the tilt mechanism. The tilt mechanism itself is exact: the
declared `pU` agrees with direct summation over the tilted table to
5.6e-17 across all 24 charge-variant-seed cells.

**3. Is either charge systematically less stable?** Yes — PDG -13 (mu+). Its
seed-to-seed spread in generated occupancy is wider at every variant
(nominal `gen/nom` [0.39, 1.32] versus [0.51, 0.90] for PDG 13; `UA_d0p9_a08`
[0.33, 0.82] versus [0.83, 1.00] on the tilted ratio), and its nominal
validation NLL spread is wider ([4.72, 6.25] versus [4.99, 5.78]). This is
not a concentration effect: PDG -13 has the *higher* `N_eff` of the two
charges under every variant. It is a tail-fidelity/seed-variance effect and
it is the main reason no single-seed claim is admissible here.

**4. How much does tilted validation improve?** Measured properly (section
8.5, same seed, same split, fixed measure): under the strong `UA_d0p1_a01`
measure the matching model beats the nominal model in **6 of 6** charge-seed
cells, by 0.20-0.67 nats (PDG 13) and 0.44-0.71 nats (PDG -13). Under their
own measures the mild tilts also win in 6 of 6 cells, but by much less:
0.04-0.28 nats (`a04`) and 0.06-0.48 nats (`a08`). The gain grows with the
tilt strength, as expected.

**5. How much does nominal validation degrade?** Essentially not at all for
the mild tilts, and mildly for the strong one. `UA_d0p9_a04` and
`UA_d0p9_a08` are *better* than the nominal model on the nominal measure in
6 of 6 charge-seed cells each (-0.01 to -0.47 nats) — at this fixture size the
mild tilt costs no measurable nominal coverage, and if anything acts as a
mild regularizer. `UA_d0p1_a01` degrades the nominal measure in 4 of 6 cells
(+0.02 to +0.56 nats) with a median of about +0.10 (PDG 13) and +0.02
(PDG -13). These deltas are paired same-seed differences; the unpaired
across-seed spread of the nominal arm alone is ~0.5-1.0 nats, so only the
paired comparison is meaningful and only the strong tilt shows a cost that
survives it.

**6. How much additional concentration and reuse does utility introduce
beyond `w_mc`?** For the mild tilts, none worth the name: `N_eff` moves by
-1.3 % to +2.5 % and maximum reuse from ~20 to 21-38 draws. For the strong
tilt, a lot: `N_eff` falls 21-47 % below the `w_mc` baseline and maximum
reuse rises 4-6x to 81-115. Crucially, the `w_mc` baseline is itself already
strongly concentrated (`N_eff / n_train ~ 0.06-0.07`) — most of the
concentration in this problem predates any utility tilt, and the mild
variants add nothing on top of it.

**7. Are conclusions consistent across seeds?** For the direction and rough
magnitude of the tilt, yes: the ordering of medians, the sign of the tilted
improvement (6/6), the sign of the mild-tilt nominal effect (6/6) and the
concentration picture reproduce across all three seeds. For any *quantitative*
occupancy claim, no: individual `gen/tilted` values swing by up to a factor
of 2.5 between seeds on PDG -13 (0.33-0.82 for `UA_d0p9_a08`). Three seeds is
enough to establish direction, not enough to fix a number.

**8. Is one moderate variant suitable for a later full-data replay?**
`UA_d0p9_a04` is the candidate, and only as **PROVISIONAL** (section 15).

*No universal winner is declared from any single seed, and none is declared
overall.* `B_toy` is not the physical endpoint, no physical background rate is
estimated, and no FairShip acceptance is claimed.

## 14. Authoritative empirical contract (unchanged)

- `w_mc` defines the current empirical **nominal** measure; nothing in this
  gate redefines it.
- The utility multiplier `r` is a separate **project-defined** quantity and is
  never called a physical weight.
- The synthetic `U_A` is **not** a calibrated physical acceptance probability,
  and `B_toy` is **not** the final physical endpoint.
- The exact downstream semantics of the source weights remain **OPEN**.
- A technical failure is not a physical negative.
- FairShip/GEANT4 remains the final downstream oracle. This gate makes no
  FairShip acceptance claim and produces no physical background rate.

## 15. Status ledger

| status | item |
| --- | --- |
| **VERIFIED** | Reproduction of the parent's portable golden failure as a pure 2-ULP float32 reduction-order drift (model bytes, `best_step`, history, structure, parameter count and probe log-probabilities all identical), and its repair under a documented ULP precision policy. Full portable suite: 707 passed, 53 skipped, **0 failures**; `reference_golden` passes under `SHIP_RUN_REFERENCE_GOLDEN=1` and skips explicitly otherwise. |
| **VERIFIED** | All twelve arena runs (4 variants x 3 model seeds) completed with finite samples, finite log-probabilities and no fit failures — `CONDITIONAL_UTILITY_ARENA_VERIFIED`. |
| **VERIFIED** | The `NOMINAL_PHYSICAL` arm reproduces the v1 nominal probability tables exactly (`r = 1`, identical table hash), so the v1 nominal contract is not regressed. |
| **VERIFIED** | No double weighting: `sample_weight_applied_to_loss = false` and `utility_weight_applied_to_loss = false` in every epoch record and manifest; every history record carries `weight_normalization = "epoch_direct_sampling_unweighted"`; the trainer raises if a sample weight is supplied on the epoch-sampler path. |
| **VERIFIED** | Equal per-charge draw counts in all 30 epochs of all 12 runs; the macro charge prior stays 1/2 -- 1/2 under every tilt. |
| **VERIFIED** | The preprocessing hash is identical across all four variants within each model seed, for all three seeds. |
| **VERIFIED** | The declared tilted `B_toy` probability agrees with direct summation over the tilted table to 5.6e-17 (max over all 24 charge-variant-seed cells). |
| **VERIFIED** | Test-payload closure: `test_payload_loaded`, `test_used_for_training`, `test_used_for_preprocessing`, `test_used_for_utility_thresholds`, `test_used_for_model_selection`, `test_used_for_evaluation` are all `false` in all 12 runs, with a regression test proving the utility construction reads only train rows. |
| **PROJECT DECISION** | One fixed *nominal* macro-weighted train-only preprocessing transform for all variants, fit once per (split, seed). Chosen to isolate the sampling measure as the only difference between arms; explicitly **not** a claim that nominal preprocessing is universally optimal. |
| **PROJECT DECISION** | The v0 arena is exactly four variants — the nominal arm plus binary-utility `UA_d0p9_a04`, `UA_d0p9_a08`, `UA_d0p1_a01` — using the repository's canonical variant ids. No `U_P` variant, no `alpha = 16` rung, no re-materialization of the historical 20-variant grid. |
| **PROJECT DECISION** | Concentration diagnostics are reported and never gate a run; no `N_eff` threshold and no automatic rejection criterion is defined. |
| **PROVISIONAL** | `UA_d0p9_a04` is the candidate moderate variant for a later full-data replay. It satisfies every condition the gate requires for that label: all three seeds completed; both charges finite in every diagnostic; generated occupancy moves consistently toward the tilted target (median `gen/tilted` 0.80 / 0.65, monotone in medians for both charges); no unresolved technical error; and the nominal cost and concentration cost are reported explicitly and are, at this fixture size, not measurable (nominal-measure delta negative in 6/6 cells; `N_eff` within +0.2 % to +1.4 % of the `w_mc` baseline). It is **provisional only**: it is a fixture result on 10 240 train rows per charge with three seeds, an eight-fold-smaller amplification than the strong arm, and a residual `gen/tilted` shortfall shared with the nominal arm. `UA_d0p9_a08` is a reasonable second candidate — higher amplification (2.25x) at still-negligible concentration cost — but its PDG -13 seed spread (`gen/tilted` 0.33-0.82) is wider than `a04`'s. `UA_d0p1_a01` is **not** recommended for replay: it is the only variant with a measurable nominal cost and a 21-47 % `N_eff` collapse. |
| **OPEN** | Variant-specific or tilted preprocessing. |
| **OPEN** | The exact downstream semantics of the source weights `w_mc`. |
| **OPEN** | Whether the residual `gen/target` shortfall (~0.6-0.9, shared by the nominal arm) is a capacity limit, a training-budget limit, or a fixture-size limit. This gate deliberately ran no hyperparameter sweep and cannot separate them. |
| **OPEN** | Any physical interpretation of `B_toy`. It is a synthetic diagnostic region; `U_A` is not a calibrated acceptance probability; FairShip/GEANT4 remains the final physical oracle. |
