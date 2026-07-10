# Feature-View Matched A/B Experiment Contract (v0)

Status: **normative for representation comparisons in the density track.**

This contract corrects the assumption that a logarithmic or slope-based representation is
already the preferred input to a Normalizing Flow. Those representations are hypotheses.
They must remain measurable experiment arms beside an untransformed Cartesian reference.

## 1. Scope and precedence

This document is an amendment to
`docs/contracts/density_problem_contract_v0.md`.

Where the density contract states that the first benchmark compares exactly two transformed
views, this contract takes precedence. The initial representation experiment has one reference
arm and two candidate arms. No candidate is promoted by construction.

This amendment also governs how feature views enter:

- D0-D5 controlled targets;
- Gaussian and Normalizing Flow baselines;
- capacity-frontier searches;
- D7 after-muon-shield evaluation;
- audit traces and run manifests;
- later utility-tilting experiments.

It does not change the raw schema:

```text
[px, py, pz, x, y, z, id, w]
```

`z`, `id`, and `w` retain the roles fixed by the density contract and are not added to the
continuous generated vector.

## 2. Scientific question

The representation question is:

> Under otherwise matched conditions, does a deterministic transform reduce the minimum
> model capacity or improve joint, tail, support, and numerical diagnostics relative to the
> untransformed Cartesian state?

Invertibility, a valid Jacobian, or visually simpler marginals are necessary engineering
properties. They are not evidence that a representation is statistically superior.

## 3. Experiment identifier and arms

The experiment identifier is:

```text
feature_view_ab_v0
```

The term A/B is used for a controlled baseline-versus-candidate experiment. There are two
candidate variants, so the concrete design is A/B1/B2.

### Arm A — reference

```text
identity_cartesian_v0 = [px, py, pz, x, y]
```

Properties:

- exact identity map;
- zero preprocessing log-Jacobian;
- accepts every finite value of the five modelled coordinates;
- carries no claim that Cartesian coordinates are optimal;
- remains the permanent regression and interpretability reference.

### Arm B1 — log-longitudinal candidate

```text
cartesian_logpz_v0 = [px, py, log(pz / pz_unit), x, y]
```

with a positive, recorded `pz_unit`, initially `1 GeV/c`.

Hypothesis:

- compressing the positive longitudinal-momentum scale may reduce asymmetry and model
  capacity.

Known risk:

- the transform has no domain at `pz <= 0` and therefore may exclude or expose problematic
  rows.

For numeric momentum coordinates stored in GeV/c:

```text
log|det dT/dx| = -log(pz)
```

### Arm B2 — slope/log-longitudinal candidate

```text
slope_logpz_v0 = [px/pz, py/pz, log(pz / pz_unit), x, y]
```

Hypothesis:

- separating directional slopes from longitudinal scale may simplify post-shield
  correlations.

Known risks:

- ratios amplify small-`pz` behaviour;
- algebraic coupling to `pz` can create heavy or unstable tails;
- a simpler visual distribution does not guarantee better proposal quality.

For numeric momentum coordinates stored in GeV/c:

```text
log|det dT/dx| = -3 * log(pz)
```

## 4. Matched-comparison rule

A feature-view comparison is valid only when all non-representation variables are matched.
Every comparison group records identical values for:

```text
raw_dataset_hash
weight_target_id
charge
split_manifest_hash
seed
model_name
model_config_hash
training_budget_id
metric_config_hash
```

The only intended difference is:

```text
feature_view_id
feature_view_config_hash
normalization_hash
```

The normalization hash is expected to differ because normalization is fitted independently on
the same training rows after each deterministic transform.

Comparisons must use:

- the same raw rows and row order;
- the same train/validation/test indices;
- the same random seeds;
- the same optimizer and stopping budget;
- the same architecture configuration when evaluating a matched model;
- the same generated-sample count and metric configuration.

A result that changes representation and architecture simultaneously is not an A/B result.
It belongs to the capacity-frontier matrix and must be labelled accordingly.

## 5. Domain and row-eligibility policy

Arm A is defined for all finite physical rows. Arms B1 and B2 require `pz > 0`.
No implementation may clip, add an epsilon, take an absolute value, or silently discard rows.

Every dataset evaluation first writes a domain-coverage report:

```text
total_rows
finite_physical_rows
valid_rows_by_arm
invalid_indices_hash_by_arm
invalid_reason_counts_by_arm
common_valid_indices_hash
```

Two results are required:

1. **Matched-support comparison** — all arms use the same `common_valid_indices`, normally the
   intersection of their declared domains. This isolates representation and model effects.
2. **Native-domain accounting** — each arm reports coverage on the complete validated raw
   dataset. Loss of relevant rows is a cost of that representation, not preprocessing noise.

For D7, a non-positive `pz` row blocks an unqualified promotion of a logarithmic arm until its
provenance is resolved. The identity arm keeps the row visible so the issue cannot disappear
through preprocessing.

## 6. Density-coordinate accounting

Every likelihood declares either:

```text
feature_space_log_prob
physical_space_log_prob
```

Cross-arm NLL comparisons are allowed only in physical coordinates after including:

- the deterministic feature-view Jacobian;
- the train-fitted normalization Jacobian;
- any model-level Jacobian.

Feature-space NLL is retained as a debugging metric. It cannot select a winning arm because
the coordinate systems differ.

Forward and inverse Jacobians must cancel within a pre-registered numerical tolerance. The
analytic feature-view Jacobian must be checked against a numerical determinant in unit tests.

## 7. Continuous parallel execution

The reference and candidate arms remain visible throughout model development.

At each density milestone, the repository must run at least one matched configuration for:

```text
identity_cartesian_v0
cartesian_logpz_v0
slope_logpz_v0
```

This applies to:

- Gaussian plumbing baselines;
- D0-D5 targets;
- affine-coupling implementation checks;
- rational-quadratic spline implementation checks;
- D7 pilot and final evaluation;
- later nominal-versus-tilted proposal tests.

A complete Cartesian product is not required at every commit. To control compute cost:

1. all arms run smoke and reference configurations;
2. successive halving may reduce expensive candidate sweeps;
3. the identity reference is never removed from reporting;
4. every eliminated arm retains its last matched evidence and elimination reason.

This design gives a continuous measurement of whether preprocessing helps, has no effect, or
creates a failure as model capacity increases.

## 8. Controlled-target requirement

D0-D5 must not be defined in a way that gives one feature arm privileged coordinates.

Each target must satisfy one of these conditions:

1. it is defined in canonical physical coordinates with exact sampling and density, then
   transformed into every arm; or
2. it provides exact, mutually consistent density evaluation under every experiment arm.

All arms receive the same underlying target samples. Target parameters, component labels,
rare-mode membership, and test masks are shared.

A target described as diagonal only after one candidate transform cannot be used to claim that
candidate requires less model capacity. That would encode the answer in the benchmark.

## 9. Evaluation and selection

No single metric selects a representation. The matched report includes at least:

- domain coverage and invalid-row reasons;
- physical-space held-out NLL;
- exact forward KL and ESS/N where available;
- C2ST;
- component and rare-mode recovery;
- tail quantile and exceedance-probability error;
- multivariate support diagnostics;
- round-trip and Jacobian stability;
- training wall time and inference throughput;
- minimum passing parameter count;
- variation across the same five seeds.

### Promotion rule

A candidate may become the default training view only when it passes all mandatory density
gates for at least four of five predefined seeds and demonstrates at least one material benefit:

- lower minimum passing parameter count;
- lower training cost at equal statistical quality;
- improved joint/tail/support metrics at equal model cost.

It must not introduce an unresolved loss of domain coverage or numerical stability.

Promotion changes the default, not the existence of Arm A. The identity reference remains in
regression experiments.

### Retirement rule

A candidate can be retired from expensive sweeps when a pre-registered matched evaluation
shows a reproducible failure, such as:

- unacceptable domain loss;
- instability near `pz = 0`;
- failure of Jacobian or inverse tolerances;
- strictly worse statistical gates and cost across two consecutive benchmark levels.

A retired arm remains implemented, tested, and documented unless its code itself is invalid.

## 10. Required artifacts

Every run adds these fields to the density artifacts:

```text
feature_view_experiment_id
feature_view_id
feature_view_role
feature_view_config_hash
comparison_group_id
domain_coverage_report_hash
common_valid_indices_hash
training_budget_id
metric_config_hash
```

The experiment-level artifact records:

```text
baseline_arm_id
candidate_arm_ids
matched_keys
arm_run_ids
eliminated_arms
elimination_reasons
promotion_status
```

The point-level audit trace continues to store the selected feature-view ID, preprocessing
Jacobian, feature point, and reconstructed physical point.

## 11. Software boundary

The density-model implementation must receive a `FeatureView` configuration or transformed
array plus its manifest. It must not hard-code log-`pz`, slopes, or Cartesian coordinates.

The feature-view module remains NumPy-only and independent of:

- PyTorch and a flow backend;
- FairShip and ROOT;
- proxy labels and `U(x)`;
- production-weight semantics.

A flow checkpoint is invalid without the feature-view config hash and normalization hash used
to train it.

## 12. Red Team

### R1 — apparent gain caused by coordinate-dependent NLL

Control: compare physical-space densities with all Jacobians included.

### R2 — candidate wins by discarding difficult rows

Control: matched-support comparison plus native-domain coverage accounting.

### R3 — target constructed in the candidate's preferred coordinates

Control: canonical physical target or exact densities for every arm.

### R4 — representation and model changes confounded

Control: matched model configurations before capacity-frontier comparisons.

### R5 — slope instability hidden by global metrics

Control: explicit small-`pz` masks, tail tests, finite checks, and per-arm domain reports.

### R6 — permanent model complexity from an untested transform

Control: Arm A remains the regression reference and candidates require measured promotion.

## 13. Non-goals

This contract does not:

- claim that logarithmic or slope variables are physically preferred;
- define a physical utility score;
- authorize silent data correction at `pz <= 0`;
- estimate SHiP background rates or FairShip speed-up;
- choose a Normalizing Flow backend;
- replace D0-D5 statistical gates with A/B significance alone;
- require an exhaustive hyperparameter product at every commit.

## 14. Definition of done

This increment is complete when:

- the identity Cartesian arm is implemented as the permanent reference;
- both transformed hypotheses are explicit candidate arms;
- all arms have forward, inverse, manifest, hash, and analytic Jacobian behaviour;
- domain differences are observable and never silently repaired;
- tests verify formulas, round trips, numerical Jacobians, and failure modes;
- future model runs can group matched arms by experiment metadata;
- no transformed arm is described as the selected representation before evidence exists.
