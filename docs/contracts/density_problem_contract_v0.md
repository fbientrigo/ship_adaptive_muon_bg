# Density Modelling and Audit Contract (v0)

Status: **specification only — no density model is implemented by this document.**

This contract defines the first statistically testable density-modelling problem for
post-muon-shield muon states. It fixes the target of the next implementation increments,
the treatment of each raw column, the auditability requirements, the controlled benchmark
curriculum, and the gates that must be passed before a model is evaluated on the full
post-shield dataset.

It does **not** define a physical utility score, simulate muon DIS, estimate a SHiP
background rate, or validate candidates against FairShip/GEANT4.

## 1. Normative scope and precedence

This document is normative for the **nominal density/proposal track** only.

Where it conflicts with density-specific planning text in
`docs/architecture/ml_skeleton_local_pkl_v0.md`, this contract takes precedence. In
particular:

- `w` is not a generative feature;
- `z` is not modelled by default for the after-muon-shield single-plane dataset;
- tiny-sample overfitting is an implementation smoke test, not a model-selection metric;
- a real flow is not promoted directly from legacy code before controlled targets and
  audit gates exist;
- density modelling of `f(x)`, utility modelling of `U(x)`, utility tilting, and simulator
  validation remain separate problems.

The existing raw-data contract remains unchanged:

```text
[px, py, pz, x, y, z, id, w]
```

This document defines views derived from that validated raw array. It does not change the
on-disk schema or its dataset hash.

## 2. Scientific question

The first question is deliberately narrower than the final adaptive-simulation objective:

> What is the smallest auditable density model that reproduces the joint distribution,
> relevant low-mass modes, and tails of observed post-shield muon states, without relying
> on marginal agreement alone?

The density model answers where observed post-shield states lie. It does not answer whether
a state is dangerous, reconstructible, veto-surviving, or useful for a downstream analysis.
Those labels must come from a `simulation_backend` and belong to the later proxy/utility
track.

The initial target is therefore a nominal density `f(x)`. The future tilted proposal

```text
p_tilde_alpha(x) proportional to f(x) * U(x)**alpha
```

is explicitly out of scope until the nominal density path and its diagnostics are green.

## 3. Raw columns and v0 decisions

| Raw column | Unit | v0 role | Decision |
| --- | --- | --- | --- |
| `px` | GeV/c | continuous state | model through a declared feature view |
| `py` | GeV/c | continuous state | model through a declared feature view |
| `pz` | GeV/c | continuous state | require `pz > 0` for logarithmic/slope views; failures are explicit |
| `x` | m | continuous state | model |
| `y` | m | continuous state | model |
| `z` | m | scoring-plane metadata | exclude from the default after-MS density view |
| `id` | PDG code | discrete condition | train separate charge models in v0 |
| `w` | dimensionless | production/sample weight candidate | never use as a generated feature |

### 3.1 `z`

The committed after-muon-shield report shows an almost fixed `z` near one scoring plane.
Modelling that small variation as an ordinary standardized feature would amplify a nearly
constant coordinate and could make numerical noise look like learned structure.

For v0:

- `z` is recorded as `plane_z_metadata` and in the dataset report;
- samples are generated on the same declared plane;
- a multi-plane future dataset must introduce an explicit `plane_id` condition or a new
  contract version before `z` is modelled.

### 3.2 `id`

The first implementation trains one density model for `id = +13` and one for `id = -13`.
This has higher auditability than an embedding and avoids making charge-conditioning a hidden
source of failure in the first capacity study.

A conditional charge model is a later compression experiment. It may replace the two-model
baseline only if it passes the same gates for each charge separately and reduces cost or
parameter count without degrading tail/mode coverage.

### 3.3 `w`

The repository does not yet contain a sufficiently explicit, machine-readable provenance
contract for the precise semantics of every observed `w` value. Therefore the training target
must never be selected silently.

Two distinct targets are permitted:

1. `row_empirical_v0`: every stored row has equal mass. This is an engineering benchmark,
   not automatically the nominal physical mixture.
2. `production_weighted_nominal_v0`: rows are weighted by a verified non-negative
   `production_weight`, normalized within the declared dataset and charge.

`production_weighted_nominal_v0` is allowed only when the run config records:

```text
weight_semantics_id
weight_source_reference
zero_weight_policy
normalization_policy
```

The following quantities must remain separate in all artifacts:

```text
production_weight
training_sample_weight
utility_tilt
nominal_log_prob
proposal_log_prob
importance_weight
```

A single generic `weight` field is insufficient downstream.

## 4. Feature views

The benchmark must compare exactly two initial five-dimensional feature views. Both are
invertible on their declared domain.

### 4.1 Cartesian-log view

```text
cartesian_log_v0 = [px, py, log(pz / pz_unit), x, y]
```

with `pz_unit = 1 GeV/c` recorded in the view config.

Inverse:

```text
pz = pz_unit * exp(log_pz)
```

### 4.2 Slope-log view

```text
slope_log_v0 = [tx, ty, log(pz / pz_unit), x, y]
tx = px / pz
ty = py / pz
```

Inverse:

```text
pz = pz_unit * exp(log_pz)
px = tx * pz
py = ty * pz
```

The slope view is not assumed to be superior. It is a falsifiable representation hypothesis
that may simplify directional correlations.

### 4.3 Density coordinates and preprocessing Jacobians

Every reported likelihood must declare its coordinate system:

- `feature_space_log_prob`, or
- `physical_space_log_prob`.

If density is reported in physical coordinates, the deterministic feature transform and its
Jacobian contribution must be included. Standardization parameters are fitted on the training
split only and their affine Jacobian is also part of the coordinate conversion.

No result may compare absolute NLL values across different feature views unless all
preprocessing Jacobians have been included consistently in the same physical coordinate
system.

## 5. Split and provenance requirements

Every run records:

```text
schema_version
contract_version
raw_dataset_hash
feature_view_id
feature_view_config_hash
weight_target_id
charge
split_manifest_hash
normalization_hash
seed
git_commit
model_config_hash
```

Minimum partitions:

- `train`;
- `validation`;
- `test_nominal`;
- `test_tail` or a deterministic tail-region mask over `test_nominal`;
- `test_sparse` or a deterministic sparse-region mask over `test_nominal`.

A row-level random split is only a temporary fallback. If source-file, production, parent-event,
or reused-muon identifiers become available, splits must be group-aware. Until then, artifacts
must state:

```text
split_leakage_risk = "row_identity_only"
```

The committed range-preserving 40k samples contain deliberately forced envelope anchors and
are intended for schema/range inspection and local development. They must not be used to make
the final capacity or fidelity claim.

## 6. Operational auditability

Auditability has three separate levels:

1. **Technical traceability**: deterministic replay and complete transformation accounting.
2. **Statistical interpretability**: explicit dependencies, conditions, density, support and
   uncertainty diagnostics.
3. **Physical interpretability**: justified relation to SHiP physics or downstream outcomes.

This contract requires levels 1 and 2. Exact invertibility does not make latent coordinates
physically meaningful. No latent dimension may be assigned a physical interpretation without
an independent validation study.

### 6.1 Required density-model operations

A future implementation must support the semantics of:

```python
fit(x, *, sample_weight=None, condition=None) -> TrainingResult
log_prob(x, *, condition=None) -> array[n]
sample(n, *, condition=None, seed) -> array[n, d]
inverse(x, *, condition=None) -> array[n, d]
trace_sample(n, *, condition=None, seed) -> AuditTraceBatch
```

The exact Python API may evolve, but implementations must not lose these capabilities.

### 6.2 Point-level audit record

Each audited generated point must be serializable with at least:

```text
audit_schema_version
candidate_id
raw_dataset_hash
feature_view_id
weight_target_id
charge
model_name
model_config_hash
checkpoint_hash
git_commit
seed
sample_index
base_distribution_name
base_component_id          # nullable
base_point_z
block_names
intermediate_states
block_log_abs_det_jacobian
preprocessing_log_abs_det_jacobian
final_feature_point
final_physical_point
feature_space_log_prob
physical_space_log_prob    # nullable only when explicitly unsupported
inverse_base_point
round_trip_max_abs_error
support_gate_results
support_classification
```

Storing an entire trace for every production sample may be prohibitively large. Production
runs may store full traces for a deterministic audited subset plus compact provenance for all
points. The audited-subset rule and selected indices must be written before sampling.

### 6.3 Support classification

The labels `interpolation`, `sparse_support`, and `extrapolation` are operational diagnostics,
not statements of physical validity.

They must be produced from gates fitted on training data only, for example:

- hard domain constraints from the feature transform;
- per-feature envelope checks;
- multivariate k-nearest-neighbour distance or another fixed local-support statistic;
- charge-conditional reference statistics.

A point inside all marginal ranges is not necessarily inside multivariate support.

## 7. Controlled density curriculum

Synthetic targets are shaped approximately like the observed numerical problem but make no
claim to reproduce SHiP physics. Each target must provide deterministic `sample` and exact
`log_prob` implementations where stated.

All target parameters, mixture weights and transforms are versioned in the benchmark config.
Train, validation and test samples are drawn independently.

### D0 — diagonal Gaussian

- Five-dimensional diagonal Gaussian in one feature view.
- Exact sampling and density.
- Falsifies plumbing, normalization, sign, shape and serialization errors.
- Minimum plausible model: diagonal Gaussian.

### D1 — correlated Gaussian

- Five-dimensional full-covariance Gaussian with non-trivial correlations.
- Exact sampling and density.
- Falsifies models or metrics that only reproduce marginal scales.
- Minimum plausible model: full-covariance Gaussian or one affine transport.

### D2 — charge-separated multimodal mixture

- Separate known Gaussian mixtures for `id = +13` and `id = -13`.
- At least two components per charge with different means and covariance matrices.
- Exact sampling, density and component labels.
- Falsifies charge mixing, mode dropping and incorrect condition accounting.
- Minimum plausible model: charge-separated Gaussian mixture.

### D3 — nonlinear curved correlations

- D2 components passed through a known smooth invertible nonlinear warp, such as a
  triangular banana-style transform.
- Exact density obtained through the known inverse and Jacobian.
- Falsifies purely affine density families.
- Minimum plausible model: small affine coupling flow, subject to measured gates.

### D4 — asymmetric heteroscedastic multimodality

- Multiple components with different scales, correlations and a known monotonic skewing
  transform.
- Exact sampling and density.
- Falsifies coupling models with insufficient local flexibility.
- Minimum plausible model: not predeclared; affine and spline coupling are compared.

### D5 — rare tail component

- D4 plus a labelled component with mass evaluated at both `1e-2` and `1e-3`.
- The rare component occupies a declared tail region and has exact component membership.
- Exact sampling and density.
- Falsifies models selected by global NLL while dropping a low-mass mode.
- No model passes if the rare component is absent from the generated sample at a rate
  inconsistent with its target mass.

### Deferred D6 — disconnected or nearly degenerate support

D6 is not part of the first implementation PR. It is added only after D0-D5 are stable.
It is intended to expose topological bridges and thin-support failures, not to debug basic
training code.

### D7 — empirical after-MS data

D7 has no known closed-form target density. It is not entered until the controlled curriculum
and capacity frontier are complete. Exact KL and exact importance weights are unavailable
unless an additional explicit reference-density assumption is introduced.

## 8. Model ladder and capacity frontier

The first ladder is deliberately small:

1. diagonal Gaussian;
2. full-covariance Gaussian;
3. Gaussian mixture baseline;
4. affine RealNVP/coupling flow;
5. rational-quadratic spline coupling flow.

Autoregressive, mixture-of-flows, heavy-tail, continuous, residual and manifold flows are not
part of the first ladder. They require a specific reproducible failure that the initial ladder
cannot resolve.

Capacity is compared by:

```text
trainable_parameter_count
number_of_transforms
conditioner_width
conditioner_depth
spline_bin_count         # where applicable
training_wall_time
peak_memory
sample_throughput
log_prob_throughput
```

The minimum sufficient model for a benchmark level is the lowest-cost configuration that
passes every mandatory gate for at least four of five predefined seeds.

No architecture is selected because it is newer, more popular, or produces better-looking
marginal histograms.

## 9. Validation metrics and gates

No single scalar metric determines success.

### 9.1 Exact-target metrics for D0-D5

Mandatory:

- held-out NLL in a declared coordinate system;
- Monte-Carlo estimate of forward KL `KL(p || q)`;
- importance-weight `ESS/N` from exact `p(x) / q(x)`;
- classifier two-sample test (C2ST) with held-out classifier evaluation;
- component-mass recovery and mode recall where labels exist;
- rare-mode recall for D5;
- tail quantile and exceedance-probability error;
- support-violation rate;
- duplicate/near-duplicate rate and nearest-neighbour spread;
- forward/inverse round-trip error;
- consistency between total `log_prob` and the base plus per-block Jacobian sum;
- seed-to-seed variation.

### 9.2 Empirical metrics for D7

Mandatory:

- held-out NLL, with coordinate/Jacobian declaration;
- C2ST;
- multivariate support diagnostics;
- conditional results per charge;
- tail and sparse-region discrepancies;
- distributional precision/recall or a documented equivalent;
- nearest-neighbour diversity;
- Jacobian and round-trip stability;
- seed-to-seed variation.

An ESS reported for D7 must identify the reference density. It may be exact only for a
well-defined discrete resampling distribution; otherwise it is explicitly labelled
model-dependent or approximate.

### 9.3 Threshold policy

Numerical pass thresholds are not hard-coded in this document because they must depend on
test sample size and Monte-Carlo resolution. Before model comparison, the benchmark config
must pre-register thresholds using:

- target-vs-target reference variability;
- bootstrap or repeated independent samples;
- the intended tail probability resolution;
- a fixed confidence level.

Hard safeguards:

- `ESS/N < 0.01` is catastrophic collapse, never a pass;
- any missing declared D5 rare mode is a failure;
- non-finite density/Jacobian values are a failure;
- a failed inverse round trip above the pre-registered numerical tolerance is a failure;
- support-violation rates above the pre-registered target-vs-target reference are a failure.

Passing the safeguards is necessary but not sufficient.

## 10. Tiny-overfit smoke test

Tiny overfit is required only to verify optimization and density plumbing.

Use fixed subsets with sizes drawn from:

```text
32, 128, 512
```

The test checks:

- deterministic behavior for a fixed seed;
- finite loss and gradients;
- decreasing train NLL relative to initialization;
- consistent `log_prob` decomposition;
- stable inverse and round trip;
- checkpoint save/load equivalence.

A near-zero NLL is not required and is not comparable across units or feature views. Tiny
training loss is never evidence of generalization, support coverage, tail fidelity or physical
validity.

## 11. Required artifacts

Every non-trivial density run writes:

```text
config_resolved.json
run_manifest.json
split_manifest.json
feature_view.json
normalization.json
training_metrics.json
density_metrics.json
support_diagnostics.json
audit_trace_manifest.json
model_manifest.json
checkpoint.*
```

Arrays use NPZ or the model backend's documented checkpoint format. JSON artifacts contain
schema versions and hashes. Large generated samples stay out of Git.

`run_manifest.json` must include environment and dependency versions. A result without seed,
dataset hash, feature view, split and model config is not trusted.

## 12. Red-team gates

The implementation and review process must explicitly test:

### R1 — good marginals, poor joint proposal

Detected by exact KL/ESS on D0-D5, C2ST, support diagnostics and joint tail tests.

### R2 — rare-mode loss

Detected by D5 component mass, rare-mode recall and exceedance probability. Global NLL cannot
waive this gate.

### R3 — out-of-distribution support

Detected with domain checks and multivariate support statistics, not marginal min/max alone.

### R4 — contaminated cost benchmark

Until FairShip is operational, only model training/inference and synthetic useful-region costs
may be reported. No FairShip speed-up claim is permitted.

### R5 — preprocessing Jacobian omission

Detected by analytic transform tests and agreement between feature-space and physical-space
densities.

### R6 — weight-semantic conflation

Detected by schema validation that keeps production, training, utility and importance weights
separate.

## 13. Implementation sequence unlocked by this contract

After review and merge, the permitted sequence is:

1. implement feature-view transforms and analytic Jacobian tests;
2. implement D0-D5 exact targets and manifests;
3. implement Gaussian baselines and metric reference tests;
4. run a short backend audit for candidate flow libraries;
5. implement an auditable affine coupling baseline;
6. implement an auditable rational-quadratic spline coupling model;
7. compute the controlled capacity frontier;
8. evaluate only surviving configurations on the full after-MS dataset;
9. add D6, heavy-tail bases, autoregressive models or mixture flows only after a
   documented failure justifies them;
10. begin the separate utility-tilting track only after the nominal density gates pass.

## 14. Non-goals

This contract does not authorize:

- a physical `U(x)` trained without simulator-produced labels;
- an estimate of SHiP background rates;
- a claim that generated states are physically valid because they pass support diagnostics;
- FairShip/ROOT imports into the NumPy-only core;
- event-level conservation constraints on independent post-shield muon rows;
- use of `w` as an input feature;
- automatic promotion of `Nflow/legacy/` code;
- CNF, residual, manifold, diffusion or production-scale model work;
- ONNX/SOFIE deployment or inference optimization;
- model selection from marginal plots alone.

## 15. Open boundary conditions

The following must be resolved or explicitly carried as limitations before the D7 claim:

- exact provenance and semantics of `w` for each source production;
- whether source-file, parent-event or reused-muon identifiers can support group-aware splits;
- the full after-MS dataset access path and immutable hash;
- the scoring-plane and geometry metadata associated with that production;
- CPU/GPU budget used to define the capacity-search stopping rule;
- pre-registered metric thresholds and test sample sizes.

These open questions do not block D0-D5 implementation. They do block an unqualified claim
that a weighted D7 model represents the nominal SHiP post-shield distribution.

## 16. Definition of done for this documentation increment

This increment is complete when reviewers agree that:

- the modelled variables and transformations are unambiguous;
- `id`, `z` and `w` cannot be silently misused;
- audit traces and density coordinate conventions are explicit;
- D0-D5 each isolate a stated failure mode;
- the capacity frontier cannot be selected from overfit or marginal plots alone;
- the next implementation PR can add targets/tests without choosing a flow backend.

No code, dependencies, models, generated datasets or physics claims belong in this increment.
