# Controlled Targets D3-D5 Contract (v0)

Status: **normative for the D3-D5 controlled-density implementation.**

Extends `docs/contracts/controlled_targets_v0.md` (D0-D2) with three
transformed targets. Implemented by
`src/ship_muon_bg/benchmarks/target_transforms.py`,
`target_regions.py` and `controlled_targets.py`
(`TransformedControlledTarget`). These are numerical controlled benchmarks,
**not** SHiP physics: no energy variable, no event-level conservation, no
FairShip/ROOT/GEANT4, no proxy/utility logic.

## 1. Coordinate convention and pz policy

All targets are defined in canonical physical coordinates `[px, py, pz, x, y]`.
Every transform **preserves `pz` exactly** (column index 2), so the
positive-`pz` policy stays auditable and no feature view is privileged. All
feature views receive the same physical target samples.

## 2. Exact invertible transform contract

`ExactTransform` implementations provide:

```
forward(base_physical) -> transformed_physical
inverse(transformed_physical) -> base_physical
forward_log_abs_det_jacobian(base_physical) -> (N,)
inverse_log_abs_det_jacobian(transformed_physical) -> (N,)
manifest() -> JSON-serializable dict
```

Requirements (all tested): exact analytic inverse (no root finding), exact
analytic log-determinant (checked against central finite differences),
float64, deterministic, no input mutation, forward/inverse round trip, `pz`
unchanged, and explicit coordinate scales in the manifest.

Implemented transforms:

- **`TriangularBananaTransform`** — `px` is the root; `py` receives a quadratic
  function of normalized `px`, `x` a linear function of normalized `px`. Unit
  Jacobian determinant (`log|det J| = 0` exactly).
- **`SinhArcsinhSkewTransform`** — elementwise `t = s·sinh(b + asinh(u/s))`;
  monotonic, closed-form inverse, `log|dt/du| = log cosh(b+asinh(z)) - ½log(1+z²)`.
  `pz` skew is fixed to 0 (exact identity, zero log-Jacobian on `pz`).
- **`ComposedTransform`** — left-to-right forward, right-to-left inverse,
  additive log-Jacobians.

`TransformedControlledTarget` wraps an exact mixture base and a transform. Its
physical `log_prob` uses the exact inverse and inverse log-Jacobian:
`log p_T(t) = log p_base(inverse(t)) + inverse_log_abs_det_jacobian(t)`.

## 3. D3 — curved nonlinear correlations

A PDG-id-conditioned correlated Gaussian mixture through the triangular banana
transform. Creates clearly nonlinear correlations (quadratic `py`–`px`
dependence) with a unit Jacobian. D3 demonstrably falsifies a full Gaussian
while remaining learnable by a small coupling flow (smoke: full-Gaussian
forward KL ≈ 0.80 vs affine-flow ≈ 0.18 on the identity view). Fixed
parameters are versioned in the manifest.

## 4. D4 — asymmetric heteroscedastic multimodality

A versioned heteroscedastic mixture (multiple components, different scales,
different full covariances, asymmetric weights, separate parameters per PDG id)
under a sinh-arcsinh skew composed with a triangular banana transform. Exact
inverse and Jacobian, no singular derivative, `pz` unchanged. Harder than D3
for an affine-only family.

## 5. D5 — labelled rare tail mode

The D4 heteroscedastic family plus one well-separated rare component, in two
versioned variants: `rare_1e-2` (rare mass `1e-2`) and `rare_1e-3` (rare mass
`1e-3`), selected via `make_controlled_target("D5", variant=...)`. Rare rows are
never forced or oversampled in the nominal sample; train/validation/test are
independent draws.

Rare region: a fixed 4σ Mahalanobis ellipsoid around the rare mode in **base**
coordinates (`MahalanobisRegion`). `region_mask` maps a transformed physical
point back through the exact inverse and tests the ellipsoid, so D5 recovery
never depends on any model's internal component labels.

The manifest records `rare_mass`, `rare_component_id`, `rare_region_definition`,
and a deterministic MC calibration (`calibration_seed`, `calibration_n`,
`target_probability_in_rare_region`, its `target_probability_stderr`, and
`main_component_contamination_in_rare_region`). Calibration constants are
versioned in code and **not** recomputed during training;
`calibrate_d5_rare_region` recomputes them deterministically for the tests. The
calibrated rare-region probability matches the rare mass to within a few
percent with zero main-mode contamination.

## 6. Extended target API

Every exact target additionally exposes `component_log_prob`,
`component_posterior`, `declared_regions` and `region_mask`. All methods
operate in canonical physical coordinates.

## 7. Acceptance tests

Enforced by `tests/test_controlled_targets_d3_d5.py`: exact deterministic
sampling; forward/inverse round trip; analytic vs numerical Jacobian; density
change-of-variables consistency; feature-view physical log-prob recovery for
A/B1/B2; component-posterior normalization; no `pz` change from transforms; D5
rare-component frequency on a large deterministic sample; D5 rare-region
calibration vs versioned manifest with low contamination; manifest/hash
determinism; distinct D5-variant hashes; import hygiene (no torch/ROOT/etc.).

## 8. Non-goals

No energy variable, no event-level conservation, no DIS/FairShip/ROOT/GEANT4,
no proxy/utility logic, no SHiP background-rate or speed-up claim, and no change
to the raw 8-column schema, the existing `FeatureView` formulas, or the D0-D2
API.
