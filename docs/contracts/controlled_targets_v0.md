# Controlled Targets D0-D2 Contract (v0)

Status: **normative for the D0-D2 controlled-density implementation.**

This document specifies the first three targets of the controlled density
curriculum defined in `docs/contracts/density_problem_contract_v0.md` section
7. It is implemented by `src/ship_muon_bg/benchmarks/controlled_targets.py`.

## 1. Purpose

D0, D1 and D2 are exact, closed-form numerical benchmark distributions used
to falsify plumbing, sign, normalization, correlation-handling and
charge/mode-mixing errors in density models before any model is evaluated on
real post-shield data. They make no claim to reproduce SHiP physics.

## 2. Coordinate convention

Every target is defined directly in canonical physical coordinates:

```text
[px, py, pz, x, y]
```

Targets are not defined in log-`pz` or slope/log-`pz` feature coordinates.
`sample` and `log_prob` accept and return exactly these five columns.

## 3. Mathematical definitions

### D0 — diagonal Gaussian

One component, `component_id = 0`, shared numerically across both charges:

```text
log N(x; mean, diag(std^2)) = sum_i [ -0.5*log(2*pi*std_i^2) - 0.5*((x_i - mean_i)/std_i)^2 ]
```

Sampling: `x = mean + std * eps`, `eps ~ N(0, I)` drawn from
`np.random.default_rng(seed)`.

### D1 — full-covariance Gaussian

One component, `component_id = 0`, shared numerically across both charges.
`covariance` is symmetric positive-definite, validated via Cholesky
decomposition `covariance = L L^T`:

```text
log N(x; mean, covariance) = -0.5*d*log(2*pi) - 0.5*log|covariance|
                              - 0.5*(x - mean)^T covariance^-1 (x - mean)
```

The quadratic form and `log|covariance|` are computed from `L` (solving
`L z = (x - mean)` and using `log|covariance| = 2*sum(log(diag(L)))`) rather
than an explicitly formed inverse.

Sampling: `x = mean + L @ eps`, `eps ~ N(0, I)`.

### D2 — charge-conditioned Gaussian mixture

Two diagonal-covariance components per charge, with independent means,
covariances and weights per charge (`13` and `-13`). Weights are positive and
sum to 1 within each charge.

```text
p(x | charge) = sum_k weight_k * N(x; mean_k, covariance_k)
log p(x | charge) = logsumexp_k( log(weight_k) + log N(x; mean_k, covariance_k) )
```

`logsumexp` is a NumPy-only, max-shifted stable implementation. Component
labels are drawn exactly (`rng.choice` on the declared weights) and returned
alongside each sample.

## 4. Fixed parameter tables (v0)

All values are exact and versioned; changing any one of them changes
`config_hash()`.

### D0 (`shared_across_charges`)

| | px | py | pz | x | y |
| --- | --- | --- | --- | --- | --- |
| mean | 0.0 | 0.0 | 50.0 | 0.0 | 0.0 |
| std  | 3.0 | 3.0 | 4.0  | 0.5 | 0.5 |

`mean_pz / std_pz = 12.5`.

### D1 (`shared_across_charges`)

mean: `[0.0, 0.0, 50.0, 0.0, 0.0]`

covariance (order `px, py, pz, x, y`):

```text
[[ 9.00, 1.80, 1.20, 0.75, 0.00 ],
 [ 1.80, 9.00, 1.20, 0.00, 0.75 ],
 [ 1.20, 1.20, 16.00, 0.10, 0.10 ],
 [ 0.75, 0.00, 0.10, 0.25, 0.05 ],
 [ 0.00, 0.75, 0.10, 0.05, 0.25 ]]
```

Non-trivial correlations: `corr(px, x) = corr(py, y) = 0.5`,
`corr(px, py) = 0.2`. `mean_pz / std_pz = 50 / 4 = 12.5`.

### D2 (`charge_conditioned_independent_mixtures`)

Charge `+13`:

| component | weight | px | py | pz | x | y | std(px,py,pz,x,y) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.6 | 2.0 | 1.0 | 45.0 | 0.3 | 0.2 | 3.0, 3.0, 4.0, 0.5, 0.5 |
| 1 | 0.4 | -2.0 | -1.0 | 60.0 | -0.3 | -0.2 | 4.0, 4.0, 5.0, 0.6, 0.6 |

Charge `-13`:

| component | weight | px | py | pz | x | y | std(px,py,pz,x,y) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.55 | -1.5 | 2.5 | 55.0 | 0.1 | -0.4 | 3.5, 3.0, 4.5, 0.5, 0.5 |
| 1 | 0.45 | 3.0 | -2.0 | 48.0 | -0.2 | 0.3 | 4.0, 3.5, 4.0, 0.6, 0.6 |

All component `mean_pz / std_pz` ratios: `11.25, 12.0` (charge `+13`),
`12.22, 12.0` (charge `-13`).

## 5. pz-domain policy

Targets are exact, untruncated Gaussians/mixtures. No clipping, rejection,
`abs()`, epsilon or silent resampling is applied at `pz <= 0`.

Every declared component satisfies `mean_pz / marginal_std_pz >= 10`; this is
validated at target-construction time and raises
`ControlledTargetConfigError` if violated. `probability_pz_nonpositive` is
computed analytically per component (and mixture-weighted per charge in
total) from the one-dimensional Gaussian marginal using `math.erf`, and is
reported in the manifest. At the declared margins this probability underflows
to exactly `0.0` in float64 (well below `1e-15`).

The frozen smoke configurations below must draw zero `pz <= 0` rows; a
violation is a test failure, not something to be repaired:

```text
(D0, charge=+13, n=4096, seed=11)
(D0, charge=-13, n=4096, seed=11)
(D1, charge=+13, n=4096, seed=7)
(D1, charge=-13, n=4096, seed=7)
(D2, charge=+13, n=4096, seed=3)
(D2, charge=-13, n=4096, seed=4)
```

## 6. Charge policy

`charge` is always an explicit call argument (`13` or `-13`), even when D0/D1
use the same numeric distribution for both charges
(`charge_parameterization = "shared_across_charges"`). D2 uses
`"charge_conditioned_independent_mixtures"`: charges have independent
component parameters and are observably distinct.

## 7. Manifest fields

Every target's `manifest()` includes at least:

```text
target_schema_version
target_id
target_description
density_coordinate            = "physical_px_py_pz_x_y"
physical_columns              = ["px", "py", "pz", "x", "y"]
supported_charges              = [13, -13]
charge_parameterization
component_count_by_charge
mixture_weights
means
covariance_matrices
probability_pz_nonpositive
exact_sample                  = true
exact_log_prob                = true
physics_claim                 = false
event_level_conservation_applied = false
```

`config_hash()` is a deterministic SHA-256 of the canonical (sorted-key) JSON
encoding of `manifest()`; any parameter change changes the hash.

## 8. Acceptance tests

Enforced by `tests/test_controlled_targets.py`:

- factory accepts `D0`/`D1`/`D2` and rejects unknown ids;
- invalid `n`, `charge`, `seed`, shapes, non-finite input and invalid
  covariance configurations fail explicitly with a typed error;
- fixed-seed sampling is bitwise-deterministic; different seeds differ;
- frozen smoke configurations have the declared shape, dtype, finiteness,
  C-contiguity and `pz > 0` for every row;
- D0/D1/D2 `log_prob` match independently computed reference formulas at
  fixed points;
- D1 covariance is symmetric positive-definite and genuinely correlated;
- D2 component labels are valid and empirical mixture fractions agree with
  declared weights within a predeclared statistical tolerance;
- D2 charge-13 and charge-(-13) samples are observably distinct;
- manifests and config hashes are deterministic, and changing one parameter
  changes the hash;
- `embed_physical_to_raw` preserves physical values exactly and sets
  `z`/`id`/`w` as specified, without mutating its input;
  the same physical samples pass through all three existing `FeatureView`
  arms (`identity_cartesian_v0`, `cartesian_logpz_v0`, `slope_logpz_v0`) and
  recover the original physical log-probability through the existing
  Jacobian accounting;
- importing `ship_muon_bg.benchmarks` pulls in no torch/ROOT/FairShip/h5py/
  scipy/pandas/sklearn.

## 9. Non-goals

This contract and its implementation do not:

- add an energy variable or any energy/momentum conservation constraint;
- model DIS products or any FairShip/ROOT/GEANT4 behavior;
- add proxy labels, `U(x)`, utility tilting, or background-rate logic;
- change the raw 8-column schema, the `DensityModel` interface, the existing
  raw-normalization implementation, or any existing `FeatureView` formula;
- select or claim a winning feature-view arm;
- truncate, clip, take `abs()`, add epsilon to, or silently resample `pz`;
- introduce D3-D7 or any model ladder beyond D0-D2.
