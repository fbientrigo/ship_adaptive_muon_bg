# D9 conditional-charge normalizing flow v0

## Scope and lineage

This branch starts at the verified Phase 1 remote tip
`a53eec41a47152f134395df460dde584f208582b` from
`experiment/d9-full-data-replay-v0`. It does not merge the historical
row-empirical D9-5N branch and does not modify the closed Phase 0 results.

The implementation is fixture-only. It does not train on the 13.8M-row full
dataset, run utility tilting, call FairShip/GEANT4, estimate a physical rate,
or evaluate a model on the test split.

## Mathematical target

The model is one shared conditional flow

\[
q_\theta(x\mid c),\qquad x=[p_x,p_y,p_z,x,y],
\]

with the explicit external condition

\[
13\mapsto c=+1,\qquad -13\mapsto c=-1.
\]

The generative law is `C ~ rho(c)`, `Z ~ N(0,I)`, and
`X = T_theta(Z; C)`. The joint target is
`Q_theta,rho(x,c) = rho(c) q_theta(x|c)`. The condition is never generated or
included in the five-dimensional transformed Jacobian.

The v0 training target is macro-balanced:

\[
L(\theta)=\frac12L_{13}(\theta)+\frac12L_{-13}(\theta),\qquad
L_c=\sum_{i:C_i=c}\frac{w_i}{\sum_jw_j}[-\log q_\theta(x_i|c)].
\]

For an even draw budget `T`, the runner draws `T/2` rows per charge with
replacement from `pi_i|c = w_i/sum(w)`, concatenates them, and deterministically
shuffles the pool. Therefore ordinary unweighted NLL on the draw pool has
expectation `L_c` within each charge, and equal charge allocation gives the
macro objective. `w_mc` is not multiplied into the loss again.

Odd total draw budgets are rejected in v0. Zero-weight rows remain in the
source table and have zero draw probability.

## Architecture and preprocessing

`AffineCouplingFlow` now accepts `condition_dim=1`. Each coupling conditioner
receives the masked x input concatenated with the external charge sign. The
invertible map remains over x only, preserving the unconditional API when
`condition_dim=0`. The pooled standardization is fitted once on the sampled
train draw pool; validation is not used to fit preprocessing.

The committed configs are:

- `configs/density_lab/conditional_charge/d9_fixture_smoke_v0.json`
- `configs/density_lab/conditional_charge/d9_fixture_pilot_v0.json`

## Fixture pilot

The pilot used the committed `data/samples/muonsFullMC_afterMS_sample.npz`,
seed 11, CPU, one shared model, 10 epochs, and 4,096 draws per charge. The
fixture manifest identifies the parent full-data canonical hash as
`44336e8e3629149c813026cf21334c6fc2db75ae9ffa12e78ed2b064f3a54579`, the
fixture subset hash as
`718b9197a3965f6842574fec1f3f6b1bf4c16949ff8a1f27dca6232817c03c32`, and the
parent file SHA-256 as
`fb251be2cbfb04aed52bee215c23cbd0898eccae989523952d3b774c004f974c`.

| PDG | condition | train / validation / test | weighted validation NLL | generated B_toy | generated finite | log-prob finite |
|---:|---:|---:|---:|---:|---:|---:|
| 13 | +1 | 12,664 / 3,166 / 3,958 | 6.8811176314 | 0.00634765625 | 1.0 | 1.0 |
| -13 | -1 | 12,936 / 3,234 / 4,042 | 6.0572568030 | 0.0068359375 | 1.0 | 1.0 |

`L_val,macro = 6.4691872172` and `L_val,worst = 6.8811176314`. The fit status
was `ok`, with 10 history steps and best step 7. These are fixture-pilot
diagnostics, not a claim of physical superiority or final acceptance.

### Sampling diagnostics

| PDG | source-table hash | alias hash | N_eff | N_eff / n_train | top-10 mass | top-100 mass | zero-probability fraction | expected unique | empirical unique / max reuse | nominal B_toy |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 13 | `5f6a4d1bcd03eb5f63f0d865d7ecb2892ac90909b7b8be342d028bd5ecd3de18` | `27baec4807ea812ed188026d7cce63042c02af93e98e059c130663133685137b` | 775.006 | 0.0612 | 0.0199513 | 0.199513 | 0 | 1459.04 | 1472 / 18 | 0.0263129 |
| -13 | `4640b30587bb882f51221cd692abd02c446b0a97d2ac8831ebea6ce1977ee656` | `c4be27435bc0430768c4f30ddf363388b095d6c6eae29daed8a7bc32b9235f1b` | 925.140 | 0.0715 | 0.0158566 | 0.158566 | 0 | 1444.07 | 1417 / 13 | 0.0272438 |

Low effective sample size is recorded as concentration, not an automatic
failure. The diagnostics distinguish concentration already present in `w_mc`
from any additional utility reweighting; no utility reweighting was used here.

## Validation and generated-distribution contract

Validation is never resampled. For each charge the runner computes
`-sum(w_i log q_theta(x_i|c))/sum(w_i)`, then reports the arithmetic macro mean
and worst charge. It also records per-charge weighted empirical summaries,
generated summaries, correlation matrices, train-only `B_toy` prevalence,
generated `B_toy` occupancy, finite fractions, and sampling diversity.

The test partition is loaded only to preserve split provenance and is excluded
from training, preprocessing, model selection, and evaluation.

## Symmetry audit

Repository evidence was searched in the density contracts, controlled-target
definitions, tests, and afterMS replay documentation. It establishes distinct
charge distributions and separate historical models, but does not define an
exact post-Muon-Shield charge-reflection plane, coordinate map, or geometry
invariance. No hard symmetry, mirroring, or augmentation is therefore used.

The audit layer always runs the no-transform empirical comparison and refuses
an undocumented transform. Current status is `OPEN`; no claim of charge
symmetry is made.

## Reproduction

```powershell
$env:PYTHONPATH = "src;."
python scripts/run_conditional_charge_nf.py `
  configs/density_lab/conditional_charge/d9_fixture_smoke_v0.json `
  --output-dir artifacts/density_lab/d9_conditional_charge_nf_v0/smoke

$env:PYTHONPATH = "src;."
python scripts/run_conditional_charge_nf.py `
  configs/density_lab/conditional_charge/d9_fixture_pilot_v0.json `
  --output-dir artifacts/density_lab/d9_conditional_charge_nf_v0/pilot
```

The pilot lineage recorded `code_commit =
6dcf1a409776208cadf17a4178a81e0062308a95`, model config hash
`82e48a048f06541425a2ef904d99a39a8e1b8652f3dabc81ee57f204666b8871`, pooled
preprocessing hash
`a2d07a87a0fdda5be83e42258ac4aab6923bc36ff0d6bcccb4d77892e42d66bc`, and
checkpoint/state fingerprint
`abb16445671f02860b94853ea4e4e0d6d032a171ada3021b70bb78fb021fcadf`.

## Artifact handoff

The pilot generated, but does not commit, the following compact files under
`artifacts/density_lab/d9_conditional_charge_nf_v0/pilot/`:

`conditional_sampling_manifest.json`, `conditional_training_metrics.json`,
`per_charge_validation.json`, `per_charge_generated_summary.json`,
`symmetry_audit.json`, and `conditional_fixture_summary.json/.csv/.md`.

The recorded byte sizes were 2,883; 5,525; 5,949; 5,888; 10,978; 32,427;
148; and 657 bytes respectively (the `.md` file is `report.md`). No generated
samples, checkpoint files, caches, or large artifacts are committed.

## Status boundaries

- **VERIFIED:** branch provenance; optional conditional architecture;
  unconditional regression compatibility; deterministic balanced sampler;
  zero-weight handling; train-only pooled preprocessing; per-charge weighted
  validation; finite fixture smoke and 10-epoch pilot; deterministic compact
  artifacts.
- **PROJECT DECISION:** one shared conditional model; `13 -> +1`, `-13 -> -1`;
  macro-balanced physical-nominal direct sampling; no double weighting; no
  hard symmetry.
- **PROVISIONAL:** one CPU fixture seed is sufficient only to test execution,
  conditioning, per-charge stability, and diagnostic production. It does not
  establish physical superiority or a symmetry.
- **OPEN:** multi-seed fixture study, full-data training, conditional-vs-
  separate-model scientific comparison, an evidence-backed reflection map,
  FairShip validation, and any physical-rate estimate.
