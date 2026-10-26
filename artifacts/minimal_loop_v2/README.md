# Minimal loop v2: iterations 3 and 4, seed variance and DIS budget (2026-10-25/26)

This is an **experiment session** on the minimal-loop pipeline. Labels: **FEATURE** / **EXPERIMENT** / **VERIFIED** /
**PROVISIONAL** / **OPEN**, as in v1. Stage: PR #1432 filter, same FairShip configuration as v0/v1 (`400c0fb`,
`TRY_2026`, filter option 1). Iteration 4 changes one option: `n_dis_per_material` 10 -> 100. Not `Y^(k)`, not
reconstruction, not `B`. Contract `SHIP-EMPIRICAL-FS-SIM-CONTRACT-V0-SOURCE`: still not on disk (OPEN).
All DIS counts below are **unweighted** (the native `wDIS` weights are recorded but their semantics are OPEN).

## What we wanted

1. **Iteration 3:** draw with U2. G3 is exhausted, so the rule moves the budget to G1 and G0. Check U2 out of
   sample there.
2. **Seed variance:** repeat already-evaluated states with fresh seeds. Is a filter PASS a property of the state or
   of the seed?
3. **Iteration 4 (added after the seed result):** is the per-run cost driven by the DIS budget? If not, raise it.

## Feature vs experiment

| item | kind |
|---|---|
| `draw-repeats`: k more executions of evaluated states, drawn per (class, previous outcome), seeds never used before (`a4b7bd0`) | FEATURE |
| groups 4/4/2/2 states x 3 repeats; budget 30 + 36 (iter. 3) and 12 + 3 (iter. 4); `n_dis_per_material = 100` | EXPERIMENT |
| per-DIS analysis `per_dis.py` -> `per_dis.json` (outside the pipeline, reads the exported Parquet) | EXPERIMENT |

Seeds: 66 (iter. 3) and 15 (iter. 4) fresh, unique, none shared with any earlier execution (VERIFIED).

## Iteration 3 (66 runs, 0 technical failures, 0 NOT_EVALUATED, 56 min)

**U2 out of sample** (`iter3/u_v3.json: evaluation_of_previous`, 30 fresh states, one run each):

| class | U2 predicted | observed | Clopper-Pearson 95 % |
|---|---:|---:|---|
| G1_upstream | 0.176 | 1/21 = 0.048 | 0.001 - 0.24 |
| G0_none | 0.038 | 0/9 | 0 - 0.34 |

Both inside the intervals; G1 came out lower than U2 predicted.
U3 = `u-cbea592c9146` (89 states): G0 0.029, G1 0.072, G2 0.333, G3 0.702.

**Seed repeats** (3 fresh-seed runs per state; "before" = all earlier runs):

| class | previous | state | p [GeV] | before | repeats |
|---|---|---|---:|---:|---:|
| G3 | pass | 12851 | 193 | 1/1 | 2/3 |
| G3 | pass | 33946 | 243 | 1/1 | 1/3 |
| G3 | pass | 34629 | 111 | 1/1 | 3/3 |
| G3 | pass | 36227 | 229 | 1/1 | 2/3 |
| G3 | fail | 21705 | 217 | 0/1 | 3/3 |
| G3 | fail | 31938 | 227 | 0/1 | 2/3 |
| G3 | fail | 3409 | 197 | 0/1 | 2/3 |
| G3 | fail | 37566 | 24 | 0/1 | 0/3 |
| G1 | pass | 18020 | 35 | 1/1 | 0/3 |
| G1 | pass | 30425 | 43 | 1/1 | 0/3 |
| G1 | fail | 4843 | 51 | 0/1 | 1/3 |
| G1 | fail | 857 | 48 | 0/1 | 0/3 |

- 9 of 12 states give mixed outcomes. In G3 a previous pass is followed by a pass in 8/12 repeats, a previous fail in
  7/12: the first outcome barely predicts the next.
- The seed drives both stages: the injection (transport) and the DIS generation. The set of materials crossed is the
  same in every repeat of a state. The path lengths are not: e.g. row 37566 AIR 13104.7 cm before vs
  13077.1-13096.0 cm in the repeats; row 29823 SBTsens 3095.7 / 3275.9 vs 3466.4 cm (raw `chain/facts.json`, found by
  review 5). These repeats show seed-dependent outcomes across the pipeline. They do not separate the transport part
  from the DIS part.

## Why: filter_any is "at least one of ~30 DIS passes"

- `filter_any == any(per-DIS filter pass)` on all 146 evaluated executions (VERIFIED, `per_dis.py`). All 7530 DIS
  carry an evaluated filter decision (143 pass); none is missing.
- With 10 DIS per material a run has 10 (G0: AIR only), 30 (AIR+MS+UBT) or up to 60 DIS. Runs with 30 DIS have 0-3
  passes; runs with 50-60 DIS have up to 6 (e.g. 29823: 6/60).
- Within G3, the per-state per-DIS counts do not reject one common rate (chi-square 19.1 on 18 dof, p = 0.39, budget-10
  evidence). This is a weak test: 16 of 19 expected counts are below 5, and it uses the Poisson variance. Not detecting
  heterogeneity does not show that the states are homogeneous.
- With `1 - (1 - 0.036)^30 = 0.66`, a single per-DIS rate reproduces the G3 per-run rate (0.65-0.75 observed).
- Within a run, passing DIS do not cluster visibly: in the 30 G3 runs with exactly 30 DIS, the counts are 11/9/9/1 runs
  with 0/1/2/3 passes, against 10.8/11.2/5.6/1.8 for a binomial (variance ratio 0.86). This covers only those runs. DIS
  of one run share a transported muon. Rates differ by material (below). The generator splits events between proton
  and neutron targets. Independence is plausible conditionally on the run; it is not established.
- So, at budget 10, a single per-run label is a noisy measurement of a state, and "distinct states with a PASS per
  hour" (v1: 29/h) partly counts lucky draws. How much of the spread is between-state variation is not determined.
  The class ordering G3 > G1 > G0 is not affected.

## Iteration 4: the DIS budget is cheap (15 runs at 100 DIS per material, 13 min)

Mean cost by number of DIS in the run (`per_dis.json: cost_by_dis_in_run`, 146 runs; different rows are different
states):

| DIS in run | runs | injection [s] | chain [s] | total [s] |
|---:|---:|---:|---:|---:|
| 10 | 33 | 35.1 | 16.1 | 51.2 |
| 30 | 78 | 34.0 | 16.0 | 50.1 |
| 100 | 5 | 33.9 | 16.6 | 50.6 |
| 300 | 8 | 34.3 | 17.0 | 51.3 |
| 400 | 1 | 35.0 | 17.8 | 52.8 |
| 600 | 1 | 36.9 | 18.4 | 55.3 |

- Fixed overhead dominates: ~34 s injection + ~16 s chain at any budget. The marginal cost is small but not zero:
  600 DIS took 5.3 s (10 %) more in total than a 30-DIS run.
- Matched states (same state, budget 10 -> 100 per material): 29823 +10.3 %, 34615 +6.5 %, 33048 +2.9 % total time.
- So one run at 100 per material gave ~10x the DIS of a budget-10 run for 3-10 % more time. 300 per material was not
  run and is an extrapolation.

Per-DIS rate of the same state at both budgets (G3 states rerun at 100 per material):

| state | 10 per material | 100 per material |
|---|---:|---:|
| 29823 | 9/120 | 37/600 |
| 34615 | 2/40 | 24/400 |
| 33048 | 0/30 (labelled fail) | 14/300 |

These three pairs are compatible with an unchanged rate. With 3 pairs this is no equivalence test, and budget effects
are not bounded.

## Per-DIS rate by class, all 146 executions (`per_dis.json`)

| class | states | runs | DIS | passing DIS | per-DIS rate, CP 95 % (binomial model, nominal) |
|---|---:|---:|---:|---:|---|
| G0_none | 38 | 38 | 830 | 0 | 0 - 0.0044 |
| G1_upstream | 43 | 55 | 3540 | 7 | 0.0020 (0.0008 - 0.0041) |
| G2_sbt | 1 | 3 | 150 | 0 | 0 - 0.024 |
| G3_tracker | 19 | 50 | 3010 | 136 | 0.045 (0.038 - 0.053) |

- These are pooled rates of this adaptively collected sample. They weight a state by its runs, its budget and its
  number of materials. In G3, 3 states supply 43 % of the DIS. The mean of the 19 per-state rates is 0.038. At budget
  10 alone, G3 is 61/1710 = 0.036; the 3 states rerun at budget 100 give 75/1300 = 0.058.
- The intervals assume i.i.d. Bernoulli DIS (see above). Their coverage is nominal.
- Passing DIS by material (unweighted):
  - G3 has passes in every material. They range from AIR 17/770 to SBT sensitive 26/160.
  - G1 has 4/1180 in AIR, 3/1180 in UBT and 0/1180 in MS.
  - G0 has 0/830, all in AIR.
- **Coverage, unchanged lesson:** G0 is 73.7 % of the generator-supported eligible pool (23087 / 31340, `p >= 2 GeV`).
  Its per-DIS upper bound (0.44 %) is still above the G1 point estimate (0.20 %), so G0 cannot yet be shown to be
  negligible.

## What this suggests for the next iterations (proposals, not implemented)

- **Endpoint:** count passing DIS out of DIS evaluated instead of "any DIS passes". It gives many more trials per run
  and does not mix the outcome with the budget the way `filter_any` does.
  - It needs a new estimator definition: unit = DIS realization, a state/material weighting, and `wDIS` weighting
    (OPEN).
  - It needs a model for DIS that share a run.
  - It needs an explicit rule for pooling budgets. Today a different budget is a different stage, by design.
  - Decision for the user.
- **Budget:** 100 per material is measured cheap (3-10 % more time). Higher budgets should be timed before relying on
  them.
- **Where to spend muons:** G0 at a high budget, to bound its per-DIS rate below G1's. If none pass in 3000 AIR DIS, the
  one-sided 95 % bound is 0.10 % (two-sided CP: 0.12 %), under the binomial model.
- **Fresh states vs repeats:** repeats measure the simulation variability of a state; fresh states also measure the
  variation between states and widen coverage. Fresh states are preferred for class rates. A few repeats per class
  keep the variability measured.

## Limits and OPEN

- Repeat groups were chosen by previous outcome, and `draw-repeats` only takes states whose earlier runs all agree
  (`scripts/run_minimal_loop.py`, `draw_repeats`). The "previous outcome" contrast is a regression-to-the-mean check,
  not a population estimate.
- Review 5 (Codex, claims): 0 BLOCKING, 5 MAJOR, 3 MINOR. All are corrected above: path lengths do change with the
  seed; pooled rates are sample-weighted; independence and homogeneity are not established; cost is not fixed; 0-3
  passes held only for 30-DIS runs; interval convention; pool denominator. The accounting (151 executions, 7530 DIS,
  143 passes, no join duplication) reproduced.
- The homogeneity tests have low power (few passes per state). Heterogeneity in G3 is not excluded (37566 at
  p = 24 GeV is 0/160).
- Unweighted DIS counts; physical weighting by `wDIS`, muon `w` and exposure: OPEN. Placement gate, `+2.5916 m`, B5,
  `Y^(k)` vs filter: unchanged from v0.
- Classes were defined after iteration 1 (v1); this session tests U2 and the noise model, not the class choice.

Raw evidence: `~/thesis/scratch/minimal-loop-v0/iter{3,4}/runs/<draw_id>/`. Here: draws, Parquet datasets and
snapshots, U3, `per_dis.py` (reads every exported dataset of v0-v2) and its output.
