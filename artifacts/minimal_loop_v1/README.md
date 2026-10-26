# Minimal loop v1: iteration 2, U refitted from data and runs allocated by strategy (2026-10-25)

This is an **experiment session** built on the minimal-loop pipeline. Labels: **FEATURE** (a pipeline behaviour that
stays and is reused by later iterations) / **EXPERIMENT** (a setting or hypothesis of this session only) /
**VERIFIED** / **PROVISIONAL** / **OPEN**. Stage: PR #1432 filter (`pr1432_filter_any_v0`) only, same FairShip
configuration as v0 (`400c0fb`, `TRY_2026`, 10 DIS/material, filter option 1). Not `Y^(k)`, not reconstruction,
not `B`. Contract `SHIP-EMPIRICAL-FS-SIM-CONTRACT-V0-SOURCE`: still not on disk (OPEN).

## What we wanted

1. **Close the loop.** Fit U from the iteration-1 evidence only, use it to choose the next states, run FairShip, and
   check U on states it never saw (the out-of-sample test that the circular pilot lacked).
2. **Spend expensive runs where they teach or yield something.** Each run costs ~50 s (34 s injection).

## Feature vs experiment

| item | kind | note |
|---|---|---|
| Straight-ray flags in `score` (`ray_upstream`, `ray_tracker`) | FEATURE | U0 and initial volume unchanged (VERIFIED equal on all rows) |
| `update-u`: U fitted only from frozen snapshots, versioned (`u_id`), with an out-of-sample check of the previous U | FEATURE | unit = distinct state, endpoint = pass per run |
| `draw-strata`: never run a known outcome (`p < 2 GeV`, already-evaluated states), without replacement, floor per class, rest proportional to U; inclusion probability recorded | FEATURE | the strategy for expensive runs |
| Classes `G0_none` / `G1_upstream` / `G2_sbt` / `G3_tracker` (ray crosses nothing / UBT / SBT / Tr1-4) | EXPERIMENT | chosen after seeing iteration-1 materials, so iteration 1 is not a test of the classes |
| U = Laplace rate per class; floor 5; budget 30 | EXPERIMENT | |

## Why these classes (iteration-1 evidence)

The materials FairShip actually traversed (`prepare` path lengths) separate the iteration-1 outcomes:
AIR+REST only, 0/19 states passed; plus MS+UBT (no SBT), 1/7; plus SBT, 2/3. On these three categories the cheap
ray agrees with the transported materials for **29/29** evaluated states (VERIFIED). The materials do not test the
tracker subdivision (G2 vs G3). Both passing SBT states also cross
`Tr1-4`, and the SBT state that failed (31238) only grazes the SBT side.

## U1 (`u_v1.json`, `u-b83071ed9d15`, fitted on 29 iteration-1 states) and allocation (`draws/draws.json`)

| class | fresh eligible, p >= 2 GeV | U1 | runs |
|---|---:|---:|---:|
| G0_none | 23068 | 0.048 | 5 |
| G1_upstream | 8226 | 0.222 | 8 |
| G2_sbt | 0 | 0.333 | 0 |
| G3_tracker | 17 | 0.750 | 17 (stratum exhausted) |

The allocation came from the pre-review rounding (5/8/0/17). True largest remainder would give 6/8/0/16, fixed
afterwards (`3f51286`). Both put a floor in every non-empty class.

## Results (30 executions, 0 technical failures, 0 NOT_EVALUATED, 25 min; `data/report.json`)

**Out-of-sample check of U1 (`u_v2.json: evaluation_of_previous`, 30 fresh states, one run each):**

| class | U1 predicted | observed | Clopper-Pearson 95 % |
|---|---:|---:|---|
| G3_tracker | 0.75 | 11/17 = 0.65 | 0.38 - 0.86 |
| G1_upstream | 0.22 | 1/8 = 0.13 | 0.003 - 0.53 |
| G0_none | 0.05 | 0/5 | 0 - 0.52 |

Each prediction lies inside its interval. The ordering G3 > G1 > G0 held on states U1 never saw. This is the first
non-circular evidence for a U in this project. It is still small-sample.

**Yield per cost (distinct states with a PASS per wall hour):**

| | runs | distinct states with a PASS | per hour |
|---|---:|---:|---:|
| iteration 1, uniform reference | 20 | 1 | 3.6 |
| iteration 1, U0 tilt (with replacement) | 20 | 2 | 7.3 |
| **iteration 2, stratified by U1** | 30 | **12** | **29.0** |

## Coverage and bias: what enrichment does not see

Plug-in estimate over the generator-supported eligible pool (31340 states, `p >= 2 GeV`):

| class | share of pool | per-run PASS rate, all states so far | contribution to the pool's PASS fraction |
|---|---:|---|---:|
| G3_tracker | 0.061 % (19) | 13/19, census: all 19 evaluated | 0.0004 |
| G2_sbt | 0.003 % (1) | 0/1 | 0 |
| G1_upstream | 26.27 % (8233) | 2/15 (nominal 0.02 - 0.40) | 0.035 (nominal upper 0.11) |
| G0_none | 73.67 % (23087) | 0/24 (nominal 0 - 0.14) | 0 (nominal upper 0.10) |

How the classes were sampled:
- In G0 and G1 every state has U0=0. Both the reference and the iteration-1 tilt are uniform there, and
  iteration 2 is uniform over the states not yet evaluated, so the evaluated set is a uniform-like sample.
- G3 was not sampled uniformly: the iteration-1 tilt favoured its 2 U0=1 states 10^4-fold. G3 is now a census.
- The intervals are nominal binomial intervals. They do not account for the adaptive design or for repeated runs
  of a state.

What this suggests, with large uncertainty:
- The ~8x observed gain in distinct PASS states per hour over the uniform reference comes mostly from **exhausting
  a 19-state class**.
- On the plug-in estimate, that class carries about 1 % of the pool's expected PASS fraction. G1 likely carries
  more. G0 cannot be bounded below the G1 point estimate: 0/24 does not show that G0 has no passes.
- Per-run efficiency and coverage of the positive population are different objectives. The exploration floor is
  what measures the second one.
- G3 is exhausted (19/19 evaluated). Under U2 the same allocation rule moves the budget to G1 and G0.

## U2 (`u_v2.json`, `u-2778ad8c2fca`, fitted on 59 states)

G0 0.038 (24 states), G1 0.176 (15), G2 0.333 (1), G3 0.667 (19).

## Reviews (Codex, read-only)

- **Review 3** (code, before the run), 0 BLOCKING:
  - U endpoint changed from "ever passed" to "per-run rate". The numbers are identical for U1 because the repeats
    were unanimous.
  - U and stage compatibility are now enforced.
  - Allocation rounding fixed.
- **Review 4** (claims, after the run), 0 BLOCKING:
  - Corrected the within-class uniformity claim (false for G3), the pool shares (denominator `p >= 2 GeV`, G2
    included), the 1/7 material count and the rounding.
  - Toned down the population conclusions.
  - Leakage check passed: all 30 states are fresh and drawn with U1's classes.

## Limits and OPEN

- Classes were defined after looking at iteration 1. Iteration 2 tests U1, not the choice of classes.
- 30 runs give wide intervals.
- One run per state, so seed variance inside a state is not measured here (v0: 32661 gave 4/4).
- Placement gate and `+2.5916 m`, B5, weights, `Y^(k)` vs filter: unchanged from v0.
- Pool is the 40000-row after-MS sample, not the full measure. No physical weights.

Raw evidence: `~/thesis/scratch/minimal-loop-v0/iter2/runs/<draw_id>/`.
Code: `scripts/run_minimal_loop.py` (`update-u`, `draw-strata`).
