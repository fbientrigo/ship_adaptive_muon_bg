# Minimal loop v0: one real FairShip iteration (2026-10-25)

Labels: **VERIFIED** (observed, check named) / **PROJECT DECISION** / **PROVISIONAL** / **OPEN**.
Authority: `SHIP-EMPIRICAL-FS-SIM-CONTRACT-V0-SOURCE` was **not on disk** during this work; semantics follow
`docs/contracts/tagging_contract_v0.md`. Any mismatch with the contract is OPEN.

## What ran

`pool (40000 after-MS rows) -> TGeo gate + U0 -> 20 draws/arm -> TTree injection -> PR #1432 prepare -> filter`
on FairShip PR #1432 `400c0fb4ae31` (`TRY_2026`, `strawDesign 10`, `--noSND`, 10 DIS/material, filter option 1).
Code: `scripts/run_minimal_loop.py` (`score | draw | run | export | report`). Branch `feat/minimal-loop-v0`.

**Endpoint: the PR #1432 filter stage (`pr1432_filter_any_v0`) only.** Not `Y^(k)`, not replay/reconstruction,
not selection/veto, not `B`.

## Gate (PROJECT DECISION, PROVISIONAL)

`score/pool_scores.json`: rescored with the PR #1432 geometry; **VERIFIED** identical (U0 and initial volume, all
39998 rows) to the 2026-10-12 `u0_main.npz`.

| status | rows |
|---|---:|
| ELIGIBLE (`cave` 32657, `Cavern` 249) | 32906 |
| INELIGIBLE_IN_SHIELD_MAGNET (`Magn7_*`, incl. 7 `MiddleMag`) | 7092 |
| INVALID_ROW (22837: p=0; 26902: w=-0) | 2 |

- U0=1: 14 rows, **9 of them start inside `Magn7_*`** (incl. 8217, 8637, 22373, 25740, 32029, 35078 of the
  earlier stage-outcome batch). Eligible U0=1: 29823, 31238, 31588, 32661, 36915.
- Being in the cave does not show a row lies on the surface PR #1432 expects (B5, OPEN). The `+2.5916 m` transform
  stays PROVISIONAL. No coordinate or geometry correction was made.

## Arms (`draws/draws.json`)

- `reference_uniform_eligible`: uniform over eligible rows. **Empirical reference, not the nominal physical
  measure**: `w` is not applied (lineage/exposure OPEN).
- `tilt_u0`: `h = [0.1 + 0.9 U0]^4` over eligible rows. 60 % of its mass is on the 5 eligible U0=1 rows
  (effective rows 13.7).
- 20 draws each, with replacement, seed 20261025; one simulation seed per draw, **VERIFIED** propagated to the
  injection (`run_simScript -s/-r`) and to `prepareEvents --seed` for all 40 runs, with 40 unique seeds
  (`runs/*/inject/request.json` against the chain `facts.json` in scratch).

## Results (`data/report.json`)

| | reference | tilt |
|---|---:|---:|
| executions / distinct states | 20 / 20 | 20 / 12 (max multiplicity 4) |
| technical failures | 0 | 0 |
| NOT_EVALUATED (generator `p < 2 GeV`) | 1 | 4 (31588, 36915 twice each) |
| valid filter evaluations | 19 | 16 |
| filter PASS executions | 1 | 6 |
| distinct states with a PASS / with a valid execution | 1 / 19 | **2 / 10** |
| wall time | 0.275 h | 0.276 h |
| PASS per wall hour (execution unit) | 3.6 | 21.8 |

- Cost: 49.6 s per execution (33.8 s injection, ~16 s prepare + filter); 40 executions in 33 min.
- The tilt's 6 PASS executions come from **two distinct source states**: 32661 (4/4) and 29823 (2/2). The 21.8/h
  counts executions. Repeated executions of one state are not new draws of the state measure, and the independence
  of their outcomes is not established. Per distinct state, 2/10 against 1/19 is descriptive: states repeated more
  often get more chances to register a PASS. It is not a demonstrated enrichment.
- **Missed region:** the one reference PASS (row 18020, U0=0, p = 34.6 GeV) shows that U0=1 is not necessary for
  a filter PASS. Under the tilt, each U0=0 row has `h` weight `1e-4` relative to a U0=1 row. Its normalized draw
  probability is 0.40 times the reference's (2.5-fold lower), and the U0=0 region keeps about 40 % of the tilt
  mass (`report.json: missed_region`). In the tilt, U0=0 gave 0/7 (one-sided 95 % bound 35 %).
- Coverage (`coverage_deciles_of_eligible_pool`): the reference reaches 9/8/8/9 of the 10 deciles of p/pT/x/y
  (20 draws). The tilt puts 11/20 draws in a single x decile and 7/20 in the top y decile, and leaves 3 p-deciles empty.
  Both arms have the same eligible support; this is reduced *realised* coverage under the concentrated proposal.
- Not repeated as evidence: the earlier 8/9 vs 0/13 result (circular, and mostly from in-magnet states).

## Data and snapshots

- `data/<arm>/{executions,realizations,candidates,observations,decisions}.parquet`: the bundle records copied row
  for row. `draws.parquet` holds proposal origin (`proposal_prob`, `u0`, `h`, seed) and `cost.parquet` holds wall
  times. `manifest.json` holds the table hashes and source `bundle_file` paths.
- `data/snapshots/snap-*/snapshot.json`: frozen stage id, estimator id (`eta_hat_positive_over_valid_v0`),
  configuration and options digest, U version (scores hash, delta, alpha), sampling policy, and the grouped split
  (by `subject_id`). `rows.parquet` holds the per-state labels, with `eta_hat = None` when nothing was valid.
- Raw evidence (ROOT, logs, chain `facts.json`, bundles) is in `~/thesis/scratch/minimal-loop-v0/runs/<draw_id>/`
  (178 MB, not in git). The paths are in each manifest.

Query example: `duckdb.sql("select * from 'data/tilt_u0/decisions.parquet' where stage_definition_id like 'pr1432_filter_any_v0@%'")`.

## Reproduce

```
env -i HOME=$HOME PATH=/usr/bin:/bin bash -c "source ~/thesis/scratch/fairship-update-1012/env_pr1432.sh && \
  python scripts/run_minimal_loop.py score --fairship ~/thesis/worktrees/FairShip-pr1432-400c0fb \
  --geo ~/thesis/scratch/fairship-update-1012/pr1432_inputs/runs/25740/geo_connector.root --out S"
PYTHONPATH=src .venv/bin/python scripts/run_minimal_loop.py draw --scores S/pool_scores.npz --out D --n 20 --seed 20261025
env -i ... python scripts/run_minimal_loop.py run --fairship ~/thesis/worktrees/FairShip-pr1432-400c0fb --draws D/draws.csv --runs R
PYTHONPATH=src .venv/bin/python scripts/run_minimal_loop.py export --draws D/draws.csv --runs R --data P   # then: report
```

## OPEN

B5 (post-shield states as MS-DIS input); the `+2.5916 m` placement; weight/exposure semantics; geometry content
identity in the pooling key; `Y^(k)` vs filter equivalence; NF-generated states (next iteration: same gate, same
contract); a learned U (2 positive states are not training data).
