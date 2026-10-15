# Stage-wise DIS outcomes on the PR #1432 chain — v0

Date: 2026-10-13. Branch `feat/stage-outcomes-v0` (worktree `worktrees/ship-stage-outcomes-v0`, from `1f16aeb`; **not committed**).
Labels: **VERIFIED** (observed, with the check named) / **PROJECT DECISION** (the user decided) / **PROVISIONAL** (engineering convention) / **OPEN**.
Authority: `SHIP-EMPIRICAL-FS-SIM-CONTRACT-V0-SOURCE` is **not on disk** (the clause families `OUTCOME-*`, `LINEAGE-*`, `FAIL-*`, `ENTITY-*`, `VERSION-*`, `DEC-LOSSLESS-01` exist nowhere locally). The basis used is the repo-local `docs/contracts/tagging_contract_v0.md` (as `canonical_post_ms_source_state.md:139-150` already directs). Any mismatch with the external text is **OPEN**.
Companion: `simplification_log.md`.

## 1. PROJECT DECISIONS carried (from the task statement)

- Distinct downstream outcomes are preserved: DIS generated/placed -> fast geometric filter -> replay + reconstruction (candidates kept as data) -> selection/veto (not implemented).
- `Y = B` stays the conceptual final endpoint and is **not evaluated** by anything here. Frozen SBT `Y^(k)` stays a separate historical/intermediate endpoint; no stage here is named or treated as `Y`.
- Goal: more useful, physically valid candidates per unit simulation cost; not a physical rate. Martina's dataset is a feasibility reference only.

## 2. What was built (smallest thing; no new schema)

| File | Role | Lines |
|---|---|---|
| `src/ship_muon_bg/adapters/pr1432_mudis.py` | prepare -> filter -> replay -> ShipReco for one source state; ROOT readers return plain dicts; pure `build_bundle(request, facts)` produces the existing records | ~340 |
| `src/ship_muon_bg/simulation/evaluation.py` | `bundle_records(bundle)`: lossless JSON dump of all five record kinds (the old manifest dropped decisions and candidates) | +27 |
| `scripts/run_pr1432_stage_smoke.py` | bounded driver: `ROW:SEED` list -> `bundle.json` per execution + `aggregates.csv` through the existing `aggregate_state_stage_evaluations` | ~75 |
| `tests/fairship/test_pr1432_mudis.py` | 16 pure tests + 3 `local_env` tests against the original ROOT files | ~190 |
| 3 one-line scope banners, 2 docs | see `simplification_log.md` S5-S6 | 6 |

Reused unchanged: `FSSimExecution`, `InteractionRealization`, `ReconstructedCandidate`, `ObservationEnvelope`, `StageDecision`, `EvaluationRequest/Bundle`, `verify_evaluation_bundle`, `aggregate_state_stage_evaluations`.

### Status vocabulary (mapping onto existing records)

| Requested | Record |
|---|---|
| PASS | `StageDecision` EVALUATED, `decision=True` |
| PHYSICS_REJECTED | `StageDecision` EVALUATED, `decision=False` |
| NOT_EVALUATED | `StageDecision` NOT_EVALUATED (never applied, or stopped by `--last-step`) |
| TECHNICAL_FAILURE | `FSSimExecution` TECHNICAL_FAILURE (only if `prepare` fails) or `StageDecision` TECHNICALLY_UNAVAILABLE (stage evidence untrustworthy) |

### Stages (execution level; each id content-addressed over FairShip commit, `n_dis_per_material`, filter option)

| Stage | PASS | PHYSICS_REJECTED | NOT_EVALUATED | TECHNICALLY_UNAVAILABLE |
|---|---|---|---|---|
| `pr1432_dis_generation_v0` | >= 1 DIS generated | muon processed, crossed no DIS target path | prepared tree empty **and** muon `p < 2 GeV` (generator support cut, `MuDISProcessor.cxx:437`; not physics) | prepared tree empty with `p >= 2`; vertex list != `nDISevt` |
| `pr1432_filter_any_v0` | >= 1 DIS kept by filter option 1 | generated >= 1, kept 0 | nothing generated | filter crashed; kept->generated vertex join not unique/ordered |
| `pr1432_replay_complete_v0` | every kept DIS replayed, `(material, disIndex)` set equals kept set, `muEntry==0` | — | nothing kept, or `--last-step filter` | replay crashed or lineage mismatch |
| `pr1432_reco_candidate_any_v0` | >= 1 `Particles` entry | ShipReco ran on all replayed events, 0 `Particles` | nothing replayed/kept, or not requested | reco crashed, or reco events != replay events |

A stage downstream of a TECHNICALLY_UNAVAILABLE stage is itself TECHNICALLY_UNAVAILABLE, not NOT_EVALUATED. Per-realization filter decisions (`pr1432_filter_dis_v0`, one per generated DIS, citing that DIS's vertex-z observation) are stored but not rolled up: the aggregation refuses realization-scoped decisions by design (`aggregation.py:558-591`). Selection/veto: **not implemented**; silence rolls up as NOT_EVALUATED.

Recorded per execution: source (`subject_id`), `execution_id` (hash of subject, configuration, options digest, seed), seed (Pythia6 and Geant4), FairShip commit, options digest, `n_dis_per_material` requested vs generated vs kept per material, path length, native `wDIS` (observation named `..._not_event_weight`), replayed events, fit tracks per realization, candidates with `candidate_index`, wall seconds per step and total, return codes and commands (`facts.json`), SHA-256 of inputs/outputs.

Candidate = one entry of ShipReco's `Particles` branch (reconstructed vertex). **PROVISIONAL**: no vertex-quality, selection or veto cut is applied; it is not an `HNL`/background claim and not `B`.

## 3. Scientific state

**VERIFIED**
- The existing records carry the whole outcome chain, lineage and the three kinds of "no answer" (tests; real runs in §4).
- PR #1432 `prepareEvents.py --seed S` is reproducible: for 25740/32029/17005 the prepared and filtered `MuonDIS` trees match the 2026-10-12 originals vertex-for-vertex (independent ROOT check, not through the adapter); same seed twice gives identical prepared, kept and fit-track results; a new seed changes them.
- Replay/reco event counts and candidate multiplicities equal the ROOT files (§4).
- A source-level "kept >= 1" depends on the trial budget and on the seed: state 25740 with 60 generated gave 7 (seed 1025740, run twice with identical results), 3 (seed 7000001) and 2 (seed 225740) kept: four executions, three distinct seeds.

**PROVISIONAL**
- The partial-failure policy (`simplification_log.md` S4); `Particles` as the candidate definition; the kept->generated join by exact vertex equality.
- Provisional coordinates: all injected states use the legacy `+2.5916 m` transform (`canonical_post_ms_source_state.md`).

**OPEN** (see §6 for the closing evidence)
- Geometry content identity is **not** part of the pooling key: the key is configuration id + options digest + stage id (commit, budget, filter). The geometry file's SHA-256 is stored in each execution's provenance (`geofile_sha256`), but `geo_connector.root` bytes differ per injection (three states, three hashes), so it cannot be a key; a TGeo-content hash is needed.
- B1 `Sigma_postMS` capture; B5 whether post-shield states are valid input for MS DIS (all runs here are injected at z ~ 31.5 m; MS DIS comes only from the 20 cm UBT-backward slice); B4 configuration pin (`f73a305` in `canonical_post_ms_source_state.md:178` vs the proposed `8df35bc` + PR #1432 `400c0fb`).
- `Y^(k)` vs the PR #1432 filter outcome (§4).
- The weight/exposure semantics (`wDIS` has no cross section and carries `w=1`; `q_i` has no field). Nothing here applies a weight.

## 4. Real FairShip evidence

Configuration: FairShip PR #1432 worktree `400c0fb4ae31` (clean; `refs/pull/1432/head` unchanged on the remote, checked read-only), pixi env of main `8df35bc`, `--shieldName TRY_2026 --strawDesign 10 --noSND`, `-d 10` DIS per material, filter option 1. Remote main has moved to `8a72b39` (not tested). Inputs: injected `sim_connector.root` (TTree connector, `w=1`). All outputs in `scratch/stage-outcomes-v0/`. No FairShip file was modified.

| Run | Result | Checked against |
|---|---|---|
| `smoke_a` 25740/32029/17005, seeds `10<row>` | generated 60/60/30, kept 7/2/0, replay 7/2/-, reco 7/2/-, 0 candidates | 2026-10-12 originals: vertices identical, kept and replay counts identical |
| `smoke_b` 25740 same seed again | identical kept (7) and fit tracks | `smoke_a` |
| `smoke_b` 25740 seed 7000001 | kept 3; one source row, 2 executions | aggregate: `execution_count=2, valid=2` |
| `smoke_b` row 20 (`p=1.54 GeV`) | prepared tree empty -> all four stages NOT_EVALUATED | `MuDISProcessor.cxx:437` |
| `smoke_b` bad geometry file | `prepare` rc=1 -> execution TECHNICAL_FAILURE, no realizations/decisions; aggregate `technical_failure=1`, `eta_hat=None` | `prepare.log` ("Failed to load geometry") |
| `batch_full` 31238/29823/8217 (full chain) | kept 0/3/25, **candidates 0/1/12** (12 in 9 of 25 replayed DIS) | `ship_reco_sim.Particles` per event read independently from the ROOT file |
| `batch_filter` 22 states, filter only (`--last-step filter`) | see below | bundles and `label_table.csv` |

Step costs (this machine): prepare 5-7 s, filter 9-11 s, replay 33-71 s (3 to 25 DIS), reco 12-31 s; TTree injection ~47 s per state. A filter-only evaluation costs ~17 s per state.

**Filter-stage labels vs pre-simulation features** (22 states: all 9 U0=1 rows with p > 3 GeV from `u0_main.npz`, 12 random U0=0 rows with p > 3 GeV (seed 20261013) and 17005; not a representative sample):

| U0 | n | filter PASS | generated DIS per state |
|---|---|---|---|
| 1 | 9 | 8 | 50-80 |
| 0 | 13 | 0 | 10-30 |

- **VERIFIED** (this sample): U0 (straight ray crosses an SBT volume) separates the filter outcome, as it separated `Y^(k)` in the 2026-10-11 pilot. Part of the separation is geometric: U0=0 muons cross fewer DIS target materials, so fewer DIS are generated at a fixed 10 per material.
- 0/13 in the U0=0 stratum leaves a one-sided 95 % upper bound of ~21 % on its pass rate, so no hard exclusion is justified; the exploration floor stays.
- `Y^(k)` vs filter: known for 10 of these states (9 U0=1 rows, `Y^(k)=1` in the main-env recheck; 17005 `Y^(k)=0`). Filter PASS in 8 of the 9 `Y^(k)=1` rows. **35078 has `Y^(k)=1` (5 SBT hits) but 0/70 DIS kept** -> the two are not equivalent at this budget (single seed). Whether `Y^(k)` is necessary for a filter PASS is **not** shown: no `Y^(k)=0` state passed, but only 17005 is known.
- Single seed per state; stochastic variance is large (above), so these are observed frequencies, not rates.

## 5. U(s): what is supportable now (deliverable C)

- **Keep the deterministic geometric `U0` + `h(U0) = (delta + (1-delta) U0)^alpha`, `delta=0.1, alpha=4`** (`docs/1011_U0_DirectedPilot/u0.py`). It is computable from the source state alone (ray vs geometry), uses no downstream outcome, and tolerates zero positives.
- **Target to record: the `pr1432_filter_any_v0` stage** (fast geometric filter, option 1). `U0` was designed for `Y^(k)`; it also tracks the filter outcome here (§4). It is **not** a predictor of replay/reco candidates or of `B`: replay + reconstruction ran on only 4 states (25740, 32029, 29823, 8217).
- **Do not carry over the hard `p0 <= 3 GeV` exclusion.** It is a necessary condition for `Y^(k)` only (energy conservation). For the DIS target the only hard support limit is the generator's own `p < 2 GeV`, which is recorded as NOT_EVALUATED, not as a rejection.
- **No learned model.** There are 22 filter labels (8 positive), selection-biased, one seed each. Training would only exercise infrastructure. Retain the heuristic.
- **Iterative update is supportable**: `batch_filter/aggregates.csv` (stamped with stage-definition id, configuration id, options digest and `n_dis_per_material`) + `label_table.csv` give per-state, per-stage counts that a stratum-wise empirical-rate update can consume (e.g. update `delta` from the U0=0 stratum once it has enough states).
- **Limits**: `U0` is a geometric ray with no field or scattering; its agreement with the endpoints is partly circular; coordinates are PROVISIONAL.

## 6. Remaining blockers (next operational milestone only)

| Blocker | Closing evidence |
|---|---|
| B1/B2/B3 canonical `Sigma_postMS` capture, schema, exact round trip | A geometry-derived plane in a pinned geometry (TGeo query, not a guessed z), a capture producer in the repo (the scratch prototype `POSTMS_GEOMETRY_DERIVED_Z_CM 3194.0` has no preserved source), and a round trip with `roundtrip_validation_plan_id` frozen before inspecting acceptance data |
| B4 configuration pin | A decision between `f73a305` and main + PR #1432 at a recorded SHA; PR #1432 merged or the pin kept at `400c0fb`; the TRY_2026 vs "24.09" geometry equality check |
| B5 MS-DIS input region | Either states captured upstream of the last shield interaction length (PR README) or a physics argument/test that post-shield states reproduce upstream MS DIS; until then the MS part of any "DIS" count is not claimed |
| Weight/exposure | Not needed for the loop; `q_i` field and `w`-propagation rule are needed only if a rate or an unbiased comparison is claimed |
| Selection/veto | A declared stage (cuts, version) and a candidate-scoped decision, with a rollup rule for realization-scoped decisions |

## 7. Next vertical slice (deliverable G)

One iteration of the loop on the filter stage, ~1 h: draw N states from `h(U0)` (`u0.py draws`) **and** N from the unbiased pool using one fixed seed policy, inject on the PR worktree (47 s/state), run `run_pr1432_stage_smoke.py --last-step filter`, and compare "filter-PASS per second of simulation" between arms; then update `delta` from the U0=0 stratum counts. This closes `source -> generate -> DIS -> evaluate -> update` with no new framework and no model.

## 8. Commands and tests (deliverable E)

```
# unit + integration (core venv)
cd worktrees/ship-stage-outcomes-v0 && ../../ship_adaptive_muon_bg/.venv/bin/python -m pytest -q -p no:cacheprovider --ignore=tests/afterms
# real-artifact tests (ROOT env)
env -i HOME=$HOME PATH=/usr/bin:/bin bash -c "source scratch/fairship-update-1012/env_pr1432.sh && export PYTHONPATH=$PWD/src && python -m pytest -q -p no:cacheprovider tests/fairship/test_pr1432_mudis.py"
# smoke
python scripts/run_pr1432_stage_smoke.py --fairship worktrees/FairShip-pr1432-400c0fb --inputs <runs dir> --out <dir> --runs ROW:SEED,... [--last-step filter|replay|reco]
```

Results (after the review fixes): core suite `658 passed, 30 skipped` (`--ignore=tests/afterms`); new file 23 passed + 3 `local_env` skipped in the core venv, 26 passed in the ROOT env (readers agree with the original ROOT files). Regression smoke `smoke_d` (25740, row 20, bad geometry) reproduces `smoke_a`/`smoke_b`. `smoke_c` is kept as evidence of an environment mistake (PYTHONPATH overwritten -> `ShipGeoConfig` missing -> prepare rc=1): the adapter recorded TECHNICAL_FAILURE, not a physics result. Pre-existing: 13 `tests/afterms` collection errors without sklearn/scipy (unchanged, listed in `simplification_log.md` S5).

## 9. Review (one adversarial Codex pass, `codex exec -s read-only`, 2026-10-13)

First pass: FAIL with 6 BLOCKING + 3 MINOR findings (full text `scratch/stage-outcomes-v0/review_codex.md`). All were in scope and fixed once, each with a test: filter exiting 0 after skipping a malformed entry now needs a log line attesting `skipped 0`; a stage is concluded only from a step recorded with `rc==0`; `GetEntry` results are checked; reader exceptions become TECHNICALLY_UNAVAILABLE with upstream kept; the recorded seed must equal the request's; execution ids are hashed (no delimiter collisions); facts survive a JSON round trip; the CSV carries stage-definition id, configuration id, options digest and budget. Not fixed, recorded as OPEN: geometry content is not in the pooling key (§3). A second review pass was not run.
