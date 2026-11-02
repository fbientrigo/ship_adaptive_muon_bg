# P2 states and their stage-1 origin (task 2, 2026-11-02)

Code: `adapters/fairship_p2_source.py`; tests: `tests/fairship/test_fairship_p2_source.py` (8 synthetic, no ROOT; 3 `local_env`, need `P2_ARTIFACTS`).
Data: FairShip `8a1cc53ec`, stage-1 seeds 201-203 -> `stage2_prod_8a1cc53/`. A P2 crossing is a *candidate* `SourceState`, carried by the existing
`TagSubject` (`subject_type = p2_crossing_state`, content-addressed `state_definition_id`). `tagging_contract_v0.md` SUBJ-02/OPEN-01 forbid code from
asserting it is THE thesis SourceState; that promotion stays a project decision. No new dataset schema: the row dict is the provenance sidecar keyed by `subject_id`.

## Hierarchy (what is NOT the same thing)

run (stage-1 file) > progenitor (stage-1 entry = one POT event) > split group (the pi/K whose decays were cloned, or the unsplit muon) > trajectory (one stage-1 muon = one clone) > state (one P2 crossing).
In this sample: 3 runs, 8 progenitors, 8 split groups (6 split parents + 2 unsplit muons), 21 trajectories, 21 states. No effective sample size is computed.

## VERIFIED

- Where the split flags go. Stage 1 sets them in `muonShieldOptimization/exitHadronAbsorber.cxx:770-771` (`stack->SetSplitSet`) and `shipdata/ShipStack.cxx:239-241` copies them into the stored `ShipMCTrack`.
  Stage 2 replays the stage-1 `MCTrack` as history with `cpg->AddTrack(pdg, p, start, mother, ..., weight, proc)` (`shipgen/MuonBackGenerator.cxx:336-345`, `368-410`): weight and process are passed, no split argument exists, and no code path of the stage-2 run calls `SetSplitSet`
  (only `exitHadronAbsorber.cxx` and tests do). Observed: every stage-2 `MCTrack` of seed 201 has `splitSet = -1`, role 0 (`test_stage2_output_has_lost_the_split_flags...`); stage 1 has 10 clones with set 0, role 2.
  Static reading plus observed output; the runtime state of `fSplitSets` in stage 2 was not inspected.
- Join by content: the injected primary of stage 2 (`procID 43`, mother chain from the crossing track) equals a stage-1 `PlaneHAPoint` momentum (< 2e-3 GeV). 21/21 resolve to exactly one stage-1 point; mutating the tolerance to 5 GeV makes 5 synthetic tests fail.
- `EventID - 1 == stage-1 entry` only for `-i 0`. Under `-i 8` the stage-2 `EventID` restarts at 1 (stock run: EventIDs 1,2,6,7 = stage-1 entries 8,9,13,14; unfiltered run: 1..7). The content join gives entries 8,9,13,14 in both; the rule needs `first_event` to agree.
- Weight of the stage-2 crossing track equals the stage-1 muon weight in 21/21 rows.
- `Detector/ScoringPlane.h` `ProcessHits` scores each track on every entry into the plane (`IsTrackEntering`), so a track can have several points. In 95 crossings of 5 datasets (stage2_prod, stock, unfiltered, two `-i 8`) all are `FIRST_FORWARD`: no repeated or backward crossing was observed. The policy below is therefore tested only on synthetic input.
- Material: at the 20 `PlaneMS` hit positions, and 5 and 20 cm upstream of them, `FindNode` returns `cave_1` / air. Being inside the z-envelope of `Magn7` (2818-3218) did not mean being in iron for these muons (x of -113 to -222, 179 and 188 cm, outside the `Magn7` half-width 171 cm). This is a point check, not the material along the trajectory.

## PROJECT DECISION (proposed)

- Repeated/backward policy: a crossing is eligible (a `TagSubject`) iff `pz > 0` and it is the first such crossing of its track in the entry. Others stay as rows with class `REPEAT_FORWARD` / `BACKWARD`; nothing is dropped.
- Non-recoverable stays non-recoverable: invalid `track_id` (< 0), no injected primary in the chain -> `UNRECOVERABLE`; zero or several stage-1 matches -> `AMBIGUOUS`. Fields stay `None`.
- Report counts and sums of the stored weight per level, labelled `WEIGHT_SEMANTICS`; never a Kish ESS.

## PROVISIONAL

- The earlier "five groups, six sums" was an error of mine: there are six split parents (sums 1.0, 0.2, 0.4, 0.1, 0.1, 0.1, in the order of the table) plus two unsplit muons. Sum of stored weights = 3.9 is arithmetic, not a physical sample size.
- `procID 43` as the marker of injected primaries is observed on 8a1cc53 only (`K_INJECTED`).
- 8 progenitors are distinct POT events (ids 5046, 5110, 6073, 6172, 17791, 7494, 18303, 14766); independence between them within one seed is assumed from distinct events, not tested.

## OPEN

- Physical meaning of the stored weight: 0.1 = 1/K of a decay alternative, 1 for unsplit; whether a split group's weights sum to a probability, and what the survivor line (not stored in these files) contributes.
- `--MuonBack` event filter excludes P2 from its keep rule (`run_simScript.py:1226`). 7 events with P2 crossings, all also with SBT hits, do not prove P2-only events are never lost. `stage2_nofilter.py` is a diagnostic tool, not an approved production interface.
- Whether the lost split flags are a FairShip omission or intended: needs the FairShip authors; a human would post any issue.
- Trajectory material: no step-by-step material along the path was computed.
- Ambiguous provenance cases in this sample: none (21 RESOLVED). Two unsplit muons have unusual stage-1 ancestry (see rows 1 and 20) but a unique origin.

## The 21 P2 states

| # | seed | st2 entry (EventID) | st2 trk | st1 entry (POT evt) | st1 mu trk | ancestry in stage 1 (pdg, newest first) | split set/role | clone w | x,y (cm) | p (GeV) | class/status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 201 | 0 (9) | 4 | 8 (5046) | 4 | 13 ← 2212 ← 2212 | - | 1 | 188, -12 | 10.43 | FIRST_FORWARD/RESOLVED |
| 2 | 201 | 1 (10) | 35 | 9 (5110) | 25 | -13 ← 211 ← 3112 ← 3214 ← 2 ← 2 ← 2112 | 0/2 (W=1) | 0.1 | -189, 5 | 25.56 | FIRST_FORWARD/RESOLVED |
| 3 | 201 | 1 (10) | 34 | 9 (5110) | 24 | -13 ← 211 ← 3112 ← 3214 ← 2 ← 2 ← 2112 | 0/2 (W=1) | 0.1 | -208, 1 | 35.56 | FIRST_FORWARD/RESOLVED |
| 4 | 201 | 1 (10) | 33 | 9 (5110) | 23 | -13 ← 211 ← 3112 ← 3214 ← 2 ← 2 ← 2112 | 0/2 (W=1) | 0.1 | -209, -16 | 46.32 | FIRST_FORWARD/RESOLVED |
| 5 | 201 | 1 (10) | 32 | 9 (5110) | 22 | -13 ← 211 ← 3112 ← 3214 ← 2 ← 2 ← 2112 | 0/2 (W=1) | 0.1 | -222, -6 | 37.15 | FIRST_FORWARD/RESOLVED |
| 6 | 201 | 1 (10) | 31 | 9 (5110) | 21 | -13 ← 211 ← 3112 ← 3214 ← 2 ← 2 ← 2112 | 0/2 (W=1) | 0.1 | -214, -17 | 33.73 | FIRST_FORWARD/RESOLVED |
| 7 | 201 | 1 (10) | 30 | 9 (5110) | 20 | -13 ← 211 ← 3112 ← 3214 ← 2 ← 2 ← 2112 | 0/2 (W=1) | 0.1 | -196, -16 | 40.19 | FIRST_FORWARD/RESOLVED |
| 8 | 201 | 1 (10) | 29 | 9 (5110) | 19 | -13 ← 211 ← 3112 ← 3214 ← 2 ← 2 ← 2112 | 0/2 (W=1) | 0.1 | -178, 18 | 20.83 | FIRST_FORWARD/RESOLVED |
| 9 | 201 | 1 (10) | 28 | 9 (5110) | 18 | -13 ← 211 ← 3112 ← 3214 ← 2 ← 2 ← 2112 | 0/2 (W=1) | 0.1 | -199, -9 | 35.58 | FIRST_FORWARD/RESOLVED |
| 10 | 201 | 1 (10) | 27 | 9 (5110) | 17 | -13 ← 211 ← 3112 ← 3214 ← 2 ← 2 ← 2112 | 0/2 (W=1) | 0.1 | -202, 2 | 37.98 | FIRST_FORWARD/RESOLVED |
| 11 | 201 | 1 (10) | 26 | 9 (5110) | 16 | -13 ← 211 ← 3112 ← 3214 ← 2 ← 2 ← 2112 | 0/2 (W=1) | 0.1 | -186, -28 | 10.72 | FIRST_FORWARD/RESOLVED |
| 12 | 201 | 2 (14) | 8 | 13 (6073) | 18 | -13 ← 211 ← 2212 ← 2212 | 0/2 (W=1) | 0.1 | -179, 37 | 5.45 | FIRST_FORWARD/RESOLVED |
| 13 | 201 | 2 (14) | 7 | 13 (6073) | 16 | -13 ← 211 ← 2212 ← 2212 | 0/2 (W=1) | 0.1 | -219, -33 | 0.62 | FIRST_FORWARD/RESOLVED |
| 14 | 201 | 3 (15) | 15 | 14 (6172) | 13 | -13 ← 321 ← 1 ← 1 ← 2112 | 0/2 (W=1) | 0.1 | -216, 13 | 39.68 | FIRST_FORWARD/RESOLVED |
| 15 | 201 | 3 (15) | 14 | 14 (6172) | 12 | -13 ← 321 ← 1 ← 1 ← 2112 | 0/2 (W=1) | 0.1 | -195, -26 | 5.40 | FIRST_FORWARD/RESOLVED |
| 16 | 201 | 3 (15) | 13 | 14 (6172) | 11 | -13 ← 321 ← 1 ← 1 ← 2112 | 0/2 (W=1) | 0.1 | -213, 24 | 32.33 | FIRST_FORWARD/RESOLVED |
| 17 | 201 | 3 (15) | 12 | 14 (6172) | 10 | -13 ← 321 ← 1 ← 1 ← 2112 | 0/2 (W=1) | 0.1 | -197, -1 | 24.03 | FIRST_FORWARD/RESOLVED |
| 18 | 201 | 4 (30) | 9 | 29 (17791) | 26 | -13 ← 211 ← 2 ← 2 ← 2 ← 2 ← 2 ← 2212 | 0/2 (W=1) | 0.1 | -113, -190 | 2.45 | FIRST_FORWARD/RESOLVED |
| 19 | 202 | 0 (16) | 7 | 15 (7494) | 11 | 13 ← -321 ← -313 ← 1 ← 1 ← 2112 | 0/2 (W=1) | 0.1 | 179, -41 | 6.44 | FIRST_FORWARD/RESOLVED |
| 20 | 202 | 1 (36) | 9 | 35 (18303) | 8 | -13 ← 22 ← 111 ← 2114 ← 1 ← 1 ← 9902210 ← 2212 | - | 1 | -191, -2 | 56.44 | FIRST_FORWARD/RESOLVED |
| 21 | 203 | 0 (26) | 7 | 25 (14766) | 18 | -13 ← 211 ← 2224 ← 1 ← 9902210 ← 2212 | 0/2 (W=1) | 0.1 | -182, -40 | 7.83 | FIRST_FORWARD/RESOLVED |
