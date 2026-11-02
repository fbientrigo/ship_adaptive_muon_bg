# P2 extraction v0: evidence and decisions (2026-11-02)

FairShip `8a1cc53ec` (release 26.10, read-only worktree `worktrees/FairShip-main-8a1cc53`). Data: three `--MuonBack` stage-2 runs
(seeds 201-203, 114 input events) under `scratch/gen-from-scratch-v0/`: `stage2_prod_8a1cc53/` (with local plane `PlaneMS`),
`p2_stock_8a1cc53/` (stock script), `p2_nofilter_8a1cc53/` (empty-event filter disabled by `stage2_nofilter.py`, raw output).
Reproduce the numbers with `p2_equivalence.py`, `clone_sets.py`, `p2_filter_audit.py`, `map_check.py` (all read-only; JSON in `p2_evidence.json`).

## VERIFIED (observed in code or ROOT)

| Fact | Evidence |
|---|---|
| P2 = `UpstreamTaggerScoringPlane2`, point branch `UpstreamTaggerScoringPoint`, `detID` = node copy number (P1=1, P2=2) | `Detector/ScoringPlane.h:41,133`; `python/shipDet_conf.py:707-722`; geofile nodes `UpstreamTaggerScoringPlane1_1`, `..Plane2_2` |
| P2 is on in the executed config; branch exists in all 3 sim files | `run.log` "UpstreamTaggerScoringPlane2: 913.2 x 856.4 x 0.1 cm3 centred at (0, 91.8, 3228) cm, PDG-filtered"; branch list |
| P2 geometry (geofile `geo_s201.root`): center (0, 91.8, 3228.0) cm, half (456.6, 428.2, 0.05), vacuum. Shield end from the magnet blocks = 3218.0 cm (`Magn7` 3018 +/- 200) = `muShield.z + length` (214 + 3004); P2 centre - shield end = 10.0 cm | `python/geometry_config.py:112-121`; `test_p2_geometry_is_read_from_the_produced_geofile` |
| P1: center (0,0,3326.95), half (220, 336.4, 0.05), all particles, tracks not kept | same file `:96-108`; geofile |
| `PlaneMS` (my local plane): center z = 3200.0, half (355.9, 169.9, 0.1). Its z lies inside the z-span of the last magnet block (2818-3218), 28 cm upstream of P2 (this envelope does not prove the muons crossed iron; that needs the material along the trajectory); laterally smaller than P2 | geofile node `exitHadronAbsorberPlaneMS_box_1` |
| Every one of the 20 `PlaneMS` crossings has a P2 partner (same event, track): `|dp| < 0.5 MeV`, `dz = 28.05 cm`, `dx` in [-14.4, +3.2] cm. One extra P2 crossing (seed 201 entry 4, p = 2.45 GeV, y = -190 cm) lies outside PlaneMS (`|y| > 169.9`) | `test_p2_contains_every_planems_crossing...` |
| P2 stores only `|pdg| = 13`; track ID valid (`>= 0`, kept in `MCTrack`) for all 21 crossings; `(entry, track_id)` is unique (no repeated crossing in this sample) | `ScoringPlane.h:82,106`; `read_crossings` |
| In the raw (unfiltered) output 7 of 114 entries have a P2 crossing and all 7 pass the keep rule; every one also has `vetoPoint` hits equal to the number of P2 crossings. The rule excludes `UpstreamTaggerScoringPoint` and `CaloScoringPlanePoint` | `macro/run_simScript.py:1226`; `test_muonback_filter_drops_no_p2_event...` |
| `MCEventHeader.EventID == entry + 1` of the stage-1 file; checked by matching the stage-2 primary muon momentum with stage-1 `PlaneHAPoint` momentum: 8/8 events, 21 primaries, unique | `map_check.py`; `test_stage2_event_id_minus_one...` |
| Stage-1 and stage-2 `ShipGeo` agree on `FairShip` sha, library `26.10`, `shieldName`, `muShield`, `hadronAbsorber`, `target`, `cave`. They differ in `SND` (true vs false), `UpstreamTagger`, `UpstreamTagger_geo`, `strawtubes_geo`; the three stage-2 `ShipGeo` hashes are equal | `p2_evidence.json` |
| Split flags of the stage-1 clones (`splitSet 0`, role 2 = decay, weight 0.1 = 1/K) are NOT present in the stage-2 `MCTrack` (all `-1/0`), although the history tracks are copied. `split_provenance.pair_weight` on stage-2 output therefore returns 0.1 (compatible) for 10 mutually exclusive alternatives; on stage-1 it returns 0 | dump of seed 201 entry 9 vs stage-2 entry 1; `shipdata/ShipMCTrack.cxx:33-35,59-61` (defaults) |
| The 21 P2 crossings come from 8 stage-1 events and 8 parent particles: 6 are decay clones of one pi/K each (sum of weights per parent 1.0, 0.2, 0.4, 0.1, 0.1, 0.1), 2 are unsplit weight-1 muons (parents p and gamma). Sum of crossing weights = 3.9 | `clone_sets.py` |

## PROJECT DECISION (proposed; nothing is merged)

- P2 is the physical post-shield extraction surface. `PlaneMS` stays a diagnostic until a larger sample repeats the pairing.
- A crossing is a row, not a sample. Independence is the stage-1 parent (split set) of the crossing, not the crossing.
- Physical weight stays the stored `MCTrack` weight, separate from any utility; identifiers are `(source file sha, stage-1 entry, stage-1 track)`, never the stage-2 entry.
- Reader/manifest live in `adapters/fairship_p2.py`; no new dataset system. Candidate rows map to existing `ReconstructedCandidate`/`StageDecision` via a later adapter.

## PROVISIONAL

- "Filter drops nothing relevant": 7 P2 events, 19 crossings, all with SBT hits. Muons that cross P2 outside the decay-vessel/SBT acceptance (P2 reaches x = +/-456 cm, y up to 520 cm) are not sampled. The raw run (`stage2_nofilter.py`) costs nothing extra here (32 kB per seed); use it for extraction campaigns until a larger sample settles this.
- The stock script and the `PlaneMS` run are not bitwise repeatable (the extra volume changes the random stream): P2 has 19 crossings in the raw run, 19 in the stock run and 21 in the `PlaneMS` run.
- Stage-1 `SND = true` vs stage-2 `SND = false`: irrelevant upstream of P2 by construction of the stage-1 score plane, not checked by geometry.

## OPEN

- Which shield design and z produced Martina's `afterMS` sample. This geofile gives the shield end at 32.18 m, not the 31.93 m used in the earlier +2.59/+3.04 m offset comparison; that comparison must be redone from the manifest.
- Whether the lost split flags in `--MuonBack` output are intended (cause inferred from default constructors, not traced to `MuonBackGenerator`/`ShipStack`). Candidate FairShip issue; a human posts it, not an agent.
- Whether `PlaneMS` overlaps magnet volumes at z = 3200 (no overlap check run).
- Behaviour on lxplus and with larger samples (`--kaon-pion-splits`, many seeds).
