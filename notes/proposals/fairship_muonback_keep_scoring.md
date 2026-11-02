# Proposal (not sent): `--MuonBackKeepScoringPlanes` for `run_simScript.py`

Status: a text for a human to review and, if they agree, to post. No agent posts to ShipSoft/FairShip (AGENT_POLICY §3). Our frozen checkout `worktrees/FairShip-main-8a1cc53` was not edited; the patch was made on a copy (`scratch/gen-from-scratch-v0/fs_proposal/`) and run from there.

## Problem (observed, FairShip 26.10, `8a1cc53ec`)

`run_simScript.py:1226` drops `UpstreamTaggerScoringPoint` and `CaloScoringPlanePoint` from the keep rule of `--MuonBack` ("scoring planes must not make an event non-empty"; `CHANGELOG.md:39` says the same). A muon that crosses P2 (detID 2, `ShieldGap` after the shield) and hits nothing else is removed, and the standard output cannot tell the user. Whether that happens depends on a side effect: with `--FollowMuon` the cavern is sensitive (`vetoPoint`, volume `Cavern_1`), so nearly every muon leaves a non-scoring point and survives.

Measured (our runs, `notes/p2_lossless_v0.md`): real stage-1 input without `--FollowMuon`: 19 P2 crossings generated, 9 kept, 10 lost (6 of the 7 events with a crossing vanish: EventIDs 9, 14, 15 of seed 201, 16 and 36 of seed 202, 26 of seed 203); with `--FollowMuon`: 19 of 19 kept.

## Change

Add `--MuonBackKeepScoringPlanes` (default off, so no behaviour change): when given, the two scoring branches stay in `pointContainers`, so an event whose only hits are scoring points is kept. 9 lines (`notes/proposals/fairship_muonback_keep_scoring.patch`).

## Evidence for the change

Same synthetic input as the loss test (12 events, 117 P2 crossings, no `--FollowMuon`): stock keeps 19, the option keeps 117 = the unfiltered count, identical `(EventID, track, crossing)` keys.

## Not shown / risks

Plane 1 (all particles, detID 1) is also a scoring point: with the option almost every event is kept, so output size grows; a per-detID rule (keep only P2) would need more code. Only run on the 8a1cc53 copy. The option name is a suggestion.
