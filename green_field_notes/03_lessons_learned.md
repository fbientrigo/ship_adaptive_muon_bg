# Lessons learned (do not lose these)

Each item: what we learned, evidence, and what the greenfield does about it. VERIFIED means measured in this work; evidence files are in `notes/` and `docs/` of the old repo and in scratch (`~/thesis/scratch/gen-from-scratch-v0/`).

## Geometry and the z offset
- VERIFIED: P2 (`UpstreamTaggerScoringPlane2`, detID 2) centre is at z=3228.0 cm = shield end (3218.0 = `Magn7` 3018+200) + 10.0 cm, read from the produced `geo_*.root` (`notes/p2_extraction_v0.md`). Read the placement from the geofile, never type it.
- The historical offsets (`+2.5916 m`, `+70.845 m`, `-68.5 m`) belong to Martina's pkl and old code paths; `docs/decisions/canonical_post_ms_source_state.md` already says not to combine them. Greenfield: no free `z_offset` parameter; a state carries the plane id and geometry hash it was taken at.
- OPEN: an earlier "+2.59/+3.04 m" comparison used 31.93 m as the shield end while the geofile says 32.18 m; redo from the manifest. Which shield design/z Martina's `afterMS` sample used is also OPEN. Not needed if we stop depending on that sample.
- `PlaneMS` (z=3200, inside `Magn7`'s z-span, air at its hits) was a diagnostic plane; every one of its 20 hits pairs with P2. Extraction must not depend on it.

## FairShip `--MuonBack` stage 2
- VERIFIED: `run_simScript.py` (26.10, `:1200-1275`) keeps only events with a non-scoring `*Point` hit, then `rm` + `mv`. A muon that only crosses P2 disappears from the standard output. Without `--FollowMuon`, 10/19 real crossings (6/7 events) and 98/117 synthetic ones were lost; with it none (the sensitive `Cavern_1` rescues every event). (`notes/p2_lossless_v0.md`)
- Lesson: never use the standard output as the source of P2 states. Use the unfiltered tree (driver `run_stage2_p2.py`, sha256-pinned) or the upstream option once it exists (draft: `notes/proposals/fairship_muonback_keep_scoring.*`, not sent).
- A run with and without `--FollowMuon` differ event by event (random stream). Loss is only meaningful inside one run.
- Greenfield: generation and processing of muons sit at well-defined places in current FairShip; the driver/option is the single interface to the post-shield states.

## Identity and provenance
- VERIFIED: stage-2 output drops stage-1 split flags (`MuonBackGenerator` replays history without them). Provenance is recovered by content: the injected primary (procID 43) matches a stage-1 `PlaneHAPoint` momentum (tol 2e-3 GeV), 21/21 unique. `EventID - 1 == stage-1 entry` holds only for `-i 0`.
- Hierarchy to keep: run > progenitor (stage-1 entry) > split group > trajectory > state. 21 P2 crossings came from 8 progenitors / 8 split groups / 21 trajectories. Ten clones of one pion are one group, not ten independent sources.
- Greenfield: give stage-2 runs a source id at injection time instead of reconstructing it afterwards (needs an upstream or generator-side change; PROPOSAL).

## Weights and statistics
- Stored `MCTrack` weight is not an event weight in a settled sense; its physical normalisation is OPEN. We report counts and sums of stored weight per level and never an ESS.
- Physical weight stays separate from utility. `wDIS` is not a weight.
- A is judged by estimation quality, which is still the fundamental problem; do not let a good-looking loss stand in for it.

## Loop findings (from the thesis memory index)
- Filter label is seed-noisy; DIS budget is cheap; G0 rate is unbounded; U0 directed pilot was circular (Y^(k) 8/13 vs 0/30) so it proves nothing about U beyond the proxy.
- PR #1432 stage-outcome chain (prepare -> filter -> replay -> reco) ran on our states; outcomes: valid generated, passed fast filter, completed sim+reco, passed selection. Keep these four distinct. `Y = B` is a conceptual endpoint; the frozen SBT `Y^(k)` is separate.

## Engineering habits worth keeping
- Run unmodified FairShip scripts (`runpy`, pin by sha256, signature check) instead of rewriting their text; a changed script is a loud failure.
- One live smoke (~1 min, two FairShip processes) as pre-flight and as regression alarm for geometry, detID, filter, provenance.
- Facts-only readers returning plain dicts; ROOT imported lazily; keep the TFile in a variable (chained `Open().Get()` segfaults; `TKey::ReadObj` on TObjString segfaults in cppyy).
- Synthetic inputs are mechanism tests, never physical samples; say so in the file.
- Max 3 FairShip processes at once on this machine.
