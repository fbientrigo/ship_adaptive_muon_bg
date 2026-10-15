# Interaction-bundle SourceState architecture — decision record v0

Answers OPEN item 2 of `canonical_post_ms_source_state.md`: whether single-muon
`SourceState` is sufficient. Design only; no code in this repo implements it yet.

## Evidence

- FairShip (pinned `f73a305`) already has ΔT/pile-up logic:
  `macro/make_time_window.py` overlays separate proton interactions into
  Poisson-sized, 1 µs windows (`:48,306,311,322`). It runs offline on
  already-simulated `cbmsim` output and is **not wired into any pipeline**.
- On `tests/data/muonback_smoke.root` (200 entries, old format, not a physics
  sample): 38 entries have 2 muons; in all 38 they share the same weight and
  mother; plane-time spread is ≤0.078 ns, versus the 1 µs pile-up window. Within
  one interaction the open question is co-presence, not timing.

## Decision

- **Level 1 — SourceState = interaction bundle.** All Σ_postMS crossings from
  one MuonBack input entry, long format (`bundle_id, source_entry_key, track_id,
  mother_track_id, crossing_index, pdg, x,y,z, px,py,pz, t_at_plane`), with
  bundle-level `bundle_weight, input_sha256, input_entry_index, fairship_commit,
  config_hash, seed, capture_execution_id`. `source_entry_key=(input_sha256,
  input_entry_index)` is the statistical cluster; `bundle_id` is unique per
  capture (`hash(source_entry_key, capture_execution_id)`) so repeated captures
  of one entry never collide. Single muon = bundle of size 1; existing
  `afterms_5d_nf_v0` / `(N,8)` row consumers are untouched.
- **Level 2 — pile-up excluded from SourceState.** Modeled later as a
  post-transport overlay, per the `make_time_window.py` precedent. Safe to defer
  for the current `Y^(k)` (an OR over vetoPoints); not safe for time-based or
  reconstruction endpoints.

## Known issues (deferred, not solved here)

- Non-muon bundle members are not replayed (`replay_policy=muons_only`; FairShip
  only re-tracks muons by default).
- Pile-up composition (Level 2) unimplemented.
- Time fidelity: unverified whether Geant4 honours injected `tof`.
- Bundle-aware NF / set-valued utility function not designed.
- Multi-muon reduction policy (tagging contract OPEN-05) unresolved.
- Legacy `.pkl` has no grouping column (`afterms/split.py`
  `LIMITATION_NO_SOURCE_LINEAGE`).

## Not yet decided (future stages, each needs its own approval)

Reinjection mechanism (no FairShip generator currently reinjects at a plane
crossing rather than a production vertex), input-entry recovery from renumbered
output events, and the crossing/membership rule's exact predicate. See
`canonical_post_ms_source_state.md` OPEN item 2.
