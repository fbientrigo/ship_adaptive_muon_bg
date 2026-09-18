# Canonical post-Muon-Shield `SourceState` — design freeze v0

This is a design freeze, not a frozen contract like `docs/contracts/*`: it
records a project decision and its current evidentiary basis, and stays valid
only until #43 lands the actual schema/round-trip.

## PROJECT DECISION

- The canonical thesis `SourceState` will **not** depend on reconstructing the
  historical coordinate provenance of Martina's `muonsFullMC_afterMS.pkl`.
- Instead, the canonical `SourceState` is generated directly from a
  pinned/current FairShip geometry, with complete provenance:

  ```
  FairShip full event
      -> explicit post-Muon-Shield scoring surface Sigma_postMS
      -> canonical SourceState
      -> exact reinjection
      -> FairShip/GEANT4
      -> UBT / SBT / DIS / reconstruction / StageDecision
  ```

- UBT/SBT/DIS/reconstruction/`StageDecision` are downstream observations of
  that pipeline, not part of the source interface itself.
- The historical `muonsFullMC_afterMS.pkl` remains legacy / training-development
  evidence only.
- The provisional transform `z_out = z_declared + 2.5916 m`
  (`afterms_nominal_plane_to_current_shield_exit_plus_1cm_f73a305_v0`) is
  legacy/provisional and must **not** define the new canonical interface.
- The historical `+70.845 m` MuonBack correction and `-68.500 m` M&M importer
  correction are separate historical code paths — they are not pieces to
  combine into the new interface.
- **#43 replaces #27 as the P0 gate** before canonical U0 label acquisition:

  ```
  #43 (freeze Sigma_postMS, freeze SourceState schema, verify lineage, round-trip)
    -> #32 (acquire/train U0)
    -> adaptive loop
  ```

  #27 is **CLOSED as `not planned`**, not as physically verified — closing it
  removed a blocker, it did not supply an answer. The historical `.pkl`
  coordinate mapping itself **remains unresolved**; it is simply no longer a
  gate for the canonical thesis pipeline or canonical U0 label acquisition.

## VERIFIED

Only claims directly backed by code or artifacts already in this repository or
FairShip:

- GitHub: #27 is `CLOSED` / `NOT_PLANNED`; #43 is `OPEN` and is the P0 gate;
  #41 points from #27 to #43; #32 is `OPEN`, downstream of #43.
- `artifacts/fairship_connector_v0/summary.json` records its single
  `coordinate_transform` block as `"status": "PROVISIONAL"` /
  `"coordinate_physics_verified": false`; this is the connector-level
  transform status, not a claim independently checked per P0/P1/P3 run.
- `src/ship_muon_bg/adapters/fairship.py:273` returns
  `"coordinate_physics_verified": False` alongside `coordinate_transform_status`
  on every mechanical-injection check — the transform has never been marked
  verified by this codebase.
- FairShip (pinned commit `f73a305a86468c320665cf41b73582ce7e533181`, the
  geometry every `..._f73a305_v0` transform name refers to) has an existing
  sensitive plane, `exitHadronAbsorber`, registered `kVETO`
  (`muonShieldOptimization/exitHadronAbsorber.cxx:56`) and instantiated in
  `macro/run_fixedTarget.py:393`. Its existence is verified; its adequacy as
  `Sigma_postMS` is not (see OPEN).

Nothing in this section is elevated from Deep Research output or from model
memory — see AGENT_POLICY §5.

## PROVISIONAL

- Candidate scoring implementation: reuse of FairShip's existing sensitive-plane
  infrastructure (`exitHadronAbsorber`, and the separate `origin/eminem_scoring_plane`
  FairShip branch) is a reasonable starting point, **but is candidate
  implementation infrastructure only — it is not already the frozen post-MS
  interface, and does not become one until #43 validates it.**
- Candidate `SourceState` field list: `x, y, z`; `px, py, pz`; PDG/charge; time
  (if required); physical source weight; `source_state_id`; execution/event
  provenance; FairShip configuration identity. Physical source weights and
  utility/proposal weights must remain distinct fields, never merged.

## OPEN — not resolved by this document

1. Exact physical definition of `Sigma_postMS` in the current/pinned TGeo
   geometry. It must be geometry-derived, not a guessed hard-coded `z`.
2. Whether a single-muon `SourceState` carries enough information, or whether
   cutting at `Sigma_postMS` discards upstream secondaries, correlated muons,
   event timing/context, shower activity, or other upstream particles relevant
   to veto behavior. If single-muon reinjection proves insufficient, the
   interface is promoted to an event/bundle state — the comparison is not
   patched to make single-muon look sufficient.
3. Exact minimal `SourceState` schema (beyond the PROVISIONAL candidate list
   above).
4. Whether lineage survives correctly through `MCTrack`, `UpstreamTaggerPoint`,
   `vetoPoint`/SBT, DIS realization (where identifiable), reconstruction, and
   `StageDecision`.
5. Empirical round-trip equivalence: full FairShip event -> capture at
   `Sigma_postMS` -> exact reinjection -> downstream FairShip, compared against
   the un-cut continuation. FairShip/GEANT4 remains the final oracle for this
   comparison. **This is not event-by-event downstream equality** —
   FairShip/GEANT4 downstream physics is stochastic, so the round-trip splits
   into two layers: (a) a deterministic/mechanical check that the captured
   state is faithfully serialized and reinjected (position/momentum/PDG/
   charge/time/config identity, lineage to downstream `MCTrack`/detector
   points); (b) a stochastic check comparing the *conditional distribution*
   `P(Y | S=s, native continuation, config C)` against
   `P(Y | S=s, reinjection, config C)` over repeated FairShip executions from
   the same `SourceState` — those repeats are conditional realizations, not
   new P0 draws. No universal numerical acceptance threshold is fixed here;
   the comparison is not to be tuned until a single-muon interface appears to
   pass.

## Constraints that this decision does not change

- `SHIP-EMPIRICAL-FS-SIM-CONTRACT-V0-SOURCE` remains authoritative for
  empirical semantics (`SourceState`, `FSSimExecution`, `DISRealization` where
  identifiable, `Candidate`, `StageDecision`, aggregates) and is not altered by
  this architecture change.
- Technical failure is never a physics negative.
- Repeated FairShip executions from one `SourceState` are conditional repeats,
  not new P0 draws.
- Physical source weights are never utility multipliers.
- The currently frozen `Y^(k)` (`docs/contracts/fairship_empirical_endpoint_v0.md`)
  is an intermediate project endpoint; `Y^(k) != B`, and nothing here claims
  otherwise.

## LEGACY VALIDATION RECORD

`artifacts/afterms_fairship_transform_validation_v0/` is published on PR #42,
which is **OPEN and unmerged** — this artifact is not part of the integrated
branch unless and until #42 is merged. It tested the historical `.pkl`-based
dataset connector against the `+2.5916 m` transform. Its
`u0_label_acquisition_permitted: false` / `common_frame_anchor: false` result
is **not rewritten**, but it is a legacy/mechanical validation record, not
strong verified evidence of a physical shared-frame mapping: the validator has
open review limitations (sidecar state not fully bound to the audited ROOT
state; the common-frame gate can depend on self-reported
`coordinate_physics_verified`; clean-checkout replay does not contain all
required reference artifacts). Its BLOCKED result remains historically useful,
but its scope is now legacy — it validates a connector this decision does not
use as the canonical interface, and it is no longer the gate for `SourceState`
generation. See #43.

## Next scientific action for #43

Derive `Sigma_postMS` from the TGeo geometry of pinned FairShip
`f73a305a86468c320665cf41b73582ce7e533181`
(`~/thesis/worktrees/FairShip-current-main-f73a305`): query the muon-shield
exit boundary from the loaded geometry itself, and evaluate whether
`exitHadronAbsorber` can be positioned at that boundary as the scoring plane.
No round-trip implementation is included in this document.
