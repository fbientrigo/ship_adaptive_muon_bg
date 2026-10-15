# First empirical FairShip endpoint `Y^(k)` v0

Status: `PROJECT DECISION` — `SBT_USABLE_FOR_FIRST_EMPIRICAL_ENDPOINT` from
the matched TGeo audit. This freezes an intermediate label only; it does not
freeze final endpoint `B`.

## Definition

`stage_definition_id` is the content-addressed
`fairship_sbt_selection_v0@sha256:1ac14cb360d7500eb652ede256b17b7cf12aa384e43fc3101e6e49eaa0eb07f2`,
exported by `ship_muon_bg.benchmarks.fairship_endpoint`. The consumed SBT
observation definition is independently content-addressed there as
`fairship_sbt_qualifying_hit_count_v0@sha256:669ffed16b93494731888d6e19be37482cfed12708bf0150c729af205fdcb42e`.

For one successfully evaluated FairShip `cbmsim` event, let `H` be the count
of `vetoPoint` entries satisfying the current-main MuonDIS SBT rule:

`1000 < detector_id < 999999 AND abs(pdg_id) == 13 AND momentum_gev > 3.0`.

The label is `Y^(k)=1` iff `H > 0`, and `Y^(k)=0` iff `H = 0`, for one
successfully evaluated, single-event `FSSimExecution`. The source candidate
(`FSSimExecution.subject_id`) is distinct from the execution id. The valid
denominator is executions with `SUCCEEDED` health and `EVALUATED` endpoint
status. Repeated executions are conditional repeats grouped under their
source subject, not independent nominal/P0 draws. Technical failures and
unavailable `vetoPoint` output are `TECHNICALLY_UNAVAILABLE` /
`NOT_EVALUATED`, never `Y^(k)=0`, and are excluded from that denominator.

This is a real FairShip/reconstruction-preprocessing outcome, not a manual
geometry mask. It is physically interpretable as eligibility for the current
SBT preprocessing path, and its cost is one existing `cbmsim.vetoPoint` scan.
The definition is independent of final endpoint `B`: `Y^(k) != B` until a
future SHiP-team decision explicitly freezes that equivalence.

## Evidence comparison

The reused fixed pilot has 12 genuine NF candidates (six per charge) and two
empirical controls intentionally selected because they reproduce positivity.
They are validation evidence only, not a random sample or a rate/efficiency
denominator. The 12-row NF cohort has no redraw or success selection. All rows use the same
FairShip commit/configuration and geometry identity from
`artifacts/fairship_tgeo_audit_v0/manifest.json`.

| candidate stage | NF `Y=1/0` | controls `Y=1/0` | cost | reproducibility / interpretation |
| --- | ---: | ---: | --- | --- |
| SBT qualifying-hit count (`H > 0`) | `0/12` | `2/0` | one `vetoPoint` scan | fixed rule/config; current SBT eligibility; selected |
| raw `vetoPoint` presence | `1/11` | `2/0` | one `vetoPoint` scan | fixed output, but less specific; not selected |
| UBT point presence | `12/0` | `2/0` | one `UpstreamTaggerPoint` scan | fixed output, no negative observed; not selected |
| MCTrack presence | `12/0` | `2/0` | one `MCTrack` scan | fixed output, no negative observed; not selected |

For the fixed pilot, all 14 audited candidate rows have a computed SBT count,
with zero unavailable or missing observations. No execution-level
technical-failure count is claimed: the candidate CSV does not store an
`FSSimExecution` id or status.

The audit records `PROVISIONAL` coordinates for the NF rows and `UNKNOWN`
coordinates for the controls; this status remains attached to the evidence.
Its candidate CSV does not store an `FSSimExecution` id or execution status, so
the table above is candidate-level audit evidence, not an execution-level
denominator. A future persisted endpoint table must carry those lineage fields.
The FairShip audit and connector remain `VERIFIED` only within their declared
runtime/configuration scope. No rate, efficiency, causal, or endpoint-`B`
claim follows from this pilot.

## Persisted lineage requirement

The committed audit CSV is candidate-level evidence only: it has no
`FSSimExecution` id or execution status. It therefore does not claim an
execution-level valid denominator or stored source-candidate lineage. A future
endpoint table must carry the source candidate/`TagSubject` id, one
single-event `FSSimExecution` id and status, the observed SBT count/status, and
the resulting `StageDecision` id/status. Only `SUCCEEDED` executions with an
`EVALUATED` decision enter the physics denominator; repeated executions remain
conditional repeats grouped by source subject. `physical_source_weight` remains
separate from any future `utility_multiplier`.

No source-candidate aggregation of repeats is frozen or allowed by this v0
contract. `Y^(k)` remains one label per successful `FSSimExecution`, while
repetitions retain their shared source-subject group identity. Any reduction
to one candidate-level value requires a future versioned `PROJECT DECISION`.
