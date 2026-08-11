# Tagging Contract (v0)

Status: **specification, except `src/ship_muon_bg/entities/`, which now
implements `TagSubject`, `FSSimExecution`/`ExecutionStatus`,
`InteractionRealization`, `ReconstructedCandidate`, `ObservationEnvelope` (+
typed payloads), `StageDecision`, and a content-addressed `definition_id`
helper (tested; see `docs/architecture/scientific_architecture_v2.md` §11).**
This document is normative, not an essay. Every clause carries a stable
anchor ID and an epistemic tag:

- **[VERIFIED]** — confirmed by reading current repository code or docs at
  the inspected SHA (citation given).
- **[PROJECT DECISION]** — a choice made by this contract, consistent with
  the frozen boundaries in the governing GOAL, not contradicted by evidence.
- **[PROVISIONAL]** — a reasonable v0 default that is expected to change;
  not a commitment future work must preserve.
- **[OPEN]** — explicitly unresolved; requires FairShip evidence, dataset
  inspection, or SHiP-team/thesis-advisor authority. Not resolved here.

Inspected repository SHA: `aca59a45d5fb9aad6629f3b7834f7e734092814d`
(branch `feat/adapter`).

## 0. Precedence

This contract governs the layer **downstream of a FairShip (or FairShip-like)
execution and upstream of a trained proxy**: observations, tags/stage
decisions, and training targets. It does not re-specify or override:

- `docs/contracts/fairship_adapter_contract_v0.md` — the FairShip boundary
  (adapter I/O, failure taxonomy, reproducibility metadata). This contract
  **refines** that taxonomy (§5) rather than replacing it.
- `docs/contracts/density_problem_contract_v0.md` — the nominal
  density/proposal track (`f(x)`, feature views, `w` semantics). That
  contract is explicitly precedence-holding for the density/proposal track
  and states the utility-tilting track begins only after its gates pass
  (density contract §13 item 10). This contract's `U_k(s)` proxy layer is
  that later utility track; it does not reopen density-track decisions.
- `AGENTS.md` — operational rules for the unrelated `afterms` D9/D9.5
  nightly training campaigns. Not affected by this contract.

Where this contract and an existing contract appear to conflict, the more
specific, earlier-dated contract wins for its own scope; raise the conflict
as a new `OPEN-*` item rather than silently resolving it.

## 1. Scope and non-goals

- **TAG-SCOPE-01** [PROJECT DECISION] This contract defines: `TagSubject`
  identity, execution/realization/candidate lineage, `ObservationEnvelope` semantics,
  `StageDecision`/`Tag` semantics, `TrainingTarget` semantics, censoring
  rules, provenance requirements, and compatibility rules for pooling data
  across runs. It defines the *shape* of these entities, not their FairShip
  serialization (that is `adapters/fairship/`'s job) and not their trained
  content (that is `Nflow`/`ProxyTagger`'s job).
- **TAG-SCOPE-02** [PROJECT DECISION] This contract does **not** define: the
  final `SourceState`, the final endpoint B, a final event-weight/rate
  formula, a multi-muon reduction policy, or any concrete `state_definition_id`
  / `stage_definition_id` / `target_definition_id` payload. Those remain
  `OPEN` (§14) or belong to future, separately reviewed definition records.
- **TAG-SCOPE-03** [PROJECT DECISION] This contract does not implement a
  FairShip runner, does not run FairShip, does not estimate SHiP background
  rates, and does not claim any current proxy equals a true physical
  efficiency. See `CLAIM-NO-*` (§13).
- **TAG-SCOPE-04** [VERIFIED] The canonical entities named in this contract
  (`TagSubject`, `FSSimExecution`, `InteractionRealization`,
  `ReconstructedCandidate`, `ObservationEnvelope`, and `StageDecision`) are
  implemented in `src/ship_muon_bg/entities/` and covered by the focused
  entity and boundary tests. `TrainingTarget`, `RunManifest`,
  `ArtifactManifest`, and the FairShip adapter remain specification-only and
  are not implemented in this slice. `MuDISPathSeed` in particular is not a
  repository symbol; it is external FairShip vocabulary referenced only by
  the GOAL and is treated here as an example of a candidate `subject_type`,
  never as a settled one (`OPEN-01`).
- **TAG-SCOPE-05** [VERIFIED] The word "reconstruction" is already used in
  this repository for an unrelated concept: `src/ship_muon_bg/afterms/d8/reconstruction.py`
  reconstructs a historical **ML model checkpoint** for deterministic
  resampling, not a detector-level reconstructed track/vertex. This
  contract's `ReconstructedCandidate` (§3) is a distinct, FairShip-detector
  concept. Implementations must not conflate the two; a future glossary
  entry should disambiguate at the point of first collision.

## 2. Canonical terminology

| Term | Meaning here | Relation to existing terms |
| --- | --- | --- |
| `TagSubject` | Neutral operational object being tracked through 0..N executions. | New. Not the same as `FlowProposalRecord` (§9, migration). |
| `FSSimExecution` | One FairShip (or FairShip-like backend) execution attempt against a subject. | Generalizes the single implicit run behind today's `SimulationResult` (`src/ship_muon_bg/simulation/types.py:56-80`, [VERIFIED]). |
| `InteractionRealization` | One interaction realization produced within an execution, typed by an explicit `interaction_type` string (e.g. `"muon_dis"`) and `interaction_definition_id` — not a DIS-specific entity name (`OPEN-09`, resolved, §3). | Generalizes today's scalar `SimulationResult.dis: Optional[bool]` (`types.py:67`, [VERIFIED]) from a 1:1 tag into an explicit 0..N, interaction-type-agnostic child entity. |
| `ReconstructedCandidate` | One reconstructed detector-level candidate (track/vertex/hit cluster) tied to an execution or realization. | New; no current analogue in `src/ship_muon_bg`. |
| `ObservationEnvelope` | A recoverable, versioned measurement attached to a subject/execution/realization/candidate. | Generalizes the free-form `metadata: Mapping[str, str]` fields on `FlowProposalRecord`/`SimulationResult` (`types.py:53,74`, [VERIFIED]) into a typed, provenanced record. |
| `StageDecision` / `Tag` | An immutable, versioned interpretation of observations at a named stage (e.g. "veto", "DIS-selection"). | Narrows today's `OutcomeCategory` (`types.py:18-29`, [VERIFIED]), which conflates execution health and one hardcoded stage ("selection") into one 3-way enum. |
| `TrainingTarget` | An explicit, versioned rule for turning `StageDecision`s into an ML label/denominator. | New; today `ProxyTagger/interfaces.py:1-21` ([VERIFIED]) states `U(x)` semantics in prose/docstring, with no machine-readable target definition or denominator rule. |
| `U_{k,phi}(s)` | A proxy model, indexed by an explicit `target_definition_id` (`k`) and its own model/version identity (`phi`). | Generalizes today's single implied "probability of DIS" reading of `ProxyScorer.score` (`ProxyTagger/interfaces.py:1-21`, [VERIFIED]). |

## 3. Entity semantics, cardinality, and lineage

- **SUBJ-01** [PROJECT DECISION] `TagSubject` has immutable fields
  `subject_id`, `subject_type`, `state_definition_id`. `subject_type` names
  *what kind of thing* is tracked (e.g. a raw post-shield row, a
  FairShip-internal path seed, a reconstructed candidate promoted to a new
  subject for re-simulation). `state_definition_id` names the exact schema
  that gives `subject_id`'s coordinates meaning (e.g. the `(N, 8)` PKL schema
  at `CONTRACT_VERSION = "0"`, `src/ship_muon_bg/data_contracts/schema.py:13`
  [VERIFIED]).
- **SUBJ-02** [PROJECT DECISION] No `subject_type` may be asserted as *the*
  thesis `SourceState` by this contract or by any code implementing it. A
  `subject_type` is one declared state definition among possibly several;
  promoting one to "the" `SourceState` is a separate, explicit project
  decision this contract does not make (`OPEN-01`).
- **SUBJ-03** [PROJECT DECISION] A subject's `state_definition_id` is
  immutable for that subject's identity. If the meaning of a subject's
  coordinates changes (units, coordinate convention, plane definition), a new
  `state_definition_id` is minted; existing subjects keep the old one. This
  mirrors the existing pattern of `CONTRACT_VERSION` bumps on breaking schema
  change (`schema.py:11-13` [VERIFIED]).
- **EXEC-01** [PROJECT DECISION] Cardinality: `TagSubject -> 0..N
  FSSimExecution`. Zero is valid (a subject may never be executed). N is
  unbounded a priori — repeated executions of the same subject (same seed
  or different seed, same or different `config`) are permitted and are
  **conditional repetitions of that subject**, never new draws from the
  nominal source measure (GOAL freeze list; this is load-bearing for
  `COMPAT-04`).
- **EXEC-02** [PROJECT DECISION] Cardinality: `FSSimExecution -> 0..N
  InteractionRealization`, and `FSSimExecution -> 0..N ReconstructedCandidate`
  (directly, when reconstruction is not keyed to a specific realization) or
  `InteractionRealization -> 0..N ReconstructedCandidate` (when it is). Neither
  cardinality may be hardcoded to exactly 1 anywhere in canonical core code.
  Evidence this is a real risk, not a hypothetical one: today's
  `SimulationResult` (`types.py:56-80` [VERIFIED]) has exactly one
  `candidate_id`, one `outcome`, and one scalar `dis: Optional[bool]` field —
  structurally a 1-subject/1-execution/1-realization assumption with no room
  for multiplicity at any level.
- **EXEC-02a** [PROJECT DECISION, added closing `OPEN-09`]
  `InteractionRealization` carries `interaction_type` (an open string, e.g.
  `"muon_dis"` for current Muon DIS — not a closed enum, so a future
  non-DIS interaction type needs no new entity or rename) and
  `interaction_definition_id` (the versioned rule that classified it as
  that type, content-addressed per `CONF-01`). This entity, not a
  DIS-specific one, is the canonical lineage node between an execution and
  its candidates (§9a of the architecture doc).
- **EXEC-03** [PROJECT DECISION] `FSSimExecution` carries an
  `execution_status` that is *only* about run health — never about physics
  outcome. The first implemented slice
  (`src/ship_muon_bg/entities/lineage.py::ExecutionStatus`) has exactly two
  values, `SUCCEEDED`/`TECHNICAL_FAILURE`; finer-grained health states
  (`TIMED_OUT`, `CRASHED`, missing-geometry, ...) remain representable as
  `TECHNICAL_FAILURE` plus a free-text `reason`/detail until a concrete
  consumer needs to branch on the distinction, per "don't build for
  hypothetical requirements." Regardless of granularity, this axis is a
  narrowing of the existing `OutcomeCategory.TECHNICAL_FAILURE` value
  (`types.py:27` [VERIFIED]) into its own axis, separated from stage
  decisions (§5). See `CENSOR-01`.
- **EXEC-04** [OPEN] Whether `ReconstructedCandidate` may itself become a new
  `TagSubject` (e.g. for a second-pass simulation) is not decided. If it is,
  the new subject's `state_definition_id` must differ from its parent's
  unless a justification is recorded that the coordinate semantics are
  identical (`SUBJ-03`).

```text
TagSubject
    -> 0..N FSSimExecution
        -> 0..N InteractionRealization
            -> 0..N ReconstructedCandidate
        -> 0..N ReconstructedCandidate   (direct, execution-level)
```

## 4. ObservationEnvelope semantics

- **OBS-01** [PROJECT DECISION] An `ObservationEnvelope` wraps anything
  recoverable from evidence: state coordinates, path lengths,
  material/volume traversal, detector hits, interaction-realization
  metadata, reconstruction multiplicity, candidate observables. Every
  envelope carries: `observation_id`, `observation_definition_id`, `units`,
  `evaluation_status` (`COMPUTED`, `NOT_APPLICABLE`, or
  `TECHNICALLY_UNAVAILABLE` — `OBS-03`), `evidence_reference` (pointer to
  the raw artifact it was read from), `config_provenance` (the
  adapter/geometry/schema versions active when it was computed), and a
  `payload` (`OBS-01a`).
- **OBS-01a** [PROJECT DECISION, mechanism fixed this continuation] The
  two-level design chosen over an unrestricted name/value/units EAV system
  (rationale: `docs/architecture/scientific_architecture_v2.md` §7a):
  `ObservationEnvelope` carries the common provenance/status metadata
  above; `payload` is a typed `ObservationPayload` (a plain frozen
  dataclass, not a dynamic/registry-based type). Implemented payload shapes
  this slice: `ScalarObservationPayload` (one float) and
  `SequenceObservationPayload` (an ordered float tuple) — chosen to *prove*
  both scalar and non-scalar payloads are supported (`OBS-02`), not to
  enumerate the eventual taxonomy. Future structured payloads (e.g. a
  path-traversal record) are additional `ObservationPayload` subclasses,
  never a redesign of the envelope. Implemented:
  `src/ship_muon_bg/entities/observation.py`.
- **OBS-02** [PROJECT DECISION] An `ObservationEnvelope`'s `payload` is not
  required to be a scalar flat-table column. Path-length sequences, hit
  collections, and per-volume traversal maps are valid `ObservationPayload`
  shapes (`OBS-01a`); a flattened scalar view (if built for training) is a
  *derived* observation with its own `observation_definition_id`, not a
  replacement for the structured one.
- **OBS-03** [PROJECT DECISION, mechanically enforced this continuation]
  `evaluation_status = TECHNICALLY_UNAVAILABLE` is distinct from a
  physically-zero or physically-absent observation (`CENSOR-02`). An
  observation extractor (the future `FairShipEvidenceExtractor`, §7 of the
  architecture doc) must set this status explicitly rather than writing a
  sentinel value (e.g. `0`, `-1`, `NaN`) into the payload field.
  `ObservationEnvelope.__post_init__` enforces this structurally, not just
  by convention: a `COMPUTED` envelope must carry a non-`None` payload; a
  `NOT_APPLICABLE`/`TECHNICALLY_UNAVAILABLE` envelope must carry `None` —
  constructing one with a sentinel payload value raises.
- **OBS-04** [VERIFIED] Today's closest analogue is the free-form
  `metadata: Mapping[str, str]` field on `SimulationResult`
  (`types.py:74` [VERIFIED]) and the `detail: str` free-text field
  (`types.py:69-70` [VERIFIED], explicitly documented as "for reports, never
  for branching logic"). Neither carries `units`, `evaluation_status`, or a
  `observation_definition_id`; both are non-normative under this contract
  and must not be read by any future `StageDecision` logic as if they were.

## 5. Stage / tag (`StageDecision`) semantics

- **DECISION-01** [PROJECT DECISION] A `StageDecision` (a.k.a. `Tag`) is
  immutable and versioned: `subject` (the `TagSubject`, `FSSimExecution`,
  `InteractionRealization`, or `ReconstructedCandidate` it decides about),
  `stage_name` (human label, e.g. `"veto"`, `"dis_selection"`,
  `"reconstruction_quality"`), `stage_definition_id` (the exact versioned
  rule that produced `decision`), `decision` (the stage's output value,
  stage-specific — boolean, category, or score), `evaluation_status`
  (`EVALUATED`, `NOT_EVALUATED`, `TECHNICALLY_UNAVAILABLE`), `reason`
  (structured, not free text where avoidable), `evidence_references`
  (`ObservationEnvelope` IDs consumed), and `configuration_provenance`.
- **DECISION-02** [PROJECT DECISION] A semantic change to how a stage decides
  (new cut, new selection logic, new veto definition) **creates a new
  `stage_definition_id`**. Historical `StageDecision` records under the old
  `stage_definition_id` are never overwritten, edited, or deleted; they
  remain valid evidence under their original definition.
- **DECISION-03** [PROJECT DECISION] There is no single `dangerous_muon`
  boolean. `physics_rejection` and `accepted_candidate` (`fairship_adapter_contract_v0.md`
  §"Failure taxonomy" [VERIFIED]) are themselves the output of one particular
  stage (an "operational selection") under one `stage_definition_id`; other
  stages (veto survival, reconstruction quality, a future analysis-specific
  cut) are independent `StageDecision`s over the same evidence, each with
  their own `stage_definition_id`. This directly answers the GOAL's
  instruction not to define one generic boolean.
- **DECISION-04** [PROJECT DECISION] `execution_status` (`EXEC-03`) is never a
  `StageDecision`. A `StageDecision.evaluation_status` may be
  `TECHNICALLY_UNAVAILABLE` *because* the parent execution had
  `execution_status = TECHNICAL_FAILURE`, but the two fields live on
  different entities and neither is derivable from the other without an
  explicit rule. See `CENSOR-01`.
- **DECISION-05** [VERIFIED] Evidence that this distinction is currently
  collapsed: `OutcomeCategory` (`types.py:18-29`) has exactly three values —
  `technical_failure`, `physics_rejection`, `accepted_candidate` — in **one**
  enum on **one** field (`SimulationResult.outcome`), and
  `docs/architecture/repo_architecture_v1.md` (lines 33 and 86 [VERIFIED],
  corrected citation — not `simulation/__init__.py` as an earlier draft of
  this contract claimed) names this "the three-way outcome taxonomy" as the
  *only* thing project code sees crossing the boundary. This is a
  reasonable v0 simplification for a not-yet-implemented adapter, but it is
  structurally a conflation of an execution-health axis with a single
  hardcoded stage-decision axis, exactly the risk the GOAL asks to
  investigate. `DECISION-04` is the fix this contract prescribes; it is not
  yet implemented (`MIGRATION` in the architecture doc).

## 6. Training-target (`TrainingTarget`) semantics

- **TARGET-01** [PROJECT DECISION] A `TrainingTarget` is distinct from any
  `Tag`/`StageDecision`. It states: `target_definition_id`,
  `state_definition_id` (which subject state it is defined over),
  `source_decision_definitions` (which `stage_definition_id`(s) it reads),
  `valid_denominator_rule` (which subjects/executions/candidates count in the
  denominator at all), `reduction_rule` (how multiple
  realizations/candidates per subject collapse to one training row, if they
  do), `missingness_rule` (`CENSOR-*`), `compatible_configuration_set`
  (`COMPAT-*`), and `statistical_unit` (subject? execution? realization?
  candidate?).
- **TARGET-02** [PROJECT DECISION] `U_k(s)` in the GOAL's mathematical chain
  means, precisely: the proxy's *target* is `target_definition_id = k`. A
  proxy is never described as estimating "danger" or "DIS probability" in
  the abstract; it estimates the specific quantity `TrainingTarget[k]`
  defines, over the specific `statistical_unit` that target declares.
- **TARGET-03** [PROJECT DECISION] `reduction_rule` must be explicit whenever
  `statistical_unit != ` the entity `DECISION-01` decisions are attached to.
  E.g. if the statistical unit is "subject" but `StageDecision`s are attached
  per-`InteractionRealization`, the target definition must state how N realizations'
  decisions become one subject-level label (e.g. "any positive", "all
  positive", "weighted by `production_weight`") — silently taking `any()` or
  the first realization is forbidden.
- **TARGET-04** [VERIFIED] Today, no `TrainingTarget` object exists.
  `ProxyTagger/interfaces.py` (docstring, lines 1-21 [VERIFIED]) fixes `U(x)`
  semantics ("0 = never DIS, 1 = DIS always") directly in prose and
  `ProxyScorer.fit`'s docstring says only "labels must be derived from
  non-technical-failure simulation outcomes" — there is no denominator rule,
  no reduction rule, and no `target_definition_id`. `docs/architecture/ml_skeleton_local_pkl_v0.md`
  §6 ([VERIFIED], "Target definition" row) already flags this as an open
  field: "(a) veto survival, (b) DIS-candidate survival, or (c) another
  explicitly named operational target... v0 contract leaves this as an open
  field." `TARGET-01`-`03` are the machine-readable contract that field was
  waiting for; they do not resolve which target is chosen (`OPEN-04`).
- **TARGET-05** [PROJECT DECISION] `U_{k,phi}(s)` (proxy layer, §"Proxy layer"
  of the GOAL) additionally fixes `phi` as the trained-model identity: model
  family/version, `state_definition_id`, `target_definition_id = k`, training
  dataset manifest hash, training config hash, seed, code commit, and
  (if used) `calibration_definition_id` and `metric_definition_id`. A persisted
  proxy artifact missing any of these fields is not a trusted artifact
  (`PROV-04`).

## 7. Missingness / censoring — technical failure is not a physics negative

- **CENSOR-01** [PROJECT DECISION, non-negotiable per GOAL] `execution_status`,
  `StageDecision.evaluation_status`, and `StageDecision.decision` are three
  separate fields. A `TrainingTarget`'s `valid_denominator_rule` must
  explicitly exclude subjects/candidates whose relevant `execution_status`
  is not a success state, or whose relevant `StageDecision.evaluation_status`
  is not `EVALUATED`. A technical failure entering the negative class of any
  `TrainingTarget` is a contract violation unless a future, explicitly-named
  target definition states and justifies the exception (GOAL, "Failure and
  missingness rule").
- **CENSOR-02** [PROJECT DECISION] Distinguish three reasons a value can be
  absent: (a) `TECHNICAL_FAILURE` — no trustworthy answer exists; (b)
  `NOT_APPLICABLE` — the observation/stage does not apply to this subject by
  construction (e.g. a veto stage for a subject that never reached the veto
  volume); (c) `PHYSICALLY_ABSENT` — a trustworthy run found nothing (e.g.
  zero DIS realizations is itself informative, not missing). Only (a) is
  censoring; (b) and (c) are valid physics information and may appear in a
  denominator or numerator per the target's explicit rule.
- **CENSOR-03** [VERIFIED, precedent] The existing `fairship_adapter_contract_v0.md`
  already states this invariant at the execution/stage level: "Counts of
  physics rejections and accepted candidates are only meaningful over the
  set of non-technical-failure outcomes" (contract, "Failure taxonomy"
  section [VERIFIED]) and the existing test
  `tests/test_architecture.py::test_technical_failure_never_carries_a_dis_tag`
  ([VERIFIED], lines 79-89) already enforces `dis is None` whenever
  `outcome is TECHNICAL_FAILURE`. `CENSOR-01`-`02` generalize this existing,
  tested invariant to the full lineage (executions, realizations,
  candidates) and to arbitrary stages, not just the one hardcoded selection
  stage.
- **CENSOR-04** [OPEN] Whether a `NOT_APPLICABLE` stage decision should ever
  count as a denominator-excluding condition (vs. a valid negative) is
  target-specific and not decided here; each `TrainingTarget`'s
  `missingness_rule` must state it explicitly per `TARGET-01`.

## 8. Configuration and immutable identifiers

- **CONF-01** [PROJECT DECISION, mechanism tightened during gauntlet review]
  Every `*_definition_id` introduced by this contract (`state_definition_id`,
  `stage_definition_id`, `target_definition_id`, `observation_definition_id`,
  `calibration_definition_id`, `metric_definition_id`) must resolve, by
  construction, to exactly one semantic definition forever. "Otherwise
  immutable" is not a second, weaker option alongside content-addressing —
  it names how the requirement is *satisfied*, not an escape from it. The
  default, required mechanism is content-addressing: hash the rule/config
  that defines the ID (mirroring `afterms/d9/contract.py`'s
  `semantic_training_hash`, `CONF-02`), exactly as `CONF-02` documents.
  Where a human-readable label is used instead (e.g. during migration,
  `"operational_selection_v0"`), it is provisional and non-compliant on its
  own; it becomes a compliant `*_definition_id` only once paired with a
  recorded content hash of the rule/config it names, so a silent in-place
  edit to that rule is mechanically detectable (`CLAIM-NO-10`). A semantic
  change mints a new ID; it never mutates an existing one in place.
- **CONF-02** [VERIFIED, precedent] This pattern already exists in this
  repository at the training-run level:
  `src/ship_muon_bg/afterms/d9/contract.py` (`semantic_training_hash`,
  lines 54-82 [VERIFIED]) hashes a *semantic fingerprint* (config +
  preprocessing contract + dataset identity + module source fingerprints),
  deliberately not the raw git commit, specifically so an unrelated commit
  leaves resume compatibility unchanged while a semantically relevant change
  invalidates it. The same module also separates `execution_policy_hash`
  (gates resume) from `evaluation_policy_hash` (never gates resume,
  `contract.py` docstring lines 130-136 [VERIFIED], corrected citation — not
  `types.py`, which is a different file, as an earlier draft of this
  contract claimed) — direct precedent for `CONF-01`'s
  "changing definition mints new ID, doesn't mutate" and for `COMPAT-*`'s
  distinction between semantically load-bearing and cosmetic changes.
- **CONF-03** [PROJECT DECISION] The producer's git commit (and dirty-state
  flag) is recorded alongside every `*_definition_id`, never substituted for
  it, mirroring `afterms/d9/contract.py`'s explicit rule (`CONF-02`).

## 9. Provenance requirements

- **PROV-01** [PROJECT DECISION] A `RunManifest`/`ArtifactManifest` sufficient
  to reconstruct a thesis figure or trained proxy back to source artifacts
  must record: project repo commit + dirty flag; FairShip repo commit + dirty
  flag; resolved configuration IDs; geometry/config identifiers; adapter
  version; `state_definition_id`/`stage_definition_id`/`target_definition_id`
  set in use; environment/dependency versions; deterministic seeds; input
  artifact hashes; output artifact hashes.
- **PROV-02** [VERIFIED, precedent] Most of this list already has a working
  precedent at the `adapters/fairship/` boundary
  (`fairship_adapter_contract_v0.md`, "Reproducibility metadata" section
  [VERIFIED]: `command`, `git_commit`, `config_hash`, `dataset_hash`, `seed`,
  `environment_profile`, `backend_name`/`backend_version`, `geometry_tag`,
  `input_schema_version`/`output_schema_version`) and at the density-track
  boundary (`density_problem_contract_v0.md` §5 and §11 [VERIFIED]: per-run
  `schema_version, contract_version, raw_dataset_hash, feature_view_id, ...,
  split_manifest_hash, normalization_hash, seed, git_commit, model_config_hash`
  and required artifacts including `run_manifest.json`). `PROV-01` extends
  this existing pattern one layer further (tag/target/proxy identity), it
  does not invent a new pattern.
- **PROV-03** [PROJECT DECISION] Large scientific artifacts (ROOT files,
  generated candidate arrays, checkpoints) stay outside Git, matching the
  existing project-wide rule (`README.md` "Development Principles": "Keep
  large datasets out of Git" [VERIFIED]). Git contains schemas, manifests,
  contracts, and configuration only.
- **PROV-04** [PROJECT DECISION] A persisted proxy artifact without the full
  field set in `TARGET-05` is not a trusted artifact and must not be loaded
  by any evaluation/proposal code without an explicit "untrusted artifact"
  flag surfaced to the caller.

## 10. Compatibility rules

- **COMPAT-01** [PROJECT DECISION] Default to **incompatible** whenever a
  candidate change might alter the conditional estimand a `TrainingTarget`
  or `StageDecision` represents. Evaluate, at minimum, changes in: random
  seed; host/runtime metadata; FairShip commit; geometry; MuDIS generation
  policy; material placement; reconstruction; selection; veto; state
  definition; tag/stage definition; target definition; physical-weight
  semantics.
- **COMPAT-02** [PROJECT DECISION] Matching *names* (`stage_name`,
  `subject_type`, config file name) are never evidence of semantic
  compatibility. Only matching `*_definition_id` values (content-addressed,
  `CONF-01`) permit pooling two runs' evidence for the same
  `StageDecision`/`TrainingTarget`.
- **COMPAT-03** [VERIFIED, precedent] This "content identity over name
  identity" pattern already exists and is tested at the training-run level:
  `afterms/d9/contract.py`'s `semantic_training_hash` (`CONF-02` above) is
  exactly this rule applied to resume compatibility; `tests/afterms/`
  ([VERIFIED path exists] — not read in full, but referenced by
  `docs/reviews/afterms_d9_5_nightly_execution_v0.md`) presumably exercises
  it. `COMPAT-01`-`02` are the same rule applied to scientific pooling rather
  than resume/checkpoint compatibility.
- **COMPAT-04** [PROJECT DECISION] A `FSSimExecution` that repeats a subject
  already executed (same or different seed) is a **conditional repetition**
  of that subject under the executing configuration, never treated as an
  independent draw from the nominal source measure `P0` when aggregating for
  a `TrainingTarget`. Any aggregation rule that implicitly averages over
  repeated executions as if they were i.i.d. `P0` draws is a contract
  violation; `reduction_rule` (`TARGET-01`) must state how repetitions are
  handled (e.g. weighted by inverse repetition count, or treated as
  additional evidence about the same subject rather than the population).
- **COMPAT-05** [OPEN] The precise set of configuration axes that count as
  "the same configuration" for pooling purposes (i.e., the exact schema of
  a compatibility key) is not fixed here; `COMPAT-01`'s list is a minimum
  checklist, not a closed schema.
- **COMPAT-06** [PROJECT DECISION, added during gauntlet review] Any derived,
  legacy-shaped projection of canonical lineage data (e.g. a
  `SimulationResult`-shaped view built from `FSSimExecution`/`StageDecision`
  records, `MIGRATION` §"SimulationResult" in the architecture doc) **must**
  embed the exact `stage_definition_id`(s) it was derived under in a
  required field of that projection (e.g. its `metadata` mapping), not only
  in the canonical records it was derived from. `COMPAT-02`'s "names are not
  evidence of compatibility" rule applies to derived projections too: two
  such projections may be pooled or compared only if their embedded
  `stage_definition_id` values match. A projection format that drops this
  field is not compliant with this contract and must not be used for
  training or reported results, only for narrow legacy-consumer
  compatibility explicitly scoped to a single run.

## 11. Weight guardrails

- **WEIGHT-01** [PROJECT DECISION] The following remain distinct fields in
  every artifact that carries any of them, and no code may multiply two of
  them together merely because both are available: source physical weight
  (`production_weight`), upstream sampling/oversampling correction
  (`training_sample_weight`), DIS/execution metadata, material/path metadata,
  replication normalization (arising from `COMPAT-04` repetitions),
  exposure normalization, and the utility multiplier `h(U)`.
- **WEIGHT-02** [VERIFIED, precedent] This list is not invented here; it
  extends an existing, already-normative list in
  `density_problem_contract_v0.md` §3.3 ([VERIFIED]): "The following
  quantities must remain separate in all artifacts:
  `production_weight, training_sample_weight, utility_tilt, nominal_log_prob,
  proposal_log_prob, importance_weight`" plus its Red-team gate **R6 —
  weight-semantic conflation** ([VERIFIED], §12). `WEIGHT-01` is the same
  rule extended to the execution/tagging layer (adds DIS/material metadata
  and the utility multiplier `h(U)`, which do not exist at the density-track
  layer).
- **WEIGHT-03** [PROJECT DECISION] No final event-weight or rate formula is
  defined, derived, or implied by this contract or by any code implementing
  it (`OPEN-06`, `CLAIM-NO-*`).

## 12. Claims enabled

- **CLAIM-EN-01** A reviewer can determine, for any persisted record, whether
  it is raw evidence, a derived observation, a versioned interpretation
  (tag/stage decision), a training target, or a trained proxy output, by its
  type alone (`ObservationEnvelope` vs `StageDecision` vs `TrainingTarget` vs proxy
  artifact).
- **CLAIM-EN-02** A reviewer can tell, for any two runs, whether their
  evidence may be pooled, by comparing `*_definition_id` values only —
  never by comparing human-readable names (`COMPAT-02`).
- **CLAIM-EN-03** A reviewer can trace any trained proxy back to: its target
  definition, its state definition, its training dataset manifest, its
  training config, its seed, and its code commit (`TARGET-05`, `PROV-01`).
- **CLAIM-EN-04** A reviewer can confirm that no technical failure has
  entered any target's negative class, by construction of
  `valid_denominator_rule` (`CENSOR-01`), not by post-hoc audit alone.
- **CLAIM-EN-05** A reviewer can identify every open scientific question this
  contract deliberately declines to resolve, and the evidence required to
  resolve it (`OPEN-*`).

## 13. Claims forbidden

- **CLAIM-NO-01** No code implementing this contract may assert that any
  current `subject_type`/`state_definition_id` is the final thesis
  `SourceState`.
- **CLAIM-NO-02** No code implementing this contract may assert that DIS
  survival (or any single stage) is the final utility endpoint `B`.
- **CLAIM-NO-03** No proxy `U_{k,phi}(s)` may be reported as equal to, or a
  calibrated estimate of, the true conditional probability `eta_k(s)` unless
  an explicit `calibration_definition_id` and supporting evidence are
  attached (`TARGET-05`).
- **CLAIM-NO-04** No artifact may combine `production_weight` and any
  utility multiplier `h(U)` into a single "final weight" field
  (`WEIGHT-01`, `WEIGHT-03`).
- **CLAIM-NO-05** No two runs' evidence may be pooled for training or
  reporting on the basis of matching human-readable names alone
  (`COMPAT-02`).
- **CLAIM-NO-06** No repeated `FSSimExecution` of one subject may be counted
  as an additional independent sample from `P0` (`COMPAT-04`).
- **CLAIM-NO-07** No `StageDecision` may be edited or overwritten in place;
  a semantic change always produces a new `stage_definition_id`
  (`DECISION-02`).
- **CLAIM-NO-08** No `execution_status` other than a defined success state
  may contribute to a `StageDecision.decision` or to a `TrainingTarget`'s
  positive or negative class (`CENSOR-01`).
- **CLAIM-NO-09** No derived legacy-shaped projection (e.g. a
  `SimulationResult`-shaped view) may be pooled or compared across records
  whose embedded `stage_definition_id` differs (`COMPAT-06`).
- **CLAIM-NO-10** No hand-chosen, version-suffixed string (e.g.
  `"operational_selection_v0"`) may be treated as a compliant, immutable
  `*_definition_id` unless it is paired with a recorded content hash of the
  rule/config it names (`CONF-01`); an unpaired label is a provisional
  display name only.

## 14. Open questions

- **OPEN-01** Final `SourceState` definition and its `subject_type`/
  `state_definition_id` — requires a project/thesis-advisor decision plus
  FairShip evidence about which intermediate state (raw after-MS row,
  FairShip-internal path seed, or something else) is the intended sampling
  point. Not resolved here (frozen by GOAL).
- **OPEN-02** Final endpoint `B` (the terminal utility target the whole
  chain optimizes toward) — frozen by GOAL; requires SHiP-team/thesis
  authority.
- **OPEN-03** Exact cardinalities in practice (how many `InteractionRealization`s or
  `ReconstructedCandidate`s a real FairShip run actually produces per
  subject) — requires FairShip evidence not available from this repository;
  `EXEC-02`'s `0..N` is a safe upper bound, not a measured distribution.
- **OPEN-04** Which `target_definition_id` (veto survival, DIS-candidate
  survival, or another named target) the first real proxy trains against —
  explicitly deferred by `docs/architecture/ml_skeleton_local_pkl_v0.md` §6
  and not resolved here.
- **OPEN-05** Multi-muon reduction policy — frozen by GOAL; requires a
  project decision informed by FairShip multi-muon event structure evidence.
- **OPEN-06** Final event-weight/rate formula and how `production_weight`,
  replication normalization, and exposure normalization combine — frozen by
  GOAL (`WEIGHT-03`).
- **OPEN-07** Whether `ReconstructedCandidate` may be promoted to a new
  `TagSubject` for re-simulation (`EXEC-04`).
- **OPEN-08** The exact compatibility-key schema beyond the minimum checklist
  in `COMPAT-01` (`COMPAT-05`).
- **OPEN-09** [RESOLVED, second continuation — kept for history, not deleted]
  Originally: the lineage entity's name (then `DISRealization`) encoded a
  DIS-specific assumption at a structurally privileged lineage level, even
  though its fields were already interaction-type-agnostic; whether a future
  non-DIS interaction realization would reuse it, need a parallel entity, or
  need a rename was left undecided. **Resolution [PROJECT DECISION]:**
  renamed to `InteractionRealization`, with explicit `interaction_type`
  (open string, e.g. `"muon_dis"`) and `interaction_definition_id` fields
  (`EXEC-02a`). Implemented as
  `src/ship_muon_bg/entities/lineage.py::InteractionRealization`. A
  DIS-specific typed view/specialization may still be layered on top later
  if a concrete consumer needs one; the base entity itself is now neutral by
  construction, so this no longer blocks modeling a second interaction
  type.
- **OPEN-10** [added during gauntlet review] Raw FairShip backend output
  retention is not specified. `PROV-03` states raw artifacts stay outside
  Git but does not commit to *any* retention policy or duration. Re-tagging
  already-extracted `ObservationEnvelope`s under a new `stage_definition_id`
  (`DECISION-02`) is always possible from stored `ObservationEnvelope.payload`
  data; re-tagging under a new `observation_definition_id` that the
  original adapter version did not extract requires re-reading raw backend
  outputs, which is only possible if they were retained. This retention
  policy requires an infrastructure/storage-budget decision not made here.
- **OPEN-11** [added during gauntlet review] `fairship_adapter_contract_v0.md`'s
  "Output contract" section normatively requires the future adapter's
  output to be exhaustively classifiable into the existing three-way
  `technical_failure`/`physics_rejection`/`accepted_candidate` taxonomy —
  a requirement on the adapter's actual output, not a legacy convenience
  view. `DECISION-03`/`DECISION-05` (this contract) instead treat that
  taxonomy as one stage's (`"operational_selection_v0"`) decision values
  among potentially several stages. These two contracts are not yet
  reconciled: either `fairship_adapter_contract_v0.md`'s output-contract
  section is amended to require the fuller lineage output instead of (or
  in addition to) the flat taxonomy, or this contract's stage model is
  scoped to be additive on top of an unchanged adapter output contract.
  Resolving this is a prerequisite for implementing
  `src/ship_muon_bg/adapters/fairship/`, not for the density/proposal-track
  work this contract otherwise unblocks.

## 15. Readiness criteria

This contract is ready for independent critique when, and only when:
(a) every entity in §3 has a stated cardinality with no hidden 1:1 assumption;
(b) `execution_status`, `StageDecision`, and `TrainingTarget` are
demonstrably on three different fields/objects, not one (§5-6, §7);
(c) every `*_definition_id` is stated to be immutable-on-change (§8);
(d) `CENSOR-01` is unconditional except for a named future exception path;
(e) `WEIGHT-01` lists every factor the GOAL named and forbids their
unconditional multiplication;
(f) every `OPEN-*` item in §14 matches an item on the GOAL's freeze list or
names the missing external evidence/authority; and
(g) this contract does not silently narrow or contradict
`density_problem_contract_v0.md` or `fairship_adapter_contract_v0.md`.

This contract is **not** ready to gate an implementation PR until the
architecture document (`docs/architecture/scientific_architecture_v2.md`)
demonstrates each entity here maps to an implementable interface without
contradiction.
