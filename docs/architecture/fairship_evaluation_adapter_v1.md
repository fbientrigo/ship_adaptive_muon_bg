# The FairShip evaluation boundary, v1

**What this document is for:** a physicist should be able to read it and then
audit the whole boundary end to end. Everything it describes runs with no ROOT,
no FairShip, no CERN account, and no LXPLUS.

**What the boundary is for:** to let the scientific core treat an expensive
simulator as an interchangeable backend, so that FairShip's current
implementation — its file formats, its geometry, its selection code, its
directory layout — cannot leak into the thesis's scientific reasoning.

---

## 1. The pipeline

```text
TagSubject
    |
    v
EvaluationRequest ---------------------> EvaluationBackend  (protocol)
                                              |
                                    +---------+---------+
                                    |                   |
                          FakeFairShipBackend   (future) FairShipAdapter
                                    |                   |
                                    +---------+---------+
                                              |
                                              v
                                       EvaluationBundle
                                    (canonical records only)
                                              |
        FSSimExecution / InteractionRealization / ReconstructedCandidate
              / ObservationEnvelope / StageDecision
                                              |
                                              v
                          tagging.aggregate_state_stage_evaluations
                                              |
                                              v
                            TaggingDataset  (proxy-ready table)
                                              |
                                              v
                                    ProxyTagger  ->  U(x)  ->  Nflow
```

Everything FairShip-specific lives to the left of `EvaluationBundle`. Everything
to the right sees canonical records and content-addressed definition ids, and
nothing else.

## 2. Where the code lives, and why

| Module | Role | May import |
| --- | --- | --- |
| `entities/` | canonical records; validation invariants only | nothing project-specific |
| `simulation/evaluation.py` | `EvaluationRequest`, `EvaluationBundle`, `EvaluationBackend`, verification helpers | `entities/` |
| `simulation/fake_fairship.py` | `FakeFairShipBackend` — deterministic test infrastructure | `entities/`, `simulation/` |
| `simulation/stub_backend.py` | `MinimalStubBackend` — the substitution control arm | `entities/`, `simulation/` |
| `tagging/aggregation.py` | records → per-state empirical summaries | `entities/` |
| `adapters/fairship/` | *not implemented*; the only ROOT/FairShip-allowed subtree | `entities/`, `simulation/`, ROOT/FairShip |

Two placement decisions carry weight:

**The new modules sit inside packages the AST import guard already sweeps.**
`tests/test_architecture_boundaries.py` walks `entities/`, `data_contracts/`,
`simulation/`, and `tagging/` and rejects any import of ROOT, FairShip, or
`ship_muon_bg.adapters.fairship`. Nothing new had to be added to enforce
backend-independence, and no future module in these packages can escape it.

**`tagging/` cannot import `simulation/`.** The aggregation layer therefore
consumes plain tuples of canonical entities rather than an `EvaluationBundle`.
It does not know a backend exists. That is the strongest available form of the
substitution guarantee: not "we checked it doesn't branch on the backend", but
"the backend layer is not reachable from it".

`simulation/fake_fairship.py` has `fairship` in its filename. The guard matches
dotted *import segments* — `fake_fairship` is not `fairship` — and the module
imports nothing FairShip-related. The name is descriptive of what it imitates.

## 3. `EvaluationRequest` — what the core asks for

Identifiers, not content. The request carries `TagSubject` identity records and,
separately, the observations describing them. It declares no kinematics field,
no weight field, and no stage semantics; those already have canonical homes.

Replication is `replications_per_subject`, and the request rejects a repeated
`subject_id`. Be precise about what that buys: it is a check on the *identifier*.
It stops a state being appended twice to a list. It does not stop one physical
state being minted under two ids — `subject_id` is caller-supplied and opaque —
so keeping ids one-per-state is the request builder's obligation, and whether
`subject_id` should be content-addressed over the coordinates is
[OPEN] (two distinct muons can share rounded coordinates).

`options` is opaque backend configuration: paths, environment profiles, resource
limits. It must never carry scientific meaning, and because "must never" is a
rule rather than a mechanism, `options_digest` content-addresses it and
`assert_requests_compatible` catches the specific hazard it affords — two runs
sharing one `fs_sim_configuration_id` whose options differ in something that
actually changed the physics.

## 4. `EvaluationBundle` — what a backend returns

A flat, normalized record set. Not a nested tree, not an aggregate: no counts,
no rates, no `eta_hat`. Parents and children join by identifier, exactly as they
would in on-disk tables.

Enforced in `__post_init__`, without needing the request:

- every identifier is unique across *all* record kinds, so an untyped
  `subject_ref` can never resolve to two different kinds of thing;
- every child points at a parent present in the same bundle, and a candidate's
  realization must belong to the candidate's own execution;
- `candidate_index` may not repeat within one `(execution_id, realization_id)` —
  otherwise it would be the first key in the schema on which an ordinary
  `groupby` silently collapses N candidates to one;
- a technically failed execution may carry evidence and censoring markers but
  **never** realizations, candidates, or an evaluated decision. A run with no
  trustworthy answer must not deposit countable physics objects. Partial or
  truncated output belongs in `FSSimExecution.provenance`.

`is_physical` is restated inside the bundle because the records are what get
persisted and eventually plotted, while the backend object does not travel with
them. It is a self-declaration: it stops an honest mistake, not a lie.

`verify_evaluation_bundle(bundle, request)` adds the request-relative half —
subject membership, configuration agreement, evidence resolvability, subject/record
id disjointness, and a cap on executions per subject. `evaluate_verified` makes
the checked path the default. A shortfall against `replications_per_subject` is
*allowed* but reported by `execution_shortfall`, because a backend that omits
crashed runs looks identical to one that was never scheduled, and only the first
is censoring.

## 5. `FakeFairShipBackend` — not a physics model

It contains no transport, no geometry, no detector response, and no SHiP
selection. Every number it produces is a hash. `is_physical` is `False` so that
statement is machine-checkable rather than a naming convention, and nothing it
produces may be cited as SHiP physics evidence.

What it is for: exercising every structural situation a real adapter will have
to represent — an absent execution, a technical failure, `0..N` realizations,
`0..N` candidates, candidates unattached to any realization, and evidence that
is computed, technically unavailable, or absent entirely.

Determinism is scoped deliberately. Outcomes depend on the seed, the
configuration id, and the subject's identity and state definition. They do
**not** depend on `request_id` or `options` — a physics answer that moves when
you rename the job is not reproducible — so relabelling a request changes the
identifiers and nothing else.

Hand-computability: explicit `FakeRunPlan` objects make the emitted records
follow a script, so a test can state the expected counts in advance.
Unscripted subjects fall back to the hash-derived structure.

Each stage carries its own observation definition and its own independently
derived value, so no stage implies another (invariant 3.4). Two stages sharing
one observable is rejected at construction.

The strongest available check that the fake is not inventing its own semantics:
`StageEvaluator` — the tagging layer, which shares no code with the backend —
re-derives every emitted decision from the emitted evidence and produces
byte-identical records, including decision ids.

`MinimalStubBackend` is a second implementation walking a fixed integer cycle,
sharing no code path with the first. It exists so the substitution test has a
genuine control arm rather than two spellings of one implementation.

## 6. Aggregation — the one genuinely scientific reduction

**The statistical unit is the execution. The grouping unit is the source state.**
Several executions of one `TagSubject` are conditional repetitions of that one
post-shield state; they are grouped under it, never spread out as independent
draws (invariant 3.1).

Each `(execution, stage)` pair lands in exactly one of five outcomes:

| Outcome | Meaning | In `eta_hat` denominator? |
| --- | --- | --- |
| `POSITIVE` | evaluated, and the stage came out positive | yes, numerator |
| `NEGATIVE` | evaluated, and the stage came out negative | yes |
| `TECHNICAL_FAILURE` | the run crashed | no |
| `TECHNICALLY_CENSORED` | ran, but stage evidence was unavailable | no |
| `NOT_EVALUATED` | the stage was never applied, or nothing says it was | no |

Collapsing any of the last three into `NEGATIVE` is the single failure mode this
layer exists to prevent.

**A negative must be attested, never inferred from silence.** An execution with
zero candidates and no decision saying the stage was applied is `NOT_EVALUATED`.
This matters more than it sounds: the commonest real partial failure is a job
that exits 0 but whose reconstruction output is empty or truncated, and from the
candidate records alone that is indistinguishable from a clean zero. Resolving
the ambiguity in favour of a countable negative would manufacture a confident
physics zero out of a silent reconstruction failure, with
`technical_failure_count = 0` in the audit columns. So a backend that really did
look says so, with an execution-level `StageDecision` — `EVALUATED False` for a
genuine zero, `TECHNICALLY_UNAVAILABLE` when the evidence could not be obtained.

That attestation has to be falsifiable, or it costs a careless adapter one free
assertion. Every `EVALUATED` decision must cite at least one `COMPUTED`
observation: report the fact you checked — the reconstructed candidate count —
as evidence, and let the decision cite it. When the count could not be read the
observation is `TECHNICALLY_UNAVAILABLE`, and the boundary then refuses any
`EVALUATED` decision resting on it. The loop closes: a silent reconstruction
failure cannot become a clean zero.

The cost of *not* attesting is on the record too. `evaluable_fraction` is
`valid_count / execution_count` on every row, because an adapter that never
attests loses every zero-candidate run from the denominator — which does not
lose precision, it changes the estimand from `P(pass)` to
`P(pass | at least one candidate)` while still printing a confident `eta_hat`.

`NOT_EVALUATED` currently merges `CENSOR-02`'s `NOT_APPLICABLE` with
plain missing evidence. Excluding both from the denominator is this layer's
conservative default; `CENSOR-04` makes that target-specific and leaves it
[OPEN], and the count is retained separately so a target that wants them can
include them.

The rollup from candidate records to one outcome per execution is named and
stored on every row (`ROLLUP_RULE`) rather than left as an invisible assumption.
Its precedence:

1. a technically failed run is `TECHNICAL_FAILURE`, whatever else is present;
2. an execution-level decision for the stage, where a backend reports one, is
   authoritative — but a contradiction under the *same* stage definition is
   **refused in both directions**, not silently resolved. An execution-level
   negative over a positive candidate, and an execution-level positive over
   candidates that were all evaluated negative, both raise. A veto is a
   different rule and needs its own stage definition. An identified positive
   candidate does still win over an execution-level *censoring* marker, because
   an identified positive is identified whatever the run-level record says;
3. otherwise: a positive candidate settles the execution regardless of what
   happened to its siblings; censoring only matters when it could still have
   changed the answer. So a positive plus a censored sibling is `POSITIVE`, but
   a negative plus a censored sibling is `TECHNICALLY_CENSORED`, not a negative.

The governing principle in one line: **identification beats non-identification,
and non-identification beats a claim of absence.** That is why the two
exceptions to rule 2 run in the same direction — an identified positive
candidate outranks an execution-level censoring marker, and an explicit
candidate-level censoring record blocks an execution-level negative. Only
*complete* candidate evidence can contradict an execution-level record, since a
censored sibling could have been the one that agreed with it.

`ROLLUP_RULE` is content-addressed over that precedence written out as data, so
changing the precedence necessarily changes the id stored on every row — a bare
version string would let two vintages pool under one label.

`decision_level` records, per execution, which records decided it. `Y` and
`1{N >= 1}` genuinely diverge on the execution-decided path: a stage whose
positive condition is the *absence* of a candidate is legitimately positive with
`N = 0`. Recording the level means a consumer reconstructing `Y` from the
multiplicity distribution never silently disagrees with `positive_count`.

`eta_hat = positive_count / valid_count`, and `None` — not `0.0` — when nothing
is evaluable. Writing zero there would fabricate a confident negative out of
missing data.

**What `eta_hat` is not.** It is the observed frequency. It is not presented as
an unbiased estimator of anything, no independence between repeated executions
of one state is asserted, and no standard error is computed, because the correct
uncertainty treatment for repeated conditional simulations of one state is
[OPEN]. The raw counts are retained so a different method can be applied later
without re-running the simulator.

Multiplicity survives aggregation: every row carries the distribution of
candidate counts and of stage-positive candidate counts over its valid
executions, so `N` remains recoverable and `Y = 1{N >= 1}` stays derivable from
`N` rather than the reverse.

Configuration is part of the grouping key — and so is the options digest each
execution records in its provenance, since a configuration *label* cannot cover
a free-form options mapping. Be precise about the claim: the rows **preserve**
the axis, they do not defend it. A caller can still average a mixed table, which
is what `require_single_configuration` and `require_single_state_definition`
exist to prevent, and why any step that concatenates, averages, or fits across
rows must call them first. `TaggingDataset` refuses duplicate
`(state, configuration, options, stage)` rows, so combining batches means
aggregating over the concatenated records rather than concatenating datasets.

A declared state that produced no execution at all does not appear in the rows.
It is reported in `unevaluated_subject_ids` instead, because if dropped work
correlates with anything physical the proxy would otherwise train on a
selection-biased population with nothing in the artifact to reveal it.

**No weight is applied anywhere in this module.** It emits counts and a
frequency. How to weight a state is a separate, explicit decision, and emitting
one opaque number here is exactly how `w_i` and `h(U_i)` get multiplied together
by accident (`WEIGHT-01`).

## 7. Substitution

`tests/evaluation/test_backend_substitution.py` runs one pipeline function
against both backends and asserts the invariants hold for both, then drives the
full path into `ProxyTagger.DummyProxy` exactly as it already exists.

An AST sweep in the same file shows `ProxyTagger/`, `Nflow/`, and
`src/ship_muon_bg/tagging/` import no `simulation`, `adapters`, or `fairship`
module at all — so they could not branch on backend identity even if someone
wanted them to.

This is evidence, not proof: neither backend is FairShip. What it establishes is
that the downstream stack contains no branch on backend identity and reads
nothing a real adapter could not also supply.

**Backend honesty is the trust boundary, and no check here can cross it.** A
backend that reports a crashed run as `SUCCEEDED`, emits one candidate record
for what were physically many, or attributes a run to the wrong subject passes
every structural check by construction. `evaluate_verified` compares a bundle's
self-declaration against the backend that produced it, which catches a wrapper
that forgot to propagate `is_physical` — an honest mistake, not a lie. A real
adapter needs its own tests against known simulator output.

## 8. What is deliberately not decided here

Endpoint `B`; the stage chain `G → D → R → S → V → B`; SBT/UBT semantics; the
geometry; real DIS execution cardinality; physical weight propagation; whether
`subject_id` should be content-addressed; the uncertainty treatment for repeated
conditional simulations. See
`docs/investigations/fairship_evaluation_adapter_v1_audit.md` §7 and
`docs/handoffs/real_fairship_adapter_next.md`.
