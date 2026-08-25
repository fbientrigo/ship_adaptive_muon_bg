# Handoff — the real FairShip adapter

**Nothing in this document is implemented.** It lists what the next mission has
to *find out* before it can write `src/ship_muon_bg/adapters/fairship/`, and
what the boundary already guarantees so that mission does not have to
re-litigate it.

Read first: `docs/architecture/fairship_evaluation_adapter_v1.md`,
`docs/contracts/fairship_adapter_contract_v0.md`,
`docs/contracts/tagging_contract_v0.md`.

---

## 1. What is already settled

The adapter's job is fixed and small: implement `EvaluationBackend`, i.e.

```python
def evaluate(self, request: EvaluationRequest) -> EvaluationBundle: ...
```

and return canonical records that pass `verify_evaluation_bundle`. If it does
that, nothing downstream changes — `tagging/aggregation.py`, `ProxyTagger`, and
`Nflow` cannot see it, because they import no simulation or adapter module at
all.

The adapter is also the **only** subtree of `src/ship_muon_bg/` allowed to
import ROOT or FairShip (`scientific_architecture_v2.md` §2a), and
`tests/test_architecture_boundaries.py` enforces that the rest of the core does
not.

Homes that already exist for FairShip-specific data, so nothing new has to be
invented:

| What the adapter has | Where it goes |
| --- | --- |
| output ROOT file paths, job ids, hostnames, container digests | `FSSimExecution.provenance` |
| the sub-seed a run actually used | `FSSimExecution.seed` |
| why a run is untrusted | `FSSimExecution.failure_reason` (only on `TECHNICAL_FAILURE`) |
| a raw artifact a value was read from | `ObservationEnvelope.evidence_reference` |
| any observable, scalar or not | `ObservationEnvelope` + a typed payload |
| site paths, geometry files, environment profiles | `EvaluationRequest.options` |
| the options digest a run was launched under (**required** on every execution, key `OPTIONS_DIGEST_PROVENANCE_KEY`) | `FSSimExecution.provenance` |

## 2. Investigation targets

Each of these is a question to answer against a **named FairShip commit** with
`file:line` evidence, per `AGENT_POLICY.md` §5. None may be answered from
memory or from "cleaner"/"better".

### T1 — The source-state surface [OPEN]

What exactly does FairShip need to start a run from a post-shield muon state,
and what is the minimal set that fully determines it? Identify the symbols that
consume it. Until this is answered, `TagSubject` stays at three fields and
coordinates travel as observations under a versioned `state_definition_id`.

Blocked question this feeds: should `subject_id` be content-addressed over the
state coordinates? Two distinct muons can share rounded coordinates, so the
answer is not obviously yes. Today the request only rejects a repeated
`subject_id`, and one-id-per-state is the request builder's obligation.

### T2 — DIS execution cardinality [OPEN]

Does one execution produce exactly one interaction realization, or `0..N`?
Find the code path that decides. The schema already supports `0..N` and assumes
nothing, so answering this narrows the adapter, not the entities.

### T3 — Reconstruction output shape [OPEN]

How many candidate objects can one event yield, what identifies them, and is
there a natural ordering? This determines how the adapter fills
`candidate_index`, which is scoped to `(execution_id, realization_id)` and must
never be relied on as a count.

### T4 — The failure taxonomy in practice [OPEN]

The single most important target. Enumerate the concrete ways a real run fails:
non-zero exit, timeout, truncated ROOT file, missing branch, NaN, empty tree,
partial event loop. For each, decide whether it is `TECHNICAL_FAILURE` or a
legitimate physics zero.

There is a **third** case, and it is the one most likely to be reached by
omission rather than by decision: a run that exited 0 but whose reconstruction
or stage evidence could not be obtained. It is not a crash and it is not a
physics zero. Report it as a `SUCCEEDED` execution carrying an execution-level
`StageDecision` with `TECHNICALLY_UNAVAILABLE`. Report a genuine clean zero as
`SUCCEEDED` with an execution-level `EVALUATED False`. Emitting neither is not neutral, and the cost is larger than it
sounds. Aggregation reads an unattested empty run as `NOT_EVALUATED` and drops
it from the denominator — which does not lose a measurement, it **redefines the
estimand** from `P(pass)` to `P(pass | at least one candidate)`, while the row
still prints a confident `eta_hat` beside it. In a rare-process regime where
most runs reconstruct nothing, that is an order-of-magnitude inflation, not a
rounding error. `evaluable_fraction` on every row is the signal: far below 1
means the `eta_hat` next to it answers a narrower question than its name
suggests. Check it on the first real table you produce.

The attestation must also be *falsifiable*. Every `EVALUATED` decision has to
cite at least one `COMPUTED` observation, so report the fact you actually
checked — the reconstructed candidate count — as an observation and let the
decision cite it. When that fact could not be read, the observation is
`TECHNICALLY_UNAVAILABLE` and the boundary then refuses any `EVALUATED`
decision resting on it. That is what stops a silent reconstruction failure
being laundered into a clean zero. The reference backends show the shape; a
fake is entitled to assert the count because it manufactured it, and you are
not.

The boundary already forbids the dangerous middle case: a `TECHNICAL_FAILURE`
execution may not carry realizations, candidates, or an evaluated decision, and
no `EVALUATED` decision may cite an observation that was never computed. So a
truncated file that yielded three tracks must be reported as a failure with
those tracks recorded in provenance, **not** as three candidates. If that turns
out to be too strict for a real failure mode, that is a contract change to
argue explicitly — not something to work around.

Related and unresolved: `ExecutionStatus` has exactly two values. Real runs meet
timeouts, OOM kills, partial output, and non-zero exits that nonetheless
produced usable events. Deciding whether that taxonomy needs to grow is part of
this target.

### T4a — Where does the adapter attach its stage decisions? [OPEN]

Today `tagging/aggregation.py` rolls up **execution-level** and
**candidate-level** decisions only. `EXEC-02a` names `InteractionRealization` as
the canonical lineage node between an execution and its candidates, which is
exactly where a DIS selection would naturally sit — and a realization-scoped
decision is currently **refused** by the aggregation rather than silently
dropped, because the right rollup semantics for it have not been defined.

If the real adapter needs realization-scoped stages, defining those semantics is
part of that mission: what does one realization passing mean for the execution,
and does `TARGET-03`'s prohibition on "silently taking `any()`" force an
explicit, named rollup at that level too?

### T5 — Stage evidence [OPEN]

Which observables does the adapter export, and which stage rules can the
tagging layer re-derive from them versus which are opaque outcomes FairShip
computed internally? The `StageDefinition` / opaque-`stage_definition_id` split
already supports both, but each stage has to be classified.

Do not hardcode historical SHiP cuts. Do not assume any implication between
stages.

### T5a — Costs of the evidence requirement, to size rather than discover [OPEN]

Every `EVALUATED` decision must cite at least one `COMPUTED` observation. Two
consequences to plan for:

- **An opaque FairShip-internal boolean has no typed payload home.** `OBS-01a`
  implements only `ScalarObservationPayload` and `SequenceObservationPayload`,
  so such an outcome travels as `0.0`/`1.0`. That is an additive
  `ObservationPayload` subclass when someone needs it, not a redesign. Note the
  requirement is not busywork even here: it converts *"FairShip said no"* into
  *"we read branch B of file F under adapter version V and it said no"*, which
  is what makes re-tagging under a new `stage_definition_id` possible without
  re-running the simulator (`DECISION-02`, `OPEN-10`).
- **Record volume.** One observation per candidate per stage is a lot of rows
  for a real campaign. They are exactly the quantities worth persisting, and
  `PROV-03`/`OPEN-10` own that question — but size it before the first large
  run rather than after.

The check is deliberately a *liveness and lineage* check, not a **relevance**
check: nothing verifies that the cited observation measures anything to do with
the stage. Closing that would mean passing `StageDefinition` objects into the
aggregation instead of bare id strings, which crosses the `tagging` →
`simulation` dependency direction and is a design change, not a fix. Recorded
as a known limit.

### T6 — Configuration identity [OPEN]

What must `fs_sim_configuration_id` content-address so that two runs sharing it
are genuinely poolable? Geometry version, physics list, FairShip commit,
container digest, magnetic field map are the obvious candidates — confirm
against the code. Anything scientifically load-bearing that ends up in
`options` instead is a hole; `options_digest` and `assert_requests_compatible`
make such drift detectable but do not prevent it.

### T7 — Physical weight propagation [OPEN]

How does a source weight `w_i` reach and survive a FairShip run? Record it as an
`ObservationEnvelope` with its own `observation_definition_id`. Under no
circumstances introduce a `final_weight` field or multiply a physical weight by
a utility multiplier `h(U_i)` (`WEIGHT-01`).

### T8 — SBT / UBT semantics [OPEN]

Absent from this repository entirely — no code, config, or data mentions SBT,
UBT, or DOCA. Nothing was assumed. This needs a SHiP-team answer before any
veto-detector stage is defined.

### T9 — Endpoint `B` [OPEN]

Not decided, not encoded. The stage set is a placeholder with no chain.

### T10 — Uncertainty on `eta_hat` [OPEN]

Repeated executions of one source state are conditional repetitions, and this
repository asserts nothing about their independence. `eta_hat` is published as
an observed frequency with no standard error. Raw counts are retained so
whatever method is chosen can be applied without re-running anything. This is a
statistics decision, not an implementation one.

## 3. Operational targets

- **Where does FairShip actually run?** LXPLUS/HTCondor, a container, or local.
  This determines whether `evaluate` is synchronous or has to model submission
  and collection. The current protocol is synchronous; an asynchronous adapter
  may need a second protocol method, which is an additive change.
- **Cost.** How long does one execution take, and what is a realistic
  `replications_per_subject`? This shapes every downstream statistical choice.
- **Reproducibility.** Is a FairShip run bit-reproducible given a seed? If not,
  the `EvaluationBackend` determinism requirement needs restating as
  "deterministic up to the simulator's own reproducibility guarantee", and that
  weakening must be written down rather than assumed.

## 3a. Known costs and sharp edges

Not defects — deliberate trades, recorded so they are not rediscovered as
surprises.

- **A legitimately inapplicable candidate discards its run's negative.**
  `DecisionEvaluationStatus` has no separate `NOT_APPLICABLE` value, so
  `CENSOR-02`'s "not applicable" travels as `NOT_EVALUATED`, and one such
  candidate now censors an otherwise attested negative. It fails safe — data
  lost, nothing fabricated — and `CENSOR-04` is open, but the first stage that
  is genuinely inapplicable to some candidates will hit this.
- **The contradiction guard is suppressible by one censored record.** It fires
  only on complete candidate evidence, because a censored sibling could have
  been the one that agreed with the execution. A backend that censors liberally
  never trips it. The reasoning is sound; the property is worth knowing.
- **A record set where no execution carries an options digest can never pass
  `require_single_configuration`.** Deliberate — an unrecorded digest is not
  evidence that the options matched — but it means any bundle persisted before
  that key existed is permanently un-poolable until re-emitted. Know this
  before writing data.
- **The single highest-value test to write next is not another invariant.** It
  is a fixture set of deliberately corrupted FairShip output — zero entries, a
  missing branch, a truncated file, a job killed mid-write — asserting which
  branch the adapter takes for each. Everything in this boundary funnels into
  one judgement per run: *did this exit-0 run genuinely reconstruct nothing, or
  could its output not be read?* The checks can verify that an adapter cited
  **an** observation while asserting an answer; they can never verify the
  answer. Both reference backends assert it from a count they manufactured, so
  the shape is modelled and the hard part is exercised nowhere.

## 4. What the next mission must not do

- Modify `ShipSoft/FairShip` or perform any remote write against it
  (`AGENT_POLICY.md` §3).
- Weaken `tests/test_architecture_boundaries.py` to let an import through.
- Add a weight or `final_weight` field anywhere.
- Let a technical failure produce a decision value of any kind.
- Use `FakeFairShipBackend` output as physics evidence — it is a hash, and
  `is_physical` is `False` on both the backend and every bundle it emits.
- Assume the boundary checks make an adapter correct. They are structural. A
  backend that reports a crashed run as `SUCCEEDED`, emits one candidate for
  what were physically many, or attributes a run to the wrong subject passes all
  of them. The adapter needs its own tests against known FairShip output.
