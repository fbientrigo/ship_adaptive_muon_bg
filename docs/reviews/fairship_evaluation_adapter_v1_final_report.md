# Mission report — the FairShip evaluation boundary, v1

## FINAL VERDICT

**Complete, reviewed, and green.** Units A–D are implemented, three adversarial
review rounds are closed, and the whole pipeline runs with no ROOT, no FairShip,
no CERN account and no LXPLUS.

One scientific question is deliberately left open and is the sole qualifier on
the substitution verdict below. It is refused loudly by the code rather than
guessed at.

## Provenance

| | |
| --- | --- |
| Starting branch | `feat/adapter` |
| Base commit | `2965ddd` — *fix(tagging): harden semantic decision identity* |
| Working branch | `feat/fairship-evaluation-adapter-v1` |
| Final commit | `5db0dbd` |
| Commits | 13 |
| Diff | 22 files, +6652 / −20 |
| Pushed | **no** — `AGENT_POLICY.md` §4 requires an explicit per-turn request |

The mission named `experiment/d9-direct-utility-sampling-v0` as the base and
said not to trust that. It does not hold: that branch is not an ancestor of
`HEAD` and contains neither `entities/` nor `tagging/`. Recorded in
`docs/investigations/fairship_evaluation_adapter_v1_audit.md` §1.

## Units completed

| Unit | Delivered |
| --- | --- |
| **A** — canonical evaluation API | `simulation/evaluation.py`: `EvaluationRequest`, `EvaluationBundle`, `EvaluationBackend`, `verify_evaluation_bundle`, `evaluate_verified`, `execution_shortfall`, `assert_requests_compatible`. Additive extensions to `entities/lineage.py`; `stage_decision_id` extracted into `entities/decision.py`. |
| **B** — fake backend | `simulation/fake_fairship.py` (`FakeFairShipBackend`) and `simulation/stub_backend.py` (`MinimalStubBackend`) — two implementations sharing no code path. |
| **C** — aggregation / tagging bridge | `tagging/aggregation.py`: `classify_executions`, `aggregate_state_stage_evaluations`, `StateStageAggregate`, `TaggingDataset`, five-way `ExecutionStageOutcome`, content-addressed `ROLLUP_RULE`. |
| **D** — contract tests | `tests/evaluation/` — 156 tests across five files plus shared fixtures. |
| Docs | D1 audit, D7 architecture, D8 handoff, and this report. |

Units A and C were **not** absent as the mission assumed. Migration steps 1–2 of
`scientific_architecture_v2.md` §11 were already implemented and tested on
`feat/adapter`; nothing in `entities/` or `tagging/` was rebuilt.

## Files changed

```text
src/ship_muon_bg/entities/{__init__,decision,lineage,observation}.py
src/ship_muon_bg/simulation/{__init__,evaluation,fake_fairship,stub_backend}.py
src/ship_muon_bg/tagging/{__init__,aggregation,evaluator}.py
tests/evaluation/{__init__,fixtures,test_aggregation,test_backend_substitution,
                  test_decision_identity,test_evaluation_boundary,test_fake_backend}.py
docs/investigations/fairship_evaluation_adapter_v1_audit.md
docs/architecture/fairship_evaluation_adapter_v1.md
docs/architecture/scientific_architecture_v2.md
docs/handoffs/real_fairship_adapter_next.md
docs/reviews/fairship_evaluation_adapter_v1_final_report.md
```

`simulation/types.py` and `simulation/backend.py` are **untouched**:
`SimulationBackend`, `SimulationResult`, `OutcomeCategory` and
`FlowProposalRecord` are byte-identical to `2965ddd`.

## Tests

Command:

```console
$ .venv/bin/python -m pytest tests/evaluation -q -p no:cacheprovider
156 passed
```

| | at `2965ddd` | at `5db0dbd` |
| --- | --- | --- |
| Full suite | 15 failed, 567 passed, 52 skipped, 13 errors | 15 failed, 723 passed, 52 skipped, 13 errors |
| Excluding `tests/afterms` | 563 passed | 612 passed, 27 skipped |
| `tests/evaluation` | — | 156 passed |

**No regressions.** The 15 failures and 13 collection errors are identical to the
baseline, all in `tests/afterms/`, all caused by `sklearn`/`scipy`/`torch` being
absent from this `.venv`. No dependency was changed to make the global number
green (mission §12: *pre-existing environment failure*, not mission regression).

## Review loop

Three rounds against two independent reviewers, run in separate model contexts
from the implementation.

| Round | Senior reviewer | Red-team | Outcome |
| --- | --- | --- | --- |
| 1 | 9 IMPORTANT | — | all fixed, each with a test |
| 2 | 2 BLOCKER, 4 IMPORTANT, 7 OPTIONAL | 2 BLOCKER, 6 IMPORTANT, 3 OPTIONAL | all fixed |
| 3 | 1 IMPORTANT, 2 OPTIONAL | 2 IMPORTANT, 3 OPTIONAL | all fixed or recorded |

Both reviewers independently found the same defect twice — round 2's
realization-scoped decision drop, and round 3's request-side evidence bypass.
Independent convergence on the same two holes is the strongest signal in this
report that the review was doing real work.

**Findings rejected:** one. The senior reviewer's `INVALID` on
`simulation/fake_fairship.py` having *fairship* in its filename — the boundary
guard matches dotted import segments, `fake_fairship` is not `fairship`, and the
module imports nothing FairShip-related. Recorded rather than renamed.

**Findings recorded rather than fixed:** five, all in
`docs/handoffs/real_fairship_adapter_next.md` §3a — an inapplicable candidate
discarding its run's negative (`CENSOR-04` is open), the contradiction guard
being suppressible by one censored record (sound reasoning, worth knowing), the
evidence check being liveness-and-lineage rather than relevance (closing it
crosses a dependency direction), opaque-boolean payload typing, and record
volume.

### What review actually caught

Three routes from a technical problem to a confidently wrong `eta_hat`. None
would have crashed; all would have produced a plausible number.

1. **A stage decision attached to an `InteractionRealization` was silently
   dropped**, turning an `EVALUATED` positive into a counted physics negative
   with `not_evaluated_count = 0`. `EXEC-02a` names the realization as the
   canonical lineage node — exactly where a real DIS selection attaches.
2. **An `EVALUATED` decision could rest on evidence that was never computed.**
   `StageEvaluator` already refused this; the boundary did not, and the
   unchecked path is the one a real adapter takes.
3. **An execution with zero candidates and no decision counted as a clean
   zero.** The commonest real partial failure is a job that exits 0 with empty
   or truncated reconstruction output, and from the records alone it is
   indistinguishable from a genuine zero.

The third is now a behaviour change, not a docstring: **a negative must be
attested, and the attestation must be falsifiable.** An `EVALUATED` decision
must cite at least one `COMPUTED` observation about the entity decided or one of
its ancestors. When the count could not be read, that observation is
`TECHNICALLY_UNAVAILABLE` and the boundary refuses any `EVALUATED` decision
resting on it. The loop closes.

That costs something, and it is on the record rather than hidden: an adapter
that never attests loses every zero-candidate run from the denominator, which
does not lose precision — it changes the estimand from `P(pass)` to
`P(pass | at least one candidate)`. `evaluable_fraction` and `reported_fraction`
are on every row so both that failure and its mirror (an adapter that *drops*
unreconstructable runs rather than reporting them) are visible in the table.

## Scientific invariants

| Invariant | How it is held |
| --- | --- |
| **Source-state semantics** | The unit of analysis is the execution; the unit of grouping is the source state. `EvaluationRequest` rejects a repeated `subject_id` and expresses repetition as `replications_per_subject`; `verify_evaluation_bundle` refuses more executions than authorized. **Limit:** this checks the *identifier*. One physical state minted under two ids is still two rows — `[OPEN]`, see below. |
| **Execution semantics** | `ExecutionStatus` is health-only. A `failure_reason` is rejected on a `SUCCEEDED` run, so run health cannot become an informal second outcome channel. |
| **Technical censoring** | Five mutually exclusive outcomes, arithmetically enforced to partition. A `TECHNICAL_FAILURE` execution may carry no realization, candidate, or evaluated decision. Explicit candidate-level censoring blocks an execution-level negative. Silence is `NOT_EVALUATED`. Nothing censored enters the `eta_hat` denominator, and `eta_hat` is `None` — never `0.0` — when nothing is evaluable. |
| **Candidate multiplicity** | Multiplicity is always the record count, never a stored number. `(execution_id, realization_id, candidate_index)` uniqueness blocks the one groupby key that collapses N to 1. Both count distributions survive aggregation, so `N` is recoverable and `Y = 1{N ≥ 1}` derivable from `N`. Where the two legitimately diverge — a veto-shaped stage positive with `N = 0` — `decision_level` records it. |
| **Stage versioning** | `stage_definition_id` is content-addressed; `stage_decision_id` lives in `entities/` so the tagging evaluator and any backend agree by construction. Decision ids are byte-identical to `2965ddd` across four rounds of edits. `ROLLUP_RULE` is content-addressed over its precedence written as data. Stages carry independent observables and imply nothing about each other. |
| **Configuration isolation** | Configuration id *and* options digest are both in the grouping key. `verify_evaluation_bundle` requires the digest on every execution; aggregation refuses a partially-recorded one; `require_single_configuration` and `require_single_state_definition` are the explicit pooling gates; `TaggingDataset` refuses duplicate rows for one state. |
| **Weight separation** | No weight field and no product of factors anywhere in the new code. `w_i` is an `ObservationEnvelope` like any other evidence, so `WEIGHT-01` is structural. `as_record()` deliberately emits no training weight. |

## BACKEND SUBSTITUTION VERDICT

> *Can the fake backend now be replaced by a future FairShip backend without
> modifying `ProxyTagger`, `Nflow`, or state-level evaluation metrics?*

## **PARTIALLY**

**Unqualified yes for `ProxyTagger`, `Nflow`, and every downstream metric.**
The qualifier is one named, deliberately-open scientific question inside the
aggregation layer.

**Evidence for what is established** (verified structurally by the reviewers,
not merely asserted by the tests):

- `tagging/aggregation.py` imports only `ship_muon_bg.entities.*`. It cannot
  reach an `EvaluationBundle`, a backend, or an adapter package. `ProxyTagger/`
  and `Nflow/` import no `simulation`, `adapters`, or `fairship` module. Backend
  identity is **unreachable** from the scientific layers, not merely unused —
  the strong form of the claim.
- One pipeline function drives two backends sharing no code path (sha256-derived
  vs. a fixed integer cycle), both exercising the full five-way censoring
  taxonomy, and reaches `DummyProxy` exactly as it already exists.
- `StageEvaluator` — which shares no code with the backends — independently
  re-derives every candidate-level decision from the emitted evidence and
  produces byte-identical records, including content-addressed ids.
- Every route found across three rounds from a technical problem to a counted
  physics negative is closed.
- 612 tests pass outside `tests/afterms` (563 at base); the boundary guard is
  green; the core contains no ROOT/FairShip/EOS/CVMFS/AFS/CERN reference.

**What would still have to change — the complete list:**

1. **`tagging/aggregation.py`, if and only if the real adapter attaches stage
   decisions to `InteractionRealization`.** `EXEC-02a` names the realization as
   the canonical lineage node between an execution and its candidates, which is
   where a DIS selection naturally sits. Such a decision is today **refused with
   an explicit error**, never silently mishandled. Defining its rollup semantics
   is handoff target **T4a**. Refusing was the right call: inventing `any()`
   semantics at the realization level would violate `TARGET-03`.
2. **Adapter-side obligations, requiring no core change** — all four documented
   in the `EvaluationBackend` protocol and handoff T4: record the options digest
   on every execution; attest empty runs three ways; make every `EVALUATED`
   decision cite a `COMPUTED` observation from its own lineage; check
   `evaluable_fraction` on the first real table.
3. **Nothing in `ProxyTagger/`, `Nflow/`, or any downstream metric.**

If the adapter attaches decisions at execution and candidate level only, a real
FairShip backend drops in today with no change to the aggregation either. That
path is documented, tested, and demonstrated by two independent backends.

## Open scientific questions

All stayed open. Each is designed *around*, never guessed.

| | Question |
| --- | --- |
| OPEN-A | The exact FairShip source-state surface |
| OPEN-B | Real DIS execution cardinality (1→1 or 1→N) |
| OPEN-C | Endpoint `B` and the `G → D → R → S → V → B` chain |
| OPEN-D | Physical weight propagation |
| OPEN-E | SBT/UBT semantics — absent from this repository entirely |
| OPEN-F | Current geometry |
| OPEN-G | Whether `subject_id` should be content-addressed over state coordinates |
| OPEN-H | The uncertainty treatment for repeated conditional simulations of one state |
| OPEN-I | Rollup semantics for realization-scoped stage decisions (new; handoff T4a) |
| `CENSOR-04` | Whether `NOT_EVALUATED` belongs in any denominator |

`SHIP-EMPIRICAL-FS-SIM-CONTRACT-V0-SOURCE` is not present locally, so per
mission §2 the implementation basis is this repository's own contracts.

## Pre-existing failures

15 failures and 13 collection errors, all in `tests/afterms/`, all from
`sklearn`/`scipy`/`torch` missing in this `.venv`. Identical before and after.
Not code defects; nothing was installed or changed to hide them.

## Technical debt introduced

- **The evidence check is liveness-and-lineage, not relevance.** Nothing
  verifies that a cited observation measures anything to do with the stage.
  Closing it means passing `StageDefinition` objects into aggregation instead of
  id strings, which crosses the `tagging → entities`-only dependency direction.
- **`ROLLUP_RULE_CONTENT` is prose kept in sync with `_classify_one` by hand.**
  Content-addressing makes a *changed* rule mint a new id; it cannot detect code
  changed without the text.
- **`DecisionEvaluationStatus` has no `NOT_APPLICABLE`**, so `CENSOR-02`'s
  not-applicable case rides `NOT_EVALUATED` and one inapplicable candidate
  discards its run's negative. Fails safe; blocked on `CENSOR-04`.
- **`ExecutionStatus` has exactly two values.** Timeouts, OOM kills and partial
  output all have to be squeezed into `TECHNICAL_FAILURE`.
- **`as_record()`'s two distributions are `int`-keyed dicts** — they survive
  `json.dumps` with string keys and have no flat CSV/Parquet column. Whatever
  persists the table is where multiplicity is most likely to be dropped.
- **A digest-less legacy bundle can never pass `require_single_configuration`.**
  Deliberate, but it means anything persisted before that key existed is
  un-poolable until re-emitted.

## Next mission

`docs/handoffs/real_fairship_adapter_next.md` — investigation targets T1–T10,
nothing implemented.

The single highest-value thing to build next is **not** another invariant. It is
a fixture set of deliberately corrupted FairShip output — zero entries, a
missing branch, a truncated file, a job killed mid-write — asserting which
branch the adapter takes for each.

Everything in this boundary funnels into one judgement per run: *did this exit-0
run genuinely reconstruct nothing, or could its output not be read?* Every check
here can verify that an adapter cited **an** observation while asserting an
answer. None can verify the answer. Both reference backends assert it from a
count they manufactured, so the shape is modelled and the hard part is exercised
nowhere.

The adapter that produces a wrong thesis number will not be one that violates
this contract. It will be one that satisfies every clause while its
`TECHNICALLY_UNAVAILABLE` branch is unreachable, because the ROOT reader raises
before it gets there — or quietly returns zero entries for a truncated file.
