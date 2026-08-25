# Audit — FairShip evaluation boundary, v1

**Status of this document:** investigation record written before implementation
and corrected against what the repository actually contained. It states what was
verified, what was assumed, and what stayed open.

Base commit: `2965ddd` (`fix(tagging): harden semantic decision identity`).
Working branch: `feat/fairship-evaluation-adapter-v1`.

---

## 1. The stated base branch was wrong — [VERIFIED]

The mission specified `experiment/d9-direct-utility-sampling-v0` as the base and
told me not to trust that assumption. It does not hold:

```console
$ git merge-base --is-ancestor origin/experiment/d9-direct-utility-sampling-v0 HEAD
NOT an ancestor of HEAD
$ git ls-tree -r --name-only origin/experiment/d9-direct-utility-sampling-v0 \
      -- src/ship_muon_bg/entities src/ship_muon_bg/tagging
(empty)
```

That branch carries neither `entities/` nor `tagging/` — the two packages this
mission builds on. Building there would have meant reimplementing the canonical
entity layer from scratch and diverging from the version already reviewed and
tested on `feat/adapter`.

**Decision:** branch from `feat/adapter` at `2965ddd`. Recorded here rather than
silently substituted, because it changes what "the mission base" means in every
later report.

`research/u-definition-fairship-audit` was read for evidence and not built on,
per the mission's instruction.

## 2. What already existed — [VERIFIED]

The mission's plan assumed Units A and C were absent. Half of that was wrong.
Migration steps 1 and 2 of `scientific_architecture_v2.md` §11 were already
implemented and tested:

| Component | State at `2965ddd` |
| --- | --- |
| `entities/subject.py` — `TagSubject` | implemented, contract-fixed at 3 fields (`SUBJ-01`) |
| `entities/lineage.py` — execution/realization/candidate | implemented, `0..N` throughout |
| `entities/observation.py` — `ObservationEnvelope` + typed payloads | implemented |
| `entities/decision.py` — `StageDecision` | implemented, censoring enforced in `__post_init__` |
| `entities/identifiers.py` — content-addressed ids | implemented |
| `tagging/definitions.py`, `tagging/evaluator.py` | implemented, generic, non-physical fixture only |
| `tests/test_architecture_boundaries.py` — AST import guard | implemented |
| Evaluation request / backend protocol / bundle | **absent** |
| Any fake or stub backend | **absent** (`git log --all --diff-filter=A` finds none) |
| State-level aggregation / `eta_hat` | **absent** |
| `docs/investigations/`, `docs/handoffs/` | **absent** (created by this mission) |

So the work reduced to: extend the entities where the boundary needed more,
build the evaluation API, build two backends, build aggregation, and test all of
it. Nothing in `entities/` or `tagging/` was rebuilt.

## 3. Where the new code had to live — [PROJECT DECISION]

`tests/test_architecture_boundaries.py:33` declares the backend-independent
package roots it sweeps:

```python
BACKEND_INDEPENDENT_PACKAGES = ("entities", "data_contracts", "simulation", "tagging")
```

Placing the evaluation API and the fake backend in `simulation/`, and the
aggregation layer in `tagging/`, means the existing AST guard already enforces
their backend-independence — no new enforcement mechanism, and no way to add one
of these modules later without it being swept.

`evaluation/` was rejected as a home. `scientific_architecture_v2.md` §2 reserves
it for campaign orchestration, which is explicitly *allowed* to import
`adapters/fairship`. Putting the canonical API there would have removed exactly
the guarantee the mission exists to establish.

The §2 table also constrains the dependency direction: `simulation/` and
`tagging/` may import `entities/` **only**. Two consequences followed:

- the fake backend may not import `tagging/`, so it receives stage identity as
  an opaque `stage_definition_id` string — which turns out to be the honest
  model anyway, since a real adapter reports the outcome of a rule it did not
  define and cannot re-derive;
- the aggregation layer may not import `simulation/`, so it consumes plain
  tuples of canonical entities rather than an `EvaluationBundle`. It therefore
  has no idea a backend exists at all.

## 4. Two deviations from the mission's literal field lists — [PROJECT DECISION]

**Candidate "reconstruction status" is a `StageDecision`, not a field.**
`DECISION-01` already names `"reconstruction_quality"` as a stage. A status field
on `ReconstructedCandidate` would recreate the `OutcomeCategory` conflation the
entity layer was built to remove: one enum mixing run health with physics
interpretation.

**Physical weight and kinematics are `ObservationEnvelope`s, not request or
subject fields.** This keeps `TagSubject` at its contract-fixed three fields
(`SUBJ-01`) and makes `WEIGHT-01` structural rather than conventional: a
production weight and a utility multiplier are separate records with separate
`observation_definition_id`s, so there is no field for a product of factors to
be written into. Confirmed by test: `EvaluationRequest` and `EvaluationBundle`
carry no field whose name contains `weight`, `momentum`, or `energy`.

## 5. Scientific authority — [VERIFIED]

`SHIP-EMPIRICAL-FS-SIM-CONTRACT-V0-SOURCE` is **not present locally**. Per
mission §2 the implementation basis is therefore the repository's own contracts,
principally `docs/contracts/tagging_contract_v0.md`. Every clause cited in the
new code refers to that file.

## 6. Evidence recovered from the audit branch — [VERIFIED]

Recovered without building on `research/u-definition-fairship-audit`:

- SBT, UBT, and DOCA appear nowhere in the repository's code, config, or data.
  No SBT/UBT semantics were assumed or encoded.
- The committed samples are unlabeled `(N, 8)` kinematics-plus-weight arrays.
  There are no physics labels anywhere in the repository today.
- `artifacts/toy_campaign/` fields are toy outputs, not physics labels.

This is why every stage used in tests is the repository's explicitly
non-physical `fixture.scalar_above_threshold` rule, and why no historical SHiP
cut appears in the new code.

## 7. Open questions this mission did not resolve — [OPEN]

These stayed open by design. None of them blocked the boundary, and each is
designed around rather than guessed at.

| # | Question | How the code avoids deciding it |
| --- | --- | --- |
| OPEN-A | The exact FairShip source-state surface | `TagSubject` stays at three fields; coordinates live in observations keyed by a versioned `state_definition_id` |
| OPEN-B | Real DIS execution cardinality (1→1 or 1→N) | `InteractionRealization` is `0..N` per execution and interaction-neutral; nothing assumes a count |
| OPEN-C | Endpoint `B` | no stage chain is encoded; stages are independent by construction |
| OPEN-D | Physical weight propagation | no weight is applied anywhere in the new code; `w_i` is an observation like any other |
| OPEN-E | SBT/UBT semantics | absent from the repository; nothing was invented |
| OPEN-F | Current geometry | reaches the backend only through `options`, whose digest is content-addressed so drift under one configuration label is at least *detectable* (`assert_requests_compatible`) |
| OPEN-G | Whether `subject_id` should be content-addressed over state coordinates | the request rejects a repeated `subject_id`, which closes the accidental case; minting one physical state under two ids remains the request builder's obligation. Two distinct muons can share rounded coordinates, so content-addressing is not obviously correct either |
| OPEN-H | The correct uncertainty treatment for repeated conditional simulations of one state | `eta_hat` is documented as an observed frequency with no independence claim and no standard error; raw counts are retained so a different method can be applied later without re-running anything |

## 8. Pre-existing environment failures — [VERIFIED]

Recorded so mission-caused regressions stay distinguishable.

```console
$ .venv/bin/python -m pytest tests/ -q --continue-on-collection-errors   # at 2965ddd
15 failed, 567 passed, 52 skipped, 13 errors
```

All 15 failures and all 13 collection errors are in `tests/afterms/` and are
caused by missing optional stacks (`sklearn`, `scipy`, `torch`) in this `.venv`.
They are not code defects, and no dependency was changed to make the global
number green.
