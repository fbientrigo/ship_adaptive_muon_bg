# ML Skeleton for Local Post-Shield Muon PKL (v0)

Status: **planning document only — not implemented.**

This document specifies the *next agile increment* after the `fairship_adapter`
contract: a **minimal, testable ML skeleton** that can ingest local post-muon-shield
muon PKL data and prepare a clean separation between a density/proposal track and a
proxy-score `U(x)` track. It defines scope, package boundaries, the local data
ingestion contract, per-track first milestones, artifacts, tests, and the
recommended next branch.

No code in this commit trains a model, claims physics performance, imports
FairShip or ROOT, or estimates SHiP background rates. This is a plan, not an ML
implementation.

The terminology in this document is fixed and must not be renamed:

- `simulator`
- `simulation_backend`
- `FairShip base simulator`
- `fairship_adapter`
- `fairship_simulator`
- `toy_simulator`

The terms `density model`, `proposal model`, `proxy`, `U(x)`, and `acquisition`
refer to ML components and are **not** simulator terms. They never run physics;
they only propose or rank candidates that a `simulation_backend` later evaluates.

---

## 1. Scope

### What this ML skeleton **will** do

- Ingest local historical muon PKL files (gzip-compressed pickle of a NumPy
  array, shape `(N, 8)`) under a single, validated data contract.
- Validate shape, columns, units sanity, and finiteness before any modelling.
- Compute a deterministic `dataset_hash` and a reproducible, seeded train/validation
  split recorded as a manifest.
- Persist feature-normalization metadata as a standalone artifact (no hidden
  global state, no re-fitting at inference time).
- Provide the **first milestone** of a density/proposal track: load → validate →
  split → train a deliberately tiny model (or placeholder baseline) → **prove it
  can overfit a tiny fixture** → save metrics and artifacts.
- Provide a **proxy `U(x)` contract** and a dummy baseline scorer (no trained
  proxy) so the acquisition interface can be exercised end to end.
- Provide **baseline acquisition policies** (random, top-k, diversity-aware top-k)
  that operate on scores without requiring a trained proxy.
- Define the JSON/NPZ artifact schemas that later increments will fill.
- Ship the guard-rail tests (fixtures, shape/units failures, deterministic split,
  tiny overfit smoke test, score schema, no-FairShip/ROOT-import check).

### What this ML skeleton will **not** do

- **No physics claim.** Nothing here estimates background rates, veto survival,
  or DIS yields. Marginal agreement is not validity.
- **No DIS/event-level conservation.** These are post-shield muon states. Energy
  and event-level conservation belong to the downstream `simulation_backend`,
  never to this skeleton.
- **No FairShip / ROOT.** Core ML modules import neither. The only path to physics
  is through the `simulation_backend` boundary defined by the `fairship_adapter`
  contract.
- **No production Normalizing Flow** in this increment (see
  [§5](#5-density--proposal-model-track) and
  [§11](#11-what-not-to-implement-yet)).
- **No trained proxy.** Labels do not yet exist; only a proxy *contract* and a
  dummy baseline are in scope.
- **No uncontrolled extrapolation.** The proposal must stay inside the observed
  post-shield support unless a controlled-extrapolation study is explicitly
  requested and labelled as such.

---

## 2. Proposed package boundaries

The README plans `src/ship_muon_bg/{data,proxy,flows,acquisition,...}`. This
increment should **introduce only the packages that have working, tested code in
the same commit that creates them.** Do not reintroduce empty packages — an empty
package is a maintenance liability and a false promise of capability.

Each package lands *with* its tests, and the packages land in **separate commits**,
one track at a time. The first implementation commit is **`data_contracts` only** —
it does **not** introduce `flows/`, `acquisition/`, or `proxy/`.

| Package | Introduce in | First responsibility |
| --- | --- | --- |
| `src/ship_muon_bg/data_contracts/` | **Commit 1 (this increment)** | PKL load, schema/units/finite validation, `dataset_hash`, split manifest, normalization metadata. **Nothing else ships in this commit.** |
| `src/ship_muon_bg/flows/` | **Commit 2** | Density/proposal **placeholder** + proposal diagnostics: tiny baseline + tiny-overfit smoke test, metrics/artifact writers. Keep the model behind a small interface so a real flow can drop in later. |
| `src/ship_muon_bg/acquisition/` | **Commit 3** | Score-driven baseline policies (random, top-k, diversity-aware top-k). Depends only on a score vector, not on a trained proxy. |
| `src/ship_muon_bg/proxy/` | **Commit 3 (dummy only); trained proxy deferred** | Dummy `U(x)` baseline + score schema in Commit 3. A **trained, calibrated proxy is deferred until labels exist from the `simulation_backend`** (see §6). |

### Implementation sequence (strict)

1. **Commit 1 — `data_contracts` only.** PKL ingestion contract, validation,
   `dataset_hash`, deterministic split manifest, train-only normalization metadata,
   the data-contract tests, and the thin `scripts/build_dataset_report.py`. No
   `flows/`, no `acquisition/`, no `proxy/`, no modelling code, no new dependencies.
2. **Commit 2 — density/proposal placeholder + proposal diagnostics.** Introduce
   `flows/` with a deliberately tiny placeholder density/proposal model, the
   tiny-overfit smoke test, and `proposal_diagnostics.json` (out-of-support rate,
   diversity, ESS-style diagnostic).
3. **Commit 3 — dummy proxy + acquisition baselines.** Introduce `proxy/` with a
   **dummy** `U(x)` baseline and the score schema, and `acquisition/` with the
   random / top-k / diversity-aware top-k baselines.
4. **Real trained proxy — only after labels exist** from the `simulation_backend`
   (today `toy_simulator`; eventually the `FairShip base simulator` via the
   `fairship_adapter`). No labels, no trained proxy.
5. **Real Normalizing Flow — only after** the placeholder density/proposal path and
   its diagnostics are green. The placeholder must prove the plumbing first; a real
   NF replaces it behind the same interface, not before.

Notes:

- `data_contracts` (not bare `data`) is the recommended name: it signals that the
  package's job is *contract enforcement and provenance*, consistent with the
  README's "treat every input dataset as suspect" principle and the existing
  `docs/contracts/` framing.
- The legacy `utils/` modules (`flow_models.py`, `data_handling.py`) stay where
  they are for now. They are **untested** and were written for HDF5 mother/daughter
  data, not the `(N, 8)` PKL contract. Do not wire them into the new core without
  a test first. The RealNVP in `utils/flow_models.py` is a *candidate* to promote
  into `flows/` **after** it has a tiny-overfit test and is adapted to the PKL
  feature layout — not before.
- `scripts/` holds thin CLI entry points (e.g. `scripts/build_dataset_report.py`)
  that call into `src/`. No business logic in scripts.

---

## 3. Minimal local PKL ingestion contract

### Accepted input format

- A **gzip-compressed pickle** file (legacy convention) whose payload is a single
  `numpy.ndarray`.
- Array shape: `(N, 8)`, `dtype` floating point, `N >= 1`.
- Loader must open via `gzip` + `pickle` and reject anything that does not
  deserialize to a 2-D float array with exactly 8 columns.
- **Security note:** `pickle` executes arbitrary code on load. The loader must
  document that PKL inputs are trusted local legacy files only, and a future
  increment should migrate the canonical on-disk format to NPZ (see
  [§7](#7-datametadata-storage)). Never unpickle untrusted/remote files.

### Required columns (fixed order)

| Index | Name | Unit | Meaning |
| --- | --- | --- | --- |
| 0 | `px` | GeV/c | momentum x |
| 1 | `py` | GeV/c | momentum y |
| 2 | `pz` | GeV/c | momentum z |
| 3 | `x` | m | position x |
| 4 | `y` | m | position y |
| 5 | `z` | m | position z |
| 6 | `id` | PDG code (int-valued) | particle id |
| 7 | `w` | dimensionless | event weight |

These are **post-shield muon states**. The contract records this fact as metadata;
it does **not** impose any inter-row or event-level energy/momentum conservation.

### Validation checks (fail fast, fail loud)

1. **Shape**: array is 2-D with exactly 8 columns; reject otherwise.
2. **Finiteness**: no `NaN` / `inf` in any column; reject otherwise.
3. **Weights**: `w` finite and `w > 0` (or `>= 0` per a documented policy); reject
   non-positive/`NaN` weights.
4. **PDG id**: `id` is integer-valued; warn if values are outside the expected
   muon set (`±13`) rather than silently accepting — record the observed id
   histogram in the report.
5. **Units sanity (bounds, not physics)**: configurable plausibility bounds per
   column (e.g. `|p| < p_max`, `|pos| < pos_max`) to catch unit mistakes
   (mm vs m, MeV vs GeV). These are *contract* bounds, **not** physics validity.
6. **Support audit**: record per-column min/max/quantiles so downstream proposal
   diversity and out-of-support checks have a reference envelope.

Validation failures raise typed errors and are surfaced in `dataset_report.json`;
they never get silently coerced.

### `dataset_hash`

- A deterministic content hash (e.g. SHA-256) over the raw decompressed array
  bytes **in a canonical layout** (C-contiguous, fixed dtype), plus the column
  schema string and the contract version.
- Recorded in every downstream artifact so any metric can be traced to its exact
  input. Two files with identical content hash identically; reordered rows hash
  differently by design (row order is part of provenance here).

### Split manifest

- Deterministic, seed-controlled `train` / `validation` split over **row indices**
  (no leakage; the split is on indices, recorded explicitly, not re-derived).
- `split_manifest.json` records: `seed`, `dataset_hash`, split fractions, the
  exact index lists (or a reproducible index-generation rule for large `N`), and
  the split strategy name.
- Optionally stratify by `id` so rare ids are represented in both splits.

### Feature-normalization metadata

- Normalization is **fit on train only**, then applied to validation — never fit
  on the full set (leakage).
- Persisted to `normalization.json`: method name, per-feature parameters
  (e.g. mean/std, or quantile-transform knots), the feature order, and the
  `dataset_hash` / split it was fit on.
- Default first method: simple standardization (per-feature mean/std) — minimal,
  invertible, and dependency-free. A `QuantileTransformer` (already used in
  `utils/`) is an option **later**, but its fitted state must be serialized as
  explicit metadata, not pickled silently.

---

## 4. Why a separate density track and proxy track

The two tracks answer different questions and must not be conflated:

- **Density / proposal (`flows/`)**: *where in phase space do post-shield muons
  live, and can I propose new candidates from that support?* Trained
  unsupervised on `(px..z, w)`; no labels needed.
- **Proxy `U(x)` (`proxy/`)**: *given a candidate, how operationally dangerous /
  worth-simulating is it?* Requires labels that only the `simulation_backend`
  (eventually the `FairShip base simulator`) can produce.

Keeping them separate prevents the classic failure of letting a likelihood stand
in for operational value, and lets the acquisition layer combine them explicitly
later.

---

## 5. Density / proposal model track

**First milestone (this increment): prove the plumbing and the overfit, not the physics.**

Pipeline for the first milestone:

1. **Load** PKL via the §3 contract.
2. **Validate** shape and finite values; reject on failure.
3. **Split** deterministically (seed + manifest).
4. **Normalize** features (fit on train, persist metadata).
5. **Train a deliberately small model or placeholder baseline** on the kinematic
   columns (`px, py, pz`, optionally derived `energy`; positions optional in v0).
6. **Prove overfitting on a tiny fixture**: on a fixed ~32–256 row fixture, a tiny
   model must drive train loss / negative-log-likelihood down to a near-zero floor.
   This is the smoke test that the gradient path and `log_prob` are wired
   correctly — *not* a generalization claim.
7. **Save metrics and artifacts**: `flow_train_metrics.json`,
   `proposal_diagnostics.json`, `model_manifest.json`.

### Model choice for v0

Do **not** implement a production NF yet. Two acceptable v0 options, in order of
preference:

1. **Placeholder baseline density**: a per-feature Gaussian (or histogram /
   independent-quantile) sampler fit on train. It supports `log_prob` and
   `sample`, trains instantly on CPU, has zero new dependencies, and gives the
   acquisition + diagnostics layers something real to consume. It will *fail* the
   diversity/support diagnostics in interesting ways — which is exactly what we
   want the diagnostics to catch.
2. **Promote the existing RealNVP** in `utils/flow_models.py` into `flows/`
   **only if** it can stay minimal and gets a tiny-overfit test in the same
   commit. It is PyTorch-based, exposes `log_prob`, runs on CPU, and is already in
   the repo (no new dependency). It is currently **untested** and was written for
   a different (HDF5) feature layout, so promoting it is real work, not a free win.

Keep whichever choice behind a tiny interface:

```python
class DensityModel(Protocol):
    def fit(self, x_train) -> None: ...
    def log_prob(self, x) -> np.ndarray: ...
    def sample(self, n, *, seed) -> np.ndarray: ...
```

so a real conditional flow (e.g. `nflows`/`zuko`/`normflows`) can drop in later
without touching the acquisition or diagnostics layers.

### Proposal diagnostics (must exist even for the baseline)

- **Support coverage / out-of-support rate**: fraction of generated samples that
  fall outside the per-feature observed envelope from §3's support audit.
- **Diversity metric**: a duplicate/near-duplicate rate and a spread measure
  (e.g. mean nearest-neighbour distance in normalized space) to catch collapse.
- **ESS placeholder**: when a `log_prob` is available, record an effective-sample-size
  style diagnostic (importance-weight ESS against the empirical sample) so the
  "good marginals, bad ESS" failure mode is observable from the start.

---

## 6. Proxy `U(x)` track

**Labels do not exist yet. This increment ships a proxy *contract* and a dummy
baseline only — no trained proxy.**

### What `U(x)` is

`U(x)` is an **operational danger / worth-simulating score**, not a likelihood and
not a physics prediction. Higher `U(x)` means "more worth spending an expensive
`simulation_backend` call on this candidate."

### Labels required before any training

| Aspect | Requirement |
| --- | --- |
| **Label source** | Outcomes from the `simulation_backend` (today `toy_simulator`; eventually the `FairShip base simulator` via `fairship_adapter`). Labels are *produced*, never assumed from legacy files. |
| **Label semantics** | A binary/ordinal target derived from the `fairship_adapter` failure taxonomy: `accepted_candidate` vs `physics_rejection`. **`technical_failure` is not a label** — it must be excluded from training, never folded into the negative class. |
| **Target definition** | One of (to be fixed before training, not silently assumed): (a) veto survival, (b) DIS-candidate survival, or (c) another explicitly named operational target. v0 contract leaves this as an open field; the dummy baseline commits to none. |
| **Calibration** | If `U(x)` is used as a probability, it must be calibrated (reliability curve, e.g. isotonic/Platt) and the calibration check must be an artifact, not an afterthought. |
| **Tail accounting** | Primary metric is **tail false-negative rate** in the dangerous region (recall on rare positives), *after* support gates — not global accuracy/AUC. A proxy with great global metrics and poor tail recall is a failure (see §9). |

### Candidate approaches to evaluate when labels arrive (not now)

- **Calibrated binary classifier** (gradient-boosted trees or a small MLP) with
  isotonic/Platt calibration — the default if labels are reasonably balanced after
  enrichment.
- **Ranking model** (pairwise/listwise) if only *relative* danger ordering is
  needed for acquisition rather than absolute probabilities.
- **Positive-Unlabeled (PU) learning** if confirmed positives are scarce and the
  negative set is really "unlabeled" — likely the realistic regime early on.
- **Uncertainty-aware acquisition** (deep ensembles / MC-dropout / GP on a
  reduced feature set) — only *after* a calibrated point proxy exists.
- **Conformal prediction** for distribution-free coverage guarantees on the
  danger score, used as a calibration/abstention check.

### v0 deliverable

Only a **dummy baseline scorer** and the score schema:

- `DummyProxy` returns a fixed or trivially-derived score (e.g. constant, or a
  monotone function of `|p|`) purely so the acquisition interface and the
  `proxy_train_metrics.json` / score-artifact schemas can be exercised and tested.
- It is explicitly labelled non-physical and must never be reported as a result.

---

## 7. Acquisition track

Acquisition consumes a score vector (from the dummy proxy now, a calibrated proxy
later) plus optional candidate features for diversity. Baseline policies:

| Policy | Needs | Notes |
| --- | --- | --- |
| **Random baseline** | nothing but `n` and a seed | The control everything else must beat. Always reported. |
| **Top-k by score** | score vector | Greedy exploitation; prone to mode collapse / duplicated picks. |
| **Diversity-aware top-k** | score + features | Greedy top-k with a diversity penalty (e.g. facility-location / k-center / max-marginal-relevance in normalized feature space) to avoid clustering. |
| **Uncertainty-aware** | calibrated proxy + uncertainty | **Deferred.** Only meaningful once a calibrated `U(x)` with usable uncertainty exists. Not in v0. |

`acquisition_report.json` records the policy, selected indices, score summary, and
a diversity summary of the selection so collapse is visible.

---

## 8. Artifacts

All artifacts are minimal JSON (or NPZ for arrays) for now — see §7 storage
rationale. Each artifact embeds `dataset_hash`, `seed`, `git_commit`, and a
`schema_version` for provenance.

| Artifact | Produced by | Contents (v0) |
| --- | --- | --- |
| `dataset_report.json` | data_contracts | row count, per-column min/max/quantiles, id histogram, validation outcomes, contract version, `dataset_hash`. |
| `split_manifest.json` | data_contracts | seed, fractions, strategy, index lists / rule, `dataset_hash`. |
| `normalization.json` | data_contracts | method, per-feature params, feature order, fit-on split, `dataset_hash`. |
| `flow_train_metrics.json` | flows | train/val loss curve, final NLL, overfit-fixture result, model config, seed. |
| `proxy_train_metrics.json` | proxy | **dummy** placeholder schema only in v0 (no real metrics); future: tail FNR, calibration, recall-after-gate. |
| `proposal_diagnostics.json` | flows | out-of-support rate, diversity/duplicate rate, ESS-style diagnostic. |
| `acquisition_report.json` | acquisition | policy, selected indices, score summary, selection-diversity summary. |
| `model_manifest.json` | flows/proxy | model type, config, artifact paths, dependency versions, seed, `dataset_hash`, `git_commit`. |

---

## 9. Storage strategy (data / metadata)

**Minimal now, heavy later — and justify every step up.**

- **Metadata** (reports, manifests, metrics, normalization, model configs): **JSON**.
  Human-diffable, git-friendly, no dependency.
- **Arrays** (features, scores, predictions, split indices): **NPZ** (`numpy`).
  Already a core dependency, compact, no schema lock-in.
- **Canonical sample format going forward**: prefer **NPZ over the legacy gzip-PKL**
  for any *new* on-disk sample, because pickle is an arbitrary-code-execution risk
  and is fragile across versions. PKL ingestion stays supported read-only for
  legacy files.
- **Avoid `pandas` / `pyarrow`** in core for now. The data is a fixed-width
  `(N, 8)` numeric array; NumPy covers it. The README's planned `.parquet`
  campaign outputs are a *later* concern, justified only once: (a) heterogeneous
  columns/strings appear, (b) datasets exceed comfortable in-memory NumPy size, or
  (c) cross-tool columnar interchange is genuinely needed. Until one of those is
  true, a `pandas`/`pyarrow` dependency is unjustified weight.
- Large datasets stay **out of git** (README principle); only hashes, manifests,
  and small fixtures are committed.

---

## 10. ML validation philosophy

1. **Overfit first.** Before any regularization, prove a tiny model can drive
   train loss to a near-zero floor on a tiny fixed fixture. If it cannot overfit,
   the model/optimizer/`log_prob` path is broken — fix that before anything else.
2. **Then regularize.** Only after overfit is demonstrated do we add capacity
   limits / weight decay / early stopping and watch the train/val gap.
3. **Always track**: train/validation split (by recorded indices), seed,
   `dataset_hash`, and the full model config — in every metrics artifact. A metric
   without these four is not reproducible and not trusted.
4. **Diagnostics over eyeballing.** Marginal-histogram agreement is necessary but
   never sufficient (this is in the README already). Out-of-support rate, diversity,
   ESS, tail recall, and calibration are the real gates.

---

## 11. Failure modes to design against

| Failure mode | Detector in this skeleton |
| --- | --- |
| Flow fits marginals but has poor ESS | ESS-style diagnostic in `proposal_diagnostics.json`; do not trust marginal plots alone. |
| Proxy has good global metrics but poor tail recall | Primary proxy metric is tail FNR / recall-after-gate, **not** AUC/accuracy; `proxy_train_metrics.json` schema enforces tail fields. |
| Proposal collapses to duplicated / low-diversity points | Duplicate + nearest-neighbour-spread diversity metric; diversity-aware acquisition; reported in proposal + acquisition artifacts. |
| Generated states leave observed post-shield support | Out-of-support rate against the §3 support envelope; proposal stays support-constrained unless a labelled controlled-extrapolation study is requested. |
| Technical failure miscounted as physics | Inherited from `fairship_adapter` taxonomy: `technical_failure` is never a proxy label. |
| Silent leakage via normalization fit on all data | Normalization fit on **train only**, recorded in `normalization.json` with the split it was fit on. |

---

## 12. Tests before ML expansion

These guard rails must exist and pass **before** any model is scaled up. They need
no FairShip, no ROOT, no GPU, and no large data — only a tiny committed PKL fixture.

1. **PKL fixture loads correctly** — a small committed `(N, 8)` gzip-PKL fixture
   loads to the expected array and schema.
2. **Invalid shape fails** — a `(N, 7)` / `(N, 9)` / 1-D array raises the typed
   shape error.
3. **Invalid units / NaN / inf fail** — arrays with `NaN`, `inf`, non-positive
   weights, or out-of-bounds magnitudes raise typed validation errors.
4. **Deterministic split** — same seed + same `dataset_hash` ⇒ identical
   `split_manifest.json`; different seed ⇒ different split.
5. **Tiny overfit smoke test** — the v0 density model drives train loss below a
   fixed threshold on the tiny fixture within a small step budget.
6. **Score artifact schema** — `proxy_train_metrics.json` / score outputs validate
   against the documented schema (even for the dummy proxy).
7. **Proposal diversity metric exists** — `proposal_diagnostics.json` contains the
   out-of-support and diversity fields and they are computed, not stubbed.
8. **No FairShip / ROOT import in core** — a test asserting core `src/ship_muon_bg/`
   modules import neither FairShip nor ROOT (mirrors the `fairship_adapter`
   contract test).

---

## 13. What not to implement yet

- **No production Normalizing Flow.** No `nflows` / `zuko` / `normflows` /
  `FrEIA` dependency in this increment. The v0 density model is a placeholder
  baseline (or the existing RealNVP, only if it stays minimal and gets a test).
- **No trained proxy.** No labels exist; ship only the contract + dummy baseline.
- **No uncertainty-aware acquisition.** It is meaningless before a calibrated proxy.
- **No conditional flow / conditioning variables.** Defer until the unconditional
  baseline and diagnostics are solid and a conditioning need is concrete.
- **No `pandas` / `pyarrow` / `parquet`** in core (see §9).
- **No FairShip / ROOT / EOS paths.** Physics stays behind the `simulation_backend`
  boundary.
- **No GPU assumptions.** Everything in v0 must run on CPU in seconds.
- **No promotion of legacy `utils/` code without a test.** `flow_models.py` and
  `data_handling.py` stay untouched until adapted to the PKL contract with tests.
- **No physics-validity language anywhere.** No "realistic", "validated", or
  "background rate" claims attached to skeleton outputs.

---

## 14. Recommended branch and commit sequence

**Branch:** `feat/ml-proposal`.

**Commit 1 scope — `data_contracts` only (smallest shippable increment):**

1. Create `src/ship_muon_bg/data_contracts/` implementing: PKL load,
   shape/units/finite/positive-weight/integer-id validation, `dataset_hash`,
   deterministic split + manifest, train-only normalization metadata.
2. Add a tiny committed gzip-PKL fixture under `tests/fixtures/`.
3. Add `tests/` covering: valid load, invalid shape, NaN/inf, non-positive
   weights, non-integer `id`, deterministic split (same seed), different split
   (different seed), normalization fit on train only, and no-FairShip/ROOT-import.
4. Add `scripts/build_dataset_report.py` that writes `dataset_report.json`,
   `split_manifest.json`, and `normalization.json` for a given PKL.
5. **Do not** introduce `flows/`, `acquisition/`, or `proxy/` in this commit.

**Commit 2** — density/proposal placeholder + proposal diagnostics (`flows/`).
**Commit 3** — dummy proxy + acquisition baselines (`proxy/` dummy, `acquisition/`).
Trained proxy and real NF follow the deferral rules in §2.

This keeps the first increment minimal, fully tested, dependency-light, and free
of any physics or FairShip claim — exactly the boundary discipline established by
the `fairship_adapter` contract.
