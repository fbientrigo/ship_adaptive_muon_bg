# U(x) Definition & FairShip Audit

Status: **investigation / audit report — not a specification, not a result.**
Branch: `research/u-definition-fairship-audit`.
Scope: read-only repo/data audit. No models trained, no Normalizing Flow, no
FairShip adapter, no physics labels invented. All findings below are backed by
direct inspection of committed files, git history, and `pytest`; commands used
are given inline so every claim is reproducible.

## 1. Executive conclusion: **Conditional Go**

The three-module architecture (`Nflow` / `ProxyTagger` / `simulation` +
`data_contracts`) and the `fairship_adapter` contract are internally sound and
already encode the right caution (typed failure taxonomy, `is_physical` flags,
tail-FNR over accuracy, `technical_failure` excluded from labels). **Go** on
continuing to build the plumbing.

**No-Go** on claiming a physically meaningful `U(x)` today, in any form. No
committed dataset, artifact, or code path in this repository contains a
reconstruction, veto, or post-selection label. A real `U(x)` — defined as an
operational danger / worth-simulating score — requires labels that only a
`simulation_backend` (FairShip, or a toy stand-in) can produce, and no such
backend is implemented yet.

**Conditional**: a provisional `U_geom(x)` (kinematics/geometry only,
explicitly non-physical) and/or toy-plumbing work can proceed now, and one
small, low-risk documentation fix (§5, §8) should happen before the next
implementation branch, to stop the repo's own docs from contradicting
themselves about what `U(x)` means.

## 2. What exists in the repo

Verified by reading `README.md`, `pyproject.toml`,
`docs/architecture/repo_architecture_v1.md`,
`docs/architecture/ml_skeleton_local_pkl_v0.md`,
`docs/contracts/fairship_adapter_contract_v0.md`, `ProxyTagger/*.py`,
`Nflow/interfaces.py`, `src/ship_muon_bg/simulation/{types,backend}.py`,
`src/ship_muon_bg/data_contracts/*.py`, `FairShip/README.md`.

- **`src/ship_muon_bg/data_contracts/`** — the only fully-implemented, tested
  package. Fixed `(N,8)` schema (`px,py,pz,x,y,z,id,w`), loaders, hashing,
  subsampling, validation (shape/finite/weight-positivity/id-integer/unit
  bounds — explicitly *not* physics-validity checks). No `U(x)`, no labels.
- **`src/ship_muon_bg/simulation/`** — `types.py` defines the three-way
  outcome taxonomy `OutcomeCategory` (`TECHNICAL_FAILURE` /
  `PHYSICS_REJECTION` / `ACCEPTED_CANDIDATE`), `FlowProposalRecord`, and
  `SimulationResult` (with a `dis: Optional[bool]` field and a runtime
  invariant: a `TECHNICAL_FAILURE` must have `dis is None`). `backend.py`
  defines the `SimulationBackend` Protocol only — **no concrete
  implementation** (no `toy_simulator`, no real FairShip adapter) exists on
  this branch.
- **`ProxyTagger/`** — `interfaces.py` defines the `ProxyScorer` Protocol and
  the canonical `U(x)` contract docstring: *"an operational danger /
  worth-simulating score in [0,1] ... Labels come only from the
  `simulation_backend` ... `technical_failure` is never a training label ...
  primary metric is tail false-negative rate in the dangerous region, not
  global accuracy/AUC."* `baseline.py` ships exactly one concrete scorer,
  `DummyProxy`: `is_physical = False`, `fit()` is a no-op, `score()` is a
  monotone function of momentum magnitude (`|p|/(|p|+scale)`) — explicitly
  documented as never to be reported as a result.
- **`Nflow/interfaces.py`** — `DensityModel` and `BiasStrategy` Protocols
  only; the biasing mechanism is deliberately left open. `Nflow/legacy/` is
  quarantined HDF5-era code, exempted from the "no ROOT/FairShip import"
  guard test, not importable from tested core.
- **`docs/contracts/fairship_adapter_contract_v0.md`** — spec-only. Defines
  the failure taxonomy and the adapter's input/output contract fields, and
  lists (as *open questions*, not resolved): exact FairShip input format,
  exact post-shield coordinate convention, exact geometry tag/version, **the
  exact selection that separates `physics_rejection` from
  `accepted_candidate`**, exact SBT/UBT fields, exact output files, weight
  propagation convention.
- **`FairShip/`** — confirmed empty, gitignored slot (only `.gitignore` +
  `README.md`); no FairShip code present, as intended.
- **`data/samples/`** — committed ≤2 MB subsamples ("Muon NTuples v1.0") of
  two real MC releases hosted on GitHub Releases (`fbientrigo/NFlow`):
  `muons_FullMC_sample` (scoring plane, upstream of shield) and
  `muonsFullMC_afterMS_sample` (after Muon Shield). 40,000 rows each, seed
  1234, deterministic `uniform_core_plus_range_anchors` subsampling.
- **`artifacts/toy_campaign/`** — present on disk, gitignored, not tracked by
  git (see §4).

## 3. What labels/data are actually available — **none, physically**

Direct inspection (`python scripts/inspect_available_labels.py`, and
independent NumPy/pickle inspection of both `.npz` and `.pkl.gz` copies of
both samples):

- Both samples are a single array of shape `(40000, 8)`, dtype `float64`,
  fully finite (no NaN/Inf), columns strictly
  `[px, py, pz, x, y, z, id, w]`. `id ∈ {13, -13}` only (roughly balanced,
  ~50/50). One `w == 0` row per file (an intentional argmin anchor, not
  missing data).
- A grep across the whole repo (excluding `.venv/`) for
  `candidate|reco|reconstructed|veto|SBT|UBT|DOCA|IP|fiducial|wall|front|side|cavern|DIS|accepted|rejected|dangerous|rho|material|weight|label|target`
  finds these terms only in **documentation/interface/contract code**
  (describing what a future backend must eventually produce), never as an
  actual column, key, or field in any committed data file.
- **`SBT`, `UBT`, `DOCA` do not appear anywhere in project code, configs, or
  data** — only in the contract's prose description of what's still needed
  from FairShip.

**Direct statement, as the task requires**: the committed data samples contain
**no** reconstruction status, veto flag, DOCA/IP cut, fiducial-volume flag,
acceptance/rejection outcome, SBT/UBT activity, DIS tag, or utility/target
score of any kind. They are unlabeled kinematic + weight data only. This
matches the (already correct) conclusion printed by
`scripts/inspect_available_labels.py`, which is kept (see §9) as the
reproducible evidence for this claim.

`artifacts/toy_campaign/proposed_candidates.csv` and `simulator_results.csv`
do contain label-*shaped* fields: a `provenance` column whose value is the
literal constant string `"{'stage': 'post_shield', 'label': 'dangerous'}"` for
all 50 rows, and a `selection_stage_reached` column with values
`reconstruction` (45 rows) / `selection` (5 rows), plus a boolean
`useful_candidate`. **These are not physics labels.** `dataset_hash.json`
records `generator: "toy_synthetic"`; `config_resolved.yaml` records
`simulator.backend: toy`, `simulator.pass_probability: 0.1`,
`geometry_version: geo-toy-v0`. The "selection" a toy candidate "reaches" is
decided by a fixed probability draw, not by any geometry, detector, or veto
model. Using any of these fields as physical evidence — for training,
for a plot, or for a thesis claim — would be a labeling error.

## 4. Is `artifacts/toy_campaign/` reproducible from source? — **No, not on this branch**

- `artifacts/` and `__pycache__/`/`*.pyc` are both gitignored
  (`.gitignore`), so nothing under `artifacts/toy_campaign/` has ever been a
  git commit on any branch (`git log --all -- artifacts/toy_campaign` is
  empty).
- `scripts/run_toy_loop.py` — referenced by `README.md`'s "First Milestone"
  section (`python scripts/run_toy_loop.py --config
  configs/campaigns/toy_adaptive_v0.yaml`) — **does not exist on this
  branch**, nor does `configs/`. `git log --all --oneline -- scripts/run_toy_loop.py`
  shows it was added once, at commit `0024f23` ("feat(loop): implement first
  end-to-end toy adaptive campaign loop"), and modified at `7077186`, both
  **only on the local, unmerged, never-pushed branch `feat/loop-v0`**
  (`git merge-base --is-ancestor feat/loop-v0 HEAD` confirms it is not an
  ancestor of the current branch).
- `feat/loop-v0` diverged from the shared history very early and was
  effectively superseded — never merged — by the three-module redesign that
  landed via `feat/architecture` (the architecture actually present on
  `main`/this branch today). Its prototype modules
  (`src/ship_muon_bg/loop/`, `evaluation/`, `physics/`, its own
  `contracts/`) were never ported into the current architecture; the
  corresponding directories on this branch contain only stale `__pycache__/`
  and no `.py` source.
- Corroborating forensic detail: the stale `.pyc` files under
  `scripts/__pycache__/run_toy_loop.*.pyc`,
  `src/ship_muon_bg/loop/__pycache__/*.pyc`, etc., carry timestamps ~7
  seconds *before* the `artifacts/toy_campaign/*` files — consistent with
  someone locally checking out `feat/loop-v0`, running
  `scripts/run_toy_loop.py --config configs/campaigns/toy_adaptive_v0.yaml`,
  then switching to this research branch, which does not delete gitignored
  files.
- **Conclusion**: `artifacts/toy_campaign/` is orphaned output from a
  different, unmerged lineage. It cannot be regenerated by anything on this
  branch. To reproduce it exactly, one would need to
  `git checkout feat/loop-v0` and rerun the same command — which is out of
  scope here and not recommended, since that code was superseded by the
  current architecture and was never subjected to the same review as
  `data_contracts`/`ProxyTagger`.

## 5. Why `U(x)` cannot be physically obtained without FairShip/reco/selection labels

This is the core semantic issue this audit was asked to resolve, and it is a
**self-inconsistency already present in the repo's own docs**, not something
missing entirely:

- `README.md` (module 3 description) and
  `docs/architecture/repo_architecture_v1.md` (ASCII diagram + module table)
  describe `U(x)` as *"the continuous 0 (never DIS) → 1 (DIS always) boundary
  estimate"* — i.e., framed as **raw DIS-occurrence** on the post-shield
  state `x`.
- `docs/architecture/ml_skeleton_local_pkl_v0.md` §6 and
  `ProxyTagger/interfaces.py` define `U(x)` more strictly as an
  **operational danger / worth-simulating score**: *"Higher `U(x)` means
  'more worth spending an expensive `simulation_backend` call on this
  candidate.'"* This version explicitly derives its label from
  `physics_rejection` vs `accepted_candidate` — i.e. **post-selection**
  outcomes, not DIS occurrence — and explicitly flags the exact target
  (veto survival? DIS-candidate survival? something else?) as **still
  undecided** ("to be fixed before training, not silently assumed").

These two framings are not equivalent. DIS occurrence is a property of the
interaction physics at the post-shield state `x` (in principle predictable,
with enough physics knowledge, from `x` and the target material alone).
Post-selection danger additionally depends on geometry/acceptance, detector
response, reconstruction efficiency, and SBT/UBT veto logic downstream of
DIS occurring — none of which is a function of `x` alone. A DIS event that
happens outside the fiducial volume, or that a working veto correctly
tags, is not a dangerous background candidate; a non-DIS-labeled state might
still matter if the label itself is mis-defined. **Whether a post-shield
state `x` is "dangerous" in the analysis sense cannot be computed from `x`
and a DIS cross-section alone — it requires running the full downstream
chain (or a calibrated proxy for it) at least once per labeled example.**

Concretely, none of the following exist anywhere in this repo today:
a reconstruction efficiency model, a fiducial/acceptance geometry
implementation, an SBT/UBT simulation or veto decision, or any historical
FairShip output to learn from. `docs/contracts/fairship_adapter_contract_v0.md`
explicitly leaves "the exact selection that separates `physics_rejection`
from `accepted_candidate`" as an open question. Until that selection is
defined and executed by a real (or toy) `simulation_backend`, `U(x)` in the
strict, thesis-relevant sense has no labels to learn from, by construction —
this is not a data-availability accident, it is what "post-selection" means.

## 6. Is `U_geom(x)` acceptable as a provisional proxy? — Yes, with discipline

A **provisional** `U_geom(x)`, computed purely from `data/samples` kinematics
(e.g. a trajectory classification — front/side/cavern-style, by position and
angle at the scoring plane — or a distance-to-acceptance-box-edge heuristic),
is legitimate to define and even publish, **provided it follows the same
discipline the repo already applies to `DummyProxy`**:

- Ship it with `is_physical = False` and a name that cannot be mistaken for
  a physics result (e.g. `geom_trajectory_proxy`, not `U`).
- State explicitly, everywhere it's mentioned, that it approximates
  *geometric* plausibility only — it says nothing about reconstruction,
  veto survival, or the DIS cross-section, and must never be reported as an
  estimate of background rate or danger probability.
- Keep it behind the existing `ProxyScorer` Protocol so swapping in a real,
  FairShip-trained `U(x)` later is a drop-in replacement, not a rewrite.
- Do not train it on toy-campaign labels (§3) — those are synthetic and
  would silently launder a random-probability draw into something that
  looks like a geometry-informed decision.

This audit does **not** implement `U_geom(x)` (out of scope per task rules);
it only certifies the idea as sound and low-risk, and scopes it as the next
branch's first candidate feature (§8).

## 7. FairShip output schema needed to train a real `U(x)`

Compiled from `docs/contracts/fairship_adapter_contract_v0.md` plus the task's
required-columns list. "Known" = already specified/decided in this repo's
docs; "Unknown" = still open, per the contract's own open-questions section.

| Field | Purpose | Status |
|---|---|---|
| `candidate_id` / source muon id | Join key back to the proposed `x` | Known (`FlowProposalRecord.candidate_id`) |
| Original post-shield features `x` (`px,py,pz,x,y,z,id,w`) | The input being labeled | Known (`data_contracts/schema.py`) |
| Geometry tag / version | Which detector geometry produced this label | **Unknown** — exact tag/pinning mechanism open |
| Backend/run metadata | Reproducibility (backend version, seed, timing) | Known shape (`backend_run_manifest`, `backend_version_metadata`, `timing` in adapter contract); values TBD |
| Technical failure flag | Exclude untrustworthy runs from labels | Known (`OutcomeCategory.TECHNICAL_FAILURE`, enforced never a label) |
| Physics rejection / acceptance category | The actual training target | Known *category names*; **exact selection logic unknown** |
| DIS occurred / forced-DIS metadata | Distinguish natural vs biased-sampling DIS, if forced DIS is used | **Unknown** — standard vs forced DIS workflow open |
| Reconstructible flag | Whether the candidate could be reconstructed at all | **Unknown** — not in current contract |
| Fiducial flag | Inside acceptance/fiducial volume | **Unknown** — not in current contract |
| DOCA | Distance of closest approach (a common SHiP selection variable) | **Unknown** — not in current contract |
| IP | Impact parameter | **Unknown** — not in current contract |
| Front/side/cavern (or equivalent trajectory class) | Geometry region tag | **Unknown** — not in current contract |
| SBT activity / energy deposit | Surrounding Background Tagger veto input | **Unknown** — "exact SBT/UBT fields" explicitly open |
| UBT activity / decision | Upstream Background Tagger veto input | **Unknown** — same open item |
| Final post-selection accepted/dangerous flag | The actual `U(x)` label | **Unknown** — depends on all of the above being resolved first |
| Event/candidate weights + meaning | Correct statistical weighting of rare labels | Known that weights must propagate (`FlowProposalRecord.weight` → `SimulationResult`); **exact propagation convention open** |

Also explicitly unknown, as flagged in the existing contract doc and
unchanged by this audit: the exact official FairShip branch/script to run,
the exact ROOT trees/branches to parse, the exact geometry tag/version to
pin, whether standard or forced-DIS generation is used, and the weight
propagation convention. None of these can be guessed or assumed without
either running FairShip once and inspecting real output, or getting them
from someone who has.

## 8. Minimal next branch (not implemented here)

In order, each step unblocked by the previous one, matching the existing
roadmap already written in `docs/architecture/repo_architecture_v1.md`
(this audit did not invent a new plan, only confirmed and sequenced it):

1. **Doc fix (small, low-risk)**: edit `README.md`'s module-3 paragraph and
   `docs/architecture/repo_architecture_v1.md`'s diagram/table so both point
   to the stricter `ml_skeleton_local_pkl_v0.md` §6 / `ProxyTagger/interfaces.py`
   definition instead of repeating "DIS-boundary score" as if it were a
   synonym. This removes the internal contradiction found in §5 without
   touching any code contract.
2. **`toy_simulator` behind `SimulationBackend`**: a cheap, deterministic,
   clearly-synthetic backend implementing the three-way outcome taxonomy
   with an explicit, documented synthetic rule (not a hidden probability
   draw as in the orphaned `feat/loop-v0` prototype). This is the first
   thing that can produce labels at all, and unblocks a first trained
   (but still explicitly non-physical) `ProxyScorer`.
3. **`U_geom(x)` as a second `ProxyScorer` baseline** (§6), to compare
   against `DummyProxy` and the eventual toy-trained scorer on tail-FNR.
4. **`fairship_adapter` dry-run**: an input-artifact writer + run-manifest
   validator against the existing contract doc, still without executing
   FairShip — closes some of the "Unknown" rows in §7 by forcing the
   input-side format to be nailed down even before real physics runs.
5. Only after 2–4: a real `fairship_simulator`, and only then a
   `U_FairShip(x)` trained on real labels.

This branch (`research/u-definition-fairship-audit`) stops at the audit; step
1 onward is deliberately left for a follow-up branch.

## 9. SOCHIFI abstract framing

**Título:** *Hacia una definición operacional de utilidad post-selección para
el modelado adaptativo de fondos inducidos por muones en SHiP*

**Resumen (borrador, ~200 palabras):**

> El experimento SHiP debe estimar fondos inducidos por dispersión profundamente
> inelástica (DIS) de muones tras el blindaje magnético, donde solo un
> subconjunto pequeño y dependiente del análisis de los estados post-blindaje
> resulta en candidatos peligrosos tras aplicar geometría, aceptancia del
> detector, reconstrucción y lógica de veto (SBT/UBT). Presentamos la
> arquitectura de un pipeline adaptativo de tres módulos — un modelo de
> densidad/propuesta, un backend de simulación con una taxonología explícita
> de resultados (falla técnica, rechazo físico, candidato aceptado), y un
> módulo de puntuación U(x) — junto con contratos de datos verificados por
> pruebas automatizadas. Realizamos una auditoría exhaustiva de los datos y
> artefactos existentes del repositorio, confirmando que las muestras
> disponibles (~40,000 estados cinemáticos post-blindaje) no contienen
> ninguna etiqueta física de reconstrucción, veto o selección, y que ningún
> artefacto de campaña existente constituye evidencia física válida. A partir
> de este hallazgo, distinguimos formalmente la ocurrencia de DIS de la
> utilidad post-selección, y especificamos el esquema exacto de salidas de
> FairShip necesario para entrenar una versión físicamente significativa de
> U(x). Reportamos el estado actual como una base honesta para el trabajo
> futuro, sin reclamar aún un resultado físico.

**Do-not-claim-yet list:**
- No claim of a physically meaningful `U(x)` or trained proxy score.
- No claim of any background-rate estimate, veto-survival rate, or DIS yield.
- No claim that `artifacts/toy_campaign/` reflects real physics, real
  reconstruction, or a real veto decision — it is synthetic scaffolding only.
- No claim that the committed `data/samples/` contain any label; they are
  raw kinematics only.
- No claim that the current `feat/loop-v0` prototype was validated or is
  part of the adopted architecture — it was superseded and never merged.
- No claim of a decided biasing mechanism (data-side vs loss-side) for the
  Normalizing Flow — still an open Protocol-level question.
- No claim that `U_geom(x)`, if built, approximates operational danger —
  only geometric plausibility.

## Appendix: commands run

```
git branch --show-current          # research/u-definition-fairship-audit
python --version                   # Python 3.12.10
python -m pytest -q                # 35 passed
python scripts/inspect_available_labels.py   # confirms zero label fields in data/samples
git log --all --oneline -- scripts/run_toy_loop.py     # only on feat/loop-v0 (0024f23, 7077186)
git merge-base --is-ancestor feat/loop-v0 HEAD         # not an ancestor
git log --all --oneline -- artifacts/toy_campaign      # empty (never tracked)
```
