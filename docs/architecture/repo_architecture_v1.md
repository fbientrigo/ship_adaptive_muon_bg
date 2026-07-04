# Repository Architecture v1 — the three-module layout

Status: **implemented** (branch `feat/architecture`).

This document describes the actual repository structure after the
architecture reorganization, how the three modules map to the adaptive
loop, how the same code runs on lxplus and Colab, and the ordered best-ROI
next steps. It complements (does not replace) the existing contracts:

- `docs/contracts/fairship_adapter_contract_v0.md` — the simulation boundary.
- `docs/architecture/ml_skeleton_local_pkl_v0.md` — the ML skeleton plan and
  its deferral rules (overfit-first, no empty packages, labels before proxy).

## The adaptive loop and its modules

```text
            ┌───────────────────────────────────────────────────────┐
            │                                                       │
            ▼                                                       │
  [1] simulation_backend            [2] Nflow                       │
  proposals x ──► FairShip run ──►  train/update proposal model ──► new proposals x'
  (DIS-marked outcomes)             biased toward DIS-likely x      │
            │                                 ▲                     │
            │ labeled outcomes                │ U(x) bias           │
            ▼                                 │                     │
  [3] ProxyTagger ────────────────────────────┘                     │
  update U(x): 0 (never DIS) → 1 (DIS always)                       │
  steers the bias; visualizes the hyperparameter landscape ─────────┘
```

| Module | Repo location | Role |
| --- | --- | --- |
| **1 — Simulation** | `src/ship_muon_bg/simulation/` (boundary) + `FairShip/` (CERN repo slot, gitignored) | Takes proposed points `x`, runs the FairShip software, saves outcomes with DIS marking. Project code sees only `FlowProposalRecord[] → SimulationBackend → SimulationResult[]` with the three-way outcome taxonomy (`technical_failure` / `physics_rejection` / `accepted_candidate`). The future `fairship_adapter` and HTCondor execution implement this interface. |
| **2 — Nflow** | `Nflow/` | The ML-heavy part: a normalizing flow learns the data distribution with a bias toward DIS-likely points. The biasing mechanism (data aggregation, modified loss, both, ...) is undecided by design and lives behind `Nflow.interfaces.BiasStrategy` so strategies are A/B-testable. Proposal models live behind `Nflow.interfaces.DensityModel`. Legacy fork code is quarantined in `Nflow/legacy/`. |
| **3 — ProxyTagger** | `ProxyTagger/` | Maintains `U(x)` ∈ [0, 1] — the continuous DIS-boundary estimate over all simulation inputs, treated as a noisy measurement. Learned only from simulation-produced labels (`technical_failure` never counts). `ProxyScorer` protocol + `DummyProxy` placeholder today. |
| **Shared core** | `src/ship_muon_bg/` | What every module relies on: `data_contracts/` (validated `(N, 8)` muon ingestion, hashing, splits, normalization — tested) and `simulation/` (the boundary types above). NumPy-only, never imports FairShip/ROOT (guard-tested). |

## Directory map

```text
ship_adaptive_muon_bg/
├── pyproject.toml            # pip install -e .   (numpy-only core; extras: flow, legacy, dev)
├── FairShip/                 # CERN ShipSoft/FairShip clone or symlink goes here (gitignored)
├── Nflow/                    # module 2: DensityModel + BiasStrategy interfaces; legacy/ quarantine
├── ProxyTagger/              # module 3: ProxyScorer interface + DummyProxy baseline
├── src/ship_muon_bg/         # shared core: data_contracts/ + simulation/ boundary
├── notebooks/colab/          # thin notebooks: clone → install → call (no function definitions)
├── scripts/                  # thin CLI entry points (no business logic)
├── tests/                    # numpy-only guard-rail suite (no GPU/ROOT/FairShip)
└── docs/                     # contracts, architecture, migration log
```

## Execution profiles: lxplus/HTCondor and Colab share one package

The modularity requirement is satisfied by a single installable package —
the *same* code runs everywhere; only the entry point differs:

| | Google Colab | lxplus (interactive GPU / HTCondor) |
| --- | --- | --- |
| **Setup** | `git clone … && pip install .` (see `notebooks/colab/quickstart.ipynb`) | `pip install -e .[flow]` in a venv on top of the SHiP environment |
| **FairShip** | not available — work stops at the `SimulationBackend` boundary (use `toy_simulator` when it exists) | clone/symlink into `FairShip/` (see `FairShip/README.md`); the future `fairship_adapter` receives its location via config |
| **Entry point** | notebook cells that only *call* installed functions | thin scripts under `scripts/` (future: `scripts/lxplus/` HTCondor submit templates calling the same functions) |
| **What is identical** | `ship_muon_bg`, `Nflow`, `ProxyTagger` — contracts, models, artifacts, seeds | same |

Rules that keep this true:

1. Notebooks never define functions; they clone, install, and call.
2. Scripts hold no business logic; they parse arguments and call `src/`-level code.
3. No hardcoded CERN/EOS paths anywhere (guard-tested); environments are
   config, not code.
4. Heavy dependencies are extras: the core (and the whole test suite) stays
   NumPy-only so any machine can run `pytest` in seconds.

## Best-ROI next steps (ordered)

Each step unblocks the ones after it; none requires FairShip until step 3b.

1. **First real proposal model (Nflow).** Adapt the legacy RealNVP
   (`Nflow/legacy/utils/flow_models.py`) to the `(N, 8)` PKL contract behind
   `DensityModel`, with the tiny-overfit smoke test in the same commit
   ("Commit 2" of the ML skeleton doc). Alternatively start with the
   per-feature Gaussian placeholder the skeleton doc prefers. Deliverables:
   `flow_train_metrics.json`, `proposal_diagnostics.json` (out-of-support
   rate, diversity, ESS).
2. **First labels (simulation).** Implement a `toy_simulator` behind
   `SimulationBackend` — cheap, deterministic, produces the three-way
   outcomes with a synthetic DIS rule. This unblocks *everything*
   downstream: trained `ProxyTagger`, acquisition, and the full loop
   rehearsal, all without FairShip.
3. **FairShip track (module 1).**
   a. `fairship_adapter` **dry-run**: input-artifact writer + run manifest
      validated against `docs/contracts/fairship_adapter_contract_v0.md`
      (no ROOT needed; resolves the contract's open questions one by one).
   b. `scripts/lxplus/` HTCondor submit templates that call the adapter on
      a node with `FairShip/` populated.
4. **First trained `U(x)` (ProxyTagger).** With toy labels from step 2:
   calibrated classifier behind `ProxyScorer`, tail-FNR as the primary
   metric, calibration curve as an artifact.
5. **A/B bias harness (Nflow).** Two `BiasStrategy` implementations —
   score-weighted resampling (data aggregation) vs score-weighted NLL
   (modified loss) — compared under identical seeds, data, and artifacts,
   plus the acquisition baselines (random / top-k / diversity-aware top-k).
6. **CI.** GitHub Actions running the NumPy-only `pytest` suite on every
   push — the suite is deliberately cheap, so this is nearly free insurance.
