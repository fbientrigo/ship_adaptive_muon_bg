# SHiP Adaptive Muon Background

Adaptive simulation tools for muon-induced background studies in the SHiP experiment.

This repository is a fork of `mferril/NFlow`, used as a clean starting point for developing a reproducible thesis pipeline around post-muon-shield muon generation, Normalizing Flows, proxy-based filtering, and FairShip-compatible simulation backends.

## Status

Early thesis-development repository.

The current goal is not to publish final physics results, but to build a reproducible software artifact that can evolve from toy studies to CERN/FairShip workflows.

## Scientific Goal

The long-term goal is to reduce the computational cost of finding useful muon-induced background candidates for SHiP by combining:

* post-muon-shield muon datasets,
* data contracts and provenance checks,
* proxy models for potentially dangerous candidates,
* Normalizing Flow proposal models,
* simulator backends,
* FairShip-compatible batch interfaces,
* reproducible experiment tracking.

The intended workflow is:

```text
post-muon-shield muon sample
        ↓
data contract + support audit
        ↓
proxy model
        ↓
biased proposal / Normalizing Flow
        ↓
candidate generation
        ↓
support and physics sanity gates
        ↓
simulation backend
        ↓
FairShip / reconstruction / selection results
        ↓
metrics, costs, failures, update loop
```

## Scope

This repository is intended to support:

1. Local development on a laptop or lab machine.
2. Toy adaptive-loop experiments.
3. Google Colab playground notebooks.
4. LXPLUS/CERN-compatible execution.
5. Future integration with FairShip workflows.

## Non-Goals

This repository does not aim to:

* replace FairShip or GEANT4;
* estimate final background rates by itself;
* treat legacy notebooks as validated physics evidence;
* assume that historical datasets are valid without provenance;
* mix document RAG tools into the simulation runtime.

Document search, PDF ingestion, and code/documentation RAG should remain separate tools.

## Terminology

This repository uses the term `simulator`, not `oracle`.

Preferred names:

* `simulator`
* `simulation_backend`
* `fairship_adapter`
* `toy_simulator`
* `fairship_simulator`

Avoid:

* `oracle`

## Repository Structure

The repo is organized around the three modules of the adaptive loop (see
`docs/architecture/repo_architecture_v1.md` for the full map and roadmap):

```text
ship_adaptive_muon_bg/
  pyproject.toml            # pip install -e .  (numpy-only core; extras: flow, legacy, dev)

  FairShip/                 # module 1 (backend): slot for the CERN ShipSoft/FairShip
                            #   clone or symlink — contents gitignored, never committed

  Nflow/                    # module 2: normalizing-flow proposal model
    interfaces.py           #   DensityModel + BiasStrategy (A/B seam for DIS biasing)
    legacy/                 #   quarantined untested code from the mferril/NFlow fork

  ProxyTagger/              # module 3: U(x) — continuous DIS-boundary score in [0, 1]
    interfaces.py           #   ProxyScorer protocol + score-artifact schema
    baseline.py             #   DummyProxy placeholder (non-physical)

  src/ship_muon_bg/         # shared core (imported by all modules; no FairShip/ROOT)
    data_contracts/         #   validated (N, 8) muon PKL ingestion (tested)
    simulation/             #   module-1 boundary: SimulationBackend, FlowProposalRecord,
                            #     SimulationResult, OutcomeCategory

  notebooks/colab/          # thin notebooks: clone → install → call (no function defs)
  scripts/                  # thin CLI entry points (no business logic)
  tests/                    # numpy-only guard-rail suite
  docs/                     # contracts, architecture, legacy migration log
```

### The three modules

1. **Simulation (`FairShip/` + `src/ship_muon_bg/simulation/`)** — takes a
   proposal of points `x`, runs the FairShip software, and saves outcomes
   with DIS marking. Project code talks only to the `SimulationBackend`
   interface; the CERN repo itself is referenced (not committed) under
   `FairShip/`.
2. **`Nflow/`** — the ML-heavy part: a normalizing flow learns to generate
   the data distribution with a bias toward DIS-likely points. The biasing
   mechanism (data aggregation, modified loss, both, ...) is an open
   question, so it stays behind the `BiasStrategy` interface for A/B
   testing.
3. **`ProxyTagger/`** — uses new simulation outcomes to improve `U(x)`, the
   continuous 0 (never DIS) → 1 (DIS always) boundary estimate over all
   simulation inputs; it steers the proposal bias and supports
   hyperparameter-landscape visualization.

## Installation

Local / lxplus (editable, with test tooling):

```bash
git clone https://github.com/fbientrigo/ship_adaptive_muon_bg
cd ship_adaptive_muon_bg
pip install -e .[dev]
pytest
```

Google Colab — open `notebooks/colab/quickstart.ipynb`, or:

```python
!git clone https://github.com/fbientrigo/ship_adaptive_muon_bg
%cd ship_adaptive_muon_bg
%pip install .
```

Extras: `.[flow]` (torch, for future flow implementations), `.[legacy]`
(everything the quarantined `Nflow/legacy/` trainer needs). The core and
the whole test suite need only NumPy — no GPU, no ROOT, no FairShip. On
lxplus, place the FairShip clone/symlink under `FairShip/` (see
`FairShip/README.md`).

## Core Components

### Data Contracts

The pipeline should treat every input dataset as suspect until its schema, units, geometry, provenance, and intended use are documented.

Expected checks include:

* variable names,
* coordinate convention,
* momentum convention,
* units,
* geometry version,
* post-muon-shield definition,
* dataset hash,
* source path,
* production metadata.

### Proxy Models

Proxy models estimate whether a generated muon candidate is worth sending to a more expensive simulation stage.

Proxy metrics must emphasize tail behavior, not only global accuracy.

Important diagnostics:

* false-negative rate in the dangerous region,
* recall after support gates,
* calibration curves,
* score stability across splits,
* failure analysis.

### Normalizing Flows

Normalizing Flows are used as proposal models for generating candidate muons in relevant regions of phase space.

Visual marginal agreement is not sufficient.

Required diagnostics should include:

* support coverage,
* effective sample size where applicable,
* tail diagnostics,
* density sanity checks,
* rejection rate after support gates,
* comparison against pilot post-shield data.

### Simulation Backends

The simulation layer should expose a common interface for different backends:

* toy simulator,
* dry-run simulator,
* FairShip adapter,
* future batch/LXPLUS execution.

Every simulation result should distinguish between:

* successful simulation,
* physics rejection,
* geometry rejection,
* format error,
* dependency error,
* runtime failure.

### Experiment Tracking

All non-trivial runs should record:

* config,
* seed,
* git commit,
* dataset hash,
* environment metadata,
* model artifact paths,
* generated candidate paths,
* simulator outputs,
* metrics,
* failure summaries.

MLflow or another lightweight tracking mechanism may be used.

## Development Principles

* Keep notebooks as interfaces, not as the main implementation.
* Keep reusable code under `src/`.
* Keep configs versioned.
* Keep large datasets out of Git.
* Keep simulator-specific code behind adapters.
* Keep LXPLUS constraints visible from the beginning.
* Preserve migration history from legacy repositories.

## Legacy Sources

This repository may reuse or adapt ideas from:

* `mferril/NFlow`;
* local legacy `nflow`;
* local legacy `ship_nflow`;
* previous Google Colab toy studies;
* FairShip documentation and scripts.

Any migrated component should be documented in:

```text
docs/legacy_migration_log.md
```

with:

```text
source repo | source file | destination file | change | reason | validation
```

## First Milestone

The first target is a reproducible toy loop:

```text
load toy data
train proxy
train/sample flow
propose candidates
run toy simulator
evaluate cost and useful-candidate yield
write campaign summary
```

Expected command:

```bash
python scripts/run_toy_loop.py --config configs/campaigns/toy_adaptive_v0.yaml
pytest
```

Expected outputs:

```text
artifacts/
  toy_campaign/
    config_resolved.yaml
    dataset_hash.json
    proxy_metrics.json
    flow_diagnostics.json
    proposed_candidates.parquet
    simulator_results.parquet
    campaign_summary.md
```

## CERN / LXPLUS Direction

The repository should be designed so that the same campaign can eventually run on LXPLUS or CERN batch infrastructure with explicit runtime configs.

Planned runtime profiles:

```text
configs/runtime/local.yaml
configs/runtime/colab.yaml
configs/runtime/lxplus.yaml
```

The LXPLUS profile should avoid assumptions that only work on a laptop or Colab.

## License

To be decided.

This repository is currently a thesis-development fork and should not be treated as a final SHiP production package.
