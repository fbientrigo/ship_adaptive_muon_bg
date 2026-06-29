# `fairship_adapter` Contract (v0)

Status: **specification only — not implemented.**

This document specifies the *future* `fairship_adapter`. It defines the
boundary, inputs, outputs, failure taxonomy, reproducibility metadata, and the
tests that must exist *before* any real FairShip integration is attempted. No
code in this commit implements the adapter, executes FairShip, or imports ROOT.

The terminology in this document is fixed. The following terms are used
consistently and must not be renamed:

- `simulator`
- `simulation_backend`
- `FairShip base simulator`
- `fairship_adapter`
- `fairship_simulator`
- `toy_simulator`

## Purpose

The `fairship_adapter` is a future infrastructure adapter. Its single
responsibility is translation across the boundary between the pure-Python
project core and a FairShip-based `simulation_backend`:

- It converts project-level records (`FlowProposalRecord[]`) into
  FairShip-compatible input artifacts.
- It converts raw FairShip outputs back into project-level
  `SimulationResult[]` records.

The adapter is the *only* component allowed to know FairShip-specific details
(input file formats, geometry tags, environment profiles, output file layout).
The scientific core stays free of ROOT, FairShip, and CERN/EOS assumptions and
talks only to the `simulation_backend` interface. The `fairship_adapter` is the
concrete bridge that lets a `fairship_simulator` (the `FairShip base simulator`
wrapped behind the adapter) satisfy that interface, exactly as the
`toy_simulator` does today.

## Non-goals

This contract explicitly does **not** cover and does **not** promise:

- **No FairShip execution in the core.** The core never runs FairShip; only a
  future adapter, behind the `simulation_backend` boundary, may do so.
- **No ROOT import in the core.** ROOT (and any ROOT-dependent code) stays
  outside `src/` core modules. Only the future adapter layer may touch ROOT.
- **No physical validity claim from the toy campaign.** The reproducible toy
  campaign exercises plumbing and provenance, not physics correctness.
- **No final background-rate estimation.** Nothing here estimates SHiP muon
  background rates.
- **No trained proxy.** This document does not define or rely on a trained
  proxy model.
- **No real Normalizing Flow yet.** `FlowProposalRecord[]` may be produced by a
  toy/placeholder proposer; a real NF is out of scope here.

## Boundary

The intended data flow across the adapter boundary is:

```text
FlowProposalRecord[]
    -> fairship_adapter
    -> FairShip-compatible input artifact
    -> FairShip base simulator / fairship_simulator
    -> raw backend outputs
    -> parsed SimulationResult[]
```

Read left to right:

1. The core hands the adapter a list of candidate proposals
   (`FlowProposalRecord[]`).
2. The `fairship_adapter` serializes them into a FairShip-compatible input
   artifact.
3. The `FairShip base simulator` (wrapped as the `fairship_simulator`) consumes
   that artifact and runs the simulation.
4. The backend writes raw outputs (ROOT files, logs, manifests).
5. The `fairship_adapter` parses those raw outputs back into
   `SimulationResult[]` records for the core.

Everything to the left of `fairship_adapter` is pure project core. Everything to
the right of it is FairShip/ROOT territory and lives behind the adapter.

## Input contract

The minimal future adapter inputs. These are the fields the adapter must
receive (directly or via a resolved campaign config) to perform a run:

| Field | Type / form | Meaning |
| --- | --- | --- |
| `candidates` | `FlowProposalRecord[]` | The proposed muon candidates to simulate. |
| `resolved_campaign_config` | mapping | Fully resolved (no unresolved references) campaign configuration. |
| `geometry_tag` | string | Identifier of the FairShip geometry to use. |
| `fairship_environment_profile` | string | Name of the FairShip environment profile (not a hardcoded path). |
| `input_format_version` | string | Version of the FairShip-compatible input artifact schema. |
| `seed` | int | Deterministic seed; never derived from `time.time()`. |
| `max_events` | int | Upper bound on events to simulate. |
| `timeout` | duration | Wall-clock limit for the backend run. |
| `working_directory` | path | Scratch directory for the run (caller-provided, not hardcoded). |
| `output_directory` | path | Where backend outputs and parsed results are written. |
| `dataset_hash` | string | Hash of the source dataset behind the candidates. |
| `campaign_id` | string | Identifier of the campaign this run belongs to. |

Notes:

- Paths (`working_directory`, `output_directory`) are supplied by the caller.
  No CERN/EOS path is hardcoded.
- `fairship_environment_profile` names a profile resolved elsewhere; the adapter
  does not embed environment-specific absolute paths.

## Output contract

The expected future adapter outputs. After a run, the adapter returns (and/or
persists) the following:

| Field | Type / form | Meaning |
| --- | --- | --- |
| `adapter_input_artifact_path` | path | Location of the FairShip-compatible input artifact the adapter wrote. |
| `backend_run_manifest` | mapping / path | Manifest describing how the backend was invoked. |
| `raw_backend_output_paths` | path[] | Paths to raw backend outputs (e.g. ROOT files, logs). |
| `results` | `SimulationResult[]` | Parsed project-level results. |
| `technical_failures` | record[] | Candidates that failed for technical reasons (see taxonomy). |
| `physics_rejections` | record[] | Candidates simulated correctly but rejected by selection. |
| `accepted_candidates` | record[] | Candidates simulated correctly and accepted by selection. |
| `timing` | mapping | Timing information (submit, run, parse durations, wall clock). |
| `backend_version_metadata` | mapping | Backend name and version metadata. |
| `geometry_metadata` | mapping | Geometry tag/version metadata actually used. |

Each parsed `SimulationResult` must be classifiable into exactly one of
`technical_failure`, `physics_rejection`, or `accepted_candidate` (see below).

## Failure taxonomy

Every candidate outcome falls into exactly one of three mutually exclusive
categories. Keeping these separate is mandatory: a technical problem must never
be silently counted as a physics result.

### `technical_failure`

The run could not produce a trustworthy physics answer for the candidate. Causes
include:

- input/output format errors;
- missing geometry;
- ROOT crash;
- timeout;
- missing expected output;
- incompatible environment profile.

A `technical_failure` says nothing about the physics of the candidate; it must
not be counted as a rejection or an acceptance.

### `physics_rejection`

The candidate was simulated correctly (the backend ran to completion and
produced valid output), but it was rejected by the operational selection. This
is a valid physics outcome.

### `accepted_candidate`

The candidate was simulated correctly **and** passes the operational selection.
This is a valid physics outcome and the signal the campaign is looking for.

Invariant: `technical_failure` is about *whether the simulation is
trustworthy*; `physics_rejection` and `accepted_candidate` are about *what a
trustworthy simulation concluded*. Counts of physics rejections and accepted
candidates are only meaningful over the set of non-technical-failure outcomes.

## Reproducibility metadata

Every adapter run must record the following metadata so the run can be audited
and reproduced:

- `command` — the exact command/invocation used;
- `git_commit` — commit hash of the project at run time;
- `config_hash` — hash of the resolved campaign config;
- `dataset_hash` — hash of the source dataset;
- `seed` — the deterministic seed used;
- `environment_profile` — the FairShip environment profile name;
- `backend_name` — name of the `simulation_backend` used;
- `backend_version` — version of that backend;
- `geometry_tag` — the geometry tag/version used;
- `input_schema_version` / `output_schema_version` — schema versions of the
  adapter input artifact and the parsed `SimulationResult`.

## Testing expectations before implementation

The following tests must exist and pass *before* any real FairShip integration
is written. They are guard rails that keep the boundary clean while the adapter
is still a specification.

- **No ROOT/FairShip imports in core** — a test asserting that core `src/`
  modules import neither ROOT nor FairShip.
- **No CERN/EOS hardcoded paths** — a test asserting that no CERN/EOS absolute
  paths appear in `src/`, `scripts/`, or `tests/`.
- **Deterministic toy campaign** — running the `toy_simulator` campaign twice
  with the same seed yields identical results (and no `time.time()` seeding).
- **Adapter dry-run contract test** — a dry-run of the `fairship_adapter`
  contract (no real FairShip) validates that inputs map to a well-formed input
  artifact and that a manifest is produced.
- **Technical failure vs physics rejection separation** — a test that the three
  outcome categories are mutually exclusive and that technical failures are not
  counted as physics outcomes.
- **Schema validation for `SimulationResult`** — parsed results validate against
  the `SimulationResult` schema.
- **Artifact manifest validation** — the backend run manifest validates against
  its expected schema.
- **Minimal fake backend roundtrip** — a fake/stub backend (no ROOT) accepts an
  adapter input artifact and returns outputs the adapter can parse back into
  `SimulationResult[]`, exercising the full boundary end to end.

These tests do not require FairShip or ROOT; they can run with the existing pure
Python core, the `toy_simulator`, and a minimal fake backend.

## Open questions

Unresolved boundary conditions that must be settled before implementation:

- **Exact FairShip input file format** — the concrete file format and schema of
  the FairShip-compatible input artifact.
- **Exact post-shield coordinate convention** — coordinate system, origin, and
  axis orientation for post-muon-shield muon kinematics.
- **Exact geometry tag/version** — which geometry tag/version the campaign
  targets and how it is pinned.
- **Exact selection definition** — the precise operational selection that
  separates `physics_rejection` from `accepted_candidate`.
- **Exact SBT/UBT fields** — which Surrounding/Upstream Background Tagger fields
  are read and how they are represented.
- **Exact output files to parse** — which raw backend output files the adapter
  parses, and in what order of precedence.
- **How weights are propagated** — how event/candidate weights flow from
  `FlowProposalRecord[]` through the backend into `SimulationResult[]`.

Until these are resolved, the `fairship_adapter` remains a specified contract
only, and the core continues to rely on the `toy_simulator` behind the
`simulation_backend` interface.
