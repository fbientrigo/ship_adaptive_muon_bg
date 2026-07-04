# FairShip/ — reference slot for the CERN FairShip repository

This directory holds **only** the CERN [`ShipSoft/FairShip`](https://github.com/ShipSoft/FairShip)
software (or a symlink to an existing installation). Its contents are
**gitignored** — nothing you put here is ever committed to this repository.

## How to populate it

On lxplus (preferred — use the central CVMFS/SHiP environment and symlink):

```bash
ln -s /path/to/your/FairShip FairShip/FairShip
```

or clone directly:

```bash
git clone https://github.com/ShipSoft/FairShip FairShip/FairShip
```

## Boundary rule

Project code **never** imports FairShip or ROOT directly (guard-rail tests
enforce this). All interaction goes through the `simulation_backend`
boundary — `src/ship_muon_bg/simulation/` (`SimulationBackend`,
`FlowProposalRecord`, `SimulationResult`, `OutcomeCategory`) — and, in the
future, the `fairship_adapter` specified in
`docs/contracts/fairship_adapter_contract_v0.md`. The adapter is the only
component allowed to know FairShip-specific details (input formats,
geometry tags, environment profiles, output layout), and it receives the
location of this directory via configuration, never as a hardcoded path.
