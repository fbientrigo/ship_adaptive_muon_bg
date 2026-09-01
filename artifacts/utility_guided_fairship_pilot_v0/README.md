# Utility-guided FairShip pilot v0

This fixed pilot sampled six states per charge from existing `UA_d0p1_a04` NF checkpoints; it did not retrain or success-select candidates.

The checkpoint, model configuration, training-dataset, sampling-seed, and
physical-space proposal-log-probability provenance for every draw is in
`candidates.csv`.  Issue #27 remains OPEN, so the declared after-MS-to-current
FairShip coordinate transform is PROVISIONAL.  No candidate had the existing
current-main qualifying SBT condition; that prevents MuDIS continuation here
without fabricating upstream context and is not a final physics label.

| PDG | generated | mechanical | SBT-hit | MuDIS eligible | DIS realizations | Geant4 | ShipReco | technical failures |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 13 | 6 | 6 | 0 | 0 | 0 | 0 | 0 | 0 |
| -13 | 6 | 6 | 0 | 0 | 0 | 0 | 0 | 0 |

All coordinates use the existing PROVISIONAL transform; this is utility-guided generation into real simulation, not physical utility enrichment or a rate/efficiency estimate.
