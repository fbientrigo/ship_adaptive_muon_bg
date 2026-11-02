# Data contract sketch (PROPOSAL; to be settled with the user)

Today: `data_contracts` enforces an `(N, 8)` array `[px, py, pz, x, y, z, id, w]` for "trusted local legacy muon files" (`data_contracts/__init__.py`). That is the Martina-era layout: it carries no plane, no geometry, no provenance. The P2 work shows what a state needs.

## A state row (candidate; `p2_crossing_state` stays provisional, not a nominal SourceState)
| Field | Meaning |
|---|---|
| `px,py,pz` GeV/c, `x,y,z` cm, `t` ns | coordinates at the entry face of the plane, FairShip global frame |
| `pdg` | particle id |
| `plane_id` + `geometry_sha256` + `fairship_version` | what the coordinates mean; replaces any z offset |
| `stored_weight` | stored MCTrack weight, semantics OPEN |
| `run_id, progenitor, split_group, trajectory` | provenance hierarchy; status RESOLVED / UNRECOVERABLE / AMBIGUOUS |
| `crossing_class` | FIRST_FORWARD (eligible) / REPEAT_FORWARD / BACKWARD |

## Who does what
- A (NF): `fit(table)`, `sample(n, seed)`, `log_prob(table)`, `export_onnx()`. Delivers samples/log-probs in the same row layout. Declares the feature view it was trained on (invertible, Jacobian known).
- B (pipeline): extracts rows from FairShip (P2 driver), validates them against the contract, consumes A's output, computes U(x) and scores.
- Transport: files (Parquet or npz + JSON manifest) and pipes; no shared Python imports at runtime. (OPEN: format.)

## Joint test (when both are together)
1. A produces a tiny deterministic delivery from a fixed seed.
2. B reads it with its own reader and checks schema version + hash, units, column order, finite values, weight column semantics string.
3. Fails if either side changes the schema without bumping its version.
