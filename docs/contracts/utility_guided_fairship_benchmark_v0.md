# Utility-guided FairShip benchmark contract v0

This contract defines one report shape for three distinct benchmark arms:

| arm | generation measure |
| --- | --- |
| `P0` | nominal source sampling |
| `PU_DIRECT` | direct sampling from declared `PU` |
| `Q_THETA` | generation from learned `Q_theta` |

The versioned configuration is
`configs/utility_guided_fairship_benchmark_v0.json`.  It pins the declared
FairShip configuration and geometry identifiers, source-state definition, and
the predeclared-cohort/no-redraw policy.  It does not define a final endpoint
`B`; an endpoint definition remains explicit provenance when one exists.

Every arm has the same three report sections:

1. `utility_tilt`: normalization/support checks, enrichment versus
   concentration, and ESS as a diagnostic only.  A high rejection fraction is
   not an OOD classification, and no universal ESS threshold exists.
2. `proposal_fidelity`: held-out fit metric, physical-space diagnostics,
   high-utility occupancy, and optional two-sample diagnostics.  Direct `PU`
   and learned `Q_theta` are never merged into one proposal.
3. `fairship_outcomes`: valid executions, technical failures, and physics
   outcomes in separate fields.  A technical failure carries no physics
   outcome.

Candidate cohorts must be declared before FairShip execution.  No
redraw-until-success is permitted.  `physical_source_weight` is source
provenance and `utility_multiplier` is a separate tilt quantity; neither is
silently substituted for the other.

The report retains the canonical lineage axes: `source_state`,
`fs_sim_execution`, `interaction_realization`, and `observation`.
Intermediate `Y^(k)` records are not renamed as endpoint `B`.

This is a contract and report skeleton, not an empirical FairShip result.
