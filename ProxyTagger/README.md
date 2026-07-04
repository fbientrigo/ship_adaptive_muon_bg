# ProxyTagger — the `U(x)` module (module 3)

Maintains `U(x)`: a continuous score in `[0, 1]` over the full simulation
input space `x`, from 0 (never DIS) to 1 (DIS always). It is learned from
outcomes the `simulation_backend` produces (module 1), treated as a noisy
boundary measurement, and used to:

- steer the `Nflow` proposal bias (via a `BiasStrategy`), and
- visualize / understand the hyperparameter landscape.

## Layout

| Path | Status | Contents |
| --- | --- | --- |
| `interfaces.py` | contract | `ProxyScorer` protocol, `U(x)` semantics, score-artifact schema (`SCORE_SCHEMA_VERSION`, `SCORE_ARTIFACT_FIELDS`). |
| `baseline.py` | tested placeholder | `DummyProxy` — deterministic, explicitly non-physical; exists only to exercise the interface and schemas end to end. |

## Hard rules (from the repo contracts)

1. **Labels come only from the `simulation_backend`.** Never assumed from
   legacy files; they are produced by simulation runs.
2. **`technical_failure` is never a training label.** Only
   `physics_rejection` vs `accepted_candidate` outcomes count; technical
   failures are excluded, not folded into the negative class
   (`ship_muon_bg.simulation.types.OutcomeCategory`).
3. **Tail metrics over global metrics.** The primary metric is tail
   false-negative rate in the dangerous region (recall after support
   gates), not accuracy/AUC. A calibrated probability requires an explicit
   calibration artifact.
4. **No physics claims from placeholders.** Anything with
   `is_physical = False` must never appear in reported results.

## Next steps for this module

Blocked on labels: implement a `toy_simulator` behind
`ship_muon_bg.simulation.SimulationBackend` first (best-ROI step 2 in
`docs/architecture/repo_architecture_v1.md`), then train the first real
`ProxyScorer` (calibrated classifier / ranking / PU-learning — see
`docs/architecture/ml_skeleton_local_pkl_v0.md`, section 6).
