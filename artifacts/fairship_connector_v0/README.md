# FairShip utility connector v0

## STATUS

`P0-A`, `P0-B`, current-main MuonDIS path A for selected empirical states,
canonical bundle parsing, and a three-state bounded batch are verified against unmodified FairShip
`f73a305a86468c320665cf41b73582ce7e533181`.  Coordinate physics is not
verified: every run used a named `PROVISIONAL` injection transform.

```text
5D proposal state
      |
      + z plane / charge / transport bookkeeping
      v
FairShip TTreeGenerator
      v
cbmsim
      v
current MuonDIS A
      v
GEANT4
      v
canonical observations
      v
future U_phys
```

## WHAT WE GENERATED

The generated physics state is exactly `(px, py, pz, x, y)`.  P0-A used an
empirical after-MS row (row 0); P0-B used a designated NF-like five-dimensional
fixture, not an NF checkpoint sample.  The end-to-end MuDIS evidence used
empirical rows 8217, 32029 (`mu+`), and 25740 (`mu-`), each in a separate
execution so current-main preprocessing cannot lose source identity.

## WHAT METADATA WE ADDED

We attached declared `z`, PDG/charge, a named coordinate transform, source
provenance, and separate bookkeeping.  For the mechanical runs, the transform
`afterms_nominal_plane_to_current_shield_exit_plus_1cm_f73a305_v0` maps
`z_out_m = z_declared_m + 2.5916`; it is `PROVISIONAL`, not a statement that
the after-MS plane and this FairShip plane are physically identified.

## WHAT FAIRSHIP RECEIVED

FairShip received only `converted_ntuple` rows with the upstream-required
schema `(px, py, pz, x, y, z, id, w)`.  Positions are metres in the importer
configuration and become centimetres in FairShip.  `w` was always `1.0`; no
upstream detector hit, parent history, interaction, or time was fabricated.

## WHAT FAIRSHIP ACTUALLY SIMULATED

The clean detached current-main worktree ran the existing importer,
`run_simScript.py --ttree`, `make_nTuple_SBT.py`, `makeMuonDIS.py`, and
`run_simScript.py --MuDIS`.  The P1 artifact has one
`MuonAndSoftInteractions` entry, two `DIS` entries, and two Geant4 `cbmsim`
entries.  The bounded batch repeated this for three source states with two DIS
realizations per state.  Current A accepts only the existing SBT-selected
muons (qualifying SBT hit and momentum threshold), so a mechanically injected
P0 state is not automatically eligible for MuDIS continuation.  A separate ShipReco run on the first MuDIS output
reconstructed two events and reported five fitted/good tracks; that is
reconstruction evidence, not an endpoint decision.

## WHAT REAL OUTPUT CAME BACK

P0 artifacts verify a readable `cbmsim` and a first `MCTrack` matching the
injected muon within the recorded tolerance.  Current-A artifacts contain
real ROOT files for preprocessing, `DIS`, and Geant4; Geant4 `cbmsim` includes
the observed `CrossSection` branch.  Each artifact directory contains the
exact commands, stdout/stderr, hashes, ROOT inventory, and verification result.

## WHAT OUR CODE EXTRACTED

`CurrentMainMuonDISRunner` produces an `EvaluationBundle` with one
`FSSimExecution` per separately staged source state, two genuinely observed
`InteractionRealization`s per successful current-A run, and computed evidence
for preprocessing, DIS, Geant4 entry counts, plus the per-realization native
FairShip `CrossSection` scalar.  There are no `StageDecision`s and no
reconstructed candidates in the canonical bundle: no authoritative endpoint
definition was applied.

## WEIGHT BOOKKEEPING

`physical_source_weight` is preserved from an empirical source row (7.6875 in
the exercised rows) and is never multiplied into simulator data.  The
NF-like fixture has `physical_source_weight = null`.  Every injected row has
`fairship_transport_weight = 1.0`, also sent as TTree `w = 1.0`; this is only
mechanical transport bookkeeping, not physical normalization.  The observed
MuonDIS `CrossSection` is retained as a native-unit scalar and is not renamed
or used as an importance weight.  Proposal density/log-probability and future
importance weighting are not supplied by this connector.

## WHAT IS VERIFIED

- An empirical row and a finite NF-like five-dimensional fixture mechanically
  enter unmodified FairShip and produce matching `MCTrack` truth.
- Selected empirical source states are traceable through TTree input,
  `cbmsim`, current-main MuDIS preprocessing, DIS generation, Geant4, and
  canonical records; selected-muon kinematics/PDG and native cross sections
  are compared at the preprocessing-to-DIS-to-Geant4 boundaries.
- The current-main A adapter treats missing/invalid output and failed commands
  as technical failure, never as a physics negative.

## WHAT IS PROVISIONAL

The after-MS-to-FairShip coordinate transform is `PROVISIONAL`; successful
Geant4 execution does not validate its physical plane.  Pythia6 in current
`makeMuonDIS.py` uses an uncontrolled wall-clock seed, so the explicit Geant4
seed does not make the DIS realization bit-reproducible.

## WHAT REMAINS OPEN

Coordinate closure is tracked in issue #27; physical reweighting for generated
muons in #24; the updated MuonDIS adapter in #25; and operational endpoint B in
#26.  This work does not estimate a background rate, natural DIS probability,
utility calibration, veto efficiency, or a final downstream label.

## CURRENT A VS FUTURE B

The scientific core receives only `EvaluationRequest` and `EvaluationBundle`.
Path A is isolated in `adapters/current_main_mudis.py`, which knows
`MuonAndSoftInteractions`, `DIS`, and current FairShip scripts.  A future B
would replace that adapter with `UpdatedMuonDISRunner` at the `cbmsim ->
EvaluationBundle` seam; no B compatibility is claimed or implemented here.

## NEXT SCIENTIFIC STEP

Validate the after-MS coordinate transform against a documented shared frame,
then define the physical reweighting and a versioned downstream endpoint before
using these artifacts to update physical utility.

## ARTIFACT MAP

This directory links the P0 and A-path packages from their isolated worktrees:
`p0a_real_row_0`, `p0b_generated_5d_0`, `p1_current_main_row_8217`, and the
four `p3_*` directories.  Large ROOT files remain ignored; each linked package
is self-contained for its execution.
