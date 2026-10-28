# Generating muons from scratch with current FairShip

Started 2026-10-28. FairShip checkout `FairShip` at `e8627d1de` (read-only; runs write to `~/thesis/scratch/gen-from-scratch-v0/`).
Labels: OBSERVED (read in code / ran), INFERENCE, OPEN. Goal (user): regenerate muons with today's FairShip so the
+2.5916 m offset question disappears; run locally first, later as an lxplus job.

## Chain

| Stage | Command (from `FairShip/`, env `pixi run --manifest-path FairShip/pixi.toml`) | Output | Label |
|---|---|---|---|
| 1. Proton on target -> hadron absorber exit | `macro/run_fixedTarget.py -n N -e <ecut GeV> --storeOnlyMuons [--kaon-pion-splits K] -rs <seed> -r <run> -f -o <dir>` | `pythia8_evtgen_Geant4_<run>_1.0.root`, tree `cbmsim`, branch `PlaneHAPoint` (pos, mom, pdg, time, parent vertex), plus `geofile_full.root` | OBSERVED: ran, exit 0 (n=20, -e 1: 73 s wall, 0 events kept) |
| 2. Muons through the muon shield | `macro/run_simScript.py --MuonBack -f <stage1 file> --FollowMuon [--FastMuon] -n N -s <seed> --shieldName TRY_2026` | `sim_*.root`; `vetoPoint` hits only | OBSERVED in code (`run_simScript.py` l.820-845); CI smoke exists (`pixi.toml` `ci-sim-muonback`) |
| 3. Record a plane after the shield | no CLI option; `exitHadronAbsorber` (PyROOT) accepts `SetZposition(z)` and then does NOT stop tracks | needs a thesis-side macro (copy of the setup in `run_fixedTarget.py` l.393-420) | OBSERVED in code (`muonShieldOptimization/exitHadronAbsorber.cxx` l.98, 487-503); not yet run |

## Facts

- `PlaneHA` is the plane just behind the hadron absorber, i.e. the entrance of the muon shield ("before the shield"). Default: tracks stop there (`exitHadronAbsorber.cxx` l.98). OBSERVED.
- `--AddPostTargetSensPlane` adds a non-stopping plane right after the target (`run_fixedTarget.py` l.362-380). OBSERVED.
- `exitHadronAbsorber` hit = detector id 111, position, momentum, time, track length, pdg, production vertex and momentum. Plane box half-sizes 3.56 m x 1.7 m, thickness 2 mm. OBSERVED.
- Without `SetUseCaveCoordinates`, `SetZposition` is local to `/MuonShieldArea_1` (l.512-516); with it, local to `TargetArea_1`. The frame of a post-shield plane must be checked by a geometry read before trusting z. OPEN.
- TRY_2026 shield: starts 2.04 m, ends 31.93 m from the target origin (`geometry_config.py`; matches TGeo). OBSERVED.
- No GAN code exists in FairShip `e8627d1de` (grep). The "same generator" is the Pythia8+Geant4 chain above. OBSERVED.
- Heavy-flavour option (`-C`/`-B`) needs a Cascade input file on EOS (`EOSSHIP`), not available locally. OBSERVED (default path in the argparse).
- A muon is rare per event: with `-e 1`, 20 events gave 0 kept events. Yield per POT is being measured (run `stage1_n3000`, kaon/pion splitting 10).
