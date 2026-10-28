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
- Yield measured (run `scratch/gen-from-scratch-v0/stage1_n3000`: `-n 3000 -e 1 --storeOnlyMuons --kaon-pion-splits 10 -rs 2 -r 2`):
  1784 s wall (0.59 s/event, single core), 4 events with activity, 13 `PlaneHAPoint` hits (8 x pdg 13, 5 x pdg -13),
  all at z = 434.7 cm, p = 1.2..4.7 GeV. So ~0.004 hits/event, ~137 s per hit at this cut. OBSERVED.
  The 13 hits come from only 4 events: split kaon/pion clones share a parent, so they are not independent muons.
  Weights of split clones are not read yet. OPEN.
- `PlaneHA` sits at z = 4.347 m (end of the first shield segment, 4.34 m in `geometry_config.py` + 1 cm), not at 2.04 m. OBSERVED.
- p of these muons is low (<5 GeV); the hard `p>3 GeV` SBT cut and the generator's `p<2 GeV` limit will remove many. INFERENCE from 13 hits.

## Production started 2026-10-28 (stage 1, local)

- Launcher `~/thesis/scratch/gen-from-scratch-v0/launch_stage1.sh <seed> <n>`; 3 detached workers, seeds 101/102/103, 20000 events each,
  `-e 1 --storeOnlyMuons --kaon-pion-splits 10`, output `prod_stage1/seed_<seed>/out/`, log `run.log` (ends with `exit=<code>`).
  Expected ~3.3 h each at 0.59 s/event (estimate from the 3000-event run, not a measurement of these runs).
- Memory: each worker held ~1.1 GB RSS (OBSERVED, `ps`); the machine has 7.8 GB and other sessions. Do not start more than 3-4.
- Outputs are scratch evidence; nothing is committed to the repo.

## GAN (what exists)

| Claim | Label | Evidence |
|---|---|---|
| SHiP published a GAN that emulates the proton-target interaction and returns muons ("fast simulation ... factor of O(10^6)") | OBSERVED (paper) | arXiv:1909.04451, JINST 2019 (Ahdida, Ratnikov, Zaytsev et al.) |
| It replaces the slow part of our stage 1 (target + absorber), not the muon shield | INFERENCE from the abstract | same |
| No GAN code or weights in FairShip `e8627d1de`, its branches or history; no public repository found by two web searches | OBSERVED (negative; absence of evidence) | local grep and `git log --grep`; searches |
| The thesis registry lists `muons_GAN.pkl` (291 MB) and `muonsGAN_afterMS.pkl` (496 MB) with geometry, geofile and POT scale all UNRESOLVED; the files are not on this machine | OBSERVED | `phd-thesis/thesis_material_audit/dataset_registry.json`; `find` found no copy |

Consequence: a GAN cannot be used from what we have. It would be a different, lower-trust source (BLOCKED in the registry) and the question
"how to increase production" is answered today only by more stage-1 CPU, FairShip's built-in boosts (`--kaon-pion-splits`, `-C/-B` with the EOS Cascade file) and lxplus.
