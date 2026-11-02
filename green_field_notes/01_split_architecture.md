# Split architecture (PROJECT DECISION on direction; details PROPOSAL)

Thesis loop: generate muons -> run FairShip -> improve guided generation -> run FairShip again to validate.

## Repo A: `nf` (classic ML; only the normalizing flow)
- Purpose: receive data, train, evaluate, hand out samples/log-probs. Nothing about FairShip, ROOT files or U(x).
- Exports: ONNX, callable from CERN ROOT SOFIE (GPU) so the team can use it without our stack. (OPEN: SOFIE operator coverage for the flow layers is unchecked; no ONNX/SOFIE code exists today, only a mention in `docs/contracts/density_problem_contract_v0.md`.)
- Works through pipes: a GPU node only trains, takes data in and delivers data out ("como un cocinero"). Experiments (autoresearch-style sweeps, configs) live here.
- Carries: `afterms` (the line that tested whether Martina Ferrillo's muons give good density estimates; estimation quality remains a fundamental problem), `density_lab`, `Nflow`, the feature-view / preprocessing code.
- May depend on torch/GPU. Must not import FairShip or ROOT.

## Repo B: `fairship-pipeline` (RL-flavoured; connection with FairShip)
- Purpose: take muons, run the thesis pipeline, take inferences and process them, compute scores such as U(x), decide what to simulate next. Most of the logic lives here.
- Carries: FairShip adapters (P2 extraction, lossless driver, provenance), entities, stage outcomes/tagging, simulation boundary, its own benchmarks (those of what comes out of / goes into FairShip).
- No GPU. Works on already-produced inference data; later it may call repo A's inference API. For now it only needs to know the data exists and how it is structured.

## Contract between them
- Both sides hold the same data contract (see `04_data_contract_sketch.md`). A has to know how to DELIVER; B has to know what to EXPECT and how to extract from FairShip.
- A joint test runs when the two are together, to detect drift when either repo changes. (PROPOSAL: a small versioned fixture + schema hash that both repos import/check; the joint test lives in whichever repo pulls the other as a dev dependency.)

## Decisions already taken about placement
- `benchmarks`: not a shared block. FairShip-facing benchmarks go to B; model benchmarks to A (user).
- `data_contracts`: depends on what each is about; the shared part is the contract in 04, not the current package.
- Martina's muons are only a development sample. New muons will be generated at another z. Never marry the design to them or to the historical z offset.
