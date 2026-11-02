# Open questions (for the user)

1. Data transport format between A and B: files (Parquet / npz + JSON manifest) or streaming pipes only?
2. Which sample is the first greenfield muon set, and at which z? (New muons are to be generated at another z; Martina's stay as development data only.)
3. Where does stage-2 source id injection live: upstream FairShip, a generator-side wrapper, or keep content-join? (Content-join works today, 21/21.)
4. SOFIE: which flow layers does it support on GPU? Needs a probe before we commit the NF architecture to ONNX.
5. History: keep git history of each half via `git subtree split` / `filter-repo` on a copy?
6. Shared contract as a third tiny package, or copied with the joint test as guard?
7. What of the old docs (34 files) is carried as history vs left behind?
8. Weight semantics and ESS: still OPEN, needed before any estimate is quoted.
9. Redo the offset comparison from the geometry manifest (shield end 32.18 m, P2 32.28 m).
10. Upstream proposal `--MuonBackKeepScoringPlanes`: a human decides whether to send it.
