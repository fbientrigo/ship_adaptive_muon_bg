# FairShip matched TGeo audit v0

Decision: `SBT_USABLE_FOR_FIRST_EMPIRICAL_ENDPOINT`.

The fixed 12-candidate genuine NF cohort and 2 empirical SBT-positive controls were read from existing ROOT outputs. Every row is matched to the same FairShip commit/configuration and one geometry anchor. `candidate_audit.csv` keeps injection state, TGeo initial volume/material, sensitive-volume encounters, detector counts, and the current preprocessing decision together; `trajectory_boundaries.csv` and `detector_observations.csv` provide the inspectable detail.

`LiSc*` is not a hand mask: it is the FairShip veto implementation's volume family created with `sens=true` (`FairShip/veto/veto.cxx:GeoSideObj/GeoCornerLiSc* sens=true`). The coordinate transform remains PROVISIONAL; this is recorded uncertainty, not a #27 activation or a causal interpretation. No candidate was redrawn or tuned and no endpoint/rate claim is made.
