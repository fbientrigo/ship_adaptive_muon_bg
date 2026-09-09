# FairShip matched TGeo audit v0

Decision: `SBT_USABLE_FOR_FIRST_EMPIRICAL_ENDPOINT`.

The fixed 12-candidate genuine NF cohort and 2 empirical SBT-positive controls were read from existing ROOT outputs. Every row is matched to the same FairShip commit/configuration and canonical `ShipGeo` identity before the shared geometry anchor is used. `candidate_audit.csv` keeps injection state, TGeo initial volume/material, straight-ray probe intersections, detector counts, and the current preprocessing decision together; `trajectory_boundaries.csv` and `detector_observations.csv` provide the inspectable detail.

`LiSc*` names are TGeo straight-ray probe intersections, not GEANT4 transport crossings or causal track assignments. The FairShip veto implementation creates this volume family with `sens=true` (`FairShip/veto/veto.cxx:GeoSideObj/GeoCornerLiSc* sens=true`). Actual GEANT4 `vetoPoint` observations remain in their separate table. Coordinate status is preserved per record (`PROVISIONAL` where supplied, `UNKNOWN` where controls omitted it); this is not a #27 activation or a causal interpretation. No candidate was redrawn or tuned and no endpoint/rate claim is made.

Reproduce with the FairShip runtime:

```sh
cd /home/fabian/thesis/worktrees/FairShip-current-main-f73a305
pixi run python /tmp/ship-sprint1-issue-30/scripts/audit_fairship_tgeo.py --pilot-dir /home/fabian/thesis/worktrees/ship-fairship-utility-connector-v0/artifacts/utility_guided_fairship_pilot_v0 --controls-dir /home/fabian/thesis/worktrees/ship-fairship-current-mudis-v0/artifacts/fairship_connector_v0 --fairship-dir /home/fabian/thesis/worktrees/FairShip-current-main-f73a305 --output /tmp/ship-sprint1-issue-30/artifacts/fairship_tgeo_audit_v0
```
