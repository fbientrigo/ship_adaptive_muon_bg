# After-MS → FairShip transform validation

Decision: `BLOCKED`. Mechanical replay used the two artifact-relative reference directories in `manifest.json`, pinned FairShip commit `f73a305a86468c320665cf41b73582ce7e533181`, and ShipGeo identity `7588f2e62432574ed2487d6c71aadc4d03758c0c7e6b03ce3699079e0f11661a`. The manifest retains the source dataset hashes and ROOT hashes.

Reproduce with `python scripts/validate_afterms_fairship_transform.py --reference-dir <fairship_connector_v0>/p3_row_25740 --reference-dir <fairship_connector_v0>/p3_row_32029 --tgeo-audit-root artifacts/fairship_tgeo_audit_v0 --output-dir artifacts/afterms_fairship_transform_validation_v0`.

The references replay the asserted transform and match their TGeo anchors, but do not independently locate the after-MS plane in the FairShip frame; coordinate_physics_verified remains false. U0 label acquisition remains disallowed while the decision is BLOCKED.
