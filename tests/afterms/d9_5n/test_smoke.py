def test_smoke_import():
    from ship_muon_bg.afterms.d9_5 import nightly_runner as nr

    assert nr.SCHEMA_ID == "d9_5_nightly_runner_state_v0"
