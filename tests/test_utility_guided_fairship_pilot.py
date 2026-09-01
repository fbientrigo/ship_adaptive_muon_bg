import importlib.util
from pathlib import Path


def test_fixed_pilot_contract_has_twelve_unit_transport_slots() -> None:
    path = Path(__file__).parents[1] / "scripts" / "run_utility_guided_fairship_pilot.py"
    spec = importlib.util.spec_from_file_location("pilot", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert module.N_PER_PDG == 6
    assert module.TRANSFORM.status == "PROVISIONAL"
    assert module.TILT_ID == "UA_d0p1_a04"
    record = module.CandidateInjectionRecord.generated("nf", px=1, py=2, pz=3, x=4, y=5, z=28.905, pdg_id=13)
    assert record.physical_source_weight is None
    assert record.fairship_transport_weight == 1.0
