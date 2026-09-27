import csv
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



def test_candidate_csv_preserves_generation_provenance(tmp_path: Path) -> None:
    path = Path(__file__).parents[1] / "scripts" / "run_utility_guided_fairship_pilot.py"
    spec = importlib.util.spec_from_file_location("pilot_csv", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    required = {"proposal_log_prob_space", "z_metadata_m", "model_checkpoint_hash"}
    assert required <= set(module.FIELDS)

    row = {field: None for field in module.FIELDS}
    row.update({
        "candidate_id": "c0",
        "proposal_log_prob_space": "physical_5d",
        "z_metadata_m": 28.905,
        "model_checkpoint_hash": "functional-hash",
    })
    module._write_tables(tmp_path, [row], [], [])

    with (tmp_path / "candidates.csv").open(newline="", encoding="utf-8") as stream:
        saved = next(csv.DictReader(stream))
    assert saved["proposal_log_prob_space"] == "physical_5d"
    assert saved["z_metadata_m"] == "28.905"
    assert saved["model_checkpoint_hash"] == "functional-hash"
