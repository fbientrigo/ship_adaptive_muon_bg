from scripts.audit_fairship_tgeo import run_config


def test_run_config_records_declared_fairship_options(tmp_path):
    commands = tmp_path / "commands.txt"
    commands.write_text(
        "run_simScript.py --shieldName TRY_2026 --strawDesign 10 "
        "--noSND --reproducible --validation\n",
        encoding="utf-8",
    )
    assert run_config(commands) == {
        "shieldName": "TRY_2026",
        "strawDesign": "10",
        "noSND": True,
        "reproducible": True,
        "validation": True,
    }
