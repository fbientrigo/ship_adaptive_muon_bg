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
        "command_signature": [[
            "run_simScript.py", "--shieldName", "TRY_2026", "--strawDesign", "10",
            "--noSND", "--reproducible", "--validation",
        ]],
    }


def test_run_config_ignores_per_run_paths_and_seeds_but_not_geometry_options(tmp_path):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text(
        "/a/eminem_importer.py /a/input.pkl --output /a/out.root --format ttree "
        "--tree-name converted_ntuple --config /a/config.json\n"
        "/a/run_simScript.py --ttree -f /a/in.root -o /a/run -n 1 -s 1 -r 1 "
        "--tag connector --reproducible --validation --shieldName TRY_2026 --strawDesign 10 --noSND\n",
        encoding="utf-8",
    )
    second.write_text(first.read_text(encoding="utf-8").replace("/a/", "/b/").replace("-s 1 -r 1", "-s 2 -r 2"), encoding="utf-8")
    assert run_config(first) == run_config(second)
    second.write_text(second.read_text(encoding="utf-8").replace("TRY_2026", "OTHER"), encoding="utf-8")
    assert run_config(first)["command_signature"] != run_config(second)["command_signature"]
