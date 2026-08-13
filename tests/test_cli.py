import os

import pytest

from coordgeo import __version__
from coordgeo.cli import main

EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "examples")


def test_cli_success(capsys):
    xyz = os.path.join(EXAMPLES, "octahedral_example.xyz")
    ret = main([xyz, "--cutoff", "2.5"])
    captured = capsys.readouterr()
    assert ret == 0
    assert "Fe (atom #1)" in captured.out
    assert "octahedral" in captured.out


def test_cli_top_option_limits_candidates(capsys):
    xyz = os.path.join(EXAMPLES, "octahedral_example.xyz")
    ret = main([xyz, "--cutoff", "2.5", "--top", "1"])
    captured = capsys.readouterr()
    assert ret == 0
    candidate_lines = [
        ln for ln in captured.out.splitlines() if ln.startswith("  ") and "shape measure" in ln
    ]
    assert len(candidate_lines) == 1


def test_cli_metal_symbol_and_index_options(capsys):
    xyz = os.path.join(EXAMPLES, "octahedral_example.xyz")
    ret = main([xyz, "--cutoff", "2.5", "--metal-symbol", "Fe"])
    assert ret == 0

    ret = main([xyz, "--cutoff", "2.5", "--metal-index", "1"])
    assert ret == 0


def test_cli_missing_file_returns_error(capsys):
    ret = main(["/no/such/file.xyz", "--cutoff", "2.5"])
    captured = capsys.readouterr()
    assert ret == 1
    assert captured.err.startswith("Error:")


def test_cli_too_small_cutoff_returns_error(capsys):
    xyz = os.path.join(EXAMPLES, "octahedral_example.xyz")
    ret = main([xyz, "--cutoff", "0.1"])
    captured = capsys.readouterr()
    assert ret == 1
    assert captured.err.startswith("Error:")


def test_cli_auto_cutoff_when_omitted(capsys):
    xyz = os.path.join(EXAMPLES, "octahedral_example.xyz")
    ret = main([xyz])
    captured = capsys.readouterr()
    assert ret == 0
    assert "CN=6" in captured.out
    assert "octahedral" in captured.out


def test_cli_tolerance_option_too_small_returns_error(capsys):
    xyz = os.path.join(EXAMPLES, "octahedral_example.xyz")
    ret = main([xyz, "--tolerance", "-10"])
    captured = capsys.readouterr()
    assert ret == 1
    assert captured.err.startswith("Error:")


def test_cli_window_option_pools_multiple_cns(tmp_path, capsys):
    # 5 core N ligands + 1 genuinely stretched N (gap-justified removal) +
    # 1 extra O outside the base cutoff (ceiling-permitted addition).
    xyz = tmp_path / "octa_plus_extra.xyz"
    xyz.write_text(
        "8\noctahedral core plus one distant extra N\n"
        "Fe 0.0 0.0 0.0\n"
        "N 2.08 0.0 0.0\n"
        "N -2.08 0.0 0.0\n"
        "N 0.0 2.08 0.0\n"
        "N 0.0 -2.08 0.0\n"
        "N 0.0 0.0 2.08\n"
        "N 0.0 0.0 -2.60\n"
        "O 0.3 0.3 3.05\n"
    )
    ret = main([str(xyz), "--cutoff", "2.65", "--window", "1"])
    captured = capsys.readouterr()
    assert ret == 0
    assert "CN=5" in captured.out
    assert "CN=6" in captured.out
    assert "CN=7" in captured.out


def test_cli_window_defaults_to_zero(capsys):
    xyz = os.path.join(EXAMPLES, "octahedral_example.xyz")
    with_flag = main([xyz, "--cutoff", "2.5", "--window", "0"])
    captured_with_flag = capsys.readouterr()
    without_flag = main([xyz, "--cutoff", "2.5"])
    captured_without_flag = capsys.readouterr()
    assert with_flag == without_flag == 0
    assert captured_with_flag.out == captured_without_flag.out


def test_cli_version_flag(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert __version__ in captured.out
