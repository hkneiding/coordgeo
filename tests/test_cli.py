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
    assert "Metal center: Fe" in captured.out
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
    assert "Cutoff: auto" in captured.out
    assert "octahedral" in captured.out


def test_cli_tolerance_option_too_small_returns_error(capsys):
    xyz = os.path.join(EXAMPLES, "octahedral_example.xyz")
    ret = main([xyz, "--tolerance", "-10"])
    captured = capsys.readouterr()
    assert ret == 1
    assert captured.err.startswith("Error:")


def test_cli_version_flag(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert __version__ in captured.out
