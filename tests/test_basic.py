import os
import numpy as np
import pytest

import coordgeo
from coordgeo.matcher import shape_measure, identify_geometry, EXACT_PERMUTATION_MAX_N
from coordgeo.geometries import GEOMETRIES, MIN_SUPPORTED_CN, MAX_SUPPORTED_CN

EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "examples")


def _random_rotation(rng):
    a = rng.standard_normal((3, 3))
    q, r = np.linalg.qr(a)
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, -1] *= -1
    return q


def test_octahedral_example():
    result = coordgeo.analyze(os.path.join(EXAMPLES, "octahedral_example.xyz"), cutoff=2.5)
    assert result.coordination_number == 6
    best = result.best_match()
    assert best.name == "octahedral"
    assert best.measure < 1e-6


def test_tetrahedral_example():
    result = coordgeo.analyze(os.path.join(EXAMPLES, "tetrahedral_example.xyz"), cutoff=2.5)
    assert result.coordination_number == 4
    best = result.best_match()
    assert best.name == "tetrahedral"
    assert best.measure < 1e-6


def test_distorted_square_planar_example():
    result = coordgeo.analyze(os.path.join(EXAMPLES, "square_planar_example.xyz"), cutoff=2.5)
    best = result.best_match()
    assert best.name == "square_planar"
    assert best.measure < 1.0  # small distortion, should still be a very good match


def test_metal_symbol_override():
    result = coordgeo.analyze(
        os.path.join(EXAMPLES, "octahedral_example.xyz"), cutoff=2.5, metal_symbol="Fe"
    )
    assert result.metal_symbol == "Fe"


def test_unknown_metal_symbol_raises():
    with pytest.raises(ValueError):
        coordgeo.analyze(
            os.path.join(EXAMPLES, "octahedral_example.xyz"), cutoff=2.5, metal_symbol="Zr"
        )


def test_too_small_cutoff_raises():
    with pytest.raises(ValueError):
        coordgeo.analyze(os.path.join(EXAMPLES, "octahedral_example.xyz"), cutoff=0.1)


def test_consistent_metal_symbol_and_index_is_accepted():
    result = coordgeo.analyze(
        os.path.join(EXAMPLES, "octahedral_example.xyz"), cutoff=2.5,
        metal_symbol="Fe", metal_index=0,
    )
    assert result.metal_symbol == "Fe"
    assert result.metal_index == 0


def test_conflicting_metal_symbol_and_index_raises():
    with pytest.raises(ValueError):
        coordgeo.analyze(
            os.path.join(EXAMPLES, "octahedral_example.xyz"), cutoff=2.5,
            metal_symbol="Zn", metal_index=0,
        )


def test_unsupported_coordination_number_raises(tmp_path):
    # 13 neighbors -> CN=13, one above MAX_SUPPORTED_CN (12).
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(13, 3))
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True) * 2.1

    lines = ["14", "CN=13 test (unsupported)", "Fe 0.0 0.0 0.0"]
    lines += [f"N {p[0]:.4f} {p[1]:.4f} {p[2]:.4f}" for p in pts]
    xyz = tmp_path / "cn13.xyz"
    xyz.write_text("\n".join(lines) + "\n")

    with pytest.raises(ValueError, match="not supported"):
        coordgeo.analyze(str(xyz), cutoff=2.5)


def test_shape_measure_self_is_zero_for_every_template():
    # Every reference template should score ~0 shape measure against itself.
    for cn, templates in GEOMETRIES.items():
        for name, verts in templates:
            measure, _ = shape_measure(verts, verts)
            assert measure < 1e-6, f"{name} (CN={cn}) failed self-match: {measure}"


def test_rotation_invariance():
    # A rotated tetrahedron should still match "tetrahedral" with ~0 measure.
    tetra = dict(GEOMETRIES[4])["tetrahedral"]
    theta = 0.7
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1],
    ])
    rotated = tetra @ R.T
    measure, _ = shape_measure(rotated, tetra)
    assert measure < 1e-6


def test_neighbor_sorting_and_vectors():
    structure = coordgeo.load_xyz(os.path.join(EXAMPLES, "tetrahedral_example.xyz"))
    metal_idx = coordgeo.find_metal_center(structure)
    neighbors = coordgeo.get_neighbors(structure, metal_idx, cutoff=2.5)
    distances = [n.distance for n in neighbors]
    assert distances == sorted(distances)
    assert len(neighbors) == 4


def test_supported_coordination_number_range():
    assert MIN_SUPPORTED_CN == 2
    assert MAX_SUPPORTED_CN == 12
    assert set(GEOMETRIES.keys()) == set(range(2, 13))


def test_icosahedral_example():
    result = coordgeo.analyze(
        os.path.join(EXAMPLES, "icosahedral_example.xyz"), cutoff=2.2, seed=0
    )
    assert result.coordination_number == 12
    best = result.best_match()
    assert best.name == "icosahedral"
    assert best.measure < 1e-3


@pytest.mark.parametrize("cn", [cn for cn in GEOMETRIES if cn > EXACT_PERMUTATION_MAX_N])
def test_icp_rotation_and_permutation_invariance_for_large_cn(cn):
    # For CN above the exact brute-force threshold, the Hungarian/ICP search
    # must still recover a ~0 shape measure for an arbitrarily rotated *and*
    # relabeled (permuted) copy of each template against itself.
    rng = np.random.default_rng(123)
    for name, verts in GEOMETRIES[cn]:
        R = _random_rotation(rng)
        rotated = verts @ R.T
        perm = rng.permutation(len(verts))
        rotated = rotated[perm]
        measure, _ = shape_measure(rotated, verts, rng=np.random.default_rng(0))
        assert measure < 1e-2, f"{name} (CN={cn}) failed rotation/permutation invariance: {measure}"


@pytest.mark.parametrize(
    "filename,expected_cn,expected_name",
    [
        ("octahedral_example.xyz", 6, "octahedral"),
        ("tetrahedral_example.xyz", 4, "tetrahedral"),
        ("square_planar_example.xyz", 4, "square_planar"),
        ("icosahedral_example.xyz", 12, "icosahedral"),
    ],
)
def test_auto_cutoff_matches_examples(filename, expected_cn, expected_name):
    # No cutoff given: falls back to covalent-radius-based auto cutoff.
    result = coordgeo.analyze(os.path.join(EXAMPLES, filename), seed=0)
    assert result.cutoff is None
    assert result.coordination_number == expected_cn
    assert result.best_match().name == expected_name
    assert result.best_match().measure < 1.0


def test_auto_cutoff_ignores_far_atom_with_unknown_element(tmp_path):
    xyz = tmp_path / "far_dummy.xyz"
    xyz.write_text(
        "4\nfar dummy atom of an unlisted element\n"
        "Fe 0.0 0.0 0.0\n"
        "N 2.1 0.0 0.0\n"
        "N -2.1 0.0 0.0\n"
        "Xx 50.0 0.0 0.0\n"
    )
    result = coordgeo.analyze(str(xyz), metal_symbol="Fe")
    assert [n.symbol for n in result.neighbors] == ["N", "N"]


def test_auto_cutoff_close_atom_with_unknown_element_raises(tmp_path):
    xyz = tmp_path / "close_dummy.xyz"
    xyz.write_text(
        "3\nclose dummy atom of an unlisted element\n"
        "Fe 0.0 0.0 0.0\n"
        "N 2.1 0.0 0.0\n"
        "Xx 2.0 2.0 0.0\n"
    )
    with pytest.raises(ValueError):
        coordgeo.analyze(str(xyz), metal_symbol="Fe")


def test_identify_geometry_seed_is_reproducible_for_large_cn():
    rng = np.random.default_rng(5)
    cn = 9
    verts = dict(GEOMETRIES[cn])["tricapped_trigonal_prismatic"]
    noisy = verts + rng.normal(scale=0.05, size=verts.shape)

    matches_a = identify_geometry(noisy, seed=42)
    matches_b = identify_geometry(noisy, seed=42)

    assert [m.measure for m in matches_a] == [m.measure for m in matches_b]
    assert [m.name for m in matches_a] == [m.name for m in matches_b]


def _octahedral_ase_atoms():
    ase = pytest.importorskip("ase")
    positions = [
        [0, 0, 0], [2.1, 0, 0], [-2.1, 0, 0], [0, 2.1, 0], [0, -2.1, 0], [0, 0, 2.1], [0, 0, -2.1],
    ]
    symbols = ["Fe", "N", "N", "N", "N", "N", "N"]
    return ase.Atoms(symbols=symbols, positions=positions)


def test_analyze_accepts_ase_atoms_with_fixed_cutoff():
    atoms = _octahedral_ase_atoms()
    result = coordgeo.analyze(atoms, cutoff=2.5)
    assert result.coordination_number == 6
    assert result.best_match().name == "octahedral"
    assert result.best_match().measure < 1e-6


def test_analyze_accepts_ase_atoms_with_auto_cutoff():
    atoms = _octahedral_ase_atoms()
    result = coordgeo.analyze(atoms)
    assert result.coordination_number == 6
    assert result.best_match().name == "octahedral"


def test_structure_from_ase_atoms_low_level():
    atoms = _octahedral_ase_atoms()
    structure = coordgeo.structure_from_ase_atoms(atoms)
    assert len(structure) == 7
    assert structure.symbols() == ["Fe", "N", "N", "N", "N", "N", "N"]


def test_analyze_accepts_pathlib_path():
    from pathlib import Path

    result = coordgeo.analyze(Path(EXAMPLES) / "octahedral_example.xyz", cutoff=2.5)
    assert result.coordination_number == 6


def test_analyze_rejects_unsupported_source_type():
    with pytest.raises(TypeError):
        coordgeo.analyze(42, cutoff=2.5)


def _octahedral_plus_extras_xyz(tmp_path):
    # 5 core N ligands near ideal octahedral positions (2.08 A, no
    # meaningful spread among themselves) + 1 genuinely stretched N (2.60 A,
    # a real Jahn-Teller-like gap from the core) + 2 extra O atoms further
    # out still (outside a 2.65 A cutoff).
    xyz = tmp_path / "octa_plus_extras.xyz"
    xyz.write_text(
        "9\ndistorted octahedron: 5 core N + 1 stretched N + 2 distant O\n"
        "Fe 0.0 0.0 0.0\n"
        "N 2.08 0.0 0.0\n"
        "N -2.08 0.0 0.0\n"
        "N 0.0 2.08 0.0\n"
        "N 0.0 -2.08 0.0\n"
        "N 0.0 0.0 2.08\n"
        "N 0.0 0.0 -2.60\n"
        "O 0.3 0.3 3.05\n"
        "O -0.3 -0.3 3.15\n"
    )
    return str(xyz)


def test_analyze_window_default_reproduces_original_behavior():
    xyz = os.path.join(EXAMPLES, "octahedral_example.xyz")
    default = coordgeo.analyze(xyz, cutoff=2.5)
    explicit_zero = coordgeo.analyze(xyz, cutoff=2.5, window=0)

    assert default.coordination_number == explicit_zero.coordination_number == 6
    assert [m.name for m in default.matches] == [m.name for m in explicit_zero.matches]
    assert [m.measure for m in default.matches] == [m.measure for m in explicit_zero.matches]
    assert default.best_match().name == "octahedral"
    assert default.best_match().measure < 1e-6


def test_analyze_with_window_pools_matches_across_coordination_numbers(tmp_path):
    xyz = _octahedral_plus_extras_xyz(tmp_path)
    result = coordgeo.analyze(xyz, cutoff=2.65, window=2)

    # `neighbors`/`coordination_number` still reflect the base (window=0) set.
    assert result.coordination_number == 6

    cns_tested = sorted({m.coordination_number for m in result.matches})
    # CN=4 is absent: removing 2 neighbors would have to cut into the
    # tightly-clustered 5-N core (no gap between them), which the
    # distance-gap requirement now blocks. CN=5 (dropping only the
    # genuinely stretched 6th N, which *does* clear the gap) is still
    # reachable.
    assert cns_tested == [5, 6, 7, 8]

    # Pooled matches are sorted best-first regardless of which CN they came from.
    assert result.matches == sorted(result.matches, key=lambda m: m.measure)

    octahedral = next(m for m in result.matches if m.name == "octahedral")
    assert octahedral.coordination_number == 6
    # ...clearly better than CN=7/8, which necessarily include the
    # poorly-fitting extra O atom(s) -- unlike the CN=5 subset (dropping
    # a genuine outlier), a same-symmetry-family smaller-CN match can
    # still coincidentally score close to or better than the full CN
    # purely from having fewer points to fit (see the caveat in
    # analyze()'s docstring: raw measures aren't fully comparable across
    # different CN even when the removal itself was well-justified).
    cn7_best = min(m.measure for m in result.matches if m.coordination_number == 7)
    cn8_best = min(m.measure for m in result.matches if m.coordination_number == 8)
    assert cn7_best > octahedral.measure
    assert cn8_best > octahedral.measure


def test_analyze_window_blocks_removal_without_a_distance_gap():
    # octahedral_example.xyz is a *perfectly* idealized octahedron -- all 6
    # Fe-N distances are exactly tied at 2.10 A, so there is no genuine gap
    # anywhere. This is the exact scenario that used to misreport
    # vacant_octahedral (CN=5) as the best match: window's remove side must
    # not offer any smaller CN here, since none of the 6 ligands are
    # meaningfully separated from the rest.
    xyz = os.path.join(EXAMPLES, "octahedral_example.xyz")
    result = coordgeo.analyze(xyz, cutoff=2.2, window=2)
    cns_tested = sorted({m.coordination_number for m in result.matches})
    assert cns_tested == [6]  # no atoms to add either, but the point here is: no removal
    assert result.best_match().name == "octahedral"
    assert result.best_match().measure < 1e-6


def test_analyze_window_allows_removal_with_a_genuine_distance_gap(tmp_path):
    # Deliberately staircased distances (0.5 A apart, each well beyond the
    # default 0.4 A tolerance) so every sequential removal step clears the
    # gap requirement, all the way down to CN=1 (invalid -- excluded from
    # matches, not raised, since the base CN=6 and several window CNs are
    # still valid).
    xyz = tmp_path / "staircase.xyz"
    xyz.write_text(
        "7\nstaircase distances to exercise full sequential removal\n"
        "Fe 0.0 0.0 0.0\n"
        "N 2.0 0.0 0.0\n"
        "N -2.5 0.0 0.0\n"
        "N 0.0 3.0 0.0\n"
        "N 0.0 -3.5 0.0\n"
        "N 0.0 0.0 4.0\n"
        "N 0.0 0.0 -4.5\n"
    )
    result = coordgeo.analyze(str(xyz), cutoff=4.5, window=5)
    cns_tested = sorted({m.coordination_number for m in result.matches})
    assert 1 not in cns_tested
    assert cns_tested == [2, 3, 4, 5, 6]


def test_analyze_negative_window_raises():
    xyz = os.path.join(EXAMPLES, "octahedral_example.xyz")
    with pytest.raises(ValueError):
        coordgeo.analyze(xyz, cutoff=2.5, window=-1)


def test_analyze_window_rescues_invalid_base_cn():
    # cutoff=1.0 gives a base CN=0 (invalid on its own), but window=5 reaches
    # out to the 6 real N ligands at 2.10 A -- several of which give valid CNs.
    xyz = os.path.join(EXAMPLES, "octahedral_example.xyz")
    result = coordgeo.analyze(xyz, cutoff=1.0, window=5)
    assert result.coordination_number == 0  # base (window=0) set is still empty
    assert result.matches                   # but the window rescued it
    cns_tested = sorted({m.coordination_number for m in result.matches})
    assert cns_tested == [2, 3, 4, 5]


def test_analyze_window_raises_only_if_nothing_in_window_is_valid():
    xyz = os.path.join(EXAMPLES, "octahedral_example.xyz")
    # window=0: no rescue possible, base CN=0 is the only thing tested.
    with pytest.raises(ValueError, match="not supported"):
        coordgeo.analyze(xyz, cutoff=1.0, window=0)
    # window=1: still nowhere near enough to reach a real ligand (2.10 A away).
    with pytest.raises(ValueError, match="not supported"):
        coordgeo.analyze(xyz, cutoff=1.0, window=1)


def test_analyze_window_add_side_respects_distance_ceiling(tmp_path):
    # Fe with 2 real N ligands at 2.10 A (base CN=2), plus one N at 6.0 A --
    # far beyond any plausible Fe-N bond, even though it's the closest
    # excluded atom. window's add side should refuse to reach for it.
    xyz = tmp_path / "ceiling_test.xyz"
    xyz.write_text(
        "4\nFe with 2 real N ligands plus one implausibly distant N\n"
        "Fe 0.0 0.0 0.0\n"
        "N 2.1 0.0 0.0\n"
        "N -2.1 0.0 0.0\n"
        "N 0.0 0.0 6.0\n"
    )
    result = coordgeo.analyze(str(xyz), cutoff=2.5, window=1)
    cns_tested = sorted({m.coordination_number for m in result.matches})
    assert cns_tested == [2]


def test_summary_header_includes_metal_center_and_no_other_sections():
    xyz = os.path.join(EXAMPLES, "octahedral_example.xyz")
    summary = coordgeo.analyze(xyz, cutoff=2.5).summary()
    lines = summary.splitlines()
    assert lines[0].startswith("Candidate geometries for Fe (atom #1)")
    # Only the header + one row per candidate -- no separate metal/cutoff/
    # window/coordination-number/neighbor-list sections.
    assert "Neighbors:" not in summary
    assert "Cutoff" not in summary
    assert "Window" not in summary
    assert all(ln == lines[0] or ln.strip().startswith("CN=") for ln in lines)


def test_summary_candidate_table_shows_cn_and_is_sorted_best_first():
    xyz = os.path.join(EXAMPLES, "octahedral_example.xyz")
    summary = coordgeo.analyze(xyz, cutoff=2.5).summary()
    candidate_lines = [ln for ln in summary.splitlines() if ln.strip().startswith("CN=")]
    assert candidate_lines[0].startswith("  CN=6 ")
    assert candidate_lines[0].endswith("<-- best match")
