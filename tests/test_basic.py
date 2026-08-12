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
