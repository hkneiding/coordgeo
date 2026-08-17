"""Idealized reference coordination geometries (vertex templates).

Each template is a set of unit-ish vectors radiating from a central point
(the metal atom sits at the origin). Templates are stored *before*
normalization; :func:`get_geometries` normalizes every template to unit
RMS radius so that shape (angular arrangement), not size, is compared.

This is not an exhaustive crystallographic reference set (see the SHAPE
program / continuous shape measures literature for the canonical, more
complete version) but covers the common coordination geometries for
CN = 2 through 12, which is enough for the vast majority of mononuclear
metal complexes. The dictionary is easy to extend -- add a new
``(name, Nx3 array)`` entry under the right coordination number.

Many higher-CN templates are built systematically from smaller ones, in
line with standard nomenclature: a "vacant" polyhedron removes a vertex
from a larger one (e.g. vacant_octahedral = octahedral minus one vertex)
and a "capped" polyhedron adds an extra vertex at a face center (e.g.
capped_octahedron = octahedral plus one vertex).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

Template = Tuple[str, np.ndarray]


def _unit(v: np.ndarray) -> np.ndarray:
    """Scale a vector to unit length.

    Parameters
    ----------
    v : array-like, shape (3,)
        Vector to normalize.

    Returns
    -------
    numpy.ndarray, shape (3,)
        `v / ||v||`.
    """
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def _ring(n: int, z: float = 0.0, r: float = 1.0, phase: float = 0.0) -> np.ndarray:
    """Generate n points evenly spaced on a ring.

    Parameters
    ----------
    n : int
        Number of points on the ring.
    z : float, default 0.0
        Height (z-coordinate) shared by every point.
    r : float, default 1.0
        Ring radius.
    phase : float, default 0.0
        Angular offset (radians) of the first point; subsequent points are
        spaced `2*pi/n` apart from it.

    Returns
    -------
    (n, 3) numpy.ndarray
        Row i is `(r*cos(theta_i), r*sin(theta_i), z)`.
    """
    angles = phase + 2 * np.pi * np.arange(n) / n
    pts = np.stack([r * np.cos(angles), r * np.sin(angles), np.full(n, z)], axis=1)
    return pts


def _raw_geometries() -> Dict[int, List[Template]]:
    """Build the un-normalized reference geometry templates.

    Returns
    -------
    dict of int to list of (str, numpy.ndarray)
        Maps each supported coordination number to its list of
        `(name, (N, 3) vertex array)` templates, before RMS-radius
        normalization (see `get_geometries`).
    """
    geoms: Dict[int, List[Template]] = {}

    # ---------------------------------------------------------------- CN=2
    geoms[2] = [
        ("linear", np.array([[1, 0, 0], [-1, 0, 0]], dtype=float)),
        ("bent", np.array([
            [np.sin(np.radians(54.75)), 0, np.cos(np.radians(54.75))],
            [-np.sin(np.radians(54.75)), 0, np.cos(np.radians(54.75))],
        ])),
    ]

    # ---------------------------------------------------------------- CN=3
    geoms[3] = [
        ("trigonal_planar", _ring(3, z=0.0)),
        ("trigonal_pyramidal", np.array([
            [np.sin(np.radians(100)) * np.cos(a), np.sin(np.radians(100)) * np.sin(a), np.cos(np.radians(100))]
            for a in (0, 2 * np.pi / 3, 4 * np.pi / 3)
        ])),
        ("t_shaped", np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0]], dtype=float)),
    ]

    # ---------------------------------------------------------------- CN=4
    geoms[4] = [
        ("tetrahedral", np.array([
            [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1],
        ], dtype=float)),
        ("square_planar", np.array([
            [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0],
        ], dtype=float)),
        ("seesaw", np.array([
            [0, 0, 1], [0, 0, -1],
            [np.cos(np.radians(60)), np.sin(np.radians(60)), 0],
            [np.cos(np.radians(60)), -np.sin(np.radians(60)), 0],
        ], dtype=float)),
        # Trigonal bipyramid with one equatorial vertex removed.
        ("vacant_trigonal_bipyramidal", np.array([
            [0, 0, 1], [0, 0, -1],
            [1, 0, 0],
            [np.cos(np.radians(120)), np.sin(np.radians(120)), 0],
        ], dtype=float)),
    ]

    # ---------------------------------------------------------------- CN=5
    geoms[5] = [
        ("trigonal_bipyramidal", np.vstack([
            [[0, 0, 1], [0, 0, -1]],
            _ring(3, z=0.0),
        ])),
        ("square_pyramidal", np.vstack([
            [[0, 0, 1.1]],
            np.array([[1, 0, -0.3], [0, 1, -0.3], [-1, 0, -0.3], [0, -1, -0.3]], dtype=float),
        ])),
        ("pentagonal_planar", _ring(5, z=0.0)),
        # Octahedron with one vertex removed.
        ("vacant_octahedral", np.array([
            [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1],
        ], dtype=float)),
    ]

    # ---------------------------------------------------------------- CN=6
    geoms[6] = [
        ("octahedral", np.array([
            [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
        ], dtype=float)),
        ("trigonal_prismatic", np.vstack([
            _ring(3, z=0.8, phase=0.0),
            _ring(3, z=-0.8, phase=0.0),
        ])),
        ("pentagonal_pyramidal", np.vstack([
            [[0, 0, 1.2]],
            _ring(5, z=-0.3),
        ])),
        ("hexagonal_planar", _ring(6, z=0.0)),
    ]

    # ---------------------------------------------------------------- CN=7
    geoms[7] = [
        ("pentagonal_bipyramidal", np.vstack([
            [[0, 0, 1], [0, 0, -1]],
            _ring(5, z=0.0),
        ])),
        ("capped_octahedron", np.vstack([
            [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
            [_unit([1, 1, 1]) * 1.0],
        ])),
        ("capped_trigonal_prism", np.vstack([
            _ring(3, z=0.8, phase=0.0),
            _ring(3, z=-0.8, phase=0.0),
            [[0, -1.3, 0]],
        ])),
        ("hexagonal_pyramidal", np.vstack([
            [[0, 0, 1.2]],
            _ring(6, z=-0.3),
        ])),
    ]

    # ---------------------------------------------------------------- CN=8
    geoms[8] = [
        ("cubic", np.array([
            [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
            [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        ], dtype=float)),
        ("square_antiprismatic", np.vstack([
            _ring(4, z=0.8, phase=0.0),
            _ring(4, z=-0.8, phase=np.pi / 4),
        ])),
        ("hexagonal_bipyramidal", np.vstack([
            [[0, 0, 1], [0, 0, -1]],
            _ring(6, z=0.0),
        ])),
        ("dodecahedral", np.array([
            [0.964, 0, 0.266], [-0.964, 0, 0.266],
            [0, 0.964, -0.266], [0, -0.964, -0.266],
            [0.628, 0.628, -0.460], [0.628, -0.628, -0.460],
            [-0.628, 0.628, -0.460], [-0.628, -0.628, -0.460],
        ], dtype=float)),
        # Trigonal prism with two of its three rectangular faces capped
        # (face centers sit at 60 degree intervals; two of the three are used).
        ("bicapped_trigonal_prismatic", np.vstack([
            _ring(3, z=0.8, phase=0.0),
            _ring(3, z=-0.8, phase=0.0),
            np.array([
                [1.3 * np.cos(np.radians(60)), 1.3 * np.sin(np.radians(60)), 0],
                [1.3 * np.cos(np.radians(180)), 1.3 * np.sin(np.radians(180)), 0],
            ], dtype=float),
        ])),
    ]

    # ---------------------------------------------------------------- CN=9
    geoms[9] = [
        # Trigonal prism with all three rectangular faces capped.
        ("tricapped_trigonal_prismatic", np.vstack([
            _ring(3, z=0.8, phase=0.0),
            _ring(3, z=-0.8, phase=0.0),
            _ring(3, z=0.0, r=1.3, phase=np.pi / 3),
        ])),
        # Square antiprism with one square face capped.
        ("capped_square_antiprismatic", np.vstack([
            _ring(4, z=0.8, phase=0.0),
            _ring(4, z=-0.8, phase=np.pi / 4),
            [[0, 0, 1.3]],
        ])),
        ("heptagonal_bipyramidal", np.vstack([
            [[0, 0, 1], [0, 0, -1]],
            _ring(7, z=0.0),
        ])),
    ]

    # --------------------------------------------------------------- CN=10
    geoms[10] = [
        ("pentagonal_prismatic", np.vstack([
            _ring(5, z=0.8, phase=0.0),
            _ring(5, z=-0.8, phase=0.0),
        ])),
        ("pentagonal_antiprismatic", np.vstack([
            _ring(5, z=0.8, phase=0.0),
            _ring(5, z=-0.8, phase=np.pi / 5),
        ])),
        # Cube with two opposite faces capped.
        ("bicapped_cube", np.vstack([
            np.array([
                [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
                [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
            ], dtype=float),
            [[0, 0, 1.6], [0, 0, -1.6]],
        ])),
    ]

    # --------------------------------------------------------------- CN=11
    geoms[11] = [
        # Pentagonal prism with one pentagonal face capped.
        ("capped_pentagonal_prismatic", np.vstack([
            _ring(5, z=0.8, phase=0.0),
            _ring(5, z=-0.8, phase=0.0),
            [[0, 0, 1.3]],
        ])),
        # Pentagonal antiprism with one pentagonal face capped.
        ("capped_pentagonal_antiprismatic", np.vstack([
            _ring(5, z=0.8, phase=0.0),
            _ring(5, z=-0.8, phase=np.pi / 5),
            [[0, 0, 1.3]],
        ])),
    ]

    # --------------------------------------------------------------- CN=12
    _phi = (1 + np.sqrt(5)) / 2
    geoms[12] = [
        ("icosahedral", np.array([
            [0, 1, _phi], [0, 1, -_phi], [0, -1, _phi], [0, -1, -_phi],
            [1, _phi, 0], [1, -_phi, 0], [-1, _phi, 0], [-1, -_phi, 0],
            [_phi, 0, 1], [_phi, 0, -1], [-_phi, 0, 1], [-_phi, 0, -1],
        ], dtype=float)),
        ("cuboctahedron", np.array([
            [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1],
            [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],
            [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
        ], dtype=float)),
        ("hexagonal_prismatic", np.vstack([
            _ring(6, z=0.8, phase=0.0),
            _ring(6, z=-0.8, phase=0.0),
        ])),
    ]

    return geoms


def get_geometries() -> Dict[int, List[Template]]:
    """Build the full set of normalized reference geometry templates.

    Every template is centered on the origin (the metal position) and
    scaled so the RMS distance of its vertices from the origin is 1.

    Returns
    -------
    dict of int to list of (str, numpy.ndarray)
        `{coordination_number: [(name, normalized_vertices), ...]}`, where
        each `normalized_vertices` is an `(N, 3)` array with unit RMS
        radius.
    """
    raw = _raw_geometries()
    normalized: Dict[int, List[Template]] = {}
    for cn, templates in raw.items():
        norm_templates = []
        for name, pts in templates:
            pts = np.asarray(pts, dtype=float)
            scale = np.sqrt(np.mean(np.sum(pts ** 2, axis=1)))
            norm_templates.append((name, pts / scale))
        normalized[cn] = norm_templates
    return normalized


GEOMETRIES = get_geometries()

MAX_SUPPORTED_CN = max(GEOMETRIES.keys())
MIN_SUPPORTED_CN = min(GEOMETRIES.keys())

GEOMETRY_BY_NAME: Dict[str, Tuple[int, np.ndarray]] = {
    name: (cn, template) for cn, templates in GEOMETRIES.items() for name, template in templates
}


def get_geometry_by_name(name: str) -> Tuple[int, np.ndarray]:
    """Look up a named reference geometry's coordination number and template.

    Parameters
    ----------
    name : str
        Geometry name as used in GEOMETRIES, e.g. "octahedral".

    Returns
    -------
    (int, numpy.ndarray)
        `(coordination_number, normalized_vertices)` for that geometry.

    Raises
    ------
    ValueError
        If `name` isn't a key in GEOMETRY_BY_NAME.
    """
    if name not in GEOMETRY_BY_NAME:
        raise ValueError(
            f"Unknown geometry name {name!r}. Valid names: {sorted(GEOMETRY_BY_NAME)}."
        )
    return GEOMETRY_BY_NAME[name]
