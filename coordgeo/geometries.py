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

Naming follows standard nomenclature: a "vacant" polyhedron removes a
vertex from a larger one (e.g. vacant_octahedral = octahedral minus one
vertex), a "capped" polyhedron adds an extra vertex at a face center (e.g.
capped_octahedron = octahedral plus one vertex), and "biaugmented" adds
two.

Most templates' vertex coordinates are sourced directly from cosymlib/
SHAPE 2.1's published reference structures (via Q-Shape's implementation),
converted to this module's metal-at-origin convention: their central-atom
point subtracted out, since cosymlib centers on the ligand+metal centroid
instead (see the coordgeo/Q-Shape methodology discussion for why coordgeo
uses the former). A handful of templates need no such conversion because
they're exact by construction regardless of source -- flat regular
n-gons (trigonal_planar, pentagonal_planar, hexagonal_planar), shapes
with only one possible angle (t_shaped, square_planar), and the Platonic
solids (tetrahedral, cubic, octahedral, icosahedral) plus cuboctahedron.

Where a named shape has multiple published variants, only one is included
-- never both, to keep the "one template per name" mapping in
GEOMETRY_BY_NAME unambiguous. Two kinds of variant pairs show up in
SHAPE/cosymlib's reference set, and they're resolved differently:

* An equal-M-L-bond-length ("ideal", spherical) version alongside a
  separate equal-edge-length Johnson-solid version of the *same* named
  shape (SHAPE gives these distinct codes, e.g. TBPY-5 vs. JTBPY-5). Here
  the equal-bond-length version is preferred, since it's the one actually
  used throughout the CShM literature and by SHAPE's own default output
  -- e.g. trigonal_bipyramidal, pentagonal_pyramidal (CN=6),
  pentagonal_bipyramidal (CN=7), biaugmented_trigonal_prismatic (CN=8),
  tricapped_trigonal_prismatic and capped_square_antiprismatic (CN=9) are
  all the equal-bond-length version, not the Johnson solid.
* A shape that exists *only* as a Johnson solid, with no equal-bond-length
  counterpart published (e.g. snub_disphenoid, bicapped_cube,
  bicapped_square_antiprismatic, capped_pentagonal_prismatic,
  capped_pentagonal_antiprismatic) -- here the Johnson solid is simply
  the shape, not a stylistic choice between two options.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

Template = Tuple[str, np.ndarray]


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
        # Vacant tetrahedron (109.47 degrees between ligands), exact by
        # construction (the tetrahedral angle is unambiguous).
        ("v_shaped", np.array([
            [np.sin(np.radians(54.75)), 0, np.cos(np.radians(54.75))],
            [-np.sin(np.radians(54.75)), 0, np.cos(np.radians(54.75))],
        ])),
        # Tetravacant octahedron (90 degrees between ligands), per
        # cosymlib/SHAPE 2.1's vOC-2 reference structure.
        ("l_shaped", np.array([[1, 0, 0], [0, 1, 0]], dtype=float)),
    ]

    # ---------------------------------------------------------------- CN=3
    geoms[3] = [
        ("trigonal_planar", _ring(3, z=0.0)),
        # Vacant tetrahedron. Vertices per cosymlib/SHAPE 2.1's vT-3
        # reference structure (109.47 degree L-M-L angle).
        ("trigonal_pyramidal", np.array([
            [1.13707, 0.0, 0.402015],
            [-0.568535, 0.984732, 0.402015],
            [-0.568535, -0.984732, 0.402015],
        ], dtype=float)),
        ("t_shaped", np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0]], dtype=float)),
        # fac-trivacant octahedron: the 3 mutually-cis vertices of an
        # octahedron (the coordinate axes), per cosymlib/SHAPE 2.1's
        # fac-vOC-3 reference structure. Distinct from trigonal_pyramidal
        # (that one is a vacant tetrahedron, 109.47 degree angles; this one
        # is 90 degrees, derived from the octahedron instead).
        ("fac_trivacant_octahedral", np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)),
    ]

    # ---------------------------------------------------------------- CN=4
    geoms[4] = [
        ("tetrahedral", np.array([
            [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1],
        ], dtype=float)),
        ("square_planar", np.array([
            [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0],
        ], dtype=float)),
        # Vertices per cosymlib/SHAPE 2.1's SS-4 reference structure.
        ("seesaw", np.array([
            [0.0, 0.0, -1.178511],
            [1.178511, 0.0, 0.0],
            [0.0, 1.178511, 0.0],
            [0.0, 0.0, 1.178511],
        ], dtype=float)),
        # Trigonal bipyramid with one equatorial vertex removed. Vertices
        # per cosymlib/SHAPE 2.1's vTBPY-4 reference structure.
        ("vacant_trigonal_bipyramidal", np.array([
            [0.0, 0.0, -1.147079],
            [1.147079, 0.0, 0.0],
            [-0.573539, 0.993399, 0.0],
            [-0.573539, -0.993399, 0.0],
        ], dtype=float)),
    ]

    # ---------------------------------------------------------------- CN=5
    geoms[5] = [
        # Equal M-L bond lengths (90/120 degree angles), the canonical
        # reference used throughout the CShM literature -- not the
        # equal-edge-length Johnson solid variant (J12), which SHAPE lists
        # separately as JTBPY-5. Vertices per cosymlib/SHAPE 2.1's
        # TBPY-5 reference structure.
        ("trigonal_bipyramidal", np.array([
            [0.0, 0.0, -1.095445],
            [1.095445, 0.0, 0.0],
            [-0.547723, 0.948683, 0.0],
            [-0.547723, -0.948683, 0.0],
            [0.0, 0.0, 1.095445],
        ], dtype=float)),
        # SPY-5, distinct from vacant_octahedral (below), which is
        # separately SHAPE's vOC-5 (= Johnson J1). Vertices per
        # cosymlib/SHAPE 2.1's SPY-5 reference structure.
        ("square_pyramidal", np.array([
            [0.0, 0.0, 1.095445],
            [1.06066, 0.0, -0.273861],
            [0.0, 1.06066, -0.273861],
            [-1.06066, 0.0, -0.273861],
            [0.0, -1.06066, -0.273861],
        ], dtype=float)),
        ("pentagonal_planar", _ring(5, z=0.0)),
        # Octahedron with one vertex removed -- this is exact by
        # construction and also happens to equal Johnson solid J1 (vOC-5).
        ("vacant_octahedral", np.array([
            [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1],
        ], dtype=float)),
    ]

    # ---------------------------------------------------------------- CN=6
    geoms[6] = [
        ("octahedral", np.array([
            [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
        ], dtype=float)),
        # Vertices per cosymlib/SHAPE 2.1's TPR-6 reference structure.
        ("trigonal_prismatic", np.array([
            [0.816497, 0.0, -0.707107],
            [-0.408248, 0.707107, -0.707107],
            [-0.408248, -0.707107, -0.707107],
            [0.816497, 0.0, 0.707107],
            [-0.408248, 0.707107, 0.707107],
            [-0.408248, -0.707107, 0.707107],
        ], dtype=float)),
        # Equal M-L bond lengths, the canonical CShM-literature reference --
        # not the equal-edge-length Johnson solid variant (J2), which SHAPE
        # lists separately as JPPY-6. Vertices per cosymlib/SHAPE 2.1's
        # PPY-6 reference structure.
        ("pentagonal_pyramidal", np.array([
            [0.0, 0.0, -1.093216],
            [1.093216, 0.0, 0.0],
            [0.337822, 1.039711, 0.0],
            [-0.884431, 0.642576, 0.0],
            [-0.884431, -0.642576, 0.0],
            [0.337822, -1.039711, 0.0],
        ], dtype=float)),
        ("hexagonal_planar", _ring(6, z=0.0)),
    ]

    # ---------------------------------------------------------------- CN=7
    geoms[7] = [
        # Equal M-L bond lengths, the canonical CShM-literature reference --
        # not the equal-edge-length Johnson solid variant (J13), which
        # SHAPE lists separately as JPBPY-7. Vertices per cosymlib/SHAPE
        # 2.1's PBPY-7 reference structure.
        ("pentagonal_bipyramidal", np.array([
            [0.0, 0.0, -1.069045],
            [1.069045, 0.0, 0.0],
            [0.330353, 1.016722, 0.0],
            [-0.864876, 0.628369, 0.0],
            [-0.864876, -0.628369, 0.0],
            [0.330353, -1.016722, 0.0],
            [0.0, 0.0, 1.069045],
        ], dtype=float)),
        # Vertices per cosymlib/SHAPE 2.1's COC-7 reference structure.
        ("capped_octahedron", np.array([
            [0.0, 0.0, 1.070845],
            [0.0, -1.046937, 0.225017],
            [0.906674, 0.523469, 0.225017],
            [-0.906674, 0.523469, 0.225017],
            [0.672965, -0.388536, -0.736796],
            [-0.672965, -0.388536, -0.736796],
            [0.0, 0.777073, -0.736796],
        ], dtype=float)),
        # Vertices per cosymlib/SHAPE 2.1's CTPR-7 reference structure.
        ("capped_trigonal_prism", np.array([
            [0.0, 0.0, 1.0704],
            [0.735248, 0.735248, 0.254124],
            [-0.735248, 0.735248, 0.254124],
            [0.735248, -0.735248, 0.254124],
            [-0.735248, -0.735248, 0.254124],
            [0.660961, 0.0, -0.841955],
            [-0.660961, 0.0, -0.841955],
        ], dtype=float)),
        # Vertices per cosymlib/SHAPE 2.1's HPY-7 reference structure.
        ("hexagonal_pyramidal", np.array([
            [0.0, 0.0, -1.07872],
            [1.07872, 0.0, 0.0],
            [0.53936, 0.934199, 0.0],
            [-0.53936, 0.934199, 0.0],
            [-1.07872, 0.0, 0.0],
            [-0.53936, -0.934199, 0.0],
            [0.53936, -0.934199, 0.0],
        ], dtype=float)),
    ]

    # ---------------------------------------------------------------- CN=8
    geoms[8] = [
        ("cubic", np.array([
            [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
            [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        ], dtype=float)),
        # Vertices per cosymlib/SHAPE 2.1's SAPR-8 reference structure.
        ("square_antiprismatic", np.array([
            [0.644649, 0.644649, -0.542083],
            [-0.644649, 0.644649, -0.542083],
            [-0.644649, -0.644649, -0.542083],
            [0.644649, -0.644649, -0.542083],
            [0.911672, 0.0, 0.542083],
            [0.0, 0.911672, 0.542083],
            [-0.911672, 0.0, 0.542083],
            [0.0, -0.911672, 0.542083],
        ], dtype=float)),
        # Vertices per cosymlib/SHAPE 2.1's HBPY-8 reference structure.
        ("hexagonal_bipyramidal", np.array([
            [0.0, 0.0, -1.06066],
            [1.06066, 0.0, 0.0],
            [0.53033, 0.918559, 0.0],
            [-0.53033, 0.918559, 0.0],
            [-1.06066, 0.0, 0.0],
            [-0.53033, -0.918559, 0.0],
            [0.53033, -0.918559, 0.0],
            [0.0, 0.0, 1.06066],
        ], dtype=float)),
        # Triangular dodecahedron / bisdisphenoid. Vertices per
        # cosymlib/SHAPE 2.1's TDD-8 reference structure.
        ("dodecahedral", np.array([
            [-0.636106, 0.0, 0.848768],
            [0.0, -0.993211, 0.372147],
            [0.636106, 0.0, 0.848768],
            [0.0, 0.993211, 0.372147],
            [-0.993211, 0.0, -0.372147],
            [0.0, -0.636106, -0.848768],
            [0.993211, 0.0, -0.372147],
            [0.0, 0.636106, -0.848768],
        ], dtype=float)),
        # Equal M-L bond lengths, the canonical CShM-literature reference --
        # not the equal-edge-length Johnson solid variant (J50), which
        # SHAPE lists separately as JBTP-8. Named biaugmented (not
        # bicapped) to match SHAPE 2.1's own terminology for this shape
        # (BTPR-8, "Biaugmented Trigonal Prism") -- unlike
        # bicapped_cube/bicapped_square_antiprismatic (CN=10), where SHAPE
        # itself does use "bicapped". Vertices per cosymlib/SHAPE 2.1's
        # BTPR-8 reference structure.
        ("biaugmented_trigonal_prismatic", np.array([
            [0.699238, 0.0, 0.80741],
            [-0.699238, 0.0, 0.80741],
            [0.699238, 0.699238, -0.403705],
            [-0.699238, 0.699238, -0.403705],
            [0.699238, -0.699238, -0.403705],
            [-0.699238, -0.699238, -0.403705],
            [0.0, 0.925005, 0.534052],
            [0.0, -0.925005, 0.534052],
        ], dtype=float)),
        # Johnson solid J84 (D2d) -- equal edge lengths by construction.
        # Vertices per cosymlib/SHAPE 2.1's JSD-8 reference structure.
        ("snub_disphenoid", np.array([
            [-0.652225622594, 0.000000000000, -1.022598826988],
            [0.652225622594, 0.000000000000, -1.022598826988],
            [0.840828401428, 0.000000000000, 0.268145244516],
            [-0.840828401428, 0.000000000000, 0.268145244516],
            [0.000000000000, -0.652225622594, 1.022598102293],
            [0.000000000000, 0.652225622594, 1.022598102293],
            [0.000000000000, -0.840828401428, -0.268144664760],
            [0.000000000000, 0.840828401428, -0.268144664760],
        ], dtype=float)),
    ]

    # ---------------------------------------------------------------- CN=9
    geoms[9] = [
        # Equal M-L bond lengths, the canonical CShM-literature reference --
        # not the equal-edge-length Johnson solid variant (J51), which
        # SHAPE lists separately as JTCTPR-9. Vertices per cosymlib/SHAPE
        # 2.1's TCTPR-9 reference structure.
        ("tricapped_trigonal_prismatic", np.array([
            [0.702728, 0.0, 0.785674],
            [-0.351364, 0.608581, 0.785674],
            [-0.351364, -0.608581, 0.785674],
            [0.702728, 0.0, -0.785674],
            [-0.351364, 0.608581, -0.785674],
            [-0.351364, -0.608581, -0.785674],
            [-1.054093, 0.0, 0.0],
            [0.527046, 0.912871, 0.0],
            [0.527046, -0.912871, 0.0],
        ], dtype=float)),
        # Equal M-L bond lengths, the canonical CShM-literature reference --
        # not the equal-edge-length Johnson solid variant (J10), which
        # SHAPE lists separately as JCSAPR-9. Vertices per cosymlib/SHAPE
        # 2.1's CSAPR-9 reference structure.
        ("capped_square_antiprismatic", np.array([
            [0.0, 0.0, 1.054093],
            [0.982654, 0.0, 0.38145],
            [0.0, 0.982654, 0.38145],
            [-0.982654, 0.0, 0.38145],
            [0.0, -0.982654, 0.38145],
            [0.59092, 0.59092, -0.642449],
            [-0.59092, 0.59092, -0.642449],
            [-0.59092, -0.59092, -0.642449],
            [0.59092, -0.59092, -0.642449],
        ], dtype=float)),
        # Vertices per cosymlib/SHAPE 2.1's HBPY-9 reference structure.
        ("heptagonal_bipyramidal", np.array([
            [0.0, 0.0, -1.054093],
            [1.054093, 0.0, 0.0],
            [0.657216, 0.824123, 0.0],
            [-0.234558, 1.027664, 0.0],
            [-0.949705, 0.457354, 0.0],
            [-0.949705, -0.457354, 0.0],
            [-0.234558, -1.027664, 0.0],
            [0.657216, -0.824123, 0.0],
            [0.0, 0.0, 1.054093],
        ], dtype=float)),
    ]

    # --------------------------------------------------------------- CN=10
    geoms[10] = [
        # Vertices per cosymlib/SHAPE 2.1's PPR-10 reference structure.
        ("pentagonal_prismatic", np.array([
            [0.904182, -0.0, -0.531465],
            [0.279408, 0.859928, -0.531465],
            [-0.731499, 0.531465, -0.531465],
            [-0.731499, -0.531465, -0.531465],
            [0.279408, -0.859928, -0.531465],
            [0.904182, -0.0, 0.531465],
            [0.279408, 0.859928, 0.531465],
            [-0.731499, 0.531465, 0.531465],
            [-0.731499, -0.531465, 0.531465],
            [0.279408, -0.859928, 0.531465],
        ], dtype=float)),
        # Vertices per cosymlib/SHAPE 2.1's PAPR-10 reference structure.
        ("pentagonal_antiprismatic", np.array([
            [0.758925, 0.551391, -0.469042],
            [-0.289884, 0.89217, -0.469042],
            [-0.938083, 0.0, -0.469042],
            [-0.289884, -0.89217, -0.469042],
            [0.758925, -0.551391, -0.469042],
            [0.938083, -0.0, 0.469042],
            [0.289884, 0.89217, 0.469042],
            [-0.758925, 0.551391, 0.469042],
            [-0.758925, -0.551391, 0.469042],
            [0.289884, -0.89217, 0.469042],
        ], dtype=float)),
        # Johnson solid J15 (equal edge lengths). Vertices per
        # cosymlib/SHAPE 2.1's JBCCU-10 reference structure.
        ("bicapped_cube", np.array([
            [0.785488, 0.0, 0.555424],
            [0.785488, 0.0, -0.555424],
            [0.0, 0.785488, 0.555424],
            [0.0, 0.785488, -0.555424],
            [-0.785488, 0.0, 0.555424],
            [-0.785488, 0.0, -0.555424],
            [-0.0, -0.785488, 0.555424],
            [-0.0, -0.785488, -0.555424],
            [0.0, 0.0, 1.340913],
            [0.0, 0.0, -1.340913],
        ], dtype=float)),
        # Johnson solid J17 (gyroelongated square bipyramid, D4d) -- equal
        # edge lengths by construction. Vertices per cosymlib/SHAPE 2.1's
        # JBCSAPR-10 reference structure.
        ("bicapped_square_antiprismatic", np.array([
            [0.831394933130, 0.000000000000, 0.494350384928],
            [0.587884995060, 0.587884995060, -0.494350384928],
            [0.000000000000, 0.831394933130, 0.494350384928],
            [-0.587884995060, 0.587884995060, -0.494350384928],
            [-0.831394933130, 0.000000000000, 0.494350384928],
            [-0.587884995060, -0.587884995060, -0.494350384928],
            [-0.000000000000, -0.831394933130, 0.494350384928],
            [0.587884995060, -0.587884995060, -0.494350384928],
            [0.000000000000, 0.000000000000, 1.325745318058],
            [0.000000000000, 0.000000000000, -1.325745318058],
        ], dtype=float)),
    ]

    # --------------------------------------------------------------- CN=11
    geoms[11] = [
        # Johnson solid J9 (equal edge lengths). Vertices per
        # cosymlib/SHAPE 2.1's JCPPR-11 reference structure.
        ("capped_pentagonal_prismatic", np.array([
            [0.900823, 0.0, 0.529491],
            [0.900823, 0.0, -0.529491],
            [0.27837, 0.856734, 0.529491],
            [0.27837, 0.856734, -0.529491],
            [-0.728781, 0.529491, 0.529491],
            [-0.728781, 0.529491, -0.529491],
            [-0.728781, -0.529491, 0.529491],
            [-0.728781, -0.529491, -0.529491],
            [0.27837, -0.856734, 0.529491],
            [0.27837, -0.856734, -0.529491],
            [0.0, 0.0, 1.08623],
        ], dtype=float)),
        # Johnson solid J11 (equal edge lengths). Vertices per
        # cosymlib/SHAPE 2.1's JCPAPR-11 reference structure.
        ("capped_pentagonal_antiprismatic", np.array([
            [0.937758, 0.0, 0.468879],
            [0.758662, 0.5512, -0.468879],
            [0.289783, 0.89186, 0.468879],
            [-0.289783, 0.89186, -0.468879],
            [-0.758662, 0.5512, 0.468879],
            [-0.937758, 0.0, -0.468879],
            [-0.758662, -0.5512, 0.468879],
            [-0.289783, -0.89186, -0.468879],
            [0.289783, -0.89186, 0.468879],
            [0.758662, -0.5512, -0.468879],
            [0.0, 0.0, -1.048445],
        ], dtype=float)),
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
        # Vertices per cosymlib/SHAPE 2.1's HPR-12 reference structure.
        ("hexagonal_prismatic", np.array([
            [0.930949, -0.0, -0.465475],
            [0.465475, 0.806226, -0.465475],
            [-0.465475, 0.806226, -0.465475],
            [-0.930949, 0.0, -0.465475],
            [-0.465475, -0.806226, -0.465475],
            [0.465475, -0.806226, -0.465475],
            [0.930949, -0.0, 0.465475],
            [0.465475, 0.806226, 0.465475],
            [-0.465475, 0.806226, 0.465475],
            [-0.930949, 0.0, 0.465475],
            [-0.465475, -0.806226, 0.465475],
            [0.465475, -0.806226, 0.465475],
        ], dtype=float)),
        # Uniform (equal-edge-length) antiprism, D6d. Vertices per
        # cosymlib/SHAPE 2.1's HAPR-12 reference structure.
        ("hexagonal_antiprismatic", np.array([
            [0.828737481092, 0.478471807796, -0.409380324284],
            [0.000000000000, 0.956943615592, -0.409380324284],
            [-0.828737481092, 0.478471807796, -0.409380324284],
            [-0.828737481092, -0.478471807796, -0.409380324284],
            [-0.000000000000, -0.956943615592, -0.409380324284],
            [0.828737481092, -0.478471807796, -0.409380324284],
            [0.956943615592, -0.000000000000, 0.409380324284],
            [0.478471807796, 0.828737481092, 0.409380324284],
            [-0.478471807796, 0.828737481092, 0.409380324284],
            [-0.956943615592, 0.000000000000, 0.409380324284],
            [-0.478471807796, -0.828737481092, 0.409380324284],
            [0.478471807796, -0.828737481092, 0.409380324284],
        ], dtype=float)),
        # Archimedean solid (Td) -- equal edge lengths by construction.
        # Vertices per cosymlib/SHAPE 2.1's TT-12 reference structure.
        ("truncated_tetrahedral", np.array([
            [0.000000000000, 0.443812682299, -0.941468871691],
            [0.443812682299, 0.887625364599, -0.313822957230],
            [-0.443812682299, 0.887625364599, -0.313822957230],
            [-0.000000000000, -0.443812682299, -0.941468871691],
            [0.443812682299, -0.887625364599, -0.313822957230],
            [-0.443812682299, -0.887625364599, -0.313822957230],
            [0.887625364599, 0.443812682299, 0.313822957230],
            [0.887625364599, -0.443812682299, 0.313822957230],
            [0.443812682299, 0.000000000000, 0.941468871691],
            [-0.887625364599, 0.443812682299, 0.313822957230],
            [-0.887625364599, -0.443812682299, 0.313822957230],
            [-0.443812682299, 0.000000000000, 0.941468871691],
        ], dtype=float)),
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

# Idealized point group (Schoenflies symbol) of each reference geometry, as
# tabulated by SHAPE 2.1 / cosymlib (via Q-Shape) -- purely informational
# metadata for display (see GeometryMatch.point_group), not used in the
# shape-matching computation itself.
POINT_GROUPS: Dict[str, str] = {
    "linear": "D∞h",
    "v_shaped": "C2v",
    "l_shaped": "C2v",
    "trigonal_planar": "D3h",
    "trigonal_pyramidal": "C3v",
    "t_shaped": "C2v",
    "fac_trivacant_octahedral": "C3v",
    "tetrahedral": "Td",
    "square_planar": "D4h",
    "seesaw": "C2v",
    "vacant_trigonal_bipyramidal": "C3v",
    "trigonal_bipyramidal": "D3h",
    "square_pyramidal": "C4v",
    "pentagonal_planar": "D5h",
    "vacant_octahedral": "C4v",
    "octahedral": "Oh",
    "trigonal_prismatic": "D3h",
    "pentagonal_pyramidal": "C5v",
    "hexagonal_planar": "D6h",
    "pentagonal_bipyramidal": "D5h",
    "capped_octahedron": "C3v",
    "capped_trigonal_prism": "C2v",
    "hexagonal_pyramidal": "C6v",
    "cubic": "Oh",
    "square_antiprismatic": "D4d",
    "hexagonal_bipyramidal": "D6h",
    "dodecahedral": "D2d",
    "biaugmented_trigonal_prismatic": "C2v",
    "snub_disphenoid": "D2d",
    "tricapped_trigonal_prismatic": "D3h",
    "capped_square_antiprismatic": "C4v",
    "heptagonal_bipyramidal": "D7h",
    "pentagonal_prismatic": "D5h",
    "pentagonal_antiprismatic": "D5d",
    "bicapped_cube": "D4h",
    "bicapped_square_antiprismatic": "D4d",
    "capped_pentagonal_prismatic": "C5v",
    "capped_pentagonal_antiprismatic": "C5v",
    "icosahedral": "Ih",
    "cuboctahedron": "Oh",
    "hexagonal_prismatic": "D6h",
    "hexagonal_antiprismatic": "D6d",
    "truncated_tetrahedral": "Td",
}

assert set(POINT_GROUPS) == set(GEOMETRY_BY_NAME), (
    "POINT_GROUPS must have exactly one entry per geometry in GEOMETRY_BY_NAME."
)


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
