# coordgeo

[![CircleCI](https://dl.circleci.com/status-badge/img/gh/hkneiding/coordgeo/tree/main.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/gh/hkneiding/coordgeo/tree/main)
[![codecov](https://codecov.io/gh/hkneiding/coordgeo/graph/badge.svg?token=JD5J1NX268)](https://codecov.io/gh/hkneiding/coordgeo)

A lightweight Python package for identifying the **coordination geometry**
of **mononuclear metal complexes** from an `.xyz` file.

Given a structure, `coordgeo` auto-detects the metal center (or you specify
it explicitly), finds its coordinating neighbors within a cutoff radius ->
coordination number (CN), and ranks idealized reference geometries for that
CN (e.g. CN=4 -> tetrahedral vs. square planar vs. seesaw) by a
**continuous shape measure (CShM)**: 0 = perfect match, larger = worse.
Supported CN: **2-12** (see `coordgeo/geometries.py` — easy to extend).

The shape-matching method mirrors and reimplements the SHAPE program in
Python, following the continuous shape measures formalism (see
[References](#references)): normalize the real ligand positions and an
idealized reference polyhedron to the same size, find the rotation and
ligand-to-vertex assignment minimizing the summed squared deviation, and
report the residual as a 0-100 measure. For CN <= 7
this assignment is found by exact brute-force search over every
permutation; above that an N! search is impractical, so a
Hungarian-algorithm-based iterative closest point (ICP) search is used
instead (see `coordgeo/matcher.py`).

## Install

```bash
pip install git+https://github.com/hkneiding/coordgeo.git
```

Dependencies: `numpy`, `scipy`.

## Command line usage

```bash
coordgeo examples/octahedral_example.xyz
```

```
Metal center: Fe (atom #1 in xyz file)
Cutoff: auto (covalent radius of Fe + covalent radius of each neighbor + 0.4 Angstrom tolerance)
Coordination number (neighbors within cutoff): 6
Neighbors:
  N   atom #2    distance = 2.100 A
  N   atom #3    distance = 2.100 A
  N   atom #4    distance = 2.100 A
  N   atom #5    distance = 2.100 A
  N   atom #6    distance = 2.100 A
  N   atom #7    distance = 2.100 A

Candidate geometries (lower shape measure = better match, 0 = perfect):
  octahedral             shape measure =   0.00  <-- best match
  trigonal_prismatic     shape measure =  17.43
  pentagonal_pyramidal   shape measure =  31.54
```

Pass `--cutoff` for a fixed distance cutoff instead:

```bash
coordgeo examples/octahedral_example.xyz --cutoff 2.5
```

Options:

- `-c, --cutoff`: fixed cutoff radius in Angstrom. If omitted, an automatic covalent-radius-based cutoff is used instead (see below).
- `--tolerance`: extra distance in Angstrom added to the summed covalent radii when `--cutoff` is not given (default: 0.4).
- `--metal-symbol` / `--metal-index`: pick the metal center explicitly (by symbol or 1-based atom index) instead of auto-detecting it.
- `--top N`: only show the top N candidate geometries.
- `--seed N`: seed for the randomized search used at CN > 7, for reproducible results.

## Python API

```python
import coordgeo

result = coordgeo.analyze("complex.xyz")  # cutoff auto-computed from covalent radii
# result = coordgeo.analyze("complex.xyz", cutoff=2.6)   # fixed cutoff instead
# result = coordgeo.analyze("complex.xyz", tolerance=0.6) # more permissive auto cutoff

print(result.coordination_number)      # e.g. 4
print(result.best_match().name)        # e.g. "square_planar"
print(result.best_match().measure)     # e.g. 0.05  (0 = perfect match)

for match in result.matches:
    print(match.name, match.measure)

print(result.summary())                # human-readable report
```

Lower-level building blocks are also available:

```python
from coordgeo import load_xyz, find_metal_center, get_neighbors, identify_geometry

structure = load_xyz("complex.xyz")
metal_idx = find_metal_center(structure)                 # auto-detect
neighbors = get_neighbors(structure, metal_idx)           # auto cutoff from covalent radii

import numpy as np
ligand_points = np.array([n.vector for n in neighbors])  # metal at origin
matches = identify_geometry(ligand_points)                # ranked GeometryMatch list
```

## How metal detection works

By default, the single atom whose element symbol is in a standard list of
metals is used as the center. Since this package only targets
**mononuclear** complexes, an error is raised if zero or more than one
metal atom is found — pass `metal_symbol=`/`metal_index=` to disambiguate.

## How the cutoff radius works

Any atom within `cutoff` of the metal center is treated
as a coordinating neighbor. **By default** (`cutoff` not given), the
cutoff is computed per neighbor as `covalent_radius(metal) +
covalent_radius(neighbor) + tolerance` (tolerance defaults to 0.4 A),
using the tabulated radii in `coordgeo/radii.py` (Cordero et al., see
[References](#references)) — the same kind of
sum-of-covalent-radii-plus-tolerance heuristic used by OpenBabel/pymatgen.
If an atom close enough to matter has no tabulated radius, a `ValueError`
is raised rather than guessing. **Pass `cutoff` explicitly** instead for a
pure fixed distance (no chemistry) — a common starting point is ~2.4-2.8 A
for first-row transition metals, larger for heavier metals/longer bonds.

## Reference geometries included

| CN | Geometries |
|----|------------|
| 2  | linear, bent |
| 3  | trigonal planar, trigonal pyramidal, T-shaped |
| 4  | tetrahedral, square planar, seesaw, vacant trigonal bipyramidal |
| 5  | trigonal bipyramidal, square pyramidal, pentagonal planar, vacant octahedral |
| 6  | octahedral, trigonal prismatic, pentagonal pyramidal, hexagonal planar |
| 7  | pentagonal bipyramidal, capped octahedron, capped trigonal prism, hexagonal pyramidal |
| 8  | cubic, square antiprismatic, hexagonal bipyramidal, dodecahedral (bisdisphenoid), bicapped trigonal prismatic |
| 9  | tricapped trigonal prismatic, capped square antiprismatic, heptagonal bipyramidal |
| 10 | pentagonal prismatic, pentagonal antiprismatic, bicapped cube |
| 11 | capped pentagonal prismatic, capped pentagonal antiprismatic |
| 12 | icosahedral, cuboctahedron, hexagonal prismatic |

These are idealized templates, not fitted to any real complex, and not
the exhaustive SHAPE reference set (which includes further Johnson-solid
and low-symmetry variants, especially at CN 8-12). Many higher-CN
templates are built systematically, per standard nomenclature: a "vacant"
polyhedron removes a vertex from a larger one, a "capped" one adds one at
a face center. Adding a geometry is just adding a `(name, Nx3 array)`
tuple in `coordgeo/geometries.py`.

## Limitations

- **Mononuclear only** — structures with more than one metal atom raise an error, by design.
- CN outside 2-12 is not supported; `analyze()` raises `ValueError` rather than guessing.
- CN <= 7 uses an exact brute-force search; CN > 7 uses the approximate ICP search above, which is not guaranteed to find the global optimum (though it reliably does in practice with many random restarts). Pass `seed=`/`--seed` for reproducible results there.
- Distance-based neighbor detection has no chemical bonding knowledge (bond orders, valence, etc.) — it is purely geometric.
- The covalent radii table covers H through Cm (the whole periodic table except synthetic superheavy elements and the actinide tail Bk-Lr); pass `cutoff=` explicitly for anything it doesn't cover.

## References

The shape-matching method implemented here follows the continuous shape
measures (CShM) formalism and the reference-polyhedra approach of the
SHAPE program:

- Pinsky, M.; Avnir, D. *Continuous Symmetry Measures. 5. The Classical Polyhedra.* Inorg. Chem. **1998**, 37, 5575–5582.
- Alvarez, S.; Alemany, P.; Casanova, D.; Cirera, J.; Llunell, M.; Avnir, D. *Shape Maps and Polyhedral Interconversion Paths in Transition Metal Chemistry.* Coord. Chem. Rev. **2005**, 249, 1693–1708.
- Llunell, M.; Casanova, D.; Cirera, J.; Alemany, P.; Alvarez, S. *SHAPE: Program for the Stereochemical Analysis of Molecular Fragments by Means of Continuous Shape Measures and Associated Tools*, v2.1; Universitat de Barcelona, 2013.

A related, independently developed web application built on a similar approach:

- Castro Silva Junior, H. *Q-Shape: Quantitative Shape Analyzer*, v1.5.0; Zenodo, **2026**. https://doi.org/10.5281/zenodo.18209621

The tabulated covalent radii used for the automatic cutoff (see "How the
cutoff radius works" above) are from:

- Cordero, B.; Gómez, V.; Platero-Prats, A. E.; Revés, M.; Echeverría, J.; Cremades, E.; Barragán, F.; Alvarez, S. *Covalent Radii Revisited.* Dalton Trans. **2008**, 2832–2838.
