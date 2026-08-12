# coordgeo

[![CircleCI](https://dl.circleci.com/status-badge/img/gh/hkneiding/coordgeo/tree/main.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/gh/hkneiding/coordgeo/tree/main)
[![codecov](https://codecov.io/gh/hkneiding/coordgeo/graph/badge.svg?token=JD5J1NX268)](https://codecov.io/gh/hkneiding/coordgeo)

A lightweight Python package for identifying the **coordination geometry**
of **mononuclear metal complexes** from an `.xyz` file.

Given a structure, `coordgeo`:

1. Auto-detects the metal center (or you can specify it explicitly).
2. Finds all neighboring atoms within a cutoff radius -> coordination number
   (CN). By default the cutoff is computed automatically per neighbor from
   covalent radii (see "How the cutoff radius works" below); you can also
   pass a fixed cutoff explicitly.
3. Compares the arrangement of those neighbors against idealized reference
   geometries for that CN (e.g. CN=4 -> tetrahedral vs. square planar vs. seesaw)
   and ranks them by a **continuous shape measure (CShM)**: 0 = perfect match,
   larger = worse match.

The shape-matching method follows the same idea as the widely used SHAPE
program / continuous shape measures approach (Avnir, Pinsky, Alvarez et
al.): normalize both the real ligand positions and the ideal reference
polyhedron to the same size, find the rotation and ligand-to-vertex
assignment that minimizes the sum of squared deviations, and report the
residual as a percentage-like measure.

For coordination number (CN) 7 and below, the ligand-to-vertex assignment
is found by exact brute-force search over every permutation (guaranteed
globally optimal). Above CN 7, an N! search is no longer practical, so a
Hungarian-algorithm-based iterative closest point (ICP) search is used
instead: it alternates optimal assignment (Hungarian algorithm, for the
current rotation) with optimal rotation (Kabsch algorithm, for the current
assignment) until convergence, restarted from many random rotations to
avoid poor local optima. See `coordgeo/matcher.py` for details.

Supported coordination numbers: **2 to 12**, covering the common shapes
for each (see `coordgeo/geometries.py` — easy to extend with more).

## Install

```bash
pip install -e .
```

Dependencies: `numpy`, `scipy` (the latter only for the Hungarian-algorithm
search used at CN > 7).

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

- `-c, --cutoff`: fixed cutoff radius in Angstrom for finding coordinating neighbors. If omitted, an automatic covalent-radius-based cutoff is used instead (see below).
- `--tolerance`: extra distance in Angstrom added to the summed covalent radii when `--cutoff` is not given (default: 0.4). Ignored if `--cutoff` is set.
- `--metal-symbol`: pick the metal center by element symbol (overrides auto-detection).
- `--metal-index`: pick the metal center by 1-based atom index in the xyz file.
- `--top N`: only show the top N candidate geometries.
- `--seed N`: seed for the randomized geometry search used at CN > 7 (ignored otherwise); set for reproducible results.

## Python API

```python
import coordgeo

result = coordgeo.analyze("complex.xyz")  # cutoff auto-computed from covalent radii

print(result.coordination_number)      # e.g. 4
print(result.best_match().name)        # e.g. "square_planar"
print(result.best_match().measure)     # e.g. 0.05  (0 = perfect match)

for match in result.matches:
    print(match.name, match.measure)

print(result.summary())                # human-readable report
```

Pass `cutoff=` for a fixed distance cutoff instead of the automatic one, and
`tolerance=` to adjust the automatic one (ignored if `cutoff=` is given):

```python
result = coordgeo.analyze("complex.xyz", cutoff=2.6)
result = coordgeo.analyze("complex.xyz", tolerance=0.6)  # more permissive auto cutoff
```

Lower-level building blocks are also available:

```python
from coordgeo import load_xyz, find_metal_center, get_neighbors, identify_geometry

structure = load_xyz("complex.xyz")
metal_idx = find_metal_center(structure)                 # auto-detect
neighbors = get_neighbors(structure, metal_idx)           # auto cutoff from covalent radii
# or: get_neighbors(structure, metal_idx, cutoff=2.6)      # fixed cutoff

import numpy as np
ligand_points = np.array([n.vector for n in neighbors])  # metal at origin
matches = identify_geometry(ligand_points)                # ranked GeometryMatch list
```

## How metal detection works

By default, the single atom whose element symbol is in a standard list of
metals (alkali/alkaline earth, transition metals, lanthanides, actinides,
post-transition metals) is used as the center. Since this package only
targets **mononuclear** complexes, an error is raised if zero or more than
one metal atom is found — in that case pass `metal_symbol=` or
`metal_index=` explicitly.

## How the cutoff radius works

Any atom (metal excluded) whose distance from the metal center is `<=
cutoff` is treated as a coordinating neighbor.

**By default** (`cutoff` not given), the cutoff is computed per neighbor as
`covalent_radius(metal) + covalent_radius(neighbor) + tolerance`
(`tolerance` defaults to 0.4 A), using the tabulated covalent radii in
`coordgeo/radii.py` (Cordero et al. 2008-style single-bond radii). This is
the same kind of sum-of-covalent-radii-plus-tolerance heuristic used for
bond perception in tools like OpenBabel and pymatgen. It requires radius
data for the metal and every candidate atom close enough to plausibly be a
neighbor; if an atom's element isn't in the table, a clear `ValueError` is
raised rather than guessing (pass `cutoff=` explicitly to bypass it, or
`tolerance=` to loosen/tighten the automatic cutoff).

**Pass `cutoff` explicitly** for a fixed distance instead — there's no
built-in "chemistry" to this mode, it's a pure distance cutoff, so choose a
value appropriate for the metal-ligand bond lengths in your system (a
common starting point is ~2.4-2.8 A for first-row transition metals,
larger for heavier metals/longer bonds).

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

These are idealized mathematical templates, not fitted to any specific
real complex — they're a reasonable, commonly used starting set, not the
exhaustive canonical SHAPE reference set (which includes further
Johnson-solid and other low-symmetry variants at each CN, especially
CN 8-12). Many of the higher-CN templates are built systematically from
smaller ones, per standard nomenclature: a "vacant" polyhedron removes a
vertex from a larger one, a "capped" polyhedron adds one at a face
center. Adding a new geometry is just adding a `(name, Nx3 array of
vertex vectors)` tuple to the relevant coordination number in
`coordgeo/geometries.py`; it will automatically be normalized and
included in matching.

## Limitations

- **Mononuclear only.** Structures with more than one metal atom raise an
  error (by design, per the current scope).
- CN outside 2-12 is not currently supported (no reference templates) and
  `analyze()` raises `ValueError` rather than guessing -- try a
  smaller/larger cutoff or tolerance to bring the detected CN into range,
  or extend `coordgeo/geometries.py` with templates for the CN you need.
- For CN <= 7, the shape measure uses an exact brute-force permutation
  search, which scales as N! and is only tractable for small N. For
  CN > 7 it instead uses a Hungarian-algorithm-based ICP search (see
  above), which is approximate: it is not guaranteed to find the global
  optimum, though in practice (with many random restarts) it reliably
  does for the templates shipped here. Pass `seed=` (Python API) or
  `--seed` (CLI) for reproducible results at CN > 7.
- Distance-based neighbor detection has no chemical bonding knowledge
  (bond orders, valence, etc.) — it is purely geometric.
- The covalent radii used for the automatic cutoff cover H through Cm
  (i.e. essentially the whole periodic table except the synthetic
  superheavy elements and the actinide tail Bk-Lr). For those, or for
  anything the table doesn't cover, pass `cutoff=` explicitly.

## Running tests

```bash
pip install pytest
pytest tests/
```
