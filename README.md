# coordgeo

[![CircleCI](https://dl.circleci.com/status-badge/img/gh/hkneiding/coordgeo/tree/main.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/gh/hkneiding/coordgeo/tree/main)
[![codecov](https://codecov.io/gh/hkneiding/coordgeo/graph/badge.svg?token=JD5J1NX268)](https://codecov.io/gh/hkneiding/coordgeo)

A lightweight Python package for identifying the **coordination geometry**
of **mononuclear metal complexes** from an `.xyz` file.

Given a structure, `coordgeo` auto-detects the metal center, finds its
coordinating neighbors within a cutoff radius -> coordination number (CN),
and ranks idealized reference geometries for that CN (e.g. CN=4 ->
tetrahedral vs. square planar vs. seesaw) by a **continuous shape measure
(CShM)**: 0 = perfect match, larger = worse. Supported CN: **2-12**.

The method mirrors and reimplements the SHAPE program in Python, following
the continuous shape measures formalism (see [References](#references)):
normalize the ligand positions and an idealized reference polyhedron to the
same size, find the rotation and ligand-to-vertex assignment minimizing the
summed squared deviation, and report the residual as a 0-100 measure. CN <=
7 uses exact brute-force search over every permutation; CN > 7 uses a
Hungarian-algorithm-based iterative closest point (ICP) search instead,
since N! is no longer tractable (see `coordgeo/matcher.py`).

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
Candidate geometries for Fe (atom #1) (lower shape measure = better match, 0 = perfect; sorted best first):
  CN=6   octahedral             shape measure =   0.00  <-- best match
  CN=6   trigonal_prismatic     shape measure =  17.43
  CN=6   pentagonal_pyramidal   shape measure =  31.54
  CN=6   hexagonal_planar       shape measure =  36.70
```

Options:

- `-c, --cutoff`: fixed cutoff radius in Angstrom. If omitted, an automatic covalent-radius-based cutoff is used (see below).
- `--tolerance`: extra distance in Angstrom added to the summed covalent radii when `--cutoff` is not given (default: 0.4); also used by `--window` (see below).
- `--window N`: also consider adding/removing up to N neighbors around the cutoff boundary, pooling ranked candidates across every CN tested (default: 0). See [Exploring nearby coordination numbers](#exploring-nearby-coordination-numbers-window).
- `--metal-symbol` / `--metal-index`: pick the metal center explicitly (by symbol or 1-based atom index) instead of auto-detecting it.
- `--top N`: only show the top N rows of the candidate geometry table.
- `--seed N`: seed for the randomized search used at CN > 7, for reproducible results.

## Python API

```python
import coordgeo

result = coordgeo.analyze("complex.xyz")  # cutoff auto-computed from covalent radii
# result = coordgeo.analyze("complex.xyz", cutoff=2.6)    # fixed cutoff instead
# result = coordgeo.analyze("complex.xyz", tolerance=0.6) # more permissive auto cutoff

print(result.coordination_number)      # e.g. 4
print(result.best_match().name)        # e.g. "square_planar"
print(result.best_match().measure)     # e.g. 0.05  (0 = perfect match)

for match in result.matches:
    print(match.coordination_number, match.name, match.measure)

print(result.summary())                # human-readable report
```

`analyze()` also accepts an in-memory [ASE](https://wiki.fysik.dtu.dk/ase/)
`Atoms` object instead of a file path -- Python-API only, since the CLI
only ever has a file path. `ase` is not a hard dependency; install it
yourself, or via `pip install "coordgeo[ase]"`.

```python
from ase import Atoms
result = coordgeo.analyze(Atoms(...), cutoff=2.6)
```

If you already have specific geometries in mind rather than wanting an
open-ended search, use `analyze_by_geometry()` to test exactly those
(Python-API only) -- each name is resolved to its own CN and matched
against that many closest atoms by plain distance; there's no
cutoff/tolerance/window here at all, and unlike `window`, no plausibility
check either, since you're specifying the hypothesis directly:

```python
result = coordgeo.analyze_by_geometry("complex.xyz", geometries=["square_planar", "octahedral"])
# result.matches has one entry per name, best (lowest measure) first
```

A name that can't be evaluated (unknown, or needs more atoms than the
structure has) is skipped with a `UserWarning` rather than aborting the
rest; it only raises if *none* of the requested geometries could be
evaluated.

Lower-level building blocks are also available:

```python
from coordgeo import load_xyz, structure_from_ase_atoms, find_metal_center, get_neighbors, identify_geometry

structure = load_xyz("complex.xyz")
# structure = structure_from_ase_atoms(atoms)             # or from an ase.Atoms object
metal_idx = find_metal_center(structure)                 # auto-detect
neighbors = get_neighbors(structure, metal_idx)           # auto cutoff from covalent radii

import numpy as np
ligand_points = np.array([n.vector for n in neighbors])  # metal at origin
matches = identify_geometry(ligand_points)                # ranked GeometryMatch list
```

## Exploring nearby coordination numbers (`window`)

A coordinating atom can sit right at the edge of the cutoff, making the
"true" CN ambiguous. `window` (Python `analyze(..., window=N)`, CLI
`--window N`) explores that: it additionally considers dropping up to N of
the cutoff-defined neighbor set's furthest neighbors, and adding up to N of
the closest atoms just outside it, pooling ranked candidates from every CN
tested (base CN +/- N) into one best-first table.

- **`window=0`** (default) reproduces the original single-CN behavior exactly.
- A neighbor is only **removable** if there's a genuine distance gap (`>
  tolerance`) between it and the kept core -- this stops e.g. a uniformly
  distorted octahedron from being reported as `vacant_octahedral` just
  because one ligand happens to be nominally furthest.
- An atom is only **addable** if it's within 1.5x the pairwise
  covalent-radius-sum distance for that atom pair -- this stops a large
  `window` on a sparse structure from reaching for chemically implausible,
  far-away atoms.
- If the base (cutoff-defined) CN itself isn't supported, `analyze()`
  still doesn't raise as long as some CN within the window is; it only
  raises once none of the CNs tested are.
- Raw shape measures aren't fully comparable across *different* CN --
  fewer points are inherently easier to fit well, so a lower-CN row
  scoring near the top isn't automatically the "more correct" answer;
  check it's a genuine, chemically sensible subset before trusting it
  over a higher-CN candidate.

```python
result = coordgeo.analyze("complex.xyz", cutoff=2.5, window=2)
print(result.summary())  # table now spans every CN tested, e.g. base CN +/- 2
```

## How metal detection works

By default, the single atom whose element symbol is in a standard list of
metals is used as the center. Since this package only targets
**mononuclear** complexes, an error is raised if zero or more than one
metal atom is found — pass `metal_symbol=`/`metal_index=` to disambiguate.

## How the cutoff radius works

Any atom within `cutoff` of the metal center is treated as a coordinating
neighbor. **By default** (`cutoff` not given), the cutoff is computed per
neighbor as `covalent_radius(metal) + covalent_radius(neighbor) +
tolerance` (tolerance defaults to 0.4 A), using the tabulated radii in
`coordgeo/radii.py` (Cordero et al., see [References](#references)) — the
same kind of sum-of-covalent-radii-plus-tolerance heuristic used by
OpenBabel/pymatgen. If an atom close enough to matter has no tabulated
radius, a `ValueError` is raised rather than guessing. **Pass `cutoff`
explicitly** instead for a pure fixed distance (no chemistry) — a common
starting point is ~2.4-2.8 A for first-row transition metals, larger for
heavier metals/longer bonds.

**Hydrogens are filtered separately, everywhere.** A candidate hydrogen is
only ever treated as a coordinating neighbor if it has no closer covalent
bond to some other atom -- a hydrogen already bonded to a carbon (e.g. an
agostic C-H...M interaction) is excluded regardless of how close it looks
to the metal, since it's the covalent partner of that other atom, not a
free/candidate hydride. A genuine terminal hydride (bonded only to the
metal) is unaffected. This applies universally: fixed or automatic
`cutoff`, `window`, and `analyze_by_geometry()` alike.

## Reference geometries included

| CN | Geometries |
|----|------------|
| 2  | linear, V-shaped, L-shaped |
| 3  | trigonal planar, trigonal pyramidal, T-shaped, fac-trivacant octahedral |
| 4  | tetrahedral, square planar, seesaw, vacant trigonal bipyramidal |
| 5  | trigonal bipyramidal, square pyramidal, pentagonal planar, vacant octahedral |
| 6  | octahedral, trigonal prismatic, pentagonal pyramidal, hexagonal planar |
| 7  | pentagonal bipyramidal, capped octahedron, capped trigonal prism, hexagonal pyramidal |
| 8  | cubic, square antiprismatic, hexagonal bipyramidal, dodecahedral (bisdisphenoid), biaugmented trigonal prismatic, snub disphenoid |
| 9  | tricapped trigonal prismatic, capped square antiprismatic, heptagonal bipyramidal |
| 10 | pentagonal prismatic, pentagonal antiprismatic, bicapped cube, bicapped square antiprismatic |
| 11 | capped pentagonal prismatic, capped pentagonal antiprismatic |
| 12 | icosahedral, cuboctahedron, hexagonal prismatic, hexagonal antiprismatic, truncated tetrahedral |

These are idealized templates, not fitted to any real complex, and not
the exhaustive SHAPE reference set (which includes further Johnson-solid
and low-symmetry variants, especially at CN 8-12). Many higher-CN
templates are built systematically, per standard nomenclature: a "vacant"
polyhedron removes a vertex from a larger one, a "capped" one adds one at
a face center. A few (l_shaped, fac_trivacant_octahedral, snub_disphenoid,
bicapped_square_antiprismatic, hexagonal_antiprismatic,
truncated_tetrahedral) are instead sourced directly from cosymlib/SHAPE
2.1's published reference structures and converted to coordgeo's
metal-at-origin convention (see [References](#references) and the
methodology note in `coordgeo/geometries.py`). Where a shape has multiple
published variants, only the equal-edge-length one is included, never
both. Adding a geometry is just adding a `(name, Nx3 array)` tuple in
`coordgeo/geometries.py`.

## Limitations

- **Mononuclear only** — structures with more than one metal atom raise an error, by design.
- CN outside 2-12 is not supported; `analyze()` raises `ValueError` rather than guessing.
- CN <= 7 uses an exact brute-force search; CN > 7 uses the approximate ICP search above, which is not guaranteed to find the global optimum (though it reliably does in practice with many random restarts). Pass `seed=`/`--seed` for reproducible results there.
- Distance-based neighbor detection has no chemical bonding knowledge (bond orders, valence, etc.) — it is purely geometric.
- The covalent radii table covers H through Cm (the whole periodic table except synthetic superheavy elements and the actinide tail Bk-Lr); pass `cutoff=` explicitly for anything it doesn't cover.
- The `window` gap/ceiling thresholds (see above) are fixed, not user-configurable.
- The hydrogen-bonded-elsewhere filter (see above) is a distance heuristic, not a real bonding analysis -- it can't be turned off, and it relies on the structure file actually including the hydrogen's real bonding partner (e.g. a heavy-atom-only file gives it nothing to check against, so no hydrogen gets excluded on that basis).

## References

The shape-matching method implemented here follows the continuous shape
measures (CShM) formalism and the reference-polyhedra approach of the
SHAPE program:

- Pinsky, M.; Avnir, D. *Continuous Symmetry Measures. 5. The Classical Polyhedra.* Inorg. Chem. **1998**, 37, 5575–5582.
- Alvarez, S.; Alemany, P.; Casanova, D.; Cirera, J.; Llunell, M.; Avnir, D. *Shape Maps and Polyhedral Interconversion Paths in Transition Metal Chemistry.* Coord. Chem. Rev. **2005**, 249, 1693–1708.
- Llunell, M.; Casanova, D.; Cirera, J.; Alemany, P.; Alvarez, S. *SHAPE: Program for the Stereochemical Analysis of Molecular Fragments by Means of Continuous Shape Measures and Associated Tools*, v2.1; Universitat de Barcelona, 2013.

A Python library implementing a similar continuous symmetry/shape measures
approach (some of coordgeo's reference geometry vertex data traces back to
this library -- see below):

- Alemany, P.; Bernuz, E.; Carreras, A.; Llunell, M. *Cosymlib: A Python Library for Continuous Symmetry Measures*, v0.9.5; Zenodo, **2021**. https://doi.org/10.5281/zenodo.4925767

A related, independently developed web application built on a similar approach:

- Castro Silva Junior, H. *Q-Shape: Quantitative Shape Analyzer*, v1.5.0; Zenodo, **2026**. https://doi.org/10.5281/zenodo.18209621

The tabulated covalent radii used for the automatic cutoff (see "How the
cutoff radius works" above) are from:

- Cordero, B.; Gómez, V.; Platero-Prats, A. E.; Revés, M.; Echeverría, J.; Cremades, E.; Barragán, F.; Alvarez, S. *Covalent Radii Revisited.* Dalton Trans. **2008**, 2832–2838.

Six reference geometry templates (`l_shaped`, `fac_trivacant_octahedral`,
`snub_disphenoid`, `bicapped_square_antiprismatic`,
`hexagonal_antiprismatic`, `truncated_tetrahedral`) use vertex coordinates
sourced from cosymlib's published reference structures via Q-Shape's
implementation, converted to coordgeo's metal-at-origin convention:

- Castro Silva Junior, H. *Q-Shape* reference geometry definitions (cosymlib-derived). `src/constants/referenceGeometries/index.js`. https://github.com/HenriqueCSJ/q-shape/blob/main/src/constants/referenceGeometries/index.js
