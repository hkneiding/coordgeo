"""Top-level orchestration for coordination geometry identification."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .elements import is_metal
from .io import Structure, load_xyz
from .matcher import GeometryMatch, identify_geometry
from .geometries import MAX_SUPPORTED_CN, MIN_SUPPORTED_CN
from .radii import COVALENT_RADII, covalent_radius

# Extra distance (Angstrom) added to the summed covalent radii of the metal
# and a candidate neighbor when `cutoff` is not given explicitly. 0.4 A is a
# commonly used tolerance for this kind of covalent-radius-sum bond
# perception (cf. OpenBabel, pymatgen).
DEFAULT_TOLERANCE = 0.4

# Largest tabulated covalent radius (Angstrom), used as a conservative upper
# bound on any candidate neighbor's radius so that far-away, chemically
# irrelevant atoms (crystallization solvent, counter-ions, dummy atoms, ...)
# don't force a covalent-radius lookup -- and thus a possible ValueError --
# for elements missing from the table, when they were never going to be
# in-range anyway.
_MAX_KNOWN_RADIUS = max(COVALENT_RADII.values())


@dataclass
class Neighbor:
    symbol: str
    index: int  # 0-based index in the original xyz file
    distance: float
    vector: np.ndarray  # position relative to the metal center


@dataclass
class AnalysisResult:
    metal_symbol: str
    metal_index: int
    cutoff: Optional[float]
    neighbors: List[Neighbor]
    tolerance: float = DEFAULT_TOLERANCE
    matches: List[GeometryMatch] = field(default_factory=list)

    @property
    def coordination_number(self) -> int:
        """Number of coordinating neighbors found.

        Returns
        -------
        int
            `len(self.neighbors)`.
        """
        return len(self.neighbors)

    def best_match(self) -> Optional[GeometryMatch]:
        """Return the top-ranked candidate geometry, if any.

        Returns
        -------
        GeometryMatch or None
            `self.matches[0]` (lowest shape measure), or None if `matches`
            is empty (e.g. an `AnalysisResult` built without ever calling
            `identify_geometry`).
        """
        return self.matches[0] if self.matches else None

    def summary(self, top_n: Optional[int] = None) -> str:
        """Render a human-readable report of the analysis.

        Parameters
        ----------
        top_n : int, optional
            Only include the top `top_n` candidate geometries. Defaults to
            showing every candidate for the coordination number found.

        Returns
        -------
        str
            Multi-line report: metal center, cutoff (fixed or auto),
            coordination number, each neighbor with its distance, and the
            ranked candidate geometries (or a note that none are available
            for this coordination number).
        """
        lines = []
        lines.append(
            f"Metal center: {self.metal_symbol} (atom #{self.metal_index + 1} in xyz file)"
        )
        if self.cutoff is not None:
            lines.append(f"Cutoff radius: {self.cutoff} Angstrom")
        else:
            lines.append(
                f"Cutoff: auto (covalent radius of {self.metal_symbol} + covalent radius "
                f"of each neighbor + {self.tolerance} Angstrom tolerance)"
            )
        lines.append(f"Coordination number (neighbors within cutoff): {self.coordination_number}")
        lines.append("Neighbors:")
        for n in self.neighbors:
            lines.append(f"  {n.symbol:<3s} atom #{n.index + 1:<4d} distance = {n.distance:.3f} A")

        if not self.matches:
            lines.append(
                f"\nNo reference geometries available for coordination number "
                f"{self.coordination_number} (supported range: "
                f"{MIN_SUPPORTED_CN}-{MAX_SUPPORTED_CN})."
            )
        else:
            lines.append("\nCandidate geometries (lower shape measure = better match, 0 = perfect):")
            shown = self.matches if top_n is None else self.matches[:top_n]
            for m in shown:
                marker = "  <-- best match" if m is self.matches[0] else ""
                lines.append(f"  {m.name:<22s} shape measure = {m.measure:6.2f}{marker}")
        return "\n".join(lines)


def find_metal_center(
    structure: Structure,
    metal_symbol: Optional[str] = None,
    metal_index: Optional[int] = None,
) -> int:
    """Return the 0-based index of the metal atom in the structure.

    If metal_index is given, it is used directly (0-based); if metal_symbol
    is *also* given, it must match the element symbol at that index -- this
    catches a stale/typo'd metal_symbol rather than silently ignoring it.
    If only metal_symbol is given, the (first) atom with that symbol is
    used. Otherwise, atoms are auto-detected by element type and there must
    be exactly one metal atom (mononuclear complex assumption).

    Parameters
    ----------
    structure : Structure
        Parsed structure to search.
    metal_symbol : str, optional
        Explicitly select the metal center by element symbol.
    metal_index : int, optional
        Explicitly select the metal center by 0-based atom index. Takes
        precedence over auto-detection; if metal_symbol is also given, it
        must agree with the symbol at this index.

    Returns
    -------
    int
        0-based index of the metal atom in `structure.atoms`.

    Raises
    ------
    ValueError
        If metal_index is out of range; if metal_index and metal_symbol
        are both given but disagree; if metal_symbol doesn't match any
        atom, or matches more than one; or if auto-detection (no
        metal_symbol/metal_index given) finds zero or more than one metal
        atom.
    """
    if metal_index is not None:
        if not (0 <= metal_index < len(structure.atoms)):
            raise ValueError(
                f"metal_index {metal_index} out of range for structure with "
                f"{len(structure.atoms)} atoms."
            )
        actual_symbol = structure.atoms[metal_index].symbol
        if metal_symbol is not None and actual_symbol != metal_symbol:
            raise ValueError(
                f"metal_index {metal_index} refers to a '{actual_symbol}' atom, which "
                f"does not match metal_symbol='{metal_symbol}'. Pass only one of "
                f"metal_symbol/metal_index, or make sure they agree."
            )
        return metal_index

    if metal_symbol is not None:
        candidates = [a.index for a in structure.atoms if a.symbol == metal_symbol]
        if not candidates:
            raise ValueError(f"No atom with symbol '{metal_symbol}' found in structure.")
        if len(candidates) > 1:
            raise ValueError(
                f"Multiple atoms with symbol '{metal_symbol}' found "
                f"(indices {candidates}); this tool only supports mononuclear "
                f"complexes. Pass metal_index explicitly to disambiguate."
            )
        return candidates[0]

    candidates = [a.index for a in structure.atoms if is_metal(a.symbol)]
    if not candidates:
        raise ValueError(
            "No metal atom auto-detected in the structure. Pass metal_symbol "
            "or metal_index explicitly."
        )
    if len(candidates) > 1:
        symbols = [structure.atoms[i].symbol for i in candidates]
        raise ValueError(
            f"Multiple metal atoms auto-detected ({list(zip(symbols, candidates))}); "
            f"this tool only supports mononuclear complexes. Pass metal_symbol or "
            f"metal_index explicitly to pick the center."
        )
    return candidates[0]


def get_neighbors(
    structure: Structure,
    metal_index: int,
    cutoff: Optional[float] = None,
    tolerance: float = DEFAULT_TOLERANCE,
) -> List[Neighbor]:
    """Return all atoms (other than the metal) treated as coordinating neighbors, sorted by distance.

    If `cutoff` is given, it is used as a single fixed distance (Angstrom)
    for every candidate atom. If `cutoff` is None (the default), a
    per-neighbor cutoff is used instead: covalent_radius(metal) +
    covalent_radius(atom) + tolerance -- this requires covalent radius data
    for the metal and every candidate atom's element (see radii.py).

    Parameters
    ----------
    structure : Structure
        Parsed structure containing the metal and its candidate neighbors.
    metal_index : int
        0-based index of the metal atom in `structure.atoms`.
    cutoff : float, optional
        Fixed cutoff radius (Angstrom) for every candidate atom. If None
        (the default), an automatic per-atom-pair cutoff based on covalent
        radii is used instead.
    tolerance : float, default DEFAULT_TOLERANCE
        Extra distance (Angstrom) added to the summed covalent radii when
        `cutoff` is None. Ignored if `cutoff` is given.

    Returns
    -------
    list of Neighbor
        Every atom within cutoff of the metal (metal excluded), sorted by
        ascending distance.

    Raises
    ------
    ValueError
        If `cutoff` is None and covalent radius data is missing for the
        metal's element, or for a candidate atom's element that is close
        enough to plausibly be a neighbor (see `covalent_radius`).
    """
    metal_symbol = structure.atoms[metal_index].symbol
    metal_coord = structure.atoms[metal_index].coord
    metal_radius = covalent_radius(metal_symbol) if cutoff is None else None
    # Any atom farther than this is guaranteed out of range regardless of its
    # (possibly unlisted) element, so it never needs a covalent_radius lookup.
    max_possible_auto_cutoff = (
        metal_radius + _MAX_KNOWN_RADIUS + tolerance if cutoff is None else None
    )

    neighbors = []
    for atom in structure.atoms:
        if atom.index == metal_index:
            continue
        vec = atom.coord - metal_coord
        dist = float(np.linalg.norm(vec))

        if cutoff is not None:
            atom_cutoff = cutoff
        elif dist > max_possible_auto_cutoff:
            continue
        else:
            atom_cutoff = metal_radius + covalent_radius(atom.symbol) + tolerance

        if dist <= atom_cutoff:
            neighbors.append(Neighbor(symbol=atom.symbol, index=atom.index, distance=dist, vector=vec))
    neighbors.sort(key=lambda n: n.distance)
    return neighbors


def analyze(
    path: str,
    cutoff: Optional[float] = None,
    metal_symbol: Optional[str] = None,
    metal_index: Optional[int] = None,
    tolerance: float = DEFAULT_TOLERANCE,
    seed: Optional[int] = None,
) -> AnalysisResult:
    """Load an xyz file, find the metal center, its neighbors, and rank candidate geometries.

    Parameters
    ----------
    path : str
        Path to a .xyz file describing a single (mononuclear) metal complex.
    cutoff : optional
        Fixed cutoff radius (Angstrom) used to decide which atoms are
        coordinating neighbors of the metal center. If omitted (the
        default), an automatic per-neighbor cutoff based on covalent radii
        is used instead: covalent_radius(metal) + covalent_radius(neighbor)
        + tolerance.
    metal_symbol, metal_index : optional
        Explicitly select the metal center instead of relying on
        auto-detection. metal_index is 0-based.
    tolerance : float
        Extra distance (Angstrom) added to the summed covalent radii when
        `cutoff` is not given. Ignored if `cutoff` is given explicitly.
    seed : optional
        Seed for the randomized ICP geometry search used for coordination
        numbers above matcher.EXACT_PERMUTATION_MAX_N (ignored for smaller
        CN, which use an exact search). Pass an explicit value for
        reproducible results.

    Returns
    -------
    AnalysisResult
        The metal center, its neighbors, and ranked candidate geometries
        (`matches`, sorted best first).

    Raises
    ------
    ValueError
        If the metal center can't be resolved (see `find_metal_center`);
        if `cutoff` is None and covalent radius data is missing for a
        relevant atom (see `get_neighbors`); if fewer than 2 neighbors are
        found; or if the coordination number found is outside the
        supported range (MIN_SUPPORTED_CN-MAX_SUPPORTED_CN).
    OSError
        If `path` doesn't exist or can't be opened (see `load_xyz`).
    """
    structure = load_xyz(path)
    m_idx = find_metal_center(structure, metal_symbol=metal_symbol, metal_index=metal_index)
    neighbors = get_neighbors(structure, m_idx, cutoff=cutoff, tolerance=tolerance)

    cn = len(neighbors)
    cutoff_desc = f"{cutoff} A cutoff" if cutoff is not None else "automatic covalent-radius cutoff"

    if cn < 2:
        raise ValueError(
            f"Only {cn} neighbor(s) found within the {cutoff_desc} of the metal "
            f"center; need at least 2 to assess a coordination geometry. Try a "
            f"larger cutoff (or a larger tolerance)."
        )
    if not (MIN_SUPPORTED_CN <= cn <= MAX_SUPPORTED_CN):
        raise ValueError(
            f"Coordination number {cn} (found within the {cutoff_desc}) is not "
            f"supported; reference geometries are only available for CN "
            f"{MIN_SUPPORTED_CN}-{MAX_SUPPORTED_CN}. Try a smaller cutoff/tolerance "
            f"to reduce the number of detected neighbors, or extend "
            f"coordgeo/geometries.py with templates for this CN."
        )

    ligand_points = np.array([n.vector for n in neighbors])
    matches = identify_geometry(ligand_points, seed=seed)

    return AnalysisResult(
        metal_symbol=structure.atoms[m_idx].symbol,
        metal_index=m_idx,
        cutoff=cutoff,
        tolerance=tolerance,
        neighbors=neighbors,
        matches=matches,
    )
