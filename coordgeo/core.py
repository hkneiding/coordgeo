"""Top-level orchestration for coordination geometry identification."""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import numpy as np

from .elements import is_metal
from .io import Structure, load_xyz, structure_from_ase_atoms
from .matcher import EXACT_PERMUTATION_MAX_N, GeometryMatch, identify_geometry, shape_measure
from .geometries import MAX_SUPPORTED_CN, MIN_SUPPORTED_CN, get_geometry_by_name
from .radii import COVALENT_RADII, covalent_radius

if TYPE_CHECKING:
    from ase import Atoms as AseAtoms

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

# Multiplier applied to the pairwise (covalent_radius(metal) +
# covalent_radius(atom) + tolerance) distance to get an absolute ceiling on
# how far the `window` parameter of analyze() is allowed to reach when
# *adding* atoms. Without this, a large enough window on a sparse structure
# could pull in chemically implausible, far-away atoms just because they
# happened to be the Nth-closest in the whole structure. 1.5x is generous
# enough to explore genuinely stretched/borderline coordination, without
# treating arbitrarily distant atoms as candidates. Not user-configurable;
# adjust here if it needs tuning. Only constrains adding atoms -- removing
# an already-included neighbor never needs this kind of sanity check.
_WINDOW_MAX_DISTANCE_FACTOR = 1.5


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
    window: int = 0
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
        """Render the ranked candidate geometry table as a human-readable report.

        Parameters
        ----------
        top_n : int, optional
            Only include the top `top_n` rows of the table. Defaults to
            showing every candidate tested.

        Returns
        -------
        str
            A header naming the metal center, followed by one row per
            candidate geometry tested (across every coordination number in
            the window, if any) with its CN and shape measure, sorted best
            (lowest measure) first -- or a note that none are available if
            `matches` is empty.
        """
        header = (
            f"Candidate geometries for {self.metal_symbol} (atom #{self.metal_index + 1}) "
            f"(lower shape measure = better match, 0 = perfect; sorted best first):"
        )
        lines = [header]
        if not self.matches:
            lines.append(
                f"  No reference geometries available for coordination number "
                f"{self.coordination_number} (supported range: "
                f"{MIN_SUPPORTED_CN}-{MAX_SUPPORTED_CN})."
            )
        else:
            shown = self.matches if top_n is None else self.matches[:top_n]
            for m in shown:
                marker = "  <-- best match" if m is self.matches[0] else ""
                lines.append(
                    f"  CN={m.coordination_number:<3d} {m.name:<22s} "
                    f"shape measure = {m.measure:6.2f}{marker}"
                )
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


def _all_neighbor_candidates(structure: Structure, metal_index: int) -> List[Neighbor]:
    """List every atom other than the metal, with no cutoff filter applied.

    Parameters
    ----------
    structure : Structure
        Parsed structure containing the metal and its candidate neighbors.
    metal_index : int
        0-based index of the metal atom in `structure.atoms`.

    Returns
    -------
    list of Neighbor
        Every non-metal atom, sorted by ascending distance from the metal.
    """
    metal_coord = structure.atoms[metal_index].coord
    candidates = []
    for atom in structure.atoms:
        if atom.index == metal_index:
            continue
        vec = atom.coord - metal_coord
        dist = float(np.linalg.norm(vec))
        candidates.append(Neighbor(symbol=atom.symbol, index=atom.index, distance=dist, vector=vec))
    candidates.sort(key=lambda n: n.distance)
    return candidates


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
    metal_radius = covalent_radius(metal_symbol) if cutoff is None else None
    # Any atom farther than this is guaranteed out of range regardless of its
    # (possibly unlisted) element, so it never needs a covalent_radius lookup.
    max_possible_auto_cutoff = (
        metal_radius + _MAX_KNOWN_RADIUS + tolerance if cutoff is None else None
    )

    neighbors = []
    # Already sorted by distance (see _all_neighbor_candidates); filtering
    # below preserves that order, so no re-sort is needed afterwards.
    for candidate in _all_neighbor_candidates(structure, metal_index):
        if cutoff is not None:
            atom_cutoff = cutoff
        elif candidate.distance > max_possible_auto_cutoff:
            continue
        else:
            atom_cutoff = metal_radius + covalent_radius(candidate.symbol) + tolerance

        if candidate.distance <= atom_cutoff:
            neighbors.append(candidate)
    return neighbors


def _load_structure_and_metal(
    source: Union[str, os.PathLike, "AseAtoms"],
    metal_symbol: Optional[str],
    metal_index: Optional[int],
) -> Tuple[Structure, int]:
    """Load a structure from `source` and resolve its metal center index.

    Shared preamble for `analyze()` and `analyze_by_geometry()`.

    Parameters
    ----------
    source : str, os.PathLike, or ase.Atoms
        Path to a .xyz file, or an in-memory `ase.Atoms` object.
    metal_symbol, metal_index : optional
        Explicitly select the metal center instead of relying on
        auto-detection. metal_index is 0-based.

    Returns
    -------
    (Structure, int)
        The parsed structure and the resolved 0-based metal atom index.

    Raises
    ------
    ValueError
        If the metal center can't be resolved (see `find_metal_center`).
    TypeError
        If `source` is neither a path nor an ase.Atoms-like object (see
        `structure_from_ase_atoms`).
    OSError
        If `source` is a path that doesn't exist or can't be opened (see
        `load_xyz`).
    """
    if isinstance(source, (str, os.PathLike)):
        structure = load_xyz(source)
    else:
        structure = structure_from_ase_atoms(source)
    m_idx = find_metal_center(structure, metal_symbol=metal_symbol, metal_index=metal_index)
    return structure, m_idx


def analyze_by_geometry(
    source: Union[str, os.PathLike, "AseAtoms"],
    geometries: List[str],
    metal_symbol: Optional[str] = None,
    metal_index: Optional[int] = None,
    seed: Optional[int] = None,
) -> AnalysisResult:
    """Score a structure against a specific, named set of reference geometries.

    Unlike `analyze()`, this doesn't search open-endedly: it tests exactly
    the geometries you name, each against its own coordination number's
    worth of closest atoms to the metal (by plain distance -- there's no
    cutoff, tolerance, or window involved, and unlike `analyze(...,
    window=...)`, no gap/ceiling plausibility check either, since you're
    specifying the hypothesis directly rather than exploring blindly).
    Use this when you already have specific candidates in mind.

    A name that can't be evaluated (unknown, or needs more atoms than the
    structure has) is skipped with a `UserWarning` rather than aborting
    the rest -- the other requested geometries are still scored.

    Parameters
    ----------
    source : str, os.PathLike, or ase.Atoms
        Path to a .xyz file describing a single (mononuclear) metal
        complex, or an in-memory `ase.Atoms` object.
    geometries : list of str
        Names of reference geometries to test, e.g.
        `['square_planar', 'octahedral']` (see
        `coordgeo.geometries.GEOMETRIES` for the full list). Each name is
        resolved to its own coordination number independently -- they
        don't need to share one.
    metal_symbol, metal_index : optional
        Explicitly select the metal center instead of relying on
        auto-detection. metal_index is 0-based.
    seed : optional
        Seed for the randomized ICP search used for coordination numbers
        above matcher.EXACT_PERMUTATION_MAX_N (ignored for smaller CN).
        Pass an explicit value for reproducible results.

    Returns
    -------
    AnalysisResult
        `matches` has one GeometryMatch per requested name that could be
        evaluated (fewer than `len(geometries)` if any were skipped),
        sorted best (lowest measure) first. `neighbors`/`coordination_number`
        reflect whichever requested geometry scored best; `cutoff` is
        always None and `window` always 0, since neither applies here.

    Warns
    -----
    UserWarning
        Once per requested name that couldn't be evaluated (unknown, or
        needs more neighbors than the structure has).

    Raises
    ------
    ValueError
        If `geometries` is empty, or none of its names could be evaluated
        (see the emitted warnings for why); or if the metal center can't
        be resolved (see `find_metal_center`).
    TypeError
        If `source` is neither a path nor an ase.Atoms-like object (see
        `structure_from_ase_atoms`).
    OSError
        If `source` is a path that doesn't exist or can't be opened (see
        `load_xyz`).
    """
    structure, m_idx = _load_structure_and_metal(source, metal_symbol, metal_index)
    return _score_named_geometries(structure, m_idx, geometries, seed)


def _score_named_geometries(
    structure: Structure,
    metal_index: int,
    geometries: List[str],
    seed: Optional[int],
) -> AnalysisResult:
    """Score a structure against a specific, named set of reference geometries.

    Bypasses cutoff-based neighbor detection entirely: each requested
    geometry is matched against that many closest atoms to the metal by
    plain distance -- no gap/ceiling plausibility checks (see
    `analyze_by_geometry` for why). A geometry that can't be evaluated
    (unknown name, or not enough atoms in the structure) is skipped with
    a `UserWarning` rather than aborting the rest -- the other requested
    geometries are still scored.

    Parameters
    ----------
    structure : Structure
        Parsed structure containing the metal and candidate neighbors.
    metal_index : int
        0-based index of the metal atom in `structure.atoms`.
    geometries : list of str
        Names of reference geometries to test (see GEOMETRY_BY_NAME).
    seed : optional
        Seed for the randomized ICP search used for coordination numbers
        above matcher.EXACT_PERMUTATION_MAX_N (ignored for smaller CN).

    Returns
    -------
    AnalysisResult
        `matches` has one GeometryMatch per requested name that could be
        evaluated (fewer than `len(geometries)` if any were skipped),
        sorted best (lowest measure) first; `neighbors`/`coordination_number`
        reflect whichever requested geometry scored best. `cutoff` is
        always None and `window` always 0 (neither applies here).

    Warns
    -----
    UserWarning
        For each requested name that can't be evaluated: not a known
        reference geometry (see `get_geometry_by_name`), or requires more
        non-metal atoms than the structure has.

    Raises
    ------
    ValueError
        If `geometries` is empty, or if none of the requested geometries
        could be evaluated (see the emitted warnings for why).
    """
    if not geometries:
        raise ValueError("geometries must be a non-empty list of geometry names.")

    all_candidates = _all_neighbor_candidates(structure, metal_index)
    rng = np.random.default_rng(seed)

    scored: List[Tuple[GeometryMatch, List[Neighbor]]] = []
    for name in geometries:
        try:
            cn, template = get_geometry_by_name(name)
            if cn > len(all_candidates):
                raise ValueError(
                    f"'{name}' requires {cn} neighbors but the structure only has "
                    f"{len(all_candidates)} non-metal atom(s)."
                )
            variant = all_candidates[:cn]
            ligand_points = np.array([n.vector for n in variant])
            measure, perm = shape_measure(
                ligand_points, template, rng=rng if cn > EXACT_PERMUTATION_MAX_N else None,
            )
        except ValueError as exc:
            warnings.warn(f"Skipping geometry {name!r}: {exc}", stacklevel=2)
            continue
        scored.append(
            (GeometryMatch(name=name, coordination_number=cn, measure=measure, permutation=perm), variant)
        )

    if not scored:
        raise ValueError(
            f"None of the requested geometries could be evaluated: {geometries}. "
            f"See the warnings above for the reason each one was skipped."
        )

    scored.sort(key=lambda pair: pair[0].measure)
    matches = [m for m, _ in scored]
    best_neighbors = scored[0][1]

    return AnalysisResult(
        metal_symbol=structure.atoms[metal_index].symbol,
        metal_index=metal_index,
        cutoff=None,
        window=0,
        neighbors=best_neighbors,
        matches=matches,
    )


def analyze(
    source: Union[str, os.PathLike, "AseAtoms"],
    cutoff: Optional[float] = None,
    window: int = 0,
    metal_symbol: Optional[str] = None,
    metal_index: Optional[int] = None,
    tolerance: float = DEFAULT_TOLERANCE,
    seed: Optional[int] = None,
) -> AnalysisResult:
    """Load a structure, find the metal center, its neighbors, and rank candidate geometries.

    A coordinating atom can sit right at the edge of the cutoff, making the
    "true" coordination number ambiguous. With the default `window=0`,
    this ranks reference geometries for exactly the cutoff-defined
    neighbor set (the base CN), as normal. With `window > 0`, it
    additionally considers shrinking that set by dropping up to `window`
    of its furthest (most marginal) neighbors, and growing it by adding up
    to `window` of the closest atoms that were just outside cutoff --
    pooling the ranked candidate geometries from *every* coordination
    number tested (base CN -/+ up to `window`) into one combined,
    best-first `matches` list, so you can see whether the best match is
    stable around the cutoff boundary or only holds for one exact CN.
    This only raises if *none* of the coordination numbers tested (not
    just the base one) turn out valid -- so a base CN that's itself too
    small/large is still rescued if some CN within the window is
    supported.

    If you already have specific candidate geometries in mind rather than
    wanting this kind of open-ended search, see `analyze_by_geometry()`
    instead.

    Parameters
    ----------
    source : str, os.PathLike, or ase.Atoms
        Path to a .xyz file describing a single (mononuclear) metal
        complex, or an in-memory `ase.Atoms` object (Python API only --
        there is no way to pass one from the CLI, which only ever has a
        file path from argv).
    cutoff : optional
        Fixed cutoff radius (Angstrom) used to decide which atoms are
        coordinating neighbors of the metal center. If omitted (the
        default), an automatic per-neighbor cutoff based on covalent radii
        is used instead: covalent_radius(metal) + covalent_radius(neighbor)
        + tolerance. Defines the base coordination number, around which
        `window` (if given) explores.
    window : int, default 0
        How many neighbors to additionally consider adding/removing from
        the cutoff boundary; 0 (the default) reproduces the original
        single-CN behavior. Tests up to `2 * window + 1` coordination
        numbers (base CN through base CN +/- window; fewer if the
        structure doesn't have enough atoms in either direction, a
        candidate atom is farther than a chemically plausible ceiling
        (see _WINDOW_MAX_DISTANCE_FACTOR; only applies to adding atoms),
        removing the furthest neighbor(s) wouldn't cross a genuine
        distance gap (> `tolerance` from the kept "core"; only applies to
        removing atoms -- this is what stops e.g. a uniformly distorted
        octahedron from being mistaken for a vacant_octahedral just
        because its 6th ligand is nominally furthest), or a candidate CN
        falls outside the supported range -- those are simply omitted
        from `matches`, without affecting the others).
    metal_symbol, metal_index : optional
        Explicitly select the metal center instead of relying on
        auto-detection. metal_index is 0-based.
    tolerance : float
        Extra distance (Angstrom) added to the summed covalent radii for
        the base neighbor detection when `cutoff` is not given (ignored
        for that part if `cutoff` is given explicitly). Also used by
        `window` regardless of `cutoff` mode: as the minimum distance gap
        required to justify removing a neighbor, and as part of the
        ceiling on how far `window` may reach when adding one (see
        `window` above).
    seed : optional
        Seed for the randomized ICP geometry search used for coordination
        numbers above matcher.EXACT_PERMUTATION_MAX_N (ignored for smaller
        CN, which use an exact search). Pass an explicit value for
        reproducible results.

    Returns
    -------
    AnalysisResult
        The metal center, its (base/window=0) neighbor list, and the
        ranked candidate geometries in `matches`.

    Raises
    ------
    ValueError
        If `window` is negative; if the metal center can't be resolved
        (see `find_metal_center`); if `cutoff` is None and covalent radius
        data is missing for a relevant atom (see `get_neighbors`); or if
        none of the coordination numbers tested (base CN -/+ up to
        `window`) fall within the supported range
        (MIN_SUPPORTED_CN-MAX_SUPPORTED_CN).
    TypeError
        If `source` is neither a path nor an ase.Atoms-like object (see
        `structure_from_ase_atoms`).
    OSError
        If `source` is a path that doesn't exist or can't be opened (see
        `load_xyz`).
    """
    if window < 0:
        raise ValueError(f"window must be >= 0, got {window}.")

    structure, m_idx = _load_structure_and_metal(source, metal_symbol, metal_index)
    neighbors = get_neighbors(structure, m_idx, cutoff=cutoff, tolerance=tolerance)

    cn = len(neighbors)
    cutoff_desc = f"{cutoff} A cutoff" if cutoff is not None else "automatic covalent-radius cutoff"

    # Atoms outside the cutoff, closest first -- candidates to add for
    # window > 0, restricted to a chemically plausible distance (see
    # _WINDOW_MAX_DISTANCE_FACTOR) regardless of whether `cutoff` is fixed
    # or automatic. Any atom whose ceiling can't be established (metal or
    # candidate element missing covalent radius data) is simply excluded
    # from candidacy rather than raising.
    excluded_candidates: List[Neighbor] = []
    if window:
        base_index_set = {n.index for n in neighbors}
        try:
            metal_radius_for_ceiling = covalent_radius(structure.atoms[m_idx].symbol)
        except ValueError:
            metal_radius_for_ceiling = None
        if metal_radius_for_ceiling is not None:
            for n in _all_neighbor_candidates(structure, m_idx):
                if n.index in base_index_set:
                    continue
                try:
                    atom_radius = covalent_radius(n.symbol)
                except ValueError:
                    continue
                ceiling = _WINDOW_MAX_DISTANCE_FACTOR * (metal_radius_for_ceiling + atom_radius + tolerance)
                if n.distance <= ceiling:
                    excluded_candidates.append(n)
            # _all_neighbor_candidates is sorted by distance and filtering
            # preserves that order, so excluded_candidates is too.

    tested_cns: List[int] = []
    matches: List[GeometryMatch] = []
    for delta in range(-window, window + 1):
        if delta < 0:
            n_remove = -delta
            if n_remove > len(neighbors):
                continue
            keep = len(neighbors) - n_remove
            if keep > 0:
                # Only strip the n_remove furthest neighbors if there's a
                # genuine distance gap separating them from the kept
                # "core" -- otherwise they're not meaningfully different
                # from the rest, and dropping them would just be gaming a
                # smaller-CN template rather than reflecting a real
                # non-coordinating outlier (see the "vacant_octahedral"
                # false-positive this guards against).
                boundary_gap = neighbors[keep].distance - neighbors[keep - 1].distance
                if boundary_gap <= tolerance:
                    continue
            variant = neighbors[:keep]
        elif delta > 0:
            n_add = delta
            if n_add > len(excluded_candidates):
                continue
            variant = sorted(neighbors + excluded_candidates[:n_add], key=lambda n: n.distance)
        else:
            variant = neighbors

        variant_cn = len(variant)
        tested_cns.append(variant_cn)
        if variant_cn >= 2 and MIN_SUPPORTED_CN <= variant_cn <= MAX_SUPPORTED_CN:
            ligand_points = np.array([n.vector for n in variant])
            matches.extend(identify_geometry(ligand_points, seed=seed))

    if not matches:
        if window == 0:
            cn_desc = f"Coordination number {cn}"
        else:
            cn_desc = (
                f"None of the coordination numbers tested ({min(tested_cns)}-{max(tested_cns)}, "
                f"base CN={cn} +/- window={window})"
            )
        raise ValueError(
            f"{cn_desc} found within the {cutoff_desc} is not supported; reference "
            f"geometries are only available for CN {MIN_SUPPORTED_CN}-{MAX_SUPPORTED_CN}. "
            f"Try a different cutoff/tolerance"
            f"{' or a larger window' if window else ''}, or extend "
            f"coordgeo/geometries.py with templates for the CN you need."
        )
    matches.sort(key=lambda m: m.measure)

    return AnalysisResult(
        metal_symbol=structure.atoms[m_idx].symbol,
        metal_index=m_idx,
        cutoff=cutoff,
        tolerance=tolerance,
        window=window,
        neighbors=neighbors,
        matches=matches,
    )
