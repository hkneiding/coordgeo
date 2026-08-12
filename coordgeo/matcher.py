"""Geometry matching via continuous shape measures (CShM).

For a given set of ligand-position vectors (metal at the origin) and an
idealized reference polyhedron of the same size, we find the permutation
of ligand-to-vertex assignment and the rotation that minimizes the sum of
squared distances between the (size-normalized) ligand points and the
reference vertices. The residual is reported as a 0-100 "shape measure",
with 0 meaning a perfect match (lower is better). This mirrors the
methodology used by the SHAPE program / continuous shape measures
literature (Avnir, Pinsky, Alvarez et al.), simplified for a lightweight
implementation.

Two search strategies are used depending on the coordination number N:

- N <= EXACT_PERMUTATION_MAX_N: exact brute-force search over every one of
  the N! ligand-to-vertex permutations, keeping the rotation-optimal
  (Kabsch) residual for each. Guaranteed globally optimal, but only
  tractable for small N.
- N > EXACT_PERMUTATION_MAX_N: approximate iterative closest point (ICP)
  search. This alternates between (a) the optimal assignment for the
  *current* rotation, solved exactly and efficiently with the Hungarian
  algorithm (``scipy.optimize.linear_sum_assignment``) on the pairwise
  squared-distance cost matrix, and (b) the optimal rotation for the
  *current* assignment, solved with Kabsch, until the assignment stops
  changing. Unlike brute force this is not guaranteed to find the global
  optimum, so it is restarted from several random initial rotations and
  the best result across restarts is kept.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .geometries import GEOMETRIES

# Coordination numbers up to and including this size use the exact
# brute-force permutation search (N!). Above it, ligand-to-vertex
# assignment switches to the Hungarian-algorithm-based ICP search, since
# N! permutations are no longer tractable (8! = 40320, 12! ~ 4.8e8).
EXACT_PERMUTATION_MAX_N = 7

# Number of random-rotation restarts and max alternations per restart for
# the ICP search used when N > EXACT_PERMUTATION_MAX_N. Lower-symmetry
# templates (e.g. capped/vacant shapes) need more restarts to reliably find
# the global optimum than highly symmetric ones (e.g. cubic, icosahedral).
ICP_RESTARTS = 100
ICP_MAX_ITER = 50


def _normalize(points: np.ndarray) -> np.ndarray:
    """Scale a point set (centered on the metal at the origin) to unit RMS radius.

    Parameters
    ----------
    points : (N, 3) array-like
        Points centered on the origin.

    Returns
    -------
    (N, 3) numpy.ndarray
        `points` divided by its RMS radius, `sqrt(mean_i(|points_i|^2))`.

    Raises
    ------
    ValueError
        If every point is at the origin (RMS radius is 0), since the scale
        factor would be undefined.
    """
    points = np.asarray(points, dtype=float)
    scale = np.sqrt(np.mean(np.sum(points ** 2, axis=1)))
    if scale == 0:
        raise ValueError("Cannot normalize a point set where every point is at the origin.")
    return points / scale


def _kabsch_rotation(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Find the optimal rotation aligning one centered point set onto another.

    Parameters
    ----------
    P, Q : (N, 3) array-like
        Centered, row-corresponding point sets (`P[i]` should map to
        `Q[i]`); same N required.

    Returns
    -------
    (3, 3) numpy.ndarray
        Rotation matrix R minimizing `||Q - P @ R||^2` (the Kabsch
        algorithm's closed-form solution, via SVD).
    """
    H = P.T @ Q
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = U @ D @ Vt
    return R


def _residual_for_assignment(P: np.ndarray, Q: np.ndarray) -> float:
    """Compute the best-fit residual for a fixed ligand-to-vertex correspondence.

    Parameters
    ----------
    P, Q : (N, 3) array-like
        Centered, row-corresponding point sets (`P[i]` should map to
        `Q[i]`); same N required.

    Returns
    -------
    float
        `sum((Q - P @ R) ** 2)`, where R is the Kabsch-optimal rotation
        (see `_kabsch_rotation`) for this fixed row correspondence.
    """
    R = _kabsch_rotation(P, Q)
    P_rot = P @ R
    return float(np.sum((Q - P_rot) ** 2))


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Draw a uniformly random (Haar-distributed) 3x3 rotation matrix.

    Parameters
    ----------
    rng : numpy.random.Generator
        Source of randomness.

    Returns
    -------
    (3, 3) numpy.ndarray
        A proper (determinant +1) rotation matrix.
    """
    a = rng.standard_normal((3, 3))
    q, r = np.linalg.qr(a)
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, -1] *= -1
    return q


def _exact_best_assignment(P: np.ndarray, Q: np.ndarray) -> Tuple[float, Tuple[int, ...]]:
    """Find the globally optimal ligand-to-vertex assignment by brute force.

    Parameters
    ----------
    P, Q : (N, 3) array-like
        Normalized, centered point sets to match (`P` = ligand points,
        `Q` = reference template vertices); same N required.

    Returns
    -------
    (float, tuple of int)
        `(best_residual, best_perm)`: the lowest residual found (see
        `_residual_for_assignment`) across every one of the N!
        permutations, and the permutation achieving it, where
        `best_perm[i]` is the template row matched to `P[i]`.
    """
    N = P.shape[0]
    best_residual = math.inf
    best_perm: Tuple[int, ...] = tuple(range(N))
    for perm in itertools.permutations(range(N)):
        residual = _residual_for_assignment(P, Q[list(perm)])
        if residual < best_residual:
            best_residual = residual
            best_perm = perm
    return best_residual, best_perm


def _icp_best_assignment(
    P: np.ndarray, Q: np.ndarray, rng: np.random.Generator
) -> Tuple[float, Tuple[int, ...]]:
    """Approximate the optimal ligand-to-vertex assignment via iterative closest point.

    Alternates Hungarian-algorithm assignment (fixed rotation) with Kabsch
    rotation (fixed assignment) until the assignment converges, from several
    random initial rotations; returns the best result found.

    Parameters
    ----------
    P, Q : (N, 3) array-like
        Normalized, centered point sets to match (`P` = ligand points,
        `Q` = reference template vertices); same N required.
    rng : numpy.random.Generator
        Source of randomness for the restart rotations (see module
        docstring and `ICP_RESTARTS`).

    Returns
    -------
    (float, tuple of int)
        `(best_residual, best_perm)`: the lowest residual found across all
        restarts (see `_residual_for_assignment`), and the permutation
        achieving it, where `best_perm[i]` is the template row matched to
        `P[i]`. Unlike `_exact_best_assignment`, this is not guaranteed to
        be the global optimum.
    """
    N = P.shape[0]
    best_residual = math.inf
    best_perm: Tuple[int, ...] = tuple(range(N))

    for restart in range(ICP_RESTARTS):
        R = np.eye(3) if restart == 0 else _random_rotation(rng)
        perm: Optional[Tuple[int, ...]] = None
        for _ in range(ICP_MAX_ITER):
            P_rot = P @ R
            cost = np.sum((P_rot[:, None, :] - Q[None, :, :]) ** 2, axis=2)
            row_ind, col_ind = linear_sum_assignment(cost)
            new_perm = tuple(col_ind[np.argsort(row_ind)])
            if new_perm == perm:
                break
            perm = new_perm
            R = _kabsch_rotation(P, Q[list(perm)])

        residual = _residual_for_assignment(P, Q[list(perm)])
        if residual < best_residual:
            best_residual = residual
            best_perm = perm

    return best_residual, best_perm


def shape_measure(
    ligand_points: np.ndarray,
    template: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, Tuple[int, ...]]:
    """Best-fit continuous shape measure between ligand points and a reference template.

    Both point sets must have the same number of rows N (same coordination
    number) and be centered on the metal / template center (origin).

    For N <= EXACT_PERMUTATION_MAX_N this is an exact brute-force search
    (global optimum). For larger N it falls back to a randomized
    Hungarian-algorithm-based ICP search (see module docstring); pass an
    explicit `rng` for reproducible results in that regime.

    Parameters
    ----------
    ligand_points : (N, 3) array-like
        Neighbor atom coordinates relative to the metal center (metal at
        the origin); not required to be pre-normalized.
    template : (N, 3) array-like
        Reference polyhedron vertices, centered on the origin; not
        required to be pre-normalized.
    rng : numpy.random.Generator, optional
        Source of randomness for the ICP search used when
        N > EXACT_PERMUTATION_MAX_N (ignored otherwise). Defaults to a
        freshly seeded generator if not given.

    Returns
    -------
    (float, tuple of int)
        `(measure, best_permutation)`: `measure` is in [0, 100] (lower is
        better; ~0 is a perfect match), and `best_permutation` maps
        `ligand_points[i] -> template[best_permutation[i]]`.

    Raises
    ------
    ValueError
        If `ligand_points` and `template` don't have the same number of
        rows, or either point set has every point at the origin (see
        `_normalize`).
    """
    N = ligand_points.shape[0]
    if template.shape[0] != N:
        raise ValueError("ligand_points and template must have the same number of points.")

    P = _normalize(ligand_points)
    Q = _normalize(template)

    if N <= EXACT_PERMUTATION_MAX_N:
        best_residual, best_perm = _exact_best_assignment(P, Q)
    else:
        if rng is None:
            rng = np.random.default_rng()
        best_residual, best_perm = _icp_best_assignment(P, Q, rng)

    # CShM convention: 100 * residual / N  (Q is unit-RMS-radius normalized,
    # so sum(|Q|^2) == N by construction).
    measure = 100.0 * best_residual / N
    return measure, best_perm


@dataclass
class GeometryMatch:
    name: str
    coordination_number: int
    measure: float  # 0 (perfect) to 100 (worst); lower is better
    permutation: Tuple[int, ...]  # ligand index i -> template vertex permutation[i]


def identify_geometry(
    ligand_points: np.ndarray, seed: Optional[int] = None
) -> List[GeometryMatch]:
    """Rank all reference geometries of the matching coordination number.

    Parameters
    ----------
    ligand_points : (N, 3) array
        Neighbor atom coordinates *relative to the metal center* (i.e.
        already translated so the metal sits at the origin).
    seed : optional
        Seed for the randomized ICP search used when N > EXACT_PERMUTATION_MAX_N
        (ignored otherwise). Pass an explicit value for reproducible results;
        by default each call draws fresh randomness.

    Returns
    -------
    list of GeometryMatch
        Sorted best (lowest measure) first.

    Raises
    ------
    ValueError
        If N (the number of rows of `ligand_points`) has no reference
        geometries in GEOMETRIES.
    """
    N = ligand_points.shape[0]
    if N not in GEOMETRIES:
        raise ValueError(
            f"No reference geometries available for coordination number {N}. "
            f"Supported: {sorted(GEOMETRIES.keys())}."
        )

    rng = np.random.default_rng(seed) if N > EXACT_PERMUTATION_MAX_N else None

    results: List[GeometryMatch] = []
    for name, template in GEOMETRIES[N]:
        measure, perm = shape_measure(ligand_points, template, rng=rng)
        results.append(GeometryMatch(name=name, coordination_number=N, measure=measure, permutation=perm))

    results.sort(key=lambda m: m.measure)
    return results
