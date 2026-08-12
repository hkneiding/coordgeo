"""Minimal .xyz file reading."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class Atom:
    symbol: str
    coord: np.ndarray  # shape (3,)
    index: int  # 0-based index in the original file


@dataclass
class Structure:
    atoms: List[Atom]
    comment: str = ""

    def __len__(self) -> int:
        """Return the number of atoms in the structure.

        Returns
        -------
        int
            `len(self.atoms)`.
        """
        return len(self.atoms)

    def coords(self) -> np.ndarray:
        """Stack every atom's coordinates into a single array.

        Returns
        -------
        (N, 3) numpy.ndarray
            Row i is `self.atoms[i].coord`.
        """
        return np.array([a.coord for a in self.atoms])

    def symbols(self) -> List[str]:
        """List every atom's element symbol, in file order.

        Returns
        -------
        list of str
            `[a.symbol for a in self.atoms]`.
        """
        return [a.symbol for a in self.atoms]


def load_xyz(path: str) -> Structure:
    """Parse a standard .xyz file.

    Format::

        <n_atoms>
        <comment line>
        <symbol> <x> <y> <z>
        ...

    Parameters
    ----------
    path : str
        Path to the .xyz file to read.

    Returns
    -------
    Structure
        The parsed atoms (0-based `index`, in file order) and comment line.

    Raises
    ------
    ValueError
        If the file has too few lines, the atom-count line isn't a valid
        integer, fewer coordinate lines are present than declared, or any
        atom line doesn't have at least 4 whitespace-separated fields
        (symbol + 3 coordinates) with parseable floats.
    OSError
        If `path` doesn't exist or can't be opened (e.g. FileNotFoundError,
        a subclass of OSError, if the file is missing).
    """
    with open(path, "r") as f:
        lines = f.read().strip().split("\n")

    if len(lines) < 2:
        raise ValueError(f"'{path}' does not look like a valid xyz file (too few lines).")

    try:
        n_atoms = int(lines[0].strip().split()[0])
    except (ValueError, IndexError) as exc:
        raise ValueError(
            f"First non-empty line of '{path}' should be the atom count, got: {lines[0]!r}"
        ) from exc

    comment = lines[1] if len(lines) > 1 else ""
    atom_lines = lines[2:2 + n_atoms]

    if len(atom_lines) < n_atoms:
        raise ValueError(
            f"'{path}' declares {n_atoms} atoms but only {len(atom_lines)} coordinate "
            f"lines were found."
        )

    atoms: List[Atom] = []
    for i, ln in enumerate(atom_lines):
        parts = ln.split()
        if len(parts) < 4:
            raise ValueError(f"Malformed atom line {i + 3} in '{path}': {ln!r}")
        symbol = parts[0]
        try:
            xyz = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
        except ValueError as exc:
            raise ValueError(f"Malformed coordinates on line {i + 3} in '{path}': {ln!r}") from exc
        atoms.append(Atom(symbol=symbol, coord=xyz, index=i))

    return Structure(atoms=atoms, comment=comment)
