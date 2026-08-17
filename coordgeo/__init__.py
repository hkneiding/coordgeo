"""coordgeo: lightweight coordination geometry identification for mononuclear metal complexes."""

from .core import (
    analyze,
    analyze_by_geometry,
    find_metal_center,
    get_neighbors,
    AnalysisResult,
    Neighbor,
    DEFAULT_TOLERANCE,
)
from .io import load_xyz, structure_from_ase_atoms, Structure, Atom
from .matcher import identify_geometry, shape_measure, GeometryMatch
from .geometries import GEOMETRIES, GEOMETRY_BY_NAME, get_geometry_by_name
from .radii import covalent_radius, COVALENT_RADII

__version__ = "0.1.0"

__all__ = [
    "analyze",
    "analyze_by_geometry",
    "find_metal_center",
    "get_neighbors",
    "AnalysisResult",
    "Neighbor",
    "DEFAULT_TOLERANCE",
    "load_xyz",
    "structure_from_ase_atoms",
    "Structure",
    "Atom",
    "identify_geometry",
    "shape_measure",
    "GeometryMatch",
    "GEOMETRIES",
    "GEOMETRY_BY_NAME",
    "get_geometry_by_name",
    "covalent_radius",
    "COVALENT_RADII",
    "__version__",
]
