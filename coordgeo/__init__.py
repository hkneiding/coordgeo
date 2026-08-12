"""coordgeo: lightweight coordination geometry identification for mononuclear metal complexes."""

from .core import analyze, find_metal_center, get_neighbors, AnalysisResult, Neighbor, DEFAULT_TOLERANCE
from .io import load_xyz, Structure, Atom
from .matcher import identify_geometry, shape_measure, GeometryMatch
from .geometries import GEOMETRIES
from .radii import covalent_radius, COVALENT_RADII

__version__ = "0.1.0"

__all__ = [
    "analyze",
    "find_metal_center",
    "get_neighbors",
    "AnalysisResult",
    "Neighbor",
    "DEFAULT_TOLERANCE",
    "load_xyz",
    "Structure",
    "Atom",
    "identify_geometry",
    "shape_measure",
    "GeometryMatch",
    "GEOMETRIES",
    "covalent_radius",
    "COVALENT_RADII",
    "__version__",
]
