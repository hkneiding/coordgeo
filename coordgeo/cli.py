"""Command line interface: coordgeo path/to/complex.xyz --cutoff 2.6"""
from __future__ import annotations

import argparse
import sys

from . import __version__
from .core import DEFAULT_TOLERANCE, analyze


def build_parser() -> argparse.ArgumentParser:
    """Build the coordgeo command-line argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser accepting `xyz_file` and the `--cutoff`/`--tolerance`/
        `--metal-symbol`/`--metal-index`/`--top`/`--seed`/`--version`
        options.
    """
    parser = argparse.ArgumentParser(
        prog="coordgeo",
        description=(
            "Identify the coordination geometry of a mononuclear metal complex "
            "from an xyz file."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}",
    )
    parser.add_argument("xyz_file", help="Path to the input .xyz file.")
    parser.add_argument(
        "-c", "--cutoff", type=float, default=None,
        help=(
            "Cutoff radius in Angstrom used to define the metal's coordinating "
            "neighbors. If omitted, a cutoff is computed automatically from the "
            "metal's and each neighbor's covalent radius (see --tolerance)."
        ),
    )
    parser.add_argument(
        "--tolerance", type=float, default=DEFAULT_TOLERANCE,
        help=(
            "Extra distance in Angstrom added to the summed covalent radii when "
            f"--cutoff is not given (default: {DEFAULT_TOLERANCE}). Ignored if "
            "--cutoff is set."
        ),
    )
    parser.add_argument(
        "--metal-symbol", default=None,
        help="Explicitly select the metal center by element symbol (overrides auto-detection).",
    )
    parser.add_argument(
        "--metal-index", type=int, default=None,
        help="Explicitly select the metal center by 1-based atom index in the xyz file.",
    )
    parser.add_argument(
        "--top", type=int, default=None,
        help="Only show the top N candidate geometries (default: show all for that CN).",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help=(
            "Seed for the randomized geometry search used for coordination numbers "
            "above 7 (ignored otherwise). Set for reproducible results."
        ),
    )
    return parser


def main(argv=None) -> int:
    """Run the coordgeo CLI: parse arguments, analyze, and print a report.

    Parameters
    ----------
    argv : list of str, optional
        Argument vector to parse instead of `sys.argv[1:]` (mainly for
        testing); passed straight through to
        `argparse.ArgumentParser.parse_args`.

    Returns
    -------
    int
        Process exit code: 0 on success, 1 if `analyze()` raised
        `ValueError` or `FileNotFoundError` (the error is printed to
        stderr in that case).

    Raises
    ------
    SystemExit
        Raised by argparse itself for `--version`/`--help`, or if `argv`
        fails to parse (e.g. a missing required argument).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    metal_index0 = args.metal_index - 1 if args.metal_index is not None else None

    try:
        result = analyze(
            args.xyz_file,
            cutoff=args.cutoff,
            metal_symbol=args.metal_symbol,
            metal_index=metal_index0,
            tolerance=args.tolerance,
            seed=args.seed,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(result.summary(top_n=args.top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
