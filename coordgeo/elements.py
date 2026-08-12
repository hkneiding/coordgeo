"""Element data used for metal-center auto-detection."""

# Alkali, alkaline earth, transition metals, lanthanides, actinides and
# post-transition metals. Metalloids (B, Si, Ge, As, Sb, Te, Po) and all
# non-metals are intentionally excluded. This list is intended to be
# "generous but sane" for auto-detecting a metal center in a molecular
# xyz file; users can always override auto-detection explicitly.
METAL_SYMBOLS = {
    # Alkali metals
    "Li", "Na", "K", "Rb", "Cs", "Fr",
    # Alkaline earth metals
    "Be", "Mg", "Ca", "Sr", "Ba", "Ra",
    # Transition metals
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn",
    # Post-transition metals
    "Al", "Ga", "In", "Sn", "Tl", "Pb", "Bi", "Nh", "Fl", "Mc", "Lv",
    # Lanthanides
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
    "Ho", "Er", "Tm", "Yb", "Lu",
    # Actinides
    "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf",
    "Es", "Fm", "Md", "No", "Lr",
}


def is_metal(symbol: str) -> bool:
    """Check whether an element symbol is treated as a metal.

    Parameters
    ----------
    symbol : str
        Element symbol, e.g. "Fe" or "fe" (case and surrounding whitespace
        are normalized before lookup).

    Returns
    -------
    bool
        True if the symbol is in METAL_SYMBOLS, False otherwise.
    """
    return symbol.strip().capitalize() in METAL_SYMBOLS
