"""
Kinetics mapping for Kindred: Eyring and Arrhenius, κ handling, and units.

Current contract
----------------
- Units handling: temperature in K (default 298.15), standard states 1 M or 1 bar.
- Eyring and Arrhenius mapping:
    Eyring unimolecular: k [1/s] = κ * (k_B*T/h) * exp(-ΔG‡/(R*T))
    Bimolecular and n-molecular: multiply by 1/(C°^(n-1)) to yield 1/(M^(n−1)*s)
    Arrhenius: k = A * exp(-Ea/(R*T))
    Units inferred from molecularity: 1/(M^(n−1)*s)
- Preview output format: default 3 sig figs, scientific `e±NN`, ROUND_HALF_UP.
- Locked conventions: constants, κ default 1.0.

Core Arrhenius/Eyring/K(ΔG°) mappings live in :mod:`kindred.core.kinetics`.
This module layers on energy normalization and preview formatting for the DSL
and GUI, and re-exports the shared core formulas for convenience.
"""

from __future__ import annotations

import math
from decimal import Decimal, ROUND_HALF_UP, localcontext

from ..units import kcalmol_to_jmol, kjmol_to_jmol
from ..kinetics import K_from_deltaG_eq, arrhenius_rate, eyring_prefactor, eyring_rate


__all__ = [
    "rate_units",
    "eyring_prefactor",
    "eyring_rate",
    "arrhenius_rate",
    "K_from_deltaG_eq",
    "normalize_energy_to_J_per_mol",
    "format_sci_half_up",
    "preview_line",
]


# ------------------------------ units ----------------------------------------


def rate_units(molecularity: int) -> str:
    """
    Return rate constant unit string given step molecularity n.

    n = 1  -> 1/s
    n = 2  -> 1/(M*s)
    n >= 3 -> 1/(M^(n-1)*s)
    """
    n = int(molecularity)
    if n <= 1:
        return "1/s"
    if n == 2:
        return "1/(M*s)"
    return f"1/(M^{n-1}*s)"


# ------------------------------ energy normalization -------------------------


def normalize_energy_to_J_per_mol(value: float, unit: str | None) -> float:
    """
    Convert energy to canonical J/mol.

    unit ∈ {"J/mol","kJ/mol","kcal/mol"}; if None, assume J/mol.
    """
    if unit is None or unit == "J/mol":
        ej = float(value)
    elif unit == "kJ/mol":
        ej = kjmol_to_jmol(float(value))
    elif unit == "kcal/mol":
        ej = kcalmol_to_jmol(float(value))
    else:
        raise ValueError(f"unsupported energy unit {unit!r}")
    if not math.isfinite(ej):
        raise ValueError("energy must be finite")
    return ej


# ------------------------------ formatting -----------------------------------


def format_sci_half_up(x: float, sig_figs: int = 3) -> str:
    """
    Scientific formatting with ROUND_HALF_UP and `e±NN`.

    Examples (sig_figs=3):
        0        -> 0.00e+00
        1.2345   -> 1.23e+00
        1234     -> 1.23e+03
        -0.01234 -> -1.23e-02
    """
    if not math.isfinite(x):
        return "nan" if math.isnan(x) else ("inf" if x > 0 else "-inf")
    if x == 0.0:
        zeros = "0." + "0" * (sig_figs - 1) if sig_figs > 1 else "0"
        return f"{zeros}e+00"

    sign = "-" if x < 0 else ""
    ax = abs(x)
    exp = int(math.floor(math.log10(ax)))
    m = ax / (10 ** exp)
    with localcontext() as ctx:
        ctx.rounding = ROUND_HALF_UP
        quant = Decimal(1).scaleb(-(sig_figs - 1))
        dm = (Decimal(m)).quantize(quant)
        if dm >= Decimal(10):
            dm = dm / Decimal(10)
            exp += 1
        mantissa = f"{dm:f}"
        if "." in mantissa:
            whole, frac = mantissa.split(".", 1)
            frac = frac + ("0" * max(0, (sig_figs - 1) - len(frac)))
            mantissa = f"{whole}.{frac}"
        else:
            mantissa = mantissa + "." + ("0" * (sig_figs - 1))
    exp_sign = "+" if exp >= 0 else "-"
    exp_abs = abs(exp)
    return f"{sign}{mantissa}e{exp_sign}{exp_abs:02d}"


# ------------------------------ preview lines --------------------------------


def preview_line(
    lhs: str,
    rhs: str,
    *,
    reversible: bool,
    kf: float,
    kr: float | None = None,
    model: str = "Eyring",               # "Eyring" | "Arrhenius"
    unit: str = "1/s",
    kappa: float = 1.0,
    T: float = 298.15,
    source: str = "explicit",
    sig_figs: int = 3,
) -> str:
    """
    Build a one-line preview using the standard DSL preview format.

    Examples:
        A <-> B ; kf=1.23e+05 ; kr=4.56e+03  # model=Eyring; unit=1/s; κ=1.0; T=298.15 K; source=explicit
        A + B <-> C ; kf=7.89e-02 ; kr=1.23e-04  # model=Arrhenius; unit=1/(M*s); κ=0.8; T=350.00 K; source=mixed(kr=derived(K))
        A -> C ; kf=3.00e+06  # model=Eyring; unit=1/s; κ=1.0; T=298.15 K; source=derived(k_fast)
    """
    arrow = "<->" if reversible else "->"
    head = f"{lhs} {arrow} {rhs} ; kf={format_sci_half_up(kf, sig_figs)}"
    if reversible:
        if kr is None:
            raise ValueError("kr must be provided for reversible preview")
        head += f" ; kr={format_sci_half_up(kr, sig_figs)}"
    # Keep κ visible even for Arrhenius previews so the output format stays consistent.
    trailer = f"# model={model}; unit={unit}; κ={float(kappa):.1f}; T={float(T):.2f} K; source={source}"
    return f"{head}  {trailer}"
