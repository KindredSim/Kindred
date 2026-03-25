"""
Physical constants and rounding policy for Kindred.

These constants are immutable and intended for scientific calculations
across algebra evaluation, simulator, and solver layers.

Units
-----
- Energies: joules (J) unless otherwise noted.
- Temperature: kelvin (K).
- Pressure: pascal (Pa).
"""

from __future__ import annotations

__all__ = [
    "k_B",
    "h",
    "hbar",
    "N_A",
    "R",
    "Rkcal",
]

# Boltzmann constant [J/K]
k_B: float = 1.380_649e-23

# Planck constant [J·s]
h: float = 6.626_070_15e-34

# Reduced Planck constant ħ [J·s]
hbar: float = 1.054_571_817e-34

# Alias for reduced Planck constant
ħ = hbar

# Avogadro constant [1/mol]
N_A: float = 6.022_140_76e23

# Gas constant [J/(mol·K)]
R: float = 8.314_462_618_153_24

# Gas constant [kcal/(mol·K)]
Rkcal: float = 1.987_204_258_640_83e-3
