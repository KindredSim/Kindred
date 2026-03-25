"""
Fast-equilibrium policy utilities.

Policy summary
--------------
- k_fast = 10 × max(explicit k at current T), fallback 1e6, clamped to [1e3, 1e12].
- For `equilibrium:` with only `K` or `ΔG°`, set
    kf = k_fast
    kr = k_fast / K
- Units: rate constants carry their dimensionality upstream; this module treats
  them as positive finite floats and does not perform unit inference.

This module is deterministic and performs no I/O.
"""

from __future__ import annotations
from .common import DSLError, FastEqResult, choose_k_fast, derive_equilibrium_rates

__all__ = [
    "choose_k_fast",
    "derive_equilibrium_rates",
    "FastEqResult",
    "DSLError",
]
