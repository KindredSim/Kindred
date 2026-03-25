from __future__ import annotations

from typing import Dict, List


def format_stoichiometry_side(side: Dict[str, float]) -> str:
    """
    Render a stoichiometry side as "A + 2B + 0.5C" with deterministic term order.

    This is a presentation helper shared by DSL parsing context strings and preview formatting.
    """

    parts: List[str] = []
    for name in sorted((side or {}).keys()):
        coeff = float(side[name])
        if abs(coeff - 1.0) < 1e-12:
            parts.append(f"{name}")
            continue
        s = f"{coeff:.12g}".rstrip("0").rstrip(".")
        parts.append(f"{s}{name}")
    return " + ".join(parts)

