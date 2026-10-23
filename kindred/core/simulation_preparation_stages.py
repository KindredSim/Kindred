"""Cohesive solver/Jacobian/ODE and temperature stages for simulation preparation.

``simulation_preparation`` remains the public preparation boundary and
coordinates these deterministic typed stages.  This module deliberately has
no worker, Fit, cache, parameter-binding, or GUI ownership.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


from kindred.core.symbolic.jacobian_execution import (
    SymbolicJacobianExecution,
)

__all__ = [
    "PreparedSolverConfig",
    "build_runtime_symbolic_jacobian",
]


@dataclass(frozen=True, slots=True)
class PreparedSolverConfig:
    """Solver policy consumed by all direct execution paths.

    Its values arrive admitted: Run and Preview from the tab settings and
    Preview points, Fit from its setup and tab settings, and Scan from its
    dialog.
    """

    solver: str
    rtol: float
    atol: float
    grid: Mapping[str, Any]
    use_analytic_jacobian: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "grid", MappingProxyType(dict(self.grid)))


def build_runtime_symbolic_jacobian(
    *,
    use_analytic_jacobian: bool,
    symbolic_structure: Any = None,
    symbolic_support: Any = None,
    coefficient_evaluator: Any,
) -> SymbolicJacobianExecution:
    """Prepare the symbolic Jacobian a Run uses, or why it uses none, as the
    accepted mechanism decided once."""

    if not use_analytic_jacobian:
        return SymbolicJacobianExecution.disabled(
            code="policy-disabled",
            reason="Analytic Jacobian preference is disabled.",
        )
    if symbolic_structure is None:
        return SymbolicJacobianExecution.unsupported(
            classified_status=dict(symbolic_support.to_status_payload()),
        )
    return SymbolicJacobianExecution.supported(
        jacobian_func=symbolic_structure.bind(coefficient_evaluator),
    )
