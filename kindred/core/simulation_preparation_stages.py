"""Cohesive solver/Jacobian/ODE and temperature stages for simulation preparation.

``simulation_preparation`` remains the public preparation boundary and
coordinates these deterministic typed stages.  This module deliberately has
no worker, Fit, cache, parameter-binding, or GUI ownership.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType
from typing import Any, Dict


from kindred.core.symbolic.errors import UnsupportedSymbolicExpressionError
from kindred.core.symbolic.jacobian_execution import (
    SymbolicJacobianExecution,
)

__all__ = [
    "PreparedSolverConfig",
    "build_runtime_symbolic_jacobian",
    "build_solver_config",
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


def build_solver_config(
    *,
    solver: str,
    rtol: Real,
    atol: Real,
    grid: Mapping[str, Any],
    use_analytic_jacobian: bool,
) -> PreparedSolverConfig:
    """Build one typed solver policy from admitted settings."""

    return PreparedSolverConfig(
        solver=solver,
        rtol=rtol,
        atol=atol,
        grid=grid,
        use_analytic_jacobian=use_analytic_jacobian,
    )


def _symbolic_jacobian_status_for_mechanism(mechanism: Any) -> Dict[str, Any]:
    from kindred.core.symbolic.jacobian import classify_symbolic_jacobian_support

    return dict(classify_symbolic_jacobian_support(mechanism).to_status_payload())


def _symbolic_jacobian_for_bind_failure(mechanism: Any) -> SymbolicJacobianExecution:
    return SymbolicJacobianExecution.from_bind_failure(
        classified_status=_symbolic_jacobian_status_for_mechanism(mechanism),
    )


def _bind_symbolic_jacobian_for_current_mechanism(
    *,
    symbolic_structure: Any = None,
    coefficient_evaluator: Any,
) -> SymbolicJacobianExecution:
    structure = symbolic_structure
    if structure is None:
        raise UnsupportedSymbolicExpressionError("symbolic structure is unsupported.")
    jacobian_func = structure.bind(coefficient_evaluator)
    return SymbolicJacobianExecution.supported(jacobian_func=jacobian_func)


def build_runtime_symbolic_jacobian(
    *,
    mechanism: Any,
    use_analytic_jacobian: bool,
    symbolic_structure: Any = None,
    coefficient_evaluator: Any,
) -> SymbolicJacobianExecution:
    """Prepare the symbolic Jacobian a Run uses, or why it uses none."""

    if not use_analytic_jacobian:
        return SymbolicJacobianExecution.disabled(
            code="policy-disabled",
            reason="Analytic Jacobian preference is disabled.",
        )
    try:
        symbolic_jacobian = _bind_symbolic_jacobian_for_current_mechanism(
            symbolic_structure=symbolic_structure,
            coefficient_evaluator=coefficient_evaluator,
        )
    except UnsupportedSymbolicExpressionError:
        symbolic_jacobian = _symbolic_jacobian_for_bind_failure(mechanism)
    return symbolic_jacobian
