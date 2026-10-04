"""Cohesive solver/Jacobian/ODE and temperature stages for simulation preparation.

``simulation_preparation`` remains the public preparation boundary and
coordinates these deterministic typed stages.  This module deliberately has
no worker, Fit, cache, parameter-binding, or GUI ownership.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Real
from types import MappingProxyType
from typing import Any, Dict

import numpy as np

from kindred.core.simulator.solvers import normalize_solver_name
from kindred.core.scipy_integrate import validate_scipy_rtol
from kindred.core.symbolic.errors import UnsupportedSymbolicExpressionError
from kindred.core.symbolic.jacobian_execution import (
    SymbolicJacobianExecution,
    symbolic_status_payload,
)
from kindred.core.temperature import TemperatureSchedule
from kindred.core.time_grid import normalize_time_grid_policy

__all__ = [
    "PreparedSolverConfig",
    "build_prepared_run_context",
    "build_solver_config",
]


@dataclass(frozen=True, slots=True)
class PreparedSolverConfig:
    """Validated solver policy consumed by all direct execution paths."""

    solver_input: str
    solver: str = field(init=False)
    rtol: float
    atol: float
    grid: Mapping[str, Any]
    use_analytic_jacobian: bool

    def __post_init__(self) -> None:
        if type(self.solver_input) is not str or not self.solver_input.strip():
            raise TypeError("Prepared solver input must be a nonblank string.")
        solver_input = self.solver_input.strip()
        solver = normalize_solver_name(solver_input)
        if type(self.atol) is bool or not isinstance(self.atol, Real):
            raise TypeError("Prepared solver atol must be a real number.")
        rtol = validate_scipy_rtol(self.rtol)
        atol = float(self.atol)
        if not (np.isfinite(atol) and atol > 0.0):
            raise ValueError("atol must be finite and > 0.")
        if not isinstance(self.grid, Mapping):
            raise TypeError("Prepared solver grid must be a mapping.")
        if type(self.use_analytic_jacobian) is not bool:
            raise TypeError("use_analytic_jacobian must be a bool.")
        object.__setattr__(self, "solver_input", solver_input)
        object.__setattr__(self, "solver", solver)
        object.__setattr__(self, "rtol", rtol)
        object.__setattr__(self, "atol", atol)
        object.__setattr__(
            self,
            "grid",
            MappingProxyType(normalize_time_grid_policy(self.grid)),
        )


def build_solver_config(
    *,
    solver_input: str,
    rtol: Real,
    atol: Real,
    grid: Mapping[str, Any],
    use_analytic_jacobian: bool,
) -> PreparedSolverConfig:
    """Validate and normalize one typed solver policy."""

    if type(solver_input) is not str:
        raise TypeError("solver_input must be a string.")
    solver_label = solver_input.strip()
    if not solver_label:
        raise ValueError("solver_input must be nonblank.")
    return PreparedSolverConfig(
        solver_input=solver_label,
        rtol=rtol,
        atol=atol,
        grid=grid,
        use_analytic_jacobian=use_analytic_jacobian,
    )


def _symbolic_jacobian_status_for_mechanism(mechanism: Any) -> Dict[str, Any]:
    from kindred.core.symbolic.jacobian import classify_symbolic_jacobian_support

    try:
        support = classify_symbolic_jacobian_support(mechanism)
    except UnsupportedSymbolicExpressionError as exc:
        return symbolic_status_payload(
            kind="jacobian",
            state="unsupported",
            code="classification-failed",
            reason=str(exc),
        )
    return dict(support.to_status_payload())


def _symbolic_jacobian_for_bind_failure(
    mechanism: Any,
    exc: UnsupportedSymbolicExpressionError,
) -> SymbolicJacobianExecution:
    return SymbolicJacobianExecution.from_bind_failure(
        classified_status=_symbolic_jacobian_status_for_mechanism(mechanism),
        exc=exc,
    )


def _bind_symbolic_jacobian_for_current_mechanism(
    *,
    mechanism: Any,
    symbolic_structure: Any = None,
    coefficient_evaluator: Any = None,
) -> SymbolicJacobianExecution:
    structure = symbolic_structure
    if structure is None:
        raise UnsupportedSymbolicExpressionError("symbolic structure is unsupported.")
    from kindred.core.ode_builder import build_effective_rate_coefficient_evaluator

    if coefficient_evaluator is None:
        coefficient_evaluator = build_effective_rate_coefficient_evaluator(mechanism)
    jacobian_func = structure.bind(coefficient_evaluator)
    return SymbolicJacobianExecution.supported(jacobian_func=jacobian_func)


def build_prepared_run_context(
    *,
    mechanism: Any,
    temperature_schedule: TemperatureSchedule | None,
    use_analytic_jacobian: bool,
    symbolic_structure: Any = None,
    coefficient_evaluator: Any = None,
) -> tuple[
    TemperatureSchedule | None,
    SymbolicJacobianExecution,
]:
    """Prepare the retained temperature schedule and symbolic Jacobian."""

    # The schedule has already been admitted by the source/metadata ingress;
    # preserve that typed object through preparation and execution.
    resolved_temperature_schedule = temperature_schedule
    if not use_analytic_jacobian:
        return (
            resolved_temperature_schedule,
            SymbolicJacobianExecution.disabled(
                code="policy-disabled",
                reason="Analytic Jacobian preference is disabled.",
            ),
        )
    symbolic_jacobian = SymbolicJacobianExecution()
    if not symbolic_jacobian.has_executable_jacobian:
        try:
            symbolic_jacobian = _bind_symbolic_jacobian_for_current_mechanism(
                mechanism=mechanism,
                symbolic_structure=symbolic_structure,
                coefficient_evaluator=coefficient_evaluator,
            )
        except UnsupportedSymbolicExpressionError as exc:
            symbolic_jacobian = _symbolic_jacobian_for_bind_failure(mechanism, exc)
    return resolved_temperature_schedule, symbolic_jacobian
