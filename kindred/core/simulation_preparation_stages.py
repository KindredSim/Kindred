"""Cohesive solver/Jacobian/ODE and temperature stages for simulation preparation.

``simulation_preparation`` remains the public preparation boundary and
coordinates these deterministic typed stages.  This module deliberately has
no worker, Fit, cache, parameter-binding, or GUI ownership.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import logging
from numbers import Real
from types import MappingProxyType
from typing import Any, Dict, Tuple

import numpy as np

from kindred.core.simulator.solvers import normalize_solver_name
from kindred.core.symbolic.errors import UnsupportedSymbolicExpressionError
from kindred.core.symbolic.jacobian_execution import SymbolicJacobianExecution
from kindred.core.symbolic.namespaces import symbolic_status_payload
from kindred.core.symbolic.structure_cache import (
    get_or_build_symbolic_jacobian_structure,
    symbolic_jacobian_structure_cache_key,
)
from kindred.core.temperature import (
    TemperatureScheduleDictProtocol,
    TemperatureScheduleProtocol,
)
from kindred.core.time_grid import normalize_time_grid_policy

logger = logging.getLogger(__name__)

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
    wegscheider_cyclicity_enabled: bool

    def __post_init__(self) -> None:
        if type(self.solver_input) is not str or not self.solver_input.strip():
            raise TypeError("Prepared solver input must be a nonblank string.")
        solver_input = self.solver_input.strip()
        solver = normalize_solver_name(solver_input)
        if type(self.rtol) is bool or not isinstance(self.rtol, Real):
            raise TypeError("Prepared solver rtol must be a real number.")
        if type(self.atol) is bool or not isinstance(self.atol, Real):
            raise TypeError("Prepared solver atol must be a real number.")
        rtol = float(self.rtol)
        atol = float(self.atol)
        if not (np.isfinite(rtol) and rtol > 0.0):
            raise ValueError("rtol must be finite and > 0.")
        if not (np.isfinite(atol) and atol > 0.0):
            raise ValueError("atol must be finite and > 0.")
        if not isinstance(self.grid, Mapping):
            raise TypeError("Prepared solver grid must be a mapping.")
        if type(self.use_analytic_jacobian) is not bool:
            raise TypeError("use_analytic_jacobian must be a bool.")
        if type(self.wegscheider_cyclicity_enabled) is not bool:
            raise TypeError("wegscheider_cyclicity_enabled must be a bool.")
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
    wegscheider_cyclicity_enabled: bool,
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
        wegscheider_cyclicity_enabled=wegscheider_cyclicity_enabled,
    )


def _symbolic_jacobian_structure_for_mechanism(*, mechanism: Any) -> Any:
    from kindred.core.symbolic.jacobian import (
        build_symbolic_jacobian_structure,
        symbolic_jacobian_structure_fingerprint_for_mechanism,
    )

    try:
        structure_fingerprint = symbolic_jacobian_structure_fingerprint_for_mechanism(
            mechanism
        )
    except UnsupportedSymbolicExpressionError:
        return None
    key = symbolic_jacobian_structure_cache_key(
        structure_fingerprint=structure_fingerprint,
    )

    def build_structure() -> Any:
        return build_symbolic_jacobian_structure(mechanism)

    try:
        return get_or_build_symbolic_jacobian_structure(key, build_structure)
    except UnsupportedSymbolicExpressionError:
        return None


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
    prepared_solver_config: PreparedSolverConfig,
    temperature_schedule: TemperatureScheduleProtocol | None,
    symbolic_structure: Any = None,
    coefficient_evaluator: Any = None,
) -> SymbolicJacobianExecution:
    if prepared_solver_config.solver.upper() not in {"BDF", "RADAU"}:
        raise UnsupportedSymbolicExpressionError(
            "Symbolic Jacobian requires BDF or Radau."
        )
    structure = (
        symbolic_structure
        if symbolic_structure is not None
        else _symbolic_jacobian_structure_for_mechanism(mechanism=mechanism)
    )
    if structure is None:
        raise UnsupportedSymbolicExpressionError("symbolic structure is unsupported.")
    from kindred.core.ode_builder import build_effective_rate_coefficient_evaluator

    if temperature_schedule is not None:
        if not isinstance(temperature_schedule, TemperatureScheduleDictProtocol):
            raise UnsupportedSymbolicExpressionError(
                "Symbolic Jacobian requires a deterministic typed temperature "
                "schedule; arbitrary callables are unsupported."
            )
    if coefficient_evaluator is None:
        coefficient_evaluator = build_effective_rate_coefficient_evaluator(mechanism)
    jacobian_func = structure.bind(
        coefficient_evaluator,
        temperature_schedule=temperature_schedule,
    )
    return SymbolicJacobianExecution.supported(
        jacobian_func=jacobian_func,
        jac_sparsity=structure.jac_sparsity,
    )


def build_prepared_run_context(
    *,
    mechanism: Any,
    solver_config: PreparedSolverConfig,
    temperature_schedule: TemperatureScheduleProtocol | None,
    metadata_view: Callable[..., Any],
    symbolic_structure: Any = None,
    coefficient_evaluator: Any = None,
) -> tuple[
    TemperatureScheduleProtocol | None,
    SymbolicJacobianExecution,
    Tuple[str, ...],
]:
    """Prepare temperature, symbolic Jacobian, and ODE metadata as one stage."""

    mech_meta = metadata_view(
        mechanism,
        temperature_schedule_override=temperature_schedule,
    )
    resolved_temperature_schedule = mech_meta.temperature_schedule
    warnings: list[str] = []
    symbolic_jacobian = SymbolicJacobianExecution()
    if (
        not symbolic_jacobian.has_executable_jacobian
        and solver_config.solver.upper() in {"RADAU", "BDF"}
    ):
        try:
            symbolic_jacobian = _bind_symbolic_jacobian_for_current_mechanism(
                mechanism=mechanism,
                prepared_solver_config=solver_config,
                temperature_schedule=resolved_temperature_schedule,
                symbolic_structure=symbolic_structure,
                coefficient_evaluator=coefficient_evaluator,
            )
        except UnsupportedSymbolicExpressionError as exc:
            symbolic_jacobian = _symbolic_jacobian_for_bind_failure(mechanism, exc)
            reason_text = str((symbolic_jacobian.status or {}).get("reason") or exc)
            symbolic_message = (
                "Symbolic Jacobian unsupported; using solver default Jacobian "
                f"handling: {reason_text}"
            )
            logger.warning("%s", symbolic_message)
            warnings.append(symbolic_message)
    return resolved_temperature_schedule, symbolic_jacobian, tuple(warnings)
