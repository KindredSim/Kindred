"""
Shared serial fitting evaluation boundary.

This module owns the fitting-side execution seam below the optimizer and above
the solver. It builds a serial evaluator from structured execution-request data
so the GUI/global-fit path no longer depends on the mutable prepared closure as
its architectural contract.
"""

from __future__ import annotations

import copy
import hashlib
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from kindred.core.exceptions import ErrorContext, FitSimulationError, SimulationCancelled
from kindred.core.mechanism_metadata import MechanismMetadataKeys
from kindred.core.simulation_preparation import (
    PreparedSimulationMetadata,
    SimulationExecutionRequest,
    _build_solver_config,
    _solve_request,
    coerce_prepared_simulation_metadata,
    metadata_view_for_mechanism,
    prepare_bound_mechanism,
    prepare_simulation_worker_run,
)
from kindred.core.simulation_series_payload import SimulationSeriesPayload, coerce_simulation_series_payload
from kindred.core.simulator.solvers import DEFAULT_SOLVER_NAME, SimulationRequest

logger = logging.getLogger(__name__)

__all__ = [
    "CallableFittingEvaluator",
    "PreparedFittingExecutionContext",
    "SerialFittingEvaluator",
    "coerce_fitting_series_evaluator",
    "prepare_fitting_execution_context",
]


class CallableFittingEvaluator:
    """Adapter that lifts a callable simulation boundary into the evaluator contract."""

    def __init__(self, simulation_func) -> None:
        if not callable(simulation_func):
            raise TypeError("simulation_func must be callable.")
        self._simulation_func = simulation_func

    @property
    def prepared_metadata(self) -> Optional[PreparedSimulationMetadata]:
        try:
            return coerce_prepared_simulation_metadata(
                getattr(self._simulation_func, "prepared_metadata", None)
                or getattr(self._simulation_func, "_kindred_prepared_simulation_meta", None)
            )
        except Exception:
            return None

    def evaluate_series(self, params: Mapping[str, float]) -> SimulationSeriesPayload:
        return coerce_simulation_series_payload(self._simulation_func(dict(params or {})))


def coerce_fitting_series_evaluator(value):
    if hasattr(value, "evaluate_series") and callable(getattr(value, "evaluate_series")):
        return value
    if callable(value):
        return CallableFittingEvaluator(value)
    raise TypeError("Fitting evaluator must expose evaluate_series(params) or be callable.")


@dataclass(frozen=True)
class PreparedFittingExecutionContext:
    """Structured execution data for the shared fitting evaluation seam."""

    execution_request: SimulationExecutionRequest
    requested_param_names: List[str]
    prepared_metadata: PreparedSimulationMetadata
    temperature_K: float
    initial_prefix: str

    def __post_init__(self) -> None:
        request = self.execution_request
        if not isinstance(request, SimulationExecutionRequest):
            request = SimulationExecutionRequest.from_mapping(dict(request))
        request_copy = SimulationExecutionRequest.from_mapping(copy.deepcopy(request.to_payload()))
        metadata_copy = coerce_prepared_simulation_metadata(self.prepared_metadata)
        if metadata_copy is None:
            raise TypeError("prepared_metadata must be a PreparedSimulationMetadata or compatible mapping.")
        object.__setattr__(self, "execution_request", request_copy)
        object.__setattr__(
            self,
            "requested_param_names",
            [str(name) for name in list(self.requested_param_names or []) if str(name).strip()],
        )
        object.__setattr__(self, "prepared_metadata", metadata_copy)
        object.__setattr__(self, "temperature_K", float(self.temperature_K))
        object.__setattr__(self, "initial_prefix", str(self.initial_prefix or "init:"))

    def clone(self) -> "PreparedFittingExecutionContext":
        return type(self)(
            execution_request=SimulationExecutionRequest.from_mapping(
                copy.deepcopy(self.execution_request.to_payload())
            ),
            requested_param_names=list(self.requested_param_names),
            prepared_metadata=self.prepared_metadata,
            temperature_K=float(self.temperature_K),
            initial_prefix=str(self.initial_prefix),
        )


def prepare_fitting_execution_context(
    *,
    mechanism_text: str,
    param_names: List[str],
    t_end: float,
    num_points: int,
    temperature_K: float = 298.15,
    solver: str = DEFAULT_SOLVER_NAME,
    rtol: float = 1e-6,
    atol: float = 1e-12,
    use_sparse_jacobian: bool = False,
    wegscheider_cyclicity_enabled: bool = False,
    initial_prefix: str = "init:",
) -> PreparedFittingExecutionContext:
    """Prepare structured execution data for serial fitting evaluation."""

    t_end = float(t_end)
    grid_n = max(2, int(num_points))
    initial_prefix = str(initial_prefix or "init:")

    prepared_solver_config = _build_solver_config(
        solver_input=str(solver or DEFAULT_SOLVER_NAME),
        rtol=rtol,
        atol=atol,
        grid={"N": int(grid_n)},
        use_sparse_jacobian=bool(use_sparse_jacobian),
        wegscheider_cyclicity_enabled=bool(wegscheider_cyclicity_enabled),
    )
    if prepared_solver_config.solver_warning:
        logger.warning(prepared_solver_config.solver_warning)

    bound = prepare_bound_mechanism(
        mechanism_text=mechanism_text,
        param_names=list(param_names or []),
        temperature_K=float(temperature_K),
        initials={},
        use_advanced_dsl=True,
        wegscheider_cyclicity_enabled=bool(wegscheider_cyclicity_enabled),
    )
    prepared_payload = dict(bound.as_serializable_execution_payload())
    prepared_payload["bindings"] = dict(bound.bindings)

    prepared_meta = PreparedSimulationMetadata(
        version=1,
        mechanism_text_sha256=hashlib.sha256((mechanism_text or "").encode("utf-8")).hexdigest(),
        mechanism_text_len=len(mechanism_text or ""),
        param_names=sorted({str(x) for x in (param_names or []) if str(x).strip()}),
        t_end=float(t_end),
        num_points=int(grid_n),
        temperature_K=float(temperature_K),
        solver_requested=str(prepared_solver_config.solver_input),
        solver_normalized=str(prepared_solver_config.solver),
        solver_warning=(
            str(prepared_solver_config.solver_warning)
            if prepared_solver_config.solver_warning
            else None
        ),
        rtol=float(prepared_solver_config.rtol),
        atol=float(prepared_solver_config.atol),
        use_sparse_jacobian=bool(prepared_solver_config.use_sparse_jacobian),
        wegscheider_cyclicity_enabled=bool(wegscheider_cyclicity_enabled),
        initial_prefix=str(initial_prefix),
    )

    execution_request = SimulationExecutionRequest(
        prepared_payload=prepared_payload,
        initials={},
        t_span=(0.0, float(t_end)),
        solver_config={
            "solver": str(prepared_solver_config.solver),
            "rtol": float(prepared_solver_config.rtol),
            "atol": float(prepared_solver_config.atol),
            "grid": {"N": int(grid_n)},
            "use_sparse_jacobian": bool(prepared_solver_config.use_sparse_jacobian),
            MechanismMetadataKeys.WEGSCHEIDER_CYCLICITY_ENABLED: bool(
                prepared_solver_config.wegscheider_cyclicity_enabled
            ),
            MechanismMetadataKeys.TEMPERATURE_K: float(temperature_K),
        },
        mechanism_text=str(mechanism_text or ""),
    )
    return PreparedFittingExecutionContext(
        execution_request=execution_request,
        requested_param_names=sorted({str(x) for x in (param_names or []) if str(x).strip()}),
        prepared_metadata=prepared_meta,
        temperature_K=float(temperature_K),
        initial_prefix=str(initial_prefix),
    )


class SerialFittingEvaluator:
    """Serial in-process evaluator built from structured fitting execution data."""

    def __init__(
        self,
        context: PreparedFittingExecutionContext,
        *,
        fixed_params: Optional[Mapping[str, float]] = None,
    ) -> None:
        self._context = context.clone()
        self._fixed_params = {
            str(name): float(value) for name, value in dict(fixed_params or {}).items() if str(name).strip()
        }
        self._prepared_run = None
        self._bindings: Dict[str, Any] = {}
        self._species_index: Dict[str, int] = {}
        self._last_shared_fp: Optional[Tuple[Tuple[str, float], ...]] = None
        self._parameter_algebra_spec = None
        self._compiled_algebra = None
        self._kindred_fitting_execution_context = self._context

    @property
    def prepared_metadata(self) -> PreparedSimulationMetadata:
        return self._context.prepared_metadata

    @property
    def context(self) -> PreparedFittingExecutionContext:
        return self._context

    def with_fixed_params(self, fixed_params: Mapping[str, float]) -> "SerialFittingEvaluator":
        merged = dict(self._fixed_params)
        for name, value in dict(fixed_params or {}).items():
            if str(name).strip():
                merged[str(name)] = float(value)
        return type(self)(self._context, fixed_params=merged)

    def __call__(self, params: Dict[str, float]) -> SimulationSeriesPayload:
        return self.evaluate_series(params)

    def evaluate_series(self, params: Mapping[str, float]) -> SimulationSeriesPayload:
        self._ensure_prepared()
        prepared_run = self._prepared_run
        if prepared_run is None:
            raise RuntimeError("Prepared fitting lane unavailable.")

        param_map = {str(name): float(value) for name, value in self._fixed_params.items()}
        param_map.update({str(name): value for name, value in dict(params or {}).items()})

        initial_overrides: Dict[str, float] = {}
        shared_values: Dict[str, float] = {}
        for key, raw_val in param_map.items():
            name = str(key)
            try:
                value = float(raw_val)
            except (TypeError, ValueError) as exc:
                raise FitSimulationError(
                    f"Invalid parameter value for {name!r}: {raw_val!r}",
                    details={"fatal": True},
                ) from exc
            if not np.isfinite(value):
                raise FitSimulationError(
                    f"Non-finite parameter value for {name!r}: {raw_val!r}",
                    details={"fatal": True},
                )
            if name.startswith(self._context.initial_prefix):
                initial_overrides[name[len(self._context.initial_prefix) :]] = value
            else:
                shared_values[name] = value

        shared_fp = tuple(sorted((name, float(val)) for name, val in shared_values.items()))
        if shared_fp != self._last_shared_fp:
            self._apply_shared_values(shared_values)
            self._last_shared_fp = shared_fp

        y0 = np.asarray(prepared_run.y0, dtype=float).copy()
        for species_name, value in initial_overrides.items():
            idx = self._species_index.get(species_name)
            if idx is None:
                continue
            y0[idx] = float(value)

        request = SimulationRequest(
            rhs=prepared_run.request.rhs,
            t_span=tuple(map(float, prepared_run.request.t_span)),
            y0=y0,
            solver=str(prepared_run.request.solver),
            rtol=float(prepared_run.request.rtol),
            atol=float(prepared_run.request.atol),
            grid=dict(prepared_run.request.grid or {}),
            jacobian_func=prepared_run.request.jacobian_func,
            temperature_schedule=prepared_run.request.temperature_schedule,
        )
        try:
            result = _solve_request(request)
        except (FitSimulationError, SimulationCancelled):
            raise
        except Exception as exc:
            raise FitSimulationError(
                f"Fitting simulation failed: {exc}",
                details={"fatal": False},
            ) from exc

        species_payload = {
            name: np.asarray(result.Y[idx, :], dtype=float).reshape(-1).copy()
            for idx, name in enumerate(prepared_run.species_names)
        }

        algebra_scalars: Dict[str, float] = {}
        if self._compiled_algebra is not None:
            from kindred.core.algebra.errors import (
                AlgebraError,
                AlgebraNameError,
                AlgebraShadowError,
                AlgebraSyntaxError,
            )
            from kindred.core.algebra.simulation_series import (
                evaluate_compiled_algebra_series_for_simulation,
            )

            try:
                initials_map = {
                    name: float(y0[idx]) for idx, name in enumerate(prepared_run.species_names)
                }
                species_series = {
                    name: np.asarray(result.Y[idx, :], dtype=float).reshape(-1)
                    for idx, name in enumerate(prepared_run.species_names)
                }
                algebra_series, algebra_scalars = evaluate_compiled_algebra_series_for_simulation(
                    prepared_run.mechanism,
                    self._compiled_algebra,
                    t=np.asarray(result.t, dtype=float).reshape(-1),
                    species_series=species_series,
                    initials=initials_map,
                    temperature_K=float(self._context.temperature_K),
                )
                for name, values in (algebra_series or {}).items():
                    if name in species_payload:
                        continue
                    species_payload[str(name)] = np.asarray(values, dtype=float).reshape(-1).copy()
            except AlgebraError as exc:
                is_fatal = isinstance(exc, (AlgebraNameError, AlgebraShadowError, AlgebraSyntaxError))
                raise FitSimulationError(
                    f"Algebra evaluation failed during fitting simulation: {exc}",
                    details={"fatal": bool(is_fatal)},
                    context=ErrorContext(line=exc.line, col=exc.col, line_text=exc.line_text),
                ) from exc
            except FitSimulationError:
                raise
            except Exception as exc:
                raise FitSimulationError(
                    f"Algebra evaluation failed during fitting simulation: {exc}",
                    details={"fatal": False},
                ) from exc

        return coerce_simulation_series_payload(
            SimulationSeriesPayload(
                t=np.asarray(result.t, dtype=float).reshape(-1).copy(),
                species=species_payload,
                algebra_scalars=dict(algebra_scalars),
            )
        )

    def _ensure_prepared(self) -> None:
        if self._prepared_run is not None:
            return
        prepared_run = prepare_simulation_worker_run(execution_request=self._context.execution_request)
        prepared_payload = dict(self._context.execution_request.prepared_payload or {})
        bindings = prepared_payload.get("bindings") or {}
        if not isinstance(bindings, Mapping):
            raise FitSimulationError("Structured fitting payload is missing mutable bindings.", details={"fatal": True})
        self._prepared_run = prepared_run
        self._bindings = dict(bindings)
        self._species_index = {name: idx for idx, name in enumerate(prepared_run.species_names)}

        from kindred.core.algebra.simulation_series import (
            CompiledAlgebraSeries,
            compile_algebra_observables,
        )
        from kindred.core.simulator.parameter_algebra import parameter_algebra_spec_from_mechanism

        self._parameter_algebra_spec = parameter_algebra_spec_from_mechanism(prepared_run.mechanism)

        compiled_algebra: Optional[CompiledAlgebraSeries] = None
        algebra_text = metadata_view_for_mechanism(prepared_run.mechanism).algebra_text
        if algebra_text:
            try:
                compiled_algebra = compile_algebra_observables(str(algebra_text))
            except Exception as exc:
                raise FitSimulationError(
                    f"Failed to parse Algebra observables for fitting: {exc}",
                    details={"fatal": True},
                ) from exc
            if compiled_algebra.time_ref_statements:
                stmt = compiled_algebra.time_ref_statements[0]
                raise FitSimulationError(
                    "Algebra baseline references like [A](T0) are not supported for fitting (v1).",
                    details={"fatal": True},
                    context=ErrorContext(line=stmt.line, col=stmt.col, line_text=stmt.line_text),
                )
        self._compiled_algebra = compiled_algebra

    def _apply_shared_values(self, shared_values: Mapping[str, float]) -> None:
        prepared_run = self._prepared_run
        if prepared_run is None:
            raise RuntimeError("Prepared fitting lane unavailable.")
        for name, value in shared_values.items():
            binding = self._bindings.get(name)
            if binding is None:
                continue
            try:
                binding.set(float(value))
            except Exception as exc:
                raise FitSimulationError(
                    f"Failed to update parameter binding {name!r}: {exc}",
                    details={"fatal": True},
                ) from exc

        from kindred.core.simulator.parameter_algebra import apply_parameter_algebra_spec_to_mechanism

        try:
            if self._parameter_algebra_spec is not None:
                _ = apply_parameter_algebra_spec_to_mechanism(
                    self._parameter_algebra_spec,
                    mechanism=prepared_run.mechanism,
                    require_mutable=True,
                )
        except Exception as exc:
            raise FitSimulationError(f"Parameter algebra failed during fitting simulation: {exc}") from exc
