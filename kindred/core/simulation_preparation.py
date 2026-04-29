"""
Simulation preparation utilities used by fitting and prepared/bound execution paths.

This module intentionally owns the "parse DSL → bind parameters → compile RHS" pipeline
so that optimization code and analysis code can depend on a narrower surface area.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple, TypedDict

import numpy as np

from kindred.core.rate_binding import RateBinding
import kindred.core.simulator.solvers as solver_api
from kindred.core.exceptions import FitSimulationError
from kindred.core.mechanism_metadata import (
    EquilibriumMetadataKeys,
    MechanismMetadataKeys,
    MechanismMetadataView,
)
from kindred.core.simulation_series_payload import SimulationSeriesPayload
from kindred.core.temperature import TemperatureScheduleProtocol, coerce_temperature_schedule
from kindred.core.runtime_defaults import (
    USE_SPARSE_JACOBIAN_DEFAULT,
    WEGSCHEIDER_CYCLICITY_ENABLED_DEFAULT,
)
from kindred.core.simulator.solvers import (
    DEFAULT_SOLVER_NAME,
    SimulationRequest,
    normalize_solver_name,
    solve_ode,
)

logger = logging.getLogger(__name__)

__all__ = [
    "BoundMechanism",
    "SimulationExecutionRequest",
    "PreparedSimulationMetadata",
    "PreparedFittingObjectiveContext",
    "PreparedSimulationRun",
    "SimulationPreparationError",
    "build_prepared_simulation_func",
    "coerce_prepared_simulation_metadata",
    "metadata_view_for_mechanism",
    "prepared_simulation_run_for_execution_request",
    "prepare_fitting_objective_context",
    "prepare_bound_mechanism",
    "prepare_simulation_worker_run",
]


@dataclass(frozen=True)
class _ParameterOverrideApplication:
    rebuild_rhs: bool
    fully_applied: bool


@dataclass
class BoundMechanism:
    """Precompiled mechanism with mutable rate bindings."""

    mechanism: Any
    rhs: Callable[..., np.ndarray]
    bindings: Dict[str, Any]
    species_names: List[str]
    y0: np.ndarray
    param_names: List[str]
    mechanism_text: str

    def as_execution_payload(
        self,
        *,
        include_rhs: bool,
    ) -> "SimulationWorkerPreparedPayloadV1 | SimulationExecutionPreparedPayloadV2":
        """Return a structured execution payload for worker or batch execution."""
        metadata = metadata_view_for_mechanism(self.mechanism)
        payload: dict[str, Any] = {
            "version": 2,
            "mechanism": self.mechanism,
            "y0": np.array(self.y0, copy=True),
            "species_names": list(self.species_names),
            "mechanism_text": self.mechanism_text,
            "temperature_schedule": metadata.temperature_schedule,
            "jacobian_func": None,
        }
        if include_rhs:
            payload["version"] = 1
            payload["rhs"] = self.rhs
        return payload

    def as_worker_payload(self) -> "SimulationWorkerPreparedPayloadV1":
        """Return a payload suitable for SimulationWorker prepared mode."""
        return self.as_execution_payload(include_rhs=True)  # type: ignore[return-value]

    def as_serializable_execution_payload(self) -> "SimulationExecutionPreparedPayloadV2":
        """Return a process-safe prepared payload without the unpicklable RHS closure."""
        return self.as_execution_payload(include_rhs=False)  # type: ignore[return-value]


class SimulationPreparationError(RuntimeError):
    """
    Error raised when preparing a SimulationWorker run payload/request fails.

    The GUI worker maps `stage` values to user-facing messages while keeping the
    preparation logic in core.
    """

    def __init__(self, stage: str, message: str) -> None:
        super().__init__(str(message))
        self.stage = str(stage or "unknown")


def _solve_request(request: SimulationRequest):
    # Resolve through the solver module so callers that monkeypatch
    # `kindred.core.simulator.solvers.solve_ode` keep working even if this
    # module was imported before the patch was applied.
    solver = getattr(solver_api, solve_ode.__name__, solve_ode)
    return solver(request)


@dataclass(frozen=True)
class PreparedSimulationRun:
    """Execution-ready simulation payload produced by core preparation utilities."""

    mechanism: Any
    rhs: Callable[..., np.ndarray]
    y0: np.ndarray
    species_names: List[str]
    solver_input: str
    solver_warning: Optional[str]
    temperature_schedule: TemperatureScheduleProtocol | None
    jacobian_func: Any
    initials_for_algebra: Optional[Dict[str, float]]
    warnings: List[str]
    request: Any


@dataclass(frozen=True)
class PreparedSimulationMetadata:
    """Typed metadata attached to prepared simulation closures for fitting workflows."""

    version: int
    mechanism_text_sha256: str
    mechanism_text_len: int
    param_names: List[str]
    t_end: float
    num_points: int
    temperature_K: float
    solver_requested: str
    solver_normalized: str
    solver_warning: Optional[str]
    rtol: float
    atol: float
    use_sparse_jacobian: bool
    wegscheider_cyclicity_enabled: bool
    initial_prefix: str

    def to_serializable_dict(self) -> Dict[str, Any]:
        return {
            "version": int(self.version),
            "mechanism_text_sha256": str(self.mechanism_text_sha256),
            "mechanism_text_len": int(self.mechanism_text_len),
            "param_names": [str(name) for name in self.param_names if str(name).strip()],
            "t_end": float(self.t_end),
            "num_points": int(self.num_points),
            "temperature_K": float(self.temperature_K),
            "solver_requested": str(self.solver_requested),
            "solver_normalized": str(self.solver_normalized),
            "solver_warning": str(self.solver_warning) if self.solver_warning else None,
            "rtol": float(self.rtol),
            "atol": float(self.atol),
            "use_sparse_jacobian": bool(self.use_sparse_jacobian),
            MechanismMetadataKeys.WEGSCHEIDER_CYCLICITY_ENABLED: bool(
                self.wegscheider_cyclicity_enabled
            ),
            "initial_prefix": str(self.initial_prefix),
        }

    @classmethod
    def from_mapping(cls, meta: Mapping[str, Any]) -> "PreparedSimulationMetadata":
        param_names_raw = meta.get("param_names") or []
        if not isinstance(param_names_raw, (list, tuple)):
            param_names_raw = []
        solver_requested = str(
            meta.get("solver_requested")
            or meta.get("solver_input")
            or meta.get("solver")
            or ""
        ).strip()
        solver_normalized = str(
            meta.get("solver_normalized")
            or meta.get("solver_used")
            or meta.get("solver")
            or ""
        ).strip()
        return cls(
            version=int(meta.get("version") or 1),
            mechanism_text_sha256=str(meta.get("mechanism_text_sha256") or ""),
            mechanism_text_len=int(meta.get("mechanism_text_len") or 0),
            param_names=sorted({str(x) for x in param_names_raw if str(x).strip()}),
            t_end=float(meta.get("t_end") or 0.0),
            num_points=int(meta.get("num_points") or 0),
            temperature_K=float(meta.get("temperature_K") or 0.0),
            solver_requested=solver_requested,
            solver_normalized=solver_normalized,
            solver_warning=str(meta.get("solver_warning")) if meta.get("solver_warning") else None,
            rtol=float(meta.get("rtol") or 0.0),
            atol=float(meta.get("atol") or 0.0),
            use_sparse_jacobian=bool(
                meta.get("use_sparse_jacobian", USE_SPARSE_JACOBIAN_DEFAULT)
            ),
            wegscheider_cyclicity_enabled=bool(
                meta.get(
                    MechanismMetadataKeys.WEGSCHEIDER_CYCLICITY_ENABLED,
                    WEGSCHEIDER_CYCLICITY_ENABLED_DEFAULT,
                )
            ),
            initial_prefix=str(meta.get("initial_prefix") or ""),
        )


@dataclass(frozen=True)
class PreparedFittingObjectiveContext:
    """Prepared context for building a fitting residual objective."""

    bound: BoundMechanism
    requested_param_names: List[str]
    request: Any
    target_species: str
    target_is_species: bool
    target_species_index: Optional[int]
    compiled_algebra: Any
    initials_for_algebra: Dict[str, float]
    temperature_K: float


@dataclass(frozen=True)
class SimulationExecutionRequest:
    """Structured execution handoff for worker and batch simulation paths."""

    prepared_payload: Optional[Mapping[str, Any]]
    initials: Dict[str, float]
    t_span: Tuple[float, float]
    solver_config: Dict[str, Any]
    mechanism_text: str = ""
    simulation_identity: Optional[Dict[str, Any]] = None
    parameter_overrides: Optional[Dict[str, float]] = None
    version: int = 1

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SimulationExecutionRequest":
        t_span_raw = payload.get("t_span") or (0.0, 0.0)
        try:
            t_span = (float(t_span_raw[0]), float(t_span_raw[1]))
        except (TypeError, ValueError, IndexError) as exc:
            raise SimulationPreparationError("execution_request", f"Invalid execution request t_span: {exc}") from exc
        prepared_payload = payload.get("prepared_payload")
        if prepared_payload is not None and not isinstance(prepared_payload, Mapping):
            raise SimulationPreparationError("execution_request", "Execution request prepared_payload must be a mapping.")
        return cls(
            version=int(payload.get("version") or 1),
            prepared_payload=dict(prepared_payload) if isinstance(prepared_payload, Mapping) else None,
            initials={str(name): float(value) for name, value in dict(payload.get("initials") or {}).items()},
            t_span=t_span,
            solver_config=dict(payload.get("solver_config") or {}),
            mechanism_text=str(payload.get("mechanism_text") or ""),
            simulation_identity=(
                dict(payload.get("simulation_identity") or {})
                if isinstance(payload.get("simulation_identity"), Mapping)
                else None
            ),
            parameter_overrides=(
                {
                    str(name): float(value)
                    for name, value in dict(payload.get("parameter_overrides") or {}).items()
                }
                if isinstance(payload.get("parameter_overrides"), Mapping)
                else None
            ),
        )

    def to_payload(self) -> Dict[str, Any]:
        prepared_payload: Optional[Dict[str, Any]] = None
        if isinstance(self.prepared_payload, Mapping):
            prepared_payload = dict(self.prepared_payload)
            if "y0" in prepared_payload:
                prepared_payload["y0"] = np.array(prepared_payload["y0"], copy=True, dtype=float).reshape(-1)
        payload = {
            "version": int(self.version),
            "prepared_payload": prepared_payload,
            "initials": {str(name): float(value) for name, value in dict(self.initials or {}).items()},
            "t_span": (float(self.t_span[0]), float(self.t_span[1])),
            "solver_config": dict(self.solver_config or {}),
            "mechanism_text": str(self.mechanism_text or ""),
            "simulation_identity": dict(self.simulation_identity or {}) if self.simulation_identity else None,
        }
        if self.parameter_overrides:
            payload["parameter_overrides"] = {
                str(name): float(value)
                for name, value in dict(self.parameter_overrides or {}).items()
            }
        return payload


class SimulationWorkerPreparedPayloadV1(TypedDict):
    version: int
    mechanism: Any
    rhs: Callable[..., np.ndarray]
    y0: np.ndarray
    species_names: List[str]
    mechanism_text: str
    temperature_schedule: TemperatureScheduleProtocol | None
    jacobian_func: Any


class SimulationExecutionPreparedPayloadV2(TypedDict):
    version: int
    mechanism: Any
    y0: np.ndarray
    species_names: List[str]
    mechanism_text: str
    temperature_schedule: TemperatureScheduleProtocol | None
    jacobian_func: Any


_MISSING = object()


def coerce_simulation_execution_request(
    value: SimulationExecutionRequest | Mapping[str, Any] | None,
) -> Optional[SimulationExecutionRequest]:
    if value is None:
        return None
    if isinstance(value, SimulationExecutionRequest):
        return value
    if isinstance(value, Mapping):
        return SimulationExecutionRequest.from_mapping(value)
    raise SimulationPreparationError("execution_request", "Execution request must be a mapping.")


def coerce_prepared_simulation_metadata(
    prepared: object,
) -> Optional[PreparedSimulationMetadata]:
    if prepared is None:
        return None
    if isinstance(prepared, PreparedSimulationMetadata):
        return prepared
    if isinstance(prepared, Mapping):
        try:
            return PreparedSimulationMetadata.from_mapping(prepared)
        except Exception:
            return None
    return None


@dataclass(frozen=True, slots=True)
class _PreparedSolverConfig:
    solver_input: str
    solver: str
    solver_warning: Optional[str]
    rtol: float
    atol: float
    grid: Mapping[str, Any]
    use_sparse_jacobian: bool
    wegscheider_cyclicity_enabled: bool


@dataclass(frozen=True, slots=True)
class _PreparedRunContext:
    solver_config: _PreparedSolverConfig
    temperature_schedule: TemperatureScheduleProtocol | None
    jacobian_func: Any
    algebra_text: Optional[str]
    warnings: Tuple[str, ...]


def _build_solver_config(
    *,
    solver_input: str,
    rtol: object,
    atol: object,
    grid: object,
    use_sparse_jacobian: bool,
    wegscheider_cyclicity_enabled: bool,
) -> _PreparedSolverConfig:
    solver_label = str(solver_input or DEFAULT_SOLVER_NAME).strip() or DEFAULT_SOLVER_NAME
    solver, solver_warning = normalize_solver_name(solver_label)

    rtol_f = float(rtol)
    atol_f = float(atol)
    if not (np.isfinite(rtol_f) and rtol_f > 0.0):
        raise ValueError("rtol must be finite and > 0.")
    if not (np.isfinite(atol_f) and atol_f > 0.0):
        raise ValueError("atol must be finite and > 0.")

    if not isinstance(grid, Mapping) or not grid:
        grid_out: Mapping[str, Any] = {"N": 100}
    else:
        grid_out = dict(grid)

    return _PreparedSolverConfig(
        solver_input=str(solver_label),
        solver=str(solver),
        solver_warning=str(solver_warning) if solver_warning else None,
        rtol=float(rtol_f),
        atol=float(atol_f),
        grid=grid_out,
        use_sparse_jacobian=bool(use_sparse_jacobian),
        wegscheider_cyclicity_enabled=bool(wegscheider_cyclicity_enabled),
    )


def _metadata_view_for_mechanism(
    mechanism: Any,
    *,
    temperature_schedule_override: object = _MISSING,
) -> Any:
    meta_raw = getattr(mechanism, "metadata", {}) or {}
    if not isinstance(meta_raw, Mapping):
        meta_raw = {}
    if temperature_schedule_override is not _MISSING:
        meta_raw = dict(meta_raw)
        meta_raw[MechanismMetadataKeys.TEMPERATURE_SCHEDULE] = temperature_schedule_override
    return MechanismMetadataView.from_metadata(meta_raw)


def metadata_view_for_mechanism(
    mechanism: Any,
    *,
    temperature_schedule_override: object = _MISSING,
) -> MechanismMetadataView:
    return _metadata_view_for_mechanism(
        mechanism,
        temperature_schedule_override=temperature_schedule_override,
    )


def _prepare_preparation_failure(stage: str, message: object) -> SimulationPreparationError:
    return SimulationPreparationError(str(stage or "unknown"), str(message or ""))


def _prepared_payload_failure(message: object) -> SimulationPreparationError:
    return _prepare_preparation_failure("prepared_payload", message)


def _preparation_failure_payload(exc: SimulationPreparationError) -> Dict[str, Any]:
    from kindred.core.simulation_failure import build_simulation_failure

    return build_simulation_failure(
        "preparation_error",
        str(exc),
        details={"stage": str(exc.stage or "unknown")},
        exc_type=exc.__class__.__name__,
    )


def _fit_simulation_error_from_preparation_error(
    exc: SimulationPreparationError,
    *,
    fatal: bool = True,
) -> FitSimulationError:
    return FitSimulationError(
        str(exc),
        details={
            "fatal": bool(fatal),
            "failure": _preparation_failure_payload(exc),
        },
    )


def _validated_prepared_worker_payload(
    prepared_payload: Mapping[str, Any],
) -> tuple[Any, Callable[..., np.ndarray] | None, list[str], np.ndarray, object, Any]:
    try:
        version = int(prepared_payload.get("version", 1))
    except Exception as exc:
        raise _prepared_payload_failure(f"Invalid prepared payload version: {exc}") from exc
    if version not in {1, 2}:
        raise _prepared_payload_failure(
            f"Unsupported prepared payload version: {version}"
        )

    try:
        mechanism = prepared_payload["mechanism"]
    except KeyError as exc:
        raise _prepared_payload_failure(f"Missing prepared payload field: {exc.args[0]}") from exc
    if version == 1:
        try:
            rhs = prepared_payload["rhs"]
        except KeyError as exc:
            raise _prepared_payload_failure(f"Missing prepared payload field: {exc.args[0]}") from exc
    else:
        rhs = None

    species_source = prepared_payload.get("species_names")
    if species_source is None:
        try:
            species_source = getattr(mechanism, "species_names")()
        except Exception as exc:
            raise _prepared_payload_failure(
                f"Unable to determine species_names from prepared payload: {exc}"
            ) from exc
    try:
        species_names = [str(name) for name in species_source]
    except Exception as exc:
        raise _prepared_payload_failure(f"Invalid prepared payload species_names: {exc}") from exc
    if not species_names:
        raise _prepared_payload_failure("Prepared payload species_names must not be empty")

    y0_source = prepared_payload.get("y0")
    if y0_source is None:
        try:
            y0 = np.array(
                [mechanism.species[sp].initial_conc for sp in species_names],
                dtype=float,
            ).reshape(-1)
        except Exception as exc:
            raise _prepared_payload_failure(
                f"Unable to derive prepared payload initial concentrations: {exc}"
            ) from exc
    else:
        try:
            y0 = np.array(y0_source, copy=True, dtype=float).reshape(-1)
        except Exception as exc:
            raise _prepared_payload_failure(f"Invalid prepared payload y0: {exc}") from exc
    if y0.size != len(species_names):
        raise _prepared_payload_failure(
            "Prepared payload y0 length does not match species_names length"
        )

    temperature_schedule_override: object = _MISSING
    if "temperature_schedule" in prepared_payload:
        temperature_schedule_override = prepared_payload.get("temperature_schedule")
        try:
            coerce_temperature_schedule(temperature_schedule_override)
        except Exception as exc:
            raise _prepared_payload_failure(
                f"Invalid prepared payload temperature_schedule: {exc}"
            ) from exc
    jacobian_func_override = prepared_payload.get("jacobian_func")
    return (
        mechanism,
        rhs,
        species_names,
        y0,
        temperature_schedule_override,
        jacobian_func_override,
    )


def _coerce_parameter_override_items(
    parameter_overrides: Mapping[str, Any] | None,
) -> list[tuple[str, float]]:
    if not isinstance(parameter_overrides, Mapping) or not parameter_overrides:
        return []
    items: list[tuple[str, float]] = []
    for raw_name, raw_value in parameter_overrides.items():
        name = str(raw_name or "").strip()
        if not name:
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError, OverflowError):
            continue
        if not np.isfinite(value):
            continue
        items.append((name, value))
    return items


def _scalar_parameter_override_known(mechanism: Any, name: str) -> bool:
    meta = getattr(mechanism, "metadata", None)
    if not isinstance(meta, dict):
        return False
    scalar_bindings = meta.get("scalar_param_bindings")
    if isinstance(scalar_bindings, dict) and name in scalar_bindings:
        setter = getattr(scalar_bindings.get(name), "set", None)
        if callable(setter):
            return True
    scalar_info = meta.get("scalar_param_info")
    if isinstance(scalar_info, Mapping) and name in scalar_info:
        return True
    scalar_params = meta.get("scalar_params")
    if isinstance(scalar_params, dict) and name in scalar_params:
        return True
    return False


def _prepared_parameter_override_can_apply(mechanism: Any, name: str) -> bool:
    from kindred.core.simulator.step_indexing import lookup_step_param_target

    target_name = _canonical_step_override_name(mechanism, name)
    target = lookup_step_param_target(mechanism, target_name)
    if target is None:
        return _scalar_parameter_override_known(mechanism, name)
    kind, idx, role, _entry = target
    candidate = None
    try:
        if kind == "reaction" and role == "k":
            candidate = getattr(mechanism.reactions[int(idx)], "rate", None)
        elif kind == "equilibrium" and role in {"kf", "kr"}:
            candidate = getattr(mechanism.equilibria[int(idx)], str(role), None)
        elif kind == "equilibrium" and role == "Keq":
            meta = getattr(mechanism.equilibria[int(idx)], "metadata", {}) or {}
            if isinstance(meta, Mapping):
                candidate = meta.get(EquilibriumMetadataKeys.KEQ_INPUT)
    except Exception:
        candidate = None
    setter = getattr(candidate, "set", None)
    return callable(setter)


def _apply_parameter_overrides_to_prepared_mechanism(
    mechanism: Any,
    parameter_overrides: Mapping[str, Any] | None,
) -> _ParameterOverrideApplication:
    """Apply slider-style values and report whether dependent runtime math changed."""
    override_items = _coerce_parameter_override_items(parameter_overrides)
    if not override_items:
        return _ParameterOverrideApplication(rebuild_rhs=False, fully_applied=True)

    for name, _value in override_items:
        if not _prepared_parameter_override_can_apply(mechanism, name):
            return _ParameterOverrideApplication(rebuild_rhs=False, fully_applied=False)

    from kindred.core.simulator.step_indexing import lookup_step_param_target

    override_applied = False
    for name, value in override_items:
        target_name = _canonical_step_override_name(mechanism, name)
        target = lookup_step_param_target(mechanism, target_name)
        if target is None:
            override_applied = (
                _apply_scalar_parameter_override_to_prepared_mechanism(
                    mechanism,
                    name,
                    float(value),
                )
                or override_applied
            )
            continue
        kind, idx, role, _entry = target
        candidate = None
        try:
            if kind == "reaction" and role == "k":
                candidate = getattr(mechanism.reactions[int(idx)], "rate", None)
            elif kind == "equilibrium" and role in {"kf", "kr"}:
                candidate = getattr(mechanism.equilibria[int(idx)], str(role), None)
            elif kind == "equilibrium" and role == "Keq":
                meta = getattr(mechanism.equilibria[int(idx)], "metadata", {}) or {}
                if isinstance(meta, Mapping):
                    candidate = meta.get(EquilibriumMetadataKeys.KEQ_INPUT)
        except Exception:
            candidate = None

        setter = getattr(candidate, "set", None)
        if callable(setter):
            setter(float(value))
            override_applied = True
    if override_applied:
        from kindred.core.simulator.parameter_algebra import (
            apply_parameter_algebra_spec_to_mechanism,
            parameter_algebra_spec_from_mechanism,
        )

        spec = parameter_algebra_spec_from_mechanism(mechanism)
        if spec is not None:
            apply_parameter_algebra_spec_to_mechanism(
                spec,
                mechanism=mechanism,
                require_mutable=False,
            )
    return _ParameterOverrideApplication(
        rebuild_rhs=bool(override_applied),
        fully_applied=True,
    )


def apply_parameter_overrides_to_prepared_mechanism(
    mechanism: Any,
    parameter_overrides: Mapping[str, Any] | None,
) -> bool:
    """Apply slider-style values and report whether dependent runtime math changed."""
    return bool(
        _apply_parameter_overrides_to_prepared_mechanism(
            mechanism,
            parameter_overrides,
        ).rebuild_rhs
    )


def _canonical_step_override_name(mechanism: Any, name: str) -> str:
    name_s = str(name or "").strip()
    if not name_s:
        return ""
    try:
        from kindred.core.simulator.parameter_namespace import build_namespace_from_mechanism

        resolution = build_namespace_from_mechanism(mechanism).resolve(name_s)
    except Exception:
        return name_s
    if resolution.equilibrium_conflict_name is not None:
        return name_s
    return str(resolution.canonical_name or name_s)


def _apply_scalar_parameter_override_to_prepared_mechanism(
    mechanism: Any,
    name: str,
    value: float,
) -> bool:
    meta = getattr(mechanism, "metadata", None)
    if not isinstance(meta, dict):
        return False
    scalar_known = False
    scalar_bindings = meta.get("scalar_param_bindings")
    if isinstance(scalar_bindings, dict) and name in scalar_bindings:
        setter = getattr(scalar_bindings.get(name), "set", None)
        if callable(setter):
            setter(float(value))
            scalar_known = True
    scalar_params = meta.get("scalar_params")
    if not isinstance(scalar_params, dict):
        scalar_params = {}
        meta["scalar_params"] = scalar_params
    scalar_info = meta.get("scalar_param_info")
    if isinstance(scalar_info, Mapping) and name in scalar_info:
        scalar_known = True
    if name in scalar_params:
        scalar_known = True
    if not scalar_known:
        return False
    scalar_params[str(name)] = float(value)
    return True


def prepared_simulation_run_for_execution_request(
    prepared: PreparedSimulationRun,
    execution_request: SimulationExecutionRequest | Mapping[str, Any],
) -> PreparedSimulationRun:
    """Reuse a prepared runtime while applying request-local initials and slider values."""
    request_payload = coerce_simulation_execution_request(execution_request)
    if request_payload is None:
        return prepared

    override_application = _apply_parameter_overrides_to_prepared_mechanism(
        prepared.mechanism,
        request_payload.parameter_overrides,
    )
    if request_payload.parameter_overrides and not override_application.fully_applied:
        return prepare_simulation_worker_run(execution_request=request_payload)
    rebuild_rhs = bool(override_application.rebuild_rhs)
    rhs = prepared.request.rhs
    jacobian_func = prepared.request.jacobian_func
    temperature_schedule = prepared.request.temperature_schedule
    warnings = list(getattr(prepared, "warnings", None) or [])
    if rebuild_rhs:
        try:
            from kindred.core.ode_builder import build_ode_rhs_from_mechanism

            rhs = build_ode_rhs_from_mechanism(prepared.mechanism)
        except Exception as exc:
            raise SimulationPreparationError("ode_build", str(exc)) from exc
        try:
            solver_config = _build_solver_config(
                solver_input=str(prepared.request.solver),
                rtol=float(prepared.request.rtol),
                atol=float(prepared.request.atol),
                grid=dict(prepared.request.grid or {}),
                use_sparse_jacobian=bool(prepared.request.jacobian_func is not None),
                wegscheider_cyclicity_enabled=bool(
                    (getattr(prepared.mechanism, "metadata", {}) or {}).get(
                        MechanismMetadataKeys.WEGSCHEIDER_CYCLICITY_ENABLED,
                        WEGSCHEIDER_CYCLICITY_ENABLED_DEFAULT,
                    )
                ),
            )
            prepared_context = _build_prepared_run_context(
                mechanism=prepared.mechanism,
                solver_config=solver_config,
                temperature_schedule_override=prepared.request.temperature_schedule,
                jacobian_func_override=None,
            )
            jacobian_func = prepared_context.jacobian_func
            temperature_schedule = prepared_context.temperature_schedule
            warnings = list(prepared_context.warnings)
        except Exception as exc:
            raise SimulationPreparationError("parameter_overrides", str(exc)) from exc

    y0 = np.asarray(prepared.y0, dtype=float).reshape(-1).copy()
    initials = dict(request_payload.initials or {})
    if initials:
        for idx, sp in enumerate(prepared.species_names):
            if sp not in initials:
                continue
            try:
                y0[idx] = float(initials[sp])
            except (TypeError, ValueError, OverflowError):
                continue

    request = replace(
        prepared.request,
        rhs=rhs,
        t_span=(float(request_payload.t_span[0]), float(request_payload.t_span[1])),
        y0=np.asarray(y0, dtype=float).reshape(-1),
        jacobian_func=jacobian_func,
        temperature_schedule=temperature_schedule,
    )
    initials_for_algebra = {
        str(sp): float(y0[idx])
        for idx, sp in enumerate(prepared.species_names)
    }
    return replace(
        prepared,
        y0=np.asarray(y0, dtype=float).reshape(-1),
        initials_for_algebra=initials_for_algebra,
        warnings=warnings,
        request=request,
    )


def _build_prepared_run_context(
    *,
    mechanism: Any,
    solver_config: _PreparedSolverConfig,
    temperature_schedule_override: object = _MISSING,
    jacobian_func_override: Any = None,
) -> _PreparedRunContext:
    mech_meta = _metadata_view_for_mechanism(
        mechanism,
        temperature_schedule_override=temperature_schedule_override,
    )
    temperature_schedule = mech_meta.temperature_schedule
    warnings: List[str] = []

    jacobian_func = jacobian_func_override
    if (
        jacobian_func is None
        and temperature_schedule is not None
        and solver_config.use_sparse_jacobian
        and str(solver_config.solver).upper() in {"RADAU", "BDF"}
    ):
        message = "Sparse Jacobian disabled for scheduled-temperature run; falling back to dense Jacobian."
        logger.warning("%s", message)
        warnings.append(message)
    elif jacobian_func is None and solver_config.use_sparse_jacobian and str(solver_config.solver).upper() in {"RADAU", "BDF"}:
        try:
            from kindred.core.sparse_jacobian import build_sparse_jacobian

            jacobian_func = build_sparse_jacobian(mechanism)
        except Exception as exc:
            message = f"Sparse Jacobian unavailable; falling back to dense Jacobian: {exc}"
            logger.warning("%s", message, exc_info=True)
            warnings.append(message)
            jacobian_func = None

    return _PreparedRunContext(
        solver_config=solver_config,
        temperature_schedule=temperature_schedule,
        jacobian_func=jacobian_func,
        algebra_text=getattr(mech_meta, "algebra_text", None),
        warnings=tuple(warnings),
    )


def prepare_simulation_worker_run(
    *,
    mechanism_text: str = "",
    initials: Optional[Mapping[str, float]] = None,
    t_span: Tuple[float, float] = (0.0, 0.0),
    solver_config: Optional[Mapping[str, Any]] = None,
    prepared_payload: Optional[Mapping[str, Any]] = None,
    execution_request: SimulationExecutionRequest | Mapping[str, Any] | None = None,
    events: Optional[Iterable[Callable[[float, np.ndarray], float]]] = None,
    progress_callback: Optional[Callable[[float, float, float], None]] = None,
) -> PreparedSimulationRun:
    """
    Prepare an execution-ready SimulationRequest for the GUI SimulationWorker.

    This function is the single owner for:
    - Temperature schedule coercion (DSL schedule takes precedence)
    - Parameter algebra application
    - Metadata flags (e.g. Wegscheider cyclicity enablement)
    - Sparse Jacobian construction (when requested and supported by solver)
    """

    request_payload = coerce_simulation_execution_request(execution_request)
    structured_prepared_request = False
    if request_payload is not None:
        request_mechanism_text = str(request_payload.mechanism_text or "")
        initials = dict(request_payload.initials or {})
        t_span = (float(request_payload.t_span[0]), float(request_payload.t_span[1]))
        solver_config = dict(request_payload.solver_config or {})
        if prepared_payload is None:
            prepared_payload = request_payload.prepared_payload
        structured_prepared_request = prepared_payload is not None and request_payload.prepared_payload is not None
        if structured_prepared_request:
            mechanism_text = request_mechanism_text
        else:
            mechanism_text = str(request_mechanism_text or mechanism_text or "")

    if initials is None:
        initials = {}
    if solver_config is None:
        solver_config = {}

    try:
        prepared_solver_config = _build_solver_config(
            solver_input=str((solver_config or {}).get("solver") or DEFAULT_SOLVER_NAME),
            rtol=(solver_config or {}).get("rtol", 1e-6),
            atol=(solver_config or {}).get("atol", 1e-12),
            grid=(solver_config or {}).get("grid", {"N": 100}) or {"N": 100},
            use_sparse_jacobian=bool(
                (solver_config or {}).get(
                    "use_sparse_jacobian",
                    USE_SPARSE_JACOBIAN_DEFAULT,
                )
            ),
            wegscheider_cyclicity_enabled=bool(
                (solver_config or {}).get(
                    MechanismMetadataKeys.WEGSCHEIDER_CYCLICITY_ENABLED,
                    WEGSCHEIDER_CYCLICITY_ENABLED_DEFAULT,
                )
            ),
        )
    except Exception as exc:
        raise SimulationPreparationError("solver_config", str(exc)) from exc

    mechanism: Any
    rhs: Callable[..., np.ndarray] | None
    species_names: List[str]
    y0: np.ndarray
    temperature_schedule_override: object = _MISSING
    jacobian_func_override = None

    if prepared_payload is not None:
        try:
            (
                mechanism,
                rhs,
                species_names,
                y0,
                temperature_schedule_override,
                jacobian_func_override,
            ) = _validated_prepared_worker_payload(prepared_payload)

            if initials:
                for idx, sp in enumerate(species_names):
                    if sp in initials:
                        try:
                            y0[idx] = float(initials[sp])
                        except (TypeError, ValueError):
                            continue
        except Exception as exc:
            if isinstance(exc, SimulationPreparationError):
                raise
            raise _prepared_payload_failure(f"Invalid prepared payload: {exc}") from exc
        require_mutable = True
    else:
        from kindred.core.units import UnitsModel
        from kindred.core.simulator.dsl import parse_dsl_to_mechanism

        temperature_K = float(
            (solver_config or {}).get(MechanismMetadataKeys.TEMPERATURE_K, 298.15)
        )
        units = UnitsModel(temperature_K=temperature_K)
        try:
            mechanism = parse_dsl_to_mechanism(mechanism_text, initials=dict(initials or {}), units=units)
        except Exception as exc:
            raise SimulationPreparationError("parse", str(exc)) from exc

        species_names = list(getattr(mechanism, "species_names")())
        y0 = np.array([mechanism.species[sp].initial_conc for sp in species_names], dtype=float)
        rhs = None
        require_mutable = False

    meta = getattr(mechanism, "metadata", None)
    if isinstance(meta, dict):
        meta[MechanismMetadataKeys.WEGSCHEIDER_CYCLICITY_ENABLED] = bool(
            prepared_solver_config.wegscheider_cyclicity_enabled
        )

    try:
        from kindred.core.simulator.parameter_algebra import (
            apply_parameter_algebra_spec_to_mechanism,
            apply_parameter_algebra_to_mechanism,
            parameter_algebra_spec_from_mechanism,
        )

        if structured_prepared_request:
            spec = parameter_algebra_spec_from_mechanism(mechanism)
            if spec is not None:
                _ = apply_parameter_algebra_spec_to_mechanism(
                    spec,
                    mechanism=mechanism,
                    require_mutable=bool(require_mutable),
                )
        elif mechanism_text:
            _ = apply_parameter_algebra_to_mechanism(
                mechanism_text,
                mechanism=mechanism,
                require_mutable=bool(require_mutable),
            )
        else:
            spec = parameter_algebra_spec_from_mechanism(mechanism)
            if spec is not None:
                _ = apply_parameter_algebra_spec_to_mechanism(
                    spec,
                    mechanism=mechanism,
                    require_mutable=bool(require_mutable),
                )
    except Exception as exc:
        raise SimulationPreparationError("parameter_algebra", str(exc)) from exc

    if request_payload is not None and request_payload.parameter_overrides:
        try:
            _bind_parameters_to_mechanism(
                mechanism,
                sorted(str(name) for name in request_payload.parameter_overrides.keys()),
            )
            apply_parameter_overrides_to_prepared_mechanism(
                mechanism,
                request_payload.parameter_overrides,
            )
        except Exception as exc:
            raise SimulationPreparationError("parameter_overrides", str(exc)) from exc

    if rhs is None:
        try:
            from kindred.core.ode_builder import build_ode_rhs_from_mechanism

            rhs = build_ode_rhs_from_mechanism(mechanism)
        except Exception as exc:
            raise SimulationPreparationError("ode_build", str(exc)) from exc

    try:
        prepared_context = _build_prepared_run_context(
            mechanism=mechanism,
            solver_config=prepared_solver_config,
            temperature_schedule_override=temperature_schedule_override,
            jacobian_func_override=jacobian_func_override,
        )
    except Exception as exc:
        raise SimulationPreparationError("temperature_schedule", str(exc)) from exc

    temperature_schedule = prepared_context.temperature_schedule
    jacobian_func = prepared_context.jacobian_func

    try:
        initials_for_algebra = {sp: float(y0[idx]) for idx, sp in enumerate(species_names)}
    except Exception as exc:
        raise SimulationPreparationError("initials_for_algebra", str(exc)) from exc

    request = SimulationRequest(
        rhs=rhs,
        t_span=(float(t_span[0]), float(t_span[1])),
        y0=np.asarray(y0, dtype=float).reshape(-1),
        solver=str(prepared_solver_config.solver),
        rtol=float(prepared_solver_config.rtol),
        atol=float(prepared_solver_config.atol),
        grid=prepared_solver_config.grid,
        jacobian_func=jacobian_func,
        events=list(events) if events is not None else None,
        temperature_schedule=temperature_schedule,
        progress_callback=progress_callback,
    )

    return PreparedSimulationRun(
        mechanism=mechanism,
        rhs=rhs,
        y0=np.asarray(y0, dtype=float).reshape(-1),
        species_names=list(species_names),
        solver_input=str(prepared_solver_config.solver_input),
        solver_warning=str(prepared_solver_config.solver_warning) if prepared_solver_config.solver_warning else None,
        temperature_schedule=temperature_schedule,
        jacobian_func=jacobian_func,
        initials_for_algebra=initials_for_algebra,
        warnings=list(prepared_context.warnings),
        request=request,
    )


def _bind_parameters_to_mechanism(mech: Any, names: List[str]) -> Dict[str, Any]:
    from kindred.core.rate_binding import RateBinding
    from kindred.core.simulator.step_indexing import lookup_step_param_target
    from kindred.core.validation import try_parse_callable_finite_float

    bindings: Dict[str, RateBinding] = {}

    for raw_name in names:
        name = _canonical_step_override_name(mech, str(raw_name))
        target = lookup_step_param_target(mech, name)
        if target is None:
            continue
        kind, idx, role, _entry = target

        binding = bindings.get(name)
        if binding is None:
            if kind == "reaction" and role == "k":
                rxn = mech.reactions[idx]
                parsed, ok = try_parse_callable_finite_float(getattr(rxn, "rate", None))
                init = float(parsed) if ok else 1.0
            elif kind == "equilibrium" and role in {"kf", "kr"}:
                eq = mech.equilibria[idx]
                parsed, ok = try_parse_callable_finite_float(getattr(eq, role, None))
                init = float(parsed) if ok else 1.0
            elif kind == "equilibrium" and role == "Keq":
                eq = mech.equilibria[idx]
                meta = getattr(eq, "metadata", {}) or {}
                parsed, ok = try_parse_callable_finite_float(
                    meta.get(EquilibriumMetadataKeys.KEQ_INPUT),
                )
                if ok:
                    init = float(parsed)
                else:
                    fallback, ok_fallback = try_parse_callable_finite_float(
                        getattr(eq, "Keq", 1.0),
                    )
                    init = float(fallback) if ok_fallback else 1.0
            else:
                init = 1.0
            binding = RateBinding(name=name, value=float(init))
            bindings[name] = binding

        if kind == "reaction" and role == "k":
            rxn = mech.reactions[idx]
            mech.reactions[idx] = replace(rxn, rate=binding)
        elif kind == "equilibrium" and role in {"kf", "kr"}:
            eq = mech.equilibria[idx]
            mech.equilibria[idx] = replace(eq, **{role: binding})
        elif kind == "equilibrium" and role == "Keq":
            eq = mech.equilibria[idx]
            meta = dict(getattr(eq, "metadata", {}) or {})
            meta[EquilibriumMetadataKeys.KEQ_INPUT] = binding
            mech.equilibria[idx] = replace(eq, metadata=meta)

    return bindings


@dataclass
class _StateNetworkEnergyBindingState:
    units: Any
    temperature_K: float
    kappa: float
    degeneracy_ratio_fwd: float
    degeneracy_ratio_rev: float
    std_ratio_fwd: float
    std_ratio_rev: float
    dG_act_fwd_J_per_mol: float
    dG_eq_J_per_mol: float
    metadata: Dict[str, Any]

    def _sync_metadata(self) -> None:
        self.metadata[EquilibriumMetadataKeys.DG_EQ_J_PER_MOL] = float(self.dG_eq_J_per_mol)
        self.metadata["dG_act_fwd_J_per_mol"] = float(self.dG_act_fwd_J_per_mol)
        self.metadata["dG_act_rev_J_per_mol"] = float(self.dG_act_rev_J_per_mol())
        self.metadata["kf"] = float(self.kf())
        self.metadata["kr"] = float(self.kr())
        self.metadata["Keq"] = float(self.Keq())

    def dG_act_rev_J_per_mol(self) -> float:
        return float(self.dG_act_fwd_J_per_mol - self.dG_eq_J_per_mol)

    def _eyring_rate(self, dG_barrier_J_per_mol: float, *, degeneracy_ratio: float, std_ratio: float) -> float:
        from kindred.core.constants import R, h, k_B

        T = float(self.temperature_K)
        exponent = math.exp(-float(dG_barrier_J_per_mol) / (float(R) * T))
        prefactor = float(self.kappa) * float(k_B) * T / float(h)
        return float(prefactor * float(degeneracy_ratio) * exponent * float(std_ratio))

    def kf(self) -> float:
        return self._eyring_rate(
            self.dG_act_fwd_J_per_mol,
            degeneracy_ratio=self.degeneracy_ratio_fwd,
            std_ratio=self.std_ratio_fwd,
        )

    def kr(self) -> float:
        return self._eyring_rate(
            self.dG_act_rev_J_per_mol(),
            degeneracy_ratio=self.degeneracy_ratio_rev,
            std_ratio=self.std_ratio_rev,
        )

    def Keq(self) -> float:
        from kindred.core.kinetics import K_from_deltaG_eq

        return float(K_from_deltaG_eq(self.dG_eq_J_per_mol, self.temperature_K))

    def binding_value(self, role: str) -> float:
        if role == "dG_act_fwd":
            return float(self.units.from_jmol(self.dG_act_fwd_J_per_mol))
        if role == "dG_eq":
            return float(self.units.from_jmol(self.dG_eq_J_per_mol))
        if role == "kf":
            return float(self.kf())
        if role == "kr":
            return float(self.kr())
        if role == "Keq":
            return float(self.Keq())
        raise KeyError(f"Unknown state-network energy binding role: {role}")

    def apply_slider_value(self, role: str, value: float) -> None:
        if role == "dG_act_fwd":
            self.dG_act_fwd_J_per_mol = float(self.units.to_jmol(value))
        elif role == "dG_eq":
            self.dG_eq_J_per_mol = float(self.units.to_jmol(value))
        else:
            raise KeyError(f"Unknown state-network energy slider role: {role}")
        self._sync_metadata()


@dataclass
class _FastEquilibriumEnergyBindingState:
    units: Any
    temperature_K: float
    kf_fixed: float
    std_ratio: float
    dG_eq_J_per_mol: float
    metadata: Dict[str, Any]
    kf_getter: Optional[Callable[[], float]] = None
    kf_setter: Optional[Callable[[float], None]] = None

    def _sync_metadata(self) -> None:
        self.metadata[EquilibriumMetadataKeys.DG_EQ_J_PER_MOL] = float(self.dG_eq_J_per_mol)
        self.metadata["kf"] = float(self.kf())
        self.metadata["kr"] = float(self.kr())
        self.metadata["Keq"] = float(self.Keq())

    def Keq(self) -> float:
        from kindred.core.kinetics import K_from_deltaG_eq

        return float(K_from_deltaG_eq(self.dG_eq_J_per_mol, self.temperature_K))

    def kf(self) -> float:
        if self.kf_getter is not None:
            return float(self.kf_getter())
        return float(self.kf_fixed)

    def kr(self) -> float:
        return float(self.kf() / (self.Keq() * self.std_ratio))

    def binding_value(self, role: str) -> float:
        if role == "dG_eq_fast":
            return float(self.units.from_jmol(self.dG_eq_J_per_mol))
        if role == "kf":
            return float(self.kf())
        if role == "kr":
            return float(self.kr())
        if role == "Keq":
            return float(self.Keq())
        raise KeyError(f"Unknown fast-equilibrium energy binding role: {role}")

    def apply_slider_value(self, role: str, value: float) -> None:
        from kindred.core.constants import R

        value_f = float(value)
        if role == "dG_eq_fast":
            self.dG_eq_J_per_mol = float(self.units.to_jmol(value_f))
        elif role == "Keq":
            if not (math.isfinite(value_f) and value_f > 0.0):
                raise ValueError("Fast-equilibrium Keq must be positive and finite")
            self.dG_eq_J_per_mol = float(-float(R) * float(self.temperature_K) * math.log(value_f))
        elif role == "kr":
            if not (math.isfinite(value_f) and value_f > 0.0):
                raise ValueError("Fast-equilibrium kr must be positive and finite")
            Keq_value = float(self.kf() / (value_f * self.std_ratio))
            if not (math.isfinite(Keq_value) and Keq_value > 0.0):
                raise ValueError("Fast-equilibrium derived Keq must be positive and finite")
            self.dG_eq_J_per_mol = float(-float(R) * float(self.temperature_K) * math.log(Keq_value))
        elif role == "kf":
            if not (math.isfinite(value_f) and value_f > 0.0):
                raise ValueError("Fast-equilibrium kf must be positive and finite")
            if self.kf_setter is not None:
                self.kf_setter(value_f)
            self.kf_fixed = float(value_f)
        else:
            raise KeyError(f"Unknown fast-equilibrium energy slider role: {role}")
        self._sync_metadata()


class _StructuredEnergyBinding(RateBinding):
    def __init__(self, *, name: str, state: Any, role: str) -> None:
        super().__init__(name=str(name), value=float(state.binding_value(role)))
        self._state = state
        self._role = str(role)

    def __call__(self) -> float:
        current = float(self._state.binding_value(self._role))
        self.value = current
        return current

    def set(self, new_value: float) -> None:
        self._state.apply_slider_value(self._role, float(new_value))
        self.value = float(self._state.binding_value(self._role))


def _canonical_fast_equilibrium_side(stoich: Mapping[str, float]) -> str:
    def _fmt_coeff(value: float) -> str:
        rounded = round(float(value))
        if abs(float(value) - float(rounded)) < 1e-12:
            return str(int(rounded))
        return f"{float(value):g}"

    parts: List[str] = []
    for name in sorted(str(key) for key in stoich.keys()):
        coeff = float(stoich[name])
        if abs(coeff - 1.0) < 1e-12:
            parts.append(str(name))
        else:
            parts.append(f"{_fmt_coeff(coeff)}{name}")
    return "_".join(parts)


def _install_energy_bindings(mechanism: Any, names: List[str]) -> Dict[str, Any]:
    from kindred.core.mechanism_metadata import MechanismMetadataView
    from kindred.core.rate_binding import RateBinding
    from kindred.core.units import UnitsModel
    from kindred.core.validation import try_parse_callable_finite_float
    from kindred.core.constants import R

    requested = {str(name) for name in (names or []) if str(name).strip()}
    if not requested:
        return {}

    mech_meta = MechanismMetadataView.from_metadata(getattr(mechanism, "metadata", {}) or {})
    units = UnitsModel(energy_unit=str(mech_meta.energy_unit), temperature_K=float(mech_meta.temperature_K))
    bindings: Dict[str, Any] = {}

    for index, eq in enumerate(getattr(mechanism, "equilibria", []) or []):
        metadata = dict(getattr(eq, "metadata", {}) or {})

        reactant = str(metadata.get("reactant") or "")
        product = str(metadata.get("product") or "")
        ts = str(metadata.get("ts") or "")
        if str(metadata.get("source") or "") == "state_network" and reactant and product and ts:
            act_name = f"dGact_fwd__{ts}__{reactant}__{product}"
            eq_name = f"dG_eq__{ts}__{reactant}__{product}"
            if act_name not in requested and eq_name not in requested:
                continue

            state = _StateNetworkEnergyBindingState(
                units=units,
                temperature_K=float(metadata.get("temperature_K") or mech_meta.temperature_K),
                kappa=float(metadata.get("kappa") or 1.0),
                degeneracy_ratio_fwd=float(metadata.get("degeneracy_ratio_fwd") or 1.0),
                degeneracy_ratio_rev=float(metadata.get("degeneracy_ratio_rev") or 1.0),
                std_ratio_fwd=float(metadata.get("std_conc_product_ts") or 1.0)
                / max(1e-300, float(metadata.get("std_conc_product_reactant") or 1.0)),
                std_ratio_rev=float(metadata.get("std_conc_product_ts") or 1.0)
                / max(1e-300, float(metadata.get("std_conc_product_product") or 1.0)),
                dG_act_fwd_J_per_mol=float(metadata.get("dG_act_fwd_J_per_mol") or 0.0),
                dG_eq_J_per_mol=float(metadata.get(EquilibriumMetadataKeys.DG_EQ_J_PER_MOL) or 0.0),
                metadata=metadata,
            )
            state._sync_metadata()
            kf_binding = _StructuredEnergyBinding(name=f"{act_name}:kf", state=state, role="kf")
            kr_binding = _StructuredEnergyBinding(name=f"{eq_name}:kr", state=state, role="kr")
            Keq_binding = _StructuredEnergyBinding(name=f"{eq_name}:Keq", state=state, role="Keq")
            metadata[EquilibriumMetadataKeys.KEQ_INPUT] = Keq_binding
            mechanism.equilibria[index] = replace(
                eq,
                kf=kf_binding,
                kr=kr_binding,
                Keq=Keq_binding,
                metadata=metadata,
            )
            if act_name in requested:
                bindings[act_name] = _StructuredEnergyBinding(name=act_name, state=state, role="dG_act_fwd")
            if eq_name in requested:
                bindings[eq_name] = _StructuredEnergyBinding(name=eq_name, state=state, role="dG_eq")
            continue

        fast_equilibrium = bool(metadata.get(EquilibriumMetadataKeys.FAST_EQUILIBRIUM, getattr(eq, "fast", False)))
        if not fast_equilibrium:
            continue
        slug = _canonical_fast_equilibrium_side(getattr(eq, "stoich_forward", {}) or {})
        slug += "__" + _canonical_fast_equilibrium_side(getattr(eq, "stoich_back", {}) or {})
        slider_name = f"dG_eq_fast__feq__{slug}"
        if slider_name not in requested:
            continue

        kf_value = getattr(eq, "kf", None)
        if isinstance(kf_value, RateBinding):
            kf_fixed = float(kf_value())
            kf_getter = kf_value
            kf_setter = kf_value.set
        else:
            kf_fixed = float(kf_value if kf_value is not None else 1.0)
            kf_getter = None
            kf_setter = None
        dG_eq_raw = metadata.get(EquilibriumMetadataKeys.DG_EQ_J_PER_MOL)
        dG_eq_parsed, dG_eq_ok = try_parse_callable_finite_float(dG_eq_raw)
        if dG_eq_ok:
            dG_eq_J_per_mol = float(dG_eq_parsed)
        else:
            Keq_input_value, Keq_input_ok = try_parse_callable_finite_float(metadata.get(EquilibriumMetadataKeys.KEQ_INPUT))
            if Keq_input_ok and math.isfinite(Keq_input_value) and Keq_input_value > 0.0:
                dG_eq_J_per_mol = float(-float(R) * float(mech_meta.temperature_K) * math.log(float(Keq_input_value)))
            else:
                dG_eq_J_per_mol = 0.0
        state = _FastEquilibriumEnergyBindingState(
            units=units,
            temperature_K=float(mech_meta.temperature_K),
            kf_fixed=float(kf_fixed),
            kf_getter=kf_getter,
            kf_setter=kf_setter,
            std_ratio=max(1e-300, float(metadata.get("std_ratio") or 1.0)),
            dG_eq_J_per_mol=float(dG_eq_J_per_mol),
            metadata=metadata,
        )
        state._sync_metadata()
        kr_binding = _StructuredEnergyBinding(name=f"{slider_name}:kr", state=state, role="kr")
        Keq_binding = _StructuredEnergyBinding(name=f"{slider_name}:Keq", state=state, role="Keq")
        metadata[EquilibriumMetadataKeys.KEQ_INPUT] = Keq_binding
        mechanism.equilibria[index] = replace(
            eq,
            kr=kr_binding,
            Keq=Keq_binding,
            metadata=metadata,
        )
        bindings[slider_name] = _StructuredEnergyBinding(name=slider_name, state=state, role="dG_eq_fast")

    return bindings


def prepare_bound_mechanism(
    mechanism_text: str,
    param_names: List[str],
    *,
    temperature_K: float = 298.15,
    initials: Optional[Dict[str, float]] = None,
    use_advanced_dsl: bool = True,
    wegscheider_cyclicity_enabled: bool = WEGSCHEIDER_CYCLICITY_ENABLED_DEFAULT,
) -> BoundMechanism:
    """
    Parse the DSL once, bind selected parameters, and compile the RHS.

    Returns a BoundMechanism holding mutable RateBinding objects so callers can
    update parameter values without re-parsing or rebuilding the ODE system.
    """
    from kindred.core.simulator.dsl import parse_dsl_to_mechanism
    from kindred.core.units import UnitsModel
    from kindred.core.ode_builder import build_ode_rhs_from_mechanism

    initials = initials or {}
    units = UnitsModel(temperature_K=temperature_K)

    if not use_advanced_dsl:
        logger.debug("Legacy basic DSL mode requested; using advanced parser instead.")
    try:
        mechanism = parse_dsl_to_mechanism(mechanism_text, initials=initials, units=units)
    except Exception as exc:
        raise _fit_simulation_error_from_preparation_error(
            _prepare_preparation_failure("parse", exc)
        ) from exc
    meta = getattr(mechanism, "metadata", None)
    if isinstance(meta, dict):
        meta[MechanismMetadataKeys.WEGSCHEIDER_CYCLICITY_ENABLED] = bool(
            wegscheider_cyclicity_enabled
        )
    species_names = mechanism.species_names()
    y0 = np.array([mechanism.species[sp].initial_conc for sp in species_names])

    try:
        from kindred.core.simulator.parameter_algebra import (
            mechanism_parameter_namespace,
            parse_parameter_algebra_spec_from_dsl_text,
        )
        from kindred.core.simulator.step_indexing import get_step_index_map

        mechanism_namespace = mechanism_parameter_namespace(mechanism)
        spec = parse_parameter_algebra_spec_from_dsl_text(
            mechanism_text,
            mechanism_namespace=mechanism_namespace,
        )
        constrained = {
            stmt.name
            for stmt in (spec.param_statements or [])
            if re.match(r"^(k|kf|kr|Keq)\d+$", str(stmt.name))
        }
        k_derived = set()
        for entry in get_step_index_map(mechanism):
            if str(entry.get("kind") or "") != "equilibrium":
                continue
            if not bool(entry.get("has_Keq_param")):
                continue
            step_idx_raw = entry.get("step_index")
            if isinstance(step_idx_raw, int):
                n = int(step_idx_raw)
            elif isinstance(step_idx_raw, str) and step_idx_raw.isdigit():
                n = int(step_idx_raw)
            else:
                continue
            derive_rate = str(entry.get("derive_rate") or "kr")
            if derive_rate not in {"kf", "kr"}:
                derive_rate = "kr"
            k_derived.add(f"{derive_rate}{n}")

        wegscheider_derived = set()
        if bool(wegscheider_cyclicity_enabled):
            from kindred.core.simulator.wegscheider import derived_parameter_names_for_cyclicity

            wegscheider_derived = derived_parameter_names_for_cyclicity(
                mechanism,
                constrained_param_names={str(x) for x in constrained},
            )
        requested = set(param_names or [])
        mech_bind_names = sorted(
            {
                str(n)
                for n in (requested | constrained | k_derived | wegscheider_derived)
                if re.match(r"^(k|kf|kr|Keq)\d+$", str(n))
            }
        )
    except Exception as exc:
        raise _fit_simulation_error_from_preparation_error(
            _prepare_preparation_failure("parameter_algebra", exc)
        ) from exc

    bindings = _bind_parameters_to_mechanism(mechanism, mech_bind_names)

    try:
        from kindred.core.simulator.parameter_algebra import apply_parameter_algebra_to_mechanism

        _ = apply_parameter_algebra_to_mechanism(mechanism_text, mechanism=mechanism, require_mutable=True)
    except Exception as exc:
        raise _fit_simulation_error_from_preparation_error(
            _prepare_preparation_failure("parameter_algebra", exc)
        ) from exc

    energy_bindings = _install_energy_bindings(mechanism, list(param_names or []))

    scalar_bindings = (getattr(mechanism, "metadata", {}) or {}).get("scalar_param_bindings") or {}
    if isinstance(scalar_bindings, dict):
        for name, binding in scalar_bindings.items():
            bindings.setdefault(str(name), binding)
    bindings.update(energy_bindings)

    try:
        rhs = build_ode_rhs_from_mechanism(mechanism)
    except Exception as exc:
        raise _fit_simulation_error_from_preparation_error(
            _prepare_preparation_failure("ode_build", exc)
        ) from exc

    return BoundMechanism(
        mechanism=mechanism,
        rhs=rhs,
        bindings=bindings,
        species_names=species_names,
        y0=y0,
        param_names=sorted(set(bindings.keys())),
        mechanism_text=mechanism_text,
    )


def prepare_fitting_objective_context(
    *,
    mechanism_text: str,
    param_names: List[str],
    t_exp: np.ndarray,
    target_species: str,
    temperature_K: float = 298.15,
    initials: Optional[Dict[str, float]] = None,
    solver: str = DEFAULT_SOLVER_NAME,
    rtol: float = 1e-6,
    atol: float = 1e-12,
    wegscheider_cyclicity_enabled: bool = WEGSCHEIDER_CYCLICITY_ENABLED_DEFAULT,
    prepare_func: Callable[..., object] | None = None,
) -> PreparedFittingObjectiveContext:
    from kindred.core.algebra.simulation_series import compile_algebra_observables
    from kindred.core.exceptions import ErrorContext

    try:
        prepared_solver_config = _build_solver_config(
            solver_input=str(solver or DEFAULT_SOLVER_NAME),
            rtol=rtol,
            atol=atol,
            grid={"N": int(np.asarray(t_exp, dtype=float).reshape(-1).size)},
            use_sparse_jacobian=USE_SPARSE_JACOBIAN_DEFAULT,
            wegscheider_cyclicity_enabled=bool(wegscheider_cyclicity_enabled),
        )
    except Exception as exc:
        raise _fit_simulation_error_from_preparation_error(
            _prepare_preparation_failure("solver_config", exc)
        ) from exc
    if prepared_solver_config.solver_warning:
        logger.warning("Solver normalization: %s (requested=%r)", prepared_solver_config.solver_warning, str(solver))

    prepare = prepare_bound_mechanism if prepare_func is None else prepare_func
    try:
        bound = prepare(
            mechanism_text=mechanism_text,
            param_names=param_names,
            temperature_K=temperature_K,
            initials=initials,
            use_advanced_dsl=True,
            wegscheider_cyclicity_enabled=bool(wegscheider_cyclicity_enabled),
        )
    except SimulationPreparationError as exc:
        raise _fit_simulation_error_from_preparation_error(exc) from exc
    if not isinstance(bound, BoundMechanism):
        raise TypeError("prepare_func must return BoundMechanism-compatible output.")

    try:
        prepared_run_context = _build_prepared_run_context(
            mechanism=bound.mechanism,
            solver_config=prepared_solver_config,
        )
    except Exception as exc:
        if isinstance(exc, SimulationPreparationError):
            raise _fit_simulation_error_from_preparation_error(exc) from exc
        raise _fit_simulation_error_from_preparation_error(
            _prepare_preparation_failure("temperature_schedule", exc)
        ) from exc
    compiled_algebra = None
    algebra_observables: set[str] = set()
    algebra_text = prepared_run_context.algebra_text
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
        algebra_observables = set(compiled_algebra.observable_names)

    target_is_species = target_species in getattr(bound.mechanism, "species", {})
    target_is_algebra = target_species in algebra_observables
    if not target_is_species and not target_is_algebra:
        raise FitSimulationError(
            f"Target species '{target_species}' not found in mechanism species or Algebra observables.",
            details={"fatal": True},
        )

    target_species_index = bound.species_names.index(target_species) if target_is_species else None
    request = SimulationRequest(
        rhs=bound.rhs,
        t_span=(float(t_exp[0]), float(t_exp[-1])),
        y0=bound.y0,
        solver=str(prepared_solver_config.solver),
        rtol=float(prepared_solver_config.rtol),
        atol=float(prepared_solver_config.atol),
        t_eval=np.asarray(t_exp, dtype=float).reshape(-1),
        temperature_schedule=prepared_run_context.temperature_schedule,
    )
    initials_for_algebra = {
        name: float(bound.y0[idx]) for idx, name in enumerate(bound.species_names)
    }
    return PreparedFittingObjectiveContext(
        bound=bound,
        requested_param_names=[str(name) for name in (param_names or [])],
        request=request,
        target_species=str(target_species),
        target_is_species=bool(target_is_species),
        target_species_index=target_species_index,
        compiled_algebra=compiled_algebra,
        initials_for_algebra=initials_for_algebra,
        temperature_K=float(temperature_K),
    )


def build_prepared_simulation_func(
    *,
    mechanism_text: str,
    param_names: List[str],
    t_end: float,
    num_points: int,
    temperature_K: float = 298.15,
    solver: str = DEFAULT_SOLVER_NAME,
    rtol: float = 1e-6,
    atol: float = 1e-12,
    use_sparse_jacobian: bool = USE_SPARSE_JACOBIAN_DEFAULT,
    wegscheider_cyclicity_enabled: bool = WEGSCHEIDER_CYCLICITY_ENABLED_DEFAULT,
    initial_prefix: str = "init:",
) -> Callable[[Dict[str, float]], SimulationSeriesPayload]:
    from kindred.core.algebra.simulation_series import (
        CompiledAlgebraSeries,
        compile_algebra_observables,
        evaluate_compiled_algebra_series_for_simulation,
    )
    from kindred.core.algebra.errors import (
        AlgebraError,
        AlgebraNameError,
        AlgebraShadowError,
        AlgebraSyntaxError,
    )
    from kindred.core.exceptions import ErrorContext
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

    bound: Optional[BoundMechanism] = None
    species_index: Dict[str, int] = {}
    temperature_schedule: Optional[TemperatureScheduleProtocol] = None
    jacobian_func = None
    last_shared_fp: Optional[Tuple[Tuple[str, float], ...]] = None
    compiled_algebra: Optional[CompiledAlgebraSeries] = None

    def _ensure_prepared() -> None:
        nonlocal bound, species_index, temperature_schedule, jacobian_func, compiled_algebra
        if bound is not None:
            return
        bound = prepare_bound_mechanism(
            mechanism_text=mechanism_text,
            param_names=list(param_names or []),
            temperature_K=float(temperature_K),
            initials={},
            use_advanced_dsl=True,
            wegscheider_cyclicity_enabled=bool(wegscheider_cyclicity_enabled),
        )
        species_index = {name: idx for idx, name in enumerate(bound.species_names)}
        prepared_context = _build_prepared_run_context(
            mechanism=bound.mechanism,
            solver_config=prepared_solver_config,
            jacobian_func_override=jacobian_func,
        )
        temperature_schedule = prepared_context.temperature_schedule
        jacobian_func = prepared_context.jacobian_func
        algebra_text = prepared_context.algebra_text
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

    def simulation_func(params: Dict[str, float]) -> SimulationSeriesPayload:
        nonlocal last_shared_fp
        _ensure_prepared()
        assert bound is not None

        initial_overrides: Dict[str, float] = {}
        shared_values: Dict[str, float] = {}
        for key, raw_val in (params or {}).items():
            name = str(key)
            try:
                value = float(raw_val)
            except (TypeError, ValueError) as exc:
                raise FitSimulationError(
                    f"Invalid parameter value for {name!r}: {raw_val!r}",
                    details={"fatal": True},
                ) from exc
            if name.startswith(initial_prefix):
                initial_overrides[name[len(initial_prefix) :]] = value
            else:
                shared_values[name] = value

        shared_fp = tuple(sorted((name, float(val)) for name, val in shared_values.items()))
        if shared_fp != last_shared_fp:
            for name, value in shared_values.items():
                binding = bound.bindings.get(name)
                if binding is None:
                    continue
                try:
                    binding.set(float(value))
                except Exception as exc:
                    raise FitSimulationError(
                        f"Failed to update parameter binding {name!r}: {exc}",
                        details={"fatal": True},
                    ) from exc
            try:
                from kindred.core.simulator.parameter_algebra import apply_parameter_algebra_to_mechanism

                _ = apply_parameter_algebra_to_mechanism(
                    mechanism_text,
                    mechanism=bound.mechanism,
                    require_mutable=True,
                )
            except Exception as exc:
                raise FitSimulationError(f"Parameter algebra failed during global simulation: {exc}") from exc
            last_shared_fp = shared_fp

        y0 = np.asarray(bound.y0, dtype=float).copy()
        for species_name, value in initial_overrides.items():
            idx = species_index.get(species_name)
            if idx is None:
                continue
            y0[idx] = float(value)

        request = SimulationRequest(
            rhs=bound.rhs,
            t_span=(0.0, float(t_end)),
            y0=y0,
            solver=str(prepared_solver_config.solver),
            rtol=float(prepared_solver_config.rtol),
            atol=float(prepared_solver_config.atol),
            grid={"N": grid_n},
            jacobian_func=jacobian_func,
            temperature_schedule=temperature_schedule,
        )
        result = _solve_request(request)
        species_payload = {name: result.Y[idx, :].copy() for idx, name in enumerate(bound.species_names)}

        algebra_scalars: Dict[str, float] = {}
        if compiled_algebra is not None:
            try:
                initials_map = {name: float(y0[idx]) for idx, name in enumerate(bound.species_names)}
                species_series = {name: result.Y[idx, :] for idx, name in enumerate(bound.species_names)}
                algebra_series, algebra_scalars = evaluate_compiled_algebra_series_for_simulation(
                    bound.mechanism,
                    compiled_algebra,
                    t=result.t,
                    species_series=species_series,
                    initials=initials_map,
                    temperature_K=float(temperature_K),
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

        return SimulationSeriesPayload(
            t=result.t.copy(),
            species=species_payload,
            algebra_scalars=dict(algebra_scalars),
        )

    simulation_func._kindred_prepared_simulation_meta = prepared_meta  # type: ignore[attr-defined]
    return simulation_func
