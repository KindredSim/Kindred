from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Optional

import numpy as np

from kindred.core.simulation_preparation import SimulationExecutionRequest, SimulationPreparationError

__all__ = [
    "SimulationAlgebraPolicy",
    "SimulationExecutionResult",
    "SimulationPlan",
]


class SimulationAlgebraPolicy(str, Enum):
    """Caller-level algebra handling policy carried by simulation plans."""

    GUI_BEST_EFFORT = "gui_best_effort"
    BATCH_BEST_EFFORT = "batch_best_effort"
    FITTING_STRICT = "fitting_strict"


def _copy_payload_value(value: Any, memo: Optional[Dict[int, Any]] = None) -> Any:
    if memo is None:
        memo = {}
    value_id = id(value)
    if value_id in memo:
        return memo[value_id]
    if isinstance(value, np.ndarray):
        copied = np.array(value, copy=True)
        memo[value_id] = copied
        return copied
    if isinstance(value, Mapping):
        copied: Dict[str, Any] = {}
        memo[value_id] = copied
        copied.update({str(key): _copy_payload_value(item, memo) for key, item in value.items()})
        return copied
    if isinstance(value, list):
        copied_list: list[Any] = []
        memo[value_id] = copied_list
        copied_list.extend(_copy_payload_value(item, memo) for item in value)
        return copied_list
    if isinstance(value, tuple):
        copied_tuple = tuple(_copy_payload_value(item, memo) for item in value)
        memo[value_id] = copied_tuple
        return copied_tuple
    try:
        return copy.deepcopy(value, memo)
    except Exception:
        return value


def _copy_optional_mapping(value: Optional[Mapping[str, Any]], *, field_name: str) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping or None.")
    return _copy_payload_value(value)


def _coerce_algebra_policy(value: SimulationAlgebraPolicy | str) -> SimulationAlgebraPolicy:
    if isinstance(value, SimulationAlgebraPolicy):
        return value
    try:
        return SimulationAlgebraPolicy(str(value))
    except ValueError as exc:
        raise ValueError(f"Invalid simulation plan algebra_policy: {value!r}") from exc


def _coerce_execution_request(value: SimulationExecutionRequest | Mapping[str, Any]) -> SimulationExecutionRequest:
    if isinstance(value, SimulationExecutionRequest):
        payload = value.to_payload()
    elif isinstance(value, Mapping):
        payload = dict(value)
    else:
        raise TypeError("execution_request must be a SimulationExecutionRequest or mapping.")
    try:
        return SimulationExecutionRequest.from_mapping(_copy_payload_value(payload))
    except SimulationPreparationError as exc:
        raise ValueError(f"Invalid simulation plan execution_request: {exc}") from exc


@dataclass(frozen=True)
class SimulationPlan:
    """Typed no-behavior-change wrapper around a simulation execution request."""

    execution_request: SimulationExecutionRequest
    execution_mode: str
    algebra_policy: SimulationAlgebraPolicy
    cache_identity_payload: Optional[Dict[str, Any]] = None
    cache_scope_payload: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: int = 1

    def __post_init__(self) -> None:
        execution_mode = str(self.execution_mode or "").strip()
        if not execution_mode:
            raise ValueError("SimulationPlan execution_mode must not be empty.")
        object.__setattr__(self, "version", int(self.version))
        object.__setattr__(self, "execution_mode", execution_mode)
        object.__setattr__(self, "algebra_policy", _coerce_algebra_policy(self.algebra_policy))
        object.__setattr__(self, "execution_request", _coerce_execution_request(self.execution_request))
        object.__setattr__(
            self,
            "cache_identity_payload",
            _copy_optional_mapping(self.cache_identity_payload, field_name="cache_identity_payload"),
        )
        object.__setattr__(
            self,
            "cache_scope_payload",
            _copy_optional_mapping(self.cache_scope_payload, field_name="cache_scope_payload"),
        )
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping.")
        object.__setattr__(self, "metadata", _copy_payload_value(self.metadata))

    @classmethod
    def from_execution_request(
        cls,
        execution_request: SimulationExecutionRequest | Mapping[str, Any],
        *,
        execution_mode: str,
        algebra_policy: SimulationAlgebraPolicy | str,
        cache_identity_payload: Optional[Mapping[str, Any]] = None,
        cache_scope_payload: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        version: int = 1,
    ) -> "SimulationPlan":
        return cls(
            version=int(version),
            execution_request=_coerce_execution_request(execution_request),
            execution_mode=str(execution_mode),
            algebra_policy=_coerce_algebra_policy(algebra_policy),
            cache_identity_payload=_copy_optional_mapping(
                cache_identity_payload,
                field_name="cache_identity_payload",
            ),
            cache_scope_payload=_copy_optional_mapping(
                cache_scope_payload,
                field_name="cache_scope_payload",
            ),
            metadata=_copy_payload_value(dict(metadata or {})),
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "SimulationPlan":
        if not isinstance(payload, Mapping):
            raise ValueError("SimulationPlan payload must be a mapping.")
        if "execution_request" not in payload:
            raise ValueError("SimulationPlan payload missing execution_request.")
        execution_request = payload.get("execution_request")
        if not isinstance(execution_request, Mapping):
            raise ValueError("SimulationPlan execution_request must be a mapping.")
        return cls(
            version=int(payload.get("version") or 1),
            execution_request=execution_request,
            execution_mode=str(payload.get("execution_mode") or ""),
            algebra_policy=_coerce_algebra_policy(payload.get("algebra_policy", "")),
            cache_identity_payload=_copy_optional_mapping(
                payload.get("cache_identity_payload"),
                field_name="cache_identity_payload",
            ),
            cache_scope_payload=_copy_optional_mapping(
                payload.get("cache_scope_payload"),
                field_name="cache_scope_payload",
            ),
            metadata=_copy_optional_mapping(payload.get("metadata") or {}, field_name="metadata") or {},
        )

    def to_payload(self) -> Dict[str, Any]:
        return {
            "version": int(self.version),
            "execution_mode": str(self.execution_mode),
            "algebra_policy": self.algebra_policy.value,
            "execution_request": _copy_payload_value(self.execution_request.to_payload()),
            "cache_identity_payload": _copy_optional_mapping(
                self.cache_identity_payload,
                field_name="cache_identity_payload",
            ),
            "cache_scope_payload": _copy_optional_mapping(
                self.cache_scope_payload,
                field_name="cache_scope_payload",
            ),
            "metadata": _copy_payload_value(self.metadata),
        }

    def to_execution_request(self) -> SimulationExecutionRequest:
        return _coerce_execution_request(self.execution_request)


_RESULT_KNOWN_FIELDS = {
    "t",
    "Y",
    "species_names",
    "algebra_scalars",
    "algebra_errors",
    "warnings",
    "solver",
    "nfev",
    "success",
    "message",
    "mechanism_text",
    "solver_config",
    "provenance",
    "fallback_occurred",
    "fallback_message",
    "base_species_count",
    "mechanism",
}


@dataclass(frozen=True)
class SimulationExecutionResult:
    """Typed wrapper for the existing worker/batch success payload shape."""

    t: Any
    Y: Any
    species_names: list[str]
    algebra_scalars: Dict[str, Any]
    algebra_errors: list[dict[str, Any]]
    warnings: list[dict[str, Any]]
    solver: str
    nfev: Any
    success: bool
    message: str
    mechanism_text: str
    solver_config: Dict[str, Any]
    provenance: Any
    fallback_occurred: bool
    fallback_message: Any
    base_species_count: Optional[int] = None
    mechanism: Any = None
    extra_fields: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "t", _copy_payload_value(self.t))
        object.__setattr__(self, "Y", _copy_payload_value(self.Y))
        object.__setattr__(self, "species_names", [str(name) for name in list(self.species_names or [])])
        object.__setattr__(self, "algebra_scalars", _copy_payload_value(dict(self.algebra_scalars or {})))
        object.__setattr__(self, "algebra_errors", _copy_payload_value(list(self.algebra_errors or [])))
        object.__setattr__(self, "warnings", _copy_payload_value(list(self.warnings or [])))
        object.__setattr__(self, "solver", str(self.solver or ""))
        object.__setattr__(self, "success", bool(self.success))
        object.__setattr__(self, "message", str(self.message or ""))
        object.__setattr__(self, "mechanism_text", str(self.mechanism_text or ""))
        object.__setattr__(self, "solver_config", _copy_payload_value(dict(self.solver_config or {})))
        object.__setattr__(self, "provenance", _copy_payload_value(self.provenance))
        object.__setattr__(self, "fallback_occurred", bool(self.fallback_occurred))
        if self.base_species_count is not None:
            object.__setattr__(self, "base_species_count", max(0, int(self.base_species_count)))
        object.__setattr__(self, "extra_fields", _copy_payload_value(dict(self.extra_fields or {})))

    @classmethod
    def from_success_payload(cls, payload: Mapping[str, Any]) -> "SimulationExecutionResult":
        if not isinstance(payload, Mapping):
            raise ValueError("SimulationExecutionResult payload must be a mapping.")
        missing = [
            key
            for key in (
                "t",
                "Y",
                "species_names",
                "algebra_scalars",
                "algebra_errors",
                "warnings",
                "solver",
                "nfev",
                "success",
                "message",
                "mechanism_text",
                "solver_config",
                "provenance",
                "fallback_occurred",
                "fallback_message",
            )
            if key not in payload
        ]
        if missing:
            raise ValueError(
                "SimulationExecutionResult success payload missing fields: "
                + ", ".join(sorted(missing))
            )
        extra_fields = {str(key): value for key, value in payload.items() if key not in _RESULT_KNOWN_FIELDS}
        return cls(
            t=payload["t"],
            Y=payload["Y"],
            species_names=list(payload.get("species_names") or []),
            algebra_scalars=dict(payload.get("algebra_scalars") or {}),
            algebra_errors=list(payload.get("algebra_errors") or []),
            warnings=list(payload.get("warnings") or []),
            solver=str(payload.get("solver") or ""),
            nfev=payload.get("nfev"),
            success=bool(payload.get("success")),
            message=str(payload.get("message") or ""),
            mechanism_text=str(payload.get("mechanism_text") or ""),
            solver_config=dict(payload.get("solver_config") or {}),
            provenance=payload.get("provenance"),
            fallback_occurred=bool(payload.get("fallback_occurred")),
            fallback_message=payload.get("fallback_message"),
            base_species_count=payload.get("base_species_count"),
            mechanism=payload.get("mechanism"),
            extra_fields=extra_fields,
        )

    def to_success_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "t": _copy_payload_value(self.t),
            "Y": _copy_payload_value(self.Y),
            "species_names": list(self.species_names),
            "algebra_scalars": _copy_payload_value(self.algebra_scalars),
            "algebra_errors": _copy_payload_value(self.algebra_errors),
            "warnings": _copy_payload_value(self.warnings),
            "solver": str(self.solver),
            "nfev": self.nfev,
            "success": bool(self.success),
            "message": str(self.message),
            "mechanism_text": str(self.mechanism_text),
            "solver_config": _copy_payload_value(self.solver_config),
            "provenance": _copy_payload_value(self.provenance),
            "fallback_occurred": bool(self.fallback_occurred),
            "fallback_message": self.fallback_message,
        }
        if self.base_species_count is not None:
            payload["base_species_count"] = int(self.base_species_count)
        if self.mechanism is not None:
            payload["mechanism"] = self.mechanism
        payload.update(_copy_payload_value(self.extra_fields))
        return payload
