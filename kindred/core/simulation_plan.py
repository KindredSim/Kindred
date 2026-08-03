from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Optional

import numpy as np

from kindred.core.intervention_schedule import (
    intervention_schedule_identity_fingerprints,
)
from kindred.core.simulation_preparation import (
    SimulationExecutionRequest,
    SimulationPreparationError,
    materialized_intervention_schedule_for_execution_request,
)

__all__ = [
    "SimulationAlgebraPolicy",
    "SimulationPlan",
]


class SimulationAlgebraPolicy(str, Enum):
    """Caller-level algebra handling policy carried by simulation plans."""

    GUI_BEST_EFFORT = "gui_best_effort"
    BATCH_BEST_EFFORT = "batch_best_effort"
    FITTING_STRICT = "fitting_strict"


def _copy_payload_value(value: Any, memo: Optional[Dict[int, Any]] = None) -> Any:
    if value is None or isinstance(value, (str, bytes, int, float, bool)):
        return value
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
    if isinstance(value, set):
        copied_set = {_copy_payload_value(item, memo) for item in value}
        memo[value_id] = copied_set
        return copied_set
    if isinstance(value, frozenset):
        copied_frozenset = frozenset(_copy_payload_value(item, memo) for item in value)
        memo[value_id] = copied_frozenset
        return copied_frozenset
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
    """Typed execution boundary carrying a nested preparation request."""

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
        object.__setattr__(
            self,
            "cache_identity_payload",
            self._cache_identity_payload_with_materialized_schedule_identity(),
        )
        self._validate_cache_identity_schedule()
        self._validate_parameter_override_cache_identity()

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

    def cache_key(self) -> str:
        payload = self.cache_identity_payload or {}
        return str(payload.get("cache_key") or "")

    def simulation_identity_payload(self) -> Dict[str, Any]:
        payload = self.cache_identity_payload or {}
        identity = payload.get("simulation_identity")
        if not isinstance(identity, Mapping):
            return {}
        return _copy_payload_value(identity)

    def preview_batch_cache_token(self) -> str:
        payload = self.cache_identity_payload or {}
        return str(payload.get("preview_batch_cache_token") or "")

    def scope_identity_payload(self) -> Dict[str, Any]:
        payload = self.cache_scope_payload or {}
        scope_identity = payload.get("scope_identity")
        if isinstance(scope_identity, Mapping):
            return _copy_payload_value(scope_identity)
        return _copy_payload_value(payload)

    def cache_queue_ids(self) -> tuple[str, ...]:
        payload = self.cache_scope_payload or {}
        queue_ids = payload.get("queue_ids")
        if not isinstance(queue_ids, (list, tuple)):
            return ()
        return tuple(str(item) for item in queue_ids if str(item))

    def _expected_schedule_identity(self) -> tuple[str, str]:
        schedule = materialized_intervention_schedule_for_execution_request(self.execution_request)
        if schedule is None:
            return "", ""
        return intervention_schedule_identity_fingerprints(schedule)

    def _cache_identity_payload_with_materialized_schedule_identity(self) -> Optional[Dict[str, Any]]:
        if not self.execution_request.has_intervention_schedule_authority:
            return self.cache_identity_payload
        payload = self.cache_identity_payload
        if not isinstance(payload, Mapping):
            return self.cache_identity_payload
        identity = payload.get("simulation_identity")
        if not isinstance(identity, Mapping):
            return self.cache_identity_payload
        if "intervention_schedule_fingerprint" in identity:
            return self.cache_identity_payload
        actual_declarative = str(identity.get("intervention_schedule_declarative_fingerprint") or "")
        actual_executable = str(identity.get("intervention_schedule_executable_fingerprint") or "")
        if actual_declarative or actual_executable:
            return self.cache_identity_payload
        expected_declarative, expected_executable = self._expected_schedule_identity()
        if not expected_declarative and not expected_executable:
            return self.cache_identity_payload
        updated_payload = _copy_payload_value(payload)
        updated_identity = dict(updated_payload.get("simulation_identity") or {})
        updated_identity["intervention_schedule_declarative_fingerprint"] = expected_declarative
        updated_identity["intervention_schedule_executable_fingerprint"] = expected_executable
        updated_payload["simulation_identity"] = updated_identity
        return updated_payload

    def _validate_cache_identity_schedule(self) -> None:
        identity = self.simulation_identity_payload()
        if not identity:
            return
        if "intervention_schedule_fingerprint" in identity:
            raise ValueError("stale intervention_schedule_fingerprint")
        actual_declarative = str(identity.get("intervention_schedule_declarative_fingerprint") or "")
        actual_executable = str(identity.get("intervention_schedule_executable_fingerprint") or "")
        if (actual_declarative or actual_executable) and not self.execution_request.has_intervention_schedule_authority:
            raise ValueError(
                "SimulationPlan cache_identity_payload simulation_identity "
                "intervention schedule fingerprints require execution_request intervention_schedule authority."
            )
        if not self.execution_request.has_intervention_schedule_authority:
            return
        expected_declarative, expected_executable = self._expected_schedule_identity()
        if actual_declarative != expected_declarative or actual_executable != expected_executable:
            raise ValueError(
                "SimulationPlan cache_identity_payload simulation_identity "
                "intervention schedule fingerprints conflict with execution_request."
            )

    def _validate_parameter_override_cache_identity(self) -> None:
        overrides = getattr(self.execution_request, "parameter_overrides", None)
        if not isinstance(overrides, Mapping) or not overrides:
            return
        request_payload = self.execution_request.to_payload()
        request_identity = request_payload.get("simulation_identity")
        cache_identity = self.simulation_identity_payload()
        if not isinstance(request_identity, Mapping) or not request_identity:
            raise ValueError(
                "SimulationPlan execution_request parameter_overrides require "
                "execution_request simulation_identity authority."
            )
        if not cache_identity:
            raise ValueError(
                "SimulationPlan execution_request parameter_overrides require "
                "cache_identity_payload simulation_identity authority."
            )
        request_fingerprint = str(request_identity.get("param_fingerprint") or "")
        cache_fingerprint = str(cache_identity.get("param_fingerprint") or "")
        if not request_fingerprint or not cache_fingerprint:
            raise ValueError(
                "SimulationPlan execution_request parameter_overrides require "
                "request and cache simulation identities with param_fingerprint."
            )
        if request_fingerprint != cache_fingerprint:
            raise ValueError(
                "SimulationPlan execution_request parameter_overrides conflict "
                "with cache_identity_payload simulation_identity param_fingerprint."
            )
        conflicts = [
            str(key)
            for key, value in dict(request_identity).items()
            if key in cache_identity and cache_identity.get(key) != value
        ]
        if conflicts:
            raise ValueError(
                "SimulationPlan execution_request parameter_overrides require "
                "execution_request and cache identity to share the same simulation_identity truth; "
                f"conflicting field(s): {', '.join(sorted(conflicts))}."
            )
