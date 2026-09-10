from __future__ import annotations

import math
from dataclasses import dataclass, replace
from numbers import Real
from types import MappingProxyType
from typing import Any, Mapping, Sequence, TypeAlias

from kindred.core.authored_source_errors import AuthoredSourceError
from kindred.core.species import (
    validate_concentration,
    validate_time_seconds,
)

__all__ = [
    "InterventionFlux",
    "InterventionHold",
    "InterventionJump",
    "InterventionProtocol",
    "InterventionProtocolOperation",
    "InterventionSchedule",
    "InterventionScheduleDependencyMetadata",
    "InterventionScheduleError",
    "InterventionScheduleFieldIdentity",
    "InterventionSet",
    "InterventionTrigger",
    "InterventionTriggerAction",
    "InterventionStopAction",
    "SpeciesTriggerCondition",
    "ObservableTriggerCondition",
    "coerce_intervention_schedule",
    "intervention_schedule_parameter_dependencies",
    "intervention_schedule_parameter_names",
]


# The declaration ordinal and (where applicable) the authored nested ordinal
# are part of the key.  Keeping the location in the key is important: two
# protocol records may use the same field name without overwriting one another.
InterventionScheduleFieldIdentity: TypeAlias = tuple[int, int | None, str]
InterventionScheduleDependencyMetadata: TypeAlias = Mapping[
    InterventionScheduleFieldIdentity, tuple[str, ...]
]


class InterventionScheduleError(AuthoredSourceError):
    """The authored intervention source is outside the final public schema."""

    __slots__ = (
        "operation",
        "declaration_ordinal",
        "field",
        "nested_location",
        "domain",
        "field_identities",
        "value_dependencies",
        "details",
        "_locked",
    )

    def __init__(
        self,
        message: str,
        *,
        operation: str | None = None,
        declaration_ordinal: int | None = None,
        field: str | None = None,
        nested_location: Mapping[str, Any] | None = None,
        domain: str | None = None,
        field_identities: Sequence[InterventionScheduleFieldIdentity] = (),
        value_dependencies: Sequence[str] = (),
        details: Mapping[str, Any] | None = None,
    ) -> None:
        """Retain producer-owned location/dependency facts immutably."""
        normalized_dependencies = tuple(
            sorted(
                {
                    str(name).strip()
                    for name in value_dependencies
                    if str(name).strip()
                }
            )
        )
        normalized_field_identities = tuple(
            (int(declaration), None if nested is None else int(nested), str(path))
            for declaration, nested, path in field_identities
        )
        payload = dict(details or {})
        if operation is not None:
            payload["operation"] = str(operation)
        if declaration_ordinal is not None:
            payload["declaration_ordinal"] = int(declaration_ordinal)
        if field is not None:
            payload["field"] = str(field)
        if nested_location is not None:
            payload["nested_location"] = nested_location
        if domain is not None:
            payload["domain"] = str(domain)
        if normalized_field_identities:
            payload["field_identities"] = normalized_field_identities
        if normalized_dependencies:
            payload["value_dependencies"] = normalized_dependencies
        object.__setattr__(self, "operation", None if operation is None else str(operation))
        object.__setattr__(
            self,
            "declaration_ordinal",
            None if declaration_ordinal is None else int(declaration_ordinal),
        )
        object.__setattr__(self, "field", None if field is None else str(field))
        object.__setattr__(self, "nested_location", _freeze_schedule_value(nested_location))
        object.__setattr__(self, "domain", None if domain is None else str(domain))
        object.__setattr__(self, "field_identities", normalized_field_identities)
        object.__setattr__(self, "value_dependencies", normalized_dependencies)
        object.__setattr__(
            self,
            "details",
            MappingProxyType(
                {
                    str(key): _freeze_schedule_value(value)
                    for key, value in payload.items()
                }
            ),
        )
        super().__init__(str(message))
        object.__setattr__(self, "_locked", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_locked", False):
            raise AttributeError("InterventionScheduleError is immutable")
        object.__setattr__(self, name, value)


def _freeze_schedule_value(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_schedule_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_schedule_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(
            _freeze_schedule_value(item) for item in sorted(value, key=repr)
        )
    return value


_PARAM_FIELDS = frozenset(
    {
        "time_param",
        "start_time_param",
        "end_time_param",
        "value_param",
        "delta_param",
        "flux_rate_param",
        "threshold_param",
    }
)
_DIRECTIONS = frozenset({"rising", "falling", "either"})
_PHASES = frozenset({"before", "during", "after"})
_AUTHORED_FLOAT_FIELDS = frozenset(
    {
        "time",
        "start_time",
        "end_time",
        "value",
        "delta",
        "flux_rate",
        "threshold",
        "every",
        "duration",
        "min_interval",
    }
)
_AUTHORED_INTEGER_FIELDS = frozenset({"count", "max_count"})


def _finite(value: object, *, field: str) -> float:
    if type(value) is bool or not isinstance(value, Real):
        raise InterventionScheduleError(f"{field} must be a finite number.")
    number = float(value)
    if not math.isfinite(number):
        raise InterventionScheduleError(f"{field} must be a finite number.")
    return number


def _nonnegative(value: object, *, field: str) -> float:
    try:
        return validate_time_seconds(value, field=field)
    except (TypeError, ValueError) as exc:
        raise InterventionScheduleError(str(exc)) from exc


def _positive(value: object, *, field: str) -> float:
    number = _nonnegative(value, field=field)
    if number <= 0.0:
        raise InterventionScheduleError(f"{field} must be greater than zero.")
    return number


def _concentration(value: object, *, field: str) -> float:
    try:
        return validate_concentration(value, field=field)
    except (TypeError, ValueError) as exc:
        raise InterventionScheduleError(str(exc)) from exc


def _positive_int(value: object, *, field: str) -> int:
    if type(value) is not int:
        raise InterventionScheduleError(f"{field} must be a positive integer.")
    number = value
    if number <= 0:
        raise InterventionScheduleError(f"{field} must be a positive integer.")
    return number


def _name(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise InterventionScheduleError(f"{field} must be a string.")
    name = value.strip()
    if not name:
        raise InterventionScheduleError(f"{field} must not be empty.")
    return name


def _enum(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise InterventionScheduleError(f"{field} must be a string.")
    text = value.strip().lower()
    if not text:
        raise InterventionScheduleError(f"{field} must not be empty.")
    return text


def _ordinal(value: object, *, field: str) -> int:
    if type(value) is not int or value < 0:
        raise InterventionScheduleError(f"{field} must be a non-negative integer.")
    return value


def _pair(
    value: object,
    parameter: object,
    *,
    field: str,
    domain: str,
) -> tuple[float | None, str | None]:
    has_value = value is not None
    has_parameter = parameter is not None
    if has_value == has_parameter:
        raise InterventionScheduleError(
            f"Exactly one of {field} and {field}_param is required."
        )
    if has_parameter:
        return None, _name(parameter, field=f"{field}_param")
    validator = {
        "signed": _finite,
        "nonnegative": _nonnegative,
        "positive": _positive,
        "concentration": _concentration,
    }[domain]
    return validator(value, field=field), None


def _resolved_payload(value, parameter, values, *, field, domain, runtime_parameters):
    if parameter in runtime_parameters:
        return {field: None, f"{field}_param": parameter}
    return {
        field: _resolved(value, parameter, values, field=field, domain=domain),
        f"{field}_param": None,
    }


def _resolved(
    value: float | None,
    parameter: str | None,
    parameter_values: Mapping[str, Any],
    *,
    field: str,
    domain: str,
) -> float:
    raw: object
    if parameter is not None:
        if parameter not in parameter_values:
            raise InterventionScheduleError(
                f"Missing intervention schedule parameter: {parameter}",
                field=field,
                domain=domain,
                value_dependencies=(parameter,),
            )
        raw = parameter_values[parameter]
    else:
        raw = value
    validator = {
        "signed": _finite,
        "nonnegative": _nonnegative,
        "positive": _positive,
        "concentration": _concentration,
    }[domain]
    try:
        return validator(raw, field=field)
    except InterventionScheduleError as exc:
        if parameter is None:
            raise
        raise InterventionScheduleError(
            str(exc),
            field=field,
            domain=domain,
            value_dependencies=(parameter,),
        ) from exc


def _put_pair(
    payload: dict[str, Any],
    *,
    field: str,
    value: float | None,
    parameter: str | None,
) -> None:
    if parameter is None:
        payload[field] = float(value if value is not None else 0.0)
    else:
        payload[f"{field}_param"] = parameter


def _reject_keys(
    payload: Mapping[str, Any],
    allowed: frozenset[str],
    *,
    context: str,
) -> None:
    unknown = [str(key) for key in payload if str(key) not in allowed]
    if unknown:
        raise InterventionScheduleError(
            f"Unsupported {context} field: {unknown[0]!r}."
        )


@dataclass(frozen=True, slots=True)
class InterventionSet:
    species: str
    time: float | None = None
    value: float | None = None
    time_param: str | None = None
    value_param: str | None = None
    declaration_ordinal: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "species", _name(self.species, field="species"))
        object.__setattr__(
            self,
            "declaration_ordinal",
            _ordinal(self.declaration_ordinal, field="declaration_ordinal"),
        )
        time, time_param = _pair(
            self.time, self.time_param, field="time", domain="nonnegative"
        )
        value, value_param = _pair(
            self.value, self.value_param, field="value", domain="concentration"
        )
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "time_param", time_param)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "value_param", value_param)

    @property
    def op(self) -> str:
        return "set"

    @property
    def is_parameterized(self) -> bool:
        return self.time_param is not None or self.value_param is not None

    def resolve_parameters(self, values: Mapping[str, Any], *, runtime_parameters: frozenset[str] = frozenset()) -> "InterventionSet":
        return InterventionSet(
            species=self.species,
            time=_resolved(
                self.time, self.time_param, values, field="time", domain="nonnegative"
            ),
            **_resolved_payload(
                self.value, self.value_param, values, field="value", domain="concentration"
            , runtime_parameters=runtime_parameters),
            declaration_ordinal=self.declaration_ordinal,
        )

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "op": "set",
            "species": self.species,
            "declaration_ordinal": self.declaration_ordinal,
        }
        _put_pair(payload, field="time", value=self.time, parameter=self.time_param)
        _put_pair(payload, field="value", value=self.value, parameter=self.value_param)
        return payload


@dataclass(frozen=True, slots=True)
class InterventionJump:
    species: str
    time: float | None = None
    delta: float | None = None
    time_param: str | None = None
    delta_param: str | None = None
    declaration_ordinal: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "species", _name(self.species, field="species"))
        object.__setattr__(
            self,
            "declaration_ordinal",
            _ordinal(self.declaration_ordinal, field="declaration_ordinal"),
        )
        time, time_param = _pair(
            self.time, self.time_param, field="time", domain="nonnegative"
        )
        delta, delta_param = _pair(
            self.delta, self.delta_param, field="delta", domain="signed"
        )
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "time_param", time_param)
        object.__setattr__(self, "delta", delta)
        object.__setattr__(self, "delta_param", delta_param)

    @property
    def op(self) -> str:
        return "jump"

    @property
    def is_parameterized(self) -> bool:
        return self.time_param is not None or self.delta_param is not None

    def resolve_parameters(self, values: Mapping[str, Any], *, runtime_parameters: frozenset[str] = frozenset()) -> "InterventionJump":
        return InterventionJump(
            species=self.species,
            time=_resolved(
                self.time, self.time_param, values, field="time", domain="nonnegative"
            ),
            **_resolved_payload(
                self.delta, self.delta_param, values, field="delta", domain="signed"
            , runtime_parameters=runtime_parameters),
            declaration_ordinal=self.declaration_ordinal,
        )

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "op": "jump",
            "species": self.species,
            "declaration_ordinal": self.declaration_ordinal,
        }
        _put_pair(payload, field="time", value=self.time, parameter=self.time_param)
        _put_pair(payload, field="delta", value=self.delta, parameter=self.delta_param)
        return payload


@dataclass(frozen=True, slots=True)
class InterventionFlux:
    species: str
    start_time: float | None = None
    end_time: float | None = None
    flux_rate: float | None = None
    start_time_param: str | None = None
    end_time_param: str | None = None
    flux_rate_param: str | None = None
    declaration_ordinal: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "species", _name(self.species, field="species"))
        object.__setattr__(
            self,
            "declaration_ordinal",
            _ordinal(self.declaration_ordinal, field="declaration_ordinal"),
        )
        for field, domain in (
            ("start_time", "nonnegative"),
            ("end_time", "nonnegative"),
            ("flux_rate", "signed"),
        ):
            value, parameter = _pair(
                getattr(self, field),
                getattr(self, f"{field}_param"),
                field=field,
                domain=domain,
            )
            object.__setattr__(self, field, value)
            object.__setattr__(self, f"{field}_param", parameter)
        if (
            self.start_time is not None
            and self.end_time is not None
            and self.end_time <= self.start_time
        ):
            raise InterventionScheduleError(
                "end_time must be greater than start_time."
            )

    @property
    def op(self) -> str:
        return "flux"

    @property
    def is_parameterized(self) -> bool:
        return any(
            value is not None
            for value in (
                self.start_time_param,
                self.end_time_param,
                self.flux_rate_param,
            )
        )

    def resolve_parameters(self, values: Mapping[str, Any], *, runtime_parameters: frozenset[str] = frozenset()) -> "InterventionFlux":
        start_time = _resolved(
            self.start_time,
            self.start_time_param,
            values,
            field="start_time",
            domain="nonnegative",
        )
        end_time = _resolved(
            self.end_time,
            self.end_time_param,
            values,
            field="end_time",
            domain="nonnegative",
        )
        if end_time <= start_time:
            raise InterventionScheduleError(
                "end_time must be greater than start_time.",
                field="end_time",
                domain="interval_order",
                field_identities=(
                    (self.declaration_ordinal, None, "start_time"),
                    (self.declaration_ordinal, None, "end_time"),
                ),
                value_dependencies=tuple(
                    name
                    for name in (self.start_time_param, self.end_time_param)
                    if name is not None
                ),
            )
        return InterventionFlux(
            species=self.species,
            start_time=start_time,
            end_time=end_time,
            **_resolved_payload(
                self.flux_rate,
                self.flux_rate_param,
                values,
                field="flux_rate",
                domain="signed",
            runtime_parameters=runtime_parameters),
            declaration_ordinal=self.declaration_ordinal,
        )

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "op": "flux",
            "species": self.species,
            "declaration_ordinal": self.declaration_ordinal,
        }
        for field in ("start_time", "end_time", "flux_rate"):
            _put_pair(
                payload,
                field=field,
                value=getattr(self, field),
                parameter=getattr(self, f"{field}_param"),
            )
        return payload


@dataclass(frozen=True, slots=True)
class InterventionHold:
    species: str
    start_time: float | None = None
    end_time: float | None = None
    value: float | None = None
    start_time_param: str | None = None
    end_time_param: str | None = None
    value_param: str | None = None
    declaration_ordinal: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "species", _name(self.species, field="species"))
        object.__setattr__(
            self,
            "declaration_ordinal",
            _ordinal(self.declaration_ordinal, field="declaration_ordinal"),
        )
        for field in ("start_time", "end_time", "value"):
            value, parameter = _pair(
                getattr(self, field),
                getattr(self, f"{field}_param"),
                field=field,
                domain="concentration" if field == "value" else "nonnegative",
            )
            object.__setattr__(self, field, value)
            object.__setattr__(self, f"{field}_param", parameter)
        if (
            self.start_time is not None
            and self.end_time is not None
            and self.end_time <= self.start_time
        ):
            raise InterventionScheduleError(
                "end_time must be greater than start_time."
            )

    @property
    def op(self) -> str:
        return "hold"

    @property
    def is_parameterized(self) -> bool:
        return any(
            value is not None
            for value in (
                self.start_time_param,
                self.end_time_param,
                self.value_param,
            )
        )

    def resolve_parameters(self, values: Mapping[str, Any], *, runtime_parameters: frozenset[str] = frozenset()) -> "InterventionHold":
        start_time = _resolved(
            self.start_time,
            self.start_time_param,
            values,
            field="start_time",
            domain="nonnegative",
        )
        end_time = _resolved(
            self.end_time,
            self.end_time_param,
            values,
            field="end_time",
            domain="nonnegative",
        )
        if end_time <= start_time:
            raise InterventionScheduleError(
                "end_time must be greater than start_time.",
                field="end_time",
                domain="interval_order",
                field_identities=(
                    (self.declaration_ordinal, None, "start_time"),
                    (self.declaration_ordinal, None, "end_time"),
                ),
                value_dependencies=tuple(
                    name
                    for name in (self.start_time_param, self.end_time_param)
                    if name is not None
                ),
            )
        return InterventionHold(
            species=self.species,
            start_time=start_time,
            end_time=end_time,
            **_resolved_payload(
                self.value,
                self.value_param,
                values,
                field="value",
                domain="concentration",
            runtime_parameters=runtime_parameters),
            declaration_ordinal=self.declaration_ordinal,
        )

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "op": "hold",
            "species": self.species,
            "declaration_ordinal": self.declaration_ordinal,
        }
        for field in ("start_time", "end_time", "value"):
            _put_pair(
                payload,
                field=field,
                value=getattr(self, field),
                parameter=getattr(self, f"{field}_param"),
            )
        return payload


@dataclass(frozen=True, slots=True)
class InterventionProtocolOperation:
    phase: str
    op: str
    species: str
    value: float | None = None
    delta: float | None = None
    flux_rate: float | None = None
    value_param: str | None = None
    delta_param: str | None = None
    flux_rate_param: str | None = None
    authored_ordinal: int = 0

    def __post_init__(self) -> None:
        phase = _enum(self.phase, field="phase")
        op = _enum(self.op, field="op")
        if phase not in _PHASES:
            raise InterventionScheduleError(f"Unsupported protocol phase: {phase!r}.")
        permitted = {"before": {"set", "jump"}, "during": {"flux", "hold"}, "after": {"set", "jump"}}
        if op not in permitted[phase]:
            raise InterventionScheduleError(
                f"Protocol phase {phase!r} does not permit op={op!r}."
            )
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "op", op)
        object.__setattr__(self, "species", _name(self.species, field="species"))
        object.__setattr__(
            self,
            "authored_ordinal",
            _ordinal(self.authored_ordinal, field="authored_ordinal"),
        )
        selected = {"set": "value", "hold": "value", "jump": "delta", "flux": "flux_rate"}[op]
        for field in ("value", "delta", "flux_rate"):
            raw_value = getattr(self, field)
            raw_parameter = getattr(self, f"{field}_param")
            if field == selected:
                value, parameter = _pair(
                    raw_value,
                    raw_parameter,
                    field=field,
                    domain="concentration" if field == "value" else "signed",
                )
            else:
                if raw_value is not None or raw_parameter is not None:
                    raise InterventionScheduleError(
                        f"Protocol op={op!r} does not permit {field}."
                    )
                value, parameter = None, None
            object.__setattr__(self, field, value)
            object.__setattr__(self, f"{field}_param", parameter)

    @property
    def is_parameterized(self) -> bool:
        return any(
            item is not None
            for item in (self.value_param, self.delta_param, self.flux_rate_param)
        )

    def resolve_parameters(
        self, values: Mapping[str, Any], *, runtime_parameters: frozenset[str] = frozenset()
    ) -> "InterventionProtocolOperation":
        selected = {"set": "value", "hold": "value", "jump": "delta", "flux": "flux_rate"}[self.op]
        kwargs: dict[str, Any] = {
            "phase": self.phase,
            "op": self.op,
            "species": self.species,
            "authored_ordinal": self.authored_ordinal,
        }
        kwargs.update(_resolved_payload(getattr(self, selected), getattr(self, f"{selected}_param"), values,
            field=selected, domain="concentration" if selected == "value" else "signed", runtime_parameters=runtime_parameters))
        return InterventionProtocolOperation(**kwargs)

    def to_payload(self) -> dict[str, Any]:
        selected = {"set": "value", "hold": "value", "jump": "delta", "flux": "flux_rate"}[self.op]
        payload: dict[str, Any] = {
            "op": self.op,
            "species": self.species,
            "authored_ordinal": self.authored_ordinal,
        }
        _put_pair(
            payload,
            field=selected,
            value=getattr(self, selected),
            parameter=getattr(self, f"{selected}_param"),
        )
        return payload


@dataclass(frozen=True, slots=True)
class InterventionProtocol:
    start_time: float | None
    every: float
    count: int
    before: tuple[InterventionProtocolOperation, ...] = ()
    during: tuple[InterventionProtocolOperation, ...] = ()
    after: tuple[InterventionProtocolOperation, ...] = ()
    duration: float | None = None
    start_time_param: str | None = None
    declaration_ordinal: int = 0

    def __post_init__(self) -> None:
        start_time, start_time_param = _pair(
            self.start_time,
            self.start_time_param,
            field="start_time",
            domain="nonnegative",
        )
        object.__setattr__(self, "start_time", start_time)
        object.__setattr__(self, "start_time_param", start_time_param)
        object.__setattr__(self, "every", _positive(self.every, field="every"))
        object.__setattr__(self, "count", _positive_int(self.count, field="count"))
        object.__setattr__(
            self,
            "declaration_ordinal",
            _ordinal(self.declaration_ordinal, field="declaration_ordinal"),
        )
        for phase in ("before", "during", "after"):
            operations = tuple(getattr(self, phase))
            if any(item.phase != phase for item in operations):
                raise InterventionScheduleError(
                    f"Protocol {phase} contains an operation from another phase."
                )
            authored_ordinals = [int(item.authored_ordinal) for item in operations]
            if len(authored_ordinals) != len(set(authored_ordinals)):
                raise InterventionScheduleError(
                    f"Protocol {phase} nested authored ordinals must be unique."
                )
            object.__setattr__(self, phase, operations)
        if not self.before and not self.during and not self.after:
            raise InterventionScheduleError(
                "Protocol requires at least one nonempty phase."
            )
        needs_duration = bool(self.during or self.after)
        if needs_duration:
            object.__setattr__(
                self, "duration", _positive(self.duration, field="duration")
            )
        elif self.duration is not None:
            raise InterventionScheduleError(
                "duration is forbidden for a before-only protocol."
            )

    @property
    def op(self) -> str:
        return "protocol"

    @property
    def kind(self) -> str:
        return "repeat"

    @property
    def is_parameterized(self) -> bool:
        return self.start_time_param is not None or any(
            operation.is_parameterized
            for operation in self.before + self.during + self.after
        )

    def resolve_parameters(self, values: Mapping[str, Any], *, runtime_parameters: frozenset[str] = frozenset()) -> "InterventionProtocol":
        resolved_phases: dict[str, tuple[InterventionProtocolOperation, ...]] = {}
        for phase in ("before", "during", "after"):
            resolved_items: list[InterventionProtocolOperation] = []
            for nested in getattr(self, phase):
                try:
                    resolved_items.append(nested.resolve_parameters(values, runtime_parameters=runtime_parameters))
                except InterventionScheduleError as exc:
                    details = dict(getattr(exc, "details", {}) or {})
                    details["nested_location"] = {
                        "phase": phase,
                        "authored_ordinal": nested.authored_ordinal,
                    }
                    dependencies = tuple(
                        getattr(exc, "value_dependencies", ())
                        or details.get("value_dependencies", ())
                    )
                    raise InterventionScheduleError(
                        str(exc),
                        operation=self.op,
                        declaration_ordinal=self.declaration_ordinal,
                        field=getattr(exc, "field", None),
                        nested_location=details["nested_location"],
                        domain=getattr(exc, "domain", None),
                        field_identities=getattr(exc, "field_identities", ()),
                        value_dependencies=dependencies,
                        details=details,
                    ) from exc
            resolved_phases[phase] = tuple(resolved_items)
        return InterventionProtocol(
            start_time=_resolved(
                self.start_time,
                self.start_time_param,
                values,
                field="start_time",
                domain="nonnegative",
            ),
            every=self.every,
            count=self.count,
            duration=self.duration,
            before=resolved_phases["before"],
            during=resolved_phases["during"],
            after=resolved_phases["after"],
            declaration_ordinal=self.declaration_ordinal,
        )

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "op": "protocol",
            "kind": "repeat",
            "every": self.every,
            "count": self.count,
            "declaration_ordinal": self.declaration_ordinal,
        }
        _put_pair(
            payload,
            field="start_time",
            value=self.start_time,
            parameter=self.start_time_param,
        )
        if self.duration is not None:
            payload["duration"] = self.duration
        for phase in ("before", "during", "after"):
            operations = getattr(self, phase)
            if operations:
                payload[phase] = [item.to_payload() for item in operations]
        return payload


@dataclass(frozen=True, slots=True)
class InterventionTriggerAction:
    op: str
    species: str
    value: float | None = None
    delta: float | None = None
    value_param: str | None = None
    delta_param: str | None = None

    def __post_init__(self) -> None:
        op = _enum(self.op, field="op")
        if op not in {"set", "jump"}:
            raise InterventionScheduleError(
                "Trigger then permits exactly op=set or op=jump."
            )
        object.__setattr__(self, "op", op)
        object.__setattr__(self, "species", _name(self.species, field="species"))
        selected = "value" if op == "set" else "delta"
        rejected = "delta" if op == "set" else "value"
        if getattr(self, rejected) is not None or getattr(self, f"{rejected}_param") is not None:
            raise InterventionScheduleError(
                f"Trigger op={op!r} does not permit {rejected}."
            )
        value, parameter = _pair(
            getattr(self, selected),
            getattr(self, f"{selected}_param"),
            field=selected,
            domain="concentration" if selected == "value" else "signed",
        )
        object.__setattr__(self, selected, value)
        object.__setattr__(self, f"{selected}_param", parameter)

    @property
    def is_parameterized(self) -> bool:
        return self.value_param is not None or self.delta_param is not None

    def resolve_parameters(
        self, values: Mapping[str, Any], *, runtime_parameters: frozenset[str] = frozenset()
    ) -> "InterventionTriggerAction":
        selected = "value" if self.op == "set" else "delta"
        return InterventionTriggerAction(
            op=self.op,
            species=self.species,
            **_resolved_payload(getattr(self, selected), getattr(self, f"{selected}_param"), values,
                field=selected, domain="concentration" if selected == "value" else "signed", runtime_parameters=runtime_parameters),
        )

    def to_payload(self) -> dict[str, Any]:
        selected = "value" if self.op == "set" else "delta"
        payload: dict[str, Any] = {"op": self.op, "species": self.species}
        _put_pair(
            payload,
            field=selected,
            value=getattr(self, selected),
            parameter=getattr(self, f"{selected}_param"),
        )
        return payload


@dataclass(frozen=True, slots=True)
class SpeciesTriggerCondition:
    name: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _name(self.name, field="watch_species"))

    def to_payload(self) -> dict[str, str]:
        return {"watch_species": self.name}


@dataclass(frozen=True, slots=True)
class ObservableTriggerCondition:
    name: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _name(self.name, field="watch_observable"))

    def to_payload(self) -> dict[str, str]:
        return {"watch_observable": self.name}


@dataclass(frozen=True, slots=True)
class InterventionStopAction:
    @property
    def op(self) -> str:
        return "stop"

    @property
    def is_parameterized(self) -> bool:
        return False

    def resolve_parameters(self, values: Mapping[str, Any], *, runtime_parameters: frozenset[str] = frozenset()) -> InterventionStopAction:
        return self

    def to_payload(self) -> dict[str, str]:
        return {"op": "stop"}


@dataclass(frozen=True, slots=True)
class InterventionTrigger:
    condition: SpeciesTriggerCondition | ObservableTriggerCondition
    then: InterventionTriggerAction | InterventionStopAction
    threshold: float | None = None
    threshold_param: str | None = None
    direction: str = "either"
    max_count: int | None = None
    min_interval: float | None = None
    declaration_ordinal: int = 0
    source_text: str | None = None

    def __post_init__(self) -> None:
        threshold, threshold_param = _pair(
            self.threshold, self.threshold_param, field="threshold",
            domain="concentration" if isinstance(self.condition, SpeciesTriggerCondition) else "signed",
        )
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(self, "threshold_param", threshold_param)
        direction = _enum(self.direction, field="direction")
        if direction not in _DIRECTIONS:
            raise InterventionScheduleError("direction must be rising, falling, or either.")
        object.__setattr__(self, "direction", direction)
        if isinstance(self.then, InterventionStopAction):
            if self.max_count is not None or self.min_interval is not None:
                raise InterventionScheduleError("Stopping triggers do not permit max_count or min_interval.")
        else:
            if not isinstance(self.condition, SpeciesTriggerCondition):
                raise InterventionScheduleError("Chemical triggers require watch_species.")
            object.__setattr__(self, "max_count", _positive_int(self.max_count, field="max_count"))
            object.__setattr__(self, "min_interval", _nonnegative(self.min_interval, field="min_interval"))
        object.__setattr__(self, "declaration_ordinal", _ordinal(self.declaration_ordinal, field="declaration_ordinal"))

    @property
    def op(self) -> str:
        return "trigger"

    @property
    def is_parameterized(self) -> bool:
        return self.threshold_param is not None or self.then.is_parameterized

    def resolve_parameters(self, values: Mapping[str, Any], *, runtime_parameters: frozenset[str] = frozenset()) -> InterventionTrigger:
        try:
            then = self.then.resolve_parameters(values, runtime_parameters=runtime_parameters)
        except InterventionScheduleError as exc:
            raise InterventionScheduleError(
                str(exc), operation=self.op, declaration_ordinal=self.declaration_ordinal,
                field=exc.field, nested_location={"path": "then"}, domain=exc.domain,
                field_identities=exc.field_identities, value_dependencies=exc.value_dependencies,
                details=exc.details,
            ) from exc
        return replace(
            self, then=then,
            **_resolved_payload(
                self.threshold, self.threshold_param, values, field="threshold",
                domain="concentration" if isinstance(self.condition, SpeciesTriggerCondition) else "signed",
            runtime_parameters=runtime_parameters),
        )

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "op": "trigger", **self.condition.to_payload(), "direction": self.direction,
            "then": self.then.to_payload(), "declaration_ordinal": self.declaration_ordinal,
        }
        if not isinstance(self.then, InterventionStopAction):
            payload.update(max_count=self.max_count, min_interval=self.min_interval)
        _put_pair(payload, field="threshold", value=self.threshold, parameter=self.threshold_param)
        return payload


InterventionOperation: TypeAlias = (
    InterventionSet
    | InterventionJump
    | InterventionFlux
    | InterventionHold
    | InterventionProtocol
    | InterventionTrigger
)


@dataclass(frozen=True, slots=True)
class InterventionSchedule:
    operations: tuple[InterventionOperation, ...] = ()

    def __post_init__(self) -> None:
        operations = tuple(self.operations)
        allowed = (
            InterventionSet,
            InterventionJump,
            InterventionFlux,
            InterventionHold,
            InterventionProtocol,
            InterventionTrigger,
        )
        if any(not isinstance(item, allowed) for item in operations):
            raise InterventionScheduleError(
                "operations contains an unsupported intervention type."
            )
        ordinals = [item.declaration_ordinal for item in operations]
        if len(ordinals) != len(set(ordinals)):
            raise InterventionScheduleError(
                "Intervention declaration ordinals must be unique."
            )
        if ordinals != sorted(ordinals):
            raise InterventionScheduleError(
                "Intervention operations must remain in authored ordinal order."
            )
        object.__setattr__(self, "operations", operations)

    def without_stops(self) -> InterventionSchedule:
        return InterventionSchedule(tuple(
            item for item in self.operations
            if not (isinstance(item, InterventionTrigger) and isinstance(item.then, InterventionStopAction))
        ))

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any] | None
    ) -> "InterventionSchedule":
        if payload is None:
            return cls()
        _reject_keys(payload, frozenset({"operations"}), context="schedule")
        raw_operations = payload.get("operations")
        if raw_operations is None:
            raw_operations = ()
        if not isinstance(raw_operations, Sequence) or isinstance(
            raw_operations, (str, bytes)
        ):
            raise InterventionScheduleError("operations must be a sequence.")
        operations: list[InterventionOperation] = []
        for index, raw in enumerate(raw_operations):
            if not isinstance(raw, Mapping):
                raise InterventionScheduleError("operations entries must be mappings.")
            data = dict(raw)
            ordinal = data.pop("declaration_ordinal", index)
            operations.append(
                _operation_from_fields(data, declaration_ordinal=ordinal, parsed=False)
            )
        return cls(tuple(operations))

    def is_empty(self) -> bool:
        return not self.operations

    @property
    def is_parameterized(self) -> bool:
        return any(item.is_parameterized for item in self.operations)

    def resolve_parameters(
        self,
        parameter_values: Mapping[str, Any] | None,
        *,
        dependency_metadata: InterventionScheduleDependencyMetadata | None = None,
        runtime_parameters: frozenset[str] = frozenset(),
    ) -> "InterventionSchedule":
        if not self.is_parameterized:
            return self
        values = dict(parameter_values or {})
        dependency_metadata = dependency_metadata or intervention_schedule_parameter_dependencies(self)
        resolved: list[InterventionOperation] = []
        for item in self.operations:
            try:
                resolved.append(item.resolve_parameters(values, runtime_parameters=runtime_parameters))
            except InterventionScheduleError as exc:
                raise _annotate_schedule_resolution_error(
                    exc,
                    operation=item,
                    dependency_metadata=dependency_metadata,
                ) from exc
        return InterventionSchedule(tuple(resolved))

    def to_payload(self) -> dict[str, Any]:
        return {"operations": [item.to_payload() for item in self.operations]}

    def validate_species(self, species_names: Sequence[str]) -> None:
        available = {str(name) for name in species_names}
        for operation in self.operations:
            names: tuple[str, ...]
            if isinstance(operation, InterventionProtocol):
                names = tuple(
                    item.species
                    for item in operation.before + operation.during + operation.after
                )
            elif isinstance(operation, InterventionTrigger):
                names = (() if isinstance(operation.condition, ObservableTriggerCondition) else (operation.condition.name,))
                if isinstance(operation.then, InterventionTriggerAction):
                    names += (operation.then.species,)
            else:
                names = (operation.species,)
            for name in names:
                if name not in available:
                    raise InterventionScheduleError(
                        f"Unknown intervention species: {name}"
                    )

    @property
    def instants(self) -> tuple[InterventionSet | InterventionJump, ...]:
        return tuple(
            item
            for item in self.operations
            if isinstance(item, (InterventionSet, InterventionJump))
        )

    @property
    def fluxes(self) -> tuple[InterventionFlux, ...]:
        return tuple(item for item in self.operations if isinstance(item, InterventionFlux))

    @property
    def holds(self) -> tuple[InterventionHold, ...]:
        return tuple(item for item in self.operations if isinstance(item, InterventionHold))

    @property
    def triggers(self) -> tuple[InterventionTrigger, ...]:
        return tuple(item for item in self.operations if isinstance(item, InterventionTrigger))


def _schedule_field_identities(
    operation: object,
    fields: Sequence[str],
    *,
    nested_location: Mapping[str, Any] | None = None,
) -> tuple[InterventionScheduleFieldIdentity, ...]:
    declaration = int(getattr(operation, "declaration_ordinal", -1))
    location = nested_location
    if location is None:
        nested_ordinal = getattr(operation, "authored_ordinal", None)
        phase = str(getattr(operation, "phase", ""))
        if nested_ordinal is not None and phase:
            location = {"phase": phase, "authored_ordinal": int(nested_ordinal)}
    nested_ordinal = None if location is None else location.get("authored_ordinal")
    phase = "" if location is None else str(location.get("phase", ""))
    path_prefix = ""
    if nested_ordinal is not None and phase:
        path_prefix = phase + "."
    elif location is not None and location.get("path"):
        path_prefix = str(location["path"]).rstrip(".") + "."
    return tuple(
        (
            declaration,
            None if nested_ordinal is None else int(nested_ordinal),
            path_prefix + str(field),
        )
        for field in fields
    )


def _annotate_schedule_resolution_error(
    exc: InterventionScheduleError,
    *,
    operation: object,
    dependency_metadata: InterventionScheduleDependencyMetadata,
) -> InterventionScheduleError:
    details = dict(getattr(exc, "details", {}) or {})
    nested_location = getattr(exc, "nested_location", None)
    if nested_location is None:
        raw_location = details.get("nested_location")
        if isinstance(raw_location, Mapping):
            nested_location = raw_location
    existing_identities = tuple(getattr(exc, "field_identities", ()) or ())
    fields = (getattr(exc, "field", None),) if getattr(exc, "field", None) else ()
    identities = existing_identities or _schedule_field_identities(
        operation,
        tuple(str(field) for field in fields),
        nested_location=nested_location,
    )
    dependencies = {
        str(name).strip()
        for name in getattr(exc, "value_dependencies", ())
        if str(name).strip()
    }
    metadata_dependencies = {
        str(name).strip()
        for identity in identities
        for name in dependency_metadata.get(identity, ())
        if str(name).strip()
    }
    if metadata_dependencies:
        dependencies = metadata_dependencies
    if identities:
        details["field_identities"] = identities
    if dependencies:
        details["value_dependencies"] = tuple(sorted(dependencies))
    return InterventionScheduleError(
        str(exc),
        operation=str(getattr(operation, "op", type(operation).__name__)),
        declaration_ordinal=int(getattr(operation, "declaration_ordinal", -1)),
        field=getattr(exc, "field", None),
        nested_location=nested_location,
        domain=getattr(exc, "domain", None),
        field_identities=identities,
        value_dependencies=tuple(sorted(dependencies)),
        details=details,
    )


def _literal_fields(fields: Mapping[str, Any], *, context: str) -> None:
    for key in fields:
        if str(key).endswith("_param") and str(key) not in _PARAM_FIELDS:
            raise InterventionScheduleError(
                f"Unsupported {context} parameter field: {key!r}."
            )


def _operation_from_fields(
    fields: Mapping[str, Any],
    *,
    declaration_ordinal: object,
    parsed: bool,
) -> InterventionOperation:
    data = dict(fields)
    _literal_fields(data, context="intervention")
    op = _enum(data.get("op"), field="op")
    ordinal = _ordinal(declaration_ordinal, field="declaration_ordinal")
    if op == "set":
        allowed = frozenset(
            {"op", "species", "time", "time_param", "value", "value_param"}
        )
        _reject_keys(data, allowed, context="set")
        return InterventionSet(
            species=data.get("species"),
            time=data.get("time"),
            time_param=data.get("time_param"),
            value=data.get("value"),
            value_param=data.get("value_param"),
            declaration_ordinal=ordinal,
        )
    if op == "jump":
        allowed = frozenset(
            {"op", "species", "time", "time_param", "delta", "delta_param"}
        )
        _reject_keys(data, allowed, context="jump")
        return InterventionJump(
            species=data.get("species"),
            time=data.get("time"),
            time_param=data.get("time_param"),
            delta=data.get("delta"),
            delta_param=data.get("delta_param"),
            declaration_ordinal=ordinal,
        )
    if op == "flux":
        allowed = frozenset(
            {
                "op",
                "species",
                "start_time",
                "start_time_param",
                "end_time",
                "end_time_param",
                "flux_rate",
                "flux_rate_param",
            }
        )
        _reject_keys(data, allowed, context="flux")
        return InterventionFlux(
            species=data.get("species"),
            start_time=data.get("start_time"),
            start_time_param=data.get("start_time_param"),
            end_time=data.get("end_time"),
            end_time_param=data.get("end_time_param"),
            flux_rate=data.get("flux_rate"),
            flux_rate_param=data.get("flux_rate_param"),
            declaration_ordinal=ordinal,
        )
    if op == "hold":
        allowed = frozenset(
            {
                "op",
                "species",
                "start_time",
                "start_time_param",
                "end_time",
                "end_time_param",
                "value",
                "value_param",
            }
        )
        _reject_keys(data, allowed, context="hold")
        return InterventionHold(
            species=data.get("species"),
            start_time=data.get("start_time"),
            start_time_param=data.get("start_time_param"),
            end_time=data.get("end_time"),
            end_time_param=data.get("end_time_param"),
            value=data.get("value"),
            value_param=data.get("value_param"),
            declaration_ordinal=ordinal,
        )
    if op == "protocol":
        allowed = frozenset(
            {
                "op",
                "kind",
                "start_time",
                "start_time_param",
                "every",
                "count",
                "duration",
                "before",
                "during",
                "after",
            }
        )
        _reject_keys(data, allowed, context="protocol")
        if _enum(data.get("kind"), field="kind") != "repeat":
            raise InterventionScheduleError("Protocol kind must be repeat.")
        phases: dict[str, tuple[InterventionProtocolOperation, ...]] = {}
        for phase in ("before", "during", "after"):
            raw = data.get(phase)
            if raw is None:
                phases[phase] = ()
            elif parsed:
                phases[phase] = _parse_phase_envelope(str(raw), phase=phase)
            elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
                phases[phase] = tuple(
                    _protocol_operation_from_payload(item, phase=phase, index=index)
                    for index, item in enumerate(raw)
                )
            else:
                raise InterventionScheduleError(
                    f"Protocol {phase} must be a sequence of typed records."
                )
        return InterventionProtocol(
            start_time=data.get("start_time"),
            start_time_param=data.get("start_time_param"),
            every=data.get("every"),
            count=data.get("count"),
            duration=data.get("duration"),
            declaration_ordinal=ordinal,
            **phases,
        )
    if op == "trigger":
        allowed = frozenset(
            {
                "op",
                "watch_species",
                "watch_observable",
                "threshold",
                "threshold_param",
                "direction",
                "max_count",
                "min_interval",
                "then",
            }
        )
        _reject_keys(data, allowed, context="trigger")
        raw_then = data.get("then")
        if parsed:
            then = _parse_trigger_envelope(str(raw_then or ""))
        elif isinstance(raw_then, Mapping):
            then = _trigger_action_from_payload(raw_then)
        else:
            raise InterventionScheduleError("then must be one typed trigger action.")
        if isinstance(then, InterventionStopAction) and ({"max_count", "min_interval"} & data.keys()):
            raise InterventionScheduleError("Stopping triggers do not permit max_count or min_interval.")
        if ("watch_species" in data) == ("watch_observable" in data):
            raise InterventionScheduleError("Exactly one of watch_species and watch_observable is required.")
        condition = (
            SpeciesTriggerCondition(data["watch_species"])
            if "watch_species" in data else ObservableTriggerCondition(data["watch_observable"])
        )
        return InterventionTrigger(
            condition=condition,
            threshold=data.get("threshold"),
            threshold_param=data.get("threshold_param"),
            direction=data.get("direction", "either"),
            max_count=data.get("max_count"),
            min_interval=data.get("min_interval"),
            then=then,
            declaration_ordinal=ordinal,
        )
    raise InterventionScheduleError(f"Unsupported intervention op: {op!r}.")


def _protocol_operation_from_payload(
    raw: object, *, phase: str, index: int
) -> InterventionProtocolOperation:
    if not isinstance(raw, Mapping):
        raise InterventionScheduleError(
            f"Protocol {phase} entries must be mappings."
        )
    data = dict(raw)
    authored = data.pop("authored_ordinal", index)
    return _protocol_operation_from_fields(
        data, phase=phase, authored_ordinal=authored
    )


def _protocol_operation_from_fields(
    fields: Mapping[str, Any], *, phase: str, authored_ordinal: object
) -> InterventionProtocolOperation:
    data = dict(fields)
    _literal_fields(data, context=f"protocol {phase}")
    op = _enum(data.get("op"), field="op")
    selected = {
        "set": "value",
        "hold": "value",
        "jump": "delta",
        "flux": "flux_rate",
    }.get(op)
    if selected is None:
        raise InterventionScheduleError(
            f"Unsupported protocol {phase} op: {op!r}."
        )
    allowed = frozenset({"op", "species", selected, f"{selected}_param"})
    _reject_keys(data, allowed, context=f"protocol {phase}")
    return InterventionProtocolOperation(
        phase=phase,
        op=op,
        species=data.get("species"),
        authored_ordinal=_ordinal(authored_ordinal, field="authored_ordinal"),
        **{
            selected: data.get(selected),
            f"{selected}_param": data.get(f"{selected}_param"),
        },
    )


def _trigger_action_from_payload(raw: Mapping[str, Any]) -> InterventionTriggerAction | InterventionStopAction:
    data = dict(raw)
    _literal_fields(data, context="trigger then")
    op = _enum(data.get("op"), field="op")
    if op == "stop":
        _reject_keys(data, frozenset({"op"}), context="stop action")
        return InterventionStopAction()
    selected = {"set": "value", "jump": "delta"}.get(op)
    if selected is None:
        raise InterventionScheduleError(
            "Trigger then permits exactly op=set or op=jump."
        )
    _reject_keys(
        data,
        frozenset({"op", "species", selected, f"{selected}_param"}),
        context="trigger then",
    )
    return InterventionTriggerAction(
        op=op,
        species=data.get("species"),
        **{
            selected: data.get(selected),
            f"{selected}_param": data.get(f"{selected}_param"),
        },
    )


def _lower_authored_fields(
    fields: Mapping[str, str], *, context: str
) -> dict[str, Any]:
    lowered: dict[str, Any] = {}
    for key, value in fields.items():
        if key in _AUTHORED_FLOAT_FIELDS:
            try:
                lowered[key] = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise InterventionScheduleError(
                    f"{context} field {key!r} must be a numeric literal."
                ) from exc
        elif key in _AUTHORED_INTEGER_FIELDS:
            try:
                lowered[key] = int(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise InterventionScheduleError(
                    f"{context} field {key!r} must be an integer literal."
                ) from exc
        else:
            lowered[key] = value
    return lowered


def _parse_record_fields(text: str, *, delimiter: str, context: str) -> dict[str, Any]:
    if not text:
        raise InterventionScheduleError(f"{context} must not be empty.")
    parts = text.split(delimiter)
    if any(not part.strip() for part in parts):
        raise InterventionScheduleError(f"{context} contains an empty field.")
    fields: dict[str, str] = {}
    for part in parts:
        if "=" not in part:
            raise InterventionScheduleError(
                f"{context} field {part.strip()!r} must use key=value."
            )
        key, value = part.split("=", 1)
        key = key.strip().lower()
        value = value.strip()
        if not key or not value:
            raise InterventionScheduleError(f"{context} contains an empty field.")
        if key in fields:
            raise InterventionScheduleError(
                f"Duplicate {context} field: {key!r}."
            )
        fields[key] = value
    return _lower_authored_fields(fields, context=context)


def _unwrap_envelope(value: str, *, context: str) -> str:
    text = str(value).strip()
    if not (text.startswith("{") and text.endswith("}")):
        raise InterventionScheduleError(
            f"{context} must be enclosed by one pair of braces."
        )
    inner = text[1:-1]
    if any(char in inner for char in "{};"):
        raise InterventionScheduleError(
            f"{context} forbids nesting and semicolons."
        )
    return inner


def _parse_phase_envelope(
    value: str, *, phase: str
) -> tuple[InterventionProtocolOperation, ...]:
    inner = _unwrap_envelope(value, context=f"Protocol {phase}")
    records = inner.split("|")
    if any(not record.strip() for record in records):
        raise InterventionScheduleError(
            f"Protocol {phase} contains an empty operation record."
        )
    return tuple(
        _protocol_operation_from_fields(
            _parse_record_fields(
                record.strip(), delimiter=",", context=f"protocol {phase} record"
            ),
            phase=phase,
            authored_ordinal=index,
        )
        for index, record in enumerate(records)
    )


def _parse_trigger_envelope(value: str) -> InterventionTriggerAction | InterventionStopAction:
    inner = _unwrap_envelope(value, context="Trigger then")
    if "|" in inner:
        raise InterventionScheduleError(
            "Trigger then requires exactly one action record."
        )
    return _trigger_action_from_payload(
        _parse_record_fields(inner, delimiter=",", context="trigger then record")
    )


def _parse_directive(rest: str, *, line_no: int) -> dict[str, Any]:
    text = str(rest).strip()
    if not text:
        raise InterventionScheduleError(
            f"Line {line_no}: intervention directive must not be empty."
        )
    fields: dict[str, str] = {}
    chunk: list[str] = []
    depth = 0
    chunks: list[str] = []
    for character in text:
        if character == "{":
            depth += 1
            if depth > 1:
                raise InterventionScheduleError(
                    f"Line {line_no}: nested braces are forbidden."
                )
        elif character == "}":
            depth -= 1
            if depth < 0:
                raise InterventionScheduleError(
                    f"Line {line_no}: unmatched closing brace."
                )
        if character == ";" and depth == 0:
            part = "".join(chunk).strip()
            if not part:
                raise InterventionScheduleError(
                    f"Line {line_no}: empty intervention field."
                )
            chunks.append(part)
            chunk = []
        else:
            chunk.append(character)
    if depth != 0:
        raise InterventionScheduleError(f"Line {line_no}: unmatched opening brace.")
    tail = "".join(chunk).strip()
    if not tail:
        raise InterventionScheduleError(
            f"Line {line_no}: empty intervention field."
        )
    chunks.append(tail)
    for part in chunks:
        if "=" not in part:
            raise InterventionScheduleError(
                f"Line {line_no}: intervention field {part!r} must use key=value."
            )
        key, value = part.split("=", 1)
        key = key.strip().lower()
        value = value.strip()
        if not key or not value:
            raise InterventionScheduleError(
                f"Line {line_no}: intervention fields must not be empty."
            )
        if key in fields:
            raise InterventionScheduleError(
                f"Line {line_no}: duplicate intervention field {key!r}."
            )
        fields[key] = value
    return _lower_authored_fields(fields, context=f"Line {line_no}: intervention")


def _admit_intervention_directives(
    text: str,
) -> InterventionSchedule | None:
    operations: list[InterventionOperation] = []
    for line_no, raw_line in enumerate(str(text or "").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#") or not line.lower().startswith("intervention:"):
            continue
        _, rest = line.split(":", 1)
        try:
            fields = _parse_directive(rest, line_no=line_no)
            operation = _operation_from_fields(fields, declaration_ordinal=len(operations), parsed=True)
            if isinstance(operation, InterventionTrigger):
                operation = replace(operation, source_text=raw_line)
            operations.append(operation)
        except InterventionScheduleError as exc:
            message = str(exc)
            if message.startswith(f"Line {line_no}:"):
                raise
            raise InterventionScheduleError(f"Line {line_no}: {message}") from exc
    schedule = InterventionSchedule(tuple(operations))
    return None if schedule.is_empty() else schedule


def intervention_schedule_parameter_names(
    schedule: InterventionSchedule | None,
) -> set[str]:
    return {
        str(name)
        for values in intervention_schedule_parameter_dependencies(schedule).values()
        for name in values
        if str(name).strip()
    }


def intervention_schedule_parameter_dependencies(
    schedule: InterventionSchedule | None,
) -> InterventionScheduleDependencyMetadata:
    """Return direct dependencies keyed by complete authored field identity."""
    if schedule is None:
        return MappingProxyType({})
    dependencies: dict[InterventionScheduleFieldIdentity, tuple[str, ...]] = {}

    def add(
        declaration: int,
        nested_ordinal: int | None,
        field_path: str,
        parameter: object,
    ) -> None:
        if not isinstance(parameter, str) or not parameter.strip():
            return
        identity = (int(declaration), nested_ordinal, str(field_path))
        dependencies[identity] = (parameter.strip(),)

    for operation in schedule.operations:
        declaration = int(getattr(operation, "declaration_ordinal", 0))
        if isinstance(operation, InterventionProtocol):
            add(declaration, None, "start_time", operation.start_time_param)
            for phase in ("before", "during", "after"):
                for nested in getattr(operation, phase):
                    selected = {
                        "set": "value",
                        "hold": "value",
                        "jump": "delta",
                        "flux": "flux_rate",
                    }[nested.op]
                    add(
                        declaration,
                        int(nested.authored_ordinal),
                        f"{phase}.{selected}",
                        getattr(nested, f"{selected}_param"),
                    )
            continue
        if isinstance(operation, InterventionTrigger):
            add(declaration, None, "threshold", operation.threshold_param)
            if isinstance(operation.then, InterventionStopAction):
                continue
            selected = "value" if operation.then.op == "set" else "delta"
            add(
                declaration,
                None,
                f"then.{selected}",
                getattr(operation.then, f"{selected}_param"),
            )
            continue
        for field in {
            "set": ("time", "value"),
            "jump": ("time", "delta"),
            "flux": ("start_time", "end_time", "flux_rate"),
            "hold": ("start_time", "end_time", "value"),
        }.get(getattr(operation, "op", ""), ()):
            add(
                declaration,
                None,
                field,
                getattr(operation, f"{field}_param", None),
            )
    return MappingProxyType(dict(dependencies))


def coerce_intervention_schedule(value: object) -> InterventionSchedule | None:
    if value is None:
        return None
    if isinstance(value, InterventionSchedule):
        return None if value.is_empty() else value
    if isinstance(value, Mapping):
        schedule = InterventionSchedule.from_payload(value)
        return None if schedule.is_empty() else schedule
    raise InterventionScheduleError(
        "Intervention schedule must be a final typed schedule or final payload."
    )
