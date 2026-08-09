from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence, TypeAlias

__all__ = [
    "InterventionFlux",
    "InterventionHold",
    "InterventionJump",
    "InterventionProtocol",
    "InterventionProtocolOperation",
    "InterventionSchedule",
    "InterventionScheduleError",
    "InterventionSet",
    "InterventionTrigger",
    "InterventionTriggerAction",
    "coerce_intervention_schedule",
    "intervention_schedule_parameter_names",
]


class InterventionScheduleError(ValueError):
    """The authored intervention source is outside the final public schema."""


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


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _finite(value: object, *, field: str) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise InterventionScheduleError(f"{field} must be a finite number.") from exc
    if not math.isfinite(number):
        raise InterventionScheduleError(f"{field} must be a finite number.")
    return number


def _nonnegative(value: object, *, field: str) -> float:
    number = _finite(value, field=field)
    if number < 0.0:
        raise InterventionScheduleError(f"{field} must be non-negative.")
    return number


def _positive(value: object, *, field: str) -> float:
    number = _finite(value, field=field)
    if number <= 0.0:
        raise InterventionScheduleError(f"{field} must be greater than zero.")
    return number


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool):
        raise InterventionScheduleError(f"{field} must be a positive integer.")
    if isinstance(value, int):
        number = value
    elif isinstance(value, str) and value.strip().isdecimal():
        number = int(value.strip())
    else:
        raise InterventionScheduleError(f"{field} must be a positive integer.")
    if number <= 0:
        raise InterventionScheduleError(f"{field} must be a positive integer.")
    return number


def _name(value: object, *, field: str) -> str:
    name = str(value or "").strip()
    if not name:
        raise InterventionScheduleError(f"{field} must not be empty.")
    return name


def _ordinal(value: object, *, field: str) -> int:
    if isinstance(value, bool):
        raise InterventionScheduleError(f"{field} must be a non-negative integer.")
    try:
        number = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise InterventionScheduleError(f"{field} must be a non-negative integer.") from exc
    if number < 0 or number != value:
        raise InterventionScheduleError(f"{field} must be a non-negative integer.")
    return number


def _pair(
    value: object,
    parameter: object,
    *,
    field: str,
    domain: str,
) -> tuple[float | None, str | None]:
    has_value = value is not None
    has_parameter = parameter is not None and bool(str(parameter).strip())
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
    }[domain]
    return validator(value, field=field), None


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
                f"Missing intervention schedule parameter: {parameter}"
            )
        raw = parameter_values[parameter]
    else:
        raw = value
    validator = {
        "signed": _finite,
        "nonnegative": _nonnegative,
        "positive": _positive,
    }[domain]
    return validator(raw, field=field)


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
            self.value, self.value_param, field="value", domain="nonnegative"
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

    def resolve_parameters(self, values: Mapping[str, Any]) -> "InterventionSet":
        return InterventionSet(
            species=self.species,
            time=_resolved(
                self.time, self.time_param, values, field="time", domain="nonnegative"
            ),
            value=_resolved(
                self.value, self.value_param, values, field="value", domain="nonnegative"
            ),
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

    def resolve_parameters(self, values: Mapping[str, Any]) -> "InterventionJump":
        return InterventionJump(
            species=self.species,
            time=_resolved(
                self.time, self.time_param, values, field="time", domain="nonnegative"
            ),
            delta=_resolved(
                self.delta, self.delta_param, values, field="delta", domain="signed"
            ),
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

    def resolve_parameters(self, values: Mapping[str, Any]) -> "InterventionFlux":
        return InterventionFlux(
            species=self.species,
            start_time=_resolved(
                self.start_time,
                self.start_time_param,
                values,
                field="start_time",
                domain="nonnegative",
            ),
            end_time=_resolved(
                self.end_time,
                self.end_time_param,
                values,
                field="end_time",
                domain="nonnegative",
            ),
            flux_rate=_resolved(
                self.flux_rate,
                self.flux_rate_param,
                values,
                field="flux_rate",
                domain="signed",
            ),
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
                domain="nonnegative",
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

    def resolve_parameters(self, values: Mapping[str, Any]) -> "InterventionHold":
        return InterventionHold(
            species=self.species,
            start_time=_resolved(
                self.start_time,
                self.start_time_param,
                values,
                field="start_time",
                domain="nonnegative",
            ),
            end_time=_resolved(
                self.end_time,
                self.end_time_param,
                values,
                field="end_time",
                domain="nonnegative",
            ),
            value=_resolved(
                self.value,
                self.value_param,
                values,
                field="value",
                domain="nonnegative",
            ),
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
        phase = str(self.phase).strip().lower()
        op = str(self.op).strip().lower()
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
                    domain="nonnegative" if field == "value" else "signed",
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
        self, values: Mapping[str, Any]
    ) -> "InterventionProtocolOperation":
        selected = {"set": "value", "hold": "value", "jump": "delta", "flux": "flux_rate"}[self.op]
        kwargs: dict[str, Any] = {
            "phase": self.phase,
            "op": self.op,
            "species": self.species,
            "authored_ordinal": self.authored_ordinal,
        }
        kwargs[selected] = _resolved(
            getattr(self, selected),
            getattr(self, f"{selected}_param"),
            values,
            field=selected,
            domain="nonnegative" if selected == "value" else "signed",
        )
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

    def resolve_parameters(self, values: Mapping[str, Any]) -> "InterventionProtocol":
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
            before=tuple(item.resolve_parameters(values) for item in self.before),
            during=tuple(item.resolve_parameters(values) for item in self.during),
            after=tuple(item.resolve_parameters(values) for item in self.after),
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
        op = str(self.op).strip().lower()
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
            domain="nonnegative" if selected == "value" else "signed",
        )
        object.__setattr__(self, selected, value)
        object.__setattr__(self, f"{selected}_param", parameter)

    @property
    def is_parameterized(self) -> bool:
        return self.value_param is not None or self.delta_param is not None

    def resolve_parameters(
        self, values: Mapping[str, Any]
    ) -> "InterventionTriggerAction":
        selected = "value" if self.op == "set" else "delta"
        return InterventionTriggerAction(
            op=self.op,
            species=self.species,
            **{
                selected: _resolved(
                    getattr(self, selected),
                    getattr(self, f"{selected}_param"),
                    values,
                    field=selected,
                    domain="nonnegative" if selected == "value" else "signed",
                )
            },
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
class InterventionTrigger:
    watch_species: str
    max_count: int
    min_interval: float
    then: InterventionTriggerAction
    threshold: float | None = None
    threshold_param: str | None = None
    direction: str = "either"
    declaration_ordinal: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "watch_species", _name(self.watch_species, field="watch_species")
        )
        threshold, threshold_param = _pair(
            self.threshold,
            self.threshold_param,
            field="threshold",
            domain="nonnegative",
        )
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(self, "threshold_param", threshold_param)
        direction = str(self.direction or "either").strip().lower()
        if direction not in _DIRECTIONS:
            raise InterventionScheduleError(
                "direction must be rising, falling, or either."
            )
        object.__setattr__(self, "direction", direction)
        object.__setattr__(
            self, "max_count", _positive_int(self.max_count, field="max_count")
        )
        object.__setattr__(
            self,
            "min_interval",
            _nonnegative(self.min_interval, field="min_interval"),
        )
        if not isinstance(self.then, InterventionTriggerAction):
            raise InterventionScheduleError(
                "then must be one typed trigger action."
            )
        object.__setattr__(
            self,
            "declaration_ordinal",
            _ordinal(self.declaration_ordinal, field="declaration_ordinal"),
        )

    @property
    def op(self) -> str:
        return "trigger"

    @property
    def is_parameterized(self) -> bool:
        return self.threshold_param is not None or self.then.is_parameterized

    def resolve_parameters(self, values: Mapping[str, Any]) -> "InterventionTrigger":
        return InterventionTrigger(
            watch_species=self.watch_species,
            threshold=_resolved(
                self.threshold,
                self.threshold_param,
                values,
                field="threshold",
                domain="nonnegative",
            ),
            direction=self.direction,
            max_count=self.max_count,
            min_interval=self.min_interval,
            then=self.then.resolve_parameters(values),
            declaration_ordinal=self.declaration_ordinal,
        )

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "op": "trigger",
            "watch_species": self.watch_species,
            "direction": self.direction,
            "max_count": self.max_count,
            "min_interval": self.min_interval,
            "then": self.then.to_payload(),
            "declaration_ordinal": self.declaration_ordinal,
        }
        _put_pair(
            payload,
            field="threshold",
            value=self.threshold,
            parameter=self.threshold_param,
        )
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
        self, parameter_values: Mapping[str, Any] | None
    ) -> "InterventionSchedule":
        if not self.is_parameterized:
            return self
        values = dict(parameter_values or {})
        return InterventionSchedule(
            tuple(item.resolve_parameters(values) for item in self.operations)
        )

    def to_payload(self) -> dict[str, Any]:
        return {"operations": [item.to_payload() for item in self.operations]}

    @property
    def fingerprint(self) -> str:
        if self.is_empty():
            return ""
        return hashlib.sha256(_canonical_bytes(self.to_payload())).hexdigest()

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
                names = (operation.watch_species, operation.then.species)
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
    def protocols(self) -> tuple[InterventionProtocol, ...]:
        return tuple(item for item in self.operations if isinstance(item, InterventionProtocol))

    @property
    def triggers(self) -> tuple[InterventionTrigger, ...]:
        return tuple(item for item in self.operations if isinstance(item, InterventionTrigger))


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
    op = str(data.get("op") or "").strip().lower()
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
        if str(data.get("kind") or "").strip().lower() != "repeat":
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
        return InterventionTrigger(
            watch_species=data.get("watch_species"),
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
    op = str(data.get("op") or "").strip().lower()
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


def _trigger_action_from_payload(raw: Mapping[str, Any]) -> InterventionTriggerAction:
    data = dict(raw)
    _literal_fields(data, context="trigger then")
    op = str(data.get("op") or "").strip().lower()
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


def _parse_record_fields(text: str, *, delimiter: str, context: str) -> dict[str, str]:
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
    return fields


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


def _parse_trigger_envelope(value: str) -> InterventionTriggerAction:
    inner = _unwrap_envelope(value, context="Trigger then")
    if "|" in inner:
        raise InterventionScheduleError(
            "Trigger then requires exactly one action record."
        )
    return _trigger_action_from_payload(
        _parse_record_fields(inner, delimiter=",", context="trigger then record")
    )


def _parse_directive(rest: str, *, line_no: int) -> dict[str, str]:
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
    return fields


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
            operations.append(
                _operation_from_fields(
                    fields,
                    declaration_ordinal=len(operations),
                    parsed=True,
                )
            )
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
    if schedule is None:
        return set()
    names: set[str] = set()

    def walk(value: object) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if str(key) in _PARAM_FIELDS and str(item or "").strip():
                    names.add(str(item))
                else:
                    walk(item)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for item in value:
                walk(item)

    walk(schedule.to_payload())
    return names


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
