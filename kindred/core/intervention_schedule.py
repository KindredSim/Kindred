from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

__all__ = [
    "InterventionInstantEvent",
    "InterventionInterval",
    "InterventionRepeatedEvent",
    "InterventionSchedule",
    "InterventionScheduleError",
    "InterventionTriggerEvent",
    "coerce_intervention_schedule",
    "intervention_schedule_fingerprint_from_dsl_text",
    "parse_intervention_schedule_from_dsl",
]


class InterventionScheduleError(ValueError):
    """Raised when a species intervention schedule is invalid."""


_ABSOLUTE_INSTANT_OPS = frozenset({"set", "clear"})
_ADDITIVE_INSTANT_OPS = frozenset({"add", "remove"})
_INSTANT_OPS = _ABSOLUTE_INSTANT_OPS | _ADDITIVE_INSTANT_OPS
_INTERVAL_KINDS = frozenset({"source", "sink", "reservoir", "clamp"})
_TRIGGER_DIRECTIONS = frozenset({"rising", "falling", "either"})


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _finite_float(value: object, *, field: str) -> float:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise InterventionScheduleError(f"{field} must be a finite number.") from exc
    if not math.isfinite(out):
        raise InterventionScheduleError(f"{field} must be a finite number.")
    return float(out)


def _positive_float(value: object, *, field: str) -> float:
    out = _finite_float(value, field=field)
    if out <= 0.0:
        raise InterventionScheduleError(f"{field} must be greater than zero.")
    return out


def _nonnegative_float(value: object, *, field: str) -> float:
    out = _finite_float(value, field=field)
    if out < 0.0:
        raise InterventionScheduleError(f"{field} must be non-negative.")
    return out


def _parameter_name(value: object, *, field: str) -> str:
    name = str(value or "").strip()
    if not name:
        raise InterventionScheduleError(f"{field} must not be empty.")
    return name


def _payload_scalar(
    payload: Mapping[str, Any],
    *,
    field: str,
    param_field: str | None = None,
    nonnegative: bool = False,
    positive: bool = False,
) -> tuple[float | None, str | None]:
    parameter_key = param_field or f"{field}_param"
    has_value = field in payload and payload.get(field) is not None
    has_param = parameter_key in payload and str(payload.get(parameter_key) or "").strip()
    if has_value and has_param:
        raise InterventionScheduleError(f"{field} and {parameter_key} are mutually exclusive.")
    if has_param:
        return None, _parameter_name(payload.get(parameter_key), field=parameter_key)
    if positive:
        return _positive_float(payload.get(field), field=field), None
    if nonnegative:
        return _nonnegative_float(payload.get(field), field=field), None
    return _finite_float(payload.get(field), field=field), None


def _resolve_scalar(
    value: float | None,
    parameter: str | None,
    parameter_values: Mapping[str, Any],
    *,
    field: str,
    nonnegative: bool = False,
    positive: bool = False,
) -> float:
    if parameter:
        if parameter not in parameter_values:
            raise InterventionScheduleError(f"Missing intervention schedule parameter: {parameter}")
        raw_value = parameter_values[parameter]
    else:
        raw_value = value
    if positive:
        return _positive_float(raw_value, field=field)
    if nonnegative:
        return _nonnegative_float(raw_value, field=field)
    return _finite_float(raw_value, field=field)


def _write_scalar_payload(
    payload: dict[str, Any],
    *,
    field: str,
    value: float | None,
    parameter: str | None,
) -> None:
    if parameter:
        payload[f"{field}_param"] = str(parameter)
    else:
        payload[field] = float(value if value is not None else 0.0)


def _species_name(value: object) -> str:
    name = str(value or "").strip()
    if not name:
        raise InterventionScheduleError("species must not be empty.")
    return name


@dataclass(frozen=True, slots=True)
class InterventionInstantEvent:
    time: float | None
    species: str
    op: str
    value: float | None = None
    amount: float | None = None
    time_param: str | None = None
    value_param: str | None = None
    amount_param: str | None = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "InterventionInstantEvent":
        op = str(payload.get("op") or "").strip().lower()
        if op not in _INSTANT_OPS:
            raise InterventionScheduleError(f"Unsupported instant intervention op: {op!r}.")
        time, time_param = _payload_scalar(payload, field="time")
        species = _species_name(payload.get("species"))
        if op in {"set"}:
            value, value_param = _payload_scalar(payload, field="value")
            return cls(
                time=time,
                species=species,
                op=op,
                value=value,
                time_param=time_param,
                value_param=value_param,
            )
        if op == "clear":
            return cls(time=time, species=species, op=op, value=0.0, time_param=time_param)
        amount, amount_param = _payload_scalar(payload, field="amount", nonnegative=True)
        return cls(
            time=time,
            species=species,
            op=op,
            amount=amount,
            time_param=time_param,
            amount_param=amount_param,
        )

    def resolve_parameters(self, parameter_values: Mapping[str, Any]) -> "InterventionInstantEvent":
        time = _resolve_scalar(self.time, self.time_param, parameter_values, field="time")
        if self.op == "set":
            return InterventionInstantEvent(
                time=time,
                species=self.species,
                op=self.op,
                value=_resolve_scalar(self.value, self.value_param, parameter_values, field="value"),
            )
        if self.op == "clear":
            return InterventionInstantEvent(time=time, species=self.species, op=self.op, value=0.0)
        return InterventionInstantEvent(
            time=time,
            species=self.species,
            op=self.op,
            amount=_resolve_scalar(
                self.amount,
                self.amount_param,
                parameter_values,
                field="amount",
                nonnegative=True,
            ),
        )

    @property
    def is_parameterized(self) -> bool:
        return bool(self.time_param or self.value_param or self.amount_param)

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "species": str(self.species),
            "op": str(self.op),
        }
        _write_scalar_payload(payload, field="time", value=self.time, parameter=self.time_param)
        if self.op in _ABSOLUTE_INSTANT_OPS:
            if self.op != "clear":
                _write_scalar_payload(payload, field="value", value=self.value, parameter=self.value_param)
        else:
            _write_scalar_payload(payload, field="amount", value=self.amount, parameter=self.amount_param)
        return payload


@dataclass(frozen=True, slots=True)
class InterventionRepeatedEvent:
    start: float | None
    every: float | None
    count: int
    species: str
    op: str = "pulse"
    amount: float | None = None
    start_param: str | None = None
    every_param: str | None = None
    amount_param: str | None = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "InterventionRepeatedEvent":
        count = int(payload.get("count") or 0)
        if count <= 0:
            raise InterventionScheduleError("count must be greater than zero.")
        op = str(payload.get("op") or "pulse").strip().lower()
        if op not in {"pulse", "add", "remove"}:
            raise InterventionScheduleError(f"Unsupported repeated intervention op: {op!r}.")
        start, start_param = _payload_scalar(payload, field="start")
        every, every_param = _payload_scalar(payload, field="every", positive=True)
        amount, amount_param = _payload_scalar(payload, field="amount", nonnegative=True)
        return cls(
            start=start,
            every=every,
            count=count,
            species=_species_name(payload.get("species")),
            op=op,
            amount=amount,
            start_param=start_param,
            every_param=every_param,
            amount_param=amount_param,
        )

    @property
    def is_parameterized(self) -> bool:
        return bool(self.start_param or self.every_param or self.amount_param)

    def resolve_parameters(self, parameter_values: Mapping[str, Any]) -> "InterventionRepeatedEvent":
        return InterventionRepeatedEvent(
            start=_resolve_scalar(self.start, self.start_param, parameter_values, field="start"),
            every=_resolve_scalar(
                self.every,
                self.every_param,
                parameter_values,
                field="every",
                positive=True,
            ),
            count=int(self.count),
            species=self.species,
            op=self.op,
            amount=_resolve_scalar(
                self.amount,
                self.amount_param,
                parameter_values,
                field="amount",
                nonnegative=True,
            ),
        )

    def expand(self) -> tuple[InterventionInstantEvent, ...]:
        if self.is_parameterized:
            raise InterventionScheduleError("Parameterized repeated intervention must be resolved before solving.")
        start = float(self.start if self.start is not None else 0.0)
        every = _positive_float(self.every, field="every")
        amount = _nonnegative_float(self.amount, field="amount")
        op = "add" if self.op == "pulse" else str(self.op)
        return tuple(
            InterventionInstantEvent(
                time=start + every * float(idx),
                species=self.species,
                op=op,
                amount=amount,
            )
            for idx in range(int(self.count))
        )

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "species": str(self.species),
            "op": str(self.op),
            "count": int(self.count),
        }
        _write_scalar_payload(payload, field="start", value=self.start, parameter=self.start_param)
        _write_scalar_payload(payload, field="every", value=self.every, parameter=self.every_param)
        _write_scalar_payload(payload, field="amount", value=self.amount, parameter=self.amount_param)
        return payload


@dataclass(frozen=True, slots=True)
class InterventionTriggerEvent:
    trigger_species: str
    threshold: float | None
    direction: str
    species: str
    op: str
    max_count: int
    min_interval: float
    value: float | None = None
    amount: float | None = None
    threshold_param: str | None = None
    value_param: str | None = None
    amount_param: str | None = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "InterventionTriggerEvent":
        direction = str(payload.get("direction") or "").strip().lower()
        if direction not in _TRIGGER_DIRECTIONS:
            raise InterventionScheduleError("direction must be rising, falling, or either.")
        action = str(payload.get("action") or payload.get("then") or "").strip().lower()
        if action not in _INSTANT_OPS:
            raise InterventionScheduleError(f"Unsupported trigger action: {action!r}.")
        max_count = int(payload.get("max_count") or 0)
        if max_count <= 0:
            raise InterventionScheduleError("max_count must be greater than zero.")
        min_interval = _nonnegative_float(payload.get("min_interval"), field="min_interval")
        threshold, threshold_param = _payload_scalar(payload, field="threshold")
        base = dict(
            trigger_species=_species_name(payload.get("trigger_species")),
            threshold=threshold,
            threshold_param=threshold_param,
            direction=direction,
            species=_species_name(payload.get("species")),
            op=action,
            max_count=max_count,
            min_interval=min_interval,
        )
        if action == "set":
            value, value_param = _payload_scalar(payload, field="value")
            return cls(**base, value=value, value_param=value_param)
        if action == "clear":
            return cls(**base, value=0.0)
        amount, amount_param = _payload_scalar(payload, field="amount", nonnegative=True)
        return cls(**base, amount=amount, amount_param=amount_param)

    @property
    def is_parameterized(self) -> bool:
        return bool(self.threshold_param or self.value_param or self.amount_param)

    def resolve_parameters(self, parameter_values: Mapping[str, Any]) -> "InterventionTriggerEvent":
        threshold = _resolve_scalar(self.threshold, self.threshold_param, parameter_values, field="threshold")
        if self.op == "set":
            return InterventionTriggerEvent(
                trigger_species=self.trigger_species,
                threshold=threshold,
                direction=self.direction,
                species=self.species,
                op=self.op,
                max_count=int(self.max_count),
                min_interval=float(self.min_interval),
                value=_resolve_scalar(self.value, self.value_param, parameter_values, field="value"),
            )
        if self.op == "clear":
            return InterventionTriggerEvent(
                trigger_species=self.trigger_species,
                threshold=threshold,
                direction=self.direction,
                species=self.species,
                op=self.op,
                max_count=int(self.max_count),
                min_interval=float(self.min_interval),
                value=0.0,
            )
        return InterventionTriggerEvent(
            trigger_species=self.trigger_species,
            threshold=threshold,
            direction=self.direction,
            species=self.species,
            op=self.op,
            max_count=int(self.max_count),
            min_interval=float(self.min_interval),
            amount=_resolve_scalar(
                self.amount,
                self.amount_param,
                parameter_values,
                field="amount",
                nonnegative=True,
            ),
        )

    def to_instant_event(self, *, time: float) -> InterventionInstantEvent:
        if self.op == "set":
            return InterventionInstantEvent(time=float(time), species=self.species, op=self.op, value=self.value)
        if self.op == "clear":
            return InterventionInstantEvent(time=float(time), species=self.species, op=self.op, value=0.0)
        return InterventionInstantEvent(time=float(time), species=self.species, op=self.op, amount=self.amount)

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "trigger_species": str(self.trigger_species),
            "direction": str(self.direction),
            "species": str(self.species),
            "action": str(self.op),
            "max_count": int(self.max_count),
            "min_interval": float(self.min_interval),
        }
        _write_scalar_payload(payload, field="threshold", value=self.threshold, parameter=self.threshold_param)
        if self.op == "set":
            _write_scalar_payload(payload, field="value", value=self.value, parameter=self.value_param)
        elif self.op in _ADDITIVE_INSTANT_OPS:
            _write_scalar_payload(payload, field="amount", value=self.amount, parameter=self.amount_param)
        return payload


@dataclass(frozen=True, slots=True)
class InterventionInterval:
    start: float | None
    end: float | None
    species: str
    kind: str
    rate: float | None = None
    value: float | None = None
    start_param: str | None = None
    end_param: str | None = None
    rate_param: str | None = None
    value_param: str | None = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "InterventionInterval":
        kind = str(payload.get("kind") or payload.get("op") or "").strip().lower()
        if kind not in _INTERVAL_KINDS:
            raise InterventionScheduleError(f"Unsupported intervention interval kind: {kind!r}.")
        start, start_param = _payload_scalar(payload, field="start")
        end, end_param = _payload_scalar(payload, field="end")
        if start_param is None and end_param is None and end <= start:
            raise InterventionScheduleError("end must be greater than start.")
        species = _species_name(payload.get("species"))
        if kind in {"source", "sink"}:
            rate, rate_param = _payload_scalar(payload, field="rate", nonnegative=True)
            return cls(
                start=start,
                end=end,
                species=species,
                kind=kind,
                rate=rate,
                start_param=start_param,
                end_param=end_param,
                rate_param=rate_param,
            )
        value, value_param = _payload_scalar(payload, field="value")
        return cls(
            start=start,
            end=end,
            species=species,
            kind=kind,
            value=value,
            start_param=start_param,
            end_param=end_param,
            value_param=value_param,
        )

    def resolve_parameters(self, parameter_values: Mapping[str, Any]) -> "InterventionInterval":
        start = _resolve_scalar(self.start, self.start_param, parameter_values, field="start")
        end = _resolve_scalar(self.end, self.end_param, parameter_values, field="end")
        if end <= start:
            raise InterventionScheduleError("end must be greater than start.")
        if self.kind in {"source", "sink"}:
            return InterventionInterval(
                start=start,
                end=end,
                species=self.species,
                kind=self.kind,
                rate=_resolve_scalar(
                    self.rate,
                    self.rate_param,
                    parameter_values,
                    field="rate",
                    nonnegative=True,
                ),
            )
        return InterventionInterval(
            start=start,
            end=end,
            species=self.species,
            kind=self.kind,
            value=_resolve_scalar(self.value, self.value_param, parameter_values, field="value"),
        )

    @property
    def is_parameterized(self) -> bool:
        return bool(self.start_param or self.end_param or self.rate_param or self.value_param)

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "species": str(self.species),
            "kind": str(self.kind),
        }
        _write_scalar_payload(payload, field="start", value=self.start, parameter=self.start_param)
        _write_scalar_payload(payload, field="end", value=self.end, parameter=self.end_param)
        if self.kind in {"source", "sink"}:
            _write_scalar_payload(payload, field="rate", value=self.rate, parameter=self.rate_param)
        else:
            _write_scalar_payload(payload, field="value", value=self.value, parameter=self.value_param)
        return payload


@dataclass(frozen=True, slots=True)
class InterventionSchedule:
    instant_events: tuple[InterventionInstantEvent, ...] = ()
    repeated_events: tuple[InterventionRepeatedEvent, ...] = ()
    trigger_events: tuple[InterventionTriggerEvent, ...] = ()
    intervals: tuple[InterventionInterval, ...] = ()
    version: int = 1

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | None) -> "InterventionSchedule":
        if not isinstance(payload, Mapping):
            return cls()
        instant_events: list[InterventionInstantEvent] = []
        for item in payload.get("instant_events") or ():
            if not isinstance(item, Mapping):
                raise InterventionScheduleError("instant_events entries must be mappings.")
            instant_events.append(InterventionInstantEvent.from_payload(item))
        repeated_events: list[InterventionRepeatedEvent] = []
        for item in payload.get("repeated_events") or ():
            if not isinstance(item, Mapping):
                raise InterventionScheduleError("repeated_events entries must be mappings.")
            repeated_events.append(InterventionRepeatedEvent.from_payload(item))
        trigger_events: list[InterventionTriggerEvent] = []
        for item in payload.get("trigger_events") or ():
            if not isinstance(item, Mapping):
                raise InterventionScheduleError("trigger_events entries must be mappings.")
            trigger_events.append(InterventionTriggerEvent.from_payload(item))
        intervals: list[InterventionInterval] = []
        for item in payload.get("intervals") or ():
            if not isinstance(item, Mapping):
                raise InterventionScheduleError("intervals entries must be mappings.")
            intervals.append(InterventionInterval.from_payload(item))
        schedule = cls(
            instant_events=tuple(sorted(instant_events, key=_instant_sort_key)),
            repeated_events=tuple(sorted(repeated_events, key=_repeated_sort_key)),
            trigger_events=tuple(sorted(trigger_events, key=_trigger_sort_key)),
            intervals=tuple(sorted(intervals, key=_interval_sort_key)),
            version=int(payload.get("version") or 1),
        )
        schedule._validate_conflicts()
        return schedule

    def is_empty(self) -> bool:
        return not self.instant_events and not self.repeated_events and not self.trigger_events and not self.intervals

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": int(self.version),
            "instant_events": [event.to_payload() for event in self.instant_events],
            "repeated_events": [event.to_payload() for event in self.repeated_events],
            "trigger_events": [event.to_payload() for event in self.trigger_events],
            "intervals": [interval.to_payload() for interval in self.intervals],
        }

    @property
    def is_parameterized(self) -> bool:
        return (
            any(event.is_parameterized for event in self.instant_events)
            or any(event.is_parameterized for event in self.repeated_events)
            or any(event.is_parameterized for event in self.trigger_events)
            or any(interval.is_parameterized for interval in self.intervals)
        )

    def resolve_parameters(self, parameter_values: Mapping[str, Any] | None) -> "InterventionSchedule":
        if not self.is_parameterized:
            return self
        values = dict(parameter_values or {})
        schedule = InterventionSchedule(
            instant_events=tuple(event.resolve_parameters(values) for event in self.instant_events),
            repeated_events=tuple(event.resolve_parameters(values) for event in self.repeated_events),
            trigger_events=tuple(event.resolve_parameters(values) for event in self.trigger_events),
            intervals=tuple(interval.resolve_parameters(values) for interval in self.intervals),
            version=int(self.version),
        )
        schedule._validate_conflicts()
        return schedule

    @property
    def fingerprint(self) -> str:
        if self.is_empty():
            return ""
        return hashlib.sha256(_canonical_json_bytes(self.to_payload())).hexdigest()

    def validate_species(self, species_names: Sequence[str]) -> None:
        available = {str(name) for name in species_names}
        for event in self.instant_events:
            if event.species not in available:
                raise InterventionScheduleError(f"Unknown intervention species: {event.species}")
        for event in self.repeated_events:
            if event.species not in available:
                raise InterventionScheduleError(f"Unknown intervention species: {event.species}")
        for event in self.trigger_events:
            if event.trigger_species not in available:
                raise InterventionScheduleError(f"Unknown intervention trigger species: {event.trigger_species}")
            if event.species not in available:
                raise InterventionScheduleError(f"Unknown intervention species: {event.species}")
        for interval in self.intervals:
            if interval.species not in available:
                raise InterventionScheduleError(f"Unknown intervention species: {interval.species}")

    def _validate_conflicts(self) -> None:
        absolute_by_key: dict[tuple[object, str], InterventionInstantEvent] = {}
        delta_keys: set[tuple[object, str]] = set()
        for event in _validation_instant_events(self):
            key = (_scalar_identity(event.time, event.time_param), str(event.species))
            if event.op not in _ABSOLUTE_INSTANT_OPS:
                delta_keys.add(key)
                if key in absolute_by_key:
                    raise InterventionScheduleError(
                        f"Cannot combine absolute and add/remove interventions for {event.species} at t={_format_scalar_identity(key[0])}."
                    )
                continue
            if key in delta_keys:
                raise InterventionScheduleError(
                    f"Cannot combine absolute and add/remove interventions for {event.species} at t={_format_scalar_identity(key[0])}."
                )
            existing = absolute_by_key.get(key)
            if existing is not None and existing.to_payload() != event.to_payload():
                raise InterventionScheduleError(
                    f"Conflicting absolute interventions for {event.species} at t={_format_scalar_identity(key[0])}."
                )
            absolute_by_key[key] = event
        clamp_like = [interval for interval in self.intervals if interval.kind in {"reservoir", "clamp"}]
        for i, left in enumerate(clamp_like):
            for right in clamp_like[i + 1 :]:
                if left.species != right.species:
                    continue
                if left.start_param or left.end_param or right.start_param or right.end_param:
                    continue
                if left.start < right.end and right.start < left.end:
                    raise InterventionScheduleError(
                        f"Overlapping reservoir/clamp intervals for {left.species} are not allowed."
                    )


def _scalar_identity(value: float | None, parameter: str | None) -> object:
    if parameter:
        return ("param", str(parameter))
    return ("fixed", float(value if value is not None else 0.0))


def _format_scalar_identity(value: object) -> str:
    if isinstance(value, tuple) and len(value) == 2 and value[0] == "param":
        return str(value[1])
    if isinstance(value, tuple) and len(value) == 2 and value[0] == "fixed":
        try:
            return f"{float(value[1]):g}"
        except (TypeError, ValueError, OverflowError):
            return str(value[1])
    return str(value)


def _instant_sort_key(event: InterventionInstantEvent) -> tuple[float, str, int, str]:
    op_order = {"set": 0, "clear": 0, "remove": 1, "add": 2}
    time = float(event.time) if event.time is not None else math.inf
    return (time, str(event.species), op_order.get(event.op, 99), str(event.op))


def _interval_sort_key(interval: InterventionInterval) -> tuple[float, float, str, str]:
    start = float(interval.start) if interval.start is not None else math.inf
    end = float(interval.end) if interval.end is not None else math.inf
    return (start, end, str(interval.species), str(interval.kind))


def _repeated_sort_key(event: InterventionRepeatedEvent) -> tuple[float, str, str]:
    start = float(event.start) if event.start is not None else math.inf
    return (start, str(event.species), str(event.op))


def _trigger_sort_key(event: InterventionTriggerEvent) -> tuple[str, str, str, str]:
    return (str(event.trigger_species), str(event.species), str(event.direction), str(event.op))


def _expanded_instant_events(schedule: InterventionSchedule) -> tuple[InterventionInstantEvent, ...]:
    events = list(schedule.instant_events)
    for repeated_event in schedule.repeated_events:
        events.extend(repeated_event.expand())
    return tuple(sorted(events, key=_instant_sort_key))


def _validation_instant_events(schedule: InterventionSchedule) -> tuple[InterventionInstantEvent, ...]:
    events = list(schedule.instant_events)
    for repeated_event in schedule.repeated_events:
        if repeated_event.is_parameterized:
            continue
        events.extend(repeated_event.expand())
    return tuple(sorted(events, key=_instant_sort_key))


def coerce_intervention_schedule(value: object) -> InterventionSchedule | None:
    if value is None:
        return None
    if isinstance(value, InterventionSchedule):
        return None if value.is_empty() else value
    if isinstance(value, Mapping):
        schedule = InterventionSchedule.from_payload(value)
        return None if schedule.is_empty() else schedule
    raise InterventionScheduleError("Intervention schedule must be a mapping or InterventionSchedule.")


def parse_intervention_schedule_from_dsl(text: str) -> InterventionSchedule | None:
    instant_events: list[dict[str, Any]] = []
    repeated_events: list[dict[str, Any]] = []
    trigger_events: list[dict[str, Any]] = []
    intervals: list[dict[str, Any]] = []
    for line_no, raw in enumerate(str(text or "").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#") or not line.lower().startswith("intervention:"):
            continue
        _, rest = line.split(":", 1)
        fields = _parse_directive_fields(rest, line_no=line_no)
        op = str(fields.get("op") or "").strip().lower()
        try:
            if op in {"set", "add", "remove", "clear"}:
                instant_events.append(fields)
            elif op == "pulse":
                repeated_events.append(fields)
            elif op == "trigger":
                trigger_events.append(fields)
            elif op in _INTERVAL_KINDS:
                interval = dict(fields)
                interval["kind"] = op
                intervals.append(interval)
            else:
                raise InterventionScheduleError(f"Unsupported intervention op: {op!r}.")
        except InterventionScheduleError as exc:
            raise InterventionScheduleError(f"Line {line_no}: {exc}") from exc
    schedule = InterventionSchedule.from_payload(
        {
            "instant_events": instant_events,
            "repeated_events": repeated_events,
            "trigger_events": trigger_events,
            "intervals": intervals,
        }
    )
    return None if schedule.is_empty() else schedule


def intervention_schedule_fingerprint_from_dsl_text(text: str) -> str:
    schedule = parse_intervention_schedule_from_dsl(text)
    return "" if schedule is None else schedule.fingerprint


def _parse_directive_fields(rest: str, *, line_no: int) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for chunk in str(rest or "").split(";"):
        part = chunk.strip()
        if not part:
            continue
        if "=" not in part:
            raise InterventionScheduleError(f"Line {line_no}: invalid intervention field {part!r}.")
        key, value = part.split("=", 1)
        fields[str(key).strip().lower()] = value.strip()
    return fields


def active_interval_boundaries(schedule: InterventionSchedule, *, t0: float, t1: float) -> list[float]:
    if schedule.is_parameterized:
        raise InterventionScheduleError("Parameterized intervention schedule must be resolved before solving.")
    boundaries = {float(t0), float(t1)}
    for event in _expanded_instant_events(schedule):
        if float(t0) <= event.time <= float(t1):
            boundaries.add(float(event.time))
    for interval in schedule.intervals:
        if float(t0) <= interval.start <= float(t1):
            boundaries.add(float(interval.start))
        if float(t0) <= interval.end <= float(t1):
            boundaries.add(float(interval.end))
    return sorted(boundaries)


def events_at_time(schedule: InterventionSchedule, time: float) -> tuple[InterventionInstantEvent, ...]:
    if schedule.is_parameterized:
        raise InterventionScheduleError("Parameterized intervention schedule must be resolved before solving.")
    return tuple(event for event in _expanded_instant_events(schedule) if float(event.time) == float(time))


def intervals_active_at(schedule: InterventionSchedule, time: float) -> tuple[InterventionInterval, ...]:
    if schedule.is_parameterized:
        raise InterventionScheduleError("Parameterized intervention schedule must be resolved before solving.")
    return tuple(
        interval
        for interval in schedule.intervals
        if float(interval.start) <= float(time) < float(interval.end)
    )
