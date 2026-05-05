from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

__all__ = [
    "InterventionInstantEvent",
    "InterventionInterval",
    "InterventionSchedule",
    "InterventionScheduleError",
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


def _species_name(value: object) -> str:
    name = str(value or "").strip()
    if not name:
        raise InterventionScheduleError("species must not be empty.")
    return name


@dataclass(frozen=True, slots=True)
class InterventionInstantEvent:
    time: float
    species: str
    op: str
    value: float | None = None
    amount: float | None = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "InterventionInstantEvent":
        op = str(payload.get("op") or "").strip().lower()
        if op not in _INSTANT_OPS:
            raise InterventionScheduleError(f"Unsupported instant intervention op: {op!r}.")
        time = _finite_float(payload.get("time"), field="time")
        species = _species_name(payload.get("species"))
        if op in {"set"}:
            return cls(time=time, species=species, op=op, value=_finite_float(payload.get("value"), field="value"))
        if op == "clear":
            return cls(time=time, species=species, op=op, value=0.0)
        amount = _nonnegative_float(payload.get("amount"), field="amount")
        return cls(time=time, species=species, op=op, amount=amount)

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "time": float(self.time),
            "species": str(self.species),
            "op": str(self.op),
        }
        if self.op in _ABSOLUTE_INSTANT_OPS:
            payload["value"] = 0.0 if self.op == "clear" else float(self.value if self.value is not None else 0.0)
        else:
            payload["amount"] = float(self.amount if self.amount is not None else 0.0)
        if self.op == "clear":
            payload.pop("value", None)
        return payload


@dataclass(frozen=True, slots=True)
class InterventionInterval:
    start: float
    end: float
    species: str
    kind: str
    rate: float | None = None
    value: float | None = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "InterventionInterval":
        kind = str(payload.get("kind") or payload.get("op") or "").strip().lower()
        if kind not in _INTERVAL_KINDS:
            raise InterventionScheduleError(f"Unsupported intervention interval kind: {kind!r}.")
        start = _finite_float(payload.get("start"), field="start")
        end = _finite_float(payload.get("end"), field="end")
        if end <= start:
            raise InterventionScheduleError("end must be greater than start.")
        species = _species_name(payload.get("species"))
        if kind in {"source", "sink"}:
            return cls(
                start=start,
                end=end,
                species=species,
                kind=kind,
                rate=_nonnegative_float(payload.get("rate"), field="rate"),
            )
        return cls(
            start=start,
            end=end,
            species=species,
            kind=kind,
            value=_finite_float(payload.get("value"), field="value"),
        )

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "start": float(self.start),
            "end": float(self.end),
            "species": str(self.species),
            "kind": str(self.kind),
        }
        if self.kind in {"source", "sink"}:
            payload["rate"] = float(self.rate if self.rate is not None else 0.0)
        else:
            payload["value"] = float(self.value if self.value is not None else 0.0)
        return payload


@dataclass(frozen=True, slots=True)
class InterventionSchedule:
    instant_events: tuple[InterventionInstantEvent, ...] = ()
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
        for item in payload.get("repeated_events") or ():
            if not isinstance(item, Mapping):
                raise InterventionScheduleError("repeated_events entries must be mappings.")
            instant_events.extend(_expand_repeated_event(item))
        intervals: list[InterventionInterval] = []
        for item in payload.get("intervals") or ():
            if not isinstance(item, Mapping):
                raise InterventionScheduleError("intervals entries must be mappings.")
            intervals.append(InterventionInterval.from_payload(item))
        schedule = cls(
            instant_events=tuple(sorted(instant_events, key=_instant_sort_key)),
            intervals=tuple(sorted(intervals, key=_interval_sort_key)),
            version=int(payload.get("version") or 1),
        )
        schedule._validate_conflicts()
        return schedule

    def is_empty(self) -> bool:
        return not self.instant_events and not self.intervals

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": int(self.version),
            "instant_events": [event.to_payload() for event in self.instant_events],
            "intervals": [interval.to_payload() for interval in self.intervals],
        }

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
        for interval in self.intervals:
            if interval.species not in available:
                raise InterventionScheduleError(f"Unknown intervention species: {interval.species}")

    def _validate_conflicts(self) -> None:
        absolute_by_key: dict[tuple[float, str], InterventionInstantEvent] = {}
        for event in self.instant_events:
            if event.op not in _ABSOLUTE_INSTANT_OPS:
                continue
            key = (float(event.time), str(event.species))
            existing = absolute_by_key.get(key)
            if existing is not None and existing.to_payload() != event.to_payload():
                raise InterventionScheduleError(
                    f"Conflicting absolute interventions for {event.species} at t={event.time:g}."
                )
            absolute_by_key[key] = event
        clamp_like = [interval for interval in self.intervals if interval.kind in {"reservoir", "clamp"}]
        for i, left in enumerate(clamp_like):
            for right in clamp_like[i + 1 :]:
                if left.species != right.species:
                    continue
                if left.start < right.end and right.start < left.end:
                    raise InterventionScheduleError(
                        f"Overlapping reservoir/clamp intervals for {left.species} are not allowed."
                    )


def _instant_sort_key(event: InterventionInstantEvent) -> tuple[float, str, int, str]:
    op_order = {"set": 0, "clear": 0, "remove": 1, "add": 2}
    return (float(event.time), str(event.species), op_order.get(event.op, 99), str(event.op))


def _interval_sort_key(interval: InterventionInterval) -> tuple[float, float, str, str]:
    return (float(interval.start), float(interval.end), str(interval.species), str(interval.kind))


def _expand_repeated_event(payload: Mapping[str, Any]) -> list[InterventionInstantEvent]:
    count = int(payload.get("count") or 0)
    if count <= 0:
        raise InterventionScheduleError("count must be greater than zero.")
    start = _finite_float(payload.get("start"), field="start")
    every = _positive_float(payload.get("every"), field="every")
    op = str(payload.get("op") or "add").strip().lower()
    if op == "pulse":
        op = "add"
    events = []
    for idx in range(count):
        event_payload = dict(payload)
        event_payload["time"] = start + every * float(idx)
        event_payload["op"] = op
        events.append(InterventionInstantEvent.from_payload(event_payload))
    return events


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
    boundaries = {float(t0), float(t1)}
    for event in schedule.instant_events:
        if float(t0) <= event.time <= float(t1):
            boundaries.add(float(event.time))
    for interval in schedule.intervals:
        if float(t0) <= interval.start <= float(t1):
            boundaries.add(float(interval.start))
        if float(t0) <= interval.end <= float(t1):
            boundaries.add(float(interval.end))
    return sorted(boundaries)


def events_at_time(schedule: InterventionSchedule, time: float) -> tuple[InterventionInstantEvent, ...]:
    return tuple(event for event in schedule.instant_events if float(event.time) == float(time))


def intervals_active_at(schedule: InterventionSchedule, time: float) -> tuple[InterventionInterval, ...]:
    return tuple(
        interval
        for interval in schedule.intervals
        if float(interval.start) <= float(time) < float(interval.end)
    )
