from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from kindred.core.intervention_schedule import (
    InterventionFlux,
    InterventionHold,
    InterventionJump,
    InterventionProtocol,
    InterventionProtocolOperation,
    InterventionSchedule,
    InterventionScheduleError,
    InterventionSet,
    InterventionTrigger,
    InterventionTriggerAction,
    coerce_intervention_schedule,
)

__all__ = [
    "ExecutableFlux",
    "ExecutableHold",
    "ExecutableInstant",
    "ExecutableTrigger",
    "InterventionExecutionContract",
    "TriggerFiring",
    "compile_intervention_schedule",
    "validate_boundary_conflicts",
]


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _finite_derived(value: object, *, context: str) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise InterventionScheduleError(
            f"{context} must remain finite."
        ) from exc
    if not math.isfinite(number):
        raise InterventionScheduleError(f"{context} must remain finite.")
    return number


def _protocol_boundary(
    start: float, every: float, index: int, *, duration: float | None = None
) -> float:
    try:
        value = start + every * index
        if duration is not None:
            value += duration
    except (OverflowError, ValueError) as exc:
        raise InterventionScheduleError(
            "Protocol boundary arithmetic must remain finite."
        ) from exc
    return _finite_derived(value, context="Protocol boundary")


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(_freeze(item) for item in value)
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_plain(item) for item in value]
    return value


def _frozen_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    frozen = _freeze(value)
    if not isinstance(frozen, Mapping):
        raise TypeError("Expected a mapping.")
    return frozen


@dataclass(frozen=True, slots=True)
class ExecutableInstant:
    op: str
    species: str
    time: float
    declaration_ordinal: int
    value: float | None = None
    delta: float | None = None
    phase: str = "ordinary"
    window_ordinal: int | None = None
    authored_ordinal: int | None = None

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "op": self.op,
            "species": self.species,
            "time": self.time,
            "declaration_ordinal": self.declaration_ordinal,
            "phase": self.phase,
        }
        payload["value" if self.op == "set" else "delta"] = (
            self.value if self.op == "set" else self.delta
        )
        if self.window_ordinal is not None:
            payload["window_ordinal"] = self.window_ordinal
        if self.authored_ordinal is not None:
            payload["authored_ordinal"] = self.authored_ordinal
        return payload


@dataclass(frozen=True, slots=True)
class ExecutableFlux:
    species: str
    start_time: float
    end_time: float
    flux_rate: float
    declaration_ordinal: int
    phase: str = "ordinary"
    window_ordinal: int | None = None
    authored_ordinal: int | None = None

    @property
    def op(self) -> str:
        return "flux"

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "op": "flux",
            "species": self.species,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "flux_rate": self.flux_rate,
            "declaration_ordinal": self.declaration_ordinal,
            "phase": self.phase,
        }
        if self.window_ordinal is not None:
            payload["window_ordinal"] = self.window_ordinal
        if self.authored_ordinal is not None:
            payload["authored_ordinal"] = self.authored_ordinal
        return payload


@dataclass(frozen=True, slots=True)
class ExecutableHold:
    species: str
    start_time: float
    end_time: float
    value: float
    declaration_ordinal: int
    phase: str = "ordinary"
    window_ordinal: int | None = None
    authored_ordinal: int | None = None

    @property
    def op(self) -> str:
        return "hold"

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "op": "hold",
            "species": self.species,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "value": self.value,
            "declaration_ordinal": self.declaration_ordinal,
            "phase": self.phase,
        }
        if self.window_ordinal is not None:
            payload["window_ordinal"] = self.window_ordinal
        if self.authored_ordinal is not None:
            payload["authored_ordinal"] = self.authored_ordinal
        return payload


@dataclass(frozen=True, slots=True)
class ExecutableTrigger:
    declaration_ordinal: int
    watch_species: str
    threshold: float
    direction: str
    max_count: int
    min_interval: float
    then: InterventionTriggerAction

    @property
    def op(self) -> str:
        return "trigger"

    def to_payload(self) -> dict[str, Any]:
        return {
            "op": "trigger",
            "declaration_ordinal": self.declaration_ordinal,
            "watch_species": self.watch_species,
            "threshold": self.threshold,
            "direction": self.direction,
            "max_count": self.max_count,
            "min_interval": self.min_interval,
            "then": self.then.to_payload(),
        }


@dataclass(frozen=True, slots=True)
class TriggerFiring:
    declaration_ordinal: int
    watch_species: str
    threshold: float
    direction: str
    root_time: float
    event_state: Mapping[str, float]
    firing_ordinal: int
    nested_action_payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.root_time)):
            raise InterventionScheduleError("Trigger root time must be finite.")
        if any(
            not math.isfinite(float(value)) for value in self.event_state.values()
        ):
            raise InterventionScheduleError(
                "Trigger event-time state must be finite."
            )
        object.__setattr__(
            self,
            "event_state",
            MappingProxyType(
                {
                    str(name): float(value)
                    for name, value in self.event_state.items()
                }
            ),
        )
        object.__setattr__(
            self,
            "nested_action_payload",
            _frozen_mapping(self.nested_action_payload),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "declaration_ordinal": self.declaration_ordinal,
            "watch_species": self.watch_species,
            "threshold": self.threshold,
            "direction": self.direction,
            "root_time": self.root_time,
            "event_state": dict(self.event_state),
            "firing_ordinal": self.firing_ordinal,
            "nested_action_payload": dict(self.nested_action_payload),
        }


@dataclass(frozen=True, slots=True)
class InterventionExecutionContract:
    declarative_payload: Mapping[str, Any]
    executable_payload: Mapping[str, Any]
    declarative_fingerprint: str
    executable_fingerprint: str
    instants: tuple[ExecutableInstant, ...] = ()
    fluxes: tuple[ExecutableFlux, ...] = ()
    holds: tuple[ExecutableHold, ...] = ()
    triggers: tuple[ExecutableTrigger, ...] = ()
    discontinuities: tuple[float, ...] = ()
    segment_identity: tuple[tuple[float, float], ...] = ()
    protocol_lineage: tuple[Mapping[str, Any], ...] = ()
    trigger_firings: tuple[TriggerFiring, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "declarative_payload", _frozen_mapping(self.declarative_payload)
        )
        object.__setattr__(
            self, "executable_payload", _frozen_mapping(self.executable_payload)
        )
        object.__setattr__(
            self,
            "protocol_lineage",
            tuple(_frozen_mapping(item) for item in self.protocol_lineage),
        )
        object.__setattr__(
            self,
            "discontinuities",
            tuple(float(value) for value in self.discontinuities),
        )
        object.__setattr__(
            self,
            "segment_identity",
            tuple((float(start), float(end)) for start, end in self.segment_identity),
        )

    @property
    def is_empty(self) -> bool:
        return not (
            self.instants or self.fluxes or self.holds or self.triggers
        )

    def with_trigger_firings(
        self, firings: Sequence[TriggerFiring]
    ) -> "InterventionExecutionContract":
        return replace(self, trigger_firings=tuple(firings))

    def discontinuities_for_span(
        self, start_time: float, end_time: float
    ) -> tuple[float, ...]:
        start = _finite_derived(start_time, context="Segment start")
        end = _finite_derived(end_time, context="Segment end")
        if end <= start:
            raise InterventionScheduleError(
                "Segment end must be greater than segment start."
            )
        return tuple(
            value for value in self.discontinuities if start < value < end
        )

    def segments_for_span(
        self, start_time: float, end_time: float
    ) -> tuple[tuple[float, float], ...]:
        start = _finite_derived(start_time, context="Segment start")
        end = _finite_derived(end_time, context="Segment end")
        boundaries = (
            start,
            *self.discontinuities_for_span(start, end),
            end,
        )
        return tuple(zip(boundaries, boundaries[1:]))

    def runtime_discontinuities_for_span(
        self, start_time: float, end_time: float
    ) -> tuple[float, ...]:
        start = _finite_derived(start_time, context="Segment start")
        end = _finite_derived(end_time, context="Segment end")
        static = self.discontinuities_for_span(start, end)
        firing_times = tuple(
            firing.root_time
            for firing in self.trigger_firings
            if start < firing.root_time < end
        )
        return tuple(sorted(set(static + firing_times)))

    def runtime_segments_for_span(
        self, start_time: float, end_time: float
    ) -> tuple[tuple[float, float], ...]:
        start = _finite_derived(start_time, context="Segment start")
        end = _finite_derived(end_time, context="Segment end")
        boundaries = (
            start,
            *self.runtime_discontinuities_for_span(start, end),
            end,
        )
        return tuple(zip(boundaries, boundaries[1:]))

    def provenance_projection(self) -> dict[str, Any]:
        return {
            "declarative_fingerprint": self.declarative_fingerprint,
            "executable_fingerprint": self.executable_fingerprint,
            "declarative": _plain(self.declarative_payload),
            "executable": _plain(self.executable_payload),
            "discontinuities": list(self.discontinuities),
            "segment_identity": [
                {"start_time": start, "end_time": end}
                for start, end in self.segment_identity
            ],
            "protocol_lineage": [
                _plain(item) for item in self.protocol_lineage
            ],
            "trigger_firings": [
                firing.to_payload() for firing in self.trigger_firings
            ],
        }

    def annotation_projection(
        self,
        *,
        execution_start: float | None = None,
        execution_end: float | None = None,
    ) -> tuple[Mapping[str, object], ...]:
        if (execution_start is None) != (execution_end is None):
            raise InterventionScheduleError(
                "Annotation execution span requires both start and end."
            )
        span: tuple[float, float] | None = None
        if execution_start is not None and execution_end is not None:
            start = _finite_derived(
                execution_start,
                context="Annotation execution start",
            )
            end = _finite_derived(
                execution_end,
                context="Annotation execution end",
            )
            if end < start:
                raise InterventionScheduleError(
                    "Annotation execution end must not precede its start."
                )
            span = (start, end)
        annotations: list[Mapping[str, object]] = []
        for instant in self.instants:
            if span is not None and not (
                span[0] <= instant.time <= span[1]
            ):
                continue
            annotations.append(
                MappingProxyType(
                    {
                        "time": instant.time,
                        "kind": (
                            "protocol-" + instant.phase
                            if instant.phase != "ordinary"
                            else "instant"
                        ),
                        "label": f"{instant.op} {instant.species}",
                        "declaration_ordinal": instant.declaration_ordinal,
                        "window_ordinal": instant.window_ordinal,
                        "authored_ordinal": instant.authored_ordinal,
                    }
                )
            )
        for interval in self.fluxes + self.holds:
            visible_start = interval.start_time
            visible_end = interval.end_time
            if span is not None:
                visible_start = max(visible_start, span[0])
                visible_end = min(visible_end, span[1])
                if visible_end <= visible_start:
                    continue
            annotations.append(
                MappingProxyType(
                    {
                        "start": visible_start,
                        "end": visible_end,
                        "declared_start": interval.start_time,
                        "declared_end": interval.end_time,
                        "kind": interval.op,
                        "label": f"{interval.op} {interval.species}",
                        "declaration_ordinal": interval.declaration_ordinal,
                        "window_ordinal": interval.window_ordinal,
                        "authored_ordinal": interval.authored_ordinal,
                    }
                )
            )
        for firing in self.trigger_firings:
            if span is not None and not (
                span[0] <= firing.root_time <= span[1]
            ):
                continue
            nested_action_payload = firing.nested_action_payload
            annotations.append(
                MappingProxyType(
                    {
                        "time": firing.root_time,
                        "kind": "trigger",
                        "label": (
                            f"trigger {firing.watch_species}: "
                            f"{nested_action_payload.get('op', '')} "
                            f"{nested_action_payload.get('species', '')}"
                        ).strip(),
                        "declaration_ordinal": firing.declaration_ordinal,
                        "firing_ordinal": firing.firing_ordinal,
                        "threshold": firing.threshold,
                        "direction": firing.direction,
                        "event_state": dict(firing.event_state),
                        "nested_action_payload": dict(
                            firing.nested_action_payload
                        ),
                    }
                )
            )
        return tuple(annotations)


def _instant_from_direct(
    operation: InterventionSet | InterventionJump,
) -> ExecutableInstant:
    if isinstance(operation, InterventionSet):
        return ExecutableInstant(
            op="set",
            species=operation.species,
            time=float(operation.time),
            value=float(operation.value),
            declaration_ordinal=operation.declaration_ordinal,
        )
    return ExecutableInstant(
        op="jump",
        species=operation.species,
        time=float(operation.time),
        delta=float(operation.delta),
        declaration_ordinal=operation.declaration_ordinal,
    )


def _protocol_instant(
    protocol: InterventionProtocol,
    operation: InterventionProtocolOperation,
    *,
    window_ordinal: int,
    time: float,
) -> ExecutableInstant:
    return ExecutableInstant(
        op=operation.op,
        species=operation.species,
        time=time,
        value=operation.value,
        delta=operation.delta,
        declaration_ordinal=protocol.declaration_ordinal,
        phase=operation.phase,
        window_ordinal=window_ordinal,
        authored_ordinal=operation.authored_ordinal,
    )


def _protocol_interval(
    protocol: InterventionProtocol,
    operation: InterventionProtocolOperation,
    *,
    window_ordinal: int,
    start_time: float,
    end_time: float,
) -> ExecutableFlux | ExecutableHold:
    common = {
        "species": operation.species,
        "start_time": start_time,
        "end_time": end_time,
        "declaration_ordinal": protocol.declaration_ordinal,
        "phase": operation.phase,
        "window_ordinal": window_ordinal,
        "authored_ordinal": operation.authored_ordinal,
    }
    if operation.op == "flux":
        return ExecutableFlux(flux_rate=float(operation.flux_rate), **common)
    return ExecutableHold(value=float(operation.value), **common)


def _lineage(
    protocol: InterventionProtocol,
    operation: InterventionProtocolOperation,
    *,
    window_ordinal: int,
    primitive: ExecutableInstant | ExecutableFlux | ExecutableHold,
) -> dict[str, Any]:
    return {
        "declaration_ordinal": protocol.declaration_ordinal,
        "window_ordinal": window_ordinal,
        "phase": operation.phase,
        "authored_ordinal": operation.authored_ordinal,
        "nested_payload": operation.to_payload(),
        "lowered_primitive": primitive.to_payload(),
        "actual_time_or_window": (
            {"time": primitive.time}
            if isinstance(primitive, ExecutableInstant)
            else {
                "start_time": primitive.start_time,
                "end_time": primitive.end_time,
            }
        ),
    }


def _execution_order(item: ExecutableInstant) -> tuple[float, int, int, int, int]:
    phase_order = {"ordinary": 0, "before": 1, "after": 2}
    return (
        item.time,
        item.declaration_ordinal,
        phase_order[item.phase],
        -1 if item.window_ordinal is None else item.window_ordinal,
        -1 if item.authored_ordinal is None else item.authored_ordinal,
    )


def compile_intervention_schedule(
    schedule: InterventionSchedule | Mapping[str, Any] | None,
) -> InterventionExecutionContract:
    schedule_obj = coerce_intervention_schedule(schedule) or InterventionSchedule()
    if schedule_obj.is_parameterized:
        raise InterventionScheduleError(
            "Parameterized intervention schedule must be resolved before compilation."
        )
    instants: list[ExecutableInstant] = []
    fluxes: list[ExecutableFlux] = []
    holds: list[ExecutableHold] = []
    triggers: list[ExecutableTrigger] = []
    lineage: list[dict[str, Any]] = []
    for operation in schedule_obj.operations:
        if isinstance(operation, (InterventionSet, InterventionJump)):
            instants.append(_instant_from_direct(operation))
        elif isinstance(operation, InterventionFlux):
            fluxes.append(
                ExecutableFlux(
                    species=operation.species,
                    start_time=float(operation.start_time),
                    end_time=float(operation.end_time),
                    flux_rate=float(operation.flux_rate),
                    declaration_ordinal=operation.declaration_ordinal,
                )
            )
        elif isinstance(operation, InterventionHold):
            holds.append(
                ExecutableHold(
                    species=operation.species,
                    start_time=float(operation.start_time),
                    end_time=float(operation.end_time),
                    value=float(operation.value),
                    declaration_ordinal=operation.declaration_ordinal,
                )
            )
        elif isinstance(operation, InterventionTrigger):
            triggers.append(
                ExecutableTrigger(
                    declaration_ordinal=operation.declaration_ordinal,
                    watch_species=operation.watch_species,
                    threshold=float(operation.threshold),
                    direction=operation.direction,
                    max_count=operation.max_count,
                    min_interval=operation.min_interval,
                    then=operation.then,
                )
            )
        elif isinstance(operation, InterventionProtocol):
            start = float(operation.start_time)
            duration = operation.duration
            _protocol_boundary(
                start,
                operation.every,
                operation.count - 1,
                duration=duration,
            )
            previous_window_start: float | None = None
            for window_ordinal in range(operation.count):
                window_start = _protocol_boundary(
                    start, operation.every, window_ordinal
                )
                if (
                    previous_window_start is not None
                    and window_start <= previous_window_start
                ):
                    raise InterventionScheduleError(
                        "Protocol windows must advance exactly in binary64."
                    )
                previous_window_start = window_start
                window_end: float | None = None
                if duration is not None:
                    window_end = _protocol_boundary(
                        start,
                        operation.every,
                        window_ordinal,
                        duration=duration,
                    )
                    if window_end <= window_start:
                        raise InterventionScheduleError(
                            "Protocol window duration must remain positive "
                            "after binary64 expansion."
                        )
                for nested in operation.before:
                    primitive = _protocol_instant(
                        operation,
                        nested,
                        window_ordinal=window_ordinal,
                        time=window_start,
                    )
                    instants.append(primitive)
                    lineage.append(
                        _lineage(
                            operation,
                            nested,
                            window_ordinal=window_ordinal,
                            primitive=primitive,
                        )
                    )
                for nested in operation.during:
                    if window_end is None:
                        raise InterventionScheduleError(
                            "Protocol during requires a finite duration."
                        )
                    primitive = _protocol_interval(
                        operation,
                        nested,
                        window_ordinal=window_ordinal,
                        start_time=window_start,
                        end_time=window_end,
                    )
                    if isinstance(primitive, ExecutableFlux):
                        fluxes.append(primitive)
                    else:
                        holds.append(primitive)
                    lineage.append(
                        _lineage(
                            operation,
                            nested,
                            window_ordinal=window_ordinal,
                            primitive=primitive,
                        )
                    )
                for nested in operation.after:
                    if window_end is None:
                        raise InterventionScheduleError(
                            "Protocol after requires a finite duration."
                        )
                    primitive = _protocol_instant(
                        operation,
                        nested,
                        window_ordinal=window_ordinal,
                        time=window_end,
                    )
                    instants.append(primitive)
                    lineage.append(
                        _lineage(
                            operation,
                            nested,
                            window_ordinal=window_ordinal,
                            primitive=primitive,
                        )
                    )
    instants.sort(key=_execution_order)
    fluxes.sort(
        key=lambda item: (
            item.start_time,
            item.declaration_ordinal,
            -1 if item.window_ordinal is None else item.window_ordinal,
            -1 if item.authored_ordinal is None else item.authored_ordinal,
        )
    )
    holds.sort(
        key=lambda item: (
            item.start_time,
            item.declaration_ordinal,
            -1 if item.window_ordinal is None else item.window_ordinal,
            -1 if item.authored_ordinal is None else item.authored_ordinal,
        )
    )
    triggers.sort(key=lambda item: item.declaration_ordinal)
    validate_boundary_conflicts(instants=instants, entering_holds=holds)
    _validate_hold_overlaps(holds)
    executable_operations = [
        *(item.to_payload() for item in instants),
        *(item.to_payload() for item in fluxes),
        *(item.to_payload() for item in holds),
        *(item.to_payload() for item in triggers),
    ]
    executable_operations.sort(
        key=lambda item: (
            int(item["declaration_ordinal"]),
            int(item.get("window_ordinal", -1)),
            {"ordinary": 0, "before": 1, "during": 2, "after": 3}.get(
                str(item.get("phase", "ordinary")), 4
            ),
            int(item.get("authored_ordinal", -1)),
        )
    )
    executable_payload = {"operations": executable_operations}
    discontinuities = sorted(
        {
            *(item.time for item in instants),
            *(item.start_time for item in fluxes),
            *(item.end_time for item in fluxes),
            *(item.start_time for item in holds),
            *(item.end_time for item in holds),
        }
    )
    if any(not math.isfinite(value) for value in discontinuities):
        raise InterventionScheduleError(
            "Executable discontinuities must remain finite."
        )
    segments = tuple(zip(discontinuities, discontinuities[1:]))
    declarative_payload = schedule_obj.to_payload()
    return InterventionExecutionContract(
        declarative_payload=declarative_payload,
        executable_payload=executable_payload,
        declarative_fingerprint=schedule_obj.fingerprint,
        executable_fingerprint=(
            "" if not executable_operations else _fingerprint(executable_payload)
        ),
        instants=tuple(instants),
        fluxes=tuple(fluxes),
        holds=tuple(holds),
        triggers=tuple(triggers),
        discontinuities=tuple(discontinuities),
        segment_identity=segments,
        protocol_lineage=tuple(lineage),
    )


def validate_boundary_conflicts(
    *,
    instants: Sequence[ExecutableInstant],
    entering_holds: Sequence[ExecutableHold],
) -> None:
    by_key: dict[tuple[float, str], list[ExecutableInstant | ExecutableHold]] = {}
    for instant in instants:
        by_key.setdefault((instant.time, instant.species), []).append(instant)
    for hold in entering_holds:
        by_key.setdefault((hold.start_time, hold.species), []).append(hold)
    for (time, species), candidates in by_key.items():
        sets = [
            item
            for item in candidates
            if isinstance(item, ExecutableInstant) and item.op == "set"
        ]
        jumps = [
            item
            for item in candidates
            if isinstance(item, ExecutableInstant) and item.op == "jump"
        ]
        holds = [item for item in candidates if isinstance(item, ExecutableHold)]
        if sets and jumps:
            raise InterventionScheduleError(
                f"set and jump conflict for {species} at t={time:g}."
            )
        absolute_values = [
            *(float(item.value) for item in sets),
            *(float(item.value) for item in holds),
        ]
        if absolute_values and any(
            value != absolute_values[0] for value in absolute_values[1:]
        ):
            raise InterventionScheduleError(
                f"Incompatible absolute/hold conflict for {species} at t={time:g}."
            )
        if holds and jumps:
            raise InterventionScheduleError(
                f"hold start and jump conflict for {species} at t={time:g}."
            )


def _validate_hold_overlaps(holds: Sequence[ExecutableHold]) -> None:
    for index, left in enumerate(holds):
        for right in holds[index + 1 :]:
            if left.species != right.species:
                continue
            if left.start_time < right.end_time and right.start_time < left.end_time:
                if left.value != right.value:
                    raise InterventionScheduleError(
                        "Overlapping holds for "
                        f"{left.species} must have equal resolved values."
                    )
