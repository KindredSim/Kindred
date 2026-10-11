from __future__ import annotations

import math
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

from kindred.core.exceptions import SimulationCancelled
from kindred.core.species import validate_time_seconds
from kindred.core.intervention_schedule import (
    InterventionFlux,
    InterventionPhysical,
    PhysicalAction,
    PhysicalQuantity,
    InterventionHold,
    InterventionJump,
    InterventionProtocol,
    InterventionProtocolOperation,
    InterventionSchedule,
    InterventionScheduleDependencyMetadata,
    InterventionScheduleFieldIdentity,
    InterventionScheduleError,
    InterventionSet,
    InterventionTrigger,
    InterventionTriggerAction,
    InterventionStopAction,
    SpeciesTriggerCondition,
    ObservableTriggerCondition,
    intervention_schedule_parameter_dependencies,
    _resolved,
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


def resolve_executable_values(item, values: Mapping[str, float], *, time: float):
    """Resolve a retained live payload at its scientific consumption point."""
    updates = {}
    for field in ("value", "delta", "flux_rate", "threshold"):
        parameter = getattr(item, f"{field}_param", None)
        if parameter is None:
            continue
        domain = (
            "concentration"
            if field == "value"
            or (
                field == "threshold"
                and isinstance(item.condition, SpeciesTriggerCondition)
            )
            else "signed"
        )
        try:
            updates[field] = _resolved(
                None, parameter, values, field=field, domain=domain
            )
        except (ValueError, ArithmeticError) as exc:
            raise InterventionScheduleError(
                str(exc),
                declaration_ordinal=item.declaration_ordinal,
                field=field,
                value_dependencies=(parameter,),
                details={"time": time},
            ) from exc
        updates[f"{field}_param"] = None
    return replace(item, **updates) if updates else item


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


def _raise_if_cancelled(
    cancellation_check: Callable[[], bool] | None,
) -> None:
    if cancellation_check is not None and bool(cancellation_check()):
        raise SimulationCancelled()


def _protocol_boundary(
    start: float,
    every: float,
    index: int,
    *,
    duration: float | None = None,
    operation: InterventionProtocol,
    dependency_metadata: InterventionScheduleDependencyMetadata,
    field: str = "start_time",
    fields: Sequence[str] = (),
) -> float:
    try:
        if duration == every:
            # Adjacent windows share exactly the same binary64 boundary.
            value = start + every * (index + 1)
        else:
            value = start + every * index
            if duration is not None:
                value += duration
    except (OverflowError, ValueError) as exc:
        raise _annotated_schedule_error(
            "Protocol boundary arithmetic must remain finite.",
            operation=operation,
            dependency_metadata=dependency_metadata,
            field=field,
            fields=fields,
            domain="finite",
        ) from exc
    try:
        return validate_time_seconds(value, field="Protocol boundary")
    except (TypeError, ValueError) as exc:
        raise _annotated_schedule_error(
            str(exc),
            operation=operation,
            dependency_metadata=dependency_metadata,
            field=field,
            fields=fields,
            domain="finite",
        ) from exc


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


@dataclass(frozen=True, slots=True)
class ExecutableInstant:
    op: str
    species: str
    time: float
    declaration_ordinal: int
    value: float | None = None
    delta: float | None = None
    value_param: str | None = None
    delta_param: str | None = None
    phase: str = "ordinary"
    window_ordinal: int | None = None
    authored_ordinal: int | None = None
    physical: PhysicalAction | None = None

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
        for field in ("value", "delta", "flux_rate"):
            parameter = getattr(self, f"{field}_param", None)
            if parameter is not None:
                payload.pop(field, None)
                payload[f"{field}_param"] = parameter
        if self.physical is not None:
            for field in ("species", "value", "delta", "flux_rate"):
                payload.pop(field, None)
            payload.update(self.physical.to_payload())
        return payload


@dataclass(frozen=True, slots=True)
class ExecutableFlux:
    species: str
    start_time: float
    end_time: float
    flux_rate: float | None
    declaration_ordinal: int
    flux_rate_param: str | None = None
    phase: str = "ordinary"
    window_ordinal: int | None = None
    authored_ordinal: int | None = None
    physical: PhysicalAction | None = None

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
        for field in ("value", "delta", "flux_rate"):
            parameter = getattr(self, f"{field}_param", None)
            if parameter is not None:
                payload.pop(field, None)
                payload[f"{field}_param"] = parameter
        if self.physical is not None:
            for field in ("species", "value", "delta", "flux_rate"):
                payload.pop(field, None)
            payload.update(self.physical.to_payload())
        return payload


@dataclass(frozen=True, slots=True)
class ExecutableHold:
    species: str
    start_time: float
    end_time: float
    value: float | None
    declaration_ordinal: int
    value_param: str | None = None
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
        for field in ("value", "delta", "flux_rate"):
            parameter = getattr(self, f"{field}_param", None)
            if parameter is not None:
                payload.pop(field, None)
                payload[f"{field}_param"] = parameter
        return payload


@dataclass(frozen=True, slots=True)
class ExecutableTrigger:
    declaration_ordinal: int
    condition: SpeciesTriggerCondition | ObservableTriggerCondition
    threshold: float | None
    direction: str
    max_count: int | None
    min_interval: float | None
    then: InterventionTriggerAction | InterventionStopAction
    source_text: str | None = None
    threshold_param: str | None = None

    @property
    def op(self) -> str:
        return "trigger"

    def to_payload(self) -> dict[str, Any]:
        return {
            "op": "trigger",
            "declaration_ordinal": self.declaration_ordinal,
            **self.condition.to_payload(),
            **({"threshold_param": self.threshold_param} if self.threshold_param else {"threshold": self.threshold}),
            "direction": self.direction,
            **({"max_count": self.max_count, "min_interval": self.min_interval} if not isinstance(self.then, InterventionStopAction) else {}),
            "then": self.then.to_payload(),
        }


@dataclass(frozen=True, slots=True)
class TriggerFiring:
    declaration_ordinal: int
    condition: SpeciesTriggerCondition | ObservableTriggerCondition
    threshold: float
    direction: str
    root_time: float
    event_state: Mapping[str, float]
    firing_ordinal: int
    nested_action_payload: Mapping[str, Any]

    def __post_init__(self) -> None:
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
            _freeze(self.nested_action_payload),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "declaration_ordinal": self.declaration_ordinal,
            **self.condition.to_payload(),
            "threshold": self.threshold,
            "direction": self.direction,
            "root_time": self.root_time,
            "event_state": dict(self.event_state),
            "firing_ordinal": self.firing_ordinal,
            "nested_action_payload": dict(self.nested_action_payload),
        }


@dataclass(frozen=True, slots=True)
class InterventionExecutionContract:
    # The authored schedule, kept for provenance and for pairing executed
    # amounts with their authored parameter names.
    authored: InterventionSchedule
    initial_volume: tuple[PhysicalQuantity, InterventionScheduleFieldIdentity] | None = None
    instants: tuple[ExecutableInstant, ...] = ()
    fluxes: tuple[ExecutableFlux, ...] = ()
    holds: tuple[ExecutableHold, ...] = ()
    triggers: tuple[ExecutableTrigger, ...] = ()
    discontinuities: tuple[float, ...] = ()
    segment_identity: tuple[tuple[float, float], ...] = ()
    protocol_lineage: tuple[Mapping[str, Any], ...] = ()
    trigger_firings: tuple[TriggerFiring, ...] = ()
    # Scheduled instants left unexecuted at a stopping boundary.
    skipped_instants: tuple[ExecutableInstant, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "protocol_lineage",
            tuple(_freeze(item) for item in self.protocol_lineage),
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

    def without_stops(self) -> InterventionExecutionContract:
        retained = tuple(item for item in self.triggers if not isinstance(item.then, InterventionStopAction))
        if len(retained) == len(self.triggers):
            return self
        return replace(self, triggers=retained, authored=self.authored.without_stops())

    def provenance_projection(self) -> dict[str, Any]:
        executable_operations = [
            *(item.to_payload() for item in self.instants),
            *(item.to_payload() for item in self.fluxes),
            *(item.to_payload() for item in self.holds),
            *(item.to_payload() for item in self.triggers),
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
        return {
            "declarative": _plain(self.authored.to_payload()),
            "executable": {"operations": executable_operations},
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
            "skipped_instants": [
                instant.to_payload() for instant in self.skipped_instants
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
            if instant in self.skipped_instants:
                continue
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
                        "physical_action": {"op": instant.op, **instant.physical.to_payload()} if instant.physical is not None else None,
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
                        "physical_action": {"op": interval.op, **interval.physical.to_payload()} if isinstance(interval, ExecutableFlux) and interval.physical is not None else None,
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
                            f"trigger {firing.condition.name}: "
                            + (
                                f"{nested_action_payload.get('op', '')} "
                                f"{nested_action_payload.get('species', '')}"
                            )
                        ).strip(),
                        "physical_action": nested_action_payload if "solution" in nested_action_payload or nested_action_payload["op"] in {"scale", "withdraw"} else None,
                        "label_prefix": f"trigger {firing.condition.name}: ",
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
    operation: InterventionSet | InterventionJump | InterventionPhysical,
) -> ExecutableInstant:
    if isinstance(operation, InterventionPhysical):
        return ExecutableInstant(op=operation.op, species="", time=float(operation.time),
                                 declaration_ordinal=operation.declaration_ordinal, physical=operation.physical)
    if isinstance(operation, InterventionSet):
        return ExecutableInstant(
            op="set",
            species=operation.species,
            time=float(operation.time),
            value=operation.value, value_param=operation.value_param,
            declaration_ordinal=operation.declaration_ordinal,
        )
    return ExecutableInstant(
        op="jump",
        species=operation.species,
        time=float(operation.time),
        delta=operation.delta, delta_param=operation.delta_param,
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
        physical=operation.physical,
        species=operation.species,
        time=time,
        value=operation.value, value_param=operation.value_param,
        delta=operation.delta, delta_param=operation.delta_param,
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
        return ExecutableFlux(flux_rate=operation.flux_rate, flux_rate_param=operation.flux_rate_param, physical=operation.physical, **common)
    return ExecutableHold(value=operation.value, value_param=operation.value_param, **common)


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


def _operation_field_identities(
    operation: InterventionProtocol | ExecutableInstant | ExecutableHold,
    *,
    fields: Sequence[str] = (),
    field: str | None = None,
) -> tuple[InterventionScheduleFieldIdentity, ...]:
    declaration = operation.declaration_ordinal
    requested = tuple(fields) or (
        () if field is None else (field,)
    )
    if isinstance(operation, InterventionProtocol):
        phase, nested_ordinal = "ordinary", None
    else:
        phase, nested_ordinal = operation.phase, operation.authored_ordinal
    identities: list[InterventionScheduleFieldIdentity] = []
    for requested_field in requested:
        if nested_ordinal is not None and phase != "ordinary":
            if requested_field in {"time", "start_time", "end_time"}:
                identities.append((declaration, None, "start_time"))
                if phase in {"during", "after"}:
                    identities.append((declaration, None, "duration"))
            else:
                identities.append(
                    (declaration, nested_ordinal, f"{phase}.{requested_field}")
                )
        else:
            identities.append((declaration, None, requested_field))
    return tuple(dict.fromkeys(identities))


def _annotated_schedule_error(
    message: str,
    *,
    operation: InterventionProtocol | None,
    dependency_metadata: InterventionScheduleDependencyMetadata,
    field: str | None = None,
    fields: Sequence[str] = (),
    operations: Sequence[ExecutableInstant | ExecutableHold] = (),
    domain: str | None = None,
) -> InterventionScheduleError:
    if operation is not None:
        identities = _operation_field_identities(operation, fields=fields, field=field)
        return InterventionScheduleError(
            message,
            operation=operation.op,
            declaration_ordinal=operation.declaration_ordinal,
            field=field,
            domain=domain,
            field_identities=identities,
            value_dependencies=tuple(
                name for identity in identities
                for name in dependency_metadata.get(identity, ())
            ),
        )
    declarations: list[int] = []
    operation_names: list[str] = []
    operation_locations: list[dict[str, object]] = []
    field_identities: list[InterventionScheduleFieldIdentity] = []
    for item in operations:
        declarations.append(item.declaration_ordinal)
        operation_names.append(item.op)
        location: dict[str, object] = {
            "operation": item.op,
            "declaration_ordinal": item.declaration_ordinal,
        }
        if item.authored_ordinal is not None:
            location["phase"] = item.phase
            location["authored_ordinal"] = item.authored_ordinal
        operation_locations.append(location)
        field_identities.extend(
            _operation_field_identities(item, fields=fields, field=field)
        )
    identities = tuple(dict.fromkeys(field_identities))
    details: dict[str, object] = {}
    if declarations:
        details["declaration_ordinals"] = tuple(sorted(set(declarations)))
    if operation_names:
        details["operations"] = tuple(operation_names)
        details["operation_locations"] = tuple(operation_locations)
    return InterventionScheduleError(
        message,
        field=field,
        field_identities=identities,
        value_dependencies=tuple(
            name for identity in identities
            for name in dependency_metadata.get(identity, ())
        ),
        details=details,
    )


def compile_intervention_schedule(
    schedule: InterventionSchedule | None,
    *,
    cancellation_check: Callable[[], bool] | None = None,
    dependency_metadata: InterventionScheduleDependencyMetadata | None = None,
    authored: InterventionSchedule | None = None,
) -> InterventionExecutionContract:
    _raise_if_cancelled(cancellation_check)
    schedule_obj = schedule or InterventionSchedule()
    dependency_metadata = dependency_metadata or intervention_schedule_parameter_dependencies(
        schedule_obj
    )
    # Only resolved schedules are compiled.
    instants: list[ExecutableInstant] = []
    fluxes: list[ExecutableFlux] = []
    holds: list[ExecutableHold] = []
    triggers: list[ExecutableTrigger] = []
    lineage: list[dict[str, Any]] = []
    for operation in schedule_obj.operations:
        _raise_if_cancelled(cancellation_check)
        if isinstance(operation, InterventionPhysical):
            if operation.op == "flux":
                fluxes.append(ExecutableFlux(species="", start_time=float(operation.start_time),
                                             end_time=float(operation.end_time), flux_rate=None,
                                             declaration_ordinal=operation.declaration_ordinal, physical=operation.physical))
            else:
                instants.append(_instant_from_direct(operation))
        elif isinstance(operation, (InterventionSet, InterventionJump)):
            instants.append(_instant_from_direct(operation))
        elif isinstance(operation, InterventionFlux):
            fluxes.append(
                ExecutableFlux(
                    species=operation.species,
                    start_time=float(operation.start_time),
                    end_time=float(operation.end_time),
                    flux_rate=operation.flux_rate, flux_rate_param=operation.flux_rate_param,
                    declaration_ordinal=operation.declaration_ordinal,
                )
            )
        elif isinstance(operation, InterventionHold):
            holds.append(
                ExecutableHold(
                    species=operation.species,
                    start_time=float(operation.start_time),
                    end_time=float(operation.end_time),
                    value=operation.value, value_param=operation.value_param,
                    declaration_ordinal=operation.declaration_ordinal,
                )
            )
        elif isinstance(operation, InterventionTrigger):
            triggers.append(
                ExecutableTrigger(
                    declaration_ordinal=operation.declaration_ordinal,
                    condition=operation.condition,
                    source_text=operation.source_text,
                    threshold=operation.threshold, threshold_param=operation.threshold_param,
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
                operation=operation,
                field="every",
                fields=("start_time", "every", "duration"),
                dependency_metadata=dependency_metadata,
            )
            previous_window_start: float | None = None
            for window_ordinal in range(operation.count):
                _raise_if_cancelled(cancellation_check)
                window_start = _protocol_boundary(
                    start,
                    operation.every,
                    window_ordinal,
                    operation=operation,
                    field="start_time",
                    fields=("start_time", "every"),
                    dependency_metadata=dependency_metadata,
                )
                if (
                    previous_window_start is not None
                    and window_start <= previous_window_start
                ):
                    raise _annotated_schedule_error(
                        "Protocol windows must advance exactly in binary64.",
                        operation=operation,
                        dependency_metadata=dependency_metadata,
                        field="every",
                        fields=("start_time", "every"),
                    )
                previous_window_start = window_start
                window_end: float | None = None
                if duration is not None:
                    window_end = _protocol_boundary(
                        start,
                        operation.every,
                        window_ordinal,
                        duration=duration,
                        operation=operation,
                        field="duration",
                        fields=("start_time", "every", "duration"),
                        dependency_metadata=dependency_metadata,
                    )
                    if window_end <= window_start:
                        raise _annotated_schedule_error(
                            "Protocol window duration must remain positive "
                            "after binary64 expansion.",
                            operation=operation,
                            dependency_metadata=dependency_metadata,
                            field="duration",
                            fields=("start_time", "every", "duration"),
                        )
                for nested in operation.before:
                    _raise_if_cancelled(cancellation_check)
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
                    _raise_if_cancelled(cancellation_check)
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
                    _raise_if_cancelled(cancellation_check)
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
    _validate_hold_overlaps(
        holds,
        cancellation_check=cancellation_check,
        dependency_metadata=dependency_metadata,
    )
    validate_boundary_conflicts(
        instants=instants,
        entering_holds=holds,
        cancellation_check=cancellation_check,
        dependency_metadata=dependency_metadata,
    )
    discontinuities = sorted(
        {
            *(item.time for item in instants),
            *(item.start_time for item in fluxes),
            *(item.end_time for item in fluxes),
            *(item.start_time for item in holds),
            *(item.end_time for item in holds),
        }
    )
    segments = tuple(zip(discontinuities, discontinuities[1:]))
    return InterventionExecutionContract(
        authored=schedule_obj if authored is None else authored,
        initial_volume=schedule_obj.initial_volume,
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
    cancellation_check: Callable[[], bool] | None = None,
    dependency_metadata: InterventionScheduleDependencyMetadata | None = None,
) -> None:
    dependency_metadata = dependency_metadata or {}
    by_key: dict[tuple[float, str], list[ExecutableInstant | ExecutableHold]] = {}
    first_set_by_time: dict[float, ExecutableInstant] = {}
    for instant in instants:
        _raise_if_cancelled(cancellation_check)
        if instant.op == "set":
            first_set_by_time.setdefault(instant.time, instant)
    for physical in instants:
        _raise_if_cancelled(cancellation_check)
        if physical.physical is not None and physical.op != "withdraw":
            absolute = first_set_by_time.get(physical.time)
            if absolute is not None:
                raise InterventionScheduleError(
                    f"set and {physical.op} conflict for {absolute.species} at t={physical.time:g}.",
                    declaration_ordinal=max(absolute.declaration_ordinal, physical.declaration_ordinal),
                )
    for instant in instants:
        _raise_if_cancelled(cancellation_check)
        if instant.physical is None:
            by_key.setdefault((instant.time, instant.species), []).append(instant)
    for hold in entering_holds:
        _raise_if_cancelled(cancellation_check)
        by_key.setdefault((hold.start_time, hold.species), []).append(hold)
    for (time, species), candidates in by_key.items():
        _raise_if_cancelled(cancellation_check)
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
            raise _annotated_schedule_error(
                f"set and jump conflict for {species} at t={time:g}.",
                operation=None,
                dependency_metadata=dependency_metadata,
                field="time",
                fields=("time", "value", "delta", "start_time", "end_time"),
                operations=tuple((*sets, *jumps)),
            )
        absolute_values = [
            *(float(item.value) for item in sets if item.value_param is None),
            *(float(item.value) for item in holds if item.value_param is None),
        ]
        if absolute_values and any(
            value != absolute_values[0] for value in absolute_values[1:]
        ):
            raise _annotated_schedule_error(
                f"Declarations set {species} to different values at t={time:g}.",
                operation=None,
                dependency_metadata=dependency_metadata,
                field="value",
                fields=("time", "value", "delta", "start_time", "end_time"),
                operations=tuple((*sets, *holds)),
            )
        if holds and jumps:
            raise _annotated_schedule_error(
                f"hold start and jump conflict for {species} at t={time:g}.",
                operation=None,
                dependency_metadata=dependency_metadata,
                field="time",
                fields=("time", "value", "delta", "start_time", "end_time"),
                operations=tuple((*holds, *jumps)),
            )


def _validate_hold_overlaps(
    holds: Sequence[ExecutableHold],
    *,
    cancellation_check: Callable[[], bool] | None = None,
    dependency_metadata: InterventionScheduleDependencyMetadata | None = None,
) -> None:
    dependency_metadata = dependency_metadata or {}
    next_by_species: dict[str, ExecutableHold] = {}
    conflict: tuple[ExecutableHold, ExecutableHold] | None = None
    # Compilation orders positive intervals by start time. The next hold of
    # the same species is the first possible conflict for each left interval.
    # Walking backwards preserves the earliest conflicting pair from the
    # nested scan, including when several species' intervals are interleaved.
    for left in reversed(holds):
        _raise_if_cancelled(cancellation_check)
        right = next_by_species.get(left.species)
        if right is not None and right.start_time < left.end_time:
            conflict = (left, right)
        next_by_species[left.species] = left
    if conflict is not None:
        left, right = conflict
        raise _annotated_schedule_error(
            f"Overlapping holds for {left.species}: "
            f"declaration {left.declaration_ordinal + 1} "
            f"[{left.start_time!r}, {left.end_time!r}) and "
            f"declaration {right.declaration_ordinal + 1} "
            f"[{right.start_time!r}, {right.end_time!r}). "
            "Only one hold per species may be active at a time.",
            operation=None,
            dependency_metadata=dependency_metadata,
            fields=("start_time", "end_time"),
            operations=(left, right),
        )
