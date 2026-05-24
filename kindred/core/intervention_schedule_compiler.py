from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from kindred.core.intervention_schedule import (
    InterventionInstantEvent,
    InterventionInterval,
    InterventionProtocol,
    InterventionProtocolOperation,
    InterventionRepeatedInterval,
    InterventionSchedule,
    InterventionScheduleError,
    coerce_intervention_schedule,
)


@dataclass(frozen=True, slots=True)
class CompiledInterventionSchedule:
    executable_schedule: InterventionSchedule
    normalized_declarative_payload: Mapping[str, Any]
    executable_payload: Mapping[str, Any]
    declarative_fingerprint: str
    executable_fingerprint: str
    lineage: tuple[Mapping[str, Any], ...] = ()
    provenance: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "normalized_declarative_payload", _copy_mapping(self.normalized_declarative_payload))
        object.__setattr__(self, "executable_payload", _copy_mapping(self.executable_payload))
        object.__setattr__(self, "lineage", tuple(_copy_mapping(item) for item in self.lineage))
        provenance = self.provenance
        if provenance is None:
            provenance = {}
        object.__setattr__(self, "provenance", _copy_mapping(provenance))


def compile_intervention_schedule(
    schedule: InterventionSchedule | Mapping[str, Any] | None,
) -> CompiledInterventionSchedule:
    schedule_obj = coerce_intervention_schedule(schedule) or InterventionSchedule()
    if schedule_obj.is_parameterized:
        raise InterventionScheduleError("Parameterized intervention schedule must be resolved before compilation.")
    if not schedule_obj.repeated_intervals and not schedule_obj.protocols and not _schedule_has_metadata(schedule_obj):
        payload = schedule_obj.to_payload()
        return CompiledInterventionSchedule(
            executable_schedule=schedule_obj,
            normalized_declarative_payload=payload,
            executable_payload=payload,
            declarative_fingerprint=schedule_obj.fingerprint,
            executable_fingerprint=schedule_obj.fingerprint,
        )

    instant_payloads = [_strip_metadata(event.to_payload()) for event in schedule_obj.instant_events]
    repeated_payloads = [_strip_metadata(event.to_payload()) for event in schedule_obj.repeated_events]
    trigger_payloads = [_strip_metadata(event.to_payload()) for event in schedule_obj.trigger_events]
    interval_payloads = [_strip_metadata(interval.to_payload()) for interval in schedule_obj.intervals]
    instant_sources: list[tuple[InterventionInstantEvent, str]] = [
        (event, "primitive instant event") for event in schedule_obj.instant_events
    ]
    for repeated_index, repeated_event in enumerate(schedule_obj.repeated_events):
        for window_index, event in enumerate(repeated_event.expand()):
            instant_sources.append((event, f"primitive repeated event {repeated_index} window {window_index}"))
    lineage: list[dict[str, Any]] = []

    for declaration_index, repeated_interval in enumerate(schedule_obj.repeated_intervals):
        for window_index, interval in enumerate(repeated_interval.expand()):
            interval_payloads.append(_strip_metadata(interval.to_payload()))
            lineage.append(
                _lineage_entry_for_repeated_interval(
                    repeated_interval,
                    declaration_index=declaration_index,
                    window_index=window_index,
                    interval=interval,
                )
            )

    for protocol in schedule_obj.protocols:
        _extend_protocol_payloads(
            protocol,
            instant_payloads=instant_payloads,
            instant_sources=instant_sources,
            interval_payloads=interval_payloads,
            lineage=lineage,
        )
    _validate_compiled_instant_conflicts(instant_sources)

    executable_schedule = InterventionSchedule.from_payload(
        {
            "instant_events": instant_payloads,
            "repeated_events": repeated_payloads,
            "trigger_events": trigger_payloads,
            "intervals": interval_payloads,
        }
    )
    declarative_payload = schedule_obj.to_payload()
    executable_payload = executable_schedule.to_payload()
    provenance: dict[str, Any] = {
        "declarative_payload": declarative_payload,
        "executable_payload": executable_payload,
        "declarative_fingerprint": schedule_obj.fingerprint,
        "executable_fingerprint": executable_schedule.fingerprint,
    }
    if schedule_obj.metadata:
        provenance["metadata"] = _metadata_dict(schedule_obj.metadata)
    primitive_metadata = _primitive_metadata_entries(schedule_obj)
    if primitive_metadata:
        provenance["primitive_metadata"] = primitive_metadata
    if lineage:
        provenance["lineage"] = [_copy_mapping(item) for item in lineage]
    if _metadata_mentions_display_unit(schedule_obj.metadata) or _lineage_mentions_display_unit(lineage) or any(
        _metadata_mentions_display_unit(entry["metadata"]) for entry in primitive_metadata
    ):
        provenance["metadata_uses_internal_numeric_values"] = True

    return CompiledInterventionSchedule(
        executable_schedule=executable_schedule,
        normalized_declarative_payload=declarative_payload,
        executable_payload=executable_payload,
        declarative_fingerprint=schedule_obj.fingerprint,
        executable_fingerprint=executable_schedule.fingerprint,
        lineage=tuple(lineage),
        provenance=provenance,
    )


def _extend_protocol_payloads(
    protocol: InterventionProtocol,
    *,
    instant_payloads: list[dict[str, Any]],
    instant_sources: list[tuple[InterventionInstantEvent, str]],
    interval_payloads: list[dict[str, Any]],
    lineage: list[dict[str, Any]],
) -> None:
    start = float(protocol.start if protocol.start is not None else 0.0)
    every = float(protocol.every if protocol.every is not None else 0.0)
    duration = float(protocol.duration if protocol.duration is not None else 0.0)
    if every <= 0.0:
        raise InterventionScheduleError("every must be greater than zero.")
    if duration <= 0.0:
        raise InterventionScheduleError("duration must be greater than zero.")

    windows = tuple(
        (
            window_index,
            start + every * float(window_index),
            start + every * float(window_index) + duration,
        )
        for window_index in range(int(protocol.count))
    )
    _validate_protocol_instant_operations(protocol, windows)

    for window_index, window_start, window_end in windows:
        for operation in protocol.before:
            event = operation.to_instant_event(time=window_start)
            instant_payloads.append(_strip_metadata(event.to_payload()))
            instant_sources.append((event, _protocol_instant_source(protocol, window_index, operation)))
            lineage.append(_lineage_entry_for_protocol(protocol, window_index, operation, primitive=event))
        for operation in protocol.during:
            interval = operation.to_interval(start=window_start, end=window_end)
            interval_payloads.append(_strip_metadata(interval.to_payload()))
            lineage.append(_lineage_entry_for_protocol(protocol, window_index, operation, primitive=interval))
        for operation in protocol.after:
            event = operation.to_instant_event(time=window_end)
            instant_payloads.append(_strip_metadata(event.to_payload()))
            instant_sources.append((event, _protocol_instant_source(protocol, window_index, operation)))
            lineage.append(_lineage_entry_for_protocol(protocol, window_index, operation, primitive=event))


def _validate_protocol_instant_operations(
    protocol: InterventionProtocol,
    windows: Sequence[tuple[int, float, float]],
) -> None:
    by_time_species: dict[tuple[float, str], tuple[int, str, InterventionProtocolOperation]] = {}
    for window_index, window_start, window_end in windows:
        for phase, time, operations in (
            ("before", window_start, protocol.before),
            ("after", window_end, protocol.after),
        ):
            for operation in operations:
                key = (float(time), str(operation.species))
                existing = by_time_species.get(key)
                if existing is None:
                    by_time_species[key] = (window_index, phase, operation)
                    continue
                existing_window_index, existing_phase, existing_operation = existing
                if _instant_operations_conflict(existing_operation, operation):
                    raise InterventionScheduleError(
                        f"Protocol {protocol.name!r} window {existing_window_index} phase {existing_phase!r} "
                        f"conflicts with window {window_index} phase {phase!r} for species {operation.species} "
                        f"at t={time:g}."
                    )


def _instant_operations_conflict(left: InterventionProtocolOperation, right: InterventionProtocolOperation) -> bool:
    absolute = {"set", "clear"}
    if left.op in absolute and right.op in absolute:
        return _operation_execution_payload(left) != _operation_execution_payload(right)
    return left.op in absolute or right.op in absolute


def _validate_compiled_instant_conflicts(instants: Sequence[tuple[InterventionInstantEvent, str]]) -> None:
    absolute_by_key: dict[tuple[float, str], tuple[InterventionInstantEvent, str]] = {}
    delta_by_key: dict[tuple[float, str], tuple[InterventionInstantEvent, str]] = {}
    for event, source in instants:
        key = (float(event.time if event.time is not None else 0.0), str(event.species))
        if event.op not in {"set", "clear"}:
            existing = absolute_by_key.get(key)
            if existing is not None:
                _raise_instant_conflict(existing, (event, source), species=event.species, time=key[0])
            delta_by_key.setdefault(key, (event, source))
            continue
        existing_delta = delta_by_key.get(key)
        if existing_delta is not None:
            _raise_instant_conflict(existing_delta, (event, source), species=event.species, time=key[0])
        existing_absolute = absolute_by_key.get(key)
        if existing_absolute is not None and _instant_events_conflict(existing_absolute[0], event):
            _raise_instant_conflict(existing_absolute, (event, source), species=event.species, time=key[0])
        absolute_by_key[key] = (event, source)


def _raise_instant_conflict(
    left: tuple[InterventionInstantEvent, str],
    right: tuple[InterventionInstantEvent, str],
    *,
    species: str,
    time: float,
) -> None:
    left_source = left[1]
    right_source = right[1]
    if right_source.startswith("protocol ") and not left_source.startswith("protocol "):
        left_source, right_source = right_source, left_source
    raise InterventionScheduleError(
        f"{left_source} conflicts with {right_source} for species {species} at t={time:g}."
    )


def _instant_events_conflict(left: InterventionInstantEvent, right: InterventionInstantEvent) -> bool:
    absolute = {"set", "clear"}
    if left.op in absolute and right.op in absolute:
        return _instant_event_execution_payload(left) != _instant_event_execution_payload(right)
    return left.op in absolute or right.op in absolute


def _instant_event_execution_payload(event: InterventionInstantEvent) -> dict[str, Any]:
    payload = event.to_payload()
    payload.pop("metadata", None)
    return payload


def _operation_execution_payload(operation: InterventionProtocolOperation) -> dict[str, Any]:
    payload = operation.to_payload()
    payload.pop("metadata", None)
    return payload


def _protocol_instant_source(
    protocol: InterventionProtocol,
    window_index: int,
    operation: InterventionProtocolOperation,
) -> str:
    return f"protocol {protocol.name!r} window {int(window_index)} phase {operation.phase!r}"


def _schedule_has_metadata(schedule: InterventionSchedule) -> bool:
    return bool(
        schedule.metadata
        or any(event.metadata for event in schedule.instant_events)
        or any(event.metadata for event in schedule.repeated_events)
        or any(event.metadata for event in schedule.trigger_events)
        or any(interval.metadata for interval in schedule.intervals)
    )


def _lineage_entry_for_repeated_interval(
    repeated_interval: InterventionRepeatedInterval,
    *,
    declaration_index: int,
    window_index: int,
    interval: InterventionInterval,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "declaration": "repeated_interval",
        "declaration_index": int(declaration_index),
        "window_index": int(window_index),
        "phase": "during",
        "species": str(repeated_interval.species),
        "primitive_kind": str(interval.kind),
        "start": float(interval.start if interval.start is not None else 0.0),
        "end": float(interval.end if interval.end is not None else 0.0),
    }
    if repeated_interval.metadata:
        entry["metadata"] = _metadata_dict(repeated_interval.metadata)
    return entry


def _lineage_entry_for_protocol(
    protocol: InterventionProtocol,
    window_index: int,
    operation: InterventionProtocolOperation,
    *,
    primitive: InterventionInstantEvent | InterventionInterval,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "protocol": str(protocol.name),
        "window_index": int(window_index),
        "phase": str(operation.phase),
        "species": str(operation.species),
        "primitive_kind": str(operation.op),
    }
    if isinstance(primitive, InterventionInterval):
        entry["start"] = float(primitive.start if primitive.start is not None else 0.0)
        entry["end"] = float(primitive.end if primitive.end is not None else 0.0)
    else:
        entry["time"] = float(primitive.time if primitive.time is not None else 0.0)
    if protocol.metadata:
        entry["protocol_metadata"] = _metadata_dict(protocol.metadata)
    if operation.metadata:
        entry["metadata"] = _metadata_dict(operation.metadata)
    return entry


def _primitive_metadata_entries(schedule: InterventionSchedule) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for index, event in enumerate(schedule.instant_events):
        if event.metadata:
            entries.append(
                {
                    "kind": "instant_event",
                    "index": int(index),
                    "species": str(event.species),
                    "metadata": _metadata_dict(event.metadata),
                }
            )
    for index, event in enumerate(schedule.repeated_events):
        if event.metadata:
            entries.append(
                {
                    "kind": "repeated_event",
                    "index": int(index),
                    "species": str(event.species),
                    "metadata": _metadata_dict(event.metadata),
                }
            )
    for index, event in enumerate(schedule.trigger_events):
        if event.metadata:
            entries.append(
                {
                    "kind": "trigger_event",
                    "index": int(index),
                    "species": str(event.species),
                    "metadata": _metadata_dict(event.metadata),
                }
            )
    for index, interval in enumerate(schedule.intervals):
        if interval.metadata:
            entries.append(
                {
                    "kind": "interval",
                    "index": int(index),
                    "species": str(interval.species),
                    "metadata": _metadata_dict(interval.metadata),
                }
            )
    return entries


def _metadata_mentions_display_unit(metadata: Mapping[str, Any] | Sequence[tuple[str, str]]) -> bool:
    if isinstance(metadata, Mapping):
        return "display_unit" in metadata
    return any(str(key) == "display_unit" for key, _value in metadata)


def _lineage_mentions_display_unit(lineage: Sequence[Mapping[str, Any]]) -> bool:
    for entry in lineage:
        for key in ("metadata", "protocol_metadata"):
            metadata = entry.get(key)
            if isinstance(metadata, Mapping) and _metadata_mentions_display_unit(metadata):
                return True
    return False


def _strip_metadata(payload: Mapping[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in payload.items():
        if str(key) == "metadata":
            continue
        if isinstance(value, Mapping):
            out[str(key)] = _strip_metadata(value)
        elif isinstance(value, list):
            out[str(key)] = [_strip_metadata(item) if isinstance(item, Mapping) else item for item in value]
        else:
            out[str(key)] = value
    return out


def _copy_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, Mapping):
            out[str(key)] = _copy_mapping(item)
        elif isinstance(item, list):
            out[str(key)] = [_copy_mapping(entry) if isinstance(entry, Mapping) else entry for entry in item]
        elif isinstance(item, tuple):
            out[str(key)] = tuple(_copy_mapping(entry) if isinstance(entry, Mapping) else entry for entry in item)
        else:
            out[str(key)] = item
    return out


def _metadata_dict(metadata: Sequence[tuple[str, str]]) -> dict[str, str]:
    return {str(key): str(value) for key, value in metadata}
