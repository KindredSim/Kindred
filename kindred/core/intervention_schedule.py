from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

__all__ = [
    "InterventionInstantEvent",
    "InterventionInterval",
    "InterventionProtocol",
    "InterventionRepeatedInterval",
    "InterventionRepeatedEvent",
    "InterventionSchedule",
    "InterventionScheduleError",
    "InterventionTriggerEvent",
    "coerce_intervention_schedule",
    "intervention_schedule_identity_fingerprints",
    "intervention_schedule_parameter_names",
    "normalized_intervention_schedule_identity_fingerprints",
    "normalized_intervention_schedule_identity_fingerprints_from_dsl_text",
    "normalized_intervention_schedule_payload",
    "parse_intervention_schedule_from_dsl",
]


class InterventionScheduleError(ValueError):
    """Raised when a species intervention schedule is invalid."""


_ABSOLUTE_INSTANT_OPS = frozenset({"set", "clear"})
_ADDITIVE_INSTANT_OPS = frozenset({"add", "remove"})
_INSTANT_OPS = _ABSOLUTE_INSTANT_OPS | _ADDITIVE_INSTANT_OPS
_INTERVAL_KINDS = frozenset({"source", "sink", "reservoir", "clamp"})
_TRIGGER_DIRECTIONS = frozenset({"rising", "falling", "either"})
_METADATA_KEYS = frozenset({"label", "intent", "quantity_kind", "display_unit"})
_QUANTITY_KINDS = frozenset({"concentration", "amount", "rate", "intensity"})
_SCHEDULE_PAYLOAD_KEYS = frozenset(
    {
        "instant_events",
        "repeated_events",
        "trigger_events",
        "intervals",
        "repeated_intervals",
        "protocols",
        "metadata",
        *_METADATA_KEYS,
    }
)


def _payload_keys(*keys: str) -> frozenset[str]:
    return frozenset((*keys, "metadata", *_METADATA_KEYS))


def _reject_unknown_payload_keys(
    payload: Mapping[str, Any],
    allowed_keys: frozenset[str],
    *,
    context: str,
) -> None:
    unknown_keys = sorted(str(key) for key in payload if str(key) not in allowed_keys)
    if unknown_keys:
        raise InterventionScheduleError(f"Unsupported {context} field: {unknown_keys[0]!r}.")


def _reject_payload_fields(
    payload: Mapping[str, Any],
    fields: frozenset[str],
    *,
    context: str,
) -> None:
    rejected = sorted(field for field in fields if field in payload)
    if rejected:
        raise InterventionScheduleError(f"Unsupported {context} field: {rejected[0]!r}.")


_INSTANT_EVENT_PAYLOAD_KEYS = _payload_keys(
    "op",
    "time",
    "time_param",
    "species",
    "value",
    "value_param",
    "amount",
    "amount_param",
)
_REPEATED_EVENT_PAYLOAD_KEYS = _payload_keys(
    "op",
    "start",
    "start_param",
    "every",
    "every_param",
    "count",
    "species",
    "amount",
    "amount_param",
)
_TRIGGER_EVENT_PAYLOAD_KEYS = _payload_keys(
    "trigger_species",
    "threshold",
    "threshold_param",
    "direction",
    "species",
    "action",
    "max_count",
    "min_interval",
    "value",
    "value_param",
    "amount",
    "amount_param",
)
_INTERVAL_PAYLOAD_KEYS = _payload_keys(
    "kind",
    "start",
    "start_param",
    "end",
    "end_param",
    "species",
    "rate",
    "rate_param",
    "value",
    "value_param",
)
_REPEATED_INTERVAL_PAYLOAD_KEYS = _payload_keys(
    "kind",
    "start",
    "start_param",
    "every",
    "every_param",
    "duration",
    "duration_param",
    "count",
    "species",
    "rate",
    "rate_param",
    "value",
    "value_param",
)
_PROTOCOL_DURING_OPERATION_PAYLOAD_KEYS = _payload_keys(
    "kind",
    "species",
    "rate",
    "rate_param",
    "value",
    "value_param",
)
_PROTOCOL_INSTANT_OPERATION_PAYLOAD_KEYS = _payload_keys(
    "op",
    "species",
    "value",
    "value_param",
    "amount",
    "amount_param",
)
_PROTOCOL_PAYLOAD_KEYS = _payload_keys(
    "kind",
    "name",
    "start",
    "start_param",
    "every",
    "every_param",
    "duration",
    "duration_param",
    "count",
    "before",
    "during",
    "after",
)


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


def _positive_int_from_payload(payload: Mapping[str, Any], *, field: str) -> int:
    raw = payload.get(field)
    if isinstance(raw, bool):
        raise InterventionScheduleError(f"{field} must be an integer.")
    if isinstance(raw, int):
        value = int(raw)
    elif isinstance(raw, str):
        raw_s = raw.strip()
        if not raw_s.isdecimal():
            raise InterventionScheduleError(f"{field} must be an integer.")
        value = int(raw_s)
    else:
        raise InterventionScheduleError(f"{field} must be an integer.")
    if value <= 0:
        raise InterventionScheduleError(f"{field} must be greater than zero.")
    return value


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool):
        raise InterventionScheduleError(f"{field} must be an integer.")
    if isinstance(value, int):
        out = int(value)
    elif isinstance(value, str):
        raw_s = value.strip()
        if not raw_s.isdecimal():
            raise InterventionScheduleError(f"{field} must be an integer.")
        out = int(raw_s)
    else:
        raise InterventionScheduleError(f"{field} must be an integer.")
    if out <= 0:
        raise InterventionScheduleError(f"{field} must be greater than zero.")
    return out


def _parameter_name(value: object, *, field: str) -> str:
    name = str(value or "").strip()
    if not name:
        raise InterventionScheduleError(f"{field} must not be empty.")
    return name


def _has_parameter(value: object) -> bool:
    return bool(str(value or "").strip())


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


def _canonical_scalar_pair(
    value: float | None,
    parameter: str | None,
    *,
    field: str,
    nonnegative: bool = False,
    positive: bool = False,
) -> tuple[float | None, str | None]:
    if _has_parameter(parameter):
        parameter_name = _parameter_name(parameter, field=f"{field}_param")
        if value is not None:
            raise InterventionScheduleError(f"{field} and {field}_param are mutually exclusive.")
        return None, parameter_name
    if positive:
        return _positive_float(value, field=field), None
    if nonnegative:
        return _nonnegative_float(value, field=field), None
    return _finite_float(value, field=field), None


def _reject_scalar_pair(
    value: float | None,
    parameter: str | None,
    *,
    field: str,
    context: str,
) -> None:
    if value is not None or _has_parameter(parameter):
        raise InterventionScheduleError(f"Unsupported {context} field: {field!r}.")


def _canonical_clear_value(value: float | None, parameter: str | None, *, context: str) -> tuple[float, None]:
    if _has_parameter(parameter):
        raise InterventionScheduleError(f"Unsupported {context} field: 'value_param'.")
    if value is None:
        return 0.0, None
    if _nonnegative_float(value, field="value") != 0.0:
        raise InterventionScheduleError(f"{context} value must be 0.0 when provided.")
    return 0.0, None


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


def _store_metadata_entry(metadata: dict[str, str], key: object, value: object) -> None:
    key_s = str(key).strip()
    if key_s not in _METADATA_KEYS:
        raise InterventionScheduleError(f"Unsupported intervention metadata key: {key_s!r}.")
    if value is None:
        return
    metadata[key_s] = str(value)


def _validated_metadata_tuple(metadata: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    quantity_kind = metadata.get("quantity_kind")
    if quantity_kind is not None and quantity_kind not in _QUANTITY_KINDS:
        raise InterventionScheduleError(f"Unsupported intervention quantity_kind: {quantity_kind!r}.")
    return tuple(sorted(metadata.items()))


def _metadata_from_payload(payload: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    metadata: dict[str, str] = {}
    raw_metadata = payload.get("metadata")
    if raw_metadata is not None:
        if not isinstance(raw_metadata, Mapping):
            raise InterventionScheduleError("metadata must be a mapping.")
        for key, value in raw_metadata.items():
            _store_metadata_entry(metadata, key, value)
    for key in _METADATA_KEYS:
        if key in payload:
            _store_metadata_entry(metadata, key, payload.get(key))
    return _validated_metadata_tuple(metadata)


def _metadata_tuple(value: Mapping[str, str] | Sequence[tuple[str, str]]) -> tuple[tuple[str, str], ...]:
    if isinstance(value, Mapping):
        value = tuple(value.items())
    metadata: dict[str, str] = {}
    try:
        for key, item in value:
            _store_metadata_entry(metadata, key, item)
    except (TypeError, ValueError) as exc:
        raise InterventionScheduleError("metadata entries must be key/value pairs.") from exc
    return _validated_metadata_tuple(metadata)


def _write_metadata_payload(payload: dict[str, Any], metadata: Sequence[tuple[str, str]]) -> None:
    if metadata:
        payload["metadata"] = {str(key): str(value) for key, value in metadata}


def intervention_schedule_parameter_names(schedule: "InterventionSchedule | None") -> set[str]:
    if schedule is None:
        return set()
    payload = schedule.to_payload()
    names: set[str] = set()

    def _walk(value: object) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if str(key).endswith("_param") and str(item or "").strip():
                    names.add(str(item))
                else:
                    _walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                _walk(item)

    _walk(payload)
    return names


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
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _metadata_tuple(self.metadata))
        time, time_param = _canonical_scalar_pair(self.time, self.time_param, field="time", nonnegative=True)
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "time_param", time_param)
        object.__setattr__(self, "species", _species_name(self.species))
        op = str(self.op).strip().lower()
        if op not in _INSTANT_OPS:
            raise InterventionScheduleError(f"Unsupported instant intervention op: {op!r}.")
        object.__setattr__(self, "op", op)
        if op == "set":
            value, value_param = _canonical_scalar_pair(self.value, self.value_param, field="value", nonnegative=True)
            object.__setattr__(self, "value", value)
            object.__setattr__(self, "value_param", value_param)
            _reject_scalar_pair(
                self.amount,
                self.amount_param,
                field="amount",
                context="set instant intervention",
            )
            object.__setattr__(self, "amount", None)
            object.__setattr__(self, "amount_param", None)
            return
        if op == "clear":
            value, value_param = _canonical_clear_value(
                self.value,
                self.value_param,
                context="clear instant intervention",
            )
            object.__setattr__(self, "value", value)
            object.__setattr__(self, "value_param", value_param)
            _reject_scalar_pair(
                self.amount,
                self.amount_param,
                field="amount",
                context="clear instant intervention",
            )
            object.__setattr__(self, "amount", None)
            object.__setattr__(self, "amount_param", None)
            return
        _reject_scalar_pair(
            self.value,
            self.value_param,
            field="value",
            context=f"{op} instant intervention",
        )
        object.__setattr__(self, "value", None)
        object.__setattr__(self, "value_param", None)
        amount, amount_param = _canonical_scalar_pair(self.amount, self.amount_param, field="amount", nonnegative=True)
        object.__setattr__(self, "amount", amount)
        object.__setattr__(self, "amount_param", amount_param)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "InterventionInstantEvent":
        _reject_unknown_payload_keys(payload, _INSTANT_EVENT_PAYLOAD_KEYS, context="instant intervention")
        op = str(payload.get("op") or "").strip().lower()
        if op not in _INSTANT_OPS:
            raise InterventionScheduleError(f"Unsupported instant intervention op: {op!r}.")
        time, time_param = _payload_scalar(payload, field="time", nonnegative=True)
        species = _species_name(payload.get("species"))
        metadata = _metadata_from_payload(payload)
        if op in {"set"}:
            _reject_payload_fields(
                payload,
                frozenset({"amount", "amount_param"}),
                context="set instant intervention",
            )
            value, value_param = _payload_scalar(payload, field="value", nonnegative=True)
            return cls(
                time=time,
                species=species,
                op=op,
                value=value,
                time_param=time_param,
                value_param=value_param,
                metadata=metadata,
            )
        if op == "clear":
            _reject_payload_fields(
                payload,
                frozenset({"value", "value_param", "amount", "amount_param"}),
                context="clear instant intervention",
            )
            return cls(time=time, species=species, op=op, value=0.0, time_param=time_param, metadata=metadata)
        _reject_payload_fields(
            payload,
            frozenset({"value", "value_param"}),
            context=f"{op} instant intervention",
        )
        amount, amount_param = _payload_scalar(payload, field="amount", nonnegative=True)
        return cls(
            time=time,
            species=species,
            op=op,
            amount=amount,
            time_param=time_param,
            amount_param=amount_param,
            metadata=metadata,
        )

    def resolve_parameters(self, parameter_values: Mapping[str, Any]) -> "InterventionInstantEvent":
        time = _resolve_scalar(self.time, self.time_param, parameter_values, field="time", nonnegative=True)
        if self.op == "set":
            return InterventionInstantEvent(
                time=time,
                species=self.species,
                op=self.op,
                value=_resolve_scalar(
                    self.value,
                    self.value_param,
                    parameter_values,
                    field="value",
                    nonnegative=True,
                ),
                metadata=tuple(self.metadata),
            )
        if self.op == "clear":
            return InterventionInstantEvent(
                time=time,
                species=self.species,
                op=self.op,
                value=0.0,
                metadata=tuple(self.metadata),
            )
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
            metadata=tuple(self.metadata),
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
        if _instant_event_is_absolute(self):
            if self.op != "clear":
                _write_scalar_payload(payload, field="value", value=self.value, parameter=self.value_param)
        else:
            _write_scalar_payload(payload, field="amount", value=self.amount, parameter=self.amount_param)
        _write_metadata_payload(payload, self.metadata)
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
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _metadata_tuple(self.metadata))
        object.__setattr__(self, "count", _positive_int(self.count, field="count"))
        op = str(self.op).strip().lower()
        if op not in {"pulse", "add", "remove"}:
            raise InterventionScheduleError(f"Unsupported repeated intervention op: {op!r}.")
        object.__setattr__(self, "op", op)
        object.__setattr__(self, "species", _species_name(self.species))
        start, start_param = _canonical_scalar_pair(self.start, self.start_param, field="start", nonnegative=True)
        every, every_param = _canonical_scalar_pair(self.every, self.every_param, field="every", positive=True)
        amount, amount_param = _canonical_scalar_pair(self.amount, self.amount_param, field="amount", nonnegative=True)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "start_param", start_param)
        object.__setattr__(self, "every", every)
        object.__setattr__(self, "every_param", every_param)
        object.__setattr__(self, "amount", amount)
        object.__setattr__(self, "amount_param", amount_param)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "InterventionRepeatedEvent":
        _reject_unknown_payload_keys(payload, _REPEATED_EVENT_PAYLOAD_KEYS, context="repeated intervention")
        count = _positive_int_from_payload(payload, field="count")
        op = str(payload.get("op") or "pulse").strip().lower()
        if op not in {"pulse", "add", "remove"}:
            raise InterventionScheduleError(f"Unsupported repeated intervention op: {op!r}.")
        start, start_param = _payload_scalar(payload, field="start", nonnegative=True)
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
            metadata=_metadata_from_payload(payload),
        )

    @property
    def is_parameterized(self) -> bool:
        return bool(self.start_param or self.every_param or self.amount_param)

    def resolve_parameters(self, parameter_values: Mapping[str, Any]) -> "InterventionRepeatedEvent":
        return InterventionRepeatedEvent(
            start=_resolve_scalar(self.start, self.start_param, parameter_values, field="start", nonnegative=True),
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
            metadata=tuple(self.metadata),
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
                metadata=tuple(self.metadata),
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
        _write_metadata_payload(payload, self.metadata)
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
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _metadata_tuple(self.metadata))
        object.__setattr__(self, "trigger_species", _species_name(self.trigger_species))
        object.__setattr__(self, "species", _species_name(self.species))
        direction = str(self.direction).strip().lower()
        if direction not in _TRIGGER_DIRECTIONS:
            raise InterventionScheduleError("direction must be rising, falling, or either.")
        object.__setattr__(self, "direction", direction)
        op = str(self.op).strip().lower()
        if op not in _INSTANT_OPS:
            raise InterventionScheduleError(f"Unsupported trigger action: {op!r}.")
        object.__setattr__(self, "op", op)
        object.__setattr__(self, "max_count", _positive_int(self.max_count, field="max_count"))
        object.__setattr__(self, "min_interval", _nonnegative_float(self.min_interval, field="min_interval"))
        threshold, threshold_param = _canonical_scalar_pair(
            self.threshold,
            self.threshold_param,
            field="threshold",
            nonnegative=True,
        )
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(self, "threshold_param", threshold_param)
        if op == "set":
            value, value_param = _canonical_scalar_pair(self.value, self.value_param, field="value", nonnegative=True)
            object.__setattr__(self, "value", value)
            object.__setattr__(self, "value_param", value_param)
            _reject_scalar_pair(
                self.amount,
                self.amount_param,
                field="amount",
                context="set trigger intervention",
            )
            object.__setattr__(self, "amount", None)
            object.__setattr__(self, "amount_param", None)
            return
        if op == "clear":
            value, value_param = _canonical_clear_value(self.value, self.value_param, context="clear trigger intervention")
            object.__setattr__(self, "value", value)
            object.__setattr__(self, "value_param", value_param)
            _reject_scalar_pair(
                self.amount,
                self.amount_param,
                field="amount",
                context="clear trigger intervention",
            )
            object.__setattr__(self, "amount", None)
            object.__setattr__(self, "amount_param", None)
            return
        _reject_scalar_pair(
            self.value,
            self.value_param,
            field="value",
            context=f"{op} trigger intervention",
        )
        object.__setattr__(self, "value", None)
        object.__setattr__(self, "value_param", None)
        amount, amount_param = _canonical_scalar_pair(self.amount, self.amount_param, field="amount", nonnegative=True)
        object.__setattr__(self, "amount", amount)
        object.__setattr__(self, "amount_param", amount_param)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "InterventionTriggerEvent":
        _reject_unknown_payload_keys(payload, _TRIGGER_EVENT_PAYLOAD_KEYS, context="trigger intervention")
        direction = str(payload.get("direction") or "").strip().lower()
        if direction not in _TRIGGER_DIRECTIONS:
            raise InterventionScheduleError("direction must be rising, falling, or either.")
        action = str(payload.get("action") or "").strip().lower()
        if action not in _INSTANT_OPS:
            raise InterventionScheduleError(f"Unsupported trigger action: {action!r}.")
        max_count = _positive_int_from_payload(payload, field="max_count")
        min_interval = _nonnegative_float(payload.get("min_interval"), field="min_interval")
        threshold, threshold_param = _payload_scalar(payload, field="threshold", nonnegative=True)
        base = dict(
            trigger_species=_species_name(payload.get("trigger_species")),
            threshold=threshold,
            threshold_param=threshold_param,
            direction=direction,
            species=_species_name(payload.get("species")),
            op=action,
            max_count=max_count,
            min_interval=min_interval,
            metadata=_metadata_from_payload(payload),
        )
        if action == "set":
            _reject_payload_fields(
                payload,
                frozenset({"amount", "amount_param"}),
                context="set trigger intervention",
            )
            value, value_param = _payload_scalar(payload, field="value", nonnegative=True)
            return cls(**base, value=value, value_param=value_param)
        if action == "clear":
            _reject_payload_fields(
                payload,
                frozenset({"value", "value_param", "amount", "amount_param"}),
                context="clear trigger intervention",
            )
            return cls(**base, value=0.0)
        _reject_payload_fields(
            payload,
            frozenset({"value", "value_param"}),
            context=f"{action} trigger intervention",
        )
        amount, amount_param = _payload_scalar(payload, field="amount", nonnegative=True)
        return cls(**base, amount=amount, amount_param=amount_param)

    @property
    def is_parameterized(self) -> bool:
        return bool(self.threshold_param or self.value_param or self.amount_param)

    def resolve_parameters(self, parameter_values: Mapping[str, Any]) -> "InterventionTriggerEvent":
        threshold = _resolve_scalar(
            self.threshold,
            self.threshold_param,
            parameter_values,
            field="threshold",
            nonnegative=True,
        )
        if self.op == "set":
            return InterventionTriggerEvent(
                trigger_species=self.trigger_species,
                threshold=threshold,
                direction=self.direction,
                species=self.species,
                op=self.op,
                max_count=int(self.max_count),
                min_interval=float(self.min_interval),
                value=_resolve_scalar(
                    self.value,
                    self.value_param,
                    parameter_values,
                    field="value",
                    nonnegative=True,
                ),
                metadata=tuple(self.metadata),
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
                metadata=tuple(self.metadata),
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
            metadata=tuple(self.metadata),
        )

    def to_instant_event(self, *, time: float) -> InterventionInstantEvent:
        if self.op == "set":
            return InterventionInstantEvent(
                time=float(time),
                species=self.species,
                op=self.op,
                value=self.value,
                metadata=tuple(self.metadata),
            )
        if self.op == "clear":
            return InterventionInstantEvent(
                time=float(time),
                species=self.species,
                op=self.op,
                value=0.0,
                metadata=tuple(self.metadata),
            )
        return InterventionInstantEvent(
            time=float(time),
            species=self.species,
            op=self.op,
            amount=self.amount,
            metadata=tuple(self.metadata),
        )

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
        _write_metadata_payload(payload, self.metadata)
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
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _metadata_tuple(self.metadata))
        start, start_param = _canonical_scalar_pair(self.start, self.start_param, field="start", nonnegative=True)
        end, end_param = _canonical_scalar_pair(self.end, self.end_param, field="end", nonnegative=True)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "start_param", start_param)
        object.__setattr__(self, "end", end)
        object.__setattr__(self, "end_param", end_param)
        if not _has_parameter(self.start_param) and not _has_parameter(self.end_param):
            if float(self.end if self.end is not None else 0.0) <= float(self.start if self.start is not None else 0.0):
                raise InterventionScheduleError("end must be greater than start.")
        object.__setattr__(self, "species", _species_name(self.species))
        kind = str(self.kind).strip().lower()
        if kind not in _INTERVAL_KINDS:
            raise InterventionScheduleError(f"Unsupported intervention interval kind: {kind!r}.")
        object.__setattr__(self, "kind", kind)
        if kind in {"source", "sink"}:
            rate, rate_param = _canonical_scalar_pair(self.rate, self.rate_param, field="rate", nonnegative=True)
            object.__setattr__(self, "rate", rate)
            object.__setattr__(self, "rate_param", rate_param)
            _reject_scalar_pair(
                self.value,
                self.value_param,
                field="value",
                context=f"{kind} interval",
            )
            object.__setattr__(self, "value", None)
            object.__setattr__(self, "value_param", None)
            return
        _reject_scalar_pair(
            self.rate,
            self.rate_param,
            field="rate",
            context=f"{kind} interval",
        )
        object.__setattr__(self, "rate", None)
        object.__setattr__(self, "rate_param", None)
        value, value_param = _canonical_scalar_pair(self.value, self.value_param, field="value", nonnegative=True)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "value_param", value_param)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "InterventionInterval":
        _reject_unknown_payload_keys(payload, _INTERVAL_PAYLOAD_KEYS, context="intervention interval")
        kind = str(payload.get("kind") or "").strip().lower()
        if kind not in _INTERVAL_KINDS:
            raise InterventionScheduleError(f"Unsupported intervention interval kind: {kind!r}.")
        start, start_param = _payload_scalar(payload, field="start", nonnegative=True)
        end, end_param = _payload_scalar(payload, field="end", nonnegative=True)
        if start_param is None and end_param is None and end <= start:
            raise InterventionScheduleError("end must be greater than start.")
        species = _species_name(payload.get("species"))
        metadata = _metadata_from_payload(payload)
        if kind in {"source", "sink"}:
            _reject_payload_fields(
                payload,
                frozenset({"value", "value_param"}),
                context=f"{kind} interval",
            )
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
                metadata=metadata,
            )
        _reject_payload_fields(
            payload,
            frozenset({"rate", "rate_param"}),
            context=f"{kind} interval",
        )
        value, value_param = _payload_scalar(payload, field="value", nonnegative=True)
        return cls(
            start=start,
            end=end,
            species=species,
            kind=kind,
            value=value,
            start_param=start_param,
            end_param=end_param,
            value_param=value_param,
            metadata=metadata,
        )

    def resolve_parameters(self, parameter_values: Mapping[str, Any]) -> "InterventionInterval":
        start = _resolve_scalar(self.start, self.start_param, parameter_values, field="start", nonnegative=True)
        end = _resolve_scalar(self.end, self.end_param, parameter_values, field="end", nonnegative=True)
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
                metadata=tuple(self.metadata),
            )
        return InterventionInterval(
            start=start,
            end=end,
            species=self.species,
            kind=self.kind,
            value=_resolve_scalar(
                self.value,
                self.value_param,
                parameter_values,
                field="value",
                nonnegative=True,
            ),
            metadata=tuple(self.metadata),
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
        _write_metadata_payload(payload, self.metadata)
        return payload


@dataclass(frozen=True, slots=True)
class InterventionRepeatedInterval:
    start: float | None
    every: float | None
    duration: float | None
    count: int
    species: str
    kind: str
    rate: float | None = None
    value: float | None = None
    start_param: str | None = None
    every_param: str | None = None
    duration_param: str | None = None
    rate_param: str | None = None
    value_param: str | None = None
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _metadata_tuple(self.metadata))
        object.__setattr__(self, "count", _positive_int(self.count, field="count"))
        object.__setattr__(self, "species", _species_name(self.species))
        kind = str(self.kind).strip().lower()
        if kind not in _INTERVAL_KINDS:
            raise InterventionScheduleError(f"Unsupported repeated interval kind: {kind!r}.")
        object.__setattr__(self, "kind", kind)
        start, start_param = _canonical_scalar_pair(self.start, self.start_param, field="start", nonnegative=True)
        every, every_param = _canonical_scalar_pair(self.every, self.every_param, field="every", positive=True)
        duration, duration_param = _canonical_scalar_pair(
            self.duration,
            self.duration_param,
            field="duration",
            positive=True,
        )
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "start_param", start_param)
        object.__setattr__(self, "every", every)
        object.__setattr__(self, "every_param", every_param)
        object.__setattr__(self, "duration", duration)
        object.__setattr__(self, "duration_param", duration_param)
        if kind in {"source", "sink"}:
            rate, rate_param = _canonical_scalar_pair(self.rate, self.rate_param, field="rate", nonnegative=True)
            object.__setattr__(self, "rate", rate)
            object.__setattr__(self, "rate_param", rate_param)
            _reject_scalar_pair(
                self.value,
                self.value_param,
                field="value",
                context=f"{kind} repeated interval",
            )
            object.__setattr__(self, "value", None)
            object.__setattr__(self, "value_param", None)
            return
        _reject_scalar_pair(
            self.rate,
            self.rate_param,
            field="rate",
            context=f"{kind} repeated interval",
        )
        object.__setattr__(self, "rate", None)
        object.__setattr__(self, "rate_param", None)
        value, value_param = _canonical_scalar_pair(self.value, self.value_param, field="value", nonnegative=True)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "value_param", value_param)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "InterventionRepeatedInterval":
        _reject_unknown_payload_keys(payload, _REPEATED_INTERVAL_PAYLOAD_KEYS, context="repeated interval")
        kind = str(payload.get("kind") or "").strip().lower()
        if kind not in _INTERVAL_KINDS:
            raise InterventionScheduleError(f"Unsupported repeated interval kind: {kind!r}.")
        count = _positive_int_from_payload(payload, field="count")
        start, start_param = _payload_scalar(payload, field="start", nonnegative=True)
        every, every_param = _payload_scalar(payload, field="every", positive=True)
        duration, duration_param = _payload_scalar(payload, field="duration", positive=True)
        base = dict(
            start=start,
            every=every,
            duration=duration,
            count=count,
            species=_species_name(payload.get("species")),
            kind=kind,
            start_param=start_param,
            every_param=every_param,
            duration_param=duration_param,
            metadata=_metadata_from_payload(payload),
        )
        if kind in {"source", "sink"}:
            _reject_payload_fields(
                payload,
                frozenset({"value", "value_param"}),
                context=f"{kind} repeated interval",
            )
            rate, rate_param = _payload_scalar(payload, field="rate", nonnegative=True)
            return cls(**base, rate=rate, rate_param=rate_param)
        _reject_payload_fields(
            payload,
            frozenset({"rate", "rate_param"}),
            context=f"{kind} repeated interval",
        )
        value, value_param = _payload_scalar(payload, field="value", nonnegative=True)
        return cls(**base, value=value, value_param=value_param)

    @property
    def is_parameterized(self) -> bool:
        return bool(
            self.start_param
            or self.every_param
            or self.duration_param
            or self.rate_param
            or self.value_param
        )

    def resolve_parameters(self, parameter_values: Mapping[str, Any]) -> "InterventionRepeatedInterval":
        start = _resolve_scalar(self.start, self.start_param, parameter_values, field="start", nonnegative=True)
        every = _resolve_scalar(self.every, self.every_param, parameter_values, field="every", positive=True)
        duration = _resolve_scalar(
            self.duration,
            self.duration_param,
            parameter_values,
            field="duration",
            positive=True,
        )
        base = dict(
            start=start,
            every=every,
            duration=duration,
            count=int(self.count),
            species=self.species,
            kind=self.kind,
            metadata=tuple(self.metadata),
        )
        if self.kind in {"source", "sink"}:
            return InterventionRepeatedInterval(
                **base,
                rate=_resolve_scalar(self.rate, self.rate_param, parameter_values, field="rate", nonnegative=True),
            )
        return InterventionRepeatedInterval(
            **base,
            value=_resolve_scalar(
                self.value,
                self.value_param,
                parameter_values,
                field="value",
                nonnegative=True,
            ),
        )

    def expand(self) -> tuple[InterventionInterval, ...]:
        if self.is_parameterized:
            raise InterventionScheduleError("Parameterized repeated interval must be resolved before compilation.")
        start = float(self.start if self.start is not None else 0.0)
        every = _positive_float(self.every, field="every")
        duration = _positive_float(self.duration, field="duration")
        intervals: list[InterventionInterval] = []
        for idx in range(int(self.count)):
            window_start = start + every * float(idx)
            base = dict(
                start=window_start,
                end=window_start + duration,
                species=self.species,
                kind=self.kind,
            )
            if self.kind in {"source", "sink"}:
                intervals.append(
                    InterventionInterval(
                        **base,
                        rate=_nonnegative_float(self.rate, field="rate"),
                        metadata=tuple(self.metadata),
                    )
                )
            else:
                intervals.append(
                    InterventionInterval(
                        **base,
                        value=_nonnegative_float(self.value, field="value"),
                        metadata=tuple(self.metadata),
                    )
                )
        return tuple(intervals)

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "species": str(self.species),
            "kind": str(self.kind),
            "count": int(self.count),
        }
        _write_scalar_payload(payload, field="start", value=self.start, parameter=self.start_param)
        _write_scalar_payload(payload, field="every", value=self.every, parameter=self.every_param)
        _write_scalar_payload(payload, field="duration", value=self.duration, parameter=self.duration_param)
        if self.kind in {"source", "sink"}:
            _write_scalar_payload(payload, field="rate", value=self.rate, parameter=self.rate_param)
        else:
            _write_scalar_payload(payload, field="value", value=self.value, parameter=self.value_param)
        _write_metadata_payload(payload, self.metadata)
        return payload


@dataclass(frozen=True, slots=True)
class InterventionProtocolOperation:
    phase: str
    species: str
    op: str
    rate: float | None = None
    value: float | None = None
    amount: float | None = None
    rate_param: str | None = None
    value_param: str | None = None
    amount_param: str | None = None
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _metadata_tuple(self.metadata))
        phase = str(self.phase).strip().lower()
        if phase not in {"before", "during", "after"}:
            raise InterventionScheduleError(f"Unsupported protocol operation phase: {phase!r}.")
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "species", _species_name(self.species))
        op = str(self.op).strip().lower()
        if phase == "during":
            if op not in _INTERVAL_KINDS:
                raise InterventionScheduleError(f"Unsupported protocol interval kind: {op!r}.")
            object.__setattr__(self, "op", op)
            _reject_scalar_pair(
                self.amount,
                self.amount_param,
                field="amount",
                context="protocol during operation",
            )
            object.__setattr__(self, "amount", None)
            object.__setattr__(self, "amount_param", None)
            if op in {"source", "sink"}:
                rate, rate_param = _canonical_scalar_pair(self.rate, self.rate_param, field="rate", nonnegative=True)
                object.__setattr__(self, "rate", rate)
                object.__setattr__(self, "rate_param", rate_param)
                _reject_scalar_pair(
                    self.value,
                    self.value_param,
                    field="value",
                    context=f"{op} protocol during",
                )
                object.__setattr__(self, "value", None)
                object.__setattr__(self, "value_param", None)
                return
            _reject_scalar_pair(
                self.rate,
                self.rate_param,
                field="rate",
                context=f"{op} protocol during",
            )
            object.__setattr__(self, "rate", None)
            object.__setattr__(self, "rate_param", None)
            value, value_param = _canonical_scalar_pair(self.value, self.value_param, field="value", nonnegative=True)
            object.__setattr__(self, "value", value)
            object.__setattr__(self, "value_param", value_param)
            return
        if op not in _INSTANT_OPS:
            raise InterventionScheduleError(f"Unsupported protocol instant op: {op!r}.")
        object.__setattr__(self, "op", op)
        _reject_scalar_pair(self.rate, self.rate_param, field="rate", context=f"protocol {phase} operation")
        object.__setattr__(self, "rate", None)
        object.__setattr__(self, "rate_param", None)
        if op == "set":
            value, value_param = _canonical_scalar_pair(self.value, self.value_param, field="value", nonnegative=True)
            object.__setattr__(self, "value", value)
            object.__setattr__(self, "value_param", value_param)
            _reject_scalar_pair(self.amount, self.amount_param, field="amount", context=f"set protocol {phase}")
            object.__setattr__(self, "amount", None)
            object.__setattr__(self, "amount_param", None)
            return
        if op == "clear":
            value, value_param = _canonical_clear_value(self.value, self.value_param, context=f"clear protocol {phase}")
            object.__setattr__(self, "value", value)
            object.__setattr__(self, "value_param", value_param)
            _reject_scalar_pair(self.amount, self.amount_param, field="amount", context=f"clear protocol {phase}")
            object.__setattr__(self, "amount", None)
            object.__setattr__(self, "amount_param", None)
            return
        _reject_scalar_pair(self.value, self.value_param, field="value", context=f"{op} protocol {phase}")
        object.__setattr__(self, "value", None)
        object.__setattr__(self, "value_param", None)
        amount, amount_param = _canonical_scalar_pair(self.amount, self.amount_param, field="amount", nonnegative=True)
        object.__setattr__(self, "amount", amount)
        object.__setattr__(self, "amount_param", amount_param)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any], *, phase: str) -> "InterventionProtocolOperation":
        phase_s = str(phase)
        if phase_s == "during":
            _reject_unknown_payload_keys(
                payload,
                _PROTOCOL_DURING_OPERATION_PAYLOAD_KEYS,
                context="protocol during operation",
            )
            op = str(payload.get("kind") or "").strip().lower()
            if op not in _INTERVAL_KINDS:
                raise InterventionScheduleError(f"Unsupported protocol interval kind: {op!r}.")
            base = dict(phase=phase_s, species=_species_name(payload.get("species")), op=op, metadata=_metadata_from_payload(payload))
            if op in {"source", "sink"}:
                _reject_payload_fields(
                    payload,
                    frozenset({"value", "value_param"}),
                    context=f"{op} protocol during",
                )
                rate, rate_param = _payload_scalar(payload, field="rate", nonnegative=True)
                return cls(**base, rate=rate, rate_param=rate_param)
            _reject_payload_fields(
                payload,
                frozenset({"rate", "rate_param"}),
                context=f"{op} protocol during",
            )
            value, value_param = _payload_scalar(payload, field="value", nonnegative=True)
            return cls(**base, value=value, value_param=value_param)
        _reject_unknown_payload_keys(
            payload,
            _PROTOCOL_INSTANT_OPERATION_PAYLOAD_KEYS,
            context=f"protocol {phase_s} operation",
        )
        op = str(payload.get("op") or "").strip().lower()
        if op not in _INSTANT_OPS:
            raise InterventionScheduleError(f"Unsupported protocol instant op: {op!r}.")
        base = dict(phase=phase_s, species=_species_name(payload.get("species")), op=op, metadata=_metadata_from_payload(payload))
        if op == "set":
            _reject_payload_fields(
                payload,
                frozenset({"amount", "amount_param"}),
                context=f"set protocol {phase_s}",
            )
            value, value_param = _payload_scalar(payload, field="value", nonnegative=True)
            return cls(**base, value=value, value_param=value_param)
        if op == "clear":
            _reject_payload_fields(
                payload,
                frozenset({"value", "value_param", "amount", "amount_param"}),
                context=f"clear protocol {phase_s}",
            )
            return cls(**base, value=0.0)
        _reject_payload_fields(
            payload,
            frozenset({"value", "value_param"}),
            context=f"{op} protocol {phase_s}",
        )
        amount, amount_param = _payload_scalar(payload, field="amount", nonnegative=True)
        return cls(**base, amount=amount, amount_param=amount_param)

    @property
    def is_parameterized(self) -> bool:
        return bool(self.rate_param or self.value_param or self.amount_param)

    def resolve_parameters(self, parameter_values: Mapping[str, Any]) -> "InterventionProtocolOperation":
        base = dict(phase=self.phase, species=self.species, op=self.op, metadata=tuple(self.metadata))
        if self.phase == "during" and self.op in {"source", "sink"}:
            return InterventionProtocolOperation(
                **base,
                rate=_resolve_scalar(self.rate, self.rate_param, parameter_values, field="rate", nonnegative=True),
            )
        if self.phase == "during" or self.op == "set":
            return InterventionProtocolOperation(
                **base,
                value=_resolve_scalar(
                    self.value,
                    self.value_param,
                    parameter_values,
                    field="value",
                    nonnegative=True,
                ),
            )
        if self.op == "clear":
            return InterventionProtocolOperation(**base, value=0.0)
        return InterventionProtocolOperation(
            **base,
            amount=_resolve_scalar(self.amount, self.amount_param, parameter_values, field="amount", nonnegative=True),
        )

    def to_payload(self) -> dict[str, Any]:
        key = "kind" if self.phase == "during" else "op"
        payload: dict[str, Any] = {key: str(self.op), "species": str(self.species)}
        if self.phase == "during" and self.op in {"source", "sink"}:
            _write_scalar_payload(payload, field="rate", value=self.rate, parameter=self.rate_param)
        elif self.phase == "during" or self.op == "set":
            _write_scalar_payload(payload, field="value", value=self.value, parameter=self.value_param)
        elif self.op in _ADDITIVE_INSTANT_OPS:
            _write_scalar_payload(payload, field="amount", value=self.amount, parameter=self.amount_param)
        _write_metadata_payload(payload, self.metadata)
        return payload

    def to_interval(self, *, start: float, end: float) -> InterventionInterval:
        if self.phase != "during":
            raise InterventionScheduleError("Only protocol during operations lower to intervals.")
        if self.is_parameterized:
            raise InterventionScheduleError("Parameterized protocol operation must be resolved before compilation.")
        if self.op in {"source", "sink"}:
            return InterventionInterval(
                start=float(start),
                end=float(end),
                species=self.species,
                kind=self.op,
                rate=_nonnegative_float(self.rate, field="rate"),
                metadata=tuple(self.metadata),
            )
        return InterventionInterval(
            start=float(start),
            end=float(end),
            species=self.species,
            kind=self.op,
            value=_nonnegative_float(self.value, field="value"),
            metadata=tuple(self.metadata),
        )

    def to_instant_event(self, *, time: float) -> InterventionInstantEvent:
        if self.phase == "during":
            raise InterventionScheduleError("Protocol during operations do not lower to instants.")
        if self.is_parameterized:
            raise InterventionScheduleError("Parameterized protocol operation must be resolved before compilation.")
        if self.op == "set":
            return InterventionInstantEvent(
                time=float(time),
                species=self.species,
                op=self.op,
                value=self.value,
                metadata=tuple(self.metadata),
            )
        if self.op == "clear":
            return InterventionInstantEvent(
                time=float(time),
                species=self.species,
                op=self.op,
                value=0.0,
                metadata=tuple(self.metadata),
            )
        return InterventionInstantEvent(
            time=float(time),
            species=self.species,
            op=self.op,
            amount=self.amount,
            metadata=tuple(self.metadata),
        )


@dataclass(frozen=True, slots=True)
class InterventionProtocol:
    kind: str
    name: str
    start: float | None
    every: float | None
    duration: float | None
    count: int
    before: tuple[InterventionProtocolOperation, ...] = ()
    during: tuple[InterventionProtocolOperation, ...] = ()
    after: tuple[InterventionProtocolOperation, ...] = ()
    start_param: str | None = None
    every_param: str | None = None
    duration_param: str | None = None
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "before", tuple(sorted(tuple(self.before), key=_protocol_operation_sort_key)))
        object.__setattr__(self, "during", tuple(sorted(tuple(self.during), key=_protocol_operation_sort_key)))
        object.__setattr__(self, "after", tuple(sorted(tuple(self.after), key=_protocol_operation_sort_key)))
        object.__setattr__(self, "metadata", _metadata_tuple(self.metadata))
        kind = str(self.kind).strip().lower()
        if kind != "repeat":
            raise InterventionScheduleError(f"Unsupported intervention protocol kind: {self.kind!r}.")
        object.__setattr__(self, "kind", kind)
        name = str(self.name or "").strip()
        if not name:
            raise InterventionScheduleError("protocol name must not be empty.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "count", _positive_int(self.count, field="count"))
        start, start_param = _canonical_scalar_pair(self.start, self.start_param, field="start", nonnegative=True)
        every, every_param = _canonical_scalar_pair(self.every, self.every_param, field="every", positive=True)
        duration, duration_param = _canonical_scalar_pair(
            self.duration,
            self.duration_param,
            field="duration",
            positive=True,
        )
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "start_param", start_param)
        object.__setattr__(self, "every", every)
        object.__setattr__(self, "every_param", every_param)
        object.__setattr__(self, "duration", duration)
        object.__setattr__(self, "duration_param", duration_param)
        if not self.before and not self.during and not self.after:
            raise InterventionScheduleError("protocol must include at least one before, during, or after operation.")
        for op in self.before:
            if not isinstance(op, InterventionProtocolOperation) or str(op.phase) != "before":
                raise InterventionScheduleError("protocol before entries must be before operations.")
        for op in self.during:
            if not isinstance(op, InterventionProtocolOperation) or str(op.phase) != "during":
                raise InterventionScheduleError("protocol during entries must be during operations.")
        for op in self.after:
            if not isinstance(op, InterventionProtocolOperation) or str(op.phase) != "after":
                raise InterventionScheduleError("protocol after entries must be after operations.")

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "InterventionProtocol":
        _reject_unknown_payload_keys(payload, _PROTOCOL_PAYLOAD_KEYS, context="protocol")
        kind = str(payload.get("kind") or "").strip().lower()
        if kind != "repeat":
            raise InterventionScheduleError(f"Unsupported intervention protocol kind: {kind!r}.")
        name = str(payload.get("name") or "").strip()
        if not name:
            raise InterventionScheduleError("protocol name must not be empty.")
        count = _positive_int_from_payload(payload, field="count")
        start, start_param = _payload_scalar(payload, field="start", nonnegative=True)
        every, every_param = _payload_scalar(payload, field="every", positive=True)
        duration, duration_param = _payload_scalar(payload, field="duration", positive=True)
        before = _protocol_operations_from_payload(payload.get("before"), phase="before")
        during = _protocol_operations_from_payload(payload.get("during"), phase="during")
        after = _protocol_operations_from_payload(payload.get("after"), phase="after")
        if not before and not during and not after:
            raise InterventionScheduleError("protocol must include at least one before, during, or after operation.")
        return cls(
            kind=kind,
            name=name,
            start=start,
            every=every,
            duration=duration,
            count=count,
            before=before,
            during=during,
            after=after,
            start_param=start_param,
            every_param=every_param,
            duration_param=duration_param,
            metadata=_metadata_from_payload(payload),
        )

    @property
    def is_parameterized(self) -> bool:
        return bool(
            self.start_param
            or self.every_param
            or self.duration_param
            or any(op.is_parameterized for op in self.before)
            or any(op.is_parameterized for op in self.during)
            or any(op.is_parameterized for op in self.after)
        )

    def resolve_parameters(self, parameter_values: Mapping[str, Any]) -> "InterventionProtocol":
        return InterventionProtocol(
            kind=self.kind,
            name=self.name,
            start=_resolve_scalar(self.start, self.start_param, parameter_values, field="start", nonnegative=True),
            every=_resolve_scalar(self.every, self.every_param, parameter_values, field="every", positive=True),
            duration=_resolve_scalar(
                self.duration,
                self.duration_param,
                parameter_values,
                field="duration",
                positive=True,
            ),
            count=int(self.count),
            before=tuple(op.resolve_parameters(parameter_values) for op in self.before),
            during=tuple(op.resolve_parameters(parameter_values) for op in self.during),
            after=tuple(op.resolve_parameters(parameter_values) for op in self.after),
            metadata=tuple(self.metadata),
        )

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"kind": str(self.kind), "name": str(self.name), "count": int(self.count)}
        _write_scalar_payload(payload, field="start", value=self.start, parameter=self.start_param)
        _write_scalar_payload(payload, field="every", value=self.every, parameter=self.every_param)
        _write_scalar_payload(payload, field="duration", value=self.duration, parameter=self.duration_param)
        if self.before:
            payload["before"] = [op.to_payload() for op in self.before]
        if self.during:
            payload["during"] = [op.to_payload() for op in self.during]
        if self.after:
            payload["after"] = [op.to_payload() for op in self.after]
        _write_metadata_payload(payload, self.metadata)
        return payload


def _protocol_operations_from_payload(value: object, *, phase: str) -> tuple[InterventionProtocolOperation, ...]:
    if value is None or value == "":
        return ()
    if not isinstance(value, (list, tuple)):
        raise InterventionScheduleError(f"protocol {phase} operations must be a list.")
    operations: list[InterventionProtocolOperation] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise InterventionScheduleError(f"protocol {phase} operations must be mappings.")
        operations.append(InterventionProtocolOperation.from_payload(item, phase=phase))
    return tuple(operations)


def _protocol_operation_sort_key(operation: InterventionProtocolOperation) -> tuple[str, str, str, bytes]:
    return (str(operation.phase), str(operation.species), str(operation.op), _sort_payload_bytes(operation))


def _parse_protocol_operations(value: str, *, phase: str) -> list[dict[str, Any]]:
    operations: list[dict[str, Any]] = []
    for raw in str(value or "").split("|"):
        part = raw.strip()
        if not part:
            continue
        pieces = [piece.strip() for piece in part.split(":")]
        if len(pieces) < 2:
            raise InterventionScheduleError(f"Invalid protocol {phase} operation: {part!r}.")
        payload: dict[str, Any] = {"species": pieces[1]}
        if phase == "during":
            payload["kind"] = pieces[0].lower()
        else:
            payload["op"] = pieces[0].lower()
        for field in pieces[2:]:
            if "=" not in field:
                raise InterventionScheduleError(f"Invalid protocol operation field: {field!r}.")
            key, raw_value = field.split("=", 1)
            key_s = str(key).strip().lower()
            if key_s in payload:
                raise InterventionScheduleError(f"Duplicate protocol operation field: {key_s!r}.")
            payload[key_s] = raw_value.strip()
        operations.append(payload)
    return operations


@dataclass(frozen=True, slots=True)
class InterventionSchedule:
    instant_events: tuple[InterventionInstantEvent, ...] = ()
    repeated_events: tuple[InterventionRepeatedEvent, ...] = ()
    trigger_events: tuple[InterventionTriggerEvent, ...] = ()
    intervals: tuple[InterventionInterval, ...] = ()
    repeated_intervals: tuple[InterventionRepeatedInterval, ...] = ()
    protocols: tuple[InterventionProtocol, ...] = ()
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "instant_events", tuple(self.instant_events))
        object.__setattr__(self, "repeated_events", tuple(self.repeated_events))
        object.__setattr__(self, "trigger_events", tuple(self.trigger_events))
        object.__setattr__(self, "intervals", tuple(self.intervals))
        object.__setattr__(self, "repeated_intervals", tuple(self.repeated_intervals))
        object.__setattr__(self, "protocols", tuple(self.protocols))
        object.__setattr__(self, "metadata", _metadata_tuple(self.metadata))
        for event in self.instant_events:
            if not isinstance(event, InterventionInstantEvent):
                raise InterventionScheduleError("instant_events entries must be InterventionInstantEvent.")
        for event in self.repeated_events:
            if not isinstance(event, InterventionRepeatedEvent):
                raise InterventionScheduleError("repeated_events entries must be InterventionRepeatedEvent.")
        for event in self.trigger_events:
            if not isinstance(event, InterventionTriggerEvent):
                raise InterventionScheduleError("trigger_events entries must be InterventionTriggerEvent.")
        for interval in self.intervals:
            if not isinstance(interval, InterventionInterval):
                raise InterventionScheduleError("intervals entries must be InterventionInterval.")
        for interval in self.repeated_intervals:
            if not isinstance(interval, InterventionRepeatedInterval):
                raise InterventionScheduleError("repeated_intervals entries must be InterventionRepeatedInterval.")
        for protocol in self.protocols:
            if not isinstance(protocol, InterventionProtocol):
                raise InterventionScheduleError("protocols entries must be InterventionProtocol.")
        object.__setattr__(self, "instant_events", tuple(sorted(self.instant_events, key=_instant_sort_key)))
        object.__setattr__(self, "repeated_events", tuple(sorted(self.repeated_events, key=_repeated_sort_key)))
        object.__setattr__(self, "trigger_events", tuple(sorted(self.trigger_events, key=_trigger_sort_key)))
        object.__setattr__(self, "intervals", tuple(sorted(self.intervals, key=_interval_sort_key)))
        object.__setattr__(
            self,
            "repeated_intervals",
            tuple(sorted(self.repeated_intervals, key=_repeated_interval_sort_key)),
        )
        object.__setattr__(self, "protocols", tuple(sorted(self.protocols, key=_protocol_sort_key)))
        if self.metadata and self._has_no_operations():
            raise InterventionScheduleError("Intervention schedule metadata requires at least one intervention.")
        self._validate_conflicts()

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | None) -> "InterventionSchedule":
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise InterventionScheduleError("Intervention schedule payload must be a mapping.")
        _reject_unknown_payload_keys(payload, _SCHEDULE_PAYLOAD_KEYS, context="intervention schedule")
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
        repeated_intervals: list[InterventionRepeatedInterval] = []
        for item in payload.get("repeated_intervals") or ():
            if not isinstance(item, Mapping):
                raise InterventionScheduleError("repeated_intervals entries must be mappings.")
            repeated_intervals.append(InterventionRepeatedInterval.from_payload(item))
        protocols: list[InterventionProtocol] = []
        for item in payload.get("protocols") or ():
            if not isinstance(item, Mapping):
                raise InterventionScheduleError("protocols entries must be mappings.")
            protocols.append(InterventionProtocol.from_payload(item))
        schedule = cls(
            instant_events=tuple(instant_events),
            repeated_events=tuple(repeated_events),
            trigger_events=tuple(trigger_events),
            intervals=tuple(intervals),
            repeated_intervals=tuple(repeated_intervals),
            protocols=tuple(protocols),
            metadata=_metadata_from_payload(payload),
        )
        return schedule

    def is_empty(self) -> bool:
        return self._has_no_operations()

    def _has_no_operations(self) -> bool:
        return (
            not self.instant_events
            and not self.repeated_events
            and not self.trigger_events
            and not self.intervals
            and not self.repeated_intervals
            and not self.protocols
        )

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "instant_events": [event.to_payload() for event in self.instant_events],
            "repeated_events": [event.to_payload() for event in self.repeated_events],
            "trigger_events": [event.to_payload() for event in self.trigger_events],
            "intervals": [interval.to_payload() for interval in self.intervals],
        }
        if self.repeated_intervals:
            payload["repeated_intervals"] = [interval.to_payload() for interval in self.repeated_intervals]
        if self.protocols:
            payload["protocols"] = [protocol.to_payload() for protocol in self.protocols]
        _write_metadata_payload(payload, self.metadata)
        return payload

    @property
    def is_parameterized(self) -> bool:
        return (
            any(event.is_parameterized for event in self.instant_events)
            or any(event.is_parameterized for event in self.repeated_events)
            or any(event.is_parameterized for event in self.trigger_events)
            or any(interval.is_parameterized for interval in self.intervals)
            or any(interval.is_parameterized for interval in self.repeated_intervals)
            or any(protocol.is_parameterized for protocol in self.protocols)
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
            repeated_intervals=tuple(interval.resolve_parameters(values) for interval in self.repeated_intervals),
            protocols=tuple(protocol.resolve_parameters(values) for protocol in self.protocols),
            metadata=tuple(self.metadata),
        )
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
        for interval in self.repeated_intervals:
            if interval.species not in available:
                raise InterventionScheduleError(f"Unknown intervention species: {interval.species}")
        for protocol in self.protocols:
            for operation in protocol.before + protocol.during + protocol.after:
                if operation.species not in available:
                    raise InterventionScheduleError(f"Unknown intervention species: {operation.species}")

    def _validate_conflicts(self) -> None:
        absolute_by_key: dict[tuple[object, str], InterventionInstantEvent] = {}
        delta_keys: set[tuple[object, str]] = set()
        for event in _validation_instant_events(self):
            key = (_scalar_identity(event.time, event.time_param), str(event.species))
            if not _instant_event_is_absolute(event):
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
            if existing is not None and _instant_events_conflict(existing, event):
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


def _instant_event_execution_payload(event: InterventionInstantEvent) -> dict[str, Any]:
    payload = event.to_payload()
    payload.pop("metadata", None)
    return payload


def _instant_event_is_absolute(event: object) -> bool:
    return str(getattr(event, "op", "")) in _ABSOLUTE_INSTANT_OPS


def _instant_events_conflict(left: InterventionInstantEvent, right: InterventionInstantEvent) -> bool:
    if _instant_event_is_absolute(left) and _instant_event_is_absolute(right):
        return _instant_event_execution_payload(left) != _instant_event_execution_payload(right)
    return _instant_event_is_absolute(left) or _instant_event_is_absolute(right)


def _format_scalar_identity(value: object) -> str:
    if isinstance(value, tuple) and len(value) == 2 and value[0] == "param":
        return str(value[1])
    if isinstance(value, tuple) and len(value) == 2 and value[0] == "fixed":
        try:
            return f"{float(value[1]):g}"
        except (TypeError, ValueError, OverflowError):
            return str(value[1])
    return str(value)


def _sort_payload_bytes(value: object) -> bytes:
    to_payload = getattr(value, "to_payload", None)
    if callable(to_payload):
        return _canonical_json_bytes(to_payload())
    return _canonical_json_bytes(str(value))


def _instant_sort_key(event: InterventionInstantEvent) -> tuple[float, str, int, str, bytes]:
    op_order = {"set": 0, "clear": 0, "remove": 1, "add": 2}
    time = float(event.time) if event.time is not None else math.inf
    return (time, str(event.species), op_order.get(event.op, 99), str(event.op), _sort_payload_bytes(event))


def _interval_sort_key(interval: InterventionInterval) -> tuple[float, float, str, str, bytes]:
    start = float(interval.start) if interval.start is not None else math.inf
    end = float(interval.end) if interval.end is not None else math.inf
    return (start, end, str(interval.species), str(interval.kind), _sort_payload_bytes(interval))


def _repeated_sort_key(event: InterventionRepeatedEvent) -> tuple[float, str, str, bytes]:
    start = float(event.start) if event.start is not None else math.inf
    return (start, str(event.species), str(event.op), _sort_payload_bytes(event))


def _trigger_sort_key(event: InterventionTriggerEvent) -> tuple[str, str, str, str, bytes]:
    return (str(event.trigger_species), str(event.species), str(event.direction), str(event.op), _sort_payload_bytes(event))


def _repeated_interval_sort_key(interval: InterventionRepeatedInterval) -> tuple[float, str, str, bytes]:
    start = float(interval.start) if interval.start is not None else math.inf
    return (start, str(interval.species), str(interval.kind), _sort_payload_bytes(interval))


def _protocol_sort_key(protocol: InterventionProtocol) -> tuple[float, str, bytes]:
    start = float(protocol.start) if protocol.start is not None else math.inf
    return (start, str(protocol.name), _sort_payload_bytes(protocol))


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


def _require_executable_primitive_schedule(schedule: InterventionSchedule) -> None:
    if schedule.repeated_intervals or schedule.protocols:
        raise InterventionScheduleError(
            "Declarative intervention schedules must be compiled with compile_intervention_schedule "
            "before primitive execution helpers are used."
        )


def coerce_intervention_schedule(value: object) -> InterventionSchedule | None:
    if value is None:
        return None
    if isinstance(value, InterventionSchedule):
        return None if value.is_empty() else value
    if isinstance(value, Mapping):
        schedule = InterventionSchedule.from_payload(value)
        return None if schedule.is_empty() else schedule
    raise InterventionScheduleError("Intervention schedule must be a mapping or InterventionSchedule.")


def intervention_schedule_identity_fingerprints(
    schedule: "InterventionSchedule | Mapping[str, Any] | None",
) -> tuple[str, str]:
    schedule_obj = coerce_intervention_schedule(schedule)
    if schedule_obj is None:
        return "", ""
    executable_fingerprint = ""
    if not schedule_obj.is_parameterized:
        from kindred.core.intervention_schedule_compiler import compile_intervention_schedule

        executable_fingerprint = compile_intervention_schedule(schedule_obj).executable_fingerprint
    return str(schedule_obj.fingerprint or ""), str(executable_fingerprint or "")


def parse_intervention_schedule_from_dsl(text: str) -> InterventionSchedule | None:
    instant_events: list[dict[str, Any]] = []
    repeated_events: list[dict[str, Any]] = []
    trigger_events: list[dict[str, Any]] = []
    intervals: list[dict[str, Any]] = []
    repeated_intervals: list[dict[str, Any]] = []
    protocols: list[dict[str, Any]] = []
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
                trigger = dict(fields)
                trigger.pop("op", None)
                trigger_events.append(trigger)
            elif op in _INTERVAL_KINDS:
                if "kind" in fields:
                    raise InterventionScheduleError(
                        "Interval directive must not include kind when op supplies the interval kind."
                    )
                interval = dict(fields)
                interval["kind"] = op
                interval.pop("op", None)
                intervals.append(interval)
            elif op == "repeated_interval":
                interval = dict(fields)
                interval["kind"] = str(interval.get("kind") or "").strip().lower()
                interval.pop("op", None)
                repeated_intervals.append(interval)
            elif op == "protocol":
                protocol = dict(fields)
                protocol["kind"] = str(protocol.get("kind") or "").strip().lower()
                for phase in ("before", "during", "after"):
                    if isinstance(protocol.get(phase), str):
                        protocol[phase] = _parse_protocol_operations(str(protocol[phase]), phase=phase)
                protocol.pop("op", None)
                protocols.append(protocol)
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
            "repeated_intervals": repeated_intervals,
            "protocols": protocols,
        }
    )
    return None if schedule.is_empty() else schedule


def _canonicalize_schedule_parameter_names(value: object, *, mechanism_namespace: object) -> object:
    if isinstance(value, Mapping):
        out: dict[str, object] = {}
        for key, item in value.items():
            key_s = str(key)
            if key_s.endswith("_param") and str(item or "").strip():
                raw_name = str(item)
                resolution = mechanism_namespace.resolve(raw_name)  # type: ignore[attr-defined]
                if resolution.canonical_name is not None:
                    out[key_s] = str(resolution.canonical_name)
                    continue
                invalid_message = mechanism_namespace.invalid_protected_indexed_identifier_message(raw_name)  # type: ignore[attr-defined]
                if invalid_message is not None:
                    raise InterventionScheduleError(invalid_message)
                out[key_s] = raw_name
                continue
            out[key_s] = _canonicalize_schedule_parameter_names(item, mechanism_namespace=mechanism_namespace)
        return out
    if isinstance(value, list):
        return [_canonicalize_schedule_parameter_names(item, mechanism_namespace=mechanism_namespace) for item in value]
    if isinstance(value, tuple):
        return tuple(_canonicalize_schedule_parameter_names(item, mechanism_namespace=mechanism_namespace) for item in value)
    return value


def normalized_intervention_schedule_payload(
    schedule: "InterventionSchedule | Mapping[str, Any] | None",
    *,
    mechanism_namespace: object,
) -> dict[str, Any] | None:
    schedule_obj = coerce_intervention_schedule(schedule)
    if schedule_obj is None:
        return None
    payload = _canonicalize_schedule_parameter_names(
        schedule_obj.to_payload(),
        mechanism_namespace=mechanism_namespace,
    )
    return InterventionSchedule.from_payload(payload).to_payload()


def normalized_intervention_schedule_identity_fingerprints(
    schedule: "InterventionSchedule | Mapping[str, Any] | None",
    *,
    mechanism_namespace: object,
) -> tuple[str, str]:
    payload = normalized_intervention_schedule_payload(
        schedule,
        mechanism_namespace=mechanism_namespace,
    )
    if payload is None:
        return "", ""
    return intervention_schedule_identity_fingerprints(InterventionSchedule.from_payload(payload))


def normalized_intervention_schedule_identity_fingerprints_from_dsl_text(text: str) -> tuple[str, str]:
    from kindred.core.simulator.dsl import parse_dsl_to_mechanism
    from kindred.core.simulator.parameter_namespace import build_namespace_from_mechanism

    schedule = parse_intervention_schedule_from_dsl(text)
    if schedule is None:
        return "", ""
    mechanism = parse_dsl_to_mechanism(str(text or ""), initials={})
    namespace = build_namespace_from_mechanism(mechanism)
    return normalized_intervention_schedule_identity_fingerprints(
        schedule,
        mechanism_namespace=namespace,
    )


def _parse_directive_fields(rest: str, *, line_no: int) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for chunk in str(rest or "").split(";"):
        part = chunk.strip()
        if not part:
            continue
        if "=" not in part:
            raise InterventionScheduleError(f"Line {line_no}: invalid intervention field {part!r}.")
        key, value = part.split("=", 1)
        key_s = str(key).strip().lower()
        if key_s in fields:
            raise InterventionScheduleError(f"Duplicate intervention field: {key_s!r}.")
        fields[key_s] = value.strip()
    return fields


def active_interval_boundaries(schedule: InterventionSchedule, *, t0: float, t1: float) -> list[float]:
    if schedule.is_parameterized:
        raise InterventionScheduleError("Parameterized intervention schedule must be resolved before solving.")
    _require_executable_primitive_schedule(schedule)
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
    _require_executable_primitive_schedule(schedule)
    return tuple(event for event in _expanded_instant_events(schedule) if float(event.time) == float(time))


def intervals_active_at(schedule: InterventionSchedule, time: float) -> tuple[InterventionInterval, ...]:
    if schedule.is_parameterized:
        raise InterventionScheduleError("Parameterized intervention schedule must be resolved before solving.")
    _require_executable_primitive_schedule(schedule)
    return tuple(
        interval
        for interval in schedule.intervals
        if float(interval.start) <= float(time) < float(interval.end)
    )
