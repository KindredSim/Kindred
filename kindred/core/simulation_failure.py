from __future__ import annotations

from typing import Any, Dict, Mapping, MutableMapping, Optional

from kindred.core.exceptions import KindredError, ErrorContext, SimulationCancelled


SimulationFailure = Dict[str, Any]


def serialize_error_context(context: ErrorContext | Mapping[str, Any] | None) -> Optional[Dict[str, Any]]:
    if context is None:
        return None
    if isinstance(context, Mapping):
        line = context.get("line")
        col = context.get("col")
        line_text = context.get("line_text")
        file_path = context.get("file_path")
        stack_trace = context.get("stack_trace")
    else:
        line = getattr(context, "line", None)
        col = getattr(context, "col", None)
        line_text = getattr(context, "line_text", None)
        file_path = getattr(context, "file_path", None)
        stack_trace = getattr(context, "stack_trace", None)
    payload = {
        "line": int(line) if line is not None else None,
        "col": int(col) if col is not None else None,
        "line_text": str(line_text) if line_text is not None else None,
        "file_path": str(file_path) if file_path is not None else None,
        "stack_trace": str(stack_trace) if stack_trace is not None else None,
    }
    if not any(value is not None for value in payload.values()):
        return None
    return payload


def build_simulation_failure(
    kind: str,
    message: str,
    *,
    code: Optional[str] = None,
    context: ErrorContext | Mapping[str, Any] | None = None,
    details: Mapping[str, Any] | None = None,
    exc_type: Optional[str] = None,
) -> SimulationFailure:
    return {
        "kind": str(kind or "simulation_error"),
        "code": str(code) if code is not None else None,
        "message": str(message or ""),
        "context": serialize_error_context(context),
        "details": dict(details or {}),
        "exc_type": str(exc_type) if exc_type is not None else None,
    }


def simulation_failure_from_exception(
    exc: BaseException,
    *,
    kind: Optional[str] = None,
    details: Mapping[str, Any] | None = None,
) -> SimulationFailure:
    if isinstance(exc, SimulationCancelled):
        resolved_kind = "cancelled"
    elif kind is not None:
        resolved_kind = str(kind)
    elif isinstance(exc, KindredError):
        resolved_kind = "simulation_error"
    else:
        resolved_kind = "simulation_error"

    combined_details: MutableMapping[str, Any] = {}
    if isinstance(exc, KindredError):
        combined_details.update(dict(getattr(exc, "details", {}) or {}))
    combined_details.update(dict(details or {}))

    message = str(getattr(exc, "message", None) or str(exc) or exc.__class__.__name__)
    code = getattr(exc, "code", None)
    context = getattr(exc, "context", None)
    return build_simulation_failure(
        resolved_kind,
        message,
        code=str(code) if code is not None else None,
        context=context,
        details=combined_details,
        exc_type=exc.__class__.__name__,
    )


def coerce_simulation_failure(value: object) -> SimulationFailure:
    if isinstance(value, dict) and "kind" in value and "message" in value:
        return build_simulation_failure(
            str(value.get("kind") or "simulation_error"),
            str(value.get("message") or ""),
            code=str(value.get("code")) if value.get("code") is not None else None,
            context=value.get("context") if isinstance(value.get("context"), Mapping) else None,
            details=value.get("details") if isinstance(value.get("details"), Mapping) else None,
            exc_type=str(value.get("exc_type")) if value.get("exc_type") is not None else None,
        )
    if isinstance(value, BaseException):
        return simulation_failure_from_exception(value)
    text = str(value or "")
    legacy_kind = "cancelled" if "cancelled" in text.lower() else "simulation_error"
    return build_simulation_failure(legacy_kind, text)


def is_cancelled_failure(value: object) -> bool:
    payload = coerce_simulation_failure(value)
    return str(payload.get("kind") or "") == "cancelled"


def simulation_failure_user_message(value: object) -> str:
    payload = coerce_simulation_failure(value)
    kind = str(payload.get("kind") or "")
    message = str(payload.get("message") or "").strip()
    details = payload.get("details") if isinstance(payload.get("details"), Mapping) else {}
    stage = str(details.get("stage") or "").strip().lower()
    if kind == "cancelled":
        return message or "Simulation cancelled by user"
    if kind == "preparation_error":
        prefixes = {
            "parse": "Failed to parse mechanism",
            "prepared_payload": "Prepared simulation payload invalid",
            "solver_config": "Invalid solver configuration",
            "parameter_algebra": "Parameter algebra failed",
            "ode_build": "Failed to build ODE system",
            "temperature_schedule": "Invalid temperature schedule",
        }
        prefix = prefixes.get(stage, "Simulation preparation failed")
        return f"{prefix}:\n\n{message}" if message else prefix
    return message or "Simulation failed"


def simulation_failure_detail_text(value: object) -> str:
    payload = coerce_simulation_failure(value)
    context = payload.get("context") if isinstance(payload.get("context"), Mapping) else None
    if not isinstance(context, Mapping):
        return ""
    return str(context.get("stack_trace") or "").strip()


def serialize_algebra_error(error: object, *, name: Optional[str] = None) -> Dict[str, Any]:
    context = serialize_error_context(getattr(error, "context", None))
    if context is None:
        context = serialize_error_context(
            {
                "line": getattr(error, "line", None),
                "col": getattr(error, "col", None),
                "line_text": getattr(error, "line_text", None),
            }
        )
    return {
        "kind": "algebra_error",
        "name": str(name) if name is not None else getattr(error, "name", None),
        "exc_type": getattr(error, "exc_type", error.__class__.__name__),
        "message": str(getattr(error, "message", None) or str(error)),
        "code": getattr(error, "code", None),
        "context": context,
        "line": context.get("line") if isinstance(context, Mapping) else None,
        "col": context.get("col") if isinstance(context, Mapping) else None,
        "line_text": context.get("line_text") if isinstance(context, Mapping) else None,
    }
