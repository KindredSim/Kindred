"""Canonical immutable factual causes for simulation and fitting boundaries.

``FactualCause`` is the internal terminal/report cause tree.  The object is
deliberately not a mapping: callers inspect typed attributes and call
``to_payload`` only at the JSON boundary.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from kindred.core.exceptions import ErrorContext

__all__ = [
    "FactualCause",
    "factual_cause_from_exception",
]


def freeze_factual_value(value: object) -> object:
    """Recursively freeze factual metadata without inventing a schema."""

    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                str(key): freeze_factual_value(item)
                for key, item in value.items()
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(freeze_factual_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(
            freeze_factual_value(item)
            for item in sorted(value, key=repr)
        )
    return str(value)


def _plain_value(value: object) -> object:
    """Convert frozen metadata at the one explicit JSON boundary."""

    if isinstance(value, Mapping):
        return {
            str(key): _plain_value(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return [_plain_value(item) for item in value]
    return value


def _context_values(context: ErrorContext | Mapping[str, Any] | None) -> Mapping[str, object]:
    if context is None:
        return {}
    if isinstance(context, Mapping):
        source = context
    else:
        source = {
            "line": getattr(context, "line", None),
            "line_text": getattr(context, "line_text", None),
        }
    def value_for(key: str) -> object:
        return source[key] if key in source else None

    values = {
        "line": (
            int(value_for("line"))
            if value_for("line") is not None
            else None
        ),
        "line_text": (
            str(value_for("line_text"))
            if value_for("line_text") is not None
            else None
        ),
    }
    return {
        key: value
        for key, value in values.items()
        if value is not None
    }


@dataclass(frozen=True, slots=True)
class FactualCause:
    """Immutable factual cause tree shared by simulation and Fit reports."""

    domain: str
    origin: str
    code: str | None
    message: str
    context: Mapping[str, object] = field(default_factory=dict)
    details: Mapping[str, object] = field(default_factory=dict)
    cause: "FactualCause | None" = None

    def location(self) -> tuple[object, object]:
        """The first authored line named along this cause chain, as (line, text)."""
        cause: FactualCause | None = self
        while cause is not None:
            if cause.context.get("line") is not None and cause.context.get("line_text") is not None:
                return cause.context["line"], cause.context["line_text"]
            cause = cause.cause
        return None, None

    def __post_init__(self) -> None:
        # Producers pass literal domains and origins and a message (an
        # exception's own text); exception attributes are normalized here.
        domain = str(self.domain).strip()
        origin = str(self.origin).strip()
        message = str(self.message).strip()
        normalized_code = (
            str(self.code).strip()
            if self.code is not None and str(self.code).strip()
            else None
        )
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "code", normalized_code)
        object.__setattr__(self, "message", message)
        object.__setattr__(self, "context", freeze_factual_value(self.context))
        object.__setattr__(self, "details", freeze_factual_value(self.details))

    def to_payload(self) -> dict[str, object]:
        """Serialize this cause tree to mutable JSON-ready data."""

        return {
            "domain": self.domain,
            "origin": self.origin,
            "code": self.code,
            "message": self.message,
            "context": _plain_value(self.context),
            "details": _plain_value(self.details),
            "cause": (
                self.cause.to_payload()
                if self.cause is not None
                else None
            ),
        }


def _linked_exception(exc: BaseException) -> BaseException | None:
    """Select the strongest available Python causal link."""

    explicit = exc.__cause__
    if isinstance(explicit, BaseException):
        return explicit
    contextual = exc.__context__
    return contextual if isinstance(contextual, BaseException) else None


def factual_cause_from_exception(
    exc: BaseException,
    *,
    domain: str = "simulation",
    origin: str = "exception",
    _seen: frozenset[int] | None = None,
) -> FactualCause:
    """Capture typed exception facts and one complete causal chain safely.

    ``__cause__`` takes precedence over ``__context__``.  Every link is visited
    at most once, so malformed custom exception graphs cannot recurse forever.
    """

    seen = frozenset() if _seen is None else _seen
    identity = id(exc)
    linked = _linked_exception(exc)
    nested = (
        factual_cause_from_exception(
            linked,
            domain=domain,
            origin=origin,
            _seen=seen | {identity},
        )
        if linked is not None
        and id(linked) not in seen
        and id(linked) != identity
        else None
    )
    raw_details = getattr(exc, "details", None)
    details = raw_details if isinstance(raw_details, Mapping) else {}
    return FactualCause(
        domain=domain,
        origin=origin,
        code=getattr(exc, "code", None),
        message=str(getattr(exc, "message", None) or "").strip() or str(exc).strip(),
        context=_context_values(getattr(exc, "context", None)),
        details=details,
        cause=nested,
    )
