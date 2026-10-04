from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .errors import UnsupportedSymbolicExpressionError


def symbolic_status_payload(
    *,
    kind: str,
    state: str,
    code: str,
    reason: str,
) -> dict[str, str]:
    return {
        "kind": str(kind),
        "state": str(state),
        "code": str(code),
        "reason": str(reason),
    }


@dataclass(frozen=True, slots=True)
class SymbolicJacobianExecution:
    jacobian_func: Any = None
    status: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        status = dict(self.status or {}) if isinstance(self.status, Mapping) else None
        state = str((status or {}).get("state") or "")
        if self.jacobian_func is None:
            if state == "supported":
                raise ValueError("Executable symbolic Jacobian state requires a Jacobian callable.")
        else:
            if state in {"unsupported", "disabled"}:
                raise ValueError(
                    "Unsupported or disabled symbolic Jacobian state cannot carry a callable."
                )
        object.__setattr__(self, "status", status)

    @property
    def has_executable_jacobian(self) -> bool:
        return self.jacobian_func is not None

    @classmethod
    def supported(
        cls,
        *,
        jacobian_func: Any,
    ) -> "SymbolicJacobianExecution":
        return cls(
            jacobian_func=jacobian_func,
            status=symbolic_status_payload(
                kind="jacobian",
                state="supported",
                code="supported",
                reason="Symbolic Jacobian supported.",
            ),
        )

    @classmethod
    def unsupported(
        cls,
        *,
        code: str,
        reason: str,
    ) -> "SymbolicJacobianExecution":
        return cls(
            status=symbolic_status_payload(
                kind="jacobian",
                state="unsupported",
                code=code,
                reason=reason,
            )
        )

    @classmethod
    def disabled(
        cls,
        *,
        code: str,
        reason: str,
    ) -> "SymbolicJacobianExecution":
        return cls(
            status=symbolic_status_payload(
                kind="jacobian",
                state="disabled",
                code=code,
                reason=reason,
            )
        )

    @classmethod
    def from_bind_failure(
        cls,
        *,
        classified_status: Mapping[str, Any],
        exc: UnsupportedSymbolicExpressionError,
    ) -> "SymbolicJacobianExecution":
        status_payload = dict(classified_status or {})
        if status_payload.get("state") == "unsupported":
            return cls(status=status_payload)
        return cls.unsupported(
            code="binding-failed",
            reason=str(exc) or "Symbolic Jacobian binding failed.",
        )

    def provenance_fields(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "symbolic_jacobian": self.has_executable_jacobian,
        }
        if self.status:
            payload["symbolic_jacobian_status"] = dict(self.status)
        return payload
