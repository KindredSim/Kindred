from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping



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
    status: Mapping[str, Any]
    jacobian_func: Any = None

    def __post_init__(self) -> None:
        # Its binder pairs a callable only with a supported state.
        object.__setattr__(self, "status", dict(self.status))

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
    ) -> "SymbolicJacobianExecution":
        # Binding fails exactly when the classifier calls the mechanism unsupported.
        return cls(status=classified_status)

    def provenance_fields(self) -> dict[str, Any]:
        return {
            "symbolic_jacobian": self.has_executable_jacobian,
            "symbolic_jacobian_status": dict(self.status),
        }
