from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .artifacts import symbolic_jacobian_identity_payload
from .errors import UnsupportedSymbolicExpressionError
from .namespaces import symbolic_status_payload


@dataclass(frozen=True, slots=True)
class SymbolicJacobianExecution:
    jacobian_func: Any = None
    jac_sparsity: Any = None
    identity: Mapping[str, Any] | None = None
    status: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        identity = dict(self.identity or {}) if isinstance(self.identity, Mapping) else None
        status = dict(self.status or {}) if isinstance(self.status, Mapping) else None
        state = str((status or {}).get("state") or "")
        if self.jacobian_func is None:
            if identity:
                raise ValueError("Symbolic Jacobian identity requires an executable Jacobian.")
            if state in {"supported", "partially_disabled"}:
                raise ValueError("Executable symbolic Jacobian state requires a Jacobian callable.")
            if state in {"unsupported", "disabled"}:
                object.__setattr__(self, "jac_sparsity", None)
        elif state in {"unsupported", "disabled"}:
            raise ValueError("Unsupported or disabled symbolic Jacobian state cannot carry a callable.")
        object.__setattr__(self, "identity", identity)
        object.__setattr__(self, "status", status)

    @property
    def has_executable_jacobian(self) -> bool:
        return self.jacobian_func is not None

    @classmethod
    def absent(cls) -> "SymbolicJacobianExecution":
        return cls()

    @classmethod
    def supported(
        cls,
        *,
        jacobian_func: Any,
        identity: Mapping[str, Any],
        jac_sparsity: Any = None,
    ) -> "SymbolicJacobianExecution":
        return cls(
            jacobian_func=jacobian_func,
            jac_sparsity=jac_sparsity,
            identity=dict(identity),
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
    def from_support_status(cls, status: Mapping[str, Any]) -> "SymbolicJacobianExecution":
        status_payload = dict(status)
        if status_payload.get("state") == "supported":
            raise ValueError("Supported preflight status is not an executable symbolic Jacobian.")
        return cls(status=status_payload)

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

    @classmethod
    def from_request_fields(
        cls,
        *,
        jacobian_func: Any,
        jac_sparsity: Any,
        status: Mapping[str, Any] | None,
    ) -> "SymbolicJacobianExecution":
        identity = symbolic_jacobian_identity_payload(jacobian_func)
        if identity:
            return cls(
                jacobian_func=jacobian_func,
                jac_sparsity=jac_sparsity,
                identity=dict(identity),
                status=dict(status or {}) if isinstance(status, Mapping) else None,
            )
        status_payload = dict(status or {}) if isinstance(status, Mapping) else None
        if status_payload and status_payload.get("state") == "supported":
            status_payload = symbolic_status_payload(
                kind="jacobian",
                state="unsupported",
                code="missing-executable",
                reason="Symbolic Jacobian status cannot be supported without an executable Jacobian.",
            )
        return cls(status=status_payload)

    def with_runtime_disabled(
        self,
        *,
        partially: bool,
        code: str,
        reason: str,
    ) -> "SymbolicJacobianExecution":
        state = "partially_disabled" if partially and self.has_executable_jacobian else "disabled"
        return SymbolicJacobianExecution(
            jacobian_func=self.jacobian_func if state == "partially_disabled" else None,
            jac_sparsity=self.jac_sparsity if state == "partially_disabled" else None,
            identity=dict(self.identity or {}) if state == "partially_disabled" and self.identity else None,
            status=symbolic_status_payload(
                kind="jacobian",
                state=state,
                code=code,
                reason=reason,
            ),
        )

    def to_request_kwargs(self) -> dict[str, Any]:
        return {
            "jacobian_func": self.jacobian_func,
            "jac_sparsity": self.jac_sparsity,
            "symbolic_jacobian_status": dict(self.status) if self.status else None,
        }

    def metadata_kwargs(self) -> dict[str, Any]:
        return {
            "symbolic_jacobian_identity": dict(self.identity) if self.identity else None,
            "symbolic_jacobian_status": dict(self.status) if self.status else None,
        }

    def provenance_fields(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "symbolic_jacobian": self.has_executable_jacobian,
            "jacobian_sparsity_hint": self.jac_sparsity is not None,
        }
        if self.identity:
            payload["symbolic_jacobian_identity"] = dict(self.identity)
        if self.status:
            payload["symbolic_jacobian_status"] = dict(self.status)
        return payload
