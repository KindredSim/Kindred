from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .errors import UnsupportedSymbolicExpressionError
from .namespaces import symbolic_status_payload


def normalize_jacobian_sparsity(value: Any) -> np.ndarray | None:
    """Materialize one SciPy-compatible immutable boolean sparsity matrix."""

    if value is None:
        return None
    try:
        dense = value.toarray()
    except AttributeError:
        dense = value
    matrix = np.asarray(dense, dtype=bool)
    if matrix.ndim != 2:
        raise ValueError("Jacobian sparsity must be a two-dimensional matrix.")
    matrix = matrix.copy()
    matrix.setflags(write=False)
    return matrix


@dataclass(frozen=True, slots=True)
class SymbolicJacobianExecution:
    jacobian_func: Any = None
    jac_sparsity: Any = None
    status: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        status = dict(self.status or {}) if isinstance(self.status, Mapping) else None
        state = str((status or {}).get("state") or "")
        if self.jacobian_func is None:
            if state in {"supported", "partially_disabled"}:
                raise ValueError("Executable symbolic Jacobian state requires a Jacobian callable.")
        else:
            if state in {"unsupported", "disabled"}:
                raise ValueError(
                    "Unsupported or disabled symbolic Jacobian state cannot carry a callable."
                )
        jac_sparsity = normalize_jacobian_sparsity(self.jac_sparsity)
        object.__setattr__(self, "jac_sparsity", jac_sparsity)
        object.__setattr__(self, "status", status)

    @property
    def has_executable_jacobian(self) -> bool:
        return self.jacobian_func is not None

    @classmethod
    def supported(
        cls,
        *,
        jacobian_func: Any,
        jac_sparsity: Any = None,
    ) -> "SymbolicJacobianExecution":
        return cls(
            jacobian_func=jacobian_func,
            jac_sparsity=jac_sparsity,
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
        jac_sparsity: Any = None,
    ) -> "SymbolicJacobianExecution":
        return cls(
            jac_sparsity=jac_sparsity,
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
            jac_sparsity=self.jac_sparsity,
            status=symbolic_status_payload(
                kind="jacobian",
                state=state,
                code=code,
                reason=reason,
            ),
        )

    def metadata_kwargs(self) -> dict[str, Any]:
        return {
            "symbolic_jacobian_status": dict(self.status) if self.status else None,
        }

    def provenance_fields(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "symbolic_jacobian": self.has_executable_jacobian,
            "jacobian_sparsity_hint": self.jac_sparsity is not None,
        }
        if self.status:
            payload["symbolic_jacobian_status"] = dict(self.status)
        return payload
