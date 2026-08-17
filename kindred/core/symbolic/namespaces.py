from __future__ import annotations

from dataclasses import dataclass

from kindred.core.simulator.parameter_algebra_spec import ParameterAlgebraSpec
from kindred.core.simulator.parameter_namespace import is_protected_indexed_identifier

from .errors import UnsupportedSymbolicExpressionError


_PROTECTED_RUNTIME_SYMBOLS = {"T"}


@dataclass(frozen=True, slots=True)
class SymbolicParameterNamespaceContext:
    scalar_input_names: frozenset[str]
    assignment_names: frozenset[str]
    canonical_names: frozenset[str]

    @classmethod
    def from_spec(cls, spec: ParameterAlgebraSpec) -> "SymbolicParameterNamespaceContext":
        return cls(
            scalar_input_names=frozenset(str(name) for name in spec.scalar_input_names),
            assignment_names=frozenset(str(name) for name in spec.param_assignment_names()),
            canonical_names=frozenset(spec.mechanism_namespace.canonical_names),
        )

    def __post_init__(self) -> None:
        object.__setattr__(self, "scalar_input_names", frozenset(str(name) for name in self.scalar_input_names))
        object.__setattr__(self, "assignment_names", frozenset(str(name) for name in self.assignment_names))
        object.__setattr__(self, "canonical_names", frozenset(str(name) for name in self.canonical_names))

    def resolve_identifier(self, name: object) -> str:
        name_s = str(name)
        if name_s in _PROTECTED_RUNTIME_SYMBOLS:
            raise UnsupportedSymbolicExpressionError(
                f"Protected runtime symbol {name_s!r} is not supported in symbolic parameter expressions."
            )
        if name_s in self.canonical_names:
            return name_s
        if is_protected_indexed_identifier(name_s):
            raise UnsupportedSymbolicExpressionError(
                f"Protected indexed mechanism parameter {name_s!r} is not present in the mechanism namespace."
            )
        if name_s in self.scalar_input_names:
            return name_s
        if name_s in self.assignment_names:
            return name_s
        raise UnsupportedSymbolicExpressionError(f"Unknown symbolic identifier {name_s!r}.")

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
