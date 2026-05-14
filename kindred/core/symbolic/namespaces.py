from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from types import MappingProxyType
from typing import Mapping, Sequence

from kindred.core.simulator.parameter_algebra_spec import ParameterAlgebraSpec, ParameterAssignment
from kindred.core.simulator.parameter_namespace import is_protected_indexed_identifier

from .errors import UnsupportedSymbolicExpressionError


_PROTECTED_RUNTIME_SYMBOLS = {"T", "T0"}


def _fingerprint(payload: Mapping[str, object]) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class SymbolicStateVectorContext:
    species_names: tuple[str, ...]
    symbol_names: tuple[str, ...]
    display_symbols: tuple[str, ...]

    @property
    def kind(self) -> str:
        return "state-vector"

    def to_payload(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "species_names": list(self.species_names),
            "symbol_names": list(self.symbol_names),
            "display_symbols": list(self.display_symbols),
        }


@dataclass(frozen=True, slots=True)
class SymbolicParameterExpressionContext:
    canonical_identifiers: tuple[str, ...] = ()

    @property
    def kind(self) -> str:
        return "parameter-expression"

    @property
    def allows_state_symbols(self) -> bool:
        return False

    def to_payload(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "allows_state_symbols": self.allows_state_symbols,
            "allowed_symbol_kinds": ["parameter", "scalar-parameter"],
            "rejected_symbol_kinds": ["state-concentration", "runtime-temperature", "logical-dynamic"],
            "canonical_identifiers": list(self.canonical_identifiers),
        }


@dataclass(frozen=True, slots=True)
class SymbolicParameterNamespaceContext:
    scalar_input_names: frozenset[str]
    assignment_names: frozenset[str]
    canonical_by_lower: Mapping[str, str]

    @classmethod
    def from_spec(cls, spec: ParameterAlgebraSpec) -> "SymbolicParameterNamespaceContext":
        return cls(
            scalar_input_names=frozenset(str(name) for name in spec.scalar_input_names),
            assignment_names=frozenset(str(name) for name in spec.param_assignment_names()),
            canonical_by_lower=dict(getattr(spec.mechanism_namespace, "canonical_by_lower", {}) or {}),
        )

    def __post_init__(self) -> None:
        object.__setattr__(self, "scalar_input_names", frozenset(str(name) for name in self.scalar_input_names))
        object.__setattr__(self, "assignment_names", frozenset(str(name) for name in self.assignment_names))
        object.__setattr__(
            self,
            "canonical_by_lower",
            MappingProxyType({str(key).lower(): str(value) for key, value in dict(self.canonical_by_lower).items()}),
        )

    def resolve_identifier(self, name: object) -> str:
        name_s = str(name)
        if name_s in _PROTECTED_RUNTIME_SYMBOLS:
            raise UnsupportedSymbolicExpressionError(f"Protected runtime symbol {name_s!r} is not supported in symbolic proof.")
        direct = self.canonical_by_lower.get(name_s.lower())
        if direct is not None:
            return direct
        if is_protected_indexed_identifier(name_s):
            raise UnsupportedSymbolicExpressionError(
                f"Protected indexed mechanism parameter {name_s!r} is not present in the mechanism namespace."
            )
        if name_s in self.scalar_input_names:
            return name_s
        if name_s in self.assignment_names:
            return name_s
        raise UnsupportedSymbolicExpressionError(f"Unknown symbolic identifier {name_s!r}.")

    def to_expression_payload(self, canonical_identifiers: Sequence[object] = ()) -> dict[str, object]:
        return make_parameter_expression_context(canonical_identifiers).to_payload() | {
            "mechanism_parameters": [self.canonical_by_lower[key] for key in sorted(self.canonical_by_lower)],
            "scalar_inputs": sorted(self.scalar_input_names),
            "assignment_names": sorted(self.assignment_names),
        }


@dataclass(frozen=True, slots=True)
class SymbolicProductIdentityProofContext:
    proof_symbols: tuple[str, ...]
    assignments: Mapping[str, ParameterAssignment]
    parameter_namespace: SymbolicParameterNamespaceContext
    proof_kind: str = "wegscheider-parameter-proof"

    def __post_init__(self) -> None:
        object.__setattr__(self, "proof_symbols", tuple(str(name) for name in self.proof_symbols))
        object.__setattr__(self, "assignments", MappingProxyType({str(name): stmt for name, stmt in dict(self.assignments).items()}))

    @classmethod
    def from_spec(
        cls,
        *,
        target_factors: Mapping[str, int],
        spec: ParameterAlgebraSpec,
    ) -> "SymbolicProductIdentityProofContext":
        return cls(
            proof_symbols=tuple(sorted(str(name) for name in target_factors)),
            assignments={str(stmt.name): stmt for stmt in spec.param_statements or []},
            parameter_namespace=SymbolicParameterNamespaceContext.from_spec(spec),
        )

    @property
    def allows_state_symbols(self) -> bool:
        return False

    def to_payload(self) -> dict[str, object]:
        return {
            "kind": self.proof_kind,
            "allows_state_symbols": self.allows_state_symbols,
            "proof_symbols": list(self.proof_symbols),
            "assignment_names": sorted(str(name) for name in self.assignments),
        }


@dataclass(frozen=True, slots=True)
class SymbolicEvaluationSnapshotContext:
    parameter_values: tuple[tuple[str, float], ...]
    fingerprint: str

    @property
    def kind(self) -> str:
        return "evaluation-snapshot"

    def to_payload(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "parameter_values": [[name, f"{value:.17g}"] for name, value in self.parameter_values],
            "fingerprint": self.fingerprint,
        }


def make_state_symbol_context(species_names: Sequence[object]) -> SymbolicStateVectorContext:
    names = tuple(str(name) for name in species_names)
    return SymbolicStateVectorContext(
        species_names=names,
        symbol_names=tuple(f"y_{idx}" for idx, _name in enumerate(names)),
        display_symbols=tuple(f"[{name}]" for name in names),
    )


def make_parameter_expression_context(
    canonical_identifiers: Sequence[object] = (),
) -> SymbolicParameterExpressionContext:
    return SymbolicParameterExpressionContext(
        canonical_identifiers=tuple(str(name) for name in canonical_identifiers),
    )


def make_parameter_namespace_context(spec: ParameterAlgebraSpec) -> SymbolicParameterNamespaceContext:
    return SymbolicParameterNamespaceContext.from_spec(spec)


def make_product_identity_proof_context(
    *,
    target_factors: Mapping[str, int],
    spec: ParameterAlgebraSpec,
) -> SymbolicProductIdentityProofContext:
    return SymbolicProductIdentityProofContext.from_spec(target_factors=target_factors, spec=spec)


def make_evaluation_snapshot_context(
    parameter_values: Sequence[tuple[str, float]],
) -> SymbolicEvaluationSnapshotContext:
    snapshot = tuple((str(name), float(value)) for name, value in parameter_values)
    fingerprint = _fingerprint(
        {
            "kind": "evaluation-snapshot",
            "parameter_values": [[name, f"{value:.17g}"] for name, value in snapshot],
        }
    )
    return SymbolicEvaluationSnapshotContext(parameter_values=snapshot, fingerprint=fingerprint)


def reject_unsupported_parameter_symbol_source(source: str) -> None:
    source_s = str(source or "").strip()
    if not source_s:
        raise UnsupportedSymbolicExpressionError("Empty symbolic expression is not supported.")
    if "[" in source_s or "]" in source_s:
        raise UnsupportedSymbolicExpressionError(
            "State concentration symbols such as [A], [A]_0, or [A](T0) are not supported in parameter/proof context."
        )
    if re.search(r"\b(if|else|and|or|not)\b", source_s):
        raise UnsupportedSymbolicExpressionError("Dynamic or logical expressions are not supported in symbolic proof.")


def canonical_parameter_identifier(name: str, spec: ParameterAlgebraSpec) -> str:
    return SymbolicParameterNamespaceContext.from_spec(spec).resolve_identifier(name)


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
