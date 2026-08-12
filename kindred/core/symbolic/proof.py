from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from kindred.core.simulator.parameter_algebra_spec import ParameterAlgebraSpec, ParameterAssignment

from .backend import get_symbolic_backend_metadata, require_sympy
from .errors import UnsupportedSymbolicExpressionError
from .identity import symbolic_fingerprint
from .namespaces import SymbolicProductIdentityProofContext
from .parameter_expression import (
    PRODUCT_IDENTITY_PROOF_POLICY,
    translate_parameter_expression,
)


@dataclass(frozen=True, slots=True)
class SymbolicProofResult:
    proven: bool
    fingerprint: str


def prove_product_identity(
    *,
    target_factors: Mapping[str, int],
    candidate: ParameterAssignment,
    spec: ParameterAlgebraSpec | None = None,
    proof_context: SymbolicProductIdentityProofContext | None = None,
) -> SymbolicProofResult:
    if proof_context is None:
        if spec is None:
            raise UnsupportedSymbolicExpressionError("Symbolic proof requires a proof context.")
        proof_context = SymbolicProductIdentityProofContext.from_spec(
            target_factors=target_factors,
            spec=spec,
        )
    sympy = require_sympy()
    metadata = get_symbolic_backend_metadata()
    assignments = dict(proof_context.assignments)
    translation_fingerprints: dict[str, str] = {}
    proof_context_payload = proof_context.to_payload()

    def expand_assignment(name: str, stack: tuple[str, ...] = ()):
        if name in stack:
            raise UnsupportedSymbolicExpressionError(
                f"Cyclic symbolic assignment dependency for {name!r}."
            )
        assignment = assignments.get(name)
        if assignment is None:
            return sympy.Symbol(str(name))
        translated_assignment = translate_parameter_expression(
            assignment,
            policy=PRODUCT_IDENTITY_PROOF_POLICY,
            namespace=proof_context.parameter_namespace,
        )
        translation_fingerprints[str(name)] = translated_assignment.fingerprint
        substitutions = {
            dep_name: expand_assignment(dep_name, stack + (name,))
            for dep_name in translated_assignment.canonical_identifiers
            if dep_name in assignments
        }
        if not substitutions:
            return translated_assignment.expression
        return translated_assignment.expression.xreplace(
            {sympy.Symbol(dep_name): expr for dep_name, expr in substitutions.items()}
        )

    try:
        translated = translate_parameter_expression(
            candidate,
            policy=PRODUCT_IDENTITY_PROOF_POLICY,
            namespace=proof_context.parameter_namespace,
        )
        translation_fingerprints[str(candidate.name)] = translated.fingerprint
        candidate_expr = translated.expression.xreplace(
            {
                sympy.Symbol(dep_name): expand_assignment(dep_name, (str(candidate.name),))
                for dep_name in translated.canonical_identifiers
                if dep_name in assignments and dep_name != str(candidate.name)
            }
        )
    except UnsupportedSymbolicExpressionError:
        fingerprint = symbolic_fingerprint(
            {
                "candidate": str(candidate.name),
                "target_factors": dict(sorted(target_factors.items())),
                "assignment_fingerprints": dict(sorted(translation_fingerprints.items())),
                "backend": metadata.to_payload(),
                "reason": "unsupported",
                "symbol_context": proof_context_payload,
            }
        )
        return SymbolicProofResult(proven=False, fingerprint=fingerprint)

    candidate_name = str(candidate.name)
    target_expr = sympy.Integer(1)
    try:
        for raw_name, raw_exponent in sorted(target_factors.items()):
            name = str(raw_name)
            exponent = int(raw_exponent)
            factor = candidate_expr if name == candidate_name else expand_assignment(name)
            target_expr *= factor ** exponent
        simplified = sympy.simplify(target_expr - 1)
    except UnsupportedSymbolicExpressionError:
        fingerprint = symbolic_fingerprint(
            {
                "candidate": candidate_name,
                "expr_fingerprint": translated.fingerprint,
                "target_factors": dict(sorted((str(k), int(v)) for k, v in target_factors.items())),
                "assignment_fingerprints": dict(sorted(translation_fingerprints.items())),
                "backend": metadata.to_payload(),
                "reason": "unsupported",
                "symbol_context": proof_context_payload,
            }
        )
        return SymbolicProofResult(proven=False, fingerprint=fingerprint)
    proven = bool(simplified == 0)
    reason = "identity" if proven else "not_identity"
    fingerprint = symbolic_fingerprint(
        {
            "candidate": candidate_name,
            "expr_fingerprint": translated.fingerprint,
            "expanded_expression": str(sympy.simplify(candidate_expr)),
            "assignment_fingerprints": dict(sorted(translation_fingerprints.items())),
            "target_factors": dict(sorted((str(k), int(v)) for k, v in target_factors.items())),
            "backend": metadata.to_payload(),
            "reason": reason,
            "symbol_context": proof_context_payload,
        }
    )
    return SymbolicProofResult(proven=proven, fingerprint=fingerprint)
