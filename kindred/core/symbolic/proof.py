from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping

from kindred.core.simulator.parameter_algebra_spec import ParameterAlgebraSpec, ParameterAssignment

from .backend import get_symbolic_backend_metadata, require_sympy
from .errors import UnsupportedSymbolicExpressionError
from .parameter_expression import translate_parameter_expression


@dataclass(frozen=True, slots=True)
class SymbolicProofResult:
    proven: bool
    reason: str
    fingerprint: str


def _fingerprint(payload: Mapping[str, object]) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _assignment_sources(assignments: Mapping[str, ParameterAssignment]) -> dict[str, dict[str, object]]:
    return {
        str(name): {
            "expr_src": str(assignment.expr_src),
            "line_number": int(getattr(assignment, "line_number", 0) or 0),
            "line_content": str(getattr(assignment, "line_content", "") or ""),
        }
        for name, assignment in sorted(assignments.items())
    }


def prove_product_identity(
    *,
    target_factors: Mapping[str, int],
    candidate: ParameterAssignment,
    spec: ParameterAlgebraSpec,
) -> SymbolicProofResult:
    sympy = require_sympy()
    metadata = get_symbolic_backend_metadata()
    assignments = {str(stmt.name): stmt for stmt in spec.param_statements or []}
    translation_fingerprints: dict[str, str] = {}

    def expand_assignment(name: str, stack: tuple[str, ...] = ()):
        if name in stack:
            raise UnsupportedSymbolicExpressionError(
                f"Cyclic symbolic assignment dependency for {name!r}."
            )
        assignment = assignments.get(name)
        if assignment is None:
            return sympy.Symbol(str(name))
        translated_assignment = translate_parameter_expression(assignment, spec=spec)
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
        translated = translate_parameter_expression(candidate, spec=spec)
        translation_fingerprints[str(candidate.name)] = translated.fingerprint
        candidate_expr = translated.expression.xreplace(
            {
                sympy.Symbol(dep_name): expand_assignment(dep_name, (str(candidate.name),))
                for dep_name in translated.canonical_identifiers
                if dep_name in assignments and dep_name != str(candidate.name)
            }
        )
    except UnsupportedSymbolicExpressionError:
        fingerprint = _fingerprint(
            {
                "candidate": str(candidate.name),
                "expr_src": str(candidate.expr_src),
                "target_factors": dict(sorted(target_factors.items())),
                "assignment_sources": _assignment_sources(assignments),
                "backend": metadata.to_payload(),
                "reason": "unsupported",
            }
        )
        return SymbolicProofResult(proven=False, reason="unsupported", fingerprint=fingerprint)

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
        fingerprint = _fingerprint(
            {
                "candidate": candidate_name,
                "expr_fingerprint": translated.fingerprint,
                "target_factors": dict(sorted((str(k), int(v)) for k, v in target_factors.items())),
                "assignment_sources": _assignment_sources(assignments),
                "backend": metadata.to_payload(),
                "reason": "unsupported",
            }
        )
        return SymbolicProofResult(proven=False, reason="unsupported", fingerprint=fingerprint)
    proven = bool(simplified == 0)
    reason = "identity" if proven else "not_identity"
    fingerprint = _fingerprint(
        {
            "candidate": candidate_name,
            "expr_fingerprint": translated.fingerprint,
            "expanded_expression": str(sympy.simplify(candidate_expr)),
            "assignment_fingerprints": dict(sorted(translation_fingerprints.items())),
            "target_factors": dict(sorted((str(k), int(v)) for k, v in target_factors.items())),
            "backend": metadata.to_payload(),
            "reason": reason,
        }
    )
    return SymbolicProofResult(proven=proven, reason=reason, fingerprint=fingerprint)
