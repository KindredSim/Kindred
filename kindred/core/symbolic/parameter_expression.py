from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Mapping

from kindred.core.algebra.parser import (
    BinaryNode,
    IdentNode,
    NumberNode,
    UnaryNode,
)
from kindred.core.algebra.symbols import PROTECTED_NAMES as ALGEBRA_PROTECTED_NAMES
from kindred.core.simulator.parameter_algebra_spec import ParameterAlgebraSpec, ParameterAssignment

from .backend import get_symbolic_backend_metadata, require_sympy
from .errors import UnsupportedSymbolicExpressionError
from .identity import symbolic_fingerprint
from .namespaces import SymbolicParameterNamespaceContext


@dataclass(frozen=True, slots=True)
class SymbolicExpression:
    expression: Any
    canonical_identifiers: tuple[str, ...]
    fingerprint: str


@dataclass(frozen=True, slots=True)
class RetainedParameterExpressionPolicy:
    """Accepted operations for one retained-AST symbolic consumer."""

    name: str
    binary_operations: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(
            self,
            "binary_operations",
            MappingProxyType(
                {
                    str(operator): str(operation)
                    for operator, operation in dict(self.binary_operations).items()
                }
            ),
        )


CALCULATOR_EXPANSION_POLICY = RetainedParameterExpressionPolicy(
    name="calculator-parameter-expansion",
    binary_operations={
        "+": "add",
        "-": "subtract",
        "*": "multiply",
        "/": "divide",
        "**": "power",
        "^": "power",
    },
)

PRODUCT_IDENTITY_PROOF_POLICY = RetainedParameterExpressionPolicy(
    name="wegscheider-product-identity-proof",
    binary_operations={
        "*": "multiply",
        "/": "divide",
        "**": "power",
        "^": "power",
    },
)


def _unsupported(message: str) -> UnsupportedSymbolicExpressionError:
    return UnsupportedSymbolicExpressionError(message)


def _coerce_parameter_namespace(
    *,
    spec: ParameterAlgebraSpec | None,
    namespace: SymbolicParameterNamespaceContext | None,
) -> SymbolicParameterNamespaceContext:
    if namespace is not None:
        return namespace
    if spec is None:
        raise _unsupported("Symbolic parameter expression requires a parameter namespace context.")
    return SymbolicParameterNamespaceContext.from_spec(spec)


def _append_identifier(names: list[str], name: str) -> None:
    if name not in names:
        names.append(name)


def _translate_retained_node(
    node: object,
    *,
    namespace: SymbolicParameterNamespaceContext,
    policy: RetainedParameterExpressionPolicy,
    identifiers: list[str],
    sympy: Any,
) -> tuple[Any, dict[str, object]]:
    """Translate a retained typed node under the requesting consumer's policy."""

    if isinstance(node, NumberNode):
        value = float(node.value)
        if not math.isfinite(value):
            raise _unsupported("Only finite retained numeric literals are supported.")
        numerator, denominator = value.as_integer_ratio()
        return (
            sympy.Rational(numerator, denominator),
            {"kind": "number", "binary64": value.hex()},
        )
    if isinstance(node, IdentNode):
        raw_name = str(node.name)
        if raw_name in ALGEBRA_PROTECTED_NAMES:
            if raw_name == "T":
                raise _unsupported(
                    "Protected runtime symbol 'T' is not supported in symbolic proof."
                )
            canonical_name = raw_name
        else:
            canonical_name = namespace.resolve_identifier(raw_name)
        _append_identifier(identifiers, canonical_name)
        return (
            sympy.Symbol(canonical_name),
            {"kind": "identifier", "canonical_name": canonical_name},
        )
    if isinstance(node, UnaryNode):
        if node.op not in {"+", "-"}:
            raise _unsupported("Only unary plus/minus are supported in symbolic proof.")
        rhs, rhs_payload = _translate_retained_node(
            node.rhs,
            namespace=namespace,
            policy=policy,
            identifiers=identifiers,
            sympy=sympy,
        )
        return (
            rhs if node.op == "+" else -rhs,
            {"kind": "unary", "operator": str(node.op), "rhs": rhs_payload},
        )
    if isinstance(node, BinaryNode):
        operator = str(node.op)
        operation = policy.binary_operations.get(operator)
        if operation is None and policy == PRODUCT_IDENTITY_PROOF_POLICY and operator in {"+", "-"}:
            raise _unsupported(
                "Additive binary64 parameter arithmetic cannot establish "
                "an exact symbolic identity."
            )
        if operation is None:
            raise _unsupported(
                f"Retained Algebra operator {operator!r} is unsupported by "
                f"the {policy.name} policy."
            )
        lhs, lhs_payload = _translate_retained_node(
            node.lhs,
            namespace=namespace,
            policy=policy,
            identifiers=identifiers,
            sympy=sympy,
        )
        rhs, rhs_payload = _translate_retained_node(
            node.rhs,
            namespace=namespace,
            policy=policy,
            identifiers=identifiers,
            sympy=sympy,
        )
        if operation == "add":
            expression = lhs + rhs
        elif operation == "subtract":
            expression = lhs - rhs
        elif operation == "multiply":
            expression = lhs * rhs
        elif operation == "divide":
            expression = lhs / rhs
        else:
            expression = lhs ** rhs
        if expression.has(sympy.zoo, sympy.oo, -sympy.oo, sympy.nan):
            raise _unsupported("Retained symbolic expression produced a non-finite result.")
        return (
            expression,
            {
                "kind": "binary",
                "operation": operation,
                "lhs": lhs_payload,
                "rhs": rhs_payload,
            },
        )
    raise _unsupported(
        "Symbolic proof requires supported nodes from the retained admitted Algebra AST; "
        f"got {type(node).__name__}."
    )


def translate_parameter_expression(
    assignment: ParameterAssignment,
    *,
    policy: RetainedParameterExpressionPolicy,
    spec: ParameterAlgebraSpec | None = None,
    namespace: SymbolicParameterNamespaceContext | None = None,
) -> SymbolicExpression:
    if not isinstance(policy, RetainedParameterExpressionPolicy):
        raise TypeError("policy must be a RetainedParameterExpressionPolicy.")
    parameter_namespace = _coerce_parameter_namespace(spec=spec, namespace=namespace)
    retained_node = assignment.expr_ast
    if retained_node is None:
        raise _unsupported("Symbolic proof requires the retained admitted Algebra AST.")
    sympy = require_sympy()
    identifier_order: list[str] = []
    expression, retained_structure = _translate_retained_node(
        retained_node,
        namespace=parameter_namespace,
        policy=policy,
        identifiers=identifier_order,
        sympy=sympy,
    )
    identifiers = tuple(identifier_order)
    symbol_context = parameter_namespace.to_expression_payload(identifiers)
    metadata = get_symbolic_backend_metadata()
    fingerprint = symbolic_fingerprint(
        {
            "name": str(assignment.name),
            "retained_structure": retained_structure,
            "policy": policy.name,
            "identifiers": identifiers,
            "symbol_context": symbol_context,
            "backend": metadata.to_payload(),
        }
    )
    return SymbolicExpression(
        expression=expression,
        canonical_identifiers=identifiers,
        fingerprint=fingerprint,
    )
