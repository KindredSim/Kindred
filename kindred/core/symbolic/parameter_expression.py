from __future__ import annotations

from dataclasses import dataclass
import ast
from typing import Any

from kindred.core.simulator.parameter_algebra_spec import ParameterAlgebraSpec, ParameterAssignment

from .backend import get_symbolic_backend_metadata, require_sympy
from .errors import UnsupportedSymbolicExpressionError
from .identity import symbolic_fingerprint
from .namespaces import (
    SymbolicParameterNamespaceContext,
    make_parameter_namespace_context,
    reject_unsupported_parameter_symbol_source,
)


@dataclass(frozen=True, slots=True)
class SymbolicExpression:
    expression: Any
    normalized_source: str
    canonical_identifiers: tuple[str, ...]
    fingerprint: str
    symbol_context: dict[str, object]


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
    return make_parameter_namespace_context(spec)


def _normalize_source(expr_src: str, namespace: SymbolicParameterNamespaceContext) -> str:
    source = str(expr_src or "").strip()
    reject_unsupported_parameter_symbol_source(source)
    source = source.replace("^", "**")
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError as exc:
        raise _unsupported(f"Invalid symbolic expression syntax: {expr_src!r}.") from exc
    _validate_ast(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            node.id = namespace.resolve_identifier(node.id)
    return ast.unparse(tree)


def _validate_ast(node: ast.AST) -> None:
    if isinstance(node, ast.Expression):
        _validate_ast(node.body)
        return
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise _unsupported("Only numeric literals are supported in symbolic proof.")
        return
    if isinstance(node, ast.Name):
        return
    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, (ast.UAdd, ast.USub)):
            raise _unsupported("Only unary plus/minus are supported in symbolic proof.")
        _validate_ast(node.operand)
        return
    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
            raise _unsupported("Only arithmetic operators are supported in symbolic proof.")
        _validate_ast(node.left)
        _validate_ast(node.right)
        return
    raise _unsupported(f"Unsupported symbolic expression node: {type(node).__name__}.")


def _identifier_order(node: ast.AST) -> tuple[str, ...]:
    names: list[str] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and child.id not in names:
            names.append(child.id)
    return tuple(names)


def translate_parameter_expression(
    assignment: ParameterAssignment,
    *,
    spec: ParameterAlgebraSpec | None = None,
    namespace: SymbolicParameterNamespaceContext | None = None,
) -> SymbolicExpression:
    parameter_namespace = _coerce_parameter_namespace(spec=spec, namespace=namespace)
    normalized = _normalize_source(assignment.expr_src, parameter_namespace)
    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError as exc:
        raise _unsupported(f"Invalid symbolic expression syntax: {assignment.expr_src!r}.") from exc
    _validate_ast(tree)
    sympy = require_sympy()
    identifiers = _identifier_order(tree)
    symbol_context = parameter_namespace.to_expression_payload(identifiers)
    locals_map = {name: sympy.Symbol(name) for name in identifiers}
    try:
        expression = sympy.sympify(normalized, locals=locals_map, rational=True)
    except Exception as exc:
        raise _unsupported(f"Could not translate symbolic expression {assignment.expr_src!r}.") from exc
    metadata = get_symbolic_backend_metadata()
    fingerprint = symbolic_fingerprint(
        {
            "name": str(assignment.name),
            "normalized_source": str(expression),
            "identifiers": identifiers,
            "symbol_context": symbol_context,
            "backend": metadata.to_payload(),
        }
    )
    return SymbolicExpression(
        expression=expression,
        normalized_source=str(expression),
        canonical_identifiers=identifiers,
        fingerprint=fingerprint,
        symbol_context=symbol_context,
    )
