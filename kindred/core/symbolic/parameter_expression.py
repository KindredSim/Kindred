from __future__ import annotations

from dataclasses import dataclass
import ast
import hashlib
import json
import re
from typing import Any

from kindred.core.simulator.parameter_algebra_spec import ParameterAlgebraSpec, ParameterAssignment

from .backend import get_symbolic_backend_metadata, require_sympy
from .errors import UnsupportedSymbolicExpressionError

_PROTECTED_NAMES = {"T", "T0"}


@dataclass(frozen=True, slots=True)
class SymbolicExpression:
    expression: Any
    normalized_source: str
    canonical_identifiers: tuple[str, ...]
    fingerprint: str


def _unsupported(message: str) -> UnsupportedSymbolicExpressionError:
    return UnsupportedSymbolicExpressionError(message)


def _canonical_identifier(name: str, spec: ParameterAlgebraSpec) -> str:
    name_s = str(name)
    if name_s in _PROTECTED_NAMES:
        raise _unsupported(f"Protected runtime symbol {name_s!r} is not supported in symbolic proof.")
    if name_s in spec.scalar_input_names:
        return name_s
    resolution = spec.mechanism_namespace.resolve(name_s)
    if resolution.canonical_name is not None:
        return resolution.canonical_name
    if name_s in spec.param_assignment_names():
        return name_s
    raise _unsupported(f"Unknown symbolic identifier {name_s!r}.")


def _normalize_source(expr_src: str, spec: ParameterAlgebraSpec) -> str:
    source = str(expr_src or "").strip()
    if not source:
        raise _unsupported("Empty symbolic expression is not supported.")
    if "[" in source or "]" in source:
        raise _unsupported("Species references are not supported in symbolic proof.")
    if re.search(r"\b(if|else|and|or|not)\b", source):
        raise _unsupported("Dynamic or logical expressions are not supported in symbolic proof.")
    source = source.replace("^", "**")
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError as exc:
        raise _unsupported(f"Invalid symbolic expression syntax: {expr_src!r}.") from exc
    _validate_ast(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            node.id = _canonical_identifier(node.id, spec)
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


def _fingerprint(payload: dict[str, object]) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def translate_parameter_expression(
    assignment: ParameterAssignment,
    *,
    spec: ParameterAlgebraSpec,
) -> SymbolicExpression:
    normalized = _normalize_source(assignment.expr_src, spec)
    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError as exc:
        raise _unsupported(f"Invalid symbolic expression syntax: {assignment.expr_src!r}.") from exc
    _validate_ast(tree)
    sympy = require_sympy()
    identifiers = _identifier_order(tree)
    locals_map = {name: sympy.Symbol(name) for name in identifiers}
    try:
        expression = sympy.sympify(normalized, locals=locals_map, rational=True)
    except Exception as exc:
        raise _unsupported(f"Could not translate symbolic expression {assignment.expr_src!r}.") from exc
    metadata = get_symbolic_backend_metadata()
    fingerprint = _fingerprint(
        {
            "name": str(assignment.name),
            "normalized_source": str(expression),
            "identifiers": identifiers,
            "backend": metadata.to_payload(),
        }
    )
    return SymbolicExpression(
        expression=expression,
        normalized_source=str(expression),
        canonical_identifiers=identifiers,
        fingerprint=fingerprint,
    )
