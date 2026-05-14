"""Canonicalize namespace-resolved identifiers in Algebra ASTs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

from kindred.core.algebra.printer import expression_to_source
from kindred.core.algebra.parser import (
    AlgebraBlock,
    BinaryNode,
    CallNode,
    ExprNode,
    IdentNode,
    LetStatement,
    NumberNode,
    SpeciesRefNode,
    UnaryNode,
    parse_algebra,
)
from kindred.core.simulator.parameter_namespace import MechanismParameterNamespace


@dataclass(frozen=True)
class CanonicalizedExpression:
    expr: ExprNode
    raw_to_canonical_identifiers: Mapping[str, str]


def _merge_identifier_maps(*maps: Mapping[str, str]) -> Dict[str, str]:
    merged: Dict[str, str] = {}
    for mapping in maps:
        for raw_name, canonical_name in mapping.items():
            existing = merged.get(raw_name)
            if existing is not None and existing != canonical_name:
                raise ValueError(
                    f"Conflicting canonical mechanism identifier mapping for {raw_name!r}: "
                    f"{existing!r} vs {canonical_name!r}"
                )
            merged[str(raw_name)] = str(canonical_name)
    return merged


def canonicalize_mechanism_identifiers(
    expr: ExprNode,
    *,
    mechanism_namespace: MechanismParameterNamespace,
) -> CanonicalizedExpression:
    if isinstance(expr, (NumberNode, SpeciesRefNode)):
        return CanonicalizedExpression(expr=expr, raw_to_canonical_identifiers={})
    if isinstance(expr, IdentNode):
        resolution = mechanism_namespace.resolve(expr.name)
        if resolution.canonical_name is not None:
            return CanonicalizedExpression(
                expr=IdentNode(str(resolution.canonical_name)),
                raw_to_canonical_identifiers={str(expr.name): str(resolution.canonical_name)},
            )
        invalid_message = mechanism_namespace.invalid_protected_indexed_identifier_message(expr.name)
        if invalid_message is not None:
            raise ValueError(invalid_message)
        return CanonicalizedExpression(expr=expr, raw_to_canonical_identifiers={})
    if isinstance(expr, UnaryNode):
        rhs = canonicalize_mechanism_identifiers(expr.rhs, mechanism_namespace=mechanism_namespace)
        return CanonicalizedExpression(
            expr=UnaryNode(op=expr.op, rhs=rhs.expr),
            raw_to_canonical_identifiers=dict(rhs.raw_to_canonical_identifiers),
        )
    if isinstance(expr, BinaryNode):
        lhs = canonicalize_mechanism_identifiers(expr.lhs, mechanism_namespace=mechanism_namespace)
        rhs = canonicalize_mechanism_identifiers(expr.rhs, mechanism_namespace=mechanism_namespace)
        return CanonicalizedExpression(
            expr=BinaryNode(op=expr.op, lhs=lhs.expr, rhs=rhs.expr),
            raw_to_canonical_identifiers=_merge_identifier_maps(
                lhs.raw_to_canonical_identifiers,
                rhs.raw_to_canonical_identifiers,
            ),
        )
    if isinstance(expr, CallNode):
        args = []
        raw_map: Dict[str, str] = {}
        for arg in expr.args:
            canonical_arg = canonicalize_mechanism_identifiers(arg, mechanism_namespace=mechanism_namespace)
            args.append(canonical_arg.expr)
            raw_map = _merge_identifier_maps(raw_map, canonical_arg.raw_to_canonical_identifiers)
        return CanonicalizedExpression(
            expr=CallNode(name=expr.name, args=tuple(args)),
            raw_to_canonical_identifiers=raw_map,
        )
    return CanonicalizedExpression(expr=expr, raw_to_canonical_identifiers={})


def canonicalize_algebra_block(
    block: AlgebraBlock,
    *,
    mechanism_namespace: MechanismParameterNamespace,
) -> AlgebraBlock:
    lines = []
    for stmt in block.lines:
        canonicalized = canonicalize_mechanism_identifiers(stmt.expr, mechanism_namespace=mechanism_namespace)
        line_text = f"let {stmt.name} = {expression_to_source(canonicalized.expr)}"
        lines.append(
            LetStatement(
                name=stmt.name,
                expr=canonicalized.expr,
                line=stmt.line,
                col=stmt.col,
                line_text=line_text,
            )
        )
    return AlgebraBlock(
        lines=lines,
        ast=[line.expr for line in lines],
        static_values=dict(block.static_values),
    )

def canonicalize_observable_rhs_source(
    expr_src: str,
    *,
    mechanism_namespace: MechanismParameterNamespace,
) -> str:
    block = parse_algebra(f"# Algebra\nlet __kindred_observable__ = {expr_src}\n")
    canonicalized = canonicalize_algebra_block(block, mechanism_namespace=mechanism_namespace)
    if not canonicalized.lines:
        return str(expr_src)
    return expression_to_source(canonicalized.lines[0].expr)
