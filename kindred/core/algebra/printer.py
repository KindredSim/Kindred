"""AST-preserving source emission for Algebra expressions."""

from __future__ import annotations

import math

from kindred.core.algebra.grammar import ASSOCIATIVITY, PRECEDENCE
from kindred.core.algebra.parser import (
    AlgebraBlock,
    BinaryNode,
    CallNode,
    ExprNode,
    IdentNode,
    NumberNode,
    SpeciesRefNode,
    UnaryNode,
)


def _node_precedence(expr: ExprNode) -> int:
    if isinstance(expr, BinaryNode):
        return int(PRECEDENCE.get(str(expr.op), 0))
    if isinstance(expr, UnaryNode):
        return int(PRECEDENCE["unary"])
    return 100


def _child_needs_parentheses(child: ExprNode, *, parent_op: str) -> bool:
    if not isinstance(child, (BinaryNode, UnaryNode)):
        return False
    parent_precedence = int(PRECEDENCE.get(str(parent_op), 0))
    child_precedence = _node_precedence(child)
    if child_precedence < parent_precedence:
        return True
    return False


def _binary_child_source(child: ExprNode, *, parent_op: str, side: str) -> str:
    text = expression_to_source(child)
    if _child_needs_parentheses(child, parent_op=parent_op):
        return f"({text})"
    if not isinstance(child, BinaryNode):
        return text
    parent_precedence = int(PRECEDENCE.get(str(parent_op), 0))
    child_precedence = int(PRECEDENCE.get(str(child.op), 0))
    if child_precedence != parent_precedence:
        return text
    associativity = str(ASSOCIATIVITY.get(str(parent_op), "left"))
    if associativity == "right":
        return f"({text})" if side == "left" else text
    return f"({text})" if side == "right" else text


def expression_to_source(expr: ExprNode) -> str:
    if isinstance(expr, NumberNode):
        value = float(expr.value)
        if not math.isfinite(value):
            raise ValueError("Cannot emit non-finite Algebra numeric literal.")
        return str(int(value)) if math.isfinite(value) and value.is_integer() else repr(value)
    if isinstance(expr, IdentNode):
        return str(expr.name)
    if isinstance(expr, SpeciesRefNode):
        if expr.kind == "init":
            return f"[{expr.name}]_0"
        if expr.kind == "T0":
            return f"[{expr.name}](T0)"
        return f"[{expr.name}]"
    if isinstance(expr, UnaryNode):
        rhs = expression_to_source(expr.rhs)
        if _child_needs_parentheses(expr.rhs, parent_op="unary"):
            rhs = f"({rhs})"
        if str(expr.op).isalpha():
            return f"{expr.op} {rhs}"
        return f"{expr.op}{rhs}"
    if isinstance(expr, BinaryNode):
        lhs = _binary_child_source(expr.lhs, parent_op=str(expr.op), side="left")
        rhs = _binary_child_source(expr.rhs, parent_op=str(expr.op), side="right")
        return f"{lhs} {expr.op} {rhs}"
    if isinstance(expr, CallNode):
        args = ", ".join(expression_to_source(arg) for arg in expr.args)
        return f"{expr.name}({args})"
    return str(expr)


def algebra_block_to_source(block: AlgebraBlock) -> str:
    lines = ["# Algebra"]
    for stmt in block.lines:
        lines.append(f"let {stmt.name} = {expression_to_source(stmt.expr)}")
    return "\n".join(lines) + "\n"
