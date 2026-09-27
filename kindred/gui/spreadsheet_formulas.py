"""Admit row-wise column patterns and lower prepared Algebra to IronCalc."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import re
from typing import Any

import ironcalc

from kindred.core import constants
from kindred.core.algebra.errors import AlgebraError
from kindred.core.algebra.parser import (
    BinaryNode, CallNode, ExprNode, IdentNode, NumberNode, UnaryNode, parse_algebra,
)
from kindred.core.algebra.unit_preparation import prepare_expression
from kindred.core.quantity_units import (
    CONSTANT_UNITS, CUBIC_METRE, MOLAR, ONE_SCALE, SECOND, Unit,
    analysis_display_unit, conversion_scale, parse_unit, scientific_unit,
)


_CONSTANTS = {name: getattr(constants, name) for name in constants.__all__}
_UNARY_FUNCTIONS = {
    "sqrt", "ln", "log10", "exp", "sin", "cos", "tan", "abs", "erf", "heaviside",
}
_FUNCTIONS = _UNARY_FUNCTIONS | {"pow", "ifelse", "clip", "min", "max"}
# IronCalc compares at 15 digits with a tolerance. The sign of the exact
# difference picks Algebra's 0 or 1 for a negative, zero or positive difference;
# an overflowed difference keeps its sign and an undefined one is an error.
_COMPARISONS = {"==": "0,1,0", "!=": "1,0,1", ">": "0,0,1", "<": "1,0,0", ">=": "0,1,1", "<=": "1,1,0"}
_ARITHMETIC = {"+", "-", "*", "/", "^", "**"}
_REFERENCE = re.compile(r"([A-Z]+)1\Z")
# Letters followed by a row number, excluding calls such as log10(x) and the
# exponent of numbers such as 1e5 or 1.e5.
_ROW_REFERENCE = re.compile(r"(?<![\w.])([A-Za-z]+)(\d+)\b(?!\s*\()")
_ROW = "\0"
_BASE_UNITS = {"concentration": MOLAR, "time": SECOND, "volume": CUBIC_METRE}


def _reference_column(name: str) -> int:
    match = _REFERENCE.fullmatch(name)
    if match is None:
        raise ValueError(f"{name!r} is not a column reference. Formulas use cells of their own row, such as B5.")
    # Letters beyond the worksheet still name a column, which admission refuses.
    column = 0
    for letter in match[1]:
        column = column * 26 + ord(letter) - 64
    return column


def _source(source: str) -> str:
    if not source:
        raise ValueError("Type an expression after =, e.g. =B5*2.")
    if "\n" in source or "\r" in source:
        raise ValueError("A formula must fit on one line.")
    return source


def pattern_from_row(text: str, row: int) -> str:
    """Return the stored row-1 pattern of a formula written in one worksheet row."""

    def reference(match: re.Match[str]) -> str:
        if int(match[2]) != row:
            raise ValueError(f"A formula can only use cells of its own row ({row}), not {match[0]}.")
        return f"{match[1].upper()}1"

    return _ROW_REFERENCE.sub(reference, text.strip().removeprefix("=").strip())


def formula_at_row(pattern: str, row: int) -> str:
    """Show a stored row-1 pattern as the formula of one worksheet row."""
    return "=" + _ROW_REFERENCE.sub(lambda match: f"{match[1]}{row}", pattern)


def _references(node: ExprNode, column_count: int) -> tuple[int, ...]:
    """Admit the supported syntax and retain authored reference order."""
    if isinstance(node, NumberNode):
        return ()
    if isinstance(node, IdentNode):
        if node.name in _CONSTANTS:
            return ()
        column = _reference_column(node.name)
        if column > column_count:
            raise ValueError(f"Column {node.name[:-1]} does not exist in this dataset.")
        return (column,)
    if isinstance(node, UnaryNode):
        if node.op not in {"+", "-", "!"}:
            raise ValueError(f"Unsupported formula operator {node.op!r}.")
        children = (node.rhs,)
    elif isinstance(node, BinaryNode):
        if node.op not in _ARITHMETIC and node.op not in _COMPARISONS:
            raise ValueError(f"Unsupported formula operator {node.op!r}.")
        children = (node.lhs, node.rhs)
    elif isinstance(node, CallNode):
        arity = (1 if node.name in _UNARY_FUNCTIONS else
                 2 if node.name == "pow" else
                 3 if node.name in {"ifelse", "clip"} else None)
        if node.name in {"min", "max"}:
            if not node.args:
                raise ValueError(f"{node.name} requires at least one argument.")
        elif arity is None:
            if node.name.lower() in _FUNCTIONS:
                raise ValueError(f"Use lower-case function names: {node.name.lower()} instead of {node.name}.")
            raise ValueError(f"Unsupported formula function {node.name!r}.")
        elif len(node.args) != arity:
            raise ValueError(f"{node.name} requires {arity} argument(s).")
        children = node.args
    else:
        raise ValueError("Formula patterns support column references, not species or context references.")
    return tuple(column for child in children for column in _references(child, column_count))


def _has_ifelse(node: ExprNode) -> bool:
    children = (node.args if isinstance(node, CallNode) else (node.lhs, node.rhs) if isinstance(node, BinaryNode)
                else (node.rhs,) if isinstance(node, UnaryNode) else ())
    return isinstance(node, CallNode) and node.name == "ifelse" or any(map(_has_ifelse, children))


def _emit(node: ExprNode, row: str) -> str:
    if isinstance(node, NumberNode):
        text = node.literal if node.literal is not None else repr(node.value)
    elif isinstance(node, IdentNode):
        text = repr(_CONSTANTS[node.name]) if node.name in _CONSTANTS else f"{node.name[:-1]}{row}"
    elif isinstance(node, UnaryNode):
        rhs = _emit(node.rhs, row)
        text = f"IF({rhs},0,1)" if node.op == "!" else f"{node.op}{rhs}"
    elif isinstance(node, BinaryNode):
        lhs, rhs = _emit(node.lhs, row), _emit(node.rhs, row)
        if node.op in _COMPARISONS:
            text = f"CHOOSE(2+SIGN({lhs}-{rhs}),{_COMPARISONS[node.op]})"
        else:
            operator = "^" if node.op == "**" else node.op
            text = f"{lhs}{operator}{rhs}"
    else:
        args = tuple(_emit(arg, row) for arg in node.args)
        if node.name == "pow":
            text = f"{args[0]}^{args[1]}"
        elif node.name == "ifelse" and _has_ifelse(node.args[0]):
            # A condition containing ifelse is bound once instead of copied; the
            # zero terms preserve Algebra's eager branch errors (IF is lazy).
            text = (f"LET(_kc,{args[0]},_ka,{args[1]},_kb,{args[2]},"
                    "((IF(_kc,_ka,_kb)+(0*_ka))+(0*_kb)))")
        elif node.name == "ifelse":
            # IronCalc's IF is lazy and its LET is slow; weighting both branches
            # by the condition's truth keeps Algebra's eager argument errors.
            weight = f"ABS(SIGN({args[0]}))"
            text = f"({weight}*{args[1]})+((1-{weight})*{args[2]})"
        elif node.name == "clip":
            # The median of the value and both bounds is Algebra's clip.
            text = f"MEDIAN({','.join(args)})"
        elif node.name == "heaviside":
            text = f"CHOOSE(2+SIGN({args[0]}),0,0,1)"
        else:
            text = f"{node.name.upper()}({','.join(args)})"
    expression = f"({text})"
    if node.unit_scale != ONE_SCALE:
        # IronCalc keeps formula numbers as 15-digit text. The factor's correctly
        # rounded value is written as the exact sum of two 14-digit numbers.
        factor = node.unit_scale.number()
        head = float(f"{factor:.14g}")
        tail = float(f"{factor - head:.14g}")
        factor_text = repr(head) if not tail else f"{head!r}+({tail!r})"
        expression = f"(({factor_text})*{expression})"
    return expression


@dataclass(frozen=True)
class CompiledPattern:
    """The facts needed to publish one admitted calculated column."""

    unit: Unit | None
    references: frozenset[int]
    _template: str

    def formula(self, row: int) -> str:
        return self._template.replace(_ROW, str(row))


def _retained_preferences(unit: Unit | None, preferences: dict[str, Unit]) -> dict[str, Unit]:
    if unit is None:
        return {}
    energy, amount, length, time, temperature = unit.dimensions
    represented = {
        "concentration": energy == 0 and length == -3 * amount and amount != 0,
        "time": time != 0,
        "volume": energy == amount == temperature == 0 and length == 3,
    }
    return {name: value for name, value in preferences.items() if represented[name]}


def compile_columns(columns: Sequence[Mapping[str, Any]]) -> dict[int, CompiledPattern]:
    """Compile all patterns together, refusing over-deep nesting as a rejection."""
    try:
        return _compile_columns(columns)
    except RecursionError:
        raise ValueError("A formula is nested too deeply to calculate.") from None


def _compile_columns(columns: Sequence[Mapping[str, Any]]) -> dict[int, CompiledPattern]:
    """Compile all patterns together, returning their dependency-ordered facts.

    Independent columns supply displayed unit declarations. Calculated columns
    always derive their units from their formula, ignoring prior display labels.
    No folded numeric value from the Algebra parser supplies a cell result.
    """
    statements = []
    calculated = []
    units = {name: CONSTANT_UNITS[name] for name in _CONSTANTS}
    preferences: dict[int, dict[str, Unit]] = {}
    for column, info in enumerate(columns, 1):
        letter = ironcalc.column_name_from_number(column)
        name = f"{letter}1"
        if info.get("formula") is not None:
            statements.append(f"let {name} = {_source(info['formula'])}")
            calculated.append(letter)
        else:
            # Stored units were admitted when declared or derived; they are not
            # re-admitted as new declarations.
            unit = parse_unit(info["unit"]) if info["unit"] is not None else None
            units[name] = unit
            preferences[column] = {
                kind: unit for kind, base in _BASE_UNITS.items()
                if unit is not None and unit.compatible(base)
            }
    # Each calculated column is one line of the joined block, so the error line
    # names the column; the block's own text and positions are never shown, and
    # its internal row-1 names read as column letters.
    try:
        block = parse_algebra("\n".join(statements) + "\n")
    except AlgebraError as exc:
        message = re.sub(r"\b([A-Z]+)1\b", r"\1", exc.msg)
        raise ValueError(f"Column {calculated[exc.line - 1]}: {message}") from exc
    references = {statement.name: _references(statement.expr, len(columns)) for statement in block.lines}
    compiled = {}
    for statement in block.statements_in_evaluation_order():
        column = _reference_column(statement.name)
        preferred = {}
        for reference in references[statement.name]:
            for kind, unit in preferences[reference].items():
                preferred.setdefault(kind, unit)
        prepared = prepare_expression(statement.expr, units.get)
        inferred = prepared.quantity_unit
        preferred = _retained_preferences(inferred, preferred)
        display = None
        if inferred is not None:
            display = analysis_display_unit(
                scientific_unit(inferred.dimensions),
                concentration=preferred.get("concentration", MOLAR),
                time=preferred.get("time", SECOND),
                volume=preferred.get("volume", CUBIC_METRE),
            )
            # Compose before float projection: individually enormous factors
            # can cancel at the root when choosing the displayed coordinate.
            factor = prepared.unit_scale * conversion_scale(inferred, display)
            factor.number()
            prepared = replace(prepared, unit_scale=factor)
        # Emitted once; each row only substitutes its number into the references.
        compiled[column] = CompiledPattern(display, frozenset(references[statement.name]), "=" + _emit(prepared, _ROW))
        units[statement.name] = display
        preferences[column] = preferred
    return compiled


def remap_pattern(source: str, old_to_new: Mapping[int, int]) -> str:
    """Rewrite admitted positional references after a column structure edit,
    keeping the authored text."""
    return _ROW_REFERENCE.sub(
        lambda match: f"{ironcalc.column_name_from_number(old_to_new[_reference_column(match[0])])}1", source)
