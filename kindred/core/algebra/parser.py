"""
AST builder for the Algebra DSL.

Current contract
----------------
This module implements the algebra grammar, builtin handling, evaluation rules,
and parser-facing error reporting used by the Algebra DSL.

Scope of this module
--------------------
- Tokenize source into a deterministic stream with locations.
- Parse the internal source format: { let IDENT = expr NEWLINE }. External
  mechanism DSL algebra declarations are normalized into this format before
  reaching this parser.
- Build a typed AST with nodes:
    Number, Ident, Unary, Binary, Call, SpeciesRef(kind="now"|"init")
- Enforce precedence and right-associative power (** and ^).
- Admit builtin argument counts from the builtin table; numeric and type
  checks belong to the evaluator.

Out of scope here
-----------------
- Dynamic evaluation (done in evaluator.py).
- Shadowing detection with species names (handled by evaluator with symbol table).

Determinism & constraints
-------------------------
- No I/O, no cwd, no registry, no network.
- Pure functions; every diagnostic is an AlgebraError* carrying its location.

"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from fractions import Fraction
from typing import List, Literal, Optional, Sequence, Tuple

from kindred.core.quantity_units import Unit, UnitScale, ONE_SCALE

from .errors import (
    AlgebraSyntaxError,
)
from .symbols import BUILTIN_FUNCTIONS
from .grammar import (
    KEYWORDS,
    TOKEN_SPEC,
    TokenKind,
)

logger = logging.getLogger(__name__)

# ------------------------------ Tokens ---------------------------------------


@dataclass(frozen=True)
class Token:
    kind: TokenKind
    text: str
    line: int
    col: int
    line_text: str


# Regex for tokenization with named groups
_TOKEN_RE = re.compile(
    "|".join(f"(?P<{tk.name}>{rx})" for tk, rx in TOKEN_SPEC) + r"|(?P<SKIP>[ \t]+)|(?P<MISMATCH>.)",
    re.ASCII,
)


def _lex(src: str) -> List[Token]:
    tokens: List[Token] = []
    line = 1
    bol = 0  # beginning-of-line index
    i = 0
    n = len(src)
    while i < n:
        m = _TOKEN_RE.match(src, i)
        if not m:
            # Shouldn't happen given MISMATCH, but guard anyway
            raise AlgebraSyntaxError("Lexical error.", line=line, col=(i - bol + 1), line_text=_line_of(src, line))
        kind_name = m.lastgroup or "MISMATCH"
        text = m.group(0)
        if kind_name == "SKIP":
            pass
        elif kind_name == "MISMATCH":
            raise AlgebraSyntaxError(f"Unexpected character {text!r}.", line=line, col=(i - bol + 1), line_text=_line_of(src, line))
        else:
            tk = TokenKind[kind_name]
            # Apply keyword recognition: if IDENT matches a keyword, use that TokenKind instead
            if tk is TokenKind.IDENT and text in KEYWORDS:
                tk = KEYWORDS[text]
            if tk is TokenKind.NEWLINE:
                tokens.append(Token(tk, text, line, (i - bol + 1), _line_of(src, line)))
                line += 1
                bol = m.end()
            else:
                tokens.append(Token(tk, text, line, (i - bol + 1), _line_of(src, line)))
        i = m.end()
    tokens.append(Token(TokenKind.EOF, "", line, 1, _line_of(src, line)))
    return tokens


def _line_of(src: str, line_no: int) -> str:
    # Simple line splitter that tolerates trailing missing lines
    lines = src.splitlines(True)
    if 1 <= line_no <= len(lines):
        return lines[line_no - 1]
    return ""


# ------------------------------ AST nodes ------------------------------------


@dataclass(frozen=True, kw_only=True)
class ExprNode:
    """Authored syntax with immutable preparation facts; printers use syntax only."""
    quantity_unit: Unit | None = field(default=None, compare=False, repr=False)
    unit_scale: UnitScale = field(default=ONE_SCALE, compare=False, repr=False)
    unit_factor: float = field(default=1.0, compare=False, repr=False)

    def has_species_ref(self) -> bool:
        return False

@dataclass(frozen=True)
class NumberNode(ExprNode):
    value: float
    literal: str | None = None

    @property
    def exact_value(self) -> Fraction | None:
        """Return the retained literal exactly when it is a finite rational."""
        try:
            return Fraction(self.literal if self.literal is not None else str(self.value))
        except (TypeError, ValueError, ZeroDivisionError, OverflowError):
            return None


@dataclass(frozen=True)
class IdentNode(ExprNode):
    name: str


@dataclass(frozen=True)
class UnaryNode(ExprNode):
    op: str
    rhs: ExprNode

    def has_species_ref(self) -> bool:
        return self.rhs.has_species_ref()

@dataclass(frozen=True)
class BinaryNode(ExprNode):
    op: str
    lhs: ExprNode
    rhs: ExprNode

    def has_species_ref(self) -> bool:
        return self.lhs.has_species_ref() or self.rhs.has_species_ref()

@dataclass(frozen=True)
class CallNode(ExprNode):
    name: str
    args: Tuple[ExprNode, ...]

    def has_species_ref(self) -> bool:
        return any(a.has_species_ref() for a in self.args)

@dataclass(frozen=True)
class SpeciesRefNode(ExprNode):
    name: str
    kind: Literal["now", "init"]

    def __post_init__(self) -> None:
        if self.kind not in {"now", "init"}:
            raise ValueError("Species reference kind must be 'now' or 'init'.")

    def has_species_ref(self) -> bool:
        return True

@dataclass(frozen=True)
class LetStatement:
    name: str
    expr: ExprNode
    line: int
    col: int
    line_text: str
    declared_unit: Unit | None = None


def _iter_identifiers(node: ExprNode):
    if isinstance(node, IdentNode):
        yield str(node.name)
        return
    if isinstance(node, SpeciesRefNode):
        return
    if isinstance(node, UnaryNode):
        yield from _iter_identifiers(node.rhs)
        return
    if isinstance(node, BinaryNode):
        yield from _iter_identifiers(node.lhs)
        yield from _iter_identifiers(node.rhs)
        return
    if isinstance(node, CallNode):
        for argument in node.args:
            yield from _iter_identifiers(argument)


def circular_definition_message(target: str, steps: Sequence[str]) -> str:
    """One sentence for a value that depends on itself, naming each read in turn."""
    joined = steps[0] if len(steps) == 1 else ", ".join(steps[:-1]) + f", and {steps[-1]}"
    return f"{target} depends on its own value: {joined}."


def _topological_evaluation_order(
    statements: Tuple[LetStatement, ...],
) -> Tuple[str, ...]:
    """Retain a stable dependency order without changing source order."""
    names = tuple(str(statement.name) for statement in statements)
    positions = {name: index for index, name in enumerate(names)}
    dependencies = {
        str(statement.name): {
            identifier
            for identifier in _iter_identifiers(statement.expr)
            if identifier in positions
        }
        for statement in statements
    }
    ordered: List[str] = []
    visiting: List[str] = []
    visited: set[str] = set()

    def visit(name: str) -> None:
        if name in visited:
            return
        if name in visiting:
            start = visiting.index(name)
            cycle = visiting[start:] + [name]
            statement = statements[positions[name]]
            raise AlgebraSyntaxError(
                circular_definition_message(name, [f"{a} reads {b}" for a, b in zip(cycle, cycle[1:])]),
                line=statement.line,
                col=statement.col,
                line_text=statement.line_text,
            )
        visiting.append(name)
        for dependency in sorted(
            dependencies.get(name, ()),
            key=positions.__getitem__,
        ):
            visit(dependency)
        visiting.pop()
        visited.add(name)
        ordered.append(name)

    for name in names:
        visit(name)
    return tuple(ordered)


@dataclass(frozen=True)
class AlgebraBlock:
    """One immutable parsed Algebra program."""

    lines: Tuple[LetStatement, ...]
    ast: Tuple[ExprNode, ...]
    evaluation_order: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        lines = tuple(self.lines)
        ast = tuple(self.ast)
        if len(lines) != len(ast):
            raise ValueError("AlgebraBlock lines and AST must have identical length.")
        if tuple(stmt.expr for stmt in lines) != ast:
            raise ValueError("AlgebraBlock AST must match its retained statements.")
        names = tuple(str(statement.name) for statement in lines)
        if len(set(names)) != len(names):
            duplicate = next(name for name in names if names.count(name) > 1)
            raise ValueError(f"AlgebraBlock assignment names must be unique: {duplicate!r}.")
        evaluation_order = tuple(str(name) for name in self.evaluation_order)
        if not evaluation_order and names:
            evaluation_order = _topological_evaluation_order(lines)
        if len(evaluation_order) != len(names) or set(evaluation_order) != set(names):
            raise ValueError(
                "AlgebraBlock evaluation order must contain every assignment name exactly once."
            )
        object.__setattr__(self, "lines", lines)
        object.__setattr__(self, "ast", ast)
        object.__setattr__(self, "evaluation_order", evaluation_order)

    def statements_in_evaluation_order(self) -> Tuple[LetStatement, ...]:
        """Return retained statements in dependency order."""
        by_name = {str(statement.name): statement for statement in self.lines}
        return tuple(by_name[name] for name in self.evaluation_order)

    # placeholder for symbol table pointer is out-of-scope here; evaluator owns it.


# ------------------------------ Parser ---------------------------------------


class Parser:
    def __init__(self, tokens: List[Token]) -> None:
        self.toks = tokens
        self.i = 0

    # Utility
    def _cur(self) -> Token:
        return self.toks[self.i]

    def _advance(self) -> Token:
        t = self.toks[self.i]
        self.i += 1
        return t

    def _match(self, *kinds: TokenKind) -> Optional[Token]:
        if self._cur().kind in kinds:
            return self._advance()
        return None

    def _expect(self, kind: TokenKind, msg: str) -> Token:
        t = self._cur()
        if t.kind is not kind:
            raise AlgebraSyntaxError(msg, line=t.line, col=t.col, line_text=t.line_text)
        return self._advance()

    # Entry: { let_stmt }
    def parse_block(self) -> AlgebraBlock:
        lines: List[LetStatement] = []
        exprs: List[ExprNode] = []

        # Accept optional blank lines
        self._consume_newlines()

        while self._cur().kind is not TokenKind.EOF:
            let_tok = self._expect(TokenKind.LET, "Expected 'let' at the beginning of the statement.")
            name_tok = self._expect(TokenKind.IDENT, "Expected a name after 'let'.")
            self._expect(TokenKind.EQUALS, "Expected '=' after the name.")
            expr = self._parse_expr()
            # Require newline after each let
            if self._cur().kind is TokenKind.NEWLINE:
                self._advance()
            elif self._cur().kind is TokenKind.EOF:
                # Allow EOF after last statement without trailing newline
                pass
            else:
                t = self._cur()
                raise AlgebraSyntaxError("Expected the end of the line.", line=t.line, col=t.col, line_text=t.line_text)

            stmt = LetStatement(name=name_tok.text, expr=expr, line=let_tok.line, col=let_tok.col, line_text=let_tok.line_text)
            lines.append(stmt)
            exprs.append(expr)

            self._consume_newlines()

        return AlgebraBlock(
            lines=lines,
            ast=exprs,
            evaluation_order=_topological_evaluation_order(tuple(lines)),
        )

    def _consume_newlines(self) -> None:
        while self._cur().kind is TokenKind.NEWLINE:
            self._advance()

    # Grammar: expr := logic_or
    def _parse_expr(self) -> ExprNode:
        return self._parse_logic_or()

    def _parse_logic_or(self) -> ExprNode:
        node = self._parse_logic_and()
        while self._match(TokenKind.OR2):
            op = "||"
            rhs = self._parse_logic_and()
            node = BinaryNode(op, node, rhs)
        return node

    def _parse_logic_and(self) -> ExprNode:
        node = self._parse_equality()
        while self._match(TokenKind.AND2):
            op = "&&"
            rhs = self._parse_equality()
            node = BinaryNode(op, node, rhs)
        return node

    def _parse_equality(self) -> ExprNode:
        node = self._parse_comparison()
        if self._cur().kind in (TokenKind.EQ2, TokenKind.NEQ):
            if isinstance(node, BinaryNode) and node.op in {"<", "<=", ">", ">="}:
                t = self._cur()
                raise AlgebraSyntaxError(
                    "Chained comparisons are not supported.",
                    line=t.line,
                    col=t.col,
                    line_text=t.line_text,
                )
            op = self._advance()
            rhs = self._parse_comparison()
            if isinstance(rhs, BinaryNode) and rhs.op in {"<", "<=", ">", ">="}:
                t = self._cur()
                raise AlgebraSyntaxError(
                    "Chained comparisons are not supported.",
                    line=t.line,
                    col=t.col,
                    line_text=t.line_text,
                )
            node = BinaryNode("==" if op.kind is TokenKind.EQ2 else "!=", node, rhs)
        if self._cur().kind in (TokenKind.EQ2, TokenKind.NEQ):
            t = self._cur()
            raise AlgebraSyntaxError(
                "Chained equality comparisons are not supported.",
                line=t.line,
                col=t.col,
                line_text=t.line_text,
            )
        return node

    def _parse_comparison(self) -> ExprNode:
        node = self._parse_term()
        if self._cur().kind in (TokenKind.LT, TokenKind.LTE, TokenKind.GT, TokenKind.GTE):
            op = self._advance()
            rhs = self._parse_term()
            operators = {
                TokenKind.LT: "<",
                TokenKind.LTE: "<=",
                TokenKind.GT: ">",
                TokenKind.GTE: ">=",
            }
            node = BinaryNode(operators[op.kind], node, rhs)
        if self._cur().kind in (TokenKind.LT, TokenKind.LTE, TokenKind.GT, TokenKind.GTE):
            t = self._cur()
            raise AlgebraSyntaxError(
                "Chained comparisons are not supported.",
                line=t.line,
                col=t.col,
                line_text=t.line_text,
            )
        return node

    def _parse_term(self) -> ExprNode:
        node = self._parse_factor()
        while True:
            if self._match(TokenKind.PLUS):
                node = BinaryNode("+", node, self._parse_factor())
            elif self._match(TokenKind.MINUS):
                node = BinaryNode("-", node, self._parse_factor())
            else:
                break
        return node

    def _parse_factor(self) -> ExprNode:
        node = self._parse_unary()
        while True:
            if self._match(TokenKind.STAR):
                node = BinaryNode("*", node, self._parse_unary())
            elif self._match(TokenKind.SLASH):
                node = BinaryNode("/", node, self._parse_unary())
            else:
                break
        return node

    def _parse_power(self) -> ExprNode:
        # Power binds tighter than a prefix unary operator.  Its RHS is a
        # unary expression so that, for example, 2^-2 means 2^(-2).
        node = self._parse_primary()
        if self._match(TokenKind.POW2):
            rhs = self._parse_unary()
            return BinaryNode("**", node, rhs)
        if self._match(TokenKind.CARET):
            rhs = self._parse_unary()
            return BinaryNode("^", node, rhs)
        return node

    def _parse_unary(self) -> ExprNode:
        if self._match(TokenKind.PLUS):
            return UnaryNode("+", self._parse_unary())
        if self._match(TokenKind.MINUS):
            return UnaryNode("-", self._parse_unary())
        if self._match(TokenKind.BANG):
            return UnaryNode("!", self._parse_unary())
        return self._parse_power()

    def _parse_primary(self) -> ExprNode:
        t = self._cur()

        # Number
        if self._match(TokenKind.NUMBER):
            try:
                val = float(t.text)
            except Exception:
                logger.debug(f"Failed to parse numeric literal: {t.text}", exc_info=True)
                raise AlgebraSyntaxError("Invalid number.", line=t.line, col=t.col, line_text=t.line_text)
            if not math.isfinite(val):
                raise AlgebraSyntaxError(
                    "Numbers must be finite.",
                    line=t.line,
                    col=t.col,
                    line_text=t.line_text,
                )
            return NumberNode(val, t.text)

        # Species refs: "[" IDENT "]" [ "_0" ]
        if self._match(TokenKind.LBRACK):
            ident_tok = self._expect(TokenKind.IDENT, "Expected a species name inside [ ].")
            self._expect(TokenKind.RBRACK, "Expected a closing ']'.")
            # Either _0 or nothing
            if self._match(TokenKind.IDENT):
                # Only legal tail is "_0" but tokenized as IDENT; validate text
                tail = self.toks[self.i - 1]
                if tail.text != "_0":
                    raise AlgebraSyntaxError("Expected '_0' after a species reference.", line=tail.line, col=tail.col, line_text=tail.line_text)
                return SpeciesRefNode(ident_tok.text, "init")
            return SpeciesRefNode(ident_tok.text, "now")

        # Ident: could be function call or bare identifier
        if self._match(TokenKind.IDENT):
            ident = t.text
            if self._match(TokenKind.LPAREN):
                args: List[ExprNode] = []
                if self._cur().kind is not TokenKind.RPAREN:
                    args.append(self._parse_expr())
                    while self._match(TokenKind.COMMA):
                        args.append(self._parse_expr())
                self._expect(TokenKind.RPAREN, "Expected ')' to close the function call.")
                builtin = BUILTIN_FUNCTIONS.get(ident)
                if builtin is not None and not builtin.admits(len(args)):
                    raise AlgebraSyntaxError(f"{ident} takes {builtin.expected()}.",
                                             line=t.line, col=t.col, line_text=t.line_text)
                return CallNode(ident, tuple(args))
            else:
                return IdentNode(ident)

        # Parenthesized
        if self._match(TokenKind.LPAREN):
            node = self._parse_expr()
            self._expect(TokenKind.RPAREN, "Expected ')'.")
            return node

        # NEWLINE inside expression is unexpected
        if t.kind is TokenKind.NEWLINE:
            raise AlgebraSyntaxError("The expression ends too early.", line=t.line, col=t.col, line_text=t.line_text)

        # Fallback
        raise AlgebraSyntaxError("Unexpected token.", line=t.line, col=t.col, line_text=t.line_text)


# ------------------------------ Public API -----------------------------------

def parse_algebra(src: str) -> AlgebraBlock:
    """
    Parse an internal synthetic algebra source string and return its
    AlgebraBlock.

    Raises Algebra* errors carrying their line and column.
    """
    tokens = _lex(src)
    parser = Parser(tokens)
    return parser.parse_block()
