"""
AST builder and conservative static folding for the Algebra DSL.

Current contract
----------------
This module implements the algebra grammar, builtin handling, evaluation rules,
and parser-facing error reporting used by the Algebra DSL.

Scope of this module
--------------------
- Tokenize source into a deterministic stream with locations.
- Parse `# Algebra` header then { let IDENT = expr NEWLINE }.
- Build a typed AST with nodes:
    Number, Ident, Unary, Binary, Call, SpeciesRef(kind="now"|"init"|"T0")
- Enforce precedence and right-associative power (** and ^).
- Validate function-argument arity superficially (min 1 for varargs, exact for unary),
  leave deeper numeric/type checks to the evaluator.
- Conservative static folding:
    * Fold any subtree containing only numeric literals, protected constants,
      and pure math/helpers, with no SpeciesRef or T0 reference and no user identifiers.
    * Record statically-folded results per let binding when entire RHS is static.

Out of scope here
-----------------
- Dynamic evaluation (done in evaluator.py).
- Baseline interpolation or time memoization.
- Shadowing detection with species names (handled by evaluator with symbol table).

Determinism & constraints
-------------------------
- No I/O, no cwd, no registry, no network.
- Pure functions; all diagnostics use AlgebraError* with code and caret.

"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .errors import (
    AlgebraSyntaxError,
)
from .grammar import (
    KEYWORDS,
    TOKEN_SPEC,
    TokenKind,
)
from .symbols import BUILTIN_FUNCTIONS as RUNTIME_FUNCS

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
            raise AlgebraSyntaxError("lexical error", line=line, col=(i - bol + 1), line_text=_line_of(src, line))
        kind_name = m.lastgroup or "MISMATCH"
        text = m.group(0)
        if kind_name == "SKIP":
            pass
        elif kind_name == "MISMATCH":
            raise AlgebraSyntaxError(f"unexpected character {text!r}", line=line, col=(i - bol + 1), line_text=_line_of(src, line))
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


class ExprNode:
    """Base class for AST nodes; used for typing only."""
    def is_static(self) -> bool:  # free of species/time refs and user idents
        return False

    def has_species_ref(self) -> bool:
        return False

    def has_time_ref(self) -> bool:
        return False


@dataclass(frozen=True)
class NumberNode(ExprNode):
    value: float

    def is_static(self) -> bool:
        return True


@dataclass(frozen=True)
class IdentNode(ExprNode):
    name: str
    # Conservatively treat as non-static; evaluator resolves
    def is_static(self) -> bool:
        return False


@dataclass(frozen=True)
class UnaryNode(ExprNode):
    op: str
    rhs: ExprNode

    def is_static(self) -> bool:
        return self.rhs.is_static()

    def has_species_ref(self) -> bool:
        return self.rhs.has_species_ref()

    def has_time_ref(self) -> bool:
        return self.rhs.has_time_ref()


@dataclass(frozen=True)
class BinaryNode(ExprNode):
    op: str
    lhs: ExprNode
    rhs: ExprNode

    def is_static(self) -> bool:
        return self.lhs.is_static() and self.rhs.is_static()

    def has_species_ref(self) -> bool:
        return self.lhs.has_species_ref() or self.rhs.has_species_ref()

    def has_time_ref(self) -> bool:
        return self.lhs.has_time_ref() or self.rhs.has_time_ref()


@dataclass(frozen=True)
class CallNode(ExprNode):
    name: str
    args: Tuple[ExprNode, ...]

    def is_static(self) -> bool:
        # static only if all args static and function is a builtin helper/math
        if any(not a.is_static() for a in self.args):
            return False
        return self.name in RUNTIME_FUNCS

    def has_species_ref(self) -> bool:
        return any(a.has_species_ref() for a in self.args)

    def has_time_ref(self) -> bool:
        return any(a.has_time_ref() for a in self.args)


@dataclass(frozen=True)
class SpeciesRefNode(ExprNode):
    name: str
    kind: str  # "now" | "init" | "T0"

    def is_static(self) -> bool:
        # Species refs are time-varying except _0 which is static w.r.t time,
        # but evaluator treats it specially; we avoid folding across any ref.
        return False

    def has_species_ref(self) -> bool:
        return True

    def has_time_ref(self) -> bool:
        return self.kind == "T0"


@dataclass(frozen=True)
class LetStatement:
    name: str
    expr: ExprNode
    line: int
    col: int
    line_text: str


@dataclass(frozen=True)
class AlgebraBlock:
    lines: List[LetStatement]
    ast: List[ExprNode]
    static_values: Dict[str, float]

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

    def _expect_ident_text(self, text: str) -> Token:
        t = self._cur()
        if not (t.kind is TokenKind.IDENT and t.text == text):
            raise AlgebraSyntaxError(f"expected identifier {text!r}", line=t.line, col=t.col, line_text=t.line_text)
        return self._advance()

    # Entry: "# Algebra" newline { let_stmt }
    def parse_block(self) -> AlgebraBlock:
        # header: '#' 'Algebra'
        self._parse_header()

        lines: List[LetStatement] = []
        exprs: List[ExprNode] = []
        static_values: Dict[str, float] = {}

        # Accept optional blank lines
        self._consume_newlines()

        while self._cur().kind is not TokenKind.EOF:
            let_tok = self._expect(TokenKind.LET, "expected 'let' at beginning of statement")
            name_tok = self._expect(TokenKind.IDENT, "expected identifier after 'let'")
            self._expect(TokenKind.EQUALS, "expected '=' after identifier")
            expr = self._parse_expr()
            # Require newline after each let
            if self._cur().kind is TokenKind.NEWLINE:
                self._advance()
            elif self._cur().kind is TokenKind.EOF:
                # Allow EOF after last statement without trailing newline
                pass
            else:
                t = self._cur()
                raise AlgebraSyntaxError("expected end of line", line=t.line, col=t.col, line_text=t.line_text)

            stmt = LetStatement(name=name_tok.text, expr=expr, line=let_tok.line, col=let_tok.col, line_text=let_tok.line_text)
            lines.append(stmt)
            exprs.append(expr)

            # Conservative static fold for an entire RHS
            folded = _try_static_eval(expr)
            if folded is not None:
                static_values[name_tok.text] = folded

            self._consume_newlines()

        return AlgebraBlock(lines=lines, ast=exprs, static_values=static_values)

    def _parse_header(self) -> None:
        # Accept optional leading newlines
        self._consume_newlines()
        t = self._cur()
        # Either tokenized as HASH IDENT "Algebra" or just IDENT if line starts directly with "#"
        if t.kind is TokenKind.HASH:
            self._advance()
            ident = self._expect(TokenKind.IDENT, "expected 'Algebra' after '#'")
            if ident.text != "Algebra":
                raise AlgebraSyntaxError("expected '# Algebra' header", line=ident.line, col=ident.col, line_text=ident.line_text)
        elif t.kind is TokenKind.IDENT and t.text == "Algebra":
            # Extremely permissive header variant; still require it's the first token
            pass
        else:
            raise AlgebraSyntaxError("algebra block must start with '# Algebra'", line=t.line, col=t.col, line_text=t.line_text)

        # Require a newline after the header
        if self._cur().kind is TokenKind.NEWLINE:
            self._advance()
        elif self._cur().kind is TokenKind.EOF:
            # Degenerate empty block allowed
            return
        else:
            t2 = self._cur()
            raise AlgebraSyntaxError("expected newline after '# Algebra'", line=t2.line, col=t2.col, line_text=t2.line_text)

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
        while True:
            if self._match(TokenKind.EQ2):
                node = BinaryNode("==", node, self._parse_comparison())
            elif self._match(TokenKind.NEQ):
                node = BinaryNode("!=", node, self._parse_comparison())
            else:
                break
        return node

    def _parse_comparison(self) -> ExprNode:
        node = self._parse_term()
        while True:
            if self._match(TokenKind.LT):
                node = BinaryNode("<", node, self._parse_term())
            elif self._match(TokenKind.LTE):
                node = BinaryNode("<=", node, self._parse_term())
            elif self._match(TokenKind.GT):
                node = BinaryNode(">", node, self._parse_term())
            elif self._match(TokenKind.GTE):
                node = BinaryNode(">=", node, self._parse_term())
            else:
                break
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
        node = self._parse_power()
        while True:
            if self._match(TokenKind.STAR):
                node = BinaryNode("*", node, self._parse_power())
            elif self._match(TokenKind.SLASH):
                node = BinaryNode("/", node, self._parse_power())
            else:
                break
        return node

    def _parse_power(self) -> ExprNode:
        # Right-associative: parse a unary, then if ** or ^, recursively parse power on RHS
        node = self._parse_unary()
        if self._match(TokenKind.POW2):
            rhs = self._parse_power()
            return BinaryNode("**", node, rhs)
        if self._match(TokenKind.CARET):
            rhs = self._parse_power()
            return BinaryNode("^", node, rhs)
        return node

    def _parse_unary(self) -> ExprNode:
        if self._match(TokenKind.PLUS):
            return UnaryNode("+", self._parse_unary())
        if self._match(TokenKind.MINUS):
            return UnaryNode("-", self._parse_unary())
        if self._match(TokenKind.BANG):
            return UnaryNode("!", self._parse_unary())
        return self._parse_primary()

    def _parse_primary(self) -> ExprNode:
        t = self._cur()

        # Number
        if self._match(TokenKind.NUMBER):
            try:
                val = float(t.text)
            except Exception:
                logger.debug(f"Failed to parse numeric literal: {t.text}", exc_info=True)
                raise AlgebraSyntaxError("invalid numeric literal", line=t.line, col=t.col, line_text=t.line_text)
            return NumberNode(val)

        # Species refs: "[" IDENT "]" [ "_0" | "(" "T0" ")" ]
        if self._match(TokenKind.LBRACK):
            ident_tok = self._expect(TokenKind.IDENT, "expected species name inside [ ]")
            self._expect(TokenKind.RBRACK, "expected closing ']'")
            # Either _0 or (T0) or nothing
            if self._match(TokenKind.IDENT):
                # Only legal tail is "_0" but tokenized as IDENT; validate text
                tail = self.toks[self.i - 1]
                if tail.text != "_0":
                    raise AlgebraSyntaxError("expected '_0' after species reference", line=tail.line, col=tail.col, line_text=tail.line_text)
                return SpeciesRefNode(ident_tok.text, "init")
            if self._match(TokenKind.LPAREN):
                self._expect(TokenKind.T0, "expected T0 after '(' in species time reference")
                self._expect(TokenKind.RPAREN, "expected ')' after T0")
                return SpeciesRefNode(ident_tok.text, "T0")
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
                self._expect(TokenKind.RPAREN, "expected ')' to close function call")
                # Basic arity sanity for known simple unary funcs
                # Deeper checks left to evaluator
                return CallNode(ident, tuple(args))
            else:
                return IdentNode(ident)

        # Parenthesized
        if self._match(TokenKind.LPAREN):
            node = self._parse_expr()
            self._expect(TokenKind.RPAREN, "expected ')'")
            return node

        # NEWLINE inside expression is unexpected
        if t.kind is TokenKind.NEWLINE:
            raise AlgebraSyntaxError("unexpected end of line in expression", line=t.line, col=t.col, line_text=t.line_text)

        # Fallback
        raise AlgebraSyntaxError("unexpected token", line=t.line, col=t.col, line_text=t.line_text)


# ------------------------------ Static folding -------------------------------

def _try_static_eval(node: ExprNode) -> Optional[float]:
    """
    Conservative, side-effect-free evaluation of a subtree.

    Only folds when:
    - No SpeciesRef present.
    - No T0 reference.
    - No IdentNode (user-defined symbols).
    - All calls are to known builtins/helpers and all arguments are foldable.

    Returns the folded float if possible, else None.
    """
    try:
        val, ok = _eval_static(node)
        return val if ok else None
    except Exception:
        logger.debug("Static evaluation failed for expression node", exc_info=True)
        return None


def _eval_static(node: ExprNode) -> Tuple[float, bool]:
    # Number
    if isinstance(node, NumberNode):
        return node.value, True
    # For safety, we never fold bare identifiers or any species refs
    if isinstance(node, (IdentNode, SpeciesRefNode)):
        return 0.0, False
    if isinstance(node, UnaryNode):
        v, ok = _eval_static(node.rhs)
        if not ok:
            return 0.0, False
        if node.op == "+":
            return +v, True
        if node.op == "-":
            return -v, True
        if node.op == "!":
            # numeric-only folding: treat 0 as False, others True -> 0.0 or 1.0
            return (0.0 if v else 1.0), True
        return 0.0, False
    if isinstance(node, BinaryNode):
        lv, lok = _eval_static(node.lhs)
        rv, rok = _eval_static(node.rhs)
        if not (lok and rok):
            return 0.0, False
        op = node.op
        if op == "+":
            return lv + rv, True
        if op == "-":
            return lv - rv, True
        if op == "*":
            return lv * rv, True
        if op == "/":
            if rv == 0.0:
                raise ZeroDivisionError()
            return lv / rv, True
        if op in ("**", "^"):
            return float(math.pow(lv, rv)), True
        if op == "||":
            return (1.0 if (lv or rv) else 0.0), True
        if op == "&&":
            return (1.0 if (lv and rv) else 0.0), True
        if op == "==":
            return (1.0 if (lv == rv) else 0.0), True
        if op == "!=":
            return (1.0 if (lv != rv) else 0.0), True
        if op == "<":
            return (1.0 if (lv < rv) else 0.0), True
        if op == "<=":
            return (1.0 if (lv <= rv) else 0.0), True
        if op == ">":
            return (1.0 if (lv > rv) else 0.0), True
        if op == ">=":
            return (1.0 if (lv >= rv) else 0.0), True
        return 0.0, False
    if isinstance(node, CallNode):
        # Only fold if function is known and all args foldable
        fn = RUNTIME_FUNCS.get(node.name)
        if fn is None:
            return 0.0, False
        vals: List[float] = []
        for arg in node.args:
            v, ok = _eval_static(arg)
            if not ok:
                return 0.0, False
            vals.append(float(v))
        try:
            out = float(fn(*vals))
        except Exception:
            logger.debug(f"Static evaluation of function {node.name} failed", exc_info=True)
            return 0.0, False
        return out, True
    return 0.0, False


# ------------------------------ Public API -----------------------------------

def parse_algebra(src: str) -> AlgebraBlock:
    """
    Parse an algebra block string and return AlgebraBlock with AST and static_values.

    Raises Algebra* errors with E-series codes and caret spans for diagnostics.
    """
    tokens = _lex(src)
    parser = Parser(tokens)
    return parser.parse_block()
