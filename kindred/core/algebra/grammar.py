"""
Algebra grammar specification and token kinds (data only).

Reference data for the Algebra DSL grammar, helpers, and protected symbols.
---------------------------------------------------------------------------
EBNF (power is right-associative):

    algebra_block  := { let_stmt }
    let_stmt       := "let" IDENT "=" expr newline
    expr           := logic_or
    logic_or       := logic_and { "||" logic_and }
    logic_and      := equality { "&&" equality }
    equality       := comparison { ("==" | "!=") comparison }
    comparison     := term { ("<"|"<="|">"|">=") term }
    term           := factor { ("+"|"-") factor }
    factor         := unary { ("*"|"/") unary }
    power          := primary [ ("**"|"^") unary ]    # right associative
    unary          := ("+"|"-"|"!") unary | power
    primary        := NUMBER | IDENT | func_call | species_ref | "(" expr ")"
    func_call      := IDENT "(" [ arg_list ] ")"
    arg_list       := expr { "," expr }
    species_ref    := "[" IDENT "]" | "[" IDENT "]_0"

Built-in functions:
    sqrt, ln, log10, log1p, exp, expm1, sin, cos, tan, abs, min, max, pow, erf
Helpers:
    heaviside(x), clip(x, lo, hi), ifelse(cond, a, b)

Protected symbols (read-only; listed here for reference only):
    k_B, h, h_bar, N_A, R, T
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Tuple


# ----------------------------- token kinds -----------------------------------

class TokenKind(str, Enum):
    # Structure
    NEWLINE = "NEWLINE"
    EOF = "EOF"

    # Literals and identifiers
    NUMBER = "NUMBER"
    IDENT = "IDENT"

    # Keywords
    LET = "LET"

    # Punctuation
    LPAREN = "LPAREN"        # (
    RPAREN = "RPAREN"        # )
    LBRACK = "LBRACK"        # [
    RBRACK = "RBRACK"        # ]
    COMMA = "COMMA"          # ,
    EQUALS = "EQUALS"        # =

    # Operators
    PLUS = "PLUS"            # +
    MINUS = "MINUS"          # -
    STAR = "STAR"            # *
    SLASH = "SLASH"          # /
    CARET = "CARET"          # ^
    POW2 = "POW2"            # **
    BANG = "BANG"            # !
    OR2 = "OR2"              # ||
    AND2 = "AND2"            # &&
    EQ2 = "EQ2"              # ==
    NEQ = "NEQ"              # !=
    LT = "LT"                # <
    LTE = "LTE"              # <=
    GT = "GT"                # >
    GTE = "GTE"              # >=

# ----------------------------- lexical spec (data only) ----------------------
# Intended for a future lexer. Order matters for maximal munch on multi-char ops.

# Regex fragments are Python-style, but this module performs no lexing itself.
# Whitespace (spaces/tabs) should be skipped by the lexer; NEWLINEs are tokens.

TOKEN_SPEC: List[Tuple[TokenKind, str]] = [
    # Newlines
    (TokenKind.NEWLINE, r"\r?\n"),
    # Multi-char operators first
    (TokenKind.POW2, r"\*\*"),
    (TokenKind.OR2, r"\|\|"),
    (TokenKind.AND2, r"&&"),
    (TokenKind.EQ2, r"=="),
    (TokenKind.NEQ, r"!="),
    (TokenKind.LTE, r"<="),
    (TokenKind.GTE, r">="),
    (TokenKind.LT, r"<"),
    (TokenKind.GT, r">"),
    # Single-char operators and punctuation
    (TokenKind.PLUS, r"\+"),
    (TokenKind.MINUS, r"-"),
    (TokenKind.STAR, r"\*"),
    (TokenKind.SLASH, r"/"),
    (TokenKind.CARET, r"\^"),
    (TokenKind.BANG, r"!"),
    (TokenKind.LPAREN, r"\("),
    (TokenKind.RPAREN, r"\)"),
    (TokenKind.LBRACK, r"\["),
    (TokenKind.RBRACK, r"\]"),
    (TokenKind.COMMA, r","),
    (TokenKind.EQUALS, r"="),
    # Literals
    # NUMBER: decimal or integer with optional exponent (e or E), no leading +
    (TokenKind.NUMBER, r"(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][+\-]?\d+)?"),
    # Identifiers: ASCII letters and underscore start; then letters/digits/underscore
    (TokenKind.IDENT, r"[A-Za-z_][A-Za-z0-9_]*"),
]

# Keywords that the lexer should fold to specific TokenKind when IDENT matches.
KEYWORDS: Dict[str, TokenKind] = {
    "let": TokenKind.LET,
}

# ----------------------------- precedence ------------------------------------

# Higher number means higher precedence.
PRECEDENCE: Dict[str, int] = {
    "||": 1,
    "&&": 2,
    "==": 3,
    "!=": 3,
    "<": 4,
    "<=": 4,
    ">": 4,
    ">=": 4,
    "+": 5,
    "-": 5,
    "*": 6,
    "/": 6,
    "unary": 7,  # synthetic precedence for unary +, -, !
    "**": 8,   # power
    "^": 8,    # power alias
}

ASSOCIATIVITY: Dict[str, str] = {
    "||": "left",
    "&&": "left",
    "==": "left",
    "!=": "left",
    "<": "left",
    "<=": "left",
    ">": "left",
    ">=": "left",
    "+": "left",
    "-": "left",
    "*": "left",
    "/": "left",
    "**": "right",  # right-associative power
    "^": "right",   # right-associative power
    "unary": "right",
}


__all__ = [
    "TokenKind",
    "TOKEN_SPEC",
    "KEYWORDS",
    "PRECEDENCE",
    "ASSOCIATIVITY",
]
