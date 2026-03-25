"""
Algebra grammar specification and token kinds (data only).

Reference data for the Algebra DSL grammar, helpers, and protected symbols.
---------------------------------------------------------------------------
EBNF (power is right-associative):

    algebra_block  := "# Algebra" newline { let_stmt }
    let_stmt       := "let" IDENT "=" expr newline
    expr           := logic_or
    logic_or       := logic_and { "||" logic_and }
    logic_and      := equality { "&&" equality }
    equality       := comparison { ("==" | "!=") comparison }
    comparison     := term { ("<"|"<="|">"|">=") term }
    term           := factor { ("+"|"-") factor }
    factor         := power { ("*"|"/") power }
    power          := unary { ("**"|"^") unary }      # right associative
    unary          := ("+"|"-"|"!") unary | primary
    primary        := NUMBER | IDENT | func_call | species_ref | "(" expr ")"
    func_call      := IDENT "(" [ arg_list ] ")"
    arg_list       := expr { "," expr }
    species_ref    := "[" IDENT "]" | "[" IDENT "]_0" | "[" IDENT "]" "(" "T0" ")"

Built-in functions:
    sqrt, ln, log10, log1p, exp, expm1, sin, cos, tan, abs, min, max, pow, erf
Helpers:
    heaviside(x), clip(x, lo, hi), ifelse(cond, a, b)

Protected symbols (read-only; listed here for reference only):
    k_B, h, ħ, N_A, R, Rkcal, T
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Tuple


# ----------------------------- EBNF (reference) ------------------------------

EBNF: str = (
    "algebra_block  := \"# Algebra\" newline { let_stmt }\n"
    "let_stmt       := \"let\" IDENT \"=\" expr newline\n"
    "expr           := logic_or\n"
    "logic_or       := logic_and { \"||\" logic_and }\n"
    "logic_and      := equality { \"&&\" equality }\n"
    "equality       := comparison { (\"==\" | \"!=\") comparison }\n"
    "comparison     := term { (\"<\"|\"<=\"|\">\"|\">=\") term }\n"
    "term           := factor { (\"+\"|\"-\") factor }\n"
    "factor         := power { (\"*\"|\"/\") power }\n"
    "power          := unary { (\"**\"|\"^\") unary }  # right associative\n"
    "unary          := (\"+\"|\"-\"|\"!\") unary | primary\n"
    "primary        := NUMBER | IDENT | func_call | species_ref | \"(\" expr \")\"\n"
    "func_call      := IDENT \"(\" [ arg_list ] \")\"\n"
    "arg_list       := expr { \",\" expr }\n"
    "species_ref    := \"[\" IDENT \"]\" | \"[\" IDENT \"]_0\" | \"[\" IDENT \"]\" \"(\" \"T0\" \")\"\n"
)


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
    T0 = "T0"

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

    # Header marker
    HASH = "HASH"            # #   (used to match "# Algebra")


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
    (TokenKind.HASH, r"#"),
    # Literals
    # NUMBER: decimal or integer with optional exponent (e or E), no leading +
    (TokenKind.NUMBER, r"(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][+\-]?\d+)?"),
    # Identifiers: ASCII letters and underscore start; then letters/digits/underscore
    (TokenKind.IDENT, r"[A-Za-z_][A-Za-z0-9_]*"),
]

# Keywords that the lexer should fold to specific TokenKind when IDENT matches.
KEYWORDS: Dict[str, TokenKind] = {
    "let": TokenKind.LET,
    "T0": TokenKind.T0,
    # "Algebra" stays IDENT; the header is recognized as "#"+"Algebra" at the parser level.
}

# Built-in function names (reference for parser validation; not enforced here).
BUILTIN_FUNCTIONS: List[str] = [
    "sqrt", "ln", "log10", "log1p", "exp", "expm1",
    "sin", "cos", "tan", "abs", "min", "max", "pow", "erf",
]

HELPER_FUNCTIONS: List[str] = ["heaviside", "clip", "ifelse"]


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
    "**": 7,   # power
    "^": 7,    # power alias
    "unary": 8,  # synthetic precedence for unary +, -, !
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
    "EBNF",
    "TokenKind",
    "TOKEN_SPEC",
    "KEYWORDS",
    "BUILTIN_FUNCTIONS",
    "HELPER_FUNCTIONS",
    "PRECEDENCE",
    "ASSOCIATIVITY",
]
