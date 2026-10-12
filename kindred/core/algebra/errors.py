"""
Algebra error taxonomy and formatting.

Error-formatting contract
------------------------
Error codes and kinds:
- E100 SyntaxError       (unexpected token)
- E110 NameError         (unknown symbol)
- E130 DomainError       (invalid domain such as sqrt of negative in real mode)
- E140 ZeroDiv           (division by zero)
- E150 TypeError         (non scalar where scalar required)
- E170 BoolCastError     (boolean in numeric-only context)

Formatting:
- The text of an error is its message. ``line`` and ``line_text`` locate the
  statement: a parse error locates a line of the program text it read, and an
  evaluation error the failing statement, which carries its authored line once
  admission has placed it. A consumer shows that location with the message.

Design
------
- We avoid clobbering Python built-ins by prefixing class names with `Algebra`.
- Column indices are 1-based externally; robust to out-of-range inputs.
- The base class carries `code`, `line`, `col`, `line_text` for consumers
  that map an error back to its statement.

This module performs no I/O and has no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

__all__ = [
    "AlgebraError",
    "AlgebraSyntaxError",
    "AlgebraNameError",
    "AlgebraDomainError",
    "AlgebraZeroDivError",
    "AlgebraTypeError",
    "AlgebraBoolCastError",
    "UNCOMPUTABLE_ERRORS",
]


# ----------------------------- base class ------------------------------------


@dataclass
class AlgebraError(Exception):
    """
    Base algebra error with code and source location.

    Attributes
    ----------
    msg : str
        Human-readable message (without code prefix).
    code : str
        Error code like "E100".
    line : int
        1-based line number.
    col : int
        1-based column.
    line_text : str | None
        Source line text for caret rendering.
    """

    msg: str
    code: str
    line: int
    col: int
    line_text: Optional[str] = None

    def __post_init__(self) -> None:
        # Normalize the column to be at least 1
        if self.col is None or self.col < 1:
            object.__setattr__(self, "col", 1)

    # Exception interface
    def __str__(self) -> str:
        return self.msg


# ----------------------------- subclasses ------------------------------------


class AlgebraSyntaxError(AlgebraError):
    def __init__(self, msg: str, *, line: int, col: int, line_text: str | None = None):
        super().__init__(msg=msg, code="E100", line=line, col=col, line_text=line_text)


class AlgebraNameError(AlgebraError):
    def __init__(self, msg: str, *, line: int, col: int, line_text: str | None = None):
        super().__init__(msg=msg, code="E110", line=line, col=col, line_text=line_text)


class AlgebraDomainError(AlgebraError):
    def __init__(self, msg: str, *, line: int, col: int, line_text: str | None = None):
        super().__init__(msg=msg, code="E130", line=line, col=col, line_text=line_text)


class AlgebraZeroDivError(AlgebraError):
    def __init__(self, msg: str, *, line: int, col: int, line_text: str | None = None):
        super().__init__(msg=msg, code="E140", line=line, col=col, line_text=line_text)


class AlgebraTypeError(AlgebraError):
    def __init__(self, msg: str, *, line: int, col: int, line_text: str | None = None):
        super().__init__(msg=msg, code="E150", line=line, col=col, line_text=line_text)


class AlgebraBoolCastError(AlgebraError):
    def __init__(self, msg: str, *, line: int, col: int, line_text: str | None = None):
        super().__init__(msg=msg, code="E170", line=line, col=col, line_text=line_text)


# A formula that cannot compute its value for these inputs: the value is
# outside its domain, for a param and a let alike.
UNCOMPUTABLE_ERRORS = (AlgebraDomainError, AlgebraZeroDivError, AlgebraTypeError)
