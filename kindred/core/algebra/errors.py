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
- E160 NonFiniteError    (a result that is not a finite number)
- E170 BoolCastError     (boolean in numeric-only context)

Formatting:
- ``message`` is the sentence; the text of an error is that sentence located
  on ``line_text``. ``line`` and ``line_text`` locate the statement: a parse
  error locates a line of the program text it read, which its admission maps
  to an authored line, and an evaluation error the failing statement, which
  carries its authored line once admission has placed it.

Design
------
- We avoid clobbering Python built-ins by prefixing class names with `Algebra`.
- Column indices are 1-based externally; robust to out-of-range inputs.
- The base class carries `code`, `line`, `col`, `line_text` for consumers
  that map an error back to its statement, and is a Kindred error, so its
  text is located on its line and owners add context to the same error.

This module performs no I/O and has no side effects.
"""

from __future__ import annotations

from kindred.core.exceptions import ErrorContext, KindredError, ValueDomainError

__all__ = [
    "AlgebraError",
    "AlgebraSyntaxError",
    "AlgebraNameError",
    "AlgebraDomainError",
    "AlgebraZeroDivError",
    "AlgebraTypeError",
    "AlgebraNonFiniteError",
    "AlgebraBoolCastError",
    "AlgebraUndefinedError",
    "UNCOMPUTABLE_ERRORS",
]


# ----------------------------- base class ------------------------------------


class AlgebraError(KindredError):
    """
    Base algebra error with code and source location.

    Attributes
    ----------
    message : str
        Human-readable message (without code prefix or location).
    code : str
        Error code like "E100".
    line : int
        1-based line number.
    col : int
        1-based column.
    line_text : str | None
        Source line text for caret rendering.

    As a Kindred error its text is the message located on that line; owners it
    passes through add what they know to the same error.
    """

    def __init__(self, msg: str, *, code: str, line: int, col: int, line_text: str | None = None):
        self.line = line
        # Normalize the column to be at least 1
        self.col = 1 if col is None or col < 1 else col
        self.line_text = line_text
        super().__init__(
            msg, code=code,
            context=None if line_text is None else ErrorContext(line=line, line_text=line_text),
        )


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


class AlgebraNonFiniteError(AlgebraError):
    def __init__(self, msg: str, *, line: int, col: int, line_text: str | None = None):
        super().__init__(msg=msg, code="E160", line=line, col=col, line_text=line_text)


class AlgebraBoolCastError(AlgebraError):
    def __init__(self, msg: str, *, line: int, col: int, line_text: str | None = None):
        super().__init__(msg=msg, code="E170", line=line, col=col, line_text=line_text)


class AlgebraUndefinedError(AlgebraNameError):
    """A let reads another let at a time point where that let is undefined."""

    def __init__(self, msg: str, *, dependency: str, line: int, col: int, line_text: str | None = None):
        super().__init__(msg, line=line, col=col, line_text=line_text)
        self.dependency = dependency


# A value that cannot be calculated for these inputs: outside its domain, for
# a param, a let, a rate constant and Keq alike.
UNCOMPUTABLE_ERRORS = (AlgebraDomainError, AlgebraZeroDivError, AlgebraNonFiniteError, ValueDomainError)
