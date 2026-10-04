"""
Algebra error taxonomy and formatting.

Error-formatting contract
------------------------
Error codes and kinds:
- E100 SyntaxError       (unexpected token)
- E110 NameError         (unknown symbol)
- E120 ShadowError       (attempted shadowing of species or builtin)
- E130 DomainError       (invalid domain such as sqrt of negative in real mode)
- E140 ZeroDiv           (division by zero)
- E150 TypeError         (non scalar where scalar required)
- E170 BoolCastError     (boolean in numeric-only context)

Formatting requirements:
- Messages include line text, caret, and `L#:C#`.
- The caret points to the offending column.

Design
------
- We avoid clobbering Python built-ins by prefixing class names with `Algebra`.
- Column indices are 1-based externally; robust to out-of-range inputs.
- Tabs in the source line are expanded to spaces for stable caret placement.
- The base class carries `code`, `line`, `col`, `line_text` and
  renders a canonical one-line message plus a three-line excerpt.

This module performs no I/O and has no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

__all__ = [
    "AlgebraError",
    "AlgebraSyntaxError",
    "AlgebraNameError",
    "AlgebraShadowError",
    "AlgebraDomainError",
    "AlgebraZeroDivError",
    "AlgebraTypeError",
    "AlgebraBoolCastError",
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
        head = f"{self.code} {self.kind()}: {self.msg} @ L{self.line}:C{self.col}"
        excerpt = self._format_excerpt()
        return head if excerpt is None else f"{head}\n{excerpt}"

    # Human-readable kind derived from class name
    def kind(self) -> str:
        name = type(self).__name__
        if name.startswith("Algebra") and name.endswith("Error"):
            name = name[len("Algebra") : -len("Error")]
        return name or "Error"

    def _format_excerpt(self) -> Optional[str]:
        if not self.line_text:
            return None
        # Expand tabs for consistent visual columns
        src = self.line_text.expandtabs(4).rstrip("\n\r")
        # Clamp within the displayable range [1, len(src)+1]
        # Allow caret after EOL to signal unexpected EOF.
        start = min(max(1, self.col), max(1, len(src) + 1))
        marker = " " * (start - 1) + "^"

        loc = f"L{self.line}:C{self.col}"
        return f"{src}\n{marker}\n{loc}"


# ----------------------------- subclasses ------------------------------------


class AlgebraSyntaxError(AlgebraError):
    def __init__(self, msg: str, *, line: int, col: int, line_text: str | None = None):
        super().__init__(msg=msg, code="E100", line=line, col=col, line_text=line_text)


class AlgebraNameError(AlgebraError):
    def __init__(self, msg: str, *, line: int, col: int, line_text: str | None = None):
        super().__init__(msg=msg, code="E110", line=line, col=col, line_text=line_text)


class AlgebraShadowError(AlgebraError):
    def __init__(self, msg: str, *, line: int, col: int, line_text: str | None = None):
        super().__init__(msg=msg, code="E120", line=line, col=col, line_text=line_text)


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
