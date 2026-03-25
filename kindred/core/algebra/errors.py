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
- E160 TimeRefError      (baseline missing or ambiguous for [A](T0))
- E170 BoolCastError     (boolean in numeric-only context)

Formatting requirements:
- Messages include line text, caret, and `L#:C#`.
- Caret should point to the offending column; when a span is known,
  highlight span with carets/tilde.

Design
------
- We avoid clobbering Python built-ins by prefixing class names with `Algebra`.
- Column indices are 1-based externally; robust to out-of-range inputs.
- Tabs in the source line are expanded to spaces for stable caret placement.
- The base class carries `code`, `line`, `col`, `end_col`, `line_text` and
  renders a canonical one-line message plus a three-line excerpt.

This module performs no I/O and has no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Protocol

class _ErrorFactory(Protocol):
    def __call__(
        self,
        msg: str,
        *,
        line: int,
        col: int,
        end_col: int | None = None,
        line_text: str | None = None,
    ) -> "AlgebraError": ...

__all__ = [
    "AlgebraError",
    "AlgebraSyntaxError",
    "AlgebraNameError",
    "AlgebraShadowError",
    "AlgebraDomainError",
    "AlgebraZeroDivError",
    "AlgebraTypeError",
    "AlgebraTimeRefError",
    "AlgebraBoolCastError",
    "ERROR_CLASSES",
    "error_from_kind",
    "make_error",
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
        1-based start column.
    end_col : int | None
        1-based end column (inclusive). If None, a single-caret marker is used.
    line_text : str | None
        Source line text for caret rendering.
    """

    msg: str
    code: str
    line: int
    col: int
    end_col: Optional[int] = None
    line_text: Optional[str] = None

    def __post_init__(self) -> None:
        # Normalize columns to be at least 1
        # Use object.__setattr__ for frozen dataclass
        if self.col is None or self.col < 1:
            object.__setattr__(self, "col", 1)
        if self.end_col is not None and self.end_col < self.col:
            object.__setattr__(self, "end_col", self.col)

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
        # Compute safe bounds (1-based columns)
        start = max(1, self.col)
        end = self.end_col if self.end_col is not None else self.col

        # Clamp within the displayable range [1, len(src)+1]
        # Allow caret after EOL to signal unexpected EOF.
        max_col = max(1, len(src) + 1)
        start = min(start, max_col)
        end = min(max(start, end), max_col)

        # Build marker line: spaces then ^ for single point, or ^~~~^ for span
        prefix_spaces = " " * (start - 1)
        if end == start:
            marker = prefix_spaces + "^"
        else:
            span_len = max(1, end - start)
            marker = prefix_spaces + "^" + ("~" * (span_len - 1)) + "^"

        loc = f"L{self.line}:C{self.col}"
        return f"{src}\n{marker}\n{loc}"


# ----------------------------- subclasses ------------------------------------


class AlgebraSyntaxError(AlgebraError):
    def __init__(self, msg: str, *, line: int, col: int, end_col: int | None = None, line_text: str | None = None):
        super().__init__(msg=msg, code="E100", line=line, col=col, end_col=end_col, line_text=line_text)


class AlgebraNameError(AlgebraError):
    def __init__(self, msg: str, *, line: int, col: int, end_col: int | None = None, line_text: str | None = None):
        super().__init__(msg=msg, code="E110", line=line, col=col, end_col=end_col, line_text=line_text)


class AlgebraShadowError(AlgebraError):
    def __init__(self, msg: str, *, line: int, col: int, end_col: int | None = None, line_text: str | None = None):
        super().__init__(msg=msg, code="E120", line=line, col=col, end_col=end_col, line_text=line_text)


class AlgebraDomainError(AlgebraError):
    def __init__(self, msg: str, *, line: int, col: int, end_col: int | None = None, line_text: str | None = None):
        super().__init__(msg=msg, code="E130", line=line, col=col, end_col=end_col, line_text=line_text)


class AlgebraZeroDivError(AlgebraError):
    def __init__(self, msg: str, *, line: int, col: int, end_col: int | None = None, line_text: str | None = None):
        super().__init__(msg=msg, code="E140", line=line, col=col, end_col=end_col, line_text=line_text)


class AlgebraTypeError(AlgebraError):
    def __init__(self, msg: str, *, line: int, col: int, end_col: int | None = None, line_text: str | None = None):
        super().__init__(msg=msg, code="E150", line=line, col=col, end_col=end_col, line_text=line_text)


class AlgebraTimeRefError(AlgebraError):
    def __init__(self, msg: str, *, line: int, col: int, end_col: int | None = None, line_text: str | None = None):
        super().__init__(msg=msg, code="E160", line=line, col=col, end_col=end_col, line_text=line_text)


class AlgebraBoolCastError(AlgebraError):
    def __init__(self, msg: str, *, line: int, col: int, end_col: int | None = None, line_text: str | None = None):
        super().__init__(msg=msg, code="E170", line=line, col=col, end_col=end_col, line_text=line_text)


# Public mapping for parsers/evaluators that dispatch by kind string
ERROR_CLASSES: Dict[str, _ErrorFactory] = {
    "SyntaxError": AlgebraSyntaxError,
    "NameError": AlgebraNameError,
    "ShadowError": AlgebraShadowError,
    "DomainError": AlgebraDomainError,
    "ZeroDiv": AlgebraZeroDivError,
    "TypeError": AlgebraTypeError,
    "TimeRefError": AlgebraTimeRefError,
    "BoolCastError": AlgebraBoolCastError,
}


def error_from_kind(kind: str) -> _ErrorFactory:
    """
    Return the AlgebraError subclass for a given spec kind name.

    Examples
    --------
    error_from_kind("SyntaxError") -> AlgebraSyntaxError
    """
    try:
        return ERROR_CLASSES[kind]
    except KeyError:
        raise KeyError(f"unknown algebra error kind {kind!r}") from None


# ----------------------------- convenience API -------------------------------


def make_error(
    kind: str,
    msg: str,
    *,
    line: int,
    col: int,
    end_col: int | None = None,
    line_text: str | None = None,
) -> AlgebraError:
    """
    Construct an AlgebraError of the given kind with location and excerpt.

    Parameters
    ----------
    kind : str
        One of: "SyntaxError","NameError","ShadowError","DomainError",
                "ZeroDiv","TypeError","TimeRefError","BoolCastError".
    msg : str
        Human-readable message (without code prefix).
    line : int
        1-based source line number.
    col : int
        1-based start column.
    end_col : int | None
        1-based inclusive end column; if None, a single-caret is rendered.
    line_text : str | None
        Source line for caret rendering.

    Returns
    -------
    AlgebraError
    """
    factory = error_from_kind(kind)
    return factory(msg, line=line, col=col, end_col=end_col, line_text=line_text)
