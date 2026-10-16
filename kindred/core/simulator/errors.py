"""Errors for DSL parsing: one sentence of what is expected, on its line."""

from __future__ import annotations

from typing import Optional

from kindred.core.exceptions import ErrorContext, ReactionValidationError


class DSLError(ReactionValidationError, ValueError):
    """DSL validation or parsing error, located on its authored line when known."""

    def __init__(
        self,
        message: str,
        *,
        line_number: Optional[int] = None,
        line_content: Optional[str] = None
    ):
        self.line_number = line_number
        self.line_content = line_content
        context = None
        if line_number is not None or line_content is not None:
            context = ErrorContext(line=line_number, line_text=line_content)
        super().__init__(message, code="E203", context=context)


def invalid_number_error(value: str, field: str) -> DSLError:
    """Error for a non-numeric value where a number is expected."""
    return DSLError(f"{field} must be a number, not '{value}'.")


def missing_arrow_error() -> DSLError:
    """Error for a reaction line whose arrow follows its first ';'."""
    return DSLError("The reaction, with '->', '<->', or '<=>', must come before the first ';'.")


def invalid_species_term_error(term: str) -> DSLError:
    """Error for malformed species term."""
    return DSLError(
        f"Invalid species term '{term}': a term is an optional positive coefficient followed by "
        "a name of letters, digits, and underscores that does not start with a digit."
    )
