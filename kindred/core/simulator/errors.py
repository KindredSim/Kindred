"""
Enhanced error messages for DSL parsing with helpful suggestions.

Provides user-friendly error messages that include:
- Clear problem description
- Contextual information
- Suggestions for common mistakes
- Examples of correct syntax
"""

from __future__ import annotations

from typing import Optional, List

from kindred.core.exceptions import ErrorContext, ReactionValidationError


class DSLError(ReactionValidationError, ValueError):
    """
    Enhanced DSL validation or parsing error with helpful context.

    Attributes
    ----------
    message : str
        Main error message
    suggestion : str, optional
        Suggested fix or common solution
    examples : list of str, optional
        Valid syntax examples
    line_number : int, optional
        Line number where error occurred
    line_content : str, optional
        Content of the problematic line
    """

    def __init__(
        self,
        message: str,
        *,
        suggestion: Optional[str] = None,
        examples: Optional[List[str]] = None,
        line_number: Optional[int] = None,
        line_content: Optional[str] = None
    ):
        self.line_number = line_number
        self.line_content = line_content
        context = None
        if line_number is not None or line_content is not None:
            context = ErrorContext(line=line_number, line_text=line_content)
        super().__init__(
            message,
            code="E203",
            suggestion=suggestion,
            examples=examples,
            context=context,
        )


def invalid_number_error(value: str, field: str) -> DSLError:
    """Error for a non-numeric value where a number is expected."""
    return DSLError(f"{field} must be a number, not '{value}'.")


def missing_arrow_error() -> DSLError:
    """Error for a reaction line whose arrow follows its first ';'."""
    return DSLError(
        "The reaction, with '->', '<->', or '<=>', must come before the first ';'.",
        examples=[
            "A -> B; k=1.0",
            "A + B -> C; dG_act=75.5",
            "A <-> B; kf=1.0; Keq=2.0",
            "A <=> B; kf=1.0; Keq=2.0",
        ],
    )


def invalid_species_term_error(term: str) -> DSLError:
    """Error for malformed species term."""
    suggestion = (
        "Use format: 'A' or '2A' or '0.5A' (optionally with '*', for example '2*A'). "
        "Species names must start with a letter/underscore and contain only letters, digits, and underscores."
    )

    return DSLError(
        f"Invalid species term: {term}.",
        suggestion=suggestion,
        examples=[
            "A",
            "2A",
            "0.5B",
            "A + 2 B -> 3 C",
            "A + 2*B -> 3*C",
        ]
    )
