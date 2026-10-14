"""
Structured exception types for Kindred user-facing and integration boundaries.

This module defines the `KindredError` hierarchy plus optional context and
error codes. Every display shows ``str(exc)``, so the text of an error is its
user-facing message, located on the authored line it names. Codes and details
are structured facts for reports and logs and never appear in the text.

Error Categories
----------------
- **Validation**: Input validation failures (E200-E299)
- **Simulation**: Runtime simulation errors (E300-E399)
- **Fitting**: Parameter estimation errors (E400-E499)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

__all__ = [
    # Base
    "KindredError",
    "ErrorContext",
    "located",
    # Validation
    "ValidationError",
    "ReactionValidationError",
    # Simulation
    "SimulationError",
    "SolverError",
    "ValueDomainError",
    # Fitting
    "FittingError",
    "FitSimulationError",
    "FittingCancelled",
]


# -------------------------------- Base Classes ---------------------------------


@dataclass
class ErrorContext:
    """
    Context information for error reporting.

    Attributes
    ----------
    line : int, optional
        Line number (1-based) where error occurred
    line_text : str, optional
        Source line text for context
    """
    line: Optional[int] = None
    line_text: Optional[str] = None


def located(message: str, line: Optional[int], line_text: Optional[str]) -> str:
    """One-line text for a message about an authored line, when the line is known."""
    return message if line is None or line_text is None else f"Line {line} ({line_text.strip()}): {message}"


class KindredError(Exception):
    """
    Base exception for all Kindred errors.

    Its text is the message, located on the authored line from ``context``
    when known. The code and details are structured attributes for reports
    and logs; they are never part of the text.

    Attributes
    ----------
    message : str
        Main error message
    code : str
        Error code (e.g., "E201")
    context : ErrorContext, optional
        Source location and context
    details : dict, optional
        Additional structured error details
    """

    def __init__(
        self,
        message: str,
        *,
        code: str = "E000",
        context: Optional[ErrorContext] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.code = code
        self.context = context
        self.details = details or {}
        super().__init__(self._build_message())

    def _build_message(self) -> str:
        if self.context is None:
            return self.message
        return located(self.message, self.context.line, self.context.line_text)

    def annotate(
        self, *, prefix: str = "", suffix: str = "", context: Optional[ErrorContext] = None, **details: Any,
    ) -> None:
        """Add what an owner the error passes through knows, keeping the error
        itself: words around the message, the authored line when it has none,
        and facts it does not record yet."""
        self.message = f"{prefix}{self.message}{suffix}"
        if self.context is None:
            self.context = context
        for name, value in details.items():
            self.details.setdefault(name, value)
        self.args = (self._build_message(),)


# ----------------------------- Validation Errors -------------------------------


class ValidationError(KindredError):
    """Base class for validation errors (E200-E299)."""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("code", "E200")
        super().__init__(message, **kwargs)


class ReactionValidationError(ValidationError):
    """Error for invalid reaction specifications."""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("code", "E203")
        super().__init__(message, **kwargs)


# ---------------------------- Simulation Errors --------------------------------


class SimulationError(KindredError):
    """Base class for simulation runtime errors (E300-E399)."""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("code", "E300")
        super().__init__(message, **kwargs)


class SolverError(SimulationError):
    """Error during ODE solver execution."""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("code", "E301")
        super().__init__(message, **kwargs)


class ValueDomainError(SimulationError, ValueError):
    """A value lies outside its scientific domain: an authored or candidate
    input, or a calculated rate constant, Keq or physical coordinate."""

    def __init__(self, field: str, message: str, **kwargs: Any):
        kwargs.setdefault("code", "E304")
        super().__init__(message, **kwargs)
        self.field = field


class SimulationCancelled(SimulationError):
    """Raised when a simulation is cancelled by user request."""
    def __init__(self, message: str = "Simulation canceled by user", **kwargs):
        kwargs.setdefault("code", "E305")
        super().__init__(message, **kwargs)


# ------------------------------ Fitting Errors ---------------------------------


class FittingError(KindredError):
    """Base class for parameter fitting errors (E400-E499)."""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("code", "E400")
        super().__init__(message, **kwargs)


class FitSimulationError(FittingError):
    """Raised when a simulation fails while evaluating a fitting objective."""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("code", "E404")
        details = dict(kwargs.pop("details", {}) or {})
        super().__init__(message, details=details, **kwargs)


class FittingCancelled(FittingError):
    """Raised when a fitting run is cancelled by user request."""

    def __init__(self, message: str = "Fit canceled by user", **kwargs):
        kwargs.setdefault("code", "E405")
        super().__init__(message, **kwargs)


# ----------------------------- Helper Functions --------------------------------


def create_solver_error(
    solver_name: str,
    t_current: float,
    reason: str,
    solver_message: str,
) -> SolverError:
    """An ODE solver failure stated by Kindred; the solver's own text stays a detail."""
    details: Dict[str, Any] = {
        "solver": solver_name,
        "time": t_current,
        "solver_message": solver_message,
    }
    return SolverError(
        f"The {solver_name} solver stopped at t = {t_current:g} s because {reason}.",
        details=details,
    )
