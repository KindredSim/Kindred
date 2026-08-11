"""
Structured exception types for Kindred user-facing and integration boundaries.

This module defines the `KindredError` hierarchy plus optional context, error
codes, suggestions, and examples for call sites that choose to surface richer
structured failures. Lower-level core helpers still use builtin exceptions for
many validation and invariant checks, so this module should be treated as the
supported structured error path rather than a repo-wide mandatory policy.

Error Categories
----------------
- **Validation**: Input validation failures (E200-E299)
- **Simulation**: Runtime simulation errors (E300-E399)
- **Fitting**: Parameter estimation errors (E400-E499)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import textwrap

__all__ = [
    # Base
    "KindredError",
    "ErrorContext",
    # Validation
    "ValidationError",
    "ReactionValidationError",
    "InitialConditionError",
    # Simulation
    "SimulationError",
    "SolverError",
    "TimeGridError",
    # Fitting
    "FittingError",
    "FitSimulationError",
    "FittingCancelled",
    "OptimizationError",
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
    col : int, optional
        Column number (1-based) where error occurred
    line_text : str, optional
        Source line text for context
    file_path : str, optional
        File path where error occurred
    stack_trace : str, optional
        Relevant stack trace information
    """
    line: Optional[int] = None
    col: Optional[int] = None
    line_text: Optional[str] = None
    file_path: Optional[str] = None
    stack_trace: Optional[str] = None


class KindredError(Exception):
    """
    Base exception for all Kindred errors.

    Provides enhanced error messages with:
    - Error code for programmatic handling
    - Clear problem description
    - Helpful suggestions
    - Valid examples
    - Source context when available

    Attributes
    ----------
    message : str
        Main error message
    code : str
        Error code (e.g., "E201")
    suggestion : str, optional
        Suggested fix or solution
    examples : list of str, optional
        Valid syntax examples
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
        suggestion: Optional[str] = None,
        examples: Optional[List[str]] = None,
        context: Optional[ErrorContext] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.code = code
        self.suggestion = suggestion
        self.examples = examples or []
        self.context = context
        self.details = details or {}

        # Build comprehensive error message
        full_message = self._build_message()
        super().__init__(full_message)

    def _build_message(self) -> str:
        """Build user-friendly error message."""
        parts = [f"{self.code} {self.__class__.__name__}: {self.message}"]

        # Add context (file, line, column)
        if self.context:
            ctx_parts = []
            if self.context.file_path:
                ctx_parts.append(f"File: {self.context.file_path}")
            if self.context.line is not None:
                loc = f"L{self.context.line}"
                if self.context.col is not None:
                    loc += f":C{self.context.col}"
                ctx_parts.append(f"Location: {loc}")

            if ctx_parts:
                parts.append("")
                parts.extend(ctx_parts)

            # Add source line with caret
            if self.context.line_text:
                parts.append("")
                parts.append(f"  {self.context.line_text}")
                if self.context.col:
                    caret_pos = self.context.col - 1
                    parts.append("  " + " " * caret_pos + "^")

        # Add suggestion
        if self.suggestion:
            parts.append("")
            parts.append(f"Suggestion: {self.suggestion}")

        # Add examples
        if self.examples:
            parts.append("")
            parts.append("Valid examples:")
            for example in self.examples:
                # Wrap long examples
                wrapped = textwrap.fill(
                    example,
                    width=70,
                    initial_indent="  • ",
                    subsequent_indent="    "
                )
                parts.append(wrapped)

        # Add details
        if self.details:
            parts.append("")
            parts.append("Details:")
            for key, value in self.details.items():
                parts.append(f"  {key}: {value}")

        return "\n".join(parts)


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


class InitialConditionError(ValidationError):
    """Error for invalid or missing initial conditions."""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("code", "E204")
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


class TimeGridError(SimulationError):
    """Error in time grid specification or generation."""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("code", "E303")
        super().__init__(message, **kwargs)


class SimulationCancelled(SimulationError):
    """Raised when a simulation is cancelled by user request."""
    def __init__(self, message: str = "Simulation cancelled by user", **kwargs):
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
    def __init__(self, message: str, *, failed_params: Optional[Dict[str, float]] = None, **kwargs):
        kwargs.setdefault("code", "E404")
        details = dict(kwargs.pop("details", {}) or {})
        super().__init__(message, details=details, **kwargs)
        self.failed_params = failed_params


class FittingCancelled(FittingError):
    """Raised when a fitting run is cancelled by user request."""

    def __init__(self, message: str = "Fit cancelled by user", **kwargs):
        kwargs.setdefault("code", "E405")
        super().__init__(message, **kwargs)


class OptimizationError(FittingError):
    """Error during optimization process."""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("code", "E401")
        super().__init__(message, **kwargs)


# ----------------------------- Helper Functions --------------------------------


def create_solver_error(
    solver_name: str,
    t_current: float,
    message: str,
    *,
    candidate_domain_eligible: bool = False,
) -> SolverError:
    """
    Factory for ODE solver errors.

    Parameters
    ----------
    solver_name : str
        Solver method name
    t_current : float
        Time at which error occurred
    message : str
        Error message from solver
    candidate_domain_eligible : bool, optional
        Whether this failure is a proven numerical candidate-domain failure.

    Returns
    -------
    SolverError
    """
    details: Dict[str, Any] = {
        "solver": solver_name,
        "time": t_current,
        "solver_message": message,
    }
    if candidate_domain_eligible is True:
        details["candidate_domain_eligible"] = True
    return SolverError(
        f"Solver '{solver_name}' failed at t={t_current:.4e}",
        suggestion="Try reducing tolerances (rtol, atol) or using a different solver method",
        examples=[
            "solver='Radau' (implicit, good for stiff systems)",
            "solver='BDF' (implicit, multistep)",
        ],
        details=details,
    )
