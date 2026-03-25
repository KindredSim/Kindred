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
- **Configuration**: Settings and profile errors (E500-E599)
- **Integration**: System integration errors (E600-E699)
- **IO**: File and data handling errors (E700-E799)
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
    "ParameterValidationError",
    "SpeciesValidationError",
    "ReactionValidationError",
    "InitialConditionError",
    # Simulation
    "SimulationError",
    "SolverError",
    "IntegrationError",
    "TimeGridError",
    "JacobianError",
    # Fitting
    "FittingError",
    "FitSimulationError",
    "FittingCancelled",
    "OptimizationError",
    "ConvergenceError",
    "DataMismatchError",
    # Configuration
    "ConfigurationError",
    "ProfileError",
    "SettingsError",
    # Integration
    "IntegrationFailureError",
    "SimulationBuilderContractError",
    "DependencyError",
    # IO
    "IOError",
    "FileFormatError",
    "ExportError",
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


class ParameterValidationError(ValidationError):
    """Error for invalid parameter values or specifications."""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("code", "E201")
        super().__init__(message, **kwargs)


class SpeciesValidationError(ValidationError):
    """Error for invalid species definitions or references."""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("code", "E202")
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


class IntegrationError(SimulationError):
    """Error during numerical integration."""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("code", "E302")
        super().__init__(message, **kwargs)


class TimeGridError(SimulationError):
    """Error in time grid specification or generation."""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("code", "E303")
        super().__init__(message, **kwargs)


class JacobianError(SimulationError):
    """Error in Jacobian computation or structure."""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("code", "E304")
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
        if failed_params:
            details.setdefault("parameters", failed_params)
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


class ConvergenceError(FittingError):
    """Error when optimization fails to converge."""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("code", "E402")
        super().__init__(message, **kwargs)


class DataMismatchError(FittingError):
    """Error when experimental data doesn't match model outputs."""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("code", "E403")
        super().__init__(message, **kwargs)


# -------------------------- Configuration Errors -------------------------------


class ConfigurationError(KindredError):
    """Base class for configuration errors (E500-E599)."""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("code", "E500")
        super().__init__(message, **kwargs)


class ProfileError(ConfigurationError):
    """Error loading or applying configuration profiles."""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("code", "E501")
        super().__init__(message, **kwargs)


class SettingsError(ConfigurationError):
    """Error in user settings or preferences."""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("code", "E502")
        super().__init__(message, **kwargs)


# -------------------------- Integration Errors ---------------------------------


class IntegrationFailureError(KindredError):
    """Base class for system integration errors (E600-E699)."""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("code", "E600")
        super().__init__(message, **kwargs)


class SimulationBuilderContractError(IntegrationFailureError):
    """Injected simulation builder does not honor the core solver contract."""

    def __init__(self, message: str | None = None, **kwargs):
        kwargs.setdefault("code", "E602")
        kwargs.setdefault(
            "suggestion",
            "Update the simulation builder to accept solver, rtol, and atol keyword arguments.",
        )
        details = dict(kwargs.pop("details", {}) or {})
        details.setdefault("contract", "(mechanism_text, param_names, *, solver, rtol, atol)")
        details.setdefault("missing_kwargs", ["solver", "rtol", "atol"])
        super().__init__(
            message
            or "Simulation builder must accept solver settings: "
            "(mechanism_text, param_names, *, solver, rtol, atol).",
            details=details,
            **kwargs,
        )


class DependencyError(IntegrationFailureError):
    """Error with missing or incompatible dependencies."""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("code", "E601")
        super().__init__(message, **kwargs)


# ------------------------------- IO Errors -------------------------------------


class IOError(KindredError):
    """Base class for I/O errors (E700-E799)."""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("code", "E700")
        super().__init__(message, **kwargs)


class FileFormatError(IOError):
    """Error parsing or writing file formats."""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("code", "E701")
        super().__init__(message, **kwargs)


class ExportError(IOError):
    """Error during data or results export."""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("code", "E702")
        super().__init__(message, **kwargs)


# ----------------------------- Helper Functions --------------------------------


def create_validation_error(
    param_name: str,
    value: Any,
    expected: str,
    *,
    examples: Optional[List[str]] = None,
) -> ParameterValidationError:
    """
    Factory for common parameter validation errors.

    Parameters
    ----------
    param_name : str
        Parameter name
    value : any
        Invalid value provided
    expected : str
        Description of expected value
    examples : list of str, optional
        Valid examples

    Returns
    -------
    ParameterValidationError
    """
    return ParameterValidationError(
        f"Invalid value for parameter '{param_name}': {value}",
        suggestion=f"Expected: {expected}",
        examples=examples or [],
        details={"parameter": param_name, "provided_value": str(value)},
    )


def create_missing_dependency_error(
    module_name: str,
    feature: str,
    install_command: str,
) -> DependencyError:
    """
    Factory for missing dependency errors.

    Parameters
    ----------
    module_name : str
        Name of missing module
    feature : str
        Feature that requires the module
    install_command : str
        Pip install command

    Returns
    -------
    DependencyError
    """
    return DependencyError(
        f"Missing optional dependency: {module_name}",
        suggestion=f"Install with: {install_command}",
        details={
            "module": module_name,
            "feature": feature,
            "install_command": install_command,
        },
    )


def create_convergence_error(
    optimizer: str,
    iterations: int,
    tolerance: float,
    final_cost: float,
) -> ConvergenceError:
    """
    Factory for optimization convergence errors.

    Parameters
    ----------
    optimizer : str
        Optimizer name
    iterations : int
        Number of iterations performed
    tolerance : float
        Target tolerance
    final_cost : float
        Final cost function value

    Returns
    -------
    ConvergenceError
    """
    return ConvergenceError(
        f"Optimization failed to converge after {iterations} iterations",
        suggestion="Try adjusting initial parameters, bounds, or increasing max iterations",
        details={
            "optimizer": optimizer,
            "iterations": iterations,
            "tolerance": tolerance,
            "final_cost": final_cost,
        },
    )


def create_solver_error(
    solver_name: str,
    t_current: float,
    message: str,
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

    Returns
    -------
    SolverError
    """
    return SolverError(
        f"Solver '{solver_name}' failed at t={t_current:.4e}",
        suggestion="Try reducing tolerances (rtol, atol) or using a different solver method",
        examples=[
            "solver='LSODA' (automatic stiffness detection)",
            "solver='Radau' (implicit, good for stiff systems)",
            "solver='BDF' (implicit, multistep)",
        ],
        details={
            "solver": solver_name,
            "time": t_current,
            "solver_message": message,
        },
    )
