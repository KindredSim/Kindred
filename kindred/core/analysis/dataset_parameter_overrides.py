"""Typed optimizer coordinates that are local to Fit dataset evaluation."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "FitCoupledVariableParamSpec",
    "FitDatasetParameterOverrides",
    "FitDatasetVariableParamSpec",
    "FitSharedVariableParamSpec",
]


# Names, values and bounds arrive admitted: parameter values, bounds and Log10
# by FitParameterVariable, initial concentrations and their bounds by
# FitInitialVariable and FitSharedInitialGroup.
@dataclass(frozen=True, slots=True)
class FitDatasetVariableParamSpec:
    """One optimizer coordinate owned by exactly one stable dataset."""

    name: str
    initial: float
    minimum: float
    maximum: float
    log10: bool = False


@dataclass(frozen=True, slots=True)
class FitSharedVariableParamSpec:
    """One optimizer coordinate shared by every dataset evaluator."""

    name: str
    initial: float
    minimum: float
    maximum: float
    log10: bool = False


# The Fit dispatch compiler assigns each parameter of a dataset to exactly one
# of fixed, variable or coupled, from uniquely named parameters, and couples
# only the unique included members of a group or initial-condition set.
@dataclass(frozen=True, slots=True)
class FitDatasetParameterOverrides:
    """Fixed and variable evaluator inputs for one stable dataset."""

    dataset_id: str
    fixed_params: tuple[tuple[str, float], ...] = ()
    variable_params: tuple[FitDatasetVariableParamSpec, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "variable_params",
            tuple(sorted(self.variable_params, key=lambda item: item.name)),
        )


@dataclass(frozen=True, slots=True)
class FitCoupledVariableParamSpec:
    """One optimizer coordinate written into multiple dataset evaluators."""

    variable_id: str
    parameter_name: str
    dataset_ids: tuple[str, ...]
    initial: float
    minimum: float
    maximum: float
    log10: bool = False
