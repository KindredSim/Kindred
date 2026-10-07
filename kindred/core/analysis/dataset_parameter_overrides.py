"""Typed optimizer coordinates that are local to Fit dataset evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "FitCoupledVariableParamSpec",
    "FitDatasetParameterOverrides",
    "FitDatasetVariableParamSpec",
    "FitSharedVariableParamSpec",
]


def _identifier(value: object, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} must be nonblank.")
    return text


def _finite(value: object, field: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite.")
    return number


def _bounds(
    *,
    initial: object,
    minimum: object,
    maximum: object,
    log10: bool,
    field: str,
) -> tuple[float, float, float]:
    value = _finite(initial, f"{field} initial")
    low = float(minimum)
    high = float(maximum)
    if math.isnan(low) or math.isnan(high) or low >= high:
        raise ValueError(f"{field} requires minimum < maximum.")
    if not low <= value <= high:
        raise ValueError(f"{field} initial value must lie inside its bounds.")
    if log10 and (
        value <= 0.0
        or low <= 0.0
        or not math.isfinite(low)
        or not math.isfinite(high)
    ):
        raise ValueError(
            f"{field} log10 fitting requires positive finite bounds and value."
        )
    return value, low, high


@dataclass(frozen=True, slots=True)
class FitDatasetVariableParamSpec:
    """One optimizer coordinate owned by exactly one stable dataset."""

    name: str
    initial: float
    minimum: float
    maximum: float
    log10: bool = False

    def __post_init__(self) -> None:
        name = _identifier(self.name, "dataset parameter name")
        initial, minimum, maximum = _bounds(
            initial=self.initial,
            minimum=self.minimum,
            maximum=self.maximum,
            log10=bool(self.log10),
            field=name,
        )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "initial", initial)
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)
        object.__setattr__(self, "log10", bool(self.log10))


@dataclass(frozen=True, slots=True)
class FitSharedVariableParamSpec:
    """One optimizer coordinate shared by every dataset evaluator."""

    name: str
    initial: float
    minimum: float
    maximum: float
    log10: bool = False

    def __post_init__(self) -> None:
        name = _identifier(self.name, "shared parameter name")
        initial, minimum, maximum = _bounds(
            initial=self.initial,
            minimum=self.minimum,
            maximum=self.maximum,
            log10=bool(self.log10),
            field=name,
        )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "initial", initial)
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)
        object.__setattr__(self, "log10", bool(self.log10))


@dataclass(frozen=True, slots=True)
class FitDatasetParameterOverrides:
    """Fixed and variable evaluator inputs for one stable dataset."""

    dataset_id: str
    fixed_params: tuple[tuple[str, float], ...] = ()
    variable_params: tuple[FitDatasetVariableParamSpec, ...] = ()

    def __post_init__(self) -> None:
        dataset_id = _identifier(self.dataset_id, "dataset ID")
        fixed: list[tuple[str, float]] = []
        fixed_names: set[str] = set()
        for raw_name, raw_value in self.fixed_params:
            name = _identifier(raw_name, "fixed dataset parameter")
            if name in fixed_names:
                raise ValueError(
                    f"Dataset {dataset_id!r} repeats fixed parameter {name!r}."
                )
            fixed_names.add(name)
            fixed.append((name, _finite(raw_value, f"{dataset_id}:{name}")))
        variables = tuple(sorted(self.variable_params, key=lambda item: item.name))
        variable_names = [item.name for item in variables]
        if len(variable_names) != len(set(variable_names)):
            raise ValueError(
                f"Dataset {dataset_id!r} repeats a variable parameter."
            )
        overlap = fixed_names.intersection(variable_names)
        if overlap:
            raise ValueError(
                f"Dataset {dataset_id!r} configures parameters as both fixed and "
                f"variable: {', '.join(sorted(overlap))}."
            )
        object.__setattr__(self, "dataset_id", dataset_id)
        object.__setattr__(self, "fixed_params", tuple(fixed))
        object.__setattr__(self, "variable_params", variables)


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

    def __post_init__(self) -> None:
        variable_id = _identifier(self.variable_id, "coupled variable ID")
        parameter_name = _identifier(
            self.parameter_name,
            "coupled parameter name",
        )
        dataset_ids = tuple(
            _identifier(item, "coupled dataset ID") for item in self.dataset_ids
        )
        if not dataset_ids or len(dataset_ids) != len(set(dataset_ids)):
            raise ValueError(
                f"Coupled variable {variable_id!r} requires unique dataset IDs."
            )
        initial, minimum, maximum = _bounds(
            initial=self.initial,
            minimum=self.minimum,
            maximum=self.maximum,
            log10=bool(self.log10),
            field=variable_id,
        )
        object.__setattr__(self, "variable_id", variable_id)
        object.__setattr__(self, "parameter_name", parameter_name)
        object.__setattr__(self, "dataset_ids", dataset_ids)
        object.__setattr__(self, "initial", initial)
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)
        object.__setattr__(self, "log10", bool(self.log10))
