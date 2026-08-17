"""Typed optimizer coordinates that are local to Fit dataset evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

__all__ = [
    "FIT_SCOPE_CENTRED_FIXED_EFFECTS",
    "FitCoupledVariableParamSpec",
    "FitCentredFixedEffectsParamSpec",
    "FitDatasetParameterOverrides",
    "FitDatasetVariableParamSpec",
    "FitSharedVariableParamSpec",
]


FIT_SCOPE_CENTRED_FIXED_EFFECTS = "centred_fixed_effects"


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
class FitCentredFixedEffectsParamSpec:
    """One mechanism parameter with an overall and centred dataset effect.

    ``initial_values`` is the admitted physical start for every included
    dataset.  The layout encodes these values on one raw scale (physical for
    linear parameters, log10 for log parameters), then centres the deviations
    on that same scale.  Dataset IDs are canonicalized by the layout, not by
    presentation order.
    """

    parameter_name: str
    dataset_ids: tuple[str, ...]
    initial_values: tuple[tuple[str, float], ...]
    minimum: float
    maximum: float
    log10: bool = False

    def __post_init__(self) -> None:
        parameter_name = _identifier(self.parameter_name, "centred parameter name")
        dataset_ids = tuple(
            _identifier(value, "centred dataset ID") for value in self.dataset_ids
        )
        if (
            not dataset_ids
            or len(dataset_ids) != len(set(dataset_ids))
        ):
            raise ValueError(
                f"Centred parameter {parameter_name!r} requires unique dataset IDs."
            )
        starts: dict[str, float] = {}
        raw_starts = (
            tuple(self.initial_values.items())
            if isinstance(self.initial_values, Mapping)
            else tuple(self.initial_values)
        )
        for raw_dataset_id, raw_value in raw_starts:
            dataset_id = _identifier(raw_dataset_id, "centred start dataset ID")
            if dataset_id in starts:
                raise ValueError(
                    f"Centred parameter {parameter_name!r} repeats dataset {dataset_id!r}."
                )
            starts[dataset_id] = _finite(
                raw_value,
                f"{parameter_name}:{dataset_id} initial",
            )
        if set(starts) != set(dataset_ids):
            raise ValueError(
                f"Centred parameter {parameter_name!r} starts must exactly match its dataset roster."
            )
        minimum = _finite(self.minimum, f"{parameter_name} minimum")
        maximum = _finite(self.maximum, f"{parameter_name} maximum")
        if minimum >= maximum:
            raise ValueError(
                f"Centred parameter {parameter_name!r} requires minimum < maximum."
            )
        if bool(self.log10) and minimum <= 0.0:
            raise ValueError(
                f"Centred parameter {parameter_name!r} log10 bounds must be positive."
            )
        if any(
            value < minimum or value > maximum
            for value in starts.values()
        ):
            raise ValueError(
                f"Centred parameter {parameter_name!r} starts must lie inside its bounds."
            )
        object.__setattr__(self, "parameter_name", parameter_name)
        dataset_ids = tuple(sorted(dataset_ids))
        object.__setattr__(self, "dataset_ids", dataset_ids)
        object.__setattr__(
            self,
            "initial_values",
            tuple((dataset_id, starts[dataset_id]) for dataset_id in dataset_ids),
        )
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
