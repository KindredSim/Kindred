from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence


@dataclass(frozen=True)
class FitDatasetVariableParamSpec:
    initial: float
    minimum: float
    maximum: float
    log10: bool = False

    def to_legacy_dict(self) -> dict[str, float | bool]:
        return {
            "initial": float(self.initial),
            "min": float(self.minimum),
            "max": float(self.maximum),
            "log10": bool(self.log10),
        }


@dataclass(frozen=True)
class FitDatasetParameterOverrides:
    dataset_id: str
    fixed_params: dict[str, float] = field(default_factory=dict)
    variable_params: dict[str, FitDatasetVariableParamSpec] = field(default_factory=dict)


def _normalize_dataset_id(value: object) -> str:
    ds_id = str(value or "").strip()
    if not ds_id:
        raise ValueError("Dataset override entries require a non-empty dataset_id.")
    return ds_id


def _coerce_fixed_params(mapping: object) -> dict[str, float]:
    if not isinstance(mapping, Mapping):
        return {}
    fixed: dict[str, float] = {}
    for key, value in mapping.items():
        name = str(key or "").strip()
        if not name:
            continue
        fixed[name] = float(value)
    return fixed


def _coerce_variable_params(mapping: object) -> dict[str, FitDatasetVariableParamSpec]:
    if not isinstance(mapping, Mapping):
        return {}
    specs: dict[str, FitDatasetVariableParamSpec] = {}
    for key, value in mapping.items():
        name = str(key or "").strip()
        if not name:
            continue
        if isinstance(value, FitDatasetVariableParamSpec):
            specs[name] = FitDatasetVariableParamSpec(
                initial=float(value.initial),
                minimum=float(value.minimum),
                maximum=float(value.maximum),
                log10=bool(value.log10),
            )
            continue
        if not isinstance(value, Mapping):
            continue
        minimum = value["minimum"] if "minimum" in value else value.get("min", float("-inf"))
        maximum = value["maximum"] if "maximum" in value else value.get("max", float("inf"))
        specs[name] = FitDatasetVariableParamSpec(
            initial=float(value.get("initial", 0.0)),
            minimum=float(minimum),
            maximum=float(maximum),
            log10=bool(value.get("log10", False)),
        )
    return specs


def _coerce_override_entry(entry: object) -> FitDatasetParameterOverrides:
    if isinstance(entry, FitDatasetParameterOverrides):
        return FitDatasetParameterOverrides(
            dataset_id=_normalize_dataset_id(entry.dataset_id),
            fixed_params=_coerce_fixed_params(entry.fixed_params),
            variable_params=_coerce_variable_params(entry.variable_params),
        )
    if not isinstance(entry, Mapping):
        raise TypeError(f"Dataset override entry must be a mapping or FitDatasetParameterOverrides; got {type(entry)!r}.")
    return FitDatasetParameterOverrides(
        dataset_id=_normalize_dataset_id(entry.get("dataset_id", entry.get("id"))),
        fixed_params=_coerce_fixed_params(entry.get("fixed_params", entry.get("dataset_params"))),
        variable_params=_coerce_variable_params(entry.get("variable_params", entry.get("dataset_variable_params"))),
    )


def coerce_fit_dataset_parameter_overrides(
    *,
    dataset_ids: Optional[Sequence[str]] = None,
    dataset_overrides: Optional[Sequence[object]] = None,
    dataset_params: Optional[Mapping[str, Mapping[str, Any]]] = None,
    dataset_variable_params: Optional[Mapping[str, Mapping[str, Mapping[str, Any]]]] = None,
) -> list[FitDatasetParameterOverrides]:
    if dataset_overrides is not None:
        merged: dict[str, FitDatasetParameterOverrides] = {}
        for entry in dataset_overrides:
            override = _coerce_override_entry(entry)
            existing = merged.get(override.dataset_id)
            if existing is None:
                merged[override.dataset_id] = override
                continue
            fixed_params = dict(existing.fixed_params)
            fixed_params.update(override.fixed_params)
            variable_params = dict(existing.variable_params)
            variable_params.update(override.variable_params)
            merged[override.dataset_id] = FitDatasetParameterOverrides(
                dataset_id=override.dataset_id,
                fixed_params=fixed_params,
                variable_params=variable_params,
            )
        if dataset_ids is not None:
            ordered: list[FitDatasetParameterOverrides] = []
            for raw_ds_id in dataset_ids:
                ds_id = str(raw_ds_id or "").strip()
                if not ds_id or ds_id not in merged:
                    continue
                ordered.append(merged[ds_id])
            return ordered
        return [merged[ds_id] for ds_id in sorted(merged)]

    ordered_ids = [str(ds_id or "").strip() for ds_id in (dataset_ids or []) if str(ds_id or "").strip()]
    if not ordered_ids:
        ordered_ids = sorted(
            {
                str(ds_id or "").strip()
                for ds_id in list((dataset_params or {}).keys()) + list((dataset_variable_params or {}).keys())
                if str(ds_id or "").strip()
            }
        )

    overrides: list[FitDatasetParameterOverrides] = []
    for ds_id in ordered_ids:
        fixed_params = _coerce_fixed_params((dataset_params or {}).get(ds_id))
        variable_params = _coerce_variable_params((dataset_variable_params or {}).get(ds_id))
        if not fixed_params and not variable_params:
            continue
        overrides.append(
            FitDatasetParameterOverrides(
                dataset_id=ds_id,
                fixed_params=fixed_params,
                variable_params=variable_params,
            )
        )
    return overrides


def split_fit_dataset_parameter_overrides(
    overrides: Sequence[FitDatasetParameterOverrides],
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, dict[str, float | bool]]]]:
    dataset_params: dict[str, dict[str, float]] = {}
    dataset_variable_params: dict[str, dict[str, dict[str, float | bool]]] = {}
    for override in overrides:
        ds_id = _normalize_dataset_id(override.dataset_id)
        fixed_params = _coerce_fixed_params(override.fixed_params)
        variable_params = _coerce_variable_params(override.variable_params)
        if fixed_params:
            dataset_params[ds_id] = fixed_params
        if variable_params:
            dataset_variable_params[ds_id] = {
                name: spec.to_legacy_dict()
                for name, spec in sorted(variable_params.items())
                if str(name).strip()
            }
    return dataset_params, dataset_variable_params
