"""Typed render projection contract for global fitting results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "FitRenderDatasetProjection",
    "FitRenderProjection",
    "build_fit_render_projection",
    "projection_from_global_fit_result",
]


def _coerce_array(values: object, *, field_name: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric and one-dimensional.") from exc
    if array.ndim != 1:
        raise ValueError(f"{field_name} must be one-dimensional.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{field_name} must contain only finite values.")
    return array.copy()


def _coerce_stats(values: Optional[Mapping[str, Any]]) -> Dict[str, float]:
    stats: Dict[str, float] = {}
    for key, value in dict(values or {}).items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            stats[str(key)] = number
    return stats


def _shared_axis_or_empty(axis_map: Optional[Mapping[str, object]]) -> np.ndarray:
    normalized = [
        _coerce_array(values, field_name="shared_axis")
        for name, values in dict(axis_map or {}).items()
        if str(name or "").strip()
    ]
    if not normalized:
        return np.asarray([], dtype=float)
    reference = normalized[0]
    for candidate in normalized[1:]:
        if candidate.size != reference.size or not np.allclose(candidate, reference):
            return np.asarray([], dtype=float)
    return reference.copy()


@dataclass(frozen=True)
class FitRenderDatasetProjection:
    """Validated render data for one committed dataset."""

    dataset_id: str
    observed_x: np.ndarray
    observed_x_label: str
    observed_series: Mapping[str, np.ndarray]
    model_x: np.ndarray
    model_series: Mapping[str, np.ndarray]
    observed_x_by_species: Mapping[str, np.ndarray] = field(default_factory=dict)
    model_x_by_species: Mapping[str, np.ndarray] = field(default_factory=dict)
    dataset_stats: Mapping[str, float] = field(default_factory=dict)
    status: str = "ok"
    diagnostics: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        dataset_id = str(self.dataset_id or "").strip()
        if not dataset_id:
            raise ValueError("FitRenderDatasetProjection.dataset_id is required.")

        status = str(self.status or "").strip() or "ok"
        observed_x = _coerce_array(self.observed_x, field_name=f"{dataset_id}.observed_x")
        observed_x_label = str(self.observed_x_label or "").strip()
        if not observed_x_label:
            observed_x_label = "Time"
        model_x = _coerce_array(self.model_x, field_name=f"{dataset_id}.model_x")
        observed_axis_source = dict(self.observed_x_by_species or {})
        observed_x_by_species: Dict[str, np.ndarray] = {}
        observed_series: Dict[str, np.ndarray] = {}
        for raw_name, raw_values in dict(self.observed_series or {}).items():
            name = str(raw_name or "").strip()
            if not name:
                continue
            values = _coerce_array(raw_values, field_name=f"{dataset_id}.{name}.observed")
            if status == "ok" and name not in observed_axis_source:
                raise ValueError(f"{dataset_id}.{name} requires observed_x_by_species.")
            axis_raw = observed_axis_source.get(name, observed_x)
            axis = _coerce_array(axis_raw, field_name=f"{dataset_id}.{name}.observed_x")
            if len(values) != len(axis):
                raise ValueError(f"{dataset_id}.{name} observed length must match its observed_x length.")
            observed_series[name] = values
            observed_x_by_species[name] = axis
        model_axis_source = dict(self.model_x_by_species or {})
        model_x_by_species: Dict[str, np.ndarray] = {}
        model_series: Dict[str, np.ndarray] = {}
        for raw_name, raw_values in dict(self.model_series or {}).items():
            name = str(raw_name or "").strip()
            if not name:
                continue
            values = _coerce_array(raw_values, field_name=f"{dataset_id}.{name}")
            if status == "ok" and name not in model_axis_source:
                raise ValueError(f"{dataset_id}.{name} requires model_x_by_species.")
            axis_raw = model_axis_source.get(name, model_x)
            axis = _coerce_array(axis_raw, field_name=f"{dataset_id}.{name}.model_x")
            if len(values) != len(axis):
                raise ValueError(f"{dataset_id}.{name} length must match its model_x length.")
            model_series[name] = values
            model_x_by_species[name] = axis
        if status == "ok" and not model_series:
            raise ValueError("FitRenderDatasetProjection.model_series is required.")
        if status == "ok" and not observed_series:
            raise ValueError("FitRenderDatasetProjection.observed_series is required.")
        if status == "ok" and not observed_x_by_species:
            raise ValueError("FitRenderDatasetProjection.observed_x_by_species is required.")
        if status == "ok" and not model_x_by_species:
            raise ValueError("FitRenderDatasetProjection.model_x_by_species is required.")
        if status == "ok" and set(observed_series) != set(model_series):
            raise ValueError(
                f"{dataset_id} observed/model species sets must match for a valid render projection."
            )

        diagnostics = tuple(str(item) for item in (self.diagnostics or ()))

        object.__setattr__(self, "dataset_id", dataset_id)
        object.__setattr__(self, "observed_x", observed_x)
        object.__setattr__(self, "observed_x_label", observed_x_label)
        object.__setattr__(self, "observed_series", observed_series)
        object.__setattr__(self, "observed_x_by_species", observed_x_by_species)
        object.__setattr__(self, "model_x", model_x)
        object.__setattr__(self, "model_series", model_series)
        object.__setattr__(self, "model_x_by_species", model_x_by_species)
        object.__setattr__(self, "dataset_stats", _coerce_stats(self.dataset_stats))
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "diagnostics", diagnostics)

    def observed_x_for_species(self, species_name: str) -> np.ndarray:
        name = str(species_name or "").strip()
        if name and name in self.observed_x_by_species:
            return np.asarray(self.observed_x_by_species[name], dtype=float).reshape(-1).copy()
        return np.asarray(self.observed_x, dtype=float).reshape(-1).copy()

    def model_x_for_species(self, species_name: str) -> np.ndarray:
        name = str(species_name or "").strip()
        if name and name in self.model_x_by_species:
            return np.asarray(self.model_x_by_species[name], dtype=float).reshape(-1).copy()
        return np.asarray(self.model_x, dtype=float).reshape(-1).copy()


@dataclass(frozen=True)
class FitRenderProjection:
    """Run-stamped live/final render projection keyed by committed dataset id."""

    phase: str
    run_stamp_hash: str
    sequence: int
    cost: Optional[float]
    datasets: Mapping[str, FitRenderDatasetProjection]

    def __post_init__(self) -> None:
        phase = str(self.phase or "").strip()
        if phase not in {"live", "final"}:
            raise ValueError("FitRenderProjection.phase must be 'live' or 'final'.")
        run_stamp_hash = str(self.run_stamp_hash or "").strip()
        if not run_stamp_hash:
            raise ValueError("FitRenderProjection.run_stamp_hash is required.")

        datasets: Dict[str, FitRenderDatasetProjection] = {}
        for raw_id, projection in dict(self.datasets or {}).items():
            if not isinstance(projection, FitRenderDatasetProjection):
                raise TypeError("FitRenderProjection.datasets values must be FitRenderDatasetProjection.")
            dataset_id = str(projection.dataset_id or raw_id or "").strip()
            if dataset_id:
                datasets[dataset_id] = projection

        cost_value: Optional[float]
        if self.cost is None:
            cost_value = None
        else:
            cost_value = float(self.cost)
            if not np.isfinite(cost_value):
                cost_value = None

        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "run_stamp_hash", run_stamp_hash)
        object.__setattr__(self, "sequence", int(self.sequence))
        object.__setattr__(self, "cost", cost_value)
        object.__setattr__(self, "datasets", datasets)

    @property
    def dataset_ids(self) -> Tuple[str, ...]:
        return tuple(self.datasets.keys())

    def model_series_by_dataset(self) -> Dict[str, Dict[str, np.ndarray]]:
        return {
            dataset_id: {
                species: values.copy()
                for species, values in projection.model_series.items()
            }
            for dataset_id, projection in self.datasets.items()
            if projection.status == "ok"
        }

    def model_x_by_dataset(self) -> Dict[str, np.ndarray]:
        return {
            dataset_id: projection.model_x.copy()
            for dataset_id, projection in self.datasets.items()
            if projection.status == "ok"
        }

    def dataset_stats_by_dataset(self) -> Dict[str, Dict[str, float]]:
        return {
            dataset_id: dict(projection.dataset_stats)
            for dataset_id, projection in self.datasets.items()
            if projection.dataset_stats
        }


def build_fit_render_projection(
    *,
    phase: str,
    run_stamp_hash: str,
    sequence: int,
    cost: Optional[float],
    observed_x_by_dataset: Mapping[str, object],
    observed_x_label_by_dataset: Optional[Mapping[str, object]] = None,
    observed_series_by_dataset: Optional[Mapping[str, Mapping[str, object]]] = None,
    observed_x_by_species_by_dataset: Optional[Mapping[str, Mapping[str, object]]] = None,
    model_x_by_dataset: Mapping[str, object],
    model_series_by_dataset: Mapping[str, Mapping[str, object]],
    model_x_by_species_by_dataset: Optional[Mapping[str, Mapping[str, object]]] = None,
    dataset_stats_by_dataset: Optional[Mapping[str, Mapping[str, Any]]] = None,
    dataset_ids: Optional[Sequence[str]] = None,
) -> FitRenderProjection:
    dataset_order = [
        str(dataset_id)
        for dataset_id in (dataset_ids if dataset_ids is not None else model_series_by_dataset.keys())
        if str(dataset_id or "").strip()
    ]
    datasets: Dict[str, FitRenderDatasetProjection] = {}
    for dataset_id in dataset_order:
        observed_x = observed_x_by_dataset.get(dataset_id)
        observed_x_label = (observed_x_label_by_dataset or {}).get(dataset_id, "Time")
        observed_series = (observed_series_by_dataset or {}).get(dataset_id)
        observed_x_by_species = (observed_x_by_species_by_dataset or {}).get(dataset_id)
        model_x = model_x_by_dataset.get(dataset_id)
        model_series = model_series_by_dataset.get(dataset_id)
        model_x_by_species = (model_x_by_species_by_dataset or {}).get(dataset_id)
        if (
            not isinstance(observed_x_by_species, Mapping)
            or not isinstance(observed_series, Mapping)
            or not isinstance(model_x_by_species, Mapping)
            or not isinstance(model_series, Mapping)
        ):
            datasets[dataset_id] = FitRenderDatasetProjection(
                dataset_id=dataset_id,
                observed_x=np.asarray([], dtype=float),
                observed_x_label=str(observed_x_label or "Time"),
                observed_series={},
                observed_x_by_species={},
                model_x=np.asarray([], dtype=float),
                model_series={},
                model_x_by_species={},
                dataset_stats=(dataset_stats_by_dataset or {}).get(dataset_id, {}),
                status="missing_projection",
                diagnostics=("Render projection arrays were not available.",),
            )
            continue
        try:
            datasets[dataset_id] = FitRenderDatasetProjection(
                dataset_id=dataset_id,
                observed_x=np.asarray([], dtype=float) if observed_x is None else observed_x,
                observed_x_label=str(observed_x_label or "Time"),
                observed_series=observed_series,
                observed_x_by_species=observed_x_by_species or {},
                model_x=np.asarray([], dtype=float) if model_x is None else model_x,
                model_series=model_series,
                model_x_by_species=model_x_by_species or {},
                dataset_stats=(dataset_stats_by_dataset or {}).get(dataset_id, {}),
                status="ok",
                diagnostics=(),
            )
        except (TypeError, ValueError):
            datasets[dataset_id] = FitRenderDatasetProjection(
                dataset_id=dataset_id,
                observed_x=np.asarray([], dtype=float),
                observed_x_label=str(observed_x_label or "Time"),
                observed_series={},
                observed_x_by_species={},
                model_x=np.asarray([], dtype=float),
                model_series={},
                model_x_by_species={},
                dataset_stats=(dataset_stats_by_dataset or {}).get(dataset_id, {}),
                status="invalid_projection",
                diagnostics=("Render projection arrays failed validation.",),
            )
    return FitRenderProjection(
        phase=phase,
        run_stamp_hash=run_stamp_hash,
        sequence=int(sequence),
        cost=cost,
        datasets=datasets,
    )


def projection_from_global_fit_result(
    result: object,
    *,
    run_stamp_hash: str,
    phase: str = "final",
    cost: Optional[float] = None,
) -> FitRenderProjection:
    dataset_ids = [
        str(info.dataset_id)
        for info in (getattr(result, "dataset_info", None) or [])
        if str(getattr(info, "dataset_id", "") or "").strip()
    ]
    dataset_stats: Dict[str, Dict[str, float]] = {}
    observed_x_by_dataset: Dict[str, np.ndarray] = {}
    observed_x_label_by_dataset: Dict[str, str] = {}
    observed_series_by_dataset = getattr(result, "plot_observed_series", {}) or {}
    observed_x_by_species_by_dataset = getattr(result, "plot_observed_x", {}) or {}
    model_x_by_dataset: Dict[str, np.ndarray] = {}
    model_x_by_species_by_dataset = getattr(result, "plot_model_x_by_species", {}) or {}
    for info in (getattr(result, "dataset_info", None) or []):
        dataset_id = str(getattr(info, "dataset_id", "") or "").strip()
        if not dataset_id:
            continue
        dataset_stats[dataset_id] = {
            "chi_squared": float(getattr(info, "chi_squared", 0.0)),
            "r_squared": float(getattr(info, "r_squared", 0.0)),
        }
        x_name = str(getattr(info, "x_name", "t") or "t").strip() or "t"
        label = "Time" if x_name == "t" else x_name
        shared_observed = _shared_axis_or_empty(observed_x_by_species_by_dataset.get(dataset_id))
        if shared_observed.size:
            observed_x_by_dataset[dataset_id] = shared_observed
        if (
            dataset_id in observed_x_by_dataset
            or isinstance(observed_x_by_species_by_dataset.get(dataset_id), Mapping)
            or dataset_id in observed_series_by_dataset
        ):
            observed_x_label_by_dataset[dataset_id] = label
        species_model_axes = model_x_by_species_by_dataset.get(dataset_id)
        if isinstance(species_model_axes, Mapping) and species_model_axes:
            model_x = _shared_axis_or_empty(species_model_axes)
            if model_x.size:
                model_x_by_dataset[dataset_id] = model_x
    return build_fit_render_projection(
        phase=phase,
        run_stamp_hash=run_stamp_hash,
        sequence=int(getattr(result, "nfev", 0) or 0),
        cost=float(cost) if cost is not None else float(getattr(result, "global_chi_squared", 0.0) or 0.0),
        observed_x_by_dataset=observed_x_by_dataset,
        observed_x_label_by_dataset=observed_x_label_by_dataset,
        observed_series_by_dataset=observed_series_by_dataset,
        observed_x_by_species_by_dataset=observed_x_by_species_by_dataset,
        model_x_by_dataset=model_x_by_dataset,
        model_series_by_dataset=getattr(result, "plot_model_series", {}) or {},
        model_x_by_species_by_dataset=model_x_by_species_by_dataset,
        dataset_stats_by_dataset=dataset_stats,
        dataset_ids=dataset_ids,
    )
