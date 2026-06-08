from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Mapping, MutableMapping, Sequence

import numpy as np


def _as_float_array(values: object) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1).copy()


def copy_observations_map(observations: Mapping[str, object] | None) -> Dict[str, Dict[str, np.ndarray]]:
    copied: Dict[str, Dict[str, np.ndarray]] = {}
    for raw_name, raw_spec in dict(observations or {}).items():
        name = str(raw_name).strip()
        if not name or not isinstance(raw_spec, Mapping):
            continue
        t_values = _as_float_array(raw_spec.get("t", []))
        y_values = _as_float_array(raw_spec.get("y", []))
        if t_values.size != y_values.size:
            raise ValueError(
                f"Observation series '{name}' has {t_values.size} time points but {y_values.size} values."
            )
        copied[name] = {"t": t_values, "y": y_values}
    return copied


def observations_have_points(observations: Mapping[str, object] | None) -> bool:
    copied = copy_observations_map(observations)
    for spec in copied.values():
        if np.asarray(spec.get("t", []), dtype=float).reshape(-1).size > 0:
            return True
        if np.asarray(spec.get("y", []), dtype=float).reshape(-1).size > 0:
            return True
    return False


def observations_from_rectangular_payload(
    *,
    t: object,
    species: Mapping[str, object] | None,
) -> Dict[str, Dict[str, np.ndarray]]:
    t_values = _as_float_array(t)
    observations: Dict[str, Dict[str, np.ndarray]] = {}
    for raw_name, raw_values in dict(species or {}).items():
        name = str(raw_name).strip()
        if not name:
            continue
        series = _as_float_array(raw_values)
        if series.size != t_values.size:
            raise ValueError(
                f"Observation series '{name}' has {series.size} points but time axis has {t_values.size}."
            )
        if t_values.size == 0:
            observations[name] = {"t": np.asarray([], dtype=float), "y": np.asarray([], dtype=float)}
            continue
        finite_mask = np.isfinite(series)
        observations[name] = {
            "t": t_values[finite_mask].copy(),
            "y": series[finite_mask].copy(),
        }
    return observations


def observations_from_payload(payload: Mapping[str, Any] | None) -> Dict[str, Dict[str, np.ndarray]]:
    source = dict(payload or {})
    raw_observations = source.get("observations")
    if isinstance(raw_observations, Mapping):
        copied = copy_observations_map(raw_observations)
        if observations_have_points(copied):
            return copied
    species = source.get("species")
    if isinstance(species, Mapping):
        return observations_from_rectangular_payload(t=source.get("t", []), species=species)
    return {}


def dense_view_from_observations(
    observations: Mapping[str, Mapping[str, object]] | None,
) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
    copied = copy_observations_map(observations)
    all_times: list[float] = []
    for spec in copied.values():
        all_times.extend(float(value) for value in np.asarray(spec["t"], dtype=float).reshape(-1))
    if not all_times:
        return np.asarray([], dtype=float), {}
    union_t = np.unique(np.asarray(all_times, dtype=float))
    dense_species: Dict[str, np.ndarray] = {}
    for name, spec in copied.items():
        dense = np.full(union_t.shape, np.nan, dtype=float)
        t_values = np.asarray(spec["t"], dtype=float).reshape(-1)
        y_values = np.asarray(spec["y"], dtype=float).reshape(-1)
        if t_values.size:
            positions = np.searchsorted(union_t, t_values)
            valid = (positions >= 0) & (positions < union_t.size) & np.isclose(union_t[positions], t_values)
            dense[positions[valid]] = y_values[valid]
        dense_species[name] = dense
    return union_t, dense_species


def dataset_payload_from_observations(
    observations: Mapping[str, Mapping[str, object]] | None,
    *,
    metadata: Mapping[str, Any] | None = None,
    extras: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    copied_observations = copy_observations_map(observations)
    payload: Dict[str, Any] = {
        "observations": copied_observations,
        "metadata": deepcopy(dict(metadata or {})),
    }
    for key, value in dict(extras or {}).items():
        if key in payload:
            continue
        payload[key] = deepcopy(value)
    return payload


def canonicalize_dataset_payload(payload: Mapping[str, Any] | None) -> Dict[str, Any]:
    source = dict(payload or {})
    metadata = dict(source.get("metadata") or {})
    extras = {
        key: value
        for key, value in source.items()
        if key not in {"observations", "t", "species", "metadata"}
    }
    observations = observations_from_payload(source)
    return dataset_payload_from_observations(observations, metadata=metadata, extras=extras)


def scale_payload_in_place(
    payload: MutableMapping[str, Any],
    *,
    time_factor: float = 1.0,
    conc_factors: Mapping[str, float] | None = None,
) -> None:
    observations = copy_observations_map(observations_from_payload(payload))
    for spec in observations.values():
        if float(time_factor) != 1.0:
            spec["t"] = np.asarray(spec["t"], dtype=float) * float(time_factor)
    for name, factor in dict(conc_factors or {}).items():
        spec = observations.get(str(name))
        if spec is None or float(factor) == 1.0:
            continue
        spec["y"] = np.asarray(spec["y"], dtype=float) * float(factor)
    refreshed = dataset_payload_from_observations(
        observations,
        metadata=dict(payload.get("metadata") or {}),
        extras={
            key: value
            for key, value in payload.items()
            if key not in {"observations", "t", "species", "metadata"}
        },
    )
    payload.clear()
    payload.update(refreshed)


def dense_view_from_payload(payload: Mapping[str, Any] | None) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
    observations = observations_from_payload(payload)
    return dense_view_from_observations(observations)


def sampled_observations_from_dense(
    *,
    t: object,
    species_data: Mapping[str, object] | None,
    species_names: Sequence[str] | None = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    dense_t = _as_float_array(t)
    chosen_names = [str(name) for name in (species_names or list(dict(species_data or {}).keys())) if str(name).strip()]
    observations: Dict[str, Dict[str, np.ndarray]] = {}
    for name in chosen_names:
        if not isinstance(species_data, Mapping) or name not in species_data:
            continue
        series = _as_float_array(species_data[name])
        if series.size != dense_t.size:
            raise ValueError(
                f"Observation series '{name}' has {series.size} points but sampled time axis has {dense_t.size}."
            )
        finite_mask = np.isfinite(series)
        observations[name] = {
            "t": dense_t[finite_mask].copy(),
            "y": series[finite_mask].copy(),
        }
    return observations


def export_rows_from_observations(
    observations: Mapping[str, Mapping[str, object]] | None,
    *,
    species_names: Sequence[str] | None = None,
) -> tuple[list[str], list[list[object]]]:
    copied = copy_observations_map(observations)
    ordered_species = [str(name) for name in (species_names or list(copied.keys())) if str(name).strip()]
    if not ordered_species:
        raise ValueError("No series selected to export.")
    selected = {name: copied[name] for name in ordered_species if name in copied}
    if not selected:
        raise ValueError("No dataset data available to export.")
    time_values = sorted(
        {
            float(t_value)
            for spec in selected.values()
            for t_value in np.asarray(spec["t"], dtype=float).reshape(-1)
        }
    )
    if not time_values:
        raise ValueError("No dataset data available to export.")
    by_species_time: Dict[str, Dict[float, list[float]]] = {}
    for name, spec in selected.items():
        grouped: Dict[float, list[float]] = {}
        t_values = np.asarray(spec["t"], dtype=float).reshape(-1)
        y_values = np.asarray(spec["y"], dtype=float).reshape(-1)
        for t_value, y_value in zip(t_values, y_values):
            grouped.setdefault(float(t_value), []).append(float(y_value))
        by_species_time[name] = grouped
    rows: list[list[object]] = []
    for t_value in time_values:
        row_count = max(
            len(by_species_time.get(name, {}).get(float(t_value), []))
            for name in ordered_species
        )
        for rep_index in range(max(1, row_count)):
            row: list[object] = [float(t_value)]
            for name in ordered_species:
                values = by_species_time.get(name, {}).get(float(t_value), [])
                if rep_index >= len(values):
                    row.append("")
                    continue
                value = float(values[rep_index])
                row.append("" if not np.isfinite(value) else value)
            rows.append(row)
    return ["Time"] + ordered_species, rows


def seeded_values_at_t0(
    observations: Mapping[str, Mapping[str, object]] | None,
    *,
    mechanism_species: Sequence[str],
    tol: float,
) -> Dict[str, float]:
    copied = copy_observations_map(observations)
    seeded: Dict[str, float] = {}
    tolerance = abs(float(tol))
    for raw_name in mechanism_species:
        name = str(raw_name).strip()
        if not name or name not in copied:
            continue
        t_values = np.asarray(copied[name]["t"], dtype=float).reshape(-1)
        y_values = np.asarray(copied[name]["y"], dtype=float).reshape(-1)
        if t_values.size == 0 or y_values.size == 0:
            continue
        matches = np.flatnonzero(np.isfinite(t_values) & (np.abs(t_values) <= tolerance))
        if matches.size == 0:
            continue
        idx = int(matches[0])
        value = float(y_values[idx])
        if np.isfinite(value):
            seeded[name] = value
    return seeded
