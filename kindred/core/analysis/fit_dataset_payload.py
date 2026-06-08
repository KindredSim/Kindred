from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np

from kindred.core.analysis.x_mapping import ALLOWED_X_MAPPING_MODES, normalize_x_mapping_mode, parse_x_mapping_mode
from kindred.core.datasets.observation_payload import (
    copy_observations_map,
    dense_view_from_observations,
    observations_have_points,
    sampled_observations_from_dense,
)


def _readonly_float_array(values: object, *, reshape_1d: bool = False) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if reshape_1d:
        array = array.reshape(-1)
    array = array.copy()
    array.setflags(write=False)
    return array


def _freeze_observations_map(
    observations: Mapping[str, Mapping[str, object]] | None,
) -> Mapping[str, Mapping[str, np.ndarray]]:
    frozen: Dict[str, Mapping[str, np.ndarray]] = {}
    for name, spec in copy_observations_map(observations).items():
        frozen[str(name)] = MappingProxyType(
            {
                "t": _readonly_float_array(spec.get("t", []), reshape_1d=True),
                "y": _readonly_float_array(spec.get("y", []), reshape_1d=True),
            }
        )
    return MappingProxyType(frozen)


def _freeze_axis_map(axis_map: Mapping[str, object] | None) -> Mapping[str, np.ndarray]:
    frozen = {
        str(name): _readonly_float_array(values, reshape_1d=True)
        for name, values in dict(axis_map or {}).items()
        if str(name).strip()
    }
    return MappingProxyType(frozen)


def _freeze_weights_map(weights: Mapping[str, object] | None) -> Mapping[str, float]:
    frozen = {
        str(name): float(value)
        for name, value in dict(weights or {}).items()
        if str(name).strip()
    }
    return MappingProxyType(frozen)


@dataclass(frozen=True)
class FitDatasetSpec(Mapping[str, Any]):
    dataset_id: str
    t_exp: np.ndarray
    species_list: Tuple[str, ...]
    y_matrix: np.ndarray
    point_count: int
    x_name: str
    x_obs: Optional[np.ndarray]
    x_mode: str
    observations: Mapping[str, Mapping[str, np.ndarray]] = field(default_factory=dict)
    x_obs_by_species: Mapping[str, np.ndarray] = field(default_factory=dict)
    target_weights: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        dataset_id = str(self.dataset_id)
        t_exp = _readonly_float_array(self.t_exp, reshape_1d=True)
        species_list = tuple(str(name) for name in (self.species_list or ()) if str(name).strip())
        y_matrix = _readonly_float_array(self.y_matrix)
        point_count = int(self.point_count)
        x_name = str(self.x_name or "t").strip() or "t"
        observations = _freeze_observations_map(self.observations)
        x_obs_by_species = _freeze_axis_map(self.x_obs_by_species)
        x_obs = None if self.x_obs is None else _readonly_float_array(self.x_obs, reshape_1d=True)
        x_mode = normalize_x_mapping_mode(self.x_mode)
        target_weights = _freeze_weights_map(self.target_weights)

        if x_name != "t" and observations:
            missing_axes = [name for name in species_list if name in observations and name not in x_obs_by_species]
            if missing_axes:
                raise ValueError(
                    f"Dataset '{dataset_id}' requires x_obs_by_species for X='{x_name}' "
                    f"({', '.join(missing_axes)})."
                )
            for name in species_list:
                if name not in observations or name not in x_obs_by_species:
                    continue
                y_values = np.asarray(observations[name]["y"], dtype=float).reshape(-1)
                axis_values = np.asarray(x_obs_by_species[name], dtype=float).reshape(-1)
                if axis_values.size != y_values.size:
                    raise ValueError(
                        f"Dataset '{dataset_id}' species '{name}' has {axis_values.size} x observations for "
                        f"X='{x_name}' but {y_values.size} observed values."
                    )
            x_obs = None

        object.__setattr__(self, "dataset_id", dataset_id)
        object.__setattr__(self, "t_exp", t_exp)
        object.__setattr__(self, "species_list", species_list)
        object.__setattr__(self, "y_matrix", y_matrix)
        object.__setattr__(self, "point_count", point_count)
        object.__setattr__(self, "x_name", x_name)
        object.__setattr__(self, "x_obs", x_obs)
        object.__setattr__(self, "x_mode", x_mode)
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "x_obs_by_species", x_obs_by_species)
        object.__setattr__(self, "target_weights", target_weights)

    def to_payload_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "id": str(self.dataset_id),
            "species": list(self.species_list),
            "observations": {
                str(name): {
                    "t": np.asarray(spec.get("t", []), dtype=float).reshape(-1).copy(),
                    "y": np.asarray(spec.get("y", []), dtype=float).reshape(-1).copy(),
                }
                for name, spec in dict(self.observations or {}).items()
            },
            "target_weights": dict(self.target_weights),
        }
        if not payload["observations"]:
            payload["t"] = np.asarray(self.t_exp, dtype=float).reshape(-1).copy()
            payload["y"] = np.asarray(self.y_matrix, dtype=float).copy()
        if str(self.x_name or "t").strip() != "t":
            payload["x_name"] = str(self.x_name or "t").strip() or "t"
            payload["x_mapping_mode"] = str(self.x_mode or "auto").strip() or "auto"
            if self.x_obs is not None and not payload["observations"]:
                payload["x_obs"] = np.asarray(self.x_obs, dtype=float).reshape(-1).copy()
            if self.x_obs_by_species:
                payload["x_obs_by_species"] = {
                    str(name): np.asarray(values, dtype=float).reshape(-1).copy()
                    for name, values in dict(self.x_obs_by_species).items()
                    if str(name).strip()
                }
        return payload

    def __getitem__(self, key: str) -> Any:
        return self.to_payload_dict()[key]

    def __iter__(self):
        return iter(self.to_payload_dict())

    def __len__(self) -> int:
        return len(self.to_payload_dict())


FitDatasetPayloadState = Literal["valid", "absent", "invalid"]


@dataclass(frozen=True, slots=True)
class FitDatasetPayloadResult:
    state: FitDatasetPayloadState
    payload: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    @classmethod
    def valid(cls, payload: Mapping[str, Any]) -> "FitDatasetPayloadResult":
        return cls("valid", payload=dict(payload), error=None)

    @classmethod
    def absent(cls) -> "FitDatasetPayloadResult":
        return cls("absent", payload=None, error=None)

    @classmethod
    def invalid(cls, error: str) -> "FitDatasetPayloadResult":
        message = str(error or "").strip() or "Dataset payload is invalid."
        return cls("invalid", payload=None, error=message)

    def as_legacy_tuple(self) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        if self.state == "valid" and self.payload is not None:
            return dict(self.payload), None
        return None, self.error


def coerce_fit_dataset_payload_result(value: object) -> FitDatasetPayloadResult:
    if isinstance(value, FitDatasetPayloadResult):
        return value
    if value is None:
        return FitDatasetPayloadResult.absent()
    if isinstance(value, Mapping):
        state = str(value.get("state") or "").strip().lower()
        payload = value.get("payload")
        error = value.get("error")
    else:
        state = str(getattr(value, "state", "") or "").strip().lower()
        payload = getattr(value, "payload", None)
        error = getattr(value, "error", None)
    if state == "valid" and isinstance(payload, Mapping):
        return FitDatasetPayloadResult.valid(payload)
    if state == "invalid":
        return FitDatasetPayloadResult.invalid(str(error or "Dataset payload is invalid."))
    if state == "absent":
        return FitDatasetPayloadResult.absent()
    return FitDatasetPayloadResult.invalid(str(error or "Dataset payload is invalid."))


def normalize_fit_dataset_dicts(datasets: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize raw dataset dicts into a consistent input shape for fitting.

    - Validates required keys ('t', 'y', 'species')
    - Adds default 'id' when missing
    - Returns shallow-copied dicts (never mutates caller mappings)
    """
    if not datasets:
        raise ValueError("At least one dataset is required")

    out: List[Dict[str, Any]] = []
    for i, ds in enumerate(datasets):
        if not isinstance(ds, Mapping):
            raise TypeError(f"Dataset {i} must be a mapping; got {type(ds)!r}.")
        if "observations" in ds and isinstance(ds.get("observations"), Mapping):
            obs = copy_observations_map(ds.get("observations"))
            if observations_have_points(obs):
                raw_species = ds.get("species")
                if isinstance(raw_species, str):
                    species_names = [str(raw_species)]
                elif isinstance(raw_species, (list, tuple)):
                    species_names = [str(name) for name in raw_species if str(name).strip()]
                else:
                    species_names = list(obs.keys())
                missing_species = [name for name in species_names if name not in obs]
                if missing_species:
                    raise ValueError(
                        f"Dataset {i} observations missing required selected species: {', '.join(missing_species)}"
                    )
                ds_norm = dict(ds)
                ds_norm["observations"] = obs
                ds_norm["species"] = species_names
                if "id" not in ds_norm:
                    ds_norm["id"] = f"dataset_{i}"
                out.append(ds_norm)
                continue
        raw_species = ds.get("species")
        if "t" in ds and isinstance(raw_species, Mapping) and "y" not in ds:
            species_names = [str(name) for name in raw_species.keys() if str(name).strip()]
            if not species_names:
                raise ValueError(f"Dataset {i} missing required 'species' field")
            ds_norm = dict(ds)
            ds_norm["species_data"] = {str(name): raw_species[name] for name in species_names}
            ds_norm["species"] = species_names
            if "id" not in ds_norm:
                ds_norm["id"] = f"dataset_{i}"
            out.append(ds_norm)
            continue
        if "t" not in ds or "y" not in ds:
            raise ValueError(f"Dataset {i} missing required 't' or 'y' fields")
        if "species" not in ds:
            raise ValueError(f"Dataset {i} missing required 'species' field")
        ds_norm = dict(ds)
        ds_norm["t"] = ds["t"]
        ds_norm["y"] = ds["y"]
        ds_norm["species"] = ds["species"]
        if "id" not in ds_norm:
            ds_norm["id"] = f"dataset_{i}"
        out.append(ds_norm)
    return out


def _matched_sample_positions(sampled_t_arr: np.ndarray, query_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if sampled_t_arr.size == 0 or query_t.size == 0:
        return np.asarray([], dtype=int), np.zeros(query_t.shape, dtype=bool)
    positions = np.searchsorted(sampled_t_arr, query_t)
    valid = (positions >= 0) & (positions < sampled_t_arr.size)
    safe_positions = np.clip(positions, 0, max(sampled_t_arr.size - 1, 0))
    valid &= np.isclose(sampled_t_arr[safe_positions], query_t)
    return safe_positions, valid


def _copy_fit_dataset_spec(spec: FitDatasetSpec) -> FitDatasetSpec:
    t_values = np.asarray(spec.t_exp, dtype=float).reshape(-1)
    species_list = tuple(str(name) for name in (spec.species_list or []) if str(name).strip())
    y_matrix = np.asarray(spec.y_matrix, dtype=float)
    x_name = str(spec.x_name or "t").strip() or "t"
    x_obs = None if spec.x_obs is None else np.asarray(spec.x_obs, dtype=float).reshape(-1)
    x_mode = normalize_x_mapping_mode(spec.x_mode)
    observations = copy_observations_map(getattr(spec, "observations", None))
    x_obs_by_species = {
        str(name): np.asarray(values, dtype=float).reshape(-1)
        for name, values in dict(getattr(spec, "x_obs_by_species", None) or {}).items()
        if str(name).strip()
    }
    target_weights = normalize_dataset_target_weights(
        dataset_id=str(spec.dataset_id),
        selected_targets=species_list,
        target_weights=getattr(spec, "target_weights", None),
    )
    point_count = int(sum(int(np.asarray(obs["y"], dtype=float).size) for obs in observations.values()))
    if point_count <= 0:
        point_count = int(np.asarray(y_matrix, dtype=float).size)
    return FitDatasetSpec(
        dataset_id=str(spec.dataset_id),
        t_exp=t_values,
        species_list=species_list,
        y_matrix=y_matrix,
        point_count=point_count,
        x_name=x_name,
        x_obs=x_obs,
        x_mode=x_mode,
        observations=observations,
        x_obs_by_species=x_obs_by_species,
        target_weights=target_weights,
    )


def _filter_observations_to_sampled_t(
    observations: Mapping[str, Mapping[str, object]],
    *,
    sampled_t: object,
) -> Dict[str, Dict[str, np.ndarray]]:
    filtered = copy_observations_map(observations)
    sampled_t_arr = np.asarray(sampled_t, dtype=float).reshape(-1)
    if sampled_t_arr.size == 0:
        return {
            name: {
                "t": np.asarray([], dtype=float),
                "y": np.asarray([], dtype=float),
            }
            for name in filtered
        }
    for name, spec in filtered.items():
        t_values = np.asarray(spec["t"], dtype=float).reshape(-1)
        y_values = np.asarray(spec["y"], dtype=float).reshape(-1)
        _positions, valid = _matched_sample_positions(sampled_t_arr, t_values)
        filtered[name] = {
            "t": t_values[valid].copy(),
            "y": y_values[valid].copy(),
        }
    return filtered


def _coerce_float_vector(values: object, *, error_message: str) -> np.ndarray:
    try:
        return np.asarray(values, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(error_message) from exc


def _coerce_species_local_x_obs(
    *,
    dataset_id: str,
    x_name: str,
    observations: Mapping[str, Mapping[str, object]],
    species_list: Sequence[str],
    sampled_t: Optional[object] = None,
    x_obs: Optional[object] = None,
    x_obs_by_species: Optional[Mapping[str, object]] = None,
) -> Dict[str, np.ndarray]:
    ds_id = str(dataset_id or "").strip() or "<dataset>"
    if str(x_name or "t").strip() == "t":
        return {}
    selected_observations = {
        name: observations[name]
        for name in species_list
        if name in observations
    }
    if not selected_observations:
        return {}
    if isinstance(x_obs_by_species, Mapping):
        normalized: Dict[str, np.ndarray] = {}
        missing_species = [name for name in species_list if name not in x_obs_by_species]
        if missing_species:
            raise ValueError(
                f"Dataset '{ds_id}' is missing per-species x observations for X='{x_name}' "
                f"({', '.join(missing_species)})."
            )
        for name in species_list:
            values = _coerce_float_vector(
                x_obs_by_species[name],
                error_message=(
                    f"Dataset '{ds_id}' species '{name}' has invalid x observations for X='{x_name}'."
                ),
            )
            y_values = np.asarray(selected_observations[name]["y"], dtype=float).reshape(-1)
            if values.size != y_values.size:
                raise ValueError(
                    f"Dataset '{ds_id}' species '{name}' has {values.size} x observations for X='{x_name}' "
                    f"but {y_values.size} sampled observations."
                )
            if not np.all(np.isfinite(values)):
                raise ValueError(
                    f"Dataset '{ds_id}' species '{name}' has non-finite x observations for X='{x_name}'."
                )
            normalized[name] = values.copy()
        return normalized

    if x_obs is None:
        raise ValueError(f"Dataset '{ds_id}' is missing x_obs for X='{x_name}'.")

    if sampled_t is None:
        sampled_t_arr, _ = dense_view_from_observations(selected_observations)
    else:
        sampled_t_arr = _coerce_float_vector(
            sampled_t,
            error_message=f"Dataset '{ds_id}' has invalid sampled time points for X='{x_name}'.",
        )
    x_obs_arr = _coerce_float_vector(
        x_obs,
        error_message=f"Dataset '{ds_id}' has invalid x_obs for X='{x_name}'.",
    )
    if x_obs_arr.size != sampled_t_arr.size:
        raise ValueError(
            f"Dataset '{ds_id}' is missing x_obs for X='{x_name}' (must match sampled time grid)."
        )

    normalized: Dict[str, np.ndarray] = {}
    for name in species_list:
        species_obs = selected_observations.get(name)
        if species_obs is None:
            continue
        obs_t = np.asarray(species_obs["t"], dtype=float).reshape(-1)
        positions, valid = _matched_sample_positions(sampled_t_arr, obs_t)
        if not np.all(valid):
            raise ValueError(
                f"Dataset '{ds_id}' species '{name}' is missing paired x observations for X='{x_name}'."
            )
        species_x = np.asarray(x_obs_arr[positions], dtype=float).reshape(-1)
        if not np.all(np.isfinite(species_x)):
            raise ValueError(
                f"Dataset '{ds_id}' species '{name}' has non-finite x observations for X='{x_name}'."
            )
        normalized[name] = species_x.copy()
    return normalized


def _coerce_fit_dataset_spec_from_mapping(
    ds: Mapping[str, Any],
    *,
    allowed_x_modes: set[str],
    already_normalized: bool = False,
) -> FitDatasetSpec:
    ds_norm = dict(ds) if already_normalized else normalize_fit_dataset_dicts([ds])[0]
    ds_id = str(ds_norm["id"])
    raw_species = ds_norm.get("species")
    if isinstance(raw_species, str):
        species_list = [str(raw_species)]
    elif isinstance(raw_species, (list, tuple)):
        species_list = [str(name) for name in raw_species if str(name).strip()]
    else:
        species_list = [
            str(name)
            for name in dict(ds_norm.get("observations") or {}).keys()
            if str(name).strip()
        ]
    if not species_list:
        raise ValueError(f"Dataset '{ds_id}' requires at least one selected species.")
    raw_observations = ds_norm.get("observations")
    observations = (
        copy_observations_map(raw_observations)
        if isinstance(raw_observations, Mapping)
        else {}
    )
    sampled_t_for_x = (
        np.asarray(ds_norm.get("t"), dtype=float).reshape(-1)
        if ds_norm.get("t") is not None
        else np.asarray([], dtype=float)
    )
    if observations_have_points(observations):
        pass
    else:
        dense_species_data = ds_norm.get("species_data")
        if not isinstance(dense_species_data, Mapping):
            raw_y = np.asarray(ds_norm.get("y", []), dtype=float)
            if len(species_list) == 1:
                dense_species_data = {
                    species_list[0]: raw_y.reshape(-1),
                }
            else:
                t_values_dense = np.asarray(ds_norm.get("t", []), dtype=float).reshape(-1)
                if raw_y.ndim != 2:
                    raise ValueError(
                        f"Dataset '{ds_id}' uses multiple species but 'y' is not a 2D array."
                    )
                if raw_y.shape[0] != len(species_list):
                    raise ValueError(
                        f"Dataset '{ds_id}' has {raw_y.shape[0]} rows but {len(species_list)} species entries."
                    )
                if raw_y.shape[1] != t_values_dense.size:
                    raise ValueError(
                        f"Dataset '{ds_id}' has {raw_y.shape[1]} points per species but time axis has {t_values_dense.size}."
                    )
                y_matrix = np.asarray(raw_y, dtype=float)
                dense_species_data = {
                    name: np.asarray(y_matrix[index], dtype=float).reshape(-1)
                    for index, name in enumerate(species_list)
                }
        observations = sampled_observations_from_dense(
            t=ds_norm.get("t", []),
            species_data=dense_species_data,
            species_names=species_list,
        )
    missing = [name for name in species_list if name not in observations]
    if missing:
        raise ValueError(f"Dataset '{ds_id}' is missing series: {', '.join(missing)}.")
    empty = [
        name
        for name in species_list
        if np.asarray(observations[name]["y"], dtype=float).reshape(-1).size == 0
    ]
    if empty:
        raise ValueError(
            f"Dataset '{ds_id}' selected series have no surviving sampled observations: {', '.join(empty)}."
        )
    t_values, dense_species = dense_view_from_observations({name: observations[name] for name in species_list})
    y_matrix = (
        np.vstack([np.asarray(dense_species[name], dtype=float).reshape(-1) for name in species_list])
        if species_list
        else np.empty((0, t_values.size), dtype=float)
    )


    x_name = str(ds_norm.get("x_name") or "t").strip() or "t"
    x_obs: Optional[np.ndarray] = None
    x_obs_by_species: Dict[str, np.ndarray] = {}
    x_mode = "auto"
    target_weights = normalize_dataset_target_weights(
        dataset_id=ds_id,
        selected_targets=species_list,
        target_weights=ds_norm.get("target_weights"),
    )
    if x_name != "t":
        mode_raw = ds_norm.get("x_mapping_mode")
        x_mode = normalize_x_mapping_mode(mode_raw)
        if x_mode not in allowed_x_modes:
            raise ValueError(
                f"Dataset '{ds_id}' has invalid x_mapping_mode '{mode_raw}'. "
                "Expected auto, monotone, or time_guided."
            )
        raw_x_obs_by_species = ds_norm.get("x_obs_by_species")
        if ds_norm.get("x_obs") is not None:
            x_obs_candidate = _coerce_float_vector(
                ds_norm.get("x_obs", []),
                error_message=f"Dataset '{ds_id}' has invalid x_obs for X='{x_name}'.",
            )
            if not isinstance(raw_x_obs_by_species, Mapping):
                expected_size = sampled_t_for_x.size if sampled_t_for_x.size else t_values.size
                if x_obs_candidate.size != expected_size:
                    raise ValueError(
                        f"Dataset '{ds_id}' x_obs has {x_obs_candidate.size} points but time axis has {expected_size}."
                    )
                if not np.all(np.isfinite(x_obs_candidate)):
                    raise ValueError(f"Dataset '{ds_id}' x_obs contains non-finite values for X='{x_name}'.")
                x_obs = x_obs_candidate
        x_obs_by_species = _coerce_species_local_x_obs(
            dataset_id=ds_id,
            x_name=x_name,
            observations=observations,
            species_list=species_list,
            sampled_t=sampled_t_for_x if sampled_t_for_x.size else t_values,
            x_obs=x_obs,
            x_obs_by_species=raw_x_obs_by_species,
        )

    return FitDatasetSpec(
        dataset_id=ds_id,
        t_exp=t_values,
        species_list=tuple(species_list),
        y_matrix=y_matrix,
        point_count=int(sum(int(np.asarray(observations[name]["y"], dtype=float).size) for name in species_list)),
        x_name=x_name,
        x_obs=x_obs,
        x_mode=x_mode,
        observations={name: observations[name] for name in species_list},
        x_obs_by_species=x_obs_by_species,
        target_weights=target_weights,
    )


def coerce_fit_dataset_specs(datasets: Sequence[object]) -> List[FitDatasetSpec]:
    """
    Coerce normalized dataset dicts into a typed, validated structure used by core fitting.
    """
    if not datasets:
        raise ValueError("At least one dataset is required")

    allowed_x_modes = set(ALLOWED_X_MAPPING_MODES)
    specs: List[FitDatasetSpec] = []
    ordered_items: list[FitDatasetSpec | None] = []
    raw_mappings: list[Mapping[str, Any]] = []
    for ds in datasets:
        if isinstance(ds, FitDatasetSpec):
            ordered_items.append(_copy_fit_dataset_spec(ds))
        else:
            if not isinstance(ds, Mapping):
                raise TypeError(f"Dataset must be a mapping or FitDatasetSpec; got {type(ds)!r}.")
            ordered_items.append(None)
            raw_mappings.append(ds)

    normalized_mappings = iter(normalize_fit_dataset_dicts(raw_mappings)) if raw_mappings else iter(())
    for item in ordered_items:
        if item is not None:
            specs.append(item)
            continue
        specs.append(
            _coerce_fit_dataset_spec_from_mapping(
                next(normalized_mappings),
                allowed_x_modes=allowed_x_modes,
                already_normalized=True,
            )
        )
    return specs




def normalize_dataset_target_weights(
    *,
    dataset_id: str,
    selected_targets: Sequence[str],
    target_weights: Optional[Mapping[str, object]],
) -> Dict[str, float]:
    ds_id = str(dataset_id or "").strip() or "<dataset>"
    selection = [str(name) for name in (selected_targets or []) if str(name).strip()]
    if target_weights is not None and not isinstance(target_weights, Mapping):
        raise ValueError(
            f"Dataset '{ds_id}' target_weights must be a mapping or None, "
            f"got {type(target_weights)!r}."
        )
    raw_weights = dict(target_weights or {})
    normalized: Dict[str, float] = {}
    for name in selection:
        raw_value = raw_weights.get(name, 1.0)
        try:
            value = float(raw_value)
        except Exception as exc:
            raise ValueError(f"Dataset '{ds_id}' target weight for '{name}' is invalid: {raw_value!r}.") from exc
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"Dataset '{ds_id}' target weight for '{name}' must be finite and positive.")
        normalized[name] = float(value)
    return normalized


def read_fit_dataset_payload(
    *,
    dataset_id: str,
    t: Optional[np.ndarray] = None,
    species_data: Optional[Mapping[str, np.ndarray]] = None,
    observations: Optional[Mapping[str, Mapping[str, object]]] = None,
    selected_species: Sequence[str],
    target_weights: Optional[Mapping[str, object]] = None,
    x_name: str = "t",
    x_obs: Optional[np.ndarray] = None,
    x_obs_by_species: Optional[Mapping[str, object]] = None,
    x_mapping_mode: str = "auto",
) -> FitDatasetPayloadResult:
    ds_id = str(dataset_id or "").strip()
    if not ds_id:
        return FitDatasetPayloadResult.invalid("Dataset id is missing.")
    selection = [str(x) for x in (selected_species or []) if str(x).strip()]
    if not selection:
        return FitDatasetPayloadResult.invalid(f"Dataset '{ds_id}' requires at least one selected series.")
    use_observation_authority = False
    observations_map: Dict[str, Dict[str, np.ndarray]]
    if observations is not None:
        try:
            observations_map = copy_observations_map(observations)
        except (TypeError, ValueError) as exc:
            return FitDatasetPayloadResult.invalid(str(exc))
        use_observation_authority = observations_have_points(observations_map)
        if use_observation_authority and t is not None:
            observations_map = _filter_observations_to_sampled_t(observations_map, sampled_t=t)
    if not use_observation_authority:
        if t is None:
            return FitDatasetPayloadResult.invalid(f"Dataset '{ds_id}' has no time points.")
        if not isinstance(species_data, Mapping) or not species_data:
            return FitDatasetPayloadResult.invalid(f"Dataset '{ds_id}' has no observed series available.")
        try:
            observations_map = sampled_observations_from_dense(
                t=t,
                species_data=species_data,
                species_names=selection,
            )
        except ValueError as exc:
            return FitDatasetPayloadResult.invalid(str(exc))
    selected_observations = {name: observations_map[name] for name in selection if name in observations_map}
    missing = [name for name in selection if name not in selected_observations]
    if missing:
        return FitDatasetPayloadResult.invalid(f"Dataset '{ds_id}' is missing series: {', '.join(missing)}.")
    if not selected_observations:
        return FitDatasetPayloadResult.invalid(f"Dataset '{ds_id}' requires at least one valid selected series.")
    empty = [
        name
        for name, spec in selected_observations.items()
        if np.asarray(spec["y"], dtype=float).reshape(-1).size == 0
    ]
    if empty:
        return FitDatasetPayloadResult.invalid(
            f"Dataset '{ds_id}' selected series have no surviving sampled observations: {', '.join(empty)}."
        )

    try:
        target_weights_norm = normalize_dataset_target_weights(
            dataset_id=ds_id,
            selected_targets=selection,
            target_weights=target_weights,
        )
    except ValueError as exc:
        return FitDatasetPayloadResult.invalid(str(exc))

    x_name_norm = str(x_name or "t").strip() or "t"
    dataset_payload: Dict[str, Any] = {
        "id": ds_id,
        "species": list(selection),
        "observations": selected_observations,
        "target_weights": dict(target_weights_norm),
    }

    if x_name_norm != "t":
        dataset_payload["x_name"] = x_name_norm
        try:
            dataset_payload["x_mapping_mode"] = parse_x_mapping_mode(x_mapping_mode)
        except ValueError:
            return FitDatasetPayloadResult.invalid(
                f"Dataset '{ds_id}' has invalid x_mapping_mode '{x_mapping_mode}'. "
                "Expected Auto, Monotone only, or Time-guided."
            )

        try:
            x_obs_by_species_norm = _coerce_species_local_x_obs(
                dataset_id=ds_id,
                x_name=x_name_norm,
                observations=selected_observations,
                species_list=selection,
                sampled_t=t,
                x_obs=x_obs,
                x_obs_by_species=x_obs_by_species,
            )
        except ValueError as exc:
            return FitDatasetPayloadResult.invalid(str(exc))
        dataset_payload["x_obs_by_species"] = x_obs_by_species_norm

    return FitDatasetPayloadResult.valid(dataset_payload)


def build_fit_dataset_payload(
    *,
    dataset_id: str,
    t: np.ndarray,
    species_data: Mapping[str, np.ndarray],
    selected_species: Sequence[str],
    target_weights: Optional[Mapping[str, object]] = None,
    x_name: str = "t",
    x_obs: Optional[np.ndarray] = None,
    x_mapping_mode: str = "auto",
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    return read_fit_dataset_payload(
        dataset_id=dataset_id,
        t=t,
        species_data=species_data,
        selected_species=selected_species,
        target_weights=target_weights,
        x_name=x_name,
        x_obs=x_obs,
        x_mapping_mode=x_mapping_mode,
    ).as_legacy_tuple()
