from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np

from kindred.core.analysis.x_mapping import ALLOWED_X_MAPPING_MODES, normalize_x_mapping_mode, parse_x_mapping_mode


@dataclass(frozen=True)
class FitDatasetSpec(Mapping[str, Any]):
    dataset_id: str
    t_exp: np.ndarray
    species_list: List[str]
    y_matrix: np.ndarray
    point_count: int
    x_name: str
    x_obs: Optional[np.ndarray]
    x_mode: str
    target_weights: Dict[str, float] = field(default_factory=dict)

    def to_payload_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "id": str(self.dataset_id),
            "t": np.asarray(self.t_exp, dtype=float).reshape(-1),
            "y": np.asarray(self.y_matrix, dtype=float),
            "species": list(self.species_list),
            "target_weights": dict(self.target_weights),
        }
        if str(self.x_name or "t").strip() != "t":
            payload["x_name"] = str(self.x_name or "t").strip() or "t"
            payload["x_mapping_mode"] = str(self.x_mode or "auto").strip() or "auto"
            if self.x_obs is not None:
                payload["x_obs"] = np.asarray(self.x_obs, dtype=float).reshape(-1)
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


def _copy_fit_dataset_spec(spec: FitDatasetSpec) -> FitDatasetSpec:
    t_values = np.asarray(spec.t_exp, dtype=float).reshape(-1)
    species_list = [str(name) for name in (spec.species_list or []) if str(name).strip()]
    y_matrix = np.asarray(spec.y_matrix, dtype=float)
    x_name = str(spec.x_name or "t").strip() or "t"
    x_obs = None if spec.x_obs is None else np.asarray(spec.x_obs, dtype=float).reshape(-1)
    x_mode = normalize_x_mapping_mode(spec.x_mode)
    target_weights = normalize_dataset_target_weights(
        dataset_id=str(spec.dataset_id),
        selected_targets=species_list,
        target_weights=getattr(spec, "target_weights", None),
    )
    return FitDatasetSpec(
        dataset_id=str(spec.dataset_id),
        t_exp=t_values,
        species_list=species_list,
        y_matrix=y_matrix,
        point_count=int(y_matrix.size),
        x_name=x_name,
        x_obs=x_obs,
        x_mode=x_mode,
        target_weights=target_weights,
    )


def _coerce_fit_dataset_spec_from_mapping(
    ds: Mapping[str, Any],
    *,
    allowed_x_modes: set[str],
    already_normalized: bool = False,
) -> FitDatasetSpec:
    ds_norm = dict(ds) if already_normalized else normalize_fit_dataset_dicts([ds])[0]
    ds_id = str(ds_norm["id"])
    t_values = np.asarray(ds_norm["t"], dtype=float).reshape(-1)

    species_list, y_matrix = normalize_dataset_species_and_y(
        dataset_id=ds_id,
        t_values=t_values,
        species=ds_norm["species"],
        y=ds_norm["y"],
    )

    x_name = str(ds_norm.get("x_name") or "t").strip() or "t"
    x_obs: Optional[np.ndarray] = None
    x_mode = "auto"
    target_weights = normalize_dataset_target_weights(
        dataset_id=ds_id,
        selected_targets=species_list,
        target_weights=ds_norm.get("target_weights"),
    )
    if x_name != "t":
        x_obs = np.asarray(ds_norm.get("x_obs", []), dtype=float).reshape(-1)
        if x_obs.size != t_values.size:
            raise ValueError(
                f"Dataset '{ds_id}' x_obs has {x_obs.size} points but time axis has {t_values.size}."
            )
        if not np.all(np.isfinite(x_obs)):
            raise ValueError(f"Dataset '{ds_id}' x_obs contains non-finite values for X='{x_name}'.")
        mode_raw = ds_norm.get("x_mapping_mode")
        x_mode = normalize_x_mapping_mode(mode_raw)
        if x_mode not in allowed_x_modes:
            raise ValueError(
                f"Dataset '{ds_id}' has invalid x_mapping_mode '{mode_raw}'. "
                "Expected auto, monotone, or time_guided."
            )

    return FitDatasetSpec(
        dataset_id=ds_id,
        t_exp=t_values,
        species_list=species_list,
        y_matrix=y_matrix,
        point_count=int(y_matrix.size),
        x_name=x_name,
        x_obs=x_obs,
        x_mode=x_mode,
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


def normalize_dataset_species_and_y(
    *,
    dataset_id: str,
    t_values: np.ndarray,
    species: object,
    y: object,
) -> Tuple[List[str], np.ndarray]:
    ds_id = str(dataset_id or "").strip() or "<dataset>"
    t_axis = np.asarray(t_values, dtype=float).reshape(-1)
    if t_axis.size == 0:
        raise ValueError(f"Dataset '{ds_id}' has no time points.")

    if isinstance(species, str):
        species_list = [str(species)]
        y_flat = np.asarray(y, dtype=float).reshape(-1)
        if y_flat.size != t_axis.size:
            raise ValueError(
                f"Dataset '{ds_id}' species '{species}' has {y_flat.size} points but time axis has {t_axis.size}."
            )
        y_matrix = y_flat.reshape(1, -1)
        return species_list, y_matrix

    if isinstance(species, (list, tuple)):
        species_list = [str(name) for name in species]
        y_matrix = np.asarray(y, dtype=float)
        if y_matrix.ndim != 2:
            raise ValueError(f"Dataset '{ds_id}' uses multiple species but 'y' is not a 2D array.")
        if y_matrix.shape[0] != len(species_list):
            raise ValueError(
                f"Dataset '{ds_id}' has {y_matrix.shape[0]} rows but {len(species_list)} species entries."
            )
        if y_matrix.shape[1] != t_axis.size:
            raise ValueError(
                f"Dataset '{ds_id}' has {y_matrix.shape[1]} points per species but time axis has {t_axis.size}."
            )
        return species_list, y_matrix

    raise ValueError(f"Dataset '{ds_id}' has unsupported 'species' type: {type(species)!r}")


def normalize_dataset_target_weights(
    *,
    dataset_id: str,
    selected_targets: Sequence[str],
    target_weights: Optional[Mapping[str, object]],
) -> Dict[str, float]:
    ds_id = str(dataset_id or "").strip() or "<dataset>"
    selection = [str(name) for name in (selected_targets or []) if str(name).strip()]
    raw_weights = dict(target_weights or {}) if isinstance(target_weights, Mapping) else {}
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
    t: np.ndarray,
    species_data: Mapping[str, np.ndarray],
    selected_species: Sequence[str],
    target_weights: Optional[Mapping[str, object]] = None,
    x_name: str = "t",
    x_obs: Optional[np.ndarray] = None,
    x_mapping_mode: str = "auto",
) -> FitDatasetPayloadResult:
    ds_id = str(dataset_id or "").strip()
    if not ds_id:
        return FitDatasetPayloadResult.invalid("Dataset id is missing.")
    t_values = np.asarray(t, dtype=float).reshape(-1)
    if t_values.size == 0:
        return FitDatasetPayloadResult.invalid(f"Dataset '{ds_id}' has no time points.")
    if not isinstance(species_data, Mapping) or not species_data:
        return FitDatasetPayloadResult.invalid(f"Dataset '{ds_id}' has no observed series available.")
    selection = [str(x) for x in (selected_species or []) if str(x).strip()]
    if not selection:
        return FitDatasetPayloadResult.invalid(f"Dataset '{ds_id}' requires at least one selected series.")

    rows: List[np.ndarray] = []
    missing: List[str] = []
    bad_len: List[str] = []
    for name in selection:
        if name not in species_data:
            missing.append(name)
            continue
        series = np.asarray(species_data[name], dtype=float).reshape(-1)
        if series.size != t_values.size:
            bad_len.append(name)
            continue
        rows.append(series)

    if missing:
        return FitDatasetPayloadResult.invalid(f"Dataset '{ds_id}' is missing series: {', '.join(missing)}.")
    if bad_len:
        return FitDatasetPayloadResult.invalid(
            f"Dataset '{ds_id}' has series with length mismatch vs time axis: {', '.join(bad_len)}."
        )
    if not rows:
        return FitDatasetPayloadResult.invalid(f"Dataset '{ds_id}' requires at least one valid selected series.")

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
        "t": t_values,
        "y": np.vstack(rows),
        "species": list(selection),
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

        if x_obs is None:
            x_obs_arr = np.asarray([], dtype=float)
        else:
            try:
                x_obs_arr = np.asarray(x_obs, dtype=float).reshape(-1)
            except Exception:
                return FitDatasetPayloadResult.invalid(
                    f"Dataset '{ds_id}' has invalid x_obs for X='{x_name_norm}' (could not convert sampled values)."
                )
        if x_obs_arr.size != t_values.size:
            return FitDatasetPayloadResult.invalid(
                f"Dataset '{ds_id}' is missing x_obs for X='{x_name_norm}' (must match sampled time grid)."
            )
        if not np.all(np.isfinite(x_obs_arr)):
            return FitDatasetPayloadResult.invalid(
                f"Dataset '{ds_id}' x_obs contains non-finite values for X='{x_name_norm}'."
            )
        dataset_payload["x_obs"] = x_obs_arr

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
