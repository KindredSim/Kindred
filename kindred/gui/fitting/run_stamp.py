from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
from kindred.core.analysis.dataset_parameter_overrides import (
    FitDatasetParameterOverrides,
    coerce_fit_dataset_parameter_overrides,
)
from kindred.core.simulation_preparation import coerce_prepared_simulation_metadata

logger = logging.getLogger(__name__)

try:
    from kindred import __version__ as KINDRED_VERSION
except Exception:  # pragma: no cover - defensive fallback
    KINDRED_VERSION = ""

__all__ = ["build_global_fit_run_stamp", "hash_global_fit_run_stamp"]

_REQUIRED_PREPARED_SIMULATION_KEYS = frozenset(
    {
        "mechanism_text_sha256",
        "mechanism_text_len",
        "param_names",
        "t_end",
        "num_points",
        "temperature_K",
        "solver_requested",
        "solver_normalized",
        "rtol",
        "atol",
        "use_sparse_jacobian",
        "wegscheider_cyclicity_enabled",
        "initial_prefix",
    }
)


def _sha256_hex(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _float_to_canonical_str(value: object) -> str:
    try:
        val = float(value)  # type: ignore[arg-type]
    except Exception:
        return str(value)
    if not np.isfinite(val):
        return str(val)
    # 12 significant digits keeps stamps compact and deterministic for common UI-edited floats.
    return f"{val:.12g}"


def _normalize_included_dataset_ids(included_ids: Sequence[str]) -> list[str]:
    return sorted({str(x).strip() for x in (included_ids or []) if str(x).strip()})


def _normalize_applied_fit_targets(
    applied_fit_targets: Dict[str, Sequence[str]],
    *,
    included_ids: set[str],
) -> dict[str, list[str]]:
    applied_norm: dict[str, list[str]] = {}
    for ds_id, targets in (applied_fit_targets or {}).items():
        key = str(ds_id).strip()
        if not key or key not in included_ids:
            continue
        applied_norm[key] = sorted({str(x).strip() for x in (targets or []) if str(x).strip()})
    return applied_norm


def _normalize_applied_target_weights(
    applied_target_weights: Optional[Dict[str, Dict[str, float]]],
    *,
    applied_fit_targets: Dict[str, List[str]],
    included_ids: set[str],
) -> dict[str, dict[str, str]]:
    raw = dict(applied_target_weights or {}) if isinstance(applied_target_weights, dict) else {}
    normalized: dict[str, dict[str, str]] = {}
    for ds_id, targets in applied_fit_targets.items():
        key = str(ds_id).strip()
        if not key or key not in included_ids:
            continue
        weight_map = raw.get(key)
        weight_map = dict(weight_map) if isinstance(weight_map, dict) else {}
        normalized[key] = {}
        for target_name in sorted({str(x).strip() for x in (targets or []) if str(x).strip()}):
            try:
                value = float(weight_map.get(target_name, 1.0))
            except Exception:
                value = 1.0
            if not np.isfinite(value) or value <= 0.0:
                value = 1.0
            normalized[key][target_name] = _float_to_canonical_str(value)
    return normalized


def _index_dataset_rows_by_id(
    dataset_rows: Sequence[Dict[str, Any]],
    *,
    included_ids: set[str],
) -> dict[str, dict[str, Any]]:
    row_by_id: dict[str, dict[str, Any]] = {}
    for row in (dataset_rows or []):
        ds_id = str((row or {}).get("id") or "").strip()
        if not ds_id or ds_id not in included_ids:
            continue
        row_by_id[ds_id] = dict(row or {})
    return row_by_id


def _build_global_fit_datasets_block(
    *,
    ordered_ids: Sequence[str],
    row_by_id: Dict[str, Dict[str, Any]],
    applied_targets: Dict[str, List[str]],
    applied_target_weights: Dict[str, Dict[str, str]],
) -> list[dict[str, Any]]:
    datasets_block: list[dict[str, Any]] = []
    for ds_id in ordered_ids:
        row = row_by_id.get(ds_id, {})
        datasets_block.append(
            {
                "id": ds_id,
                "label": str(row.get("label") or ds_id),
                "weight_input": _float_to_canonical_str(row.get("weight", 1.0)),
                "fit_targets_applied": list(applied_targets.get(ds_id, [])),
                "target_weights_applied": dict(applied_target_weights.get(ds_id, {})),
            }
        )
    return datasets_block


def _normalize_global_fit_weights_used(weights_used: Optional[Dict[str, float]]) -> Optional[dict[str, str]]:
    if weights_used is None:
        return None
    return {str(k): _float_to_canonical_str(v) for k, v in sorted(weights_used.items()) if str(k).strip()}


def _normalize_fit_config_dict(fit_config: Dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    config = dict(fit_config or {})
    parameters = dict(config.get("parameters") or {}) if isinstance(config.get("parameters"), dict) else {}
    config["parameters"] = dict(parameters)
    fixed_params = dict(config.get("fixed_params") or {}) if isinstance(config.get("fixed_params"), dict) else {}
    config["fixed_params"] = dict(fixed_params)
    bounds = dict(config.get("bounds") or {}) if isinstance(config.get("bounds"), dict) else {}
    config["bounds"] = dict(bounds)
    log10_params = dict(config.get("log10_params") or {}) if isinstance(config.get("log10_params"), dict) else {}
    config["log10_params"] = dict(log10_params)
    config["method"] = str(config.get("method") or "")
    try:
        config["max_nfev"] = int(config.get("max_nfev") or 0)
    except Exception:
        config["max_nfev"] = 0
    config["seed"] = config.get("seed")
    try:
        config["parallel_starts"] = int(config.get("parallel_starts") or 0)
    except Exception:
        config["parallel_starts"] = 0
    return config, {"parameters": parameters, "fixed_params": fixed_params, "bounds": bounds, "log10_params": log10_params}


def _normalize_bounds_block(bounds: Dict[str, Any]) -> dict[str, list[str]]:
    bounds_norm: dict[str, list[str]] = {}
    invalid_bound_keys: list[str] = []
    for name, pair in sorted((bounds or {}).items()):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            invalid_bound_keys.append(str(name))
            continue
        lo = pair[0]
        hi = pair[1]
        bounds_norm[str(name)] = [_float_to_canonical_str(lo), _float_to_canonical_str(hi)]
    if invalid_bound_keys:
        invalid_bound_keys_sorted = ", ".join(sorted(set(invalid_bound_keys)))
        logger.debug("Ignoring invalid bounds entries when stamping a global fit run: %s", invalid_bound_keys_sorted)
    return bounds_norm


def _normalize_shared_params_block(
    *,
    parameters: Dict[str, Any],
    fixed_params: Dict[str, Any],
    log10_params: Dict[str, Any],
    bounds_norm: Dict[str, List[str]],
) -> dict[str, Any]:
    shared_fit_initial = {
        str(name): _float_to_canonical_str(value)
        for name, value in sorted((parameters or {}).items())
        if str(name).strip()
    }
    shared_fixed = {
        str(name): _float_to_canonical_str(value)
        for name, value in sorted((fixed_params or {}).items())
        if str(name).strip()
    }
    shared_log10 = {
        str(name): bool((log10_params or {}).get(name))
        for name in sorted({str(k) for k in (log10_params or {}).keys() if str(k).strip()})
    }
    return {
        "fit_initial": shared_fit_initial,
        "fixed": shared_fixed,
        "log10": shared_log10,
        "bounds": bounds_norm,
    }


def _normalize_dataset_params_block(
    *,
    dataset_overrides: Sequence[FitDatasetParameterOverrides],
) -> dict[str, dict[str, str]]:
    dataset_params_norm: dict[str, dict[str, str]] = {}
    for override in dataset_overrides:
        ds_id = str(override.dataset_id or "").strip()
        if not ds_id or not override.fixed_params:
            continue
        dataset_params_norm[ds_id] = {
            str(k): _float_to_canonical_str(v)
            for k, v in sorted(override.fixed_params.items())
            if str(k).strip()
        }
    return dataset_params_norm


def _normalize_dataset_variable_params_block(
    *,
    dataset_overrides: Sequence[FitDatasetParameterOverrides],
) -> dict[str, dict[str, dict[str, Any]]]:
    dataset_variable_norm: dict[str, dict[str, dict[str, Any]]] = {}
    for override in dataset_overrides:
        ds_id = str(override.dataset_id or "").strip()
        if not ds_id or not override.variable_params:
            continue
        normalized_specs: dict[str, dict[str, Any]] = {}
        for param_name, spec in sorted(override.variable_params.items()):
            if not str(param_name).strip():
                continue
            normalized_specs[str(param_name)] = {
                "initial": _float_to_canonical_str(spec.initial),
                "min": _float_to_canonical_str(spec.minimum),
                "max": _float_to_canonical_str(spec.maximum),
                "log10": bool(spec.log10),
            }
        if normalized_specs:
            dataset_variable_norm[ds_id] = normalized_specs
    return dataset_variable_norm


def _normalize_prepared_simulation_block(prepared_simulation: Optional[object]) -> Optional[dict[str, Any]]:
    if isinstance(prepared_simulation, Mapping):
        missing = sorted(_REQUIRED_PREPARED_SIMULATION_KEYS.difference(prepared_simulation.keys()))
        if missing:
            raise ValueError(
                "Incomplete prepared simulation metadata: missing "
                + ", ".join(missing)
            )
    prepared_meta = coerce_prepared_simulation_metadata(prepared_simulation)
    if prepared_meta is None:
        return None
    serialized = prepared_meta.to_serializable_dict()
    return {
        "mechanism_text_sha256": str(serialized.get("mechanism_text_sha256") or ""),
        "mechanism_text_len": int(serialized.get("mechanism_text_len") or 0),
        "param_names": sorted({str(x) for x in (serialized.get("param_names") or []) if str(x).strip()}),
        "t_end": _float_to_canonical_str(serialized.get("t_end")),
        "num_points": int(serialized.get("num_points") or 0),
        "temperature_K": _float_to_canonical_str(serialized.get("temperature_K")),
        "solver_requested": str(serialized.get("solver_requested") or ""),
        "solver_normalized": str(serialized.get("solver_normalized") or ""),
        "rtol": _float_to_canonical_str(serialized.get("rtol")),
        "atol": _float_to_canonical_str(serialized.get("atol")),
        "use_sparse_jacobian": bool(serialized["use_sparse_jacobian"]),
        "wegscheider_cyclicity_enabled": bool(serialized["wegscheider_cyclicity_enabled"]),
        "initial_prefix": str(serialized.get("initial_prefix") or ""),
    }


def build_global_fit_run_stamp(
    *,
    dataset_rows: Sequence[Dict[str, Any]],
    included_ids: Sequence[str],
    applied_fit_targets: Dict[str, Sequence[str]],
    weights_used: Optional[Dict[str, float]],
    weight_mode: str,
    applied_target_weights: Optional[Dict[str, Dict[str, float]]] = None,
    fit_config: Dict[str, Any],
    mechanism_text: str,
    reactions_text: str,
    prepared_simulation: Optional[object] = None,
    dataset_overrides: Optional[Sequence[object]] = None,
    dataset_params: Optional[Dict[str, Dict[str, float]]] = None,
    dataset_variable_params: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    ordered_ids = _normalize_included_dataset_ids(included_ids)
    included = set(ordered_ids)
    applied_norm = _normalize_applied_fit_targets(applied_fit_targets, included_ids=included)
    applied_target_weights_norm = _normalize_applied_target_weights(
        applied_target_weights,
        applied_fit_targets=applied_norm,
        included_ids=included,
    )
    row_by_id = _index_dataset_rows_by_id(dataset_rows, included_ids=included)
    datasets_block = _build_global_fit_datasets_block(
        ordered_ids=ordered_ids,
        row_by_id=row_by_id,
        applied_targets=applied_norm,
        applied_target_weights=applied_target_weights_norm,
    )
    weights_block = _normalize_global_fit_weights_used(weights_used)

    config, extracted = _normalize_fit_config_dict(fit_config)
    parameters = extracted["parameters"]
    fixed_params = extracted["fixed_params"]
    bounds = extracted["bounds"]
    log10_params = extracted["log10_params"]

    bounds_norm = _normalize_bounds_block(bounds)
    shared_params_block = _normalize_shared_params_block(
        parameters=parameters,
        fixed_params=fixed_params,
        log10_params=log10_params,
        bounds_norm=bounds_norm,
    )
    overrides = coerce_fit_dataset_parameter_overrides(
        dataset_ids=ordered_ids,
        dataset_overrides=dataset_overrides,
        dataset_params=dataset_params,
        dataset_variable_params=dataset_variable_params,
    )
    dataset_params_norm = _normalize_dataset_params_block(dataset_overrides=overrides)
    dataset_variable_norm = _normalize_dataset_variable_params_block(
        dataset_overrides=overrides,
    )
    prepared_block = _normalize_prepared_simulation_block(prepared_simulation)

    return {
        "version": 3,
        "mode": "global",
        "kindred_version": str(KINDRED_VERSION or ""),
        "datasets": datasets_block,
        "fit_targets_applied": {k: applied_norm[k] for k in ordered_ids if k in applied_norm},
        "target_weights_applied": {k: applied_target_weights_norm[k] for k in ordered_ids if k in applied_target_weights_norm},
        "weight_mode": str(weight_mode or ""),
        "weights_used": weights_block,
        "algorithm": {
            "method": str(config.get("method") or ""),
            "max_nfev": int(config.get("max_nfev") or 0),
            "seed": (int(config["seed"]) if config.get("seed") is not None else None),
            "parallel_starts": int(config.get("parallel_starts") or 0),
        },
        "parameters": {
            "fit": sorted({str(k) for k in parameters.keys() if str(k).strip()}),
            "fixed": sorted({str(k) for k in fixed_params.keys() if str(k).strip()}),
            "bounds": bounds_norm,
        },
        "shared_params": shared_params_block,
        "dataset_params": dataset_params_norm,
        "dataset_variable_params": dataset_variable_norm,
        "prepared_simulation": prepared_block,
        "mechanism_sha256": _sha256_hex(mechanism_text),
        "reactions_sha256": _sha256_hex(reactions_text),
    }


def hash_global_fit_run_stamp(stamp: Dict[str, Any]) -> str:
    canonical = json.dumps(stamp or {}, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return _sha256_hex(canonical)
