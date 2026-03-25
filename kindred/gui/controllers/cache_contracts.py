from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Mapping, NotRequired, Optional, TypedDict, cast

import numpy as np

from kindred.core.simulation_identity import SimulationIdentity, coerce_simulation_identity


class BatchCacheEntryV1(TypedDict):
    version: int
    t: np.ndarray
    series: Dict[str, np.ndarray]
    algebra_scalars: Dict[str, float]
    mechanism: Any
    mechanism_text: str
    simulation_identity: Dict[str, Any]
    solver_config: Dict[str, Any]
    preview_batch_cache_token: str
    fallback_occurred: bool
    fallback_message: Any


class PlotOverlayEntryV1(TypedDict):
    label: str
    t: np.ndarray
    series: Dict[str, np.ndarray]
    set_id: NotRequired[str]
    curve_role: NotRequired[str]


@dataclass(frozen=True, slots=True)
class BatchCacheEntryReadResult:
    state: Literal["valid", "missing", "invalid"]
    entry: Optional[BatchCacheEntryV1] = None


def _coerce_1d_float_array(values: object) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)


def _try_float(value: object) -> Optional[float]:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _coerce_series_map(series: object) -> Dict[str, np.ndarray]:
    if not isinstance(series, Mapping):
        raise TypeError("series must be a mapping")
    out: Dict[str, np.ndarray] = {}
    for k, v in series.items():
        key = str(k)
        out[key] = _coerce_1d_float_array(v)
    return out


def build_batch_cache_entry(
    *,
    t: object,
    series: Mapping[str, object],
    algebra_scalars: Optional[Mapping[str, object]] = None,
    mechanism: Any = None,
    mechanism_text: str = "",
    simulation_identity: Mapping[str, Any] | SimulationIdentity | None = None,
    solver_config: Optional[Mapping[str, Any]] = None,
    preview_batch_cache_token: Optional[str] = None,
    fallback_occurred: bool = False,
    fallback_message: Any = None,
) -> BatchCacheEntryV1:
    scalars: Dict[str, float] = {}
    if isinstance(algebra_scalars, Mapping):
        for k, v in algebra_scalars.items():
            scalar_value = _try_float(v)
            if scalar_value is None:
                continue
            scalars[str(k)] = float(scalar_value)

    identity = coerce_simulation_identity(simulation_identity)

    return {
        "version": 1,
        "t": _coerce_1d_float_array(t),
        "series": _coerce_series_map(series),
        "algebra_scalars": scalars,
        "mechanism": mechanism,
        "mechanism_text": str(mechanism_text or ""),
        "simulation_identity": identity.to_payload() if identity is not None else {},
        "solver_config": dict(solver_config or {}),
        "preview_batch_cache_token": str(preview_batch_cache_token or ""),
        "fallback_occurred": bool(fallback_occurred),
        "fallback_message": fallback_message,
    }


def read_batch_cache_entry(payload: object) -> BatchCacheEntryReadResult:
    if payload is None:
        return BatchCacheEntryReadResult("missing")
    if not isinstance(payload, Mapping):
        return BatchCacheEntryReadResult("invalid")
    t = payload.get("t")
    series = payload.get("series")
    if t is None or series is None:
        return BatchCacheEntryReadResult("invalid")
    try:
        coerced = build_batch_cache_entry(
            t=t,
            series=cast(Mapping[str, object], series),
            algebra_scalars=cast(Optional[Mapping[str, object]], payload.get("algebra_scalars")),
            mechanism=payload.get("mechanism"),
            mechanism_text=str(payload.get("mechanism_text") or ""),
            simulation_identity=cast(Optional[Mapping[str, Any]], payload.get("simulation_identity")),
            solver_config=cast(Optional[Mapping[str, Any]], payload.get("solver_config")),
            preview_batch_cache_token=cast(Optional[str], payload.get("preview_batch_cache_token")),
            fallback_occurred=bool(payload.get("fallback_occurred")),
            fallback_message=payload.get("fallback_message"),
        )
    except Exception:
        return BatchCacheEntryReadResult("invalid")
    return BatchCacheEntryReadResult("valid", entry=coerced)


def coerce_batch_cache_entry(payload: object) -> Optional[BatchCacheEntryV1]:
    return read_batch_cache_entry(payload).entry


def build_overlay_entry(
    *,
    label: str,
    entry: BatchCacheEntryV1,
    set_id: str | None = None,
    curve_role: str | None = None,
) -> PlotOverlayEntryV1:
    overlay: PlotOverlayEntryV1 = {
        "label": str(label or ""),
        "t": np.asarray(entry["t"], dtype=float).reshape(-1),
        "series": {str(k): np.asarray(v, dtype=float).reshape(-1) for k, v in (entry.get("series") or {}).items()},
    }
    if set_id is not None and str(set_id):
        overlay["set_id"] = str(set_id)
    if curve_role is not None and str(curve_role):
        overlay["curve_role"] = str(curve_role)
    return overlay
