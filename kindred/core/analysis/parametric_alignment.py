"""Strict, discontinuity-safe parametric-X alignment for Fit."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import numpy as np

from kindred.core.exceptions import FitSimulationError

__all__ = [
    "DeterministicParametricResolutionError",
    "align_y_on_x_obs",
    "align_y_on_x_obs_time_guided",
]


class DeterministicParametricResolutionError(FitSimulationError):
    """Monotone mapping is validly formed but cannot resolve one exact branch."""


def _vector(name: str, values: object) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise FitSimulationError(f"Invalid {name} array: {exc}") from exc
    if not np.all(np.isfinite(array)):
        raise FitSimulationError(f"{name} contains non-finite values.")
    return array


def _inputs(
    *,
    t_obs: object,
    x_obs: object,
    t_sim: Optional[np.ndarray],
    x_model: object,
    y_model: object,
    segment_ids: Optional[np.ndarray],
    segment_keys: Sequence[str],
    dataset_label: str,
    x_name: str,
    y_name: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[str, ...],
]:
    dataset = str(dataset_label or "dataset")
    observed_t = _vector("t_obs", t_obs)
    observed_x = _vector("x_obs", x_obs)
    if observed_t.size == 0 or observed_t.size != observed_x.size:
        raise FitSimulationError(
            f"Dataset {dataset!r}: observed time and X rows must align."
        )
    if t_sim is None:
        raise FitSimulationError(
            f"Dataset {dataset!r}: simulation time is required for X={x_name!r}."
        )
    simulated_t = _vector("t_sim", t_sim)
    simulated_x = _vector("x_model", x_model)
    simulated_y = _vector("y_model", y_model)
    if (
        simulated_t.size < 2
        or simulated_x.size != simulated_t.size
        or simulated_y.size != simulated_t.size
    ):
        raise FitSimulationError(
            f"Dataset {dataset!r}: simulated X={x_name!r}, Y={y_name!r}, and "
            "time must have the same length of at least two."
        )
    if segment_ids is None:
        segments = np.zeros(simulated_t.size, dtype=np.int64)
    else:
        segments = np.asarray(segment_ids, dtype=np.int64).reshape(-1)
        if segments.size != simulated_t.size:
            raise FitSimulationError(
                f"Dataset {dataset!r}: discontinuity segment identity is misaligned."
            )
    if np.any(segments < 0) or np.any(np.diff(segments) < 0):
        raise FitSimulationError(
            f"Dataset {dataset!r}: discontinuity segments must be ordered "
            "and nonnegative."
        )
    keys = tuple(str(value or "").strip() for value in segment_keys)
    if len(keys) != simulated_t.size or any(not value for value in keys):
        raise FitSimulationError(
            f"Dataset {dataset!r}: exact simulation segment keys are "
            "missing or misaligned."
        )
    same_segment = (
        (segments[1:] == segments[:-1])
        & np.fromiter(
            (left == right for left, right in zip(keys, keys[1:])),
            dtype=bool,
            count=max(0, len(keys) - 1),
        )
    )
    if np.any(np.diff(simulated_t)[same_segment] <= 0.0):
        raise FitSimulationError(
            f"Dataset {dataset!r}: simulation time must increase inside "
            "each uninterrupted segment."
        )
    return (
        observed_t,
        observed_x,
        simulated_t,
        simulated_x,
        simulated_y,
        segments,
        keys,
    )


def _window_bounds(observed_t: np.ndarray) -> tuple[float, float, float]:
    low = float(np.min(observed_t))
    high = float(np.max(observed_t))
    scale = max(1.0, abs(low), abs(high))
    pad = 1e-12 * scale
    return low, high, pad


def _overlapping_intervals(
    simulated_t: np.ndarray,
    segments: np.ndarray,
    *,
    low: float,
    high: float,
    pad: float,
) -> np.ndarray:
    return (
        (segments[1:] == segments[:-1])
        & (simulated_t[1:] >= low - pad)
        & (simulated_t[:-1] <= high + pad)
    )


def align_y_on_x_obs(
    *,
    t_obs: np.ndarray,
    x_obs: np.ndarray,
    t_sim: Optional[np.ndarray],
    x_model: np.ndarray,
    y_model: np.ndarray,
    segment_ids: Optional[np.ndarray] = None,
    segment_keys: Sequence[str],
    dataset_label: str,
    x_name: str,
    y_name: str,
) -> np.ndarray:
    """Invert X independently inside each uniquely eligible segment.

    Observation time is deliberately not used to disambiguate Monotone mode.
    Each observed X value must belong to exactly one uninterrupted model
    segment, and that segment's X trajectory must be strictly monotone.
    """

    obs_t, obs_x, sim_t, sim_x, sim_y, segments, keys = _inputs(
        t_obs=t_obs,
        x_obs=x_obs,
        t_sim=t_sim,
        x_model=x_model,
        y_model=y_model,
        segment_ids=segment_ids,
        segment_keys=segment_keys,
        dataset_label=dataset_label,
        x_name=x_name,
        y_name=y_name,
    )
    x_tolerance = 1e-12 * max(
        1.0,
        float(np.max(np.abs(sim_x))),
        float(np.max(np.abs(obs_x))),
    )
    segment_models: list[
        tuple[str, np.ndarray, np.ndarray, bool]
    ] = []
    for segment_key in dict.fromkeys(keys):
        mask = np.fromiter(
            (value == segment_key for value in keys),
            dtype=bool,
            count=len(keys),
        )
        x_values = sim_x[mask]
        y_values = sim_y[mask]
        if x_values.size < 2:
            continue
        differences = np.diff(x_values)
        increasing = bool(np.all(differences > x_tolerance))
        decreasing = bool(np.all(differences < -x_tolerance))
        if decreasing:
            x_values = x_values[::-1]
            y_values = y_values[::-1]
        segment_models.append(
            (
                segment_key,
                x_values,
                y_values,
                increasing or decreasing,
            )
        )
    if not segment_models:
        raise DeterministicParametricResolutionError(
            f"Dataset {dataset_label!r}: no uninterrupted model segment has "
            "enough points for Monotone parametric-X mapping."
        )

    aligned = np.empty(obs_x.size, dtype=float)
    for index, observation_x in enumerate(obs_x):
        eligible = [
            item
            for item in segment_models
            if (
                float(observation_x) >= float(np.min(item[1])) - x_tolerance
                and float(observation_x) <= float(np.max(item[1])) + x_tolerance
            )
        ]
        if len(eligible) != 1:
            raise DeterministicParametricResolutionError(
                f"Dataset {dataset_label!r}: observed X={float(observation_x):.6g} "
                f"for {x_name!r} belongs to {len(eligible)} uninterrupted "
                "model segments; Monotone mode requires exactly one."
            )
        segment_key, x_values, y_values, monotone = eligible[0]
        if not monotone:
            raise DeterministicParametricResolutionError(
                f"Dataset {dataset_label!r}: X(t) for {x_name!r} is not "
                "strictly monotone in uninterrupted segment "
                f"{segment_key!r}."
            )
        aligned[index] = float(
            np.interp(float(observation_x), x_values, y_values)
        )
    return aligned


def align_y_on_x_obs_time_guided(
    *,
    t_obs: np.ndarray,
    x_obs: np.ndarray,
    t_sim: Optional[np.ndarray],
    x_model: np.ndarray,
    y_model: np.ndarray,
    segment_ids: Optional[np.ndarray] = None,
    segment_keys: Sequence[str],
    dataset_label: str,
    x_name: str,
    y_name: str,
) -> np.ndarray:
    """Choose the exact X crossing closest to each observation time."""

    obs_t, obs_x, sim_t, sim_x, sim_y, segments, keys = _inputs(
        t_obs=t_obs,
        x_obs=x_obs,
        t_sim=t_sim,
        x_model=x_model,
        y_model=y_model,
        segment_ids=segment_ids,
        segment_keys=segment_keys,
        dataset_label=dataset_label,
        x_name=x_name,
        y_name=y_name,
    )
    low, high, pad = _window_bounds(obs_t)
    overlapping = _overlapping_intervals(
        sim_t,
        segments,
        low=low,
        high=high,
        pad=pad,
    )
    overlapping &= np.fromiter(
        (left == right for left, right in zip(keys, keys[1:])),
        dtype=bool,
        count=max(0, len(keys) - 1),
    )
    x_tolerance = 1e-12 * max(
        1.0,
        float(np.max(np.abs(sim_x))),
        float(np.max(np.abs(obs_x))),
    )
    aligned = np.empty(obs_x.size, dtype=float)
    for index, (observation_time, observation_x) in enumerate(zip(obs_t, obs_x)):
        candidates: list[tuple[float, float]] = []
        for left in range(sim_t.size - 1):
            right = left + 1
            if not overlapping[left]:
                continue
            x_left = float(sim_x[left])
            x_right = float(sim_x[right])
            if (
                observation_x < min(x_left, x_right) - x_tolerance
                or observation_x > max(x_left, x_right) + x_tolerance
            ):
                continue
            delta_x = x_right - x_left
            if abs(delta_x) <= x_tolerance:
                if abs(float(observation_x) - x_left) > x_tolerance:
                    continue
                t_star = min(
                    max(float(observation_time), float(sim_t[left])),
                    float(sim_t[right]),
                )
                fraction = (
                    (t_star - float(sim_t[left]))
                    / (float(sim_t[right]) - float(sim_t[left]))
                )
            else:
                fraction = (float(observation_x) - x_left) / delta_x
                t_star = float(
                    sim_t[left]
                    + fraction * (sim_t[right] - sim_t[left])
                )
            if t_star < low - pad or t_star > high + pad:
                continue
            y_star = float(
                sim_y[left] + fraction * (sim_y[right] - sim_y[left])
            )
            candidates.append((t_star, y_star))
        if not candidates:
            raise FitSimulationError(
                f"Dataset {dataset_label!r}: no uninterrupted crossing exists "
                f"for X={x_name!r} at {float(observation_x):.6g}."
            )
        chosen_t, chosen_y = min(
            candidates,
            key=lambda item: (abs(item[0] - float(observation_time)), item[0]),
        )
        if not np.isfinite(chosen_t) or not np.isfinite(chosen_y):
            raise FitSimulationError(
                f"Dataset {dataset_label!r}: parametric-X alignment produced "
                "a non-finite value."
            )
        aligned[index] = chosen_y
    return aligned
