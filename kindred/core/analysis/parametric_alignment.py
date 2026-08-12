"""Strict, discontinuity-safe parametric-X alignment for Fit."""

from __future__ import annotations

from collections.abc import Sequence
import numpy as np

from kindred.core.exceptions import FitCandidateDomainError, FitSimulationError

__all__ = [
    "CandidateParametricAlignmentError",
    "DeterministicParametricResolutionError",
    "align_y_on_x_obs",
    "align_y_on_x_obs_time_guided",
]


class CandidateParametricAlignmentError(FitCandidateDomainError):
    """An admitted candidate cannot map the captured parametric-X rows."""

    def __init__(self, message: str) -> None:
        super().__init__(message, stage="parametric_x_alignment")


class DeterministicParametricResolutionError(CandidateParametricAlignmentError):
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
    t_sim: np.ndarray,
    x_model: object,
    y_model: object,
    segment_ids: np.ndarray,
    segment_keys: Sequence[str],
    discontinuity_ids: np.ndarray,
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
    np.ndarray,
]:
    dataset = str(dataset_label or "dataset")
    observed_t = _vector("t_obs", t_obs)
    observed_x = _vector("x_obs", x_obs)
    if observed_t.size == 0 or observed_t.size != observed_x.size:
        raise FitSimulationError(
            f"Dataset {dataset!r}: observed time and X rows must align."
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
    discontinuities = np.asarray(discontinuity_ids, dtype=np.int64).reshape(-1)
    if discontinuities.size != simulated_t.size or np.any(discontinuities < 0):
        raise FitSimulationError(
            f"Dataset {dataset!r}: discontinuity identity is misaligned."
        )
    if simulated_t.size < 2:
        raise FitSimulationError(
            f"Dataset {dataset!r}: parametric alignment requires two model rows."
        )
    same_segment = (
        (segments[1:] == segments[:-1])
        & (discontinuities[1:] == discontinuities[:-1])
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
        discontinuities,
    )


_FRACTION_ENDPOINT_TOLERANCE = 16.0 * np.finfo(float).eps


def _selected_point(
    *,
    observation_x: float,
    fraction: float,
    left: int,
    right: int,
    simulated_x: np.ndarray,
    simulated_y: np.ndarray,
    segments: np.ndarray,
    keys: tuple[str, ...],
    discontinuities: np.ndarray,
) -> tuple[float, float, int, str, int, str]:
    if abs(float(fraction)) <= _FRACTION_ENDPOINT_TOLERANCE:
        chosen = int(left)
    elif abs(float(fraction) - 1.0) <= _FRACTION_ENDPOINT_TOLERANCE:
        chosen = int(right)
    else:
        if (
            segments[left] != segments[right]
            or keys[left] != keys[right]
            or discontinuities[left] != discontinuities[right]
        ):
            raise FitSimulationError(
                "Parametric interpolation cannot cross a segment or discontinuity."
            )
        y_value = float(
            simulated_y[left]
            + float(fraction) * (simulated_y[right] - simulated_y[left])
        )
        return (
            float(observation_x),
            y_value,
            int(segments[left]),
            keys[left],
            int(discontinuities[left]),
            "interpolated",
        )
    return (
        float(simulated_x[chosen]),
        float(simulated_y[chosen]),
        int(segments[chosen]),
        keys[chosen],
        int(discontinuities[chosen]),
        "direct",
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
    t_sim: np.ndarray,
    x_model: np.ndarray,
    y_model: np.ndarray,
    segment_ids: np.ndarray,
    segment_keys: Sequence[str],
    discontinuity_ids: np.ndarray,
    dataset_label: str,
    x_name: str,
    y_name: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[str, ...],
    np.ndarray,
    tuple[str, ...],
]:
    """Invert X independently inside each uniquely eligible segment.

    Observation time is deliberately not used to disambiguate Monotone mode.
    Each observed X value must belong to exactly one uninterrupted model
    segment, and that segment's X trajectory must be strictly monotone.
    """

    (
        obs_t,
        obs_x,
        sim_t,
        sim_x,
        sim_y,
        segments,
        keys,
        discontinuities,
    ) = _inputs(
        t_obs=t_obs,
        x_obs=x_obs,
        t_sim=t_sim,
        x_model=x_model,
        y_model=y_model,
        segment_ids=segment_ids,
        segment_keys=segment_keys,
        discontinuity_ids=discontinuity_ids,
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
        tuple[str, int, np.ndarray, np.ndarray, np.ndarray, bool]
    ] = []
    for segment_key, discontinuity_id in dict.fromkeys(
        zip(keys, discontinuities, strict=True)
    ):
        mask = np.fromiter(
            (
                key == segment_key and int(discontinuity) == int(discontinuity_id)
                for key, discontinuity in zip(
                    keys,
                    discontinuities,
                    strict=True,
                )
            ),
            dtype=bool,
            count=len(keys),
        )
        x_values = sim_x[mask]
        y_values = sim_y[mask]
        indices = np.flatnonzero(mask)
        if x_values.size < 2:
            continue
        differences = np.diff(x_values)
        increasing = bool(np.all(differences > x_tolerance))
        decreasing = bool(np.all(differences < -x_tolerance))
        if decreasing:
            x_values = x_values[::-1]
            y_values = y_values[::-1]
            indices = indices[::-1]
        segment_models.append(
            (
                segment_key,
                int(discontinuity_id),
                x_values,
                y_values,
                indices,
                increasing or decreasing,
            )
        )
    if not segment_models:
        raise DeterministicParametricResolutionError(
            f"Dataset {dataset_label!r}: no uninterrupted model segment has "
            "enough points for Monotone parametric-X mapping."
        )

    selected = []
    for observation_x in obs_x:
        eligible = [
            item
            for item in segment_models
            if (
                float(observation_x) >= float(np.min(item[2])) - x_tolerance
                and float(observation_x) <= float(np.max(item[2])) + x_tolerance
            )
        ]
        usable: list[tuple[float, float, int, str, int, str]] = []
        for (
            segment_key,
            _discontinuity_id,
            x_values,
            _y_values,
            indices,
            monotone,
        ) in eligible:
            if not monotone:
                raise DeterministicParametricResolutionError(
                    f"Dataset {dataset_label!r}: X(t) for {x_name!r} is not "
                    "strictly monotone in uninterrupted segment "
                    f"{segment_key!r}."
                )
            right_offset = int(
                np.searchsorted(x_values, float(observation_x), side="left")
            )
            if right_offset <= 0:
                left_offset, right_offset = 0, 1
            elif right_offset >= x_values.size:
                left_offset, right_offset = x_values.size - 2, x_values.size - 1
            else:
                left_offset = right_offset - 1
            left = int(indices[left_offset])
            right = int(indices[right_offset])
            width = float(sim_x[right] - sim_x[left])
            if width == 0.0:
                raise DeterministicParametricResolutionError(
                    f"Dataset {dataset_label!r}: X(t) is not invertible in "
                    f"segment {segment_key!r}."
                )
            fraction = float((float(observation_x) - sim_x[left]) / width)
            usable.append(
                _selected_point(
                    observation_x=float(observation_x),
                    fraction=fraction,
                    left=left,
                    right=right,
                    simulated_x=sim_x,
                    simulated_y=sim_y,
                    segments=segments,
                    keys=keys,
                    discontinuities=discontinuities,
                )
            )
        if len(usable) != 1:
            raise DeterministicParametricResolutionError(
                f"Dataset {dataset_label!r}: observed X={float(observation_x):.6g} "
                f"for {x_name!r} belongs to {len(usable)} selectable uninterrupted "
                "model segments; Monotone mode requires exactly one."
            )
        selected.append(usable[0])
    return (
        np.asarray([item[0] for item in selected], dtype=float),
        np.asarray([item[1] for item in selected], dtype=float),
        np.asarray([item[2] for item in selected], dtype=np.int64),
        tuple(item[3] for item in selected),
        np.asarray([item[4] for item in selected], dtype=np.int64),
        tuple(item[5] for item in selected),
    )


def align_y_on_x_obs_time_guided(
    *,
    t_obs: np.ndarray,
    x_obs: np.ndarray,
    t_sim: np.ndarray,
    x_model: np.ndarray,
    y_model: np.ndarray,
    segment_ids: np.ndarray,
    segment_keys: Sequence[str],
    discontinuity_ids: np.ndarray,
    dataset_label: str,
    x_name: str,
    y_name: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[str, ...],
    np.ndarray,
    tuple[str, ...],
]:
    """Choose the exact X crossing closest to each observation time."""

    (
        obs_t,
        obs_x,
        sim_t,
        sim_x,
        sim_y,
        segments,
        keys,
        discontinuities,
    ) = _inputs(
        t_obs=t_obs,
        x_obs=x_obs,
        t_sim=t_sim,
        x_model=x_model,
        y_model=y_model,
        segment_ids=segment_ids,
        segment_keys=segment_keys,
        discontinuity_ids=discontinuity_ids,
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
    overlapping &= discontinuities[1:] == discontinuities[:-1]
    x_tolerance = 1e-12 * max(
        1.0,
        float(np.max(np.abs(sim_x))),
        float(np.max(np.abs(obs_x))),
    )
    selected = []
    for observation_time, observation_x in zip(obs_t, obs_x, strict=True):
        candidates: list[tuple[float, int, int, float]] = []
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
            candidates.append((t_star, left, right, float(fraction)))
        if not candidates:
            raise CandidateParametricAlignmentError(
                f"Dataset {dataset_label!r}: no uninterrupted crossing exists "
                f"for X={x_name!r} at {float(observation_x):.6g}."
            )
        chosen_t, left, right, fraction = min(
            candidates,
            key=lambda item: (abs(item[0] - float(observation_time)), item[0]),
        )
        point = _selected_point(
            observation_x=float(observation_x),
            fraction=fraction,
            left=left,
            right=right,
            simulated_x=sim_x,
            simulated_y=sim_y,
            segments=segments,
            keys=keys,
            discontinuities=discontinuities,
        )
        if not np.isfinite(chosen_t) or not np.isfinite(point[1]):
            raise CandidateParametricAlignmentError(
                f"Dataset {dataset_label!r}: parametric-X alignment produced "
                "a non-finite value."
            )
        selected.append(point)
    return (
        np.asarray([item[0] for item in selected], dtype=float),
        np.asarray([item[1] for item in selected], dtype=float),
        np.asarray([item[2] for item in selected], dtype=np.int64),
        tuple(item[3] for item in selected),
        np.asarray([item[4] for item in selected], dtype=np.int64),
        tuple(item[5] for item in selected),
    )
