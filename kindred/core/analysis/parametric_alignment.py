"""Canonical discontinuity-safe parametric-X projections.

The projection is the single alignment authority shared by objective evaluation
and terminal replay.  It carries physical source-row identity all the way to
the selected model sample and never manufactures an extrapolated point.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from kindred.core.exceptions import FitCandidateDomainError, FitSimulationError

__all__ = [
    "CandidateParametricAlignmentError",
    "DeterministicParametricResolutionError",
    "ParametricXProjection",
    "align_parametric_x",
    "project_parametric_y",
]


class CandidateParametricAlignmentError(FitCandidateDomainError):
    """An admitted candidate cannot produce a scientific X projection."""

    def __init__(
        self,
        message: str,
        *,
        dataset_label: str | None = None,
        x_name: str | None = None,
        failed_observed_value: float | None = None,
        candidate_parameter_facts: Mapping[str, object] | None = None,
        value_dependencies: Sequence[str] = (),
    ) -> None:
        details: dict[str, object] = {
            "producer": "parametric_alignment",
            "stage": "parametric_x_alignment",
            "message": str(message),
        }
        if dataset_label is not None:
            details["dataset"] = str(dataset_label)
        if x_name is not None:
            details["x_identity"] = {"name": str(x_name)}
        if failed_observed_value is not None:
            details["failed_observed_value"] = float(failed_observed_value)
        if candidate_parameter_facts:
            details["candidate_parameter_facts"] = dict(candidate_parameter_facts)
        dependencies = tuple(
            sorted({str(item).strip() for item in value_dependencies if str(item).strip()})
        )
        if dependencies:
            details["value_dependencies"] = dependencies
            details["parameter_names"] = dependencies
        super().__init__(message, stage="parametric_x_alignment", details=details)


class DeterministicParametricResolutionError(CandidateParametricAlignmentError):
    """Monotone mode cannot resolve a unique physical branch."""


def _vector(name: str, values: object) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise FitSimulationError(f"Invalid {name} array: {exc}") from exc
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise FitSimulationError(f"{name} must be finite and nonempty.")
    return array


def _freeze_float(values: object) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1).copy()
    array.setflags(write=False)
    return array


def _freeze_int(values: object) -> np.ndarray:
    array = np.asarray(values, dtype=np.int64).reshape(-1).copy()
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class ParametricXProjection:
    """Immutable one-row-per-canonical-X projection."""

    source_rows: tuple[int, ...]
    observation_time: np.ndarray
    observed_x: np.ndarray
    selected_time: np.ndarray
    selected_model_x: np.ndarray
    delta_x: np.ndarray
    segment_ids: np.ndarray
    segment_keys: tuple[str, ...]
    discontinuity_ids: np.ndarray
    match_kinds: tuple[Literal["exact_crossing", "finite_projection"], ...]
    sample_kinds: tuple[Literal["direct", "interpolated"], ...]
    model_rows: tuple[int, ...]

    def __post_init__(self) -> None:
        rows = tuple(int(item) for item in self.source_rows)
        if (
            not rows
            or len(rows) != len(set(rows))
            or any(item <= 0 for item in rows)
        ):
            raise ValueError("Parametric X projection requires unique positive source rows.")
        arrays = {
            "observation_time": _freeze_float(self.observation_time),
            "observed_x": _freeze_float(self.observed_x),
            "selected_time": _freeze_float(self.selected_time),
            "selected_model_x": _freeze_float(self.selected_model_x),
            "delta_x": _freeze_float(self.delta_x),
            "segment_ids": _freeze_int(self.segment_ids),
            "discontinuity_ids": _freeze_int(self.discontinuity_ids),
        }
        size = len(rows)
        if any(value.size != size for value in arrays.values()):
            raise ValueError("Parametric X projection arrays must have one row per source row.")
        keys = tuple(str(item or "").strip() for item in self.segment_keys)
        match = tuple(str(item) for item in self.match_kinds)
        sample = tuple(str(item) for item in self.sample_kinds)
        model_rows = tuple(int(item) for item in self.model_rows)
        if (
            len(keys) != size
            or len(match) != size
            or len(sample) != size
            or len(model_rows) != size
            or any(not item for item in keys)
            or any(item not in {"exact_crossing", "finite_projection"} for item in match)
            or any(item not in {"direct", "interpolated"} for item in sample)
            or any(item < 0 for item in arrays["segment_ids"])
            or any(item < 0 for item in arrays["discontinuity_ids"])
            or any(item < 0 for item in model_rows)
        ):
            raise ValueError("Parametric X projection identity is malformed.")
        if not np.allclose(
            arrays["delta_x"],
            arrays["selected_model_x"] - arrays["observed_x"],
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError("Parametric X projection delta_x is not canonical.")
        for name, value in arrays.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "source_rows", rows)
        object.__setattr__(self, "segment_keys", keys)
        object.__setattr__(self, "match_kinds", match)
        object.__setattr__(self, "sample_kinds", sample)
        object.__setattr__(self, "model_rows", model_rows)

    @property
    def size(self) -> int:
        return len(self.source_rows)

    def subset(self, source_rows: Sequence[int]) -> "ParametricXProjection":
        """Return the same projection restricted by physical source-row order."""

        positions = {row: index for index, row in enumerate(self.source_rows)}
        try:
            indexes = tuple(positions[int(row)] for row in source_rows)
        except KeyError as exc:
            raise FitSimulationError("Y references an X source row outside the projection.") from exc
        return ParametricXProjection(
            source_rows=tuple(int(row) for row in source_rows),
            observation_time=self.observation_time[list(indexes)],
            observed_x=self.observed_x[list(indexes)],
            selected_time=self.selected_time[list(indexes)],
            selected_model_x=self.selected_model_x[list(indexes)],
            delta_x=self.delta_x[list(indexes)],
            segment_ids=self.segment_ids[list(indexes)],
            segment_keys=tuple(self.segment_keys[index] for index in indexes),
            discontinuity_ids=self.discontinuity_ids[list(indexes)],
            match_kinds=tuple(self.match_kinds[index] for index in indexes),
            sample_kinds=tuple(self.sample_kinds[index] for index in indexes),
            model_rows=tuple(self.model_rows[index] for index in indexes),
        )


def _validate_model(
    *,
    t_model: object,
    x_model: object,
    segment_ids: object,
    segment_keys: Sequence[str],
    discontinuity_ids: object,
    dataset_label: str,
    x_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...], np.ndarray]:
    t = _vector("model time", t_model)
    x = _vector(f"model X {x_name!r}", x_model)
    segments = np.asarray(segment_ids, dtype=np.int64).reshape(-1)
    discontinuities = np.asarray(discontinuity_ids, dtype=np.int64).reshape(-1)
    keys = tuple(str(item or "").strip() for item in segment_keys)
    if (
        t.size < 2
        or x.size != t.size
        or segments.size != t.size
        or discontinuities.size != t.size
        or len(keys) != t.size
        or any(not item for item in keys)
        or np.any(segments < 0)
        or np.any(discontinuities < 0)
        or np.any(np.diff(segments) < 0)
        or np.any(np.diff(discontinuities) < 0)
    ):
        raise FitSimulationError(
            f"Dataset {dataset_label!r}: model X/time/segment identity is malformed."
        )
    same = (
        (segments[1:] == segments[:-1])
        & (discontinuities[1:] == discontinuities[:-1])
        & np.fromiter(
            (left == right for left, right in zip(keys, keys[1:])),
            dtype=bool,
            count=t.size - 1,
        )
    )
    if np.any(np.diff(t)[same] <= 0.0):
        raise FitSimulationError(
            f"Dataset {dataset_label!r}: model time must increase inside each "
            "uninterrupted segment."
        )
    return t, x, segments, keys, discontinuities


def _intervals(
    t: np.ndarray,
    segments: np.ndarray,
    keys: tuple[str, ...],
    discontinuities: np.ndarray,
) -> list[tuple[int, int]]:
    valid = (
        (segments[1:] == segments[:-1])
        & (discontinuities[1:] == discontinuities[:-1])
        & np.fromiter(
            (left == right for left, right in zip(keys, keys[1:])),
            dtype=bool,
            count=t.size - 1,
        )
    )
    return [(index, index + 1) for index in np.flatnonzero(valid)]


def align_parametric_x(
    *,
    observation_time: object,
    observed_x: object,
    source_rows: Sequence[int],
    model_time: object,
    model_x: object,
    segment_ids: object,
    segment_keys: Sequence[str],
    discontinuity_ids: object,
    mode: str,
    dataset_label: str,
    x_name: str,
) -> ParametricXProjection:
    """Resolve one canonical X roster without extrapolation or branch switching."""

    obs_t = _vector("observation time", observation_time)
    obs_x = _vector("observed X", observed_x)
    rows = tuple(int(item) for item in source_rows)
    if (
        obs_t.size != obs_x.size
        or len(rows) != obs_t.size
        or len(rows) != len(set(rows))
        or any(item <= 0 for item in rows)
    ):
        raise FitSimulationError(
            f"Dataset {dataset_label!r}: X observation identity is malformed."
        )
    mode_norm = str(mode or "").strip().lower()
    if mode_norm not in {"monotone", "time_guided"}:
        raise FitSimulationError(f"Dataset {dataset_label!r}: unsupported X mode {mode!r}.")
    t, x, segments, keys, discontinuities = _validate_model(
        t_model=model_time,
        x_model=model_x,
        segment_ids=segment_ids,
        segment_keys=segment_keys,
        discontinuity_ids=discontinuity_ids,
        dataset_label=dataset_label,
        x_name=x_name,
    )
    intervals = _intervals(t, segments, keys, discontinuities)
    if not intervals:
        raise DeterministicParametricResolutionError(
            f"Dataset {dataset_label!r}: no uninterrupted model interval can resolve X.",
            dataset_label=dataset_label,
            x_name=x_name,
        )
    # Comparisons live in the physical X unit, while endpoint/direct
    # classification lives in the dimensionless interpolation fraction.  A
    # large X scale must not turn a genuine midpoint into a direct sample.
    scale = max(1.0, float(np.max(np.abs(x))), float(np.max(np.abs(obs_x))))
    x_tolerance = 16.0 * np.finfo(float).eps * scale
    fraction_tolerance = 16.0 * np.finfo(float).eps
    # Monotone means every uninterrupted branch is strictly monotone and time
    # cannot be used to settle a duplicate branch.
    if mode_norm == "monotone":
        for branch in dict.fromkeys((keys[left], int(discontinuities[left])) for left, _ in intervals):
            branch_rows = [
                index
                for index, branch_value in enumerate(zip(keys, discontinuities))
                if (branch_value[0], int(branch_value[1])) == branch
            ]
            if len(branch_rows) >= 2:
                differences = np.diff(x[branch_rows])
                if not np.all(differences > x_tolerance) and not np.all(differences < -x_tolerance):
                    raise DeterministicParametricResolutionError(
                        f"Dataset {dataset_label!r}: X(t) is not strictly monotone.",
                        dataset_label=dataset_label,
                        x_name=x_name,
                    )

    selected: list[tuple[float, float, float, int, str, int, str, str, int]] = []
    for value_t, value_x in zip(obs_t, obs_x, strict=True):
        exact: list[tuple[float, float, float, int, str, int, str, str, int]] = []
        mismatch: list[tuple[float, float, float, int, str, int, str, str, int]] = []
        for left, right in intervals:
            x_left, x_right = float(x[left]), float(x[right])
            low, high = min(x_left, x_right), max(x_left, x_right)
            if value_x >= low - x_tolerance and value_x <= high + x_tolerance:
                width = x_right - x_left
                if abs(width) <= x_tolerance:
                    continue
                fraction = min(1.0, max(0.0, float((value_x - x_left) / width)))
                selected_t = float(t[left] + fraction * (t[right] - t[left]))
                selected_x = float(x[left] + fraction * width)
                sample = "direct" if fraction <= fraction_tolerance or 1.0 - fraction <= fraction_tolerance else "interpolated"
                model_row = left if fraction < 0.5 else right
                exact.append(
                    (
                        abs(selected_x - float(value_x)),
                        selected_t,
                        selected_x,
                        model_row,
                        keys[left],
                        int(segments[left]),
                        int(discontinuities[left]),
                        sample,
                        0,
                    )
                )
            # Bounded projection onto the interval; no extrapolation.
            fraction = 0.0 if abs(x_right - x_left) <= x_tolerance else min(
                1.0, max(0.0, float((value_x - x_left) / (x_right - x_left)))
            )
            projected_x = float(x_left + fraction * (x_right - x_left))
            projected_t = float(t[left] + fraction * (t[right] - t[left]))
            sample = "direct" if fraction <= fraction_tolerance or 1.0 - fraction <= fraction_tolerance else "interpolated"
            model_row = left if fraction < 0.5 else right
            mismatch.append(
                (
                    abs(projected_x - float(value_x)),
                    projected_t,
                    projected_x,
                    model_row,
                    keys[left],
                    int(segments[left]),
                    int(discontinuities[left]),
                    sample,
                    1,
                )
            )
        def deduplicate(
            values: list[tuple[float, float, float, int, str, int, str, str, int]],
        ) -> list[tuple[float, float, float, int, str, int, str, str, int]]:
            unique: dict[tuple[float, float, int, str, int], tuple[float, float, float, int, str, int, str, str, int]] = {}
            for item in values:
                key = (item[1], item[2], item[5], item[4], item[6])
                current = unique.get(key)
                if current is None or item[3] < current[3]:
                    unique[key] = item
            return list(unique.values())

        candidates = deduplicate(exact or mismatch)
        if not candidates:
            raise CandidateParametricAlignmentError(
                f"Dataset {dataset_label!r}: no bounded model projection exists for X={float(value_x):.6g}.",
                dataset_label=dataset_label,
                x_name=x_name,
                failed_observed_value=float(value_x),
            )
        if mode_norm == "time_guided":
            chosen = min(candidates, key=lambda item: (abs(item[1] - float(value_t)), item[1], item[3]))
        else:
            best_distance = min(item[0] for item in candidates)
            tied = [item for item in candidates if abs(item[0] - best_distance) <= x_tolerance]
            if len(tied) != 1:
                raise DeterministicParametricResolutionError(
                    f"Dataset {dataset_label!r}: X={float(value_x):.6g} has unresolved equal branches.",
                    dataset_label=dataset_label,
                    x_name=x_name,
                    failed_observed_value=float(value_x),
                )
            chosen = tied[0]
        selected.append(chosen)

    return ParametricXProjection(
        source_rows=rows,
        observation_time=obs_t,
        observed_x=obs_x,
        selected_time=np.asarray([item[1] for item in selected]),
        selected_model_x=np.asarray([item[2] for item in selected]),
        delta_x=np.asarray([item[2] - value for item, value in zip(selected, obs_x, strict=True)]),
        segment_ids=np.asarray([item[5] for item in selected], dtype=np.int64),
        segment_keys=tuple(item[4] for item in selected),
        discontinuity_ids=np.asarray([item[6] for item in selected], dtype=np.int64),
        match_kinds=tuple("exact_crossing" if item[8] == 0 else "finite_projection" for item in selected),
        sample_kinds=tuple(item[7] for item in selected),
        model_rows=tuple(item[3] for item in selected),
    )


def project_parametric_y(
    *,
    projection: ParametricXProjection,
    model_time: object,
    model_y: object,
    segment_ids: object,
    segment_keys: Sequence[str],
    discontinuity_ids: object,
) -> np.ndarray:
    """Evaluate one model Y at an existing X projection, within its segment."""

    t = _vector("model time", model_time)
    y = _vector("model Y", model_y)
    segments = np.asarray(segment_ids, dtype=np.int64).reshape(-1)
    discontinuities = np.asarray(discontinuity_ids, dtype=np.int64).reshape(-1)
    keys = tuple(str(item or "").strip() for item in segment_keys)
    if (
        y.size != t.size
        or segments.size != t.size
        or discontinuities.size != t.size
        or len(keys) != t.size
    ):
        raise FitSimulationError("Parametric Y projection received malformed model identity.")
    result: list[float] = []
    for selected_t, segment, key, discontinuity, row in zip(
        projection.selected_time,
        projection.segment_ids,
        projection.segment_keys,
        projection.discontinuity_ids,
        projection.model_rows,
        strict=True,
    ):
        if row < 0 or row >= t.size:
            raise FitSimulationError("Parametric X projection model row is out of bounds.")
        if (
            int(segments[row]) != int(segment)
            or keys[row] != key
            or int(discontinuities[row]) != int(discontinuity)
        ):
            raise FitSimulationError("Parametric projection crossed a discontinuity.")
        if abs(float(selected_t) - float(t[row])) <= 16.0 * np.finfo(float).eps:
            result.append(float(y[row]))
            continue
        if selected_t >= t[row] and row + 1 < t.size:
            other = row + 1
        elif row > 0:
            other = row - 1
        else:
            raise FitSimulationError("Parametric projection cannot resolve model Y.")
        if (
            int(segments[other]) != int(segment)
            or keys[other] != key
            or int(discontinuities[other]) != int(discontinuity)
        ):
            raise FitSimulationError("Parametric Y interpolation crossed a discontinuity.")
        width = float(t[other] - t[row])
        if width == 0.0:
            raise FitSimulationError("Parametric model time interval is degenerate.")
        fraction = float((selected_t - t[row]) / width)
        result.append(float(y[row] + fraction * (y[other] - y[row])))
    values = np.asarray(result, dtype=float)
    if not np.all(np.isfinite(values)):
        raise FitSimulationError("Parametric Y projection produced non-finite values.")
    return values
