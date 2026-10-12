"""Joint weighted matching to uninterrupted linear simulation segments.

A location is a geometric projection, not an identified experimental time.
Ties retain the first equally good location without asserting uniqueness.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields

import numpy as np

from kindred.core.analysis.fit_dataset_payload import FitObservationSeries
from kindred.core.exceptions import FitSimulationError, FittingCancelled
from kindred.core.simulator.solver_types import SimulationCancellationControl

__all__ = ["ParametricProjection", "align_parametric", "project_parametric_y"]


@dataclass(frozen=True, slots=True)
class ParametricProjection:
    """One shared curve location per dataset row within the owning dataset."""

    source_rows: tuple[int, ...]
    delta_x: np.ndarray
    left_rows: np.ndarray
    right_rows: np.ndarray
    fractions: np.ndarray

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, np.ndarray):
                value.setflags(write=False)

    def subset(self, source_rows: Sequence[int]) -> "ParametricProjection":
        positions = {row: index for index, row in enumerate(self.source_rows)}
        indexes = [positions[row] for row in source_rows]
        return ParametricProjection(**{
            field.name: (tuple(getattr(self, field.name)[i] for i in indexes)
                         if field.name == "source_rows"
                         else getattr(self, field.name)[indexes])
            for field in fields(self)
        })


def align_parametric(
    *,
    observed_x: np.ndarray,
    source_rows: Sequence[int],
    observations: Mapping[str, FitObservationSeries],
    model_x: np.ndarray,
    model_y: Mapping[str, np.ndarray],
    model_time: np.ndarray,
    segment_ids: np.ndarray,
    segment_keys: Sequence[str],
    discontinuity_ids: np.ndarray,
    multipliers: Mapping[str, float],
    x_name: str,
    cancellation_control: SimulationCancellationControl | None,
) -> ParametricProjection:
    """Minimize sum((weight multiplier * discrepancy)**2) on each segment.

    Every available selected quantity contributes once. Recorded observation
    time is deliberately absent. Endpoints also participate, including isolated
    samples beside a jump. No segment crosses an executor-issued boundary.
    """
    if cancellation_control is not None and cancellation_control.cancel_requested():
        raise FittingCancelled()
    names = tuple(sorted(observations))
    model = np.column_stack((model_x, *(model_y[name] for name in names)))
    segments = np.asarray(segment_ids)
    discontinuities = np.asarray(discontinuity_ids)
    keys = tuple(segment_keys)
    continuous = ((segments[1:] == segments[:-1])
                  & (discontinuities[1:] == discontinuities[:-1])
                  & np.asarray([a == b for a, b in zip(keys, keys[1:])]))
    if np.any(np.diff(model_time)[continuous] <= 0.0):
        raise FitSimulationError("Model time must increase within uninterrupted segments.")
    # Direct samples guarantee bounded matches even when no adjacent pair is valid.
    samples = np.arange(len(model))
    adjacent = np.flatnonzero(continuous)
    left = np.concatenate((samples, adjacent))
    right = np.concatenate((samples, adjacent + 1))
    observed_by_row = {
        name: dict(zip(observations[name].source_rows, observations[name].y, strict=True))
        for name in names
    }
    weight = np.asarray([multipliers[x_name], *(multipliers[name] for name in names)])
    chosen_rows: list[int] = []
    chosen_fractions: list[float] = []
    for row, x_value in zip(source_rows, observed_x, strict=True):
        if cancellation_control is not None and cancellation_control.cancel_requested():
            raise FittingCancelled()
        observed = np.asarray([x_value, *(observed_by_row[name].get(row, np.nan) for name in names)])
        available = np.isfinite(observed)
        start = model[left][:, available]
        direction = (model[right][:, available] - start) * weight[available]
        offset = (observed[available] - start) * weight[available]
        # Scale the calculation together, preserving the weighted minimizer.
        scale = max(float(np.max(np.abs(direction))), float(np.max(np.abs(offset))))
        if scale:
            direction = direction / scale
            offset = offset / scale
        denominator = np.sum(direction * direction, axis=1)
        fraction = np.clip(np.divide(
            np.sum(offset * direction, axis=1), denominator,
            out=np.zeros_like(denominator), where=denominator > 0.0,
        ), 0.0, 1.0)
        errors = offset - fraction[:, None] * direction
        costs = np.sum(errors * errors, axis=1)
        chosen = int(np.argmin(costs))
        chosen_rows.append(chosen)
        chosen_fractions.append(float(fraction[chosen]))
    selected_left = left[chosen_rows]
    selected_right = right[chosen_rows]
    fractions = np.asarray(chosen_fractions)
    selected_x = model_x[selected_left] + fractions * (model_x[selected_right] - model_x[selected_left])
    return ParametricProjection(
        source_rows=tuple(source_rows), delta_x=selected_x - observed_x,
        left_rows=selected_left, right_rows=selected_right, fractions=fractions,
    )


def project_parametric_y(*, projection: ParametricProjection, model_y: np.ndarray) -> np.ndarray:
    """Use the already selected location for another quantity of the same row."""
    left = model_y[projection.left_rows]
    return left + projection.fractions * (model_y[projection.right_rows] - left)
