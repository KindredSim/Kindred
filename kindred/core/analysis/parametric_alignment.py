"""Joint weighted matching to uninterrupted linear simulation segments.

A location is a geometric projection, not an identified experimental time.
Ties retain the first equally good location without asserting uniqueness.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields

import numpy as np

from kindred.core.analysis.fit_dataset_payload import FitObservationSeries
from kindred.core.exceptions import FitCandidateDomainError, FitSimulationError, FittingCancelled
from kindred.core.simulator.solver_types import SimulationCancellationControl

__all__ = ["CandidateParametricAlignmentError", "ParametricProjection",
           "align_parametric", "project_parametric_y"]


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


@dataclass(frozen=True, slots=True)
class ParametricProjection:
    """One shared curve location per dataset row within the owning dataset."""

    source_rows: tuple[int, ...]
    selected_model_x: np.ndarray
    delta_x: np.ndarray
    segment_ids: np.ndarray
    segment_keys: tuple[str, ...]
    discontinuity_ids: np.ndarray
    left_rows: np.ndarray
    right_rows: np.ndarray
    fractions: np.ndarray

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, np.ndarray):
                value.setflags(write=False)

    @property
    def size(self) -> int:
        return len(self.source_rows)

    @property
    def sample_kinds(self) -> tuple[str, ...]:
        return tuple("direct" if f == 0.0 or f == 1.0 else "interpolated"
                     for f in self.fractions)

    def subset(self, source_rows: Sequence[int]) -> "ParametricProjection":
        positions = {row: index for index, row in enumerate(self.source_rows)}
        indexes = [positions[row] for row in source_rows]
        return ParametricProjection(**{
            field.name: (tuple(getattr(self, field.name)[i] for i in indexes)
                         if field.name in {"source_rows", "segment_keys"}
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
    dataset_label: str,
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
    if not model.size or not np.all(np.isfinite(model)):
        raise FitSimulationError(f"Dataset {dataset_label!r}: model quantities must be finite.")
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
        if not np.isfinite(costs[chosen]):
            raise CandidateParametricAlignmentError(
                "Joint curve matching produced nonfinite discrepancies.",
                dataset_label=dataset_label, x_name=x_name,
            )
        chosen_rows.append(chosen)
        chosen_fractions.append(float(fraction[chosen]))
    selected_left = left[chosen_rows]
    selected_right = right[chosen_rows]
    fractions = np.asarray(chosen_fractions)
    selected_x = model_x[selected_left] + fractions * (model_x[selected_right] - model_x[selected_left])
    return ParametricProjection(
        source_rows=tuple(source_rows), selected_model_x=selected_x,
        delta_x=selected_x - observed_x,
        segment_ids=segments[selected_left],
        segment_keys=tuple(keys[i] for i in selected_left),
        discontinuity_ids=discontinuities[selected_left],
        left_rows=selected_left, right_rows=selected_right, fractions=fractions,
    )


def project_parametric_y(*, projection: ParametricProjection, model_y: np.ndarray) -> np.ndarray:
    """Use the already selected location for another quantity of the same row."""
    left = model_y[projection.left_rows]
    return left + projection.fractions * (model_y[projection.right_rows] - left)
