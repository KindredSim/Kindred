"""Project retained physical coverage and ordered endpoints for quantity consumers."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from kindred.core.project_presentation_state import (
    SimulationSeriesAxisSelection, VesselVolumeAxisSelection,
)


def axis_values(result, selection):
    if isinstance(selection, VesselVolumeAxisSelection):
        return None if result.volume is None else (
            result.volume.first_index, result.volume.values, "Vessel volume",
        )
    if isinstance(selection, SimulationSeriesAxisSelection):
        name = selection.series_name
        values = result.t if name == "t" else result.series.get(name)
        if values is None and name in result.scalars:
            values = np.full(result.t.shape, result.scalars[name])
        return None if values is None else (0, values, name)
    return None


@dataclass(frozen=True, slots=True)
class ProjectedCurveChunk:
    """An ordered continuous region or instantaneous path, with canonical time."""

    time: np.ndarray
    x: np.ndarray
    series: tuple[tuple[str, np.ndarray], ...]
    instantaneous: bool = False


def physical_curve_chunks(result, x_axis, series_axes, indices):
    """Use execution endpoints directly; never sort or deduplicate by axis value."""
    x_data = axis_values(result, x_axis)
    if x_data is None:
        return ()
    ys = [(name, axis, axis_values(result, axis)) for name, axis in series_axes]
    if any(data is None for _, _, data in ys):
        return ()
    boundaries = {float(b.time): b for b in result.presentation_boundaries}
    # Exact canonical row identity is owned by the solver, including transitions.
    boundary_rows = {i for i, time in enumerate(result.t) if float(time) in boundaries}
    transitions = {i for i in range(1, len(result.t)) if (
        result.segment_ids[i] != result.segment_ids[i-1]
        or result.discontinuity_ids[i] != result.discontinuity_ids[i-1]
    )}
    indices = sorted(set(map(int, indices)) | boundary_rows | transitions)
    chunks, rows = [], []

    def emit(rows, instantaneous=False):
        if rows:
            matrix = np.asarray(rows, dtype=float)
            chunks.append(ProjectedCurveChunk(
                matrix[:, 0], matrix[:, 1],
                tuple((name, matrix[:, i+2]) for i, (name, _, _) in enumerate(ys)), instantaneous,
            ))

    def canonical(data, index):
        return np.nan if index < data[0] else data[1][index-data[0]]

    def endpoint(boundary, state, index):
        concentrations = dict(state.concentrations)
        matches = (state.temperature_K == result.temperature_K[index] and all(
            result.series[name][index] == value for name, value in state.concentrations
        ))

        def value(axis):
            if isinstance(axis, VesselVolumeAxisSelection):
                return np.nan if state.volume is None else state.volume
            name = axis.series_name
            if name == "t":
                return boundary.time
            if name in concentrations:
                return concentrations[name]
            if name in result.scalars:
                return result.scalars[name]
            if matches and name in result.series:
                return result.series[name][index]
            return np.nan
        return [boundary.time, value(x_axis), *(value(axis) for _, axis, _ in ys)]

    for index in indices:
        time = float(result.t[index])
        boundary = boundaries.get(time)
        row = [time, canonical(x_data, index), *(canonical(data, index) for _, _, data in ys)]
        if boundary is not None:
            endpoints = [endpoint(boundary, state, index) for state in boundary.states]
            if rows:
                rows.append(endpoints[0])
                emit(rows)
            emit(endpoints, True)
            # Canonical outputs stay authoritative even when added sides are unknown.
            rows = [row]
        else:
            rows.append(row)
            if index in transitions:
                emit(rows)
                rows = [row]
    emit(rows)
    return tuple(chunks)
