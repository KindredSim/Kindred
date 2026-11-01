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
    boundary_rows = np.flatnonzero(np.isin(result.t, tuple(boundaries)))
    transitions = 1 + np.flatnonzero(
        (result.segment_ids[1:] != result.segment_ids[:-1])
        | (result.discontinuity_ids[1:] != result.discontinuity_ids[:-1])
    )
    breaks = np.union1d(boundary_rows, transitions)
    indices = np.union1d(np.fromiter(indices, dtype=np.int64), breaks)

    def canonical(data):
        column = np.full(indices.size, np.nan)
        known = indices >= data[0]
        column[known] = data[1][indices[known] - data[0]]
        return column

    matrix = np.column_stack((result.t[indices], canonical(x_data), *(canonical(data) for _, _, data in ys)))
    chunks = []

    def emit(rows, instantaneous=False):
        if len(rows):
            rows = np.array(rows, dtype=float)
            chunks.append(ProjectedCurveChunk(
                rows[:, 0], rows[:, 1],
                tuple((name, rows[:, i+2]) for i, (name, _, _) in enumerate(ys)), instantaneous,
            ))

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

    start = 0
    for position in np.searchsorted(indices, breaks):
        index = indices[position]
        boundary = boundaries.get(float(result.t[index]))
        if boundary is not None:
            endpoints = [endpoint(boundary, state, index) for state in boundary.states]
            if position > start:
                emit(np.vstack((matrix[start:position], np.asarray(endpoints[0], dtype=float))))
            emit(endpoints, True)
            # Canonical outputs stay authoritative even when added sides are unknown.
            start = position
        else:
            emit(matrix[start:position+1])
            start = position
    emit(matrix[start:])
    return tuple(chunks)


def alignment_rows(result, x_axis, series_axes):
    """Flatten the drawn continuous curve for measured-X matching.

    Rows come from the same chunks the plot draws, so known pre-boundary
    endpoints participate while instantaneous jump paths never connect.  Each
    row carries its chunk index as region identity.  A quantity unknown at a row
    (an uncomputable calculated endpoint, or a time point where a calculated
    quantity is undefined) stays NaN there; matching places an observation only
    where every quantity it measures is known, as the plot breaks each curve
    where it is unknown.
    """
    chunks = physical_curve_chunks(result, x_axis, series_axes, range(len(result.t)))
    names = tuple(name for name, _axis in series_axes)
    xs, regions = [], []
    ys = {name: [] for name in names}
    for region, chunk in enumerate(chunk for chunk in chunks if not chunk.instantaneous):
        xs.append(chunk.x)
        regions.append(np.full(chunk.time.size, region, dtype=np.int64))
        for name, values in chunk.series:
            ys[name].append(values)

    def joined(parts, dtype=float):
        return np.concatenate(parts) if parts else np.asarray([], dtype=dtype)

    return (
        joined(xs),
        {name: joined(ys[name]) for name in names},
        joined(regions, np.int64),
    )
