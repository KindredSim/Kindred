"""Pure focused Fit quantities, unit conversion, and retained curve projection."""
from __future__ import annotations

import numpy as np

from kindred.core.analysis.global_fit_execution import fit_result_target_values
from kindred.core.project_presentation_state import SimulationSeriesAxisSelection
from kindred.core.quantity_units import MOLAR, SECOND, conversion_scale, presentation_unit
from kindred.gui.result_curve_projection import physical_curve_chunks


def focused_units(quantity_units, preferred_units, x_name, targets):
    """Choose concentration presentation once from the admitted setup's units."""
    units, preferred = dict(quantity_units), dict(preferred_units)
    names = tuple(dict.fromkeys((x_name, *targets)))
    concentrations = [name for name in names if units.get(name) is not None and units[name].compatible(MOLAR)]
    choices = [preferred.get(name) or MOLAR for name in concentrations]
    common = choices[0] if choices and all(unit == choices[0] for unit in choices) else MOLAR
    return tuple((name, SECOND if name == "t" else common if name in concentrations else units.get(name)) for name in names)


def prepared_fit_units(prepared, x_name, targets):
    spec = prepared.parameter_algebra_spec_for_execution()
    units = dict(spec.quantity_units)
    units.update({name: MOLAR for name in prepared.analysis.species_names})
    units["t"] = SECOND
    declarations = {line.name: line.declared_unit for line in spec.observable_block.lines}
    preferred = {name: presentation_unit(unit, declarations.get(name)) for name, unit in units.items()}
    return tuple(units.items()), focused_units(units, preferred, x_name, targets)


def observed_fit_units(model_display, x_name, x_unit, y_units):
    """Convert compatible measurements to the model display, retaining other meanings."""
    display = dict(model_display)
    return tuple(
        (name, display[name] if display.get(name) is not None and unit.compatible(display[name]) else unit)
        for name, unit in ((x_name, x_unit), *y_units)
    )


def shared_axis_unit(units):
    """A shared label describes every model and measured quantity on the axis."""
    units = tuple(units)
    return units[0] if units and units[0] is not None and all(unit == units[0] for unit in units) else None


def scale_values(values, source, destination):
    if source is None or destination is None:
        return tuple(float(value) for value in values)
    scale = conversion_scale(source, destination)
    # Model gaps retain their presentation meaning; observations are admitted finite.
    return tuple(scale.apply(float(value)) if np.isfinite(value) else float(value) for value in values)


def focused_model_chunks(dataset):
    result = dataset.simulation
    targets = tuple(name for name, _ in dataset.observed_y)
    for name in dict.fromkeys((dataset.x_name, *targets)):
        if name != "t":
            fit_result_target_values(result, name, dataset_id=dataset.dataset_id)
    return physical_curve_chunks(
        result, SimulationSeriesAxisSelection(dataset.x_name),
        tuple((name, SimulationSeriesAxisSelection(name)) for name in targets),
        range(len(result.t)),
    )


def focused_model_rows(dataset):
    """Full numerical rows, including known endpoints; no renderer gap rows."""
    rows = []
    for chunk in focused_model_chunks(dataset):
        matrix = np.column_stack((chunk.time, chunk.x, *(values for _, values in chunk.series)))
        for row in matrix:
            if not rows or not np.array_equal(rows[-1], row, equal_nan=True):
                rows.append(row)
    return np.asarray(rows, dtype=float)
