"""Pure focused Fit quantities, unit conversion, and retained curve projection."""
from __future__ import annotations

import numpy as np

from kindred.core.analysis.global_fit_execution import fit_result_target_values
from kindred.core.project_presentation_state import SimulationSeriesAxisSelection, VesselVolumeAxisSelection
from kindred.core.quantity_units import CUBIC_METRE, MOLAR, SECOND, conversion_scale
from kindred.core.result_curve_projection import physical_curve_chunks
from kindred.core.simulation_result_finalization import result_quantity_units

VESSEL_VOLUME = "Vessel volume"


def focused_units(quantity_units, preferred_units, x_name, targets):
    """Choose concentration presentation once from the admitted setup's units."""
    units, preferred = dict(quantity_units), dict(preferred_units)
    names = tuple(dict.fromkeys((x_name, *targets)))
    concentrations = [name for name in names if units.get(name) is not None and units[name].compatible(MOLAR)]
    choices = [preferred.get(name) or MOLAR for name in concentrations]
    common = choices[0] if choices and all(unit == choices[0] for unit in choices) else MOLAR
    return tuple((name, SECOND if name == "t" else common if name in concentrations else units.get(name)) for name in names)


def fit_display_units(units, preferred_units, x_name, targets, x_unit, y_units):
    """Model and measured display units; an axis without one shared unit stays scientific."""
    units = dict(units)
    display = dict(focused_units(units, preferred_units, x_name, targets))
    for axis in ((x_name,), tuple(targets)):
        measured = dict(observed_fit_units(display, x_name, x_unit, y_units))
        if shared_axis_unit(unit for name in axis for unit in (display[name], measured[name])) is None:
            display.update((name, SECOND if name == "t" else units.get(name)) for name in axis)
    return display, dict(observed_fit_units(display, x_name, x_unit, y_units))


def prepared_fit_units(prepared, x_name, targets, x_unit, y_units):
    units, preferred = result_quantity_units(
        prepared.parameter_algebra_spec_for_execution(), prepared.analysis.species_names
    )
    return fit_display_units(units, preferred, x_name, targets, x_unit, y_units)


def observed_fit_units(model_display, x_name, x_unit, y_units):
    """Convert compatible measurements to the model display, retaining other meanings."""
    display = dict(model_display)
    return tuple(
        (name, display[name] if unit is not None and display.get(name) is not None and unit.compatible(display[name]) else unit)
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


def simulated_quantities(result):
    """Every quantity a retained result holds, in result order; failed lets are absent."""
    names = list(result.series)
    names.extend(name for name in result.succeeded_algebra_outputs
                 if name in result.scalars and name not in result.series)
    if result.volume is not None:
        names.append(VESSEL_VOLUME)
    return tuple(names)


def simulated_units(result, names, targets_display):
    """Source and display unit of quantities drawn beside the fitted targets.

    The targets keep their own units; others never influence them. Concentrations
    take the targets' concentration unit, volume uses its presentation unit, and
    any other quantity keeps its retained scientific unit, as non-concentration
    targets do.
    """
    units = dict(result.quantity_units)
    concentration = next((unit for unit in targets_display.values()
                          if unit is not None and unit.compatible(MOLAR)), MOLAR)
    shown = {}
    for name in names:
        if name == VESSEL_VOLUME:
            shown[name] = (CUBIC_METRE, dict(result.presentation_units)[VESSEL_VOLUME])
            continue
        source = units.get(name)
        display = concentration if source is not None and source.compatible(MOLAR) else source
        shown[name] = (source, display)
    return shown


def _quantity_axis(name):
    return VesselVolumeAxisSelection() if name == VESSEL_VOLUME else SimulationSeriesAxisSelection(name)


def model_chunks(dataset, names):
    """Project quantities present in the dataset's retained result against its X."""
    result = dataset.simulation
    return physical_curve_chunks(
        result, SimulationSeriesAxisSelection(dataset.x_name),
        tuple((name, _quantity_axis(name)) for name in names),
        range(len(result.t)),
    )


def focused_model_chunks(dataset):
    result = dataset.simulation
    targets = tuple(name for name, _ in dataset.observed_y)
    for name in dict.fromkeys((dataset.x_name, *targets)):
        if name != "t":
            fit_result_target_values(result, name, dataset_id=dataset.dataset_id)
    return model_chunks(dataset, targets)


def focused_model_rows(dataset, names=None):
    """Full numerical rows, including known endpoints; no renderer gap rows.

    Without names the rows hold the fitted targets; with names, those quantities.
    """
    rows = []
    for chunk in focused_model_chunks(dataset) if names is None else model_chunks(dataset, names):
        matrix = np.column_stack((chunk.time, chunk.x, *(values for _, values in chunk.series)))
        for row in matrix:
            if not rows or not np.array_equal(rows[-1], row, equal_nan=True):
                rows.append(row)
    return np.asarray(rows, dtype=float)
