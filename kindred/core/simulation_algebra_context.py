"""Scientific symbol bindings shared by execution and result Algebra."""
from __future__ import annotations

from typing import Callable, Mapping

import numpy as np

from kindred.core.equilibrium_rate_authority import effective_equilibrium_keq
from kindred.core.kinetics import (
    ArrheniusRateModel,
    EyringRateModel,
    forward_rate_model_for_equilibrium,
    forward_rate_model_for_reaction,
)
from kindred.core.mechanism_metadata import EquilibriumMetadataKeys, MechanismMetadataView
from kindred.core.ode_builder import build_effective_rate_coefficient_evaluator
from kindred.core.simulator.parameter_namespace import build_namespace_from_mechanism
from kindred.core.simulator.step_indexing import get_step_index_map, iter_canonical_parameters
from kindred.core.temperature import TemperatureScheduleProtocol
from kindred.core.units import UnitsModel


def _prepared_static_scalar_snapshot(
    mechanism: object,
    parameter_algebra_spec: object,
) -> dict[str, float]:
    """Read the complete static scalar map already bound during preparation."""

    metadata = getattr(mechanism, "metadata", {}) or {}
    if not isinstance(metadata, Mapping):
        raise TypeError("prepared mechanism metadata must be a mapping.")
    if parameter_algebra_spec is None:
        raise TypeError(
            "prepared scalar snapshot requires its retained parameter algebra spec."
        )
    assignment_names = getattr(
        parameter_algebra_spec,
        "param_assignment_names",
        None,
    )
    if not callable(assignment_names):
        raise TypeError(
            "prepared parameter Algebra spec must expose assignment names."
        )
    mechanism_names = {
        str(name)
        for name in (
            getattr(parameter_algebra_spec, "mechanism_param_names", ()) or ()
        )
    }
    scalar_names = tuple(
        sorted(
            str(name)
            for name in assignment_names()
            if str(name) not in mechanism_names
        )
    )
    raw_values = metadata.get("scalar_params")
    scalar_values = raw_values if isinstance(raw_values, Mapping) else {}
    raw_bindings = metadata.get("scalar_param_bindings")
    scalar_bindings = raw_bindings if isinstance(raw_bindings, Mapping) else {}
    values: dict[str, float] = {}
    for name in scalar_names:
        if name in scalar_bindings:
            binding = scalar_bindings[name]
            raw_value = binding() if callable(binding) else binding
        elif name in scalar_values:
            raw_value = scalar_values[name]
        else:
            raise ValueError(
                f"prepared static scalar snapshot is missing {name!r}."
            )
        if type(raw_value) is bool:
            raise TypeError(
                f"prepared static scalar {name!r} must be numeric."
            )
        try:
            value = float(raw_value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError(
                f"prepared static scalar {name!r} must be numeric."
            ) from exc
        if not np.isfinite(value):
            raise ValueError(
                f"prepared static scalar {name!r} must be finite."
            )
        values[name] = value
    # Parser-admitted physical source coordinates are static Algebra symbols.
    # Their values come from the typed model/thermodynamic authority; only
    # calculated indexed readouts remain row-dependent below.
    metadata_view = MechanismMetadataView.from_metadata(metadata)
    energy_units = UnitsModel(energy_unit=metadata_view.energy_unit)
    reactions = tuple(getattr(mechanism, "reactions", ()) or ())
    equilibria = tuple(getattr(mechanism, "equilibria", ()) or ())
    step_map = get_step_index_map(mechanism)
    for coordinate in build_namespace_from_mechanism(mechanism).source_coordinates:
        role = str(coordinate.role)
        if role not in {
            "A",
            "Ea",
            "dG_act",
            "dG_eq",
            "kappa",
            "degeneracy",
        }:
            continue
        source_index = coordinate.source_index
        if source_index is None or not 0 <= int(source_index) < len(step_map):
            continue
        entry = step_map[int(source_index)]
        kind = str(entry.get("kind") or "")
        index_key = "reaction_index" if kind == "reaction" else "equilibrium_index"
        try:
            index = int(entry[index_key])
        except (KeyError, TypeError, ValueError):
            continue
        model = None
        if kind == "reaction" and 0 <= index < len(reactions):
            model = forward_rate_model_for_reaction(
                reactions[index],
                default_standard_conc_M=float(metadata_view.standard_conc_M),
            )
        elif kind == "equilibrium" and 0 <= index < len(equilibria):
            model = forward_rate_model_for_equilibrium(
                equilibria[index],
                temperature_K=None,
                default_standard_conc_M=float(metadata_view.standard_conc_M),
            )
        value: float | None = None
        if role == "A" and isinstance(model, ArrheniusRateModel):
            value = float(model.A)
        elif role == "Ea" and isinstance(model, ArrheniusRateModel):
            value = float(energy_units.from_jmol(model.Ea_J_per_mol))
        elif role == "dG_act" and isinstance(model, EyringRateModel):
            value = float(energy_units.from_jmol(model.dG_act_J_per_mol))
        elif role == "kappa" and isinstance(model, EyringRateModel):
            value = float(model.kappa)
        elif role == "degeneracy" and isinstance(model, EyringRateModel):
            value = float(model.degeneracy)
        elif role == "dG_eq" and kind == "equilibrium":
            raw_dg = dict(getattr(equilibria[index], "metadata", {}) or {}).get(
                EquilibriumMetadataKeys.DG_EQ_J_PER_MOL
            )
            try:
                value = float(energy_units.from_jmol(float(raw_dg)))
            except (TypeError, ValueError, OverflowError):
                value = None
        if value is not None and np.isfinite(value):
            values[str(coordinate.canonical_name)] = float(value)
    return values


def prepare_simulation_algebra_symbols(
    mechanism: object,
    *,
    parameter_algebra_spec: object,
    temperature_schedule: TemperatureScheduleProtocol | None,
) -> tuple[dict[str, float], tuple[str, ...], Callable[[float], dict[str, float]]]:
    """Bind the scientific symbol owners once for point and row evaluation."""
    metadata = getattr(mechanism, "metadata", {}) or {}
    nominal_temperature = float(MechanismMetadataView.from_metadata(metadata).temperature_K)
    coefficient_evaluator = build_effective_rate_coefficient_evaluator(mechanism)
    reactions = tuple(getattr(mechanism, "reactions", ()) or ())
    equilibria = tuple(getattr(mechanism, "equilibria", ()) or ())
    canonical_parameters = tuple(iter_canonical_parameters(mechanism))

    def symbols(time: float) -> dict[str, float]:
        temperature = float(temperature_schedule(time)) if temperature_schedule is not None else nominal_temperature
        coefficients = coefficient_evaluator(temperature)
        values = {"T": temperature}
        for name, entry, role in canonical_parameters:
            kind = str(entry.get("kind") or "")
            if kind == "reaction" and role == "k":
                value = coefficients.forward[int(entry["reaction_index"])]
            elif kind == "equilibrium":
                index = int(entry["equilibrium_index"])
                coefficient_index = len(reactions) + index
                if role == "kf":
                    value = coefficients.forward[coefficient_index]
                elif role == "kr":
                    value = coefficients.reverse[coefficient_index]
                elif role == "Keq":
                    value = effective_equilibrium_keq(equilibria[index], temperature_K=temperature)
                else:
                    continue
            else:
                continue
            if value is None:
                raise ValueError(f"chemistry coefficient readout {name!r} is unavailable.")
            values[str(name)] = float(value)
        return values

    names = ("T", *(str(name) for name, entry, role in canonical_parameters
                    if (entry.get("kind") == "reaction" and role == "k")
                    or (entry.get("kind") == "equilibrium" and role in {"kf", "kr", "Keq"})))
    return _prepared_static_scalar_snapshot(mechanism, parameter_algebra_spec), names, symbols
