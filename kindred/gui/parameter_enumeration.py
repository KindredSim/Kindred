"""Typed GUI coordinates for mechanism parameters.

The GUI never scans DSL text to invent parameter ownership.  This module
parses the accepted source once, projects the core's typed rate families, and
returns the one coordinate representation consumed by Slider and Fit.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Mapping

from kindred.core.equilibrium_rate_authority import (
    EquilibriumRateAuthorityKind,
    effective_equilibrium_keq,
    effective_equilibrium_reverse_rate,
    normalize_existing_equilibrium_rate_authority,
    step_entry_authored_role_is_derived,
    step_entry_authored_role_is_editable,
)
from kindred.core.kinetics import (
    ForwardRateModel,
    forward_rate_at_temperature,
    forward_rate_family,
    forward_rate_model_for_equilibrium,
    forward_rate_model_for_reaction,
)
from kindred.core.mechanism_metadata import (
    MechanismMetadataKeys,
    MechanismMetadataView,
)
from kindred.core.simulator.dsl_parse import PreparedDslRequestSet
from kindred.core.simulator.parameter_algebra import (
    apply_parameter_algebra_spec_to_mechanism,
    solver_parameter_units_from_mechanism,
)
from kindred.core.simulator.parameter_namespace import build_namespace_from_mechanism
from kindred.core.simulator.step_indexing import get_step_index_map
from kindred.core.validation import try_parse_callable_finite_float, try_parse_int
from kindred.core.species import validate_temperature
from kindred.core.units import (
    KEQ_MIN,
    KEQ_MAX,
    RATE_MIN,
    RATE_MAX,
    admit_keq,
    admit_rate,
)

__all__ = [
    "GuiParameterCoordinate",
    "GuiParameterScientificCategory",
    "admit_gui_parameter_value",
    "enumerate_parameter_coordinates",
]


GuiParameterScientificCategory = Literal[
    "rate",
    "equilibrium_constant",
    "finite_scalar",
]


def admit_gui_parameter_value(
    value: object,
    *,
    scientific_category: GuiParameterScientificCategory,
    allow_exact_zero: bool,
    field: str,
) -> float:
    """Admit one GUI/Fit scalar through its accepted scientific authority."""

    if type(allow_exact_zero) is not bool:
        raise TypeError("allow_exact_zero must be a bool.")
    if scientific_category == "rate":
        return admit_rate(
            value,
            field=field,
            allow_zero=allow_exact_zero,
        )
    if scientific_category == "equilibrium_constant":
        if allow_exact_zero:
            raise ValueError("Equilibrium constants cannot permit exact zero.")
        return admit_keq(value, field=field)
    if scientific_category == "finite_scalar":
        if not allow_exact_zero:
            raise ValueError("Finite scalar coordinates must permit exact zero.")
        if type(value) is bool or isinstance(value, str):
            raise TypeError(f"{field} must be numeric.")
        try:
            parsed = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError(f"{field} must be numeric.") from exc
        if not math.isfinite(parsed):
            raise ValueError(f"{field} must be finite.")
        return float(parsed)
    raise ValueError(f"Unknown GUI parameter scientific category: {scientific_category!r}.")


def _finite(value: object) -> float | None:
    parsed, ok = try_parse_callable_finite_float(value)
    return float(parsed) if ok else None


def _positive_bounds(value: float, *, minimum: float = RATE_MIN, maximum: float = RATE_MAX) -> tuple[float, float]:
    if value > 0.0:
        low = max(value * 0.01, minimum)
        high = min(value * 100.0, maximum)
        if high > low:
            return float(low), float(high)
    return 0.0, max(1.0, abs(value) * 10.0)


def _signed_bounds(value: float) -> tuple[float, float]:
    span = max(1.0, abs(value) * 0.5)
    low = value - span
    high = value + span
    if not math.isfinite(low) or low >= value:
        low = value
    if not math.isfinite(high) or high <= value:
        high = value
    if low == high:
        low = math.nextafter(value, -math.inf)
        high = math.nextafter(value, math.inf)
    if not math.isfinite(low):
        low = value
    if not math.isfinite(high):
        high = value
    if not low < high:
        raise ValueError("Finite scalar has no representable finite Fit interval.")
    return float(low), float(high)


@dataclass(frozen=True, slots=True)
class GuiParameterCoordinate:
    """One stable GUI/Fit coordinate backed by a typed scientific owner."""

    name: str
    display_name: str
    value: float
    minimum: float
    maximum: float
    log10_allowed: bool
    family: str
    role: str
    writable: bool
    derived: bool
    scientific_category: GuiParameterScientificCategory
    allow_exact_zero: bool
    unit: str = "1"

    def __post_init__(self) -> None:
        name = str(self.name or "").strip()
        if not name:
            raise ValueError("GUI parameter name must be nonblank.")
        if type(self.allow_exact_zero) is not bool:
            raise TypeError("GUI parameter exact-zero authority must be a bool.")
        value = admit_gui_parameter_value(
            self.value,
            scientific_category=self.scientific_category,
            allow_exact_zero=self.allow_exact_zero,
            field=f"{name} value",
        )
        minimum = admit_gui_parameter_value(
            self.minimum,
            scientific_category=self.scientific_category,
            allow_exact_zero=self.allow_exact_zero,
            field=f"{name} minimum",
        )
        maximum = admit_gui_parameter_value(
            self.maximum,
            scientific_category=self.scientific_category,
            allow_exact_zero=self.allow_exact_zero,
            field=f"{name} maximum",
        )
        if minimum >= maximum or not minimum <= value <= maximum:
            raise ValueError(
                f"{name} requires minimum <= value <= maximum."
            )
        if self.log10_allowed and (minimum <= 0.0 or value <= 0.0):
            raise ValueError(f"{name} cannot offer Log10 outside a positive domain.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)


def enumerate_parameter_coordinates(
    prepared_dsl: PreparedDslRequestSet,
    *,
    wegscheider_cyclicity_enabled: bool,
) -> tuple[GuiParameterCoordinate, ...]:
    """Enumerate writable coordinates from one accepted typed DSL authority."""

    if not isinstance(prepared_dsl, PreparedDslRequestSet):
        raise TypeError("prepared_dsl must be a PreparedDslRequestSet.")
    temperature = float(prepared_dsl.analysis.temperature_K)
    temperature = validate_temperature(
        temperature,
        field="Prepared DSL nominal temperature",
    )
    if type(wegscheider_cyclicity_enabled) is not bool:
        raise TypeError("wegscheider_cyclicity_enabled must be a bool.")
    mechanism = prepared_dsl.mechanism_for_execution()
    metadata = getattr(mechanism, "metadata", {}) or {}
    if isinstance(metadata, dict):
        metadata[MechanismMetadataKeys.WEGSCHEIDER_CYCLICITY_ENABLED] = bool(
            wegscheider_cyclicity_enabled
        )
    metadata_view = MechanismMetadataView.from_metadata(metadata)
    apply_parameter_algebra_spec_to_mechanism(
        prepared_dsl.parameter_algebra_spec_for_execution(),
        mechanism=mechanism,
        require_mutable=False,
    )
    namespace = build_namespace_from_mechanism(mechanism)
    step_by_index: dict[int, Mapping[str, object]] = {}
    for entry in get_step_index_map(mechanism):
        parsed, ok = try_parse_int(entry.get("step_index"))
        if ok:
            step_by_index[int(parsed)] = entry
    units = solver_parameter_units_from_mechanism(mechanism)
    constrained = {
        str(name)
        for name in dict(metadata.get("constrained_params") or {})
    }
    reactions = tuple(getattr(mechanism, "reactions", ()) or ())
    equilibria = tuple(getattr(mechanism, "equilibria", ()) or ())
    coordinates: list[GuiParameterCoordinate] = []

    for item in namespace.ordered_items:
        info = item.info
        if info.step_index is None:
            continue
        step_index = int(info.step_index)
        entry = step_by_index.get(step_index, {})
        kind = str(info.step_kind or "")
        role = str(info.role or "")
        name = str(item.canonical_name)
        if name in constrained:
            continue
        context = str(entry.get("context") or "").strip()
        family = "Constant"
        derived = False
        writable = True
        scientific_category: GuiParameterScientificCategory
        allow_exact_zero = False
        model: ForwardRateModel | None = None
        value: float | None = None

        if kind == "reaction":
            raw_index, ok = try_parse_int(entry.get("reaction_index", -1))
            if not ok or not (0 <= raw_index < len(reactions)):
                continue
            model = forward_rate_model_for_reaction(
                reactions[int(raw_index)],
                default_standard_conc_M=metadata_view.standard_conc_M,
            )
            family = forward_rate_family(model)
            value = forward_rate_at_temperature(model, temperature)
            scientific_category = "rate"
            allow_exact_zero = family == "Constant"
        elif kind == "equilibrium":
            raw_index, ok = try_parse_int(entry.get("equilibrium_index", -1))
            if not ok or not (0 <= raw_index < len(equilibria)):
                continue
            equilibrium = equilibria[int(raw_index)]
            authority = normalize_existing_equilibrium_rate_authority(equilibrium)
            editable = step_entry_authored_role_is_editable(entry, role)
            derived_truth = step_entry_authored_role_is_derived(entry, role)
            writable = bool(
                authority.authored_role_is_editable(role)
                if editable is None
                else editable
            )
            derived = bool(
                authority.authored_role_is_derived(role)
                if derived_truth is None
                else derived_truth
            )
            if role == "kf":
                model = forward_rate_model_for_equilibrium(
                    equilibrium,
                    temperature_K=temperature,
                    default_standard_conc_M=metadata_view.standard_conc_M,
                )
                family = forward_rate_family(model)
                value = forward_rate_at_temperature(model, temperature)
                scientific_category = "rate"
                allow_exact_zero = (
                    authority.kind == EquilibriumRateAuthorityKind.KEQ
                    and family == "Constant"
                )
            elif role == "kr":
                family = "reverse"
                value = effective_equilibrium_reverse_rate(
                    equilibrium,
                    temperature_K=temperature,
                )
                scientific_category = "rate"
                allow_exact_zero = False
            elif role == "Keq":
                family = "reverse"
                value = effective_equilibrium_keq(
                    equilibrium,
                    temperature_K=temperature,
                )
                scientific_category = "equilibrium_constant"
                allow_exact_zero = False
            else:
                continue
        if value is None or not math.isfinite(float(value)):
            continue
        # Fit and Preview consume this API directly: derived readouts never
        # enter their request parameter maps.
        if not writable or derived:
            continue
        bound_minimum, bound_maximum = (
            (KEQ_MIN, KEQ_MAX) if role == "Keq" else (RATE_MIN, RATE_MAX)
        )
        low, high = (
            _positive_bounds(
                float(value),
                minimum=bound_minimum,
                maximum=bound_maximum,
            )
            if float(value) >= 0.0
            else _signed_bounds(float(value))
        )
        display_name = name
        if model is not None:
            if family in {"Arrhenius", "Eyring"}:
                display_name = f"{name} (k@{temperature:g} K)"
        if context:
            display_name = f"{display_name} — {context}"
        coordinates.append(
            GuiParameterCoordinate(
                name=name,
                display_name=display_name,
                value=float(value),
                minimum=low,
                maximum=high,
                log10_allowed=bool(float(value) > 0.0 and low > 0.0),
                family=family,
                role=role,
                writable=True,
                derived=False,
                scientific_category=scientific_category,
                allow_exact_zero=allow_exact_zero,
                unit=str(units.get(name, "1")),
            )
        )

    scalar_params = dict(metadata.get("scalar_params") or {})
    scalar_info = dict(metadata.get("scalar_param_info") or {})
    for raw_name, raw_value in scalar_params.items():
        name = str(raw_name)
        value = _finite(raw_value)
        info = dict(scalar_info.get(name) or {})
        if (
            value is None
            or name in constrained
            or info.get("derived") is True
            or info.get("editable") is False
        ):
            continue
        low, high = _signed_bounds(value)
        coordinates.append(
            GuiParameterCoordinate(
                name=name,
                display_name=name,
                value=value,
                minimum=low,
                maximum=high,
                log10_allowed=bool(value > 0.0 and low > 0.0),
                family="scalar",
                role="scalar",
                writable=True,
                derived=False,
                scientific_category="finite_scalar",
                allow_exact_zero=True,
                unit=str(units.get(name, "1")),
            )
        )
    return tuple(coordinates)
