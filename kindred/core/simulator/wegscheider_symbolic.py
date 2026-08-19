from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from math import exp, gcd, isclose, isfinite, log
from functools import reduce
import re
from typing import Mapping, Sequence

from kindred.core.constants import R
from kindred.core.equilibrium_rate_authority import (
    EquilibriumRateAuthorityKind,
    effective_equilibrium_forward_rate,
    effective_equilibrium_keq,
    effective_equilibrium_reverse_rate,
    normalize_existing_equilibrium_rate_authority,
)
from kindred.core.kinetics import (
    ArrheniusRateModel,
    ConstantRateModel,
    EyringRateModel,
    ForwardRateModel,
    forward_rate_model_for_equilibrium,
)
from kindred.core.mechanism_metadata import (
    MechanismMetadataView,
)
from kindred.core.ode_builder import (
    EffectiveRateCoefficientEvaluator,
    build_effective_rate_coefficient_evaluator,
)
from kindred.core.rate_binding import ForwardRateBinding, RateBinding
from kindred.core.simulator.errors import DSLError
from kindred.core.simulator.step_indexing import get_step_index_map
from kindred.core.temperature import TemperatureScheduleDictProtocol
from kindred.core.units import UnitsModel

__all__ = [
    "UnresolvedWegscheiderCyclicityError",
    "UnsupportedWegscheiderCorrectionError",
    "WegscheiderExecutableCycleSatisfaction",
    "WegscheiderExecutableSatisfactionError",
    "WegscheiderExecutableSatisfactionReport",
    "WegscheiderCycle",
    "WegscheiderCyclicityReport",
    "WegscheiderResolutionUpdate",
    "WegscheiderTopologyReport",
    "analyze_bound_wegscheider_satisfaction",
    "analyze_wegscheider_cyclicity",
    "analyze_wegscheider_topology",
    "apply_wegscheider_resolution_to_reactions_text",
    "build_wegscheider_resolution_updates",
    "require_wegscheider_admission_satisfied",
    "validate_bound_wegscheider_satisfaction",
]


@dataclass(frozen=True)
class WegscheiderCycle:
    cycle_id: str
    step_indices: tuple[int, ...]
    equilibrium_indices: tuple[int, ...]
    coefficients: tuple[int, ...]
    parameter_names: tuple[str, ...]
    # The cycle coefficients above remain aligned with the selected
    # equilibrium steps for runtime temperature checks.  Correction targets
    # may expose more than one writable coordinate per step (kf+kr), so their
    # exponents are carried separately and aligned with parameter_names.
    correction_exponents: tuple[int, ...] = ()
    authority_parameter_names: tuple[str, ...] = ()
    authority_exponents: tuple[int, ...] = ()
    resolved_by: str | None = None
    selected_step_stoichiometric_vectors: tuple[
        tuple[tuple[str, Fraction], ...], ...
    ] = ()
    admission_status: str = "unavailable"
    admission_reason: str | None = None
    admission_log_product: float | None = None
    admission_parameter_values: tuple[float | None, ...] = ()


@dataclass(frozen=True)
class WegscheiderTopologyReport:
    """Readable cycle topology derived from canonical reversible stoichiometry."""

    cycles: tuple[WegscheiderCycle, ...]


@dataclass(frozen=True)
class WegscheiderCyclicityReport:
    cycles: tuple[WegscheiderCycle, ...]

    @property
    def unresolved_cycles(self) -> tuple[WegscheiderCycle, ...]:
        return tuple(cycle for cycle in self.cycles if cycle.resolved_by is None)

    @property
    def is_resolved(self) -> bool:
        return not self.unresolved_cycles

@dataclass(frozen=True)
class WegscheiderExecutableCycleSatisfaction:
    cycle_id: str
    status: str
    temperatures_K: tuple[float, ...] = ()
    log_products: tuple[float, ...] = ()
    reason: str | None = None
    parameter_names: tuple[str, ...] = ()
    authority_parameter_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "cycle_id", str(self.cycle_id))
        object.__setattr__(self, "status", str(self.status))
        object.__setattr__(
            self,
            "parameter_names",
            tuple(str(name) for name in self.parameter_names if str(name).strip()),
        )
        object.__setattr__(
            self,
            "authority_parameter_names",
            tuple(
                str(name)
                for name in self.authority_parameter_names
                if str(name).strip()
            ),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "cycle_id": self.cycle_id,
            "status": self.status,
            "temperatures_K": tuple(float(value) for value in self.temperatures_K),
            "log_products": tuple(float(value) for value in self.log_products),
            "reason": self.reason,
            "parameter_names": self.parameter_names,
            "authority_parameter_names": self.authority_parameter_names,
        }


@dataclass(frozen=True)
class WegscheiderExecutableSatisfactionReport:
    """Later satisfaction against one bound executable request/candidate."""

    admission_report: WegscheiderCyclicityReport
    cycles: tuple[WegscheiderExecutableCycleSatisfaction, ...]

    @property
    def failed_cycles(self) -> tuple[WegscheiderExecutableCycleSatisfaction, ...]:
        return tuple(cycle for cycle in self.cycles if cycle.status != "satisfied")

    @property
    def is_satisfied(self) -> bool:
        return not self.failed_cycles

@dataclass(frozen=True)
class WegscheiderResolutionUpdate:
    cycle_id: str
    parameter_name: str
    source_value: float | None = None
    # Source-owned dG_eq corrections are materialized by the native Reactions
    # writer.  Keep the exact linear source-coordinate relation alongside the
    # display expression so the writer never has to reparse accepted source or
    # reconstruct the detailed-balance report.
    source_terms: tuple[tuple[str, Fraction], ...] = ()

class UnresolvedWegscheiderCyclicityError(DSLError):
    def __init__(
        self,
        cycles: Sequence[WegscheiderCycle],
        *,
        report: WegscheiderCyclicityReport | None = None,
    ) -> None:
        unresolved = tuple(cycles)
        names = ", ".join(
            f"{cycle.cycle_id} ({', '.join(cycle.parameter_names)})"
            for cycle in unresolved
        )
        super().__init__(
            f"unresolved Wegscheider cyclicity: {names}",
            suggestion=(
                "Adjust one admitted directional-rate or thermodynamic "
                "coordinate per independent cycle so the effective "
                "coefficients satisfy the structural cycle identity."
            ),
        )
        self.stage = "wegscheider_admission"
        self.cycles = unresolved
        self.report = report


class WegscheiderExecutableSatisfactionError(DSLError):
    def __init__(self, report: WegscheiderExecutableSatisfactionReport) -> None:
        failed = report.failed_cycles
        details = ", ".join(
            f"{cycle.cycle_id}:{cycle.status}"
            for cycle in failed
        )
        super().__init__(
            f"bound executable Wegscheider satisfaction failed: {details}",
            suggestion=(
                "Use a source-admitted detailed-balance correction or change the "
                "request/candidate temperature and writable canonical coordinates."
            ),
        )
        self.stage = "wegscheider_executable"
        self.report = report
        self.cycles = failed


class UnsupportedWegscheiderCorrectionError(ValueError):
    """The selected correction has no canonical source coordinate to rewrite."""


@dataclass(frozen=True)
class _ReversibleStep:
    step_index: int
    equilibrium_index: int
    vector: Mapping[str, Fraction]
    parameter_name: str


@dataclass(frozen=True, slots=True)
class _LogTemperatureRelation:
    """log(kf/kr) = log_constant + log_temperature_power*log(T) + inverse_temperature/T."""

    log_constant: float
    log_temperature_power: float
    inverse_temperature: float
    source: str


def _as_fraction_stoich_map(values: Mapping[object, object]) -> dict[str, Fraction]:
    out: dict[str, Fraction] = {}
    for raw_name, raw_value in (values or {}).items():
        try:
            value = Fraction(str(raw_value))
        except (TypeError, ValueError) as exc:
            raise DSLError(f"Invalid stoichiometric coefficient for {raw_name!r}: {raw_value!r}") from exc
        if value:
            out[str(raw_name)] = value
    return out


def _reversible_steps(mechanism: object) -> tuple[_ReversibleStep, ...]:
    eqs = list(getattr(mechanism, "equilibria", []) or [])
    authored_by_equilibrium_index: dict[int, int] = {}
    for entry in get_step_index_map(mechanism):
        try:
            step_index = int(entry.get("step_index"))
        except (TypeError, ValueError):
            continue
        if str(entry.get("kind") or "") != "equilibrium":
            continue
        try:
            equilibrium_index = int(entry.get("equilibrium_index"))
        except (TypeError, ValueError):
            continue
        if 0 <= equilibrium_index < len(eqs):
            authored_by_equilibrium_index[equilibrium_index] = step_index

    steps: list[_ReversibleStep] = []
    for equilibrium_index, step_index in sorted(
        authored_by_equilibrium_index.items(),
        key=lambda item: item[1],
    ):
        parameter_name = f"Keq{step_index}"
        eq = eqs[equilibrium_index]
        forward = _as_fraction_stoich_map(getattr(eq, "stoich_forward", {}) or {})
        back = _as_fraction_stoich_map(getattr(eq, "stoich_back", {}) or {})
        species = sorted(set(forward) | set(back))
        vector = {
            name: back.get(name, Fraction(0)) - forward.get(name, Fraction(0))
            for name in species
            if back.get(name, Fraction(0)) - forward.get(name, Fraction(0)) != 0
        }
        steps.append(
            _ReversibleStep(
                step_index=step_index,
                equilibrium_index=equilibrium_index,
                vector=vector,
                parameter_name=parameter_name,
            )
        )
    return tuple(sorted(steps, key=lambda step: step.step_index))


def _correction_coordinates(
    mechanism: object,
    selected_steps: Sequence[_ReversibleStep],
    cycle_coefficients: Sequence[int],
) -> tuple[
    tuple[str, ...],
    tuple[int, ...],
    tuple[str, ...],
    tuple[int, ...],
]:
    """Return the writable authority coordinates for one cycle.

    Wegscheider's scientific identity is expressed in equilibrium ratios, but
    the correction UI must write the accepted authority basis.  Explicit
    ``kf+kr`` contributes both coordinates to the identity, but only a literal
    constant ``kf`` is offered as a source correction.  ``kr`` remains the
    canonical correction for a typed Arrhenius/Eyring forward model.
    """

    equilibria = tuple(getattr(mechanism, "equilibria", ()) or ())
    selectable_names: list[str] = []
    selectable_exponents: list[int] = []
    authority_names: list[str] = []
    authority_exponents: list[int] = []
    metadata = MechanismMetadataView.from_metadata(
        getattr(mechanism, "metadata", {}) or {}
    )
    for step, coefficient in zip(selected_steps, cycle_coefficients, strict=True):
        if not (0 <= int(step.equilibrium_index) < len(equilibria)):
            raise ValueError("Wegscheider cycle references an invalid equilibrium index.")
        equilibrium = equilibria[int(step.equilibrium_index)]
        authority = normalize_existing_equilibrium_rate_authority(equilibrium)
        c = int(coefficient)
        if authority.kind == EquilibriumRateAuthorityKind.KR:
            kf_name = f"kf{step.step_index}"
            kr_name = f"kr{step.step_index}"
            authority_names.extend((kf_name, kr_name))
            authority_exponents.extend((c, -c))
            try:
                model = forward_rate_model_for_equilibrium(
                    equilibrium,
                    temperature_K=float(metadata.temperature_K),
                    default_standard_conc_M=float(metadata.standard_conc_M),
                )
            except (TypeError, ValueError, OverflowError):
                model = None
            if isinstance(model, ConstantRateModel):
                selectable_names.append(kf_name)
                selectable_exponents.append(c)
            selectable_names.append(kr_name)
            selectable_exponents.append(-c)
        elif authority.dG_eq_J_per_mol is not None and not authority.has_explicit_keq_param:
            name = f"dG_eq{step.step_index}"
            selectable_names.append(name)
            selectable_exponents.append(c)
            authority_names.append(name)
            authority_exponents.append(c)
        else:
            name = f"Keq{step.step_index}"
            selectable_names.append(name)
            selectable_exponents.append(c)
            authority_names.append(name)
            authority_exponents.append(c)
    return (
        tuple(selectable_names),
        tuple(selectable_exponents),
        tuple(authority_names),
        tuple(authority_exponents),
    )


def _cycle_correction_terms(
    cycle: WegscheiderCycle,
) -> tuple[tuple[str, int], ...]:
    names = cycle.authority_parameter_names or cycle.parameter_names
    exponents = cycle.authority_exponents or cycle.correction_exponents
    if len(exponents) != len(names):
        raise ValueError(
            f"{cycle.cycle_id} authority-coordinate metadata is not aligned."
        )
    return tuple(
        (str(name), int(exponent))
        for name, exponent in zip(names, exponents, strict=True)
        if int(exponent) != 0
    )


def _lcm(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return abs(a * b) // gcd(abs(a), abs(b))


def _normalize_integer_vector(values: Sequence[Fraction]) -> tuple[int, ...]:
    denom_lcm = 1
    for value in values:
        denom_lcm = _lcm(denom_lcm, value.denominator)
    ints = [int(value * denom_lcm) for value in values]
    divisor = reduce(gcd, (abs(value) for value in ints if value), 0) or 1
    ints = [value // divisor for value in ints]
    first = next((value for value in ints if value), 0)
    if first < 0:
        ints = [-value for value in ints]
    return tuple(int(value) for value in ints)


def _nullspace_integer_basis(matrix: Sequence[Sequence[Fraction]]) -> tuple[tuple[int, ...], ...]:
    if not matrix:
        return ()
    rows = [[Fraction(value) for value in row] for row in matrix]
    row_count = len(rows)
    col_count = len(rows[0]) if rows else 0
    pivot_cols: list[int] = []
    pivot_row = 0
    for col in range(col_count):
        pivot = None
        for row in range(pivot_row, row_count):
            if rows[row][col] != 0:
                pivot = row
                break
        if pivot is None:
            continue
        rows[pivot_row], rows[pivot] = rows[pivot], rows[pivot_row]
        scale = rows[pivot_row][col]
        rows[pivot_row] = [value / scale for value in rows[pivot_row]]
        for row in range(row_count):
            if row == pivot_row or rows[row][col] == 0:
                continue
            factor = rows[row][col]
            rows[row] = [
                current - factor * pivot_value
                for current, pivot_value in zip(rows[row], rows[pivot_row])
            ]
        pivot_cols.append(col)
        pivot_row += 1
        if pivot_row == row_count:
            break

    free_cols = [col for col in range(col_count) if col not in pivot_cols]
    basis: list[tuple[int, ...]] = []
    for free_col in free_cols:
        vec = [Fraction(0) for _ in range(col_count)]
        vec[free_col] = Fraction(1)
        for row, pivot_col in enumerate(pivot_cols):
            vec[pivot_col] = -rows[row][free_col]
        normalized = _normalize_integer_vector(vec)
        if any(normalized):
            basis.append(normalized)
    return tuple(basis)


def _dG_source_terms(
    terms: Sequence[tuple[str, int]],
    dependent_parameter: str,
) -> tuple[tuple[str, Fraction], ...]:
    """Return the source-unit linear relation for a dependent dG_eq."""

    names = tuple(name for name, _coefficient in terms)
    if dependent_parameter not in names:
        raise ValueError(f"{dependent_parameter!r} is not part of the cycle.")
    dependent_coefficient = next(
        int(coefficient)
        for name, coefficient in terms
        if name == dependent_parameter
    )
    if dependent_coefficient == 0:
        raise ValueError(f"{dependent_parameter!r} has zero cycle coefficient.")
    return tuple(
        (name, Fraction(-int(coefficient), dependent_coefficient))
        for name, coefficient in terms
        if name != dependent_parameter and int(coefficient) != 0
    )


def _dependent_correction_value(
    cycle: WegscheiderCycle,
    dependent_parameter: str,
) -> float:
    terms = _cycle_correction_terms(cycle)
    names = tuple(name for name, _coefficient in terms)
    if dependent_parameter not in names:
        raise ValueError(f"{dependent_parameter!r} is not part of {cycle.cycle_id}.")
    if dependent_parameter not in cycle.parameter_names:
        raise UnsupportedWegscheiderCorrectionError(
            f"{dependent_parameter!r} is not a supported canonical source correction."
        )
    if len(cycle.admission_parameter_values) != len(names):
        raise UnsupportedWegscheiderCorrectionError(
            f"{cycle.cycle_id} has no complete admitted source-coordinate values."
        )
    dependent_index = names.index(dependent_parameter)
    dependent_exponent = int(terms[dependent_index][1])
    if dependent_exponent == 0:
        raise UnsupportedWegscheiderCorrectionError(
            f"{dependent_parameter!r} has no correction exponent."
        )

    log_other_product = 0.0
    for index, (_name, exponent) in enumerate(terms):
        if index == dependent_index:
            continue
        value = cycle.admission_parameter_values[index]
        if value is None or not isfinite(value) or value <= 0.0:
            raise UnsupportedWegscheiderCorrectionError(
                f"{cycle.cycle_id} cannot correct {dependent_parameter!r}: "
                "another canonical source coordinate is not positive and finite."
            )
        log_other_product += int(exponent) * log(float(value))
    try:
        value = exp(-log_other_product / dependent_exponent)
    except (OverflowError, ZeroDivisionError) as exc:
        raise UnsupportedWegscheiderCorrectionError(
            f"{cycle.cycle_id} correction for {dependent_parameter!r} is not finite."
        ) from exc
    if not isfinite(value) or value <= 0.0:
        raise UnsupportedWegscheiderCorrectionError(
            f"{cycle.cycle_id} correction for {dependent_parameter!r} is not positive and finite."
        )
    return float(value)


def build_wegscheider_resolution_updates(
    report: WegscheiderCyclicityReport,
    dependent_parameters: Mapping[str, str],
) -> tuple[WegscheiderResolutionUpdate, ...]:
    cycles_by_id = {cycle.cycle_id: cycle for cycle in report.cycles}
    updates: list[WegscheiderResolutionUpdate] = []
    requested_parameters: dict[str, str] = {}
    for cycle_id, parameter_name in dependent_parameters.items():
        cycle = cycles_by_id.get(str(cycle_id))
        if cycle is None:
            raise ValueError(f"Unknown Wegscheider cycle id {cycle_id!r}.")
        parameter_key = str(parameter_name)
        previous_cycle = requested_parameters.get(parameter_key)
        if previous_cycle is not None:
            raise ValueError(
                f"Wegscheider dependent parameter {parameter_name!r} was selected for both "
                f"{previous_cycle!r} and {cycle_id!r}; choose one dependent parameter per cycle."
            )
        requested_parameters[parameter_key] = str(cycle_id)
        terms = _cycle_correction_terms(cycle)
        is_dg = str(parameter_name).startswith("dG_eq")
        dg_terms = tuple(
            name.startswith("dG_eq")
            for name, _coefficient in terms
        )
        if any(dg_terms) and not all(dg_terms):
            raise UnsupportedWegscheiderCorrectionError(
                f"{cycle.cycle_id} mixes energy and concentration/rate source authorities; "
                "no one-way canonical correction is available."
            )
        if is_dg != bool(dg_terms and all(dg_terms)):
            raise UnsupportedWegscheiderCorrectionError(
                f"{parameter_name!r} does not belong to a supported canonical correction basis."
            )
        updates.append(
            WegscheiderResolutionUpdate(
                cycle_id=str(cycle_id),
                parameter_name=str(parameter_name),
                source_value=(
                    None
                    if is_dg
                    else _dependent_correction_value(
                        cycle,
                        str(parameter_name),
                    )
                ),
                source_terms=(
                    _dG_source_terms(
                        terms,
                        str(parameter_name),
                    )
                    if is_dg
                    else ()
                ),
            )
        )
    return tuple(updates)


def apply_wegscheider_resolution_to_reactions_text(
    dsl_text: str,
    updates: Sequence[WegscheiderResolutionUpdate],
    *,
    units: UnitsModel,
    admitted_analysis: object,
) -> str:
    if type(units) is not UnitsModel:
        raise TypeError("Detailed-balance correction units must be UnitsModel.")
    normalized_updates = tuple(updates)
    names = tuple(str(update.parameter_name) for update in normalized_updates)
    if len(names) != len(set(names)):
        raise ValueError("Detailed-balance source updates must have unique targets.")
    if not normalized_updates:
        return str(dsl_text or "")
    return _materialize_admitted_canonical_source_updates(
        str(dsl_text or ""),
        normalized_updates,
        units=units,
        admitted_analysis=admitted_analysis,
    )


def _materialize_admitted_canonical_source_updates(
    dsl_text: str,
    updates: Sequence[WegscheiderResolutionUpdate],
    *,
    units: UnitsModel,
    admitted_analysis: object,
) -> str:
    """Rewrite only native source coordinates; never manufacture Algebra proof."""

    from kindred.core.simulator.dsl_derivation import (
        authored_step_origin,
        coordinate_source_parts,
        coordinate_token_matches,
        origin_line_positions,
        rewrite_coordinate_source_line,
    )
    from kindred.core.mechanism_source import (
        MechanismAuthoringSource,
        source_matches_analysis,
    )

    if not source_matches_analysis(
        MechanismAuthoringSource.from_parts(reactions_text=dsl_text),
        admitted_analysis,
    ):
        raise ValueError(
            "Detailed-balance source does not match its admitted analysis."
        )
    analysis = admitted_analysis
    if (
        str(units.energy_unit) != str(analysis.energy_unit)
        or float(units.temperature_K) != float(analysis.temperature_K)
    ):
        raise ValueError(
            "Detailed-balance correction units do not match the admitted analysis."
        )
    raw_lines = str(dsl_text).splitlines(keepends=True)
    line_contents = [line.rstrip("\r\n") for line in raw_lines]
    line_endings = [line[len(content):] for line, content in zip(raw_lines, line_contents)]
    if not raw_lines and dsl_text:
        line_contents = [dsl_text]
        line_endings = [""]
    origin_positions = origin_line_positions(analysis, dsl_text)
    def source_coordinate_location(name: str) -> tuple[int, str, float]:
        match = re.fullmatch(r"(kf|kr|Keq|dG_eq)(\d+)", str(name))
        if match is None:
            raise UnsupportedWegscheiderCorrectionError(
                f"Unsupported native detailed-balance coordinate {name!r}."
            )
        role = str(match.group(1))
        step_index = int(match.group(2))
        try:
            origin = authored_step_origin(analysis, step_index=step_index)
        except LookupError as exc:
            raise UnsupportedWegscheiderCorrectionError(
                f"Source has no authored coordinate {name!r}."
            ) from exc
        try:
            line_index = origin_positions[origin.authored_ordinal]
        except KeyError as exc:
            raise UnsupportedWegscheiderCorrectionError(
                f"Source has no exact origin for {name!r}."
            ) from exc
        if not 0 <= line_index < len(line_contents):
            raise UnsupportedWegscheiderCorrectionError(
                f"Source origin for {name!r} is outside the exact source."
            )
        source_token = "Keq" if role == "keq" else "dG_eq" if role == "dg_eq" else role
        _prefix, tokens, _comment = coordinate_source_parts(line_contents[line_index])
        matches = [
            pair for pair in tokens if coordinate_token_matches(pair[0], source_token)
        ]
        if len(matches) != 1:
            raise UnsupportedWegscheiderCorrectionError(
                f"Source does not contain one supported literal coordinate {name!r}."
            )
        try:
            value = float(matches[0][1])
        except (TypeError, ValueError, OverflowError) as exc:
            raise UnsupportedWegscheiderCorrectionError(
                f"Source {name} value is not numeric."
            ) from exc
        if not isfinite(value):
            raise UnsupportedWegscheiderCorrectionError(
                f"Source {name} value is not finite."
            )
        return line_index, source_token, value

    coordinate_names: set[str] = {
        str(update.parameter_name)
        for update in updates
    }
    for update in updates:
        coordinate_names.update(
            str(name)
            for name, _factor in update.source_terms
        )

    admitted_coordinates: dict[str, tuple[int, str, float]] = {}
    for coordinate_name in sorted(coordinate_names):
        admitted_coordinates[coordinate_name] = source_coordinate_location(
            coordinate_name
        )

    # All values are retained in the one admitted source energy unit.  Do not
    # convert through J/mol here: the canonical source directive remains in
    # ``analysis.energy_unit`` and the rendered coordinate must stay in it.
    _source_energy_unit = analysis.energy_unit
    if not _source_energy_unit:
        raise ValueError("Admitted DSL source has no energy render unit.")
    materialized_values: dict[str, float] = {}
    for update in updates:
        target = str(update.parameter_name)
        coordinate = admitted_coordinates.get(target)
        if coordinate is None:
            raise ValueError(
                f"Source does not contain authored coordinate {update.parameter_name!r}."
            )
        if update.source_terms:
            materialized = 0.0
            for source_name, factor in update.source_terms:
                source_key = str(source_name)
                source = admitted_coordinates.get(source_key)
                if source is None:
                    raise UnsupportedWegscheiderCorrectionError(
                        f"Source does not contain authored coordinate {source_name!r}."
                    )
                materialized += float(factor) * source[2]
        elif update.source_value is not None:
            materialized = float(update.source_value)
        else:
            raise UnsupportedWegscheiderCorrectionError(
                f"Detailed-balance update {update.parameter_name!r} has no canonical source value."
            )
        if not isfinite(materialized):
            raise UnsupportedWegscheiderCorrectionError(
                f"Detailed-balance correction for {update.parameter_name!r} is not finite."
            )
        materialized_values[target] = float(materialized)

    for update in updates:
        target = str(update.parameter_name)
        line_index, source_token, _value = admitted_coordinates[target]
        line_contents[line_index] = rewrite_coordinate_source_line(
            line_contents[line_index],
            token=source_token,
            value=materialized_values[target],
        )
    return "".join(
        content + ending
        for content, ending in zip(line_contents, line_endings, strict=True)
    )


def _forward_model_temperature_exponents(
    model: ForwardRateModel,
) -> tuple[float, float]:
    if isinstance(model, ConstantRateModel):
        return 0.0, 0.0
    if isinstance(model, ArrheniusRateModel):
        return 0.0, -float(model.Ea_J_per_mol) / float(R)
    if isinstance(model, EyringRateModel):
        return 1.0, -float(model.dG_act_J_per_mol) / float(R)
    raise TypeError("Unsupported typed forward-rate model.")


def _runtime_temperature_domain(
    mechanism: object,
) -> tuple[str, tuple[float, ...], Mapping[str, object] | None]:
    metadata = MechanismMetadataView.from_metadata(
        getattr(mechanism, "metadata", {}) or {}
    )
    schedule = metadata.temperature_schedule
    if schedule is None:
        return "finite", (float(metadata.temperature_K),), None
    if not isinstance(schedule, TemperatureScheduleDictProtocol):
        return "continuum", (), None

    payload = dict(schedule.to_dict())
    schedule_type = str(payload.get("type") or "")
    if schedule_type == "constant":
        return "finite", (float(payload["temperature"]),), payload
    if schedule_type == "piecewise":
        intervals = tuple(payload.get("intervals") or ())
        temperatures: list[float] = []
        if intervals:
            first = intervals[0]
            if (
                isinstance(first, Mapping)
                and float(first.get("t_start", 0.0)) > 0.0
            ):
                temperatures.append(float(payload["baseline_temperature"]))
        for interval in intervals:
            if not isinstance(interval, Mapping):
                return "continuum", (), payload
            temperatures.append(float(interval["temperature"]))
        return "finite", tuple(sorted(set(temperatures))), payload
    if schedule_type == "response":
        baseline = float(payload["baseline_temperature"])
        setpoints = tuple(
            float(value) for value in tuple(payload.get("temperatures") or ())
        )
        if all(value == baseline for value in setpoints):
            return "finite", (baseline,), payload
        return "continuum", (), payload
    return "continuum", (), payload


def _effective_equilibrium_ratio(
    evaluator: EffectiveRateCoefficientEvaluator,
    *,
    reaction_count: int,
    equilibrium_index: int,
    temperature_K: float,
) -> float | None:
    coefficients = evaluator(float(temperature_K))
    coefficient_index = int(reaction_count) + int(equilibrium_index)
    forward = float(coefficients.forward[coefficient_index])
    reverse = float(coefficients.reverse[coefficient_index])
    if (
        not isfinite(forward)
        or not isfinite(reverse)
        or forward <= 0.0
        or reverse <= 0.0
    ):
        return None
    ratio = forward / reverse
    return float(ratio) if isfinite(ratio) and ratio > 0.0 else None


def _step_log_temperature_relation(
    mechanism: object,
    step: _ReversibleStep,
    *,
    evaluator: EffectiveRateCoefficientEvaluator,
    nominal_temperature_K: float,
    nominal_ratio: float,
) -> _LogTemperatureRelation | None:
    equilibria = tuple(getattr(mechanism, "equilibria", ()) or ())
    equilibrium = equilibria[step.equilibrium_index]
    log_temperature_power = 0.0
    inverse_temperature = 0.0
    source = ""

    authority = normalize_existing_equilibrium_rate_authority(equilibrium)
    if authority.kind == EquilibriumRateAuthorityKind.KEQ:
        if (
            authority.dG_eq_J_per_mol is not None
            and not authority.has_explicit_keq_param
        ):
            if callable(authority.dG_eq_J_per_mol):
                return None
            try:
                inverse_temperature = -float(
                    authority.dG_eq_J_per_mol
                ) / float(R)
            except (TypeError, ValueError, OverflowError):
                return None
            source = "authored-delta-g"
        else:
            if (
                callable(authority.Keq)
                and bool(
                    getattr(
                        authority.Keq,
                        "accepts_runtime_temperature",
                        False,
                    )
                )
            ):
                return None
            source = "authored-explicit-keq"
    elif authority.kind == EquilibriumRateAuthorityKind.KR:
        forward_binding = getattr(equilibrium, "kf", None)
        if isinstance(forward_binding, ForwardRateBinding):
            forward_model = forward_binding.model
            forward_exponents = _forward_model_temperature_exponents(
                forward_model
            )
        elif isinstance(forward_binding, RateBinding):
            forward_exponents = (0.0, 0.0)
        else:
            try:
                forward_model = forward_rate_model_for_equilibrium(
                    equilibrium,
                    temperature_K=nominal_temperature_K,
                    default_standard_conc_M=MechanismMetadataView.from_metadata(
                        getattr(mechanism, "metadata", {}) or {}
                    ).standard_conc_M,
                )
            except (TypeError, ValueError):
                return None
            forward_exponents = _forward_model_temperature_exponents(
                forward_model
            )

        reverse_binding = authority.kr
        if isinstance(reverse_binding, ForwardRateBinding):
            reverse_exponents = _forward_model_temperature_exponents(
                reverse_binding.model
            )
        elif (
            callable(reverse_binding)
            and bool(
                getattr(
                    reverse_binding,
                    "accepts_runtime_temperature",
                    False,
                )
            )
        ):
            return None
        else:
            reverse_exponents = (0.0, 0.0)
        log_temperature_power = (
            forward_exponents[0] - reverse_exponents[0]
        )
        inverse_temperature = (
            forward_exponents[1] - reverse_exponents[1]
        )
        source = "authored-forward-reverse-rates"
    else:
        return None

    if not isfinite(nominal_ratio) or nominal_ratio <= 0.0:
        return None
    log_constant = (
        log(float(nominal_ratio))
        - log_temperature_power * log(float(nominal_temperature_K))
        - inverse_temperature / float(nominal_temperature_K)
    )
    if not all(
        isfinite(value)
        for value in (
            log_constant,
            log_temperature_power,
            inverse_temperature,
        )
    ):
        return None
    return _LogTemperatureRelation(
        log_constant=float(log_constant),
        log_temperature_power=float(log_temperature_power),
        inverse_temperature=float(inverse_temperature),
        source=source,
    )


def _combined_cycle_relation(
    mechanism: object,
    cycle: WegscheiderCycle,
    selected_steps: Sequence[_ReversibleStep],
    *,
    evaluator: EffectiveRateCoefficientEvaluator,
    nominal_temperature_K: float,
    nominal_ratios: Mapping[int, float],
) -> tuple[_LogTemperatureRelation, tuple[_LogTemperatureRelation, ...]] | None:
    relations: list[_LogTemperatureRelation] = []
    for step in selected_steps:
        relation = _step_log_temperature_relation(
            mechanism,
            step,
            evaluator=evaluator,
            nominal_temperature_K=nominal_temperature_K,
            nominal_ratio=nominal_ratios[step.equilibrium_index],
        )
        if relation is None:
            return None
        relations.append(relation)
    combined = _LogTemperatureRelation(
        log_constant=sum(
            int(coefficient) * relation.log_constant
            for coefficient, relation in zip(
                cycle.coefficients,
                relations,
                strict=True,
            )
        ),
        log_temperature_power=sum(
            int(coefficient) * relation.log_temperature_power
            for coefficient, relation in zip(
                cycle.coefficients,
                relations,
                strict=True,
            )
        ),
        inverse_temperature=sum(
            int(coefficient) * relation.inverse_temperature
            for coefficient, relation in zip(
                cycle.coefficients,
                relations,
                strict=True,
            )
        ),
        source="combined-cycle",
    )
    return combined, tuple(relations)


def analyze_wegscheider_topology(mechanism: object) -> WegscheiderTopologyReport:
    """Find structural cycles before evaluating any rate or equilibrium value."""

    steps = _reversible_steps(mechanism)
    if len(steps) < 2:
        return WegscheiderTopologyReport(cycles=())
    species = sorted({species_name for step in steps for species_name in step.vector.keys()})
    if not species:
        return WegscheiderTopologyReport(cycles=())
    matrix = [
        [step.vector.get(species_name, Fraction(0)) for step in steps]
        for species_name in species
    ]
    basis = _nullspace_integer_basis(matrix)
    cycles: list[WegscheiderCycle] = []
    for idx, coeffs in enumerate(basis, start=1):
        selected = [
            (step, coeff)
            for step, coeff in zip(steps, coeffs)
            if int(coeff) != 0
        ]
        if len(selected) < 2:
            continue
        selected_steps = tuple(step for step, _coefficient in selected)
        selected_coefficients = tuple(
            int(coefficient) for _step, coefficient in selected
        )
        cycle = WegscheiderCycle(
            cycle_id=f"cycle_{idx}",
            step_indices=tuple(step.step_index for step, _coeff in selected),
            equilibrium_indices=tuple(step.equilibrium_index for step, _coeff in selected),
            coefficients=selected_coefficients,
            parameter_names=(),
            selected_step_stoichiometric_vectors=tuple(
                tuple(sorted((str(name), Fraction(value)) for name, value in step.vector.items()))
                for step in selected_steps
            ),
        )
        cycles.append(cycle)
    return WegscheiderTopologyReport(cycles=tuple(cycles))


def _steps_for_cycle(
    steps: Sequence[_ReversibleStep],
    cycle: WegscheiderCycle,
) -> tuple[_ReversibleStep, ...]:
    by_equilibrium_index = {step.equilibrium_index: step for step in steps}
    try:
        return tuple(
            by_equilibrium_index[index]
            for index in cycle.equilibrium_indices
        )
    except KeyError as exc:
        raise ValueError(
            f"{cycle.cycle_id} is not present in the canonical reversible topology."
        ) from exc


def _canonical_admitted_ratio(
    mechanism: object,
    step: _ReversibleStep,
    *,
    temperature_K: float,
) -> float | None:
    equilibria = tuple(getattr(mechanism, "equilibria", ()) or ())
    equilibrium = equilibria[step.equilibrium_index]
    value = effective_equilibrium_keq(
        equilibrium,
        temperature_K=temperature_K,
    )
    if value is None or not isfinite(value) or value <= 0.0:
        return None
    return float(value)


def _admission_correction_values(
    mechanism: object,
    selected_steps: Sequence[_ReversibleStep],
    *,
    temperature_K: float,
) -> tuple[float | None, ...]:
    equilibria = tuple(getattr(mechanism, "equilibria", ()) or ())
    values: list[float | None] = []
    for step in selected_steps:
        equilibrium = equilibria[step.equilibrium_index]
        authority = normalize_existing_equilibrium_rate_authority(equilibrium)
        if authority.kind == EquilibriumRateAuthorityKind.KR:
            values.extend(
                (
                    effective_equilibrium_forward_rate(
                        equilibrium,
                        temperature_K=temperature_K,
                    ),
                    effective_equilibrium_reverse_rate(
                        equilibrium,
                        temperature_K=temperature_K,
                    ),
                )
            )
        elif (
            authority.dG_eq_J_per_mol is not None
            and not authority.has_explicit_keq_param
        ):
            value = authority.dG_eq_J_per_mol
            try:
                values.append(
                    float(value(temperature_K)) if callable(value) else float(value)
                )
            except (TypeError, ValueError, OverflowError):
                values.append(None)
        else:
            values.append(
                effective_equilibrium_keq(
                    equilibrium,
                    temperature_K=temperature_K,
                )
            )
    return tuple(
        float(value) if value is not None and isfinite(value) else None
        for value in values
    )


def analyze_wegscheider_cyclicity(
    mechanism: object,
) -> WegscheiderCyclicityReport:
    """Evaluate source admission after topology and before downstream mutation."""

    topology = analyze_wegscheider_topology(mechanism)
    if not topology.cycles:
        return WegscheiderCyclicityReport(
            cycles=(),
        )
    steps = _reversible_steps(mechanism)
    nominal_temperature_K = float(
        MechanismMetadataView.from_metadata(
            getattr(mechanism, "metadata", {}) or {}
        ).temperature_K
    )
    cycles: list[WegscheiderCycle] = []
    for structural_cycle in topology.cycles:
        selected_steps = _steps_for_cycle(steps, structural_cycle)
        (
            correction_names,
            correction_exponents,
            authority_names,
            authority_exponents,
        ) = _correction_coordinates(
            mechanism,
            selected_steps,
            structural_cycle.coefficients,
        )
        ratios = tuple(
            _canonical_admitted_ratio(
                mechanism,
                step,
                temperature_K=nominal_temperature_K,
            )
            for step in selected_steps
        )
        correction_values = _admission_correction_values(
            mechanism,
            selected_steps,
            temperature_K=nominal_temperature_K,
        )
        if any(value is None for value in ratios):
            cycles.append(
                replace(
                    structural_cycle,
                    parameter_names=correction_names,
                    correction_exponents=correction_exponents,
                    authority_parameter_names=authority_names,
                    authority_exponents=authority_exponents,
                    admission_status="unavailable",
                    admission_reason=(
                        "canonical admitted equilibrium authority is not positive and finite"
                    ),
                    admission_parameter_values=correction_values,
                )
            )
            continue
        log_product = sum(
            coefficient * log(float(ratio))
            for coefficient, ratio in zip(
                structural_cycle.coefficients,
                ratios,
                strict=True,
            )
        )
        satisfied = isfinite(log_product) and isclose(
            log_product,
            0.0,
            rel_tol=0.0,
            abs_tol=1e-10,
        )
        cycles.append(
            replace(
                structural_cycle,
                parameter_names=correction_names,
                correction_exponents=correction_exponents,
                authority_parameter_names=authority_names,
                authority_exponents=authority_exponents,
                resolved_by=(
                    "canonical-admitted-equilibrium-authority"
                    if satisfied
                    else None
                ),
                admission_status="satisfied" if satisfied else "unsatisfied",
                admission_reason=(
                    None
                    if satisfied
                    else "canonical admitted cycle product is not unity"
                ),
                admission_log_product=float(log_product),
                admission_parameter_values=correction_values,
            )
        )
    return WegscheiderCyclicityReport(
        cycles=tuple(cycles),
    )


def require_wegscheider_admission_satisfied(
    report: WegscheiderCyclicityReport,
) -> WegscheiderCyclicityReport:
    if not isinstance(report, WegscheiderCyclicityReport):
        raise TypeError("Wegscheider admission requires its typed staged report.")
    if report.unresolved_cycles:
        raise UnresolvedWegscheiderCyclicityError(
            report.unresolved_cycles,
            report=report,
        )
    return report


def _finite_executable_cycle_satisfaction(
    cycle: WegscheiderCycle,
    selected_steps: Sequence[_ReversibleStep],
    *,
    evaluator: EffectiveRateCoefficientEvaluator,
    reaction_count: int,
    temperatures_K: Sequence[float],
) -> WegscheiderExecutableCycleSatisfaction:
    log_products: list[float] = []
    for temperature in temperatures_K:
        ratios = tuple(
            _effective_equilibrium_ratio(
                evaluator,
                reaction_count=reaction_count,
                equilibrium_index=step.equilibrium_index,
                temperature_K=temperature,
            )
            for step in selected_steps
        )
        if any(ratio is None for ratio in ratios):
            return WegscheiderExecutableCycleSatisfaction(
                cycle_id=cycle.cycle_id,
                status="unavailable",
                temperatures_K=tuple(float(value) for value in temperatures_K),
                log_products=tuple(log_products),
                reason=(
                    "bound forward/reverse coefficients are not positive and finite"
                ),
                parameter_names=cycle.parameter_names,
                authority_parameter_names=cycle.authority_parameter_names,
            )
        log_product = sum(
            coefficient * log(float(ratio))
            for coefficient, ratio in zip(
                cycle.coefficients,
                ratios,
                strict=True,
            )
        )
        log_products.append(float(log_product))
        if not isfinite(log_product) or not isclose(
            log_product,
            0.0,
            rel_tol=0.0,
            abs_tol=1e-10,
        ):
            return WegscheiderExecutableCycleSatisfaction(
                cycle_id=cycle.cycle_id,
                status="unsatisfied",
                temperatures_K=tuple(float(value) for value in temperatures_K),
                log_products=tuple(log_products),
                reason="bound cycle product is not unity",
                parameter_names=cycle.parameter_names,
                authority_parameter_names=cycle.authority_parameter_names,
            )
    return WegscheiderExecutableCycleSatisfaction(
        cycle_id=cycle.cycle_id,
        status="satisfied",
        temperatures_K=tuple(float(value) for value in temperatures_K),
        log_products=tuple(log_products),
        parameter_names=cycle.parameter_names,
        authority_parameter_names=cycle.authority_parameter_names,
    )


def _continuum_executable_cycle_satisfaction(
    mechanism: object,
    cycle: WegscheiderCycle,
    selected_steps: Sequence[_ReversibleStep],
    *,
    evaluator: EffectiveRateCoefficientEvaluator,
) -> WegscheiderExecutableCycleSatisfaction:
    nominal_temperature_K = float(evaluator.nominal_temperature_K)
    reaction_count = len(tuple(getattr(mechanism, "reactions", ()) or ()))
    nominal_ratios = {
        step.equilibrium_index: _effective_equilibrium_ratio(
            evaluator,
            reaction_count=reaction_count,
            equilibrium_index=step.equilibrium_index,
            temperature_K=nominal_temperature_K,
        )
        for step in selected_steps
    }
    if any(value is None for value in nominal_ratios.values()):
        return WegscheiderExecutableCycleSatisfaction(
            cycle_id=cycle.cycle_id,
            status="unavailable",
            temperatures_K=(nominal_temperature_K,),
            reason="bound coefficients cannot establish a continuum identity",
            parameter_names=cycle.parameter_names,
            authority_parameter_names=cycle.authority_parameter_names,
        )
    relations = _combined_cycle_relation(
        mechanism,
        cycle,
        selected_steps,
        evaluator=evaluator,
        nominal_temperature_K=nominal_temperature_K,
        nominal_ratios={key: float(value) for key, value in nominal_ratios.items() if value is not None},
    )
    if relations is None:
        return WegscheiderExecutableCycleSatisfaction(
            cycle_id=cycle.cycle_id,
            status="unavailable",
            temperatures_K=(nominal_temperature_K,),
            reason="bound temperature dependence has no exhaustive typed relation",
            parameter_names=cycle.parameter_names,
            authority_parameter_names=cycle.authority_parameter_names,
        )
    combined, _step_relations = relations
    identity_holds = (
        isclose(combined.log_constant, 0.0, rel_tol=0.0, abs_tol=1e-10)
        and isclose(
            combined.log_temperature_power,
            0.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and isclose(
            combined.inverse_temperature,
            0.0,
            rel_tol=0.0,
            abs_tol=1e-10,
        )
    )
    return WegscheiderExecutableCycleSatisfaction(
        cycle_id=cycle.cycle_id,
        status="satisfied" if identity_holds else "unsatisfied",
        temperatures_K=(nominal_temperature_K,),
        log_products=(
            float(
                combined.log_constant
                + combined.log_temperature_power * log(nominal_temperature_K)
                + combined.inverse_temperature / nominal_temperature_K
            ),
        ),
        reason=(None if identity_holds else "bound cycle identity varies over temperature"),
        parameter_names=cycle.parameter_names,
        authority_parameter_names=cycle.authority_parameter_names,
    )


def analyze_bound_wegscheider_satisfaction(
    mechanism: object,
    admission_report: WegscheiderCyclicityReport,
    *,
    temperatures_K: Sequence[float] | None = None,
) -> WegscheiderExecutableSatisfactionReport:
    """Evaluate actual-cycle satisfaction without changing admission truth."""

    if not isinstance(admission_report, WegscheiderCyclicityReport):
        raise TypeError("bound Wegscheider analysis requires its admission report.")
    current_topology = analyze_wegscheider_topology(mechanism)
    topology_matches = len(current_topology.cycles) == len(admission_report.cycles)
    if topology_matches:
        topology_matches = all(
            (
                current.step_indices == admitted.step_indices
                and current.equilibrium_indices == admitted.equilibrium_indices
                and current.coefficients == admitted.coefficients
                and current.selected_step_stoichiometric_vectors
                == admitted.selected_step_stoichiometric_vectors
            )
            for current, admitted in zip(
                current_topology.cycles,
                admission_report.cycles,
                strict=True,
            )
        )
    if not topology_matches:
        raise ValueError(
            "bound mechanism topology does not match its admitted Wegscheider topology."
        )
    if not admission_report.cycles:
        return WegscheiderExecutableSatisfactionReport(
            admission_report=admission_report,
            cycles=(),
        )

    blocked = tuple(
        WegscheiderExecutableCycleSatisfaction(
            cycle_id=cycle.cycle_id,
            status="blocked-by-admission",
            reason=(
                "downstream mutation cannot rescue an unsatisfied or unavailable admission"
            ),
            parameter_names=cycle.parameter_names,
            authority_parameter_names=cycle.authority_parameter_names,
        )
        for cycle in admission_report.cycles
        if cycle.resolved_by is None
    )
    if blocked:
        satisfied_ids = {
            cycle.cycle_id
            for cycle in admission_report.cycles
            if cycle.resolved_by is not None
        }
    else:
        satisfied_ids = {cycle.cycle_id for cycle in admission_report.cycles}

    try:
        evaluator = build_effective_rate_coefficient_evaluator(mechanism)
    except (TypeError, ValueError, OverflowError) as exc:
        unavailable = tuple(
            WegscheiderExecutableCycleSatisfaction(
                cycle_id=cycle.cycle_id,
                status="unavailable",
                reason=f"bound coefficient authority is unavailable: {exc}",
                parameter_names=cycle.parameter_names,
                authority_parameter_names=cycle.authority_parameter_names,
            )
            for cycle in admission_report.cycles
            if cycle.cycle_id in satisfied_ids
        )
        return WegscheiderExecutableSatisfactionReport(
            admission_report=admission_report,
            cycles=blocked + unavailable,
        )

    if temperatures_K is None:
        domain_kind, domain_temperatures, _payload = _runtime_temperature_domain(mechanism)
        explicit_temperatures = domain_temperatures if domain_kind == "finite" else ()
    else:
        domain_kind = "finite"
        explicit_temperatures = tuple(float(value) for value in temperatures_K)
        if not explicit_temperatures or any(
            not isfinite(value) or value <= 0.0 for value in explicit_temperatures
        ):
            raise ValueError(
                "bound Wegscheider temperatures must be a nonempty positive finite sequence."
            )
    reaction_count = len(tuple(getattr(mechanism, "reactions", ()) or ()))
    steps = _reversible_steps(mechanism)
    results: list[WegscheiderExecutableCycleSatisfaction] = list(blocked)
    for cycle in admission_report.cycles:
        if cycle.cycle_id not in satisfied_ids:
            continue
        selected_steps = _steps_for_cycle(steps, cycle)
        if domain_kind == "finite" and explicit_temperatures:
            result = _finite_executable_cycle_satisfaction(
                cycle,
                selected_steps,
                evaluator=evaluator,
                reaction_count=reaction_count,
                temperatures_K=explicit_temperatures,
            )
        else:
            result = _continuum_executable_cycle_satisfaction(
                mechanism,
                cycle,
                selected_steps,
                evaluator=evaluator,
            )
        results.append(result)
    by_id = {result.cycle_id: result for result in results}
    ordered = tuple(by_id[cycle.cycle_id] for cycle in admission_report.cycles)
    return WegscheiderExecutableSatisfactionReport(
        admission_report=admission_report,
        cycles=ordered,
    )


def validate_bound_wegscheider_satisfaction(
    mechanism: object,
    admission_report: WegscheiderCyclicityReport,
    *,
    temperatures_K: Sequence[float] | None = None,
) -> WegscheiderExecutableSatisfactionReport:
    report = analyze_bound_wegscheider_satisfaction(
        mechanism,
        admission_report,
        temperatures_K=temperatures_K,
    )
    if report.failed_cycles:
        raise WegscheiderExecutableSatisfactionError(report)
    return report
