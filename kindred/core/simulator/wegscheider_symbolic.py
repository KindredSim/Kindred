from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from math import gcd, isclose, isfinite, log
from functools import reduce
import re
from typing import Any, Mapping, Sequence

from kindred.core.constants import R
from kindred.core.equilibrium_rate_authority import (
    EquilibriumRateAuthorityKind,
    GeneratedEquilibriumAuthorityKind,
    generated_equilibrium_authority_kind,
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
    EquilibriumMetadataKeys,
    MechanismMetadataView,
)
from kindred.core.ode_builder import (
    EffectiveRateCoefficientEvaluator,
    build_effective_rate_coefficient_evaluator,
)
from kindred.core.rate_binding import ForwardRateBinding, RateBinding
from kindred.core.simulator.algebra_section import upsert_lines_into_algebra_section
from kindred.core.simulator.errors import DSLError
from kindred.core.simulator.parameter_namespace import build_namespace_from_mechanism
from kindred.core.simulator.parameter_algebra_spec import (
    ParameterAlgebraSpec,
    ParameterAssignment,
    strip_inline_comment,
)
from kindred.core.simulator.step_indexing import get_step_index_map
from kindred.core.symbolic.backend import get_symbolic_backend_metadata
from kindred.core.symbolic.identity import symbolic_fingerprint
from kindred.core.symbolic.proof import prove_product_identity
from kindred.core.temperature import TemperatureScheduleDictProtocol

__all__ = [
    "UnresolvedWegscheiderCyclicityError",
    "WegscheiderCycle",
    "WegscheiderCyclicityReport",
    "WegscheiderResolutionUpdate",
    "analyze_wegscheider_cyclicity",
    "apply_wegscheider_resolution_to_reactions_text",
    "build_wegscheider_resolution_updates",
    "validate_wegscheider_cyclicity_resolved",
]


@dataclass(frozen=True)
class WegscheiderCycle:
    cycle_id: str
    step_indices: tuple[int, ...]
    equilibrium_indices: tuple[int, ...]
    coefficients: tuple[int, ...]
    parameter_names: tuple[str, ...]
    generated_step_flags: tuple[bool, ...] = ()
    resolved_by: str | None = None
    resolved_proof_fingerprint: str | None = None


@dataclass(frozen=True)
class WegscheiderCyclicityReport:
    cycles: tuple[WegscheiderCycle, ...]

    @property
    def unresolved_cycles(self) -> tuple[WegscheiderCycle, ...]:
        return tuple(cycle for cycle in self.cycles if cycle.resolved_by is None)

    @property
    def is_resolved(self) -> bool:
        return not self.unresolved_cycles

    @property
    def symbolic_identity(self) -> dict[str, Any]:
        metadata = get_symbolic_backend_metadata()
        cycles_payload = [
            {
                "cycle_id": str(cycle.cycle_id),
                "step_indices": list(cycle.step_indices),
                "equilibrium_indices": list(cycle.equilibrium_indices),
                "coefficients": list(cycle.coefficients),
                "parameter_names": list(cycle.parameter_names),
                "generated_step_flags": list(
                    cycle.generated_step_flags
                ),
                "resolved_by": cycle.resolved_by,
                "proof_fingerprint": cycle.resolved_proof_fingerprint,
            }
            for cycle in self.cycles
        ]
        source_cycles_payload = [
            {
                "cycle_id": str(cycle.cycle_id),
                "step_indices": list(cycle.step_indices),
                "equilibrium_indices": list(cycle.equilibrium_indices),
                "coefficients": list(cycle.coefficients),
                "parameter_names": list(cycle.parameter_names),
                "generated_step_flags": list(
                    cycle.generated_step_flags
                ),
            }
            for cycle in self.cycles
        ]
        payload: dict[str, Any] = {
            "kind": "wegscheider_cyclicity",
            "backend_name": metadata.backend_name,
            "backend_version": metadata.backend_version,
            "profile_version": metadata.profile_version,
            "cycles": cycles_payload,
            "resolved": bool(self.is_resolved),
        }
        source_payload = {
            "kind": "wegscheider_cyclicity_source",
            "cycles": source_cycles_payload,
        }
        payload["source_fingerprint"] = symbolic_fingerprint(source_payload)
        payload["artifact_fingerprint"] = symbolic_fingerprint(
            {
                "kind": "wegscheider_cyclicity_proof",
                "resolved": bool(self.is_resolved),
                "proof_fingerprints": [
                    item.get("proof_fingerprint")
                    for item in cycles_payload
                ],
            }
        )
        payload["fingerprint"] = symbolic_fingerprint(payload)
        return payload


@dataclass(frozen=True)
class WegscheiderResolutionUpdate:
    cycle_id: str
    parameter_name: str
    expr_src: str

    @property
    def line(self) -> str:
        return f"param {self.parameter_name} = {self.expr_src}"


class UnresolvedWegscheiderCyclicityError(DSLError):
    def __init__(self, cycles: Sequence[WegscheiderCycle]) -> None:
        unresolved = tuple(cycles)
        names = ", ".join(
            f"{cycle.cycle_id} ({', '.join(cycle.parameter_names)})"
            for cycle in unresolved
        )
        super().__init__(
            f"unresolved Wegscheider cyclicity: {names}",
            suggestion=(
                "Add a symbolic parameter-algebra dependency such as "
                "'param Keq3 = 1 / (Keq1 * Keq2)' for each independent thermodynamic cycle."
            ),
        )
        self.stage = "wegscheider_cyclicity"
        self.cycles = unresolved


@dataclass(frozen=True)
class _ReversibleStep:
    step_index: int
    equilibrium_index: int
    vector: Mapping[str, Fraction]
    parameter_name: str
    generated: bool = False


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
    highest_authored_step = 0
    for entry in get_step_index_map(mechanism):
        try:
            step_index = int(entry.get("step_index"))
        except (TypeError, ValueError):
            continue
        highest_authored_step = max(highest_authored_step, step_index)
        if str(entry.get("kind") or "") != "equilibrium":
            continue
        try:
            equilibrium_index = int(entry.get("equilibrium_index"))
        except (TypeError, ValueError):
            continue
        if 0 <= equilibrium_index < len(eqs):
            authored_by_equilibrium_index[equilibrium_index] = step_index

    steps: list[_ReversibleStep] = []
    generated_ordinal = 0
    for equilibrium_index, eq in enumerate(eqs):
        authored_step_index = authored_by_equilibrium_index.get(
            equilibrium_index
        )
        if authored_step_index is None:
            generated_ordinal += 1
            step_index = highest_authored_step + generated_ordinal
            parameter_name = (
                f"generated_effective_Keq_equilibrium_{equilibrium_index + 1}"
            )
        else:
            step_index = authored_step_index
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
                generated=authored_step_index is None,
            )
        )
    return tuple(sorted(steps, key=lambda step: step.step_index))


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


def _cycle_expression(cycle: WegscheiderCycle, dependent_parameter: str) -> str:
    if any(cycle.generated_step_flags):
        raise ValueError(
            "Wegscheider cycles containing generated steps cannot be "
            "rewritten through authored parameter algebra."
        )
    if dependent_parameter not in cycle.parameter_names:
        raise ValueError(f"{dependent_parameter!r} is not part of {cycle.cycle_id}.")
    idx = cycle.parameter_names.index(dependent_parameter)
    dependent_coeff = int(cycle.coefficients[idx])
    if dependent_coeff == 0:
        raise ValueError(f"{dependent_parameter!r} has zero coefficient in {cycle.cycle_id}.")
    numerator_terms: list[str] = []
    denominator_terms: list[str] = []
    for name, coeff in zip(cycle.parameter_names, cycle.coefficients):
        if name == dependent_parameter or coeff == 0:
            continue
        exponent = Fraction(-int(coeff), dependent_coeff)
        target_terms = numerator_terms if exponent > 0 else denominator_terms
        magnitude = abs(exponent)
        if magnitude.denominator == 1:
            power = magnitude.numerator
            term = str(name) if power == 1 else f"{name}**{power}"
        else:
            term = f"{name}**({magnitude.numerator}/{magnitude.denominator})"
        target_terms.append(term)

    numerator = " * ".join(numerator_terms) if numerator_terms else "1"
    if not denominator_terms:
        return numerator
    denominator = " * ".join(denominator_terms)
    return f"{numerator} / ({denominator})"


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
        parameter_key = str(parameter_name).lower()
        previous_cycle = requested_parameters.get(parameter_key)
        if previous_cycle is not None:
            raise ValueError(
                f"Wegscheider dependent parameter {parameter_name!r} was selected for both "
                f"{previous_cycle!r} and {cycle_id!r}; choose one dependent parameter per cycle."
            )
        requested_parameters[parameter_key] = str(cycle_id)
        updates.append(
            WegscheiderResolutionUpdate(
                cycle_id=str(cycle_id),
                parameter_name=str(parameter_name),
                expr_src=_cycle_expression(cycle, str(parameter_name)),
            )
        )
    return tuple(updates)


def apply_wegscheider_resolution_to_reactions_text(
    dsl_text: str,
    updates: Sequence[WegscheiderResolutionUpdate],
) -> str:
    pending = {str(update.parameter_name).lower(): update for update in updates}
    if not pending:
        return str(dsl_text or "")

    lines = str(dsl_text or "").splitlines()
    rewritten: list[str] = []
    replaced: set[str] = set()
    param_re = re.compile(r"^\s*param\s+([A-Za-z_][A-Za-z0-9_]*)\s*=", re.IGNORECASE)
    for raw in lines:
        code = strip_inline_comment(raw).strip()
        match = param_re.match(code)
        if match is None:
            rewritten.append(raw)
            continue
        key = str(match.group(1)).lower()
        update = pending.get(key)
        if update is None:
            rewritten.append(raw)
            continue
        if key in replaced:
            continue
        rewritten.append(update.line)
        replaced.add(key)

    missing = [
        update.line
        for key, update in pending.items()
        if key not in replaced
    ]
    rewritten_text = "\n".join(rewritten).rstrip("\n")
    if not missing:
        return rewritten_text + "\n"
    return upsert_lines_into_algebra_section(rewritten_text, missing)


def _assignment_by_name(spec: ParameterAlgebraSpec | None) -> dict[str, ParameterAssignment]:
    if spec is None:
        return {}
    return {str(stmt.name): stmt for stmt in spec.param_statements or []}


def _cycle_candidate_proof_fingerprint(
    *,
    cycle: WegscheiderCycle,
    assignment: ParameterAssignment,
    spec: ParameterAlgebraSpec,
) -> str | None:
    candidate = str(assignment.name)
    if candidate not in cycle.parameter_names:
        return None
    proof = prove_product_identity(
        target_factors={
            str(name): int(coeff)
            for name, coeff in zip(cycle.parameter_names, cycle.coefficients)
            if int(coeff) != 0
        },
        candidate=assignment,
        spec=spec,
    )
    return str(proof.fingerprint or "") if bool(proof.proven) else None


def _parameter_algebra_resolution(
    cycle: WegscheiderCycle,
    *,
    parameter_algebra_spec: ParameterAlgebraSpec | None,
) -> tuple[str | None, str | None]:
    assignments = _assignment_by_name(parameter_algebra_spec)
    for name in cycle.parameter_names:
        assignment = assignments.get(name)
        if assignment is None or parameter_algebra_spec is None:
            continue
        proof_fingerprint = _cycle_candidate_proof_fingerprint(
            cycle=cycle,
            assignment=assignment,
            spec=parameter_algebra_spec,
        )
        if proof_fingerprint:
            return name, proof_fingerprint
    return None, None


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
    metadata = getattr(equilibrium, "metadata", {}) or {}
    generated_kind = (
        generated_equilibrium_authority_kind(metadata)
        if isinstance(metadata, Mapping)
        else ""
    )
    log_temperature_power = 0.0
    inverse_temperature = 0.0
    source = ""

    if generated_kind:
        model = metadata.get(EquilibriumMetadataKeys.GENERATED_RATE_MODEL)
        if not isinstance(model, Mapping):
            return None
        try:
            if (
                generated_kind
                == GeneratedEquilibriumAuthorityKind.TRANSITION_STATE_BARRIER_RATES
            ):
                dG_forward = float(model["dG_act_fwd_J_per_mol"])
                dG_reverse = float(model["dG_act_rev_J_per_mol"])
                inverse_temperature = -(
                    dG_forward - dG_reverse
                ) / float(R)
                source = "generated-transition-state"
            elif (
                generated_kind
                == GeneratedEquilibriumAuthorityKind.DIRECT_THERMODYNAMIC_EQUILIBRIUM
            ):
                inverse_temperature = -float(
                    model["dG_eq_J_per_mol"]
                ) / float(R)
                source = "generated-direct-thermodynamic"
            else:
                return None
        except (KeyError, TypeError, ValueError, OverflowError):
            return None
    else:
        authority = normalize_existing_equilibrium_rate_authority(
            equilibrium
        )
        if authority.kind == EquilibriumRateAuthorityKind.KEQ:
            if (
                callable(authority.reverse_std_ratio)
                and bool(
                    getattr(
                        authority.reverse_std_ratio,
                        "accepts_runtime_temperature",
                        False,
                    )
                )
            ):
                return None
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
                        default_kappa=MechanismMetadataView.from_metadata(
                            getattr(mechanism, "metadata", {}) or {}
                        ).kappa_global,
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


def _runtime_domain_resolution(
    mechanism: object,
    cycle: WegscheiderCycle,
    selected_steps: Sequence[_ReversibleStep],
    *,
    evaluator: EffectiveRateCoefficientEvaluator,
    nominal_temperature_K: float,
    nominal_ratios: Mapping[int, float],
    parameter_algebra_spec: ParameterAlgebraSpec | None,
) -> tuple[str | None, str | None]:
    domain_kind, temperatures, schedule_payload = (
        _runtime_temperature_domain(mechanism)
    )
    reaction_count = len(tuple(getattr(mechanism, "reactions", ()) or ()))
    if domain_kind == "finite" and temperatures:
        log_products: list[float] = []
        ratios_by_temperature: list[tuple[float, ...]] = []
        for temperature in temperatures:
            ratios: list[float] = []
            for step in selected_steps:
                ratio = _effective_equilibrium_ratio(
                    evaluator,
                    reaction_count=reaction_count,
                    equilibrium_index=step.equilibrium_index,
                    temperature_K=temperature,
                )
                if ratio is None:
                    return None, None
                ratios.append(ratio)
            log_product = sum(
                int(coefficient) * log(float(ratio))
                for coefficient, ratio in zip(
                    cycle.coefficients,
                    ratios,
                    strict=True,
                )
            )
            if not isfinite(log_product) or not isclose(
                log_product,
                0.0,
                rel_tol=0.0,
                abs_tol=1e-10,
            ):
                break
            log_products.append(float(log_product))
            ratios_by_temperature.append(tuple(ratios))
        else:
            fingerprint = symbolic_fingerprint(
                {
                    "kind": "exhaustive-runtime-temperature-domain",
                    "cycle_id": cycle.cycle_id,
                    "schedule": schedule_payload,
                    "temperatures_K": temperatures,
                    "ratios": ratios_by_temperature,
                    "log_products": log_products,
                    "coefficients": cycle.coefficients,
                }
            )
            return "runtime-temperature-domain", fingerprint

    relation_result = _combined_cycle_relation(
        mechanism,
        cycle,
        selected_steps,
        evaluator=evaluator,
        nominal_temperature_K=nominal_temperature_K,
        nominal_ratios=nominal_ratios,
    )
    if relation_result is None:
        return None, None
    combined, relations = relation_result
    temperature_invariant = (
        isclose(
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
    if not temperature_invariant:
        return None, None
    algebra_name, algebra_fingerprint = _parameter_algebra_resolution(
        cycle,
        parameter_algebra_spec=parameter_algebra_spec,
    )
    constant_identity = isclose(
        combined.log_constant,
        0.0,
        rel_tol=0.0,
        abs_tol=1e-10,
    )
    if not constant_identity and algebra_fingerprint is None:
        return None, None
    fingerprint = symbolic_fingerprint(
        {
            "kind": "structural-runtime-temperature-identity",
            "cycle_id": cycle.cycle_id,
            "schedule": schedule_payload,
            "combined_relation": {
                "log_constant": combined.log_constant,
                "log_temperature_power": (
                    combined.log_temperature_power
                ),
                "inverse_temperature": combined.inverse_temperature,
            },
            "step_relations": [
                {
                    "log_constant": relation.log_constant,
                    "log_temperature_power": (
                        relation.log_temperature_power
                    ),
                    "inverse_temperature": relation.inverse_temperature,
                    "source": relation.source,
                }
                for relation in relations
            ],
            "algebra_proof": algebra_fingerprint,
        }
    )
    return (
        algebra_name or "structural-temperature-identity",
        fingerprint,
    )


def analyze_wegscheider_cyclicity(
    mechanism: object,
    *,
    parameter_algebra_spec: ParameterAlgebraSpec | None = None,
) -> WegscheiderCyclicityReport:
    if parameter_algebra_spec is None:
        meta = getattr(mechanism, "metadata", {}) or {}
        if isinstance(meta, dict):
            maybe_spec = meta.get("parameter_algebra_spec")
            if isinstance(maybe_spec, ParameterAlgebraSpec):
                parameter_algebra_spec = maybe_spec
            elif str(meta.get("algebra_text") or "").strip():
                from kindred.core.simulator.parameter_algebra_spec import (
                    parse_parameter_algebra_spec_from_dsl_text,
                )

                parameter_algebra_spec = parse_parameter_algebra_spec_from_dsl_text(
                    str(meta.get("algebra_text") or ""),
                    mechanism_namespace=build_namespace_from_mechanism(mechanism),
                )

    steps = _reversible_steps(mechanism)
    if len(steps) < 2:
        return WegscheiderCyclicityReport(cycles=())
    try:
        evaluator = build_effective_rate_coefficient_evaluator(mechanism)
        nominal_temperature_K = float(
            evaluator.nominal_temperature_K
        )
        reaction_count = len(
            tuple(getattr(mechanism, "reactions", ()) or ())
        )
        nominal_ratios = {
            step.equilibrium_index: _effective_equilibrium_ratio(
                evaluator,
                reaction_count=reaction_count,
                equilibrium_index=step.equilibrium_index,
                temperature_K=nominal_temperature_K,
            )
            for step in steps
        }
    except (TypeError, ValueError, OverflowError) as exc:
        raise DSLError(
            "Wegscheider analysis could not evaluate the canonical "
            "effective rate coefficients."
        ) from exc
    if any(value is None for value in nominal_ratios.values()):
        raise DSLError(
            "Wegscheider analysis requires finite positive effective "
            "forward and reverse coefficients."
        )
    resolved_nominal_ratios = {
        equilibrium_index: float(value)
        for equilibrium_index, value in nominal_ratios.items()
        if value is not None
    }
    species = sorted({species_name for step in steps for species_name in step.vector.keys()})
    if not species:
        return WegscheiderCyclicityReport(cycles=())
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
        cycle = WegscheiderCycle(
            cycle_id=f"cycle_{idx}",
            step_indices=tuple(step.step_index for step, _coeff in selected),
            equilibrium_indices=tuple(step.equilibrium_index for step, _coeff in selected),
            coefficients=tuple(int(coeff) for _step, coeff in selected),
            parameter_names=tuple(step.parameter_name for step, _coeff in selected),
            generated_step_flags=tuple(
                bool(step.generated)
                for step, _coeff in selected
            ),
        )
        resolved_by, proof_fingerprint = _runtime_domain_resolution(
            mechanism,
            cycle,
            tuple(step for step, _coefficient in selected),
            evaluator=evaluator,
            nominal_temperature_K=nominal_temperature_K,
            nominal_ratios=resolved_nominal_ratios,
            parameter_algebra_spec=parameter_algebra_spec,
        )
        cycles.append(
            replace(
                cycle,
                resolved_by=resolved_by,
                resolved_proof_fingerprint=proof_fingerprint,
            )
        )
    return WegscheiderCyclicityReport(cycles=tuple(cycles))


def validate_wegscheider_cyclicity_resolved(
    mechanism: object,
    *,
    parameter_algebra_spec: ParameterAlgebraSpec | None = None,
) -> WegscheiderCyclicityReport:
    report = analyze_wegscheider_cyclicity(
        mechanism,
        parameter_algebra_spec=parameter_algebra_spec,
    )
    if report.unresolved_cycles:
        raise UnresolvedWegscheiderCyclicityError(report.unresolved_cycles)
    return report
