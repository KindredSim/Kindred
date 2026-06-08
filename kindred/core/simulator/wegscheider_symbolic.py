from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from math import gcd
from functools import reduce
import re
from typing import Any, Mapping, Sequence

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
    steps: list[_ReversibleStep] = []
    for entry in get_step_index_map(mechanism):
        if str(entry.get("kind") or "") != "equilibrium":
            continue
        try:
            step_index = int(entry.get("step_index"))
            equilibrium_index = int(entry.get("equilibrium_index"))
        except (TypeError, ValueError):
            continue
        if not (0 <= equilibrium_index < len(eqs)):
            continue
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
                parameter_name=f"Keq{step_index}",
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


def _resolved_by_for_cycle(
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
        )
        resolved_by, proof_fingerprint = _resolved_by_for_cycle(
            cycle,
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
