"""Pure global planning of source-backed exact correction coordinates."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations

from kindred.core.simulator.exact_constraint_composition import (
    ExactMonomial,
    ExactConstraintComposition,
)
from kindred.core.simulator.dsl_build_input import source_coordinate_capabilities

__all__ = [
    "CorrectionTarget",
    "CorrectionPlan",
    "CorrectionPlanner",
    "plan_correction_basis",
    "build_correction_expressions",
]


@dataclass(frozen=True, slots=True)
class CorrectionTarget:
    """Canonical source/editability facts for one admitted coordinate."""

    name: str
    family: str
    source_backed: bool = True
    editable: bool = True
    locked: bool = False
    derived: bool = False

    @property
    def eligible(self) -> bool:
        return bool(
            isinstance(self.name, str)
            and bool(self.name.strip())
            and isinstance(self.family, str)
            and bool(self.family.strip())
            and self.source_backed
            and self.editable
            and not self.locked
            and not self.derived
            and _correction_role_is_admitted(self.family)
        )


@dataclass(frozen=True, slots=True)
class CorrectionPlan:
    basis: tuple[str, ...]
    assignments: tuple[tuple[str, str], ...]
    available: bool
    reason: str | None = None


def _rank(rows: Sequence[Sequence[Fraction]]) -> int:
    if not rows:
        return 0
    matrix = [list(map(Fraction, row)) for row in rows]
    width = len(matrix[0]) if matrix else 0
    rank = 0
    for column in range(width):
        pivot = next(
            (row for row in range(rank, len(matrix)) if matrix[row][column]),
            None,
        )
        if pivot is None:
            continue
        matrix[rank], matrix[pivot] = matrix[pivot], matrix[rank]
        scale = matrix[rank][column]
        matrix[rank] = [value / scale for value in matrix[rank]]
        for row in range(len(matrix)):
            if row == rank or not matrix[row][column]:
                continue
            factor = matrix[row][column]
            matrix[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(matrix[row], matrix[rank])
            ]
        rank += 1
        if rank == len(matrix):
            break
    return rank


def _inverse(
    matrix: Sequence[Sequence[Fraction]],
) -> tuple[tuple[Fraction, ...], ...] | None:
    size = len(matrix)
    if not size or any(len(row) != size for row in matrix):
        return None
    augmented = [
        [Fraction(value) for value in row]
        + [Fraction(1 if row_index == column else 0) for column in range(size)]
        for row_index, row in enumerate(matrix)
    ]
    for column in range(size):
        pivot = next(
            (row for row in range(column, size) if augmented[row][column]),
            None,
        )
        if pivot is None:
            return None
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(size):
            if row == column or not augmented[row][column]:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(augmented[row], augmented[column])
            ]
    return tuple(tuple(row[size:]) for row in augmented)


def _preference(target: CorrectionTarget, source_index: int) -> tuple[int, int, int]:
    """Rank correction families, then retain authenticated source order."""

    family = str(target.family)
    if family in {"kf+kr", "kf", "kr"}:
        return (0, 0 if family == "kr" else 1, source_index)
    if family == "Keq":
        return (1, 0, source_index)
    if family == "dG_eq":
        return (2, 0, source_index)
    if family in {"A", "Ea", "dG_act"}:
        return (3, {"A": 0, "Ea": 1, "dG_act": 2}[family], source_index)
    if family == "kappa":
        return (4, 0, source_index)
    return (5, 0, source_index)


def _correction_role_is_admitted(role: str) -> bool:
    try:
        return source_coordinate_capabilities(str(role)).correction_target
    except ValueError:
        return False


def _correction_constraints(
    composition: ExactConstraintComposition,
) -> tuple[object, ...]:
    """Return every meaningful resolved/unsatisfied row in the composition."""

    return tuple(
        item
        for item in composition.constraints
        if item.status in {"resolved", "unsatisfied"}
        and (
            item.status == "unsatisfied"
            or not _is_exact_identity(item.monomial)
        )
    )


def _is_exact_identity(value: ExactMonomial) -> bool:
    return bool(
        value.coefficient == 1
        and not value.powers
        and not value.exponentials
        and not value.sum_terms
    )


def _direct_source_power(item: object, target_name: str) -> Fraction:
    """Return only a multiplicative authored factor's power."""

    return Fraction(
        dict(item.monomial.powers).get(
            str(target_name),
            Fraction(0),
        )
    )


def _has_unsupported_source_factor(item: object) -> bool:
    """Reject factors that cannot be authored as a static correction RHS."""

    monomial = item.monomial
    if monomial.exponentials or monomial.sum_terms:
        return True
    if monomial.coefficient == 0:
        return True
    return any(
        (
            str(name) == "T"
            or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(name)) is None
        )
        for name, _power in monomial.powers
    )


def _simplify_generated_factors(
    factors: Mapping[tuple[str, str | Fraction], Fraction],
) -> dict[tuple[str, str | Fraction], Fraction]:
    """Combine exact RHS factors and drop every cancelled zero power."""

    simplified: dict[tuple[str, str | Fraction], Fraction] = {}
    for key, raw_power in factors.items():
        power = Fraction(raw_power)
        if power:
            simplified[key] = simplified.get(key, Fraction(0)) + power
    return {
        key: power
        for key, power in simplified.items()
        if power
    }


def _has_unmaterializable_generated_factor(
    factors: Mapping[tuple[str, str | Fraction], Fraction],
) -> bool:
    """Reject only surviving factors without an exact authored RHS value."""

    for (kind, value), power in factors.items():
        if not power or kind != "symbol":
            continue
        name = str(value)
        if name == "C0" or re.fullmatch(r"degeneracy[1-9]\d*", name):
            return True
    return False


def _solve_correction_factors(
    composition: ExactConstraintComposition,
    basis: Sequence[str],
) -> tuple[
    tuple[tuple[Fraction, ...], ...],
    tuple[dict[tuple[str, str | Fraction], Fraction], ...],
] | None:
    """Solve one basis and aggregate its exact generated RHS factors."""

    constraints = _correction_constraints(composition)
    inverse = _inverse(
        [
            tuple(_direct_source_power(item, name) for name in basis)
            for item in constraints
        ]
    )
    if inverse is None:
        return None
    selected_names = {str(name) for name in basis}
    generated: list[dict[tuple[str, str | Fraction], Fraction]] = []
    for target_index in range(len(basis)):
        factors: dict[tuple[str, str | Fraction], Fraction] = {}
        for row_index, constraint in enumerate(constraints):
            exponent = inverse[target_index][row_index]
            if not exponent:
                continue
            monomial = constraint.monomial
            constraint_coefficient = Fraction(monomial.coefficient)
            if constraint_coefficient != 1:
                key = ("coefficient", constraint_coefficient)
                factors[key] = factors.get(key, Fraction(0)) - exponent
            for name, power in monomial.powers:
                if not power:
                    continue
                key = ("symbol", str(name))
                factors[key] = factors.get(key, Fraction(0)) - Fraction(power) * exponent
        generated.append(
            _simplify_generated_factors(
                {
                    key: power
                    for key, power in factors.items()
                    if power
                    and not (
                        key[0] == "symbol" and key[1] in selected_names
                    )
                }
            )
        )
    return inverse, tuple(generated)


def _plan_for_basis(
    constraints: Sequence[object],
    basis: Sequence[CorrectionTarget],
) -> CorrectionPlan:
    return CorrectionPlan(
        basis=tuple(target.name for target in basis),
        assignments=tuple(
            (constraint.cycle_id, target.name)
            for target, constraint in zip(basis, constraints, strict=True)
        ),
        available=True,
    )


def _choose_multiplicative_basis(
    composition: ExactConstraintComposition,
    constraints: Sequence[object],
    eligible: Sequence[CorrectionTarget],
) -> tuple[CorrectionTarget, ...] | None:
    """Choose a full-rank basis whose RHS is an exact product of factors."""

    rows = [
        tuple(_direct_source_power(item, target.name) for target in eligible)
        for item in constraints
    ]
    if _rank(rows) != len(constraints):
        return None
    for candidate in combinations(eligible, len(constraints)):
        candidate_rows = [
            tuple(_direct_source_power(item, target.name) for target in candidate)
            for item in constraints
        ]
        if _rank(candidate_rows) != len(constraints):
            continue
        solution = _solve_correction_factors(
            composition,
            tuple(target.name for target in candidate),
        )
        if solution is None:
            continue
        _inverse_matrix, generated_factors = solution
        if any(
            _has_unmaterializable_generated_factor(factors)
            for factors in generated_factors
        ):
            # Keep searching for a basis that does not require a symbolic-only
            # C0 or degeneracy RHS.
            continue
        return tuple(candidate)
    return None


_ENERGY_FAMILIES = frozenset({"Ea", "dG_act", "dG_eq"})
_ENERGY_COMMON_POWERS = frozenset({"T"})


def _energy_term_data(
    term: ExactMonomial,
    energy_names: frozenset[str],
) -> tuple[str, Fraction, tuple[tuple[str, Fraction], ...]] | None:
    """Split one additive exponent term into energy and common factors."""

    if term.sum_terms or term.exponentials:
        return None
    energy_powers = [
        (str(name), Fraction(power))
        for name, power in term.powers
        if str(name) in energy_names
    ]
    if len(energy_powers) != 1 or energy_powers[0][1] != 1:
        return None
    energy_name, _energy_power = energy_powers[0]
    common_powers = tuple(
        (str(name), Fraction(power))
        for name, power in term.powers
        if str(name) != energy_name
    )
    if any(name not in _ENERGY_COMMON_POWERS for name, _power in common_powers):
        return None
    return energy_name, Fraction(term.coefficient), common_powers


def _energy_rows(
    constraints: Sequence[object],
    energy_names: frozenset[str],
) -> tuple[dict[str, Fraction], ...] | None:
    """Return normalized linear energy coefficients for every exact row."""

    rows: list[dict[str, Fraction]] = []
    for item in constraints:
        monomial = item.monomial
        if (
            monomial.coefficient != 1
            or monomial.powers
            or monomial.sum_terms
            or len(monomial.exponentials) != 1
        ):
            return None
        argument = monomial.exponentials[0]
        terms = argument.sum_terms or (argument,)
        parsed = [
            _energy_term_data(term, energy_names)
            for term in terms
        ]
        if not parsed or any(value is None for value in parsed):
            return None
        typed = tuple(value for value in parsed if value is not None)
        # All terms must share one exact non-energy factor.  Their rational
        # coefficients become the linear row coefficients after factoring it.
        common_powers = typed[0][2]
        if any(value[2] != common_powers for value in typed):
            return None
        scale = typed[0][1]
        if not scale:
            return None
        row: dict[str, Fraction] = {}
        for name, coefficient, _powers in typed:
            row[name] = row.get(name, Fraction(0)) + coefficient / scale
        rows.append({name: value for name, value in row.items() if value})
    return tuple(rows)


def _choose_energy_basis(
    constraints: Sequence[object],
    eligible: Sequence[CorrectionTarget],
) -> tuple[CorrectionTarget, ...] | None:
    """Choose an exact linear basis for exponential energy residuals."""

    if not eligible:
        return None
    energy_names = frozenset(
        str(target.name)
        for target in eligible
    )
    rows = _energy_rows(constraints, energy_names)
    if rows is None:
        return None
    candidate_rows = [
        tuple(row.get(target.name, Fraction(0)) for target in eligible)
        for row in rows
    ]
    if _rank(candidate_rows) != len(constraints):
        return None
    for candidate in combinations(eligible, len(constraints)):
        matrix = [
            tuple(row.get(target.name, Fraction(0)) for target in candidate)
            for row in rows
        ]
        if _rank(matrix) != len(constraints) or _inverse(matrix) is None:
            continue
        return tuple(candidate)
    return None


def _render_linear_term(name: str, coefficient: Fraction) -> str:
    coefficient = Fraction(coefficient)
    if coefficient == 1:
        return str(name)
    if coefficient == -1:
        return f"-{name}"
    return f"{_render_fraction(coefficient)} * {name}"


def _render_linear_expression(terms: Mapping[str, Fraction]) -> str:
    nonzero = [
        (str(name), Fraction(coefficient))
        for name, coefficient in terms.items()
        if Fraction(coefficient)
    ]
    if not nonzero:
        return "0"
    rendered = _render_linear_term(*nonzero[0])
    for name, coefficient in nonzero[1:]:
        coefficient = Fraction(coefficient)
        if coefficient < 0:
            rendered += " - " + _render_linear_term(name, -coefficient)
        else:
            rendered += " + " + _render_linear_term(name, coefficient)
    return rendered


def _build_energy_correction_expressions(
    constraints: Sequence[object],
    basis: Sequence[str],
) -> dict[str, str] | None:
    energy_names = frozenset(
        str(name)
        for item in constraints
        for name in item.monomial.coordinate_power_map()
        if re.fullmatch(r"(?:Ea|dG_act|dG_eq)[1-9][0-9]*", str(name))
    )
    rows = _energy_rows(constraints, energy_names)
    if rows is None:
        return None
    matrix = [
        tuple(row.get(str(name), Fraction(0)) for name in basis)
        for row in rows
    ]
    inverse = _inverse(matrix)
    if inverse is None:
        return None
    selected = set(str(name) for name in basis)
    output: dict[str, str] = {}
    for target_index, target_name in enumerate(basis):
        terms: dict[str, Fraction] = {}
        for row_index, row in enumerate(rows):
            multiplier = -inverse[target_index][row_index]
            if not multiplier:
                continue
            for name, coefficient in row.items():
                if name in selected:
                    continue
                terms[name] = terms.get(name, Fraction(0)) + multiplier * coefficient
        output[str(target_name)] = _render_linear_expression(terms)
    return output


class CorrectionPlanner:
    """Choose one acyclic, source-backed, full-rank global basis."""

    def __init__(
        self,
        composition: ExactConstraintComposition,
        targets: Mapping[str, CorrectionTarget] | Sequence[CorrectionTarget],
    ) -> None:
        self.composition = composition
        raw_targets = targets.values() if isinstance(targets, Mapping) else targets
        admitted: dict[str, CorrectionTarget] = {}
        for target in raw_targets:
            if not isinstance(target, CorrectionTarget):
                raise TypeError("Correction targets must be canonical CorrectionTarget values.")
            name = str(target.name)
            if name in admitted:
                raise ValueError(f"Duplicate correction target {name!r}.")
            admitted[name] = target
        self.targets = admitted

    def plan(self) -> CorrectionPlan:
        unresolved = tuple(
            item for item in self.composition.constraints if item.status == "unresolved"
        )
        if unresolved:
            reasons = "; ".join(
                f"{item.cycle_id}: {item.reason or 'unresolved exact constraint'}"
                for item in unresolved
            )
            return CorrectionPlan((), (), False, reasons)
        constraints = _correction_constraints(self.composition)
        unsatisfied = tuple(
            item for item in constraints if item.status == "unsatisfied"
        )
        if not unsatisfied:
            return CorrectionPlan((), (), True)
        eligible = [
            target
            for target in self.targets.values()
            if target.eligible
        ]
        eligible.sort(
            key=lambda target: _preference(
                target,
                tuple(self.targets).index(target.name),
            )
        )
        # A direct multiplicative basis remains the preferred route for
        # authored k/kf/kr/Keq/A/kappa coordinates.  Exponential residuals
        # cannot be represented by this solver; leave those rows for the
        # bounded energy projection below.
        if not any(_has_unsupported_source_factor(item) for item in constraints):
            basis = _choose_multiplicative_basis(self.composition, constraints, eligible)
            if basis is not None:
                return _plan_for_basis(constraints, basis)

        energy_eligible = [
            target
            for target in eligible
            if target.family in {"Ea", "dG_act", "dG_eq"}
        ]
        energy_basis = _choose_energy_basis(constraints, energy_eligible)
        if energy_basis is not None:
            return _plan_for_basis(constraints, energy_basis)

        if any(_has_unsupported_source_factor(item) for item in constraints):
            return CorrectionPlan(
                (),
                (),
                False,
                "exact source correction requires a bounded linear energy projection",
            )
        return CorrectionPlan(
            (),
            (),
            False,
            "no source-backed editable unlocked global full-rank correction basis is available",
        )


def plan_correction_basis(
    composition: ExactConstraintComposition,
    targets: Mapping[str, CorrectionTarget] | Sequence[CorrectionTarget],
) -> CorrectionPlan:
    return CorrectionPlanner(composition, targets).plan()


def _render_fraction(value: Fraction) -> str:
    value = Fraction(value)
    return (
        str(value.numerator)
        if value.denominator == 1
        else f"({value.numerator}/{value.denominator})"
    )


def _render_factor(name: str, power: Fraction) -> str:
    return name if power == 1 else f"{name}**({_render_fraction(power)})"


def build_correction_expressions(
    composition: ExactConstraintComposition,
    plan: CorrectionPlan,
) -> dict[str, str]:
    """Render the exact coupled solution for the selected global basis."""

    if not plan.available or not plan.basis:
        return {}
    constraints = _correction_constraints(composition)
    basis = tuple(plan.basis)
    assignment_map = dict(plan.assignments)
    if tuple(assignment_map.get(item.cycle_id) for item in constraints) != basis:
        raise ValueError("Correction plan assignments do not match its global basis.")

    if any(_has_unsupported_source_factor(item) for item in constraints):
        expressions = _build_energy_correction_expressions(constraints, basis)
        if expressions is None:
            raise ValueError(
                "Exact source correction requires a bounded linear energy projection."
            )
        return expressions

    solution = _solve_correction_factors(composition, basis)
    if solution is None:
        raise ValueError("Correction basis is not invertible.")
    _inverse_matrix, generated_factors = solution
    if any(
        _has_unmaterializable_generated_factor(factors)
        for factors in generated_factors
    ):
        raise ValueError(
            "Exact source correction requires a basis whose simplified RHS "
            "does not depend on symbolic-only degeneracy or C0."
        )
    output: dict[str, str] = {}
    for target_index, target_name in enumerate(basis):
        factors: list[str] = []
        for (kind, value), power in generated_factors[target_index].items():
            rendered_name = (
                _render_fraction(value)
                if kind == "coefficient"
                else str(value)
            )
            factors.append(_render_factor(rendered_name, power))
        output[target_name] = " * ".join(factors) if factors else "1"
    return output
