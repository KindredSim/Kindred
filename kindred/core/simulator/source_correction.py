"""Pure global planning of source-backed exact correction coordinates."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction

from kindred.core.simulator.exact_constraint_composition import (
    ExactMonomial,
    ExactConstraintComposition,
)
from kindred.core.simulator.dsl_build_input import source_coordinate_capabilities

__all__ = [
    "CorrectionTarget",
    "CorrectionPlan",
    "CorrectionPlanner",
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
    constraint_basis: tuple[str, ...]
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
    if family in {"kf", "kr"}:
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
        if not power:
            continue
        if kind == "coefficient":
            if ExactMonomial(Fraction(value)).power(power) is None:
                return True
            continue
        if kind != "symbol":
            return True
        name = str(value)
        if name == "C0" or re.fullmatch(r"degeneracy[1-9]\d*", name):
            return True
    return False


def _solve_correction_factors(
    constraints: Sequence[object],
    basis: Sequence[str],
) -> tuple[
    tuple[tuple[Fraction, ...], ...],
    tuple[dict[tuple[str, str | Fraction], Fraction], ...],
] | None:
    """Solve one basis and aggregate its exact generated RHS factors."""

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
        constraint_basis=tuple(constraint.cycle_id for constraint in constraints),
        available=True,
    )


def _multiplicative_basis_is_admissible(
    constraints: Sequence[object],
    candidate: Sequence[CorrectionTarget],
) -> bool:
    """Apply the one multiplicative rank/materialization admission rule."""

    if len(candidate) != len(constraints):
        return False
    candidate_rows = [
        tuple(_direct_source_power(item, target.name) for target in candidate)
        for item in constraints
    ]
    if _rank(candidate_rows) != len(constraints):
        return False
    solution = _solve_correction_factors(
        constraints,
        tuple(target.name for target in candidate),
    )
    if solution is None:
        return False
    _inverse_matrix, generated_factors = solution
    return not any(
        _has_unmaterializable_generated_factor(factors)
        for factors in generated_factors
    )


_ENERGY_FAMILIES = frozenset({"Ea", "dG_act", "dG_eq"})
_ENERGY_COMMON_POWERS = frozenset({"T"})


def _energy_term_data(
    term: ExactMonomial,
) -> tuple[str, Fraction, tuple[tuple[str, Fraction], ...]] | None:
    """Split one linear energy term into its symbolic leaf and common factors."""

    if term.sum_terms or term.exponentials:
        return None
    symbolic_powers = [
        (str(name), Fraction(power))
        for name, power in term.powers
        if str(name) not in _ENERGY_COMMON_POWERS
    ]
    if len(symbolic_powers) != 1 or symbolic_powers[0][1] != 1:
        return None
    energy_name, _energy_power = symbolic_powers[0]
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
            _energy_term_data(term)
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


def _energy_basis_is_admissible(
    constraints: Sequence[object],
    candidate: Sequence[CorrectionTarget],
) -> bool:
    """Apply the one bounded linear-energy basis admission rule."""

    if len(candidate) != len(constraints) or not candidate:
        return False
    if any(target.family not in _ENERGY_FAMILIES for target in candidate):
        return False
    rows = _energy_rows(constraints)
    if rows is None:
        return False
    matrix = [
        tuple(row.get(target.name, Fraction(0)) for target in candidate)
        for row in rows
    ]
    return _rank(matrix) == len(constraints) and _inverse(matrix) is not None


def _independent_row_indices(
    rows: Sequence[Sequence[Fraction]],
) -> tuple[int, ...]:
    """Return the first source-ordered row basis of one exact matrix."""

    selected: list[int] = []
    current_rank = 0
    for index, row in enumerate(rows):
        candidate = [rows[item] for item in selected] + [row]
        candidate_rank = _rank(candidate)
        if candidate_rank > current_rank:
            selected.append(index)
            current_rank = candidate_rank
    return tuple(selected)


def _span_coefficients(
    basis_rows: Sequence[Sequence[Fraction]],
    row: Sequence[Fraction],
) -> tuple[Fraction, ...] | None:
    """Express one row in an independent row basis, if possible."""

    if not basis_rows:
        return () if not any(row) else None
    required = len(basis_rows)
    pivot_columns: list[int] = []
    current_rank = 0
    for column in range(len(row)):
        projected = [
            tuple(source[item] for item in (*pivot_columns, column))
            for source in basis_rows
        ]
        projected_rank = _rank(projected)
        if projected_rank > current_rank:
            pivot_columns.append(column)
            current_rank = projected_rank
            if current_rank == required:
                break
    if len(pivot_columns) != required:
        return None
    matrix = [
        tuple(source[column] for column in pivot_columns)
        for source in basis_rows
    ]
    inverse = _inverse(matrix)
    if inverse is None:
        return None
    pivot_values = tuple(Fraction(row[column]) for column in pivot_columns)
    coefficients = tuple(
        sum(
            (pivot_values[column] * inverse[column][basis_index]
             for column in range(required)),
            Fraction(0),
        )
        for basis_index in range(required)
    )
    reconstructed = tuple(
        sum(
            (coefficient * Fraction(source[column])
             for coefficient, source in zip(coefficients, basis_rows, strict=True)),
            Fraction(0),
        )
        for column in range(len(row))
    )
    return coefficients if reconstructed == tuple(map(Fraction, row)) else None


def _multiplicative_rows_are_consistent(
    constraints: Sequence[object],
    target_rows: Sequence[Sequence[Fraction]],
    independent_indices: Sequence[int],
) -> bool:
    """Prove that every exact product row follows from the retained row basis."""

    basis_rows = tuple(target_rows[index] for index in independent_indices)
    basis_constraints = tuple(constraints[index] for index in independent_indices)
    for constraint, row in zip(constraints, target_rows, strict=True):
        coefficients = _span_coefficients(basis_rows, row)
        if coefficients is None:
            return False
        combined = ExactMonomial()
        for basis_constraint, coefficient in zip(
            basis_constraints,
            coefficients,
            strict=True,
        ):
            powered = basis_constraint.monomial.power(coefficient)
            if powered is None:
                return False
            combined = combined.multiply(powered)
        if combined != constraint.monomial:
            return False
    return True


def _linear_rows_are_consistent(
    rows: Sequence[Mapping[str, Fraction]],
    target_rows: Sequence[Sequence[Fraction]],
    independent_indices: Sequence[int],
) -> bool:
    """Prove that every linear energy row follows from the retained row basis."""

    basis_target_rows = tuple(target_rows[index] for index in independent_indices)
    basis_rows = tuple(rows[index] for index in independent_indices)
    for row, target_row in zip(rows, target_rows, strict=True):
        coefficients = _span_coefficients(basis_target_rows, target_row)
        if coefficients is None:
            return False
        names = set(row)
        for basis_row in basis_rows:
            names.update(basis_row)
        reconstructed = {
            name: value
            for name in names
            if (
                value := sum(
                    (
                        coefficient * basis_row.get(name, Fraction(0))
                        for coefficient, basis_row in zip(
                            coefficients,
                            basis_rows,
                            strict=True,
                        )
                    ),
                    Fraction(0),
                )
            )
        }
        if reconstructed != dict(row):
            return False
    return True


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
    rows = _energy_rows(constraints)
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
        self._constraints = _correction_constraints(composition)
        self._eligible = self._ordered_eligible_targets()
        (
            self._system_kind,
            self._system_constraints,
            self._system_rows,
            self._system_targets,
            self._system_reason,
        ) = self._build_correction_system()
        self._completion_cache: dict[frozenset[str], bool] = {}

    @property
    def required_basis_size(self) -> int:
        """Number of dependent coordinates required by the exact system."""

        return len(self._system_constraints)

    def _ordered_eligible_targets(self) -> tuple[CorrectionTarget, ...]:
        source_order = tuple(self.targets)
        eligible = [target for target in self.targets.values() if target.eligible]
        eligible.sort(
            key=lambda target: _preference(
                target,
                source_order.index(target.name),
            )
        )
        return tuple(eligible)

    def _build_correction_system(
        self,
    ) -> tuple[
        str,
        tuple[object, ...],
        tuple[dict[str, Fraction], ...],
        tuple[CorrectionTarget, ...],
        str | None,
    ]:
        """Build the one independent exact system used by every planner API."""

        constraints = self._constraints
        if not constraints:
            return "multiplicative", (), (), self._eligible, None

        if not any(_has_unsupported_source_factor(item) for item in constraints):
            kind = "multiplicative"
            system_targets = self._eligible
            rows = tuple(dict(item.monomial.powers) for item in constraints)
        else:
            kind = "energy"
            system_targets = tuple(
                target
                for target in self._eligible
                if target.family in _ENERGY_FAMILIES
            )
            energy_rows = _energy_rows(constraints)
            if energy_rows is None:
                return (
                    kind,
                    (),
                    (),
                    system_targets,
                    "exact source correction requires a bounded linear energy projection",
                )
            rows = energy_rows

        target_names = tuple(target.name for target in system_targets)
        target_rows = tuple(
            tuple(row.get(name, Fraction(0)) for name in target_names)
            for row in rows
        )
        independent_indices = _independent_row_indices(target_rows)
        if not independent_indices:
            return (
                kind,
                (),
                (),
                system_targets,
                "no source-backed editable unlocked global full-rank correction basis is available",
            )
        consistent = (
            _multiplicative_rows_are_consistent(
                constraints,
                target_rows,
                independent_indices,
            )
            if kind == "multiplicative"
            else _linear_rows_are_consistent(
                rows,
                target_rows,
                independent_indices,
            )
        )
        if not consistent:
            return (
                kind,
                (),
                (),
                system_targets,
                "the exact constraints contain independent requirements without "
                "enough editable correction coordinates",
            )
        return (
            kind,
            tuple(constraints[index] for index in independent_indices),
            tuple(rows[index] for index in independent_indices),
            system_targets,
            None,
        )

    def _column_rank(self, targets: Sequence[CorrectionTarget]) -> int:
        return _rank(
            [
                tuple(row.get(target.name, Fraction(0)) for target in targets)
                for row in self._system_rows
            ]
        )

    def _basis_is_admissible(
        self,
        candidate: Sequence[CorrectionTarget],
    ) -> bool:
        if self._system_reason is not None:
            return False
        if self._system_kind == "multiplicative":
            return _multiplicative_basis_is_admissible(
                self._system_constraints,
                candidate,
            )
        return _energy_basis_is_admissible(self._system_constraints, candidate)

    def _selected_targets(
        self,
        selected_names: Sequence[str],
    ) -> tuple[CorrectionTarget, ...] | None:
        selected = tuple(selected_names)
        selected_set = set(selected)
        if len(selected_set) != len(selected):
            return None
        if not selected_set.issubset(target.name for target in self._system_targets):
            return None
        return tuple(
            target for target in self._system_targets if target.name in selected_set
        )

    def _rank_only_completion_is_safe(self) -> bool:
        if self._system_kind == "energy":
            return True
        return all(
            item.monomial.coefficient == 1
            and not any(
                name == "C0" or re.fullmatch(r"degeneracy[1-9]\d*", name)
                for name, _power in item.monomial.powers
            )
            for item in self._system_constraints
        )

    def _selection_can_complete(
        self,
        selected: tuple[CorrectionTarget, ...],
    ) -> bool:
        selected_names = frozenset(target.name for target in selected)
        cached = self._completion_cache.get(selected_names)
        if cached is not None:
            return cached
        required = self.required_basis_size
        selected_rank = self._column_rank(selected)
        if selected_rank != len(selected):
            result = False
        elif len(selected) == required:
            result = self._basis_is_admissible(selected)
        elif self._rank_only_completion_is_safe():
            result = True
        else:
            result = any(
                self._selection_can_complete((*selected, target))
                for target in self._system_targets
                if target.name not in selected_names
                and self._column_rank((*selected, target)) == len(selected) + 1
            )
        self._completion_cache[selected_names] = result
        return result

    def partial_selection_can_complete(
        self,
        selected_names: Sequence[str],
    ) -> bool:
        """Whether one dependent-target selection extends to an exact basis."""

        if any(item.status == "unresolved" for item in self.composition.constraints):
            return False
        if self._system_reason is not None:
            return False
        required = self.required_basis_size
        selected = self._selected_targets(selected_names)
        if selected is None or len(selected) > required:
            return False
        return self._selection_can_complete(selected)

    def viable_next_targets(
        self,
        selected_names: Sequence[str],
    ) -> tuple[str, ...]:
        """Return every unchecked target preserving at least one completion."""

        selected = tuple(selected_names)
        selected_set = set(selected)
        if len(selected) >= self.required_basis_size:
            return ()
        return tuple(
            target.name
            for target in self._system_targets
            if target.name not in selected_set
            and self.partial_selection_can_complete((*selected, target.name))
        )

    def plan_selected_basis(
        self,
        selected_names: Sequence[str],
    ) -> CorrectionPlan:
        """Validate and build the exact plan for one user-selected basis."""

        unresolved = tuple(
            item for item in self.composition.constraints if item.status == "unresolved"
        )
        if unresolved:
            reasons = "; ".join(
                f"{item.cycle_id}: {item.reason or 'unresolved exact constraint'}"
                for item in unresolved
            )
            return CorrectionPlan((), (), False, reasons)
        if self._system_reason is not None:
            return CorrectionPlan((), (), False, self._system_reason)
        selected = self._selected_targets(selected_names)
        if selected is None:
            return CorrectionPlan(
                (),
                (),
                False,
                "selected correction targets must be distinct and eligible",
            )
        if len(selected) != self.required_basis_size:
            return CorrectionPlan(
                (),
                (),
                False,
                f"select exactly {self.required_basis_size} dependent parameters",
            )
        if not self._basis_is_admissible(selected):
            return CorrectionPlan(
                (),
                (),
                False,
                "selected correction targets do not form an admissible full-rank basis",
            )
        return _plan_for_basis(self._system_constraints, selected)

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
        unsatisfied = tuple(
            item for item in self._constraints if item.status == "unsatisfied"
        )
        if not unsatisfied:
            return CorrectionPlan((), (), True)
        if self._system_reason is not None:
            return CorrectionPlan((), (), False, self._system_reason)
        basis: list[CorrectionTarget] = []
        for target in self._system_targets:
            candidate = (*basis, target)
            if self._column_rank(candidate) == len(candidate):
                basis.append(target)
                if len(basis) == self.required_basis_size:
                    break
        if self._basis_is_admissible(basis):
            return _plan_for_basis(self._system_constraints, basis)
        if self._selection_can_complete(()) and not self._rank_only_completion_is_safe():
            for target in self._system_targets:
                if self._selection_can_complete((target,)):
                    basis = [target]
                    while len(basis) < self.required_basis_size:
                        next_target = next(
                            item
                            for item in self._system_targets
                            if item not in basis
                            and self._selection_can_complete((*basis, item))
                        )
                        basis.append(next_target)
                    return _plan_for_basis(self._system_constraints, basis)
        return CorrectionPlan(
            (),
            (),
            False,
            "no source-backed editable unlocked global full-rank correction basis is available",
        )


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
    constraints_by_id = {
        item.cycle_id: item
        for item in _correction_constraints(composition)
    }
    try:
        constraints = tuple(
            constraints_by_id[cycle_id]
            for cycle_id in plan.constraint_basis
        )
    except KeyError as exc:
        raise ValueError("Correction plan constraint basis is absent from the composition.") from exc
    basis = tuple(plan.basis)
    if len(constraints) != len(basis):
        raise ValueError("Correction plan constraint and parameter bases differ in size.")

    if any(_has_unsupported_source_factor(item) for item in constraints):
        expressions = _build_energy_correction_expressions(constraints, basis)
        if expressions is None:
            raise ValueError(
                "Exact source correction requires a bounded linear energy projection."
            )
        return expressions

    solution = _solve_correction_factors(constraints, basis)
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
