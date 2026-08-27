"""Pure global planning of source-backed exact correction coordinates."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations

from kindred.core.simulator.exact_constraint_composition import (
    ExactConstraintComposition,
)

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
    algebra_owned: bool = False
    derived: bool = False
    dependencies: tuple[str, ...] = ()
    canonical_identity: tuple[str, str] | None = None
    source_origin: str | None = None

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
            and not self.algebra_owned
            and not self.derived
            and self.canonical_identity is not None
            and self.canonical_identity[0] == self.name
            and self.canonical_identity[1]
            in {"A", "Ea", "dG_act", "dG_eq", "kf", "kr", "kf+kr", "Keq", "k"}
            and isinstance(self.source_origin, str)
            and bool(self.source_origin)
            and "\n" not in self.source_origin
            and "\r" not in self.source_origin
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
    return (4, 0, source_index)


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
            or bool(item.powers)
        )
    )


def _direct_source_power(item: object, target_name: str) -> Fraction:
    """Return only a multiplicative authored factor's power."""

    return Fraction(dict(item.powers).get(str(target_name), Fraction(0)))


def _has_unsupported_source_factor(item: object) -> bool:
    """Reject factors that cannot be authored as a static correction RHS."""

    if item.coefficient == 0:
        return True
    return any(
        (
            str(name).startswith("exp(")
            or str(name) == "T"
            or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(name)) is None
        )
        for name, _power in item.powers
    )


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
            constraint_coefficient = Fraction(constraint.coefficient)
            if constraint_coefficient != 1:
                key = ("coefficient", constraint_coefficient)
                factors[key] = factors.get(key, Fraction(0)) - exponent
            for name, power in constraint.powers:
                if not power:
                    continue
                key = ("symbol", str(name))
                factors[key] = factors.get(key, Fraction(0)) - Fraction(power) * exponent
        generated.append(
            {
                key: power
                for key, power in factors.items()
                if power and not (key[0] == "symbol" and key[1] in selected_names)
            }
        )
    return inverse, tuple(generated)


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
        if any(not item.powers for item in unsatisfied):
            return CorrectionPlan(
                (),
                (),
                False,
                "an unsatisfied exact constraint has no source-backed powers",
            )
        if any(_has_unsupported_source_factor(item) for item in constraints):
            return CorrectionPlan(
                (),
                (),
                False,
                "exact source correction requires an unsupported non-multiplicative inverse",
            )
        eligible = [target for target in self.targets.values() if target.eligible]
        eligible.sort(
            key=lambda target: _preference(
                target,
                tuple(self.targets).index(target.name),
            )
        )
        rows = [
            tuple(_direct_source_power(item, target.name) for target in eligible)
            for item in constraints
        ]
        if _rank(rows) != len(constraints):
            return CorrectionPlan((), (), False, "exact constraints are rank-deficient")
        selected: tuple[CorrectionTarget, ...] | None = None
        cycle_reason: str | None = None
        for candidate in combinations(eligible, len(constraints)):
            candidate_rows = [
                tuple(
                    _direct_source_power(item, item_target.name)
                    for item_target in candidate
                )
                for item in constraints
            ]
            if _rank(candidate_rows) != len(constraints):
                continue
            solution = _solve_correction_factors(
                self.composition,
                tuple(target.name for target in candidate),
            )
            if solution is None:
                continue
            _inverse_matrix, generated_factors = solution
            selected_names = {target.name for target in candidate}
            generated_by_target = dict(
                zip(
                    (target.name for target in candidate),
                    generated_factors,
                    strict=True,
                )
            )
            dependency_graph = {
                target.name: tuple(
                    dict.fromkeys(
                        (
                            tuple(str(dependency) for dependency in target.dependencies)
                            + (
                                tuple(
                                    str(value)
                                    for (kind, value), power in generated_by_target[target.name].items()
                                    if kind == "symbol" and power
                                )
                                if target.name in selected_names
                                else ()
                            )
                        )
                    )
                )
                for target in self.targets.values()
            }
            visiting: list[str] = []
            visited: set[str] = set()

            def visit(name: str) -> str | None:
                if name not in dependency_graph:
                    # Symbols outside the admitted target set are ordinary
                    # authored leaves, not dependency edges owned here.
                    return None
                if name in visiting:
                    start = visiting.index(name)
                    return " -> ".join((*visiting[start:], name))
                if name in visited:
                    return None
                visiting.append(name)
                for dependency in dependency_graph[name]:
                    cycle = visit(dependency)
                    if cycle is not None:
                        return cycle
                visiting.pop()
                visited.add(name)
                return None

            cycle: str | None = None
            for name in (target.name for target in candidate):
                cycle = visit(name)
                if cycle is not None:
                    break
            if cycle is None:
                selected = candidate
                break
            cycle_reason = cycle
        if selected is None:
            if cycle_reason is not None:
                return CorrectionPlan(
                    (),
                    (),
                    False,
                    f"generated correction dependency cycle: {cycle_reason}",
                )
            return CorrectionPlan(
                (),
                (),
                False,
                "no source-backed editable unlocked global full-rank generated-expression-acyclic basis is available",
            )
        return CorrectionPlan(
            basis=tuple(target.name for target in selected),
            assignments=tuple(
                (constraint.cycle_id, target.name)
                for target, constraint in zip(selected, constraints, strict=True)
            ),
            available=True,
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
    if any(_has_unsupported_source_factor(item) for item in constraints):
        raise ValueError("Exact source correction cannot invert an exponential factor.")
    basis = tuple(plan.basis)
    solution = _solve_correction_factors(composition, basis)
    if solution is None:
        raise ValueError("Correction basis is not invertible.")
    _inverse_matrix, generated_factors = solution
    assignment_map = dict(plan.assignments)
    if tuple(assignment_map.get(item.cycle_id) for item in constraints) != basis:
        raise ValueError("Correction plan assignments do not match its global basis.")
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
