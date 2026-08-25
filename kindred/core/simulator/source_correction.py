"""Pure global planning of source-backed exact correction coordinates."""

from __future__ import annotations

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
            and self.canonical_identity[1] in {
                "A", "Ea", "dG_act", "dG_eq", "kf", "kr", "kf+kr", "Keq", "k"
            }
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
        pivot = next((row for row in range(rank, len(matrix)) if matrix[row][column]), None)
        if pivot is None:
            continue
        matrix[rank], matrix[pivot] = matrix[pivot], matrix[rank]
        scale = matrix[rank][column]
        matrix[rank] = [value / scale for value in matrix[rank]]
        for row in range(len(matrix)):
            if row == rank or not matrix[row][column]:
                continue
            factor = matrix[row][column]
            matrix[row] = [value - factor * pivot for value, pivot in zip(matrix[row], matrix[rank])]
        rank += 1
        if rank == len(matrix):
            break
    return rank


def _inverse(matrix: Sequence[Sequence[Fraction]]) -> tuple[tuple[Fraction, ...], ...] | None:
    size = len(matrix)
    if not size or any(len(row) != size for row in matrix):
        return None
    augmented = [
        [Fraction(value) for value in row]
        + [Fraction(1 if row_index == column else 0) for column in range(size)]
        for row_index, row in enumerate(matrix)
    ]
    for column in range(size):
        pivot = next((row for row in range(column, size) if augmented[row][column]), None)
        if pivot is None:
            return None
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(size):
            if row == column or not augmented[row][column]:
                continue
            factor = augmented[row][column]
            augmented[row] = [value - factor * pivot for value, pivot in zip(augmented[row], augmented[column])]
    return tuple(tuple(row[size:]) for row in augmented)


def _preference(target: CorrectionTarget, source_index: int) -> tuple[int, int, int]:
    """Rank correction families, then retain authenticated source order.

    Source order is supplied by ProjectDocument from the admitted authored
    mechanism.  It is deliberately not reconstructed from a coordinate name:
    ``k2`` and ``k10`` are distinct identities whose stable tie break is their
    authored order.
    """

    family = str(target.family)
    if family in {"kf+kr", "kf", "kr"}:
        return (0, 1 if target.name.startswith("kf") else 0, source_index)
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
            or bool(item.source_power_map())
            or bool(item.source_factor_map())
        )
    )


class CorrectionPlanner:
    """Choose one acyclic, source-backed, full-rank global basis."""

    def __init__(self, composition: ExactConstraintComposition, targets: Mapping[str, CorrectionTarget] | Sequence[CorrectionTarget]) -> None:
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
        if any(not item.source_powers for item in unsatisfied):
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
        if not constraints:
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
        rows = [
            tuple(_direct_source_power(item, target.name) for target in eligible)
            for item in constraints
        ]
        if _rank(rows) != len(constraints):
            return CorrectionPlan((), (), False, "exact constraints are rank-deficient")
        selected: tuple[CorrectionTarget, ...] | None = None
        # Search combinations in preference order.  Greedy selection can
        # choose a cyclic tempting coordinate and incorrectly reject a later
        # acyclic full-rank basis.
        for candidate in combinations(eligible, len(constraints)):
            candidate_rows = [
                tuple(_direct_source_power(item, item_target.name) for item_target in candidate)
                for item in constraints
            ]
            if _rank(candidate_rows) != len(constraints):
                continue
            selected_names = {target.name for target in candidate}
            visiting: set[str] = set()
            visited: set[str] = set()

            def visit(name: str) -> bool:
                if name in visiting:
                    return False
                if name in visited:
                    return True
                visiting.add(name)
                for dependency in self.targets[name].dependencies:
                    dependency = str(dependency)
                    if dependency not in self.targets:
                        return False
                    if not visit(dependency):
                        return False
                visiting.remove(name)
                visited.add(name)
                return True

            if all(visit(name) for name in selected_names):
                selected = candidate
                break
        if selected is None:
            return CorrectionPlan(
                (),
                (),
                False,
                "no source-backed editable unlocked global full-rank acyclic basis is available",
            )
        matrix = [
            tuple(_direct_source_power(item, target.name) for target in selected)
            for item in constraints
        ]
        if _inverse(matrix) is None:
            return CorrectionPlan((), (), False, "selected correction basis is not invertible")
        return CorrectionPlan(
            basis=tuple(target.name for target in selected),
            assignments=tuple((constraint.cycle_id, target.name) for target, constraint in zip(selected, constraints, strict=True)),
            available=True,
        )


def plan_correction_basis(composition: ExactConstraintComposition, targets: Mapping[str, CorrectionTarget] | Sequence[CorrectionTarget]) -> CorrectionPlan:
    return CorrectionPlanner(composition, targets).plan()


def _render_fraction(value: Fraction) -> str:
    value = Fraction(value)
    return str(value.numerator) if value.denominator == 1 else f"({value.numerator}/{value.denominator})"


def _render_factor(name: str, power: Fraction) -> str:
    return name if power == 1 else f"{name}**({_render_fraction(power)})"


def _direct_source_power(item: object, target_name: str) -> Fraction:
    """Return only a multiplicative authored factor's power.

    A coordinate appearing solely inside ``exp(...)`` cannot be solved by
    multiplying factors.  Treating its dependency projection as a monomial
    power creates self-referential or dynamically invalid source edits, so
    such a target is deliberately unavailable until a canonical inverse is
    admitted.
    """

    source_factors = tuple(getattr(item, "source_factors", ()) or ())
    if source_factors:
        return Fraction(dict(source_factors).get(str(target_name), Fraction(0)))
    return Fraction(getattr(item, "source_power_map")().get(str(target_name), Fraction(0)))


def _has_unsupported_source_factor(item: object) -> bool:
    """Reject factors that cannot be authored as a static correction RHS."""

    return any(
        str(name).startswith("exp(") or str(name) == "T"
        for name, _power in item.source_factor_map().items()
    )


def build_correction_expressions(composition: ExactConstraintComposition, plan: CorrectionPlan) -> dict[str, str]:
    """Render the exact coupled solution for the selected global basis."""
    if not plan.available or not plan.basis:
        return {}
    constraints = _correction_constraints(composition)
    if any(_has_unsupported_source_factor(item) for item in constraints):
        raise ValueError("Exact source correction cannot invert an exponential factor.")
    basis = tuple(plan.basis)
    inverse = _inverse(
        [
            tuple(_direct_source_power(item, name) for name in basis)
            for item in constraints
        ]
    )
    if inverse is None:
        raise ValueError("Correction basis is not invertible.")
    assignment_map = dict(plan.assignments)
    if tuple(assignment_map.get(item.cycle_id) for item in constraints) != basis:
        raise ValueError("Correction plan assignments do not match its global basis.")
    output: dict[str, str] = {}
    for target_index, target_name in enumerate(basis):
        factors: list[str] = []
        for row_index, constraint in enumerate(constraints):
            exponent = inverse[target_index][row_index]
            if not exponent:
                continue
            # The source-coordinate row is already the exact authored
            # equation.  ``constraint.coefficient`` belongs to the fully
            # substituted residual and must not be substituted into this
            # symbolic solve a second time; doing so applies the current
            # residual twice (for example 36/25 instead of 6/5).
            source_coefficient = Fraction(constraint.source_coefficient)
            if source_coefficient != 1:
                factors.append(_render_factor(_render_fraction(source_coefficient), -exponent))
            for name, power in constraint.source_factor_map().items():
                if name not in basis and power:
                    factors.append(_render_factor(name, -power * exponent))
        output[target_name] = " * ".join(factors) if factors else "1"
    return output
