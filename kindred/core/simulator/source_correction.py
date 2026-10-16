"""Pure global planning of source-backed exact correction coordinates.

Each reversible cycle requires ``sum(c_i * ln Keq_i) = 0`` over its steps.  A
correction makes one target per independent unsatisfied cycle dependent and
solves the composed requirements for those targets exactly.  A
``kf``/``kr``/``Keq`` closure is a product of names: every ``KeqN`` that does
not depend on a target stays a name, one that does is written out through the
formulas that depend on a target, and an energy those formulas name is written
as its step's ``KeqN``.  A ``dG_eq`` closure is a sum of energies where one
exists: every value is written out down to energies, and an energy that does
not depend on a target stays a name.  Otherwise a step whose ``KeqN`` does not
come from energies enters as ``R*T*ln`` of its ``KeqN``.  Every written closure
is read back as the cycle check reads an admitted ``param`` line.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction

from kindred.core.algebra.parser import BinaryNode, CallNode, IdentNode, NumberNode, UnaryNode
from kindred.core.algebra.printer import expression_to_source
from kindred.core.algebra.unit_preparation import authored_projection, prepare_expression
from kindred.core.simulator.dsl_build_input import source_coordinate_name_parts
from kindred.core.simulator.exact_constraint_composition import (
    ExactConstraint,
    ExactConstraintComposition,
    ExactMonomial,
    cycle_message,
    names_other_step,
)

__all__ = [
    "CorrectionTarget",
    "CorrectionPlan",
    "CorrectionPlanner",
    "build_correction_expressions",
]


@dataclass(frozen=True, slots=True)
class CorrectionTarget:
    """Canonical source/editability facts for one cycle coordinate.

    ``param_line`` marks a coordinate set by its own ``param`` line.  Correction
    rewrites only reaction-line values without a ``param`` line.
    """

    name: str
    family: str
    param_line: bool = False
    # The step's reaction text, as Sliders show it after the name.
    context: str = ""


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


_FAMILY_ORDER = ("kr", "kf", "Keq", "dG_eq")
_ENERGY_SCALE = (("R", Fraction(-1)), ("T", Fraction(-1)))


def _is_energy(name: str) -> bool:
    parts = source_coordinate_name_parts(name)
    return parts is not None and parts[0] == "dG_eq"


def _target_step(name: str) -> int:
    return source_coordinate_name_parts(name)[1]


def _partner(name: str) -> str | None:
    family, step = source_coordinate_name_parts(name)
    return {"kr": f"kf{step}", "kf": f"kr{step}"}.get(family)


def _atom(name: str) -> tuple:
    """The log atom whose coefficient is a target's unknown."""

    if _is_energy(name):
        return ("exp", tuple(sorted((*_ENERGY_SCALE, (name, Fraction(1))))), (), None)
    return ("power", name)


def _atom_names(key: tuple) -> set[str]:
    if key[0] == "power":
        return {key[1]}
    return {name for name, _power in key[1]}.union(
        *(item.names() for item in key[2]), {key[3]} - {None}
    )


def _without(value: ExactMonomial, atoms: set[tuple]) -> ExactMonomial:
    """``value`` with the given log atoms removed."""

    powers = tuple(item for item in value.powers if ("power", item[0]) not in atoms)
    terms = tuple(
        term
        for argument in value.exponentials
        for term in argument.sum_terms or (argument,)
        if ("exp", term.powers, term.exponentials, term.logarithm) not in atoms
    )
    exponentials = (
        ()
        if not terms
        else (terms[0],) if len(terms) == 1 else (ExactMonomial(sum_terms=terms),)
    )
    return ExactMonomial(value.coefficient, powers, exponentials)


def _energy_terms(value: ExactMonomial) -> list[tuple[ExactMonomial, str]] | None:
    """Each ``c*E/(R*T)`` term in ``value``'s exponentials with its energy ``E``.

    ``None`` when any other term appears there.
    """

    found = []
    for argument in value.exponentials:
        for term in argument.sum_terms or (argument,):
            energies = [item for item in term.powers if item not in _ENERGY_SCALE]
            if (
                term.exponentials
                or term.logarithm
                or len(term.powers) != 3
                or len(energies) != 1
                or energies[0][1] != 1
            ):
                return None
            found.append((term, energies[0][0]))
    return found


def _named_energies(
    composition: ExactConstraintComposition,
    value: ExactMonomial,
    forbidden: set[str],
) -> ExactMonomial | None:
    """``value`` with each ``dG_eqN`` energy written as its step's ``KeqN``.

    A step written out through a formula in energies names other steps'
    ``dG_eq``; a product closure names their ``KeqN`` instead, so both
    spellings of one value cancel.  ``None`` when an energy has no such name.
    """

    terms = _energy_terms(value)
    if terms is None:
        return None
    result = ExactMonomial(value.coefficient, value.powers)
    for term, energy in terms:
        keq = f"Keq{_target_step(energy)}" if _is_energy(energy) else None
        relation = composition.relations.get(keq)
        if (
            relation is None
            or keq in forbidden
            or relation.exponentials != (ExactMonomial(Fraction(-1), term.powers),)
        ):
            return None
        # KeqN = rest * exp(-dG_eqN/(R*T)), so exp(c*dG_eqN/(R*T)) = (rest/KeqN)^c.
        lifted = ExactMonomial(
            relation.coefficient, (*relation.powers, (keq, Fraction(-1)))
        ).power(term.coefficient)
        if lifted is None:
            return None
        result = result.multiply(lifted)
    return result


def _energies_only(value: ExactMonomial | None) -> bool:
    """Whether ``value`` is ``exp`` of energies, with ``C0`` its only power."""

    return (
        value is not None
        and not value.sum_terms
        and value.coefficient == 1
        and all(name == "C0" for name, _power in value.powers)
        and _energy_terms(value) is not None
    )


def _closures(
    composition: ExactConstraintComposition,
    cycles: Sequence[ExactConstraint],
    basis: Sequence[str],
    integral: bool = False,
) -> dict[str, str] | None:
    """Solve the cycles for the basis and write each closure, or ``None``.

    For a ``kf``/``kr``/``Keq`` closure each step's ``KeqN`` stays a name
    unless it depends on a basis target; then it is written out through the
    formulas that depend on one, and each energy they name becomes its step's
    ``KeqN`` (:func:`_named_energies`).  A ``dG_eq`` closure writes a step whose
    ``KeqN`` comes from energies as those energies, an energy that does not
    depend on a target staying a name, and any other step as ``R*T*ln`` of its
    ``KeqN``.  A closure therefore never reads a basis target, and the composed
    requirement is solved exactly.  Every closure must name another step's
    value (:func:`names_other_step`), and is offered only when the cycle check
    and unit preparation read the written line as closing the cycles.
    ``integral`` also refuses a closure with a fractional power or weight.
    """

    selected = set(basis)
    readers = {name for name, reads in composition.reads.items() if reads & selected}
    forbidden = readers | selected
    # An energy closure writes every other value down to energies; an energy
    # that does not depend on a target stays a name.
    through_energies = readers | {
        name for name in composition.relations if not _is_energy(name)
    }
    atoms = [_atom(name) for name in basis]
    atom_set = set(atoms)
    steps = sorted({step for cycle in cycles for step, _c in cycle.step_coefficients})
    views: dict[tuple[int, str], ExactMonomial | None] = {}

    def view(step: int, mode: str) -> ExactMonomial | None:
        """``KeqN`` as a product (``"product"``), written down to energies
        (``"energy"``), or as energies where it comes from energies and a name
        otherwise (``"mixed"``)."""

        key = (step, mode)
        if key not in views:
            name = f"Keq{step}"
            value = composition.compose(name, through_energies) if mode != "product" else None
            if mode == "product" or (
                mode == "mixed" and name not in forbidden and not _energies_only(value)
            ):
                value = (
                    ExactMonomial(Fraction(1), ((name, Fraction(1)),))
                    if name not in forbidden
                    else composition.compose(name, readers)
                )
            views[key] = value
        return views[key]

    def combine(weights: Mapping[int, Fraction], mode: str) -> ExactMonomial | None:
        product = ExactMonomial()
        for step, weight in weights.items():
            if not weight:
                continue
            value = view(step, mode)
            if value is None or value.sum_terms:
                return None
            rest = _without(value, atom_set)
            if rest.names() & forbidden:
                return None
            powered = rest.power(weight)
            if powered is None:
                return None
            product = product.multiply(powered)
        return product

    vectors = {}
    for step in steps:
        value = view(step, "product")
        if value is None or value.sum_terms:
            return None
        vectors[step] = _log_atoms(value)
    matrix = [
        [
            sum(
                (Fraction(c) * vectors[step].get(atom, Fraction(0)) for step, c in cycle.step_coefficients),
                Fraction(0),
            )
            for atom in atoms
        ]
        for cycle in cycles
    ]
    inverse = _inverse(matrix)
    if inverse is None:
        return None
    units = dict(composition.coordinate_units)

    def admitted(name: str, expression, named: set[str]):
        """``name``'s closure line as admission reads it, or ``None``."""

        # A param line cannot read the standard concentration ``C0``.
        if "C0" in named or not names_other_step(
            composition.step_names, named, _target_step(name)
        ):
            return None
        expression = authored_projection(expression, units.get)
        line = prepare_expression(expression, units.get)
        if line.quantity_unit is not None and not line.quantity_unit.compatible(units[name]):
            return None
        return expression, line

    written: dict[str, str] = {}
    prepared = {}
    for name, inverse_row in zip(basis, inverse, strict=True):
        weights: dict[int, Fraction] = {}
        for multiplier, cycle in zip(inverse_row, cycles, strict=True):
            for step, c in cycle.step_coefficients:
                weights[step] = weights.get(step, Fraction(0)) - multiplier * c
        own_step = _target_step(name)
        if _is_energy(name):
            # A sum of energies when one exists, else R*T*ln of the other
            # steps' KeqN, writing as energies those that come from energies.
            # A form is used only when admission accepts its line.
            for mode in ("energy", "mixed", "product"):
                product = combine(weights, mode)
                terms = (
                    None
                    if product is None
                    or product.coefficient != 1
                    or (mode == "energy" and product.powers)
                    else _energy_terms(product)
                )
                if terms is None:
                    continue
                energies: dict[str, Fraction] = {}
                for term, item in terms:
                    energies[item] = energies.get(item, Fraction(0)) + term.coefficient
                energies = {key: value for key, value in energies.items() if value}
                logarithm = ExactMonomial(Fraction(1), product.powers)
                # Every form carries the same exponents, so a fractional one
                # fails the integral search before admission is consulted.
                if integral and any(
                    value.denominator != 1 for value in (*energies.values(), *dict(product.powers).values())
                ):
                    return None
                closure = admitted(
                    name,
                    _energy_closure(energies, logarithm, own_step),
                    {*energies, *dict(product.powers)},
                )
                if closure is not None:
                    break
            else:
                return None
        else:
            product = combine(weights, "product")
            product = None if product is None else _named_energies(composition, product, forbidden)
            if product is None:
                return None
            if integral and any(value.denominator != 1 for value in dict(product.powers).values()):
                return None
            closure = admitted(name, _monomial_closure(product, own_step), set(dict(product.powers)))
            if closure is None:
                return None
        expression, line = closure
        written[name] = expression_to_source(expression)
        prepared[name] = line
    return written if composition.closes(cycles, prepared) else None


def _axpy(
    vector: Mapping[object, Fraction],
    factor: Fraction,
    other: Mapping[object, Fraction],
) -> dict[object, Fraction]:
    result = dict(vector)
    for key, value in other.items():
        result[key] = result.get(key, Fraction(0)) + factor * value
    return {key: value for key, value in result.items() if value}


def _log_atoms(monomial: ExactMonomial) -> dict[object, Fraction]:
    """Exponents of a composed product's independent factors."""

    atoms: dict[object, Fraction] = {("power", name): power for name, power in monomial.powers}
    for argument in monomial.exponentials:
        for term in argument.sum_terms or (argument,):
            key = ("exp", term.powers, term.exponentials, term.logarithm)
            atoms[key] = atoms.get(key, Fraction(0)) + term.coefficient
    return {key: value for key, value in atoms.items() if value}


def _independent_cycles(
    cycles: Sequence[ExactConstraint],
) -> tuple[tuple[ExactConstraint, ...], str | None]:
    """Keep the cycles whose composed requirements are independent.

    Authored relationships can make several topology cycles impose the same
    requirement.  A dependent cycle is closed by closing the kept ones only
    when its product equals the matching product of theirs, coefficient
    included.  Otherwise no correction exists, and the reason says why.
    """

    basis: list[tuple[object, dict[object, Fraction], dict[object, Fraction]]] = []
    kept: list[int] = []
    dependent: list[tuple[int, dict[object, Fraction]]] = []
    for index, cycle in enumerate(cycles):
        vector = _log_atoms(cycle.monomial)
        combination: dict[object, Fraction] = {}
        for pivot, row, row_combination in basis:
            factor = vector.get(pivot)
            if factor:
                vector = _axpy(vector, -factor, row)
                combination = _axpy(combination, factor, row_combination)
        if not vector:
            dependent.append((index, combination))
            continue
        pivot = next(iter(vector))
        scale = vector[pivot]
        basis.append((
            pivot,
            {key: value / scale for key, value in vector.items()},
            {key: value / scale for key, value in _axpy({index: Fraction(1)}, Fraction(-1), combination).items()},
        ))
        kept.append(index)
    for index, combination in dependent:
        cycle = cycles[index]
        if not combination:
            return (), "No rewritten value can change a constant written in formulas."
        involved = (cycle, *(cycles[other] for other in combination))
        lines = tuple(dict.fromkeys(line for item in involved for line in item.lines))
        product = ExactMonomial()
        for kept_index, exponent in combination.items():
            powered = cycles[kept_index].monomial.power(exponent)
            if powered is None:
                return (), cycle_message(
                    "These reversible cycles repeat each other only up to a root "
                    "that Kindred cannot write exactly.",
                    lines,
                )
            product = product.multiply(powered)
        if product.coefficient != cycle.monomial.coefficient or _log_atoms(product) != _log_atoms(cycle.monomial):
            return (), cycle_message(
                "These reversible cycles need the same values to differ, so no set "
                "of rewritten values closes them all.",
                lines,
            )
    return tuple(cycles[index] for index in kept), None


class CorrectionPlanner:
    """Choose one acyclic, source-backed, full-rank global basis.

    Planning uses the inconsistent cycles whose composed requirements are
    independent; closing those closes every dependent one.  A target is a
    candidate when its unknown is one log factor of those composed products,
    wherever its value enters them.  One exact predicate, ``_closures``,
    decides admissibility for the automatic plan, the user's selection and
    the chooser, and renders the proposal.  Cycles sharing steps, or whose
    steps read a common candidate, form blocks that complete independently.
    Where every candidate enters only through its own step, no ``dG_eq``
    candidate exists and every readout admits any root, rank alone decides;
    otherwise the block's subsets are searched.
    """

    def __init__(
        self,
        composition: ExactConstraintComposition,
        targets: Sequence[CorrectionTarget],
    ) -> None:
        self._composition = composition
        self._blocking = tuple(
            item
            for item in composition.constraints
            if item.status == "unresolved" and not item.kept
        )
        open_cycles = tuple(
            item
            for item in composition.constraints
            if item.status == "unsatisfied" or (item.status == "unresolved" and item.kept)
        )
        self._open_cycles = open_cycles
        self._cycles: tuple[ExactConstraint, ...] = ()
        self._contradiction: str | None = None
        if not self._blocking:
            self._cycles, self._contradiction = _independent_cycles(open_cycles)
        self._vectors = tuple(_log_atoms(cycle.monomial) for cycle in self._cycles)
        self._steps = {step.step_index: step for step in composition.steps}
        self._reads = composition.reads
        self._candidates: tuple[CorrectionTarget, ...] = ()
        self._blocks: tuple[
            tuple[tuple[ExactConstraint, ...], tuple[CorrectionTarget, ...], bool], ...
        ] = ()
        self._block_of: dict[str, int] = {}
        self._cache: dict[tuple[int, frozenset[str], bool], bool] = {}
        if self._blocking or self._contradiction:
            return
        source_order = tuple(target.name for target in targets)
        candidates = tuple(filter(self._is_candidate, targets))
        self._touched = {target.name: self._touched_steps(target) for target in candidates}
        self._candidates = tuple(
            sorted(
                candidates,
                key=lambda target: (
                    _FAMILY_ORDER.index(target.family),
                    source_order.index(target.name),
                ),
            )
        )
        self._blocks = self._build_blocks()

    def _is_candidate(self, target: CorrectionTarget) -> bool:
        """Whether a target's unknown is one separable log factor of the cycles."""

        if target.param_line:
            return False
        atom = _atom(target.name)
        return any(vector.get(atom) for vector in self._vectors) and not any(
            target.name in _atom_names(key)
            for vector in self._vectors
            for key in vector
            if key != atom
        )

    def _touched_steps(self, target: CorrectionTarget) -> set[int]:
        return {
            step
            for cycle in self._cycles
            for step, _c in cycle.step_coefficients
            if f"Keq{step}" == target.name
            or target.name in self._reads.get(f"Keq{step}", frozenset())
        }

    def _direct(self, target: CorrectionTarget) -> bool:
        return self._touched[target.name] == {_target_step(target.name)} and (
            target.name not in self._reads.get(_partner(target.name), frozenset())
        )

    def _admits_any_root(self, step: int) -> bool:
        keq = self._steps[step].keq
        return keq.coefficient == 1 and keq.power(Fraction(1, 2)) is not None

    def _build_blocks(
        self,
    ) -> tuple[tuple[tuple[ExactConstraint, ...], tuple[CorrectionTarget, ...], bool], ...]:
        """Group cycles sharing steps or a candidate read by their steps."""

        parent = list(range(len(self._cycles)))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def join(indices: Sequence[int]) -> None:
            for index in indices[1:]:
                parent[find(index)] = find(indices[0])

        step_cycles: dict[int, list[int]] = {}
        for index, cycle in enumerate(self._cycles):
            for step, _c in cycle.step_coefficients:
                step_cycles.setdefault(step, []).append(index)
        for indices in step_cycles.values():
            join(indices)
        touched = self._touched
        for steps in touched.values():
            join([index for step in steps for index in step_cycles[step]])
        groups: dict[int, list[int]] = {}
        for index in range(len(self._cycles)):
            groups.setdefault(find(index), []).append(index)
        blocks = []
        for indices in groups.values():
            cycles = tuple(self._cycles[index] for index in indices)
            steps = {step for cycle in cycles for step, _c in cycle.step_coefficients}
            candidates = tuple(
                target for target in self._candidates if touched[target.name] & steps
            )
            for target in candidates:
                self._block_of[target.name] = len(blocks)
            rank_decides = all(
                target.family != "dG_eq" and self._direct(target) for target in candidates
            ) and all(self._admits_any_root(step) for step in steps)
            blocks.append((cycles, candidates, rank_decides))
        return tuple(blocks)

    def _columns(
        self,
        cycles: Sequence[ExactConstraint],
        targets: Sequence[CorrectionTarget],
    ) -> list[list[Fraction]]:
        vectors = [_log_atoms(cycle.monomial) for cycle in cycles]
        return [
            [vector.get(_atom(target.name), Fraction(0)) for target in targets]
            for vector in vectors
        ]

    def _block_can_complete(
        self,
        index: int,
        selected: tuple[CorrectionTarget, ...],
        integral: bool = False,
    ) -> bool:
        cycles, candidates, rank_decides = self._blocks[index]
        # Where rank decides, a step's kf and kr give closures with the same
        # powers, so the integral search visits each set of steps once.
        by_step = integral and rank_decides

        def member(name: str) -> str:
            return _target_step(name) if by_step else name

        members = frozenset(member(target.name) for target in selected)
        if len(members) != len(selected):
            # Two targets of one step can never both be dependent.
            return False
        key = (index, members, integral)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        if len(selected) > len(cycles) or _rank(self._columns(cycles, selected)) != len(selected):
            result = False
        elif rank_decides and not integral:
            result = _rank(self._columns(cycles, candidates)) == len(cycles)
        elif len(selected) == len(cycles):
            result = _closures(
                self._composition, cycles, tuple(target.name for target in selected), integral
            ) is not None
        else:
            result = any(
                self._block_can_complete(index, (*selected, target), integral)
                for target in candidates
                if member(target.name) not in members
            )
        self._cache[key] = result
        return result

    def _selection_can_complete(
        self, selected: tuple[CorrectionTarget, ...], integral: bool = False
    ) -> bool:
        grouped: list[list[CorrectionTarget]] = [[] for _block in self._blocks]
        for target in selected:
            index = self._block_of.get(target.name)
            if index is None:
                return False
            grouped[index].append(target)
        return all(
            self._block_can_complete(index, tuple(group), integral)
            for index, group in enumerate(grouped)
        )

    @property
    def required_basis_size(self) -> int:
        """Number of dependent coordinates required by the exact system."""

        return len(self._cycles)

    def _selected_targets(
        self,
        selected_names: Sequence[str],
    ) -> tuple[CorrectionTarget, ...] | None:
        selected = set(selected_names)
        if len(selected) != len(selected_names):
            return None
        chosen = tuple(target for target in self._candidates if target.name in selected)
        return chosen if len(chosen) == len(selected) else None

    def partial_selection_can_complete(
        self,
        selected_names: Sequence[str],
    ) -> bool:
        """Whether one dependent-target selection extends to an exact basis."""

        if self._blocking or self._contradiction:
            return False
        selected = self._selected_targets(selected_names)
        if selected is None or len(selected) > self.required_basis_size:
            return False
        return self._selection_can_complete(selected)

    def viable_next_targets(
        self,
        selected_names: Sequence[str],
    ) -> tuple[str, ...]:
        """Return every unchecked target preserving at least one completion."""

        selected = tuple(selected_names)
        if len(selected) >= self.required_basis_size:
            return ()
        return tuple(
            target.name
            for target in self._candidates
            if target.name not in selected
            and self.partial_selection_can_complete((*selected, target.name))
        )

    def _unavailable_plan(self) -> CorrectionPlan | None:
        if self._blocking:
            return CorrectionPlan(
                (), (), False, "\n\n".join(item.reason for item in self._blocking)
            )
        if self._contradiction:
            return CorrectionPlan((), (), False, self._refusal(self._contradiction))
        return None

    def _refusal(self, conclusion: str) -> str:
        """Every inconsistent cycle's sentence, then why no rewrite closes them."""

        return "\n\n".join((*(item.reason for item in self._open_cycles), conclusion))

    def _plan_for_basis(self, basis: Sequence[CorrectionTarget]) -> CorrectionPlan:
        return CorrectionPlan(
            basis=tuple(target.name for target in basis),
            constraint_basis=tuple(cycle.cycle_id for cycle in self._cycles),
            available=True,
        )

    def plan_selected_basis(
        self,
        selected_names: Sequence[str],
    ) -> CorrectionPlan:
        """Validate and build the exact plan for one user-selected basis."""

        unavailable = self._unavailable_plan()
        if unavailable is not None:
            return unavailable
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
        if not self._selection_can_complete(selected):
            return CorrectionPlan(
                (),
                (),
                False,
                "selected correction targets do not form an admissible full-rank basis",
            )
        return self._plan_for_basis(selected)

    def plan(self) -> CorrectionPlan:
        unavailable = self._unavailable_plan()
        if unavailable is not None:
            return unavailable
        if not self._cycles:
            return CorrectionPlan((), (), True)
        # The automatic plan prefers closures without fractional powers, then
        # kr, kf, Keq and dG_eq targets in source order.
        for integral in (True, False):
            basis: tuple[CorrectionTarget, ...] = ()
            while len(basis) < self.required_basis_size:
                following = next(
                    (
                        target
                        for target in self._candidates
                        if target not in basis
                        and self._selection_can_complete((*basis, target), integral)
                    ),
                    None,
                )
                if following is None:
                    break
                basis = (*basis, following)
            if len(basis) == self.required_basis_size:
                return self._plan_for_basis(basis)
        reason = (
            "No kf, kr, Keq, or dG_eq written on a reaction line without its "
            "own param line can be rewritten to close every inconsistent cycle. "
            "A rewrite must be written from other steps' values without reading "
            "itself: a kf, kr, or Keq as a product with exact powers, a dG_eq "
            "as a sum of energies and R*T*ln of other steps' values."
        )
        return CorrectionPlan((), (), False, self._refusal(reason))


def _fraction_node(value: Fraction):
    numerator = NumberNode(float(value.numerator), str(value.numerator))
    return numerator if value.denominator == 1 else BinaryNode(
        "/", numerator, NumberNode(float(value.denominator), str(value.denominator)))


def _factor_order(own_step: int):
    """Order factors: the target's own step, then steps, then other names."""

    def key(item: tuple[str, Fraction]) -> tuple:
        parts = source_coordinate_name_parts(item[0])
        if parts is None:
            return (2, 0, item[0])
        return (0 if parts[1] == own_step else 1, parts[1], item[0])

    return key


def _product(factors: Sequence[tuple[str, Fraction]]):
    node = None
    for name, power in factors:
        factor = IdentNode(name)
        if power != 1:
            factor = BinaryNode("^", factor, _fraction_node(power))
        node = factor if node is None else BinaryNode("*", node, factor)
    return node


def _monomial_closure(value: ExactMonomial, own_step: int):
    """Render ``coefficient * prod(name ** power)`` as one ratio."""

    ordered = sorted(value.powers, key=_factor_order(own_step))
    numerator = _product([(name, power) for name, power in ordered if power > 0])
    denominator = _product([(name, -power) for name, power in ordered if power < 0])
    if value.coefficient.numerator != 1:
        number = _fraction_node(Fraction(value.coefficient.numerator))
        numerator = number if numerator is None else BinaryNode("*", number, numerator)
    if value.coefficient.denominator != 1:
        number = _fraction_node(Fraction(value.coefficient.denominator))
        denominator = number if denominator is None else BinaryNode("*", number, denominator)
    numerator = numerator if numerator is not None else NumberNode(1.0, "1")
    return numerator if denominator is None else BinaryNode("/", numerator, denominator)


def _energy_closure(weights: dict[str, Fraction], logarithm: ExactMonomial, own_step: int):
    """Render ``sum(weight * energy) + R*T*ln(logarithm)``, signs in the operators."""

    node = None
    for name, weight in sorted(weights.items(), key=_factor_order(own_step)):
        identifier = IdentNode(name)
        magnitude = abs(weight) if node is not None else weight
        if magnitude == 1:
            term = identifier
        elif magnitude == -1:
            term = UnaryNode("-", identifier)
        else:
            term = BinaryNode("*", _fraction_node(magnitude), identifier)
        node = term if node is None else BinaryNode("+" if weight > 0 else "-", node, term)
    if logarithm.powers:
        negative = all(power < 0 for _name, power in logarithm.powers)
        argument = _monomial_closure(logarithm.power(-1) if negative else logarithm, own_step)
        gas = UnaryNode("-", IdentNode("R")) if negative and node is None else IdentNode("R")
        term = BinaryNode("*", BinaryNode("*", gas, IdentNode("T")), CallNode("ln", (argument,)))
        node = term if node is None else BinaryNode("-" if negative else "+", node, term)
    return node if node is not None else NumberNode(0.0, "0")


def build_correction_expressions(
    composition: ExactConstraintComposition,
    plan: CorrectionPlan,
) -> dict[str, str]:
    """Render the exact coupled solution for the selected global basis."""

    if not plan.available or not plan.basis:
        return {}
    cycles_by_id = {item.cycle_id: item for item in composition.constraints}
    closures = _closures(
        composition,
        tuple(cycles_by_id[cycle_id] for cycle_id in plan.constraint_basis),
        plan.basis,
    )
    if closures is None:
        raise ValueError("Kindred cannot write a closure for the selected parameters.")
    return closures
