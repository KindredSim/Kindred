"""Pure exact composition of canonical symbolic mechanism constraints."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

from kindred.core.constants import R, h, k_B
from kindred.core.equilibrium_rate_authority import equilibrium_rate_symbolic_relations
from kindred.core.simulator.step_indexing import get_step_index_map
from kindred.core.simulator.wegscheider_symbolic import WegscheiderConstraintReport, analyze_wegscheider_constraints

__all__ = [
    "ExactMonomial",
    "ExactConstraint",
    "ExactConstraintComposition",
    "compose_exact_constraints",
    "exact_monomial_from_ast",
]

_EXACT_CONSTANTS = {"R": Fraction(str(R)), "k_B": Fraction(str(k_B)), "h": Fraction(str(h))}


@dataclass(frozen=True, slots=True)
class ExactMonomial:
    coefficient: Fraction = Fraction(1)
    powers: tuple[tuple[str, Fraction], ...] = ()
    exponentials: tuple["ExactMonomial", ...] = ()
    # A compact immutable additive expression is needed only as an
    # exponential argument (for example ``exp(-Ea/(R*T) + dG)``).  General
    # products of sums remain unresolved rather than becoming a second
    # symbolic engine.
    sum_terms: tuple["ExactMonomial", ...] = ()

    def __post_init__(self) -> None:
        coefficient = Fraction(self.coefficient)
        powers = {str(name): Fraction(power) for name, power in self.powers if Fraction(power) != 0}
        sum_terms = tuple(self.sum_terms)
        if sum_terms:
            flat_terms: list[ExactMonomial] = []
            for term in sum_terms:
                flat_terms.extend(term.sum_terms or (term,))
            grouped_terms: dict[
                tuple[tuple[tuple[str, Fraction], ...], tuple["ExactMonomial", ...]],
                Fraction,
            ] = {}
            for term in flat_terms:
                key = (term.powers, term.exponentials)
                grouped_terms[key] = grouped_terms.get(key, Fraction(0)) + term.coefficient
            normalized_terms = tuple(
                ExactMonomial(coefficient, powers, nested)
                for (powers, nested), coefficient in grouped_terms.items()
                if coefficient != 0
            )
            coefficient = Fraction(1)
            powers = {}
            exponentials: tuple["ExactMonomial", ...] = ()
            sum_terms = normalized_terms
            if not sum_terms:
                coefficient = Fraction(0)
        elif coefficient == 0:
            powers = {}
            exponentials: tuple["ExactMonomial", ...] = ()
        else:
            exponentials = tuple(self.exponentials)
        object.__setattr__(self, "coefficient", coefficient)
        object.__setattr__(self, "powers", tuple(sorted(powers.items())))
        object.__setattr__(self, "exponentials", exponentials)
        object.__setattr__(self, "sum_terms", sum_terms)

    @property
    def base_powers(self) -> tuple[tuple[str, Fraction], ...]:
        return self.powers

    def power_map(self) -> dict[str, Fraction]:
        result = dict(self.powers)
        for argument in self.exponentials:
            result[f"exp({_render_monomial(argument)})"] = Fraction(1)
        return result

    def scale(self, factor: Fraction) -> "ExactMonomial":
        """Scale a scalar expression by an exact factor.

        Exponential arguments are part of the expression tree.  Scaling only
        their top-level coefficient (while retaining their variable powers)
        is the algebraically correct operation for ``q * (a*T**p)``.
        """

        factor = Fraction(factor)
        if self.sum_terms:
            return ExactMonomial(
                sum_terms=tuple(term.scale(factor) for term in self.sum_terms)
            )
        return ExactMonomial(
            self.coefficient * factor,
            self.powers,
            tuple(argument.scale(factor) for argument in self.exponentials),
        )

    @staticmethod
    def _aggregate_exponentials(
        exponentials: tuple["ExactMonomial", ...],
    ) -> tuple["ExactMonomial", ...]:
        grouped: dict[
            tuple[tuple[tuple[str, Fraction], ...], tuple["ExactMonomial", ...]],
            Fraction,
        ] = {}
        additive: list[ExactMonomial] = []
        for argument in exponentials:
            if argument.sum_terms:
                opposite = -argument
                match = next(
                    (index for index, current in enumerate(additive) if current == opposite),
                    None,
                )
                if match is None:
                    additive.append(argument)
                else:
                    additive.pop(match)
                continue
            key = (argument.base_powers, argument.exponentials)
            grouped[key] = grouped.get(key, Fraction(0)) + argument.coefficient
        return tuple(
            ExactMonomial(coefficient, powers, nested)
            for (powers, nested), coefficient in sorted(grouped.items(), key=lambda item: repr(item[0]))
            if coefficient != 0
        ) + tuple(additive)

    def multiply(self, other: "ExactMonomial") -> "ExactMonomial":
        if self.sum_terms or other.sum_terms:
            return None  # type: ignore[return-value]
        powers = dict(self.powers)
        for name, power in other.powers:
            powers[name] = powers.get(name, Fraction(0)) + power
        return ExactMonomial(
            self.coefficient * other.coefficient,
            tuple(powers.items()),
            self._aggregate_exponentials(self.exponentials + other.exponentials),
        )

    def divide(self, other: "ExactMonomial") -> "ExactMonomial | None":
        if self.sum_terms or other.sum_terms:
            return None
        if other.coefficient == 0:
            return None
        powers = dict(self.powers)
        for name, power in other.powers:
            powers[name] = powers.get(name, Fraction(0)) - power
        return ExactMonomial(
            self.coefficient / other.coefficient,
            tuple(powers.items()),
            self._aggregate_exponentials(
                self.exponentials
                + tuple(-argument for argument in other.exponentials)
            ),
        )

    def __mul__(self, other: "ExactMonomial") -> "ExactMonomial":
        return self.multiply(other)

    def __rmul__(self, other: "ExactMonomial") -> "ExactMonomial":
        return self.multiply(other)

    def __truediv__(self, other: "ExactMonomial") -> "ExactMonomial":
        result = self.divide(other)
        if result is None:
            raise ZeroDivisionError("exact monomial division by zero")
        return result

    def __neg__(self) -> "ExactMonomial":
        if self.sum_terms:
            return ExactMonomial(sum_terms=tuple(-term for term in self.sum_terms))
        return ExactMonomial(-self.coefficient, self.powers, self.exponentials)

    def __pow__(self, exponent: "ExactMonomial") -> "ExactMonomial | None":
        if exponent.powers or exponent.exponentials or exponent.sum_terms:
            return None
        return self.power(exponent.coefficient)

    def power(self, exponent: Fraction) -> "ExactMonomial" | None:
        exponent = Fraction(exponent)
        if self.sum_terms:
            return self if exponent == 1 else None
        if self.coefficient == 0 and exponent == 0:
            return None
        if self.coefficient == 0:
            if exponent <= 0:
                return None
            return ExactMonomial(
                0,
                tuple((name, power * exponent) for name, power in self.powers),
                tuple(argument.scale(exponent) for argument in self.exponentials),
            )
        numerator, denominator = exponent.numerator, exponent.denominator
        root = _exact_rational_root(self.coefficient, denominator)
        if root is None:
            return None
        try:
            coefficient = root ** numerator
        except (OverflowError, ZeroDivisionError, ValueError):
            return None
        return ExactMonomial(
            coefficient,
            tuple((name, power * exponent) for name, power in self.powers),
            tuple(argument.scale(exponent) for argument in self.exponentials),
        )


def _exact_integer_root(value: int, degree: int) -> int | None:
    if value < 0 or degree <= 0:
        return None
    if value in {0, 1} or degree == 1:
        return value
    low, high = 0, 1 << ((value.bit_length() + degree - 1) // degree + 1)
    while low + 1 < high:
        middle = (low + high) // 2
        power = middle ** degree
        if power == value:
            return middle
        if power < value:
            low = middle
        else:
            high = middle
    return low if low ** degree == value else None


def _exact_rational_root(value: Fraction, degree: int) -> Fraction | None:
    if degree <= 0:
        return None
    if degree == 1:
        return value
    if value < 0:
        if degree % 2 == 0:
            return None
        root = _exact_rational_root(-value, degree)
        return None if root is None else -root
    numerator = _exact_integer_root(value.numerator, degree)
    denominator = _exact_integer_root(value.denominator, degree)
    if numerator is None or denominator is None:
        return None
    return Fraction(numerator, denominator)


def exact_monomial_from_ast(
    expr: object,
    *,
    canonical_names: set[str] | frozenset[str],
    constant_values: Mapping[str, Fraction] | None = None,
) -> ExactMonomial | None:
    """Lower only retained exact AST arithmetic; unsupported nodes stay unresolved."""

    constants = dict(_EXACT_CONSTANTS)
    if constant_values:
        constants.update({str(name): Fraction(value) for name, value in constant_values.items()})
    # The AST is admitted and owned by the Algebra parser.  Composition only
    # consumes its retained node shape; importing parser/evaluator authority
    # here would turn this pure lowering pass into another Algebra boundary.
    node_kind = type(expr).__name__
    if node_kind == "NumberNode":
        exact = getattr(expr, "exact_value", None)
        return None if exact is None else ExactMonomial(exact)
    if node_kind == "IdentNode":
        name = str(getattr(expr, "name", ""))
        if name in constants:
            return ExactMonomial(constants[name])
        if name in canonical_names:
            return ExactMonomial(Fraction(1), ((name, Fraction(1)),))
        return None
    if node_kind == "SpeciesRefNode":
        return None
    if node_kind == "CallNode":
        # Function calls remain owned by the admitted Algebra evaluator.  In
        # particular, do not create a local ``exp`` scientific special case:
        # canonical kinetic/equilibrium relations already arrive as typed
        # ExactMonomial trees.
        return None
    if node_kind == "UnaryNode":
        value = exact_monomial_from_ast(getattr(expr, "rhs", None), canonical_names=canonical_names, constant_values=constants)
        if value is None:
            return None
        if getattr(expr, "op", None) == "+":
            return value
        if getattr(expr, "op", None) == "-":
            return -value
        return None
    if node_kind != "BinaryNode":
        return None
    op = getattr(expr, "op", None)
    left = exact_monomial_from_ast(getattr(expr, "lhs", None), canonical_names=canonical_names, constant_values=constants)
    right = exact_monomial_from_ast(getattr(expr, "rhs", None), canonical_names=canonical_names, constant_values=constants)
    if op == "*" and left is not None and right is not None:
        return left.multiply(right)
    if op == "/" and left is not None and right is not None:
        return left.divide(right)
    if op == "**" and left is not None:
        exponent = exact_monomial_from_ast(
            getattr(expr, "rhs", None),
            canonical_names=canonical_names,
            constant_values=constants,
        )
        # A power is admissible only when the *whole* exponent is one exact
        # scalar.  A scalar coefficient beside a residual symbol, additive
        # term, or exponential is not a scalar exponent: accepting it would
        # silently erase the unprovable part (for example ``x**(y + 1)``).
        if (
            exponent is None
            or exponent.powers
            or exponent.exponentials
            or exponent.sum_terms
        ):
            return None
        return left.power(exponent.coefficient)
    if (
        op in {"+", "-"}
        and left is not None
        and right is not None
        and left.powers == right.powers
        and left.exponentials == right.exponentials
        and not left.sum_terms
        and not right.sum_terms
    ):
        coefficient = left.coefficient + (right.coefficient if op == "+" else -right.coefficient)
        return ExactMonomial(coefficient, left.powers, left.exponentials)
    if op in {"+", "-"} and left is not None and right is not None:
        left_terms = left.sum_terms or (left,)
        right_terms = right.sum_terms or (right,)
        if op == "-":
            right_terms = tuple(-term for term in right_terms)
        return ExactMonomial(sum_terms=tuple(left_terms) + tuple(right_terms))
    return None


@dataclass(frozen=True, slots=True)
class ExactConstraint:
    cycle_id: str
    powers: tuple[tuple[str, Fraction], ...]
    expression: str
    status: str = "resolved"
    reason: str | None = None
    coefficient: Fraction = Fraction(1)


@dataclass(frozen=True, slots=True)
class ExactConstraintComposition:
    constraints: tuple[ExactConstraint, ...]


def _canonical_names(mechanism: object, spec: object | None, report: WegscheiderConstraintReport) -> set[str]:
    names: set[str] = set()
    names.update(name for constraint in report.constraints for name, _power in constraint.terms)
    for entry in get_step_index_map(mechanism):
        step_index = entry.get("step_index")
        kind = str(entry.get("kind") or "")
        if kind == "equilibrium":
            names.update({f"kf{step_index}", f"kr{step_index}", f"Keq{step_index}"})
        elif kind == "reaction":
            names.add(f"k{step_index}")
        source_coordinates = entry.get("source_coordinates") or {}
        if isinstance(source_coordinates, Mapping):
            names.update(str(name) for name in source_coordinates.values())
    if spec is not None:
        block = getattr(spec, "parameter_block", None)
        statements = (
            block.statements_in_evaluation_order()
            if block is not None
            and callable(getattr(block, "statements_in_evaluation_order", None))
            else ()
        )
        names.update(str(statement.name) for statement in statements)
    return names


def _render_monomial(value: ExactMonomial) -> str:
    if value.sum_terms:
        return "(" + " + ".join(_render_monomial(term) for term in value.sum_terms) + ")"
    factors: list[str] = []
    if value.coefficient != 1 or not value.powers:
        factors.append(str(value.coefficient))
    factors.extend(
        name
        if power == 1
        else f"{name}**({power.numerator}/{power.denominator})"
        if power.denominator != 1
        else f"{name}**({power.numerator})"
        for name, power in value.powers
    )
    factors.extend(
        f"exp({_render_monomial(argument)})"
        for argument in value.exponentials
    )
    return " * ".join(factors) if factors else "1"


def _mechanism_relations(
    mechanism: object,
    names: set[str],
) -> dict[str, ExactMonomial]:
    """Build only source-coordinate symbolic equilibrium relations.

    Equilibrium authority is lowered from admitted ``kf``/``kr``/``Keq`` or
    thermodynamic source coordinates.  Numeric rate bindings are downstream
    simulation state and are intentionally not consulted here.
    """

    relations: dict[str, ExactMonomial] = {}
    equilibria = tuple(getattr(mechanism, "equilibria", ()) or ())
    metadata = getattr(mechanism, "metadata", {}) or {}
    standard_conc = metadata.get("standard_conc_M")
    if standard_conc is None:
        return relations
    temperature = ExactMonomial(Fraction(1), (("T", Fraction(1)),))

    def literal(value: float) -> ExactMonomial:
        return ExactMonomial(Fraction(str(value)))

    def exponential(value: ExactMonomial) -> ExactMonomial:
        return ExactMonomial(Fraction(1), (), (value,))

    for entry in get_step_index_map(mechanism):
        if str(entry.get("kind") or "") != "equilibrium":
            continue
        try:
            index = int(entry.get("equilibrium_index"))
            step = int(entry.get("step_index"))
        except (TypeError, ValueError):
            continue
        if not 0 <= index < len(equilibria):
            continue
        kf, kr, keq = f"kf{step}", f"kr{step}", f"Keq{step}"
        if not {kf, kr, keq}.issubset(names):
            continue
        source_coordinates = {
            str(role): str(canonical_name)
            for role, canonical_name in dict(
                entry.get("source_coordinates") or {}
            ).items()
        }

        authority = entry.get("equilibrium_authority")
        authority = authority if isinstance(authority, Mapping) else {}
        has_authored_kf = bool(entry.get("user_provided_kf"))
        has_authored_kr = bool(entry.get("user_provided_kr"))
        has_authored_keq = bool(authority.get("has_explicit_keq_param"))

        def source_coordinate(role: str) -> ExactMonomial:
            name = source_coordinates.get(role)
            # Only independently authored roles may become source symbols.
            # Derived nominal values are never promoted from typed state.
            if name is None and role == "kf" and has_authored_kf:
                name = f"kf{step}"
            elif name is None and role == "kr" and has_authored_kr:
                name = f"kr{step}"
            elif name is None and role == "Keq" and has_authored_keq:
                name = f"Keq{step}"
            if name is None:
                raise ValueError(
                    f"missing canonical source coordinate for {role!r}"
                )
            return ExactMonomial(Fraction(1), ((name, Fraction(1)),))

        if not has_authored_kf:
            continue
        # Equilibrium DSL authority always authors kf directly.  Keep it as a
        # source coordinate; numeric bindings and kinetic model projections
        # belong to downstream simulation preparation, not this analysis.
        try:
            projected = source_coordinate("kf")
        except ValueError:
            continue
        try:
            equilibrium_relations = equilibrium_rate_symbolic_relations(
                equilibria[index],
                forward_relation=projected,
                temperature=temperature,
                standard_conc_M=standard_conc,
                literal=literal,
                exponential=exponential,
                source_coordinate=source_coordinate,
            )
        except (TypeError, ValueError, ZeroDivisionError, OverflowError):
            continue
        for name, relation in (
            (kf, equilibrium_relations.kf),
            (kr, equilibrium_relations.kr),
            (keq, equilibrium_relations.Keq),
        ):
            # A canonical coordinate with no defining relation is free.
            # Leaving that identity out lets substitution
            # terminate at the retained coordinate instead of recursing
            # into ``kf1 = kf1`` (and likewise for kr/Keq).
            if relation != ExactMonomial(Fraction(1), ((name, Fraction(1)),)):
                relations[name] = relation
    return relations


def _algebra_relations(spec: object | None, names: set[str]) -> tuple[dict[str, ExactMonomial], set[str]]:
    if spec is None or spec.parameter_block is None:
        return {}, set()
    statements = spec.parameter_block.statements_in_evaluation_order()
    allowed = set(names) | {str(statement.name) for statement in statements}
    relations: dict[str, ExactMonomial] = {}
    unresolved: set[str] = set()

    def identifiers(node: object) -> set[str]:
        if node is None:
            return set()
        if type(node).__name__ == "IdentNode":
            return {str(getattr(node, "name", ""))}
        found: set[str] = set()
        for attribute in ("lhs", "rhs"):
            found.update(identifiers(getattr(node, attribute, None)))
        for argument in getattr(node, "args", ()) or ():
            found.update(identifiers(argument))
        return found

    for statement in statements:
        # A literal-only Algebra declaration is an independent authored
        # coordinate.  Its numeric default is not a symbolic relation and
        # must never decide whether a cycle is balanced.
        if not identifiers(statement.expr):
            continue
        value = exact_monomial_from_ast(statement.expr, canonical_names=allowed)
        if value is None:
            unresolved.add(str(statement.name))
        else:
            relations[str(statement.name)] = value
    return relations, unresolved


def _substitute(
    monomial: ExactMonomial,
    relations: Mapping[str, ExactMonomial],
    unresolved: set[str],
) -> ExactMonomial | None:
    """Recursively lower retained exact expressions in evaluation order.

    Names are resolved through the admitted relation graph.  Exponential
    arguments are traversed as expression trees (rather than rendered and
    reparsed), so ``exp(-Ea/(R*T))`` retains its exact coefficient, fraction,
    symbolic temperature, and constants.  A sum is valid inside an
    exponential argument but not as a top-level multiplicative factor.
    """

    cache: dict[str, ExactMonomial | None] = {}
    visiting: set[str] = set()

    def resolve(name: str) -> ExactMonomial | None:
        if name in cache:
            return cache[name]
        if name in unresolved or name in visiting:
            cache[name] = None
            return None
        relation = relations.get(name)
        if relation is None:
            value = ExactMonomial(Fraction(1), ((name, Fraction(1)),))
            cache[name] = value
            return value
        visiting.add(name)
        result = substitute_value(relation, allow_sum=False)
        visiting.remove(name)
        cache[name] = result
        return result

    def substitute_value(
        value: ExactMonomial,
        *,
        allow_sum: bool,
    ) -> ExactMonomial | None:
        if value.sum_terms:
            if not allow_sum:
                return None
            terms: list[ExactMonomial] = []
            for term in value.sum_terms:
                substituted = substitute_value(term, allow_sum=False)
                if substituted is None:
                    return None
                terms.append(substituted)
            return ExactMonomial(sum_terms=tuple(terms))

        arguments: list[ExactMonomial] = []
        for argument in value.exponentials:
            substituted = substitute_value(argument, allow_sum=True)
            if substituted is None:
                return None
            arguments.append(substituted)
        result = ExactMonomial(value.coefficient, (), tuple(arguments))
        for name, power in value.base_powers:
            resolved = resolve(name)
            powered = resolved.power(power) if resolved is not None else None
            if powered is None:
                return None
            result = result.multiply(powered)
            if result is None:
                return None
        return result

    return substitute_value(monomial, allow_sum=False)


def _render_constraint(monomial: ExactMonomial) -> str:
    if monomial.sum_terms:
        return f"product({_render_monomial(monomial)}) = 1"
    factors: list[str] = []
    if monomial.coefficient != 1 or not monomial.powers:
        factors.append(str(monomial.coefficient))
    factors.extend(
        name
        if power == 1
        else f"{name}**({power.numerator}/{power.denominator})"
        if power.denominator != 1
        else f"{name}**({power.numerator})"
        for name, power in monomial.powers
    )
    factors.extend(
        f"exp({_render_monomial(argument)})"
        for argument in monomial.exponentials
    )
    return f"product({' * '.join(factors)}) = 1"


def _rebind_equilibrium_relations(
    mechanism: object,
    relations: dict[str, ExactMonomial],
) -> None:
    """Apply the admitted step-map dependency direction without new authority.

    The accepted equilibrium authority is already recorded on each canonical
    step-map entry.  Composition only follows that fact to keep derived
    ``Keq``/``kr`` coordinates from becoming a second source; it does not
    inspect rendered source or evaluate a numerical residual.
    """

    def identity(name: str) -> ExactMonomial:
        return ExactMonomial(Fraction(1), ((name, Fraction(1)),))

    for entry in get_step_index_map(mechanism):
        if str(entry.get("kind") or "") != "equilibrium":
            continue
        try:
            step = int(entry.get("step_index"))
        except (TypeError, ValueError):
            continue
        kf, kr, keq = f"kf{step}", f"kr{step}", f"Keq{step}"
        authority = entry.get("equilibrium_authority")
        if not isinstance(authority, Mapping):
            continue
        kind = str(authority.get("kind") or "")
        if kind == "kr_authoritative":
            derived = relations.get(kf, identity(kf)).divide(
                relations.get(kr, identity(kr))
            )
            if derived is not None:
                relations[keq] = derived
            continue
        if kind != "Keq_authoritative":
            continue
        derived = relations.get(kf, identity(kf)).divide(
            relations.get(keq, identity(keq))
        )
        if derived is not None:
            relations[kr] = derived


def compose_exact_constraints(
    mechanism: object,
    *,
    parameter_algebra_spec: object | None = None,
    constraint_report: WegscheiderConstraintReport | None = None,
) -> ExactConstraintComposition:
    """Compose exact cycle products using one symbolic relation graph."""

    report = constraint_report or analyze_wegscheider_constraints(mechanism)
    names = _canonical_names(mechanism, parameter_algebra_spec, report)
    relations = _mechanism_relations(mechanism, names)
    algebra, unresolved = _algebra_relations(parameter_algebra_spec, names)
    relations.update(algebra)
    _rebind_equilibrium_relations(mechanism, relations)
    result: list[ExactConstraint] = []
    for constraint in report.constraints:
        initial = ExactMonomial(Fraction(1), tuple((name, Fraction(power)) for name, power in constraint.terms))
        composed = _substitute(initial, relations, unresolved)
        if composed is None:
            result.append(
                ExactConstraint(
                    constraint.cycle_id,
                    initial.powers,
                    constraint.expression,
                    "unresolved",
                    "constraint depends on an unsupported, cyclic, or unprovable relation",
                )
            )
            continue
        if composed.coefficient != 1:
            status = "unsatisfied"
            reason = "exact multiplicative product is not an identity"
        elif composed.powers or composed.exponentials or composed.sum_terms:
            status = "unsatisfied"
            reason = "exact symbolic product is not an identity"
        else:
            status = "resolved"
            reason = None
        result.append(
            ExactConstraint(
                constraint.cycle_id,
                tuple(sorted(composed.power_map().items())),
                _render_constraint(composed),
                status,
                reason,
                composed.coefficient,
            )
        )
    return ExactConstraintComposition(tuple(result))
