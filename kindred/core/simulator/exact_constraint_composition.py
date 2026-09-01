"""Pure exact composition of canonical symbolic mechanism constraints."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

from kindred.core.algebra.parser import (
    BinaryNode,
    CallNode,
    IdentNode,
    NumberNode,
    SpeciesRefNode,
    UnaryNode,
)
from kindred.core.constants import N_A, R, Rkcal, h, hbar, k_B
from kindred.core.equilibrium_rate_authority import equilibrium_rate_symbolic_relations
from kindred.core.kinetics import (
    forward_rate_model_for_equilibrium,
    forward_rate_symbolic_relation,
)
from kindred.core.simulator.step_indexing import (
    get_step_index_map,
    lookup_source_coordinate_target,
)
from kindred.core.simulator.wegscheider_symbolic import analyze_wegscheider_topology

__all__ = [
    "ExactMonomial",
    "ExactConstraint",
    "ExactConstraintComposition",
    "compose_exact_constraints",
]

_EXACT_CONSTANTS = {
    "k_B": Fraction(str(k_B)),
    "h": Fraction(str(h)),
    "hbar": Fraction(str(hbar)),
    "ħ": Fraction(str(hbar)),
    "N_A": Fraction(str(N_A)),
    "R": Fraction(str(R)),
    "Rkcal": Fraction(str(Rkcal)),
}


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
            normalized_exponentials: tuple["ExactMonomial", ...] = ()
            sum_terms = normalized_terms
            if not sum_terms:
                coefficient = Fraction(0)
        elif coefficient == 0:
            powers = {}
            normalized_exponentials: tuple["ExactMonomial", ...] = ()
        else:
            normalized_exponentials = tuple(self.exponentials)
        object.__setattr__(self, "coefficient", coefficient)
        object.__setattr__(self, "powers", tuple(sorted(powers.items())))
        object.__setattr__(self, "exponentials", normalized_exponentials)
        object.__setattr__(self, "sum_terms", sum_terms)

    @property
    def base_powers(self) -> tuple[tuple[str, Fraction], ...]:
        return self.powers

    def coordinate_power_map(self) -> dict[str, Fraction]:
        """Return powers for every symbol retained in this expression.

        The top-level ``powers`` tuple is the multiplicative projection used
        by the direct rate-coordinate solver.  Exact energy coordinates live
        in exponential arguments, however, so consumers that need to find
        authored coordinates must walk the retained expression tree instead
        of inspecting rendered ``exp(...)`` text.
        """

        result: dict[str, Fraction] = {}

        def visit(value: ExactMonomial) -> None:
            for name, power in value.powers:
                result[name] = result.get(name, Fraction(0)) + Fraction(power)
            for argument in value.exponentials:
                visit(argument)
            for term in value.sum_terms:
                visit(term)

        visit(self)
        return {
            name: power
            for name, power in result.items()
            if power
        }

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
        if not exponentials:
            return ()

        # exp(a) * exp(b) is one exact exp(a + b).  Flattening the additive
        # arguments before grouping is what makes e.g.
        # exp(-x) * exp(-y) * exp(x + y) cancel structurally.  The argument
        # remains a compact ExactMonomial tree; no rendered expression is
        # reparsed and no general symbolic algebra is introduced.
        terms: list[ExactMonomial] = []
        for argument in exponentials:
            terms.extend(argument.sum_terms or (argument,))
        argument = (
            terms[0]
            if len(terms) == 1
            else ExactMonomial(sum_terms=tuple(terms))
        )
        if (
            argument.coefficient == 0
            and not argument.powers
            and not argument.exponentials
            and not argument.sum_terms
        ):
            return ()
        return (argument,)

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


def _exact_monomial_from_ast(
    expr: object,
    *,
    canonical_names: set[str] | frozenset[str],
) -> ExactMonomial | None:
    """Lower supported arithmetic from one retained, admitted Algebra AST.

    This is deliberately a value projection only.  Assignment dependencies,
    evaluation order, and cycle handling are owned by ``ParameterAlgebraSpec``
    and are not inferred from this AST walk.
    """

    if isinstance(expr, NumberNode):
        exact = expr.exact_value
        return None if exact is None else ExactMonomial(exact)
    if isinstance(expr, IdentNode):
        name = str(expr.name)
        if name in _EXACT_CONSTANTS:
            return ExactMonomial(_EXACT_CONSTANTS[name])
        if name == "T" or name in canonical_names:
            return ExactMonomial(Fraction(1), ((name, Fraction(1)),))
        return None
    if isinstance(expr, SpeciesRefNode) or isinstance(expr, CallNode):
        return None
    if isinstance(expr, UnaryNode):
        value = _exact_monomial_from_ast(
            expr.rhs,
            canonical_names=canonical_names,
        )
        if value is None:
            return None
        if expr.op == "+":
            return value
        if expr.op == "-":
            return -value
        return None
    if not isinstance(expr, BinaryNode):
        return None

    left = _exact_monomial_from_ast(expr.lhs, canonical_names=canonical_names)
    right = _exact_monomial_from_ast(expr.rhs, canonical_names=canonical_names)
    if left is None or right is None:
        return None
    if expr.op == "*":
        return left.multiply(right)
    if expr.op == "/":
        return left.divide(right)
    if expr.op in {"**", "^"}:
        # A power is exact only when its full exponent is a scalar literal;
        # silently dropping a symbolic or additive exponent would change the
        # admitted relation.
        if right.powers or right.exponentials or right.sum_terms:
            return None
        return left.power(right.coefficient)
    if expr.op in {"+", "-"}:
        if (
            left.powers == right.powers
            and left.exponentials == right.exponentials
            and not left.sum_terms
            and not right.sum_terms
        ):
            coefficient = left.coefficient + (
                right.coefficient if expr.op == "+" else -right.coefficient
            )
            return ExactMonomial(coefficient, left.powers, left.exponentials)
        left_terms = left.sum_terms or (left,)
        right_terms = right.sum_terms or (right,)
        if expr.op == "-":
            right_terms = tuple(-term for term in right_terms)
        return ExactMonomial(sum_terms=tuple(left_terms) + tuple(right_terms))
    return None


@dataclass(frozen=True, slots=True)
class ExactConstraint:
    cycle_id: str
    monomial: ExactMonomial
    expression: str
    status: str = "resolved"
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class ExactConstraintComposition:
    constraints: tuple[ExactConstraint, ...]


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
) -> dict[str, ExactMonomial]:
    """Build the exact relation graph from admitted source authorities.

    A typed forward model owns the forward relation.  The indexed ``kfN``
    value is therefore a source only for a Constant model; Arrhenius and
    Eyring models retain their physical authored coordinates.  Reverse
    authority is then lowered by the canonical equilibrium relation helper.
    Numeric rate bindings and the nominal calculated forward rate are never
    promoted into an independent symbolic source.
    """

    relations: dict[str, ExactMonomial] = {}
    equilibria = tuple(getattr(mechanism, "equilibria", ()) or ())
    temperature = ExactMonomial(Fraction(1), (("T", Fraction(1)),))
    standard_concentration = ExactMonomial(
        Fraction(1), (("C0", Fraction(1)),)
    )

    def literal(value: object) -> ExactMonomial:
        try:
            return ExactMonomial(Fraction(str(value)))
        except (TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
            raise ValueError(f"Unsupported exact literal {value!r}.") from exc

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
        source_coordinates = {
            str(role): str(canonical_name)
            for role, canonical_name in dict(
                entry.get("source_coordinates") or {}
            ).items()
        }

        def source_coordinate(role: str) -> ExactMonomial | None:
            """Resolve one admitted source role through the source owner."""

            name = source_coordinates.get(role)
            if name is None:
                if role in {"kappa", "degeneracy"}:
                    # Omitted Eyring coordinates retain their exact defaults.
                    return None
                raise ValueError(
                    f"missing canonical source coordinate for {role!r}"
                )

            # The source owner decides whether the named coordinate is an
            # admitted authored source.  This is a trusted in-process lookup;
            # exact composition does not duplicate source admission or span
            # validation here.
            # Every accepted source role, including Constant ``kfN`` and an
            # explicit reverse authority, is resolved through the producer's
            # source map.  A calculated indexed readout is not source-backed
            # merely because it has the same canonical spelling.
            target = lookup_source_coordinate_target(mechanism, name)
            if target is None:
                raise ValueError(
                    f"source coordinate {name!r} is not an admitted authored role"
                )
            return ExactMonomial(Fraction(1), ((name, Fraction(1)),))

        try:
            forward_model = forward_rate_model_for_equilibrium(
                equilibria[index],
                temperature_K=None,
                default_standard_conc_M=1.0,
            )
            projected = forward_rate_symbolic_relation(
                forward_model,
                temperature=temperature,
                literal=literal,
                exponential=exponential,
                source_coordinate=source_coordinate,
                standard_concentration=standard_concentration,
            )
            equilibrium_relations = equilibrium_rate_symbolic_relations(
                equilibria[index],
                forward_relation=projected,
                temperature=temperature,
                literal=literal,
                exponential=exponential,
                source_coordinate=source_coordinate,
                standard_concentration=standard_concentration,
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


def _substitute(
    monomial: ExactMonomial,
    relations: Mapping[str, ExactMonomial],
    unresolved: set[str] | frozenset[str] = frozenset(),
) -> ExactMonomial | None:
    """Recursively lower retained exact expressions in evaluation order.

    Names are resolved through the admitted relation graph.  Exponential
    arguments are traversed as expression trees (rather than rendered and
    reparsed), so ``exp(-Ea/(R*T))`` retains its exact coefficient, fraction,
    symbolic temperature, and constants.  A sum is valid inside an
    exponential argument but not as a top-level multiplicative factor.
    """

    cache: dict[tuple[str, bool], ExactMonomial | None] = {}
    visiting: set[str] = set()

    def resolve(name: str, *, allow_sum: bool) -> ExactMonomial | None:
        cache_key = (name, bool(allow_sum))
        if cache_key in cache:
            return cache[cache_key]
        if name in unresolved or name in visiting:
            cache[cache_key] = None
            return None
        relation = relations.get(name)
        if relation is None:
            value = ExactMonomial(Fraction(1), ((name, Fraction(1)),))
            cache[cache_key] = value
            return value
        visiting.add(name)
        result = substitute_value(relation, allow_sum=allow_sum)
        visiting.remove(name)
        cache[cache_key] = result
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
            resolved = resolve(name, allow_sum=allow_sum)
            powered = resolved.power(power) if resolved is not None else None
            if powered is None:
                return None
            if result.sum_terms or powered.sum_terms:
                if not allow_sum:
                    return None
                result = _multiply_with_sums(result, powered)
            else:
                result = result.multiply(powered)
            if result is None:
                return None
        return result

    def _multiply_with_sums(
        left: ExactMonomial,
        right: ExactMonomial,
    ) -> ExactMonomial | None:
        """Distribute only when a sum is explicitly inside an expression."""

        left_terms = left.sum_terms or (left,)
        right_terms = right.sum_terms or (right,)
        products: list[ExactMonomial] = []
        for left_term in left_terms:
            for right_term in right_terms:
                product = left_term.multiply(right_term)
                if product is None:
                    return None
                products.append(product)
        return ExactMonomial(sum_terms=tuple(products))

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


def _algebra_relations(
    spec: object | None,
) -> tuple[dict[str, ExactMonomial], set[str]]:
    """Lower retained parameter assignments in their admitted order."""

    if spec is None:
        return {}, set()
    block = getattr(spec, "parameter_block", None)
    dependencies = getattr(spec, "parameter_dependencies", None)
    assignment_kind = getattr(spec, "parameter_assignment_kind", None)
    if (
        block is None
        or not isinstance(dependencies, Mapping)
        or not callable(assignment_kind)
    ):
        return {}, set()
    statements_in_order = getattr(block, "statements_in_evaluation_order", None)
    if not callable(statements_in_order):
        return {}, set()

    # Build one direct name -> retained statement view.  The spec's dependency
    # mapping supplies the already-admitted topological order; no identifier
    # walk or local cycle analysis is performed here.
    statement_by_name = {
        str(statement.name): statement
        for statement in tuple(statements_in_order())
    }
    assignment_names = tuple(str(name) for name in dependencies)
    canonical_names = set(assignment_names)
    namespace = getattr(spec, "mechanism_namespace", None)
    flat_names = getattr(namespace, "flat_names", None)
    if callable(flat_names):
        canonical_names.update(str(name) for name in flat_names())
    canonical_names.update(_EXACT_CONSTANTS)
    canonical_names.add("T")

    relations: dict[str, ExactMonomial] = {}
    unresolved: set[str] = set()
    for name in assignment_names:
        statement = statement_by_name.get(name)
        if statement is None:
            unresolved.add(name)
            continue
        kind = str(assignment_kind(name))
        if kind == "independent scalar":
            # An independent scalar is an exact symbolic leaf.  Its numeric
            # execution seed is intentionally not consumed here.
            continue
        if kind not in {"derived scalar", "mechanism-bound"}:
            unresolved.add(name)
            continue
        lowered = _exact_monomial_from_ast(
            statement.expr,
            canonical_names=canonical_names,
        )
        if lowered is None:
            unresolved.add(name)
        else:
            relations[name] = lowered
    return relations, unresolved


def compose_exact_constraints(
    mechanism: object,
    *,
    parameter_algebra_spec: object | None = None,
) -> ExactConstraintComposition:
    """Compose exact cycle products using one symbolic relation graph."""

    topology = analyze_wegscheider_topology(mechanism)
    relations = _mechanism_relations(mechanism)
    algebra, unresolved = _algebra_relations(parameter_algebra_spec)
    relations.update(algebra)
    result: list[ExactConstraint] = []
    for cycle in topology.cycles:
        initial = ExactMonomial(
            Fraction(1),
            tuple(
                (f"Keq{step_index}", Fraction(coefficient))
                for step_index, coefficient in zip(
                    cycle.step_indices,
                    cycle.coefficients,
                    strict=True,
                )
                if int(coefficient) != 0
            ),
        )
        composed = _substitute(initial, relations, unresolved)
        if composed is None:
            result.append(
                ExactConstraint(
                    cycle_id=cycle.cycle_id,
                    monomial=initial,
                    expression=_render_constraint(initial),
                    status="unresolved",
                    reason=(
                        "constraint depends on an unsupported, cyclic, or "
                        "unprovable relation"
                    ),
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
                cycle_id=cycle.cycle_id,
                monomial=composed,
                expression=_render_constraint(composed),
                status=status,
                reason=reason,
            )
        )
    return ExactConstraintComposition(tuple(result))
