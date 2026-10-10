"""Pure exact composition of canonical symbolic mechanism constraints."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from fractions import Fraction

from kindred.core.algebra.parser import (
    BinaryNode,
    CallNode,
    IdentNode,
    NumberNode,
    SpeciesRefNode,
    UnaryNode,
    _iter_identifiers,
)
from kindred.core.quantity_units import Unit
from kindred.core._bounded_rational import ExactWorkLimit, literal_fraction, rational_binary
from kindred.core.equilibrium_rate_authority import equilibrium_rate_symbolic_relations
from kindred.core.kinetics import forward_rate_symbolic_relation
from kindred.core.simulator.dsl_build_input import source_coordinate_name_parts
from kindred.core.simulator.step_indexing import (
    get_step_index_map,
    lookup_source_coordinate_target,
)
from kindred.core.simulator.wegscheider_symbolic import analyze_wegscheider_topology

if TYPE_CHECKING:
    from kindred.core.simulator.parameter_algebra_spec import ParameterAlgebraSpec

__all__ = [
    "ExactMonomial",
    "ExactConstraint",
    "ExactConstraintComposition",
    "compose_exact_constraints",
    "cycle_message",
]

# Roles whose admitted values are always positive, so a root of their powers
# never hides a sign.  Energies and free Algebra parameters may be negative.
_POSITIVE_ROLES = frozenset({"k", "kf", "kr", "Keq", "A", "kappa", "degeneracy"})


def _known_positive(name: str) -> bool:
    if name in {"T", "C0"} or name in _EXACT_CONSTANTS:
        return True
    parts = source_coordinate_name_parts(name)
    return parts is not None and parts[0] in _POSITIVE_ROLES


_EXACT_CONSTANTS = frozenset({"k_B", "h", "h_bar", "N_A", "R"})


def _constant(name: str) -> ExactMonomial:
    """Lower a physical constant as a positive symbol.

    ``R`` is exactly ``k_B * N_A`` (both defined exactly in SI), so ``k_B``
    is lowered as ``R / N_A`` and closures may use either spelling.
    """

    if name == "k_B":
        return ExactMonomial(Fraction(1), (("N_A", Fraction(-1)), ("R", Fraction(1))))
    return ExactMonomial(Fraction(1), ((name, Fraction(1)),))


@dataclass(frozen=True, slots=True)
class ExactMonomial:
    coefficient: Fraction = Fraction(1)
    powers: tuple[tuple[str, Fraction], ...] = ()
    exponentials: tuple["ExactMonomial", ...] = ()
    # A compact immutable additive expression is needed only as an
    # exponential argument (for example ``exp(-Ea/(R*T) + dG)``).  Products,
    # and quotients by a single term, are multiplied out over sums; a sum as
    # a multiplicative factor is never expanded (its name stays a free
    # symbol), so this does not become a second symbolic engine.
    sum_terms: tuple["ExactMonomial", ...] = ()
    # One ``ln(name)`` factor of a sum term.  It enters only through ``ln`` and
    # is consumed by an exponential: ``exp(c*ln(x))`` is the power ``x**c``,
    # and any other term reading it stays its own exponential factor.
    logarithm: str | None = None

    def __post_init__(self) -> None:
        coefficient = Fraction(self.coefficient)
        powers: dict[str, Fraction] = {}
        for name, power in self.powers:
            powers[str(name)] = Fraction(power)
        logarithm = self.logarithm
        sum_terms = tuple(self.sum_terms)
        if sum_terms:
            flat_terms: list[ExactMonomial] = []
            for term in sum_terms:
                flat_terms.extend(term.sum_terms or (term,))
            grouped_terms: dict[
                tuple[tuple[tuple[str, Fraction], ...], tuple["ExactMonomial", ...], str | None],
                Fraction,
            ] = {}
            for term in flat_terms:
                key = (term.powers, term.exponentials, term.logarithm)
                grouped_terms[key] = grouped_terms.get(key, Fraction(0)) + term.coefficient
            normalized_terms = tuple(
                ExactMonomial(coefficient, powers, nested, logarithm=log)
                for (powers, nested, log), coefficient in grouped_terms.items()
                if coefficient != 0
            )
            coefficient = Fraction(1)
            powers = {}
            logarithm = None
            normalized_exponentials: tuple["ExactMonomial", ...] = ()
            sum_terms = normalized_terms
            if not sum_terms:
                coefficient = Fraction(0)
        elif coefficient == 0:
            powers = {}
            normalized_exponentials: tuple["ExactMonomial", ...] = ()
            logarithm = None
        else:
            normalized_exponentials = ()
            for argument in self.exponentials:
                terms = argument.sum_terms or (argument,)
                rest = []
                for term in terms:
                    if term.logarithm is not None and not term.powers and not term.exponentials:
                        name = term.logarithm
                        powers[name] = powers.get(name, Fraction(0)) + term.coefficient
                    else:
                        rest.append(term)
                if len(rest) == len(terms):
                    normalized_exponentials += (argument,)
                elif rest:
                    normalized_exponentials += (
                        rest[0] if len(rest) == 1 else ExactMonomial(sum_terms=tuple(rest)),
                    )
        object.__setattr__(self, "coefficient", coefficient)
        object.__setattr__(
            self, "powers", tuple(sorted((name, power) for name, power in powers.items() if power))
        )
        object.__setattr__(self, "exponentials", normalized_exponentials)
        object.__setattr__(self, "sum_terms", sum_terms)
        object.__setattr__(self, "logarithm", logarithm)

    def names(self) -> set[str]:
        """Every name this expression reads, inside exponentials and sums too."""
        found = {name for name, _power in self.powers}
        if self.logarithm is not None:
            found.add(self.logarithm)
        for child in (*self.exponentials, *self.sum_terms):
            found |= child.names()
        return found

    @property
    def base_powers(self) -> tuple[tuple[str, Fraction], ...]:
        return self.powers

    def scale(self, factor: Fraction) -> "ExactMonomial":
        """Scale a scalar expression by an exact factor.

        Scalar multiplication changes the coefficient, preserving variable
        powers and exponential factors: ``q * (c*exp(E)) = (q*c)*exp(E)``.
        Raising an expression to a power scales its exponential arguments
        separately in ``power``.
        """

        factor = Fraction(factor)
        if self.sum_terms:
            return ExactMonomial(
                sum_terms=tuple(term.scale(factor) for term in self.sum_terms)
            )
        return ExactMonomial(
            self.coefficient * factor,
            self.powers,
            self.exponentials,
            logarithm=self.logarithm,
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
        if self.sum_terms or other.sum_terms or (self.logarithm and other.logarithm):
            return None  # type: ignore[return-value]
        powers = dict(self.powers)
        for name, power in other.powers:
            powers[name] = powers.get(name, Fraction(0)) + power
        return ExactMonomial(
            self.coefficient * other.coefficient,
            tuple(powers.items()),
            self._aggregate_exponentials(self.exponentials + other.exponentials),
            logarithm=self.logarithm or other.logarithm,
        )

    def divide(self, other: "ExactMonomial") -> "ExactMonomial | None":
        if self.sum_terms or other.sum_terms or other.logarithm:
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
            logarithm=self.logarithm,
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
        return ExactMonomial(
            -self.coefficient, self.powers, self.exponentials, logarithm=self.logarithm
        )

    def __pow__(self, exponent: "ExactMonomial") -> "ExactMonomial | None":
        if exponent.powers or exponent.exponentials or exponent.sum_terms:
            return None
        return self.power(exponent.coefficient)

    def power(self, exponent: Fraction) -> "ExactMonomial" | None:
        exponent = Fraction(exponent)
        if self.sum_terms or self.logarithm:
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
        if denominator != 1 and not self._factors_positive():
            return None
        root = _exact_rational_root(self.coefficient, denominator)
        if root is None:
            return None
        try:
            coefficient = rational_binary("^", root, Fraction(numerator))
        except (OverflowError, ZeroDivisionError, ValueError):
            return None
        return ExactMonomial(
            coefficient,
            tuple((name, power * exponent) for name, power in self.powers),
            tuple(argument.scale(exponent) for argument in self.exponentials),
        )

    def _factors_positive(self) -> bool:
        """Whether a positive value has only positive factors.

        (x**p)**q == x**(p*q) and ln(x*y) == ln(x) + ln(y) only when no
        factor can be negative: an even power, or a product of signed values,
        would hide the sign that the numerical root or logarithm removes.
        """

        signed = [
            power
            for name, power in self.powers
            if power.denominator == 1 and not _known_positive(name)
        ]
        return not (
            self.coefficient < 0
            or len(signed) > 1
            or any(power.numerator % 2 == 0 for power in signed)
        )

    def ln(self) -> "ExactMonomial | None":
        """``ln`` of a product: a sum of ``power*ln(name)`` and its exponents.

        ``None`` for a sum, a logarithm, or a number other than 1, whose
        logarithm has no exact form here.
        """

        if self.sum_terms or self.logarithm or self.coefficient != 1 or not self._factors_positive():
            return None
        terms = [ExactMonomial(power, logarithm=name) for name, power in self.powers]
        for argument in self.exponentials:
            terms.extend(argument.sum_terms or (argument,))
        return ExactMonomial(sum_terms=tuple(terms)) if terms else ExactMonomial(Fraction(0))


def _exact_integer_root(value: int, degree: int) -> int | None:
    if value < 0 or degree <= 0:
        return None
    if value in {0, 1} or degree == 1:
        return value
    if degree > value.bit_length():
        return None
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


def _power(base: ExactMonomial | None, exponent: ExactMonomial | None) -> ExactMonomial | None:
    # A power is exact only when its full exponent is a scalar literal;
    # silently dropping a symbolic or additive exponent would change the
    # admitted relation.
    if base is None or exponent is None or exponent.powers or exponent.exponentials or exponent.sum_terms:
        return None
    return base.power(exponent.coefficient)


def _exact_monomial_from_ast(expr, *, canonical_names):
    try:
        value = _unscaled_monomial_from_ast(expr, canonical_names=canonical_names)
        if value is None:
            return None
        scale = expr.unit_scale.rational()
        if scale is None:
            return None
        rational_binary("*", value.coefficient, scale)
        return value.scale(scale)
    except ExactWorkLimit:
        return None


def _unscaled_monomial_from_ast(
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
        try:
            return ExactMonomial(literal_fraction(expr.literal if expr.literal is not None else str(expr.value)))
        except ValueError:
            return None
    if isinstance(expr, IdentNode):
        name = str(expr.name)
        if name in _EXACT_CONSTANTS:
            return _constant(name)
        if name == "T" or name in canonical_names:
            return ExactMonomial(Fraction(1), ((name, Fraction(1)),))
        return None
    if isinstance(expr, CallNode):
        if expr.name not in {"exp", "sqrt", "ln"} or len(expr.args) != 1:
            return None
        argument = _exact_monomial_from_ast(expr.args[0], canonical_names=canonical_names)
        if argument is None:
            return None
        if expr.name == "sqrt":
            return argument.power(Fraction(1, 2))
        if expr.name == "ln":
            return argument.ln()
        return ExactMonomial(
            Fraction(1), (), ExactMonomial._aggregate_exponentials((argument,))
        )
    if isinstance(expr, SpeciesRefNode):
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
    if expr.op in {"+", "-", "*", "/"} and (expr.op != "/" or right.coefficient):
        rational_binary(expr.op, left.coefficient, right.coefficient)
    if expr.op == "*":
        if left.sum_terms or right.sum_terms:
            products = [
                left_term.multiply(right_term)
                for left_term in left.sum_terms or (left,)
                for right_term in right.sum_terms or (right,)
            ]
            if any(product is None for product in products):
                return None
            return ExactMonomial(sum_terms=tuple(products))
        return left.multiply(right)
    if expr.op == "/":
        if left.sum_terms and not right.sum_terms:
            quotients = [term.divide(right) for term in left.sum_terms]
            if any(quotient is None for quotient in quotients):
                return None
            return ExactMonomial(sum_terms=tuple(quotients))
        return left.divide(right)
    if expr.op in {"**", "^"}:
        return _power(left, right)
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
    # The authored reaction lines of the cycle's steps.
    lines: tuple[str, ...]
    status: str = "resolved"
    # The user sentence naming the cycle by its lines, when it is not an identity.
    reason: str | None = None
    step_coefficients: tuple[tuple[int, int], ...] = ()
    # Defined names kept as free symbols that survive in an unresolved product.
    kept: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ExactStep:
    """One cycle step's composed ``KeqN`` readout."""

    step_index: int
    keq: ExactMonomial | None


@dataclass(frozen=True, slots=True)
class ExactConstraintComposition:
    constraints: tuple[ExactConstraint, ...]
    coordinate_units: tuple[tuple[str, Unit | None], ...] = ()
    steps: tuple[ExactStep, ...] = ()
    # Every name each defined name reads through the admitted evaluation
    # graph, transitively.
    reads: Mapping[str, frozenset[str]] = field(default_factory=dict)
    relations: Mapping[str, ExactMonomial] = field(default_factory=dict)
    unresolved: frozenset[str] = frozenset()
    # The steps each coordinate or helper line stands for (``_step_names``).
    step_names: Mapping[str, frozenset[int]] = field(default_factory=dict)

    def compose(
        self,
        name: str,
        through: Collection[str] | None = None,
    ) -> ExactMonomial | None:
        """Lower ``name`` through the relations of ``through`` (all when omitted).

        Every other name stays its own symbol, so a value can be written in
        terms of the names that do not depend on ``through``.
        """

        relations = (
            self.relations
            if through is None
            else {key: value for key, value in self.relations.items() if key in through}
        )
        return _substitute(
            ExactMonomial(Fraction(1), ((name, Fraction(1)),)), relations, self.unresolved
        )

    def closes(
        self,
        cycles: Collection[ExactConstraint],
        lines: Mapping[str, object],
    ) -> bool:
        """Whether new unit-prepared ``param`` lines make every cycle an identity.

        Each line is lowered and composed exactly as the cycle check reads an
        admitted closure, so correction offers only what admission accepts.
        """

        relations = dict(self.relations)
        for name, expr in lines.items():
            lowered = _exact_monomial_from_ast(
                expr, canonical_names=set(_iter_identifiers(expr))
            )
            if lowered is None:
                return False
            relations[name] = lowered
        return all(
            _substitute(_cycle_product(cycle.step_coefficients), relations, self.unresolved)
            == ExactMonomial()
            for cycle in cycles
        )


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
    equilibria = tuple(mechanism.equilibria)
    reactions = tuple(mechanism.reactions)
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
        kind = str(entry.get("kind") or "")
        try:
            index = int(entry.get("reaction_index" if kind == "reaction" else "equilibrium_index"))
            step = int(entry.get("step_index"))
        except (TypeError, ValueError):
            continue
        if not 0 <= index < len(reactions if kind == "reaction" else equilibria):
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
            forward_model = reactions[index].forward_model if kind == "reaction" else equilibria[index].forward_model
            projected = forward_rate_symbolic_relation(
                forward_model,
                temperature=temperature,
                literal=literal,
                exponential=exponential,
                source_coordinate=source_coordinate,
                standard_concentration=standard_concentration,
                constant=_constant,
            )
            if kind == "reaction":
                name = f"k{step}"
                if projected != ExactMonomial(Fraction(1), ((name, Fraction(1)),)):
                    relations[name] = projected
                continue
            # Reverse and equilibrium relations name ``kfN`` so a step's
            # readout can be written through its own coordinates.
            equilibrium_relations = equilibrium_rate_symbolic_relations(
                equilibria[index],
                forward_relation=ExactMonomial(Fraction(1), ((kf, Fraction(1)),)),
                temperature=temperature,
                literal=literal,
                exponential=exponential,
                source_coordinate=source_coordinate,
                standard_concentration=standard_concentration,
                constant=_constant,
            )
        except (TypeError, ValueError, ZeroDivisionError, OverflowError):
            continue
        for name, relation in (
            (kf, projected),
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


class _CyclicRelation(Exception):
    """A relation reaches itself; it is never kept as a free symbol."""


def _substitute(
    monomial: ExactMonomial,
    relations: Mapping[str, ExactMonomial],
    unresolved: set[str] | frozenset[str] = frozenset(),
    kept: set[str] | None = None,
) -> ExactMonomial | None:
    """Recursively lower retained exact expressions in evaluation order.

    Names are resolved through the admitted relation graph.  Exponential
    arguments are traversed as expression trees (rather than rendered and
    reparsed), so ``exp(-Ea/(R*T))`` retains its exact coefficient, fraction,
    symbolic temperature, and constants.  A sum is valid inside an
    exponential argument but not as a top-level multiplicative factor.
    A defined name whose value cannot be represented where it is used
    (unlowered, a multiplicative or powered sum, an inexact root) stays its
    own free symbol and is recorded in ``kept``: treating one value as
    independent can lose an identity but never create one.  A cyclic
    relation fails.
    """

    cache: dict[tuple[str, bool], ExactMonomial | None] = {}
    visiting: set[str] = set()
    kept = set() if kept is None else kept

    def resolve(name: str, *, allow_sum: bool) -> ExactMonomial | None:
        cache_key = (name, bool(allow_sum))
        if cache_key in cache:
            return cache[cache_key]
        if name in visiting:
            raise _CyclicRelation(name)
        if name in unresolved:
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
            # Terms are always traversed so a cycle inside a sum fails even
            # where the sum itself is kept as a named symbol.
            terms = tuple(
                substitute_value(term, allow_sum=True) for term in value.sum_terms
            )
            if not allow_sum or any(term is None for term in terms):
                return None
            return ExactMonomial(sum_terms=terms)

        arguments = tuple(
            substitute_value(argument, allow_sum=True)
            for argument in value.exponentials
        )
        if any(argument is None for argument in arguments):
            return None
        result = ExactMonomial(value.coefficient, (), arguments)
        factors = []
        for name, power in value.base_powers:
            resolved = resolve(name, allow_sum=allow_sum)
            powered = resolved.power(power) if resolved is not None else None
            if powered is None or (powered.sum_terms and not allow_sum):
                kept.add(name)
                powered = ExactMonomial(Fraction(1), ((name, power),))
            factors.append(powered)
        if value.logarithm is not None:
            # ln(x) is written through x's value; a value without an exact
            # logarithm keeps ln(x) as its own factor.
            resolved = resolve(value.logarithm, allow_sum=False)
            logarithm = resolved.ln() if resolved is not None else None
            if logarithm is None:
                kept.add(value.logarithm)
                logarithm = ExactMonomial(logarithm=value.logarithm)
            factors.append(logarithm)
        for powered in factors:
            if result.sum_terms or powered.sum_terms:
                result = _multiply_with_sums(result, powered)
            else:
                result = result.multiply(powered)
            if result is None or (result.sum_terms and not allow_sum):
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

    try:
        return substitute_value(monomial, allow_sum=False)
    except _CyclicRelation:
        return None


def _step_names(
    statements: Mapping[str, object],
    coordinates: set[str] | frozenset[str],
) -> dict[str, frozenset[int]]:
    """The steps each coordinate or helper ``param`` line stands for.

    A coordinate stands for its own step.  A helper line stands for every step
    whose coordinate it names, directly or through other helper lines; the
    walk stops at coordinates, so a coordinate's own line never adds steps.
    """

    found = {
        name: frozenset({source_coordinate_name_parts(name)[1]}) for name in coordinates
    }

    def steps(name: str) -> frozenset[int]:
        if name not in found:
            found[name] = frozenset()
            if name in statements:
                found[name] = frozenset().union(
                    *map(steps, _iter_identifiers(statements[name].expr))
                )
        return found[name]

    for name in statements:
        steps(name)
    return {name: value for name, value in found.items() if value}


def names_other_step(
    step_names: Mapping[str, frozenset[int]],
    names: Collection[str],
    step: int,
) -> bool:
    """Whether a value of ``step`` written from ``names`` is a closure.

    Only a formula naming a coordinate of another step, directly or through
    helper lines, relates steps; numbers, ``T`` and parameters inside a
    formula naming no other step never decide a reversible cycle.
    """

    return any(step_names.get(name, frozenset()) - {step} for name in names)


def _free_statements(
    spec: ParameterAlgebraSpec,
    statements: Mapping[str, object],
    coordinates: set[str] | frozenset[str],
    step_names: Mapping[str, frozenset[int]],
) -> frozenset[str]:
    """Name every ``param`` line that authors a free symbol.

    An independent scalar is a free symbol.  A line setting a step coordinate
    is a free symbol unless it is a closure (:func:`names_other_step`).
    """

    free = set()
    for name, statement in statements.items():
        if name in coordinates:
            if not names_other_step(
                step_names,
                set(_iter_identifiers(statement.expr)),
                source_coordinate_name_parts(name)[1],
            ):
                free.add(name)
        elif spec.parameter_assignment_kind(name) == "independent scalar":
            free.add(name)
    return frozenset(free)


def _algebra_relations(
    spec: ParameterAlgebraSpec,
    identities: Collection[str],
) -> tuple[dict[str, ExactMonomial], set[str], dict[str, frozenset[int]], set[str]]:
    """Lower the retained ``param`` lines that relate other values."""

    statements = {
        str(statement.name): statement
        for statement in spec.parameter_block.statements_in_evaluation_order()
    }
    coordinates = {str(name) for name in spec.mechanism_param_names}
    canonical_names = set(statements) | coordinates | set(_EXACT_CONSTANTS) | {"T"}
    step_names = _step_names(statements, coordinates)
    free = _free_statements(spec, statements, coordinates, step_names) | set(identities)

    relations: dict[str, ExactMonomial] = {}
    unresolved: set[str] = set()
    for name, statement in statements.items():
        if name in free:
            # A free symbol: its formula and numbers are never consumed here.
            continue
        lowered = _exact_monomial_from_ast(
            statement.expr,
            canonical_names=canonical_names,
        )
        if lowered is None:
            unresolved.add(name)
        else:
            relations[name] = lowered
    return relations, unresolved, step_names, canonical_names


def proves_redundant_parameter(
    mechanism: object,
    spec: ParameterAlgebraSpec,
    target: str,
    identities: Collection[str],
) -> bool:
    """Whether ``target``'s line only restates its value through chemistry.

    Every other line is read exactly as the cycle check reads it
    (:func:`_algebra_relations`), so a free line stays its own symbol whatever
    its formula and its numbers never make an identity.
    """
    relations = _mechanism_relations(mechanism)
    algebra, unresolved, _step_names_, canonical_names = _algebra_relations(spec, identities)
    relations.update(algebra)
    relations.pop(target, None)
    statement = next(line for line in spec.parameter_block.lines if line.name == target)
    rhs = _exact_monomial_from_ast(statement.expr, canonical_names=canonical_names)
    return rhs is not None and _substitute(rhs, relations, unresolved) == ExactMonomial(
        Fraction(1), ((target, Fraction(1)),)
    )


def cycle_message(sentence: str, lines: Collection[str]) -> str:
    """One sentence about reversible cycles, followed by their authored lines."""

    return sentence + "\n" + "\n".join(lines)


def _cycle_product(step_coefficients: Collection[tuple[int, int]]) -> ExactMonomial:
    return ExactMonomial(
        Fraction(1),
        tuple(
            (f"Keq{step_index}", Fraction(coefficient))
            for step_index, coefficient in step_coefficients
            if coefficient
        ),
    )


def compose_exact_constraints(
    mechanism: object,
    *,
    parameter_algebra_spec: ParameterAlgebraSpec,
) -> ExactConstraintComposition:
    """Compose exact cycle products using one symbolic relation graph."""

    topology = analyze_wegscheider_topology(mechanism)
    relations = _mechanism_relations(mechanism)
    algebra, unresolved, step_names, _names = _algebra_relations(
        parameter_algebra_spec, parameter_algebra_spec.identity_checks
    )
    relations.update(algebra)
    line_of = {
        entry["step_index"]: entry["line_text"].strip()
        for entry in mechanism.metadata["step_index_map"]
    }
    # A step's own param lines belong to the cycle text after its reaction lines.
    param_lines = tuple(
        (source_coordinate_name_parts(line.name)[1], line.line_text.strip())
        for line in parameter_algebra_spec.parameter_block.lines
        if source_coordinate_name_parts(line.name) is not None
    )
    result: list[ExactConstraint] = []
    for cycle in topology.cycles:
        step_coefficients = tuple(zip(cycle.step_indices, cycle.coefficients, strict=True))
        lines = tuple(line_of[step] for step in cycle.step_indices) + tuple(
            text for step, text in param_lines if step in cycle.step_indices
        )
        initial = _cycle_product(step_coefficients)
        kept: set[str] = set()
        composed = _substitute(initial, relations, unresolved, kept)
        status, sentence = "unresolved", None
        surviving: tuple[str, ...] = ()
        if composed is None:
            composed = initial
            sentence = (
                "The values around this reversible cycle are calculated from "
                "each other in a loop, so Kindred cannot check the cycle."
            )
        elif surviving := tuple(sorted(kept & composed.names())):
            # A kept value is treated as independent, so the product cannot
            # prove the cycle inconsistent.
            sentence = (
                ", ".join(surviving)
                + (
                    " is used as its own value because its formula cannot"
                    if len(surviving) == 1
                    else " are used as their own values because their formulas cannot"
                )
                + " be combined into this reversible cycle."
            )
        elif composed.coefficient != 1 and not (
            composed.powers or composed.exponentials or composed.sum_terms
        ):
            status = "unsatisfied"
            sentence = (
                "The formulas around this reversible cycle make its equilibrium "
                f"constants multiply to {composed.coefficient}, not 1."
            )
        elif composed.coefficient != 1 or composed.powers or composed.exponentials or composed.sum_terms:
            status = "unsatisfied"
            sentence = "This reversible cycle is not closed."
        else:
            status = "resolved"
        result.append(
            ExactConstraint(
                cycle_id=cycle.cycle_id,
                monomial=composed,
                lines=lines,
                status=status,
                reason=None if sentence is None else cycle_message(sentence, lines),
                step_coefficients=step_coefficients,
                kept=surviving,
            )
        )
    cycle_steps = sorted({step for cycle in topology.cycles for step in cycle.step_indices})
    steps = tuple(
        ExactStep(
            step_index,
            _substitute(
                ExactMonomial(Fraction(1), ((f"Keq{step_index}", Fraction(1)),)),
                relations,
                unresolved,
            ),
        )
        for step_index in cycle_steps
    )
    # Admission empties a discharged identity's edges; its statement still
    # reads its operands.
    graph = {
        name: set(dependencies)
        for name, dependencies in parameter_algebra_spec.evaluation_dependencies.items()
    }
    statements = {statement.name: statement for statement in parameter_algebra_spec.parameter_block.lines}
    for name in parameter_algebra_spec.identity_checks:
        graph.setdefault(name, set()).update(_iter_identifiers(statements[name].expr))

    def transitive_reads(name: str) -> frozenset[str]:
        found: set[str] = set()
        pending = list(graph.get(name, ()))
        while pending:
            dependency = pending.pop()
            if dependency not in found:
                found.add(dependency)
                pending.extend(graph.get(dependency, ()))
        return frozenset(found)

    reads = {name: transitive_reads(name) for name in graph}
    return ExactConstraintComposition(
        tuple(result),
        tuple(parameter_algebra_spec.quantity_units),
        steps,
        reads,
        relations,
        frozenset(unresolved),
        step_names,
    )
