from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Mapping

import numpy as np

from kindred.core.mechanism import Equilibrium, Mechanism
from kindred.core.ode_builder import (
    EffectiveRateCoefficientEvaluator,
)
from kindred.core.kinetics import forward_rate_family
from kindred.core.simulator.parameter_namespace import (
    build_namespace_from_mechanism,
    canonical_name_for_mechanism_step_parameter,
)

from .backend import require_sympy
from .errors import UnsupportedSymbolicExpressionError
from .jacobian_execution import symbolic_status_payload


@dataclass(frozen=True, slots=True)
class SymbolicJacobianStructure:
    species_names: tuple[str, ...]
    parameter_symbols: tuple[str, ...]
    rhs_expressions: tuple[str, ...]
    jacobian_expressions: tuple[tuple[str, ...], ...]
    _compiled: Callable[..., Any]
    jac_sparsity: tuple[tuple[bool, ...], ...]

    def bind(
        self,
        coefficient_evaluator: EffectiveRateCoefficientEvaluator,
    ) -> Callable[..., np.ndarray]:
        if not isinstance(
            coefficient_evaluator,
            EffectiveRateCoefficientEvaluator,
        ):
            raise TypeError(
                "coefficient_evaluator must be "
                "EffectiveRateCoefficientEvaluator."
            )
        coefficient_parameter_symbols = (
            _coefficient_parameter_symbols_from_step_kinds(
                coefficient_evaluator.step_kinds
            )
        )
        def jacobian_func(t: float, y: np.ndarray, *, T: float) -> np.ndarray:
            values = np.asarray(y, dtype=float).reshape(-1)
            if values.size != len(self.species_names):
                raise ValueError(f"symbolic jacobian expected {len(self.species_names)} state values, got {values.size}")
            coefficients = coefficient_evaluator(T)
            coefficient_values = _coefficient_values_by_parameter_name(
                coefficient_evaluator,
                coefficients.forward,
                coefficients.reverse,
            )
            parameter_tuple = tuple(
                coefficient_values[name]
                for name in coefficient_parameter_symbols
            )
            return np.asarray(
                self._compiled(*values, *parameter_tuple),
                dtype=float,
            )

        return jacobian_func


@dataclass(frozen=True, slots=True)
class _SymbolicMechanismExpressionModel:
    species_names: tuple[str, ...]
    parameter_symbols: tuple[str, ...]
    rhs_expressions: tuple[Any, ...]
    jacobian_expressions: tuple[tuple[Any, ...], ...]
    state_symbols: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class SymbolicMechanismPublicProjection:
    """Canonical public view of the single private symbolic mechanism graph."""

    species_names: tuple[str, ...]
    parameter_symbols: tuple[str, ...]
    rhs_expressions: tuple[Any, ...]
    jacobian_expressions: tuple[tuple[Any, ...], ...]
    state_symbols: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class SymbolicJacobianSupport:
    supported: bool
    code: str
    reason: str

    def raise_if_unsupported(self) -> None:
        if not self.supported:
            raise UnsupportedSymbolicExpressionError(self.reason)

    def to_status_payload(self) -> dict[str, str]:
        state = "supported" if self.supported else "unsupported"
        reason = self.reason if self.reason else "Symbolic Jacobian supported."
        return symbolic_status_payload(kind="jacobian", state=state, code=self.code, reason=reason)


def _coefficient_parameter_name(
    step_index: int,
    direction: str,
) -> str:
    role = "kf" if direction == "forward" else "kr"
    return f"__kindred_effective_{role}_{int(step_index) + 1}"


def _symbolic_coefficient_parameter_symbols(
    mechanism: Mechanism,
) -> tuple[str, ...]:
    names: list[str] = []
    step_index = 0
    for _reaction in mechanism.reactions:
        names.append(
            _coefficient_parameter_name(step_index, "forward")
        )
        step_index += 1
    for _equilibrium in mechanism.equilibria:
        names.extend(
            (
                _coefficient_parameter_name(step_index, "forward"),
                _coefficient_parameter_name(step_index, "reverse"),
            )
        )
        step_index += 1
    return tuple(sorted(names))


def _coefficient_parameter_symbols_from_step_kinds(
    step_kinds: tuple[str, ...],
) -> tuple[str, ...]:
    names: list[str] = []
    for step_index, kind in enumerate(step_kinds):
        names.append(_coefficient_parameter_name(step_index, "forward"))
        if str(kind) == "equilibrium":
            names.append(_coefficient_parameter_name(step_index, "reverse"))
    return tuple(sorted(names))


def _coefficient_values_by_parameter_name(
    evaluator: EffectiveRateCoefficientEvaluator,
    forward: tuple[float, ...],
    reverse: tuple[float, ...],
) -> dict[str, float]:
    if not (
        len(evaluator.step_kinds)
        == len(forward)
        == len(reverse)
    ):
        raise UnsupportedSymbolicExpressionError(
            "Effective rate coefficient shape conflicts with symbolic topology."
        )
    values: dict[str, float] = {}
    for step_index, kind in enumerate(evaluator.step_kinds):
        values[
            _coefficient_parameter_name(step_index, "forward")
        ] = float(forward[step_index])
        if kind == "equilibrium":
            values[
                _coefficient_parameter_name(step_index, "reverse")
            ] = float(reverse[step_index])
    return values


@dataclass(frozen=True, slots=True)
class _EquilibriumParameterNames:
    kf_name: str
    kr_name: str
    keq_name: str


def _canonical_step_parameter_name(
    mechanism: Mechanism,
    *,
    kind: str,
    item_index: int,
    role: str,
    expected_name: str,
) -> str:
    canonical_name = canonical_name_for_mechanism_step_parameter(
        mechanism,
        kind=kind,
        item_index=int(item_index),
        role=role,
        expected_name=str(expected_name),
    )
    return canonical_name


def _finite_scalar(value: object, *, label: str) -> float:
    if callable(value):
        raise UnsupportedSymbolicExpressionError(f"Dynamic callable {label} is not supported for symbolic Jacobian.")
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise UnsupportedSymbolicExpressionError(f"{label} must be a finite scalar.") from exc
    if not math.isfinite(out):
        raise UnsupportedSymbolicExpressionError(f"{label} must be finite.")
    return out


def _stoichiometric_coefficient_expr(sympy: Any, coeff: float, *, exact: bool) -> Any:
    if bool(exact):
        return sympy.Rational(str(float(coeff)))
    return sympy.Float(float(coeff))


def _power_product(sympy: Any, symbols: Mapping[str, Any], powers: Mapping[str, object]) -> Any:
    expr = sympy.Integer(1)
    for species_name, raw_power in sorted((powers or {}).items()):
        power = _finite_scalar(raw_power, label=f"stoichiometric power for {species_name}")
        if species_name not in symbols:
            raise UnsupportedSymbolicExpressionError(f"Unknown species {species_name!r} in symbolic Jacobian.")
        expr *= symbols[str(species_name)] ** sympy.Rational(str(power))
    return expr


def _equilibrium_parameter_names(
    mechanism: Mechanism,
    eq: Equilibrium,
    *,
    equilibrium_index: int,
) -> _EquilibriumParameterNames:
    kf_name = _canonical_step_parameter_name(
        mechanism,
        kind="equilibrium",
        item_index=int(equilibrium_index),
        role="kf",
        expected_name=f"kf{int(equilibrium_index) + 1}",
    )
    kr_name = _canonical_step_parameter_name(
        mechanism,
        kind="equilibrium",
        item_index=int(equilibrium_index),
        role="kr",
        expected_name=f"kr{int(equilibrium_index) + 1}",
    )
    keq_name = _canonical_step_parameter_name(
        mechanism,
        kind="equilibrium",
        item_index=int(equilibrium_index),
        role="Keq",
        expected_name=f"Keq{int(equilibrium_index) + 1}",
    )

    return _EquilibriumParameterNames(
        kf_name=kf_name,
        kr_name=kr_name,
        keq_name=keq_name,
    )


def _unsupported_support(code: str, reason: str) -> SymbolicJacobianSupport:
    return SymbolicJacobianSupport(
        supported=False,
        code=str(code),
        reason=str(reason),
    )


def classify_symbolic_jacobian_support(mechanism: Mechanism) -> SymbolicJacobianSupport:
    species_names = tuple(str(name) for name in mechanism.species_names())
    if not species_names:
        return _unsupported_support(
            "missing-species",
            "Symbolic Jacobian requires at least one species.",
        )

    concentration_exponents: list[tuple[str, str, float]] = []
    for idx, reaction in enumerate(
        mechanism.reactions,
        start=1,
    ):
        concentration_exponents.extend(
            (
                f"reaction {idx}",
                str(species_name),
                float(exponent),
            )
            for species_name, exponent in (
                reaction.rate_orders
            ).items()
        )
    for idx, equilibrium in enumerate(
        mechanism.equilibria,
        start=1,
    ):
        for direction, side in (
            ("forward", equilibrium.stoich_forward),
            ("reverse", equilibrium.stoich_back),
        ):
            concentration_exponents.extend(
                (
                    f"equilibrium {idx} {direction}",
                    str(species_name),
                    float(exponent),
                )
                for species_name, exponent in side.items()
            )
    for location, species_name, exponent in concentration_exponents:
        if not exponent.is_integer():
            return _unsupported_support(
                "nonintegral-concentration-exponent",
                "Symbolic Jacobian is not used because "
                f"{location} concentration exponent for {species_name!r} is "
                f"non-integral ({exponent:g}).",
            )

    try:
        for reaction in mechanism.reactions:
            forward_rate_family(reaction.forward_model)
        for equilibrium in mechanism.equilibria:
            forward_rate_family(equilibrium.forward_model)
    except (TypeError, ValueError) as exc:
        return _unsupported_support("unsupported-rate-authority", str(exc))
    return SymbolicJacobianSupport(
        supported=True,
        code="supported",
        reason="",
    )


def _coefficient_public_parameter_names(
    mechanism: Mechanism,
) -> tuple[str, ...]:
    public_by_internal: dict[str, str] = {}
    reactions = tuple(mechanism.reactions)
    equilibria = tuple(mechanism.equilibria)
    try:
        for reaction_index, reaction in enumerate(reactions):
            internal_name = _coefficient_parameter_name(
                reaction_index,
                "forward",
            )
            public_by_internal[internal_name] = _canonical_step_parameter_name(
                mechanism,
                kind="reaction",
                item_index=reaction_index,
                role="k",
                expected_name=f"k{reaction_index + 1}",
            )
        reaction_count = len(reactions)
        for equilibrium_index, equilibrium in enumerate(equilibria):
            step_index = reaction_count + equilibrium_index
            source = _equilibrium_parameter_names(
                mechanism,
                equilibrium,
                equilibrium_index=equilibrium_index,
            )
            if source.kf_name is None or source.kr_name is None:
                raise UnsupportedSymbolicExpressionError(
                    "Equilibrium public coefficient names are incomplete."
                )
            public_by_internal[
                _coefficient_parameter_name(step_index, "forward")
            ] = source.kf_name
            public_by_internal[
                _coefficient_parameter_name(step_index, "reverse")
            ] = source.kr_name
    except ValueError as exc:
        raise UnsupportedSymbolicExpressionError(str(exc)) from exc
    internal_names = _symbolic_coefficient_parameter_symbols(mechanism)
    try:
        return tuple(public_by_internal[name] for name in internal_names)
    except KeyError as exc:
        raise UnsupportedSymbolicExpressionError(
            "Canonical symbolic coefficient projection is incomplete."
        ) from exc


def _canonical_public_parameter_symbols(
    mechanism: Mechanism,
) -> tuple[str, ...]:
    try:
        namespace = build_namespace_from_mechanism(mechanism)
    except ValueError as exc:
        raise UnsupportedSymbolicExpressionError(str(exc)) from exc
    return tuple(
        str(item.canonical_name)
        for item in namespace.ordered_items
    )


def _build_symbolic_mechanism_expression_model(
    mechanism: Mechanism,
    *,
    exact_stoichiometric_coefficients: bool = False,
) -> _SymbolicMechanismExpressionModel:
    support = classify_symbolic_jacobian_support(mechanism)
    support.raise_if_unsupported()
    sympy = require_sympy()
    species_names = tuple(str(name) for name in mechanism.species_names())
    if not species_names:
        raise UnsupportedSymbolicExpressionError("Symbolic Jacobian requires at least one species.")
    # Plain private state symbols keep lambdify from dummifying every
    # argument.  Their prefix is disjoint from the private coefficient
    # symbols, and species names never enter this private graph.
    state_symbols = tuple(
        sympy.Symbol(f"__kindred_state_{idx}")
        for idx, _name in enumerate(species_names)
    )
    symbol_by_species = dict(zip(species_names, state_symbols))
    rhs = [sympy.Integer(0) for _name in species_names]
    species_index = {name: idx for idx, name in enumerate(species_names)}

    for reaction_index, rxn in enumerate(mechanism.reactions):
        coefficient = sympy.Symbol(
            _coefficient_parameter_name(
                reaction_index,
                "forward",
            )
        )
        rate_expr = coefficient * _power_product(
            sympy,
            symbol_by_species,
            rxn.rate_orders,
        )
        for species_name, coeff in rxn.net_stoich.items():
            rhs[species_index[str(species_name)]] += _stoichiometric_coefficient_expr(
                sympy,
                float(coeff),
                exact=bool(exact_stoichiometric_coefficients),
            ) * rate_expr

    reaction_count = len(tuple(mechanism.reactions))
    for equilibrium_index, eq in enumerate(mechanism.equilibria):
        step_index = reaction_count + equilibrium_index
        forward_coefficient = sympy.Symbol(
            _coefficient_parameter_name(step_index, "forward")
        )
        reverse_coefficient = sympy.Symbol(
            _coefficient_parameter_name(step_index, "reverse")
        )
        forward_expr = forward_coefficient * _power_product(
            sympy,
            symbol_by_species,
            eq.stoich_forward,
        )
        reverse_expr = reverse_coefficient * _power_product(
            sympy,
            symbol_by_species,
            eq.stoich_back,
        )
        rate_expr = forward_expr - reverse_expr
        for species_name in species_names:
            coeff = float(eq.stoich_back.get(species_name, 0.0)) - float(
                eq.stoich_forward.get(species_name, 0.0)
            )
            if coeff:
                rhs[species_index[species_name]] += _stoichiometric_coefficient_expr(
                    sympy,
                    coeff,
                    exact=bool(exact_stoichiometric_coefficients),
                ) * rate_expr

    rhs_matrix = sympy.Matrix(rhs)
    jacobian_matrix = rhs_matrix.jacobian(sympy.Matrix(state_symbols))
    parameter_symbols = _symbolic_coefficient_parameter_symbols(mechanism)
    return _SymbolicMechanismExpressionModel(
        species_names=species_names,
        parameter_symbols=parameter_symbols,
        rhs_expressions=tuple(sympy.simplify(expr) for expr in rhs_matrix),
        jacobian_expressions=tuple(
            tuple(sympy.simplify(jacobian_matrix[i, j]) for j in range(len(species_names)))
            for i in range(len(species_names))
        ),
        state_symbols=state_symbols,
    )


def _project_symbolic_mechanism_expression_model(
    mechanism: Mechanism,
    *,
    model: _SymbolicMechanismExpressionModel,
) -> SymbolicMechanismPublicProjection:
    """Project the single private graph onto canonical public chemistry atoms."""

    sympy = require_sympy()
    public_state_symbols = tuple(
        sympy.Symbol(str(name))
        for name in model.species_names
    )
    if len(public_state_symbols) != len(model.state_symbols):
        raise UnsupportedSymbolicExpressionError(
            "Symbolic state projection is inconsistent."
        )

    internal_parameter_names = tuple(model.parameter_symbols)
    coefficient_public_names = _coefficient_public_parameter_names(mechanism)
    if len(coefficient_public_names) != len(internal_parameter_names):
        raise UnsupportedSymbolicExpressionError(
            "Symbolic coefficient projection is inconsistent."
        )
    public_parameter_symbols = _canonical_public_parameter_symbols(mechanism)
    public_symbols_by_name = {
        name: sympy.Symbol(name)
        for name in public_parameter_symbols
    }
    replacements = {
        internal: public
        for internal, public in zip(
            model.state_symbols,
            public_state_symbols,
            strict=True,
        )
    }
    replacements.update(
        {
            sympy.Symbol(internal_name): public_symbols_by_name[public_name]
            for internal_name, public_name in zip(
                internal_parameter_names,
                coefficient_public_names,
                strict=True,
            )
        }
    )

    rhs_expressions = tuple(
        sympy.simplify(expression.xreplace(replacements))
        for expression in model.rhs_expressions
    )
    jacobian_expressions = tuple(
        tuple(
            sympy.simplify(expression.xreplace(replacements))
            for expression in row
        )
        for row in model.jacobian_expressions
    )

    return SymbolicMechanismPublicProjection(
        species_names=tuple(model.species_names),
        parameter_symbols=public_parameter_symbols,
        rhs_expressions=rhs_expressions,
        jacobian_expressions=jacobian_expressions,
        state_symbols=public_state_symbols,
    )


def build_symbolic_mechanism_public_projection(
    mechanism: Mechanism,
    *,
    exact_stoichiometric_coefficients: bool = False,
) -> SymbolicMechanismPublicProjection:
    """Build the single graph and return only its canonical public projection."""

    model = _build_symbolic_mechanism_expression_model(
        mechanism,
        exact_stoichiometric_coefficients=bool(
            exact_stoichiometric_coefficients
        ),
    )
    return _project_symbolic_mechanism_expression_model(
        mechanism,
        model=model,
    )


def build_symbolic_jacobian_structure(mechanism: Mechanism) -> SymbolicJacobianStructure:
    sympy = require_sympy()
    model = _build_symbolic_mechanism_expression_model(
        mechanism,
        exact_stoichiometric_coefficients=False,
    )
    projection = _project_symbolic_mechanism_expression_model(
        mechanism,
        model=model,
    )
    species_names = model.species_names
    state_symbols = model.state_symbols
    jacobian_matrix = sympy.Matrix(model.jacobian_expressions)
    rhs_strings = tuple(
        str(expression)
        for expression in projection.rhs_expressions
    )
    jacobian_strings = tuple(
        tuple(
            str(expression)
            for expression in row
        )
        for row in projection.jacobian_expressions
    )
    jac_sparsity = tuple(
        tuple(bool(expression != 0) for expression in row)
        for row in jacobian_matrix.tolist()
    )
    coefficient_parameter_symbols = model.parameter_symbols
    parameter_symbols = projection.parameter_symbols
    parameter_sympy_symbols = tuple(
        sympy.Symbol(name)
        for name in coefficient_parameter_symbols
    )
    compiled = sympy.lambdify((*state_symbols, *parameter_sympy_symbols), jacobian_matrix, modules="numpy")
    return SymbolicJacobianStructure(
        species_names=species_names,
        parameter_symbols=parameter_symbols,
        rhs_expressions=rhs_strings,
        jacobian_expressions=jacobian_strings,
        _compiled=compiled,
        jac_sparsity=jac_sparsity,
    )
