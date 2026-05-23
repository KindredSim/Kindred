from __future__ import annotations

from decimal import Decimal, InvalidOperation
import math
import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from kindred.core.algebra.parser import (
    BinaryNode,
    CallNode,
    IdentNode,
    NumberNode,
    SpeciesRefNode,
    UnaryNode,
    parse_algebra,
)
from kindred.core.algebra.grammar import BUILTIN_FUNCTIONS as ALGEBRA_BUILTIN_FUNCTIONS
from kindred.core.algebra.grammar import HELPER_FUNCTIONS as ALGEBRA_HELPER_FUNCTIONS
from kindred.core.algebra.printer import expression_to_source
from kindred.core.algebra.symbols import PROTECTED_NAMES as ALGEBRA_PROTECTED_NAMES
from kindred.core.mechanism_metadata import MechanismMetadataKeys, MechanismMetadataView
from kindred.core.simulator.parameter_algebra import parameter_algebra_spec_from_mechanism
from kindred.core.simulator.parameter_algebra_spec import (
    ParameterAlgebraSpec,
    classify_parameter_algebra_declaration,
    parse_parameter_algebra_spec_from_dsl_text,
)
from kindred.core.simulator.parameter_namespace import (
    MechanismParameterNamespace,
    build_namespace_from_mechanism,
)

from .backend import require_sympy
from .errors import UnsupportedSymbolicExpressionError
from .jacobian import _build_symbolic_mechanism_expression_model, classify_symbolic_jacobian_support
from .parameter_expression import translate_parameter_expression

_NUMERIC_LITERAL_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:\d[\d_]*\.\d[\d_]*|\d[\d_]*\.|\.\d[\d_]*|\d[\d_]*)(?:[eE][+\-]?\d[\d_]*)?(?![A-Za-z0-9_])"
)
_SUPPORTED_TRANSFORMS = frozenset({"simplify", "factor", "expand", "cancel"})
_RESERVED_ALGEBRA_IDENTIFIERS = frozenset(
    str(name)
    for name in (
        set(ALGEBRA_PROTECTED_NAMES)
        | set(ALGEBRA_BUILTIN_FUNCTIONS)
        | set(ALGEBRA_HELPER_FUNCTIONS)
        | {"T0"}
    )
)


class SymbolicCalculatorError(UnsupportedSymbolicExpressionError):
    """Raised when a calculator query is syntactically valid but unsupported."""


class SymbolicCalculatorUnavailable(SymbolicCalculatorError):
    """Raised when the committed mechanism cannot be represented truthfully."""


@dataclass(frozen=True, slots=True)
class SymbolicCalculatorResult:
    query: str
    result_text: str
    assumptions: tuple[str, ...]
    symbol_legend: Mapping[str, Sequence[str]]
    mechanism_source: str = ""
    parameter_definitions: Mapping[str, str] = field(default_factory=dict)
    let_definitions: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "assumptions", tuple(str(item) for item in self.assumptions))
        object.__setattr__(
            self,
            "symbol_legend",
            MappingProxyType(
                {
                    str(kind): tuple(str(name) for name in names)
                    for kind, names in dict(self.symbol_legend).items()
                }
            ),
        )
        object.__setattr__(
            self,
            "parameter_definitions",
            MappingProxyType({str(name): str(expr) for name, expr in dict(self.parameter_definitions).items()}),
        )
        object.__setattr__(
            self,
            "let_definitions",
            MappingProxyType({str(name): str(expr) for name, expr in dict(self.let_definitions).items()}),
        )

    def compact_copy_text(self) -> str:
        return str(self.result_text)

    def full_context_copy_text(self) -> str:
        parameters = ", ".join(str(name) for name in self.symbol_legend.get("parameters", ())) or "none"
        species = ", ".join(str(name) for name in self.symbol_legend.get("species", ())) or "none"
        assumptions = "\n".join(f"- {item}" for item in self.assumptions) or "- none"
        definitions_by_kind = [
            (f"param {name}", expr)
            for name, expr in sorted(dict(self.parameter_definitions).items())
        ]
        definitions_by_kind.extend(
            (f"let {name}", expr)
            for name, expr in sorted(dict(self.let_definitions).items())
        )
        definitions = "\n".join(f"- {name} := {expr}" for name, expr in definitions_by_kind) or "- none"
        mechanism_source = str(self.mechanism_source or "").strip() or "(not provided)"
        return "\n".join(
            [
                f"Query: {self.query}",
                f"Result: {self.result_text}",
                "Assumptions:",
                assumptions,
                "Canonical symbol legend:",
                f"Species: {species}",
                f"Parameters: {parameters}",
                "Definitions:",
                definitions,
                "Mechanism source:",
                mechanism_source,
            ]
        )


@dataclass(frozen=True, slots=True)
class _CalculatorContext:
    mechanism: Any
    species_names: tuple[str, ...]
    parameter_symbols: tuple[str, ...]
    rhs_by_species: Mapping[str, Any]
    jacobian_by_species: Mapping[tuple[str, str], Any]
    expression_symbols: Mapping[str, Any]
    parameter_namespace: MechanismParameterNamespace
    internal_to_display: Mapping[Any, Any]
    rendered_parameter_definitions: Mapping[str, str]
    let_sources: Mapping[str, str]
    mechanism_source: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "species_names", tuple(str(name) for name in self.species_names))
        object.__setattr__(self, "parameter_symbols", tuple(str(name) for name in self.parameter_symbols))
        object.__setattr__(
            self,
            "rhs_by_species",
            MappingProxyType({str(name): expr for name, expr in dict(self.rhs_by_species).items()}),
        )
        object.__setattr__(
            self,
            "jacobian_by_species",
            MappingProxyType(
                {
                    (str(row), str(col)): expr
                    for (row, col), expr in dict(self.jacobian_by_species).items()
                }
            ),
        )
        object.__setattr__(self, "expression_symbols", MappingProxyType(dict(self.expression_symbols)))
        object.__setattr__(self, "internal_to_display", MappingProxyType(dict(self.internal_to_display)))
        object.__setattr__(
            self,
            "rendered_parameter_definitions",
            MappingProxyType({str(name): str(expr) for name, expr in dict(self.rendered_parameter_definitions).items()}),
        )
        object.__setattr__(
            self,
            "let_sources",
            MappingProxyType({str(name): str(expr) for name, expr in dict(self.let_sources).items()}),
        )
        object.__setattr__(self, "mechanism_source", str(self.mechanism_source or ""))


def evaluate_symbolic_query(
    mechanism: Any,
    query: str,
    *,
    mechanism_source: str = "",
) -> SymbolicCalculatorResult:
    context = _build_context(mechanism, mechanism_source=mechanism_source)
    query_s = str(query or "").strip()
    if not query_s:
        raise SymbolicCalculatorError("Symbolic calculator query is empty.")
    expr_or_lines = _evaluate_query(context, query_s)
    if isinstance(expr_or_lines, tuple):
        result_text = "\n".join(str(line) for line in expr_or_lines)
    else:
        result_text = _render_expr(context, expr_or_lines)
    return SymbolicCalculatorResult(
        query=query_s,
        result_text=result_text,
        assumptions=(
            "Symbolic calculator uses the committed mechanism's elementary steps and supported Algebra definitions.",
            "Scheduled interventions, scheduled temperature, runtime overrides, and hybrid execution behavior are not included in symbolic calculations.",
        ),
        symbol_legend={
            "species": list(context.species_names),
            "parameters": list(context.parameter_symbols),
        },
        mechanism_source=str(mechanism_source or ""),
        parameter_definitions=context.rendered_parameter_definitions,
        let_definitions=_supported_let_sources(context),
    )


def symbolic_calculator_available(mechanism: Any) -> tuple[bool, str]:
    try:
        _build_context(mechanism, mechanism_source="")
        support = classify_symbolic_jacobian_support(mechanism)
    except SymbolicCalculatorUnavailable as exc:
        return False, str(exc)
    if not bool(support.supported):
        return False, str(support.reason or "Symbolic calculator unavailable.")
    return True, ""


def _check_available(mechanism: Any) -> None:
    raw_metadata = getattr(mechanism, "metadata", {}) or {}
    metadata = MechanismMetadataView.from_metadata(raw_metadata)
    if metadata.temperature_schedule is not None:
        raise SymbolicCalculatorUnavailable(
            "Symbolic calculator is unavailable for mechanisms with scheduled temperature."
        )
    schedule = metadata.intervention_schedule
    if schedule is not None and not bool(schedule.is_empty()):
        raise SymbolicCalculatorUnavailable(
            "Symbolic calculator is unavailable for mechanisms with scheduled interventions."
        )
    if isinstance(raw_metadata, Mapping) and raw_metadata.get(MechanismMetadataKeys.STATE_NETWORK):
        raise SymbolicCalculatorUnavailable(
            "Symbolic calculator is unavailable for mechanisms with state-network definitions."
        )


def _build_context(mechanism: Any, *, mechanism_source: str) -> _CalculatorContext:
    _check_available(mechanism)
    parameter_namespace = build_namespace_from_mechanism(mechanism)
    _check_symbol_name_collisions(mechanism, parameter_namespace=parameter_namespace)
    sympy = require_sympy()
    try:
        model = _build_symbolic_mechanism_expression_model(
            mechanism,
            exact_stoichiometric_coefficients=True,
        )
    except (UnsupportedSymbolicExpressionError, ValueError) as exc:
        raise SymbolicCalculatorUnavailable(str(exc)) from exc
    species_names = tuple(str(name) for name in model.species_names)
    display_symbols = {name: sympy.Symbol(name) for name in species_names}
    internal_to_display = {
        internal: display_symbols[name]
        for name, internal in zip(species_names, model.state_symbols)
    }
    expression_symbols: dict[str, Any] = {
        name: internal
        for name, internal in zip(species_names, model.state_symbols)
    }
    expression_symbols.update({name: sympy.Symbol(name) for name in model.parameter_symbols})
    rhs_by_species = {
        name: model.rhs_expressions[idx]
        for idx, name in enumerate(species_names)
    }
    jacobian_by_species = {
        (row_name, col_name): model.jacobian_expressions[row_idx][col_idx]
        for row_idx, row_name in enumerate(species_names)
        for col_idx, col_name in enumerate(species_names)
    }
    rendered_definitions = _parameter_definition_sources(mechanism)
    expression_symbols.update({name: sympy.Symbol(name) for name in rendered_definitions})
    let_sources = _let_sources(mechanism)
    expression_symbols.update({name: sympy.Symbol(name) for name in let_sources})
    return _CalculatorContext(
        mechanism=mechanism,
        species_names=species_names,
        parameter_symbols=tuple(str(name) for name in model.parameter_symbols),
        rhs_by_species=rhs_by_species,
        jacobian_by_species=jacobian_by_species,
        expression_symbols=expression_symbols,
        parameter_namespace=parameter_namespace,
        internal_to_display=internal_to_display,
        rendered_parameter_definitions=rendered_definitions,
        let_sources=let_sources,
        mechanism_source=str(mechanism_source or ""),
    )


def _check_symbol_name_collisions(
    mechanism: Any,
    *,
    parameter_namespace: MechanismParameterNamespace | None = None,
) -> None:
    species_names = tuple(str(name) for name in _mechanism_species_names(mechanism))
    namespace = parameter_namespace or build_namespace_from_mechanism(mechanism)
    parameter_names = set(namespace.canonical_names)
    parameter_names.update(_parameter_definition_sources(mechanism))
    let_names = set(_let_sources(mechanism))
    reserved_names = parameter_names | let_names
    collisions: list[str] = []
    for name in species_names:
        if name in reserved_names or name in _RESERVED_ALGEBRA_IDENTIFIERS:
            collisions.append(name)
            continue
        resolved = namespace.resolve(name)
        if resolved.canonical_name is not None:
            collisions.append(name)
            continue
        if namespace.invalid_protected_indexed_identifier(name) is not None:
            collisions.append(name)
            continue
    collisions = sorted(set(collisions))
    if collisions:
        raise SymbolicCalculatorUnavailable(
            "Symbolic calculator is unavailable because species names collide with symbolic "
            f"parameter or Algebra names: {', '.join(collisions)}."
        )


def _mechanism_species_names(mechanism: Any) -> tuple[str, ...]:
    species_names = getattr(mechanism, "species_names", None)
    if not callable(species_names):
        raise SymbolicCalculatorUnavailable(
            "Symbolic calculator requires a Kindred mechanism with species_names()."
        )
    return tuple(str(name) for name in species_names())


def _evaluate_query(context: _CalculatorContext, query: str) -> Any:
    return _parse_calculator_expression(context, query)


def _evaluate_zero_arg_query(context: _CalculatorContext, name: str) -> tuple[str, ...]:
    if name == "odes":
        return tuple(
            f"d{name}/dt = {_render_expr(context, context.rhs_by_species[name])}"
            for name in context.species_names
        )
    if name == "jacobian":
        rows: list[str] = []
        for row_name in context.species_names:
            rendered = [
                _render_expr(context, context.jacobian_by_species[(row_name, col_name)])
                for col_name in context.species_names
            ]
            rows.append("[" + ", ".join(rendered) + "]")
        return tuple(rows)
    raise SymbolicCalculatorError(f"Unsupported calculator operation {name!r}.")


def _reject_nonfinite_symbolic_result(sympy: Any, expr: Any) -> None:
    if expr.has(sympy.zoo, sympy.oo, -sympy.oo, sympy.nan):
        raise SymbolicCalculatorError("Calculator expression produced a non-finite symbolic result.")


def _evaluate_call(context: _CalculatorContext, name: str, args: Sequence[Any]) -> Any:
    sympy = require_sympy()
    if name in _SUPPORTED_TRANSFORMS:
        if len(args) != 1:
            raise SymbolicCalculatorError(f"{name} requires exactly one argument.")
        expr = args[0]
        if isinstance(expr, tuple):
            raise SymbolicCalculatorError(f"{name} requires a scalar symbolic expression.")
        return getattr(sympy, name)(expr)
    if name == "collect":
        if len(args) != 2:
            raise SymbolicCalculatorError("collect requires an expression and a symbol.")
        expr = args[0]
        if isinstance(expr, tuple):
            raise SymbolicCalculatorError("collect requires a scalar symbolic expression.")
        return sympy.collect(expr, args[1])
    if name == "expand_params":
        if len(args) != 1:
            raise SymbolicCalculatorError("expand_params requires exactly one expression.")
        expr = args[0]
        if isinstance(expr, tuple):
            raise SymbolicCalculatorError("expand_params requires a scalar symbolic expression.")
        parameter_definitions, _rendered = _translated_parameter_definitions(context.mechanism)
        substitutions = {
            sympy.Symbol(name): value
            for name, value in parameter_definitions.items()
        }
        expanded = sympy.expand(expr.xreplace(substitutions))
        _reject_nonfinite_symbolic_result(sympy, expanded)
        return expanded
    raise SymbolicCalculatorError(f"Unsupported calculator operation {name!r}.")


def _require_scalar_expression(expr: Any, *, operation: str) -> Any:
    if isinstance(expr, tuple):
        raise SymbolicCalculatorError(f"{operation} requires a scalar symbolic expression.")
    return expr


def _parse_calculator_expression(context: _CalculatorContext, source: str) -> Any:
    parser = _CalculatorExpressionParser(context, source)
    expr = parser.parse()
    if not isinstance(expr, tuple):
        _reject_nonfinite_symbolic_result(require_sympy(), expr)
    return expr


@dataclass(frozen=True, slots=True)
class _Token:
    kind: str
    text: str
    pos: int


class _CalculatorExpressionParser:
    def __init__(self, context: _CalculatorContext, source: str) -> None:
        self._context = context
        self._source = str(source or "")
        self._tokens = self._tokenize(self._source)
        self._idx = 0

    def parse(self) -> Any:
        expr = self._parse_additive()
        if not self._at("EOF"):
            token = self._peek()
            raise SymbolicCalculatorError(f"Unexpected calculator token {token.text!r}.")
        return expr

    @staticmethod
    def _tokenize(source: str) -> list[_Token]:
        tokens: list[_Token] = []
        idx = 0
        while idx < len(source):
            char = source[idx]
            if char.isspace():
                idx += 1
                continue
            if char.isalpha() or char == "_":
                start = idx
                idx += 1
                while idx < len(source) and (source[idx].isalnum() or source[idx] == "_"):
                    idx += 1
                tokens.append(_Token("IDENT", source[start:idx], start))
                continue
            if char.isdigit() or char == ".":
                start = idx
                if char == "." and (idx + 1 >= len(source) or not source[idx + 1].isdigit()):
                    raise SymbolicCalculatorError(f"Unexpected calculator token {char!r}.")
                idx += 1
                while idx < len(source) and (source[idx].isdigit() or source[idx] in "_."):
                    idx += 1
                if idx < len(source) and source[idx] in "eE":
                    idx += 1
                    if idx < len(source) and source[idx] in "+-":
                        idx += 1
                    if idx >= len(source) or not source[idx].isdigit():
                        raise SymbolicCalculatorError("Invalid numeric literal in calculator expression.")
                    while idx < len(source) and (source[idx].isdigit() or source[idx] == "_"):
                        idx += 1
                literal = source[start:idx]
                _validate_numeric_literal_text(literal)
                tokens.append(_Token("NUMBER", literal, start))
                continue
            if char in "+-/^(),":
                tokens.append(_Token(char, char, idx))
                idx += 1
                continue
            if char == "*":
                if idx + 1 < len(source) and source[idx + 1] == "*":
                    tokens.append(_Token("**", "**", idx))
                    idx += 2
                else:
                    tokens.append(_Token("*", "*", idx))
                    idx += 1
                continue
            raise SymbolicCalculatorError(f"Unexpected calculator token {char!r}.")
        tokens.append(_Token("EOF", "", len(source)))
        return tokens

    def _parse_additive(self) -> Any:
        expr = self._parse_multiplicative()
        while self._at("+") or self._at("-"):
            op = self._advance().kind
            rhs = self._parse_multiplicative()
            expr = _require_scalar_expression(expr, operation="Arithmetic composition")
            rhs = _require_scalar_expression(rhs, operation="Arithmetic composition")
            if op == "+":
                expr = expr + rhs
            else:
                expr = expr - rhs
        return expr

    def _parse_multiplicative(self) -> Any:
        expr = self._parse_unary()
        while self._at("*") or self._at("/"):
            op = self._advance().kind
            rhs = self._parse_unary()
            expr = _require_scalar_expression(expr, operation="Arithmetic composition")
            rhs = _require_scalar_expression(rhs, operation="Arithmetic composition")
            if op == "*":
                expr = expr * rhs
            else:
                expr = expr / rhs
                _reject_nonfinite_symbolic_result(require_sympy(), expr)
        return expr

    def _parse_power(self) -> Any:
        expr = self._parse_primary()
        if self._at("^") or self._at("**"):
            self._advance()
            rhs = self._parse_unary()
            expr = _require_scalar_expression(expr, operation="Exponentiation")
            rhs = _require_scalar_expression(rhs, operation="Exponentiation")
            expr = expr ** rhs
            _reject_nonfinite_symbolic_result(require_sympy(), expr)
        return expr

    def _parse_unary(self) -> Any:
        if self._at("+"):
            self._advance()
            return _require_scalar_expression(self._parse_unary(), operation="Unary plus")
        if self._at("-"):
            self._advance()
            return -_require_scalar_expression(self._parse_unary(), operation="Unary minus")
        return self._parse_power()

    def _parse_primary(self) -> Any:
        if self._at("NUMBER"):
            literal = self._advance().text.replace("_", "")
            try:
                return require_sympy().Rational(literal)
            except Exception as exc:
                raise SymbolicCalculatorError("Invalid numeric literal in calculator expression.") from exc
        if self._at("("):
            self._advance()
            expr = self._parse_additive()
            self._expect(")")
            return expr
        if self._at("IDENT"):
            return self._parse_identifier_primary()
        token = self._peek()
        raise SymbolicCalculatorError(f"Unexpected calculator token {token.text!r}.")

    def _parse_identifier_primary(self) -> Any:
        token = self._advance()
        name = token.text
        if name == "d" and self._at("("):
            return self._parse_field_derivative_after_d()
        if self._should_parse_compact_derivative(name):
            return self._parse_derivative_after_numerator(name)
        if self._at("("):
            return self._parse_call_after_name(name)
        return self._symbol_for_identifier(name)

    def _should_parse_compact_derivative(self, numerator: str) -> bool:
        if not numerator.startswith("d") or not self._at("/"):
            return False
        if numerator in self._context.expression_symbols or numerator in self._context.let_sources:
            return False
        denominator_index = self._idx + 1
        if denominator_index >= len(self._tokens):
            return False
        denominator_token = self._tokens[denominator_index]
        if denominator_token.kind != "IDENT":
            return False
        denominator = denominator_token.text
        numerator_species = numerator[1:]
        if denominator == "dt":
            return numerator_species in self._context.rhs_by_species
        if denominator.startswith("d"):
            if denominator in self._context.expression_symbols or denominator in self._context.let_sources:
                return False
            denominator_species = denominator[1:]
            return (
                numerator_species in self._context.rhs_by_species
                and denominator_species in self._context.rhs_by_species
            )
        return False

    def _parse_field_derivative_after_d(self) -> Any:
        self._expect("(")
        if not self._at("IDENT"):
            raise SymbolicCalculatorError("Field derivative requires a rate expression.")
        numerator = self._advance().text
        if not self._is_unambiguous_derivative_symbol(numerator):
            raise SymbolicCalculatorError("Field derivative requires a rate expression.")
        self._expect("/")
        denominator_time = self._expect("IDENT").text
        if denominator_time != "dt":
            raise SymbolicCalculatorError("Field derivative requires a rate expression.")
        self._expect(")")
        self._expect("/")
        denominator = self._expect("IDENT").text
        if not self._is_unambiguous_derivative_symbol(denominator):
            raise SymbolicCalculatorError("Field derivative requires a species denominator.")
        num = _require_species(self._context, numerator[1:])
        den = _require_species(self._context, denominator[1:])
        return require_sympy().diff(
            self._context.rhs_by_species[num],
            _display_symbol_for_species(self._context, den),
        )

    def _is_unambiguous_derivative_symbol(self, name: str) -> bool:
        return (
            name.startswith("d")
            and name not in self._context.expression_symbols
            and name not in self._context.let_sources
            and name[1:] in self._context.rhs_by_species
        )

    def _parse_derivative_after_numerator(self, numerator: str) -> Any:
        self._expect("/")
        denominator = self._expect("IDENT").text
        if denominator == "dt":
            return self._context.rhs_by_species[_require_species(self._context, numerator[1:])]
        if denominator.startswith("d"):
            num = _require_species(self._context, numerator[1:])
            den = _require_species(self._context, denominator[1:])
            denominator_expr = self._context.rhs_by_species[den]
            if denominator_expr == 0:
                raise SymbolicCalculatorError(f"d{den}/dt is zero, so d{num}/d{den} is undefined.")
            return require_sympy().simplify(self._context.rhs_by_species[num] / denominator_expr)
        raise SymbolicCalculatorError(f"Unknown symbolic identifier {denominator!r}.")

    def _parse_call_after_name(self, name: str) -> Any:
        if name not in _SUPPORTED_TRANSFORMS and name not in {"collect", "expand_params", "odes", "jacobian"}:
            raise SymbolicCalculatorError(f"Unsupported calculator operation {name!r}.")
        self._expect("(")
        if name == "odes" or name == "jacobian":
            if not self._at(")"):
                raise SymbolicCalculatorError(f"{name} takes no arguments.")
            self._expect(")")
            return _evaluate_zero_arg_query(self._context, name)
        args: list[Any] = []
        if not self._at(")"):
            if name == "collect":
                args.append(self._parse_additive())
                if not self._at(","):
                    raise SymbolicCalculatorError("collect requires an expression and a symbol.")
                self._advance()
                args.append(self._parse_collect_symbol())
            else:
                args.append(self._parse_additive())
                while self._at(","):
                    self._advance()
                    args.append(self._parse_additive())
        self._expect(")")
        return _evaluate_call(self._context, name, args)

    def _parse_collect_symbol(self) -> Any:
        if not self._at("IDENT"):
            raise SymbolicCalculatorError("collect requires a symbol as its second argument.")
        name = _canonicalize_identifier(self._context, self._advance().text)
        if name not in self._context.expression_symbols:
            raise SymbolicCalculatorError(f"Unknown collect symbol {name!r}.")
        return self._context.expression_symbols[name]

    def _symbol_for_identifier(self, name: str) -> Any:
        if name in self._context.let_sources:
            return _translate_let_by_name(self._context, name, stack=())
        canonical = _canonicalize_identifier(self._context, name)
        if canonical not in self._context.expression_symbols:
            raise SymbolicCalculatorError(f"Unknown symbolic identifier {canonical!r}.")
        return self._context.expression_symbols[canonical]

    def _at(self, kind: str) -> bool:
        return self._peek().kind == kind

    def _peek(self) -> _Token:
        return self._tokens[self._idx]

    def _advance(self) -> _Token:
        token = self._peek()
        self._idx += 1
        return token

    def _expect(self, kind: str) -> _Token:
        if not self._at(kind):
            token = self._peek()
            expected = kind if kind != "IDENT" else "identifier"
            actual = token.text or "end of query"
            raise SymbolicCalculatorError(f"Expected {expected}, got {actual!r}.")
        return self._advance()


def _canonicalize_identifier(context: _CalculatorContext, raw_name: str) -> str:
    name = str(raw_name)
    if name in context.expression_symbols:
        return name
    resolved = context.parameter_namespace.resolve(name)
    if resolved.canonical_name is not None and resolved.canonical_name in context.expression_symbols:
        return resolved.canonical_name
    invalid_message = context.parameter_namespace.invalid_protected_indexed_identifier_message(name)
    if invalid_message is not None:
        raise SymbolicCalculatorError(invalid_message)
    return name


def _validate_numeric_literals_in_source(source: str) -> None:
    for match in _NUMERIC_LITERAL_RE.finditer(str(source or "")):
        _validate_numeric_literal_text(match.group(0))


def _validate_numeric_literal_text(literal: str) -> None:
    literal_s = str(literal).replace("_", "")
    try:
        decimal_value = Decimal(literal_s)
    except InvalidOperation:
        return
    try:
        float_value = float(literal_s)
    except (OverflowError, ValueError):
        raise SymbolicCalculatorError(
            "Numeric literal is too large to preserve safely in calculator expressions."
        )
    if not math.isfinite(float_value):
        raise SymbolicCalculatorError("Only finite numeric literals are supported in calculator expressions.")
    if decimal_value.is_finite() and decimal_value != 0 and float_value == 0.0:
        raise SymbolicCalculatorError(
            "Numeric literal is too small to preserve safely in calculator expressions."
        )


def _parameter_definition_sources(mechanism: Any) -> dict[str, str]:
    spec = _parameter_spec(mechanism)
    if spec is None:
        return {}
    return {
        str(assignment.name): str(assignment.expr_src).replace(" ", "")
        for assignment in spec.param_statements
    }


def _translated_parameter_definitions(mechanism: Any) -> tuple[dict[str, Any], dict[str, str]]:
    spec = _parameter_spec(mechanism)
    if spec is None:
        return {}, {}
    sympy = require_sympy()
    translated_by_name: dict[str, Any] = {}
    rendered: dict[str, str] = {}
    for assignment in spec.param_statements:
        name = str(assignment.name)
        _validate_numeric_literals_in_source(str(assignment.expr_src))
        try:
            translated = translate_parameter_expression(assignment, spec=spec)
        except UnsupportedSymbolicExpressionError as exc:
            raise SymbolicCalculatorError(str(exc)) from exc
        _reject_nonfinite_symbolic_result(sympy, translated.expression)
        translated_by_name[name] = translated
        rendered[name] = translated.normalized_source

    expanded_by_name: dict[str, Any] = {}

    def expand_assignment(name: str, stack: tuple[str, ...] = ()) -> Any:
        if name in expanded_by_name:
            return expanded_by_name[name]
        if name in stack:
            raise SymbolicCalculatorError(f"Cyclic symbolic parameter dependency for {name!r}.")
        translated = translated_by_name[name]
        substitutions = {
            sympy.Symbol(dep_name): expand_assignment(dep_name, stack + (name,))
            for dep_name in translated.canonical_identifiers
            if dep_name in translated_by_name
        }
        expr = translated.expression
        if substitutions:
            expr = sympy.expand(expr.xreplace(substitutions))
        _reject_nonfinite_symbolic_result(sympy, expr)
        expanded_by_name[name] = expr
        return expr

    expressions = {
        name: expand_assignment(name)
        for name in translated_by_name
    }
    return expressions, rendered


def _parameter_spec(mechanism: Any) -> ParameterAlgebraSpec | None:
    spec = parameter_algebra_spec_from_mechanism(mechanism)
    if spec is not None:
        return spec
    metadata = MechanismMetadataView.from_metadata(getattr(mechanism, "metadata", {}) or {})
    algebra_text = str(metadata.algebra_text or "").strip()
    if not algebra_text:
        return None
    namespace = build_namespace_from_mechanism(mechanism)
    parsed = parse_parameter_algebra_spec_from_dsl_text(
        algebra_text,
        mechanism_namespace=namespace,
    )
    if not parsed.param_statements:
        return parsed if parsed.observable_names else None
    return parsed


def _let_sources(mechanism: Any) -> dict[str, str]:
    metadata = MechanismMetadataView.from_metadata(getattr(mechanism, "metadata", {}) or {})
    algebra_text = str(metadata.algebra_text or "").strip()
    if not algebra_text:
        return {}
    out: dict[str, Any] = {}
    for line_no, line in enumerate(algebra_text.splitlines(), start=1):
        classification = classify_parameter_algebra_declaration(line, line_number=line_no)
        if classification.kind != "let":
            continue
        code = str(classification.code)
        rhs = code.split("=", 1)[1].strip()
        out[str(classification.raw_name)] = rhs
    return out


def _supported_let_sources(context: _CalculatorContext) -> dict[str, str]:
    supported: dict[str, str] = {}
    for name, source in context.let_sources.items():
        try:
            _translate_let_by_name(context, name, stack=())
        except SymbolicCalculatorError:
            continue
        supported[str(name)] = str(source)
    return supported


def _translate_let_by_name(context: _CalculatorContext, name: str, *, stack: Sequence[str]) -> Any:
    if name in stack:
        raise SymbolicCalculatorError(f"Unsupported let expression cycle at {name!r}.")
    source = context.let_sources.get(name)
    if source is None:
        raise SymbolicCalculatorError(f"Unknown let observable {name!r}.")
    _validate_numeric_literals_in_source(source)
    block = parse_algebra(f"# Algebra\nlet {name} = {source}\n")
    if not block.lines:
        raise SymbolicCalculatorError(f"Unsupported let expression {name!r}.")
    result = _translate_let_node(
        block.lines[0].expr,
        context=context,
        stack=tuple(stack) + (name,),
    )
    _reject_nonfinite_symbolic_result(require_sympy(), result)
    return result


def _translate_let_node(node: Any, *, context: _CalculatorContext, stack: Sequence[str]) -> Any:
    sympy = require_sympy()
    if isinstance(node, NumberNode):
        try:
            literal_source = expression_to_source(node)
        except ValueError as exc:
            raise SymbolicCalculatorError(str(exc)) from exc
        _validate_numeric_literals_in_source(literal_source)
        return sympy.Rational(str(float(node.value)))
    if isinstance(node, IdentNode):
        name = str(node.name)
        if name in context.let_sources:
            return _translate_let_by_name(context, name, stack=stack)
        name = _canonicalize_identifier(context, name)
        if name not in context.expression_symbols:
            raise SymbolicCalculatorError(f"Unsupported let expression identifier {name!r}.")
        return context.expression_symbols[name]
    if isinstance(node, SpeciesRefNode):
        if node.kind != "now":
            raise SymbolicCalculatorError("Unsupported let expression species reference.")
        name = str(node.name)
        if name not in context.expression_symbols:
            raise SymbolicCalculatorError(f"Unsupported let expression species {name!r}.")
        return context.expression_symbols[name]
    if isinstance(node, UnaryNode):
        rhs = _translate_let_node(node.rhs, context=context, stack=stack)
        if node.op == "+":
            return rhs
        if node.op == "-":
            return -rhs
        raise SymbolicCalculatorError("Unsupported let expression unary operator.")
    if isinstance(node, BinaryNode):
        lhs = _translate_let_node(node.lhs, context=context, stack=stack)
        rhs = _translate_let_node(node.rhs, context=context, stack=stack)
        if node.op == "+":
            return lhs + rhs
        if node.op == "-":
            return lhs - rhs
        if node.op == "*":
            return lhs * rhs
        if node.op == "/":
            result = lhs / rhs
            _reject_nonfinite_symbolic_result(sympy, result)
            return result
        if node.op in {"^", "**"}:
            return lhs ** rhs
        raise SymbolicCalculatorError("Unsupported let expression binary operator.")
    if isinstance(node, CallNode):
        raise SymbolicCalculatorError("Unsupported let expression function call.")
    raise SymbolicCalculatorError(f"Unsupported let expression node {type(node).__name__}.")


def _display_expr(context: _CalculatorContext, expr: Any) -> Any:
    return expr.xreplace(dict(context.internal_to_display))


def _render_expr(context: _CalculatorContext, expr: Any) -> str:
    displayed = _display_expr(context, expr)
    return str(displayed)


def _require_species(context: _CalculatorContext, name: str) -> str:
    name_s = str(name)
    if name_s not in context.species_names:
        raise SymbolicCalculatorError(f"Unknown species {name_s!r}.")
    return name_s


def _display_symbol_for_species(context: _CalculatorContext, name: str) -> Any:
    return context.expression_symbols[_require_species(context, name)]
