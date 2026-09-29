from __future__ import annotations

from kindred.core.quantity_units import Unit, parse_declared_unit

from dataclasses import dataclass, field, replace
import re
from types import MappingProxyType
from typing import List, Mapping, Sequence, Set, Tuple

from kindred.core.algebra.parser import (
    AlgebraBlock,
    BinaryNode,
    CallNode,
    ExprNode,
    IdentNode,
    LetStatement,
    NumberNode,
    SpeciesRefNode,
    UnaryNode,
    parse_algebra,
)
from kindred.core.algebra.symbols import SymbolTable
from kindred.core.simulator.errors import DSLError
from kindred.core.simulator.parameter_namespace import (
    MechanismParameterNamespace,
    is_protected_indexed_identifier,
)
from .dsl_build_input import (
    DslSourceCoordinate,
    source_coordinate_capabilities,
    source_coordinate_name_pattern,
)

_PARAM_STMT_RE = re.compile(r"^\s*(?i:param)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?)\s*$")
_LET_STMT_RE = re.compile(r"^\s*(?i:let)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?)\s*$")
_ASSIGN_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=")
_ARROW_RE = re.compile(r"<->|<=>|->|=>")
_MECH_PARAM_RE = source_coordinate_name_pattern()
_PROTECTED_STEP_KEY_IDENTIFIERS = frozenset({"k", "kf", "kr", "keq"})
_NON_ALGEBRA_ASSIGNMENT_PREFIXES = {
    "energy",
    "init",
    "initial",
    "t",
    "temp_response",
    "temp_step",
    "time",
    "c0",
}


def _raise_invalid_protected_indexed_identifier_error(
    raw_name: str,
    *,
    suggested_names: Sequence[str],
    line_number: int,
    line_content: str,
) -> None:
    suggestions = ", ".join(str(name) for name in suggested_names)
    raise DSLError(
        f"{raw_name!r} is not a valid indexed parameter identifier.",
        suggestion=(
            "Use the canonical indexed parameter name"
            + (f" for this step: {suggestions}." if suggestions else ".")
            + " Exact protected indexed names must resolve to an existing mechanism parameter."
        ),
        line_number=line_number,
        line_content=line_content,
    )


def invalid_parameter_algebra_identifier_reference_message(
    raw_name: str,
    *,
    mechanism_namespace: MechanismParameterNamespace | None = None,
    reject_unresolved_protected_indexed: bool = False,
) -> str | None:
    name = str(raw_name or "").strip()
    if not name:
        return None
    if is_protected_step_key_identifier(name):
        return (
            f"{name!r} is a step-local DSL key and cannot be used as an Algebra identifier. "
            "Use a canonical indexed mechanism parameter such as k1, kf1, kr1, or Keq1, "
            "or choose a longer ordinary name such as 'K1_test'."
        )
    if mechanism_namespace is not None:
        if mechanism_namespace.resolve(name).canonical_name is not None:
            return None
        invalid_protected = mechanism_namespace.invalid_protected_indexed_identifier(name)
        if invalid_protected is not None:
            suggestions = ", ".join(str(suggestion) for suggestion in invalid_protected.suggested_names)
            suggestion_clause = (
                f" Use canonical indexed parameter name(s) for this step: {suggestions}."
                if suggestions
                else " Use an existing canonical indexed mechanism parameter, or choose a longer ordinary name."
            )
            return f"{name!r} is not a valid indexed parameter identifier.{suggestion_clause}"
        return None
    if reject_unresolved_protected_indexed and is_protected_indexed_identifier(name):
        return (
            f"{name!r} is a protected indexed parameter identifier. "
            "Exact protected indexed names must resolve through the mechanism namespace."
        )
    return None


def is_protected_step_key_identifier(raw_name: str) -> bool:
    name = str(raw_name or "").strip()
    return name == "K" or name.lower() in _PROTECTED_STEP_KEY_IDENTIFIERS


def _raise_protected_observable_identifier_error(
    raw_name: str,
    *,
    line_number: int,
    line_content: str,
) -> None:
    raise DSLError(
        f"{raw_name!r} is a protected indexed parameter identifier and cannot be declared as an observable.",
        suggestion=(
            "Use 'param name = expr' for mechanism parameter algebra, or choose a longer observable name "
            "such as 'K1_test' if this is not a mechanism parameter."
        ),
        line_number=line_number,
        line_content=line_content,
    )


def _raise_protected_step_key_identifier_error(
    raw_name: str,
    *,
    line_number: int,
    line_content: str,
) -> None:
    raise DSLError(
        f"{raw_name!r} is a step-local DSL key and cannot be declared as a parameter or observable.",
        suggestion=(
            "Use a canonical indexed mechanism parameter such as k1, kf1, kr1, or Keq1, "
            "or choose a longer ordinary name such as 'K1_test'."
        ),
        line_number=line_number,
        line_content=line_content,
    )


def _raise_unsupported_bare_assignment_error(
    raw_name: str,
    *,
    line_number: int,
    line_content: str,
) -> None:
    raise DSLError(
        f"Bare algebra assignment {raw_name!r} is not supported.",
        suggestion="Use 'let name = expr' or 'param name = expr'.",
        line_number=line_number,
        line_content=line_content,
    )


def _reserved_name_message(name: str, declaration: str) -> str:
    """Say which built-in a rejected parameter or let name would shadow."""
    if name == "T":
        reserved = "the built-in temperature"
    elif name in SymbolTable().functions():
        reserved = "a built-in function"
    else:
        reserved = "a built-in constant"
    return f"{name!r} is {reserved} and cannot be redefined as a {declaration}."


@dataclass(frozen=True)
class ParameterAssignment:
    name: str
    line_number: int
    line_content: str
    source_coordinate: DslSourceCoordinate | None = None
    declared_unit: Unit | None = None


@dataclass(frozen=True)
class ParameterAlgebraDeclarationClassification:
    kind: str
    raw_name: str = ""
    code: str = ""
    line_number: int = 0
    line_content: str = ""
    canonical_name: str | None = None
    invalid_protected_indexed_identifier: bool = False
    invalid_protected_indexed_suggestions: Tuple[str, ...] = ()


def classify_parameter_algebra_declaration(
    line: str,
    *,
    line_number: int = 0,
    mechanism_namespace: MechanismParameterNamespace | None = None,
    allow_non_algebra_bare_assignment: bool = True,
) -> ParameterAlgebraDeclarationClassification:
    original = str(line or "").rstrip("\r\n")
    stripped = original.strip()
    if not stripped or stripped.startswith("#"):
        return ParameterAlgebraDeclarationClassification(
            kind="empty",
            line_number=int(line_number or 0),
            line_content=original,
        )
    code = strip_inline_comment(original).strip()
    if not code:
        return ParameterAlgebraDeclarationClassification(
            kind="empty",
            line_number=int(line_number or 0),
            line_content=original,
        )
    for kind, regex in (("param", _PARAM_STMT_RE), ("let", _LET_STMT_RE)):
        match = regex.match(code)
        if match is None:
            continue
        raw_name = str(match.group(1))
        rhs = str(match.group(2) or "").lstrip()
        if rhs.startswith("{"):
            return ParameterAlgebraDeclarationClassification(
                kind="non_algebra",
                code=code,
                line_number=int(line_number or 0),
                line_content=original,
            )
        if is_protected_step_key_identifier(raw_name):
            return ParameterAlgebraDeclarationClassification(
                kind="invalid_step_key_identifier",
                raw_name=raw_name,
                code=code,
                line_number=int(line_number or 0),
                line_content=original,
            )
        canonical_name = None
        invalid_protected_found = False
        invalid_suggestions: Tuple[str, ...] = ()
        if mechanism_namespace is not None:
            resolution = mechanism_namespace.resolve(raw_name)
            canonical_name = resolution.canonical_name
            if canonical_name is None:
                invalid_protected = mechanism_namespace.invalid_protected_indexed_identifier(raw_name)
                if invalid_protected is not None:
                    invalid_protected_found = True
                    invalid_suggestions = tuple(str(name) for name in invalid_protected.suggested_names)
        return ParameterAlgebraDeclarationClassification(
            kind=kind,
            raw_name=raw_name,
            code=code,
            line_number=int(line_number or 0),
            line_content=original,
            canonical_name=canonical_name,
            invalid_protected_indexed_identifier=bool(invalid_protected_found),
            invalid_protected_indexed_suggestions=invalid_suggestions,
        )
    match = None if _ARROW_RE.search(code) else _ASSIGN_RE.match(code)
    if match is not None:
        rhs = code[match.end():].lstrip()
        raw_name = str(match.group(1))
        if (
            allow_non_algebra_bare_assignment
            and (rhs.startswith("{") or raw_name.lower() in _NON_ALGEBRA_ASSIGNMENT_PREFIXES)
        ):
            return ParameterAlgebraDeclarationClassification(
                kind="non_algebra",
                code=code,
                line_number=int(line_number or 0),
                line_content=original,
            )
        return ParameterAlgebraDeclarationClassification(
            kind="unsupported_bare_assignment",
            raw_name=raw_name,
            code=code,
            line_number=int(line_number or 0),
            line_content=original,
        )
    return ParameterAlgebraDeclarationClassification(
        kind="non_algebra",
        code=code,
        line_number=int(line_number or 0),
        line_content=original,
    )


@dataclass(frozen=True)
class ParameterAlgebraNamespace:
    mechanism_param_names: Set[str]
    param_assignment_names: Set[str]
    scalar_input_names: Set[str]
    observable_names: Set[str]
    builtin_function_names: Set[str]
    protected_symbol_names: Set[str]

    def __post_init__(self) -> None:
        for field_name in (
            "mechanism_param_names",
            "param_assignment_names",
            "scalar_input_names",
            "observable_names",
            "builtin_function_names",
            "protected_symbol_names",
        ):
            object.__setattr__(self, field_name, frozenset(getattr(self, field_name)))

    def solver_param_names(self) -> Set[str]:
        return set(self.mechanism_param_names) | set(self.param_assignment_names)


@dataclass(frozen=True)
class ParameterAlgebraSpec:
    """
    Parsed `param` statements from mechanism DSL text.

    Semantics
    ---------
    - Scalar base parameters: `param a = 4` declares an adjustable solver parameter
      with a default value. It is NOT a constraint and may be overridden by sliders/fitting.
    - Derived parameters (constraints): any `param <name> = <expr>` that references
      other solver parameters OR targets a mechanism parameter (k1, k2, kf1, ...).
      Derived parameters are re-evaluated whenever solver parameters change.
    """

    param_statements: Tuple[ParameterAssignment, ...]
    observable_names: Set[str]
    mechanism_namespace: MechanismParameterNamespace
    scalar_input_names: Set[str] = field(default_factory=set)
    parameter_block: AlgebraBlock | None = None
    observable_block: AlgebraBlock | None = None
    # Authored-assignment projection for symbolic/source consumers. Numerical
    # ordering and classification use evaluation_dependencies, which also
    # includes chemistry readouts and their canonical source coordinates.
    parameter_dependencies: Mapping[str, Tuple[str, ...]] = field(
        default_factory=dict
    )
    evaluation_dependencies: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)
    chemistry_dependencies: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)
    temperature_dependent_names: frozenset[str] = frozenset()
    identity_checks: tuple[str, ...] = ()
    quantity_units: tuple[tuple[str, Unit | None], ...] = ()
    input_units: tuple[tuple[str, Unit], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "param_statements", tuple(self.param_statements))
        object.__setattr__(self, "observable_names", frozenset(self.observable_names))
        object.__setattr__(self, "scalar_input_names", frozenset(self.scalar_input_names))
        dependencies = {
            str(name): tuple(dict.fromkeys(str(dep) for dep in values))
            for name, values in dict(self.parameter_dependencies).items()
        }
        object.__setattr__(
            self,
            "parameter_dependencies",
            MappingProxyType(dependencies),
        )
        for field_name in ("evaluation_dependencies", "chemistry_dependencies"):
            object.__setattr__(self, field_name, MappingProxyType({name: tuple(deps) for name, deps in getattr(self, field_name).items()}))
        object.__setattr__(self, "temperature_dependent_names", frozenset(self.temperature_dependent_names))
        object.__setattr__(self, "identity_checks", tuple(self.identity_checks))

    def __deepcopy__(self, memo: dict[int, object]) -> "ParameterAlgebraSpec":
        import copy

        clone = type(self)(
            self.param_statements,
            self.observable_names,
            copy.deepcopy(self.mechanism_namespace, memo),
            self.scalar_input_names,
            copy.deepcopy(self.parameter_block, memo),
            copy.deepcopy(self.observable_block, memo),
            {
                name: tuple(dependencies)
                for name, dependencies in self.parameter_dependencies.items()
            },
            dict(self.evaluation_dependencies), dict(self.chemistry_dependencies),
            self.temperature_dependent_names, self.identity_checks,
            self.quantity_units, self.input_units,
        )
        memo[id(self)] = clone
        return clone

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        return (
            type(self),
            (
                self.param_statements,
                self.observable_names,
                self.mechanism_namespace,
                self.scalar_input_names,
                self.parameter_block,
                self.observable_block,
                {
                    name: tuple(dependencies)
                    for name, dependencies in self.parameter_dependencies.items()
                },
                dict(self.evaluation_dependencies), dict(self.chemistry_dependencies),
                self.temperature_dependent_names, self.identity_checks,
                self.quantity_units, self.input_units,
            ),
        )

    @property
    def mechanism_param_names(self) -> Set[str]:
        return self.mechanism_namespace.flat_names()

    def param_assignment_names(self) -> Set[str]:
        return {p.name for p in self.param_statements}

    def parameter_assignment_kind(self, name: str) -> str:
        """Classify one admitted assignment from the retained graph."""

        requested = str(name)
        if requested not in self.parameter_dependencies:
            raise KeyError(f"Unknown parameter assignment {requested!r}.")
        if requested in self.identity_checks:
            return "source identity"
        if requested in self.mechanism_param_names:
            return "mechanism-bound"
        if self.evaluation_dependencies.get(requested, self.parameter_dependencies[requested]):
            return "derived scalar"
        return "independent scalar"

    def solver_param_names(self) -> Set[str]:
        return self.namespace_model().solver_param_names()

    def namespace_model(self) -> ParameterAlgebraNamespace:
        symtab = SymbolTable()
        return ParameterAlgebraNamespace(
            mechanism_param_names=self.mechanism_namespace.flat_names(),
            param_assignment_names=self.param_assignment_names(),
            scalar_input_names=set(self.scalar_input_names),
            observable_names=set(self.observable_names),
            builtin_function_names=set(symtab.functions().keys()),
            protected_symbol_names=set(symtab.protected_names()),
        )


def admit_parameter_evaluation_graph(
    spec: ParameterAlgebraSpec, mechanism: object
) -> ParameterAlgebraSpec:
    """Compose authored definitions with typed chemistry before publication."""
    from graphlib import CycleError, TopologicalSorter
    from .parameter_coordinates import (
        build_parameter_target_map,
        chemistry_parameter_dependencies,
    )
    from .exact_constraint_composition import proves_redundant_parameter

    targets = build_parameter_target_map(mechanism)
    chemistry = chemistry_parameter_dependencies(mechanism, targets)
    statements = {statement.name: statement for statement in spec.parameter_block.lines}
    names = set(targets) | set(statements) | {"T"}
    graph = {name: () for name in sorted(names)}
    graph.update(chemistry)
    for name, statement in statements.items():
        if name in chemistry:
            raise DSLError(
                f"{name} is a chemistry-derived readout, not an editable source coordinate.",
                line_number=statement.line,
                line_content=statement.line_text,
            )
        graph[name] = tuple(
            dict.fromkeys(
                dep for dep in _iter_expr_identifiers(statement.expr) if dep in names
            )
        )
    from .parameter_units import prepare_spec_units
    spec = prepare_spec_units(spec, mechanism, graph)
    statements = {statement.name: statement for statement in spec.parameter_block.lines}
    identities: list[str] = []
    while True:
        try:
            order = tuple(TopologicalSorter(graph).static_order())
            break
        except CycleError as exc:
            cycle = exc.args[1]
            candidate = next(
                (
                    name
                    for name in sorted(set(cycle))
                    if name in statements
                    and name in targets
                    and name not in chemistry
                    and proves_redundant_parameter(
                        mechanism, statements, name, identities
                    )
                ),
                None,
            )
            if candidate is None:
                raise DSLError(
                    "Circular parameter definition through chemistry: "
                    + " -> ".join(cycle)
                ) from exc
            graph[candidate] = ()
            identities.append(candidate)
    dynamic = {"T"}
    for name in order:
        if any(dep in dynamic for dep in graph.get(name, ())):
            dynamic.add(name)
    return replace(
        spec,
        evaluation_dependencies={name: graph.get(name, ()) for name in order},
        chemistry_dependencies=chemistry,
        temperature_dependent_names=frozenset(dynamic),
        identity_checks=tuple(identities),
    )


def _iter_expr_identifiers(expr: ExprNode):
    if isinstance(expr, IdentNode):
        yield str(expr.name)
    elif isinstance(expr, (NumberNode, SpeciesRefNode)):
        return
    elif isinstance(expr, UnaryNode):
        yield from _iter_expr_identifiers(expr.rhs)
    elif isinstance(expr, BinaryNode):
        yield from _iter_expr_identifiers(expr.lhs)
        yield from _iter_expr_identifiers(expr.rhs)
    elif isinstance(expr, CallNode):
        for arg in expr.args:
            yield from _iter_expr_identifiers(arg)


def _iter_call_names(expr: ExprNode):
    if isinstance(expr, CallNode):
        yield str(expr.name)
        for arg in expr.args:
            yield from _iter_call_names(arg)
    elif isinstance(expr, UnaryNode):
        yield from _iter_call_names(expr.rhs)
    elif isinstance(expr, BinaryNode):
        yield from _iter_call_names(expr.lhs)
        yield from _iter_call_names(expr.rhs)


def _canonicalize_parameter_expr(
    expr: ExprNode,
    *,
    mechanism_namespace: MechanismParameterNamespace,
    scalar_input_names: Set[str],
    assignment: ParameterAssignment,
) -> ExprNode:
    """Canonicalize mechanism identifiers while retaining the parsed AST."""
    if isinstance(expr, (NumberNode, SpeciesRefNode)):
        return expr
    if isinstance(expr, IdentNode):
        raw_name = str(expr.name)
        resolution = mechanism_namespace.resolve(raw_name)
        invalid = (
            mechanism_namespace.invalid_protected_indexed_identifier(raw_name)
            if resolution.canonical_name is None
            else None
        )
        if invalid is not None:
            _raise_invalid_protected_indexed_identifier_error(
                raw_name,
                suggested_names=invalid.suggested_names,
                line_number=assignment.line_number,
                line_content=assignment.line_content,
            )
        if resolution.canonical_name is None and is_protected_indexed_identifier(raw_name):
            _raise_invalid_protected_indexed_identifier_error(
                raw_name,
                suggested_names=(),
                line_number=assignment.line_number,
                line_content=assignment.line_content,
            )
        if resolution.canonical_name is None or raw_name in scalar_input_names:
            return expr
        source_info = mechanism_namespace.source_coordinate_info(
            str(resolution.canonical_name)
        )
        if source_info is not None and not source_coordinate_capabilities(
            source_info.role
        ).algebra_target:
            raise DSLError(
                f"Source coordinate {raw_name!r} is symbolic-only and cannot be used in Parameter Algebra.",
                suggestion="Use an authored continuous coordinate such as A1, Ea1, dG_act1, dG_eq1, or kappa1.",
            )
        return IdentNode(resolution.canonical_name)
    if isinstance(expr, UnaryNode):
        rhs = _canonicalize_parameter_expr(
            expr.rhs,
            mechanism_namespace=mechanism_namespace,
            scalar_input_names=scalar_input_names,
            assignment=assignment,
        )
        return UnaryNode(expr.op, rhs)
    if isinstance(expr, BinaryNode):
        lhs = _canonicalize_parameter_expr(
            expr.lhs,
            mechanism_namespace=mechanism_namespace,
            scalar_input_names=scalar_input_names,
            assignment=assignment,
        )
        rhs = _canonicalize_parameter_expr(
            expr.rhs,
            mechanism_namespace=mechanism_namespace,
            scalar_input_names=scalar_input_names,
            assignment=assignment,
        )
        return BinaryNode(expr.op, lhs, rhs)
    if isinstance(expr, CallNode):
        args: list[ExprNode] = []
        for arg in expr.args:
            canonical_arg = _canonicalize_parameter_expr(
                arg,
                mechanism_namespace=mechanism_namespace,
                scalar_input_names=scalar_input_names,
                assignment=assignment,
            )
            args.append(canonical_arg)
        return CallNode(expr.name, tuple(args))
    return expr


def _compile_retained_parameter_block(
    assignments: Sequence[ParameterAssignment],
    *,
    mechanism_namespace: MechanismParameterNamespace,
    scalar_input_names: Set[str],
    parsed: AlgebraBlock,
) -> AlgebraBlock:
    """Canonicalize the parameter slice of the retained admission AST."""
    canonical_lines: list[LetStatement] = []
    for assignment, statement in zip(assignments, parsed.lines):
        expr = _canonicalize_parameter_expr(
            statement.expr,
            mechanism_namespace=mechanism_namespace,
            scalar_input_names=set(scalar_input_names),
            assignment=assignment,
        )
        canonical_lines.append(
            LetStatement(
                name=assignment.name,
                expr=expr,
                line=int(assignment.line_number or statement.line),
                col=int(statement.col),
                line_text=assignment.line_content,
                declared_unit=assignment.declared_unit,
            )
        )
    block = AlgebraBlock(
        lines=tuple(canonical_lines),
        ast=tuple(stmt.expr for stmt in canonical_lines),
        static_values=parsed.static_values,
        evaluation_order=parsed.evaluation_order,
    )
    return block


def _validate_retained_parameter_program(
    spec: ParameterAlgebraSpec,
    *,
    assignments: Sequence[ParameterAssignment],
    block: AlgebraBlock,
) -> Mapping[str, Tuple[str, ...]]:
    """Validate names, roles, and dependency cycles before acceptance."""
    namespace = spec.namespace_model()
    defined = {str(assignment.name) for assignment in assignments}
    # Parameter algebra is static solver-parameter algebra. Runtime state,
    # row-local time, and temperature belong to dynamic ``let`` observables and
    # are rejected here before any source or mechanism mutation.
    allowed_constants = set(namespace.protected_symbol_names) - {"T"}
    # Keep the parser's admitted expression order.  This is the sole graph
    # retained by the spec; consumers must not rediscover it from the AST.
    dependencies: dict[str, tuple[str, ...]] = {}
    for assignment, statement in zip(assignments, block.lines):
        identifier_order = tuple(dict.fromkeys(_iter_expr_identifiers(statement.expr)))
        idents = set(identifier_order)
        if statement.expr.has_species_ref() or "T" in idents or "t" in idents:
            raise DSLError(
                f"Parameter expression for {assignment.name!r} must be static.",
                suggestion="Use numeric literals, other static parameters, and supported constants; put dynamic state expressions in a 'let'.",
                line_number=assignment.line_number,
                line_content=assignment.line_content,
            )
        dynamic_scalars = sorted(idents & set(namespace.scalar_input_names))
        if dynamic_scalars:
            raise DSLError(
                "Parameter expression cannot reference dynamic scalar input(s): "
                + ", ".join(repr(name) for name in dynamic_scalars),
                suggestion="Use admitted solver parameters in 'param' expressions and consume dynamic scalars from a 'let'.",
                line_number=assignment.line_number,
                line_content=assignment.line_content,
            )
        if not (idents - allowed_constants):
            # Literal/constant-only parameter expressions are statically
            # admitted; reject domain/division failures before mutation.
            try:
                import numpy as np

                from kindred.core.algebra.evaluator import EvaluationContext, evaluate_block

                evaluate_block(
                    AlgebraBlock(
                        lines=(statement,),
                        ast=(statement.expr,),
                        static_values={},
                    ),
                    EvaluationContext(
                        t=np.array([0.0]),
                        species_series={},
                        initials={},
                        species_names=set(),
                        symtab=SymbolTable(),
                    ),
                )
            except Exception as exc:
                raise DSLError(
                    f"Static parameter expression for {assignment.name!r} is invalid: {exc}",
                    line_number=assignment.line_number,
                    line_content=assignment.line_content,
                ) from exc
        unknown_functions = {
            name
            for name in _iter_call_names(statement.expr)
            if name not in namespace.builtin_function_names
        }
        if unknown_functions:
            raise DSLError(
                "Unknown function(s) in parameter expression: "
                + ", ".join(repr(name) for name in sorted(unknown_functions)),
                suggestion="Use a supported Algebra math function.",
                line_number=assignment.line_number,
                line_content=assignment.line_content,
            )
        unknown = {
            name
            for name in idents
            if name not in defined
            and name not in namespace.solver_param_names()
            and name not in allowed_constants
        }
        if unknown:
            raise DSLError(
                "Unknown name(s) in parameter expression: "
                + ", ".join(repr(name) for name in sorted(unknown)),
                suggestion="Parameter expressions may only reference other parameters and supported constants.",
                line_number=assignment.line_number,
                line_content=assignment.line_content,
            )
        dependencies[assignment.name] = tuple(
            name for name in identifier_order if name in defined
        )

    visiting: set[str] = set()
    visited: set[str] = set()
    ordered_names: list[str] = []

    def visit(name: str, stack: list[str]) -> None:
        if name in visited:
            return
        if name in visiting:
            cycle = stack[stack.index(name):] + [name]
            raise DSLError(
                "Parameter algebra has a dependency cycle: " + " -> ".join(cycle),
                suggestion="Rewrite the param definitions so they form an acyclic dependency graph.",
            )
        visiting.add(name)
        for dependency in dependencies.get(name, ()):
            visit(dependency, stack + [name])
        visiting.remove(name)
        visited.add(name)
        ordered_names.append(name)

    for name in block.evaluation_order:
        visit(name, [])

    # Defaults with no parameter references are admitted above. Derived
    # expressions are checked when consumed, even when their dependencies are
    # temperature-independent; an unused intervention must not force them.

    # ``ordered_names`` is the admitted evaluation order, and each tuple is
    # the direct identifier order retained from that assignment's AST.
    return {
        name: tuple(dependencies.get(name, ()))
        for name in ordered_names
    }


def _validate_retained_observable_program(
    spec: ParameterAlgebraSpec,
    *,
    block: AlgebraBlock,
) -> None:
    """Validate retained ``let`` names, domains, and dependency cycles.

    This is admission-only validation.  Dynamic concentration, ``t``/``T``,
    coefficient, and scalar references are intentionally left for the live
    execution context rather than evaluated against a nominal value.
    """
    namespace = spec.namespace_model()
    defined_params = set(namespace.param_assignment_names)
    defined_lets = {str(statement.name) for statement in block.lines}
    allowed_constants = set(namespace.protected_symbol_names)
    dynamic_identifiers = {"t", "T"}
    dependencies: dict[str, set[str]] = {}

    for statement in block.lines:
        name = str(statement.name)
        idents = set(_iter_expr_identifiers(statement.expr))
        dynamic = bool(
            statement.expr.has_species_ref()
            or (idents & dynamic_identifiers)
        )
        if not dynamic and not (idents - allowed_constants):
            try:
                import numpy as np

                from kindred.core.algebra.evaluator import EvaluationContext, evaluate_block

                evaluate_block(
                    AlgebraBlock(
                        lines=(statement,),
                        ast=(statement.expr,),
                        static_values={},
                    ),
                    EvaluationContext(
                        t=np.array([0.0]),
                        species_series={},
                        initials={},
                        species_names=set(),
                        symtab=SymbolTable(),
                    ),
                )
            except Exception as exc:
                raise DSLError(
                    f"Static observable expression for {name!r} is invalid: {exc}",
                    line_number=statement.line,
                    line_content=statement.line_text,
                ) from exc

        unknown_functions = {
            function_name
            for function_name in _iter_call_names(statement.expr)
            if function_name not in namespace.builtin_function_names
        }
        if unknown_functions:
            raise DSLError(
                "Unknown function(s) in observable expression: "
                + ", ".join(repr(function_name) for function_name in sorted(unknown_functions)),
                suggestion="Use a supported Algebra math function.",
                line_number=statement.line,
                line_content=statement.line_text,
            )

        unknown = {
            identifier
            for identifier in idents
            if identifier not in defined_params
            and identifier not in defined_lets
            and identifier not in namespace.mechanism_param_names
            and spec.mechanism_namespace.resolve(identifier).canonical_name is None
            and identifier not in namespace.scalar_input_names
            and identifier not in allowed_constants
            and identifier not in dynamic_identifiers
        }
        if unknown:
            raise DSLError(
                "Unknown name(s) in observable expression: "
                + ", ".join(repr(identifier) for identifier in sorted(unknown)),
                suggestion=(
                    "In `let` expressions, use `[X]` for the current concentration of species `X` "
                    "and `[X]_0` for its initial concentration. Bare names may reference declared "
                    "parameters (for example, `a` from `param a = 5`), available reaction parameters "
                    "(such as `k1`, `kf1`, `kr1`, `Keq1`, `A1`, `Ea1`, `dG_act1`, `dG_eq1`, or "
                    "`kappa1`), values defined by other `let` statements, time `t`, temperature `T`, "
                    "or the constants `k_B`, `h`, `h_bar`, `N_A`, `R`."
                ),
                line_number=statement.line,
                line_content=statement.line_text,
            )
        dependencies[name] = {
            identifier
            for identifier in idents
            if identifier in defined_lets
        }

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(name: str, stack: list[str]) -> None:
        if name in visited:
            return
        if name in visiting:
            cycle = stack[stack.index(name):] + [name]
            raise DSLError(
                "Observable algebra has a dependency cycle: " + " -> ".join(cycle),
                suggestion="Rewrite the let definitions so they form an acyclic dependency graph.",
            )
        visiting.add(name)
        for dependency in sorted(dependencies.get(name, ())):
            visit(dependency, stack + [name])
        visiting.remove(name)
        visited.add(name)

    for name in sorted(defined_lets):
        visit(name, [])

    static_names: set[str] = set()
    statement_by_name = {str(statement.name): statement for statement in block.lines}
    changed = True
    while changed:
        changed = False
        for name, statement in statement_by_name.items():
            if name in static_names:
                continue
            idents = set(_iter_expr_identifiers(statement.expr))
            if statement.expr.has_species_ref() or (idents & dynamic_identifiers):
                continue
            # A let that depends on a retained static parameter is valid, but
            # its domain is decided only after the prepared scalar snapshot is
            # available. Do not attempt nominal admission-time evaluation.
            external = idents - defined_lets - allowed_constants
            if external:
                continue
            if all(dependency in static_names for dependency in dependencies.get(name, ())):
                static_names.add(name)
                changed = True
    if static_names:
        ordered_static_names: list[str] = []
        ordered_seen: set[str] = set()

        def append_static(name: str) -> None:
            if name in ordered_seen:
                return
            for dependency in sorted(dependencies.get(name, ())):
                if dependency in static_names:
                    append_static(dependency)
            ordered_seen.add(name)
            ordered_static_names.append(name)

        for name in statement_by_name:
            if name in static_names:
                append_static(name)
        static_statements = tuple(
            statement_by_name[name]
            for name in ordered_static_names
        )
        try:
            import numpy as np

            from kindred.core.algebra.evaluator import EvaluationContext, evaluate_block

            evaluate_block(
                AlgebraBlock(
                    lines=static_statements,
                    ast=tuple(statement.expr for statement in static_statements),
                    static_values={},
                ),
                EvaluationContext(
                    t=np.array([0.0]),
                    species_series={},
                    initials={},
                    species_names=set(),
                    symtab=SymbolTable(),
                ),
            )
        except Exception as exc:
            name = next(
                (
                    str(statement.name)
                    for statement in static_statements
                    if int(getattr(exc, "line", 0) or 0) == int(statement.line)
                ),
                str(static_statements[0].name),
            )
            raise DSLError(
                f"Static observable expression for {name!r} is invalid: {exc}",
                line_number=statement_by_name[name].line,
                line_content=statement_by_name[name].line_text,
            ) from exc


def _compile_retained_algebra_program(
    algebra_lines: Sequence[Tuple[int, str]],
) -> tuple[AlgebraBlock, tuple[tuple[str, int, str], ...]]:
    """Parse all accepted Algebra declarations in one admission pass.

    The returned declaration descriptors are ``(kind, source_line, source)``
    and align one-for-one with the parsed block statements.  Parameter and
    observable views are partitioned from this retained block; no later layer
    needs to rebuild source text.
    """
    source_lines: list[str] = []
    descriptors: list[tuple[str, int, str]] = []
    for line_number, raw in algebra_lines:
        classification = classify_parameter_algebra_declaration(
            raw,
            line_number=line_number,
            allow_non_algebra_bare_assignment=False,
        )
        if classification.kind not in {"param", "let"}:
            continue
        match = _PARAM_STMT_RE.match(classification.code) if classification.kind == "param" else _LET_STMT_RE.match(classification.code)
        if match is None:
            raise DSLError(
                f"Malformed {classification.kind} declaration.",
                line_number=line_number,
                line_content=raw,
            )
        name = str(match.group(1))
        if name == "t":
            raise DSLError(
                "'t' is reserved for simulation time and cannot be declared as a parameter or observable.",
                line_number=line_number,
                line_content=raw,
            )
        expr_src, _unit = _split_unit_clause(str(match.group(2) or ""), line_number, raw)
        source_lines.append(f"let {name} = {expr_src}")
        descriptors.append((classification.kind, int(line_number), str(raw)))
    try:
        block = parse_algebra("\n".join(source_lines) + "\n")
    except Exception as exc:
        statement_index = int(getattr(exc, "line", 0) or 0) - 1
        if 0 <= statement_index < len(descriptors):
            _kind, line_number, line_content = descriptors[statement_index]
            raise DSLError(
                str(exc),
                line_number=line_number,
                line_content=line_content,
            ) from exc
        raise DSLError(str(exc)) from exc
    block = replace(
        block,
        lines=tuple(
            replace(
                statement,
                line=line_number,
                col=len(raw) - len(raw.lstrip()) + 1,
                line_text=raw,
                declared_unit=_split_unit_clause(strip_inline_comment(raw).split("=", 1)[1], line_number, raw)[1],
            )
            for statement, (_kind, line_number, raw) in zip(block.lines, descriptors)
        ),
    )
    return block, tuple(descriptors)


def _split_unit_clause(rhs: str, line_number: int, line_content: str) -> tuple[str, Unit | None]:
    parts = rhs.split(";")
    if len(parts) == 1:
        return rhs.strip(), None
    try:
        if len(parts) != 2 or "=" not in parts[1]:
            raise ValueError("Expected one trailing '; unit = ...' declaration.")
        key, value = parts[1].split("=", 1)
        if key.strip() != "unit":
            raise ValueError(f"Unknown parameter/observable annotation {key.strip()!r}; expected 'unit'.")
        return parts[0].strip(), parse_declared_unit(value)
    except ValueError as exc:
        raise DSLError(str(exc), line_number=line_number, line_content=line_content) from exc


def strip_inline_comment(line: str) -> str:
    """
    Strip inline comments using the same conservative heuristic as the GUI:
    remove the first '#' that is not inside [...]. (Algebra has no strings.)
    """
    if "#" not in line:
        return line
    bracket_depth = 0
    for i, ch in enumerate(line):
        if ch == "[":
            bracket_depth += 1
        elif ch == "]":
            bracket_depth = max(0, bracket_depth - 1)
        elif ch == "#" and bracket_depth == 0:
            return line[:i].rstrip()
    return line


def collect_algebra_section_lines(dsl_text: str) -> List[Tuple[int, str]]:
    """
    Return (line_number, raw_line) entries for algebra lines anywhere in the DSL text.
    """
    out: List[Tuple[int, str]] = []
    for ln, raw_with_eol in enumerate(dsl_text.splitlines(keepends=True), start=1):
        raw = raw_with_eol.rstrip("\r\n")
        stripped = raw.strip()
        classification = classify_parameter_algebra_declaration(raw, line_number=ln)
        if stripped and not stripped.startswith("#") and (
            classification.kind in {
                "param",
                "let",
                "invalid_step_key_identifier",
                "unsupported_bare_assignment",
            }
        ):
            out.append((ln, raw))
    return out


def _scalar_source_coordinate(
    line_number: int,
    line_content: str,
    *,
    name: str,
) -> DslSourceCoordinate:
    """Capture one admitted scalar RHS span from exact line content."""

    code = strip_inline_comment(line_content)
    equals_index = code.find("=")
    if equals_index < 0:
        raise DSLError(
            "Malformed param declaration.",
            line_number=line_number,
            line_content=line_content,
        )
    rhs = code[equals_index + 1 :]
    value_start = equals_index + 1 + (len(rhs) - len(rhs.lstrip()))
    value_end = len(code.split(";", 1)[0].rstrip())
    if value_end <= value_start:
        raise DSLError(
            "Param declaration requires a nonempty right-hand side.",
            line_number=line_number,
            line_content=line_content,
        )
    return DslSourceCoordinate(
        canonical_name=str(name),
        role="param",
        physical_line_index=int(line_number) - 1,
        value_start=value_start,
        value_end=value_end,
    )


def extract_parameter_assignments_from_algebra_lines(
    algebra_lines: Sequence[Tuple[int, str]],
    *,
    mechanism_namespace: MechanismParameterNamespace,
) -> List[ParameterAssignment]:
    assignments: List[ParameterAssignment] = []
    seen: Set[str] = set()
    mechanism_param_names = mechanism_namespace.flat_names()

    for line_no, raw in algebra_lines:
        original = raw.rstrip("\r\n")
        stripped = original.strip()
        if not stripped or stripped.startswith("#"):
            continue

        classification = classify_parameter_algebra_declaration(
            original,
            line_number=line_no,
            mechanism_namespace=mechanism_namespace,
        )
        code = classification.code
        if not code:
            continue

        if classification.kind == "param":
            raw_name = classification.raw_name
            resolution = mechanism_namespace.resolve(raw_name)
            if classification.invalid_protected_indexed_identifier:
                _raise_invalid_protected_indexed_identifier_error(
                    raw_name,
                    suggested_names=classification.invalid_protected_indexed_suggestions,
                    line_number=line_no,
                    line_content=original,
                )
            name = resolution.canonical_name or raw_name
            source_info = mechanism_namespace.source_coordinate_info(str(name))
            if source_info is not None and not source_coordinate_capabilities(
                source_info.role
            ).algebra_target:
                raise DSLError(
                    f"Source coordinate {raw_name!r} is symbolic-only and cannot be a Parameter Algebra target.",
                    suggestion="Use an authored continuous coordinate for Parameter Algebra.",
                    line_number=line_no,
                    line_content=original,
                )
            declared_unit = _split_unit_clause(_PARAM_STMT_RE.match(code).group(2), line_no, original)[1]
            if name in SymbolTable().protected_names() or name in SymbolTable().functions().keys():
                raise DSLError(
                    _reserved_name_message(name, "parameter"),
                    suggestion="Choose a name that does not shadow protected constants/functions (e.g., 'a', 'Ea', 'scale').",
                    line_number=line_no,
                    line_content=original,
                )
            m_mech = _MECH_PARAM_RE.match(raw_name)
            if m_mech and resolution.canonical_name is None and name not in mechanism_param_names:
                raise DSLError(
                    f"Unknown mechanism parameter {raw_name!r} in Algebra param statement",
                    suggestion="Use an existing mechanism parameter (e.g., k1, k2, kf1, kr1, Keq1) or define the parameter on a reaction line.",
                    examples=["A -> B; k=1.0", "param k1 = 4*k2"],
                    line_number=line_no,
                    line_content=original,
                )
            if name in seen:
                raise DSLError(
                    f"Duplicate derived parameter definition for {name!r}",
                    suggestion="Define each derived parameter only once.",
                    line_number=line_no,
                    line_content=original,
                )
            seen.add(name)
            assignments.append(
                ParameterAssignment(
                    name=name,
                    declared_unit=declared_unit,
                    line_number=line_no,
                    line_content=original,
                    source_coordinate=(
                        None
                        if source_info is not None
                        else _scalar_source_coordinate(
                            line_no,
                            original,
                            name=name,
                        )
                    ),
                )
            )
            continue

        if classification.kind == "invalid_step_key_identifier":
            _raise_protected_step_key_identifier_error(
                classification.raw_name,
                line_number=line_no,
                line_content=original,
            )

        if classification.kind == "let":
            target_raw = classification.raw_name
            resolution = mechanism_namespace.resolve(target_raw)
            target = resolution.canonical_name or target_raw
            if resolution.canonical_name is not None and target in mechanism_param_names:
                raise DSLError(
                    f"{target_raw!r} is a rate/equilibrium parameter; use 'param {target_raw} = ...' for parameter algebra",
                    suggestion=f"Replace this with: param {target_raw} = ...",
                    examples=[f"param {target_raw} = 4*k2"],
                    line_number=line_no,
                    line_content=original,
                )
            if is_protected_indexed_identifier(target_raw):
                _raise_protected_observable_identifier_error(
                    target_raw,
                    line_number=line_no,
                    line_content=original,
                )
            continue

        if classification.kind == "unsupported_bare_assignment":
            _raise_unsupported_bare_assignment_error(
                classification.raw_name,
                line_number=line_no,
                line_content=original,
            )

    return assignments


def extract_observable_names_from_algebra_lines(algebra_lines: Sequence[Tuple[int, str]]) -> Set[str]:
    out: Set[str] = set()
    for _line_no, raw in algebra_lines:
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        code = strip_inline_comment(stripped).strip()
        if not code:
            continue
        classification = classify_parameter_algebra_declaration(raw, line_number=_line_no)
        if classification.kind == "param":
            continue
        if classification.kind == "let":
            if is_protected_indexed_identifier(classification.raw_name):
                _raise_protected_observable_identifier_error(
                    classification.raw_name,
                    line_number=_line_no,
                    line_content=raw.rstrip("\n"),
                )
            if is_protected_step_key_identifier(classification.raw_name):
                _raise_protected_step_key_identifier_error(
                    classification.raw_name,
                    line_number=_line_no,
                    line_content=raw.rstrip("\n"),
                )
            symbols = SymbolTable()
            if (
                classification.raw_name in symbols.protected_names()
                or classification.raw_name in symbols.functions()
            ):
                # Evaluation would drop the LET and later references would
                # silently read the built-in constant or function instead.
                raise DSLError(
                    _reserved_name_message(classification.raw_name, "let value"),
                    suggestion="Choose a name that does not shadow protected constants/functions (e.g., 'ratio', 'rate_A').",
                    line_number=_line_no,
                    line_content=raw.rstrip("\n"),
                )
            out.add(classification.raw_name)
            continue
        if classification.kind == "invalid_step_key_identifier":
            _raise_protected_step_key_identifier_error(
                classification.raw_name,
                line_number=_line_no,
                line_content=raw.rstrip("\n"),
            )
        if classification.kind == "unsupported_bare_assignment":
            _raise_unsupported_bare_assignment_error(
                classification.raw_name,
                line_number=_line_no,
                line_content=raw.rstrip("\n"),
            )
    return out


def parse_parameter_algebra_spec_from_dsl_text(
    dsl_text: str,
    *,
    mechanism_namespace: MechanismParameterNamespace,
    scalar_input_names: Set[str] | None = None,
) -> ParameterAlgebraSpec:
    lines = collect_algebra_section_lines(dsl_text)
    assignments = extract_parameter_assignments_from_algebra_lines(lines, mechanism_namespace=mechanism_namespace)
    observables = extract_observable_names_from_algebra_lines(lines)
    for assignment in assignments:
        if assignment.name in observables:
            raise DSLError(
                f"Name {assignment.name!r} is defined as both a solver parameter (param) and an observable (let)",
                suggestion=(
                    f"Use only 'param {assignment.name} = ...' for solver parameters; "
                    f"use 'let {assignment.name}_obs = ...' for observables."
                ),
                line_number=assignment.line_number,
                line_content=assignment.line_content,
            )
    initial_spec = ParameterAlgebraSpec(
        param_statements=list(assignments),
        observable_names=set(observables),
        mechanism_namespace=mechanism_namespace,
        scalar_input_names=set(scalar_input_names or ()),
    )
    full_block, descriptors = _compile_retained_algebra_program(lines)
    parameter_indices = [
        index for index, (kind, _line_number, _line_content) in enumerate(descriptors)
        if kind == "param"
    ]
    observable_indices = [
        index for index, (kind, _line_number, _line_content) in enumerate(descriptors)
        if kind == "let"
    ]
    parameter_raw_block = AlgebraBlock(
        lines=tuple(full_block.lines[index] for index in parameter_indices),
        ast=tuple(full_block.ast[index] for index in parameter_indices),
        static_values={
            name: value
            for name, value in full_block.static_value_map().items()
            if name in {assignments[i].name for i in range(len(assignments))}
        },
    )
    if assignments:
        block = _compile_retained_parameter_block(
            assignments,
            mechanism_namespace=mechanism_namespace,
            scalar_input_names=set(scalar_input_names or ()),
            parsed=parameter_raw_block,
        )
        parameter_dependencies = _validate_retained_parameter_program(
            initial_spec,
            assignments=assignments,
            block=block,
        )
        retained_assignments = assignments
    else:
        # Keep an explicit empty parameter block so let-only Algebra still
        # carries one accepted, partitioned program through the pipeline.
        block = parameter_raw_block
        retained_assignments = []
        parameter_dependencies = {}

    observable_block = AlgebraBlock(
        lines=tuple(full_block.lines[index] for index in observable_indices),
        ast=tuple(full_block.ast[index] for index in observable_indices),
        static_values={
            name: value
            for name, value in full_block.static_value_map().items()
            if name in {observables_name for observables_name in observables}
        },
    )
    _validate_retained_observable_program(initial_spec, block=observable_block)
    return replace(
        initial_spec,
        param_statements=retained_assignments,
        parameter_block=block,
        observable_block=observable_block,
        parameter_dependencies=parameter_dependencies,
    )
