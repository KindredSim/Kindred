from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, List, Mapping, Sequence, Set

import numpy as np

from kindred.core.algebra.errors import AlgebraError
from kindred.core.algebra.evaluator import EvaluationContext, evaluate_block
from kindred.core.algebra.parser import (
    AlgebraBlock,
    BinaryNode,
    CallNode,
    ExprNode,
    IdentNode,
    SpeciesRefNode,
    UnaryNode,
    parse_algebra,
)
from kindred.core.algebra.symbols import SymbolTable
from kindred.core.simulator.errors import DSLError
from kindred.core.simulator.parameter_algebra_spec import (
    ParameterAlgebraNamespace,
    ParameterAlgebraSpec,
    ParameterAssignment,
    mechanism_parameter_name_pattern,
)


def _iter_identifiers(expr: ExprNode) -> Iterable[str]:
    if isinstance(expr, IdentNode):
        yield expr.name
        return
    if isinstance(expr, SpeciesRefNode):
        return
    if isinstance(expr, UnaryNode):
        yield from _iter_identifiers(expr.rhs)
        return
    if isinstance(expr, BinaryNode):
        yield from _iter_identifiers(expr.lhs)
        yield from _iter_identifiers(expr.rhs)
        return
    if isinstance(expr, CallNode):
        for arg in expr.args:
            yield from _iter_identifiers(arg)
        return


def _toposort(names: Sequence[str], deps: Dict[str, Set[str]]) -> List[str]:
    order: List[str] = []
    visiting: Set[str] = set()
    visited: Set[str] = set()
    stack: List[str] = []

    def visit(name: str) -> None:
        if name in visited:
            return
        if name in visiting:
            if name in stack:
                j = stack.index(name)
                cycle = stack[j:] + [name]
            else:
                cycle = [name, name]
            raise DSLError(
                "Parameter algebra has a dependency cycle: " + " -> ".join(cycle),
                suggestion="Remove the cycle by rewriting the param definitions so they form an acyclic dependency graph.",
            )
        visiting.add(name)
        stack.append(name)
        for dep in sorted(deps.get(name, set())):
            visit(dep)
        stack.pop()
        visiting.remove(name)
        visited.add(name)
        order.append(name)

    for name in names:
        visit(name)
    return order


def _parse_param_block(assignments: Sequence[ParameterAssignment]) -> AlgebraBlock:
    lines = ["# Algebra"]
    for assignment in assignments:
        lines.append(f"let {assignment.name} = {assignment.expr_src}")
    block_src = "\n".join(lines) + "\n"
    return parse_algebra(block_src)


def _map_algebra_error_to_assignment(
    err: AlgebraError,
    *,
    order: Sequence[ParameterAssignment],
) -> DSLError:
    stmt_idx = int(getattr(err, "line", 0)) - 2
    if 0 <= stmt_idx < len(order):
        assignment = order[stmt_idx]
        return DSLError(
            str(err),
            line_number=assignment.line_number,
            line_content=assignment.line_content,
        )
    return DSLError(str(err))


@dataclass(frozen=True)
class ParameterAlgebraEvaluationModel:
    namespace: ParameterAlgebraNamespace
    identifier_names_by_assignment: Mapping[str, frozenset[str]]
    referenced_scalar_inputs: frozenset[str]
    solver_dependencies_by_assignment: Mapping[str, frozenset[str]]
    assignment_dependencies_by_assignment: Mapping[str, frozenset[str]]

    def registerable_input_names(self) -> Set[str]:
        return self.namespace.solver_param_names() | set(self.referenced_scalar_inputs)


@dataclass(frozen=True)
class ParameterAlgebraAssignmentAnalysis:
    identifier_names: frozenset[str]
    referenced_scalar_inputs: frozenset[str]
    solver_dependencies: frozenset[str]
    assignment_dependencies: frozenset[str]
    error: DSLError | None


def _build_assignment_analysis(
    namespace: ParameterAlgebraNamespace,
    *,
    assignment: ParameterAssignment,
    expr: ExprNode,
    defined_names: Set[str],
    base_values: Mapping[str, float],
) -> ParameterAlgebraAssignmentAnalysis:
    solver_params = namespace.solver_param_names()
    allowed_constants = set(namespace.protected_symbol_names) - {"T"}
    idents = set(_iter_identifiers(expr))
    referenced_scalar_inputs = frozenset(name for name in idents if name in namespace.scalar_input_names)
    solver_dependencies = frozenset(name for name in idents if name in solver_params and name != assignment.name)
    assignment_dependencies = frozenset(name for name in idents if name in defined_names and name != assignment.name)

    def _analysis(error: DSLError | None = None) -> ParameterAlgebraAssignmentAnalysis:
        return ParameterAlgebraAssignmentAnalysis(
            identifier_names=frozenset(idents),
            referenced_scalar_inputs=referenced_scalar_inputs,
            solver_dependencies=solver_dependencies,
            assignment_dependencies=assignment_dependencies,
            error=error,
        )

    if expr.has_species_ref():
        return _analysis(
            DSLError(
                "Parameter expressions cannot reference concentrations like [A], [A]_0, or [A](T0)",
                suggestion="Use only numeric literals, math functions, and other parameters (k1, k2, ...).",
                line_number=assignment.line_number,
                line_content=assignment.line_content,
            )
        )
    if expr.has_time_ref():
        return _analysis(
            DSLError(
                "Parameter expressions cannot reference baseline/time-dependent terms (T0)",
                suggestion="Use only time-independent expressions for parameter algebra.",
                line_number=assignment.line_number,
                line_content=assignment.line_content,
            )
        )
    if "T" in idents:
        return _analysis(
            DSLError(
                "Parameter expressions cannot reference temperature 'T' (keep parameter algebra static)",
                suggestion="If you need temperature-dependent rates, use Arrhenius/Eyring reaction models instead.",
                line_number=assignment.line_number,
                line_content=assignment.line_content,
            )
        )

    for name in sorted(idents):
        if name in namespace.observable_names:
            return _analysis(
                DSLError(
                    f"{name!r} is an observable (defined with 'let') and cannot be used in parameter algebra",
                    suggestion=f"Define it as a solver parameter instead: param {name} = <number>",
                    examples=[f"param {name} = 4", "param k2 = a*k1"],
                    line_number=assignment.line_number,
                    line_content=assignment.line_content,
                )
            )

    referenced_reserved_scalar_names = sorted(
        name
        for name in idents
        if name in namespace.scalar_input_names and name in namespace.reserved_identifier_names()
    )
    if referenced_reserved_scalar_names:
        names = ", ".join(repr(name) for name in referenced_reserved_scalar_names)
        return _analysis(
            DSLError(
                f"Scalar input name(s) {names} shadow protected algebra names and cannot be referenced in parameter expressions",
                suggestion="Rename the top-level scalar assignment so it does not shadow a protected constant or builtin function.",
                line_number=assignment.line_number,
                line_content=assignment.line_content,
            )
        )

    unknown: Set[str] = set()
    for name in idents:
        if name in defined_names:
            continue
        if name in solver_params:
            continue
        if name in namespace.scalar_input_names:
            continue
        if name in allowed_constants:
            continue
        unknown.add(name)
    if unknown:
        unknown_sorted = ", ".join(sorted(repr(x) for x in unknown))
        return _analysis(
            DSLError(
                f"Unknown name(s) in parameter expression: {unknown_sorted}",
                suggestion="Parameter expressions may only reference other parameters (k1, k2, ...) and supported math/constants.",
                line_number=assignment.line_number,
                line_content=assignment.line_content,
            )
        )

    referenced_input_names = sorted(
        name
        for name in idents
        if (name in solver_params or name in namespace.scalar_input_names) and name not in defined_names
    )
    missing = [name for name in referenced_input_names if name not in base_values]
    if missing:
        return _analysis(
            DSLError(
                f"Undefined parameter value(s) referenced in expression: {', '.join(missing)}",
                suggestion="Ensure those parameters exist in the mechanism and have numeric values.",
                line_number=assignment.line_number,
                line_content=assignment.line_content,
            )
        )

    nonfinite: list[str] = []
    for name in referenced_input_names:
        try:
            value = float(base_values[name])
        except (TypeError, ValueError):
            nonfinite.append(name)
            continue
        if not math.isfinite(value):
            nonfinite.append(name)
    if nonfinite:
        return _analysis(
            DSLError(
                f"Non-finite parameter value(s) referenced in expression: {', '.join(nonfinite)}",
                suggestion="Set those referenced parameters to finite numeric values before using them in parameter algebra.",
                line_number=assignment.line_number,
                line_content=assignment.line_content,
            )
        )

    return _analysis()


def analyze_parameter_algebra_assignments(
    spec: ParameterAlgebraSpec,
    *,
    assignments: Sequence[ParameterAssignment],
    ast_by_name: Mapping[str, ExprNode] | None = None,
    base_values: Mapping[str, float],
) -> Mapping[str, ParameterAlgebraAssignmentAnalysis]:
    namespace = spec.namespace_model()
    defined_names = {assignment.name for assignment in assignments}
    ast_map = dict(ast_by_name or {})
    analyses: Dict[str, ParameterAlgebraAssignmentAnalysis] = {}

    for assignment in assignments:
        expr = ast_map.get(assignment.name)
        if expr is None:
            try:
                parsed_block = _parse_param_block([assignment])
            except AlgebraError as err:
                analyses[assignment.name] = ParameterAlgebraAssignmentAnalysis(
                    identifier_names=frozenset(),
                    referenced_scalar_inputs=frozenset(),
                    solver_dependencies=frozenset(),
                    assignment_dependencies=frozenset(),
                    error=_map_algebra_error_to_assignment(err, order=[assignment]),
                )
                continue
            expr = parsed_block.lines[0].expr
        analyses[assignment.name] = _build_assignment_analysis(
            namespace,
            assignment=assignment,
            expr=expr,
            defined_names=defined_names,
            base_values=base_values,
        )
    return dict(analyses)


def build_parameter_algebra_evaluation_model(
    spec: ParameterAlgebraSpec,
    *,
    assignments: Sequence[ParameterAssignment],
    ast_by_name: Mapping[str, ExprNode] | None = None,
    base_values: Mapping[str, float],
) -> ParameterAlgebraEvaluationModel:
    if ast_by_name is None:
        try:
            parsed_block = _parse_param_block(assignments)
        except AlgebraError as err:
            raise _map_algebra_error_to_assignment(err, order=list(assignments)) from err
        ast_by_name = {stmt.name: stmt.expr for stmt in parsed_block.lines}
    namespace = spec.namespace_model()
    analyses = analyze_parameter_algebra_assignments(
        spec,
        assignments=assignments,
        ast_by_name=ast_by_name,
        base_values=base_values,
    )

    identifier_names_by_assignment: Dict[str, frozenset[str]] = {}
    solver_dependencies_by_assignment: Dict[str, frozenset[str]] = {}
    assignment_dependencies_by_assignment: Dict[str, frozenset[str]] = {}
    referenced_scalar_inputs: Set[str] = set()

    for assignment in assignments:
        analysis = analyses[assignment.name]
        if analysis.error is not None:
            raise analysis.error
        identifier_names_by_assignment[assignment.name] = analysis.identifier_names
        solver_dependencies_by_assignment[assignment.name] = analysis.solver_dependencies
        assignment_dependencies_by_assignment[assignment.name] = analysis.assignment_dependencies
        referenced_scalar_inputs.update(analysis.referenced_scalar_inputs)

    return ParameterAlgebraEvaluationModel(
        namespace=namespace,
        identifier_names_by_assignment=dict(identifier_names_by_assignment),
        referenced_scalar_inputs=frozenset(referenced_scalar_inputs),
        solver_dependencies_by_assignment=dict(solver_dependencies_by_assignment),
        assignment_dependencies_by_assignment=dict(assignment_dependencies_by_assignment),
    )


def _build_symbol_table(
    evaluation_model: ParameterAlgebraEvaluationModel,
    *,
    base_values: Mapping[str, float],
) -> SymbolTable:
    symtab = SymbolTable()
    for name, value in base_values.items():
        if name not in evaluation_model.registerable_input_names():
            continue
        try:
            symtab.define_user(name, float(value), species_names=set())
        except ValueError as exc:
            raise DSLError(f"Invalid parameter algebra input name {name!r}: {exc}") from exc
    return symtab


def evaluate_parameter_algebra(
    spec: ParameterAlgebraSpec,
    *,
    base_values: Dict[str, float],
) -> Dict[str, float]:
    assignments = list(spec.param_statements)
    if not assignments:
        return {}

    try:
        parsed_block = _parse_param_block(assignments)
    except AlgebraError as err:
        raise _map_algebra_error_to_assignment(err, order=list(assignments)) from err

    ast_by_name: Dict[str, ExprNode] = {stmt.name: stmt.expr for stmt in parsed_block.lines}
    evaluation_model = build_parameter_algebra_evaluation_model(
        spec,
        assignments=assignments,
        ast_by_name=ast_by_name,
        base_values=base_values,
    )

    scalar_defaults: Dict[str, float] = {}
    derived_statements: List[ParameterAssignment] = []
    depends_on_map: Dict[str, Set[str]] = {}
    mech_param_re = mechanism_parameter_name_pattern()

    for assignment in assignments:
        depends_on = set(evaluation_model.solver_dependencies_by_assignment.get(assignment.name, frozenset()))
        depends_on_map[assignment.name] = set(depends_on)

        is_mech_target = mech_param_re.match(assignment.name) is not None
        if not is_mech_target and not depends_on:
            try:
                tmp_spec = ParameterAlgebraSpec(
                    param_statements=[assignment],
                    observable_names=set(spec.observable_names),
                    mechanism_param_names=set(spec.mechanism_param_names),
                    scalar_input_names=set(spec.scalar_input_names),
                )
                val_map = evaluate_parameter_algebra_in_context(
                    tmp_spec,
                    base_values=dict(base_values),
                    enforce_defaults=False,
                )
                scalar_defaults[assignment.name] = float(val_map[assignment.name])
            except DSLError:
                raise
            except Exception as exc:
                raise DSLError(
                    f"Failed to evaluate default value for parameter {assignment.name!r}: {exc}",
                    line_number=assignment.line_number,
                    line_content=assignment.line_content,
                ) from exc
        else:
            derived_statements.append(assignment)

    for name, default_val in scalar_defaults.items():
        if name not in base_values:
            base_values[name] = float(default_val)

    if not derived_statements:
        return {}

    derived_set = {assignment.name for assignment in derived_statements}
    deps: Dict[str, Set[str]] = {}
    for assignment in derived_statements:
        deps[assignment.name] = {
            name for name in depends_on_map.get(assignment.name, set()) if name in derived_set
        }

    topo_names = _toposort(sorted(derived_set), deps)
    topo_assignments = [next(assignment for assignment in derived_statements if assignment.name == name) for name in topo_names]

    try:
        block = _parse_param_block(topo_assignments)
    except AlgebraError as err:
        raise _map_algebra_error_to_assignment(err, order=topo_assignments) from err

    symtab = _build_symbol_table(evaluation_model, base_values=base_values)

    ctx = EvaluationContext(
        t=np.array([0.0]),
        species_series={},
        initials={},
        species_names=set(),
        symtab=symtab,
        baseline=None,
    )

    try:
        series, scalars = evaluate_block(block, ctx)
    except AlgebraError as err:
        raise _map_algebra_error_to_assignment(err, order=topo_assignments) from err

    out: Dict[str, float] = {}
    for name in topo_names:
        if name in scalars:
            value = float(scalars[name])
        else:
            arr = series.get(name)
            if arr is None or arr.size < 1:
                raise DSLError(f"Failed to evaluate derived parameter {name!r}")
            value = float(arr[0])
        if not math.isfinite(value):
            raise DSLError(f"Derived parameter {name!r} evaluated to a non-finite value: {value!r}")
        out[name] = value
    return out


def evaluate_parameter_algebra_in_context(
    spec: ParameterAlgebraSpec,
    *,
    base_values: Dict[str, float],
    enforce_defaults: bool,
) -> Dict[str, float]:
    statements = list(spec.param_statements)
    if not statements:
        return {}

    try:
        parsed_block = _parse_param_block(statements)
    except AlgebraError as err:
        raise _map_algebra_error_to_assignment(err, order=statements) from err

    ast_by_name: Dict[str, ExprNode] = {stmt.name: stmt.expr for stmt in parsed_block.lines}
    evaluation_model = build_parameter_algebra_evaluation_model(
        spec,
        assignments=statements,
        ast_by_name=ast_by_name,
        base_values=base_values,
    )

    deps: Dict[str, Set[str]] = {}
    for assignment in statements:
        deps[assignment.name] = set(
            evaluation_model.assignment_dependencies_by_assignment.get(assignment.name, frozenset())
        )

    topo = _toposort(sorted(evaluation_model.namespace.param_assignment_names), deps)
    topo_assignments = [next(assignment for assignment in statements if assignment.name == name) for name in topo]
    try:
        block = _parse_param_block(topo_assignments)
    except AlgebraError as err:
        raise _map_algebra_error_to_assignment(err, order=topo_assignments) from err

    symtab = _build_symbol_table(evaluation_model, base_values=base_values)
    ctx = EvaluationContext(
        t=np.array([0.0]),
        species_series={},
        initials={},
        species_names=set(),
        symtab=symtab,
        baseline=None,
    )
    try:
        series, scalars = evaluate_block(block, ctx)
    except AlgebraError as err:
        raise _map_algebra_error_to_assignment(err, order=topo_assignments) from err

    out: Dict[str, float] = {}
    for name in topo:
        if name in scalars:
            value = float(scalars[name])
        else:
            arr = series.get(name)
            if arr is None or arr.size < 1:
                raise DSLError(f"Failed to evaluate derived parameter {name!r}")
            value = float(arr[0])
        if not math.isfinite(value):
            raise DSLError(f"Derived parameter {name!r} evaluated to a non-finite value: {value!r}")
        out[name] = value
    return out
