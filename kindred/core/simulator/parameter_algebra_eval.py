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
    LetStatement,
    NumberNode,
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
    _build_mechanism_param_lookup,
    _raise_equilibrium_constant_alias_error,
    _resolve_mechanism_param_name,
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
class _CanonicalizedExpr:
    expr: ExprNode
    raw_to_canonical_identifiers: Mapping[str, str]


@dataclass(frozen=True)
class _CanonicalizedParameterBlock:
    block: AlgebraBlock
    raw_to_canonical_identifiers_by_assignment: Mapping[str, Mapping[str, str]]


def _merge_raw_identifier_maps(*maps: Mapping[str, str]) -> Dict[str, str]:
    merged: Dict[str, str] = {}
    for mapping in maps:
        for raw_name, canonical_name in mapping.items():
            existing = merged.get(raw_name)
            if existing is not None and existing != canonical_name:
                raise ValueError(
                    f"Conflicting canonical mechanism identifier mapping for {raw_name!r}: {existing!r} vs {canonical_name!r}"
                )
            merged[raw_name] = canonical_name
    return merged


def _canonicalize_mechanism_param_identifiers(
    expr: ExprNode,
    *,
    canonical_by_lower: Mapping[str, str],
    scalar_input_names: Set[str],
    assignment: ParameterAssignment,
) -> _CanonicalizedExpr:
    if isinstance(expr, NumberNode):
        return _CanonicalizedExpr(expr=expr, raw_to_canonical_identifiers={})
    if isinstance(expr, SpeciesRefNode):
        return _CanonicalizedExpr(expr=expr, raw_to_canonical_identifiers={})
    if isinstance(expr, IdentNode):
        # Exact-case scalar names keep their original binding.
        if expr.name in scalar_input_names:
            return _CanonicalizedExpr(expr=expr, raw_to_canonical_identifiers={})
        resolution = _resolve_mechanism_param_name(expr.name, canonical_by_lower=canonical_by_lower)
        if resolution.equilibrium_conflict_name is not None:
            _raise_equilibrium_constant_alias_error(
                expr.name,
                equilibrium_name=resolution.equilibrium_conflict_name,
                line_number=assignment.line_number,
                line_content=assignment.line_content,
            )
        if resolution.canonical_name is None:
            return _CanonicalizedExpr(expr=expr, raw_to_canonical_identifiers={})
        return _CanonicalizedExpr(
            expr=IdentNode(resolution.canonical_name),
            raw_to_canonical_identifiers={expr.name: resolution.canonical_name},
        )
    if isinstance(expr, UnaryNode):
        rhs = _canonicalize_mechanism_param_identifiers(
            expr.rhs,
            canonical_by_lower=canonical_by_lower,
            scalar_input_names=scalar_input_names,
            assignment=assignment,
        )
        return _CanonicalizedExpr(
            expr=UnaryNode(op=expr.op, rhs=rhs.expr),
            raw_to_canonical_identifiers=dict(rhs.raw_to_canonical_identifiers),
        )
    if isinstance(expr, BinaryNode):
        lhs = _canonicalize_mechanism_param_identifiers(
            expr.lhs,
            canonical_by_lower=canonical_by_lower,
            scalar_input_names=scalar_input_names,
            assignment=assignment,
        )
        rhs = _canonicalize_mechanism_param_identifiers(
            expr.rhs,
            canonical_by_lower=canonical_by_lower,
            scalar_input_names=scalar_input_names,
            assignment=assignment,
        )
        return _CanonicalizedExpr(
            expr=BinaryNode(op=expr.op, lhs=lhs.expr, rhs=rhs.expr),
            raw_to_canonical_identifiers=_merge_raw_identifier_maps(
                lhs.raw_to_canonical_identifiers,
                rhs.raw_to_canonical_identifiers,
            ),
        )
    if isinstance(expr, CallNode):
        canonical_args: List[ExprNode] = []
        raw_map: Dict[str, str] = {}
        for arg in expr.args:
            canonical_arg = _canonicalize_mechanism_param_identifiers(
                arg,
                canonical_by_lower=canonical_by_lower,
                scalar_input_names=scalar_input_names,
                assignment=assignment,
            )
            canonical_args.append(canonical_arg.expr)
            raw_map = _merge_raw_identifier_maps(raw_map, canonical_arg.raw_to_canonical_identifiers)
        return _CanonicalizedExpr(
            expr=CallNode(name=expr.name, args=tuple(canonical_args)),
            raw_to_canonical_identifiers=raw_map,
        )
    return _CanonicalizedExpr(expr=expr, raw_to_canonical_identifiers={})


def _parse_canonicalized_param_block(
    spec: ParameterAlgebraSpec,
    assignments: Sequence[ParameterAssignment],
) -> _CanonicalizedParameterBlock:
    parsed_block = _parse_param_block(assignments)
    canonical_by_lower = _build_mechanism_param_lookup(spec.mechanism_param_names)
    scalar_input_names = set(spec.scalar_input_names)
    lines: List[LetStatement] = []
    raw_to_canonical_identifiers_by_assignment: Dict[str, Mapping[str, str]] = {}

    for assignment, stmt in zip(assignments, parsed_block.lines):
        canonicalized_expr = _canonicalize_mechanism_param_identifiers(
            stmt.expr,
            canonical_by_lower=canonical_by_lower,
            scalar_input_names=scalar_input_names,
            assignment=assignment,
        )
        lines.append(
            LetStatement(
                name=stmt.name,
                expr=canonicalized_expr.expr,
                line=stmt.line,
                col=stmt.col,
                line_text=stmt.line_text,
            )
        )
        raw_to_canonical_identifiers_by_assignment[assignment.name] = dict(
            canonicalized_expr.raw_to_canonical_identifiers
        )

    return _CanonicalizedParameterBlock(
        block=AlgebraBlock(
            lines=lines,
            ast=[line.expr for line in lines],
            static_values=dict(parsed_block.static_values),
        ),
        raw_to_canonical_identifiers_by_assignment=dict(raw_to_canonical_identifiers_by_assignment),
    )


def _display_identifier_names(
    identifier_names: Iterable[str],
    *,
    raw_to_canonical_identifiers: Mapping[str, str],
) -> List[str]:
    raw_names_by_canonical: Dict[str, List[str]] = {}
    for raw_name, canonical_name in raw_to_canonical_identifiers.items():
        raw_names_by_canonical.setdefault(canonical_name, []).append(raw_name)

    ordered: List[str] = []
    for identifier_name in identifier_names:
        raw_names = sorted(set(raw_names_by_canonical.get(identifier_name, [])))
        if raw_names:
            ordered.extend(raw_names)
            continue
        ordered.append(identifier_name)
    return list(dict.fromkeys(ordered))


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
    raw_to_canonical_identifiers: Mapping[str, str],
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
            display_name = _display_identifier_names(
                [name],
                raw_to_canonical_identifiers=raw_to_canonical_identifiers,
            )[0]
            return _analysis(
                DSLError(
                    f"{display_name!r} is an observable (defined with 'let') and cannot be used in parameter algebra",
                    suggestion=f"Define it as a solver parameter instead: param {display_name} = <number>",
                    examples=[f"param {display_name} = 4", "param k2 = a*k1"],
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
        display_names = _display_identifier_names(
            referenced_reserved_scalar_names,
            raw_to_canonical_identifiers=raw_to_canonical_identifiers,
        )
        names = ", ".join(repr(name) for name in display_names)
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
        unknown_display_names = _display_identifier_names(
            sorted(unknown),
            raw_to_canonical_identifiers=raw_to_canonical_identifiers,
        )
        unknown_sorted = ", ".join(repr(name) for name in unknown_display_names)
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
        missing_display_names = _display_identifier_names(
            missing,
            raw_to_canonical_identifiers=raw_to_canonical_identifiers,
        )
        return _analysis(
            DSLError(
                f"Undefined parameter value(s) referenced in expression: {', '.join(missing_display_names)}",
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
        nonfinite_display_names = _display_identifier_names(
            nonfinite,
            raw_to_canonical_identifiers=raw_to_canonical_identifiers,
        )
        return _analysis(
            DSLError(
                f"Non-finite parameter value(s) referenced in expression: {', '.join(nonfinite_display_names)}",
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
    raw_to_canonical_identifiers_by_assignment: Mapping[str, Mapping[str, str]] | None = None,
    base_values: Mapping[str, float],
) -> Mapping[str, ParameterAlgebraAssignmentAnalysis]:
    namespace = spec.namespace_model()
    defined_names = {assignment.name for assignment in assignments}
    ast_map = dict(ast_by_name or {})
    raw_identifier_map = {
        name: dict(mapping)
        for name, mapping in (raw_to_canonical_identifiers_by_assignment or {}).items()
    }
    analyses: Dict[str, ParameterAlgebraAssignmentAnalysis] = {}

    for assignment in assignments:
        expr = ast_map.get(assignment.name)
        raw_to_canonical_identifiers = raw_identifier_map.get(assignment.name, {})
        if expr is None:
            try:
                parsed_block = _parse_canonicalized_param_block(spec, [assignment])
            except AlgebraError as err:
                analyses[assignment.name] = ParameterAlgebraAssignmentAnalysis(
                    identifier_names=frozenset(),
                    referenced_scalar_inputs=frozenset(),
                    solver_dependencies=frozenset(),
                    assignment_dependencies=frozenset(),
                    error=_map_algebra_error_to_assignment(err, order=[assignment]),
                )
                continue
            expr = parsed_block.block.lines[0].expr
            raw_to_canonical_identifiers = parsed_block.raw_to_canonical_identifiers_by_assignment.get(
                assignment.name,
                {},
            )
        analyses[assignment.name] = _build_assignment_analysis(
            namespace,
            assignment=assignment,
            expr=expr,
            raw_to_canonical_identifiers=raw_to_canonical_identifiers,
            defined_names=defined_names,
            base_values=base_values,
        )
    return dict(analyses)


def build_parameter_algebra_evaluation_model(
    spec: ParameterAlgebraSpec,
    *,
    assignments: Sequence[ParameterAssignment],
    ast_by_name: Mapping[str, ExprNode] | None = None,
    raw_to_canonical_identifiers_by_assignment: Mapping[str, Mapping[str, str]] | None = None,
    base_values: Mapping[str, float],
) -> ParameterAlgebraEvaluationModel:
    if ast_by_name is None:
        try:
            parsed_block = _parse_canonicalized_param_block(spec, assignments)
        except AlgebraError as err:
            raise _map_algebra_error_to_assignment(err, order=list(assignments)) from err
        ast_by_name = {stmt.name: stmt.expr for stmt in parsed_block.block.lines}
        raw_to_canonical_identifiers_by_assignment = parsed_block.raw_to_canonical_identifiers_by_assignment
    namespace = spec.namespace_model()
    analyses = analyze_parameter_algebra_assignments(
        spec,
        assignments=assignments,
        ast_by_name=ast_by_name,
        raw_to_canonical_identifiers_by_assignment=raw_to_canonical_identifiers_by_assignment,
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
        parsed_block = _parse_canonicalized_param_block(spec, assignments)
    except AlgebraError as err:
        raise _map_algebra_error_to_assignment(err, order=list(assignments)) from err

    ast_by_name: Dict[str, ExprNode] = {stmt.name: stmt.expr for stmt in parsed_block.block.lines}
    evaluation_model = build_parameter_algebra_evaluation_model(
        spec,
        assignments=assignments,
        ast_by_name=ast_by_name,
        raw_to_canonical_identifiers_by_assignment=parsed_block.raw_to_canonical_identifiers_by_assignment,
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
        block = _parse_canonicalized_param_block(spec, topo_assignments).block
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
        parsed_block = _parse_canonicalized_param_block(spec, statements)
    except AlgebraError as err:
        raise _map_algebra_error_to_assignment(err, order=statements) from err

    ast_by_name: Dict[str, ExprNode] = {stmt.name: stmt.expr for stmt in parsed_block.block.lines}
    evaluation_model = build_parameter_algebra_evaluation_model(
        spec,
        assignments=statements,
        ast_by_name=ast_by_name,
        raw_to_canonical_identifiers_by_assignment=parsed_block.raw_to_canonical_identifiers_by_assignment,
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
        block = _parse_canonicalized_param_block(spec, topo_assignments).block
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
