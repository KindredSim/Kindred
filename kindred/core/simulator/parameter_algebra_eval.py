"""Bind the admitted chemistry/Algebra order to current candidate values."""
from __future__ import annotations

from collections.abc import Mapping

from kindred.core.algebra.errors import AlgebraError
from kindred.core.algebra.evaluator import bind_scalar_expression_evaluator
from kindred.core.exceptions import ErrorContext, KineticRuntimeDomainError
from kindred.core.mechanism_metadata import MechanismMetadataView
from kindred.core.simulator.errors import DSLError
from kindred.core.algebra.parser import _iter_identifiers
from kindred.core.simulator.parameter_algebra_spec import ParameterAlgebraSpec
from kindred.core.simulator.parameter_coordinates import (
    build_parameter_target_map, read_mechanism_parameter_values, bind_chemistry_readout,
    _admit_coordinate_value,
)


def bind_parameter_evaluation(
    spec: ParameterAlgebraSpec,
    mechanism: object,
    base_values: Mapping[str, float] | None = None,
):
    """Bind once; evaluate requested dependency closures at the current temperature."""
    targets = build_parameter_target_map(mechanism)
    graph = spec.evaluation_dependencies
    statements = {statement.name: statement for statement in spec.parameter_block.lines}
    # A mechanism input's param line was seeded into its coordinate, so the
    # coordinate is read like any other authored leaf.
    inputs = {
        name for name in statements
        if spec.parameter_assignment_kind(name) == "mechanism input"
    }
    leaves = {
        name for name, deps in graph.items() if not deps and name not in statements
    } | set(spec.identity_checks) | inputs
    values = read_mechanism_parameter_values(
        mechanism, names=leaves, parameter_target_map=targets
    )
    metadata = mechanism.metadata
    independent = {
        name for name in spec.parameter_dependencies
        if spec.parameter_assignment_kind(name) == "independent scalar"
    }
    for name, value in dict(metadata.get("scalar_params") or {}).items():
        if name in independent:
            values[name] = float(value)
    values.update(base_values or {})
    values["T"] = float(MechanismMetadataView.from_metadata(metadata).temperature_K)
    expression = bind_scalar_expression_evaluator(spec.parameter_block, values)
    operations = {}
    for name in graph:
        if name in spec.identity_checks or name in inputs:
            continue
        if name in statements:
            if (
                spec.parameter_assignment_kind(name) == "independent scalar"
                and name in values
            ):
                continue
            operations[name] = lambda statement=statements[name]: expression(statement)
        elif name in spec.chemistry_dependencies:
            operations[name] = bind_chemistry_readout(mechanism, targets[name], values)

    def dependencies(name: str) -> tuple[str, ...]:
        result = set()

        def visit(current):
            deps = graph.get(current, ())
            if not deps:
                result.add(current)
            else:
                for dependency in deps:
                    visit(dependency)

        visit(name)
        return tuple(sorted(result))

    input_units = dict(spec.input_units)

    def calculate(name: str, operation, *, check: bool = False) -> None:
        try:
            value = float(operation())
            if name in targets and name in statements:
                _admit_coordinate_value(mechanism, name, value, targets[name], unit=input_units[name].symbol)
            if not check:
                values[name] = value
        except AlgebraError as exc:
            error = DSLError(
                str(exc),
                line_number=statements[name].line,
                line_content=statements[name].line_text,
            )
            error.details.update(
                assignment=name,
                value_dependencies=dependencies(name),
                parameter_names=dependencies(name),
            )
            raise error from exc
        except (ArithmeticError, ValueError) as exc:
            # The refusal is shown on the line that calculates the value: its
            # param line, else its reaction line.
            statement = statements.get(name)
            entry = targets[name][4] if statement is None else None
            raise KineticRuntimeDomainError(
                str(exc),
                parameter_names=dependencies(name),
                details={"assignment": name, "value_dependencies": dependencies(name)},
                context=(ErrorContext(line=statement.line, line_text=statement.line_text)
                         if statement is not None
                         else ErrorContext(line=entry["line"], line_text=entry["line_text"])),
            ) from exc

    dynamic = set(operations) & spec.temperature_dependent_names
    check_dependencies = {
        name: tuple(_iter_identifiers(statements[name].expr))
        for name in spec.identity_checks
    }
    dynamic_checks = {
        name for name, deps in check_dependencies.items()
        if set(deps) & spec.temperature_dependent_names
    }
    closures: dict[tuple[str, ...], tuple[str, ...]] = {}
    completed: set[str] = set()
    checked: set[str] = set()
    last_temperature = None

    def evaluate(temperature: float, requested_names: tuple[str, ...]) -> dict[str, float]:
        nonlocal last_temperature
        if temperature != last_temperature:
            completed.difference_update(dynamic)
            checked.difference_update(dynamic_checks)
            for name in dynamic:
                values.pop(name, None)
            values["T"] = float(temperature)
            last_temperature = temperature
        if requested_names not in closures:
            required: set[str] = set()

            def visit(name: str) -> None:
                if name in required:
                    return
                required.add(name)
                for dependency in graph.get(name, ()):
                    visit(dependency)

            for name in requested_names:
                visit(name)
            for deps in check_dependencies.values():
                for name in deps:
                    visit(name)
            closures[requested_names] = tuple(name for name in operations if name in required)
        for name in closures[requested_names]:
            if name in completed:
                continue
            calculate(name, operations[name])
            completed.add(name)
        for name in spec.identity_checks:
            if name in checked:
                continue
            calculate(name, lambda name=name: expression(statements[name]), check=True)
            checked.add(name)
        return {name: values[name] for name in requested_names}

    return evaluate
