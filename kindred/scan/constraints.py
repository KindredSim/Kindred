"""Constructive affine scan constraints over retained authored Algebra."""
from __future__ import annotations

from kindred.core._bounded_rational import literal_fraction

from dataclasses import dataclass, replace
from fractions import Fraction as F
import math
from itertools import product
from typing import Callable

from kindred.core.algebra.parser import (
    AlgebraBlock, NumberNode, IdentNode, SpeciesRefNode, UnaryNode, BinaryNode, CallNode,
)
from kindred.core.algebra.symbols import POWER_DOMAIN, SymbolTable, BUILTIN_FUNCTIONS
from kindred.core.algebra.evaluator import EvaluationContext, evaluate_block
from kindred.core.simulation_parameters import enumerate_simulation_parameters
from kindred.core.quantity_units import conversion_scale, convert_value, presentation_unit
import numpy as np
from kindred.scan.config import START, END, InputRange, ScanConfig, initial_key, input_label, parameter_key
from kindred.scan.intervals import Interval, normalize, representable, sample


class InputGenerationError(ValueError):
    def __init__(self, message, values):
        super().__init__(message)
        self.values = dict(values)


@dataclass(frozen=True)
class Affine:
    terms: dict[str, F]
    constant: F = F(0)

    def scale(self, factor: F) -> Affine:
        return Affine({k: v * factor for k, v in self.terms.items() if v * factor}, self.constant * factor)

    def add(self, other: Affine) -> Affine:
        terms = dict(self.terms)
        for key, value in other.terms.items():
            terms[key] = terms.get(key, F(0)) + value
        return Affine({k: v for k, v in terms.items() if v}, self.constant + other.constant)

    def substitute(self, replacements: dict[str, Affine]) -> Affine:
        result = Affine({}, self.constant)
        for key, value in self.terms.items():
            result = result.add(replacements.get(key, Affine({key: F(1)})).scale(value))
        return result

    def evaluate(self, values: dict[str, float]) -> float:
        # A dependent assignment, deliberately not a reconstructed-equality test.
        result = float(self.constant)
        for key, coefficient in self.terms.items():
            result += float(coefficient) * values[key]
        return result


@dataclass(frozen=True)
class Inequality:
    expression: Affine  # expression <= 0 (or < 0)
    strict: bool = False


def scientific_constraints(spec, constraints):
    statements = {s.name: s for s in spec.observable_block.lines}
    result = []
    for condition in constraints:
        statement = statements[condition.observable]
        unit = statement.expr.quantity_unit
        shown = presentation_unit(unit, statement.declared_unit)
        result.append(condition if unit is None else replace(condition,
            lower=convert_value(condition.lower, shown, unit),
            upper=convert_value(condition.upper, shown, unit)))
    return tuple(result)


_AFFINE_FORMS = ("Constraints accept sums and differences of inputs, each multiplied by a fixed number or "
                 "divided by a nonzero one, without comparisons or logic.")

def compile_constraints(prepared, config: ScanConfig, fixed: dict[str, float]):
    spec = prepared.parameter_algebra_spec_for_execution()
    statements = {}
    for block in (spec.parameter_block, spec.observable_block):
        if block:
            statements.update({s.name: s.expr for s in block.lines})
    available = {parameter_key(p.name) for p in enumerate_simulation_parameters(prepared)}
    available.update(initial_key(s) for s in prepared.analysis.species_names)
    inputs, quantities = dict(spec.input_units), dict(spec.quantity_units)
    symbols = SymbolTable()
    memo: dict[str, Affine] = {}

    def coordinate(key: str) -> Affine:
        if key in fixed:
            return Affine({}, F(fixed[key]))
        return Affine({key: F(1)})

    def named(name: str) -> Affine:
        if name in {"t", "T"}:
            raise ValueError(f"{name} is an execution-dependent symbol.")
        key = parameter_key(name)
        if key in available:
            scale = conversion_scale(inputs[name], quantities[name])
            factor = scale.rational()
            return coordinate(key).scale(factor if factor is not None else F(scale.number()))
        if name in memo:
            return memo[name]
        if name in statements:
            memo[name] = lower(statements[name])
            return memo[name]
        if symbols.has(name):
            return Affine({}, F(symbols.get(name)))
        raise ValueError(f"{name} is not a supported independent input or input-only definition.")

    def lower(node) -> Affine:
        value = unscaled(node)
        factor = node.unit_scale.rational()
        return value.scale(factor if factor is not None else F(node.unit_factor))

    def unscaled(node) -> Affine:
        if isinstance(node, NumberNode):
            return Affine({}, literal_fraction(node.literal if node.literal is not None else str(node.value)))
        if isinstance(node, SpeciesRefNode):
            if node.kind != "init":
                raise ValueError(f"[{node.name}] depends on simulated concentrations. Use an input-only quantity.")
            return coordinate(initial_key(node.name))
        if isinstance(node, IdentNode):
            return named(node.name)
        if isinstance(node, UnaryNode):
            if node.op not in {"+", "-"}:
                raise ValueError(_AFFINE_FORMS)
            return lower(node.rhs).scale(F(-1 if node.op == "-" else 1))
        if isinstance(node, BinaryNode):
            left, right = lower(node.lhs), lower(node.rhs)
            if node.op == "+":
                return left.add(right)
            if node.op == "-":
                return left.add(right.scale(F(-1)))
            if node.op == "*" and not (left.terms and right.terms):
                return left.scale(right.constant) if not right.terms else right.scale(left.constant)
            if node.op == "/" and not right.terms and right.constant:
                return left.scale(1 / right.constant)
            if node.op in {"^", "**"} and not left.terms and not right.terms:
                # As Kindred evaluates a power, never into complex numbers.
                try:
                    return Affine({}, F(math.pow(float(left.constant), float(right.constant))))
                except ValueError:
                    raise ValueError("^ " + POWER_DOMAIN) from None
            raise ValueError(_AFFINE_FORMS)
        if isinstance(node, CallNode):
            if node.name == "ifelse":
                # Only the selected value is calculated, as in a Run.
                condition = lower(node.args[0])
                if condition.terms:
                    raise ValueError("ifelse(...) reads a scanned input in its condition; constraints accept a fixed condition.")
                return lower(node.args[1] if condition.constant else node.args[2])
            arguments = [lower(arg) for arg in node.args]
            if any(arg.terms for arg in arguments):
                raise ValueError(f"{node.name}(...) reads a scanned input; constraints accept functions only of fixed values.")
            return Affine({}, F(BUILTIN_FUNCTIONS[node.name](*(float(a.constant) for a in arguments))))

    equations, dependents, inequalities, conditional = [], [], [], []
    for condition in scientific_constraints(spec, config.constraints):
        if condition.dependent not in config.ranges:
            raise ValueError(f"Select a scanned independent input for {condition.observable}.")
        try:
            expression = lower(statements[condition.observable])
        except (ValueError, KeyError, OverflowError, ZeroDivisionError) as exc:
            raise ValueError(f"{condition.observable}: {exc}") from exc
        if condition.dependent not in expression.terms:
            raise ValueError(f"{condition.observable} does not depend on the selected input.")
        if condition.condition == "equals":
            equations.append(expression.add(Affine({}, -F(condition.lower))))
            if condition.dependent not in dependents:
                dependents.append(condition.dependent)
        else:
            conditional.append((condition, expression))
            inequalities.extend((
                Inequality(expression.add(Affine({}, -F(condition.upper))), not condition.include_upper),
                Inequality(expression.scale(F(-1)).add(Affine({}, F(condition.lower))), not condition.include_lower),
            ))
    derived = solve_equalities(equations, dependents)
    precedence = set()
    for condition, expression in conditional:
        if condition.dependent in derived:
            raise ValueError(f"{condition.observable}: {input_label(condition.dependent)} is already calculated by an equality. Select a free input to sample conditionally.")
        for key in expression.substitute(derived).terms:
            if key != condition.dependent:
                precedence.add((key, condition.dependent))
    # Execution bounds are ordinary request semantics, not chemical equations.
    time_order = Affine({START: F(1), END: F(-1)}).substitute(
        {key: Affine({}, F(value)) for key, value in fixed.items()})
    inequalities.append(Inequality(time_order, True))
    # So is each flux and hold ending after it starts, when params place its ends.
    schedule = prepared.parameterized_intervention_schedule_for_binding()
    for item in () if schedule is None else schedule.intervals:
        try:
            start, end = (Affine({}, F(getattr(item, field))) if getattr(item, f"{field}_param") is None
                          else named(getattr(item, f"{field}_param")) for field in ("start_time", "end_time"))
        except (ValueError, KeyError, OverflowError, ZeroDivisionError):
            # Not affine in the scanned inputs: the run refuses a reversed draw.
            continue
        inequalities.append(Inequality(start.add(end.scale(F(-1))), True))
    return derived, tuple(Inequality(row.expression.substitute(derived), row.strict) for row in inequalities), precedence


def conditional_order(keys, precedence):
    """Honor all conditional selections, retaining stable ties by input order."""
    remaining = list(keys)
    order = []
    while remaining:
        ready = next((key for key in remaining
                      if not any(target == key and source in remaining for source, target in precedence)), None)
        if ready is None:
            raise ValueError("Conditional input selections form a cycle: " + ", ".join(remaining))
        remaining.remove(ready)
        order.append(ready)
    return tuple(order)


class IntervalChecks:
    """Use retained Algebra for the numerical admission of interval quantities.

    Equalities are assignments and deliberately do not enter this check.
    """

    def __init__(self, prepared, config, first_keys=None):
        conditions = tuple(c for c in config.constraints if c.condition == "range")
        self.blocks = {}
        if not conditions:
            return
        spec = prepared.parameter_algebra_spec_for_execution()
        conditions = scientific_constraints(spec, conditions)
        self.input_units, self.quantity_units = dict(spec.input_units), dict(spec.quantity_units)
        statements = {s.name: s for block in (spec.parameter_block, spec.observable_block) if block for s in block.lines}
        independent = {p.name for p in enumerate_simulation_parameters(prepared)}
        dependencies = {}

        def visit(node, needed, inputs):
            if isinstance(node, IdentNode):
                if node.name in independent:
                    inputs.add(parameter_key(node.name))
                elif node.name in statements:
                    names, keys = collect(node.name)
                    needed.update(names)
                    inputs.update(keys)
            elif isinstance(node, SpeciesRefNode):
                inputs.add(initial_key(node.name))
            elif isinstance(node, UnaryNode):
                visit(node.rhs, needed, inputs)
            elif isinstance(node, BinaryNode):
                visit(node.lhs, needed, inputs)
                visit(node.rhs, needed, inputs)
            elif isinstance(node, CallNode):
                for argument in node.args:
                    visit(argument, needed, inputs)

        def collect(name):
            if name not in dependencies:
                needed, inputs = {name}, set()
                visit(statements[name].expr, needed, inputs)
                dependencies[name] = needed, inputs
            return dependencies[name]

        groups = {True: ([], set()), False: ([], set())}
        for condition in conditions:
            needed, inputs = collect(condition.observable)
            # Retained dependencies include terms cancelled by affine lowering.
            early = first_keys is not None and inputs <= first_keys
            selected, names = groups[early]
            selected.append(condition)
            names.update(needed)
        for early, (selected, names) in groups.items():
            if selected:
                lines = tuple(statement for name, statement in statements.items() if name in names)
                self.blocks[early] = AlgebraBlock(lines), tuple(selected)

    def admit(self, values, *, first_group=False):
        if first_group not in self.blocks:
            return
        block, conditions = self.blocks[first_group]
        context = EvaluationContext(t=np.array([0.0]), species_series={},
                                    initials={k.removeprefix("initial:"): v for k, v in values.items() if k.startswith("initial:")},
                                    symtab=SymbolTable(),
                                    symbol_values={k.removeprefix("parameter:"): convert_value(v, self.input_units[k.removeprefix("parameter:")], self.quantity_units[k.removeprefix("parameter:")]) for k, v in values.items() if k.startswith("parameter:")})
        _, scalars = evaluate_block(block, context)
        for condition in conditions:
            value = scalars[condition.observable]
            if not Interval(condition.lower, condition.upper, condition.include_lower, condition.include_upper).contains(value):
                raise ValueError(f"{condition.observable}: numerical evaluation is outside the configured interval.")


def solve_equalities(equations: list[Affine], dependents: list[str]) -> dict[str, Affine]:
    rows = list(equations)
    derived = {}
    for index, key in enumerate(dependents):
        pivot = next((j for j in range(index, len(rows)) if rows[j].terms.get(key)), None)
        if pivot is None:
            raise ValueError("The selected calculated inputs do not have independent determining equations.")
        rows[index], rows[pivot] = rows[pivot], rows[index]
        rows[index] = rows[index].scale(1 / rows[index].terms[key])
        for j in range(len(rows)):
            if j != index and rows[j].terms.get(key):
                rows[j] = rows[j].add(rows[index].scale(-rows[j].terms[key]))
    for row in rows[len(dependents):]:
        if row.terms:
            raise ValueError("These equalities require additional calculated inputs.")
        if row.constant:
            raise ValueError("The equality constraints are inconsistent.")
    for index, key in enumerate(dependents):
        row = rows[index]
        derived[key] = Affine({k: -v for k, v in row.terms.items() if k != key}, -row.constant)
    return derived


def simplify(rows: tuple[Inequality, ...]) -> tuple[Inequality, ...] | None:
    unique: dict[tuple, Inequality] = {}
    for row in rows:
        expr = row.expression
        if not expr.terms:
            if expr.constant > 0 or expr.constant == 0 and row.strict:
                return None
            continue
        scale = abs(next(iter(sorted(expr.terms.items())))[1])
        normalized = expr.scale(1 / scale)
        key = (tuple(sorted(normalized.terms.items())), normalized.constant)
        prior = unique.get(key)
        unique[key] = Inequality(normalized, row.strict or bool(prior and prior.strict))
    return tuple(unique.values())


def eliminate(rows: tuple[Inequality, ...], key: str, check: Callable[[], None]):
    positive, negative, neutral = [], [], []
    for row in rows:
        coefficient = row.expression.terms.get(key, F(0))
        (positive if coefficient > 0 else negative if coefficient < 0 else neutral).append(row)
    for upper in positive:
        check()
        for lower in negative:
            combined = upper.expression.scale(1 / upper.expression.terms[key]).add(
                lower.expression.scale(-1 / lower.expression.terms[key]))
            neutral.append(Inequality(combined, upper.strict or lower.strict))
    return simplify(tuple(neutral))


def interval_rows(expression: Affine, interval: Interval) -> tuple[Inequality, ...]:
    return (Inequality(expression.add(Affine({}, -F(interval.upper))), not interval.include_upper),
            Inequality(expression.scale(F(-1)).add(Affine({}, F(interval.lower))), not interval.include_lower))


class FeasibleSpace:
    """Union of projected affine regions, sampled by conditional intervals."""

    def __init__(self, keys: tuple[str, ...], pieces: list[tuple[Inequality, ...]], check: Callable[[], None]):
        self.keys = keys
        self.check = check
        self.levels = []
        self.pieces = []
        for piece in pieces:
            check()
            reduced = simplify(piece)
            if reduced is None:
                continue
            levels = [reduced]
            for key in reversed(keys):
                reduced = eliminate(reduced, key, check)
                if reduced is None:
                    break
                levels.append(reduced)
            if reduced is not None:
                self.pieces.append(piece)
                self.levels.append(tuple(reversed(levels[:-1])))
        if not self.pieces:
            raise ValueError("The configured constraints and ranges have an empty feasible region.")

    def project(self, keys: tuple[str, ...]) -> FeasibleSpace:
        pieces = []
        for piece in self.pieces:
            reduced = piece
            for key in self.keys:
                if key not in keys:
                    reduced = eliminate(reduced, key, self.check)
                    if reduced is None:
                        break
            if reduced is not None:
                pieces.append(reduced)
        return FeasibleSpace(keys, pieces, self.check)

    def marginal_ranges(self, key):
        projected = self.project((key,))
        intervals = []
        for levels in projected.levels:
            low, high, lc, uc = None, None, True, True
            for row in levels[0]:
                coefficient = row.expression.terms.get(key, F(0))
                if not coefficient:
                    continue
                bound = -row.expression.constant / coefficient
                if coefficient > 0:
                    if high is None or bound < high:
                        high, uc = bound, not row.strict
                    elif bound == high:
                        uc = uc and not row.strict
                else:
                    if low is None or bound > low:
                        low, lc = bound, not row.strict
                    elif bound == low:
                        lc = lc and not row.strict
            if low is not None and high is not None:
                try:
                    intervals.append(representable(low, high, lc, uc))
                except ValueError:
                    continue
        if not intervals:
            raise ValueError(f"No representable feasible value exists for {input_label(key)}.")
        return normalize(tuple(intervals))

    def draw(self, units: dict[str, float], selections: dict[str, InputRange]) -> dict[str, float]:
        values = {}
        for index, key in enumerate(self.keys):
            self.check()
            intervals = []
            for levels in self.levels:
                low, high, lc, uc = None, None, True, True
                valid = True
                for row in levels[index]:
                    expr = row.expression
                    coefficient = expr.terms.get(key, F(0))
                    residual = expr.constant + sum(v * F(values[k]) for k, v in expr.terms.items() if k != key)
                    if not coefficient:
                        if residual > 0 or residual == 0 and row.strict:
                            valid = False
                            break
                        continue
                    bound = -residual / coefficient
                    if coefficient > 0:
                        if high is None or bound < high:
                            high, uc = bound, not row.strict
                        elif bound == high:
                            uc = uc and not row.strict
                    else:
                        if low is None or bound > low:
                            low, lc = bound, not row.strict
                        elif bound == low:
                            lc = lc and not row.strict
                if valid and low is not None and high is not None and (low < high or low == high and lc and uc):
                    try:
                        intervals.append(representable(low, high, lc, uc))
                    except ValueError:
                        pass
            if not intervals:
                raise InputGenerationError(f"No representable feasible value remains for {input_label(key)}.", values)
            selection = selections[key]
            values[key] = sample(tuple(intervals), units[key], selection.allocation, selection.log_scale)
        return values


def build_spaces(ranges, derived, inequalities, check, order):
    """Separate disconnected components before interval-union decomposition."""
    free = tuple(key for key in order if key in ranges and key not in derived)
    groups = [{key} for key in free]
    expressions = [row.expression for row in inequalities] + list(derived.values())
    for expression in expressions:
        involved = set(expression.terms)
        touched = [group for group in groups if group & involved]
        if touched:
            groups = [group for group in groups if group not in touched]
            groups.append(set().union(*touched))
    constant_rows = tuple(row for row in inequalities if not row.expression.terms)
    if simplify(constant_rows) is None:
        raise ValueError("The configured constraints have no feasible inputs.")
    spaces = []
    for group in groups:
        keys = tuple(key for key in free if key in group)
        base = tuple(row for row in inequalities if set(row.expression.terms) & group)
        domains = [(derived.get(key, Affine({key: F(1)})), intervals)
                   for key, intervals in ranges.items()
                   if set(derived.get(key, Affine({key: F(1)})).terms) & group]
        pieces = []
        for combination in product(*(intervals for _, intervals in domains)):
            check()
            rows = base + tuple(row for (expression, _), interval in zip(domains, combination)
                                for row in interval_rows(expression, interval))
            if simplify(rows) is not None:
                pieces.append(rows)
        spaces.append(FeasibleSpace(keys, pieces, check))
    for key, expression in derived.items():
        if not expression.terms and not any(i.contains(float(expression.constant)) for i in ranges[key]):
            raise ValueError(f"Calculated {input_label(key)} is outside its allowed ranges.")
    return spaces
