"""Seeded, exact-count scan designs with constructive affine restrictions."""
from __future__ import annotations

from fractions import Fraction as F
from pathlib import Path
import math
import json
import tempfile
import warnings

import numpy as np
from scipy.stats import qmc

from kindred.core.algebra.errors import AlgebraError

from kindred.scan.config import ScanConfig, RunInput, input_label
from kindred.scan.inputs import input_catalog, admitted_ranges
from kindred.scan.intervals import Interval
from kindred.scan.constraints import Affine, Inequality, IntervalChecks, InputGenerationError, compile_constraints, build_spaces, conditional_order, FeasibleSpace, interval_rows


def unit_samples(count, keys, method, seed, directory, check):
    """One design, not independent per-chunk LHS designs."""
    if not keys:
        for _ in range(count):
            check()
            yield {}
        return
    if method == "sobol":
        engine = qmc.Sobol(d=len(keys), scramble=True, seed=seed)
        remaining = count
        while remaining:
            check()
            size = min(remaining, 1024)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="The balance properties of Sobol.*", category=UserWarning)
                block = engine.random(size)
            for row in block:
                yield dict(zip(keys, map(float, row)))
            remaining -= size
        return
    rng = np.random.default_rng(seed)
    path = Path(directory) / f"lhs-{seed}.bin"
    matrix = np.memmap(path, dtype="float64", mode="w+", shape=(count, len(keys)))
    try:
        for column in range(len(keys)):
            check()
            permutation = rng.permutation(count)
            for start in range(0, count, 4096):
                check()
                end = min(count, start + 4096)
                matrix[start:end, column] = (permutation[start:end] + rng.random(end - start)) / count
        for start in range(0, count, 1024):
            check()
            for row in np.array(matrix[start:start + 1024]):
                yield dict(zip(keys, map(float, row)))
    finally:
        matrix._mmap.close()
        path.unlink(missing_ok=True)


def validate_endpoint_budget(endpoints, quota, count):
    if quota < 0 or quota > count:
        raise ValueError("Endpoint attempts must be between zero and the total count of this design.")
    if endpoints and not quota:
        raise ValueError("Selected endpoints require a nonzero endpoint budget.")
    if quota and not endpoints:
        raise ValueError("Select at least one included endpoint for the endpoint budget.")
    if quota < len(endpoints):
        raise ValueError("The endpoint budget must cover every selected endpoint at least once.")


class InputDesign:
    def __init__(self, ranges, derived, spaces, selections, endpoints, quota, check):
        self.ranges, self.derived, self.selections = ranges, derived, selections
        # Unit coordinates remain associated with catalog keys, independently
        # of the conditional order used inside the feasible spaces.
        self.keys = tuple(key for key in ranges if key not in derived)
        self.spaces = spaces
        self.marginals = {key: space.marginal_ranges(key) for space in spaces for key in space.keys}
        self.faces = []
        self.quota = quota
        for key, endpoint in endpoints:
            expression = derived.get(key, Affine({key: F(1)}))
            try:
                if not expression.terms and float(expression.constant) != endpoint:
                    raise ValueError("The dependent expression does not materialize the exact requested endpoint.")
                face = []
                for space in spaces:
                    extra = interval_rows(expression, Interval(endpoint, endpoint)) if set(expression.terms) & set(space.keys) else ()
                    projected = FeasibleSpace(space.keys, [piece + extra for piece in space.pieces], check)
                    for coordinate in projected.keys:
                        projected.marginal_ranges(coordinate)
                    face.append(projected)
            except ValueError as exc:
                raise ValueError(f"Selected endpoint {input_label(key)} = {endpoint:g}: {exc}") from exc
            self.faces.append((key, endpoint, face))

    def _draw(self, spaces, units):
        values = {}
        try:
            for space in spaces:
                values.update(space.draw(units, self.selections))
            for key, expression in self.derived.items():
                values[key] = expression.evaluate(values)
            for key, value in values.items():
                if not any(interval.contains(value) for interval in self.ranges[key]):
                    raise ValueError(f"Calculated {input_label(key)} = {value:g} cannot be represented inside its allowed ranges.")
        except (ValueError, OverflowError) as exc:
            if isinstance(exc, InputGenerationError):
                values.update(exc.values)
            raise InputGenerationError(str(exc), {k: v for k, v in values.items() if math.isfinite(v)}) from exc
        return values

    def draw(self, index, units):
        if index < self.quota:
            key, endpoint, spaces = self.faces[index % len(self.faces)]
            try:
                values = self._draw(spaces, units)
                if values[key] != endpoint:
                    raise InputGenerationError("The dependent expression did not produce the exact endpoint.", values)
            except InputGenerationError as exc:
                raise InputGenerationError(f"Selected endpoint {input_label(key)} = {endpoint:g}: {exc}", exc.values) from exc
            return values
        return self._draw(self.spaces, units)


class _SampledSets:
    """Disk-backed reusable inputs and per-slot materialization diagnostics."""

    def __init__(self, directory, name, count, keys):
        self.keys = keys
        self.values = self.offsets = self.errors = None
        self.unavailable = ""
        try:
            self.values = np.memmap(Path(directory) / f"{name}.bin", dtype="float64", mode="w+", shape=(count, max(1, len(keys))))
            self.values[:] = np.nan
            self.offsets = np.memmap(Path(directory) / f"{name}-errors.bin", dtype="int64", mode="w+", shape=(count,))
            self.offsets[:] = -1
            self.errors = tempfile.TemporaryFile(mode="w+b", dir=directory)
        except BaseException:
            self.close()
            raise

    def record(self, index, values, error=""):
        for column, key in enumerate(self.keys):
            if key in values:
                self.values[index, column] = values[key]
        if error:
            self.offsets[index] = self.errors.tell()
            self.errors.write((json.dumps(error) + "\n").encode("utf-8"))

    def slot(self, index):
        values = {key: float(self.values[index, column]) for column, key in enumerate(self.keys)
                  if math.isfinite(self.values[index, column])}
        error = self.unavailable
        if self.offsets[index] >= 0:
            self.errors.seek(int(self.offsets[index]))
            error = json.loads(self.errors.readline())
        return values, error

    def close(self):
        for matrix in (self.values, self.offsets):
            if matrix is not None:
                matrix._mmap.close()
        self.values = self.offsets = None
        if self.errors is not None:
            self.errors.close()
            self.errors = None


class PreparedScanDesign:
    def __init__(self, config: ScanConfig, prepared, check):
        self.config, self.prepared, self.check = config, prepared, check
        self.storage = tempfile.TemporaryDirectory(prefix="kindred-scan-design-")
        self.groups = {}
        try:
            self._prepare()
        except BaseException:
            self.close()
            raise

    def _prepare(self):
        cfg = self.config
        if cfg.requested <= 0 or cfg.seed < 0:
            raise ValueError("Request a positive simulation count and a nonnegative seed.")
        if not cfg.outputs:
            raise ValueError("Select at least one exported species or authored quantity.")
        self.check()
        prepared = self.prepared
        self.catalog = input_catalog(prepared, cfg)
        all_ranges = admitted_ranges(self.catalog, cfg)
        self.fixed = {item.key: item.value for item in self.catalog if item.key not in cfg.ranges}
        ranges = {key: intervals for key, intervals in all_ranges.items() if key in cfg.ranges}
        derived, inequalities, precedence = compile_constraints(prepared, cfg, self.fixed)
        singleton_values = {key: Affine({}, F(intervals[0].lower)) for key, intervals in ranges.items()
                            if key not in derived and len(intervals) == 1 and intervals[0].lower == intervals[0].upper}
        derived = {key: expression.substitute(singleton_values) for key, expression in derived.items()}
        inequalities = tuple(Inequality(row.expression.substitute(singleton_values), row.strict) for row in inequalities)
        endpoints = [(key, value) for key, selection in cfg.ranges.items() for value in selection.endpoints]
        if cfg.grouped:
            if cfg.parameter_sets <= 0 or cfg.initial_sets <= 0 or cfg.parameter_sets * cfg.initial_sets != cfg.requested:
                raise ValueError("Parameter/time sets × initial-condition sets must equal the requested simulation count.")
            for initial, quota, count in ((False, cfg.parameter_boundaries, cfg.parameter_sets),
                                          (True, cfg.initial_boundaries, cfg.initial_sets)):
                validate_endpoint_budget([(k, v) for k, v in endpoints if k.startswith("initial:") == initial], quota, count)
        else:
            validate_endpoint_budget(endpoints, cfg.boundary_attempts, cfg.requested)
        self.effective_ranges = all_ranges
        self.derived = derived
        order = conditional_order(tuple(key for key in ranges if key not in derived), precedence)
        spaces = build_spaces(ranges, derived, inequalities, self.check, order)
        if not cfg.grouped:
            self.interval_checks = IntervalChecks(prepared, cfg)
            self.joint = InputDesign(ranges, derived, spaces, cfg.ranges, endpoints,
                                     cfg.boundary_attempts, self.check)
            self.effective_ranges.update(self.joint.marginals)
            return
        for key, expression in derived.items():
            if any(k.startswith("initial:") != key.startswith("initial:") for k in expression.terms):
                raise ValueError(f"Calculated {input_label(key)} depends on the other group. These inputs cannot be reused unchanged. Choose Joint sampling in Exploration.")

        def group_of(key):
            return "initial" if key.startswith("initial:") else "parameter"

        group_precedence = {(group_of(source), group_of(target)) for source, target in precedence
                            if source in order and target in order and group_of(source) != group_of(target)}
        first, second = conditional_order(("parameter", "initial"), group_precedence)
        counts = {"parameter": cfg.parameter_sets, "initial": cfg.initial_sets}
        quotas = {"parameter": cfg.parameter_boundaries, "initial": cfg.initial_boundaries}
        seeds = {"parameter": cfg.seed, "initial": cfg.seed + 1}
        keys = {group: tuple(key for key in ranges if group_of(key) == group) for group in (first, second)}
        self.interval_checks = IntervalChecks(prepared, cfg, set(self.fixed) | set(keys[first]))
        for group in (first, second):
            self.groups[group] = _SampledSets(self.storage.name, group, counts[group], keys[group])

        first_spaces = [space.project(tuple(key for key in space.keys if key in keys[first]))
                        for space in spaces if set(space.keys) & set(keys[first])]
        first_design = InputDesign({key: ranges[key] for key in keys[first]},
                                   {key: value for key, value in derived.items() if key in keys[first]},
                                   first_spaces, cfg.ranges, [(k, v) for k, v in endpoints if k in keys[first]],
                                   quotas[first], self.check)
        self.effective_ranges.update(first_design.marginals)
        maxima = [None] * len(inequalities)
        successes = 0
        for values in self._materialize(first_design, self.groups[first], counts[first], seeds[first], first_group=True):
            successes += 1
            for index, row in enumerate(inequalities):
                contribution = sum(coef * F(values[key]) for key, coef in row.expression.terms.items() if key in keys[first])
                maxima[index] = contribution if maxima[index] is None else max(maxima[index], contribution)
        if not successes:
            self.groups[second].unavailable = f"The reusable {second} design could not be materialized because every {first} set failed."
            return
        common = tuple(Inequality(Affine({key: coef for key, coef in row.expression.terms.items() if key in keys[second]},
                                         row.expression.constant + maxima[index]), row.strict)
                       for index, row in enumerate(inequalities))
        second_ranges = {key: ranges[key] for key in keys[second]}
        second_derived = {key: value for key, value in derived.items() if key in keys[second]}
        try:
            second_spaces = build_spaces(second_ranges, second_derived, common, self.check, order)
            second_design = InputDesign(second_ranges, second_derived, second_spaces, cfg.ranges,
                                        [(k, v) for k, v in endpoints if k in keys[second]],
                                        quotas[second], self.check)
        except ValueError as exc:
            raise ValueError(f"No reusable {second} sets fit every generated {first} set. Choose Joint sampling in Exploration.") from exc
        self.effective_ranges.update(second_design.marginals)
        for _ in self._materialize(second_design, self.groups[second], counts[second], seeds[second]):
            pass

    def _materialize(self, design, storage, count, seed, *, first_group=False):
        for index, unit in enumerate(unit_samples(count, design.keys, self.config.sampler,
                                                  seed, self.storage.name, self.check)):
            values = {}
            try:
                values = design.draw(index, unit)
                if first_group:
                    self.interval_checks.admit(self.fixed | values, first_group=True)
            except (ValueError, OverflowError, AlgebraError) as exc:
                if isinstance(exc, InputGenerationError):
                    values.update(exc.values)
                storage.record(index, values, str(exc))
            else:
                storage.record(index, values)
                yield values

    def inputs(self):
        cfg = self.config
        if cfg.grouped:
            for p in range(cfg.parameter_sets):
                for c in range(cfg.initial_sets):
                    self.check()
                    parameters, parameter_error = self.groups["parameter"].slot(p)
                    initials, initial_error = self.groups["initial"].slot(c)
                    values = self.fixed | parameters | initials
                    messages = [message for message in (parameter_error, initial_error) if message]
                    # Two messages are joined as sentences; one passes through as it is.
                    error = " ".join(
                        message if message.endswith(".") else message + "." for message in messages
                    ) if len(messages) > 1 else "".join(messages)
                    if not error:
                        try:
                            self.interval_checks.admit(values)
                        except (ValueError, OverflowError, AlgebraError) as exc:
                            error = str(exc)
                    yield RunInput(p * cfg.initial_sets + c + 1, values, p + 1, c + 1, error)
        else:
            for index, unit in enumerate(unit_samples(cfg.requested, self.joint.keys, cfg.sampler,
                                                      cfg.seed, self.storage.name, self.check)):
                values = {}
                try:
                    values = self.joint.draw(index, unit)
                    self.interval_checks.admit(self.fixed | values)
                except (ValueError, OverflowError, AlgebraError) as exc:
                    if isinstance(exc, InputGenerationError):
                        values.update(exc.values)
                    yield RunInput(index + 1, self.fixed | values, generation_error=str(exc))
                else:
                    yield RunInput(index + 1, self.fixed | values)

    def close(self):
        for group in self.groups.values():
            group.close()
        self.groups.clear()
        self.storage.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
