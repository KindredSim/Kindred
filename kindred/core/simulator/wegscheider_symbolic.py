from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from functools import reduce
from math import gcd
from typing import Sequence


__all__ = [
    "WegscheiderCycle",
    "WegscheiderTopologyReport",
    "analyze_wegscheider_topology",
]


@dataclass(frozen=True)
class WegscheiderCycle:
    cycle_id: str
    step_indices: tuple[int, ...]
    coefficients: tuple[int, ...]


@dataclass(frozen=True)
class WegscheiderTopologyReport:
    """Readable cycle topology derived from canonical reversible stoichiometry."""

    cycles: tuple[WegscheiderCycle, ...]


@dataclass(frozen=True)
class _ReversibleStep:
    step_index: int
    vector: Mapping[str, Fraction]


def _as_fraction_stoich_map(values: Mapping[str, float]) -> dict[str, Fraction]:
    # Coefficients are parser-admitted positive reals of a nonempty side.
    return {name: Fraction(str(value)) for name, value in values.items()}


def _canonical_reversible_steps(mechanism: object) -> tuple[tuple[int, int], ...]:
    """Return every equilibrium's authoritative global step identity.

    The DSL builder owns the global map, with one entry for every authored
    step, so each ``KeqN`` name identifies its admitted source reaction.
    """

    return tuple(
        (entry["equilibrium_index"], entry["step_index"])
        for entry in mechanism.metadata["step_index_map"]
        if entry["kind"] == "equilibrium"
    )


def _reversible_steps(mechanism: object) -> tuple[_ReversibleStep, ...]:
    eqs = list(mechanism.equilibria)
    if not eqs:
        return ()
    authored_steps = _canonical_reversible_steps(mechanism)

    steps: list[_ReversibleStep] = []
    for equilibrium_index, step_index in authored_steps:
        eq = eqs[equilibrium_index]
        forward = _as_fraction_stoich_map(eq.stoich_forward)
        back = _as_fraction_stoich_map(eq.stoich_back)
        species = sorted(set(forward) | set(back))
        vector = {
            name: back.get(name, Fraction(0)) - forward.get(name, Fraction(0))
            for name in species
            if back.get(name, Fraction(0)) - forward.get(name, Fraction(0)) != 0
        }
        steps.append(
            _ReversibleStep(
                step_index=step_index,
                vector=vector,
            )
        )
    return tuple(sorted(steps, key=lambda step: step.step_index))


def _lcm(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return abs(a * b) // gcd(abs(a), abs(b))


def _normalize_integer_vector(values: Sequence[Fraction]) -> tuple[int, ...]:
    denom_lcm = 1
    for value in values:
        denom_lcm = _lcm(denom_lcm, value.denominator)
    ints = [int(value * denom_lcm) for value in values]
    divisor = reduce(gcd, (abs(value) for value in ints if value), 0) or 1
    ints = [value // divisor for value in ints]
    first = next((value for value in ints if value), 0)
    if first < 0:
        ints = [-value for value in ints]
    return tuple(int(value) for value in ints)


def _nullspace_integer_basis(
    matrix: Sequence[Sequence[Fraction]],
) -> tuple[tuple[int, ...], ...]:
    if not matrix:
        return ()
    rows = [[Fraction(value) for value in row] for row in matrix]
    row_count = len(rows)
    col_count = len(rows[0]) if rows else 0
    pivot_cols: list[int] = []
    pivot_row = 0
    for col in range(col_count):
        pivot = None
        for row in range(pivot_row, row_count):
            if rows[row][col] != 0:
                pivot = row
                break
        if pivot is None:
            continue
        rows[pivot_row], rows[pivot] = rows[pivot], rows[pivot_row]
        scale = rows[pivot_row][col]
        rows[pivot_row] = [value / scale for value in rows[pivot_row]]
        for row in range(row_count):
            if row == pivot_row or rows[row][col] == 0:
                continue
            factor = rows[row][col]
            rows[row] = [
                current - factor * pivot_value
                for current, pivot_value in zip(rows[row], rows[pivot_row])
            ]
        pivot_cols.append(col)
        pivot_row += 1
        if pivot_row == row_count:
            break

    free_cols = [col for col in range(col_count) if col not in pivot_cols]
    basis: list[tuple[int, ...]] = []
    for free_col in free_cols:
        vec = [Fraction(0) for _ in range(col_count)]
        vec[free_col] = Fraction(1)
        for row, pivot_col in enumerate(pivot_cols):
            vec[pivot_col] = -rows[row][free_col]
        normalized = _normalize_integer_vector(vec)
        if any(normalized):
            basis.append(normalized)
    return tuple(basis)


def analyze_wegscheider_topology(mechanism: object) -> WegscheiderTopologyReport:
    """Find structural cycles before evaluating any rate or equilibrium value."""

    steps = _reversible_steps(mechanism)
    if not steps:
        return WegscheiderTopologyReport(cycles=())
    species = sorted(
        {species_name for step in steps for species_name in step.vector.keys()}
    )
    if not species:
        basis = tuple(
            tuple(1 if index == column else 0 for index in range(len(steps)))
            for column in range(len(steps))
        )
    else:
        matrix = [
            [step.vector.get(species_name, Fraction(0)) for step in steps]
            for species_name in species
        ]
        basis = _nullspace_integer_basis(matrix)
    cycles: list[WegscheiderCycle] = []
    for idx, coeffs in enumerate(basis, start=1):
        selected = [
            (step, coeff)
            for step, coeff in zip(steps, coeffs)
            if int(coeff) != 0
        ]
        selected_coefficients = tuple(
            int(coefficient) for _step, coefficient in selected
        )
        cycle = WegscheiderCycle(
            cycle_id=f"cycle_{idx}",
            step_indices=tuple(step.step_index for step, _coeff in selected),
            coefficients=selected_coefficients,
        )
        cycles.append(cycle)
    return WegscheiderTopologyReport(cycles=tuple(cycles))
