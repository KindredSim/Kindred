from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from functools import reduce
from math import gcd
from typing import Sequence

from kindred.core.simulator.errors import DSLError

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


def _as_fraction_stoich_map(
    values: Mapping[object, object],
    *,
    label: str,
) -> dict[str, Fraction]:
    if not isinstance(values, Mapping):
        raise DSLError(
            f"Canonical equilibrium {label} stoichiometry must be a mapping."
        )
    if not values:
        raise DSLError(
            f"Canonical equilibrium {label} stoichiometry cannot be empty."
        )
    out: dict[str, Fraction] = {}
    for raw_name, raw_value in values.items():
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise DSLError(
                f"Canonical equilibrium {label} has an invalid species name: "
                f"{raw_name!r}."
            )
        try:
            value = Fraction(str(raw_value))
        except (TypeError, ValueError) as exc:
            raise DSLError(
                f"Invalid canonical stoichiometric coefficient for {raw_name!r}: "
                f"{raw_value!r}."
            ) from exc
        if value <= 0:
            raise DSLError(
                f"Canonical equilibrium {label} coefficient for {raw_name!r} "
                f"must be positive, got {raw_value!r}."
            )
        out[raw_name] = value
    return out


def _canonical_index(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DSLError(
            f"Canonical Wegscheider step map has an invalid {label}: {value!r}."
        )
    return int(value)


def _canonical_reversible_steps(
    mechanism: object,
    equilibria: Sequence[object],
) -> tuple[tuple[int, int], ...]:
    """Return every equilibrium's authoritative global step identity.

    The DSL builder owns the global map.  Wegscheider analysis deliberately
    does not invent a local numbering when that map is unavailable or
    incomplete: the resulting ``KeqN`` name would no longer identify the
    admitted source reaction.
    """

    reactions = tuple(getattr(mechanism, "reactions", ()) or ())
    metadata = getattr(mechanism, "metadata", None)
    if not isinstance(metadata, Mapping):
        raise DSLError(
            "Canonical mechanism metadata is missing; Wegscheider analysis "
            "requires the admitted step_index_map."
        )
    raw_mapping = metadata.get("step_index_map")
    if not isinstance(raw_mapping, list):
        raise DSLError(
            "Canonical mechanism step_index_map is missing or invalid; "
            "Wegscheider analysis cannot infer reaction identities."
        )
    expected_count = len(reactions) + len(equilibria)
    if len(raw_mapping) != expected_count:
        raise DSLError(
            "Canonical mechanism step_index_map is incomplete; Wegscheider "
            "analysis requires one entry for every authored kinetic step."
        )

    seen_step_indices: set[int] = set()
    seen_reaction_indices: set[int] = set()
    seen_equilibrium_indices: set[int] = set()
    equilibrium_steps: list[tuple[int, int]] = []
    for position, raw_entry in enumerate(raw_mapping):
        if not isinstance(raw_entry, Mapping):
            raise DSLError(
                f"Canonical mechanism step_index_map entry {position} is invalid."
            )
        step_index = _canonical_index(
            raw_entry.get("step_index"),
            label=f"step_index at entry {position}",
        )
        if step_index <= 0 or step_index in seen_step_indices:
            raise DSLError(
                "Canonical mechanism step_index_map has duplicate or non-positive "
                f"step_index {step_index}."
            )
        seen_step_indices.add(step_index)
        kind = raw_entry.get("kind")
        if kind == "reaction":
            reaction_index = _canonical_index(
                raw_entry.get("reaction_index"),
                label=f"reaction_index at entry {position}",
            )
            if not 0 <= reaction_index < len(reactions):
                raise DSLError(
                    "Canonical mechanism step_index_map reaction_index is out "
                    f"of range: {reaction_index}."
                )
            if reaction_index in seen_reaction_indices:
                raise DSLError(
                    "Canonical mechanism step_index_map repeats reaction_index "
                    f"{reaction_index}."
                )
            seen_reaction_indices.add(reaction_index)
            continue
        if kind != "equilibrium":
            raise DSLError(
                "Canonical mechanism step_index_map has unsupported step kind "
                f"{kind!r}."
            )
        equilibrium_index = _canonical_index(
            raw_entry.get("equilibrium_index"),
            label=f"equilibrium_index at entry {position}",
        )
        if not 0 <= equilibrium_index < len(equilibria):
            raise DSLError(
                "Canonical mechanism step_index_map equilibrium_index is out "
                f"of range: {equilibrium_index}."
            )
        if equilibrium_index in seen_equilibrium_indices:
            raise DSLError(
                "Canonical mechanism step_index_map repeats equilibrium_index "
                f"{equilibrium_index}."
            )
        seen_equilibrium_indices.add(equilibrium_index)
        equilibrium_steps.append((equilibrium_index, step_index))

    expected_step_indices = set(range(1, expected_count + 1))
    if seen_step_indices != expected_step_indices:
        raise DSLError(
            "Canonical mechanism step_index_map does not cover the complete "
            "authored global step order."
        )
    if seen_reaction_indices != set(range(len(reactions))):
        raise DSLError(
            "Canonical mechanism step_index_map omits an authored reaction step."
        )
    if seen_equilibrium_indices != set(range(len(equilibria))):
        raise DSLError(
            "Canonical mechanism step_index_map omits an authored equilibrium step."
        )
    return tuple(sorted(equilibrium_steps, key=lambda item: item[1]))


def _reversible_steps(mechanism: object) -> tuple[_ReversibleStep, ...]:
    eqs = list(getattr(mechanism, "equilibria", []) or [])
    if not eqs:
        return ()
    authored_steps = _canonical_reversible_steps(mechanism, eqs)

    steps: list[_ReversibleStep] = []
    for equilibrium_index, step_index in authored_steps:
        eq = eqs[equilibrium_index]
        forward = _as_fraction_stoich_map(
            getattr(eq, "stoich_forward", None),
            label="forward",
        )
        back = _as_fraction_stoich_map(
            getattr(eq, "stoich_back", None),
            label="backward",
        )
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
