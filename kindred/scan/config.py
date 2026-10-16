"""Operation-local scan configuration; no project or scientific state ownership."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from kindred.scan.intervals import Interval


@dataclass(frozen=True)
class InputRange:
    intervals: tuple[Interval, ...]
    allocation: Literal["width", "equal"] = "width"
    endpoints: tuple[float, ...] = ()
    log_scale: bool = False


@dataclass(frozen=True)
class ObservableConstraint:
    observable: str
    condition: Literal["equals", "range"]
    lower: float
    upper: float
    dependent: str
    include_lower: bool = True
    include_upper: bool = True


@dataclass(frozen=True)
class ScanConfig:
    source: str
    initials: dict[str, float]
    parameters: dict[str, float]
    ranges: dict[str, InputRange]
    outputs: tuple[str, ...]
    t_start: float
    t_end: float
    solver: str
    points: int
    rtol: float
    atol: float
    analytic_jacobian: bool
    use_authored_stops: bool
    requested: int
    destination: str
    seed: int = 1729
    sampler: Literal["sobol", "lhs"] = "sobol"
    # The user-selected exploration mode; never inferred from the counts, so
    # zero counts in grouped mode reach the grouped-count admission error.
    grouped: bool = False
    parameter_sets: int = 0
    initial_sets: int = 0
    boundary_attempts: int = 0
    parameter_boundaries: int = 0
    initial_boundaries: int = 0
    workers: int = 1
    constraints: tuple[ObservableConstraint, ...] = ()
    energy_unit: str = "kJ/mol"
    temperature_K: float = 298.15
    standard_conc_M: float = 1.0


@dataclass
class ScanProgress:
    requested: int
    attempted: int = 0
    succeeded: int = 0
    failed: int = 0
    cancelled: int = 0
    saved: int = 0
    export_errors: int = 0
    running: int = 0
    phase: str = "Preparing scan"
    latest_failure: str = ""
    path: str = ""
    state: str = "running"
    error: str = ""


@dataclass(frozen=True)
class RunInput:
    run_id: int
    values: dict[str, float]
    parameter_set: int | None = None
    initial_set: int | None = None
    generation_error: str = ""


def initial_key(name: str) -> str:
    return f"initial:{name}"


def parameter_key(name: str) -> str:
    return f"parameter:{name}"


START = "time:start"
END = "time:end"


def input_label(key: str) -> str:
    """The name an input key is shown under."""
    if key == START:
        return "Start time"
    if key == END:
        return "End time"
    kind, _, name = key.partition(":")
    return f"Initial {name}" if kind == "initial" else name
