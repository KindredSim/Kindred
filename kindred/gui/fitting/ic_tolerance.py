"""Typed per-cell initial-condition tolerance for Fit setup."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from kindred.core.species import validate_concentration

__all__ = ["FitInitialTolerance", "FitInitialToleranceMode"]

FitInitialToleranceMode = Literal[
    "disabled",
    "symmetric_percent",
    "asymmetric_percent",
    "absolute",
]


def _finite(value: object, field: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be finite.")
    return parsed


@dataclass(frozen=True, slots=True)
class FitInitialTolerance:
    """One complete bound-entry form for a local ``(set_id, species)`` IC.

    ``lower`` and ``upper`` are percentages for percentage modes and absolute
    bounds for ``absolute``.  Disabled cells deliberately own no dormant
    bounds; changing mode authors a fresh tolerance.
    """

    mode: FitInitialToleranceMode = "disabled"
    lower: float | None = None
    upper: float | None = None

    def __post_init__(self) -> None:
        mode = str(self.mode or "").strip().lower()
        if mode not in {
            "disabled",
            "symmetric_percent",
            "asymmetric_percent",
            "absolute",
        }:
            raise ValueError("Unknown Vary IC mode.")
        object.__setattr__(self, "mode", mode)
        if mode == "disabled":
            if self.lower is not None or self.upper is not None:
                raise ValueError("No Vary IC mode cannot retain bounds.")
            return
        lower = _finite(self.lower, "Vary IC lower value")
        upper = _finite(self.upper, "Vary IC upper value")
        if mode == "symmetric_percent":
            if lower < 0.0 or upper != lower or lower == 0.0:
                raise ValueError(
                    "Symmetric Vary IC requires one positive percentage."
                )
        elif mode == "asymmetric_percent":
            if lower < 0.0 or upper < 0.0 or (lower == 0.0 and upper == 0.0):
                raise ValueError(
                    "Asymmetric Vary IC percentages must be nonnegative "
                    "and not both zero."
                )
        elif lower < 0.0 or lower >= upper:
            raise ValueError(
                "Absolute Vary IC requires nonnegative minimum < maximum."
            )
        if mode in {"symmetric_percent", "asymmetric_percent"} and lower >= 100.0:
            raise ValueError(
                "Vary IC lower percentage must be below 100; use Absolute "
                "Minimum / Maximum (in M) to allow zero."
            )
        if mode == "absolute":
            lower = validate_concentration(
                lower,
                field="Vary IC absolute minimum",
            )
            upper = validate_concentration(
                upper,
                field="Vary IC absolute maximum",
            )
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    @classmethod
    def symmetric_percent(cls, percent: object) -> "FitInitialTolerance":
        value = _finite(percent, "Vary IC percentage")
        return cls("symmetric_percent", value, value)

    @classmethod
    def asymmetric_percent(
        cls,
        lower_percent: object,
        upper_percent: object,
    ) -> "FitInitialTolerance":
        return cls("asymmetric_percent", lower_percent, upper_percent)

    @classmethod
    def absolute(
        cls,
        minimum: object,
        maximum: object,
    ) -> "FitInitialTolerance":
        return cls("absolute", minimum, maximum)

    @property
    def enabled(self) -> bool:
        return self.mode != "disabled"

    def bounds_for(self, initial: object) -> tuple[float, float]:
        """Resolve bounds from the current editable nominal value."""

        nominal = validate_concentration(
            _finite(initial, "IC initial value"),
            field="IC initial value",
        )
        if not self.enabled:
            return nominal, nominal
        assert self.lower is not None and self.upper is not None
        if self.mode in {"symmetric_percent", "asymmetric_percent"}:
            if nominal == 0.0:
                raise ValueError(
                    "A zero Starting IC cannot move through percentage "
                    "bounds; use meaningful absolute bounds."
                )
            # Preserve the exact authored percentage formula.
            minimum = nominal * (1.0 - self.lower / 100.0)
            maximum = nominal * (1.0 + self.upper / 100.0)
        else:
            minimum, maximum = self.lower, self.upper
        if not math.isfinite(minimum) or not math.isfinite(maximum):
            raise ValueError(
                "Vary IC resolves to non-finite bounds; reduce the "
                "percentage or use explicit finite absolute bounds."
            )
        minimum = validate_concentration(
            minimum,
            field="Vary IC minimum",
        )
        maximum = validate_concentration(
            maximum,
            field="Vary IC maximum",
        )
        if minimum >= maximum or not minimum <= nominal <= maximum:
            raise ValueError(
                "Vary IC must resolve to minimum <= Starting IC <= maximum."
            )
        return float(minimum), float(maximum)
