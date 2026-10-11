"""Typed per-cell initial-condition tolerance for Fit setup."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from kindred.core.species import validate_concentration

__all__ = ["FitInitialTolerance", "FitInitialToleranceMode", "ZERO_IC_PERCENT_MESSAGE"]

ZERO_IC_PERCENT_MESSAGE = "A zero Starting IC can vary only with Absolute Minimum / Maximum (in M)."

FitInitialToleranceMode = Literal[
    "disabled",
    "symmetric_percent",
    "asymmetric_percent",
    "absolute",
]


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
        mode = self.mode
        if mode == "disabled":
            return
        lower = self.lower
        upper = self.upper
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
                "Vary IC lower percentage must be below 100. Use Absolute "
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
    def symmetric_percent(cls, percent: float) -> "FitInitialTolerance":
        return cls("symmetric_percent", percent, percent)

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

    def bounds_for(self, nominal: float) -> tuple[float, float]:
        """Resolve bounds from the current editable nominal value."""

        if not self.enabled:
            return nominal, nominal
        assert self.lower is not None and self.upper is not None
        if self.mode in {"symmetric_percent", "asymmetric_percent"}:
            if nominal == 0.0:
                raise ValueError(ZERO_IC_PERCENT_MESSAGE)
            # Preserve the exact authored percentage formula.
            minimum = nominal * (1.0 - self.lower / 100.0)
            maximum = nominal * (1.0 + self.upper / 100.0)
        else:
            minimum, maximum = self.lower, self.upper
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
