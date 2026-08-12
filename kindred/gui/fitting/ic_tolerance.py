"""Typed per-cell initial-condition tolerance for Fit setup."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

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
            raise ValueError("Unknown initial-condition tolerance mode.")
        object.__setattr__(self, "mode", mode)
        if mode == "disabled":
            if self.lower is not None or self.upper is not None:
                raise ValueError("Disabled IC tolerance cannot retain bounds.")
            return
        lower = _finite(self.lower, "IC tolerance lower value")
        upper = _finite(self.upper, "IC tolerance upper value")
        if mode == "symmetric_percent":
            if lower < 0.0 or upper != lower or lower == 0.0:
                raise ValueError(
                    "Symmetric IC tolerance requires one positive percentage."
                )
        elif mode == "asymmetric_percent":
            if lower < 0.0 or upper < 0.0 or (lower == 0.0 and upper == 0.0):
                raise ValueError(
                    "Asymmetric IC tolerance percentages must be nonnegative "
                    "and not both zero."
                )
        elif lower < 0.0 or lower >= upper:
            raise ValueError(
                "Absolute IC tolerance requires nonnegative minimum < maximum."
            )
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    @classmethod
    def symmetric_percent(cls, percent: object) -> "FitInitialTolerance":
        value = _finite(percent, "IC tolerance percentage")
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

        nominal = _finite(initial, "IC initial value")
        if nominal < 0.0:
            raise ValueError("Initial concentration must be nonnegative.")
        if not self.enabled:
            return nominal, nominal
        assert self.lower is not None and self.upper is not None
        if self.mode in {"symmetric_percent", "asymmetric_percent"}:
            if nominal == 0.0:
                raise ValueError(
                    "A zero IC initial guess cannot move through percentage "
                    "bounds; use meaningful absolute bounds."
                )
            minimum = max(0.0, nominal * (1.0 - self.lower / 100.0))
            maximum = nominal * (1.0 + self.upper / 100.0)
        else:
            minimum, maximum = self.lower, self.upper
        if not math.isfinite(minimum) or not math.isfinite(maximum):
            raise ValueError(
                "IC tolerance resolves to non-finite bounds; reduce the "
                "percentage or use explicit finite absolute bounds."
            )
        if minimum >= maximum or not minimum <= nominal <= maximum:
            raise ValueError(
                "IC tolerance must resolve to minimum <= initial <= maximum."
            )
        return float(minimum), float(maximum)
