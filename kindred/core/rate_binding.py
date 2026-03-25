from __future__ import annotations

from dataclasses import dataclass

__all__ = ["RateBinding"]


@dataclass
class RateBinding:
    """
    Lightweight binding that allows rate constants (or scalar parameters) to be updated in-place.

    Callers hold references to this object and read its value via calling the instance
    (``binding()``) while writers update it via ``binding.set(...)``.
    """

    name: str
    value: float

    def __call__(self) -> float:
        return float(self.value)

    def set(self, new_value: float) -> None:
        self.value = float(new_value)

