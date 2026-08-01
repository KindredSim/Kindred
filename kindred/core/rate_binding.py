from __future__ import annotations

from dataclasses import dataclass

from .kinetics import (
    ForwardRateModel,
    forward_rate_at_temperature,
    forward_rate_family,
    write_reference_rate,
)

__all__ = ["ForwardRateBinding", "RateBinding"]


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

    def __float__(self) -> float:
        return float(self.value)

    def set(self, new_value: float) -> None:
        self.value = float(new_value)


class ForwardRateBinding(RateBinding):
    """Mutable k@Tref coordinate that retains its authored rate family."""

    accepts_runtime_temperature = True

    def __init__(
        self,
        *,
        name: str,
        model: ForwardRateModel,
        reference_temperature_K: float,
    ) -> None:
        self._model = model
        value = forward_rate_at_temperature(
            model,
            reference_temperature_K,
        )
        self.reference_temperature_K = float(reference_temperature_K)
        super().__init__(name=str(name), value=float(value))

    @property
    def model(self) -> ForwardRateModel:
        return self._model

    @property
    def family(self) -> str:
        return forward_rate_family(self._model)

    def __call__(self, temperature_K: float | None = None) -> float:
        temperature = (
            self.reference_temperature_K
            if temperature_K is None
            else temperature_K
        )
        return forward_rate_at_temperature(self._model, temperature)

    def __float__(self) -> float:
        return self()

    def set(self, new_value: float) -> None:
        self._model = write_reference_rate(
            self._model,
            reference_rate=new_value,
            reference_temperature_K=self.reference_temperature_K,
        )
        self.value = forward_rate_at_temperature(
            self._model,
            self.reference_temperature_K,
        )
