"""Execution-local vessel arithmetic and continuous trajectory ownership."""
from __future__ import annotations

from dataclasses import replace
import math

import numpy as np

from kindred.core.intervention_schedule import InterventionScheduleError, PhysicalQuantity
from kindred.core.volume import PhysicalDomainError, VolumeTrajectory, volume_input_in_range


class VolumeExecution:
    def __init__(self, *, species_names, parameter_values, end_time):
        self.names = tuple(species_names)
        self.parameters = parameter_values
        self.end_time = end_time
        self.value = None
        self.initial_value = None
        self.initial_time = None
        self.rows = {}
        self.segments = []

    def _error(self, item, time, message, **operands):
        return PhysicalDomainError(
            message, time=time, declaration=item.declaration_ordinal,
            operation=item.physical.op, phase=item.phase,
            authored_ordinal=item.authored_ordinal, window_ordinal=item.window_ordinal,
            **operands,
        )

    def _value(self, quantity, temperature):
        value = (quantity.value if quantity.parameter is None else
                 self.parameters(temperature, (quantity.parameter,))[quantity.parameter])
        return float(value)

    def _quantity(self, item, path, quantity, time, temperature, *, positive=False):
        value = self._value(quantity, temperature)
        if path in {"volume", "solution.volume", "solution.volume_rate"} and not volume_input_in_range(value, allow_zero=True):
            unit = "m³/s" if path == "solution.volume_rate" else "m³"
            raise self._error(item, time,
                f"{path} must be zero or from 1e-30 through 1e30 {unit}.",
                field=path, value=value, unit=unit, parameter=quantity.parameter)
        if not math.isfinite(value) or value < 0 or (positive and value == 0):
            unit = "M" if ".stock." in path else "1" if path == "factor" else "m³/s" if "rate" in path else "m³"
            raise self._error(item, time,
                f"{path} must be {'positive' if positive else 'nonnegative'} and finite.",
                field=path, value=value, unit=unit, parameter=quantity.parameter)
        return value

    def initialize(self, declaration, time, temperature):
        quantity, identity = declaration
        value = self._value(quantity, temperature)
        if not volume_input_in_range(value):
            raise PhysicalDomainError(
                "initial_volume must be from 1e-30 through 1e30 m³.", time=time, declaration=identity[0],
                operation="initial_volume", field=identity[2], authored_ordinal=identity[1],
                value=value, unit="m³", parameter=quantity.parameter,
            )
        self.value = self.initial_value = value
        self.initial_time = time
        self.rows[time] = value

    def require_volume(self, item, time):
        if self.value is None:
            raise InterventionScheduleError(
                "A volume-aware intervention requires an initial_volume declaration.",
                declaration_ordinal=item.declaration_ordinal, details={"time": time},
            )

    def apply(self, item, state, *, time, temperature, held_indices):
        physical = item.physical
        amount_path = ("solution." if physical.op == "jump" else "") + physical.amount_field
        amount = self._quantity(item, amount_path, physical.amount, time, temperature,
                                positive=physical.op == "scale")
        output = state.copy()
        resolved_stock = []
        old_volume = self.value
        with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
            if physical.op == "jump":
                stock = np.zeros(len(self.names))
                for name, quantity in physical.stock:
                    value = self._quantity(item, f"solution.stock.{name}.value", quantity, time, temperature)
                    stock[self.names.index(name)] = value
                    resolved_stock.append((name, PhysicalQuantity(value)))
                new_volume = old_volume + amount
                # Weighted concentrations avoid overflowing intermediate amounts.
                output = output * (old_volume / new_volume) + stock * (amount / new_volume)
            elif physical.op == "scale":
                output *= amount
                new_volume = None if old_volume is None else old_volume / amount
            else:
                new_volume = old_volume - amount
        if new_volume is not None and (not math.isfinite(new_volume) or new_volume <= 0):
            raise self._error(item, time, "Physical intervention must leave positive finite vessel volume.",
                field="volume", unit="m³", volume_before=old_volume, amount=amount, volume_after=new_volume)
        for index in held_indices:
            output[index] = state[index]
        if np.any(~np.isfinite(output)):
            raise self._error(item, time, "Physical intervention produced nonfinite concentrations.",
                field="concentrations", unit="M", amount=amount, volume_before=old_volume,
                concentrations_before=tuple(map(float, state)), concentrations_after=tuple(map(float, output)))
        self.value = new_volume
        if new_volume is not None:
            self.rows[time] = new_volume
        resolved = replace(physical, amount=PhysicalQuantity(amount), stock=tuple(resolved_stock), initial_volume=None)
        return output, replace(item, physical=resolved)

    def flow(self, items, time, temperature, vessel, concentrations, held_indices):
        rate = 0.0
        contribution = np.zeros(len(self.names))
        for item in items:
            if item.physical is None:
                continue
            if not math.isfinite(vessel) or vessel <= 0:
                raise self._error(item, time, "Solution inflow requires positive finite vessel volume.",
                                  field="volume", value=vessel, unit="m³")
            q = self._quantity(item, "solution.volume_rate", item.physical.amount, time, temperature)
            rate += q
            stock = np.zeros(len(self.names))
            for name, quantity in item.physical.stock:
                stock[self.names.index(name)] = self._quantity(item, f"solution.stock.{name}.value", quantity, time, temperature)
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                contribution += (q / vessel) * (stock - concentrations)
            for index in held_indices:
                contribution[index] = 0.0
            if not math.isfinite(rate) or np.any(~np.isfinite(contribution)):
                raise self._error(item, time, "Solution inflow must produce finite volume and concentration rates.",
                    field="inflow", volume=vessel, volume_unit="m³", volume_rate=rate, volume_rate_unit="m³/s",
                    mixing=tuple(map(float, contribution)), mixing_unit="M/s")
        return rate, contribution

    def record_segment(self, start, end, dense):
        self.segments.append((start, end, dense))
        self.value = float(dense(end)[-1])
        self.rows[end] = self.value

    def trajectory(self, times):
        if self.initial_time is None:
            return None
        first = int(np.searchsorted(times, self.initial_time))
        values = np.empty(len(times) - first)
        segment_index = 0
        for index, time in enumerate(times[first:]):
            if time in self.rows:
                values[index] = self.rows[time]
                continue
            while segment_index < len(self.segments) and self.segments[segment_index][1] < time:
                segment_index += 1
            start, end, dense = self.segments[segment_index]
            if not start <= time <= end:
                raise ValueError("Volume samples must belong to an executed segment.")
            values[index] = dense(time)[-1]
        return VolumeTrajectory(first, values)
